"""
End-to-end test for spectroscopic ND filter calibration (prism mode).

Starts from L1 FITS files produced by corgisim and exercises the full
corgidrp pipeline:

    L1 → L2a (l1_to_l2a_basic.json)
       → L2b (l2a_to_l2b_spec.json)
       → NDSpectroscopy calibration product (l2b_to_nd_filter_spec.json)
              which internally runs:
                divide_by_exptime
                determine_wave_zeropoint
                add_wavelength_map
                extract_spec
                create_nd_filter_cal_spec

Expected data layout under e2edata_path:
    <e2edata_path>/
        ND_SPEC/
            L1/               ← L1 FITS files from corgisim
                cgi_*_l1_.fits  (dim-star frames with FPAMNAME=HOLE)
                cgi_*_l1_.fits  (bright-star frames with FPAMNAME=ND225)
        TV-36_Coronagraphic_Data/
            Cals/             ← standard detector calibration files

Baseline configuration:
    FPAM  : ND225
    Band  : 3 (CFAM 3F or similar)
    DPAM  : PRISM3
"""
import argparse
import logging
import os
import shutil
import traceback
import warnings

import astropy.io.fits as fits
import astropy.time as time
import numpy as np
import pytest

import corgidrp
import corgidrp.caldb as caldb
import corgidrp.check as check
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
from corgidrp.check import fix_hdrs_for_tvac
from corgidrp.darks import build_synthesized_dark

thisfile_dir = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_eacq_to_center(filelist):
    """
    Set EACQ_ROW/EACQ_COL to the centre of each (cropped) L2b frame.

    L2b frames have been cropped by prescan_biassub but the EACQ keywords
    still hold full-frame coordinates.  The L2b→L3 crop step uses EACQ as
    its centre, so updating to the image midpoint prevents the crop window
    from falling outside the data.
    """
    for path in filelist:
        with fits.open(path, mode='update') as hdul:
            h = hdul[1].header
            n1, n2 = int(h['NAXIS1']), int(h['NAXIS2'])
            h['EACQ_ROW'] = (n2 - 1) / 2.0
            h['EACQ_COL'] = (n1 - 1) / 2.0


def _setup_caldb(l1_datadir, processed_cal_path, calibrations_dir, logger):
    """
    Build a temporary CalDB populated with standard detector calibrations
    and the corgidrp default spectroscopy calibrations.

    Parameters
    ----------
    l1_datadir : str
        Directory containing the L1 input files (used to create mock
        input_dataset headers for calibration products).
    processed_cal_path : str
        Directory containing detector calibration flat files (dark, flat,
        nonlinearity table, noise maps, bad pixel map).
    calibrations_dir : str
        Output directory where mock calibration FITS products are saved.
    logger : logging.Logger

    Returns
    -------
    this_caldb : corgidrp.caldb.CalDB
        Populated temporary calibration database.
    is_pc_data : bool
        True if the L1 data is photon-counted.
    """
    # Use a temporary CSV so we don't pollute the user's real CalDB.
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_nd_spec_e2e_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()

    # Pull in default spectroscopy calibrations (DispersionModel,
    # SpecFilterOffset, etc.) bundled with corgidrp.
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    logger.info(f"Loaded default calibrations from {corgidrp.default_cal_dir}")

    # Paths to flat detector calibration files
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_path   = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path   = os.path.join(processed_cal_path, "flat.fits")
    fpn_path    = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path    = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_path     = os.path.join(processed_cal_path, "bad_pix.fits")

    # Build a minimal mock input_dataset from a couple of L1 files so that
    # calibration products can record their provenance.
    all_l1 = sorted(f for f in os.listdir(l1_datadir)
                    if f.endswith('l1.fits') or f.endswith('l1_.fits'))
    mock_cal_files = [os.path.join(l1_datadir, f) for f in all_l1[-2:]]
    mock_cal_dir   = os.path.join(os.path.dirname(calibrations_dir), 'mock_cal_input')
    os.makedirs(mock_cal_dir, exist_ok=True)
    mock_cal_files = [
        shutil.copy2(f, os.path.join(mock_cal_dir, os.path.basename(f)))
        for f in mock_cal_files
    ]
    mock_cal_files = fix_hdrs_for_tvac(mock_cal_files, mock_cal_dir)
    for f in mock_cal_files:
        with fits.open(f, mode='update') as hdul:
            if 'ISPC' in hdul[1].header:
                hdul[1].header['ISPC'] = int(hdul[1].header['ISPC'])
    mock_input_dataset = data.Dataset(mock_cal_files)

    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr['DRPCTIME'] = time.Time.now().isot
    ext_hdr['DRPVERSN'] = corgidrp.__version__

    # Determine whether the data are photon-counted
    sample_hdr = fits.getheader(mock_cal_files[0], ext=1)
    is_pc_data = bool(int(sample_hdr.get('ISPC', 0)))

    # Nonlinearity
    nonlin_dat  = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(
        nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format([nonlinear_cal], calibrations_dir, "nln_cal")
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    signal_array = np.linspace(0, 50)
    noise_array  = np.sqrt(signal_array)
    ext_hdr['RN']    = 100.0
    ext_hdr['RN_ERR'] = 0.0
    kgain = data.KGain(
        8.7, ptc=np.column_stack([signal_array, noise_array]),
        pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format([kgain], calibrations_dir, "krn_cal")
    this_caldb.create_entry(kgain)

    # Noise map (FPN + CIC + dark)
    import corgidrp.detector as detector
    fpn_dat  = fits.getdata(fpn_path)
    cic_dat  = fits.getdata(cic_path)
    dark_dat = fits.getdata(dark_path)
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    nm_dat = np.zeros((3,
                       detector.detector_areas['SCI']['frame_rows'],
                       detector.detector_areas['SCI']['frame_cols']))
    nm_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = np.array([fpn_dat, cic_dat, dark_dat])
    err_hdr_nm = fits.Header()
    err_hdr_nm['BUNIT'] = 'detected electron'
    ext_hdr['B_O']     = 0.
    ext_hdr['B_O_ERR'] = 0.
    noise_map = data.DetectorNoiseMaps(
        nm_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=mock_input_dataset,
        err=np.zeros([1] + list(nm_dat.shape)),
        dq=np.zeros(nm_dat.shape, dtype=int),
        err_hdr=err_hdr_nm)
    mocks.rename_files_to_cgi_format([noise_map], calibrations_dir, "dnm_cal")
    this_caldb.create_entry(noise_map)

    # Dark (analog only; PC dark is created later from L2a frames)
    if not is_pc_data:
        exptime  = float(sample_hdr['EXPTIME'])
        emgain_c = float(sample_hdr['EMGAIN_C'])
        tmp_ds = data.Dataset(mock_cal_files[:1])
        tmp_ds.frames[0].ext_hdr['EXPTIME']  = exptime
        tmp_ds.frames[0].ext_hdr['EMGAIN_C'] = emgain_c
        dark_cal = build_synthesized_dark(tmp_ds, noise_map)
        mocks.rename_files_to_cgi_format([dark_cal], calibrations_dir, "drk_cal")
        this_caldb.create_entry(dark_cal)
        logger.info("Analog dark calibration created.")
    else:
        logger.info("PC dark will be created from L2a frames.")

    # Flat field
    flat = data.FlatField(
        fits.getdata(flat_path),
        pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format([flat], calibrations_dir, "flt_cal")
    this_caldb.create_entry(flat)

    # Bad pixel map
    bp_map = data.BadPixelMap(
        fits.getdata(bp_path),
        pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format([bp_map], calibrations_dir, "bpm_cal")
    this_caldb.create_entry(bp_map)

    logger.info("Calibration database populated.")
    return this_caldb, is_pc_data


# ---------------------------------------------------------------------------
# Core test logic (separated so it can also be called from __main__)
# ---------------------------------------------------------------------------

def run_nd_filter_spec_e2e(l1_datadir, processed_cal_path, outputdir, logger):
    """
    Execute the full ND filter spectroscopy calibration pipeline and validate
    the resulting NDSpectroscopy product.

    Parameters
    ----------
    l1_datadir : str
        Directory containing the L1 input FITS files.
    processed_cal_path : str
        Directory containing detector calibration files.
    outputdir : str
        Root output directory for all intermediate and final products.
    logger : logging.Logger

    Returns
    -------
    nd_spec_cal : corgidrp.data.NDSpectroscopy
        The recovered spectroscopic ND filter calibration product.
    """
    # ------------------------------------------------------------------ #
    # 1. Calibration database                                             #
    # ------------------------------------------------------------------ #
    calibrations_dir = os.path.join(outputdir, 'calibrations')
    os.makedirs(calibrations_dir, exist_ok=True)

    this_caldb, is_pc_data = _setup_caldb(
        l1_datadir, processed_cal_path, calibrations_dir, logger)

    # ------------------------------------------------------------------ #
    # 2. Prepare L1 input files                                           #
    # ------------------------------------------------------------------ #
    input_l1_dir = os.path.join(outputdir, 'input_l1')
    os.makedirs(input_l1_dir, exist_ok=True)

    raw_l1_files = sorted(
        os.path.join(l1_datadir, f)
        for f in os.listdir(l1_datadir)
        if f.endswith('l1_.fits') or f.endswith('l1.fits')
    )
    if not raw_l1_files:
        raise FileNotFoundError(f"No L1 files found in {l1_datadir}")

    logger.info(f"Found {len(raw_l1_files)} L1 input files.")

    # Copy + fix headers for corgisim-generated files
    l1_filelist = fix_hdrs_for_tvac(raw_l1_files, input_l1_dir)
    for f in l1_filelist:
        with fits.open(f, mode='update') as hdul:
            if 'ISPC' in hdul[1].header:
                hdul[1].header['ISPC'] = int(hdul[1].header['ISPC'])

    # ------------------------------------------------------------------ #
    # 3. L1 → L2a                                                        #
    # ------------------------------------------------------------------ #
    logger.info("Running L1 → L2a …")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l1_filelist, "", outputdir)

    l2a_filelist = sorted(
        os.path.join(outputdir, f)
        for f in os.listdir(outputdir) if f.endswith('_l2a.fits')
    )
    logger.info(f"L1 → L2a complete: {len(l2a_filelist)} L2a files produced.")

    # PC dark (only needed for photon-counted data)
    if is_pc_data and l2a_filelist:
        from corgidrp.photon_counting import get_pc_mean
        num_dark = max(len(l2a_filelist), 10)
        _, dark_l1_ds, _, _ = mocks.create_photon_countable_frames(
            Nbrights=1, Ndarks=num_dark)
        dark_l1_dir = os.path.join(outputdir, 'pc_dark_l1')
        os.makedirs(dark_l1_dir, exist_ok=True)
        dark_l1_ds.save(filedir=dark_l1_dir)
        dark_l1_files = sorted(
            os.path.join(dark_l1_dir, f)
            for f in os.listdir(dark_l1_dir) if f.endswith('_l1_.fits')
        )
        dark_l2a_dir = os.path.join(outputdir, 'pc_dark_l2a')
        os.makedirs(dark_l2a_dir, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            walker.walk_corgidrp(dark_l1_files, "", dark_l2a_dir,
                                 template="l1_to_l2a_basic.json")
        dark_l2a_files = sorted(
            os.path.join(dark_l2a_dir, f)
            for f in os.listdir(dark_l2a_dir) if f.endswith('_l2a.fits')
        )
        pc_dark = get_pc_mean(data.Dataset(dark_l2a_files), inputmode='darks')
        mocks.rename_files_to_cgi_format([pc_dark], calibrations_dir, "drk_cal")
        this_caldb.create_entry(pc_dark)
        logger.info("PC dark created.")

    # ------------------------------------------------------------------ #
    # 4. L2a → L2b (spectroscopy recipe)                                 #
    # ------------------------------------------------------------------ #
    logger.info("Running L2a → L2b (spec) …")
    if is_pc_data:
        recipe = walker.autogen_recipe(l2a_filelist, outputdir)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            out1 = walker.run_recipe(recipe[0], save_recipe_file=True)
        recipe[1]['inputs'] = out1
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            out2 = walker.run_recipe(recipe[1], save_recipe_file=True)
        recipe[2]['inputs'] = out2
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            walker.run_recipe(recipe[2], save_recipe_file=True)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            walker.walk_corgidrp(l2a_filelist, "", outputdir)

    l2b_filelist = sorted(
        os.path.join(outputdir, f)
        for f in os.listdir(outputdir) if f.endswith('_l2b.fits')
    )
    logger.info(f"L2a → L2b complete: {len(l2b_filelist)} L2b files produced.")

    # Patch EACQ to image centre (workaround for cropped-frame coordinates)
    _patch_eacq_to_center(l2b_filelist)

    # ------------------------------------------------------------------ #
    # 5. L2b → NDSpectroscopy calibration product                        #
    #    (via l2b_to_nd_filter_spec.json:                                 #
    #     divide_by_exptime → determine_wave_zeropoint →                 #
    #     add_wavelength_map → extract_spec →                            #
    #     create_nd_filter_cal_spec → save)                              #
    # ------------------------------------------------------------------ #
    logger.info("Running L2b → NDSpectroscopy …")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l2b_filelist, "", outputdir)

    # ------------------------------------------------------------------ #
    # 6. Locate and load the NDSpectroscopy product                      #
    # ------------------------------------------------------------------ #
    nd_spec_files = sorted(
        os.path.join(outputdir, f)
        for f in os.listdir(outputdir) if f.endswith('_nd_spec_cal.fits')
    )
    if not nd_spec_files:
        raise AssertionError(
            f"No _nd_spec_cal.fits file found in {outputdir}. "
            "Check that the pipeline ran and the walker routing is correct."
        )

    nd_spec_cal = data.NDSpectroscopy(nd_spec_files[0])
    logger.info(f"NDSpectroscopy product loaded: {nd_spec_files[0]}")

    # ------------------------------------------------------------------ #
    # 7. Validation                                                       #
    # ------------------------------------------------------------------ #
    logger.info("Validating NDSpectroscopy product …")

    # Data shape: (2, M)
    assert nd_spec_cal.data.ndim == 2 and nd_spec_cal.data.shape[0] == 2, \
        f"Unexpected data shape: {nd_spec_cal.data.shape}"
    M = nd_spec_cal.data.shape[1]
    logger.info(f"  Data shape: (2, {M})  ✓")

    # Wavelengths should be monotonically increasing and in the band 3 range
    wave = nd_spec_cal.wavelengths
    assert np.all(np.diff(wave) > 0), "Wavelength grid is not monotonically increasing."
    assert wave[0] > 500 and wave[-1] < 1100, \
        f"Wavelengths {wave[0]:.1f}–{wave[-1]:.1f} nm outside expected range."
    logger.info(f"  Wavelength range: {wave[0]:.1f}–{wave[-1]:.1f} nm  ✓")

    # OD values should be positive and finite
    od = nd_spec_cal.od_spectrum
    assert np.all(np.isfinite(od)), "OD spectrum contains non-finite values."
    assert np.all(od > 0), f"OD spectrum contains non-positive values (min={od.min():.3f})."
    logger.info(f"  OD range: {od.min():.3f}–{od.max():.3f}  ✓")

    # Headers
    assert nd_spec_cal.ext_hdr['DATATYPE'] == 'NDSpectroscopy'
    assert nd_spec_cal.ext_hdr.get('FPAMNAME', '').startswith('ND'), \
        f"Expected FPAMNAME to start with 'ND', got '{nd_spec_cal.ext_hdr.get('FPAMNAME')}'"
    assert nd_spec_cal.ext_hdr.get('DPAMNAME', '').startswith('PRISM'), \
        f"Expected DPAMNAME to start with 'PRISM', got '{nd_spec_cal.ext_hdr.get('DPAMNAME')}'"
    assert nd_spec_cal.ext_hdr['DATALVL'] == 'CAL'
    logger.info("  Header keywords  ✓")

    # Remove temporary CalDB
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_nd_spec_e2e_caldb.csv')
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)

    logger.info("NDSpectroscopy E2E test PASSED.")
    return nd_spec_cal


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_nd_filter_spec_e2e(e2edata_path, e2eoutput_path):
    """
    Pytest wrapper for the spectroscopic ND filter calibration E2E test.

    Expected data layout::

        <e2edata_path>/
            ND_SPEC/
                L1/               ← L1 FITS files from corgisim
            TV-36_Coronagraphic_Data/
                Cals/             ← detector calibration files
    """
    l1_datadir        = os.path.join(e2edata_path, "ND_SPEC", "L1")
    processed_cal_path = os.path.join(
        e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    outputdir = os.path.join(e2eoutput_path, "nd_filter_spec_e2e")

    # Skip gracefully if test data have not been provided yet
    if not os.path.isdir(l1_datadir):
        pytest.skip(
            f"ND_SPEC L1 data not found at {l1_datadir}. "
            "Generate them with corgisim and re-run."
        )

    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    log_file = os.path.join(outputdir, 'nd_filter_spec_e2e.log')
    logger   = logging.getLogger('nd_filter_spec_e2e')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)

    try:
        run_nd_filter_spec_e2e(l1_datadir, processed_cal_path, outputdir, logger)
    except Exception as e:
        logger.error(f"Test FAILED: {e}")
        logger.error(traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Spectroscopic ND filter calibration E2E test (L1 → NDSpectroscopy)"
    )
    ap.add_argument(
        "-tvac", "--e2edata_dir",
        default=os.path.join(thisfile_dir, "test_data"),
        help="Root directory containing ND_SPEC/L1/ and TV-36.../Cals/ sub-folders"
    )
    ap.add_argument(
        "-o", "--outputdir",
        default=thisfile_dir,
        help="Directory to write all output products and logs"
    )
    args = ap.parse_args()

    l1_datadir         = os.path.join(args.e2edata_dir, "ND_SPEC", "L1")
    processed_cal_path = os.path.join(
        args.e2edata_dir, "TV-36_Coronagraphic_Data", "Cals")
    outputdir = os.path.join(args.outputdir, "nd_filter_spec_e2e")

    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    log_file = os.path.join(outputdir, 'nd_filter_spec_e2e.log')
    logger   = logging.getLogger('nd_filter_spec_e2e')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_nd_filter_spec_e2e(l1_datadir, processed_cal_path, outputdir, logger)
