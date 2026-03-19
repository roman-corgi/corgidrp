"""
End-to-end test for spectroscopic ND filter calibration (prism mode).

Measures optical density as a function of wavelength, OD(lambda), for the
ND225 focal-plane filter in Band 3 spectroscopy (PRISM3) mode.

Pipeline
--------
L1 → L2a  (l1_to_l2a_basic.json)
   → L2b  (l2a_to_l2b_spec.json)
   → NDSpectroscopy cal product  (l2b_to_nd_filter_spec.json)
        steps: divide_by_exptime
               determine_wave_zeropoint   [needs 3D + 3F frames]
               add_wavelength_map         [needs DispersionModel from CalDB]
               extract_spec
               create_nd_filter_cal_spec  [needs SpecFluxCal or dim-star frames]

Required L1 input files  (<e2edata_path>/ND_SPEC/L1/)
-----------------------------------------------------
All frames must share the same EXPTIME, EMGAIN_C, and KGAINPAR so that
a single synthesized dark can be subtracted.  Generate them with
corgisim/make_nd_spec_l1_data.py.

  1. Narrowband dim-star frames  (CFAMNAME=3D, FPAMNAME=OPEN_34, DPAMNAME=PRISM3)
       VISTYPE  = CGIVST_CAL_ABSFLUX_FAINT
       TARGET   = <dim CALSPEC star, e.g. "tyc 4424-1286-1">
       Purpose  : wavelength zero-point via determine_wave_zeropoint
       Minimum  : 1 frame (2 recommended)

  2. Broadband dim-star frames   (CFAMNAME=3F, FPAMNAME=OPEN_34, DPAMNAME=PRISM3)
       VISTYPE  = CGIVST_CAL_ABSFLUX_FAINT
       TARGET   = <same dim CALSPEC star>
       Purpose  : spectral flux calibration C(lambda) via spec_fluxcal,
                  which converts CALSPEC SED [erg/s/cm^2/AA] to detector
                  counts [e-/s/bin] at each wavelength
       Minimum  : 1 frame (3 recommended)

  3. Broadband bright-star frames (CFAMNAME=3F, FPAMNAME=ND225, DPAMNAME=PRISM3)
       VISTYPE  = CGIVST_CAL_ABSFLUX_BRIGHT
       TARGET   = <bright CALSPEC star, e.g. "109 vir">
       Purpose  : OD(lambda) measurement — pipeline computes
                  OD(lambda) = -log10(counts_ND / expected_counts)
                  where expected_counts = SED_bright(lambda) / C(lambda)
       Minimum  : 1 frame (3 recommended)

TARGET must be a key in corgidrp.fluxcal.calspec_names so that the
CALSPEC SED can be auto-downloaded from STScI.

Required calibration files
--------------------------
A. Detector calibrations — flat files under <e2edata_path>/TV-36_Coronagraphic_Data/Cals/:
     nonlin_table_240322.txt    — nonlinearity correction table
     dark_current_20240322.fits — per-pixel dark current [e-/s]
     flat.fits                  — flat field
     fpn_20240322.fits          — fixed-pattern noise map
     cic_20240322.fits          — clock-induced charge map
     bad_pix.fits               — bad pixel map
   These are used to build NonLinearityCalibration, KGain, DetectorNoiseMaps,
   SynthesizedDark, FlatField, and BadPixelMap products in a temporary CalDB.

B. Spectroscopy calibrations — loaded automatically from corgidrp.default_cal_dir
   (~/.corgidrp/default_calibs/):
     DispersionModel_*.fits     — maps pixel position → wavelength (nm)
                                  for each band/prism combination
     SpecFilterOffset_*.fits    — narrowband/broadband filter centroid offsets,
                                  used by determine_wave_zeropoint

   These files ship with corgidrp and are created on first use by
   corgidrp.caldb.create_default_calibrations().  No manual action needed
   unless the default_calibs directory is missing or corrupted.

Output
------
A single NDSpectroscopy FITS product (*_nd_spec_cal.fits) containing:
    data[0, :]  — wavelength grid (nm)
    data[1, :]  — OD(lambda) spectrum
    err[0, 0, :] — wavelength uncertainty (nm)
    err[0, 1, :] — OD uncertainty (1-sigma)
with header keywords DATATYPE='NDSpectroscopy', FPAMNAME='ND225',
DPAMNAME='PRISM3', DATALVL='CAL'.

Baseline configuration tested here:
    FPAM  : ND225  (OD ~ 2.25)
    Band  : 3  (CFAMNAME = 3F / 3D)
    DPAM  : PRISM3
    Stars : TYC 4424-1286-1 (dim, Vmag~12) + 109 Vir (bright, Vmag~3.7)
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
    Build a temporary CalDB populated with detector calibrations and the
    corgidrp default spectroscopy calibrations.

    The CalDB is written to a temporary CSV (tmp_nd_spec_e2e_caldb.csv) so
    it does not interfere with the user's real CalDB.

    Parameters
    ----------
    l1_datadir : str
        Directory containing the L1 input files.  Two files are copied to
        create provenance headers for the calibration products.
    processed_cal_path : str
        Directory containing flat detector calibration files.  The following
        filenames are expected (TV-36 TVAC naming convention):
            nonlin_table_240322.txt    — nonlinearity correction table (CSV)
            dark_current_20240322.fits — per-pixel dark current image
            flat.fits                  — flat field image
            fpn_20240322.fits          — fixed-pattern noise image
            cic_20240322.fits          — clock-induced charge image
            bad_pix.fits               — bad pixel map image
    calibrations_dir : str
        Output directory where the built calibration FITS files are saved.
    logger : logging.Logger

    Returns
    -------
    this_caldb : corgidrp.caldb.CalDB
        Populated temporary calibration database containing:
            NonLinearityCalibration, KGain, DetectorNoiseMaps,
            SynthesizedDark (analog mode only), FlatField, BadPixelMap,
            DispersionModel, SpecFilterOffset  (from corgidrp.default_cal_dir)
    is_pc_data : bool
        True if the L1 data are photon-counted (ISPC=1).
        If True, a PC dark is built later from L2a frames rather than here.
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
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table.txt")
    dark_path   = os.path.join(processed_cal_path, "dark_current.fits")
    flat_path   = os.path.join(processed_cal_path, "flat.fits")
    fpn_path    = os.path.join(processed_cal_path, "fpn.fits")
    cic_path    = os.path.join(processed_cal_path, "cic.fits")
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

    # Determine whether the dataset contains any photon-counted frames.
    # Check all L1 files, not just mock_cal_files, because the last files
    # written (bright star through ND) are in analog mode (ISPC=0) while
    # the dim star frames are in PC mode (ISPC=1).
    sample_hdr = fits.getheader(mock_cal_files[0], ext=1)
    is_pc_data = any(
        bool(int(fits.getheader(os.path.join(l1_datadir, f), ext=1).get('ISPC', 0)))
        for f in all_l1
    )

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
        # Build one synthesized dark per unique (EXPTIME, EMGAIN_C) so that
        # dim-star and bright-star frames can use different exposure times.
        seen_configs = {}
        for fname in all_l1:
            h = fits.getheader(os.path.join(l1_datadir, fname), ext=1)
            key = (float(h['EXPTIME']), float(h['EMGAIN_C']))
            if key not in seen_configs:
                seen_configs[key] = fname
        for (exptime, emgain_c), fname in seen_configs.items():
            src = os.path.join(l1_datadir, fname)
            tmp_f = shutil.copy2(
                src, os.path.join(mock_cal_dir, os.path.basename(src)))
            tmp_fixed = fix_hdrs_for_tvac([tmp_f], mock_cal_dir)
            if not tmp_fixed:
                tmp_fixed = [tmp_f]
            with fits.open(tmp_fixed[0], mode='update') as hdul:
                if 'ISPC' in hdul[1].header:
                    hdul[1].header['ISPC'] = int(hdul[1].header['ISPC'])
            tmp_ds = data.Dataset(tmp_fixed)
            tmp_ds.frames[0].ext_hdr['EXPTIME']  = exptime
            tmp_ds.frames[0].ext_hdr['EMGAIN_C'] = emgain_c
            dark_cal = build_synthesized_dark(tmp_ds, noise_map)
            mocks.rename_files_to_cgi_format([dark_cal], calibrations_dir, "drk_cal")
            this_caldb.create_entry(dark_cal)
            logger.info(f"Analog dark created: EXPTIME={exptime}s, EMGAIN_C={emgain_c}.")
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

    Steps
    -----
    1. Build a temporary CalDB from detector flat files + corgidrp default
       spectroscopy calibrations (DispersionModel, SpecFilterOffset).
    2. Copy and fix L1 FITS headers (fix_hdrs_for_tvac).
    3. L1 → L2a  via walker (prescan_biassub, cosmic ray detection,
       nonlinearity correction).
    4. Optionally build a PC dark from L2a frames (photon-counted data only).
    5. L2a → L2b  via walker (convert_to_electrons, em_gain_division,
       dark_subtraction, desmear, CTI correction, bad-pixel correction).
       Patches EACQ_ROW/EACQ_COL to image centre after this step.
    6. L2b → NDSpectroscopy  via walker (divide_by_exptime,
       determine_wave_zeropoint, add_wavelength_map, extract_spec,
       create_nd_filter_cal_spec).
    7. Validate the output NDSpectroscopy product (shape, wavelength range,
       OD positivity, header keywords).

    Parameters
    ----------
    l1_datadir : str
        Directory containing L1 FITS files.  Must include at least:
          - Narrowband dim-star frames  (CFAMNAME=3D, FPAMNAME=OPEN_34,
            DPAMNAME=PRISM3, VISTYPE=CGIVST_CAL_ABSFLUX_FAINT)
          - Broadband dim-star frames   (CFAMNAME=3F, FPAMNAME=OPEN_34,
            DPAMNAME=PRISM3, VISTYPE=CGIVST_CAL_ABSFLUX_FAINT)
          - Broadband bright-star frames (CFAMNAME=3F, FPAMNAME=ND225,
            DPAMNAME=PRISM3, VISTYPE=CGIVST_CAL_ABSFLUX_BRIGHT)
        All frames must share the same EXPTIME, EMGAIN_C, and KGAINPAR.
        TARGET primary-header keyword must be a key in
        corgidrp.fluxcal.calspec_names for auto CALSPEC lookup.
        Generate with corgisim/make_nd_spec_l1_data.py.
    processed_cal_path : str
        Directory containing flat detector calibration files; see
        _setup_caldb() for the expected filenames.
    outputdir : str
        Root output directory.  Intermediate L2a/L2b files and the final
        NDSpectroscopy product are written here.
    logger : logging.Logger

    Returns
    -------
    nd_spec_cal : corgidrp.data.NDSpectroscopy
        Validated spectroscopic ND filter calibration product with:
            .wavelengths  — wavelength grid (nm), shape (M,)
            .od_spectrum  — OD(lambda),           shape (M,)
            .od_err       — 1-sigma OD error,     shape (M,)
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
        walker.walk_corgidrp(l1_filelist, "", outputdir,
                             template="l1_to_l2a_basic.json")

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
        # get_pc_mean requires a single VISTYPE per run.  Split by VISTYPE
        # (CGIVST_CAL_ABSFLUX_FAINT / CGIVST_CAL_ABSFLUX_BRIGHT) and run
        # the three PC spec recipes independently for each group.
        l2a_dataset = data.Dataset(l2a_filelist, no_data=True, no_err=True, no_dq=True)
        vistype_groups, _ = l2a_dataset.split_dataset(prihdr_keywords=['VISTYPE'])
        for group in vistype_groups:
            group_files = [f.filepath for f in group.frames]
            vt = group.frames[0].pri_hdr['VISTYPE']
            group_ispc = int(group.frames[0].ext_hdr.get('ISPC', 1))
            logger.info(f"  L2a→L2b: VISTYPE={vt}, ISPC={group_ispc} ({len(group_files)} files)")
            if group_ispc == 1:
                # Photon-counting path (dim star, ISPC=1)
                recipe = walker.autogen_recipe(group_files, outputdir)
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
                # Analog path (bright star through ND filter, ISPC=0)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    walker.walk_corgidrp(group_files, "", outputdir,
                                         template="l2a_to_l2b_spec.json")
    else:
        # Split by EXPTIME so each group is paired with its matching dark.
        # Dim-star and bright-star frames may use different exposure times.
        l2a_ds = data.Dataset(l2a_filelist, no_data=True, no_err=True, no_dq=True)
        exptime_groups, _ = l2a_ds.split_dataset(exthdr_keywords=['EXPTIME'])
        for group in exptime_groups:
            group_files = [f.filepath for f in group.frames]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                walker.walk_corgidrp(group_files, "", outputdir,
                                     template="l2a_to_l2b_spec.json")

    l2b_filelist = sorted(
        os.path.join(outputdir, f)
        for f in os.listdir(outputdir) if f.endswith('_l2b.fits')
    )
    logger.info(f"L2a → L2b complete: {len(l2b_filelist)} L2b files produced.")

    # Patch EACQ to image centre (workaround for cropped-frame coordinates)
    _patch_eacq_to_center(l2b_filelist)

    # ------------------------------------------------------------------ #
    # Diagnostic: inspect L2b frame signal levels before NDSpec recipe   #
    # ------------------------------------------------------------------ #
    for fpath in sorted(l2b_filelist):
        hdr  = fits.getheader(fpath, ext=1)
        data_arr = fits.getdata(fpath, ext=1).astype(float)
        cfam = hdr.get('CFAMNAME', '?')
        fpam = hdr.get('FPAMNAME', '?')
        logger.info(
            f"L2b {os.path.basename(fpath)}: "
            f"CFAM={cfam} FPAM={fpam} "
            f"shape={data_arr.shape} "
            f"min={data_arr.min():.3g} max={data_arr.max():.3g} "
            f"sum={data_arr.sum():.3g} mean={data_arr.mean():.3g} "
            f"median={np.median(data_arr):.3g}"
        )
        if cfam == '3D':
            ypeak, xpeak = np.unravel_index(np.argmax(data_arr), data_arr.shape)
            logger.info(f"  NB peak pixel: ({xpeak}, {ypeak}) = {data_arr[ypeak, xpeak]:.3g}")

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
        walker.walk_corgidrp(l2b_filelist, "", outputdir,
                             template="l2b_to_nd_filter_spec.json")

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

    Skips gracefully if the L1 data directory does not exist.

    Expected directory layout::

        <e2edata_path>/
            ND_SPEC/
                L1/
                    cgi_*_l1_.fits   (3D dim-star frames,  CFAMNAME=3D, FPAMNAME=OPEN_34)
                    cgi_*_l1_.fits   (3F dim-star frames,  CFAMNAME=3F, FPAMNAME=OPEN_34)
                    cgi_*_l1_.fits   (3F bright-star frames, CFAMNAME=3F, FPAMNAME=ND225)
                Cals/
                    nonlin_table.txt
                    dark_current.fits
                    flat.fits
                    fpn.fits
                    cic.fits
                    bad_pix.fits

    Generate the L1 files with::

        python corgisim/make_nd_spec_l1_data.py -o <e2edata_path>
    """
    l1_datadir        = os.path.join(e2edata_path, "ND_SPEC", "L1")
    processed_cal_path = os.path.join(e2edata_path, "ND_SPEC", "Cals")
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
        default="/Users/jmilton/Documents/CGI/E2E_Test_Data2",
        help="Root directory containing ND_SPEC/L1/ and TV-36.../Cals/ sub-folders"
    )
    ap.add_argument(
        "-o", "--outputdir",
        default=thisfile_dir,
        help="Directory to write all output products and logs"
    )
    args = ap.parse_args()

    l1_datadir         = os.path.join(args.e2edata_dir, "ND_SPEC", "L1")
    processed_cal_path = os.path.join(args.e2edata_dir, "ND_SPEC", "Cals")
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
