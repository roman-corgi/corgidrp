"""
End-to-end test starting at L1 for spectroscopy flux calibration.

Measures the spectral flux calibration factor

Pipeline
--------
L1 -> L2a  (l1_to_l2a_basic.json)
   -> L2b  (l2a_to_l2b_spec.json)
   -> SpecFluxCal cal product  (l2b_to_spec_flux.json)
        steps: divide_by_exptime
               determine_wave_zeropoint   (needs 3D + 3F frames)
               add_wavelength_map         (needs DispersionModel from CalDB)
               extract_spec
               spec_fluxcal

Required L1 input files (generated from corgisim)
-------------------------------------------------
Exploit the dim stars simulation from ND_SPEC thanks to Julia
Dim star frames must all have the same EXPTIME, EMGAIN_C, and KGAINPAR so that
a single dark can be subtracted.  

  1. Narrowband dim star frames  (CFAMNAME=3D, FPAMNAME=OPEN_34, DPAMNAME=PRISM3)
       VISTYPE  = CGIVST_CAL_ABSFLUX_FAINT
       TARGET   = dim CALSPEC star "tyc 4424-1286-1"
       Purpose  : wavelength zero-point via determine_wave_zeropoint

  2. Broadband dim-star frames   (CFAMNAME=3F, FPAMNAME=OPEN_34, DPAMNAME=PRISM3)
       VISTYPE  = CGIVST_CAL_ABSFLUX_FAINT
       TARGET   = dim CALSPEC star "tyc 4424-1286-1"
       Purpose  : spectral flux calibration C(lambda) via spec_fluxcal,
                  which converts CALSPEC SED (erg/s/cm^2/AA) to detector
                  counts (e-/s/bin) at each wavelength


TARGET should be in corgidrp.fluxcal.calspec_names so that the CALSPEC SED can be 
downloaded.

Required calibration files
--------------------------
A. Detector calibrations (generated from corgisim):
     nonlin_table.txt    — nonlinearity correction table
     dark_current.fits   — per-pixel dark current [e-/s]
     flat.fits           — flat field
     fpn.fits            — fixed-pattern noise map
     cic.fits            — clock-induced charge map
     bad_pix.fits        — bad pixel map
   These are used to build NonLinearityCalibration, KGain, DetectorNoiseMaps,
   SynthesizedDark, FlatField, and BadPixelMap cal products in a temporary CalDB.

B. Spectroscopy calibrations (loaded automatically from corgidrp.default_cal_dir):
     DispersionModel_*.fits     — maps pixel position to wavelength (nm)
                                  for each band/prism combination
     SpecFilterOffset_*.fits    — narrowband/broadband filter centroid offsets,
                                  used by determine_wave_zeropoint

Output
------
A single SpecFluxCal FITS product (*_nds_cal.fits) containing:
    data[0, :]  — wavelength grid (nm)
    data[1, :]  — spectrum of flux calibration factors
    err[0, 0, :] — wavelength uncertainty (nm)
    err[0, 1, :] — flux calibration factor uncertainty (1-sigma)


The test here does:
    Band  : 3  (CFAMNAME = 3F / 3D)
    DPAM  : PRISM3
    Star : TYC 4424-1286-1 (dim, Vmag~12)
"""
import argparse
import os
import shutil
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
from corgidrp.photon_counting import get_pc_mean
from corgidrp.darks import build_synthesized_dark
import corgidrp.detector as detector
import corgidrp.fluxcal as fluxcal
import corgidrp.l2b_to_l3 as l2b_to_l3
import corgidrp.spec as spec
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_caldb(l1_datadir, processed_cal_path, calibrations_dir):
    """
    Make a temporary CalDB populated with detector calibrations and the
    default spectroscopy calibrations.

    Args:
        l1_datadir (str): Directory containing the L1 input files. 
        processed_cal_path (str):  Directory containing flat detector calibration files.  
            The following filenames are expected:
                nonlin_table.txt    — nonlinearity correction table (CSV)
                dark_current.fits   — per-pixel dark current image
                flat.fits           — flat field image
                fpn.fits            — fixed-pattern noise image
                cic.fits            — clock-induced charge image
                bad_pix.fits        — bad pixel map image
        calibrations_dir (str): Output directory where the built calibration FITS files 
            are saved.

    Returns:
        this_caldb (corgidrp.caldb.CalDB): Populated temporary calibration database 
            containing:
                NonLinearityCalibration, KGain, DetectorNoiseMaps,
                SynthesizedDark (analog mode only), FlatField, BadPixelMap,
                DispersionModel, SpecFilterOffset  (from corgidrp.default_cal_dir)
        is_pc_data (bool): True if the L1 data are photon-counted (ISPC=1).
            If True, a PC dark is built later from L2a frames rather than here.
    """
    # Use a temporary CSV to avoid issues with real CalDB
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_spec_flux_e2e_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()

    # Get default spectroscopy calibrations 
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    print(f"Loaded default calibrations from {corgidrp.default_cal_dir}")

    # Paths to detector calibration files
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table.txt")
    dark_path   = os.path.join(processed_cal_path, "dark_current.fits")
    flat_path   = os.path.join(processed_cal_path, "flat.fits")
    fpn_path    = os.path.join(processed_cal_path, "fpn.fits")
    cic_path    = os.path.join(processed_cal_path, "cic.fits")
    bp_path     = os.path.join(processed_cal_path, "bad_pix.fits")

    # Build a mock input_dataset from a couple of L1 files so that
    # calibration products can record them
    all_l1 = sorted(f for f in os.listdir(l1_datadir)
                    if f.endswith('l1.fits') or f.endswith('l1_.fits'))
    mock_cal_files = [os.path.join(l1_datadir, f) for f in all_l1[-2:]]
    mock_cal_dir   = os.path.join(os.path.dirname(calibrations_dir), 'mock_cal_input')
    os.makedirs(mock_cal_dir, exist_ok=True)
    mock_cal_files = [
        shutil.copy2(f, os.path.join(mock_cal_dir, os.path.basename(f)))
        for f in mock_cal_files
    ]
    mock_input_dataset = data.Dataset(mock_cal_files)

    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr['DRPCTIME'] = time.Time.now().isot
    ext_hdr['DRPVERSN'] = corgidrp.__version__

    # Determine whether the dataset contains any photon-counted frames
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

    # Noise map
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

    # Synthesized dark: caldb drk entry only for analog. PC pipeline dark
    # comes from L2a, but we still need a drk FITS for BadPixelMap input_dataset.
    fname0 = all_l1[0]
    h0 = fits.getheader(os.path.join(l1_datadir, fname0), ext=1)
    exptime = float(h0['EXPTIME'])
    emgain_c = float(h0['EMGAIN_C'])
    src = os.path.join(l1_datadir, fname0)
    tmp_f = shutil.copy2(
        src, os.path.join(mock_cal_dir, os.path.basename(src)))
    tmp_ds = data.Dataset([tmp_f])
    tmp_ds.frames[0].ext_hdr['EXPTIME'] = exptime
    tmp_ds.frames[0].ext_hdr['EMGAIN_C'] = emgain_c
    dark_cal = build_synthesized_dark(tmp_ds, noise_map)
    mocks.rename_files_to_cgi_format([dark_cal], calibrations_dir, "drk_cal")
    if not is_pc_data:
        this_caldb.create_entry(dark_cal)
    bpm_drk_input_path = dark_cal.filepath

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
        input_dataset=data.Dataset([bpm_drk_input_path]))
    mocks.rename_files_to_cgi_format([bp_map], calibrations_dir, "bpm_cal")
    this_caldb.create_entry(bp_map)

    print("Calibration database populated.")
    return this_caldb, is_pc_data


# ---------------------------------------------------------------------------
# Test 
# ---------------------------------------------------------------------------

def run_spec_flux_e2e(l1_datadir, processed_cal_path, outputdir):
    """
    Run and validate the spectroscopy flux calibration pipeline.

    1. Build a temporary CalDB from detector files + corgidrp default
       spectroscopy calibrations 
    2. L1 -> SpecFluxCal  via walker 
    3. Validate the output SpecfluxCal product (shape, wavelength range,
       values, header keywords)

    Args:
    l1_datadir (str): Directory containing L1 FITS files
    processed_cal_path (str): Directory containing detector calibration files
    outputdir (str): Root output directory.  Intermediate L2a/L2b files and the 
        final SpecFluxCal product are written here.

    Returns: 
    spec_flux_cal (corgidrp.data.SpecFluxCal): Spectroscopy flux calibration 
        product with:
            .wavelengths  — wavelength grid (nm), shape (M,)
            .spec_fluxcal  — (spec fluxcal, lambda), shape (M,)
            .spec_fluxcal_err  — 1-sigma fluxcal error, shape (M,)
    """
    # ------------------------------------------------------------------ 
    # 1. Caldb                                             
    # ------------------------------------------------------------------ 
    calibrations_dir = os.path.join(outputdir, 'calibrations')
    os.makedirs(calibrations_dir, exist_ok=True)

    this_caldb, is_pc_data = setup_caldb(
        l1_datadir, processed_cal_path, calibrations_dir)

    # ------------------------------------------------------------------
    # 2. Prepare L1 input files (only 5 of the faint star without ND filter)
    # ------------------------------------------------------------------
    l1_filelist = sorted(
        os.path.join(l1_datadir, f)
        for f in os.listdir(l1_datadir)
        if f.endswith('l1_.fits') or f.endswith('l1.fits')
    )[0:5]
    if not l1_filelist:
        raise FileNotFoundError(f"No L1 files found in {l1_datadir}")

    print(f"Found {len(l1_filelist)} L1 input files.")

    # ------------------------------------------------------------------ 
    # 3. L1 ->SpecFluxCal                                                        
    # ------------------------------------------------------------------ 
    print("Running L1 -> SpecFluxCal")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l1_filelist, "", outputdir)

    print(f"L1 -> SpecFluxCal complete.")

    # ------------------------------------------------------------------ 
    # 4. Find and load the SpecFluxCal (sfl) product                      
    # ------------------------------------------------------------------ 
    spec_flux_files = sorted(
        os.path.join(outputdir, f)
        for f in os.listdir(outputdir) if f.endswith('_sfl_cal.fits')
    )
    if not spec_flux_files:
        raise AssertionError(
            f"No _sfc_cal.fits file found in {outputdir}. "
            "Check that the pipeline ran and the walker routing is correct."
        )

    spec_flux_cal = data.SpecFluxCal(spec_flux_files[0])
    print(f"SpecFluxCal product loaded: {spec_flux_files[0]}")

    # ------------------------------------------------------------------ 
    # 5. Validation                                                       
    # ------------------------------------------------------------------ 
    print("Validating SpecFluxCal product")

    # Data shape: (2, M)
    assert spec_flux_cal.data.ndim == 2 and spec_flux_cal.data.shape[0] == 2, \
        f"Unexpected data shape: {spec_flux_cal.data.shape}"
    M = spec_flux_cal.data.shape[1]
    print(f"  Data shape: (2, {M})  PASSED")

    # Wavelengths should be in the band 3 range
    wave = spec_flux_cal.wavelength
    sort_idx = np.argsort(wave)
    spec_wave  = wave[sort_idx]
    assert np.all(np.diff(spec_wave) > 0), "Wavelength grid is not monotonically increasing."
    assert spec_wave[0] > 500 and spec_wave[-1] < 1100, \
        f"Wavelengths {spec_wave[0]:.1f}–{spec_wave[-1]:.1f} nm outside expected range."
    print(f"  Wavelength range: {spec_wave[0]:.1f}–{spec_wave[-1]:.1f} nm PASSED")

    # spec fluxcal values should be positive and finite
    specflux = spec_flux_cal.specflux
    assert np.all(np.isfinite(specflux)), "spectrum fluxcal contains non-finite values."
    assert np.all(specflux > 0), f"spectrum fluxcal contains non-positive values (min={specflux.min():.3f})."
    
    specflux_err = spec_flux_cal.specflux_err
    assert np.all(np.isfinite(specflux_err)), "spectrum fluxcal errors contains non-finite values."
    assert np.all(specflux_err > 0), f"spectrum fluxcal errors contains non-positive values (min={specflux_err.min():.3f})."
    
    #estimate flux cal factor from one l2b image
    l2b_filelist = sorted(
        os.path.join(outputdir, f)
        for f in os.listdir(outputdir) if f.endswith('_l2b.fits')
    )
    l2b_image = data.Image(l2b_filelist[2])
    target = l2b_image.pri_hdr["TARGET"]
    exptime = l2b_image.ext_hdr["EXPTIME"]
    calspec_filepath, _ = fluxcal.get_calspec_file(target)
    flux_ref = fluxcal.read_cal_spec(calspec_filepath,spec_wave * 10.)
    # Sum the same box extract_spec used, by hand: the recipe crops before extraction, so the
    # zero point is at (WV0_X, WV0_Y) of the cropped frame, the box is 2 * halfwidth + 1 = 5
    # columns wide, and it spans redheight rows toward -Y (red for PRISM3) and blueheight toward +Y.
    l2b_cropped = l2b_to_l3.crop(data.Dataset([l2b_image]))[0]
    xcent = int(round(spec_flux_cal.ext_hdr["WV0_X"]))
    ycent = int(round(spec_flux_cal.ext_hdr["WV0_Y"]))
    redheight, blueheight = spec.DEFAULT_SPEC_EXTRACT_HEIGHTS[l2b_image.ext_hdr["DPAMNAME"]]
    box = l2b_cropped.data[ycent - redheight:ycent + blueheight + 1, xcent - 2:xcent + 3]
    spec_counts = np.sum(np.mean(box, 0))/exptime
    est_fluxfac = np.mean(flux_ref)/spec_counts
    dev = (est_fluxfac - np.mean(specflux))/est_fluxfac

    assert dev < 0.1, f"deviation of the estimated spec flux cal factor is bigger than 10 %: {dev * 100:.3f}"
    print(f"deviation from estimated spec flux calibration factors is {dev *100:.3f} %, therefore spec flux cal values PASSED")

    # Headers
    assert spec_flux_cal.ext_hdr['DATATYPE'] == 'SpecFluxCal'
    assert spec_flux_cal.ext_hdr.get('DPAMNAME', '').startswith('PRISM'), \
        f"Expected DPAMNAME to start with 'PRISM', got '{spec_flux_cal.ext_hdr.get('DPAMNAME')}'"
    assert spec_flux_cal.ext_hdr['DATALVL'] == 'CAL'
    print("  Header keywords PASSED")

    # Remove temporary CalDB
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_spec_flux_e2e_caldb.csv')
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)

    print("SpecFluxCal E2E test PASSED.")
    return spec_flux_cal


# ---------------------------------------------------------------------------
# Pytest entry
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_spec_fluxcal_e2e(e2edata_path, e2eoutput_path):
    """
    Pytest wrapper for the spectroscopy flux calibration E2E test.

    Args:
        e2edata_path (str): Path to the E2E test data
        e2eoutput_path (str): Path to the E2E test output

    """
    l1_datadir        = os.path.join(e2edata_path, "ND_SPEC", "L1")
    processed_cal_path = os.path.join(e2edata_path, "ND_SPEC", "Cals")
    outputdir = os.path.join(e2eoutput_path, "l1_to_spec_fluxcal_e2e")

    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    run_spec_flux_e2e(l1_datadir, processed_cal_path, outputdir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir
    ap = argparse.ArgumentParser(
        description="Spectroscopy flux calibration E2E test (L1 -> SpecFluxCal)"
    )
    ap.add_argument(
        "-d", "--e2edata_dir",
        default="/home/schreiber/DataCopy/E2E_Test_Data",
        help="Root directory containing ND_SPEC/L1/ and ND_SPEC/Cals/ sub-folders"
    )
    ap.add_argument(
        "-o", "--outputdir",
        default=thisfile_dir,
        help="Directory to write all output products"
    )
    args = ap.parse_args()

    test_spec_fluxcal_e2e(e2edata_dir, outputdir)