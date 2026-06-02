"""
End-to-end test starting at L1 for spectroscopy ND filter calibration.

Measures optical density as a function of wavelength, OD(lambda), for the
ND225 focal plane filter in Band 3 spectroscopy (PRISM3) mode.

Pipeline
--------
L1 -> L2a  (l1_to_l2a_basic.json)
   -> L2b  (l2a_to_l2b_spec.json)
   -> NDSpectroscopy cal product  (l2b_to_nd_filter_spec.json)
        steps: divide_by_exptime
               determine_wave_zeropoint   (needs 3D + 3F frames)
               add_wavelength_map         (needs DispersionModel from CalDB)
               extract_spec
               create_nd_filter_cal_spec  (needs SpecFluxCal or dim-star frames)

Required L1 input files (generated from corgisim)
-------------------------------------------------
Dim star frames must all have the same EXPTIME, EMGAIN_C, and KGAINPAR so that
a single dark can be subtracted.  
Bright star frames also should have consistent EXPTIME, EMGAIN_C, and KGAINPAR so
that a separate dark can be subtracted.

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

  3. Broadband bright-star frames (CFAMNAME=3F, FPAMNAME=ND225, DPAMNAME=PRISM3)
       VISTYPE  = CGIVST_CAL_ABSFLUX_BRIGHT
       TARGET   = bright CALSPEC star, "109 vir"
       Purpose  : OD(lambda) = -log10(counts_ND / expected_counts)
                  where expected_counts = SED_bright(lambda) / C(lambda)

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
A single NDSpectroscopy FITS product (*_nds_cal.fits) containing:
    data[:, :, 0]  — OD(lambda) spectrum
    data[:, :, 1]  — wavelength grid (nm)
    data[:, :, 2]  - EXCAM star position in x-direction
    data[:, :, 3]  - EXCAM star position in y-direction
    err[0, :, :, 0] — OD uncertainty (1-sigma)
    err[0, :, :, 1] — wavelength uncertainty (nm)
    err[0, :, :, 2]  - Star position uncertainty in x-direction
    err[0, :, :, 3]  - Star position uncertainty in y-direction
    


The test here does:
    FPAM  : ND225  (OD ~ 2.20)
    Band  : 3  (CFAMNAME = 3F / 3D)
    DPAM  : PRISM3
    Stars : TYC 4413-304-1 (dim, Vmag~12.8) + eta Uma (bright, Vmag~1.9)
"""
import argparse
import os, sys
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
from corgidrp.fluxcal import get_calspec_file
import corgidrp.detector as detector
import corgidrp.nd_filter_calibration as nd_filter_calibration

thisfile_dir = os.path.dirname(__file__)


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
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_nd_spec_e2e_caldb.csv')
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

    # Dark (analog only. PC dark is created later from L2a frames)
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
            tmp_ds = data.Dataset([tmp_f])
            tmp_ds.frames[0].ext_hdr['EXPTIME']  = exptime
            tmp_ds.frames[0].ext_hdr['EMGAIN_C'] = emgain_c
            dark_cal = build_synthesized_dark(tmp_ds, noise_map)
            mocks.rename_files_to_cgi_format([dark_cal], calibrations_dir, "drk_cal")
            this_caldb.create_entry(dark_cal)
            print(f"Analog dark created: EXPTIME={exptime}s, EMGAIN_C={emgain_c}.")
    else:
        print("PC dark will be created from L2a frames.")

    # Flat field
    flat = data.FlatField(
        fits.getdata(flat_path),
        pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format([flat], calibrations_dir, "flt_cal")
    this_caldb.create_entry(flat)

    # Bad pixel map requires a dark(-like) frame
    if not is_pc_data:
        dark_cals = []
        for f in os.listdir(calibrations_dir):
            if f.endswith('_drk_cal.fits'):
                dark_cals.append(os.path.join(calibrations_dir, f))
        if dark_cals:
            bp_dark = data.Dark(dark_cals[0])
        else:
            bp_dark = build_synthesized_dark(data.Dataset([flat]), noise_map)
    else:
        bp_dark = build_synthesized_dark(data.Dataset([flat]), noise_map)

    bp_map_inputs = data.Dataset([bp_dark, flat])
    bp_map = data.BadPixelMap(
        fits.getdata(bp_path),
        pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=bp_map_inputs)
    mocks.rename_files_to_cgi_format([bp_map], calibrations_dir, "bpm_cal")
    this_caldb.create_entry(bp_map)

    print("Calibration database populated.")
    return this_caldb, is_pc_data


# ---------------------------------------------------------------------------
# Test 
# ---------------------------------------------------------------------------

def run_nd_filter_spec_e2e(l1_datadir, processed_cal_path, outputdir):
    """
    Run and validate the ND filter spectroscopy calibration pipeline.

    1. Build a temporary CalDB from detector files + corgidrp default
       spectroscopy calibrations 
    2. L1 -> L2a  via walker 
    3. Optionally build a PC dark from L2a frames (photon-counting only)
    4. L2a -> L2b  via walker
    5. L2b -> NDSpectroscopy via walker 
    6. Validate the output NDSpectroscopy product (shape, wavelength range,
       positive OD values, header keywords)

    Args:
    l1_datadir (str): Directory containing L1 FITS files
    processed_cal_path (str): Directory containing detector calibration files
    outputdir (str): Root output directory.  Intermediate L2a/L2b files and the 
        final NDSpectroscopy product are written here.

    Returns: 
    nd_spec_cal (corgidrp.data.NDSpectroscopy): Spectroscopy ND filter calibration 
        product with:
            .wavelengths  — wavelength grid (nm),         shape (M, N)
            .od_spectra   — OD(lambda),                   shape (M, N)
            .x_values     — EXCAM star position in x,     shape (M, N)
            .y_values     — EXCAM star position in y,     shape (M, N)
    """
    # ------------------------------------------------------------------ 
    # 1. Caldb                                             
    # ------------------------------------------------------------------ 
    calibrations_dir = os.path.join(outputdir, 'calibrations')
    os.makedirs(calibrations_dir, exist_ok=True)

    this_caldb, is_pc_data = setup_caldb(
        l1_datadir, processed_cal_path, calibrations_dir)

    # ------------------------------------------------------------------
    # 2. Prepare L1 input files
    # ------------------------------------------------------------------
    l1_filelist = sorted(
        os.path.join(l1_datadir, f)
        for f in os.listdir(l1_datadir)
        if f.endswith('l1_.fits') or f.endswith('l1.fits')
    )
    if not l1_filelist:
        raise FileNotFoundError(f"No L1 files found in {l1_datadir}")

    print(f"Found {len(l1_filelist)} L1 input files.")

    input_l1_datadir = os.path.join(outputdir,"input_l1")
    os.makedirs(input_l1_datadir)

    # Copy all L1 files to output_dir before alterations to header keywords
    shutil.copytree(l1_datadir,input_l1_datadir, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.txt', '*.png'))
    
    l1_filelist = sorted(
        os.path.join(input_l1_datadir, f)
        for f in os.listdir(input_l1_datadir)
        if f.endswith('l1_.fits') or f.endswith('l1.fits')
    )
    
    # Separating file into brightstar and faint star dataset and assigning a different visit id to the former.
    bright_star_filelist = sorted(f for f in l1_filelist if fits.getheader(f, ext=1)['FPAMNAME'] == 'ND225')
    faint_star_filelist = sorted(f for f in l1_filelist if fits.getheader(f, ext=1)['FPAMNAME'] != 'ND225')
    changed_visitid = False
    if fits.getheader(bright_star_filelist[0])['VISITID'] ==  fits.getheader(faint_star_filelist[0])['VISITID']:
        changed_visitid = True
        brightdata_visitid = fits.getheader(faint_star_filelist[0])['VISITID']
        for f in bright_star_filelist:
            fits.setval(f, 'VISITID', value = '0200001001001002001', ext=0)
    
    # Changing TARGET if original TARGET not in calspec list
    changed_bright_target = False
    changed_faint_target = False
    try:
        bright_target = fits.getheader(bright_star_filelist[0], ext=0)['TARGET']
        _, _ = get_calspec_file(bright_target)
    except:
        changed_bright_target = True
        for f in bright_star_filelist:
            fits.setval(f, 'TARGET', value = 'eta uma', ext=0)
    try:
        faint_target = fits.getheader(faint_star_filelist[0], ext=0)['TARGET']
        _, _ = get_calspec_file(bright_target)
    except:
        changed_faint_target = True
        for f in faint_star_filelist:
            fits.setval(f, 'TARGET', value = 'tyc 4413-304-1', ext=0)


    # L2b files were saturated for faint star dataset.
    for file in faint_star_filelist:
        with fits.open(file, mode='update') as fits_file:
            fits_file[1].header['EXPTIME'] = fits_file[1].header['EXPTIME'] * 0.75
            fits_file[1].data = fits_file[1].data * 0.75


    # ------------------------------------------------------------------ 
    # 3. L1 -> L2a                                                        
    # ------------------------------------------------------------------ 
    print("Running L1 -> L2a …")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l1_filelist, "", outputdir,
                             template="l1_to_l2a_basic.json")

    l2a_filelist = sorted(
        os.path.join(outputdir, f)
        for f in os.listdir(outputdir) if f.endswith('_l2a.fits')
    )
    print(f"L1 -> L2a complete: {len(l2a_filelist)} L2a files produced.")

    # PC dark (only needed for photon-counted data)
    if is_pc_data and l2a_filelist:
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
            walker.walk_corgidrp(dark_l1_files, "", dark_l2a_dir)
        dark_l2a_files = sorted(
            os.path.join(dark_l2a_dir, f)
            for f in os.listdir(dark_l2a_dir) if f.endswith('_l2a.fits')
        )
        pc_dark = get_pc_mean(data.Dataset(dark_l2a_files), inputmode='darks')
        mocks.rename_files_to_cgi_format([pc_dark], calibrations_dir, "drk_cal")
        this_caldb.create_entry(pc_dark)
        print("PC dark created.")

    # ------------------------------------------------------------------ #
    # 4. L2a -> L2b (spectroscopy recipe)                                 #
    # ------------------------------------------------------------------ #
    print("Running L2a → L2b (spec) …")
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
            print(f"  L2a→L2b: VISTYPE={vt}, ISPC={group_ispc} ({len(group_files)} files)")
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
                    walker.walk_corgidrp(group_files, "", outputdir)
    else:
        # Split by EXPTIME so each group is paired with its matching dark.
        # Dim-star and bright-star frames may use different exposure times.
        l2a_ds = data.Dataset(l2a_filelist, no_data=True, no_err=True, no_dq=True)
        exptime_groups, _ = l2a_ds.split_dataset(exthdr_keywords=['EXPTIME'])
        for group in exptime_groups:
            group_files = [f.filepath for f in group.frames]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                walker.walk_corgidrp(group_files, "", outputdir)

    l2b_filelist = sorted(
        os.path.join(outputdir, f)
        for f in os.listdir(outputdir) if f.endswith('_l2b.fits')
    )
    print(f"L2a -> L2b complete: {len(l2b_filelist)} L2b files produced.")

    # ------------------------------------------------------------------ 
    # 5. L2b -> NDSpectroscopy calibration product                                                 
    # ------------------------------------------------------------------ 
    print("Running L2b -> NDSpectroscopy …")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l2b_filelist, "", outputdir)

    # ------------------------------------------------------------------ 
    # 6. Find and load the NDSpectroscopy product                      
    # ------------------------------------------------------------------ 
    nd_spec_files = sorted(
        os.path.join(outputdir, f)
        for f in os.listdir(outputdir) if f.endswith('_nds_cal.fits')
    )
    if not nd_spec_files:
        raise AssertionError(
            f"No _nds_cal.fits file found in {outputdir}. "
            "Check that the pipeline ran and the walker routing is correct."
        )

    nd_spec_cal = data.NDSpectroscopy(nd_spec_files[0])
    print(f"NDSpectroscopy loaded: {nd_spec_files[0]}")

    # ------------------------------------------------------------------ 
    # 7. Validation                                                       
    # ------------------------------------------------------------------ 
    print("Validating NDSpectroscopy product")

    # "Sweet-spot" dataset should have multiple FSMX and FSMY
    bright_dataset = data.Dataset(bright_star_filelist)
    _, fsm = bright_dataset.split_dataset(exthdr_keywords=["FSMX","FSMY"])
    assert len(fsm) > 1, "Only 1 (FSMX, FSMY) combination in bright star dataset. Sweet-spot dataset with ND filter should be dithered over multiple (FSMX, FSMY) positions."
    print(f"  Sweet-spot dataset is dithered over {len(fsm)} (FSMX, FSMY) positions PASSED")

    # Data shape: (M, N, 4)
    assert nd_spec_cal.data.ndim == 3 and nd_spec_cal.data.shape[2] == 4, \
        f"Unexpected data shape: {nd_spec_cal.data.shape}"
    M = nd_spec_cal.data.shape[0]
    N = nd_spec_cal.data.shape[1]
    print(f"  Data shape: ({M}, {N}, 4)  PASSED")

    # Wavelengths should be monotonically increasing and in the band 3 range
    wave = nd_spec_cal.wavelengths
    for i, wv in enumerate(wave):
        assert np.all(np.diff(wv) > 0), f"Wavelength grid for dither {i+1} is not monotonically increasing."
        assert wv[0] > 500 and wv[-1] < 1100, \
            f"Wavelengths {wv[0]:.1f}–{wv[-1]:.1f} nm for dither {i+1} outside expected range."
        print(f"  Wavelength range: {wv[0]:.1f}–{wv[-1]:.1f} nm for dither {i+1} PASSED")

    # OD values should be positive and finite
    od_all = nd_spec_cal.od_spectra
    for i, od in enumerate(od_all):
        assert np.all(np.isfinite(od)), f"OD spectrum for dither {i+1} contains non-finite values."
        assert np.all(od > 0), f"OD spectrum for dither {i+1} contains non-positive values (min={od.min():.3f})."
        print(f"  OD range: {od.min():.3f}–{od.max():.3f} for dither {i+1}")

    od_expected = 2.20
    od_tolerance = 0.05
    od_median = np.median(od)
    assert abs(od_median - od_expected) < od_tolerance, (
        f"Median OD {od_median:.3f} is more than {od_tolerance} away from "
        f"nominal ND225 value of {od_expected}."
    )
    print(f"  Median OD: {od_median:.3f} (nominal {od_expected}, tol ±{od_tolerance})  PASSED")

    # Headers
    assert nd_spec_cal.ext_hdr['DATATYPE'] == 'NDSpectroscopy'
    assert nd_spec_cal.ext_hdr.get('FPAMNAME', '').startswith('ND'), \
        f"Expected FPAMNAME to start with 'ND', got '{nd_spec_cal.ext_hdr.get('FPAMNAME')}'"
    assert nd_spec_cal.ext_hdr.get('DPAMNAME', '').startswith('PRISM'), \
        f"Expected DPAMNAME to start with 'PRISM', got '{nd_spec_cal.ext_hdr.get('DPAMNAME')}'"
    assert nd_spec_cal.ext_hdr['DATALVL'] == 'CAL'
    assert nd_spec_cal.ext_hdr['FPAM_H'] > 0.0
    assert nd_spec_cal.ext_hdr['FPAM_V'] > 0.0
    print("  Header keywords PASSED")

    # Make a 5x5 mock 'clean_spec_image' with the wavelength zeropoint at (2,2)
    # Shift it by (3,3) => final location (5,5).
    spec_wave = np.linspace(np.max(nd_spec_cal.wavelengths[:,0]), np.min(nd_spec_cal.wavelengths[:,-1]), num=nd_spec_cal.wavelengths.shape[1], dtype=float)
    spec_values = np.ones_like(spec_wave, dtype=float)
    spec_err = 0.1*np.ones_like(spec_wave, dtype=float)
    
    clean_spec_image = mocks.make_1d_spec_image(spec_values, spec_err, spec_wave, pa_aper_deg=0, exp_time=30)
    # Choosing some values that will help predict the expected value of the OD
    # when using the bilinear OD interpolation in nd_filter_calibration.interpolate_od()
    # These values ensure that the shift in EXCAM pixels is (3,3)
    clean_spec_image.ext_hdr["FPAM_H"] = -24.42 + nd_spec_cal.ext_hdr["FPAM_H"]
    clean_spec_image.ext_hdr["FPAM_V"] = 24.42 + nd_spec_cal.ext_hdr["FPAM_V"]
    clean_spec_image.ext_hdr["WV0_X"] = 509.
    clean_spec_image.ext_hdr["WV0_Y"] = 514.

    # Default FPAM/FSAM transformations (use mock instead of loading from file which
    # seems to be inconsistent)
    fpamfsamcal = mocks.create_mock_fpamfsam_cal(save_file=False)    

    # Call the function under test
    interpolated_od = nd_filter_calibration.calculate_od_spec_at_new_location(
        clean_spec_image=clean_spec_image,
        fpamfsamcal=fpamfsamcal,
        ndspectroscopy_dataset=nd_spec_cal)

    # Expect the final location = (512+3, 516+3) = (515,519).
    fpam2excam_matrix = fits.getdata(os.path.join(os.path.dirname(__file__), '../test_data',
        'fpam_to_excam_modelbased.fits'))
    # Check final position is (5,5)
    final_excam_pos = (np.array([clean_spec_image.ext_hdr["WV0_X"],clean_spec_image.ext_hdr["WV0_Y"]]) + fpam2excam_matrix @
        np.array([-24.42,24.42]))
    # Single precision because the FPAM_H/V values were set to be close to
    # produce a change of 3 EXCAM pixels within single precision
    assert np.all(np.abs(final_excam_pos - np.array([512,517])) < 1e-7), \
    f"Final EXCAM position should be (512.,517.), but is ({final_excam_pos[0]:.1f},{final_excam_pos[1]:.1f})."
    print("Final EXCAM position is (512.,517.) as expected PASSED")

    expected_value = 2.17
    atol_nd = 0.05
    for i, wave in enumerate(spec_wave):
        test_result_od_accuracy = abs(interpolated_od[i] - expected_value) < atol_nd

        assert test_result_od_accuracy, (
            f"Expected OD={expected_value} +/- {atol_nd}, got {interpolated_od[i]:.3f} at wavelength {wave:.3f} nm"
        )
        print('')
        print(
            f"test_calculate_od_spec_at_new_location PASSED: "
            f"estimated OD = {interpolated_od[i]:.3f} at wavelength {wave:.3f} nm, expected OD = {expected_value} +/- {atol_nd}"
        )

    # Remove temporary CalDB
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_nd_spec_e2e_caldb.csv')
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)

    print("NDSpectroscopy E2E test PASSED.")
    return nd_spec_cal


# ---------------------------------------------------------------------------
# Pytest entry
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_nd_filter_spec_e2e(e2edata_path, e2eoutput_path):
    """
    Pytest wrapper for the spectroscopy ND filter calibration E2E test.

    Args:
        e2edata_path (str): Path to the E2E test data
        e2eoutput_path (str): Path to the E2E test output

    """
    l1_datadir        = os.path.join(e2edata_path, "ND_SPEC", "SPEC_NOM_L1")
    processed_cal_path = os.path.join(e2edata_path, "ND_SPEC", "Cals")
    outputdir = os.path.join(e2eoutput_path, "nd_filter_spec_e2e")

    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    run_nd_filter_spec_e2e(l1_datadir, processed_cal_path, outputdir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Spectroscopy ND filter calibration E2E test (L1 -> NDSpectroscopy)"
    )
    ap.add_argument(
        "-d", "--e2edata_dir",
        default="/home/ababuraj/roman/E2E_Test_Data",
        help="Root directory containing ND_SPEC/L1/ and ND_SPEC/Cals/ sub-folders"
    )
    ap.add_argument(
        "-o", "--outputdir",
        default=thisfile_dir,
        help="Directory to write all output products"
    )
    args = ap.parse_args()

    l1_datadir         = os.path.join(args.e2edata_dir, "ND_SPEC", "SPEC_NOM_L1")
    processed_cal_path = os.path.join(args.e2edata_dir, "ND_SPEC", "Cals")
    outputdir = os.path.join(args.outputdir, "l1_to_nd_filter_spec_e2e")

    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    run_nd_filter_spec_e2e(l1_datadir, processed_cal_path, outputdir)
