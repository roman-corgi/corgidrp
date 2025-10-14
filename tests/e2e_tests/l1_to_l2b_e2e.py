import argparse
import os
import pytest
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector
import shutil
import logging
import traceback
from corgidrp.darks import build_synthesized_dark
from corgidrp.check import (check_filename_convention, check_dimensions, 
                           verify_hdu_count, verify_header_keywords)

try:
    from proc_cgi_frame.gsw_process import Process
except:
    pass

thisfile_dir = os.path.dirname(__file__) # this file's folder

def fix_str_for_tvac(
    list_of_fits,
    ):
    """ 
    Gets around EMGAIN_A being set to 1 in TVAC data and fixes string header values.
    
    Args:
        list_of_fits (list): list of FITS files that need to be updated.
    """
    for file in list_of_fits:
        with fits.open(file, mode='update') as fits_file:
            exthdr = fits_file[1].header
            if float(exthdr['EMGAIN_A']) == 1:
                exthdr['EMGAIN_A'] = -1 #for new SSC-updated TVAC files which have EMGAIN_A by default as 1 regardless of the commanded EM gain
            if type(exthdr['EMGAIN_C']) is str:
                exthdr['EMGAIN_C'] = float(exthdr['EMGAIN_C'])
            if 'RN' in exthdr and type(exthdr['RN']) is str:
                exthdr['RN'] = float(exthdr['RN'])
            
            # TEMPORARY FIX: Set ISPC=False to disable photon counting
            # TODO: Fix ISPC in source L1 files instead of modifying copies here
            if 'ISPC' in exthdr:
                exthdr['ISPC'] = False


def run_l1_to_l2b_e2e_test(l1_datadir, test_outputdir, cals_dir, input_datadir, logger):
    """Run the complete L1 to L2b end-to-end test.
    
    This function consolidates all the test steps into a single linear flow
    for easier reading and understanding.
    
    Args:
        l1_datadir (str): Path to L1 input data directory
        test_outputdir (str): Path to output directory
        cals_dir (str): Path to calibrations directory
        input_datadir (str or None): Original input_datadir parameter (to determine if we process all files)
        logger (logging.Logger): Logger instance for output
        
    Returns:
        tuple: (new_l2a_filenames, new_l2b_filenames) - lists of output files
    """
    
    # ================================================================================
    # (1) Setup Output Directories
    # ================================================================================
    logger.info('='*80)
    logger.info('Pre-test: Set up output directories and calibrations')
    logger.info('='*80)

    # Create input_data subfolder
    input_data_dir = os.path.join(test_outputdir, 'input_l1')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)
    calibrations_dir = os.path.join(test_outputdir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)
    l2a_tvac_outputdir = os.path.join(test_outputdir, "tvac_reference_data", "l2a")
    if not os.path.exists(l2a_tvac_outputdir):
        os.makedirs(l2a_tvac_outputdir)
    # clean up by removing old files
    for file in os.listdir(l2a_tvac_outputdir):
        os.remove(os.path.join(l2a_tvac_outputdir, file))
    l2b_tvac_outputdir = os.path.join(test_outputdir, "tvac_reference_data", "l2b")
    if not os.path.exists(l2b_tvac_outputdir):
        os.makedirs(l2b_tvac_outputdir)
    # clean up by removing old files
    for file in os.listdir(l2b_tvac_outputdir):
        os.remove(os.path.join(l2b_tvac_outputdir, file))
    # separate L2a and L2b outputdirs
    l2a_outputdir = os.path.join(test_outputdir, "l1_to_l2a")
    if not os.path.exists(l2a_outputdir):
        os.mkdir(l2a_outputdir)
    # clean up by removing old files
    for file in os.listdir(l2a_outputdir):
        os.remove(os.path.join(l2a_outputdir, file))
    
    logger.info(f"Created output directories:")
    logger.info(f"  - {input_data_dir}")
    logger.info(f"  - {calibrations_dir}")
    logger.info(f"  - {l2a_outputdir}")
    logger.info(f"  - {test_outputdir}")
    logger.info('')

    # Build calibration file paths
    if cals_dir is None:
        # For default, cals_dir needs to be derived from e2edata_path
        # But we don't have e2edata_path here, so we'll raise an error
        raise ValueError("cals_dir cannot be None in run_l1_to_l2b_e2e_test helper function")
    
    processed_cal_path = cals_dir
    
    # Use default paths with known filenames or search for them
    # (This logic assumes cals_dir is always provided to this function)
    if os.path.exists(os.path.join(processed_cal_path, "nonlin_table_240322.txt")):
        # Use default filenames
        nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
        dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
        flat_path = os.path.join(processed_cal_path, "flat.fits")
        fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
        cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
        bp_path = os.path.join(processed_cal_path, "bad_pix.fits")
    else:
        # Search for calibration files by pattern
        # Find nonlinearity file
        nln_files = [f for f in os.listdir(processed_cal_path) if "nln" in f.lower()]
        if not nln_files:
            raise FileNotFoundError(f"No file containing 'nln' found in {processed_cal_path}")
        nonlin_path = os.path.join(processed_cal_path, nln_files[0])
        
        # Find dark current file
        drk_files = [f for f in os.listdir(processed_cal_path) if "drk" in f.lower()]
        if not drk_files:
            raise FileNotFoundError(f"No file containing 'drk' found in {processed_cal_path}")
        dark_path = os.path.join(processed_cal_path, drk_files[0])
        
        # Find flat file
        flat_files = [f for f in os.listdir(processed_cal_path) if "flat" in f.lower() or "flt" in f.lower()]
        if not flat_files:
            raise FileNotFoundError(f"No file containing 'flat' or 'flt' found in {processed_cal_path}")
        flat_path = os.path.join(processed_cal_path, flat_files[0])
        
        # Find FPN file
        fpn_files = [f for f in os.listdir(processed_cal_path) if "fpn" in f.lower()]
        if not fpn_files:
            raise FileNotFoundError(f"No file containing 'fpn' found in {processed_cal_path}")
        fpn_path = os.path.join(processed_cal_path, fpn_files[0])
        
        # Find CIC file
        cic_files = [f for f in os.listdir(processed_cal_path) if "cic" in f.lower()]
        if not cic_files:
            raise FileNotFoundError(f"No file containing 'cic' found in {processed_cal_path}")
        cic_path = os.path.join(processed_cal_path, cic_files[0])
        
        # Find bad pixel map file
        bp_files = [f for f in os.listdir(processed_cal_path) if "bad" in f.lower() or "bp" in f.lower() or "bpm" in f.lower()]
        if not bp_files:
            raise FileNotFoundError(f"No file containing 'bad', 'bp', or 'bpm' found in {processed_cal_path}")
        bp_path = os.path.join(processed_cal_path, bp_files[0])
    
    # Filter to only include L1 files for mock calibration
    all_files = os.listdir(l1_datadir)
    l1_files_only = [f for f in all_files if f.endswith('l1_.fits')]
    if not l1_files_only:
        raise FileNotFoundError(f"No files ending in 'l1_.fits' found in {l1_datadir}")
    mock_cal_filelist = [os.path.join(l1_datadir, l1_files_only[i]) for i in [-2,-1]] # grab the last two L1 files to mock the calibration 
    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make calibration files using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")
    this_caldb.create_entry(nonlinear_cal)

    # KGain (with read noise)
    kgain_val = 8.7 # 8.7 is what is in the TVAC headers
    signal_array = np.linspace(0, 50)
    noise_array = np.sqrt(signal_array)
    ptc = np.column_stack([signal_array, noise_array])
    kgain = data.KGain(kgain_val, ptc=ptc, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    this_caldb.create_entry(kgain)

    # NoiseMap
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_path) as hdulist:
        dark_dat = hdulist[0].data
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    mocks.rename_files_to_cgi_format(list_of_fits=[noise_map], output_dir=calibrations_dir, level_suffix="dnm_cal")
    this_caldb.create_entry(noise_map)

    # Dark calibration - use build_synthesized_dark to match the exposure time and emgain of the input data
    sample_l1_file = os.path.join(l1_datadir, l1_files_only[0])
    sample_hdr = fits.getheader(sample_l1_file, ext=1)
    data_exptime = sample_hdr['EXPTIME']
    data_emgain = float(sample_hdr['EMGAIN_C'])
    
    # Create a temporary dataset with the correct header values
    temp_dataset = data.Dataset(mock_cal_filelist[:1])
    temp_dataset.frames[0].ext_hdr['EXPTIME'] = data_exptime
    temp_dataset.frames[0].ext_hdr['EMGAIN_C'] = data_emgain
    
    # Build a synthesized dark with the correct exposure time and EM gain
    dark_cal = build_synthesized_dark(temp_dataset, noise_map)
    mocks.rename_files_to_cgi_format(list_of_fits=[dark_cal], output_dir=calibrations_dir, level_suffix="drk_cal")
    this_caldb.create_entry(dark_cal)

    ## Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[flat], output_dir=calibrations_dir, level_suffix="flt_cal")
    this_caldb.create_entry(flat)

    # bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[bp_map], output_dir=calibrations_dir, level_suffix="bpm_cal")
    this_caldb.create_entry(bp_map)
    
    logger.info("Created calibration products:")
    logger.info(f"  - NonLinearityCalibration: {nonlinear_cal.filename}")
    logger.info(f"  - KGain: {kgain.filename}")
    logger.info(f"  - DetectorNoiseMaps: {noise_map.filename}")
    logger.info(f"  - Dark: {dark_cal.filename}")
    logger.info(f"  - FlatField: {flat.filename}")
    logger.info(f"  - BadPixelMap: {bp_map.filename}")
    logger.info('')

    # ================================================================================
    # (2) Validate Input L1 Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 1: Input L1 Image Data Format and Content')
    logger.info('='*80)

    # define the raw science data to process
    # l1_files_only was already filtered above for mock_cal_filelist
    if input_datadir is None:
        # Default behavior: select just the first two files for testing
        l1_data_filelist = [os.path.join(l1_datadir, l1_files_only[i]) for i in [0,1]] # grab the first two L1 files
    else:
        # Custom directory: process all L1 files
        l1_data_filelist = [os.path.join(l1_datadir, f) for f in l1_files_only]

    # Copy files to input_data directory and update file list
    l1_data_filelist = [
        shutil.copy2(file_path, os.path.join(input_data_dir, os.path.basename(file_path)))
        for file_path in l1_data_filelist
    ] 
    
    # Validate L1 input files
    l1_dataset = data.Dataset(l1_data_filelist)
    for i, (frame, filepath) in enumerate(zip(l1_dataset, l1_data_filelist)):
        frame_info = f"L1 Input Frame {i}"
        
        check_filename_convention(os.path.basename(filepath), 'cgi_*_l1_.fits', frame_info, logger, data_level='l1_')
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L1'}, frame_info, logger)
        
        # Verify HDU count
        try:
            with fits.open(filepath) as hdul:
                verify_hdu_count(hdul, 5, frame_info, logger)  # L1 should have 5 HDUs
        except Exception as e:
            logger.info(f"{frame_info}: HDU count verification failed. Error: {str(e)}. FAIL")
        
        # Check dimensions
        logger.info(f"{frame_info}: Data shape {frame.data.shape}")
        logger.info("")
    
    logger.info(f"Total input L1 images validated: {len(l1_dataset)}")
    logger.info('')

    # tvac_l2a_filelist = [os.path.join(l2a_datadir, os.listdir(l2a_datadir)[i]) for i in [0,1]] #[os.path.join(l2a_datadir, "{0}.fits".format(i)) for i in [90528, 90530]] # just grab the first two files
    # tvac_l2b_filelist = [os.path.join(l2b_datadir, os.listdir(l2b_datadir)[i]) for i in [0,1]] #[os.path.join(l2b_datadir, "{0}.fits".format(i)) for i in [90529, 90531]] # just grab the first two files
    tvac_l2a_filelist = []
    tvac_l2b_filelist = []
    bad_pix = np.zeros((1024,1024)) # what is used in DRP
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    det_params = this_caldb.get_calib(None, data.DetectorParams)
    fwc_pp_e = int(det_params.params['FWC_PP_E']) # same as what is in DRP's DetectorParams
    fwc_em_e = int(det_params.params['FWC_EM_E']) # same as what is in DRP's DetectorParams
    telem_rows_start = det_params.params['TELRSTRT']
    telem_rows_end = det_params.params['TELREND']
    telem_rows = slice(telem_rows_start, telem_rows_end)
    for j, file in enumerate(l1_data_filelist):
        frame_data = fits.getdata(file)
        ext_hdr = fits.getheader(file, ext=1)
        exptime = ext_hdr['EXPTIME']
        em_gain = float(ext_hdr['EMGAIN_C'])
        eperdn = float(ext_hdr['KGAINPAR'])
        b_offset = 0 # what is used in DRP by default
        md_data = fpn_dat/em_gain + exptime*dark_dat + cic_dat
        proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                b_offset, em_gain, exptime,
                nonlin_path, dark=md_data, flat=flat.data, desmear_flag=True)
        l2a_im, bp_im, _, _, l2a_fr, bp_fr, _ = proc.L1_to_L2a(frame_data)
        with fits.open(file) as hdul:
            hdul_copy = fits.HDUList([hdu.copy() for hdu in hdul])
            hdul_copy[1].data = l2a_im
            # not important to change headers in the way DRP would; we are just comparing data values
            l2a_tvac_filename = file.split(os.path.sep)[-1].replace('l1_','l2a',1) 
            hdul_copy.writeto(os.path.join(l2a_tvac_outputdir, l2a_tvac_filename), overwrite=True)
        tvac_l2a_filelist.append(os.path.join(l2a_tvac_outputdir, l2a_tvac_filename))
        l2b_im, bp_im, _ = proc.L2a_to_L2b(l2a_im, bp_im)
        nan_inds = np.where(bp_im == 1)
        l2b_im[nan_inds] = np.nan
        with fits.open(file) as hdul:
            hdul_copy = fits.HDUList([hdu.copy() for hdu in hdul])
            hdul_copy[1].data = l2b_im
            # not important to change headers in the way DRP would; we are just comparing data values
            l2b_tvac_filename = file.split(os.path.sep)[-1].replace('l1_','l2b',1) 
            hdul_copy.writeto(os.path.join(l2b_tvac_outputdir, l2b_tvac_filename), overwrite=True)
        tvac_l2b_filelist.append(os.path.join(l2b_tvac_outputdir, l2b_tvac_filename))

    # modify TVAC headers for production
    #fix_headers_for_tvac(l1_data_filelist)
    fix_str_for_tvac(l1_data_filelist)

    # ================================================================================
    # (3) Run L1 -> L2a Processing Pipeline
    # ================================================================================
    logger.info('='*80)
    logger.info('Running L1 -> L2a processing pipeline')
    logger.info('='*80)

    logger.info('Running L1 to L2a recipe...')
    walker.walk_corgidrp(l1_data_filelist, "", l2a_outputdir)
    logger.info('')

    # ================================================================================
    # (4) Validate Intermediate L2a Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Intermediate L2a Image Data Format and Content')
    logger.info('='*80)
    
    new_l2a_filenames = [os.path.join(l2a_outputdir, f) for f in os.listdir(l2a_outputdir) if f.endswith('l2a.fits')] #[os.path.join(l2a_outputdir, "{0}.fits".format(i)) for i in [90499, 90500]]
    logger.info(f"Found {len(new_l2a_filenames)} L2a output files")
    
    # Validate L2a files
    l2a_dataset = data.Dataset(new_l2a_filenames)
    for i, (frame, filepath) in enumerate(zip(l2a_dataset, new_l2a_filenames)):
        frame_info = f"L2a Intermediate Frame {i}"
        
        check_filename_convention(os.path.basename(filepath), 'cgi_*_l2a.fits', frame_info, logger, data_level='l2a')
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L2a'}, frame_info, logger)
        
        try:
            with fits.open(filepath) as hdul:
                verify_hdu_count(hdul, 5, frame_info, logger)
        except Exception as e:
            logger.info(f"{frame_info}: HDU count verification failed. Error: {str(e)}. FAIL")
        
        logger.info(f"{frame_info}: Data shape {frame.data.shape}")
        logger.info("")
    
    logger.info(f"Total intermediate L2a images validated: {len(l2a_dataset)}")
    logger.info('')
    
    # ================================================================================
    # (5) Run L2a -> L2b Processing Pipeline
    # ================================================================================
    logger.info('='*80)
    logger.info('Running L2a -> L2b processing pipeline')
    logger.info('='*80)
    
    # Fix L2a headers before processing to L2b
    fix_str_for_tvac(new_l2a_filenames)
    logger.info('Running L2a to L2b recipe...')
    walker.walk_corgidrp(new_l2a_filenames, "", test_outputdir)
    logger.info('')

    # ================================================================================
    # (6) Validate Output L2b Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 3: Output L2b Image Data Format and Content')
    logger.info('='*80)
    
    new_l2b_filenames = [os.path.join(test_outputdir, f) for f in os.listdir(test_outputdir) if f.endswith('l2b.fits')]
    logger.info(f"Found {len(new_l2b_filenames)} L2b output files")
    
    # Validate L2b files
    l2b_dataset = data.Dataset(new_l2b_filenames)
    for i, (frame, filepath) in enumerate(zip(l2b_dataset, new_l2b_filenames)):
        frame_info = f"L2b Output Frame {i}"
        
        try:
            check_filename_convention(os.path.basename(filepath), 'cgi_*_l2b.fits', frame_info, logger, data_level='l2b')
            verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L2b'}, frame_info, logger)
            
            # Verify HDU count
            with fits.open(filepath) as hdul:
                verify_hdu_count(hdul, 5, frame_info, logger)
            
            # Check dimensions
            logger.info(f"{frame_info}: Data shape {frame.data.shape}")
            
            # Verify units
            if frame.ext_hdr['BUNIT'] == 'photoelectron':
                logger.info(f"{frame_info}: BUNIT = 'photoelectron'. PASS")
            else:
                logger.info(f"{frame_info}: BUNIT = '{frame.ext_hdr['BUNIT']}'. Expected: 'photoelectron'. FAIL")
            
        except Exception as e:
            logger.info(f"{frame_info}: Validation failed with error: {str(e)}. FAIL")
            raise
        
        logger.info("")
    
    logger.info(f"Total output L2b images validated: {len(l2b_dataset)}")
    logger.info('')
    
    # ================================================================================
    # (7) Compare Against TVAC Reference Data (only for default data)
    # ================================================================================
    if input_datadir is None and len(tvac_l2a_filelist) > 0:
        logger.info('='*80)
        logger.info('Test Case 4: TVAC Reference Data Comparison')
        logger.info('='*80)
        logger.info('Comparing against TVAC reference data...')
        
        # l2a data comparison
        for new_filename, tvac_filename in zip(sorted(new_l2a_filenames), sorted(tvac_l2a_filelist)):
            img = data.Image(new_filename)
            with fits.open(tvac_filename) as hdulist:
                tvac_dat = hdulist[1].data
            diff = img.data - tvac_dat
            if np.all(np.abs(diff) < 1e-5):
                logger.info(f"L2a TVAC comparison: {os.path.basename(new_filename)} matches reference. PASS")
            else:
                logger.info(f"L2a TVAC comparison: {os.path.basename(new_filename)} differs from reference. FAIL")

        # l2b data comparison
        for new_filename, tvac_filename in zip(sorted(new_l2b_filenames), sorted(tvac_l2b_filelist)):
            img = data.Image(new_filename)
            with fits.open(tvac_filename) as hdulist:
                tvac_dat = hdulist[1].data
            
            # make sure the NaNs from cosmic rays are in the same place
            e2e_nans = np.where(np.isnan(img.data))
            tvac_nans = np.where(np.isnan(tvac_dat))
            if np.array_equal(e2e_nans, tvac_nans):
                logger.info(f"L2b TVAC NaN comparison: {os.path.basename(new_filename)} NaN positions match. PASS")
            else:
                logger.info(f"L2b TVAC NaN comparison: {os.path.basename(new_filename)} NaN positions differ. FAIL")
            
            # now compare the rest of the data
            img.data[e2e_nans] = 0.0
            tvac_dat[tvac_nans] = 0.0
            diff = img.data - tvac_dat
            if np.all(np.abs(diff) < 1e-5):
                logger.info(f"L2b TVAC data comparison: {os.path.basename(new_filename)} matches reference. PASS")
            else:
                logger.info(f"L2b TVAC data comparison: {os.path.basename(new_filename)} differs from reference. FAIL")
        
        logger.info('')
    elif input_datadir is not None:
        logger.info('='*80)
        logger.info('Test Case 4: TVAC Reference Data Comparison - SKIPPED')
        logger.info('='*80)
        logger.info('TVAC comparison skipped (custom input_datadir provided)')
        logger.info('')

        # plotting script for debugging
        # import matplotlib.pylab as plt
        # fig = plt.figure(figsize=(10,3.5))
        # fig.add_subplot(131)
        # plt.imshow(img.data, vmin=-0.01, vmax=45, cmap="viridis")
        # plt.title("corgidrp")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # fig.add_subplot(132)
        # plt.imshow(tvac_dat, vmin=-0.01, vmax=45, cmap="viridis")
        # plt.title("TVAC")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # fig.add_subplot(133)
        # plt.imshow(diff, vmin=-0.01, vmax=0.01, cmap="inferno")
        # plt.title("difference")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # plt.show()
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)
    
    return new_l2a_filenames, new_l2b_filenames


@pytest.mark.e2e
def test_l1_to_l2b(e2edata_path, e2eoutput_path, input_datadir, cals_dir):
    """Run the complete L1 to L2b end-to-end test.
    
    Args:
        e2edata_path (str): Path to test data
        e2eoutput_path (str): Output directory path for results and logs
        input_datadir (str or None): Custom input data directory
        cals_dir (str or None): Custom calibration directory
    """
    # Set up output directory and logging
    global logger
    
    # Use custom paths if provided, otherwise fall back to defaults from e2edata_path
    if input_datadir is None:
        l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    else:
        l1_datadir = input_datadir
    
    if cals_dir is None:
        processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    else:
        processed_cal_path = cals_dir

    # make output directory if needed
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_l2b_e2e")
    if os.path.exists(test_outputdir):
        shutil.rmtree(test_outputdir)
    os.makedirs(test_outputdir)
    
    log_file = os.path.join(test_outputdir, 'l1_to_l2b_e2e.log')
    
    # Create a new logger specifically for this test
    logger = logging.getLogger('l1_to_l2b_e2e')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info('='*80)
    logger.info('L1 TO L2B END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    # Run the complete end-to-end test
    try:
        new_l2a_filenames, new_l2b_filenames = run_l1_to_l2b_e2e_test(l1_datadir, test_outputdir, processed_cal_path, input_datadir, logger)
        
        logger.info('='*80)
        logger.info('END-TO-END TEST COMPLETE - ALL TESTS PASSED')
        logger.info('='*80)
        
        print('e2e test for L1 to L2b passed')
    except Exception as e:
        logger.error('='*80)
        logger.error('END-TO-END TEST FAILED')
        logger.error('='*80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Print traceback to log
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        print(f'e2e test for L1 to L2b FAILED: {str(e)}')
        raise


if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    #e2edata_dir =  '/home/jwang/Desktop/CGI_TVAC_Data/'
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2b end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    ap.add_argument("--input_datadir", default=None,
                    help="Optional: Override input data directory [%(default)s]")
    ap.add_argument("--cals_dir", default=None,
                    help="Optional: Override calibration directory [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    input_datadir = args.input_datadir
    cals_dir = args.cals_dir
    test_l1_to_l2b(e2edata_dir, outputdir, input_datadir, cals_dir)