#This tests combines l1_to_l3_pol_e2e.py with l3_to_l4_pol_e2e.py to test the entire flow from l1 to l4 with policy in place.
#The input data is all noisy data, but with the correct headers (I think). The test is just a function tests, and probably shouldn't overwrite 
#the need for l3_to_l4_pol_e2e.py. 

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
import corgidrp.check as check
from corgidrp import corethroughput
import shutil
import logging
import traceback
from corgidrp.check import (check_filename_convention,
                           verify_hdu_count, verify_header_keywords,
                           )
import warnings

thisfile_dir = os.path.dirname(__file__) # this file's folder

def run_l1_to_l4_e2e_test(l1_datadir, l4_outputdir, processed_cal_path, logger):
    """Run the complete L1 to L3 polarimetry data end-to-end test.
    
    Args:
        l1_datadir (str): Path to L1 input data directory
        l4_outputdir (str): Path to output directory
        processed_cal_path (str): Path to processed calibration files directory
        logger (logging.Logger): Logger instance for output
        
    Returns:
        list: List of L3 output filenames
    """
    
    # ================================================================================
    # (1) Setup Calibrations
    # ================================================================================
    logger.info('='*80)
    logger.info('Pre-test: Set up calibration files')
    logger.info('='*80)

    # Create calibrations subfolder
    calibrations_dir = os.path.join(l4_outputdir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)

    # Initialize a connection to the calibration database
    # Use a test-specific caldb filepath
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_l1_to_l4_pol_e2e_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()
    
    # Scan default calibration directory 
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    logger.info(f"Loaded default calibrations from {corgidrp.default_cal_dir}")

    # Calibration file paths
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")
    
    # Create mock headers for calibration products
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] = corgidrp.__version__
    
    # Get some L1 files to mock the input_dataset for calibrations
    all_files = os.listdir(l1_datadir)
    l1_files = [f for f in all_files if f.endswith('l1_.fits')]
    if len(l1_files) >= 2:
        mock_cal_filelist = [os.path.join(l1_datadir, l1_files[i]) for i in [-2, -1]]
    else:
        mock_cal_filelist = [os.path.join(l1_datadir, f) for f in l1_files]
    # Copy and fix mock cal headers
    mock_cal_dir = os.path.join(l4_outputdir, 'mock_cal_input')
    os.makedirs(mock_cal_dir, exist_ok=True)
    mock_cal_filelist = [
        shutil.copy2(f, os.path.join(mock_cal_dir, os.path.basename(f)))
        for f in mock_cal_filelist
    ]
    mock_cal_filelist = check.fix_hdrs_for_tvac(mock_cal_filelist, mock_cal_dir)

    mock_input_dataset = data.Dataset(mock_cal_filelist)

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")
    this_caldb.create_entry(nonlinear_cal)

    # KGain (with read noise)
    kgain_val = 8.7  # Standard value from TVAC headers
    signal_array = np.linspace(0, 50)
    noise_array = np.sqrt(signal_array)
    ptc = np.column_stack([signal_array, noise_array])
    kgain = data.KGain(kgain_val, ptc=ptc, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                      input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    this_caldb.create_entry(kgain)

    # NoiseMap (FPN + CIC + Dark)
    import corgidrp.detector as detector
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
    ext_hdr['B_O'] = 0.
    ext_hdr['B_O_ERR'] = 0.
    noisemap_ext_hdr = ext_hdr.copy()
    noisemap_ext_hdr['DRPNFILE'] = 100
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=noisemap_ext_hdr,
                                       input_dataset=mock_input_dataset, err=noise_map_noise,
                                       dq=noise_map_dq, err_hdr=err_hdr)
    mocks.rename_files_to_cgi_format(list_of_fits=[noise_map], output_dir=calibrations_dir, level_suffix="dnm_cal")
    this_caldb.create_entry(noise_map)

    # Dark calibration - build synthesized dark matching input data
    from corgidrp.darks import build_synthesized_dark
    # Get exposure time and emgain from a sample L1 file
    sample_l1_files = [f for f in os.listdir(l1_datadir) if f.endswith('l1_.fits')]
    if sample_l1_files:
        sample_l1_file = os.path.join(l1_datadir, sample_l1_files[0])
        sample_hdr = fits.getheader(sample_l1_file, ext=1)
        data_exptime = sample_hdr['EXPTIME']
        data_emgain = float(sample_hdr['EMGAIN_C'])
        
        # Create temp dataset with correct header values
        temp_dataset = data.Dataset(mock_cal_filelist[:1])
        temp_dataset.frames[0].ext_hdr['EXPTIME'] = data_exptime
        temp_dataset.frames[0].ext_hdr['EMGAIN_C'] = data_emgain
        
        dark_cal = build_synthesized_dark(temp_dataset, noise_map)
        mocks.rename_files_to_cgi_format(list_of_fits=[dark_cal], output_dir=calibrations_dir, level_suffix="drk_cal")
        this_caldb.create_entry(dark_cal)

    # Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    # Pol flats are taken with an open FP mask (OPEN_12 for band 1)
    # Create separate input datasets for each pol flat with the correct DPAMNAME and FPAMNAME
    mock_dataset_flat0 = data.Dataset([frame.copy() for frame in mock_input_dataset])
    for frame in mock_dataset_flat0:
        frame.ext_hdr['DPAMNAME'] = 'POL0'
        frame.ext_hdr['FPAMNAME'] = 'OPEN_12'
    pri_hdr_flat45 = pri_hdr.copy()
    pri_hdr_visitID = str(int(pri_hdr['VISITID']) + 1)  #Change the visit id so the filename will be different
    pri_hdr_flat45['VISITID'] = pri_hdr_visitID
    mock_dataset_flat45 = data.Dataset([frame.copy() for frame in mock_input_dataset])
    for frame in mock_dataset_flat45:
        frame.ext_hdr['DPAMNAME'] = 'POL45'
        frame.ext_hdr['FPAMNAME'] = 'OPEN_12'
    flat0 = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_dataset_flat0)
    flat45 = data.FlatField(flat_dat, pri_hdr=pri_hdr_flat45, ext_hdr=ext_hdr, input_dataset=mock_dataset_flat45)
    mocks.rename_files_to_cgi_format(list_of_fits=[flat0, flat45], output_dir=calibrations_dir, level_suffix="flt_cal")
    this_caldb.create_entry(flat0)
    this_caldb.create_entry(flat45)


    # Bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    # Make sure BPM includes a dark(-like) frame
    bp_map_inputs = data.Dataset([dark_cal, flat0])
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=bp_map_inputs)
    mocks.rename_files_to_cgi_format(list_of_fits=[bp_map], output_dir=calibrations_dir, level_suffix="bpm_cal")
    this_caldb.create_entry(bp_map)

    ########################################
    ########### Create astrom cal ##########
    ########################################

    # CREATE DISTORTED POL DATASET
    # Create a bad pixel map for testing
    datashape = (2, 1024, 1024)  # pol data shape
    bpixmap = np.zeros(datashape)

    field_path = os.path.join(os.path.dirname(__file__), "..","test_data", "JWST_CALFIELD2020.csv")

    distortion_coeffs_path = os.path.join(os.path.dirname(__file__), "..","test_data", "distortion_expected_coeffs.csv")
    distortion_coeffs = np.genfromtxt(distortion_coeffs_path)
    distortion_dataset_base = mocks.create_astrom_data(field_path, distortion_coeffs_path=distortion_coeffs_path,
                                                       bpix_map=bpixmap[0], sim_err_map=True)


    astromcal_data = np.concatenate((np.array([80.553428801, -69.514096821, 21.8, 45, 0, 0]), distortion_coeffs),
                                    axis=0)
    astrom_cal = data.AstrometricCalibration(astromcal_data,
                                             pri_hdr=distortion_dataset_base[0].pri_hdr,
                                             ext_hdr=distortion_dataset_base[0].ext_hdr,
                                             input_dataset=distortion_dataset_base)
    mocks.rename_files_to_cgi_format(list_of_fits=[astrom_cal], output_dir=calibrations_dir, level_suffix="ast_cal")
    
    
    this_caldb.create_entry(astrom_cal)

    ###########################
    #### Make dummy CT cal ####
    ###########################

    # Dataset with some CT profile defined in create_ct_interp
    # Pupil image
    pupil_image = np.zeros([1024, 1024])
    # Set it to some known value for a selected range of pixels
    pupil_image[510:530, 510:530]=1
    prhd, exthd_pupil, errhdr, dqhdr = mocks.create_default_L3_headers()
    # DRP
    # cfam filter
    exthd_pupil['CFAMNAME'] = '1F'
    # Add specific values for pupil images:
    # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'

    data_psf, psf_loc_in, half_psf = mocks.create_ct_psfs(50, cfam_name='1F',
    n_psfs=100)
    ct_dataset0 = data_psf[0]
    ct_dataset0.ext_hdr['FPAMNAME'] = 'HLC12_C2R1'  # set FPM to a coronagraphic one
    
    err = np.ones([1024,1024])
    data_ct_interp = [ct_dataset0, data.Image(pupil_image,pri_hdr = prhd,
        ext_hdr = exthd_pupil, err = err)]
    # Set of off-axis PSFs with a CT profile defined in create_ct_interp
    # First, we need the CT FPM center to create the CT radial profile
    # We can use a miminal dataset to get to know it
    ct_cal_tmp = corethroughput.generate_ct_cal(data.Dataset(data_ct_interp))
    mocks.rename_files_to_cgi_format(list_of_fits=[ct_cal_tmp], output_dir=calibrations_dir, level_suffix="ctm_cal")
    this_caldb.create_entry(ct_cal_tmp)

    ##########################################
    #### Generate a flux calibration file ####
    ##########################################

    #Create a mock flux calibration file
    fluxcal_factor = 2e-12
    fluxcal_factor_error = 1e-14
    prhd, exthd, errhd, dqhd = mocks.create_default_L3_headers()
    # Set consistent header values for flux calibration factor
    exthd['CFAMNAME'] = '1F'
    exthd['DPAMNAME'] = 'POL0'
    exthd['FPAMNAME'] = 'HLC12_C2R1'
    exthd['DRPNFILE'] = 0  # mock calibration, no input files
    fluxcal_fac = data.FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, err_hdr = errhd)

    mocks.rename_files_to_cgi_format(list_of_fits=[fluxcal_fac], output_dir=calibrations_dir, level_suffix="abf_cal")
    this_caldb.create_entry(fluxcal_fac)

    #################################################
    ########### Create Mueller Matrix cals ##########
    #################################################

        # define mueller matrices and stokes vectors
    nd_mueller_matrix = np.array([
        [0.8, 0.1, 0, 0],
        [0.05, -0.75, 0, 0],
        [0.05, 0.05, 0.75, 0],
        [0, 0, 0, 0.95]
    ])
    system_mueller_matrix = np.array([
        [0.9, -0.02, 0, 0],
        [0.01, -0.8, 0, 0],
        [0, 0, 0.8, 0.005],
        [0, 0, -0.01, 0.9]
    ])

    #Create dataset because we need to input something: 
    prihdr, exthdr = mocks.create_default_L1_headers()
    dummy1 = data.Image(np.zeros((10,10)), pri_hdr=prihdr, ext_hdr=exthdr)
    dummy2 = data.Image(np.zeros((10,10)), pri_hdr=prihdr, ext_hdr=exthdr)
    input_dataset = data.Dataset([dummy1, dummy2])

    mm_prihdr, mm_exthdr, _, _ = mocks.create_default_calibration_product_headers()
    system_mm_cal = data.MuellerMatrix(system_mueller_matrix, pri_hdr=mm_prihdr.copy(), ext_hdr=mm_exthdr.copy(), input_dataset=input_dataset)
    nd_exthdr = mm_exthdr.copy()
    for dataset in input_dataset.frames:
        dataset.ext_hdr['FPAMNAME'] = 'ND475'
    nd_mm_cal = data.NDMuellerMatrix(nd_mueller_matrix, pri_hdr=mm_prihdr.copy(), ext_hdr=nd_exthdr, input_dataset=input_dataset)

    mocks.rename_files_to_cgi_format(list_of_fits=[system_mm_cal], output_dir=calibrations_dir, level_suffix="mmx_cal")
    mocks.rename_files_to_cgi_format(list_of_fits=[nd_mm_cal], output_dir=calibrations_dir, level_suffix="ndm_cal")

    this_caldb.create_entry(system_mm_cal)
    this_caldb.create_entry(nd_mm_cal)
    
    logger.info("Created calibration products:")
    logger.info(f"  - NonLinearityCalibration: {nonlinear_cal.filename}")
    logger.info(f"  - KGain: {kgain.filename}")
    logger.info(f"  - DetectorNoiseMaps: {noise_map.filename}")
    if sample_l1_files:
        logger.info(f"  - Dark: {dark_cal.filename}")
    logger.info(f"  - FlatFieldPOL0: {flat0.filename}")
    logger.info(f"  - FlatFieldPOL45: {flat45.filename}")
    logger.info(f"  - BadPixelMap: {bp_map.filename}")
    logger.info(f"  - AstrometricCalibration: {astrom_cal.filename}")
    logger.info(f"  - CTM Calibration: {ct_cal_tmp.filename}")
    logger.info(f"  - Flux Calibration Factor: {fluxcal_fac.filename}")
    logger.info(f"  - System Mueller Matrix: {system_mm_cal.filename}")
    logger.info(f"  - ND Mueller Matrix: {nd_mm_cal.filename}")
    logger.info('')

    # ================================================================================
    # (2) Validate Input Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 1: Input L1 Image Data Format and Content')
    logger.info('='*80)
    
    # Filter to only include L1 files as inputs
    all_files = os.listdir(l1_datadir)
    input_files = [f for f in all_files if f.endswith('l1_.fits')]
    if not input_files:
        raise FileNotFoundError(f"No files ending in 'l1_.fits' found in {l1_datadir}")
    
    input_data_filelist = [os.path.join(l1_datadir, f) for f in input_files]
    
    # Create input_data subfolder
    input_data_dir = os.path.join(l4_outputdir, 'input_l1')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    # Update headers
    # TO DO: pol sims should have the correct VISTYPE, currently undefined
    input_data_filelist = check.fix_hdrs_for_tvac(input_data_filelist, input_data_dir)

    for f in input_data_filelist:
        hdulist = fits.open(f)
        hdulist[1].header['FSMPRFL'] = 'NFOV'
        hdulist.writeto(f, overwrite=True)

    # Validate all input images
    input_dataset = data.Dataset(input_data_filelist)
    
    for i, (frame, filepath) in enumerate(zip(input_dataset, input_data_filelist)):
        frame_info = f"L1 Input Frame {i}"
        
        check_filename_convention(os.path.basename(filepath), 'cgi_*_l1_.fits', frame_info, logger, data_level='l1_')
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L1'}, frame_info, logger)
        
        
        # Verify HDU count
        try:
            with fits.open(filepath) as hdul:
                verify_hdu_count(hdul, 2, frame_info, logger) 
        except Exception as e:
            logger.info(f"{frame_info}: HDU count verification failed. Error: {str(e)}. FAIL")
        
        # Check dimensions
        logger.info(f"{frame_info}: Data shape {frame.data.shape}")
        
        logger.info("")
    
    logger.info(f"Total input images validated: {len(input_dataset)}")
    logger.info('')
    
    # ================================================================================
    # (3) Run Processing Pipeline
    # ================================================================================
    logger.info('='*80)
    logger.info('Running L1 -> L2b -> L3 polarimetry data processing pipeline')
    logger.info('='*80)
    
    # # Step 1: L1 -> L2b
    logger.info('Step 1: Running L1 to L2a recipe...')
    with warnings.catch_warnings():  
        warnings.filterwarnings('ignore', category=UserWarning)# prevent UserWarning: Number of frames which made the DetectorNoiseMaps product is less than the number of frames in input_dataset
        walker.walk_corgidrp(input_data_filelist, "", l4_outputdir)
    
    l2a_files = [f for f in os.listdir(l4_outputdir) if f.endswith('_l2a.fits')]
    l2a_filelist = [os.path.join(l4_outputdir, f) for f in l2a_files]
    logger.info(f'L1 to L2a complete. Generated {len(l2a_filelist)} L2a files.')
    logger.info('')
    
    logger.info('Step 2: Running L2a to L2b recipe...')
    with warnings.catch_warnings():  
        warnings.filterwarnings('ignore', category=UserWarning)# prevent UserWarning: Number of frames which made the DetectorNoiseMaps product is less than the number of frames in input_dataset
        walker.walk_corgidrp(l2a_filelist, "", l4_outputdir)


    l2b_files = [f for f in os.listdir(l4_outputdir) if f.endswith('_l2b.fits')]
    l2b_filelist = [os.path.join(l4_outputdir, f) for f in l2b_files]
    logger.info(f'L2a to L2b complete. Generated {len(l2b_filelist)} L2b files.')
    logger.info('')
    
    # # Step 2: L2b -> L3 
    logger.info('Step 3: Running L2b to L3 polarimetry recipe...')
    walker.walk_corgidrp(l2b_filelist, "", l4_outputdir)
    l3_filelist = [os.path.join(l4_outputdir, f) for f in os.listdir(l4_outputdir) if f.endswith('_l3_.fits')]
    logger.info(f'L2b to L3 complete. Generated {len(l3_filelist)} L3 files.')
    logger.info('')

    ##############################################
    # Step 1/2 b: Do the stap spot data. 
    logger.info('Running L1 to L2a on the polarimetry satspots...')
    # Ensure subsequent walker calls use this test's CalDB
    corgidrp.caldb_filepath = tmp_caldb_csv
    input_satspots_filenames = os.listdir(os.path.join(l1_datadir,"sat_spots"))
    input_files = np.array(sorted([os.path.join(l1_datadir, "sat_spots", f) for f in input_satspots_filenames if f.endswith('l1_.fits')]))
    logger.info(f'Found {len(input_files)} sat spot files in input directory: {l1_datadir}/sat_spots')
    sat_spot_dir = os.path.join(l4_outputdir,"sat_spots")

    #Split the input data by targets
    input_targets = np.array([fits.getheader(f)['TARGET'] for f in input_files])
    unique_targets = np.unique(input_targets)
    # import IPython; IPython.embed()
    for target in unique_targets:
        filelist = input_files[np.where(input_targets == target)]
        walker.walk_corgidrp(list(filelist), "", sat_spot_dir)
    l2a_satspot_files = np.array([os.path.join(sat_spot_dir, f) for f in os.listdir(sat_spot_dir) if f.endswith('_l2a.fits')])
    logger.info(f'L1 to L2a complete for satspots. Generated {len(l2a_satspot_files)} L2a files.')

    logger.info('Running L2a to L2b on the polarimetry satspots...')
    input_targets = np.array([fits.getheader(f)['TARGET'] for f in l2a_satspot_files])
    unique_targets = np.unique(input_targets)
    for target in unique_targets:
        filelist = l2a_satspot_files[input_targets == target]
        walker.walk_corgidrp(list(filelist), "", sat_spot_dir)
    l2b_satspot_files = [os.path.join(sat_spot_dir, f) for f in os.listdir(sat_spot_dir) if f.endswith('_l2b.fits')]
    logger.info(f'L2a to L2b complete for satspots. Generated {len(l2b_satspot_files)} L2b files.')    

    #Don't need to split them for L3 - data can be more inhomogeneous. 
    logger.info('Running L2b to L3 on the polarimetry satspots...')
    walker.walk_corgidrp(l2b_satspot_files, "", sat_spot_dir)    
    logger.info('L2b to L3 sat spots complete')
    l3_satspot_files = [os.path.join(sat_spot_dir, f) for f in os.listdir(sat_spot_dir) if f.endswith('_l3_.fits')]

    # # # Append the satspot files to the l3_filelist
    l3_filelist.extend(l3_satspot_files)


    ##############################################
    # Step 1/2 b: Do the unocculted data. 
    logger.info('Running L1 to L2a on the unocculted images...')
    input_offaxis_filenames = os.listdir(os.path.join(l1_datadir,"off_axis"))
    logger.info(f'Found {len(input_offaxis_filenames)} off-axis files in input directory: {l1_datadir}/off_axis')
    input_files = np.array(sorted([os.path.join(l1_datadir, "off_axis", f) for f in input_offaxis_filenames if f.endswith('l1_.fits')]))
    off_axis_dir = os.path.join(l4_outputdir,"off_axis")
    
    #Split the input data by targets
    input_targets = np.array([fits.getheader(f)['TARGET'] for f in input_files])
    unique_targets = np.unique(input_targets)
    for target in unique_targets:
        filelist = input_files[np.where(input_targets == target)[0]]
        walker.walk_corgidrp(list(filelist), "", off_axis_dir)
    l2a_unocculted_files = np.array([os.path.join(off_axis_dir, f) for f in os.listdir(off_axis_dir) if f.endswith('_l2a.fits')])
    logger.info(f'L1 to L2a complete for unocculted images. Generated {len(l2a_unocculted_files)} L2a files.')

    logger.info('Running L2a to L2b on the unocculted images...')
    input_targets = np.array([fits.getheader(f)['TARGET'] for f in l2a_unocculted_files])
    unique_targets = np.unique(input_targets)
    for target in unique_targets:
        filelist = l2a_unocculted_files[np.where(input_targets == target)[0]]
        walker.walk_corgidrp(list(filelist), "", off_axis_dir)
    l2b_offaxis_files = [os.path.join(off_axis_dir, f) for f in os.listdir(off_axis_dir) if f.endswith('_l2b.fits')]   
    logger.info(f'L2a to L2b complete for unocculted images. Generated {len(l2b_offaxis_files)} L2b files.')

    logger.info('Running L2b to L3 on the unocculted images...')
    walker.walk_corgidrp(l2b_offaxis_files, "", off_axis_dir)    
    logger.info('L1 to L3 off-axis complete')
    l3_offaxis_files = [os.path.join(off_axis_dir, f) for f in os.listdir(off_axis_dir) if f.endswith('_l3_.fits')]

    # # Append the satspot files to the l3_filelist
    l3_filelist.extend(l3_offaxis_files)

    # Step 3: L3 -> L4 (with policy)
    logger.info('Step 3: Running L3 to L4 polarimetry recipe...')
    walker.walk_corgidrp(l3_filelist, "", l4_outputdir, template="l3_to_l4_pol.json")
    logger.info('L3 to L4 with policy complete.')
    logger.info('')
    
    # ================================================================================
    # (4) Validate Output L3 Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Output L4 Image Data Format and Content')
    logger.info('='*80)
    
    # Filter out calibration files and only get L4 data files
    all_files = [f for f in os.listdir(l4_outputdir) if f.endswith('.fits')]
    new_l4_filenames = [os.path.join(l4_outputdir, f) for f in all_files if '_l4' in f and '_cal' not in f]

    # Basic validation: check that L4 files were created
    if len(new_l4_filenames) == 0:
        logger.info("No L4 files created. FAIL")
        raise AssertionError("No L4 files were created")

    logger.info(f"Found {len(new_l4_filenames)} L4 output files")
    for fname in new_l4_filenames:
        logger.info(f"  - {os.path.basename(fname)}")
    logger.info('')

    # Check that each L4 file has proper headers and data
    for i, l4_filename in enumerate(new_l4_filenames):
        frame_info = f"L4 Output Frame {i}"
        
        try:
            img = data.Image(l4_filename)
            
            # Verify filename
            check_filename_convention(os.path.basename(l4_filename), 'cgi_*_l4_.fits', frame_info, logger, data_level='l4_')

            # Verify HDU count
            with fits.open(l4_filename) as hdul:
                verify_hdu_count(hdul, 6, frame_info, logger)  # L4 should have 5 HDUs

            # Verify data level
            verify_header_keywords(img.ext_hdr, {'DATALVL': 'L4'}, frame_info, logger)

            # Check this is polarimetry data
            # dpam = img.ext_hdr.get('DPAMNAME', '')
            # if dpam in ('POL0', 'POL45'):
            #     logger.info(f"{frame_info}: DPAMNAME = '{dpam}' (polarimetry). PASS")
            # else:
            #     logger.info(f"{frame_info}: DPAMNAME = '{dpam}'. Expected POL0 or POL45. FAIL")
            
            # Check data dimensions - should always be polarimetry datacube (2, N, N)
            if len(img.data.shape) == 3 and img.data.shape[0] == 4:
                logger.info(f"{frame_info}: Polarimetry datacube shape {img.data.shape}. PASS")
            else:
                logger.info(f"{frame_info}: Expected polarimetry datacube (4, N, N), got {img.data.shape}. FAIL")
                raise AssertionError(f"{frame_info}: Expected polarimetry datacube (4, N, N), got {img.data.shape}")
            
            # Verify WCS headers exist (from create_wcs step)
            wcs_keys = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2']
            missing_wcs = [k for k in wcs_keys if k not in img.ext_hdr]
            if not missing_wcs:
                logger.info(f"{frame_info}: WCS headers present ({', '.join(wcs_keys)}). PASS")
            else:
                logger.info(f"{frame_info}: WCS headers incomplete. Missing: {', '.join(missing_wcs)}). FAIL")
            
            # Verify data has been divided by exposure time (should be in photoelectrons/s)
            if img.ext_hdr['BUNIT'] == 'photoelectron/s':
                logger.info(f"{frame_info}: BUNIT = 'photoelectron/s'. PASS")
            else:
                logger.info(f"{frame_info}: BUNIT = '{img.ext_hdr['BUNIT']}'. Expected: 'photoelectron/s'. FAIL")
            
        except Exception as e:
            logger.info(f"{frame_info}: Validation failed with error: {str(e)}. FAIL")
        
        logger.info('')

    logger.info(f"Total output L4 images validated: {len(new_l4_filenames)}")
    logger.info('')
    
    # remove temporary caldb file if exists
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)

    return new_l4_filenames

@pytest.mark.e2e
def test_l1_to_l4(e2edata_path, e2eoutput_path):
    """Run the complete L1 to L4 polarimetry data end-to-end test with recipe chaining.

    Args:
        e2edata_path (str): Path to test data (expects L1 files)
        e2eoutput_path (str): Output directory path for results and logs
    """
    # Set up output directory
    l4_outputdir = os.path.join(e2eoutput_path, "l1_to_l4_pol_e2e")
    if os.path.exists(l4_outputdir):
        shutil.rmtree(l4_outputdir)
    os.makedirs(l4_outputdir)

    analog_outputdir = os.path.join(l4_outputdir, "analog")
    pc_outputdir = os.path.join(l4_outputdir, "pc")
    sat_spots_outputdir = os.path.join(analog_outputdir, "sat_spots")
    off_axis_outputdir = os.path.join(analog_outputdir, "off_axis")
    if not os.path.exists(analog_outputdir):
        os.makedirs(analog_outputdir)
    if not os.path.exists(pc_outputdir):
        os.makedirs(pc_outputdir)
    if not os.path.exists(sat_spots_outputdir):
        os.makedirs(sat_spots_outputdir)
    if not os.path.exists(off_axis_outputdir):
        os.makedirs(off_axis_outputdir)

    log_file = os.path.join(l4_outputdir, 'l1_to_l4_pol_e2e.log')

    # Create a new logger specifically for this test
    global logger
    logger = logging.getLogger('l1_to_l4_pol_e2e')
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
    logger.info('L1 TO L3 POLARIMETRY DATA END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    # L1 data directories
    analog_datadir = os.path.join(e2edata_path, "POL_sims", "l1s_for_l4", "analog_data")
    # pc_datadir = os.path.join(e2edata_path, "POL_sims", "L1", "PC_data")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    
    # Run the complete end-to-end test
    try:
        logger.info('='*80)
        logger.info('ANALOG POLARIMETRY DATA TEST')
        logger.info('='*80)
        new_l4_analog_filenames = run_l1_to_l4_e2e_test(analog_datadir, analog_outputdir, processed_cal_path, logger)
        
        # logger.info('='*80)
        # logger.info('PC POLARIMETRY DATA TEST')
        # logger.info('='*80)
        # new_l4_pc_filenames = run_l1_to_l4_e2e_test(pc_datadir, pc_outputdir, processed_cal_path, logger)
        
        logger.info('='*80)
        logger.info('END-TO-END TEST COMPLETE')
        logger.info('='*80)

        for new_filename in new_l4_analog_filenames:
            check.compare_to_mocks_hdrs(new_filename)
        # for new_filename in new_l4_pc_filenames:
        #     check.compare_to_mocks_hdrs(new_filename)

        print('e2e test for L1 to L4 polarimetry passed')
    except Exception as e:
        logger.error('='*80)
        logger.error('END-TO-END TEST FAILED')
        logger.error('='*80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Print traceback to log
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())

        print(f'e2e test for L1 to L4 polarimetry FAILED: {str(e)}')
        raise

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    outputdir = '/Users/jmilton/Github/corgidrp/tests/e2e_tests'

    ap = argparse.ArgumentParser(description="run the l1->l4 polarimetry end-to-end test with recipe chaining")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")

    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir

    test_l1_to_l4(e2edata_dir, outputdir)