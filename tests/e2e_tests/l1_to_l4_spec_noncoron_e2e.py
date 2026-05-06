import argparse
import os, sys
import json
import pytest
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.astrom as astrom
import corgidrp.check as check
import shutil
import logging
import traceback
from corgidrp.data import (Dataset, Image)
from corgidrp.data import (NonLinearityCalibration, BadPixelMap, 
                            KGain, DetectorNoiseMaps, FlatField, FluxcalFactor)
from corgidrp.check import (check_filename_convention, check_dimensions, 
                           verify_hdu_count, verify_header_keywords)
from corgidrp.darks import build_synthesized_dark
from corgidrp.photon_counting import get_pc_mean
import warnings

thisfile_dir = os.path.dirname(__file__) # this file's folder


def run_l1_to_l4_e2e_test(l1_datadir, l4_outputdir, processed_cal_path, logger):
    """Run the complete L1 to L4 spectroscopy data end-to-end test.
    
    Args:
        l1_datadir (str): Path to L1 input data directory
        l4_outputdir (str): Path to output directory
        processed_cal_path (str): Path to calibration files directory
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
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
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
    l1_files = [f for f in all_files if f.endswith('l1.fits') or f.endswith('l1_.fits')]
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
    # fix ISPC to int 
    for file in mock_cal_filelist:
        with fits.open(file, mode='update') as fits_file:
            if 'ISPC' in fits_file[1].header:
                fits_file[1].header['ISPC'] = int(fits_file[1].header['ISPC'])

    mock_input_dataset = Dataset(mock_cal_filelist)

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")
    this_caldb.create_entry(nonlinear_cal)

    # KGain (with read noise)
    kgain_val = 8.7  # Standard value from TVAC headers
    signal_array = np.linspace(0, 50)
    noise_array = np.sqrt(signal_array)
    ptc = np.column_stack([signal_array, noise_array])
    ext_hdr['RN'] = 100.0
    ext_hdr['RN_ERR'] = 0.0
    kgain = KGain(kgain_val, ptc=ptc, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
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
    noise_map = DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                       input_dataset=mock_input_dataset, err=noise_map_noise,
                                       dq=noise_map_dq, err_hdr=err_hdr)
    mocks.rename_files_to_cgi_format(list_of_fits=[noise_map], output_dir=calibrations_dir, level_suffix="dnm_cal")
    this_caldb.create_entry(noise_map)

    # Check if data is PC or analog to determine which type of dark to create
    sample_l1_files = [f for f in os.listdir(l1_datadir) if f.endswith('l1.fits') or f.endswith('l1_.fits')]
    is_pc_data = False
    if sample_l1_files:
        sample_l1_file = os.path.join(l1_datadir, sample_l1_files[0])
        with fits.open(sample_l1_file) as hdul:
            exthdr = hdul[1].header
            if 'ISPC' in exthdr:
                ispc_val = int(exthdr.get('ISPC'))
                if ispc_val == 1:
                    is_pc_data = True
                elif ispc_val == 0:
                    is_pc_data = False
                else:
                    raise ValueError(f"Expected 0 or 1 for ISPC value in header: {ispc_val}.")
            else:
                raise ValueError("Missing ISPC keyword in L1 header. Cannot determine PC vs analog.")
    logger.info(f'Photon counting mode = {is_pc_data}')
    # Dark calibration - create appropriate dark based on data type
    dark_cal = None
    if not is_pc_data:
        # Create analog dark for analog data
        # Get exposure time and emgain from a sample L1 file
        if sample_l1_files:
            sample_l1_file = os.path.join(l1_datadir, sample_l1_files[0])
            sample_hdr = fits.getheader(sample_l1_file, ext=1)
            data_exptime = sample_hdr['EXPTIME']
            data_emgain = float(sample_hdr['EMGAIN_C'])
            
            # Create dataset with correct header values
            temp_dataset = Dataset(mock_cal_filelist[:1])
            temp_dataset.frames[0].ext_hdr['EXPTIME'] = data_exptime
            temp_dataset.frames[0].ext_hdr['EMGAIN_C'] = data_emgain
            
            dark_cal = build_synthesized_dark(temp_dataset, noise_map)
            mocks.rename_files_to_cgi_format(list_of_fits=[dark_cal], output_dir=calibrations_dir, level_suffix="drk_cal")
            this_caldb.create_entry(dark_cal)
            logger.info('Analog dark calibration created.')
    else:
        # PC dark will be created later from L2a frames (need L2a files first)
        logger.info('PC dark will be created from L2a frames after L1->L2a processing')

    # Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[flat], output_dir=calibrations_dir, level_suffix="flt_cal")
    this_caldb.create_entry(flat)

    # Bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    # Make sure BPM includes a dark(-like) frame
    bp_dark = dark_cal
    if bp_dark is None:
        bp_dark = build_synthesized_dark(Dataset([flat]), noise_map)
    bp_map_inputs = Dataset([bp_dark, flat])
    bp_map = BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=bp_map_inputs)
    mocks.rename_files_to_cgi_format(list_of_fits=[bp_map], output_dir=calibrations_dir, level_suffix="bpm_cal")
    this_caldb.create_entry(bp_map)

    # Create mock astrometric calibration
    field_path = os.path.join(os.path.dirname(__file__), "..", "test_data", "JWST_CALFIELD2020.csv")
    astrom_input_dir = os.path.join(l4_outputdir, 'astrom_cal_input')
    if not os.path.exists(astrom_input_dir):
        os.makedirs(astrom_input_dir)
    
    mock_dataset = mocks.create_astrom_data(field_path=field_path, filedir=None)
    mock_dataset.save(filedir=astrom_input_dir)
    astrom_cal = astrom.boresight_calibration(input_dataset=mock_dataset, field_path=field_path, find_threshold=5)
    mocks.rename_files_to_cgi_format(list_of_fits=[astrom_cal], output_dir=calibrations_dir, level_suffix="ast_cal")
    this_caldb.create_entry(astrom_cal)

    #Using a specfluxcal calib file created using DIP data
    specflux_cal = corgidrp.data.SpecFluxCal(os.path.join(processed_cal_path,'cgi_0200001001001001001_20260319t1146350_sfl_cal.fits'))

    mocks.rename_files_to_cgi_format(list_of_fits=[specflux_cal], output_dir=calibrations_dir, level_suffix="sfl_cal")
    this_caldb.create_entry(specflux_cal)

    logger.info("Created calibration products:")
    logger.info(f"  - NonLinearityCalibration: {nonlinear_cal.filename}")
    logger.info(f"  - KGain: {kgain.filename}")
    logger.info(f"  - DetectorNoiseMaps: {noise_map.filename}")
    # Log dark calibration - only exists for analog data
    if not is_pc_data and sample_l1_files:
        logger.info(f"  - Dark: {dark_cal.filename}")
    elif is_pc_data:
        logger.info(f"  - Dark: (PC dark will be created from L2a frames)")
    logger.info(f"  - FlatField: {flat.filename}")
    logger.info(f"  - BadPixelMap: {bp_map.filename}")
    logger.info(f"  - AstrometricCalibration: {astrom_cal.filename}")
    logger.info(f"  - SpecFluxCal: {specflux_cal.filename}")
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

    # Validate all input images
    input_dataset = Dataset(input_data_filelist)

    for i, (frame, filepath) in enumerate(zip(input_dataset, input_data_filelist)):
        frame_info = f"L1 Input Frame {i}"
        
        check_filename_convention(os.path.basename(filepath), 'cgi_*_l1_.fits', frame_info, logger, data_level='l1_')
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L1'}, frame_info, logger)
        
        # Verify HDU count
        expected_hdus = 2
        try:
            with fits.open(filepath) as hdul:
                verify_hdu_count(hdul, expected_hdus, frame_info, logger)
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
    logger.info('Running L1 -> L2a -> L2b -> L3 spectroscopy data processing pipeline')
    logger.info('='*80)
    
    # Step 1: L1 -> L2a (generic processing)
    logger.info('Step 1: Running L1 to L2a recipe...')
    with warnings.catch_warnings():  
        warnings.filterwarnings('ignore', category=UserWarning)# prevent UserWarning: Number of frames which made the DetectorNoiseMaps product is less than the number of frames in input_dataset
        walker.walk_corgidrp(input_data_filelist, "", l4_outputdir)
    
    # Find the L2a output files
    l2a_files = [f for f in os.listdir(l4_outputdir) if f.endswith('_l2a.fits')]
    l2a_filelist = [os.path.join(l4_outputdir, f) for f in l2a_files]
    logger.info(f'L1 to L2a complete. Generated {len(l2a_filelist)} L2a files.')
    
    '''
    if l2a_filelist:
        
        # Dark calibration: Create PC master dark from L2a frames (for PC data only)
        # This completes the calibration setup that began in the calibration products section above
        if is_pc_data:
            logger.info('Creating photon-counted master dark from mocked L1 dark frames...')
            try:
                # 1) Generate photon-countable L1 dark frames using mocks
                num_illum_frames = len(l2a_filelist)
                num_dark_frames = max(num_illum_frames, 10)
                _, dark_l1_dataset, _, _ = mocks.create_photon_countable_frames(
                    Nbrights=1, Ndarks=num_dark_frames)

                # 2) Save L1 darks to disk
                dark_l1_dir = os.path.join(l4_outputdir, 'pc_dark_l1')
                if not os.path.exists(dark_l1_dir):
                    os.makedirs(dark_l1_dir)
                dark_l1_dataset.save(filedir=dark_l1_dir)

                # 3) Convert L1 darks -> L2a using the walker
                dark_l1_files = [os.path.join(dark_l1_dir, f) for f in os.listdir(dark_l1_dir) if f.endswith('_l1_.fits')]
                dark_l2a_outdir = os.path.join(l4_outputdir, 'pc_dark_l2a')
                if not os.path.exists(dark_l2a_outdir):
                    os.makedirs(dark_l2a_outdir)
                with warnings.catch_warnings():  
                    warnings.filterwarnings('ignore', category=UserWarning)# prevent UserWarning: Number of frames which made the DetectorNoiseMaps product is less than the number of frames in input_dataset
                    walker.walk_corgidrp(dark_l1_files, "", dark_l2a_outdir, template="l1_to_l2a_basic.json")

                # 4) Collect resulting L2a dark frames
                dark_l2a_filelist = [os.path.join(dark_l2a_outdir, f) for f in os.listdir(dark_l2a_outdir) if f.endswith('_l2a.fits')]
                if len(dark_l2a_filelist) == 0:
                    raise Exception('No L2a darks produced from mocked L1 dark frames')

                # 5) Build PC master dark from L2a darks
                from corgidrp.photon_counting import get_pc_mean
                dark_l2a_dataset = Dataset(dark_l2a_filelist)
                pc_dark = get_pc_mean(dark_l2a_dataset, inputmode='darks')
                if pc_dark.ext_hdr.get('PC_STAT') != 'photon-counted master dark':
                    raise Exception('Failed to create valid photon-counted master dark')
                calibrations_dir = os.path.join(l4_outputdir, 'calibrations')
                if not os.path.exists(calibrations_dir):
                    os.makedirs(calibrations_dir)
                mocks.rename_files_to_cgi_format(list_of_fits=[pc_dark], output_dir=calibrations_dir, level_suffix="drk_cal")
                this_caldb.create_entry(pc_dark)
                logger.info(f'Photon-counted master dark created from {len(dark_l2a_filelist)} L2a dark frames.')
            except Exception as e:
                logger.warning(f'Could not create photon-counted master dark: {e}.')
                import traceback
                logger.warning(traceback.format_exc())
    '''

    # Step 2: L2a -> L2b (auto-detect spectroscopy recipe)

    logger.info('Step 2: Running L2a to L2b recipe...')
    '''
    if is_pc_data:
        recipe = walker.autogen_recipe(l2a_filelist, l4_outputdir)
        ### Modify keyword to so that the PC master dark is used
        for step in recipe[0]['steps']:
            if step['name'] == "dark_subtraction":
                step['calibs']['Dark'] = pc_dark.filepath
        with warnings.catch_warnings():  
            warnings.filterwarnings('ignore', category=UserWarning)# prevent UserWarning: Number of frames which made the DetectorNoiseMaps product is less than the number of frames in input_dataset
            output_filepaths = walker.run_recipe(recipe[0], save_recipe_file=True)
        recipe[1]['inputs'] = output_filepaths
        for step in recipe[1]['steps']:
            if step['name'] == "get_pc_mean":
                step['calibs']['Dark'] = pc_dark.filepath
        with warnings.catch_warnings():  
            warnings.filterwarnings('ignore', category=UserWarning)# prevent UserWarning: Number of frames which made the DetectorNoiseMaps product is less than the number of frames in input_dataset
            output_filepaths1 = walker.run_recipe(recipe[1], save_recipe_file=True)
            # files are overwritten with same filenames
            recipe[2]['inputs'] = output_filepaths1
            walker.run_recipe(recipe[2], save_recipe_file=True)
    '''
    with warnings.catch_warnings():  
        warnings.filterwarnings('ignore', category=UserWarning)# prevent UserWarning: Number of frames which made the DetectorNoiseMaps product is less than the number of frames in input_dataset
        walker.walk_corgidrp(l2a_filelist, "", l4_outputdir)
    
    # Find the L2b output files
    l2b_files = [f for f in os.listdir(l4_outputdir) if f.endswith('_l2b.fits')]
    l2b_filelist = [os.path.join(l4_outputdir, f) for f in l2b_files]
    logger.info(f'L2a to L2b complete. Generated {len(l2b_filelist)} L2b files.')

    # Step 3: L2b -> L3 (using output from step 2)
    logger.info('Step 3: Running L2b to L3 spectroscopy recipe...')
    walker.walk_corgidrp(l2b_filelist, "", l4_outputdir)
    l3_filelist = [os.path.join(l4_outputdir, f) for f in os.listdir(l4_outputdir) if f.endswith('_l3_.fits')]
    logger.info('L2b to L3 complete.')
    logger.info('')
    logger.info(f"Generated and saved {len(l3_filelist)} L3 input files")

    ##################################################################################
    # Step 3: L3 -> L4 (with policy)
    logger.info('Step 3: Running L3 to L4 non-coronographic spectroscopy recipe...')
    walker.walk_corgidrp(l3_filelist, "", os.path.join(l4_outputdir,'../'))
    logger.info('L3 to L4 complete.')
    logger.info('')

    # ================================================================================
    # (5) Validate Output L3 Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Output L3 Image Data Format and Content')
    logger.info('='*80)
    
    # Filter out calibration files and only get L3 data files
    all_files = [f for f in os.listdir(l4_outputdir) if f.endswith('.fits')]
    new_l3_filenames = [os.path.join(l4_outputdir, f) for f in all_files if '_l3' in f and '_cal' not in f]

    # Basic validation: check that L3 files were created
    if len(new_l3_filenames) == 0:
        logger.info("No L3 files created. FAIL")
        raise AssertionError("No L3 files were created")
    
    logger.info(f"Found {len(new_l3_filenames)} L3 output files")
    for fname in new_l3_filenames:
        logger.info(f"  - {os.path.basename(fname)}")
    logger.info('')

    # Check that each L3 file has proper headers and data
    for i, l3_filename in enumerate(new_l3_filenames):
        frame_info = f"L3 Output Frame {i}"
        
        try:
            img = Image(l3_filename)
            
            # Verify filename
            check_filename_convention(os.path.basename(l3_filename), 'cgi_*_l3_.fits', frame_info, logger, data_level='l3_')
            
            # Verify HDU count
            expected_hdus = 4
            with fits.open(l3_filename) as hdul:
                verify_hdu_count(hdul, expected_hdus, frame_info, logger)
            
            # Verify data level
            verify_header_keywords(img.ext_hdr, {'DATALVL': 'L3'}, frame_info, logger)

            # Verify analog or PC
            verify_header_keywords(img.ext_hdr, {'ISPC': is_pc_data}, frame_info, logger)

            # Check this is spectroscopy data
            dpam = img.ext_hdr.get('DPAMNAME', '')
            if dpam in ('PRISM3'):
                logger.info(f"{frame_info}: DPAMNAME = '{dpam}' (spectroscopy). PASS")
            else:
                logger.info(f"{frame_info}: DPAMNAME = '{dpam}'. Expected PRISM3. FAIL")
            
            # Check data dimensions
            check_dimensions(img.data, (125,125), frame_info, logger)
            
            # Verify WCS headers exist (from create_wcs step)
            wcs_keys = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2','PLTSCALE','NORTHANG' ]
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
            raise
        
        logger.info('')
    
    logger.info(f"Total output L3 images validated: {len(new_l3_filenames)}")
    logger.info('')

   # ================================================================================
    # (4) Validate Output L4 Image Data
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Output Output L4 Image Data Format and Content')
    logger.info('='*80)

    # Validate output product
    out_file = check.get_latest_cal_file(os.path.join(l4_outputdir,'../'), '*_l4_.fits', logger)
    check_filename_convention(os.path.basename(out_file), 'cgi_*_l4_.fits', "spec l4 output product", logger, data_level = "l4_")

    with fits.open(out_file) as hdul:        
        verify_hdu_count(hdul, 11, "spec l4 output product", logger)
        
        # Verify HDU0 (header only)
        hdu0 = hdul[0]
        if hdu0.data is None:
            logger.info("HDU0: Header only. Expected: header only. PASS.")
        else:
            logger.info(f"HDU0: Contains data with shape {hdu0.data.shape}. Expected: header only. FAIL.")
            
        # Verify HDU1 (spec data)
        hdu1 = hdul[1]
        data = hdu1.data
        check_dimensions(data, (81, 81), "HDU1 Data Array: 2D array containing the 2D spectral distribution", logger)
        # Verify HDU2 (err)
        hdu2 = hdul[2]
        data = hdu2.data
        check_dimensions(data, (1,81, 81), "HDU2 Data Array: 3D array containing the 3D spectral uncertainty", logger)
        # Verify HDU3 (DQ)
        hdu3 = hdul[3]
        data = hdu3.data
        check_dimensions(data, (81, 81), "HDU3 Data Array: 2D array containing the 2D data quality", logger)
        # Verify HDU4 (WAVE)
        hdu4 = hdul[4]
        data = hdu4.data
        check_dimensions(data, (81, 81), "HDU4 Data Array: 2D array containing the 2D wavelength distribution", logger)
        
        # PASS/FAIL checks for wavelength map (CGI-REQT-5464)
        logger.info("")
        logger.info("Wavelength-to-pixel calibration map (CGI-REQT-5464) validation:")
        
        # Check data type is float64
        if data.dtype.name == "float64":
            logger.info(f"    Data type: {data.dtype.name}. Expected: float64. PASS")
        else:
            logger.error(f"    Data type: {data.dtype.name}. Expected: float64. FAIL")
        
        # Check header contains expected expected keywords
        expected_keywords = {'BUNIT', 'REFWAVE', 'XREFWAV', 'YREFWAV'}
        hdu4_header_keys = set(hdu4.header.keys())
        missing_keywords = expected_keywords - hdu4_header_keys
        if missing_keywords:
            logger.error(f"    Header missing expected keywords: {missing_keywords}. FAIL")
        else:
            logger.info(f"    Header contains expected expected keywords (BUNIT, REFWAVE, XREFWAV, YREFWAV). PASS")
        
        # Confirm that wavelength map dimensions exactly match the L3 image dimensions in HDU1
        hdu1_data = hdul[1].data
        if data.shape == hdu1_data.shape:
            logger.info(f"    Wavelength map dimensions {data.shape} exactly match L3 image dimensions {hdu1_data.shape} in HDU1. PASS")
        else:
            logger.error(f"    Wavelength map dimensions {data.shape} do not match L3 image dimensions {hdu1_data.shape} in HDU1. FAIL")
        
        # Confirm no NaNs/inf values in wavelength map
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        if has_nan:
            nan_count = np.isnan(data).sum()
            logger.error(f"    Contains {nan_count} NaN values. Expected: no NaNs. FAIL")
        else:
            logger.info(f"    No NaN values in wavelength map. PASS")
        if has_inf:
            inf_count = np.isinf(data).sum()
            logger.error(f"    Contains {inf_count} inf values. Expected: no infs. FAIL")
        else:
            logger.info(f"    No inf values in wavelength map. PASS")
        
        # Print min/max wavelength values from wavelength map for inspection
        valid_data = data[np.isfinite(data)]
        if len(valid_data) > 0:
            min_wave = np.min(valid_data)
            max_wave = np.max(valid_data)
            logger.info(f"    Minimum wavelength value: {min_wave} nm")
            logger.info(f"    Maximum wavelength value: {max_wave} nm")
        else:
            logger.error(f"    No valid (finite) wavelength values found for min/max calculation. FAIL")
        
        # Check and print wavelength zero-point values
        hdu1_header = hdul[1].header
        wv0_keywords = {'WAVLEN0', 'WV0_X', 'WV0_Y', 'WV0_XERR', 'WV0_YERR', 'WV0_DIMX', 'WV0_DIMY'}
        missing_wv0 = wv0_keywords - set(hdu1_header.keys())
        if missing_wv0:
            logger.error(f"    Wavelength zero-point keywords missing: {missing_wv0}. FAIL")
        else:
            logger.info(f"    Wavelength zero-point values present:")
            logger.info(f"        WAVLEN0 = {hdu1_header.get('WAVLEN0')} nm")
            logger.info(f"        WV0_X = {hdu1_header.get('WV0_X')} pixels")
            logger.info(f"        WV0_Y = {hdu1_header.get('WV0_Y')} pixels")
            logger.info(f"        WV0_XERR = {hdu1_header.get('WV0_XERR')} pixels")
            logger.info(f"        WV0_YERR = {hdu1_header.get('WV0_YERR')} pixels")
            logger.info(f"        WV0_DIMX = {hdu1_header.get('WV0_DIMX')} pixels")
            logger.info(f"        WV0_DIMY = {hdu1_header.get('WV0_DIMY')} pixels")
            logger.info(f"    Wavelength zero-point values present. PASS")
        
        # Verify HDU5 (WAVE_ERR)
        hdu5 = hdul[5]
        data = hdu5.data
        check_dimensions(data, (81, 81), "HDU5 Data Array: 2D array containing the 2D wavelength uncertainty distribution", logger)
        #verify HDU6
        hdu6 = hdul[6]
        check_dimensions(hdu6.data, (19,), "HDU6 Data Array: containing the 1D spectral distribution", logger)
        if np.isnan(hdu6.data).any() is True:
            logger.info(f"HDU6 Data Array: Contains NANs in the data. Expected: no NANs. FAIL.")
        else:
            logger.info(f"HDU6 Data Array: No NANs in the data. Expected: no NANs. PASS.")
        if np.isinf(hdu6.data).any() is True:
            logger.info(f"HDU6 Data Array: Contains INFs in the data. Expected: no INFs. FAIL.")
        else:
            logger.info(f"HDU6 Data Array: No INFs in the data. Expected: no INFs. PASS.")
        # Verify HDU7 (error)
        hdu7 = hdul[7]
        err = hdu7.data
        check_dimensions(err, (1, 19), "HDU7 Data Array: 1D array with the corresponding spectral uncertainty", logger)
        # Verify HDU8 (dq)
        hdu8 = hdul[8]
        dq = hdu8.data
        check_dimensions(dq, (19,), "HDU8 Data Array: 1D array with the corresponding spectral data quality", logger)
        
        # Verify HDU9 (wavelength)
        hdu9 = hdul[9]
        wave = hdu9.data
        check_dimensions(wave, (19,), "HDU9 Data Array: 1D array with the corresponding wavelength", logger)
        
        # Verify HDU10 (wavelength uncertainties)
        hdu10 = hdul[10]
        wave_err = hdu10.data
        check_dimensions(wave_err, (19,), "HDU10 Data Array: 1D array with the corresponding wavelength uncertainty", logger)
        
        # Verify header keywords
        verify_header_keywords(hdul[1].header, {'DATALVL': 'L4', 'CFAMNAME' : '3', 'FSAMNAME': 'R1C2', 'DPAMNAME':'PRISM3', 'BUNIT' : 'photoelectron/s'},
                                               "spec output product", logger)
        verify_header_keywords(hdul[1].header, {'WAVLEN0', 'WV0_X', 'WV0_Y', 'WV0_DIMX', 'WV0_DIMY'},
                                               "spec output product", logger)
        verify_header_keywords(hdul[1].header, {'STARLOCX', 'STARLOCY', 'CRPIX1', 'CRPIX2', 'CTCALFN', 'FLXCALFN'},
                                               "spec output product", logger)
        verify_header_keywords(hdul[6].header, {'BUNIT' : 'photoelectron/s/bin'},
                                               "spec output product", logger)

    logger.info("")
    
    check.compare_to_mocks_hdrs(out_file)
    # ================================================================================
    # (5) Baseline Performance Checks
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 3: Baseline Performance Checks')
    logger.info('='*80)

    # Load and display spec output product results
    spec_out = Image(out_file)
    sed = spec_out.hdu_list["SPEC"].data
    wave = spec_out.hdu_list["SPEC_WAVE"].data
    logger.info(f"wavelengths: {wave} nm")
    logger.info(f"spectrum: {sed}")
    logger.info("")
    
    # Clean up temporary caldb file
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    
    return spec_out

@pytest.mark.e2e
def test_l1_to_l4(e2edata_path, e2eoutput_path):
    """Run the complete L1 to L4 non-coronographic spectroscopy data end-to-end test with recipe chaining.
    
    Args:
        e2edata_path (str): Path to test data (expects L1 files)
        e2eoutput_path (str): Output directory path for results and logs
    """
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    l1_datadir_analog = os.path.join(e2edata_path, "SPEC_NOM_sims/NON_CORON_SPEC","analog")
    #l1_datadir_pc = os.path.join(e2edata_path, "NON_CORON_SPEC_sims","pc")

    l4_outputdir = os.path.join(e2eoutput_path, "l1_to_l4_spec_noncoron_e2e")
    if os.path.exists(l4_outputdir):
        shutil.rmtree(l4_outputdir)
    os.makedirs(l4_outputdir)

    analog_outputdir = os.path.join(l4_outputdir, "analog")
    sat_spots_outputdir = os.path.join(analog_outputdir, "sat_spots")

    if not os.path.exists(analog_outputdir):
        os.makedirs(analog_outputdir)
    if not os.path.exists(sat_spots_outputdir):
        os.makedirs(sat_spots_outputdir)

    #pc_outputdir = os.path.join(l4_outputdir, "pc")
    #pc_sat_spots_outputdir = os.path.join(pc_outputdir, "sat_spots")
    
    #if not os.path.exists(pc_outputdir):
    #    os.makedirs(pc_outputdir)
    #if not os.path.exists(pc_sat_spots_outputdir):
    #    os.makedirs(pc_sat_spots_outputdir)
    
    log_file = os.path.join(l4_outputdir, 'l1_to_l4_spec_noncoron_e2e.log')
    
    # Create a new logger specifically for this test
    global logger
    logger = logging.getLogger('l1_to_l4_spec_noncoron_e2e')
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
    logger.info('L1 TO L4 NON-CORONOGRAPHIC SPECTROSCOPY DATA END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    # Run the complete end-to-end test
    try:
        logger.info('='*80)
        logger.info('ANALOG SPECTROSCOPY DATA TEST')
        logger.info('='*80)
        spec_out = run_l1_to_l4_e2e_test(l1_datadir_analog, analog_outputdir, processed_cal_path, logger)
        
        #logger.info('='*80)
        #logger.info('PC SPECTROSCOPY DATA TEST')
        #logger.info('='*80)
        #spec_out = run_l1_to_l4_e2e_test(l1_datadir_pc, pc_outputdir, processed_cal_path, logger)
        
        logger.info('='*80)
        logger.info('END-TO-END TEST COMPLETE')
        logger.info('='*80)
        
        

        print('e2e test for L1 to L4 non-coronographic spectroscopy passed')
    except Exception as e:
        logger.error('='*80)
        logger.error('END-TO-END TEST FAILED')
        logger.error('='*80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Print traceback to log
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        print(f'e2e test for L1 to L4 non-coronographic spectroscopy FAILED: {str(e)}')
        raise

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    thisfile_dir = os.path.dirname(__file__)
    e2edata_dir = '/home/ababuraj/roman/E2E_Test_Data'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l4 non-coronographic spectroscopy end-to-end test with recipe chaining")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")

    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir

    test_l1_to_l4(e2edata_dir, outputdir)

