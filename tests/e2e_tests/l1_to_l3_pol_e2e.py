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
import corgidrp.astrom as astrom
import shutil
import logging
import traceback
from corgidrp.check import (check_filename_convention, check_dimensions, 
                           verify_hdu_count, verify_header_keywords, 
                           get_latest_cal_file)

thisfile_dir = os.path.dirname(__file__) # this file's folder

def fix_headers(
    list_of_fits,
    ):
    """ 
    Gets around EMGAIN_A being set to 1 in TVAC data and fixes string header values.
    Also adds missing EACQ_ROW/COL headers for L2b files if needed.
    
    Args:
        list_of_fits (list): list of FITS files that need to be updated.
    """
    for file in list_of_fits:
        with fits.open(file, mode='update') as fits_file:
            exthdr = fits_file[1].header
            if 'EMGAIN_A' in exthdr and float(exthdr['EMGAIN_A']) == 1:
                exthdr['EMGAIN_A'] = -1 
            if 'EMGAIN_C' in exthdr and type(exthdr['EMGAIN_C']) is str:
                exthdr['EMGAIN_C'] = float(exthdr['EMGAIN_C'])
            
            # TO DO: flag sims bug that misspells EACQ_ROW/COL
            if exthdr.get('DATALVL') == 'L2b':
                naxis1 = exthdr.get('NAXIS1', 1024)
                naxis2 = exthdr.get('NAXIS2', 1024)
                if 'EACQ_ROW' not in exthdr or exthdr['EACQ_ROW'] == 0:
                    exthdr['EACQ_ROW'] = naxis2 // 2
                if 'EACQ_COL' not in exthdr or exthdr.get('EACQ_COL', 0) == 0:
                    exthdr['EACQ_COL'] = naxis1 // 2


def run_l1_to_l3_e2e_test(l1_datadir, l3_outputdir, processed_cal_path, logger):
    """Run the complete L1 to L3 polarimetry data end-to-end test.
    
    Args:
        l1_datadir (str): Path to L1 input data directory
        l3_outputdir (str): Path to output directory
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
    calibrations_dir = os.path.join(l3_outputdir, 'calibrations')
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
    l1_files = [f for f in all_files if f.endswith('l1_.fits')]
    if len(l1_files) >= 2:
        mock_cal_filelist = [os.path.join(l1_datadir, l1_files[i]) for i in [-2, -1]]
    else:
        mock_cal_filelist = [os.path.join(l1_datadir, f) for f in l1_files]
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
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
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
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[flat], output_dir=calibrations_dir, level_suffix="flt_cal")
    this_caldb.create_entry(flat)

    # Bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[bp_map], output_dir=calibrations_dir, level_suffix="bpm_cal")
    this_caldb.create_entry(bp_map)

    # Create mock astrometric calibration
    field_path = os.path.join(os.path.dirname(__file__), "..", "test_data", "JWST_CALFIELD2020.csv")
    astrom_input_dir = os.path.join(l3_outputdir, 'astrom_cal_input')
    if not os.path.exists(astrom_input_dir):
        os.makedirs(astrom_input_dir)
    
    mock_dataset = mocks.create_astrom_data(field_path=field_path, filedir=None)
    mock_dataset.save(filedir=astrom_input_dir)
    astrom_cal = astrom.boresight_calibration(input_dataset=mock_dataset, field_path=field_path, find_threshold=5)
    mocks.rename_files_to_cgi_format(list_of_fits=[astrom_cal], output_dir=calibrations_dir, level_suffix="ast_cal")
    this_caldb.create_entry(astrom_cal)
    
    logger.info("Created calibration products:")
    logger.info(f"  - NonLinearityCalibration: {nonlinear_cal.filename}")
    logger.info(f"  - KGain: {kgain.filename}")
    logger.info(f"  - DetectorNoiseMaps: {noise_map.filename}")
    if sample_l1_files:
        logger.info(f"  - Dark: {dark_cal.filename}")
    logger.info(f"  - FlatField: {flat.filename}")
    logger.info(f"  - BadPixelMap: {bp_map.filename}")
    logger.info(f"  - AstrometricCalibration: {astrom_cal.filename}")
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
    input_data_dir = os.path.join(l3_outputdir, 'input_l1')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    # Copy files to input_data directory and update file list
    input_data_filelist = [
        shutil.copy2(file_path, os.path.join(input_data_dir, os.path.basename(file_path)))
        for file_path in input_data_filelist
    ] 
    
    # Validate all input images
    input_dataset = data.Dataset(input_data_filelist)
    
    for i, (frame, filepath) in enumerate(zip(input_dataset, input_data_filelist)):
        frame_info = f"L1 Input Frame {i}"
        
        check_filename_convention(os.path.basename(filepath), 'cgi_*_l1_.fits', frame_info, logger, data_level='l1_')
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L1'}, frame_info, logger)
        
        
        # Verify HDU count
        try:
            with fits.open(filepath) as hdul:
                verify_hdu_count(hdul, 4, frame_info, logger) 
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
    
    # Step 1: L1 -> L2b
    logger.info('Step 1: Running L1 to L2b recipe...')
    walker.walk_corgidrp(input_data_filelist, "", l3_outputdir, template="l1_to_l2b.json")
    
    l2b_files = [f for f in os.listdir(l3_outputdir) if f.endswith('_l2b.fits')]
    l2b_filelist = [os.path.join(l3_outputdir, f) for f in l2b_files]
    logger.info(f'L1 to L2b complete. Generated {len(l2b_filelist)} L2b files.')
    logger.info('')
    
    # Step 2: L2b -> L3 
    logger.info('Step 2: Running L2b to L3 polarimetry recipe...')
    walker.walk_corgidrp(l2b_filelist, "", l3_outputdir, template="l2b_to_l3_pol.json")
    logger.info('L2b to L3 complete.')
    logger.info('')
    
    # ================================================================================
    # (4) Validate Output L3 Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Output L3 Image Data Format and Content')
    logger.info('='*80)
    
    # Filter out calibration files and only get L3 data files
    all_files = [f for f in os.listdir(l3_outputdir) if f.endswith('.fits')]
    new_l3_filenames = [os.path.join(l3_outputdir, f) for f in all_files if '_l3' in f and '_cal' not in f]

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
            img = data.Image(l3_filename)
            
            # Verify filename
            check_filename_convention(os.path.basename(l3_filename), 'cgi_*_l3_.fits', frame_info, logger, data_level='l3_')
            
            # Verify HDU count
            with fits.open(l3_filename) as hdul:
                verify_hdu_count(hdul, 5, frame_info, logger)  # L3 should have 5 HDUs
            
            # Verify data level
            verify_header_keywords(img.ext_hdr, {'DATALVL': 'L3'}, frame_info, logger)
            
            # Check this is polarimetry data
            dpam = img.ext_hdr.get('DPAMNAME', '')
            if dpam in ('POL0', 'POL45'):
                logger.info(f"{frame_info}: DPAMNAME = '{dpam}' (polarimetry). PASS")
            else:
                logger.info(f"{frame_info}: DPAMNAME = '{dpam}'. Expected POL0 or POL45. FAIL")
            
            # Check data dimensions - should always be polarimetry datacube (2, N, N)
            if len(img.data.shape) == 3 and img.data.shape[0] == 2:
                logger.info(f"{frame_info}: Polarimetry datacube shape {img.data.shape}. PASS")
            else:
                logger.info(f"{frame_info}: Expected polarimetry datacube (2, N, N), got {img.data.shape}. FAIL")
            
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
    
    logger.info(f"Total output L3 images validated: {len(new_l3_filenames)}")
    logger.info('')
    
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)
    
    return new_l3_filenames


@pytest.mark.e2e
def test_l2b_to_l3(e2edata_path, e2eoutput_path):
    """Run the complete L1 to L3 polarimetry data end-to-end test with recipe chaining.
    
    Args:
        e2edata_path (str): Path to test data (expects L1 files)
        e2eoutput_path (str): Output directory path for results and logs
    """
    # Set up output directory
    l3_outputdir = os.path.join(e2eoutput_path, "l1_to_l3_pol_e2e")
    if os.path.exists(l3_outputdir):
        shutil.rmtree(l3_outputdir)
    os.makedirs(l3_outputdir)

    analog_outputdir = os.path.join(l3_outputdir, "analog")
    pc_outputdir = os.path.join(l3_outputdir, "pc")
    if not os.path.exists(analog_outputdir):
        os.makedirs(analog_outputdir)
    if not os.path.exists(pc_outputdir):
        os.makedirs(pc_outputdir)
    
    log_file = os.path.join(l3_outputdir, 'l1_to_l3_pol_e2e.log')
    
    # Create a new logger specifically for this test
    global logger
    logger = logging.getLogger('l1_to_l3_pol_e2e')
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
    analog_datadir = os.path.join(e2edata_path, "POL_sims", "L1", "analog_data")
    pc_datadir = os.path.join(e2edata_path, "POL_sims", "L1", "PC_data")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    
    # Run the complete end-to-end test
    try:
        logger.info('='*80)
        logger.info('ANALOG POLARIMETRY DATA TEST')
        logger.info('='*80)
        new_l3_analog_filenames = run_l1_to_l3_e2e_test(analog_datadir, analog_outputdir, processed_cal_path, logger)
        
        logger.info('='*80)
        logger.info('PC POLARIMETRY DATA TEST')
        logger.info('='*80)
        new_l3_pc_filenames = run_l1_to_l3_e2e_test(pc_datadir, pc_outputdir, processed_cal_path, logger)
        
        logger.info('='*80)
        logger.info('END-TO-END TEST COMPLETE')
        logger.info('='*80)
        
        print('e2e test for L1 to L3 polarimetry passed')
    except Exception as e:
        logger.error('='*80)
        logger.error('END-TO-END TEST FAILED')
        logger.error('='*80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Print traceback to log
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        print(f'e2e test for L1 to L3 polarimetry FAILED: {str(e)}')
        raise

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l3 polarimetry end-to-end test with recipe chaining")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")

    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir

    test_l2b_to_l3(e2edata_dir, outputdir)

