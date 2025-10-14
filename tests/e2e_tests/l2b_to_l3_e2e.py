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


def run_l2b_to_l3_e2e_test(l2b_datadir, l3_outputdir, cals_dir, logger):
    """Run the complete L2b to L3 end-to-end test.
    
    Args:
        l2b_datadir (str): Path to L2b input data directory
        l3_outputdir (str): Path to output directory
        cals_dir (str or None): Path to calibrations directory (None = use mocks)
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

    if cals_dir is None:
        logger.info("Creating mock astrometric calibration...")
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
        logger.info(f"Mock astrometric calibration created: {astrom_cal.filename}")
    else:
        logger.info(f"Loading astrometric calibration from {cals_dir}...")
        # Search for files with ast_cal filename pattern
        astrom_files = [f for f in os.listdir(cals_dir) if "ast_cal" in f.lower()]
        if not astrom_files:
            raise FileNotFoundError(f"No file containing 'ast_cal' found in {cals_dir}")
        astrom_path = os.path.join(cals_dir, astrom_files[0])
        astrom_cal = data.AstrometricCalibration(astrom_path)
        this_caldb.create_entry(astrom_cal)
        logger.info(f"Loaded astrometric calibration: {os.path.basename(astrom_path)}")
    
    logger.info('')

    # ================================================================================
    # (2) Validate Input Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 1: Input L2b Image Data Format and Content')
    logger.info('='*80)

    # Filter to only include L2b files as inputs
    all_files = os.listdir(l2b_datadir)
    l2b_files_only = [f for f in all_files if f.endswith('l2b.fits')]
    if not l2b_files_only:
        raise FileNotFoundError(f"No files ending in 'l2b.fits' found in {l2b_datadir}")
    
    l2b_data_filelist = [os.path.join(l2b_datadir, f) for f in l2b_files_only]
    
    # Create input_data subfolder
    input_data_dir = os.path.join(l3_outputdir, 'input_l2b')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    # Copy files to input_data directory and update file list
    l2b_data_filelist = [
        shutil.copy2(file_path, os.path.join(input_data_dir, os.path.basename(file_path)))
        for file_path in l2b_data_filelist
    ] 

    # fix headers
    fix_headers(l2b_data_filelist)
    
    # Validate all input images
    l2b_dataset = data.Dataset(l2b_data_filelist)
    for i, (frame, filepath) in enumerate(zip(l2b_dataset, l2b_data_filelist)):
        frame_info = f"L2b Input Frame {i}"
        
        check_filename_convention(os.path.basename(filepath), 'cgi_*_l2b.fits', frame_info, logger, data_level='l2b')
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L2b'}, frame_info, logger)
        
        # Verify HDU count
        try:
            with fits.open(filepath) as hdul:
                verify_hdu_count(hdul, 5, frame_info, logger)  # L2b should have 5 HDUs
        except Exception as e:
            logger.info(f"{frame_info}: HDU count verification failed. Error: {str(e)}. FAIL")
        
        # Check dimensions (L2b images can vary in size)
        logger.info(f"{frame_info}: Data shape {frame.data.shape}")
        
        logger.info("")
    
    logger.info(f"Total input images validated: {len(l2b_dataset)}")
    logger.info('')
    
    # ================================================================================
    # (3) Run Processing Pipeline
    # ================================================================================
    logger.info('='*80)
    logger.info('Running L2b -> L3 processing pipeline')
    logger.info('='*80)

    logger.info('Running L2b to L3 recipe...')
    walker.walk_corgidrp(l2b_data_filelist, "", l3_outputdir)
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
            
            # Check data dimensions (will just report dimensions)
            check_dimensions(img.data, (125,125), frame_info, logger)
            
            # Verify WCS headers exist (from create_wcs step)
            wcs_keys = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2']
            missing_wcs = [k for k in wcs_keys if k not in img.ext_hdr]
            if not missing_wcs:
                logger.info(f"{frame_info}: WCS headers present ({', '.join(wcs_keys)}). PASS")
            else:
                logger.info(f"{frame_info}: WCS headers incomplete. Missing: {', '.join(missing_wcs)}. FAIL")
            
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
    
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)
    
    return new_l3_filenames


@pytest.mark.e2e
def test_l2b_to_l3(e2edata_path, e2eoutput_path, input_datadir, cals_dir):
    """Run the complete L2b to L3 end-to-end test.
    
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
        l2b_datadir = os.path.join(e2edata_path, "SPEC_targetstar_slit_prism", "L2b")
    else:
        l2b_datadir = input_datadir
    
    l3_outputdir = os.path.join(e2eoutput_path, "l2b_to_l3_e2e")
    if os.path.exists(l3_outputdir):
        shutil.rmtree(l3_outputdir)
    os.makedirs(l3_outputdir)
    
    log_file = os.path.join(l3_outputdir, 'l2b_to_l3_e2e.log')
    
    # Create a new logger specifically for this test
    logger = logging.getLogger('l2b_to_l3_e2e')
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
    logger.info('L2B TO L3 END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    # Run the complete end-to-end test
    try:
        new_l3_filenames = run_l2b_to_l3_e2e_test(l2b_datadir, l3_outputdir, cals_dir, logger)
        
        logger.info('='*80)
        logger.info('END-TO-END TEST COMPLETE - ALL TESTS PASSED')
        logger.info('='*80)
        
        print('e2e test for L2b to L3 passed')
    except Exception as e:
        logger.error('='*80)
        logger.error('END-TO-END TEST FAILED')
        logger.error('='*80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Print traceback to log
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        print(f'e2e test for L2b to L3 FAILED: {str(e)}')
        raise

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l2b->l3 end-to-end test")
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
    test_l2b_to_l3(e2edata_dir, outputdir, input_datadir, cals_dir)

