import os
import sys
import glob
import numpy as np
import astropy.io.fits as fits
from datetime import datetime, timedelta
import logging
import pytest
import argparse
import warnings
from astropy.io.fits.verify import VerifyWarning

from corgidrp.data import Dataset, DispersionModel
from corgidrp.spec import compute_psf_centroid, calibrate_dispersion_model
from astropy.table import Table
from corgidrp.data import Image
from corgidrp.mocks import create_default_L2b_headers, rename_files_to_cgi_format
from corgidrp.walker import walk_corgidrp
from corgidrp.check import (check_filename_convention, check_dimensions, 
                           verify_hdu_count, verify_header_keywords, 
                           validate_binary_table_fields, get_latest_cal_file, compare_to_mocks_hdrs)
import corgidrp
import corgidrp.caldb as caldb



# ================================================================================
# Main Spec Prism Disp E2E Test Function
# ================================================================================

def run_spec_prism_disp_e2e_test(e2edata_path, e2eoutput_path):
    """Run the complete spectroscopy prism dispersion end-to-end test.
    
    This function consolidates all the test steps into a single linear flow
    for easier reading and understanding.
    
    Args:
        e2edata_path (str): Path to input data directory
        e2eoutput_path (str): Path to output directory
        
    Returns:
        tuple: (disp_model, coeffs, angle) from the baseline performance checks
    """
    
    # ================================================================================
    # (1) Setup Input Files
    # ================================================================================
    logger.info('='*80)
    logger.info('Pre-test: set up input files and save to disk')
    logger.info('='*80)

    # Check if input folder already contains the expected files
    existing_files = sorted(glob.glob(os.path.join(e2edata_path, 'cgi_*_l2b.fits')))
    
    if existing_files:
        logger.info(f"Found {len(existing_files)} existing L2b files in {e2edata_path}")
        logger.info("Using existing input files (skipping generation)")
        saved_files = existing_files
        l2b_dataset_with_filenames = Dataset(saved_files)
    else:
        logger.info("No existing input files found. Generating new input files...")
        
        # Load test data
        datadir = os.path.join(os.path.dirname(__file__), '../test_data/spectroscopy')
        file_path = os.path.join(datadir, "g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep_withoffsets.fits")

        assert os.path.exists(file_path), f'Test file not found: {file_path}'

        psf_array = fits.getdata(file_path, ext=0)
        psf_table = Table(fits.getdata(file_path, ext=1))

        # Create dataset with mock headers and noise
        pri_hdr, ext_hdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
        ext_hdr["DPAMNAME"] = 'PRISM3'
        ext_hdr["FSAMNAME"] = 'OPEN'

        # Add random noise for reproducibility
        np.random.seed(5)
        read_noise = 200
        noisy_data_array = (np.random.poisson(np.abs(psf_array) / 2) + 
                            np.random.normal(loc=0, scale=read_noise, size=psf_array.shape))

        # Create Image objects
        psf_images = []
        for i in range(noisy_data_array.shape[0]):
            image = Image(
                data_or_filepath=np.copy(noisy_data_array[i]),
                pri_hdr=pri_hdr.copy(),
                ext_hdr=ext_hdr.copy(),
                err=np.zeros_like(noisy_data_array[i]),
                dq=np.zeros_like(noisy_data_array[i], dtype=int)
            )
            image.ext_hdr['CFAMNAME'] = psf_table['CFAM'][i].upper().strip()
            psf_images.append(image)

        basetime = datetime.now()
        for i, img in enumerate(psf_images):
            # Set unique FILETIME for each frame to ensure unique filenames
            time_offset = timedelta(milliseconds=i)
            unique_time = basetime + time_offset
            img.ext_hdr['FILETIME'] = unique_time.isoformat()
        
        # Rename Image objects directly to CGI format
        renamed_files = rename_files_to_cgi_format(list_of_fits=psf_images, output_dir=e2edata_path, level_suffix="l2b", pattern=None)

        # Load saved files back into dataset
        saved_files = sorted(glob.glob(os.path.join(e2edata_path, 'cgi_*_l2b.fits')))
        assert len(saved_files) > 0, f'No saved L2b files found in {e2edata_path}!'

        l2b_dataset_with_filenames = Dataset(saved_files)
        logger.info(f"Generated and saved {len(saved_files)} new input files")

    logger.info('')
    
    # ================================================================================
    # (2) Validate Input Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 1: Input Image Data Format and Content')
    logger.info('='*80)

    # Validate all input images
    for i, frame in enumerate(l2b_dataset_with_filenames):
        frame_info = f"Frame {i}"
        
        check_filename_convention(getattr(frame, 'filename', None), 'cgi_*_l2b.fits', frame_info, logger)
        check_dimensions(frame.data, (81, 81), frame_info, logger)
        verify_header_keywords(frame.ext_hdr, ['CFAMNAME'], frame_info, logger)
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L2b'}, frame_info, logger)
        logger.info("")

    logger.info(f"Total input images validated: {len(l2b_dataset_with_filenames)}")
    logger.info("")
    
    # Create a temporary caldb and add the default DispersionModel calibration
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()
    
    # Scan for default calibrations
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    
    # ================================================================================
    # (3) Run Processing Pipeline
    # ================================================================================
    logger.info('='*80)
    logger.info('Running processing pipeline')
    logger.info('='*80)

    logger.info('Running e2e recipe...')
    recipe = walk_corgidrp(
        filelist=saved_files, 
        CPGS_XML_filepath="",
        outputdir=e2eoutput_path,
        template="l2b_to_spec_prism_disp.json"
    )
    logger.info("")
    
    # ================================================================================
    # (4) Validate Output Calibration Product
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Output Calibration Product Data Format and Content')
    logger.info('='*80)

    # Validate output calibration product
    cal_file = get_latest_cal_file(e2eoutput_path, '*_dpm_cal.fits', logger)
    check_filename_convention(os.path.basename(cal_file), 'cgi_*_dpm_cal.fits', "DPM calibration product", logger)

    compare_to_mocks_hdrs(cal_file)

    with fits.open(cal_file) as hdul:
        verify_hdu_count(hdul, 2, "DPM calibration product", logger)
        
        # Verify HDU0 (header only)
        hdu0 = hdul[0]
        if hdu0.data is None:
            logger.info("HDU0: Header only. Expected: header only. PASS.")
        else:
            logger.info(f"HDU0: Contains data with shape {hdu0.data.shape}. Expected: header only. FAIL.")
        
        # Verify HDU1 (binary table with required fields)
        if len(hdul) > 1:
            validate_binary_table_fields(hdul[1], ['clocking_angle', 'pos_vs_wavlen_polycoeff'], logger)
        else:
            logger.info("HDU1: Missing. Expected: HDU1 present. FAIL.")
            # Report all field failures when HDU1 is missing
            for field in ['clocking_angle', 'pos_vs_wavlen_polycoeff']:
                logger.info(f"HDU1: Field '{field}' missing. Expected: field present. FAIL.")
                if field == 'pos_vs_wavlen_polycoeff':
                    logger.info(f"HDU1: Field '{field}' missing. Expected: 64-bit float. FAIL.")
                    logger.info(f"HDU1: Field '{field}' missing. Expected: 1×4 array. FAIL.")
        
        # Verify header keywords
        if len(hdul) > 1:
            verify_header_keywords(hdul[1].header, {'DATALVL': 'CAL', 'DATATYPE': 'DispersionModel'}, "DPM calibration product", logger)

    logger.info("")
    
    # ================================================================================
    # (5) Baseline Performance Checks
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 3: Baseline Performance Checks')
    logger.info('='*80)

    # Load and display dispersion model results
    cal_file = get_latest_cal_file(e2eoutput_path, '*_dpm_cal.fits', logger)
    disp_model = DispersionModel(cal_file)

    coeffs = disp_model.pos_vs_wavlen_polycoeff
    angle = disp_model.clocking_angle
    logger.info(f"Dispersion axis orientation angle (clocking_angle): {angle} deg")
    logger.info(f"Polynomial coefficients (pos_vs_wavlen_polycoeff): {coeffs}")
    logger.info("")
    
    return disp_model, coeffs, angle



# ================================================================================
# Pytest Test Function
# ================================================================================
@pytest.mark.e2e
def test_run_end_to_end(e2edata_path, e2eoutput_path):
    """Run the complete end-to-end test.
    
    Args:
        e2edata_path (str): Path to input data directory
        e2eoutput_path (str): Output directory path for results and logs.

    """
    # Set up output directory and logging
    global logger
    
    spec_prism_outputdir = os.path.join(e2eoutput_path, "spec_prism_disp_cal_e2e")
    if not os.path.exists(spec_prism_outputdir):
        os.makedirs(spec_prism_outputdir)
    # clean out any files from a previous run
    for f in os.listdir(spec_prism_outputdir):
        file_path = os.path.join(spec_prism_outputdir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    

    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv

    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    
    # Create input_l2b subfolder for input data
    input_l2b_dir = os.path.join(spec_prism_outputdir, 'input_l2b')
    os.makedirs(input_l2b_dir, exist_ok=True)
    
    # Use proper paths for input generation and output
    input_data_dir = input_l2b_dir
    output_dir = spec_prism_outputdir
    
    log_file = os.path.join(output_dir, 'spec_prism_disp_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('spec_prism_disp_cal_e2e')
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
    logger.info('SPECTROSCOPY PRISM SCALE AND DISPERSION END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    # Run the complete end-to-end test
    disp_model, coeffs, angle = run_spec_prism_disp_e2e_test(input_data_dir, output_dir)
    
    logger.info('='*80)
    logger.info('END-TO-END TEST COMPLETE')
    logger.info('='*80)
    
    # Clean up temporary caldb file
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    
    print('e2e test for spectroscopy prism dispersion calibration passed')
    


# Run the test if this script is executed directly
if __name__ == "__main__":
    thisfile_dir = os.path.dirname(__file__)
    # Default output directory name
    outputdir = thisfile_dir
    e2edata_dir = '/Users/kevinludwick/Documents/DRP_E2E_Test_Files_v2/E2E_Test_Data'#'/Users/jmilton/Documents/CGI/E2E_Test_Data2'  # Default input data path

    ap = argparse.ArgumentParser(description="run the spectroscopy prism dispersion end-to-end test")
    ap.add_argument("-i", "--e2edata_dir", default=e2edata_dir,
                    help="directory to get input files from [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    
    # Run the e2e test with the same nested structure logic
    test_run_end_to_end(args.e2edata_dir, args.outputdir)


