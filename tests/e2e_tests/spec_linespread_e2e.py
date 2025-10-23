import os
import glob
import numpy as np
import astropy.io.fits as fits
from datetime import datetime, timedelta
import logging
import pytest
import argparse
import warnings
from astropy.io.fits.verify import VerifyWarning

from corgidrp.data import Dataset, LineSpread, DispersionModel
from corgidrp.data import Image
from corgidrp.mocks import create_default_L2b_headers
from corgidrp.walker import walk_corgidrp
import corgidrp
import corgidrp.caldb as caldb
from corgidrp.check import (check_filename_convention, check_dimensions, 
                           verify_hdu_count, verify_header_keywords, 
                           get_latest_cal_file)



# ================================================================================
# Main Spec Linespread E2E Test Function
# ================================================================================

def run_spec_linespread_e2e_test(e2edata_path, e2eoutput_path):
    """Run the complete spectroscopy linespread end-to-end test.
    
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
        file_path_spot = os.path.join(datadir, "g0v_vmag6_spc-spec_band3_unocc_CFAM3d_R1C2SLIT_PRISM3_offset_array.fits")

        assert os.path.exists(file_path_spot), f'Test file not found: {file_path_spot}'

        psf_array_spot = fits.getdata(file_path_spot, ext=0)
        # Create dataset with mock headers and noise
        pri_hdr, ext_hdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
        ext_hdr["DPAMNAME"] = 'PRISM3'
        ext_hdr["FSAMNAME"] = 'R1C2'
        # Add random noise for reproducibility
        np.random.seed(5)
        read_noise = 200
        noisy_data_array = (np.random.poisson(np.abs(psf_array_spot) / 2) + 
                            np.random.normal(loc=0, scale=read_noise, size=psf_array_spot.shape))
        
        # Create Image objects
        psf_images = []
        for i in range(int(len(psf_array_spot)/2)):
            image_spot = Image(
                data_or_filepath=np.copy(noisy_data_array[i]),
                pri_hdr=pri_hdr.copy(),
                ext_hdr=ext_hdr.copy(),
                err=np.zeros_like(noisy_data_array[i]),
                dq=np.zeros_like(noisy_data_array[i], dtype=int)
            )
            image_spot.ext_hdr["CFAMNAME"] = "3D"
            psf_images.append(image_spot)
        
        # Save images to disk with timestamped filenames
        def get_formatted_filename(pri_hdr, dt, suffix="l2b"):
            visitid = pri_hdr.get('VISITID', '0000000000000000000')
            now = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
            return f"cgi_{visitid}_{now}_{suffix}.fits"

        basetime = datetime.now()
        for i, img in enumerate(psf_images):
            fname = get_formatted_filename(img.pri_hdr, basetime + timedelta(seconds=i), suffix="l2b")
            fpath = os.path.join(e2edata_path, fname)
            
            # Save as FITS
            primary_hdu = fits.PrimaryHDU(header=img.pri_hdr)
            image_hdu = fits.ImageHDU(data=img.data, header=img.ext_hdr)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=VerifyWarning)
                fits.HDUList([primary_hdu, image_hdu]).writeto(fpath, overwrite=True)

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
        frame_info = f"L2b Frame {i}"
        
        check_filename_convention(getattr(frame, 'filename', None), 'cgi_*_l2b.fits', frame_info, logger)
        check_dimensions(frame.data, (81, 81), frame_info, logger)
        verify_header_keywords(frame.ext_hdr, {'DPAMNAME', 'CFAMNAME', 'FSAMNAME'}, frame_info, logger)
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
        template="l2b_to_spec_linespread.json"
    )
    logger.info("")
    
    # ================================================================================
    # (4) Validate Output Calibration Product
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Output Calibration Product Data Format and Content')
    logger.info('='*80)

    # Validate output calibration product
    cal_file = get_latest_cal_file(e2eoutput_path, '*_line_spread.fits', logger)
    check_filename_convention(os.path.basename(cal_file), 'cgi_*_line_spread.fits', "LineSpread calibration product", logger)

    with fits.open(cal_file) as hdul:
        verify_hdu_count(hdul, 3, "linespread calibration product", logger)
        
        # Verify HDU0 (header only)
        hdu0 = hdul[0]
        if hdu0.data is None:
            logger.info("HDU0: Header only. Expected: header only. PASS.")
        else:
            logger.info(f"HDU0: Contains data with shape {hdu0.data.shape}. Expected: header only. FAIL.")
        #verify HDU1
        hdu1 = hdul[1]
        check_dimensions(hdu1.data, (2,19), "HDU1 Data Array: containing the 1D wavelengths and the line spread function", logger)
        if np.isnan(hdu1.data).any() is True:
            logger.info(f"HDU1 Data Array: Contains NANs in the data. Expected: no NANs. FAIL.")
        else:
            logger.info(f"HDU1 Data Array: No NANs in the data. Expected: no NANs. PASS.")
        if np.isinf(hdu1.data).any() is True:
            logger.info(f"HDU1 Data Array: Contains INFs in the data. Expected: no INFs. FAIL.")
        else:
            logger.info(f"HDU1 Data Array: No INFs in the data. Expected: no INFs. PASS.")
        #verify that the line spread function is normalized to 1
        if np.float32(np.sum(hdu1.data[1,:])) != 1.:
            logger.info(f"HDU1 Data Array: sum of the line spread function is not approx. 1. Expected: line spread function normalized to 1. FAIL.")
        else:
            logger.info(f"HDU1 Data Array: sum of the line spread function is approx. 1. Expected: line spread function normalized to 1. PASS.")
        # Verify HDU2 (Gaussian parameters)
        gauss_par = hdul[2].data
        check_dimensions(gauss_par, (6,), "HDU2 Data Array: 1D array with the Gaussian fit parameters", logger)
        
        # Verify header keywords
        verify_header_keywords(hdul[1].header, {'DATALVL': 'CAL', 'DATATYPE': 'LineSpread', 'CFAMNAME' : '3D', 'FSAMNAME': 'R1C2', 'DPAMNAME':'PRISM3'}, "linespread calibration product", logger)

    logger.info("")
    
    # ================================================================================
    # (5) Baseline Performance Checks
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 3: Baseline Performance Checks')
    logger.info('='*80)

    # Load and display dispersion model results
    linespread = LineSpread(cal_file)
    wavlens = linespread.data[0, :]
    flux_profile = linespread.data[1, :]
    logger.info(f"wavelengths: {wavlens} nm")
    logger.info(f"flux profile: {flux_profile}")
    logger.info(f"Gaussian fit parameters:") 
    logger.info(f"amplitude: {linespread.amplitude} +- {linespread.amp_err}")
    logger.info(f"mean_wave: {linespread.mean_wave} +- {linespread.wave_err} nm")
    logger.info(f"fwhm: {linespread.fwhm} +- {linespread.fwhm_err} nm")
    logger.info("")
    
    # Clean up temporary caldb file
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    
    return wavlens, flux_profile, gauss_par



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
    
    # Create the spec_linespread_e2e subfolder regardless
    input_top_level = os.path.join(e2edata_path, 'spec_linespread_e2e')
    output_top_level = os.path.join(e2eoutput_path, 'spec_linespread_e2e')
    
    os.makedirs(input_top_level, exist_ok=True)
    os.makedirs(output_top_level, exist_ok=True)
    
    # Update paths to use the subfolder structure
    e2edata_path = input_top_level
    e2eoutput_path = output_top_level
    
    log_file = os.path.join(e2eoutput_path, 'spec_linespread_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('spec_linespread_e2e')
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
    logger.info('SPECTROSCOPY LINESPREAD FUNCTION END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    # Run the complete end-to-end test
    linespread = run_spec_linespread_e2e_test(e2edata_path, e2eoutput_path)
    
    logger.info('='*80)
    logger.info('END-TO-END TEST COMPLETE')
    logger.info('='*80)
    


# Run the test if this script is executed directly
if __name__ == "__main__":
    thisfile_dir = os.path.dirname(__file__)
    # Create top-level spec_linespread_e2e folder
    top_level_dir = os.path.join(thisfile_dir, 'spec_linespread_e2e')
    outputdir = os.path.join(top_level_dir, 'output')
    e2edata_dir = os.path.join(top_level_dir, 'input_data')

    ap = argparse.ArgumentParser(description="run the spectroscopy linespread end-to-end test")
    ap.add_argument("-i", "--e2edata_dir", default=e2edata_dir,
                    help="directory to get input files from [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    
    # Run the e2e test with the same nested structure logic
    test_run_end_to_end(args.e2edata_dir, args.outputdir)


