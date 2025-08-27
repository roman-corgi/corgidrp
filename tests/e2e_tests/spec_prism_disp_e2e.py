import os
import sys
import glob
import numpy as np
import astropy.io.fits as fits
from datetime import datetime, timedelta
import logging
import pytest
import argparse

from corgidrp.data import Dataset, DispersionModel
from corgidrp.spec import compute_psf_centroid, calibrate_dispersion_model
from astropy.table import Table
from corgidrp.data import Image
from corgidrp.mocks import create_default_L2b_headers
from corgidrp.walker import walk_corgidrp

# ================================================================================
# Validation functions - can be used in other E2E tests
# ================================================================================

def check_filename_convention(filename, expected_pattern, frame_info=""):
    """Check if filename follows the expected naming convention.

    Args:
        filename (str): Filename to check
        expected_pattern (str): Expected pattern (e.g., 'cgi_*_l2b.fits')
        frame_info (str): Additional info for logging (e.g., "Frame 0")

    Returns:
        bool: True if filename matches convention
    """
    if not filename:
        logger.info(f"{frame_info}: No filename. Naming convention FAIL.")
        return False
    
    # Basic pattern check
    if expected_pattern == 'cgi_*_l2b.fits':
        parts = filename.split('_')
        valid = (len(parts) >= 4 and 
                parts[0] == 'cgi' and 
                len(parts[2]) == 16 and parts[2][8] == 't' and 
                parts[2][:8].isdigit() and parts[2][9:].isdigit() and
                filename.endswith('_l2b.fits'))
    elif expected_pattern == 'cgi_*_dpm_cal.fits':
        valid = filename.startswith('cgi_') and '_dpm_cal.fits' in filename
    else:
        valid = expected_pattern in filename
    
    status = "PASS" if valid else "FAIL"
    logger.info(f"{frame_info}: Filename: {filename}. Naming convention {status}.")
    return valid

def check_dimensions(data, expected_shape, frame_info=""):
    """Check if data has expected dimensions.

    Args:
        data (numpy.ndarray): Data array to check
        expected_shape (tuple): Expected shape tuple
        frame_info (str): Additional info for logging (e.g., "Frame 0")

    Returns:
        bool: True if dimensions match
    """
    if data.shape == expected_shape:
        logger.info(f"{frame_info}: Shape={data.shape}. Expected: {expected_shape}. PASS.")
        return True
    else:
        logger.info(f"{frame_info}: Shape={data.shape}. Expected: {expected_shape}. FAIL.")
        return False

def verify_hdu_count(hdul, expected_count, frame_info=""):
    """Verify that the number of HDUs in the FITS file is as expected.

    Args:
        hdul (astropy.io.fits.HDUList): FITS HDUList object
        expected_count (int): Expected number of HDUs
        frame_info (str): Additional info for logging (e.g., "Frame 0")

    Returns:
        bool: True if HDU count matches expected count
    """
    if len(hdul) == expected_count:
        logger.info(f"{frame_info}: HDU count={len(hdul)}. Expected: {expected_count}. PASS.")
        return True

def verify_header_keywords(header, required_keywords, frame_info=""):
    """Verify that required header keywords are present and have expected values.

    Args:
        header (astropy.io.fits.Header): FITS header object
        required_keywords (dict or list): Dictionary of {keyword: expected_value} or list of keywords
        frame_info (str): Additional info for logging (e.g., "Frame 0")

    Returns:
        bool: True if all keywords are valid
    """
    all_valid = True
    
    if isinstance(required_keywords, dict):
        # Check keyword-value pairs
        for keyword, expected_value in required_keywords.items():
            if keyword not in header:
                logger.error(f"{frame_info}: Missing required keyword {keyword}!")
                all_valid = False
            else:
                actual_value = header[keyword]
                if actual_value == expected_value:
                    logger.info(f"{frame_info}: {keyword}={actual_value}. Expected {keyword}: {expected_value}. PASS.")
                else:
                    logger.info(f"{frame_info}: {keyword}={actual_value}. Expected {keyword}: {expected_value}. FAIL.")
                    all_valid = False
    else:
        # Just check if keywords exist
        for keyword in required_keywords:
            if keyword not in header:
                logger.error(f"{frame_info}: Missing required keyword {keyword}!")
                all_valid = False
            else:
                logger.info(f"{frame_info}: {keyword}={header[keyword]},")
    
    return all_valid

def validate_binary_table_fields(hdu1, required_fields):
    """Validate binary table fields with consistent error reporting.

    Args:
        hdu1 (astropy.io.fits.BinTableHDU): FITS binary table HDU
        required_fields (list): List of required field names

    Returns:
        bool: True if all fields are valid
    """
    if isinstance(hdu1, fits.BinTableHDU):
        logger.info("HDU1: Binary table format. Expected: BinTableHDU. PASS.")
        
        for field in required_fields:
            if field in hdu1.data.names:
                data = hdu1.data[field]
                # Check if dtype is 64-bit float (ignoring endianness)
                is_float64 = (data.dtype.kind == 'f' and data.dtype.itemsize == 8)
                status = "PASS" if is_float64 else "FAIL"
                logger.info(f"HDU1: Table field '{field}' present. Data type {data.dtype}. Expected: 64-bit float. {status}.")
                
                # Additional shape validation for polynomial coefficients
                if field == 'pos_vs_wavlen_polycoeff':
                    expected_shape = (1, 4)
                    shape_status = "PASS" if data.shape == expected_shape else "FAIL"
                    logger.info(f"HDU1: {field} shape {data.shape}. Expected: {expected_shape}. {shape_status}.")
            else:
                logger.info(f"HDU1: Field '{field}' missing. Expected: field present. FAIL.")
        return True
    else:
        logger.info(f"HDU1: Format {type(hdu1)}. Expected: BinTableHDU. FAIL.")
        # Report all field failures when format is wrong
        for field in required_fields:
            logger.info(f"HDU1: Field '{field}' missing. Expected: field present. FAIL.")
            if field == 'pos_vs_wavlen_polycoeff':
                logger.info(f"HDU1: Field '{field}' missing. Expected: 64-bit float. FAIL.")
                logger.info(f"HDU1: Field '{field}' missing. Expected: 1×4 array. FAIL.")
        return False

def get_latest_cal_file(e2eoutput_path, pattern):
    """Get the most recent calibration file matching the pattern.

    Args:
        e2eoutput_path (str): Directory to search for calibration files
        pattern (str): Pattern to match (e.g., '*_dpm_cal.fits')

    Returns:
        str: Path to the most recent calibration file
    """
    cal_files = sorted(glob.glob(os.path.join(e2eoutput_path, pattern)), key=os.path.getmtime, reverse=True)
    assert len(cal_files) > 0, f'No {pattern} files found in {e2eoutput_path}!'
    return cal_files[0]

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
            image.ext_hdr['CFAMNAME'] = psf_table['CFAM'][i]
            psf_images.append(image)

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
        frame_info = f"Frame {i}"
        
        check_filename_convention(getattr(frame, 'filename', None), 'cgi_*_l2b.fits', frame_info)
        check_dimensions(frame.data, (81, 81), frame_info)
        verify_header_keywords(frame.ext_hdr, ['CFAMNAME'], frame_info)
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L2b'}, frame_info)
        logger.info("")

    logger.info(f"Total input images validated: {len(l2b_dataset_with_filenames)}")
    logger.info("")
    
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
    cal_file = get_latest_cal_file(e2eoutput_path, '*_dpm_cal.fits')
    check_filename_convention(os.path.basename(cal_file), 'cgi_*_dpm_cal.fits', "DPM calibration product")

    with fits.open(cal_file) as hdul:
        verify_hdu_count(hdul, 2, "DPM calibration product")
        
        # Verify HDU0 (header only)
        hdu0 = hdul[0]
        if hdu0.data is None:
            logger.info("HDU0: Header only. Expected: header only. PASS.")
        else:
            logger.info(f"HDU0: Contains data with shape {hdu0.data.shape}. Expected: header only. FAIL.")
        
        # Verify HDU1 (binary table with required fields)
        if len(hdul) > 1:
            validate_binary_table_fields(hdul[1], ['clocking_angle', 'pos_vs_wavlen_polycoeff'])
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
            verify_header_keywords(hdul[1].header, {'DATALVL': 'CAL', 'DATATYPE': 'DispersionModel'}, "DPM calibration product")

    logger.info("")
    
    # ================================================================================
    # (5) Baseline Performance Checks
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 3: Baseline Performance Checks')
    logger.info('='*80)

    # Load and display dispersion model results
    cal_file = get_latest_cal_file(e2eoutput_path, '*_dpm_cal.fits')
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

    Returns:
        tuple: (disp_model, coeffs, angle)
    """
    # Set up output directory and logging
    global logger
    
    # Create the spec_prism_disp_e2e subfolder regardless
    input_top_level = os.path.join(e2edata_path, 'spec_prism_disp_e2e')
    output_top_level = os.path.join(e2eoutput_path, 'spec_prism_disp_e2e')
    
    os.makedirs(input_top_level, exist_ok=True)
    os.makedirs(output_top_level, exist_ok=True)
    
    # Update paths to use the subfolder structure
    e2edata_path = input_top_level
    e2eoutput_path = output_top_level
    
    log_file = os.path.join(e2eoutput_path, 'spec_prism_disp_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('spec_prism_disp_e2e')
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
    disp_model, coeffs, angle = run_spec_prism_disp_e2e_test(e2edata_path, e2eoutput_path)
    
    logger.info('='*80)
    logger.info('END-TO-END TEST COMPLETE')
    logger.info('='*80)
    
    return disp_model, coeffs, angle


# Run the test if this script is executed directly
if __name__ == "__main__":
    thisfile_dir = os.path.dirname(__file__)
    # Create top-level spec_prism_disp_e2e folder
    top_level_dir = os.path.join(thisfile_dir, 'spec_prism_disp_e2e')
    outputdir = os.path.join(top_level_dir, 'output')
    e2edata_dir = os.path.join(top_level_dir, 'input_data')

    ap = argparse.ArgumentParser(description="run the spectroscopy prism dispersion end-to-end test")
    ap.add_argument("-i", "--e2edata_dir", default=e2edata_dir,
                    help="directory to get input files from [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    
    # Run the e2e test with the same nested structure logic
    test_run_end_to_end(args.e2edata_dir, args.outputdir)


