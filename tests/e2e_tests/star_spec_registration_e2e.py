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
from corgidrp.spec import star_spec_registration
from astropy.table import Table
from corgidrp.data import Image
from corgidrp.mocks import create_default_L2b_headers
from corgidrp.walker import walk_corgidrp

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

def run_star_spec_registration_e2e_test(e2edata_path, e2eoutput_path):
    """Run the complete star spectrum registration end-to-end test.
    
    This function consolidates all the test steps into a single linear flow
    for easier reading and understanding.
    
    Args:
        e2edata_path (str): Path to input data directory
        e2eoutput_path (str): Path to output directory
        
    Returns:
        Dispersed star image whose PSF-to-FSAM slit alignment most closely matches
      that of the target source
    """
    
    # ================================================================================
    # (1) Setup Input Files
    # ================================================================================
    logger.info('='*80)
    logger.info('Pre-test: set up input files and save to disk')
    logger.info('='*80)

    # CFAM. Accommodate band2 wherever its is straightforward
    cfam_test = '3F'

    # Check if input folder already contains the expected files
    existing_files_template = sorted(glob.glob(os.path.join(e2edata_path,
        'template', 'cgi_*_l2b.fits')))
    existing_files_data = sorted(glob.glob(os.path.join(e2edata_path,
        'data', 'cgi_*_l2b.fits')))
    
    if existing_files_template and existing_files_data:
        logger.info(f'Found {len(existing_files_template)} existing L2b files in {e2edata_path}/template/')
        logger.info(f'Found {len(existing_files_template)} existing L2b files in {e2edata_path}/data/')
        logger.info('Using existing input files (skipping generation)')
        dataset_template = Dataset(existing_files_template)
        dataset_data = Dataset(existing_files_data)
    else:
        logger.info('No existing input files found. Generating new input files...')
        
        # Load test data
        test_datadir = os.path.join(os.path.dirname(__file__), '../test_data/spectroscopy')

        # Create some mock data for the template with spectra
        file_path = os.path.join(test_datadir,
                'g0v_vmag6_spc-spec_band3_unocc_CFAM3_R1C2SLIT_PRISM3_offset_array.fits')
        assert os.path.exists(file_path), f'Test FITS file not found: {file_path}'

        with fits.open(file_path) as hdul:
            psf_array = hdul[0].data
            psf_table = Table(hdul[1].data)

        # Create dataset with mock headers and noise
        pri_hdr, ext_hdr = create_default_L2b_headers()[0:2]

        # Instrumental setup
        cfam_name = '3F'
        dpam_name = 'PRISM3'
        spam_name = 'SPEC'
        lsam_name = 'SPEC'
        fsam_name = 'R1C2'
        fpam_name = 'ND225'

        # Seeded random generator
        rng = np.random.default_rng(seed=0)
        # Choose (arbitrarily) which template will be the correct stellar spectrum
        slit_ref = 4

        # Add an initial guess of where the centroid is found
        initial_cent = {
            'xcent': np.array(psf_table['xcent']),
            'ycent': np.array(psf_table['ycent'])
        }

        # Add wavelength zero-point. In this test, we set it in a way that
        # matches one of the slices, so that we can predict which one is the
        # best image later
        ext_hdr['WV0_X'] = initial_cent['xcent'][slit_ref]
        ext_hdr['WV0_Y'] = initial_cent['ycent'][slit_ref]

        # Update Setup header key values
        ext_hdr['CFAMNAME'] = cfam_name
        ext_hdr['DPAMNAME'] = dpam_name
        ext_hdr['SPAMNAME'] = spam_name
        ext_hdr['LSAMNAME'] = lsam_name
        ext_hdr['FSAMNAME'] = fsam_name
        ext_hdr['FPAMNAME'] = fpam_name
        template_images = []
        data_images = []
        for i in range(psf_array.shape[0]):
            data_2d = np.copy(psf_array[i])
            err = np.zeros_like(data_2d)
            dq = np.zeros_like(data_2d, dtype=int)
            # Store template data as a DRP object
            image_template = Image(
                data_or_filepath=data_2d,
                pri_hdr=pri_hdr,
                ext_hdr=ext_hdr,
                err=err,
                dq=dq
            )
            template_images.append(image_template)
            # Some noisy version for the simulated data without blowing it
            # unreasonably. The one with slit_ref has no additional
            # noise to test that this is the one outputted by star_spec_registration()
            # Collected data have different FSM values
            ext_hdr['FSMX'] = i // 5
            ext_hdr['FSMY'] = i - 5 * (i // 5)
            image_data = Image(
                data_or_filepath=data_2d + rng.normal(0,
                    0.1*np.abs(i-slit_ref)*data_2d.std(), data_2d.shape),
                pri_hdr=pri_hdr,
                ext_hdr=ext_hdr,
                err=err,
                dq=dq
            )
            data_images.append(image_data)
        # Save images to disk with timestamped filenames
        def get_formatted_filename(pri_hdr, dt, suffix="l2b"):
            visitid = pri_hdr.get('VISITID', '0000000000000000000')
            now = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
            return f'cgi_{visitid}_{now}_{suffix}.fits'

        basetime = datetime.now()
        for i, img in enumerate(template_images):
            fname = get_formatted_filename(img.pri_hdr, basetime + timedelta(seconds=i), suffix='l2b')
            fpath = os.path.join(e2edata_path, 'template', fname)
            # Save as FITS
            primary_hdu = fits.PrimaryHDU(header=img.pri_hdr)
            image_hdu = fits.ImageHDU(data=img.data, header=img.ext_hdr)
            fits.HDUList([primary_hdu, image_hdu]).writeto(fpath, overwrite=True)
        for i, img in enumerate(data_images):
            fname = get_formatted_filename(img.pri_hdr, basetime + timedelta(seconds=i), suffix='l2b')
            fpath = os.path.join(e2edata_path, 'data', fname)
            # Save as FITS
            primary_hdu = fits.PrimaryHDU(header=img.pri_hdr)
            image_hdu = fits.ImageHDU(data=img.data, header=img.ext_hdr)
            fits.HDUList([primary_hdu, image_hdu]).writeto(fpath, overwrite=True)        

        # Load saved files back into dataset
        saved_template_files = sorted(glob.glob(os.path.join(e2edata_path, 'template', 'cgi_*_l2b.fits')))
        assert len(saved_template_files) > 0, f'No saved L2b files found in {e2edata_path}/template/!'

        dataset_template = Dataset(saved_template_files)
        logger.info(f"Generated and saved {len(saved_template_files)} new template files")

        saved_data_files = sorted(glob.glob(os.path.join(e2edata_path, 'data', 'cgi_*_l2b.fits')))
        assert len(saved_data_files) > 0, f'No saved L2b files found in {e2edata_path}/data/!'

        dataset_fsm = Dataset(saved_data_files)
        logger.info(f"Generated and saved {len(saved_data_files)} new data files")

    breakpoint()
    # Double check and set up rest of PAM enumvalues

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
    
    # Adjust folders
    # Create the e2e subfolder regardless
    input_top_level = os.path.join(e2edata_path)
    output_top_level = os.path.join(e2eoutput_path)
    
    os.makedirs(input_top_level, exist_ok=True)
    os.makedirs(output_top_level, exist_ok=True)

    # Add subfloders for template (reference spectra) and data (FSM values)
    os.makedirs(os.path.join(input_top_level, 'template'), exist_ok=True)
    os.makedirs(os.path.join(input_top_level, 'data'), exist_ok=True)
    
    # Update paths to use the subfolder structure
    e2edata_path = input_top_level
    e2eoutput_path = output_top_level
    
    log_file = os.path.join(e2eoutput_path, 'star_spec_registration_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('star_spec_registration_e2e')
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
    logger.info('STAR SPECTRUM REGISTRATION END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    # Run the complete end-to-end test
    best_img = run_star_spec_registration_e2e_test(e2edata_path, e2eoutput_path)
    
    logger.info('='*80)
    logger.info('END-TO-END TEST COMPLETE')
    logger.info('='*80)
    
    return disp_model, coeffs, angle


# Run the test if this script is executed directly
if __name__ == "__main__":
    thisfile_dir = os.path.dirname(__file__)
    # Create top-level e2e folder
    top_level_dir = os.path.join(thisfile_dir, 'star_spec_registration_e2e')
    outputdir = os.path.join(top_level_dir, 'output')
    e2edata_dir = os.path.join(top_level_dir, 'input')

    ap = argparse.ArgumentParser(description="run the spectroscopy prism dispersion end-to-end test")
    ap.add_argument("-i", "--e2edata_dir", default=e2edata_dir,
                    help="directory to get input files from [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    
    # Run the e2e test with the same nested structure logic
    test_run_end_to_end(args.e2edata_dir, args.outputdir)


