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

from corgidrp.data import Dataset
from corgidrp.data import Image
from corgidrp.mocks import create_default_calibration_product_headers
from corgidrp.mocks import rename_files_to_cgi_format
from corgidrp.walker import walk_corgidrp
import corgidrp
import corgidrp.caldb as caldb
from corgidrp.check import (check_filename_convention, check_dimensions, 
                           verify_hdu_count, verify_header_keywords, 
                           get_latest_cal_file)


# first lift the L1 simulations to L3



# ================================================================================
# Main Spec L3 to L4 E2E Test Function PSF subtracted
# ================================================================================

def run_spec_l3_to_l4_psfsub_e2e_test(e2edata_path, e2eoutput_path):
    """Run the complete spectroscopy psfsub l3 to l4 end-to-end test.
    
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
    psfref_files = sorted(glob.glob(os.path.join(e2edata_path, "SPEC_refstar_slit_prism", "L2a", 'cgi_*_l2a.fits')))
    target_files = sorted(glob.glob(os.path.join(e2edata_path, "SPEC_targetstar_slit_prism", "L2b", 'cgi_*_l2b.fits')))
    logger.info(f"Found {len(target_files)} existing L2b target files in {e2edata_path}")
    logger.info(f"Found {len(psfref_files)} existing L2a reference files in {e2edata_path}")
    
    all_frames = []
    l2a_dataset_with_psfref = Dataset(psfref_files)
    for frame in l2a_dataset_with_psfref:
        frame.ext_hdr["PSFREF"] = 1
        all_frames.append(frame)
    l2b_dataset_with_target = Dataset(target_files)
    for frame in l2b_dataset_with_target:
        all_frames.append(frame)
        
    l3_dataset = Dataset(all_frames)

    logger.info('')
    
    # ================================================================================
    # (2) Validate Input Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 1: Input Image Data Format and Content')
    logger.info('='*80)

    # Validate all input images
    for i, frame in enumerate(l3_dataset):
        frame_info = f"L3 Frame {i}"
        
        check_filename_convention(getattr(frame, 'filename', None), 'cgi_*_l3_.fits', frame_info, logger, data_level = 'l3_')
        check_dimensions(frame.data, (81, 81), frame_info, logger)
        verify_header_keywords(frame.ext_hdr, {'DPAMNAME', 'CFAMNAME', 'FSAMNAME'}, frame_info, logger)
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L3'}, frame_info, logger)
        logger.info("")

    logger.info(f"Total input images validated: {len(l3_dataset)}")
    logger.info("")
    
    # Create a temporary caldb and add the default DispersionModel calibration
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()
    
    # Create calibrations subfolder
    calibrations_dir = os.path.join(e2eoutput_path, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)
    #Create a mock flux calibration file
    fluxcal_factor = 2e-12
    fluxcal_factor_error = 1e-14
    prhd, exthd, errhd, dqhd = create_default_calibration_product_headers()
    # Set consistent header values for flux calibration factor
    exthd['CFAMNAME'] = '3'
    exthd['DPAMNAME'] = 'PRISM3'
    exthd['FSAMNAME'] = 'R1C2'
    fluxcal_fac = corgidrp.data.FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, err_hdr = errhd, input_dataset = l2b_dataset_with_target)

    rename_files_to_cgi_format(list_of_fits=[fluxcal_fac], output_dir=calibrations_dir, level_suffix="abf_cal")
    this_caldb.create_entry(fluxcal_fac)
    
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
        template="l2_to_l4_psfsub_spec.json"
    )
    logger.info("")
    
    # ================================================================================
    # (4) Validate Output Calibration Product
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Output Calibration Product Data Format and Content')
    logger.info('='*80)

    # Validate output product
    out_file = get_latest_cal_file(e2eoutput_path, '*_l4_.fits', logger)
    check_filename_convention(os.path.basename(out_file), 'cgi_*_l4_.fits', "spec l4 output product", logger, data_level = "l4_")

    with fits.open(out_file) as hdul:        
        verify_hdu_count(hdul, 6, "spec l4 output product", logger)
        
        # Verify HDU0 (header only)
        hdu0 = hdul[0]
        if hdu0.data is None:
            logger.info("HDU0: Header only. Expected: header only. PASS.")
        else:
            logger.info(f"HDU0: Contains data with shape {hdu0.data.shape}. Expected: header only. FAIL.")
        #verify HDU1
        hdu1 = hdul[1]
        check_dimensions(hdu1.data, (19,), "HDU1 Data Array: containing the 1D spectral distribution", logger)
        if np.isnan(hdu1.data).any() is True:
            logger.info(f"HDU1 Data Array: Contains NANs in the data. Expected: no NANs. FAIL.")
        else:
            logger.info(f"HDU1 Data Array: No NANs in the data. Expected: no NANs. PASS.")
        if np.isinf(hdu1.data).any() is True:
            logger.info(f"HDU1 Data Array: Contains INFs in the data. Expected: no INFs. FAIL.")
        else:
            logger.info(f"HDU1 Data Array: No INFs in the data. Expected: no INFs. PASS.")
        # Verify HDU2 (error)
        hdu2 = hdul[2]
        err = hdu2.data
        check_dimensions(err, (1, 19), "HDU2 Data Array: 1D array with the corresponding spectral uncertainty", logger)
        # Verify HDU3 (dq)
        hdu3 = hdul[3]
        dq = hdu3.data
        check_dimensions(dq, (19,), "HDU3 Data Array: 1D array with the corresponding spectral data quality", logger)
        
        # Verify HDU4 (wavelength)
        hdu4 = hdul[4]
        wave = hdu4.data
        check_dimensions(wave, (19,), "HDU4 Data Array: 1D array with the corresponding wavelength", logger)
        
        # Verify HDU5 (wavelength uncertainties)
        hdu5 = hdul[5]
        wave_err = hdu5.data
        check_dimensions(wave_err, (19,), "HDU5 Data Array: 1D array with the corresponding wavelength uncertainty", logger)
        
        # Verify header keywords
        verify_header_keywords(hdul[1].header, {'DATALVL': 'L4', 'CFAMNAME' : '3', 'FSAMNAME': 'R1C2', 'DPAMNAME':'PRISM3', 'BUNIT' : 'photoelectron/s/bin'},
                                               "spec output product", logger)
        verify_header_keywords(hdul[1].header, {'WAVLEN0', 'WV0_X', 'WV0_Y', 'WV0_DIMX', 'WV0_DIMY'},
                                               "spec output product", logger)

    logger.info("")
    
    # ================================================================================
    # (5) Baseline Performance Checks
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 3: Baseline Performance Checks')
    logger.info('='*80)

    # Load and display spec output product results
    spec_out = Image(out_file)
    sed = spec_out.data
    wave = spec_out.hdu_list["WAVE"].data
    logger.info(f"wavelengths: {wave} nm")
    logger.info(f"spectrum: {sed}")
    logger.info("")
    
    # Clean up temporary caldb file
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    
    return spec_out



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
    
    # Create the spec_l3_to_l4_e2e subfolder regardless
    output_top_level = os.path.join(e2eoutput_path, 'spec_l3_to_l4_psfsub_e2e')
    
    os.makedirs(output_top_level, exist_ok=True)
    
    # Update paths to use the subfolder structure
    e2eoutput_path = output_top_level
    
    log_file = os.path.join(e2eoutput_path, 'spec_l3_to_l4_psfsub_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('spec_l3_to_l4_psfsub_e2e')
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
    logger.info('SPECTROSCOPY PSFSUB L3 to L4 END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    # Run the complete end-to-end test
    spec_out = run_spec_l3_to_l4_psfsub_e2e_test(e2edata_path, e2eoutput_path)
    
    logger.info('='*80)
    logger.info('END-TO-END TEST COMPLETE')
    logger.info('='*80)
    


# Run the test if this script is executed directly
if __name__ == "__main__":
    thisfile_dir = os.path.dirname(__file__)
    # Create top-level e2e folder
    top_level_dir = os.path.join(thisfile_dir, 'spec_l3_to_l4_psfsub_e2e')
    outputdir = os.path.join(top_level_dir, 'output')
    e2edata_dir = "/home/schreiber/spec_sim"

    ap = argparse.ArgumentParser(description="run the spectroscopy l3 to l4 end-to-end test")
    ap.add_argument("-i", "--e2edata_dir", default=e2edata_dir,
                    help="directory to get input files from [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    
    # Run the e2e test with the same nested structure logic
    test_run_end_to_end(args.e2edata_dir, args.outputdir)


