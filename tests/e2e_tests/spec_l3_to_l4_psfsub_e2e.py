import os
import glob
import numpy as np
import astropy.io.fits as fits
import logging
import pytest
import argparse

from corgidrp import corethroughput
from corgidrp.data import Dataset
from corgidrp.data import Image
from corgidrp.mocks import create_default_calibration_product_headers, create_default_L2b_headers
from corgidrp.mocks import rename_files_to_cgi_format, create_ct_psfs
from corgidrp.walker import walk_corgidrp
import corgidrp
import corgidrp.caldb as caldb
from corgidrp.check import (check_filename_convention, check_dimensions, 
                           verify_hdu_count, verify_header_keywords, 
                           get_latest_cal_file, compare_to_mocks_hdrs)
from l1_to_l3_spec_e2e import run_l1_to_l3_e2e_test
import warnings


def patch_eacq_to_center_if_missing(filelist):
    """Set EACQ_ROW/EACQ_COL to image center when missing or (0,0).

    This should be done in the sims, but is done here for now to avoid breaking the e2e tests.

    Args:
        filelist (list): List of file paths to patch
    """
    for path in filelist:
        with fits.open(path, mode='update') as hdul:
            h = hdul[1].header
            n1, n2 = int(h['NAXIS1']), int(h['NAXIS2'])
            row = h.get('EACQ_ROW')
            col = h.get('EACQ_COL')
            missing = row is None or col is None
            both_zero = not missing and float(row) == 0 and float(col) == 0
            if missing or both_zero:
                h['EACQ_ROW'] = (n2 - 1) / 2.0
                h['EACQ_COL'] = (n1 - 1) / 2.0


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
        
    psfref_satspot_path = os.path.join(e2edata_path, "SPEC_refstar_satspot", "Analog", "L1")
    target_satspot_path = os.path.join(e2edata_path, "SPEC_targetstar_satspot", "L1", "analog")
    psfref_satspot_files = sorted(glob.glob(os.path.join(psfref_satspot_path, "cgi_*l1_.fits")))
    target_satspot_files = sorted(glob.glob(os.path.join(target_satspot_path, "cgi_*l1_.fits")))
    psfref_files_path = os.path.join(e2edata_path, "SPEC_refstar_slit_prism", "Analog", "L1")
    psfref_files = sorted(glob.glob(os.path.join(psfref_files_path, "cgi_*l1_.fits")))
    target_files_path = os.path.join(e2edata_path, "SPEC_targetstar_slit_prism", "L1", "analog")
    target_files = sorted(glob.glob(os.path.join(target_files_path, "cgi_*l1_.fits")))
    logger.info(f"Found {len(target_files)} existing L1 target files in {e2edata_path}")
    logger.info(f"Found {len(psfref_files)} existing L1 reference files in {e2edata_path}")
    logger.info(f"Found {len(target_satspot_files)} existing L1 target satspot files in {e2edata_path}")
    logger.info(f"Found {len(psfref_satspot_files)} existing L1 reference satspot files in {e2edata_path}")
    
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    ref_l3_output_dir = os.path.join(e2edata_path, "SPEC_refstar_slit_prism", "L3")
    target_l3_output_dir = os.path.join(e2edata_path, "SPEC_targetstar_slit_prism", "L3")
    ref_spot_l3_output_dir = os.path.join(e2edata_path, "SPEC_refstar_slit_prism", "L3", "satspot")
    target_spot_l3_output_dir = os.path.join(e2edata_path, "SPEC_targetstar_slit_prism", "L3", "satspot")

    cpgs_xml_filepath = os.path.join(os.path.dirname(__file__), "..", "test_data", "cpgs_mock.xml")

    # Patch EACQ_ROW/EACQ_COL on L1s. TODO: fix this in the sims
    for flist in (psfref_satspot_files, psfref_files, target_satspot_files, target_files):
        patch_eacq_to_center_if_missing(flist)

    run_l1_to_l3_e2e_test(psfref_satspot_path, ref_spot_l3_output_dir, processed_cal_path, logger)
    run_l1_to_l3_e2e_test(psfref_files_path, ref_l3_output_dir, processed_cal_path, logger)
    run_l1_to_l3_e2e_test(target_satspot_path, target_spot_l3_output_dir, processed_cal_path, logger)
    run_l1_to_l3_e2e_test(target_files_path, target_l3_output_dir, processed_cal_path, logger)
    
    l3_files = []
    l3_psfref = sorted(glob.glob(os.path.join(ref_l3_output_dir, "cgi_*l3_.fits")))
    l3_files.extend(l3_psfref)
    l3_psfref_spot = sorted(glob.glob(os.path.join(ref_spot_l3_output_dir, "cgi_*l3_.fits")))
    l3_files.extend(l3_psfref_spot)
    l3_target = sorted(glob.glob(os.path.join(target_l3_output_dir, "cgi_*l3_.fits")))
    l3_files.extend(l3_target)
    l3_target_spot = sorted(glob.glob(os.path.join(target_spot_l3_output_dir, "cgi_*l3_.fits")))
    l3_files.extend(l3_target_spot)
    l3_dataset = Dataset(l3_files)
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
        #simulation seems to be done without coronagraph
        frame.ext_hdr['FSMLOS'] = "1"
        check_filename_convention(getattr(frame, 'filename', None), 'cgi_*_l3_.fits', frame_info, logger, data_level = 'l3_')
        check_dimensions(frame.data, (125, 125), frame_info, logger)
        verify_header_keywords(frame.ext_hdr, {'DPAMNAME', 'CFAMNAME', 'FSAMNAME'}, frame_info, logger)
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L3', 'FSMLOS' : "1"}, frame_info, logger)
        verify_header_keywords(frame.pri_hdr, {'PSFREF'}, frame_info, logger)
        verify_header_keywords(frame.ext_hdr, {'SATSPOTS'}, frame_info, logger)
        logger.info("")
    
    l3_files_dir = os.path.join(e2eoutput_path, "L3")
    if not os.path.exists(l3_files_dir):
        os.makedirs(l3_files_dir)
    l3_dataset.save(filedir = l3_files_dir)
    l3_files_input = sorted(glob.glob(os.path.join(l3_files_dir, "cgi_*_l3_.fits")))
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
    fluxcal_fac = corgidrp.data.FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, err_hdr = errhd, input_dataset = l3_dataset)

    rename_files_to_cgi_format(list_of_fits=[fluxcal_fac], output_dir=calibrations_dir, level_suffix="abf_cal")
    this_caldb.create_entry(fluxcal_fac)
    
    ###########################
    #### Make dummy CT cal ####
    ###########################

    # Dataset with some CT profile defined in create_ct_interp
    # Pupil image
    pupil_image = np.zeros([1024, 1024])
    # Set it to some known value for a selected range of pixels
    pupil_image[510:530, 510:530]=1
    # Add specific values for pupil images:
    # DPAM = PUPIL, FSAM = OPEN, LSAM=OPEN and FPAM=OPEN_12
    exthd['DPAMNAME'] = 'PUPIL'
    exthd['LSAMNAME'] = 'OPEN'
    exthd['FSAMNAME'] = 'OPEN'
    exthd['FPAMNAME'] = 'OPEN_12'

    data_psf, psf_loc_in, half_psf = create_ct_psfs(50, cfam_name='3', n_psfs=10)
    
    err = np.ones([1024,1024]) 
    data_ct_interp = [Image(pupil_image,pri_hdr = prhd,
        ext_hdr = exthd, err = err)]
    # Set of off-axis PSFs with a CT profile defined in create_ct_interp
    # First, we need the CT FPM center to create the CT radial profile
    # We can use a miminal dataset to get to know it
    data_ct_interp += [data_psf[0]]
    ct_cal_tmp = corethroughput.generate_ct_cal(Dataset(data_ct_interp))
    rename_files_to_cgi_format(list_of_fits=[ct_cal_tmp], output_dir=calibrations_dir, level_suffix="ctm_cal")
    this_caldb.create_entry(ct_cal_tmp)
    
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
        filelist=l3_files_input, 
        CPGS_XML_filepath=cpgs_xml_filepath,
        outputdir=e2eoutput_path
    )
    logger.info("")
    
    # ================================================================================
    # (4) Validate Output Calibration Product
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Output Calibration Product Data Format and Content')
    logger.info('='*80)

    # Validate output product
    out_files = sorted(glob.glob(os.path.join(e2eoutput_path, '*_l4_.fits')))
    if len(out_files) == 1:
        logger.info("Expected: only one combined l4 file. PASS.")
    else:
        logger.info(f"Expected: only one combined l4 file but contains {len(out_files)} files. FAIL.")
    out_file = out_files[0]
    check_filename_convention(os.path.basename(out_file), 'cgi_*_l4_.fits', "spec l4 output product", logger, data_level = "l4_")
    
    compare_to_mocks_hdrs(out_file)
    
    with fits.open(out_file) as hdul:        
        verify_hdu_count(hdul, 12, "spec l4 output product", logger)
        
        # Verify HDU0 (header only)
        hdu0 = hdul[0]
        if hdu0.data is None:
            logger.info("HDU0: Header only. Expected: header only. PASS.")
        else:
            logger.info(f"HDU0: Contains data with shape {hdu0.data.shape}. Expected: header only. FAIL.")
            
        # Verify HDU1 (spec data)
        hdu1 = hdul[1]
        data = hdu1.data
        check_dimensions(data, (125, 125), "HDU1 Data Array: 2D array containing the 2D spectral distribution", logger)
        # Verify HDU2 (err)
        hdu2 = hdul[2]
        data = hdu2.data
        check_dimensions(data, (1, 125, 125), "HDU2 Data Array: 3D array containing the 3D spectral uncertainty", logger)
        # Verify HDU3 (DQ)
        hdu3 = hdul[3]
        data = hdu3.data
        check_dimensions(data, (125, 125), "HDU3 Data Array: 2D array containing the 2D data quality", logger)
        # Verify HDU4 (WAVE)
        hdu4 = hdul[4]
        data = hdu4.data
        check_dimensions(data, (125, 125), "HDU4 Data Array: 2D array containing the 2D wavelength distribution", logger)
        
        # PASS/FAIL checks for wavelength map (CGI-REQT-5464)
        logger.info("")
        logger.info("Wavelength-to-pixel calibration map (CGI-REQT-5464) validation:")
        
        # Check data type is float64
        if data.dtype.name == "float64":
            logger.info(f"    Data type: {data.dtype.name}. Expected: float64. PASS")
        else:
            logger.error(f"    Data type: {data.dtype.name}. Expected: float64. FAIL")
        
        # Check header contains expected keywords
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
        
        # Confirm no NaNs/inf values in HDU4
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
        
        # Print min/max wavelength values from HDU4 for inspection
        valid_data = data[np.isfinite(data)]
        if len(valid_data) > 0:
            min_wave = np.min(valid_data)
            max_wave = np.max(valid_data)
            logger.info(f"    Minimum wavelength value: {min_wave} nm")
            logger.info(f"    Maximum wavelength value: {max_wave} nm")
        else:
            logger.error(f"    No valid (finite) wavelength values found for min/max calculation. FAIL")
        
        # Check and print wavelength zero-point values (CGI-REQT-5474)
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
        check_dimensions(data, (125, 125), "HDU5 Data Array: 2D array containing the 2D wavelength uncertainty distribution", logger)
        # Verify HDU6 (ALGO_THRU)
        hdu6 = hdul[6]
        check_dimensions(hdu6.data, (19,), "HDU6 Data Array: containing the algorithm throughput", logger)
        # verify HDU7 (SPEC)
        hdu7 = hdul[7]
        check_dimensions(hdu7.data, (19,), "HDU7 Data Array: containing the 1D spectral distribution", logger)
        if np.isnan(hdu7.data).any() is True:
            logger.info(f"HDU7 Data Array: Contains NANs in the data. Expected: no NANs. FAIL.")
        else:
            logger.info(f"HDU7 Data Array: No NANs in the data. Expected: no NANs. PASS.")
        if np.isinf(hdu7.data).any() is True:
            logger.info(f"HDU7 Data Array: Contains INFs in the data. Expected: no INFs. FAIL.")
        else:
            logger.info(f"HDU7 Data Array: No INFs in the data. Expected: no INFs. PASS.")
        # Verify HDU8 (error)
        hdu8 = hdul[8]
        err = hdu8.data
        check_dimensions(err, (1, 19), "HDU8 Data Array: 1D array with the corresponding spectral uncertainty", logger)
        # Verify HDU9 (dq)
        hdu9 = hdul[9]
        dq = hdu9.data
        check_dimensions(dq, (19,), "HDU9 Data Array: 1D array with the corresponding spectral data quality", logger)
        
        # Verify HDU10 (wavelength)
        hdu10 = hdul[10]
        wave = hdu10.data
        check_dimensions(wave, (19,), "HDU10 Data Array: 1D array with the corresponding wavelength", logger)
        
        # Verify HDU11 (wavelength uncertainties)
        hdu11 = hdul[11]
        wave_err = hdu11.data
        check_dimensions(wave_err, (19,), "HDU11 Data Array: 1D array with the corresponding wavelength uncertainty", logger)
        # Verify header keywords
        verify_header_keywords(hdul[1].header, {'DATALVL': 'L4', 'CFAMNAME' : '3F', 'FSAMNAME': 'R1C2', 'DPAMNAME':'PRISM3', 'BUNIT' : 'photoelectron/s'},
                                               "spec output product", logger)
        verify_header_keywords(hdul[1].header, {'WAVLEN0', 'WV0_X', 'WV0_Y', 'WV0_DIMX', 'WV0_DIMY'},
                                               "spec output product", logger)
        verify_header_keywords(hdul[1].header, {'STARLOCX', 'STARLOCY', 'CRPIX1', 'CRPIX2', 'CTCALFN', 'FLXCALFN'},
                                               "spec output product", logger)
        verify_header_keywords(hdul[7].header, {'BUNIT' : 'photoelectron/s/bin'},
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
    sed = spec_out.hdu_list["SPEC"].data
    wave = spec_out.hdu_list["SPEC_WAVE"].data
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
    output_top_level = os.path.join(e2eoutput_path, "l3_to_l4_spec_psfsub_e2e")
    if not os.path.exists(output_top_level):
        os.makedirs(output_top_level)
    # clean out any files from a previous run
    for f in os.listdir(output_top_level):
        file_path = os.path.join(output_top_level, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    
    os.makedirs(output_top_level, exist_ok=True)
    
    # Update paths to use the subfolder structure
    e2eoutput_path = output_top_level
    
    log_file = os.path.join(e2eoutput_path, 'l3_to_l4_spec_psfsub_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('l3_to_l4_spec_psfsub_e2e')
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
    with warnings.catch_warnings():  
        warnings.filterwarnings('ignore', category=RuntimeWarning)# prevent UserWarning: Number of frames which made the DetectorNoiseMaps product is less than the number of frames in input_dataset
        spec_out = run_spec_l3_to_l4_psfsub_e2e_test(e2edata_path, e2eoutput_path)
    
    logger.info('='*80)
    logger.info('END-TO-END TEST COMPLETE')
    logger.info('='*80)
    


# Run the test if this script is executed directly
if __name__ == "__main__":
    thisfile_dir = os.path.dirname(__file__)
    # Create top-level e2e folder
    outputdir = thisfile_dir
    e2edata_dir = '/Users/kevinludwick/Documents/DRP_E2E_Test_Files_v2/E2E_Test_Data'

    ap = argparse.ArgumentParser(description="run the spectroscopy l3 to l4 end-to-end test")
    ap.add_argument("-i", "--e2edata_dir", default=e2edata_dir,
                    help="directory to get input files from [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    
    # Run the e2e test with the same nested structure logic
    test_run_end_to_end(args.e2edata_dir, args.outputdir)


