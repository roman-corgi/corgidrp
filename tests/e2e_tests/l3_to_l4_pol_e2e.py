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
                           verify_hdu_count, verify_header_keywords)

thisfile_dir = os.path.dirname(__file__) # this file's folder


def run_l3_to_l4_e2e_test(l3_datadir, l4_outputdir, processed_cal_path, logger):
    """Run the complete L3 to L4 polarimetry data end-to-end test.
    
    Args:
        l3_datadir (str): Path to L3 input data directory
        l4_outputdir (str): Path to output directory
        processed_cal_path (str): Path to processed calibration files directory
        logger (logging.Logger): Logger instance for output
        
    Returns:
        list: List of L4 output filenames
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

    # Create mock headers for calibration products
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] = corgidrp.__version__
    
    # Get some L3 files to mock the input_dataset for calibrations
    all_files = os.listdir(l3_datadir)
    l3_files = [f for f in all_files if f.endswith('_l3_.fits')]
    if len(l3_files) >= 2:
        mock_cal_filelist = [os.path.join(l3_datadir, l3_files[i]) for i in [-2, -1]]
    else:
        mock_cal_filelist = [os.path.join(l3_datadir, f) for f in l3_files]
    mock_input_dataset = data.Dataset(mock_cal_filelist)

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

    # Create mock CoreThroughputMap
    # CoreThroughputMap expects data with shape (3, N) where rows are [x, y, ct_value]
    # Create a simple grid of x, y positions with CT values
    n_points = 47 * 47  # Grid of 47x47 points
    x_coords = np.linspace(-23, 23, 47)
    y_coords = np.linspace(-23, 23, 47)
    xx, yy = np.meshgrid(x_coords, y_coords)
    # Create radial falloff for CT values (center = 0.8, edge = 0.1)
    r = np.sqrt(xx**2 + yy**2)
    ct_values = 0.8 * np.exp(-r/10.0) + 0.1
    ct_values = np.clip(ct_values, 0.0, 1.0)
    
    # Reshape to (3, N) format: [x, y, ct]
    ct_map_data = np.array([xx.flatten(), yy.flatten(), ct_values.flatten()])
    ct_map = data.CoreThroughputMap(ct_map_data, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[ct_map], output_dir=calibrations_dir, level_suffix="ctm_cal")
    this_caldb.create_entry(ct_map)

    # Create mock FluxcalFactor
    # FluxcalFactor needs a value, error, and ext_hdr with filter info
    fluxcal_value = np.array([1.0])  # Single band value
    fluxcal_error = np.array([0.01])
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'erg/(s*cm^2*AA)/(photoelectron/s)'
    # Need filter name in ext_hdr
    fluxcal_ext_hdr = ext_hdr.copy()
    fluxcal_ext_hdr['CFAMNAME'] = '3'  # Band 3 for spectroscopy
    fluxcal_factor = data.FluxcalFactor(fluxcal_value, err=fluxcal_error, pri_hdr=pri_hdr, 
                                       ext_hdr=fluxcal_ext_hdr, err_hdr=err_hdr,
                                       input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[fluxcal_factor], output_dir=calibrations_dir, level_suffix="fcf_cal")
    this_caldb.create_entry(fluxcal_factor)
    
    logger.info("Created calibration products:")
    logger.info(f"  - AstrometricCalibration: {astrom_cal.filename}")
    logger.info(f"  - CoreThroughputMap: {ct_map.filename}")
    logger.info(f"  - FluxcalFactor: {fluxcal_factor.filename}")
    logger.info('')

    # ================================================================================
    # (2) Validate Input Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 1: Input L3 Image Data Format and Content')
    logger.info('='*80)

    # Filter to only include L3 files as inputs
    input_files = [f for f in all_files if f.endswith('_l3_.fits')]
    if not input_files:
        raise FileNotFoundError(f"No files ending in '_l3_.fits' found in {l3_datadir}")
    
    input_data_filelist = [os.path.join(l3_datadir, f) for f in input_files]
    
    # Create input_data subfolder
    input_data_dir = os.path.join(l4_outputdir, 'input_l3')
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
        frame_info = f"L3 Input Frame {i}"
        
        check_filename_convention(os.path.basename(filepath), 'cgi_*_l3_.fits', frame_info, logger, data_level='l3_')
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L3'}, frame_info, logger)
        
        # Verify HDU count
        with fits.open(filepath) as hdul:
            verify_hdu_count(hdul, 5, frame_info, logger)
        
        # Check data dimensions - should be polarimetry datacube (2, N, N)
        if len(frame.data.shape) == 3 and frame.data.shape[0] == 2:
            logger.info(f"{frame_info}: Polarimetry datacube shape {frame.data.shape}. PASS")
        else:
            logger.info(f"{frame_info}: Expected polarimetry datacube (2, N, N), got {frame.data.shape}. FAIL")
        
        logger.info("")
    
    logger.info(f"Total input images validated: {len(input_dataset)}")
    logger.info('')
    
    # ================================================================================
    # (3) Run Processing Pipeline
    # ================================================================================
    logger.info('='*80)
    logger.info('Running L3 -> L4 polarimetry data processing pipeline')
    logger.info('='*80)
    
    logger.info('Running L3 to L4 polarimetry recipe...')
    walker.walk_corgidrp(input_data_filelist, "", l4_outputdir)
    logger.info('L3 to L4 complete.')
    logger.info('')
    
    # ================================================================================
    # (4) Validate Output L4 Images
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
                verify_hdu_count(hdul, 5, frame_info, logger)  # L4 should have 5 HDUs
            
            # Verify data level
            verify_header_keywords(img.ext_hdr, {'DATALVL': 'L4'}, frame_info, logger)

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
            
            # Verify WCS headers exist (should be preserved from L3)
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
    
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)
    
    return new_l4_filenames


@pytest.mark.e2e
def test_l3_to_l4(e2edata_path, e2eoutput_path):
    """Run the complete L3 to L4 polarimetry data end-to-end test.
    
    Args:
        e2edata_path (str): Path to test data (expects L3 files)
        e2eoutput_path (str): Output directory path for results and logs
    """
    # Set up output directory
    l4_outputdir = os.path.join(e2eoutput_path, "l3_to_l4_pol_e2e")
    if os.path.exists(l4_outputdir):
        shutil.rmtree(l4_outputdir)
    os.makedirs(l4_outputdir)

    analog_outputdir = os.path.join(l4_outputdir, "analog")
    pc_outputdir = os.path.join(l4_outputdir, "pc")
    if not os.path.exists(analog_outputdir):
        os.makedirs(analog_outputdir)
    if not os.path.exists(pc_outputdir):
        os.makedirs(pc_outputdir)
    
    log_file = os.path.join(l4_outputdir, 'l3_to_l4_pol_e2e.log')
    
    # Create a new logger specifically for this test
    global logger
    logger = logging.getLogger('l3_to_l4_pol_e2e')
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
    logger.info('L3 TO L4 POLARIMETRY DATA END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    # L3 data directories (assuming they're in POL_sims/L3/analog_data and POL_sims/L3/PC_data)
    analog_datadir = os.path.join(e2edata_path, "POL_sims", "L3", "analog_data")
    pc_datadir = os.path.join(e2edata_path, "POL_sims", "L3", "PC_data")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    
    # Run the complete end-to-end test
    try:
        logger.info('='*80)
        logger.info('ANALOG POLARIMETRY DATA TEST')
        logger.info('='*80)
        new_l4_analog_filenames = run_l3_to_l4_e2e_test(analog_datadir, analog_outputdir, processed_cal_path, logger)
        
        logger.info('='*80)
        logger.info('PC POLARIMETRY DATA TEST')
        logger.info('='*80)
        new_l4_pc_filenames = run_l3_to_l4_e2e_test(pc_datadir, pc_outputdir, processed_cal_path, logger)
        
        logger.info('='*80)
        logger.info('END-TO-END TEST COMPLETE')
        logger.info('='*80)
        
        print('e2e test for L3 to L4 polarimetry passed')
    except Exception as e:
        logger.error('='*80)
        logger.error('END-TO-END TEST FAILED')
        logger.error('='*80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Print traceback to log
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        print(f'e2e test for L3 to L4 polarimetry FAILED: {str(e)}')
        raise

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l3->l4 polarimetry end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")

    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir

    test_l3_to_l4(e2edata_dir, outputdir)

