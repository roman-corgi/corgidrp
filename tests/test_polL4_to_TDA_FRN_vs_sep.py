from corgidrp.mocks import create_psfsub_dataset,create_ct_cal, create_default_L4_headers
from corgidrp.l3_to_l4 import do_psf_subtraction
from corgidrp.l4_to_tda import compute_flux_ratio_noise
import corgidrp.data as data
import corgidrp.mocks as mocks
import logging

import os, shutil
from corgidrp.check import (check_dimensions, verify_hdu_count,
                            verify_header_keywords,validate_binary_table_fields)

import numpy as np

#=========================================================================================
# Polarimetry L4-> TDA VAP Test 1: Flux Ratio Noise vs Separation (Issue #637)
# Main Test Function
#=========================================================================================

def run_pol_L4_TDA_VAP_Test1():

    #=====================================================================================
    # Set up logging
    global logger
    log_file = 'test_FRN_vs_sep.log'
    
    # Create a new logger for this test
    logger = logging.getLogger('test_FRN_vs_sep')
    logger.setLevel(logging.INFO)
    
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
    #=====================================================================================
    # Setup: Generate the input data for l4_to_tda.compute_flux_ratio_noise
    logger.info('='*80)
    logger.info('Pre-test: set up input data')
    logger.info('='*80)
    
    # Step 1: Generate a mock polarization data cube (4,n,m) using 
    #         mocks.create_mock_stokes_image_l4
    
    # Set up the input arguments
    companion_counts = 5.0e4
    col_cor = 1.2
    # Create the mock image
    pol_datacube = mocks.create_mock_stokes_i_image(companion_counts,'COMP',
                                                    col_cor = col_cor, seed = 2,
                                                    wv0_x = 2.0, wv0_y = -1.0,
                                                    is_coronagraphic = True)
                                                    
    # Create the corresponding mock dataset
    pol_dataset = data.Dataset([pol_datacube])
    
    # Step 2: Construct an unocculted star dataset
    host_counts = 2.5e5
    star_image = mocks.create_mock_stokes_i_image(host_counts,'HOST',
                                                     seed=1, wv0_x=-1.0, wv0_y=0.5,
                                                     is_coronagraphic=True)
    star_dataset = data.Dataset([star_image]) # data is (1,64,64)
    
    # Step 3: Construct a mock ND calibration, following the approach used in 
    # test_expected_flux_ratio_noise()
    data_shape = (64,64)
    nd_x, nd_y = np.meshgrid(np.linspace(0, data_shape[1], 5), np.linspace(0, data_shape[0], 5))
    nd_x = nd_x.ravel()
    nd_y = nd_y.ravel()
    nd_od = np.ones(nd_y.shape) * 1e-2
    pri_hdr, ext_hdr, errhdr, dqhdr, biashdr = mocks.create_default_L2b_headers()
    nd_cal = data.NDFilterSweetSpotDataset(np.array([nd_od, nd_x, nd_y]).T, pri_hdr=pri_hdr,
                                      ext_hdr=ext_hdr)
    #=====================================================================================
    # Validate the input images
    
    # Expected data format (4,n,m)
    # Define the expected n and m
    n = 64
    m = 64
    
    # Start logging
    logger.info('='*80)
    logger.info('Data format checks')
    logger.info('='*80)
    
    for i, frame in enumerate(pol_dataset):
        frame_info=f"Frame {i}"
        
        # Check that L4 data input complies with cgi format ##FIX THIS--Shape fails
        # Check that the data for each frame is (4,n,m)
        check_dimensions(frame.data,(4,n,m), frame_info, logger)
        # Check that DATALVL = L4
        verify_header_keywords(frame.ext_hdr,{'DATALVL':'L4'},frame_info,logger)
        # Check that BUNIT = photoelectron/s
        verify_header_keywords(frame.ext_hdr,{'BUNIT':'photoelectron/s'},frame_info,logger)
        logger.info("")
        
    #=====================================================================================
    # Run compute_flux_ratio_noise()
    
    # Add to the log
    logger.info('='*80)
    logger.info('Running compute_flux_ratio_noise()')
    logger.info('='*80)
    
    # Extract the first slice from the datacube (0,n,m) and pass into the function
    # compute_flux_ratio_noise

	# FIX THIS. SOMETHING IS WRONG WITH THE INPUT DATA TYPE
    frn_dataset_nostarloc = compute_flux_ratio_noise(pol_datacube.data[0,:,:], nd_cal, star_dataset[0,:,:], halfwidth=3)
        
                   
    
    # now see what the step function gives, with and without a supplied star location guess:
    #
    #frn_dataset_starloc = compute_flux_ratio_noise(psfsub_dataset_rdi, nd_cal, star_dataset, unocculted_star_loc=np.array([[17],[15]]), halfwidth=3)
    
if __name__ == '__main__':
    run_pol_L4_TDA_VAP_Test1()     