import argparse
import os, shutil
import glob
import pytest
import numpy as np
import logging
from datetime import datetime, timedelta

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.fluxcal as fluxcal
from corgidrp import caldb
from corgidrp.check import (check_filename_convention, check_dimensions, verify_header_keywords, compare_to_mocks_hdrs)

@pytest.mark.e2e
def test_expected_results_e2e(e2edata_path, e2eoutput_path):
    # create output dir first
    output_dir = os.path.join(e2eoutput_path, 'fluxcal_pol_e2e')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # set up logging
    global logger
    log_file = os.path.join(output_dir, 'fluxcal_pol_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('fluxcal_pol_e2e')
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
    
    logger.info('='*80)
    logger.info('Polarization Absolute Flux Calibration END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    logger.info('='*80)
    logger.info('Pre-test: set up input files and save to disk')
    logger.info('='*80)

    #mock a point source image
    fwhm = 3
    star_flux = 1.5e-09 #erg/(s*cm^2*AA)
    cal_factor = star_flux/200
    # split the flux unevenly by polarization
    star_flux_left = 0.6 * star_flux
    star_flux_right = 0.4 * star_flux
    
    # Create WP1 images with unique timestamps
    base_time = datetime.now()
    flux_image_WP1_1 = mocks.create_pol_flux_image(star_flux_left, star_flux_right, fwhm, cal_factor, dpamname='POL0', filedir=None, file_save=False)
    flux_image_WP1_1.ext_hdr['BUNIT'] = 'photoelectron'
    flux_image_WP1_1.ext_hdr['FILETIME'] = base_time.isoformat()
    
    flux_image_WP1_2 = mocks.create_pol_flux_image(star_flux_left, star_flux_right, fwhm, cal_factor, dpamname='POL0', filedir=None, file_save=False)
    flux_image_WP1_2.ext_hdr['BUNIT'] = 'photoelectron'
    flux_image_WP1_2.ext_hdr['FILETIME'] = (base_time + timedelta(milliseconds=100)).isoformat()
    
    flux_dataset_WP1 = data.Dataset([flux_image_WP1_1, flux_image_WP1_2])
    
    # Create WP2 images with unique timestamps
    flux_image_WP2_1 = mocks.create_pol_flux_image(star_flux_left, star_flux_right, fwhm, cal_factor, dpamname='POL45', filedir=None, file_save=False)
    flux_image_WP2_1.ext_hdr['BUNIT'] = 'photoelectron'
    flux_image_WP2_1.ext_hdr['FILETIME'] = (base_time + timedelta(milliseconds=200)).isoformat()
    
    flux_image_WP2_2 = mocks.create_pol_flux_image(star_flux_left, star_flux_right, fwhm, cal_factor, dpamname='POL45', filedir=None, file_save=False)
    flux_image_WP2_2.ext_hdr['BUNIT'] = 'photoelectron'
    flux_image_WP2_2.ext_hdr['FILETIME'] = (base_time + timedelta(milliseconds=300)).isoformat()
    
    flux_dataset_WP2 = data.Dataset([flux_image_WP2_1, flux_image_WP2_2])

    logger.info('='*80)
    logger.info('Test Case 1: Input Image Data Format and Content for WP1')
    logger.info('='*80)

    # check input dataset info for WP1
    for i, frame in enumerate(flux_dataset_WP1):
        frame_info = f"Frame {i}"
        frame_name = getattr(frame, 'filename', None)
        check_filename_convention(frame_name, 'cgi_*_l2b.fits', frame_info=frame_info, logger=logger)
        check_dimensions(frame.data, (1024, 1024), frame_info=frame_info, logger=logger)
        # check all images have the same CFAMNAME value
        verify_header_keywords(frame.ext_hdr, {'CFAMNAME': flux_dataset_WP1.frames[0].ext_hdr['CFAMNAME']}, frame_info=frame_info, logger=logger)
        # check all images have POL0 as DPAMNAME value
        verify_header_keywords(frame.ext_hdr, {'DPAMNAME': 'POL0'}, frame_info=frame_info, logger=logger)
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L2b'}, frame_info=frame_info, logger=logger)
        # log CFAMNAME
        logger.info(f"Frame {frame_name} in flux_dataset_WP1 have CFAMNAME {frame.ext_hdr['CFAMNAME']}")
        logger.info("")
    logger.info(f"Total input images validated: {len(flux_dataset_WP1)}")

    # Create proper folder structure: WP1 folder with input_l2b subdirectory

    output_dir_WP1 = os.path.join(output_dir, 'WP1')
    os.makedirs(output_dir_WP1)
    input_l2b_dir_WP1 = os.path.join(output_dir_WP1, 'input_l2b')
    os.makedirs(input_l2b_dir_WP1)
    
    # Save files with proper CGI naming convention
    data_filelist_WP1 = mocks.rename_files_to_cgi_format(
        list_of_fits=list(flux_dataset_WP1),
        output_dir=input_l2b_dir_WP1,
        level_suffix="l2b"
    )

    ####### Run the DRP walker for WP1
    logger.info('='*80)
    logger.info('Running processing pipeline')
    logger.info('='*80)

    logger.info('Running walker for WP1')
    walker.walk_corgidrp(data_filelist_WP1, '', output_dir_WP1)
    fluxcal_file_WP1 = glob.glob(os.path.join(output_dir_WP1, '*abf_cal*.fits'))[0]
    fluxcal_image_WP1 = data.Image(fluxcal_file_WP1)

    logger.info('='*80)
    logger.info('Test Case 2: Output Calibration Product Data Format and Content for WP1')
    logger.info('='*80)
    ## check that the calibration file is configured correctly
    # check HDU0 have no data
    if not hasattr(fluxcal_image_WP1.pri_hdr, "data"):
        logger.info("HDU0: Header only. Expected: header only. PASS.")
    else:
        logger.info(f"HDU0: Contains data. Expected: header only. FAIL.")
    # check HDU1 data is a single float
    if fluxcal_image_WP1.data.dtype.type == corgidrp.image_dtype:
        logger.info(f"HDU1 dtype: float. Expected: float. PASS.")
    else:
        logger.info(f"HDU1 dtype: {fluxcal_image_WP1.data.dtype.type}. Expected: float. FAIL.")
    # check err and dq have the right dimension
    if fluxcal_image_WP1.err.shape == (1,):
        logger.info(f"Err data shape: (1,). Expected: (1,). PASS.")
    else:
        logger.info(f"Err data shape: {fluxcal_image_WP1.err.shape}. Expected: (1,). FAIL.")
    if fluxcal_image_WP1.dq.shape == (1,):
        logger.info(f"DQ data shape: (1,). Expected: (1,). PASS.")
    else:
        logger.info(f"DQ data shape: {fluxcal_image_WP1.dq.shape}. Expected: (1,). FAIL.")
    # check filename convention
    check_filename_convention(getattr(fluxcal_image_WP1, 'filename', None), 'abf_cal.fits', logger=logger)
    # check header keyword values match with what is expected
    verify_header_keywords(fluxcal_image_WP1.ext_hdr, {'DATALVL': 'CAL'}, logger=logger)
    verify_header_keywords(fluxcal_image_WP1.ext_hdr, {'DATATYPE': 'FluxcalFactor'}, logger=logger)
    verify_header_keywords(fluxcal_image_WP1.ext_hdr, {'DPAMNAME': flux_image_WP1_1.ext_hdr['DPAMNAME']}, logger=logger)
    verify_header_keywords(fluxcal_image_WP1.ext_hdr, {'CFAMNAME': flux_image_WP1_1.ext_hdr['CFAMNAME']}, logger=logger)
    logger.info("")

    # baseline performance check WP1
    logger.info('='*80)
    logger.info('Test Case 3: Baseline Performance Checks for WP1')
    logger.info('='*80)
    flux_fac_WP1 = data.FluxcalFactor(fluxcal_file_WP1)
    logger.info(f"used color filter {flux_fac_WP1.filter}")
    logger.info(f"used ND filter {flux_fac_WP1.nd_filter}")
    logger.info(f"fluxcal factor {flux_fac_WP1.fluxcal_fac}")
    logger.info(f"fluxcal factor error {flux_fac_WP1.fluxcal_err}")
    for i, frame in enumerate(flux_dataset_WP1):
        frame_info = f"Frame {i}"
        frame_name = getattr(frame, 'filename', None)
        # log input FSMX and FSMY
        logger.info(f"Frame {frame_name} in flux_dataset_WP1 have FSMX {frame.ext_hdr['FSMX']}")
        logger.info(f"Frame {frame_name} in flux_dataset_WP1 have FSMY {frame.ext_hdr['FSMY']}")
    # log output FSMX and FSMY
    logger.info(f"Output fluxcal file have FSMX {flux_fac_WP1.ext_hdr['FSMX']}")
    logger.info(f"Output fluxcal file have FSMY {flux_fac_WP1.ext_hdr['FSMY']}")
    logger.info("")
    assert flux_fac_WP1.fluxcal_fac == pytest.approx(cal_factor, abs = 1.5 * flux_fac_WP1.fluxcal_err)

    logger.info('='*80)
    logger.info('Test Case 4: Input Image Data Format and Content for WP2')
    logger.info('='*80)

    # check input dataset info for WP2
    for i, frame in enumerate(flux_dataset_WP2):
        frame_name = getattr(frame, 'filename', None)
        check_filename_convention(frame_name, 'cgi_*_l2b.fits', frame_info=frame_info, logger=logger)
        check_dimensions(frame.data, (1024, 1024), frame_info=frame_info, logger=logger)
        # check all images have the same CFAMNAME value
        verify_header_keywords(frame.ext_hdr, {'CFAMNAME': flux_dataset_WP2.frames[0].ext_hdr['CFAMNAME']}, frame_info=frame_info, logger=logger)
        # check all images have POL45 as DPAMNAME value
        verify_header_keywords(frame.ext_hdr, {'DPAMNAME': 'POL45'}, frame_info=frame_info, logger=logger)
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L2b'}, frame_info=frame_info, logger=logger)
        # print CFAMNAME
        logger.info(f"Frame {frame_name} in flux_dataset_WP2 have CFAMNAME {frame.ext_hdr['CFAMNAME']}")
        logger.info("")
    logger.info(f"Total input images validated: {len(flux_dataset_WP2)}")

    # Create proper folder structure: WP2 folder with input_l2b subdirectory
    # Output calibration files will be saved to WP2 root
    output_dir_WP2 = os.path.join(output_dir, 'WP2')
    os.makedirs(output_dir_WP2)
    input_l2b_dir_WP2 = os.path.join(output_dir_WP2, 'input_l2b')
    os.makedirs(input_l2b_dir_WP2)
    
    # Save files with proper CGI naming convention
    data_filelist_WP2 = mocks.rename_files_to_cgi_format(
        list_of_fits=list(flux_dataset_WP2),
        output_dir=input_l2b_dir_WP2,
        level_suffix="l2b"
    )

    ####### Run the DRP walker for WP2
    logger.info('Running walker for WP2')
    walker.walk_corgidrp(data_filelist_WP2, '', output_dir_WP2)
    fluxcal_file_WP2 = glob.glob(os.path.join(output_dir_WP2, '*abf_cal*.fits'))[0]
    fluxcal_image_WP2 = data.Image(fluxcal_file_WP2)

    logger.info('='*80)
    logger.info('Test Case 5: Output Calibration Product Data Format and Content for WP2')
    logger.info('='*80)
    ## check that the calibration file is configured correctly
    # check HDU0 have no data
    if not hasattr(fluxcal_image_WP2.pri_hdr, "data"):
        logger.info("HDU0: Header only. Expected: header only. PASS.")
    else:
        logger.info(f"HDU0: Contains data. Expected: header only. FAIL.")
    # check HDU1 data is a single float
    if fluxcal_image_WP2.data.dtype.type == corgidrp.image_dtype:
        logger.info(f"HDU1 dtype: float. Expected: float. PASS.")
    else:
        logger.info(f"HDU1 dtype: {fluxcal_image_WP2.data.dtype.type}. Expected: float. FAIL.")
    # check err and dq have the right dimension
    if fluxcal_image_WP2.err.shape == (1,):
        logger.info(f"Err data shape: (1,). Expected: (1,). PASS.")
    else:
        logger.info(f"Error data shape: {fluxcal_image_WP2.err.shape}. Expected: (1,). FAIL.")
    if fluxcal_image_WP2.dq.shape == (1,):
        logger.info(f"DQ data shape: (1,). Expected: (1,). PASS.")
    else:
        logger.info(f"DQ data shape: {fluxcal_image_WP2.dq.shape}. Expected: (1,). FAIL.")
    # check filename convention
    check_filename_convention(getattr(fluxcal_image_WP2, 'filename', None), 'abf_cal.fits', logger=logger)
    # check header keyword values match with what is expected
    verify_header_keywords(fluxcal_image_WP2.ext_hdr, {'DATALVL': 'CAL'}, logger=logger)
    verify_header_keywords(fluxcal_image_WP2.ext_hdr, {'DATATYPE': 'FluxcalFactor'}, logger=logger)
    verify_header_keywords(fluxcal_image_WP2.ext_hdr, {'DPAMNAME': flux_image_WP2_1.ext_hdr['DPAMNAME']}, logger=logger)
    verify_header_keywords(fluxcal_image_WP2.ext_hdr, {'CFAMNAME': flux_image_WP2_1.ext_hdr['CFAMNAME']}, logger=logger)
    logger.info("")

    # baseline performance check WP2
    logger.info('='*80)
    logger.info('Test Case 6: Baseline Performance Checks for WP2')
    logger.info('='*80)
    flux_fac_WP2 = data.FluxcalFactor(fluxcal_file_WP2)
    logger.info(f"used color filter {flux_fac_WP2.filter}")
    logger.info(f"used ND filter {flux_fac_WP2.nd_filter}")
    logger.info(f"fluxcal factor {flux_fac_WP2.fluxcal_fac}")
    logger.info(f"fluxcal factor error {flux_fac_WP2.fluxcal_err}")
    for i, frame in enumerate(flux_dataset_WP2):
        frame_info = f"Frame {i}"
        frame_name = getattr(frame, 'filename', None)
        # log input FSMX and FSMY
        logger.info(f"Frame {frame_name} in flux_dataset_WP2 have FSMX {frame.ext_hdr['FSMX']}")
        logger.info(f"Frame {frame_name} in flux_dataset_WP2 have FSMY {frame.ext_hdr['FSMY']}")
    # log output FSMX and FSMY
    logger.info(f"Output fluxcal file have FSMX {flux_fac_WP2.ext_hdr['FSMX']}")
    logger.info(f"Output fluxcal file have FSMY {flux_fac_WP2.ext_hdr['FSMY']}")
    logger.info("")
    assert flux_fac_WP2.fluxcal_fac == pytest.approx(cal_factor, abs = 1.5 * flux_fac_WP2.fluxcal_err)

    #check the flux values are similar regardless of the wollaston used
    assert flux_fac_WP1.fluxcal_fac == pytest.approx(flux_fac_WP2.fluxcal_fac, rel=0.05)

    compare_to_mocks_hdrs(fluxcal_file_WP2)

    # clean up
    this_caldb = caldb.CalDB()
    this_caldb.remove_entry(flux_fac_WP1)
    this_caldb.remove_entry(flux_fac_WP2)
    logger.info('='*80)
    logger.info('END-TO-END TEST COMPLETE')
    logger.info('='*80)


if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir
    e2edata_dir =  "/home/ericshen/corgi/E2E_Test_Data/"

    ap = argparse.ArgumentParser(description="run the l2b-> PolFluxcalFactor end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_e2e(e2edata_dir, outputdir)