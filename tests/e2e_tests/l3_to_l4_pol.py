import os, shutil
import pytest
import logging
import numpy as np
import argparse

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.caldb as caldb

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_l3_to_l4_pol_e2e(e2edata_path, e2eoutput_path):
    # create output dir first
    output_dir = os.path.join(e2eoutput_path, 'polcal_e2e')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # set up logging
    global logger
    log_file = os.path.join(output_dir, 'polcal_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('polcal_e2e')
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

    ## Create the astrometric calibration


    #Need: 
    # - Astrometric cal
    # - System mueller matrix cal without ND filter
    # - System mueller matrix cal with ND filter
    # - L3 data with unocculted observations of science and reference stars
    # - L3 data with sat spots for science and reference stars
    # - L3 science and reference data. 

    # Create calibrations subfolder for mock calibration products
    calibrations_dir = os.path.join(output_dir, "calibrations")
    if not os.path.exists(calibrations_dir):
        os.mkdir(calibrations_dir)

    ###### Setup necessary calibration files
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()  # connection to cal DB
    
    ########################################
    ########### Create astrom cal ##########
    ########################################

    # CREATE DISTORTED POL DATASET
    # Create a bad pixel map for testing
    datashape = (2, 1024, 1024)  # pol data shape
    bpixmap = np.zeros(datashape)

    field_path = os.path.join(os.path.dirname(__file__), "..","test_data", "JWST_CALFIELD2020.csv")

    distortion_coeffs_path = os.path.join(os.path.dirname(__file__), "..","test_data", "distortion_expected_coeffs.csv")
    distortion_coeffs = np.genfromtxt(distortion_coeffs_path)
    distortion_dataset_base = mocks.create_astrom_data(field_path, distortion_coeffs_path=distortion_coeffs_path,
                                                       bpix_map=bpixmap[0], sim_err_map=True)


    astromcal_data = np.concatenate((np.array([80.553428801, -69.514096821, 21.8, 45, 0, 0]), distortion_coeffs),
                                    axis=0)
    astrom_cal = data.AstrometricCalibration(astromcal_data,
                                             pri_hdr=distortion_dataset_base[0].pri_hdr,
                                             ext_hdr=distortion_dataset_base[0].ext_hdr,
                                             input_dataset=distortion_dataset_base)
    mocks.rename_files_to_cgi_format(list_of_fits=[astrom_cal], output_dir=calibrations_dir, level_suffix="ast_cal")
    
    
    this_caldb.create_entry(astrom_cal)

    #################################################
    ########### Create Mueller Matrix cals ##########
    #################################################

        # define mueller matrices and stokes vectors
    nd_mueller_matrix = np.array([
        [0.5, 0.1, 0, 0],
        [0.1, -0.5, 0, 0],
        [0.05, 0.05, 0.5, 0],
        [0, 0, 0, 0.5]
    ])
    system_mueller_matrix = np.array([
        [0.9, -0.02, 0, 0],
        [0.01, -0.8, 0, 0],
        [0, 0, 0.8, 0.005],
        [0, 0, -0.01, 0.9]
    ])

    #Create dataset because we need to input something: 
    prihdr, exthdr = mocks.create_default_L1_headers()
    dummy1 = data.Image(np.zeros((10,10)), pri_hdr=prihdr, ext_hdr=exthdr)
    dummy2 = data.Image(np.zeros((10,10)), pri_hdr=prihdr, ext_hdr=exthdr)
    input_dataset = data.Dataset([dummy1, dummy2])

    mm_prihdr, mm_exthdr, _, _ = mocks.create_default_calibration_product_headers()
    system_mm_cal = data.MuellerMatrix(system_mueller_matrix, pri_hdr=mm_prihdr.copy(), ext_hdr=mm_exthdr.copy(), input_dataset=input_dataset)
    nd_mm_cal = data.MuellerMatrix(nd_mueller_matrix, pri_hdr=mm_prihdr.copy(), ext_hdr=mm_exthdr.copy(), input_dataset=input_dataset)

    mocks.rename_files_to_cgi_format(list_of_fits=[system_mm_cal], output_dir=calibrations_dir, level_suffix="mmx_cal")
    mocks.rename_files_to_cgi_format(list_of_fits=[nd_mm_cal], output_dir=calibrations_dir, level_suffix="ndm_cal")

    this_caldb.create_entry(system_mm_cal)
    this_caldb.create_entry(nd_mm_cal)

    ########################################


if __name__ == "__main__":
    #e2edata_dir = "/Users/macuser/Roman/corgidrp_develop/calibration_notebooks/TVAC"
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2b->boresight end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_l3_to_l4_pol_e2e(e2edata_dir, outputdir)