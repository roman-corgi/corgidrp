import argparse
import os
import numpy as np
from corgidrp.data import AstrometricCalibration

import corgidrp
import corgidrp.caldb as caldb
import corgidrp.mocks as mocks
import corgidrp.astrom as astrom
import corgidrp.walker as walker
import pytest
import glob


thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_l2b_to_distortion(e2edata_path, e2eoutput_path):
    '''

    An end-to-end test that creates L2b level data and runs it through the astrometric calibration distortion map solution step.

        It checks that: 
            - Distortion map coefficients are recorded correctly
            - The distortion map error in the central 1" x 1" detector region is < 4 [mas] (~0.1835 [pixel])

        Data needed: 
            - L2b level dataset - created in the test
    
    Args:
        e2edata_path (str): Path to the test data
        e2eoutput_path (str): Path to the output directory


    '''

    distortion_outputdir = os.path.join(e2eoutput_path, "l2b_to_distortion_output")
    if not os.path.exists(distortion_outputdir):
        os.mkdir(distortion_outputdir)

    e2e_mockdata_path = os.path.join(distortion_outputdir, "astrom_distortion")
    if not os.path.exists(e2e_mockdata_path):
        os.mkdir(e2e_mockdata_path)


    #################################
    #### Generate a mock dataset ####
    #################################

    field_path = os.path.join(os.path.dirname(__file__),"..","test_data", "JWST_CALFIELD2020.csv")
    distortion_coeffs_path = os.path.join(os.path.dirname(__file__),"..","test_data", "distortion_expected_coeffs.csv")

    #Create the mock dataset
    mock_dataset = mocks.create_astrom_data(field_path=field_path, filedir=e2e_mockdata_path, rotation=20, distortion_coeffs_path=distortion_coeffs_path, dither_pointings=3)
    # update headers to be L2b level
    l2b_pri_hdr, l2b_ext_hdr, errhdr, dqhdr, biashdr = mocks.create_default_L2b_headers()
    for mock_image in mock_dataset:
        mock_image.pri_hdr = l2b_pri_hdr
        mock_image.pri_hdr['RA'], mock_image.pri_hdr['DEC'] = 80.553428801, -69.514096821
        mock_image.ext_hdr = l2b_ext_hdr

    # expected_platescale, expected_northangle = 21.8, 20.
    expected_coeffs = np.genfromtxt(distortion_coeffs_path)

    #####################################
    #### Pass the data to the walker ####
    #####################################

    l2b_data_filelist = sorted(glob.glob(os.path.join(e2e_mockdata_path, "*.fits")))
    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"corgidrp","recipe_templates","l2b_to_distortion.json")

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)

    # template_path = '/Users/macuser/Roman/corgidrp/corgidrp/recipe_templates/l2b_to_distortion.json'
    walker.walk_corgidrp(l2b_data_filelist, "", distortion_outputdir, template=template_path)

    #Read in th Astrometric Calibration file
    ast_cal_filename = glob.glob(os.path.join(distortion_outputdir, "*ast_cal.fits"))[0]
    ast_cal = AstrometricCalibration(ast_cal_filename)

    #Check that distortion map error within the central 1" x 1" region of the detector is <4 [mas] (~0.1835 [pixel])
    comp_coeffs = ast_cal.distortion_coeffs[:-1]
    comp_order = int(ast_cal.distortion_coeffs[-1])
    comp_shape = mock_dataset[0].data.shape
    xdiff, ydiff = astrom.transform_coeff_to_map(comp_coeffs, comp_order, comp_shape)

    true_coeffs = expected_coeffs[:-1]
    true_order = int(expected_coeffs[-1])
    true_shape = np.array([1024, 1024])
    true_xdiff, true_ydiff = astrom.transform_coeff_to_map(true_coeffs, true_order, true_shape)

        # check only the central square arcsecond
    lower_lim, upper_lim = int((1024//2) - ((1000/21.8)//2)), int((1024//2) + ((1000/21.8)//2))
    central_1arcsec_x = xdiff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]
    central_1arcsec_y = ydiff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]

    true_1arcsec_x = true_xdiff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]
    true_1arcsec_y = true_ydiff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]

    assert np.all(np.abs(central_1arcsec_x - true_1arcsec_x) < 0.1835)
    assert np.all(np.abs(central_1arcsec_y - true_1arcsec_y) < 0.1835)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.

    outputdir = thisfile_dir
    e2edata_dir = '/Users/macuser/Roman/corgidrp_develop/calibration_notebooks/TVAC'

    ap = argparse.ArgumentParser(description="run the l2b->distortion end-to-end test")

    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir

    test_l2b_to_distortion(e2edata_dir, outputdir)
