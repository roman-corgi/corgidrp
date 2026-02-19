""" Module to test the generation of both nonlin and kgain from the same PUPILIMG dataset """
import os
import glob
import argparse
import pytest
import numpy as np
from astropy import time
from astropy.io import fits
import matplotlib.pyplot as plt

import corgidrp
from corgidrp import data
from corgidrp import mocks
from corgidrp import walker
from corgidrp import caldb
from corgidrp import check
import shutil

thisfile_dir = os.path.dirname(__file__)  # this file's folder

def set_vistype_for_tvac(
    list_of_fits,
    ):
    """ Adds proper values to VISTYPE for non-linearity calibration.

    This function is unnecessary with future data because data will have
    the proper values in VISTYPE. Hence, the "tvac" string in its name.

    Args:
    list_of_fits (list): list of FITS files that need to be updated.
    """
    print("Adding VISTYPE='CGIVST_CAL_PUPIL_IMAGING' to TVAC data")
    for file in list_of_fits:
        fits_file = fits.open(file)
        prihdr = fits_file[0].header
        # Adjust VISTYPE
        if prihdr['VISTYPE'] == 'N/A':
            prihdr['VISTYPE'] = 'CGIVST_CAL_PUPIL_IMAGING'
            prihdr['VISTYPE'] = 'CGIVST_CAL_PUPIL_IMAGING'
        exthdr = fits_file[1].header
        if exthdr['EMGAIN_A'] == 1:
            exthdr['EMGAIN_A'] = -1 #for new SSC-updated TVAC files which have EMGAIN_A by default as 1 regardless of the commanded EM gain
        # Update FITS file
        fits_file.writeto(file, overwrite=True)


@pytest.mark.e2e
def test_nonlin_and_kgain_e2e(
    e2edata_path,
    e2eoutput_path,
    ):
    """ 
    Performs the e2e test to generate both nonlin and kgain calibrations from the same
    L1 pupilimg dataset.
    NOTE:  The original II&T code for nonlin calibration did not have a restriction on the number of
        frames per EM gain, but the CORGI DRP does, and the default number is 20.  
        For this e2e test, we use 3 EM gains, and for one of those EM gains, there 
        are only 14 frames in the e2e test data.  So, we set the keyword n_cal=14 below 
        before running the steps through the walker.

    Args:
        e2edata_path (str): Location of L1 data. Folders for both kgain and nonlin
        e2eoutput_path (str): Location of the output products: recipes, non-linearity
            calibration FITS file, and kgain fits file

    """

    # figure out paths, assuming everything is located in the same relative location
    nonlin_l1_datadir = os.path.join(e2edata_path,
        'TV-20_EXCAM_noise_characterization', 'nonlin')
    kgain_l1_datadir = os.path.join(e2edata_path,
        'TV-20_EXCAM_noise_characterization', 'nonlin', 'kgain')

    e2eoutput_path = os.path.join(e2eoutput_path, 'nonlin_kgain_cal_e2e')

    if not os.path.exists(nonlin_l1_datadir):
        raise FileNotFoundError('Please store L1 data used to calibrate non-linearity',
            f'in {nonlin_l1_datadir}')
    if not os.path.exists(kgain_l1_datadir):
        raise FileNotFoundError('Please store L1 data used to calibrate kgain',
            f'in {kgain_l1_datadir}')

    if os.path.exists(e2eoutput_path):
        shutil.rmtree(e2eoutput_path)
    os.makedirs(e2eoutput_path)

    # Create input_data subfolder
    input_data_dir = os.path.join(e2eoutput_path, 'input_l1')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()

    # Define the raw science data to process
    nonlin_l1_list = glob.glob(os.path.join(nonlin_l1_datadir, "*.fits"))
    nonlin_l1_list.sort()
    kgain_l1_list = glob.glob(os.path.join(kgain_l1_datadir, "*.fits"))
    kgain_l1_list.sort()
    
    pupilimg_l1_list = list(nonlin_l1_list)
    for filepath in kgain_l1_list:
        if filepath.lower().endswith('.fits'):
            pupilimg_l1_list.append(filepath)

    # Copy files to input_data directory
    pupilimg_l1_list = [
        shutil.copy2(file_path, os.path.join(input_data_dir, os.path.basename(file_path)))
        for file_path in pupilimg_l1_list
    ]

    # Fix headers for TVAC
    pupilimg_l1_list = check.fix_hdrs_for_tvac(pupilimg_l1_list, input_data_dir)

    # Set TVAC data to have VISTYPE=CGIVST_CAL_PUPIL_IMAGING (flight data should have these values)
    set_vistype_for_tvac(pupilimg_l1_list)

   # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    # Run the walker on some test_data
    print('Running walker')
    #walker.walk_corgidrp(pupilimg_l1_list, '', e2eoutput_path)
    recipe = walker.autogen_recipe(pupilimg_l1_list, e2eoutput_path)
    ### Modify they keywords of some of the steps
    for step in recipe[1]['steps']:
        if step['name'] == "calibrate_kgain":
            step['keywords']['apply_dq'] = False #do not apply the cosmics in e2etests
    walker.run_recipe(recipe[1], save_recipe_file=True)
    for step in recipe[0]['steps']:
        if step['name'] == "calibrate_nonlin":
            step['keywords']['apply_dq'] = False #do not apply the cosmics in e2etests
            step['keywords']['n_cal'] = 14 #fewer SSC frames found, and this works fine for II&T code
    walker.run_recipe(recipe[0], save_recipe_file=True)

    # check that files can be loaded from disk successfully. no need to check correctness as done in other e2e tests
    # NL from CORGIDRP
    possible_nonlin_files = glob.glob(os.path.join(e2eoutput_path, '*_nln_cal*.fits'))
    nonlin_drp_filepath = max(possible_nonlin_files, key=os.path.getmtime) # get the one most recently modified
    nonlin = data.NonLinearityCalibration(nonlin_drp_filepath)

    # kgain from corgidrp
    possible_kgain_files = glob.glob(os.path.join(e2eoutput_path, '*_krn_cal*.fits'))
    kgain_filepath = max(possible_kgain_files, key=os.path.getmtime) # get the one most recently modified
    kgain = data.KGain(kgain_filepath)

    check.compare_to_mocks_hdrs(nonlin_drp_filepath, mocks.create_default_L2a_headers)
    check.compare_to_mocks_hdrs(kgain_filepath, mocks.create_default_L2a_headers)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

    # Print success message
    print('e2e test for nonlin_kgain calibration passed')

if __name__ == "__main__":

    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.

    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'#"/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/Working_Folder/"#'/home/jwang/Desktop/CGI_TVAC_Data/'
    #e2edata_dir = "/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/Working_Folder/"
    OUTPUT_DIR = thisfile_dir

    ap = argparse.ArgumentParser(description="run the non-linearity end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=OUTPUT_DIR,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    # Run the e2e test
    test_nonlin_and_kgain_e2e(args.e2edata_dir, args.outputdir)
