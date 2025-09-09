""" Module to test the generation of both nonlin and kgain from the same PUPILIMG dataset """
import os
import glob
import argparse
import pytest
import numpy as np
from astropy import time
from astropy.io import fits
from datetime import datetime
import matplotlib.pyplot as plt
import shutil

import corgidrp
from corgidrp import data
from corgidrp import mocks
from corgidrp import walker
from corgidrp import caldb

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
    print("Adding VISTYPE='PUPILIMG' to TVAC data")
    for file in list_of_fits:
        fits_file = fits.open(file)
        prihdr = fits_file[0].header
        # Adjust VISTYPE
        if prihdr['VISTYPE'] == 'N/A':
            prihdr['VISTYPE'] = 'PUPILIMG'
        exthdr = fits_file[1].header
        if exthdr['EMGAIN_A'] == 1:
            exthdr['EMGAIN_A'] = -1 #for new SSC-updated TVAC files which have EMGAIN_A by default as 1 regardless of the commanded EM gain
        # Update FITS file
        fits_file.writeto(file, overwrite=True)

def fix_headers_for_tvac(
    list_of_fits,
    ):
    """ 
    Fixes TVAC headers to be consistent with flight headers. 
    Writes headers back to disk

    Args:
        list_of_fits (list): list of FITS files that need to be updated.
    """
    print("Fixing TVAC headers")
    for file in list_of_fits:
        fits_file = fits.open(file)
        prihdr = fits_file[0].header
        exthdr = fits_file[1].header
        # Adjust VISTYPE
        prihdr['OBSNUM'] = prihdr['OBSID']
        exthdr['EMGAIN_C'] = exthdr['CMDGAIN']
        exthdr['EMGAIN_A'] = -1
        exthdr['DATALVL'] = exthdr['DATA_LEVEL']
        prihdr["OBSNAME"] = prihdr['OBSTYPE']
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

    e2eoutput_path = os.path.join(e2eoutput_path, 'nonlin_kgain_output')

    if not os.path.exists(nonlin_l1_datadir):
        raise FileNotFoundError('Please store L1 data used to calibrate non-linearity',
            f'in {nonlin_l1_datadir}')
    if not os.path.exists(kgain_l1_datadir):
        raise FileNotFoundError('Please store L1 data used to calibrate kgain',
            f'in {kgain_l1_datadir}')

    if not os.path.exists(e2eoutput_path):
        os.mkdir(e2eoutput_path)
    # clean up output directory
    for f in os.listdir(e2eoutput_path):
        os.remove(os.path.join(e2eoutput_path, f))

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()

    # Create input_data subfolder
    input_data_dir = os.path.join(e2eoutput_path, "input_data")
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    # Define the raw science data to process
    nonlin_l1_list = glob.glob(os.path.join(nonlin_l1_datadir, "*.fits"))
    nonlin_l1_list.sort()
    kgain_l1_list = glob.glob(os.path.join(kgain_l1_datadir, "*.fits"))
    kgain_l1_list.sort()
    
    pupilimg_l1_list = nonlin_l1_list # start with the nonlin filelist
    for filepath in kgain_l1_list:
        if filepath.lower().endswith('.fits'):
            pupilimg_l1_list.append(filepath)

    # Rename and save input files to input_data subfolder with proper L1 filename convention
    renamed_pupilimg_list = []
    base_time = datetime.now()
    for i, file_path in enumerate(pupilimg_l1_list):
        # Extract visitid from original filename or use index
        current_filename = os.path.basename(file_path)
        if '_L1_' in current_filename:
            # Extract the frame number after '_L1_'
            frame_number = current_filename.split('_L1_')[-1].replace('.fits', '')
            visitid = frame_number.zfill(19)  # Pad with zeros to make 19 digits
        else:
            visitid = f"{i:019d}"  # Fallback - use file index padded to 19 digits
        
        # Generate unique timestamp for each file
        # Handle second and minute rollover properly
        new_second = (base_time.second + i) % 60
        new_minute = (base_time.minute + ((base_time.second + i) // 60)) % 60
        new_hour = (base_time.hour + ((base_time.minute + ((base_time.second + i) // 60)) // 60))
        file_time = base_time.replace(hour=new_hour, minute=new_minute, second=new_second)
        time_str = data.format_ftimeutc(file_time.isoformat())
        
        # Create new filename with proper L1 convention
        new_filename = f'cgi_{visitid}_{time_str}_l1_.fits'
        new_file_path = os.path.join(input_data_dir, new_filename)
        
        # Copy the file to input_data with new name
        shutil.copy2(file_path, new_file_path)
        renamed_pupilimg_list.append(new_file_path)
    
    # Use the renamed files for DRP processing
    pupilimg_l1_list = renamed_pupilimg_list


    # Set TVAC data to have VISTYPE=PUPILIMG (flight data should have these values)
    set_vistype_for_tvac(pupilimg_l1_list)
    #fix_headers_for_tvac(pupilimg_l1_list)

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
    possible_nonlin_files = glob.glob(os.path.join(e2eoutput_path, '*_nln_cal*.fits'))
    nonlin_drp_filepath = max(possible_nonlin_files, key=os.path.getmtime) # get the one most recently modified
    nonlin = data.NonLinearityCalibration(nonlin_drp_filepath)

    # kgain from corgidrp
    possible_kgain_files = glob.glob(os.path.join(e2eoutput_path, '*_krn_cal*.fits'))
    kgain_filepath = max(possible_kgain_files, key=os.path.getmtime) # get the one most recently modified
    kgain = data.KGain(kgain_filepath)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)


   # Print success message
    print('e2e test for NL passed')

if __name__ == "__main__":

    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.

    e2edata_dir = '/Users/jmilton/Documents/CGI/CGI_TVAC_Data/'
    OUTPUT_DIR = thisfile_dir

    ap = argparse.ArgumentParser(description="run the non-linearity end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=OUTPUT_DIR,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    # Run the e2e test
    test_nonlin_and_kgain_e2e(args.e2edata_dir, args.outputdir)
