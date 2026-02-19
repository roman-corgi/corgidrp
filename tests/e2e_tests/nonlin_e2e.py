""" Module to test the generation of the non-linearity calibration """
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

@pytest.mark.e2e
def test_nonlin_cal_e2e(
    e2edata_path,
    e2eoutput_path,
    ):
    """ Performs the e2e test to generate a non-linearity calibration object
        from raw L1 data and compares with the existing TVAC correction for the
        same data.  
        NOTE:  The original II&T code did not have a restriction on the number of
        frames per EM gain, but the CORGI DRP does, and the default number is 20.  
        For this e2e test, we use 3 EM gains, and for one of those EM gains, there 
        are only 14 frames in the e2e test data.  So, we set the keyword n_cal=14 below 
        before running the steps through the walker.

        Args:
        e2edata_path (str): Location of L1 data used to generate the non-linearity
            calibration.
        e2eoutput_path (str): Location of the output products: recipe, non-linearity
            calibration FITS file and summary figure with a comparison of the NL
            coefficients for different values of DN and EM is stored.

    """

    # figure out paths, assuming everything is located in the same relative location
    nonlin_l1_datadir = os.path.join(e2edata_path,
        'TV-20_EXCAM_noise_characterization', 'nonlin')
    kgain_l1_datadir = os.path.join(e2edata_path,
        'TV-20_EXCAM_noise_characterization', 'nonlin', 'kgain')
    tvac_caldir = os.path.join(e2edata_path, 'TV-36_Coronagraphic_Data', 'Cals')
    e2eoutput_path = os.path.join(e2eoutput_path, 'nonlin_cal_e2e')

    if not os.path.exists(nonlin_l1_datadir):
        raise FileNotFoundError('Please store L1 data used to calibrate non-linearity',
            f'in {nonlin_l1_datadir}')
    if not os.path.exists(kgain_l1_datadir):
        raise FileNotFoundError('Please store L1 data used to calibrate kgain',
            f'in {kgain_l1_datadir}')
    if not os.path.exists(tvac_caldir):
        raise FileNotFoundError(f'Please store L1 calibration data in {tvac_caldir}')

    if os.path.exists(e2eoutput_path):
        shutil.rmtree(e2eoutput_path)
    os.makedirs(e2eoutput_path)

    # Create input_data subfolder
    input_data_dir = os.path.join(e2eoutput_path, 'input_l1')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)
    
    # Create calibrations subfolder
    calibrations_dir = os.path.join(e2eoutput_path, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)
    
    # Create tvac_reference subfolder and copy reference nonlinearity calibration
    tvac_reference_dir = os.path.join(e2eoutput_path, 'tvac_reference')
    if not os.path.exists(tvac_reference_dir):
        os.makedirs(tvac_reference_dir)
    
    # Copy TVAC reference nonlinearity calibration file
    tvac_nonlin_file = os.path.join(tvac_caldir, 'nonlin_8_11_25.fits')
    if os.path.exists(tvac_nonlin_file):
        mocks.rename_files_to_cgi_format(list_of_fits=[tvac_nonlin_file], output_dir=tvac_reference_dir, level_suffix="nln_cal")
    else:
        raise FileNotFoundError(f"TVAC reference nonlinearity file not found at {tvac_nonlin_file}")

    # Define the raw science data to process
    nonlin_l1_list = glob.glob(os.path.join(nonlin_l1_datadir, "*.fits"))
    nonlin_l1_list.sort()
    kgain_l1_list = glob.glob(os.path.join(kgain_l1_datadir, "*.fits"))
    kgain_l1_list.sort()
    nonlin_l1_list = nonlin_l1_list + kgain_l1_list

    # Copy files to input_data directory and update file list
    nonlin_l1_list = [
        shutil.copy2(file_path, os.path.join(input_data_dir, os.path.basename(file_path)))
        for file_path in nonlin_l1_list
    ]

    # Update headers for TVAC files
    nonlin_l1_list = check.fix_hdrs_for_tvac(
        nonlin_l1_list,
        input_data_dir,
        header_template=mocks.create_default_L1_headers,
    )
    for file in nonlin_l1_list:
        with fits.open(file, mode='update') as fits_file:
            prihdr = fits_file[0].header
            prihdr['VISTYPE'] = 'CGIVST_CAL_PUPIL_IMAGING'
            prihdr['PHTCNT'] = 0

    # Non-linearity calibration file used to compare the output from CORGIDRP:
    # We are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    #nonlin_table_from_eng = 'nonlin_table_091224.txt'
    #nonlin_dat = np.genfromtxt(os.path.join(tvac_caldir,nonlin_table_from_eng),
        #delimiter=",")
    nonlin_table_from_eng = 'nonlin_8_11_25.fits' #II&T code run on updated SSC TVAC files
    nonlin_dat = fits.getdata(os.path.join(tvac_caldir,nonlin_table_from_eng))
    pri_hdr, ext_hdr = mocks.create_default_L1_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(nonlin_l1_list)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")

    # KGain
    kgain_val = 8.7
    kgain = data.KGain(kgain_val, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    
    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(kgain)
    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    # Run the walker on some test_data
    print('Running walker')
    #walker.walk_corgidrp(nonlin_l1_list, '', e2eoutput_path, "l1_to_l2a_nonlin.json")
    recipe = walker.autogen_recipe(nonlin_l1_list, e2eoutput_path)
    ### Modify they keywords of some of the steps
    for step in recipe[0]['steps']:
        if step['name'] == "calibrate_nonlin":
            step['keywords']['apply_dq'] = False # full shaped pupil FOV
            step['keywords']['n_cal'] = 14 #fewer SSC frames found, and this works fine for II&T code
    walker.run_recipe(recipe[0], save_recipe_file=True)
    # Compare results
    print('Comparing the results with TVAC')
    # NL from CORGIDRP
    possible_nonlin_files = glob.glob(os.path.join(e2eoutput_path, '*_nln_cal*.fits'))
    nonlin_drp_filepath = max(possible_nonlin_files, key=os.path.getmtime) # get the one most recently modified
    nonlin_drp_filename = nonlin_drp_filepath.split(os.path.sep)[-1]

    nonlin_out = fits.open(nonlin_drp_filepath)
    nonlin_out_table = nonlin_out[1].data
    n_emgain = nonlin_out_table.shape[1]

    # NL from TVAC - find the actual reference file
    nonlin_tvac_files = glob.glob(os.path.join(tvac_reference_dir, '*nln_cal.fits'))
    if not nonlin_tvac_files:
        raise FileNotFoundError(f"No nonlinearity calibration file found in {tvac_reference_dir}")
    nonlin_tvac_file = nonlin_tvac_files[0]
    nonlin_tvac = fits.open(nonlin_tvac_file)
    nonlin_tvac_table = nonlin_tvac[1].data

    # Check
    if (nonlin_out_table.shape[0] != nonlin_tvac_table.shape[0] or
        n_emgain != nonlin_tvac_table.shape[1]):
        raise ValueError('Non-linearity table from CORGI DRP has a different',
            'format than the one from TVAC')

    rel_out_tvac_perc = 100*(nonlin_out_table[1:,1:]/nonlin_tvac_table[1:,1:]-1)

    # Summary figure
    plt.figure(figsize=(10,6))
    em_list = nonlin_out_table[0,1:]
    for i_em, em_val in enumerate(em_list):
        plt.plot(nonlin_out_table[1:,0], rel_out_tvac_perc[:,i_em], label=f'EM={em_val:.1f}')
    plt.xlabel('DN value', fontsize=14)
    plt.ylabel('Relative difference (%)', fontsize=14)
    plt.title('Comparison of ENG/CORGI DRP NL table for a given DN and EM value',
        fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(e2eoutput_path,nonlin_drp_filename[:-5]+".png"))
    print(f'NL differences wrt ENG/TVAC delivered code ({nonlin_table_from_eng}): ' +
        f'max={np.abs(rel_out_tvac_perc).max():1.1e} %, ' + 
        f'rms={np.std(rel_out_tvac_perc):1.1e} %')
    print(f'Figure saved: {os.path.join(e2eoutput_path,nonlin_drp_filename[:-5])}.png')

    # Set a quantitative test for the comparison
    assert np.less(np.abs(rel_out_tvac_perc).max(), 1e-4)

    check.compare_to_mocks_hdrs(nonlin_drp_filepath, mocks.create_default_L2a_headers)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)
    # Print success message
    print('e2e test for nonlin calibration passed')

if __name__ == "__main__":

    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.

    e2edata_dir = '/Users/kevinludwick/Documents/DRP_E2E_Test_Files_v2/E2E_Test_Data'# '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    #e2edata_dir = "/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/Working_Folder/"#'/home/jwang/Desktop/CGI_TVAC_Data/'
    OUTPUT_DIR = thisfile_dir

    ap = argparse.ArgumentParser(description="run the non-linearity end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=OUTPUT_DIR,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    # Run the e2e test
    test_nonlin_cal_e2e(args.e2edata_dir, args.outputdir)
