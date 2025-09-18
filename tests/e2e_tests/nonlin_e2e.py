""" Module to test the generation of the non-linearity calibration """
import os
import glob
import argparse
import pytest
import numpy as np
from astropy import time
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import corgidrp
from corgidrp import data
from corgidrp import mocks
from corgidrp import walker
from corgidrp import caldb
from corgidrp.check import generate_fits_excel_documentation

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
        prihdr['VISTYPE'] = 'PUPILIMG'
        # Adjust other keywords
        prihdr['OBSNUM'] = prihdr['OBSID']
        exthdr['EMGAIN_C'] = exthdr['CMDGAIN']
        exthdr['EMGAIN_A'] = -1
        exthdr['DATALVL'] = exthdr['DATA_LEVEL']
        prihdr["OBSNAME"] = prihdr['OBSTYPE']
        exthdr['DATALVL'] = exthdr['DATA_LEVEL']
        # Update FITS file
        fits_file.writeto(file, overwrite=True)

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
        import shutil
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
    
    # Create tvac_reference subfolder
    tvac_reference_dir = os.path.join(e2eoutput_path, 'tvac_reference')
    if not os.path.exists(tvac_reference_dir):
        os.makedirs(tvac_reference_dir)

    # Define the raw science data to process
    nonlin_l1_list = glob.glob(os.path.join(nonlin_l1_datadir, "*.fits"))
    nonlin_l1_list.sort()
    kgain_l1_list = glob.glob(os.path.join(kgain_l1_datadir, "*.fits"))
    kgain_l1_list.sort()
    nonlin_l1_list = nonlin_l1_list + kgain_l1_list

    # Generate proper filenames with visitid and current time
    current_time = datetime.now().strftime('%Y%m%dt%H%M%S%f')[:-5]
    # Extract visit ID from primary header VISITID keyword of first file
    if nonlin_l1_list:
        with fits.open(nonlin_l1_list[0]) as hdulist:
            prihdr = hdulist[0].header
            visitid = prihdr.get('VISITID', None)
            if visitid is not None:
                # Convert to string and pad to 19 digits
                visitid = str(visitid).zfill(19)
            else:
                # Fallback: use default visitid
                visitid = "0000000000000000000"
    else:
        visitid = "0000000000000000000"

    # Copy files to input_data directory with proper naming
    for i, file_path in enumerate(nonlin_l1_list):
        # Generate unique timestamp by incrementing by 0.1 seconds each time
        unique_time = (datetime.now() + timedelta(milliseconds=i*100)).strftime('%Y%m%dt%H%M%S%f')[:-5]
        new_filename = f'cgi_{visitid}_{unique_time}_l1_.fits'
        new_file_path = os.path.join(input_data_dir, new_filename)
        import shutil
        shutil.copy2(file_path, new_file_path)
    
    # Update nonlin_l1_list to point to new files
    nonlin_l1_list = []
    for f in os.listdir(input_data_dir):
        if f.endswith('.fits'):
            nonlin_l1_list.append(os.path.join(input_data_dir, f))

    # Set TVAC OBSNAME to MNFRAME/NONLIN (flight data should have these values)
    #fix_headers_for_tvac(nonlin_l1_list)
    set_vistype_for_tvac(nonlin_l1_list)

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
    # Generate timestamp for TVAC reference NonLinearityCalibration
    base_time = datetime.now()
    tvac_time_str = data.format_ftimeutc((base_time.replace(second=(base_time.second + 2) % 60)).isoformat())
    tvac_nln_filename = f"cgi_0000000000000000000_{tvac_time_str}_nln_cal.fits"
    nonlinear_cal.save(filedir=tvac_reference_dir, filename=tvac_nln_filename)

    # KGain
    kgain_val = 8.7
    kgain = data.KGain(kgain_val, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    # Generate timestamp for KGain calibration
    kgain_time_str = data.format_ftimeutc((base_time.replace(second=(base_time.second + 1) % 60)).isoformat())
    kgain_filename = f"cgi_0000000000000000000_{kgain_time_str}_krn_cal.fits"
    kgain.save(filedir=calibrations_dir, filename=kgain_filename)
    
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

    # NL from TVAC
    nonlin_tvac = fits.open(os.path.join(tvac_reference_dir, tvac_nln_filename))
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

    # Generate Excel documentation for the nonlinearity calibration product
    nonlin_file = os.path.join(e2eoutput_path, nonlin_drp_filename)
    excel_output_path = os.path.join(e2eoutput_path, "nln_cal_documentation.xlsx")
    generate_fits_excel_documentation(nonlin_file, excel_output_path)
    print(f"Excel documentation generated: {excel_output_path}")

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

    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
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
