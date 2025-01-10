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

thisfile_dir = os.path.dirname(__file__)  # this file's folder

def set_vistype_for_tvac(
    list_of_fits,
    ):
    """ Adds proper values to VISTYPE for non-linearity calibration.

    This function is unnecessary with future data because data will have
    the proper values in VISTYPE. Hence, the "tvac" string in its name.
    For reference, TVAC data used to calibrate non-linearity were the
    following 382 files with IDs: 51841-51870 (30: mean frame). And NL:
    51731-51840 (110), 51941-51984 (44), 51986-52051 (66), 55122-55187 (66),
    55191-55256 (66)  

    Args:
    list_of_fits (list): list of FITS files that need to be updated.
    """
    print("Adding VISTYPE='PUPILIMG' to TVAC data")
    for file in list_of_fits:
        fits_file = fits.open(file)
        prihdr = fits_file[0].header
        # Adjust VISTYPE
        prihdr['VISTYPE'] = 'PUPILIMG'
        # Update FITS file
        fits_file.writeto(file, overwrite=True)


@pytest.mark.e2e
def test_nonlin_cal_e2e(
    tvacdata_path,
    e2eoutput_path,
    ):
    """ Performs the e2e test to generate a non-linearity calibration object
        from raw L1 data and compares with the existing TVAC correction for the
        same data.

        Args:
        tvacdata_path (str): Location of L1 data used to generate the non-linearity
            calibration.
        e2eoutput_path (str): Location of the output products: recipe, non-linearity
            calibration FITS file and summary figure with a comparison of the NL
            coefficients for different values of DN and EM is stored.

    """

    # figure out paths, assuming everything is located in the same relative location
    nonlin_l1_datadir = os.path.join(tvacdata_path,
        'TV-20_EXCAM_noise_characterization', 'nonlin')
    tvac_caldir = os.path.join(tvacdata_path, 'TV-36_Coronagraphic_Data', 'Cals')
    e2eoutput_path = os.path.join(e2eoutput_path, 'l1_to_nonlin_output')

    if not os.path.exists(nonlin_l1_datadir):
        raise FileNotFoundError('Please store L1 data used to calibrate non-linearity',
            f'in {nonlin_l1_datadir}')

    if not os.path.exists(tvac_caldir):
        raise FileNotFoundError(f'Please store L1 calibration data in {tvac_caldir}')

    if not os.path.exists(e2eoutput_path):
        os.mkdir(e2eoutput_path)

    # Define the raw science data to process
    nonlin_l1_list = glob.glob(os.path.join(nonlin_l1_datadir, "*.fits"))
    nonlin_l1_list.sort()

    # Set TVAC OBSTYPE to MNFRAME/NONLIN (flight data should have these values)
    set_vistype_for_tvac(nonlin_l1_list)

    # Non-linearity calibration file used to compare the output from CORGIDRP:
    # We are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    nonlin_table_from_eng = 'nonlin_table_091224.txt'
    nonlin_dat = np.genfromtxt(os.path.join(tvac_caldir,nonlin_table_from_eng),
        delimiter=",")
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(nonlin_l1_list)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=e2eoutput_path, filename="nonlin_tvac.fits" )
    
    
    # KGain
    kgain_val = 8.7
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=e2eoutput_path, filename="mock_kgain.fits")
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(kgain)

    # Run the walker on some test_data
    print('Running walker')
    walker.walk_corgidrp(nonlin_l1_list, '', e2eoutput_path, "l1_to_l2a_nonlin.json")

    # Compare results
    print('Comparing the results with TVAC')
    # NL from CORGIDRP
    possible_nonlin_files = glob.glob(os.path.join(e2eoutput_path, '*_NonLinearityCalibration.fits'))
    nonlin_drp_filepath = max(possible_nonlin_files, key=os.path.getmtime) # get the one most recently modified
    nonlin_drp_filename = nonlin_drp_filepath.split(os.path.sep)[-1]

    nonlin_out = fits.open(nonlin_drp_filepath)
    nonlin_out_table = nonlin_out[1].data
    n_emgain = nonlin_out_table.shape[1]

    # NL from TVAC
    nonlin_tvac = fits.open(os.path.join(e2eoutput_path,'nonlin_tvac.fits'))
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

    # remove entry from caldb
    nonlin_entry = data.NonLinearityCalibration(os.path.join(e2eoutput_path, nonlin_drp_filename))
    this_caldb.remove_entry(nonlin_entry)
    this_caldb.remove_entry(kgain)
   # Print success message
    print('e2e test for NL passed')

if __name__ == "__main__":

    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.

    TVACDATA_DIR = '/home/jwang/Desktop/CGI_TVAC_Data/'
    OUTPUT_DIR = thisfile_dir

    ap = argparse.ArgumentParser(description="run the non-linearity end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=TVACDATA_DIR,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--output_dir", default=OUTPUT_DIR,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    # Run the e2e test
    test_nonlin_cal_e2e(args.tvacdata_dir, args.output_dir)
