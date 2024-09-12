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

thisfile_dir = os.path.dirname(__file__)  # this file's folder


def set_obstype_for_tvac(
    list_of_fits,
    ):
    """ Adds proper values to OBSTYPE for the non-linearity calibration: NONLIN,
    (data used to calibrate the non-linearity) or MNFRAME (data used to build
    a mean frame).

    This function is unnecessary with future data because data will have
    the proper values in OBSTYPE. The TVAC data used must be the
    following 382 files with IDs: 51841-51870 (30: mean frame). And NL:
    51731-51840 (110), 51941-51984 (44), 51986-52051 (66), 55122-55187 (66),
    55191-55256 (66)  

    Args:
    list_of_fits (list): list of FITS files that need to be updated.

    """
    # Folder with files
    nonlin_dir = list_of_fits[0][0:len(list_of_fits[0]) - list_of_fits[0][::-1].find('/')]
    # TVAC files
    tvac_file_0 = [
        'CGI_EXCAM_L1_0000051841.fits',
        'CGI_EXCAM_L1_0000051731.fits',
        'CGI_EXCAM_L1_0000051941.fits',
        'CGI_EXCAM_L1_0000051986.fits',
        'CGI_EXCAM_L1_0000055122.fits',
        'CGI_EXCAM_L1_0000055191.fits',
        ]

    n_files = [30, 110, 44, 66, 66, 66]
    if len(tvac_file_0) != len(n_files):
        raise ValueError(f'Inconsistent number of files{n_files} and stacks {len(tvac_file_0)}')

    for i_group, file in enumerate(tvac_file_0):
        l1_number = int(file[file.find('L1_')+3:file.find('L1_')+13])
        print(f'Group of {n_files[i_group]} files starting with {file}')
        for i_file in range(n_files[i_group]):
            file_name = f'CGI_EXCAM_L1_00000{l1_number+i_file}.fits'
            # Additional check
            if np.any([nonlin_dir+file_name == file for file in list_of_fits]) is False:
                raise IOError(f'The file {nonlin_dir+file} is not part of the calibration data')
            fits_file = fits.open(nonlin_dir+file_name)
            prihdr = fits_file[0].header
            # Adjust OBSTYPE
            if n_files[i_group] == 30:
                prihdr['OBSTYPE'] = 'MNFRAME'
            else:
                prihdr['OBSTYPE'] = 'NONLIN'
            # Update FITS file
            fits_file.writeto(nonlin_dir+file_name, overwrite=True)


def get_first_nonlin_file(
    list_of_fits,
    ):
    """ Returns the first FITS file with the NONLIN value on OBSTYPE in a list
        of FITS files.

        Remember that FITS files used for NL calibration must have DATETIME in
        ascending order.

        Args:
        list_of_fits (list): list of FITS files that need to be updated.

        Returns:
        first_fits_file (str): First FITS file with OBSTYPE set to NONLIN.

    """
    first_fits_file = 'NONLIN not found'
    for file in list_of_fits:
        fits_file = fits.open(file)
        if fits_file[0].header['OBSTYPE'] == 'NONLIN':
            first_fits_file = fits_file.filename()
            break
    return first_fits_file

@pytest.mark.e2e
def test_nonlin_cal_e2e(
    tvacdata_dir,
    output_dir,
    ):
    """ Performs the e2e test to generate a non-linearity calibration object
        from raw L1 data and compares with the existing TVAC correction for the
        same data.

        Args:
        tvacdata_dir (str): Location of L1 data used to generate the non-linearity
            calibration.
        output_dir (str): Location of the output products: recipe, non-linearity
            calibration FITS file and summary figure with a comparison of the NL
            coefficients for different values of DN and EM is stored.

    """

    # figure out paths, assuming everything is located in the same relative location
    nonlin_l1_datadir = os.path.join(tvacdata_dir,
        'TV-20_EXCAM_noise_characterization', 'nonlin')
    tvac_caldir = os.path.join(tvacdata_dir, 'TV-36_Coronagraphic_Data', 'Cals')
    output_dir = os.path.join(output_dir, 'l1_to_l2a_output')

    if not os.path.exists(nonlin_l1_datadir):
        raise FileNotFoundError('Please store L1 data used to calibrate non-linearity',
            f'in {nonlin_l1_datadir}')

    if not os.path.exists(tvac_caldir):
        raise FileNotFoundError(f'Please store L1 calibration data in {tvac_caldir}')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Define the raw science data to process
    nonlin_l1_list = glob.glob(os.path.join(nonlin_l1_datadir, "*.fits"))
    nonlin_l1_list.sort()

    # Set TVAC OBSTYPE to MNFRAME/NONLIN (flight data should have these values)
    set_obstype_for_tvac(nonlin_l1_list)

    first_nonlin_file = get_first_nonlin_file(nonlin_l1_list)

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
    nonlinear_cal.save(filedir=output_dir, filename="nonlin_tvac.fits" )

    # Run the walker on some test_data
    print('Running walker')
    walker.walk_corgidrp(nonlin_l1_list, '', output_dir)

    # Compare results
    print('Comparing the results with TVAC')
    # NL from CORGIDRP
    nonlin_out_filename = first_nonlin_file[len(first_nonlin_file) -
        first_nonlin_file[::-1].find(os.path.sep):]
    if nonlin_out_filename.find('fits') == -1:
        raise IOError('Data files must be FITS files')
    nonlin_out_filename = nonlin_out_filename[0:nonlin_out_filename.find('fits')-1]
    nonlin_out_filename += '_NonLinearityCalibration.fits'
    nonlin_out = fits.open(os.path.join(output_dir, nonlin_out_filename))
    if nonlin_out[0].header['OBSTYPE'] != 'NONLIN':
        raise ValueError('Calibration type is not NL')
    nonlin_out_table = nonlin_out[1].data

    # NL from TVAC
    nonlin_tvac = fits.open(os.path.join(output_dir,'nonlin_tvac.fits'))
    nonlin_tvac_table = nonlin_tvac[1].data

    # Check
    if (nonlin_out_table.shape[0] != nonlin_tvac_table.shape[0] or
        nonlin_out_table.shape[1] != nonlin_tvac_table.shape[1]):
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
    plt.savefig(os.path.join(output_dir,nonlin_out_filename[:-5]))
    print(f'NL differences wrt ENG/TVAC delivered code ({nonlin_table_from_eng}): ' +
        f'max={np.abs(rel_out_tvac_perc).max():1.1e} %, ' + 
        f'rms={np.std(rel_out_tvac_perc):1.1e} %')
    print(f'Figure saved: {os.path.join(output_dir,nonlin_out_filename[:-5])}.png')

    # Set a quantitative test for the comparison
    assert np.less(np.abs(rel_out_tvac_perc).max(), 1e-4)
   # Print success message
    print('e2e test for NL passed')

if __name__ == "__main__":

    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.

    TVACDATA_DIR = "/Users/srhildeb/Documents/GitHub/CGI_TVAC_Data/"
    OUTPUT_DIR = thisfile_dir

    ap = argparse.ArgumentParser(description="run the non-linearity end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=TVACDATA_DIR,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--output_dir", default=OUTPUT_DIR,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    # Run the e2e test
    test_nonlin_cal_e2e(args.tvacdata_dir, args.output_dir)
