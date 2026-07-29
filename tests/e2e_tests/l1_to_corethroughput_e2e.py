# E2E Test Code for l1 to CoreThroughput Calibration

import argparse
import os, shutil
import glob
import pytest
import warnings
import numpy as np
from astropy.io import fits

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.detector as detector
import corgidrp.corethroughput as corethroughput
from corgidrp import caldb
from corgidrp import check
from corgidrp.check import compare_to_mocks_hdrs

# this file's folder
thisfile_dir = os.path.dirname(__file__)

@pytest.mark.e2e
def test_expected_results_band1_nfov_e2e(e2edata_path, e2eoutput_path):
    """Test corethroughput calibration with mock data

    Args:
        e2edata_path (str): Path to the test data
        e2eoutput_path (str): Path to the output directory

    """
    
    # create output directory if none exists
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_corethroughput_e2e")
    if os.path.exists(test_outputdir):
        shutil.rmtree(test_outputdir)
    os.makedirs(test_outputdir)
    
    # grab required sims and mock calibration data
    sci_datadir = os.path.join(e2edata_path, "corethroughput_sims_HLC_B1")

    l2b_outputdir = os.path.join(test_outputdir, "l2b_results")
    if not os.path.exists(l2b_outputdir):
        os.mkdir(l2b_outputdir)
    # clean up by removing old files
    for file in os.listdir(l2b_outputdir):
        os.remove(os.path.join(l2b_outputdir, file))
    
    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir) # grab default calibrations

    # define the raw science data to process
    l1_data_filelist = [os.path.join(sci_datadir, os.listdir(sci_datadir)[i]) for i in range(len(os.listdir(sci_datadir))) if os.listdir(sci_datadir)[i].endswith("l1_.fits")] 

    ## delete some headers introduced in corgisim that aren't part of real data
    for filename in l1_data_filelist:
        with fits.open(filename) as hdulist:
            prihdr = hdulist[0].header
            if 'SATSPOTS' in prihdr:
                del prihdr['SATSPOTS']
            if 'COMMENT' in prihdr:
                del prihdr['COMMENT']
            hdulist.writeto(filename, overwrite=True)
            

    # run the walker
    with warnings.catch_warnings():
        # suppress warning about number of detectornoisemap frames
        warnings.simplefilter("ignore", category=UserWarning)
        walker.walk_corgidrp(l1_data_filelist, "", l2b_outputdir)

    # Load in the output data. It should be the latest ctp_cal file produced.
    corethroughput_drp_file = glob.glob(os.path.join(l2b_outputdir,
        '*ctp_cal.fits'))[0]
    ct_cal_drp = data.CoreThroughputCalibration(corethroughput_drp_file)
    
    # check that the calibration finds the correct PSF locations
    ct_x, ct_y, ct_vals = ct_cal_drp.ct_excam
    recovered_psf_locs = [(ct_x[i], ct_y[i]) for i in range(len(ct_x))] # put into list of tuples
    # using simulation parameters, compute where we expect the PSFs to be
    platescale = 21.8 # detector platescale in mas
    xcen, ycen = 512, 512 # detector center
    # angle and separation parameters for PSF dithering in simulation
    step_rad = [0, 120 * (np.pi/180), 240 * (np.pi/180)] # 0, 120, 240 are the dither angles in degrees, we convert it to radians for computation directly here
    sep_mas = [160, 300, 470]
    # compute expected excam PSF position for each combination of angles and separations
    expected_psf_locs = []
    for theta in step_rad:
        for r in sep_mas:
            # get x and y offset in mas first, then convert to pixels
            dx_pix = - (r * np.cos(theta)) / platescale # negative sign to account for axis reflection between RA and pixel coordinates
            dy_pix = (r * np.sin(theta)) / platescale
            # add offset to center to get expected positions
            expected_psf_locs.append((xcen + dx_pix, ycen + dy_pix))
    # since the recovered psf locations might be ordered differently
    # for each recovered location, we compare it to the list of expected locations and get the closest distance match
    # we then check that this distance match is within a certain tolerance to verify the recovered location is correct
    for loc in recovered_psf_locs:
        # keep track of the minimum distance between the recovered location and the expected psf locations
        min_dist = np.inf # initialize it to some large number to start
        min_dist_idx = -1
        for i in range(len(expected_psf_locs)):
            expected_loc = expected_psf_locs[i]
            # distance between this particular recovered psf location to this particular expected psf location
            dist = np.sqrt((loc[0] - expected_loc[0])**2 + (loc[1] - expected_loc[1])**2)
            # check if this is smaller than the current minimum distance measurement, update if so
            if dist < min_dist:
                min_dist = dist
                min_dist_idx = i
        # assert the minimum distance difference is within a small tolerance
        tol = 3
        assert min_dist < tol
        # remove the corresponding index from the expected psf locations list to prevent accidental double counting
        del expected_psf_locs[min_dist_idx]

    # check headers
    check.compare_to_mocks_hdrs(corethroughput_drp_file, header_template=mocks.create_default_calibration_product_headers)
    assert ct_cal_drp.ext_hdr["DATATYPE"] == "CoreThroughputCalibration"
    assert ct_cal_drp.ext_hdr["DATALVL"] == "CAL"

    # load in the new corethroughput calibration file, call walker again to create CT map
    this_caldb.scan_dir_for_new_entries(l2b_outputdir)

    # test corethroughput map
    ctmap_l1_input_dir = os.path.join(e2edata_path, "ctmap_sims")

    l2a_outputdir = os.path.join(test_outputdir, "l2a_sci_output")
    ctmap_outputdir = os.path.join(test_outputdir, "ctmap_output")
    if not os.path.exists(l2a_outputdir):
        os.mkdir(l2a_outputdir)
    for file in os.listdir(l2a_outputdir):
        os.remove(os.path.join(ctmap_outputdir, file))
    if not os.path.exists(ctmap_outputdir):
        os.mkdir(ctmap_outputdir)
    for file in os.listdir(ctmap_outputdir):
        os.remove(os.path.join(ctmap_outputdir, file))
    
    ctmap_l1_input_filelist = [os.path.join(ctmap_l1_input_dir, os.listdir(ctmap_l1_input_dir)[i]) for i in range(len(os.listdir(ctmap_l1_input_dir))) if os.listdir(ctmap_l1_input_dir)[i].endswith("l1_.fits")]

    # run the walker
    walker.walk_corgidrp(ctmap_l1_input_filelist, "", l2a_outputdir, template="l1_to_l2a_basic.json")
    l2a_input_filelist = [os.path.join(l2a_outputdir, os.listdir(l2a_outputdir)[i]) for i in range(len(os.listdir(l2a_outputdir))) if os.listdir(l2a_outputdir)[i].endswith("l2a.fits")]
    walker.walk_corgidrp(l2a_input_filelist, "", ctmap_outputdir, template="l2a_to_corethroughput_map.json")

    # grab ctmap file
    ctmap_file = glob.glob(os.path.join(ctmap_outputdir,
        '*ctm_cal.fits'))[0]
    ct_map = data.CoreThroughputMap(ctmap_file)

    # check headers
    check.compare_to_mocks_hdrs(ctmap_file)
    assert ct_map.ext_hdr["DATATYPE"] == "CoreThroughputMap"
    assert ct_cal_drp.ext_hdr["DATALVL"] == "CAL"

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

    # Print success message
    print('e2e test for corethroughput calibration with mock data passed')

    

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    outputdir = thisfile_dir
    e2edata_path = '/home/eshen12345/dev/E2E_Test_Data'

    ap = argparse.ArgumentParser(description='run the l1 to CoreThroughput end-to-end test')
    ap.add_argument('-e2e', '--e2edata_dir', default=e2edata_path,
                    help='Path to CGI_TVAC_Data Folder [%(default)s]')
    ap.add_argument('-o', '--outputdir', default=outputdir,
                    help='directory to write results to [%(default)s]')
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_band1_nfov_e2e(args.e2edata_dir, args.outputdir)
