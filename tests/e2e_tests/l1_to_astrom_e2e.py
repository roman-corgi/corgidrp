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
import corgidrp.astrom as astrom
from corgidrp import caldb
from corgidrp import check

# this file's folder
thisfile_dir = os.path.dirname(__file__)

@pytest.mark.e2e
def test_l1_to_astrom_e2e(e2edata_path, e2eoutput_path):
    """Test astrometric calibration with mock data

    Args:
        e2edata_path (str): Path to the test data
        e2eoutput_path (str): Path to the output directory

    """

    # grab input L1 data 
    l1_input_data_dir = os.path.join(e2edata_path, "astrom_sims")
    l1_input_data_list = sorted(glob.glob(os.path.join(l1_input_data_dir, "*_l1_*.fits")))


    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)

    # grab default pipeline calibrations
    db = caldb.CalDB()
    db.scan_dir_for_new_entries(corgidrp.default_cal_dir) 

    # create output directory
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_astrom_e2e")
    if os.path.exists(test_outputdir):
        shutil.rmtree(test_outputdir)
    os.makedirs(test_outputdir)
    l2b_outputdir = os.path.join(test_outputdir, "l2b_results")
    os.makedirs(l2b_outputdir)

    # run pipeline
    with warnings.catch_warnings():
        # suppress warnings about the three input field having different EM gain configurations
        warnings.simplefilter("ignore", category=RuntimeWarning)
        walker.walk_corgidrp(l1_input_data_list, "", l2b_outputdir)

    # expected values from simulation input
    expected_platescale = 21.8 # mas/pixel
    expected_north_angle = -45
    # compute the expected ra and dec offset due to detector placement at (532, 505) instead of (512, 512)
    dx_pix, dy_pix = 532 - 512, 505 - 512 
    # get expected offsets in ra and dec, units of mas
    expected_ra_offset  = dx_pix * expected_platescale
    expected_dec_offset = dy_pix* expected_platescale

    # check that the recovered platescale, north angle, and offsets match up
    astrom_cal_file = glob.glob(os.path.join(l2b_outputdir, '*_ast_cal.fits'))[0]
    astrom_cal = data.AstrometricCalibration(astrom_cal_file)
    actual_platescale = astrom_cal.platescale
    actual_north_angle = astrom_cal.northangle
    actual_ra_offset  = astrom_cal.avg_offset[0] * 3.6e6 # convert from deg to mas
    actual_dec_offset = astrom_cal.avg_offset[1] * 3.6e6
    assert expected_platescale == pytest.approx(astrom_cal.platescale, rel=0.05)
    assert expected_north_angle == pytest.approx(actual_north_angle, abs=0.05)
    assert expected_ra_offset == pytest.approx(actual_ra_offset, abs=35)
    assert expected_dec_offset == pytest.approx(actual_dec_offset, abs=35)

    # check headers
    check.compare_to_mocks_hdrs(astrom_cal_file)
    assert astrom_cal.ext_hdr["DATATYPE"] == "AstrometricCalibration"
    assert astrom_cal.ext_hdr["DATALVL"] == "CAL"

if __name__ == "__main__":
    outputdir = thisfile_dir
    e2edata_path = '/home/eshen12345/dev/E2E_Test_Data'

    ap = argparse.ArgumentParser(description='run the l1 to astrometric calibration end-to-end test')
    ap.add_argument('-e2e', '--e2edata_dir', default=e2edata_path,
                    help='Path to test Data Folder [%(default)s]')
    ap.add_argument('-o', '--outputdir', default=outputdir,
                    help='directory to write results to [%(default)s]')
    args = ap.parse_args()
    outputdir = args.outputdir
    test_l1_to_astrom_e2e(args.e2edata_dir, args.outputdir)
