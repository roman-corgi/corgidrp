import argparse
import os, shutil
import glob
import pytest
import warnings
import numpy as np
from astropy.io import fits
import astropy
from astropy.coordinates import SkyCoord

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
        walker.walk_corgidrp(l1_input_data_list, "", l2b_outputdir, template='l1_to_boresight_offset.json')

    # load in an l2b to get the target RA, Dec values from the header
    l2b_filenames = glob.glob(l2b_outputdir+'/*_l2b.fits')
    l2b_dataset = data.Dataset(l2b_filenames)
    expected_pointing = l2b_dataset[0].pri_hdr['RA'], l2b_dataset[0].pri_hdr['DEC']

    # expected values from simulation input
    expected_platescale = 21.8 # mas/pixel
    expected_north_angle = -45
    # compute the expected ra and dec offset due to detector placement at (532, 505) instead of (512, 512) using an astropy wcs from the true platescale and northangle
    vert_ang = np.radians(expected_north_angle)
    pc = np.array([[-np.cos(vert_ang), np.sin(vert_ang)], [np.sin(vert_ang), np.cos(vert_ang)]])
    cdmatrix = pc * (expected_platescale * 0.001) / 3600.
    new_hdr = {}
    new_hdr['CD1_1'] = cdmatrix[0,0]
    new_hdr['CD1_2'] = cdmatrix[0,1]
    new_hdr['CD2_1'] = cdmatrix[1,0]
    new_hdr['CD2_2'] = cdmatrix[1,1]
    new_hdr['CRPIX1'] = 533.    # true pixel value at the target pointing
    new_hdr['CRPIX2'] = 506.
    new_hdr['CTYPE1'] = 'RA---TAN'
    new_hdr['CTYPE2'] = 'DEC--TAN'
    new_hdr['CDELT1'] = (expected_platescale * 0.001) / 3600.
    new_hdr['CDELT2'] = (expected_platescale * 0.001) / 3600.
    new_hdr['CRVAL1'] = expected_pointing[0]    # true target pointing
    new_hdr['CRVAL2'] = expected_pointing[1]
    w = astropy.wcs.WCS(new_hdr)

    # use astropy wcs to find the true coordinate value of detector center (512., 512.)
    expected_center_skycoord = astropy.wcs.utils.pixel_to_skycoord(512., 512., wcs=w, origin=1)

    # check that the recovered platescale, north angle, and offsets match up
    astrom_cal_file = glob.glob(os.path.join(l2b_outputdir, '*_ast_cal.fits'))[0]
    astrom_cal = data.AstrometricCalibration(astrom_cal_file)
    actual_platescale = astrom_cal.platescale
    actual_north_angle = astrom_cal.northangle
    assert expected_platescale == pytest.approx(actual_platescale, rel=0.05)
    assert expected_north_angle == pytest.approx(actual_north_angle, abs=0.05)
    # measure how well we recover the center coordinate
    actual_center_skycoord = SkyCoord(ra=astrom_cal.boresight[0], dec=astrom_cal.boresight[1], unit='deg')
    error_ra, error_dec = actual_center_skycoord.spherical_offsets_to(expected_center_skycoord)
    assert error_ra.mas == pytest.approx(0, abs=10)    # make sure we are in [mas]
    assert error_dec.mas == pytest.approx(0, abs=10)

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
