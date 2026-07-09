# E2E test for sim L1 pol data to mueller matrix calibration
import os, shutil, glob, argparse
import pytest
import warnings
import numpy as np
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.walker as walker
from corgidrp import caldb, check

# this file's folder
thisfile_dir = os.path.dirname(__file__)

@pytest.mark.e2e
def test_l1_to_polcal_e2e(e2edata_path, e2eoutput_path):
    # grab input L1 data, consisting of unocculted unpolarized and polarized stars using ND225 
    l1_input_data_dir = os.path.join(e2edata_path, "mueller_matrix_sims", "L1")
    l1_input_data_list = glob.glob(os.path.join(l1_input_data_dir, "*_l1_*.fits"))

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)

    # grab polarimetric calibration files
    cal_dir =os.path.join(e2edata_path, "mueller_matrix_sims", "Cals") 
    db = caldb.CalDB()
    db.scan_dir_for_new_entries(cal_dir) # polarimetric calibration files
    # db.scan_dir_for_new_entries(corgidrp.default_cal_dir) # grab other default files needed

    # create empty output directory
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_polcal_e2e")
    if os.path.exists(test_outputdir):
        shutil.rmtree(test_outputdir)
    os.makedirs(test_outputdir)
    l2b_outputdir = os.path.join(test_outputdir, "l2b_results")
    os.makedirs(l2b_outputdir)

    with warnings.catch_warnings():
        # suppress warning about number of detectornoisemap frames
        warnings.simplefilter("ignore", category=UserWarning)
        # suppress warnings about frames having different header values
        # warnings.simplefilter("ignore", category=RuntimeWarning)
        walker.walk_corgidrp(l1_input_data_list, "", l2b_outputdir)
    

    # grab calibration file
    mm_file = glob.glob(os.path.join(l2b_outputdir, '*ndm_cal*.fits'))[0]
    mueller_matrix = data.NDMuellerMatrix(mm_file)
    mm = mueller_matrix.data

    # check the mueller matrix elements against the normalized mueller matrix used in corgisim to create the input files
    # also skip circular polarization elements (row/column 4) since that cannot be detected and calibrated for
    assert mm[0,0] == 1 # I->I should be normalized to 1
    # for main diagonal elements, check it is within 5% accuracy
    rtol = 0.05
    assert mm[1,1] == pytest.approx(-0.99995, rel=rtol)
    assert mm[2,2] == pytest.approx(0.99455, rel=rtol)

    # for off-diagonal elements which are basically 0 and noisier, check with absolute tolerance
    atol = 0.1
    assert mm[0,1] == pytest.approx(0.00926, abs=atol)
    assert mm[0,2] == pytest.approx(0.00000, abs=atol)
    assert mm[1,0] == pytest.approx(-0.00926, abs=atol)
    assert mm[1,2] == pytest.approx(0.00001, abs=atol)
    assert mm[2,0] == pytest.approx(0.00000, abs=atol)
    assert mm[2,1] == pytest.approx(0.00000, abs=atol)

    # check headers
    check.compare_to_mocks_hdrs(mm_file)
    assert mueller_matrix.ext_hdr["FPAMNAME"] == "ND225" # input data uses ND225 filter
    assert mueller_matrix.ext_hdr["DATALVL"] == "CAL"
    assert mueller_matrix.ext_hdr["DATATYPE"] == "NDMuellerMatrix"

    os.remove(tmp_caldb_csv)

    print("L1 to polcal E2E test passed")

if __name__ == "__main__":
    outputdir = thisfile_dir
    e2edata_path = '/home/eshen12345/dev/E2E_Test_Data'

    ap = argparse.ArgumentParser(description='run the l1 to mueller matrix calibration end-to-end test')
    ap.add_argument('-e2e', '--e2edata_dir', default=e2edata_path,
                    help='Path to test Data Folder [%(default)s]')
    ap.add_argument('-o', '--outputdir', default=outputdir,
                    help='directory to write results to [%(default)s]')
    args = ap.parse_args()
    outputdir = args.outputdir
    test_l1_to_polcal_e2e(args.e2edata_dir, args.outputdir)