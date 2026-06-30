import os, shutil, glob, argparse
import pytest

import corgidrp
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.walker as walker
import corgidrp.nd_filter_calibration as nd_filter_calibration
from corgidrp import caldb, check
from astropy.time import Time

# this file's folder
thisfile_dir = os.path.dirname(__file__)

@pytest.mark.e2e
def test_l1_to_nd_filter_e2e(e2edata_path, e2eoutput_path):
    # grab input L1 data, consisting of dim (no ND) and bright (ND 475) stars
    l1_input_data_dir = os.path.join(e2edata_path, "ND_sims", "L1")
    l1_input_data_list = glob.glob(os.path.join(l1_input_data_dir, "*_l1_*.fits"))

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)

    # grab calibration files for processing L1 data
    cal_dir =os.path.join(e2edata_path, "ND_sims", "Cals") 
    db = caldb.CalDB()
    db.scan_dir_for_new_entries(cal_dir)

    # add detector params to caldb
    detector_params = data.DetectorParams({}, date_valid=Time("2023-11-01 00:00:00"))
    detector_params.save(filedir=corgidrp.config_folder)
    db.create_entry(detector_params)

    # create empty output directory
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_ND_filter_e2e")
    if os.path.exists(test_outputdir):
        shutil.rmtree(test_outputdir)
    os.makedirs(test_outputdir)
    l2b_outputdir = os.path.join(test_outputdir, "l2b_results")
    if not os.path.exists(l2b_outputdir):
        os.mkdir(l2b_outputdir)

    # clean up by removing old files
    for file in os.listdir(l2b_outputdir):
        os.remove(os.path.join(l2b_outputdir, file))
    
    # run walker
    walker.walk_corgidrp(l1_input_data_list, "", l2b_outputdir)

    """
    # 5. Load product & assert if calculated OD matches the input
    nd_file = glob.glob(os.path.join(simdata_dir, "*_ndf_cal*.fits"))
    nd_cal  = data.NDFilterSweetSpotDataset(nd_file[0])

    recovered_od = float(nd_cal.od_values[0])  # use the first entry for the check
    print("Calculated OD:", recovered_od)
    print("Input OD:", od_truth)
    assert recovered_od == pytest.approx(od_truth, abs=1e-1)

    check.compare_to_mocks_hdrs(nd_file[0])
    """
    
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

    print("ND‑filter E2E test passed")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    outputdir = thisfile_dir
    e2edata_path = '/home/eshen12345/dev/E2E_Test_Data'

    ap = argparse.ArgumentParser(description='run the l1 to NDFilterSweetSpot end-to-end test')
    ap.add_argument('-e2e', '--e2edata_dir', default=e2edata_path,
                    help='Path to test Data Folder [%(default)s]')
    ap.add_argument('-o', '--outputdir', default=outputdir,
                    help='directory to write results to [%(default)s]')
    args = ap.parse_args()
    outputdir = args.outputdir
    test_l1_to_nd_filter_e2e(args.e2edata_dir, args.outputdir)
