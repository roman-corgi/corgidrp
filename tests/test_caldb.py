import os
import glob
import numpy as np
import pytest
import shutil
from astropy.io import fits
import corgidrp
import corgidrp.caldb as caldb
import corgidrp.data as data
import corgidrp.mocks as mocks

datadir = os.path.join(os.path.dirname(__file__), "simdata")
calibdir = os.path.join(os.path.dirname(__file__), "testcalib")

if not os.path.exists(datadir):
    os.mkdir(datadir)
if not os.path.exists(calibdir):
    os.mkdir(calibdir)

testcaldb_filepath = os.path.join(calibdir, "test_caldb.csv")

# make some fake test data to use
np.random.seed(456)
dark_dataset = mocks.create_dark_calib_files()
master_dark = data.Dark(dark_dataset[0].data, dark_dataset[0].pri_hdr, dark_dataset[0].ext_hdr, dark_dataset)
# save master dark to disk to be loaded later
master_dark.save(filedir=calibdir, filename="mockdark.fits")

def test_caldb_init():
    """
    Tests that caldb has been initialized. It has to be if it's being imported.
    """
    assert caldb.initialized


def test_caldb_create_default():
    """
    Test caldb creation when no filepath is passed in (uses default path)
    """
    # remove any stranded testcaldb if needed
    if os.path.exists(testcaldb_filepath):
        os.remove(testcaldb_filepath)
    assert(not os.path.exists(testcaldb_filepath))

    # modify default path so we don't mess up the real thing
    old_path = corgidrp.caldb_filepath
    corgidrp.caldb_filepath = testcaldb_filepath

    # create the caldb and check it's saved to disk and empty
    testcaldb = caldb.CalDB()
    assert(testcaldb.filepath == testcaldb_filepath)
    assert(os.path.exists(testcaldb_filepath))
    assert(len(testcaldb._db.index) == 0)

    # remove db and restore path
    os.remove(testcaldb_filepath)
    corgidrp.caldb_filepath = old_path


def test_caldb_custom_filepath():
    """
    Test caldb creation when filepath is passed in (should be an edge case)
    """
    # remove any stranded testcaldb if needed
    if os.path.exists(testcaldb_filepath):
        os.remove(testcaldb_filepath)
    assert(not os.path.exists(testcaldb_filepath))

    # create the caldb and check it's saved to disk and empty
    testcaldb = caldb.CalDB(filepath=testcaldb_filepath)
    assert(testcaldb.filepath == testcaldb_filepath)
    assert(testcaldb.filepath != corgidrp.caldb_filepath)
    assert(os.path.exists(testcaldb_filepath))
    assert(len(testcaldb._db.index) == 0)

    # remove db and restore path
    os.remove(testcaldb_filepath)

def test_caldb_insert_and_remove():
    """
    Tests the ability to add and remove an entry successfully
    """
    # remove any stranded testcaldb if needed
    if os.path.exists(testcaldb_filepath):
        os.remove(testcaldb_filepath)
    assert(not os.path.exists(testcaldb_filepath))

    # create custom caldb for testing
    testcaldb = caldb.CalDB(filepath=testcaldb_filepath)
    assert(len(testcaldb._db.index) == 0)

    # add dark file into database and check that the values are stored correctly
    testcaldb.create_entry(master_dark)
    assert(len(testcaldb._db.index) == 1)
    assert(testcaldb._db['NAXIS1'][0] == master_dark.data.shape[1])
    orig_exptime = master_dark.ext_hdr['EXPTIME']
    assert(testcaldb._db['EXPTIME'][0] == orig_exptime)

    # test update
    master_dark.ext_hdr['EXPTIME'] = 2*orig_exptime
    testcaldb.create_entry(master_dark)
    assert(len(testcaldb._db.index) == 1)
    assert(testcaldb._db['NAXIS1'][0] == master_dark.data.shape[1])
    assert(testcaldb._db['EXPTIME'][0] == 2*orig_exptime)

    # test remove
    testcaldb.remove_entry(master_dark)
    assert(len(testcaldb._db.index) == 0)

    # reset everything
    master_dark.ext_hdr['EXPTIME'] = orig_exptime
    os.remove(testcaldb_filepath)

def test_get_calib():
    """
    Tests ability to load a calibration file from disk
    """
    # remove any stranded testcaldb if needed
    if os.path.exists(testcaldb_filepath):
        os.remove(testcaldb_filepath)
    assert(not os.path.exists(testcaldb_filepath))

    # create custom caldb for testing
    testcaldb = caldb.CalDB(filepath=testcaldb_filepath)
    assert(len(testcaldb._db.index) == 0)

    # add dark file into database and check that the values are stored correctly
    testcaldb.create_entry(master_dark)
    assert(len(testcaldb._db.index) == 1)

    # grab the only dark in the caldb, so it should be the one we put in
    auto_dark = testcaldb.get_calib(dark_dataset[2], data.Dark)
    assert(auto_dark.filepath == master_dark.filepath)

    with pytest.raises(ValueError):
        _ = testcaldb.get_calib(dark_dataset[2], data.DetectorNoiseMaps)

    # make a second one dark
    master_dark_2 = data.Dark(dark_dataset[1].data, dark_dataset[1].pri_hdr, dark_dataset[0].ext_hdr, dark_dataset)
    # save master dark to disk to be loaded later
    master_dark_2.save(filedir=calibdir, filename="mockdark2.fits")
    testcaldb.create_entry(master_dark_2)

    # test that with no input data, we get the most recent dark
    auto_dark_2 = testcaldb.get_calib(None, data.Dark)
    assert(auto_dark_2.filepath == master_dark_2.filepath)
        
    # reset everything
    os.remove(testcaldb_filepath)



def test_caldb_scan():
    """
    Tests ability to scan a folder to look for calibration files
    """
    # remove any stranded testcaldb if needed
    if os.path.exists(testcaldb_filepath):
        os.remove(testcaldb_filepath)
    assert(not os.path.exists(testcaldb_filepath))

    # create custom caldb for testing
    testcaldb = caldb.CalDB(filepath=testcaldb_filepath)
    assert(len(testcaldb._db.index) == 0)

    # there should be no calibration files in "./simdata", just unprocessed data
    testcaldb.scan_dir_for_new_entries(datadir)
    assert(len(testcaldb._db.index) == 0)

    # there should be at least the master dark in "./testcalib"
    testcaldb.scan_dir_for_new_entries(calibdir)
    assert(len(testcaldb._db.index) > 0)

    # reset everything
    os.remove(testcaldb_filepath)

def test_default_calibs():
    """
    Tests that the default calibration files are created if they don't exist.
    """
    # Copy all files in corgidrp.default_cal_dir to a temporary directory, 
    # then clear out corgidrp.default_cal_dir for this test and restore it at the end
    current_dir = os.path.dirname(__file__)
    temp_dir = os.path.join(current_dir, "temp_test_dir")
    shutil.copy2(corgidrp.caldb_filepath, os.path.join(corgidrp.config_folder, "temp_caldb.csv"))
    os.makedirs(temp_dir, exist_ok=True)
    for filename in os.listdir(corgidrp.default_cal_dir):
        src = os.path.join(corgidrp.default_cal_dir, filename)
        dst = os.path.join(temp_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    # Remove all files in corgidrp.default_cal_dir
    for filename in os.listdir(corgidrp.default_cal_dir):
        file_path = os.path.join(corgidrp.default_cal_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    default_cal_files_before = glob.glob(os.path.join(corgidrp.default_cal_dir, "*.fits"))
    assert(len(default_cal_files_before) == 0)
    # initialize (same thing happens at import, but we want to re-run it)
    caldb.initialize()
    testcaldb = caldb.CalDB(filepath=testcaldb_filepath)
    testcaldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    default_cal_files_after = glob.glob(os.path.join(corgidrp.default_cal_dir, "*.fits"))
    assert(len(default_cal_files_after) > 0)
    assert(len(testcaldb._db.index) == len(default_cal_files_after))
    # check that the default cals were generated
    cal_type_list = []
    for filename in default_cal_files_after:
        with fits.open(filename) as hdul:
            cal_type_list.append(hdul[1].header['DATATYPE'])
    assert(set(testcaldb._db['Type']) == set(cal_type_list))

    # reset everything
    os.remove(testcaldb_filepath)
    # Remove all files just created in corgidrp.default_cal_dir
    for filename in os.listdir(corgidrp.default_cal_dir):
        file_path = os.path.join(corgidrp.default_cal_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # Copy back the original files
    for filename in os.listdir(temp_dir):
        src = os.path.join(temp_dir, filename)
        dst = os.path.join(corgidrp.default_cal_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    shutil.rmtree(temp_dir)
    shutil.copy2(os.path.join(corgidrp.config_folder, "temp_caldb.csv"), corgidrp.caldb_filepath)
    os.remove(os.path.join(corgidrp.config_folder, "temp_caldb.csv"))

def test_caldb_filter():
    '''
    test that the filter function works correctly to select the best
    calibration file 
    '''

    # create mock calibration files
    ct_cal_nfov = mocks.create_ct_cal(3)
    ct_cal_nfov.ext_hdr['FPAMNAME'] = "HLC12_C2R1"
    ct_cal_wfov = mocks.create_ct_cal(3)
    ct_cal_wfov.ext_hdr['FPAMNAME'] = "SPC12_R1C1"
    ct_cal_nd = mocks.create_ct_cal(3)
    ct_cal_nd.ext_hdr['FPAMNAME'] = "ND475"
    ct_cal_nfov.save(filedir=calibdir, filename=('mock_ct_cal_nfov.fits'))
    ct_cal_wfov.save(filedir=calibdir, filename=('mock_ct_cal_wfov.fits'))
    ct_cal_nd.save(filedir=calibdir, filename=('mock_ct_cal_nd.fits'))

    # remove any stranded testcaldb if needed
    if os.path.exists(testcaldb_filepath):
        os.remove(testcaldb_filepath)
    assert(not os.path.exists(testcaldb_filepath))

    # create custom caldb for testing
    testcaldb = caldb.CalDB(filepath=testcaldb_filepath)
    assert(len(testcaldb._db.index) == 0)

    # add mock ct cal files with different filter configurations
    testcaldb.create_entry(ct_cal_nfov)
    assert(len(testcaldb._db.index) == 1)
    testcaldb.create_entry(ct_cal_wfov)
    assert(len(testcaldb._db.index) == 2)
    testcaldb.create_entry(ct_cal_nd)
    assert(len(testcaldb._db.index) == 3)

    # create mock image to input into caldb.get_calib()
    img_nfov, loc_nfov, val_nfov = mocks.create_ct_psfs(3, n_psfs=1)
    img_nfov[0].ext_hdr['FPAMNAME'] = 'HLC12_C2R1'
    img_wfov, loc_wfov, val_wfov = mocks.create_ct_psfs(3, n_psfs=1)
    img_wfov[0].ext_hdr['FPAMNAME'] = 'SPC12_R1C1'

    # check that the returned calibration file uses the hlc focal plane msk
    returned_cal_file = testcaldb.get_calib(img_nfov[0], data.CoreThroughputCalibration)
    assert returned_cal_file.ext_hdr['FPAMNAME'] == 'HLC12_C2R1'

    # check again with a different input to confirm caldb isn't just picking the most recent file
    returned_cal_file = testcaldb.get_calib(img_wfov[0], data.CoreThroughputCalibration)
    assert returned_cal_file.ext_hdr['FPAMNAME'] == 'SPC12_R1C1'

    # reset everything
    os.remove(testcaldb_filepath)

if __name__ == "__main__":
    test_default_calibs()
    test_caldb_init()
    test_get_calib()
    test_caldb_create_default()
    test_caldb_custom_filepath()
    test_caldb_insert_and_remove()
    test_caldb_scan()
    test_caldb_filter()