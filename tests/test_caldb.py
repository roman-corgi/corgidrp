import os
import numpy as np
import pytest
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

if __name__ == "__main__":
    test_caldb_init()
    test_get_calib()
    test_caldb_create_default()
    test_caldb_custom_filepath()
    test_caldb_insert_and_remove()
    test_caldb_scan()