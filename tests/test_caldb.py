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



def test_create_entry_file_not_on_disk():
    """
    Tests that create_entry raises FileNotFoundError when the file does not exist on disk.
    """
    if os.path.exists(testcaldb_filepath):
        os.remove(testcaldb_filepath)
    testcaldb = caldb.CalDB(filepath=testcaldb_filepath)

    # Point the dark at a path that doesn't exist
    fake_dark = data.Dark(dark_dataset[0].data, dark_dataset[0].pri_hdr, dark_dataset[0].ext_hdr, dark_dataset)
    fake_dark.filedir = calibdir
    fake_dark.filename = "nonexistent_dark.fits"

    with pytest.raises(FileNotFoundError):
        testcaldb.create_entry(fake_dark)

    os.remove(testcaldb_filepath)


def test_get_calib_missing_file():
    """
    Tests that get_calib falls back to the next best calibration when the
    best-matching file no longer exists on disk, rather than crashing.
    """
    if os.path.exists(testcaldb_filepath):
        os.remove(testcaldb_filepath)
    testcaldb = caldb.CalDB(filepath=testcaldb_filepath)

    # Create two darks using the same headers so EXPTIME/EMGAIN_C match the reference frame
    dark1 = data.Dark(dark_dataset[0].data, dark_dataset[0].pri_hdr, dark_dataset[0].ext_hdr, dark_dataset)
    dark1.save(filedir=calibdir, filename="mockdark_miss1.fits")
    dark2 = data.Dark(dark_dataset[1].data, dark_dataset[1].pri_hdr, dark_dataset[0].ext_hdr, dark_dataset)
    dark2.save(filedir=calibdir, filename="mockdark_miss2.fits")

    testcaldb.create_entry(dark1)
    testcaldb.create_entry(dark2)

    # Manually set MJDs so dark1 is closer to the reference frame than dark2
    ref_frame = dark_dataset[2]
    ref_mjd = float(ref_frame.ext_hdr['MJDSRT'])
    dark1_abs = os.path.abspath(dark1.filepath)
    dark2_abs = os.path.abspath(dark2.filepath)
    testcaldb._db.loc[testcaldb._db['Filepath'] == dark1_abs, 'MJD'] = ref_mjd + 1.0
    testcaldb._db.loc[testcaldb._db['Filepath'] == dark2_abs, 'MJD'] = ref_mjd + 100.0
    testcaldb.save()

    # With both files present, dark1 is selected (closest in time)
    result = testcaldb.get_calib(ref_frame, data.Dark)
    assert result.filepath == dark1_abs

    # Remove dark1 from disk to simulate user moving/deleting the file
    os.remove(dark1_abs)

    # get_calib should fall back to dark2 without crashing
    result = testcaldb.get_calib(ref_frame, data.Dark)
    assert result.filepath == dark2_abs

    # Remove dark2 too; now all matching calibrations are gone
    os.remove(dark2_abs)
    with pytest.raises(ValueError):
        testcaldb.get_calib(ref_frame, data.Dark)

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
    # Ensure the test caldb starts fresh (a failing earlier test may leave it behind)
    if os.path.exists(testcaldb_filepath):
        os.remove(testcaldb_filepath)

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

def test_dispersion_model_dpam_match():
    """
    get_calib for DispersionModel should select the model whose DPAMNAME matches the frame's
    (PRISM2 vs PRISM3), not merely the closest in time. Both defaults ship in default_cal_dir.
    """
    cdb = caldb.CalDB()

    def make_ref(dpam):
        prihdr, exthdr = mocks.create_default_L2b_headers()[:2]
        exthdr['DPAMNAME'] = dpam
        exthdr['CFAMNAME'] = '3F'
        exthdr['FSAMNAME'] = 'OPEN'
        exthdr['MJDSRT'] = 60000.0
        return data.Image(np.zeros((50, 50)), pri_hdr=prihdr, ext_hdr=exthdr)

    dm2 = cdb.get_calib(make_ref('PRISM2'), data.DispersionModel)
    assert dm2.ext_hdr['DPAMNAME'] == 'PRISM2'
    assert float(dm2.ext_hdr['REFWAVE']) == 660.

    dm3 = cdb.get_calib(make_ref('PRISM3'), data.DispersionModel)
    assert dm3.ext_hdr['DPAMNAME'] == 'PRISM3'
    assert float(dm3.ext_hdr['REFWAVE']) == 730.


def test_normalize_spec_cfam():
    """
    CalDB._normalize_spec_cfam should map spectroscopy CFAMNAME sub-bands to their
    parent broadband value, and leave already-broadband or unrelated values unchanged.
    """
    cdb = caldb.CalDB()
    assert cdb._normalize_spec_cfam('2A') == '2F'
    assert cdb._normalize_spec_cfam('2C') == '2F'
    assert cdb._normalize_spec_cfam('3D') == '3F'
    assert cdb._normalize_spec_cfam('2F') == '2F'
    assert cdb._normalize_spec_cfam('3F') == '3F'
    assert cdb._normalize_spec_cfam('CLEAR') == 'CLEAR'


def test_select_reference_frame_mixed_pam():
    """
    Per issue #820, get_calib() needs to correctly resolve a representative frame from
    a Dataset that intentionally mixes frames with different PAM configurations.
    CalDB._select_reference_frame should pick an ND-filter-in frame
    (FPAMNAME starting with 'ND') for ND-based calibration types, even if it isn't the
    first frame in the dataset, and should just use the first frame for every other
    calibration type. A single frame (not a Dataset) should always be returned
    unchanged, regardless of calibration type.
    """
    cdb = caldb.CalDB()

    def make_frame(fpam, cfam='2F', dpam='PRISM2'):
        prihdr, exthdr = mocks.create_default_L2b_headers()[:2]
        exthdr['FPAMNAME'] = fpam
        exthdr['CFAMNAME'] = cfam
        exthdr['DPAMNAME'] = dpam
        return data.Image(np.zeros((5, 5)), pri_hdr=prihdr, ext_hdr=exthdr)

    non_nd_frame = make_frame('OPEN_12')
    nd_frame = make_frame('ND225')
    mixed_dataset = data.Dataset([non_nd_frame, nd_frame])

    # ND-based calibration types should pick the ND-filter-in frame, even though
    # it isn't first in the dataset
    for dtype_label in caldb.CalDB._ND_FILTER_CAL_TYPES:
        picked = cdb._select_reference_frame(mixed_dataset, dtype_label)
        assert picked is nd_frame, "{0} should pick the ND-filter-in frame".format(dtype_label)

    # every other calibration type should just use the first frame
    assert cdb._select_reference_frame(mixed_dataset, 'Dark') is non_nd_frame

    # a single frame (not a Dataset) is always returned unchanged
    assert cdb._select_reference_frame(non_nd_frame, 'NDMuellerMatrix') is non_nd_frame
    assert cdb._select_reference_frame(None, 'Dark') is None

    # if no ND-filter-in frame exists in the dataset, falls back to the first frame
    all_non_nd_dataset = data.Dataset([make_frame('OPEN_12'), make_frame('OPEN_34')])
    picked = cdb._select_reference_frame(all_non_nd_dataset, 'NDMuellerMatrix')
    assert picked is all_non_nd_dataset[0]


def test_get_calib_ndfiltersweetspot_mixed_dataset():
    """
    get_calib() for NDFilterSweetSpot should correctly select a calibration entry when
    given a full Dataset that mixes ND-filter-out and ND-filter-in frames (as happens in
    real ND filter calibration processing, which uses dim-star frames with no ND filter
    alongside bright-star frames observed through the ND filter). Using the first frame
    in the dataset (which may not be the ND-filter-in one) would incorrectly filter by
    the wrong FPAMNAME, or fail entirely.

    Also exercises the CFAM sub-band fallback: the science frame uses a narrowband
    CFAMNAME ('2A') while the calibration entry is tagged with the parent broadband
    ('2F'); NDFilterSweetSpot's fallback-to-broadband logic should still match it.
    """
    cdb = caldb.CalDB()

    # register an NDFilterSweetSpot calibration entry
    prihdr, exthdr = mocks.create_default_L2b_headers()[:2]
    exthdr['FPAMNAME'] = 'ND225'
    exthdr['DPAMNAME'] = 'PRISM2'
    exthdr['CFAMNAME'] = '2F'
    exthdr['MJDSRT'] = 60000.0
    nd_cal = data.NDFilterSweetSpotDataset(
        np.array([[2.0, 10.0, 10.0]]), pri_hdr=prihdr, ext_hdr=exthdr
    )
    nd_cal_filepath = os.path.join(calibdir, "test_ndfiltersweetspot_cal.fits")
    nd_cal.save(filedir=calibdir, filename="test_ndfiltersweetspot_cal.fits")
    cdb.create_entry(nd_cal)

    try:
        # build a mixed dataset: a dim-star (ND-filter-out) frame first, then the
        # bright-star (ND-filter-in) frame that should actually drive the lookup
        dim_prihdr, dim_exthdr = mocks.create_default_L2b_headers()[:2]
        dim_exthdr['FPAMNAME'] = 'OPEN_12'
        dim_exthdr['DPAMNAME'] = 'PRISM2'
        dim_exthdr['CFAMNAME'] = '2F'
        dim_frame = data.Image(np.zeros((5, 5)), pri_hdr=dim_prihdr, ext_hdr=dim_exthdr)

        bright_prihdr, bright_exthdr = mocks.create_default_L2b_headers()[:2]
        bright_exthdr['FPAMNAME'] = 'ND225'
        bright_exthdr['DPAMNAME'] = 'PRISM2'
        bright_exthdr['CFAMNAME'] = '2A'  # narrowband sub-band of the registered 2F entry
        bright_frame = data.Image(np.zeros((5, 5)), pri_hdr=bright_prihdr, ext_hdr=bright_exthdr)

        mixed_dataset = data.Dataset([dim_frame, bright_frame])

        result = cdb.get_calib(mixed_dataset, data.NDFilterSweetSpotDataset)
        assert result.ext_hdr['FPAMNAME'] == 'ND225'
        assert result.ext_hdr['CFAMNAME'] == '2F'
    finally:
        cdb.remove_entry(nd_cal)
        os.remove(nd_cal_filepath)


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
    test_dispersion_model_dpam_match()
    test_normalize_spec_cfam()
    test_select_reference_frame_mixed_pam()
    test_get_calib_ndfiltersweetspot_mixed_dataset()
    test_caldb_filter()