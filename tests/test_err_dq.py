import os
import pytest
import numpy as np
import astropy.io.fits as fits
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.detector as detector
from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset, DetectorParams
import corgidrp.caldb as caldb
from corgidrp.darks import build_trad_dark

np.random.seed(123)

data = np.ones([1024,1024]) * 2
err = np.zeros([1024,1024])
err1 = np.ones([1024,1024])
err2 = err1.copy()
err3 = np.ones([1,1024,1024]) * 0.5
dq = np.zeros([1024,1024], dtype = int)
dq1 = dq.copy()
dq1[0,0] = 1
prhd, exthd = create_default_headers()
errhd = fits.Header()
errhd["CASE"] = "test"
dqhd = fits.Header()
dqhd["CASE"] = "test"

old_err_tracking = corgidrp.track_individual_errors
# use default parameters
detector_params = DetectorParams({})

def test_err_dq_creation():
    """
    test the initialization of error and dq attributes of the Image class including saving and loading

    Test assuming track individual error terms is on
    """
    corgidrp.track_individual_errors = True

    image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd)
    assert hasattr(image1, "err")
    assert hasattr(image1, "dq")
    assert image1.data.shape == data.shape
    assert image1.data.shape == image1.err.shape[-2:]
    assert image1.err.shape == (1, 1024, 1024)
    assert image1.data.shape == image1.dq.shape
    #test the initial error and dq headers
    assert hasattr(image1, "err_hdr")
    assert hasattr(image1, "dq_hdr")
    assert np.mean(image1.dq) == 0
    image1.save(filename='test_image1.fits')

    image2 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq1, err_hdr = errhd, dq_hdr = dqhd)
    print("data", image2.data)
    print("error", image2.err)
    print("dq", image2.dq)
    print("err_hdr", image2.err_hdr)
    print("dq_hdr", image2.dq_hdr)
    # test the user defined error and dq headers
    assert image2.err_hdr["CASE"] == errhd["CASE"]
    assert image2.dq_hdr["CASE"] == dqhd["CASE"]
    #check the correct saving and loading of fits files
    image2.save(filename='test_image2.fits')
    image3 = Image('test_image2.fits')
    assert np.array_equal(image3.data, image2.data)
    assert np.array_equal(image3.err, image2.err)
    assert np.array_equal(image3.dq, image2.dq)
    assert image3.err_hdr["CASE"] == errhd["CASE"]
    assert image3.dq_hdr["CASE"] == dqhd["CASE"]

    #check the overwriting of err and dq parameters of the fits extensions arrays
    image_test = Image('test_image2.fits', err = err, dq = dq)
    assert np.array_equal(image_test.dq, dq)
    assert np.array_equal(image_test.err[0,:,:], err)
    #test 3d input error array
    image_test2 = Image('test_image2.fits', err = err3)
    assert np.array_equal(image_test2.err, err3)

def test_err_dq_copy():
    """
    test the copying of the err and dq attributes

    Runs assuming tracking individual errors
    """
    corgidrp.track_individual_errors = True

    image2 = Image('test_image2.fits')
    image3 = image2.copy()
    assert np.array_equal(image3.data, image2.data)
    assert np.array_equal(image3.err, image2.err)
    assert np.array_equal(image3.dq, image2.dq)
    #check the copying of the headers of the extensions
    assert image2.err_hdr == image3.err_hdr
    assert image2.dq_hdr == image3.dq_hdr

def test_add_error_term():
    """
    test the add_error_term function

    Runs assuming tracking individual errors
    """
    corgidrp.track_individual_errors = True

    image1 = Image('test_image1.fits')
    image1.add_error_term(err1, "error_noid")
    assert image1.err[0,0,0] == err1[0,0]
    image1.add_error_term(err2, "error_nuts")
    assert image1.err.shape == (3,1024,1024)
    assert image1.err[0,0,0] == np.sqrt(err1[0,0]**2 + err2[0,0]**2)
    image1.save(filename="test_image0.fits")

    image_test = Image('test_image0.fits')
    assert np.array_equal(image_test.dq, dq)
    assert np.array_equal(image_test.err, image1.err)
    assert image_test.err.shape == (3,1024,1024)
    assert image_test.err_hdr["Layer_1"] == "combined_error"
    assert image_test.err_hdr["Layer_2"] == "error_noid"
    assert image_test.err_hdr["Layer_3"] == "error_nuts"

def test_err_dq_dataset():
    """
    test the behavior of the err and data arrays in the dataset

    Runs assuming tracking individual errors
    """
    corgidrp.track_individual_errors = True

    dataset = Dataset(["test_image1.fits", "test_image2.fits"])
    assert np.array_equal(dataset[0].data, dataset[1].data)
    assert np.array_equal(dataset[0].err, dataset[1].err)
    assert not np.array_equal(dataset[0].dq, dataset[1].dq)
    assert dataset.all_data.shape[0] == len(dataset)
    assert dataset.all_err.ndim == 4
    assert dataset.all_dq.ndim == 3
    assert dataset.all_err.shape[2] == err.shape[0]
    assert dataset.all_dq.shape[2] == dq.shape[1]

def test_get_masked_data():
    """
    test the masked array
    """
    image2 = Image('test_image2.fits')
    masked_data = image2.get_masked_data()
    print("masked data", masked_data.data)
    print("mask", masked_data.mask)
    assert masked_data.data[0,1] == 2
    #check that pixel 0,0 is masked and not considered
    assert masked_data.mask[0,0] == True
    assert masked_data.mean()==2
    assert masked_data.sum()==image2.data.sum()-2


def test_err_adderr_notrack():
    """
    test the initialization of error and adding errors when we are not tracking
    individual errors. There should always only be a single 2-D map.
    """
    corgidrp.track_individual_errors = False

    image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd)
    assert hasattr(image1, "err")
    assert image1.data.shape == data.shape
    assert image1.data.shape == image1.err.shape[-2:]
    assert image1.err.shape == (1, 1024, 1024)
    #test the initial error and dq headers
    assert hasattr(image1, "err_hdr")

    image1.add_error_term(err1, "error_noid")
    assert image1.err[0,0,0] == err1[0,0]
    image1.add_error_term(err2, "error_nuts")
    assert image1.err.shape == (1,1024,1024)
    assert image1.err[0,0,0] == np.sqrt(err1[0,0]**2 + err2[0,0]**2)


def test_read_many_errors_notrack():
    """
    Check that we can successfully discard errors when reading in a frame with multiple errors

    """
    corgidrp.track_individual_errors = False

    image_test = Image('test_image0.fits')
    assert image_test.err.shape == (1,1024,1024)
    assert image_test.err_hdr["Layer_1"] == "combined_error"
    with pytest.raises(KeyError):
        assert image_test.err_hdr["Layer_2"] == "error_noid"
    with pytest.raises(KeyError):
        assert image_test.err_hdr["Layer_3"] == "error_nuts"


def test_err_array_sizes():
    '''
    Check that we're robust to 2D error arrays

    Creates a dark calibration and then forces the error array to be 2D

    Makes sure that we're robust to that.
    '''

    ##### Create a master Dark #####
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    dark_dataset = mocks.create_dark_calib_files(filedir=datadir)
    dark_frame = build_trad_dark(dark_dataset, detector_params, detector_regions=None, full_frame=True)

    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    dark_filename = "sim_dark_calib.fits"
    if not os.path.exists(calibdir):
            os.mkdir(calibdir)
    dark_frame.save(filedir=calibdir, filename=dark_filename)


    ##### Scan the caldb ##### - This tests for previous bug that darks weren't in the right format.
    testcaldb_filepath = os.path.join(calibdir, "test_caldb.csv")
    testcaldb = caldb.CalDB(filepath=testcaldb_filepath)
    testcaldb.scan_dir_for_new_entries(calibdir)

    ##### Force it to be 2D ##### - This tests to maks sure we're robust to 2D error arrays in general
    dark_frame.err = np.ones(dark_frame.data.shape)
    dark_frame.save(filedir=calibdir, filename=dark_filename)
    testcaldb.scan_dir_for_new_entries(calibdir)



def teardown_module():
    """
    Runs automatically at the end. ONLY IN PYTEST

    Removes new FITS files and restores track individual error setting.
    """
    for i in range(3):
        os.remove('test_image{0}.fits'.format(i))

    corgidrp.track_individual_errors = old_err_tracking


# for debugging. does not run with pytest!!
if __name__ == '__main__':
    test_err_array_sizes()
    test_err_dq_creation()
    test_err_dq_copy()
    test_add_error_term()
    test_err_dq_dataset()
    test_get_masked_data()

    for i in range(3):
        os.remove('test_image{0}.fits'.format(i))