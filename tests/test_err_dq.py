import os
import pytest
import numpy as np
import astropy.io.fits as fits
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.detector as detector
from corgidrp.mocks import create_default_L2a_headers
from corgidrp.data import Image, Dataset, DetectorParams
import corgidrp.caldb as caldb
from corgidrp.darks import build_trad_dark

np.random.seed(123)

float_data = 2.
float_err = 0.5
data = np.ones([1024,1024]) * float_data
err = np.zeros([1024,1024])
err1 = np.ones([1024,1024])
err2 = err1.copy()
err3 = np.ones([1,1024,1024]) * float_err
dq = np.zeros([1024,1024], dtype = int)
dq1 = dq.copy()
dq1[0,0] = 1
prhd, exthd, errhd, dqhd, biashdr = create_default_L2a_headers()
errhd["CASE"] = "test"
dqhd["CASE"] = "test"

data_3d = np.ones([2,1024,1024]) * float_data
err_3d = np.zeros([2,1024,1024])
err1_3d = np.ones([2,1024,1024])
err2_3d = err1_3d.copy()
err3_3d = np.ones([2,1,1024,1024]) * float_err
dq_3d = np.zeros([2,1024,1024], dtype = int)
dq1_3d = dq_3d.copy()
dq1_3d[0,0,0] = 1

old_err_tracking = corgidrp.track_individual_errors
# use default parameters
detector_params = DetectorParams({})

def test_single_float_data():
    """
    test the initialization of error of the Image class including saving and loading in case of single float data and err input

    Test assuming track individual error terms is on
    """
    corgidrp.track_individual_errors = True

    image1 = Image(float_data, err = float_err, pri_hdr = prhd, ext_hdr = exthd)
    assert hasattr(image1, "err")
    assert np.shape(image1.data) == (1,)
    assert np.shape(image1.err) == (1,)
    assert image1.data[0] == float_data
    assert image1.err[0] == float_err
    #test the initial error and dq headers
    assert hasattr(image1, "err_hdr")
    #test saving and copying
    image1.save(filename='test_image3.fits')
    image2 = Image('test_image3.fits')
    assert image2.data[0] == float_data
    assert image2.err[0] == float_err
    image3 = image1.copy()
    assert image3.data[0] == float_data
    assert image3.err[0] == float_err
    
    image4 = Image('test_image3.fits', err = float_err/2.)
    assert image4.data[0] == float_data
    assert image4.err[0] == float_err/2.
    

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

    # Use err_3d.copy() so that add_error_term modifying self.err in-place
    # doesn't also modify the original err_3d array (they no longer share memory
    # because Image.__init__ upcasts float arrays to float64, creating a copy)
    image_3d = Image(data_3d,prhd,exthd,err_3d.copy(),dq_3d,errhd,dqhd)
    image_3d.add_error_term(err1_3d, "error_noid")
    # Combined error should be sqrt(original^2 + added^2)
    assert image_3d.err[0,0,0,0] == np.sqrt(err_3d[0,0,0]**2 + err1_3d[0,0,0]**2)
    image_3d.add_error_term(err2_3d, "error_nuts")
    assert image_3d.err.shape == (3,2,1024,1024)
    assert image_3d.err[0,0,0,0] == np.sqrt(err1_3d[0,0,0]**2 + err2_3d[0,0,0]**2)
    image_3d.save(filename="test_image4.fits")

    image_test_3d = Image('test_image4.fits')
    assert np.array_equal(image_test_3d.dq, dq_3d)
    assert np.array_equal(image_test_3d.err, image_3d.err)
    assert image_test_3d.err.shape == (3,2,1024,1024)
    assert image_test_3d.err_hdr["Layer_1"] == "combined_error"
    assert image_test_3d.err_hdr["Layer_2"] == "error_noid"
    assert image_test_3d.err_hdr["Layer_3"] == "error_nuts"


def test_rescale_error():
    """
    test the rescale_error function

    Runs assuming tracking individual errors
    """
    corgidrp.track_individual_errors = True

    image_test = Image('test_image0.fits')
    scale_factor = 8.1
    image_test.rescale_error(scale_factor, "test_factor")
    assert np.allclose(image_test.err[1], err1 * scale_factor, rtol=1e-6)
    scale_factor = np.ones([1024,1024]) * 8.1
    image_test.rescale_error(scale_factor, "test_factor")
    assert np.allclose(image_test.err[1], err1 * scale_factor * scale_factor, rtol=1e-6)
    assert "test_factor" in str(image_test.err_hdr["HISTORY"])
    

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
    
    #test add_error_term for datasets
    dataset.add_error_term(err1, "err_add")
    print(dataset[0].err.shape)
    assert dataset[0].err.shape == (2, 1024, 1024)
    assert dataset[1].err.shape == (2, 1024, 1024)
    assert dataset[0].err_hdr["Layer_2"] == "err_add"
    
    #test rescale_error for datasets
    scale_factor = 8.1
    dataset.rescale_error(scale_factor, "scale_test")
    assert np.array_equal(dataset[0].err[1], err1 * scale_factor)
    assert np.array_equal(dataset[1].err[1], err1 * scale_factor)
    assert "scale_test" in str(dataset[1].err_hdr["HISTORY"])

def test_get_masked_data():
    """
    test the masked array
    """
    image2 = Image('test_image2.fits')
    masked_data = image2.get_masked_data()
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
    # clean out directory of old cals .fits files
    for filename in os.listdir(calibdir):
        if filename.endswith(".fits"):
            os.remove(os.path.join(calibdir, filename))
            
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
    for i in range(5):
        os.remove('test_image{0}.fits'.format(i))

    corgidrp.track_individual_errors = old_err_tracking


# for debugging. does not run with pytest!!
if __name__ == '__main__':
    test_single_float_data()
    test_err_dq_creation()
    test_err_dq_copy()
    test_add_error_term()
    test_rescale_error()
    test_err_dq_dataset()
    test_get_masked_data()
    test_err_adderr_notrack()
    test_read_many_errors_notrack()


    for i in range(5):
        os.remove('test_image{0}.fits'.format(i))