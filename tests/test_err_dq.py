import os
import numpy as np
from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset
import pytest
import glob


data = np.ones([1024,1024]) * 2
err = np.zeros([1024,1024]) 
err1 = np.ones([1024,1024])
err2 = err1.copy()
dq = np.zeros([1024,1024], dtype = int)
dq1 = dq.copy()
dq1[0,0] = 1 
prhd, exthd = create_default_headers()

def test_err_dq_creation():
    """
     test the initialization of error and dq attributes of the Image class
    """
    image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd)
    assert hasattr(image1, "err")
    assert hasattr(image1, "dq")
    assert image1.data.shape == image1.err.shape[-2:]
    assert image1.data.shape == image1.dq.shape
    assert hasattr(image1, "errhd")
    assert hasattr(image1, "dqhd")
    assert np.mean(image1.dq) == 0
    image1.save('test_image1.fits')
    
    image2 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq1)
    print("data", image2.data)
    print("error", image2.err)
    print("dq", image2.dq)
    #check the correct saving and loading of fits files
    image2.save('test_image2.fits')
    image3 = Image('test_image2.fits')
    assert np.array_equal(image3.data, image2.data)
    assert np.array_equal(image3.err, image2.err)
    assert np.array_equal(image3.dq, image2.dq)
    
    #check the overwriting of err and dq parameters of the fits extensions arrays 
    image_test = Image('test_image2.fits', err = err, dq = dq)
    assert np.array_equal(image_test.dq, dq)
    assert np.array_equal(image_test.err[0,:,:], err)
    
def test_err_dq_copy():
    """
    test the copying of the err and dq attributes
    """
    image2 = Image('test_image2.fits')
    image3 = image2.copy()
    assert np.array_equal(image3.data, image2.data)
    assert np.array_equal(image3.err, image2.err)
    assert np.array_equal(image3.dq, image2.dq)
    #check the copying of the headers of the extensions
    assert image2.errhd[:] == image3.errhd[:]
    assert image2.dqhd[:] == image3.dqhd[:]
    
def test_add_error_term():
    """
    test the add_error_term function
    """
    image1 = Image('test_image1.fits')
    image1.add_error_term(err1, "error_noid")
    assert image1.err[0,0,0] == err1[0,0]
    image1.add_error_term(err2, "error_nuts")
    assert image1.err.shape == (3,1024,1024)
    assert image1.err[0,0,0] == np.sqrt(err1[0,0]**2 + err2[0,0]**2)
    image1.save("test_image0.fits")
    
    image_test = Image('test_image0.fits')
    assert np.array_equal(image_test.dq, dq)
    assert np.array_equal(image_test.err, image1.err)
    assert image_test.err.shape == (3,1024,1024)
    assert image_test.errhd["Layer_1"] == "combined_error"
    assert image_test.errhd["Layer_2"] == "error_noid"
    assert image_test.errhd["Layer_3"] == "error_nuts"
 
def test_err_dq_dataset():
    """
    test the behavior of the err and data arrays in the dataset      
    """
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
    
    
if __name__ == '__main__':
    test_err_dq_creation()
    test_err_dq_copy()
    test_add_error_term()
    test_err_dq_dataset()
    test_get_masked_data()
    for i in range(3):
        os.remove('test_image{0}.fits'.format(i))