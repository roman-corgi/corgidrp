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
dq = np.zeros([1024,1024], dtype = np.uint16)
dq1 = dq.copy()
dq1[0,0] = 1 
prhd, exthd = create_default_headers()

def test_error_dq():
    print("test Image class")
    image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd)
    assert hasattr(image1, "err")
    assert hasattr(image1, "dq")
    assert image1.data.shape == image1.err.shape
    assert image1.data.shape == image1.dq.shape
    assert np.mean(image1.dq) == 0
    image1.save('test_image1.fits')
    
    image2 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq1)
    print("data", image2.data)
    print("error", image2.err)
    print("dq", image2.dq)
    image2.save('test_image2.fits')
    image3 = Image('test_image2.fits')
    assert np.array_equal(image3.data, image2.data)
    assert np.array_equal(image3.err, image2.err)
    assert np.array_equal(image3.dq, image2.dq)
    
    image_test = Image('test_image2.fits', err = err, dq = dq)
    assert(np.array_equal(image_test.dq, dq))
    assert(np.array_equal(image_test.err, err))
    
    #test copy
    image4 = image2.copy()
    assert np.array_equal(image4.data, image2.data)
    assert np.array_equal(image4.err, image2.err)
    assert np.array_equal(image4.dq, image2.dq)
    
    #test add_error_term
    image1.add_error_term(err1, "error_noid")
    assert(image1.err[0,0,0] == err1[0,0])
    image1.add_error_term(err2, "error_nuts")
    assert(image1.err.shape == (3,1024,1024))
    assert(image1.err[0,0,0] == np.sqrt(err1[0,0]**2 + err2[0,0]**2))
    image1.save("test_image0.fits")
    
    image_test = Image('test_image0.fits')
    assert(np.array_equal(image_test.dq, dq))
    assert(np.array_equal(image_test.err, image1.err))
    assert(image_test.err.shape == (3,1024,1024))
    assert(image_test.errhd["Layer_1"] == "combined_error")
    assert(image_test.errhd["Layer_2"] == "error_noid")
    assert(image_test.errhd["Layer_3"] == "error_nuts")
    
    
    image4.add_error_term(err1, "error_noid")
    assert(image4.err[0,0,0] == err1[0,0])
    image4.add_error_term(err2, "error_nuts")
    assert(image4.err.shape == (3,1024,1024))
    assert(image4.err[0,0,0] == np.sqrt(err1[0,0]**2 + err2[0,0]**2))
    image5 = image4.copy()
    assert(image5.errhd[:] == image4.errhd[:])
    assert(image5.dqhd[:] == image4.dqhd[:])
    assert(np.array_equal(image5.err, image4.err))
    
    image4.save("test_image3.fits")
    
    
 
    #test dataset with error and dq
    dataset = Dataset(["test_image1.fits", "test_image2.fits"])
    assert np.array_equal(dataset[0].data, dataset[1].data)
    assert np.array_equal(dataset[0].err, dataset[1].err)
    assert not np.array_equal(dataset[0].dq, dataset[1].dq)
    assert dataset.all_data.shape[0] == len(dataset)
    assert dataset.all_err.shape[1] == err.shape[0]
    assert dataset.all_dq.shape[2] == dq.shape[1]
    
    dataset = Dataset(["test_image0.fits", "test_image3.fits"])
    assert np.array_equal(dataset[0].data, dataset[1].data)
    assert np.array_equal(dataset[0].err, dataset[1].err)
    assert dataset.all_err.ndim == 4
    assert not np.array_equal(dataset[0].dq, dataset[1].dq)
    assert dataset.all_data.shape[0] == len(dataset)
    print("shape of all_err", dataset.all_err.shape)
    assert dataset.all_err.shape[2] == err.shape[0]
    assert dataset.all_dq.shape[2] == dq.shape[1]
   
    #test get_masked_data
    masked_data = image2.get_masked_data()
    print("masked data", masked_data.data)
    print("mask", masked_data.mask)
    assert masked_data.data[0,1] == 2
    assert masked_data.mask[0,0] == True
    assert masked_data.mean()==2
    assert masked_data.sum()==image1.data.sum()-2
    
    #for i in range(4):
    #    os.remove('test_image{0}.fits'.format(i))
    
if __name__ == '__main__':
    test_error_dq()
