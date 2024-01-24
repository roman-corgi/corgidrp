import os
import numpy as np
from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset
import pytest


data = np.ones([1024,1024])
err = np.zeros([1024,1024])
dq = np.zeros([1024,1024], dtype = np.uint16)
dq[0,0] = 1 
prhd, exthd = create_default_headers()
    
def test_error_dq():
    print("test Image class")
    image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd)
    print("data",image1.data)
    image2 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq)
    print("data", image2.data)
    print("error", image2.err)
    print("dq", image2.dq)
    image2.save('test_image.fits')
    image3 = Image('test_image.fits')
    assert np.array_equal(image3.data, image2.data)
    assert np.array_equal(image3.err, image2.err)
    assert np.array_equal(image3.dq, image2.dq)
        
    image4 = image2.copy()
    assert np.array_equal(image4.data, image2.data)
    assert np.array_equal(image4.err, image2.err)
    assert np.array_equal(image4.dq, image2.dq)
 
    dataset = Dataset(["test_image.fits", "test_image.fits"])
    assert(np.array_equal(dataset[0].data, dataset[1].data))
    assert(np.array_equal(dataset[0].err, dataset[1].err))
    assert(np.array_equal(dataset[0].dq, dataset[1].dq))
    assert (dataset.all_data.shape[0] == len(dataset))
    assert(dataset.all_err.shape[1] == err.shape[0])
    assert(dataset.all_dq.shape[2] == dq.shape[1])
    os.remove('test_image.fits')
    pytest.raises(TypeError, image1.get_masked_data)
    masked_data = image2.get_masked_data()
    print("masked data", masked_data.data)
    print("mask", masked_data.mask)
    assert(masked_data.data[0,1] == 1)
    assert(masked_data.mask[0,0] == True)
    assert(masked_data.mean()==1)
    assert(masked_data.sum()==image1.data.sum()-1)
    
if __name__ == '__main__':
    test_error_dq()
