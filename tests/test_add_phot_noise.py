import os
import numpy as np
from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset
from corgidrp.l2a_to_l2b import add_photon_noise
import pytest


data = np.ones([1024,1024])*2.
err = np.ones([1024,1024]) *0.5
dq = np.zeros([1024,1024], dtype = np.uint16)
prhd, exthd = create_default_headers()

def test_add_phot_noise():
    print("test add_photon_noise pipeline step")
    image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq)
    image2 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq)
    
    dataset = Dataset([image1, image2])
    dataset_add = add_photon_noise(dataset)
    all_err = dataset.all_err
    all_err1 = dataset_add.all_err
    assert not np.array_equal(all_err, all_err1)
    assert np.array_equal(all_err1[0,1], np.sqrt(data))
    assert np.allclose(all_err1[0,0], np.sqrt(data + np.square(err)))
    assert "noise" in str(dataset_add.frames[0].ext_hdr["HISTORY"])
    
if __name__ == '__main__':
    test_add_phot_noise()
