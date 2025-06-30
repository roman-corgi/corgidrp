import os
import numpy as np
import corgidrp
from corgidrp.mocks import create_default_L2b_headers
from corgidrp.data import Image, Dataset
from corgidrp.l2a_to_l2b import add_photon_noise
import pytest

old_err_tracking = corgidrp.track_individual_errors

data = np.ones([1024,1024])*2.
err = np.ones([1024,1024]) *0.5
dq = np.zeros([1024,1024], dtype = np.uint16)
# TO DO: Check to confirm this is correct data level
prhd, exthd, errhdr, dqhdr, biashdr = create_default_L2b_headers()

def test_add_phot_noise():
    corgidrp.track_individual_errors = True

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
    assert "photnoise_error" == dataset_add.frames[0].err_hdr["Layer_2"]
    #check that excess noise is applied
    dataset[0].ext_hdr["EMGAIN_C"] = 3000
    dataset_add1 = add_photon_noise(dataset)
    all_err2 = dataset_add1.all_err
    assert np.array_equal(all_err2[0,1], np.sqrt(data)*np.sqrt(2))
    
    corgidrp.track_individual_errors = old_err_tracking
    
if __name__ == '__main__':
    test_add_phot_noise()
