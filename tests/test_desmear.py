import os
import pytest
import corgidrp
import numpy as np
import corgidrp.detector as detector
from astropy.time import Time
from corgidrp.l2a_to_l2b import desmear
from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset, BadPixelMap

old_err_tracking = corgidrp.track_individual_errors

##rowreadtime_sec = detector.get_rowreadtime_sec()
rowreadtime_sec = 223.5e-6

data = np.ones([1024,1024])
err = np.ones([1024,1024]) *0.5
dq = np.zeros([1024,1024], dtype = np.uint16)
prhd, exthd = create_default_headers()

def test_desmear():
    # Tolerance for comparisons
    tol = 1e-13

    corgidrp.track_individual_errors = True

    print("test desmear step")

    image1 = Image(data, pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq)
    dataset = Dataset([image1])

    corgidrp.track_individual_errors = old_err_tracking

    data_cube = dataset.all_data # 3D data cube for all frames in the dataset
    dq_cube = dataset.all_dq # 3D DQ array cube for all frames in the dataset

    # Add some smear effect
    e_t=60
    data_cube_smear = e_t*data_cube + rowreadtime_sec

    history_msg = "Added pixels affected by smearing"
    dataset.update_after_processing_step(history_msg, new_all_data=data_cube_smear,
        new_all_dq=dq_cube)

    assert type(dataset) == corgidrp.data.Dataset

    # Apply desmear correction
    dataset_desmear = desmear(dataset)

    assert type(dataset_desmear) == corgidrp.data.Dataset

#    assert(np.max(np.abs(desmeared_frame - i1)) < tol)
    
if __name__ == '__main__':
    test_desmear()
    
