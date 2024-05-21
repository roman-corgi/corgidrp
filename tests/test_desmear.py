import os
import pytest
import corgidrp
import numpy as np
from corgidrp.mocks import create_default_headers
from corgidrp.l2a_to_l2b import desmear
from corgidrp.data import Image, Dataset, BadPixelMap

old_err_tracking = corgidrp.track_individual_errors

data = np.ones([1024,1024])*2.
err = np.ones([1024,1024]) *0.5
dq = np.zeros([1024,1024], dtype = np.uint16)
prhd, exthd = create_default_headers()

def test_desmear():

    corgidrp.track_individual_errors = True

    print("test desmear step")

    image1 = Image(data, pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq)
    dataset = Dataset([image1])

    corgidrp.track_individual_errors = old_err_tracking

    data_cube = dataset.all_data # 3D data cube for all frames in the dataset
    dq_cube = dataset.all_dq # 3D DQ array cube for all frames in the dataset

    # Add some smear effect? (Peter Williams)

    history_msg = "Pixels affected by Smear added"
    dataset.update_after_processing_step(history_msg, new_all_data=data_cube,
        new_all_dq=dq_cube)

    assert type(dataset) == corgidrp.data.Dataset

    # Apply desmear correction
    new_dataset = desmear(dataset)

    assert type(new_dataset) == corgidrp.data.Dataset

    # Test desmear correction (Peter Williams)
    

    
