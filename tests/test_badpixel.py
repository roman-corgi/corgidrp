import os
import pytest
import corgidrp
import numpy as np
from corgidrp.mocks import create_default_headers
from corgidrp.l2a_to_l2b import correct_bad_pixels
from corgidrp.data import Image, Dataset, BadPixelMap

old_err_tracking = corgidrp.track_individual_errors

data = np.ones([1024,1024])*2.
err = np.ones([1024,1024]) *0.5
dq = np.zeros([1024,1024], dtype = np.uint16)
prhd, exthd = create_default_headers()

def test_bad_pixels():

    corgidrp.track_individual_errors = True

    print("test correct bad pixels pipeline step")

    image1 = Image(data, pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq)
    dataset = Dataset([image1])

    corgidrp.track_individual_errors = old_err_tracking

    data_cube = dataset.all_data # 3D data cube for all frames in the dataset
    dq_cube = dataset.all_dq # 3D DQ array cube for all frames in the dataset
    # Add some CR
    col_cr = [12, 123, 234, 456, 678, 890]
    row_cr = [546, 789, 123, 43, 547, 675]
    for i_col in col_cr:
        for i_row in row_cr:
            dq_cube[0, i_col, i_row] += 128

    history_msg = "Pixels affected by CR added"
    dataset.update_after_processing_step(history_msg, new_all_data=data_cube,
        new_all_dq=dq_cube)

    assert type(dataset) == corgidrp.data.Dataset

    # Generate bad pixel detector mask
    breakpoint()
    bp_mask = BadPixelMap(np.zeros([1024,1024], dtype = np.uint16), pri_hdr = prhd, ext_hdr = exthd)

    # Add some Bad Detector Pixels
    col_bp = [12, 120, 234, 450, 678, 990]
    row_bp = [546, 89, 123, 243, 447, 675]
    for i_col in col_bp:
        for i_row in row_bp:
            bp_mask[i_col, i_row] += 4

    assert type(dataset) == corgidrp.data.BadPixelMap

    new_dataset = correct_bad_pixels(dataset, bp_mask)

    assert type(new_dataset) == corgidrp.data.Dataset

    # Checking that NaN are only at the expected locations
    nan_data = np.where(new_dataset.all_data[0] == np.nan)
    for ii in nan_data[0]:
        assert ii in col_cr or ii in col_bp
    for jj in nan_data[1]:
        assert jj in row_cr or jj in row_bp

    # Checking that CR are at the expected locations
    cr_dq = np.where(new_dataset.all_dq[0] == 4)
    for ii in cr_dq[0]:
        assert ii in col_cr
    for jj in cr_dq[1]:
        assert jj in row_cr

    # Checking that bad pixels are at the expected locations
    bp_dq = np.where(new_dataset.all_dq[0] == 128)
    for ii in bp_dq[0]:
        assert ii in col_bp
    for jj in bp_dq[1]:
        assert jj in row_bp

    # Checking that coincident bad pixels and CR are at the expected locations
    bp_cr_dq = np.where(new_dataset.all_dq[0] == 132)
    for ii in bp_cr_dq[0]:
        assert ii in col_bp and ii in col_cr
    for jj in bp_cr_dq[1]:
        assert jj in row_bp and jj in row_cr

    breakpoint()

if __name__ == '__main__':
    test_bad_pixels()
