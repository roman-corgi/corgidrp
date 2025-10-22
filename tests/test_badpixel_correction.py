import os
import pickle
import pytest
import corgidrp
import numpy as np
import corgidrp.mocks as mocks
from corgidrp.mocks import create_default_calibration_product_headers
from corgidrp.l2a_to_l2b import correct_bad_pixels
from corgidrp.data import Image, Dataset, BadPixelMap

old_err_tracking = corgidrp.track_individual_errors

data = np.ones([1024,1024])*2.
err = np.ones([1024,1024]) *0.5
dq = np.zeros([1024,1024], dtype = np.uint16)
prhd, exthd, errhdr, dqhdr = create_default_calibration_product_headers()

def test_bad_pixels():

    corgidrp.track_individual_errors = False

    print("UT for pipeline step correct_bad_pixels")

    image1 = Image(data, pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq)
    dataset = Dataset([image1])

    corgidrp.track_individual_errors = old_err_tracking

    data_cube = dataset.all_data # 3D data cube for all frames in the dataset
    dq_cube = dataset.all_dq # 3D DQ array cube for all frames in the dataset
    # Add some CR
    col_cr_test = [12, 123, 234, 456, 678, 890]
    row_cr_test = [546, 789, 123, 43, 547, 675]
    for i_col in col_cr_test:
        for i_row in row_cr_test:
            dq_cube[0, i_col, i_row] = np.bitwise_or(dq_cube[0, i_col, i_row].astype(np.uint8), 128)

    history_msg = "Pixels affected by CR added"
    dataset.update_after_processing_step(history_msg, new_all_data=data_cube,
        new_all_dq=dq_cube)

    assert type(dataset) == corgidrp.data.Dataset

    # Generate bad pixel detector mask
    datadir = os.path.join(os.path.dirname(__file__), "testcalib")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    outputdir = os.path.join(os.path.dirname(__file__), "testcalib")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    col_bp_test=[12, 120, 234, 450, 678, 990]
    row_bp_test=[546, 89, 123, 243, 447, 675]
    bp_mask = mocks.create_badpixelmap_files(filedir=datadir,
        col_bp=col_bp_test, row_bp=row_bp_test)
    new_bp_mask = BadPixelMap(bp_mask.all_data[0], pri_hdr=bp_mask[0].pri_hdr.copy(),
                     ext_hdr=bp_mask[0].ext_hdr.copy(), input_dataset=bp_mask)

    assert type(new_bp_mask) == corgidrp.data.BadPixelMap
    
    # check the bpmap can be pickled (for CTC operations)
    pickled = pickle.dumps(new_bp_mask)
    pickled_bpmap = pickle.loads(pickled)
    assert np.all((new_bp_mask.data == pickled_bpmap.data))


    new_dataset = correct_bad_pixels(dataset, new_bp_mask)

    assert type(new_dataset) == corgidrp.data.Dataset

    # Checking that NaN are only at the expected locations
    nan_data = np.argwhere(np.isnan(new_dataset.all_data[0]))
    for ii in nan_data[:,0]:
        assert ii in col_cr_test or ii in col_bp_test
    for jj in nan_data[:,1]:
        assert jj in row_cr_test or jj in row_bp_test

    # Checking that bad pixels are only at the expected locations
    replaced_pix = np.argwhere(new_dataset.all_dq[0] != 0)
    for ii in replaced_pix[:,0]:
        assert ii in col_cr_test or ii in col_bp_test
    for jj in replaced_pix[:,1]:
        assert jj in row_cr_test or jj in row_bp_test

    # Checking that DQ bit for replaced pixels is set to 1 (Big-Endian)
    for ii in replaced_pix[:,0]:
        for jj in replaced_pix[:,1]:
            # Only bad pixels
            if new_dataset.all_dq[0][ii][jj] != 0:
                assert np.unpackbits(new_dataset.all_dq[0][ii][jj].astype('uint8'))[6] == 1

    # Checking that bad pixels are at the expected locations
    bp_dq = np.where(new_dataset.all_dq[0] == 4)
    for ii in bp_dq[0]:
        assert ii in col_bp_test
    for jj in bp_dq[1]:
        assert jj in row_bp_test

    # Checking that CR are at the expected locations
    cr_dq = np.where(new_dataset.all_dq[0] == 128)
    for ii in cr_dq[0]:
        assert ii in col_cr_test
    for jj in cr_dq[1]:
        assert jj in row_cr_test

    # Checking that coincident bad pixels and CR are at the expected locations
    bp_cr_dq = np.where(new_dataset.all_dq[0] == 132)
    for ii in bp_cr_dq[0]:
        assert ii in col_bp_test and ii in col_cr_test
    for jj in bp_cr_dq[1]:
        assert jj in row_bp_test and jj in row_cr_test


    # save and reload bad pixel map
    new_bp_mask.save(filedir=outputdir, filename="sim_bp_map_cal.fits")
    new_bp_mask_2 = BadPixelMap(os.path.join(outputdir, "sim_bp_map_cal.fits"))
    
    # check the bpmap can be pickled (for CTC operations)
    pickled = pickle.dumps(new_bp_mask_2)
    pickled_bpmap = pickle.loads(pickled)
    assert np.all((new_bp_mask_2.data == pickled_bpmap.data))

    print("UT passed")

if __name__ == '__main__':
    test_bad_pixels()
