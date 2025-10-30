import os
import pickle
import pytest
import corgidrp
import numpy as np
import corgidrp.mocks as mocks
from corgidrp.mocks import create_default_calibration_product_headers
from corgidrp.l2a_to_l2b import correct_bad_pixels
from corgidrp.data import Image, Dataset, BadPixelMap
from corgidrp.l3_to_l4 import replace_bad_pixels

old_err_tracking = corgidrp.track_individual_errors

data = np.ones([1024, 1024]) * 2.
err = np.ones([1024, 1024]) * 0.5
dq = np.zeros([1024, 1024], dtype=np.uint16)
prhd, exthd, errhdr, dqhdr = create_default_calibration_product_headers()
constant = 10.
xgradient = np.arange(30)


def test_bad_pixels():
    corgidrp.track_individual_errors = False

    print("UT for pipeline step correct_bad_pixels")

    image1 = Image(data, pri_hdr=prhd, ext_hdr=exthd, err=err, dq=dq)
    dataset = Dataset([image1])

    corgidrp.track_individual_errors = old_err_tracking

    data_cube = dataset.all_data  # 3D data cube for all frames in the dataset
    dq_cube = dataset.all_dq  # 3D DQ array cube for all frames in the dataset
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
    col_bp_test = [12, 120, 234, 450, 678, 990]
    row_bp_test = [546, 89, 123, 243, 447, 675]
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
    for ii in nan_data[:, 0]:
        assert ii in col_cr_test or ii in col_bp_test
    for jj in nan_data[:, 1]:
        assert jj in row_cr_test or jj in row_bp_test

    # Checking that bad pixels are only at the expected locations
    replaced_pix = np.argwhere(new_dataset.all_dq[0] != 0)
    for ii in replaced_pix[:, 0]:
        assert ii in col_cr_test or ii in col_bp_test
    for jj in replaced_pix[:, 1]:
        assert jj in row_cr_test or jj in row_bp_test

    # Checking that DQ bit for replaced pixels is set to 1 (Big-Endian)
    for ii in replaced_pix[:, 0]:
        for jj in replaced_pix[:, 1]:
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


def test_replace_bps_2d():
    """Test that the replace_bad_pixels correctly patches bad pixels, and
    the error array, and does not modify the dq array, given a uniform 2d data array.
    """
    # Create a clean dataset with constant values in data & err
    input_dataset_clean, _ = mocks.create_psfsub_dataset(2, 0, [0, 0], data_shape=[30, 20])
    input_dataset_clean.all_data[:, :, :] = constant
    input_dataset_clean.all_err[:, :, :, :] = constant

    # Flag some bad pixels and assign erroneous values in data
    input_dataset_bad = input_dataset_clean.copy()

    # Pixel on the edge
    input_dataset_bad.all_dq[0, 0, 1] = 1
    input_dataset_bad.all_data[0, 0, 1] = 100.
    input_dataset_bad.all_err[0, :, 0, 1] = 100.

    # Pixel near the middle
    input_dataset_bad.all_dq[1, 9, 9] = 1
    input_dataset_bad.all_data[1, 9, 9] = 100.
    input_dataset_bad.all_err[1, :, 9, 9] = 100.

    # Patch of 4 pixels
    input_dataset_bad.all_dq[1, 15:17, 15:17] = 1
    input_dataset_bad.all_data[1, 15:17, 15:17] = 100.
    input_dataset_bad.all_err[1, :, 15:17, 15:17] = 100.

    # Run bad pixel cleaning
    cleaned_dataset = replace_bad_pixels(input_dataset_bad)

    if not cleaned_dataset.all_data == pytest.approx(input_dataset_clean.all_data):
        raise Exception("Cleaned data array does not match input data array for 2D uniform data.")
    if not cleaned_dataset.all_err == pytest.approx(input_dataset_clean.all_err):
        raise Exception("Cleaned error array does not match input error array for 2D uniform data.")
    if not cleaned_dataset.all_dq == pytest.approx(input_dataset_bad.all_dq):
        raise Exception("Output DQ array does not match input DQ array for 2D uniform data.")


def test_replace_bps_3d():
    """Test that the replace_bad_pixels correctly patches bad pixels, and
    the error array, and does not modify the dq array, given a uniform 3d data array."""
    # Create a clean dataset with constant values in data & err
    input_dataset_clean, _ = mocks.create_psfsub_dataset(2, 0, [0, 0], data_shape=[3, 30, 20])
    input_dataset_clean.all_data[:, :, :, :] = constant
    input_dataset_clean.all_err[:, :, :, :, :] = constant

    # Flag some bad pixels and assign erroneous values in data
    input_dataset_bad = input_dataset_clean.copy()

    # Pixel on the edge
    input_dataset_bad.all_dq[0, 2, 0, 1] = 1
    input_dataset_bad.all_data[0, 2, 0, 1] = 100.
    input_dataset_bad.all_err[0, :, 2, 0, 1] = 100.

    # Pixel near the middle
    input_dataset_bad.all_dq[1, 2, 9, 9] = 1
    input_dataset_bad.all_data[1, 2, 9, 9] = 100.
    input_dataset_bad.all_err[1, :, 2, 9, 9] = 100.

    # Run bad pixel cleaning
    cleaned_dataset = replace_bad_pixels(input_dataset_bad)

    if not cleaned_dataset.all_data == pytest.approx(input_dataset_clean.all_data):
        raise Exception("Cleaned data array does not match input data array for 3D uniform data.")
    if not cleaned_dataset.all_err == pytest.approx(input_dataset_clean.all_err):
        raise Exception("Cleaned error array does not match input error array for 3D uniform data.")
    if not cleaned_dataset.all_dq == pytest.approx(input_dataset_bad.all_dq):
        raise Exception("Output DQ array does not match input DQ array for 3D uniform data.")


def test_replace_bps_nonuniform():
    """Test that the replace_bad_pixels correctly patches bad pixels, and
    the error array, and does not modify the dq array, given a nonuniform (linear gradient)
    2d data array."""
    # Create a clean dataset with constant values in data & err
    input_dataset_clean, _ = mocks.create_psfsub_dataset(2, 0, [0, 0], data_shape=[30, 20])

    input_dataset_clean.all_data[:, :, :] = xgradient
    input_dataset_clean.all_err[:, :, :, :] = constant

    # Add bad pixel below dq threshold which we won't change
    input_dataset_clean.all_dq[1, -1, -1] = 0.1
    input_dataset_clean.all_data[1, -1, -1] = 50

    # Flag some bad pixels and assign erroneous values in data
    input_dataset_bad = input_dataset_clean.copy()

    # Pixel on the edge
    input_dataset_bad.all_dq[0, 0, 1] = 1
    input_dataset_bad.all_data[0, 0, 1] = 100.
    input_dataset_bad.all_err[0, :, 0, 1] = 100.

    # Pixel near the middle
    input_dataset_bad.all_dq[1, 9, 9] = 1
    input_dataset_bad.all_data[1, 9, 9] = 100.
    input_dataset_bad.all_err[1, :, 9, 9] = 100.

    # Another bad pixel
    input_dataset_bad.all_dq[1, 15, 15] = 1
    input_dataset_bad.all_data[1, 15, 15] = 100.
    input_dataset_bad.all_err[1, :, 15, 15] = 100.

    # Run bad pixel cleaning
    cleaned_dataset = replace_bad_pixels(input_dataset_bad)

    # for f,frame in enumerate(input_dataset_bad):
    #     import matplotlib.pyplot as plt
    #     fig,ax = plt.subplots(1,2,figsize=[10,5])
    #     ax[0].imshow(frame.data,vmin=0,vmax=30)
    #     ax[0].set_title('Input Data')

    #     ax[1].imshow(cleaned_dataset[f].data,vmin=0,vmax=30)
    #     ax[1].set_title('Cleaned Data')

    if not cleaned_dataset.all_data == pytest.approx(input_dataset_clean.all_data):
        raise Exception("Cleaned data array does not match input data array for 2D nonuniform data.")
    if not cleaned_dataset.all_err == pytest.approx(input_dataset_clean.all_err):
        raise Exception("Cleaned error array does not match input error array for 2D nonuniform data.")
    if not cleaned_dataset.all_dq == pytest.approx(input_dataset_bad.all_dq):
        raise Exception("Output DQ array does not match input DQ array for 2D nonuniform data.")


import copy


def create_pol_dataset(n_frames=1):
    """
    Creates a dataset for polarization data of shape (n_frames, 2, 1024, 1024).

    Args:
        n_frames (int): Number of frames to include in the dataset.

    Returns:
        np.ndarray: Polarization dataset with simulated error and DQ maps.
    """
    datashape = (2, 1024, 1024)  # pol frame shape
    bpixmap = np.zeros(datashape)
    ## CREATE POL DATASET
    field_path = os.path.join(os.path.dirname(__file__), "test_data", "JWST_CALFIELD2020.csv")
    # Create the first dataset as a template
    input_dataset_pol = mocks.create_astrom_data(field_path, bpix_map=bpixmap[0], sim_err_map=True)
    # Convert frames to a list (otherwise can't append)
    frames_list = list(input_dataset_pol.frames)

    for i in range(len(frames_list)):
        # Stack to create (2, 1024, 1024) for each frame
        frames_list[i].data = np.stack([frames_list[i].data,
                                        frames_list[i].data])
        original_err = frames_list[i].err
        frames_list[i].err = np.stack([original_err, original_err], axis=1)
        frames_list[i].dq = np.stack([frames_list[i].dq,
                                      frames_list[i].dq])

    # Duplicate existing frames if n_frames > 1
    if n_frames > len(frames_list):
        frames_to_add = n_frames - len(frames_list)
        for _ in range(frames_to_add):
            # Create a copy of the first frame
            new_frame = copy.deepcopy(frames_list[0])
            frames_list.append(new_frame)

    # Put new frames in dataset
    input_dataset_pol.frames = np.array(frames_list)
    # Stack all frames
    input_dataset_pol.all_data = np.array([frame.data for frame in input_dataset_pol.frames])
    input_dataset_pol.all_err = np.array([frame.err for frame in input_dataset_pol.frames])
    input_dataset_pol.all_dq = np.array([frame.dq for frame in input_dataset_pol.frames])

    return input_dataset_pol


def test_replace_bps_pol():
    """Test that the replace_bad_pixels correctly patches bad pixels, and
    the error array, and does not modify the dq array, given a uniform pol data array.
    """

    # test this w/ 1 frame
    input_dataset_pol = create_pol_dataset()
    # Set to uniform constant values
    input_dataset_clean = input_dataset_pol.copy()
    input_dataset_clean.all_data[:, :, :, :] = constant
    input_dataset_clean.all_err[:, :, :, :, :] = constant

    # Flag some bad pixels and assign erroneous values in data
    input_dataset_bad = input_dataset_clean.copy()

    # Pixel on the edge of first pol state
    input_dataset_bad.all_dq[0, 0, 0, 1] = 1
    input_dataset_bad.all_data[0, 0, 0, 1] = 100.
    input_dataset_bad.all_err[0, :, 0, 0, 1] = 100.

    # Pixel near the middle of second pol state
    input_dataset_bad.all_dq[0, 1, 9, 9] = 1
    input_dataset_bad.all_data[0, 1, 9, 9] = 100.
    input_dataset_bad.all_err[0, :, 1, 9, 9] = 100.

    # Patch of 4 pixels in second pol state
    input_dataset_bad.all_dq[0, 1, 15:17, 15:17] = 1
    input_dataset_bad.all_data[0, 1, 15:17, 15:17] = 100.
    input_dataset_bad.all_err[0, :, 1, 15:17, 15:17] = 100.

    # Run bad pixel cleaning
    cleaned_dataset = replace_bad_pixels(input_dataset_bad)

    if not cleaned_dataset.all_data == pytest.approx(input_dataset_clean.all_data):
        raise Exception("Cleaned data array does not match input data array for pol uniform data.")
    if not cleaned_dataset.all_err == pytest.approx(input_dataset_clean.all_err):
        raise Exception("Cleaned error array does not match input error array for pol uniform data.")
    if not cleaned_dataset.all_dq == pytest.approx(input_dataset_bad.all_dq):
        raise Exception("Output DQ array does not match input DQ array for pol uniform data.")

    print("UT passed for pol data")


def test_replace_bps_pol_2frames():
    """Test that the replace_bad_pixels correctly patches bad pixels, and
    the error array, and does not modify the dq array, given a uniform pol data array.
    """

    # test this w/ 2 frames
    input_dataset_pol = create_pol_dataset(2)
    # Set to uniform constant values
    input_dataset_clean = input_dataset_pol.copy()
    input_dataset_clean.all_data[:, :, :, :] = constant
    input_dataset_clean.all_err[:, :, :, :, :] = constant

    # Flag some bad pixels and assign erroneous values in data
    input_dataset_bad = input_dataset_clean.copy()

    # Pixel on the edge of first pol state
    input_dataset_bad.all_dq[0, 0, 0, 1] = 1
    input_dataset_bad.all_data[0, 0, 0, 1] = 100.
    input_dataset_bad.all_err[0, :, 0, 0, 1] = 100.

    # Pixel near the middle of second pol state
    input_dataset_bad.all_dq[0, 1, 9, 9] = 1
    input_dataset_bad.all_data[0, 1, 9, 9] = 100.
    input_dataset_bad.all_err[0, :, 1, 9, 9] = 100.

    # Patch of 4 pixels in second frame,  second pol state
    input_dataset_bad.all_dq[1, 1, 15:17, 15:17] = 1
    input_dataset_bad.all_data[1, 1, 15:17, 15:17] = 100.
    input_dataset_bad.all_err[1, :, 1, 15:17, 15:17] = 100.

    # pixel on edge of second frame, first pol state
    input_dataset_bad.all_dq[1, 0, 0, 1] = 1
    input_dataset_bad.all_data[1, 0, 0, 1] = 100.
    input_dataset_bad.all_err[1, :, 0, 0, 1] = 100.

    # Run bad pixel cleaning
    cleaned_dataset = replace_bad_pixels(input_dataset_bad)

    if not cleaned_dataset.all_data == pytest.approx(input_dataset_clean.all_data):
        raise Exception("Cleaned data array does not match input data array for pol uniform data.")
    if not cleaned_dataset.all_err == pytest.approx(input_dataset_clean.all_err):
        raise Exception("Cleaned error array does not match input error array for pol uniform data.")
    if not cleaned_dataset.all_dq == pytest.approx(input_dataset_bad.all_dq):
        raise Exception("Output DQ array does not match input DQ array for pol uniform data.")

    print("UT passed for pol data")


if __name__ == '__main__':
    test_bad_pixels()
    test_replace_bps_2d()
    test_replace_bps_3d()
    test_replace_bps_nonuniform()
    test_replace_bps_pol()
    test_replace_bps_pol_2frames()
