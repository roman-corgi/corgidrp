import os
import pytest
import numpy as np
import astropy.io.fits as fits
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.combine as combine
from corgidrp.mocks import create_psfsub_dataset

img1 = np.ones([100, 100])
err1 = np.ones([100, 100])
dq = np.zeros([100, 100], dtype = int)
prhd, exthd = mocks.create_default_L1_headers()
errhd = fits.Header()
errhd["CASE"] = "test"
dqhd = fits.Header()
dqhd["CASE"] = "test"


def test_mean_combine_subexposures():
    """
    Test mean combine of subexposures
    """

    image1 = data.Image(img1, err=err1, dq=dq, pri_hdr = prhd, ext_hdr = exthd)
    image1.filename = "1.fits"
    image2 = image1.copy()
    image2.filename = "2.fits"
    image3 = image1.copy()
    image3.filename = "3.fits"
    image4 = image1.copy()
    image4.filename = "4.fits"

    dataset = data.Dataset([image1, image2, image3, image4])

    combined_dataset = combine.combine_subexposures(dataset, 2)

    assert(len(combined_dataset) == 2)
    assert(np.all(combined_dataset[0].data == 2))
    assert(np.all(combined_dataset[0].err == pytest.approx(np.sqrt(2))))
    assert(np.all(combined_dataset[0].dq == 0))

    assert combined_dataset[0].ext_hdr['DRPNFILE'] == 2
    assert combined_dataset[0].ext_hdr['FILE0'] in ["2.fits", "1.fits"]
    assert combined_dataset[0].ext_hdr['FILE1'] in ["2.fits", "1.fits"]
    assert combined_dataset[0].ext_hdr['NUM_FR'] == 2

    assert combined_dataset[1].ext_hdr['DRPNFILE'] == 2
    assert combined_dataset[1].ext_hdr['FILE0'] in ["3.fits", "4.fits"]
    assert combined_dataset[1].ext_hdr['FILE1'] in ["3.fits", "4.fits"]
    assert combined_dataset[1].ext_hdr['NUM_FR'] == 2
    # combine again
    combined_dataset_2 = combine.combine_subexposures(combined_dataset, 2)
     
    assert(len(combined_dataset_2) == 1)
    assert(np.all(combined_dataset_2[0].data == 4))
    assert(np.all(combined_dataset_2[0].err == pytest.approx(2)))
    assert(np.all(combined_dataset_2[0].dq == 0))

    assert combined_dataset_2[0].ext_hdr['DRPNFILE'] == 4
    assert combined_dataset_2[0].ext_hdr['FILE0'] in ["2.fits", "1.fits", "3.fits", "4.fits"]
    assert combined_dataset_2[0].ext_hdr['FILE1'] in ["2.fits", "1.fits", "3.fits", "4.fits"]
    assert combined_dataset_2[0].ext_hdr['FILE2'] in ["2.fits", "1.fits", "3.fits", "4.fits"]
    assert combined_dataset_2[0].ext_hdr['FILE3'] in ["2.fits", "1.fits", "3.fits", "4.fits"]

def test_mean_combine_subexposures_without_scaling():
    """
    Test mean combine of subexposures for case where num_frames_scaling=False
    """

    image1 = data.Image(img1, err=err1, dq=dq, pri_hdr = prhd, ext_hdr = exthd)
    image1.filename = "1.fits"
    image2 = image1.copy()
    image2.filename = "2.fits"
    image3 = image1.copy()
    image3.filename = "3.fits"
    image4 = image1.copy()
    image4.filename = "4.fits"

    dataset = data.Dataset([image1, image2, image3, image4])

    combined_dataset = combine.combine_subexposures(dataset, 2, num_frames_scaling=False)

    # Check that data and error values are not scaled up
    assert(len(combined_dataset) == 2)
    assert(np.all(combined_dataset[0].data == 1))
    assert(np.all(combined_dataset[0].err == pytest.approx(1/np.sqrt(2))))
    assert(np.all(combined_dataset[0].dq == 0))

    # combine again
    combined_dataset_2 = combine.combine_subexposures(combined_dataset, 2, num_frames_scaling=False)
     
    assert(len(combined_dataset_2) == 1)
    assert(np.all(combined_dataset_2[0].data == 1))
    assert(np.all(combined_dataset_2[0].err == pytest.approx((1/np.sqrt(2))/np.sqrt(2))))
    assert(np.all(combined_dataset_2[0].dq == 0))

def test_mean_combine_subexposures_with_bad():
    """
    Test mean combine of subexposures with bad pixels
    """
    # use copies since we are going to modify their values
    image1 = data.Image(np.copy(img1), err=np.copy(err1), dq=np.copy(dq), 
                        pri_hdr = prhd, ext_hdr = exthd)
    image1.filename = "1.fits"
    image2 = image1.copy()
    image2.filename = "2.fits"
    image3 = image1.copy()
    image3.filename = "3.fits"
    image4 = image1.copy()
    image4.filename = "4.fits"

    # (0,0) has one bad frame
    image1.dq[0][0] = 1
    # (0,1) has both pixels bad
    image1.dq[0][1] = 1
    image2.dq[0][1] = 1

    dataset = data.Dataset([image1, image2, image3, image4])

    combined_dataset = combine.combine_subexposures(dataset, 2)

    assert(len(combined_dataset) == 2)
    # the pixel with one bad pixel should have same value but higher error
    assert combined_dataset[0].data[0][0] == 2
    assert combined_dataset[0].err[0][0][0] == pytest.approx(2)
    assert combined_dataset[0].dq[0][0] == 0 # 0 because one of the two frames had a good value
    # compare against a pixel without any bad pixels
    assert combined_dataset[1].data[0][0] == 2
    assert combined_dataset[1].err[0][0][0] == pytest.approx(np.sqrt(2))
    assert combined_dataset[1].dq[0][0] == 0

    # the pixel with two bad pixels should be nan
    assert np.isnan(combined_dataset[0].data[0][1])
    assert np.isnan(combined_dataset[0].err[0][0][1])
    assert combined_dataset[0].dq[0][1] == 1

def test_median_combine_subexposures():
    """
    Test median combine of subexposures. And tests default case where num_frames_per_group isn't specified. 
    """

    image1 = data.Image(img1, err=err1, dq=dq, pri_hdr = prhd, ext_hdr = exthd)
    image1.filename = "1.fits"
    image2 = image1.copy()
    image2.filename = "2.fits"
    image3 = image1.copy()
    image3.filename = "3.fits"
    image4 = image1.copy()
    image4.filename = "4.fits"

    dataset = data.Dataset([image1, image2, image3, image4])

    combined_dataset = combine.combine_subexposures(dataset, collapse="median")

    assert(len(combined_dataset) == 1)
    assert(np.all(combined_dataset[0].data == 4))
    assert(np.all(combined_dataset[0].err == pytest.approx(np.sqrt(2*np.pi))))
    assert(np.all(combined_dataset[0].dq == 0))

def test_median_combine_subexposures_with_bad():
    """
    Test median combine of subexposures with bad pixels over multiple combinations
    """
    # use copies since we are going to modify their values
    image1 = data.Image(np.copy(img1), err=np.copy(err1), dq=np.copy(dq), 
                        pri_hdr = prhd, ext_hdr = exthd)
    image1.filename = "1.fits"
    image2 = image1.copy()
    image2.filename = "2.fits"
    image3 = image1.copy()
    image3.filename = "3.fits"
    image4 = image1.copy()
    image4.filename = "4.fits"

    # (0,0) has one bad frame
    image1.dq[0][0] = 1
    # (0,1) has both pixels bad
    image1.dq[0][1] = 1
    image2.dq[0][1] = 1

    dataset = data.Dataset([image1, image2, image3, image4])

    combined_dataset = combine.combine_subexposures(dataset, 2, collapse="median")

    assert(len(combined_dataset) == 2)
    # the pixel with one bad pixel should have same value but higher error. In both cases the error should be inflated by np.sqrt(np.pi/2) compared to mean error.
    assert combined_dataset[0].data[0][0] == 2
    assert combined_dataset[0].err[0][0][0] == pytest.approx(2 * np.sqrt(np.pi/2))
    assert combined_dataset[0].dq[0][0] == 0 # 0 because one of the two frames had a good value
    # compare against a pixel without any bad pixels
    assert combined_dataset[1].data[0][0] == 2
    assert combined_dataset[1].err[0][0][0] == pytest.approx(np.sqrt(2) * np.sqrt(np.pi/2))
    assert combined_dataset[1].dq[0][0] == 0

    # the pixel with two bad pixels should be nan
    assert np.isnan(combined_dataset[0].data[0][1])
    assert np.isnan(combined_dataset[0].err[0][0][1])
    assert combined_dataset[0].dq[0][1] == 1

    # combine again
    combined_dataset_2 = combine.combine_subexposures(combined_dataset, 2, collapse="median")
     
    assert(len(combined_dataset_2) == 1)
    assert(np.all(combined_dataset_2[0].data == 4))

    # error for pixel with no bad pixels in original data (i.e. most pixels in data)
    assert combined_dataset_2[0].err[0][5][0] == pytest.approx(np.pi)

    # error for pixel with one bad pixel in original data (i.e. no nans after first combination)
    assert combined_dataset_2[0].err[0][0][0] == pytest.approx(0.5 * np.pi * np.sqrt(6))

    # error for pixel with two bad pixels in original data (i.e. 1 nan after first combination)
    assert combined_dataset_2[0].err[0][0][1] == pytest.approx(np.pi * np.sqrt(2))

    assert(np.all(combined_dataset_2[0].dq == 0))

def test_combine_different_values():
    """
    Test whether the function correctly combines different values.
    """

    # use copies since we are going to modify their values
    image1 = data.Image(np.copy(img1), err=np.copy(err1), dq=np.copy(dq), 
                        pri_hdr = prhd, ext_hdr = exthd)
    image1.filename = "1.fits"
    image2 = image1.copy()
    image2.filename = "2.fits"
    image3 = image1.copy()
    image3.filename = "3.fits"
    image4 = image1.copy()
    image4.filename = "4.fits"

    # (0,0) has different values in each frame. Some are bad pixels.
    image1.data[0][0] = 5
    image2.data[0][0] = 6
    image3.data[0][0] = 9
    image4.data[0][0] = 19

    image2.dq[0][0] = 1
    image4.dq[0][0] = 1

    # (0,1) is a bad pixel in every frame
    image1.dq[0][1] = 1
    image2.dq[0][1] = 1
    image3.dq[0][1] = 1
    image4.dq[0][1] = 1

    dataset = data.Dataset([image1, image2, image3, image4])

    combined_dataset = combine.combine_subexposures(dataset, collapse="median")

    assert(len(combined_dataset) == 1)

    # Most pixels had good values of 1 in all frames
    assert combined_dataset[0].data[0][2] == 4
    assert combined_dataset[0].err[0][0][2] == pytest.approx(2*np.sqrt(np.pi/2))

    # (0,0) has a different median value calculated ignoring nans
    assert combined_dataset[0].data[0][0] == 7 * 4 # median value scaled by number of images
    assert combined_dataset[0].err[0][0][0] == pytest.approx(2 * np.sqrt(np.pi))

    # (0,1) is a nan
    assert np.isnan(combined_dataset[0].data[0][1])
    assert np.isnan(combined_dataset[0].err[0][0][1])

    # the updated bad pixel map only contains one bad pixel (i.e. the pixel for which there were no good values)
    assert combined_dataset[0].dq[0][0] == 0
    assert combined_dataset[0].dq[0][1] == 1

def test_not_divisible():
    """
    Tests that function correctly fails when the length of the dataset is not divisible by num_frames_per_group.
    """

    image1 = data.Image(img1, err=err1, dq=dq, pri_hdr = prhd, ext_hdr = exthd)
    image1.filename = "1.fits"
    image2 = image1.copy()
    image2.filename = "2.fits"
    image3 = image1.copy()
    image3.filename = "3.fits"
    image4 = image1.copy()
    image4.filename = "4.fits"

    dataset = data.Dataset([image1, image2, image3, image4])

    with pytest.raises(ValueError):
        combined_dataset = combine.combine_subexposures(dataset, 3) # Should fail as 4 % 3 != 0

def test_invalid_collapse():
    """
    Tests that function correctly fails when collapse type is not valid.
    """

    image1 = data.Image(img1, err=err1, dq=dq, pri_hdr = prhd, ext_hdr = exthd)
    image1.filename = "1.fits"
    image2 = image1.copy()
    image2.filename = "2.fits"
    image3 = image1.copy()
    image3.filename = "3.fits"
    image4 = image1.copy()
    image4.filename = "4.fits"

    dataset = data.Dataset([image1, image2, image3, image4])

    with pytest.raises(ValueError):
        combined_dataset = combine.combine_subexposures(dataset, collapse="invalid_option")

def test_prop_err_dq():
    """
    Test that the expected dq and err arrays are produced in RDI, ADI, and ADI+RDI cases.
    """

    # RDI
    rolls = [0,0,0]
    mode = 'RDI'
    dq_thresh = 1
    mock_sci,mock_ref = create_psfsub_dataset(1,2,rolls)

    mock_ref.all_dq[:,1,1] = 1      # This should become flagged bc all ref data will be flagged
    mock_ref.all_dq[0,55,55] = 1    # This should not become flagged
    mock_sci.all_dq[:,5,5] = 1      # This should become flagged bc all science data will be flagged
    mock_sci.all_dq[:,2,7] = 0.1    # This should not become flagged

    expected_dqs = np.zeros_like(mock_sci.all_dq[0])
    expected_dqs[1,1] = 1
    expected_dqs[5,5] = 1

    dqs, errs = combine.prop_err_dq(mock_sci,mock_ref,mode,dq_thresh)

    # Check shape of output dq & err arrays
    assert dqs.shape == mock_sci.all_dq[0].shape
    assert errs.shape == mock_sci.all_err[0].shape

    # Check values of output dq & error arrays
    assert dqs == pytest.approx(expected_dqs)
    assert np.all(np.isnan(errs))

    # ADI
    rolls = [0,90]
    mode = 'ADI'
    dq_thresh = 4
    mock_sci,mock_ref = create_psfsub_dataset(2,0,rolls)

    mock_sci.all_dq[:,5,5] = 4     # This should become flagged bc all science data after derotation will be flagged
    mock_sci.all_dq[:,94,5] = 4    # This should become flagged bc all science data after derotation will be flagged
    mock_sci.all_dq[:,94,94] = 4    # This should not become flagged
    mock_sci.all_dq[:,18,30] = 4   # This should not become flagged
    mock_sci.all_dq[0,98,1] = 4    # This should not become flagged
    mock_sci.all_dq[:,44,45] = 1   # This should not become flagged
    mock_sci.all_dq[1,98,98] = 4   # This should not become flagged
    
    expected_dqs = np.zeros_like(mock_sci.all_dq[0])
    expected_dqs[5,5] = 1
    expected_dqs[94,5] = 1

    dqs, errs = combine.prop_err_dq(mock_sci,mock_ref,mode,dq_thresh)

    # Check shape of output dq & err arrays
    assert dqs.shape == mock_sci.all_dq[0].shape
    assert errs.shape == mock_sci.all_err[0].shape

    # Check values of output dq & error arrays
    assert dqs == pytest.approx(expected_dqs)
    assert np.all(np.isnan(errs))

    # ADI+RDI
    rolls = [0,90,0,0]
    mode = 'ADI+RDI'
    dq_thresh = 4
    mock_sci,mock_ref = create_psfsub_dataset(2,2,rolls)

    mock_sci.all_dq[:,5,5] = 4     # This should become flagged bc all science data after derotation will be flagged
    mock_sci.all_dq[:,94,5] = 4    # This should become flagged bc all science data after derotation will be flagged
    mock_sci.all_dq[:,94,94] = 4    # This should not become flagged
    mock_sci.all_dq[:,18,30] = 4   # This should not become flagged
    mock_sci.all_dq[0,98,1] = 4    # This should not become flagged
    mock_sci.all_dq[:,44,45] = 1   # This should not become flagged
    mock_sci.all_dq[1,98,98] = 4   # This should not become flagged
    
    mock_ref.all_dq[:,1,1] = 4  # This should become flagged bc all ref data after derotation will be flagged
    mock_ref.all_dq[:,98,1] = 4  # This should become flagged bc all ref data after derotation will be flagged
    mock_ref.all_dq[:,98,98] = 4  # This should not become flagged
    mock_ref.all_dq[:,7,9] = 1  # This should not become flagged
    mock_ref.all_dq[0,3,9] = 4  # This should not become flagged

    expected_dqs = np.zeros_like(mock_sci.all_dq[0])
    expected_dqs[5,5] = 1
    expected_dqs[94,5] = 1
    expected_dqs[1,1] = 1
    expected_dqs[98,1] = 1

    dqs, errs = combine.prop_err_dq(mock_sci,mock_ref,mode,dq_thresh)

    # Check shape of output dq & err arrays
    assert dqs.shape == mock_sci.all_dq[0].shape
    assert errs.shape == mock_sci.all_err[0].shape

    # Check values of output dq & error arrays
    assert dqs == pytest.approx(expected_dqs)
    assert np.all(np.isnan(errs))

    pass


def test_derotate_arr_2d():
    """Tests that 2d arrays can be rotated, 
    tests that the array remains unchanged when rotating by 0 degrees,
    tests that the array can be rotated around positions other than the center.
    """
    test_arr = np.full((10,10),1.)
    test_arr[1,1] = 10.
    test_arr[5,5] = 10.
    test_arr[9,9] = np.nan

    xcen, ycen = 4.5, 4.5

    # Check that the array stays the same if roll angle is 0
    roll_angle = 0
    derotated_arr = combine.derotate_arr(test_arr,roll_angle, xcen,ycen)
    
    assert derotated_arr[~np.isnan(derotated_arr)] == pytest.approx(test_arr[~np.isnan(test_arr)])

    roll_angle = 90
    derotated_arr = combine.derotate_arr(test_arr,roll_angle, xcen,ycen)
    expected_arr = np.ones_like(test_arr)
    expected_arr[1,8] = 10.
    expected_arr[5,4] = 10.
    expected_arr[9,0] = np.nan
    assert derotated_arr[~np.isnan(derotated_arr)] == pytest.approx(expected_arr[~np.isnan(expected_arr)])

    # Check that rotating around a location other than the array center works
    xcen, ycen = 3.5, 3.5

    roll_angle = 90
    derotated_arr = combine.derotate_arr(test_arr,roll_angle, xcen,ycen)
    expected_arr = np.ones_like(test_arr)
    expected_arr[1,6] = 10.
    expected_arr[5,2] = 10.
    expected_arr[:,8:] = np.nan
    assert derotated_arr[~np.isnan(derotated_arr)] == pytest.approx(expected_arr[~np.isnan(expected_arr)])
    
    pass


def test_derotate_arr_nd():
    """Tests that 3D arrays can be derotated, and that dq arrays are treated correctly.
    """
    test_arr = np.full((2,10,10),1.)
    test_arr[:,1,1] = 10.
    test_arr[1,5,5] = 10.
    test_arr[:,9,9] = np.nan

    # Check usual usage
    xcen, ycen = 4.5, 4.5
    roll_angle = 90
    
    expected_arr = np.ones_like(test_arr)
    expected_arr[:,1,8] = 10.
    expected_arr[1,5,4] = 10.
    expected_arr[:,9,0] = np.nan

    derotated_arr = combine.derotate_arr(test_arr,roll_angle, xcen,ycen)
    assert derotated_arr[~np.isnan(derotated_arr)] == pytest.approx(expected_arr[~np.isnan(expected_arr)])

    # Test dq array
    test_dq = np.full((2,10,10),0.)
    test_dq[:,1,1] = 1.
    test_dq[1,5,5] = 1.
    test_dq = test_dq.astype(np.int64)
    
    # Check usual usage
    xcen, ycen = 4.5, 4.5
    roll_angle = 90
    
    expected_dq = np.zeros_like(test_arr)
    expected_dq[:,1,8] = 1.
    expected_dq[1,5,4] = 1.
    expected_dq = expected_dq.astype(int)
    
    derotated_dq = combine.derotate_arr(test_dq,roll_angle, xcen,ycen,
                                        is_dq=True)
    assert derotated_dq == pytest.approx(expected_dq)
    assert derotated_dq.dtype == np.int64

    pass


if __name__ == "__main__":
    test_derotate_arr_2d()
    test_derotate_arr_nd()
    test_prop_err_dq()
