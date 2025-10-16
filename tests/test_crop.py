import numpy as np
import pytest
import warnings
from corgidrp.data import Dataset, Image
from corgidrp.l2b_to_l3 import crop
from corgidrp.mocks import create_default_L3_headers


def make_test_dataset(test_arr,centxy=None,set_cen_kws=True):
    """
    Make 2D or 3D test data.

    Args:
        test_arr (np.array): input frame data array (must be 2D or 3D).
        centxy (arraylike, optional): location of 4 pixel dot. Defaults to center of array.
        set_cen_kws (bool, optional): if false, delete center keywords.

    Returns:
        corgidrp.data.Dataset: test dataset with center related keywords if desired.
    """
    shape = np.array(test_arr.shape)

    if centxy is None:
        cent = np.array(shape)/2 - 0.5
    else:
        cent = [centxy[-i] for i in np.array(range(len(centxy)))+1]
        
    prihdr,exthdr, errhdr, dqhdr = create_default_L3_headers()
    exthdr['LSAM_H'] = cent[1]
    exthdr['LSAM_V'] = cent[0]
    exthdr['LSAMNAME'] = 'NFOV'

    # To test missing keyword case
    if set_cen_kws:
        exthdr['EACQ_COL'] = cent[1]
        exthdr['EACQ_ROW'] = cent[0]
    else:
        del exthdr['EACQ_COL']
        del exthdr['EACQ_ROW']
    
    if test_arr.ndim == 2:
        err_arr = test_arr[np.newaxis,:,:]
    elif test_arr.ndim == 3:
        err_arr = test_arr[np.newaxis,:,:,:]
    
    test_dataset = Dataset([Image(test_arr,prihdr,exthdr,dq=test_arr,err=err_arr)])

    return test_dataset


input_arr_even = np.zeros((100,100))
input_arr_even[49:51,49:51] = 1
goal_arr_even = np.zeros((10,10))
goal_arr_even[4:6,4:6] = 1

input_rect_arr_even = np.zeros((100,200))
input_rect_arr_even[49:51,99:101] = 1
goal_rect_arr_even = np.zeros((10,20))
goal_rect_arr_even[4:6,9:11] = 1

input_rect_arr_odd = np.zeros((101,201))
input_rect_arr_odd[50,100] = 1
goal_rect_arr_odd = np.zeros((11,21))
goal_rect_arr_odd[5,10] = 1

input_rect_arr_mixed = np.zeros((100,201))
input_rect_arr_mixed[49:51,100] = 1
goal_rect_arr_mixed = np.zeros((10,21))
goal_rect_arr_mixed[4:6,10] = 1


def test_2d_square_center_crop():
    """ Test cropping to the center of a square using the header keywords "EACQ_ROW/COL".
    """

    test_dataset = make_test_dataset(input_arr_even,centxy=[49.5,49.5])
    test_dataset[0].ext_hdr["CRPIX1"] = 50.5
    test_dataset[0].ext_hdr["CRPIX2"] = 50.5
    test_dataset[0].ext_hdr["STARLOCX"] = 50.5
    test_dataset[0].ext_hdr["STARLOCY"] = 50.5
    
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=None)

    # Check that data, err, and dq were cropped correctly
    assert cropped_test_dataset[0].data == pytest.approx(goal_arr_even) , "Unexpected result for 2D square crop test."
    assert cropped_test_dataset[0].err[0] == pytest.approx(goal_arr_even), "Unexpected result in err for 2D square crop test."
    assert cropped_test_dataset[0].dq == pytest.approx(goal_arr_even), "Unexpected result for 2D square crop test."
    
    # Test that headers were updated correctly
    assert cropped_test_dataset[0].ext_hdr["EACQ_COL"] == 4.5, "Frame header kw EACQ_COL not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["EACQ_ROW"] == 4.5, "Frame header kw EACQ_ROW not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["CRPIX1"] == 5.5, "Frame header kw CRPIX1 not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["CRPIX2"] == 5.5, "Frame header kw CRPIX2 not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["STARLOCX"] == 5.5, "Frame header kw STARLOCX not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["STARLOCY"] == 5.5, "Frame header kw STARLOCY not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["NAXIS1"] == 10, "Frame header kw NAXIS1 not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["NAXIS2"] == 10, "Frame header kw NAXIS2 not updated correctly."
    assert cropped_test_dataset[0].err_hdr["NAXIS1"] == 10, "Frame err header kw NAXIS1 not updated correctly."
    assert cropped_test_dataset[0].err_hdr["NAXIS2"] == 10, "Frame err header kw NAXIS2 not updated correctly."
    assert cropped_test_dataset[0].dq_hdr["NAXIS1"] == 10, "Frame dq header kw NAXIS1 not updated correctly."
    assert cropped_test_dataset[0].dq_hdr["NAXIS2"] == 10, "Frame dq header kw NAXIS2 not updated correctly."


def test_manual_center_crop():
    """ Test overriding crop location using centerxy argument and make sure 
    DETPIX0X/Y header keyword is updated correctly.
    """

    test_arr = np.zeros((12,12))
    test_arr[5:7,5:7] = 1
    test_dataset = make_test_dataset(test_arr,centxy=[5.5,5.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=[6.5,6.5])

    offset_goal_arr = np.zeros((10,10))
    offset_goal_arr[3:5,3:5] = 1

    expected_detpix_xy = (2,2)

    assert (cropped_test_dataset[0].ext_hdr["DETPIX0X"],
            cropped_test_dataset[0].ext_hdr["DETPIX0Y"]) == expected_detpix_xy, "Extension header DETPIX0X/Y not updated correctly."
    

    assert cropped_test_dataset[0].data == pytest.approx(offset_goal_arr), "Unexpected result for manual crop test."
    assert cropped_test_dataset[0].err[0] == pytest.approx(offset_goal_arr), "Unexpected result in errfor manual crop test."
    assert cropped_test_dataset[0].dq == pytest.approx(offset_goal_arr), "Unexpected result in dq for manual crop test."


def test_2d_square_offcenter_crop():
    """ Test cropping off-center square data.
    """
    test_arr = np.zeros((100,100))
    test_arr[24:26,49:51] = 1
    test_dataset = make_test_dataset(test_arr,centxy=[49.5,24.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=None)

    assert cropped_test_dataset[0].data == pytest.approx(goal_arr_even), "Unexpected result for 2D square offcenter crop test."


def test_2d_rect_offcenter_crop():
    """ Tests cropping off-center non-square data.
    """
    test_arr = np.zeros((100,200))
    test_arr[24:26,49:51] = 1
    test_dataset = make_test_dataset(test_arr,centxy=[49.5,24.5,])
    cropped_test_dataset = crop(test_dataset,sizexy=[20,10],centerxy=None)

    assert cropped_test_dataset[0].data == pytest.approx(goal_rect_arr_even), "Unexpected result for 2D rect offcenter crop test."


def test_3d_rect_offcenter_crop():
    """ Tests cropping 3D off-center non-square data.
    """
    test_arr = np.zeros((3,100,200))
    test_arr[:,24:26,49:51] = 1
    test_dataset = make_test_dataset(test_arr,centxy=[49.5,24.5,])

    test_dataset[0].ext_hdr["CRPIX1"] = 50.5
    test_dataset[0].ext_hdr["CRPIX2"] = 25.5
    cropped_test_dataset = crop(test_dataset,sizexy=[20,10],centerxy=None)

    # Check that data was cropped correctly
    goal_rect_arr3d = np.array([goal_rect_arr_even,goal_rect_arr_even,goal_rect_arr_even])
    assert cropped_test_dataset[0].data == pytest.approx(goal_rect_arr3d), "Unexpected result for 2D rect offcenter crop test."
    assert cropped_test_dataset[0].err[0] == pytest.approx(goal_rect_arr3d), "Unexpected result for 2D rect offcenter crop test."
    assert cropped_test_dataset[0].dq == pytest.approx(goal_rect_arr3d), "Unexpected result for 2D rect offcenter crop test."
        
    # Check that headers were updated correctly
    assert cropped_test_dataset[0].ext_hdr["EACQ_COL"] == 9.5, "Frame header kw EACQ_COL not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["EACQ_ROW"] == 4.5, "Frame header kw EACQ_ROW not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["CRPIX1"] == 10.5, "Frame header kw CRPIX1 not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["CRPIX2"] == 5.5, "Frame header kw CRPIX2 not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["NAXIS1"] == 20, "Frame header kw NAXIS1 not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["NAXIS2"] == 10, "Frame header kw NAXIS2 not updated correctly."
    assert cropped_test_dataset[0].ext_hdr["NAXIS3"] == 3, "Frame header kw NAXIS3 not updated correctly."
    assert cropped_test_dataset[0].dq_hdr["NAXIS1"] == 20, "Frame dq header kw NAXIS1 not updated correctly."
    assert cropped_test_dataset[0].dq_hdr["NAXIS2"] == 10, "Frame dq header kw NAXIS2 not updated correctly."
    assert cropped_test_dataset[0].dq_hdr["NAXIS3"] == 3, "Frame dq header kw NAXIS3 not updated correctly."
    assert cropped_test_dataset[0].err_hdr["NAXIS1"] == 20, "Frame err header kw NAXIS1 not updated correctly."
    assert cropped_test_dataset[0].err_hdr["NAXIS2"] == 10, "Frame err header kw NAXIS2 not updated correctly."
    assert cropped_test_dataset[0].err_hdr["NAXIS3"] == 3, "Frame err header kw NAXIS3 not updated correctly."


def test_edge_of_detector():
    """ Tests that trying to crop a region right at the edge of the 
    detector succeeds.
    """

    test_arr = np.zeros((100,100))
    test_arr[94:96,94:96] = 1.
    test_dataset = make_test_dataset(test_arr,centxy=[94.5,94.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=None)

    assert cropped_test_dataset[0].data == pytest.approx(goal_arr_even), "Unexpected result for edge of FOV crop test."


def test_outside_detector_edge():
    """ Tests cropping to a region partially outside the detector.
    """

    
    # Upper right edge
    test_arr = np.zeros((100,100))
    test_arr[-1,-1] = 1
    test_dataset = make_test_dataset(test_arr)
    goal_arr = np.full((11,11),np.nan)
    goal_arr[:6,:6] = 0
    goal_arr[5,5] = 1
    cropped_test_dataset = crop(test_dataset,sizexy=11,centerxy=[99,99])

    # Replace nans with a finite value for comparison purposes
    cropped_test_dataset[0].data = np.where(np.isnan(cropped_test_dataset[0].data),100.,cropped_test_dataset[0].data)
    goal_arr = np.where(np.isnan(goal_arr),100.,goal_arr)
    assert cropped_test_dataset[0].data == pytest.approx(goal_arr), "Unexpected result for cropping off top-right edge."
    
    # Lower right edge
    test_arr = np.zeros((100,100))
    test_arr[0,-1] = 1
    test_dataset = make_test_dataset(test_arr)
    goal_arr = np.full((11,11),np.nan)
    goal_arr[5:,:6] = 0
    goal_arr[5,5] = 1
    cropped_test_dataset = crop(test_dataset,sizexy=11,centerxy=[99,0])

    # Replace nans with a finite value for comparison purposes
    cropped_test_dataset[0].data = np.where(np.isnan(cropped_test_dataset[0].data),100.,cropped_test_dataset[0].data)
    goal_arr = np.where(np.isnan(goal_arr),100.,goal_arr)
    assert cropped_test_dataset[0].data == pytest.approx(goal_arr), "Unexpected result for cropping off lower-right edge."
    
    # Lower left edge
    test_arr = np.zeros((100,100))
    test_arr[0,0] = 1
    test_dataset = make_test_dataset(test_arr)
    goal_arr = np.full((11,11),np.nan)
    goal_arr[5:,5:] = 0
    goal_arr[5,5] = 1
    cropped_test_dataset = crop(test_dataset,sizexy=11,centerxy=[0,0])

    # Replace nans with a finite value for comparison purposes
    cropped_test_dataset[0].data = np.where(np.isnan(cropped_test_dataset[0].data),100.,cropped_test_dataset[0].data)
    goal_arr = np.where(np.isnan(goal_arr),100.,goal_arr)
    assert cropped_test_dataset[0].data == pytest.approx(goal_arr), "Unexpected result for cropping off lower-left edge."

    # Upper left edge
    test_arr = np.zeros((100,100))
    test_arr[99,0] = 1
    test_dataset = make_test_dataset(test_arr)
    goal_arr = np.full((11,11),np.nan)
    goal_arr[:6,5:] = 0
    goal_arr[5,5] = 1
    cropped_test_dataset = crop(test_dataset,sizexy=11,centerxy=[0,99])

    # Replace nans with a finite value for comparison purposes
    cropped_test_dataset[0].data = np.where(np.isnan(cropped_test_dataset[0].data),100.,cropped_test_dataset[0].data)
    goal_arr = np.where(np.isnan(goal_arr),100.,goal_arr)
    assert cropped_test_dataset[0].data == pytest.approx(goal_arr), "Unexpected result for cropping off top-left edge."
    

    # Zoom out
    test_arr = np.zeros((99,99))
    test_arr[49,49] = 1
    test_dataset = make_test_dataset(test_arr)
    goal_arr = np.full((101,101),np.nan)
    goal_arr[1:100,1:100] = 0
    goal_arr[50,50] = 1
    cropped_test_dataset = crop(test_dataset,sizexy=101,centerxy=[49,49])

    # Replace nans with a finite value for comparison purposes
    cropped_test_dataset[0].data = np.where(np.isnan(cropped_test_dataset[0].data),100.,cropped_test_dataset[0].data)
    goal_arr = np.where(np.isnan(goal_arr),100.,goal_arr)
    assert cropped_test_dataset[0].data == pytest.approx(goal_arr), "Unexpected result for zoom out."


def test_nonhalfinteger_centxy():
    """ Tests that trying to crop data to a center that is not at the intersection 
    of 4 pixels results in centering on the nearest pixel intersection.
    """
    test_dataset = make_test_dataset(input_arr_even,centxy=[49.5,49.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=[49.7,49.7])

    assert cropped_test_dataset[0].data == pytest.approx(goal_arr_even), "Unexpected result for non half-integer crop test."


def test_unsupported_input():
    """ Crop function is not configured for certain observations and should
    fail if the Lyot stop or filter is not the supported positions, unless the desired
    image size is provided manually.
    """

    test_dataset = make_test_dataset(input_arr_even,centxy=[50.5,50.5])
    for frame in test_dataset:
        frame.ext_hdr['LSAMNAME'] = 'UNKNOWN'

    try:
        _ = crop(test_dataset,sizexy=20,centerxy=None)
    except:
        raise ValueError('Unable to determine image size, please change instrument configuration or provide a sizexy value')
    
    
    with pytest.raises(UserWarning):
        _ = crop(test_dataset,sizexy=None,centerxy=None)

def test_noncoron():
    """ Crop function should not automatically crop non-coronagraphic data.
    """

    test_dataset = make_test_dataset(input_arr_even,centxy=[50.5,50.5])
    for frame in test_dataset:
        frame.ext_hdr['LSAMNAME'] = 'OPEN'

    cropped_dataset = crop(test_dataset,sizexy=None,centerxy=None)

    assert cropped_dataset.all_data == pytest.approx(test_dataset.all_data)

def test_detpix0_nonzero():
    """ Tests that the detector pixel header keyword is updated correctly if it 
    already exists and is nonzero.
    """
    test_arr = np.zeros((100,40))
    test_arr[49:51,24:26] = 1
    test_dataset = make_test_dataset(test_arr,centxy=[24.5,49.5])
    test_dataset[0].ext_hdr.set('DETPIX0X',24)
    test_dataset[0].ext_hdr.set('DETPIX0Y',34)

    expected_detpix_xy = (39,79)
    
    cropped_test_dataset = crop(test_dataset,sizexy=[20,10],centerxy=None)

    assert cropped_test_dataset[0].data == pytest.approx(goal_rect_arr_even), "Unexpected result for 2D rect offcenter crop test with nonzero DETPIX0X/Y."
    
    assert (cropped_test_dataset[0].ext_hdr["DETPIX0X"],
            cropped_test_dataset[0].ext_hdr["DETPIX0Y"]) == expected_detpix_xy, "Extension header DETPIX0X/Y not updated correctly."


def test_non_nfov_input():
    '''
    test that for a non nfov input, if no sizexy parameter is inputted, the size
    of the cropped image is calculated correctly
    '''
    test_arr = np.zeros((250,250))
    test_arr[124:126,124:126] = 1
    test_dataset = make_test_dataset(test_arr,centxy=[124.5,124.5])
    #test WFOV band 1
    for frame in test_dataset:
        frame.ext_hdr['LSAMNAME'] = 'WFOV'
        frame.ext_hdr['CFAMNAME'] = '1F'
    cropped_data_WFOV_band_1 = crop(test_dataset)
    assert cropped_data_WFOV_band_1[0].data.shape == (103, 103)

    #test WFOV band 4
    for frame in test_dataset:
        frame.ext_hdr['LSAMNAME'] = 'WFOV'
        frame.ext_hdr['CFAMNAME'] = '4F'
    cropped_data_WFOV_band_4 = crop(test_dataset)
    assert cropped_data_WFOV_band_4[0].data.shape == (143, 143)

    #test SPEC band 2
    for frame in test_dataset:
        frame.ext_hdr['LSAMNAME'] = 'SPEC'
        frame.ext_hdr['CFAMNAME'] = '2F'
    cropped_data_spec_band_2 = crop(test_dataset)
    assert cropped_data_spec_band_2[0].data.shape == (59, 59)

    #test SPEC band 3
    for frame in test_dataset:
        frame.ext_hdr['LSAMNAME'] = 'SPEC'
        frame.ext_hdr['CFAMNAME'] = '3F'
    cropped_data_spec_band_3 = crop(test_dataset)
    assert cropped_data_spec_band_3[0].data.shape == (65, 65)

    #test SPEC with slit/prism
    for frame in test_dataset:
        frame.ext_hdr['FSAMNAME'] = 'R6C5'
    cropped_data_spec_band_3_slit = crop(test_dataset)
    assert cropped_data_spec_band_3_slit[0].data.shape == (125, 125)


def test_default_crop():

    test_arr = input_rect_arr_odd
    test_dataset = make_test_dataset(test_arr)

    cropped_test_dataset = crop(test_dataset)

    goal_arr = np.zeros((61,61))
    goal_arr[30,30] = 1
    assert cropped_test_dataset[0].data == pytest.approx(goal_arr), "Unexpected result for default crop test."


def test_mixed_oddeven_crop():

    test_dataset = make_test_dataset(input_rect_arr_mixed)

    cropped_test_dataset = crop(test_dataset,sizexy=[21,10])

    assert cropped_test_dataset[0].data == pytest.approx(goal_rect_arr_mixed), "Unexpected result for mixed odd-even size crop test."


def test_missing_cen_kws():
    "Crop function should fail if center keywords do not exist and centerxy is not provided."
    test_dataset = make_test_dataset(input_arr_even,centxy=[49.5,49.5],set_cen_kws=False)
    
    with pytest.raises(ValueError):
        _ = crop(test_dataset,sizexy=10,centerxy=None)


if __name__ == "__main__":
    test_2d_square_center_crop()
    test_manual_center_crop()
    test_2d_square_offcenter_crop()
    test_2d_rect_offcenter_crop()
    test_3d_rect_offcenter_crop()
    test_edge_of_detector()
    test_outside_detector_edge()
    test_nonhalfinteger_centxy()
    test_non_nfov_input()
    test_detpix0_nonzero()
    test_unsupported_input()
    test_noncoron()
    test_default_crop()
    test_mixed_oddeven_crop()
    test_missing_cen_kws()
