import numpy as np
import pytest
import warnings
from corgidrp.data import Dataset, Image
from corgidrp.l3_to_l4 import crop
from corgidrp.mocks import create_default_L3_headers

def make_test_dataset(shape=[100,100],centxy=None):
    """
    Make 2D or 3D test data.

    Args:
        shape (arraylike, optional): data shape. Defaults to [100,100].
        centxy (arraylike,optional): location of 4 pixel dot. Defaults to center of array.

    Returns:
        corgidrp.data.Dataset: test data with a 2x2 "PSF" at location centxy.
    """
    shape = np.array(shape)

    test_arr = np.zeros(shape)
    if centxy is None:
        cent = np.array(shape)/2 - 0.5
    else:
        cent = [centxy[-i] for i in np.array(range(len(centxy)))+1]
        
    prihdr,exthdr, errhdr, dqhdr = create_default_L3_headers()
    exthdr['STARLOCX'] = cent[1]
    exthdr['STARLOCY'] = cent[0]
    exthdr['LSAM_H'] = cent[1]
    exthdr['LSAM_V'] = cent[0]
    exthdr['LSAMNAME'] = 'NFOV'
    
    if len(shape) == 2:
        test_arr[int(cent[0]-0.5):int(cent[0]+1.5),int(cent[1]-0.5):int(cent[1]+1.5)] = 1

    elif len(shape) == 3:
        test_arr[:,int(cent[0]-0.5):int(cent[0]+1.5),int(cent[1]-0.5):int(cent[1]+1.5)] = 1

    test_dataset = Dataset([Image(test_arr,prihdr,exthdr)])

    return test_dataset

goal_arr = np.zeros((10,10))
goal_arr[4:6,4:6] = 1

goal_rect_arr = np.zeros((10,20))
goal_rect_arr[4:6,9:11] = 1

def test_2d_square_center_crop():
    """ Test cropping to the center of a square using the header keywords "STARLOCX/Y".
    """

    test_dataset = make_test_dataset(shape=[100,100],centxy=[49.5,49.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=None)

    if not cropped_test_dataset[0].data == pytest.approx(goal_arr):
        raise Exception("Unexpected result for 2D square crop test.")

def test_manual_center_crop():
    """ Test overriding crop location using centerxy argument and make sure 
    DETPIX0X/Y header keyword is updated correctly.
    """

    test_dataset = make_test_dataset(shape=[12,12],centxy=[5.5,5.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=[6.5,6.5])

    offset_goal_arr = np.zeros((10,10))
    offset_goal_arr[3:5,3:5] = 1

    expected_detpix_xy = (2,2)

    if not (cropped_test_dataset[0].ext_hdr["DETPIX0X"],
            cropped_test_dataset[0].ext_hdr["DETPIX0Y"]) == expected_detpix_xy:
        raise Exception("Extension header DETPIX0X/Y not updated correctly.")
    

    if not cropped_test_dataset[0].data == pytest.approx(offset_goal_arr):
        raise Exception("Unexpected result for manual crop test.")

def test_2d_square_offcenter_crop():
    """ Test cropping off-center square data.
    """

    test_dataset = make_test_dataset(shape=[100,100],centxy=[24.5,49.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=None)

    if not cropped_test_dataset[0].data == pytest.approx(goal_arr):
        raise Exception("Unexpected result for 2D square offcenter crop test.")

def test_2d_rect_offcenter_crop():
    """ Tests cropping off-center non-square data.
    """
    test_dataset = make_test_dataset(shape=[100,40],centxy=[24.5,49.5])
    cropped_test_dataset = crop(test_dataset,sizexy=[20,10],centerxy=None)

    if not cropped_test_dataset[0].data == pytest.approx(goal_rect_arr):
        raise Exception("Unexpected result for 2D rect offcenter crop test.")

def test_3d_rect_offcenter_crop():
    """ Tests cropping 3D off-center non-square data.
    """
    test_dataset = make_test_dataset(shape=[3,100,40],centxy=[24.5,49.5])
    cropped_test_dataset = crop(test_dataset,sizexy=[20,10],centerxy=None)

    goal_rect_arr3d = np.array([goal_rect_arr,goal_rect_arr,goal_rect_arr])

    if not cropped_test_dataset[0].data == pytest.approx(goal_rect_arr3d):
        raise Exception("Unexpected result for 2D rect offcenter crop test.")

def test_edge_of_detector():
    """ Tests that trying to crop a region right at the edge of the 
    detector succeeds.
    """
    test_dataset = make_test_dataset(shape=[100,100],centxy=[94.5,94.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=None)

    if not cropped_test_dataset[0].data == pytest.approx(goal_arr):
        raise Exception("Unexpected result for edge of FOV crop test.")

def test_outside_detector_edge():
    """ Tests that trying to crop a region outside the detector fails.
    """

    test_dataset = make_test_dataset(shape=[100,100],centxy=[95.5,95.5])

    with pytest.raises(ValueError):
        _ = crop(test_dataset,sizexy=10,centerxy=None)

def test_nonhalfinteger_centxy():
    """ Tests that trying to crop data to a center that is not at the intersection 
    of 4 pixels results in centering on the nearest pixel intersection.
    """
    test_dataset = make_test_dataset(shape=[100,100],centxy=[49.5,49.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=[49.7,49.7])

    if not cropped_test_dataset[0].data == pytest.approx(goal_arr):
        raise Exception("Unexpected result for non half-integer crop test.")

def test_header_updates_2d():
    """ Tests that cropping works, and updates header values related to 
    pixel positions and data shapes correctly, for 3D data.
    """
    
    test_dataset = make_test_dataset(shape=[100,100],centxy=[49.5,49.5])
    test_dataset[0].ext_hdr["STARLOCX"] = 49.5
    test_dataset[0].ext_hdr["STARLOCY"] = 49.5
    test_dataset[0].pri_hdr["CRPIX1"] = 50.5
    test_dataset[0].pri_hdr["CRPIX2"] = 50.5
    
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=None)

    if not cropped_test_dataset[0].ext_hdr["STARLOCX"] == 4.5:
        raise Exception("Frame header kw STARLOCX not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["STARLOCY"] == 4.5:
        raise Exception("Frame header kw STARLOCY not updated correctly.")
    if not cropped_test_dataset[0].pri_hdr["CRPIX1"] == 5.5:
        raise Exception("Frame header kw CRPIX1 not updated correctly.")
    if not cropped_test_dataset[0].pri_hdr["CRPIX2"] == 5.5:
        raise Exception("Frame header kw CRPIX2 not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["NAXIS1"] == 10:
        raise Exception("Frame header kw NAXIS1 not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["NAXIS2"] == 10:
        raise Exception("Frame header kw NAXIS2 not updated correctly.")
    if not cropped_test_dataset[0].err_hdr["NAXIS1"] == 10:
        raise Exception("Frame err header kw NAXIS1 not updated correctly.")
    if not cropped_test_dataset[0].err_hdr["NAXIS2"] == 10:
        raise Exception("Frame err header kw NAXIS2 not updated correctly.")
    if not cropped_test_dataset[0].dq_hdr["NAXIS1"] == 10:
        raise Exception("Frame dq header kw NAXIS1 not updated correctly.")
    if not cropped_test_dataset[0].dq_hdr["NAXIS2"] == 10:
        raise Exception("Frame dq header kw NAXIS2 not updated correctly.")
    
def test_header_updates_3d():
    """ Tests that cropping works, and updates header values related to pixel 
    positions and data shapes correctly, for 3D data.
    """
    
    test_dataset = make_test_dataset(shape=[3,100,100],centxy=[49.5,49.5])
    test_dataset[0].ext_hdr["STARLOCX"] = 49.5
    test_dataset[0].ext_hdr["STARLOCY"] = 49.5
    test_dataset[0].pri_hdr["CRPIX1"] = 50.5
    test_dataset[0].pri_hdr["CRPIX2"] = 50.5
    
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=None)

    if not cropped_test_dataset[0].ext_hdr["STARLOCX"] == 4.5:
        raise Exception("Frame header kw STARLOCX not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["STARLOCY"] == 4.5:
        raise Exception("Frame header kw STARLOCY not updated correctly.")
    if not cropped_test_dataset[0].pri_hdr["CRPIX1"] == 5.5:
        raise Exception("Frame header kw CRPIX1 not updated correctly.")
    if not cropped_test_dataset[0].pri_hdr["CRPIX2"] == 5.5:
        raise Exception("Frame header kw CRPIX2 not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["NAXIS1"] == 10:
        raise Exception("Frame header kw NAXIS1 not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["NAXIS2"] == 10:
        raise Exception("Frame header kw NAXIS2 not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["NAXIS3"] == 3:
        raise Exception("Frame header kw NAXIS3 not updated correctly.")
    if not cropped_test_dataset[0].dq_hdr["NAXIS1"] == 10:
        raise Exception("Frame dq header kw NAXIS1 not updated correctly.")
    if not cropped_test_dataset[0].dq_hdr["NAXIS2"] == 10:
        raise Exception("Frame dq header kw NAXIS2 not updated correctly.")
    if not cropped_test_dataset[0].dq_hdr["NAXIS3"] == 3:
        raise Exception("Frame dq header kw NAXIS3 not updated correctly.")
    if not cropped_test_dataset[0].err_hdr["NAXIS1"] == 10:
        raise Exception("Frame err header kw NAXIS1 not updated correctly.")
    if not cropped_test_dataset[0].err_hdr["NAXIS2"] == 10:
        raise Exception("Frame err header kw NAXIS2 not updated correctly.")
    if not cropped_test_dataset[0].err_hdr["NAXIS3"] == 3:
        raise Exception("Frame err header kw NAXIS3 not updated correctly.")
    
def test_non_nfov_input():
    """ Crop function is not configured for non-NFOV observations and should
    fail if the Lyot stop is not in the narrow FOV position, unless the desired
    image size is provided manually.
    """

    test_dataset = make_test_dataset(shape=[100,100],centxy=[50.5,50.5])
    for frame in test_dataset:
        frame.ext_hdr['LSAMNAME'] = 'WFOV'

    try:
        _ = crop(test_dataset,sizexy=20,centerxy=None)
    except:
        raise ValueError('Cropping a non-NFOV observation failed even though sizexy was provided')
    
    
    with pytest.raises(UserWarning):
        _ = crop(test_dataset,sizexy=None,centerxy=None)

def test_detpix0_nonzero():
    """ Tests that the detector pixel header keyword is updated correctly if it 
    already exists and is nonzero.
    """
    test_dataset = make_test_dataset(shape=[100,40],centxy=[24.5,49.5])
    test_dataset[0].ext_hdr.set('DETPIX0X',24)
    test_dataset[0].ext_hdr.set('DETPIX0Y',34)

    expected_detpix_xy = (39,79)
    
    cropped_test_dataset = crop(test_dataset,sizexy=[20,10],centerxy=None)

    if not cropped_test_dataset[0].data == pytest.approx(goal_rect_arr):
        raise Exception("Unexpected result for 2D rect offcenter crop test with nonzero DETPIX0X/Y.")
    
    if not (cropped_test_dataset[0].ext_hdr["DETPIX0X"],
            cropped_test_dataset[0].ext_hdr["DETPIX0Y"]) == expected_detpix_xy:
        raise Exception("Extension header DETPIX0X/Y not updated correctly.")
    

if __name__ == "__main__":
    # test_2d_square_center_crop()
    test_manual_center_crop()
    # test_2d_square_offcenter_crop()
    # test_2d_rect_offcenter_crop()
    # test_3d_rect_offcenter_crop()
    # test_edge_of_detector()
    # test_outside_detector_edge()
    # test_nonhalfinteger_centxy()
    # test_header_updates_2d()
    # test_header_updates_3d()
    # test_non_nfov_input()
    test_detpix0_nonzero()

