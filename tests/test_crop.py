import numpy as np
import pytest
from corgidrp.data import Dataset, Image
from corgidrp.l3_to_l4 import crop
from corgidrp.mocks import create_default_headers

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
        
    prihdr,exthdr = create_default_headers()
    exthdr['STARLOCX'] = cent[1]
    exthdr['STARLOCY'] = cent[0]
    
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
    """ Test overriding crop location using centerxy argument.
    """

    test_dataset = make_test_dataset(shape=[100,100],centxy=[49.5,49.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=[50.5,50.5])

    offset_goal_arr = np.zeros((10,10))
    offset_goal_arr[3:5,3:5] = 1

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
    """ Test cropping off-center non-square data.
    """
    test_dataset = make_test_dataset(shape=[100,40],centxy=[24.5,49.5])
    cropped_test_dataset = crop(test_dataset,sizexy=[20,10],centerxy=None)

    if not cropped_test_dataset[0].data == pytest.approx(goal_rect_arr):
        raise Exception("Unexpected result for 2D rect offcenter crop test.")

def test_3d_rect_offcenter_crop():
    """ Test cropping 3D off-center non-square data.
    """
    test_dataset = make_test_dataset(shape=[3,100,40],centxy=[24.5,49.5])
    cropped_test_dataset = crop(test_dataset,sizexy=[20,10],centerxy=None)

    goal_rect_arr3d = np.array([goal_rect_arr,goal_rect_arr,goal_rect_arr])

    if not cropped_test_dataset[0].data == pytest.approx(goal_rect_arr3d):
        raise Exception("Unexpected result for 2D rect offcenter crop test.")


def test_edge_of_FOV():
    """ Test cropping right at the edge of the data array.
    """
    test_dataset = make_test_dataset(shape=[100,100],centxy=[94.5,94.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=None)

    if not cropped_test_dataset[0].data == pytest.approx(goal_arr):
        raise Exception("Unexpected result for edge of FOV crop test.")

def test_outside_FOV():
    """ Test cropping over the edge of the data array.
    """

    test_dataset = make_test_dataset(shape=[100,100],centxy=[95.5,95.5])

    with pytest.raises(ValueError):
        _ = crop(test_dataset,sizexy=10,centerxy=None)

def test_nonhalfinteger_centxy():
    """ Test trying to center the crop not on a pixel intersection.
    """
    test_dataset = make_test_dataset(shape=[100,100],centxy=[49.5,49.5])
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=[49.7,49.7])

    if not cropped_test_dataset[0].data == pytest.approx(goal_arr):
        raise Exception("Unexpected result for non half-integer crop test.")

def test_header_updates():
    """ Test that the header values are updated correctly.
    """
    
    test_dataset = make_test_dataset(shape=[100,100],centxy=[49.5,49.5])
    test_dataset[0].ext_hdr["MASKLOCX"] = 49.5
    test_dataset[0].ext_hdr["MASKLOCY"] = 49.5
    test_dataset[0].pri_hdr["PSFCENTX"] = 49.5
    test_dataset[0].pri_hdr["PSFCENTY"] = 49.5
    test_dataset[0].pri_hdr["CRPIX1"] = 50.5
    test_dataset[0].pri_hdr["CRPIX2"] = 50.5
    
    cropped_test_dataset = crop(test_dataset,sizexy=10,centerxy=None)

    if not cropped_test_dataset[0].ext_hdr["STARLOCX"] == 4.5:
        raise Exception("Frame header kw STARLOCX not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["STARLOCY"] == 4.5:
        raise Exception("Frame header kw STARLOCY not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["MASKLOCX"] == 4.5:
        raise Exception("Frame header kw MASKLOCX not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["MASKLOCY"] == 4.5:
        raise Exception("Frame header kw MASKLOCY not updated correctly.")
    if not cropped_test_dataset[0].pri_hdr["PSFCENTX"] == 4.5:
        raise Exception("Frame header kw PSFCENTX not updated correctly.")
    if not cropped_test_dataset[0].pri_hdr["PSFCENTY"] == 4.5:
        raise Exception("Frame header kw PSFCENTY not updated correctly.")
    if not cropped_test_dataset[0].pri_hdr["CRPIX1"] == 5.5:
        raise Exception("Frame header kw CRPIX1 not updated correctly.")
    if not cropped_test_dataset[0].pri_hdr["CRPIX2"] == 5.5:
        raise Exception("Frame header kw CRPIX2 not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["NAXIS1"] == 10:
        raise Exception("Frame header kw NAXIS1 not updated correctly.")
    if not cropped_test_dataset[0].ext_hdr["NAXIS2"] == 10:
        raise Exception("Frame header kw NAXIS2 not updated correctly.")
    

if __name__ == "__main__":
    test_2d_square_center_crop()
    test_2d_square_offcenter_crop()
    test_2d_rect_offcenter_crop()
    test_edge_of_FOV()
    test_outside_FOV()
    test_nonhalfinteger_centxy()
    test_header_updates()
