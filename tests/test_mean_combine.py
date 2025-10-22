import pytest

import numpy as np
from corgidrp.darks import mean_combine

ones = np.ones((10, 10))
zeros = np.zeros((10, 10), dtype=int)
vals = [1, 0, 4]
check_ims = []
check_masks = []
one_fr_mask = []
for val in vals:
    check_ims.append(ones*val)
    check_masks.append(zeros.copy())
    one_fr_mask.append(zeros.copy())
    pass

all_masks_i = (0, 0)
single_mask_i = (0, 1)
# Mask one pixel across all frames
for m in check_masks:
    m[all_masks_i[0], all_masks_i[1]] = 1
    pass

# Mask another pixel in only one frame
check_masks[0][single_mask_i[0], single_mask_i[1]] = 1
# Mask one pixel in one frame for one_fr_mask
one_fr_mask[0][single_mask_i[0], single_mask_i[1]] = 1


def test_mean_im():
    """Verify method calculates mean image and error term."""
    tol = 1e-13

    check_combined_im = np.mean(check_ims, axis=0)
    check_combined_im_err = np.sqrt(np.sum(np.array(check_ims)**2,
                                            axis=0)/len(check_ims))
    # For the pixel that is masked throughout
    check_combined_im[all_masks_i] = 0
    check_combined_im_err[all_masks_i] = 0
    unmasked_vals = np.delete(vals, single_mask_i[0])
    # For pixel that is only masked once
    check_combined_im[single_mask_i] = np.mean(unmasked_vals)
    check_combined_im_err[single_mask_i] = \
        np.sqrt(np.sum(unmasked_vals**2)/unmasked_vals.size)

    combined_im, _, _, _ = mean_combine(check_ims, check_masks)
    combined_im_err, _, _, _ = mean_combine(check_ims,
                                            check_masks, err=True)

    assert (np.max(np.abs(combined_im - check_combined_im)) < tol)
    assert (np.max(np.abs(combined_im_err - check_combined_im_err))
                                            < tol)


def test_mean_mask():
    """Verify method calculates correct mean mask."""
    _, combined_mask, _, _ = mean_combine(check_ims, check_masks)
    check_combined_mask = np.zeros_like(combined_mask)
    # Only this pixel should be combined
    check_combined_mask[all_masks_i] = 1

    assert ((combined_mask == check_combined_mask).all())


def test_darks_exception():
    """Half or more of the frames for a given pixel are masked."""
    # all frames for pixel (0,0) masked for inputs below
    _, _, _, enough_for_rn = mean_combine(check_ims, check_masks)
    assert (enough_for_rn == False)


def test_darks_map_im():
    """map_im as expected."""
    _, _, map_im, _  = mean_combine(check_ims,
                                            one_fr_mask)
    # 99 pixels with no mask on any of the 3 frames, one with one
    # frame masked
    # one pixel masked on one frame:
    assert(np.where(map_im == 2)[0].size == 1)
    # 99 pixels not masked on all 3 frames:
    assert(np.where(map_im == 3)[0].size == map_im.size - 1)
    expected_mean_num_good_fr = (3*99 + 2)/100
    assert (np.mean(map_im) == expected_mean_num_good_fr)


def test_invalid_image_list():
    """Invalid inputs caught"""

    bpmap_list = [np.zeros((3, 3), dtype=int), np.eye(3, dtype=int)]

    # for image_list
    perrlist = [
        'txt', None, 1j, 0, (5,), # not list
        (np.ones((3, 3)), np.ones((3, 3))), # not list
        [np.eye(3), np.eye(3), np.eye(3)], # length mismatch
        [np.eye(3)], # length mismatch
        [np.eye(4), np.eye(3)], # array size mismatch
        [np.eye(4), np.eye(4)], # array size mismatch
        [np.ones((1, 3, 3)), np.ones((1, 3, 3))], # not 2D
        ]

    for perr in perrlist:
        with pytest.raises(TypeError):
            mean_combine(perr, bpmap_list)
        pass

    # special empty case
    with pytest.raises(TypeError):
        mean_combine([], [])

    pass


def test_invalid_bpmap_list():
    """Invalid inputs caught"""

    image_list = [np.zeros((3, 3)), np.eye(3)]

    # for image_list
    perrlist = [
        'txt', None, 1j, 0, (5,), # not list
        (np.ones((3, 3), dtype=int), np.ones((3, 3), dtype=int)), #not list
        [np.eye(3, dtype=int), np.eye(3, dtype=int),
            np.eye(3, dtype=int)], # length mismatch
        [np.eye(3, dtype=int)], # length mismatch
        [np.eye(4, dtype=int), np.eye(3, dtype=int)], # array size mismatch
        [np.eye(4, dtype=int), np.eye(4, dtype=int)], # array size mismatch
        [np.ones((1, 3, 3), dtype=int),
            np.ones((1, 3, 3), dtype=int)], # not 2D
        [np.eye(3)*1.0, np.eye(3)*1.0], # not int
        ]

    for perr in perrlist:
        with pytest.raises(TypeError):
            mean_combine(image_list, perr)
        pass
    pass

def test_bpmap_list_element_not_0_or_1():
    """Catch case where bpmap is not 0 or 1"""

    image_list = [np.zeros((3, 3))]
    bpmap_list = [np.zeros((3, 3))]

    for val in [-1, 2]:
        bpmap_list[0][0, 0] = val
        with pytest.raises(TypeError):
            mean_combine(image_list, bpmap_list)
        pass
    pass

def test_accommodation_of_ndarray_inputs():
    """If inputs is a array_like (whether a single frame or a stack),
    the function will accommodate and convert each to a list of arrays."""

    # single frame
    image_list = np.ones((3,3))
    bpmap_list = np.zeros((3,3)).astype(int)
    # runs with no issues:
    mean_combine(image_list, bpmap_list)

    # stack
    image_list = np.stack([np.ones((3,3)), 2*np.ones((3,3))])
    bpmap_list = np.stack([np.zeros((3,3)).astype(int),
                            np.zeros((3,3)).astype(int)])
    # runs with no issues:
    mean_combine(image_list, bpmap_list)

if __name__ == '__main__':
    test_mean_im()
    test_mean_mask()
    test_darks_exception()
    test_darks_map_im()
    test_invalid_image_list()
    test_invalid_bpmap_list()
    test_bpmap_list_element_not_0_or_1()
    test_accommodation_of_ndarray_inputs()





