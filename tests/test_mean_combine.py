import unittest

import numpy as np
from corgidrp.mean_combine import mean_combine

class TestMeanCombine(unittest.TestCase):
    """Unit tests for mean_combine method."""

    def setUp(self):
        ones = np.ones((10, 10))
        zeros = np.zeros((10, 10), dtype=int)
        self.vals = [1, 0, 4]
        self.check_ims = []
        self.check_masks = []
        self.one_fr_mask = []
        for val in self.vals:
            self.check_ims.append(ones*val)
            self.check_masks.append(zeros.copy())
            self.one_fr_mask.append(zeros.copy())
            pass

        self.all_masks_i = (0, 0)
        self.single_mask_i = (0, 1)
        # Mask one pixel across all frames
        for m in self.check_masks:
            m[self.all_masks_i[0], self.all_masks_i[1]] = 1
            pass

        # Mask another pixel in only one frame
        self.check_masks[0][self.single_mask_i[0], self.single_mask_i[1]] = 1
        # Mask one pixel in one frame for one_fr_mask
        self.one_fr_mask[0][self.single_mask_i[0], self.single_mask_i[1]] = 1


    def test_mean_im(self):
        """Verify method calculates mean image and erorr term."""
        tol = 1e-13

        check_combined_im = np.mean(self.check_ims, axis=0)
        check_combined_im_err = np.sqrt(np.sum(np.array(self.check_ims)**2,
                                               axis=0))/len(self.check_ims)
        # For the pixel that is masked throughout
        check_combined_im[self.all_masks_i] = 0
        check_combined_im_err[self.all_masks_i] = 0
        unmasked_vals = np.delete(self.vals, self.single_mask_i[0])
        # For pixel that is only masked once
        check_combined_im[self.single_mask_i] = np.mean(unmasked_vals)
        check_combined_im_err[self.single_mask_i] = \
            np.sqrt(np.sum(unmasked_vals**2))/unmasked_vals.size

        combined_im, _, _, _ = mean_combine(self.check_ims, self.check_masks)
        combined_im_err, _, _, _ = mean_combine(self.check_ims,
                                                self.check_masks, err=True)

        self.assertTrue(np.max(np.abs(combined_im - check_combined_im)) < tol)
        self.assertTrue(np.max(np.abs(combined_im_err - check_combined_im_err))
                                                < tol)


    def test_mean_mask(self):
        """Verify method calculates correct mean mask."""
        _, combined_mask, _, _ = mean_combine(self.check_ims, self.check_masks)
        check_combined_mask = np.zeros_like(combined_mask)
        # Only this pixel should be combined
        check_combined_mask[self.all_masks_i] = 1

        self.assertTrue((combined_mask == check_combined_mask).all())


    def test_darks_exception(self):
        """Half or more of the frames for a given pixel are masked."""
        # all frames for pixel (0,0) masked for inputs below
        _, _, _, enough_for_rn = mean_combine(self.check_ims, self.check_masks)
        self.assertTrue(enough_for_rn == False)


    def test_darks_map_im(self):
        """map_im as expected."""
        _, _, map_im, _  = mean_combine(self.check_ims,
                                                self.one_fr_mask)
        # 99 pixels with no mask on any of the 3 frames, one with one
        # frame masked
        # one pixel masked on one frame:
        self.assertTrue(np.where(map_im == 2)[0].size == 1)
        # 99 pixels not masked on all 3 frames:
        self.assertTrue(np.where(map_im == 3)[0].size == map_im.size - 1)
        expected_mean_num_good_fr = (3*99 + 2)/100
        self.assertEqual(np.mean(map_im), expected_mean_num_good_fr)


    def test_invalid_image_list(self):
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
            with self.assertRaises(TypeError):
                mean_combine(perr, bpmap_list)
            pass

        # special empty case
        with self.assertRaises(TypeError):
            mean_combine([], [])

        pass


    def test_invalid_bpmap_list(self):
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
            with self.assertRaises(TypeError):
                mean_combine(image_list, perr)
            pass
        pass

    def test_bpmap_list_element_not_0_or_1(self):
        """Catch case where bpmap is not 0 or 1"""

        image_list = [np.zeros((3, 3))]
        bpmap_list = [np.zeros((3, 3))]

        for val in [-1, 2]:
            bpmap_list[0][0, 0] = val
            with self.assertRaises(TypeError):
                mean_combine(image_list, bpmap_list)
            pass
        pass

    def test_accommodation_of_ndarray_inputs(self):
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
    unittest.main()