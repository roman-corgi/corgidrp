"""
Unit test suite for the calibrate_darks module.

Data was generated for test in the if __name__ == '__main__' part of
calibrate_darks_lsq.py.  Smaller frames (for faster running) is in
the testdata_small.
There is a set of stacks in the testdata_small folder for N=30 in each
sub-stack (useful for fast-running tests, but the fits give an average adjusted
R^2 of about 0.19) and N=600 (useful for getting a higher average adjusted
R^2 of about 0.75).
"""

import os
import unittest
from pathlib import Path
import warnings
import numpy as np

from cal.util.read_metadata import Metadata as MetadataWrapper
import cal.util.ut_check as ut_check
from cal.calibrate_darks.calibrate_darks_lsq import (calibrate_darks_lsq,
            CalDarksLSQException)

here = Path(os.path.dirname(os.path.abspath(__file__)))

meta_path = Path(here,'..', 'util', 'metadata.yaml')
meta_path_sub = Path(here,'..', 'util', 'metadata_test.yaml')
meta = MetadataWrapper(meta_path)
meta_sub = MetadataWrapper(meta_path_sub)
# for fluxmap
# im_rows, im_cols, im_r0c0 = \
#     meta._unpack_geom('image')
nonlin_path = Path(here, '..', 'util', 'nonlin_sample.csv')

#for raw dark, photons/s
# fluxmap = np.zeros((im_rows, im_cols))
# specified parameters; data created in the
# if __name__ == '__main__' part of calibrate_darks_lsq.py
fwc_em_e = 90000 #e-
fwc_pp_e = 50000 #e-
# dark_current = 8.33e-4 #e-/pix/s
# cic=0.02  # e-/pix/frame
# read_noise=100 # e-/pix/frame
# bias=2000 # e-
# qe=0.9  # quantum efficiency, e-/photon
# cr_rate=0.  # hits/cm^2/s
# pixel_pitch=13e-6  # m
eperdn = 7 # e-/DN conversion; used in this example for all stacks
# nbits=64 # number of ADU bits
Nem=604 #number of gain register elements

g_picks = (np.linspace(2, 5000, 7))
t_picks = (np.linspace(2, 100, 7)) # 7x7 = 49 data points
grid = np.meshgrid(g_picks, t_picks)
g_arr = grid[0].ravel()
t_arr = grid[1].ravel()
k_arr = eperdn*np.ones_like(g_arr) # all the same
#added in after emccd_detect makes the frames (see below)
FPN = 21 # e
# number of frames in each sub-stack of stack_arr:
N = 30#600 #30; can also replace with 30 to use those sub-stacks in the
#testdata_small folder; for 30, R_map will be smaller per pixel b/c of big
# variance, and F, C, and D will have more variance with respect to their
# expected values.  No need to test both here;  just N=600 case is sufficient.
# image area, including "shielded" rows and cols:
imrows, imcols, imr0c0 = meta_sub._imaging_area_geom()

# load in test data for sub-frame
stack_list_sub = []
test_data_path_sub = Path(here, 'testdata_small')
for i in range(len(g_arr)):
    load = os.path.join(test_data_path_sub, 'g_'+str(int(g_arr[i]))+'_t_'+
    str(int(t_arr[i]))+'_N_'+str(N)+'stack.npy')
    stack_list_sub.append(np.load(load))
    # simulate a constant FPN in image area (not in prescan
    # so that it isn't removed when bias is removed)
    stack_list_sub[i] = stack_list_sub[i].astype('float64')
    stack_list_sub[i][:,imr0c0[0]:imr0c0[0]+imrows,imr0c0[1]:
               imr0c0[1]+imcols] += FPN/k_arr[i] # in DN

stack_arr_sub_fr = np.stack(stack_list_sub)
# simulate telemetry rows, with the last 5 column entries with high counts
stack_arr_sub_fr[:,:,-4:,-5] = 100000 #DN
# to make more like actual frames from detector
stack_arr_sub_fr = stack_arr_sub_fr.astype('uint16')

class TestCalibrateDarksLSQ(unittest.TestCase):
    """Unit tests for calibrate_darks_lsq method."""
    def setUp(self):
        self.meta_path_sub = meta_path_sub
        self.nonlin_path = nonlin_path

        self.fwc_em_e = fwc_em_e
        self.fwc_pp_e = fwc_pp_e
        self.Nem=Nem

        self.g_arr = g_arr
        self.t_arr = t_arr
        self.k_arr = k_arr
        self.N = N

        # filter out expected warnings
        warnings.filterwarnings('ignore', category=UserWarning,
            module='cal.calibrate_darks.calibrate_darks_lsq')

    def test_expected_results_sub(self):
        """Outputs are as expected, for smaller-sized frames."""
        nonlin_path = Path(here, '..', 'util', 'testdata',
                'ut_nonlin_array_ones.txt') # does no corrections
        (F_map, C_map, D_map, bias_offset,
        F_image_map, C_image_map, D_image_map,
        Fvar, Cvar, Dvar, read_noise, R_map,
        F_image_mean, C_image_mean, D_image_mean) = \
        calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, self.t_arr,
            self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
            nonlin_path, self.Nem)
        # F
        self.assertTrue(np.isclose(np.mean(F_image_map), FPN//eperdn*eperdn, 
                                   atol=FPN/10))
        self.assertTrue(np.isclose(F_image_mean, FPN//eperdn*eperdn, 
                                   atol=FPN/10))
        # No FPN was inserted in non-image areas (so that bias subtraction
        #wouldn't simply remove it), so it should be 0 in prescan
        F_prescan_map = meta_sub.slice_section(F_map, 'prescan')
        self.assertTrue(np.isclose(np.nanmean(F_prescan_map), 0, atol=2))
        # C
        self.assertTrue(np.isclose(np.nanmean(C_map), 0.02, atol=0.01))
        self.assertTrue(np.isclose(np.mean(C_image_map), 0.02, atol=0.01))
        self.assertTrue(np.isclose(C_image_mean, 0.02, atol=0.01))
        self.assertTrue(len(C_map[C_map < 0]) == 0)
        self.assertTrue(len(C_image_map[C_image_map < 0]) == 0)
        # D
        self.assertTrue(np.isclose(np.mean(D_image_map),
                        8.33e-4, atol=2e-4))
        self.assertTrue(len(D_map[D_map < 0]) == 0)
        self.assertTrue(len(D_image_map[D_image_map < 0]) == 0)
        self.assertTrue(np.isclose(D_image_mean, 8.33e-4, atol=2e-4))
        # D_map: 0 everywhere except image area
        im_rows, im_cols, r0c0 = meta_sub._imaging_area_geom()
        # D_nonimg = D_map[~D_map[r0c0[0]:r0c0[0]+im_rows,
        #             r0c0[1]:r0c0[1]+im_cols]]
        D_map[r0c0[0]:r0c0[0]+im_rows,
                    r0c0[1]:r0c0[1]+im_cols] = 0
        # now whole map should be 0
        self.assertTrue(np.nanmin(D_map) == 0)
        # read_noise
        self.assertTrue(np.isclose(read_noise, 100, rtol=0.1))
        # adjusted R^2:  acceptable fit (the higher N is, the better the fit)
        if N == 30:
            # bias_offset (tolerance of 5 for N=30 since I used a small number
            # of frames for that dataset, especially for the high-gain ones)
            self.assertTrue(np.isclose(bias_offset, 0, atol=5)) #in DN
            self.assertTrue(np.nanmean(R_map) > 0.1)
        if N == 600:
            self.assertTrue(np.isclose(bias_offset, 0, atol=1)) #in DN
            self.assertTrue(np.nanmean(R_map) > 0.7)

    def test_telem_rows_success(self):
        '''Successful run with telem_rows specified.'''
        calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, self.t_arr,
            self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
            nonlin_path, self.Nem, telem_rows=slice(-5,120))

    def test_4D(self):
        """stack_arr should be 4-D."""
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack_arr_sub_fr[0], self.g_arr, self.t_arr,
            self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
            self.nonlin_path, self.Nem)

    def test_sub_stack_len(self):
        """stack_arr should have at least 4 sub-stacks."""
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack_arr_sub_fr[0:2], self.g_arr,
            self.t_arr, self.k_arr, self.fwc_em_e, self.fwc_pp_e,
            self.meta_path_sub, self.nonlin_path, self.Nem)

    def test_g_arr(self):
        """g_arr should be a 1-D, real array."""
        for perr in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                calibrate_darks_lsq(stack_arr_sub_fr, perr, self.t_arr,
                    self.k_arr, self.fwc_em_e, self.fwc_pp_e,
                    self.meta_path_sub, self.nonlin_path, self.Nem)
        for perr in ut_check.rarraylist:
            with self.assertRaises(TypeError):
                calibrate_darks_lsq(stack_arr_sub_fr, perr, self.t_arr,
                    self.k_arr, self.fwc_em_e, self.fwc_pp_e,
                    self.meta_path_sub, self.nonlin_path, self.Nem)

    def test_g_arr_unique(self):
        '''g_arr must have at least 2 unique elements.'''
        g_arr = 2*np.ones(len(stack_arr_sub_fr))
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack_arr_sub_fr, g_arr, self.t_arr,
                self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, self.Nem)

    def test_g_arr_len(self):
        '''g_arr and stack_arr must have the same length.'''
        g_arr = np.array([2,3,4,5,6])
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack_arr_sub_fr, g_arr, self.t_arr,
                self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, self.Nem)

    def test_g_gtr_1(self):
        '''g_arr elements must all be bigger than 1.'''
        g_arr = np.arange(1,31) #same length as stack_arr, but includes 1
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack_arr_sub_fr, g_arr, self.t_arr,
                self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, self.Nem)

    def test_t_arr(self):
        """t_arr should be a 1-D, real array."""
        for perr in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, perr,
                    self.k_arr, self.fwc_em_e, self.fwc_pp_e,
                    self.meta_path_sub, self.nonlin_path, self.Nem)
        for perr in ut_check.rarraylist:
            with self.assertRaises(TypeError):
                calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, perr,
                    self.k_arr, self.fwc_em_e, self.fwc_pp_e,
                    self.meta_path_sub, self.nonlin_path, self.Nem)

    def test_t_arr_unique(self):
        '''t_arr must have at least 2 unique elements.'''
        t_arr = 2*np.ones(len(stack_arr_sub_fr))
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, t_arr,
                self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, self.Nem)

    def test_t_arr_len(self):
        '''t_arr and stack_arr must have the same length.'''
        t_arr = np.array([2,3,4,5,6])
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, t_arr,
                self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, self.Nem)

    def test_t_gtr_0(self):
        '''t_arr elements must all be bigger than 0.'''
        t_arr = np.arange(0,30) #same length as stack_arr, but contains 0
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, t_arr,
                self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, self.Nem)

    def test_k_arr(self):
        """k_arr should be a 1-D, real array."""
        for perr in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, self.t_arr,
                    perr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                    self.nonlin_path, self.Nem)
        for perr in ut_check.rarraylist:
            with self.assertRaises(TypeError):
                calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, self.t_arr,
                    perr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                    self.nonlin_path, self.Nem)

    def test_k_arr_len(self):
        '''k_arr and stack_arr must have the same length.'''
        k_arr = np.array([2,3,4,5,6])
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, self.t_arr,
                k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, self.Nem)

    def test_k_gtr_0(self):
        '''k_arr elements must all be bigger than 0.'''
        k_arr = np.arange(0,30) #same length as stack_arr, but contains 0
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, self.t_arr,
                k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, self.Nem)

    def test_unique_g_t_combos(self):
        '''The EM gain and frame time combos for the sub-stacks must be
        unique.'''
        g_arr = np.arange(2,32) # same length as stack_arr
        t_arr = np.arange(2,32) # same length as stack_arr
        t_arr[0] = 31 # same as last of t_arr
        g_arr[0] = 31 # same as last of g_arr; so two of the same pairs
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack_arr_sub_fr, g_arr, t_arr,
                self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, self.Nem)

    def test_psi(self):
        """These three below must be positive scalar integers."""
        check_list = ut_check.psilist
        # fwc_em_e
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, self.t_arr,
                self.k_arr, perr, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, self.Nem)

        # fwc_pp_e
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, self.t_arr,
                self.k_arr, self.fwc_em_e, perr, self.meta_path_sub,
                self.nonlin_path, self.Nem)

        # Nem
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, self.t_arr,
                self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, perr)

    def test_telem_rows(self):
        '''telem_rows must be a slice or None.'''
        with self.assertRaises(TypeError):
            calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, self.t_arr,
                self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                nonlin_path, self.Nem, telem_rows=(-3,-1))

    def test_telem_rows_image(self):
        '''telem_rows cannot specify rows in image area.'''
        with self.assertRaises(ValueError):
            calibrate_darks_lsq(stack_arr_sub_fr, self.g_arr, self.t_arr,
                self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                nonlin_path, self.Nem, telem_rows=slice(90,91))

    def test_mean_num(self):
        '''If too many masked in a stack for a given pixel, exception raised.
        '''
        stack = np.random.randint(0, 200, size=(3,3,
                                    meta_sub.frame_rows, meta_sub.frame_cols))
        # now tag 2 of the 3 sub-stacks with cosmic-affected pixels all the
        # way through for one pixel
        stack[0][0:2,0,0] = self.fwc_pp_e/self.k_arr[0]
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(stack, self.g_arr, self.t_arr,
                self.k_arr, self.fwc_em_e, self.fwc_pp_e, self.meta_path_sub,
                self.nonlin_path, self.Nem)




if __name__ == '__main__':
    unittest.main()

