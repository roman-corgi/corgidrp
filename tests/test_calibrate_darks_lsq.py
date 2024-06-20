import os
import unittest
import warnings
import numpy as np
from pathlib import Path

from corgidrp.calibrate_darks_lsq import (calibrate_darks_lsq,
            CalDarksLSQException)
from corgidrp.detector import Metadata
from corgidrp.mocks import create_synthesized_master_dark_calib
from corgidrp.data import NoiseMap

here = Path(os.path.dirname(os.path.abspath(__file__)))
one_up = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
meta_path = Path(one_up, 'corgidrp', 'util', 'metadata.yaml')
meta_path_sub = Path(one_up, 'corgidrp', 'util', 'metadata_test.yaml')
meta = Metadata(meta_path)
meta_sub = Metadata(meta_path_sub)

# specified parameters used in simulated data from
# mocks.create_synthesized_master_dark_calib:
fwc_em_e = 90000 #e-
fwc_pp_e = 50000 #e-
dark_current = 8.33e-4 #e-/pix/s
cic=0.02  # e-/pix/frame
rn=100 # e-/pix/frame
bias=2000 # e-
eperdn = 7 # e-/DN conversion; used in this example for all stacks

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
# expected values.  No need to test both here;  just N=30 case is sufficient.
# image area, including "shielded" rows and cols:
imrows, imcols, imr0c0 = meta_sub._imaging_area_geom()


class TestCalibrateDarksLSQ(unittest.TestCase):
    """Unit tests for calibrate_darks_lsq method."""
    def setUp(self):
        self.meta_path_sub = meta_path_sub
        self.datasets = create_synthesized_master_dark_calib()

        # filter out expected warnings
        warnings.filterwarnings('ignore', category=UserWarning,
            module='corgidrp.calibrate_darks_lsq')
        # filter out expected warning when unreliable_pix_map has a pixel
        # masked for all sub-stacks (which makes Rsq NaN from a division by 0)
        warnings.filterwarnings('ignore', category=RuntimeWarning,
            module='corgidrp.calibrate_darks_lsq')

    def test_expected_results_sub(self):
        """Outputs are as expected, for smaller-sized frames."""

        (F_map, C_map, D_map, bias_offset, bias_offset_up, bias_offset_low,
         F_image_map, C_image_map,
            D_image_map, Fvar, Cvar, Dvar, read_noise, R_map, F_image_mean,
            C_image_mean, D_image_mean, unreliable_pix_map, F_std_map,
            C_std_map, D_std_map, stacks_err, F_noise_map, C_noise_map,
            D_noise_map) = \
                calibrate_darks_lsq(self.datasets, self.meta_path_sub)
        # F
        self.assertTrue(np.isclose(np.mean(F_image_map), FPN//eperdn*eperdn,
                                   atol=FPN/5))
        self.assertTrue(np.isclose(F_image_mean, FPN//eperdn*eperdn,
                                   atol=FPN/5))
        # No FPN was inserted in non-image areas (so that bias subtraction
        #wouldn't simply remove it), so it should be 0 in prescan
        F_prescan_map = meta_sub.slice_section(F_map, 'prescan')
        self.assertTrue(np.isclose(np.nanmean(F_prescan_map), 0, atol=5))
        # C
        self.assertTrue(np.isclose(np.nanmean(C_map), cic, atol=0.01))
        self.assertTrue(np.isclose(np.mean(C_image_map), cic, atol=0.01))
        self.assertTrue(np.isclose(C_image_mean, cic, atol=0.01))
        self.assertTrue(len(C_map[C_map < 0]) == 0)
        self.assertTrue(len(C_image_map[C_image_map < 0]) == 0)
        # D
        self.assertTrue(np.isclose(np.mean(D_image_map),
                        dark_current, atol=2e-4))
        self.assertTrue(len(D_map[D_map < 0]) == 0)
        self.assertTrue(len(D_image_map[D_image_map < 0]) == 0)
        self.assertTrue(np.isclose(D_image_mean, dark_current, atol=2e-4))
        # D_map: 0 everywhere except image area
        im_rows, im_cols, r0c0 = meta_sub._imaging_area_geom()
        # D_nonimg = D_map[~D_map[r0c0[0]:r0c0[0]+im_rows,
        #             r0c0[1]:r0c0[1]+im_cols]]
        D_map[r0c0[0]:r0c0[0]+im_rows,
                    r0c0[1]:r0c0[1]+im_cols] = 0
        # now whole map should be 0
        self.assertTrue(np.nanmin(D_map) == 0)
        # read_noise
        self.assertTrue(np.isclose(read_noise, rn, rtol=0.1))
        # adjusted R^2:  acceptable fit (the higher N is, the better the fit)
        if N == 30:
            # bias_offset (tolerance of 5 for N=30 since I used a small number
            # of frames for that dataset, especially for the high-gain ones)
            self.assertTrue(np.isclose(bias_offset, 0, atol=5)) #in DN
            self.assertTrue(np.nanmean(R_map) > 0.1)
        if N == 600:
            self.assertTrue(np.isclose(bias_offset, 0, atol=1)) #in DN
            self.assertTrue(np.nanmean(R_map) > 0.7)
        # dark current only in image area
        D_std_map_im = D_std_map[r0c0[0]:r0c0[0]+im_rows,
                                 r0c0[1]:r0c0[1]+im_cols]
        # assert that the std dev coming from the fit itself is < noise itself
        self.assertTrue(np.nanmean(D_std_map_im) < np.nanmean(D_image_map))
        self.assertTrue(np.nanmean(C_std_map) < np.nanmean(C_map))
        self.assertTrue(np.nanmean(F_std_map) < np.nanmean(F_map))


        ntypes = ['FPN', 'CIC', 'DC']
        nmaps = [F_noise_map, C_noise_map, D_noise_map]
        for nm in nmaps:
            # save noise map
            calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
            nm_filename = nm.filename
            if not os.path.exists(calibdir):
                os.mkdir(calibdir)
            nm.save(filedir=calibdir, filename=nm_filename)
            nm_filepath = os.path.join(calibdir, nm_filename)
            ind = nmaps.index(nm)
            nm_f = NoiseMap(nm_filepath, noise_type=ntypes[ind])
            # tests the copy method, from filepath way of creating class
            # instance, too
            nm_open = nm_f.copy()
            self.assertTrue(np.array_equal(nm_open.data, nm.data, equal_nan=True))

            # check headers
            self.assertTrue(ntypes[ind] in nm.filename)
            self.assertTrue(nm.ext_hdr["BUNIT"] == "detected electrons")
            self.assertTrue(nm.err_hdr["BUNIT"] == "detected electrons")
            self.assertTrue("NoiseMap" in str(nm.ext_hdr["HISTORY"]))

            self.assertTrue(ntypes[ind] in nm_open.filename)
            self.assertTrue(nm_open.ext_hdr["BUNIT"] == "detected electrons")
            self.assertTrue(nm_open.err_hdr["BUNIT"] == "detected electrons")
            self.assertTrue("NoiseMap" in str(nm_open.ext_hdr["HISTORY"]))

    def test_sub_stack_len(self):
        """datasets should have at least 4 sub-stacks."""
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(self.datasets[0:2])

    def test_g_arr_unique(self):
        '''EM gains must have at least 2 unique elements.'''
        ds = self.datasets.copy()
        for j in range(len(ds)):
            for d in ds[j].frames:
                d.ext_hdr['CMDGAIN'] = 4
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(ds, meta_path_sub)

    def test_g_gtr_1(self):
        '''EM gains must all be bigger than 1.'''
        ds = self.datasets.copy()
        ds[0].frames[0].ext_hdr['CMDGAIN'] = 1
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(ds, meta_path_sub)

    def test_t_arr_unique(self):
        '''Exposure times must have at least 2 unique elements.'''
        ds = self.datasets.copy()
        for j in range(len(ds)):
            for d in ds[j].frames:
                d.ext_hdr['EXPTIME'] = 4
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(ds, meta_path_sub)

    def test_t_gtr_0(self):
        '''Exposure times must all be bigger than 0.'''
        ds = self.datasets.copy()
        ds[0].frames[0].ext_hdr['EXPTIME'] = 0
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(ds, meta_path_sub)


    def test_k_gtr_0(self):
        '''K gains must all be bigger than 0.'''
        ds = self.datasets.copy()
        ds[0].frames[0].ext_hdr['KGAIN'] = 0
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(ds, meta_path_sub)

    def test_unique_g_t_combos(self):
        '''The EM gain and frame time combos for the sub-stacks must be
        unique.'''
        ds = self.datasets.copy()
        for j in [1,2]:
            for d in ds[j].frames:
                d.ext_hdr['EXPTIME'] = 4
                d.ext_hdr['CMDGAIN'] = 5
        with self.assertRaises(CalDarksLSQException):
            calibrate_darks_lsq(ds, meta_path_sub)

    def test_mean_num(self):
        '''If too many masked in a stack for a given pixel, exception raised.
        '''
        ds = self.datasets.copy()
        # tag 48 of the 49 sub-stacks with cosmic-affected pixels all the
        # way through for one pixel (7,8)
        for i in range(48):
            ds[i].all_dq[:,7,8] = 1
        with self.assertWarns(UserWarning):
            calibrate_darks_lsq(ds, meta_path_sub)

if __name__ == '__main__':
    unittest.main()

