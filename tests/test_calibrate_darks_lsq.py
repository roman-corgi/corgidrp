import os
import pytest
import pickle
import warnings
import numpy as np

from corgidrp.darks import (calibrate_darks_lsq,
            CalDarksLSQException)
from corgidrp.detector import slice_section, imaging_area_geom, imaging_slice
from corgidrp.mocks import create_synthesized_master_dark_calib
from corgidrp.mocks import detector_areas_test as dat
from corgidrp.data import DetectorNoiseMaps, DetectorParams, Dataset

# make test reproducible
np.random.seed(4567)
# use default parameters
detector_params = DetectorParams({})
# specified parameters simulated in simulated data from
# mocks.create_synthesized_master_dark_calib:
dark_current = 8.33e-4 #e-/pix/s
cic=0.02  # e-/pix/frame
rn=100 # e-/pix/frame
bias=2000 # e-
eperdn = 7 # e-/DN conversion; used in this example for all stacks

EMgain_picks = (np.linspace(2, 5000, 7))
exptime_picks = (np.linspace(2, 100, 7)) # 7x7 = 49 data points
grid = np.meshgrid(EMgain_picks, exptime_picks)
EMgain_arr = grid[0].ravel()
exptime_arr = grid[1].ravel()
kgain_arr = eperdn*np.ones_like(EMgain_arr) # all the same
#added in after emccd_detect makes the frames (see below)
FPN = 21 # e
# number of frames in each sub-stack of stack_arr:
N = 30#600 #30; can also replace with 30 to use those sub-stacks in the
#testdata_small folder; for 30, R_map will be smaller per pixel b/c of big
# variance, and F, C, and D will have more variance with respect to their
# expected values.  No need to test both here;  just N=30 case is sufficient.
# image area, including "shielded" rows and cols:
imrows, imcols, imr0c0 = imaging_area_geom('SCI', dat)

dataset = create_synthesized_master_dark_calib(dat)

# filter out expected warnings
warnings.filterwarnings('ignore', category=UserWarning,
    module='corgidrp.darks')
# filter out expected warning when unreliable_pix_map has a pixel
# masked for all sub-stacks (which makes Rsq NaN from a division by 0)
warnings.filterwarnings('ignore', category=RuntimeWarning,
    module='corgidrp.darks')


def test_expected_results_sub():
    """Outputs are as expected, for smaller-sized frames."""

    noise_maps = calibrate_darks_lsq(dataset, detector_params, dat)
    F_image_map = imaging_slice('SCI', noise_maps.FPN_map, dat)
    # F
    assert (np.isclose(np.mean(F_image_map), FPN//eperdn*eperdn,
                                atol=FPN/5))
    # No FPN was inserted in non-image areas (so that bias subtraction
    #wouldn't simply remove it), so it should be 0 in prescan
    F_prescan_map = slice_section(noise_maps.FPN_map, 'SCI', 'prescan', dat)
    assert(np.isclose(np.nanmean(F_prescan_map), 0, atol=5))
    # C
    assert(np.isclose(np.nanmean(noise_maps.CIC_map), cic, atol=0.01))
    assert(len(noise_maps.CIC_map[noise_maps.CIC_map < 0]) == 0)
    # D
    D_image_map = imaging_slice('SCI', noise_maps.DC_map, dat)
    assert(np.isclose(np.mean(D_image_map),
                    dark_current, atol=2e-4))
    assert(len(noise_maps.DC_map[noise_maps.DC_map < 0]) == 0)
    # D_map: 0 everywhere except image area
    im_rows, im_cols, r0c0 = imaging_area_geom('SCI', dat)
    # D_nonimg = D_map[~D_map[r0c0[0]:r0c0[0]+im_rows,
    #             r0c0[1]:r0c0[1]+im_cols]]
    D_map = noise_maps.DC_map.copy()
    D_map[r0c0[0]:r0c0[0]+im_rows,
                r0c0[1]:r0c0[1]+im_cols] = 0
    # now whole map should be 0
    assert(np.nanmin(D_map) == 0)
    if N == 30:
        # bias_offset (tolerance of 5 for N=30 since I used a small number
        # of frames for that dataset, especially for the high-gain ones)
        assert(np.isclose(noise_maps.bias_offset, 0, atol=5)) #in DN
    if N == 600:
        assert(np.isclose(noise_maps.bias_offset, 0, atol=1)) #in DN
    # dark current only in image area
    D_std_map_im = noise_maps.DC_err[r0c0[0]:r0c0[0]+im_rows,
                                r0c0[1]:r0c0[1]+im_cols]
    # assert that the std dev coming from the fit itself is < noise itself
    assert(np.nanmean(D_std_map_im) < np.nanmean(D_image_map))
    # noise_maps.CIC_err accounts for the err from the input frames and the
    # statistical error (std dev across frames), so skip the assertion for CIC here
    assert(np.nanmean(noise_maps.FPN_err) < np.nanmean(noise_maps.FPN_map))

    # check the noisemap can be pickled (for CTC operations)
    pickled = pickle.dumps(noise_maps)
    pickled_noisemap = pickle.loads(pickled)
    assert np.all((noise_maps.data == pickled_noisemap.data) | np.isnan(noise_maps.data))

    # save noise map
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    nm_filename = noise_maps.filename
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    noise_maps.save(filedir=calibdir, filename=nm_filename)
    nm_filepath = os.path.join(calibdir, nm_filename)
    nm_f = DetectorNoiseMaps(nm_filepath)

    # check the noisemap can be pickled (for CTC operations)
    pickled = pickle.dumps(nm_f)
    pickled_noisemap = pickle.loads(pickled)
    assert np.all((noise_maps.data == pickled_noisemap.data) | np.isnan(noise_maps.data))

    # tests the copy method, from filepath way of creating class
    # instance, too
    nm_open = nm_f.copy()
    assert(np.array_equal(nm_open.data, noise_maps.data, equal_nan=True))

    # check headers
    assert(noise_maps.data.ndim == 3)
    assert('DetectorNoiseMaps' in noise_maps.filename)
    assert(noise_maps.ext_hdr["BUNIT"] == "detected electrons")
    assert(noise_maps.err_hdr["BUNIT"] == "detected electrons")
    assert("DetectorNoiseMaps" in str(noise_maps.ext_hdr["HISTORY"]))
    assert(noise_maps.ext_hdr['B_O_UNIT'] == 'DN')

    assert(nm_open.data.ndim == 3)
    assert('DetectorNoiseMaps' in nm_open.filename)
    assert(nm_open.ext_hdr["BUNIT"] == "detected electrons")
    assert(nm_open.err_hdr["BUNIT"] == "detected electrons")
    assert("DetectorNoiseMaps" in str(nm_open.ext_hdr["HISTORY"]))
    assert(nm_open.ext_hdr['B_O_UNIT'] == 'DN')

def test_sub_stack_len():
    """datasets should have at least 4 sub-stacks."""
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN', 'KGAIN'])
    # make a dataset with only 2 distinct combos
    dataset_few = Dataset([ds[0].frames[k] for k in range(len(ds[0]))] + [ds[1].frames[k] for k in range(len(ds[1]))])
    with pytest.raises(CalDarksLSQException):
        calibrate_darks_lsq(dataset_few, detector_params, dat)

def test_g_arr_unique():
    '''EM gains must have at least 2 unique elements.'''
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN', 'KGAIN'])
    for j in range(len(ds)):
        for d in ds[j].frames:
            d.ext_hdr['CMDGAIN'] = 4
    with pytest.raises(CalDarksLSQException):
        calibrate_darks_lsq(data_set, detector_params, dat)

def test_g_gtr_1():
    '''EM gains must all be bigger than 1.'''
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN', 'KGAIN'])
    ds[0].frames[0].ext_hdr['CMDGAIN'] = 1
    with pytest.raises(CalDarksLSQException):
        calibrate_darks_lsq(data_set, detector_params, dat)

def test_t_arr_unique():
    '''Exposure times must have at least 2 unique elements.'''
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN', 'KGAIN'])
    for j in range(len(ds)):
        for d in ds[j].frames:
            d.ext_hdr['EXPTIME'] = 4
    with pytest.raises(CalDarksLSQException):
        calibrate_darks_lsq(data_set, detector_params, dat)

def test_t_gtr_0():
    '''Exposure times must all be bigger than 0.'''
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN', 'KGAIN'])
    ds[0].frames[0].ext_hdr['EXPTIME'] = 0
    with pytest.raises(CalDarksLSQException):
        calibrate_darks_lsq(data_set, detector_params, dat)


def test_k_gtr_0():
    '''K gains must all be bigger than 0.'''
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN', 'KGAIN'])
    ds[0].frames[0].ext_hdr['KGAIN'] = 0
    with pytest.raises(CalDarksLSQException):
        calibrate_darks_lsq(data_set, detector_params, dat)

def test_mean_num():
    '''If too many masked in a stack for a given pixel, warning raised. Checks
    that dq values are as expected, too.
    '''
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN', 'KGAIN'])
    # tag 48 of the 49 sub-stacks as bad pixel all the
    # way through for one pixel (7,8)
    # And mask (10,12) to get flag value of 256
    for i in range(48):
        ds[i].all_dq[:,7,8] = 4
        ds[i].all_dq[:int(1+len(ds[i])/2),10,12] = 2

    with pytest.warns(UserWarning):
        nm_out = calibrate_darks_lsq(data_set, detector_params, dat)
    # last of out is the DetectorNoiseMaps instance
    # And dq is really a 3-frame stack, and all 3 are the same.  So pick one of them.
    assert nm_out.dq[0,7,8] == 1
    assert nm_out.dq[0,10,12] == 256


if __name__ == '__main__':
    test_mean_num()
    test_expected_results_sub()
    test_sub_stack_len()
    test_g_arr_unique()
    test_g_gtr_1()
    test_t_arr_unique()
    test_t_gtr_0()
    test_k_gtr_0()



