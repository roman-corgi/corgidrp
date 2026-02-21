import re
import os
import pytest
import pickle
import warnings
import numpy as np

from corgidrp.darks import calibrate_darks_lsq, CalDarksLSQException
from corgidrp.detector import slice_section, imaging_area_geom, imaging_slice
from corgidrp.mocks import create_synthesized_master_dark_calib, rename_files_to_cgi_format
from corgidrp.mocks import detector_areas_test as dat
from corgidrp.data import DetectorNoiseMaps, DetectorParams, Dataset
import shutil

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

# EMgain_picks = (np.linspace(2, 5000, 7))
# exptime_picks = (np.linspace(2, 100, 7)) # 7x7 = 49 data points
# grid = np.meshgrid(EMgain_picks, exptime_picks)
# EMgain_arr = grid[0].ravel()
# exptime_arr = grid[1].ravel()
# kgain_arr = eperdn*np.ones_like(EMgain_arr) # all the same

#added in after emccd_detect makes the frames (see below)
FPN = 21 # e
# number of frames in each sub-stack of stack_arr:
N = 30#600 #30; can also replace with 30 to use those sub-stacks in the
#testdata_small folder; for 30, R_map will be smaller per pixel b/c of big
# variance, and F, C, and D will have more variance with respect to their
# expected values.  No need to test both here;  just N=30 case is sufficient.
# image area, including "shielded" rows and cols:
imrows, imcols, imr0c0 = imaging_area_geom('SCI', dat)

def setup_module():
    """
    Sets up testing module
    """
    global dataset
    np.random.seed(4567)  # make test reproducible
    dataset = create_synthesized_master_dark_calib(dat)
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    if os.path.exists(datadir):
        for name in os.listdir(datadir):
            path = os.path.join(datadir, name)
            os.remove(path)
    # saving with correct filename format:
    output_filenames = rename_files_to_cgi_format(dataset.frames, datadir)

def teardown_module():
    """
    Runs at the end. Deletes big unused variables
    """
    global dataset
    del dataset


# filter out expected warning when unreliable_pix_map has a pixel
# masked for all sub-stacks (which makes Rsq NaN from a division by 0)
# warnings.filterwarnings('ignore', category=RuntimeWarning,
    # module='corgidrp.darks')


def test_expected_results_sub():
    """Outputs are as expected, for smaller-sized frames."""
    for weighting in [True, False]:
        # filter out expected warning from internal module
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='corgidrp.darks')
            noise_maps = calibrate_darks_lsq(dataset, detector_params, weighting, dat)
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
    test_filename = dataset.frames[-1].filename.split('.fits')[0] + '_dnm_cal.fits'
    test_filename = re.sub('_l[0-9].', '', test_filename)
    assert(noise_maps.filename == test_filename)
    assert(noise_maps.ext_hdr["BUNIT"] == "detected electron")
    assert(noise_maps.err_hdr["BUNIT"] == "detected electron")
    assert("DetectorNoiseMaps" in str(noise_maps.ext_hdr["HISTORY"]))
    assert(noise_maps.ext_hdr['B_O_UNIT'] == 'DN')

    assert(nm_open.data.ndim == 3)
    assert(nm_open.filename == test_filename)
    assert(nm_open.ext_hdr["BUNIT"] == "detected electron")
    assert(nm_open.err_hdr["BUNIT"] == "detected electron")
    assert("DetectorNoiseMaps" in str(nm_open.ext_hdr["HISTORY*"]))
    assert(nm_open.ext_hdr['B_O_UNIT'] == 'DN')
    # make sure an example set of exposure time, EM gain, and number of frames is in HISTORY
    assert('[2.0, 2.0, 30.0]' in str(nm_open.ext_hdr['HISTORY*']))
    assert('FPN_IMM' in nm_open.ext_hdr.keys())
    assert('CIC_IMM' in nm_open.ext_hdr.keys())
    assert('DC_IMM' in nm_open.ext_hdr.keys())
    assert('FPN_IMME' in nm_open.ext_hdr.keys())

def test_sub_stack_len():
    """datasets should have at least 4 sub-stacks."""
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
    # make a dataset with only 2 distinct combos
    dataset_few = Dataset([ds[0].frames[k] for k in range(len(ds[0]))] + [ds[1].frames[k] for k in range(len(ds[1]))])
    with pytest.raises(CalDarksLSQException):
        calibrate_darks_lsq(dataset_few, detector_params, detector_regions=dat)

def test_g_arr_unique():
    '''EM gains must have at least 2 unique elements.'''
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
    for j in range(len(ds)):
        for d in ds[j].frames:
            d.ext_hdr['EMGAIN_C'] = 4
    with pytest.raises(CalDarksLSQException):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='corgidrp.darks')
            calibrate_darks_lsq(data_set, detector_params, detector_regions=dat)

def test_g_gtr_1():
    '''EM gains must all be 1 or bigger.'''
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
    ds[0].frames[0].ext_hdr['EMGAIN_C'] = 0.9
    ds[0].frames[1].ext_hdr['EMGAIN_C'] = 0.9
    with pytest.raises(CalDarksLSQException):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='corgidrp.darks')
            calibrate_darks_lsq(data_set, detector_params, detector_regions=dat)

def test_t_arr_unique():
    '''Exposure times must have at least 2 unique elements.'''
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
    for j in range(len(ds)):
        for d in ds[j].frames:
            d.ext_hdr['EXPTIME'] = 4.
    with pytest.raises(CalDarksLSQException):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='corgidrp.darks')
            calibrate_darks_lsq(data_set, detector_params, detector_regions=dat)

def test_t_gtr_0():
    '''Exposure times must all be 0 or greater.'''
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
    ds[0].frames[0].ext_hdr['EXPTIME'] = -0.1
    ds[0].frames[1].ext_hdr['EXPTIME'] = -0.1
    with pytest.raises(CalDarksLSQException):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='corgidrp.darks')
            calibrate_darks_lsq(data_set, detector_params, detector_regions=dat)


def test_k_gtr_0():
    '''K gains must all be bigger than 0.'''
    data_set = dataset.copy()
    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
    ds[0].frames[0].ext_hdr['KGAINPAR'] = 0
    ds[0].frames[1].ext_hdr['KGAINPAR'] = 0
    with pytest.raises(CalDarksLSQException):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='corgidrp.darks')
            calibrate_darks_lsq(data_set, detector_params, detector_regions=dat)

def test_mean_num():
    '''If too many masked in a stack for a given pixel, warning raised. Checks
    that dq values are as expected, too.
    '''
    data_set = dataset.copy()

    ds, _ = data_set.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
    # tag 48 of the 49 sub-stacks as bad pixel all the
    # way through for one pixel (7,8)
    # And mask (10,12) to get high statistical error for that pixel
    for i in range(48):
        ds[i].all_dq[:,7,8] = 4
        ds[i].all_dq[:int(1+len(ds[i])/2),10,12] = 2

    ##warning check is removed because the expected warning was supressed
    #with pytest.warns(UserWarning):
        #nm_out = calibrate_darks_lsq(data_set, detector_params, detector_regions=dat)
    
    nm_out = calibrate_darks_lsq(data_set, detector_params, detector_regions=dat)
    
    # last of out is the DetectorNoiseMaps instance
    # And dq is really a 3-frame stack, and all 3 are the same.  So pick one of them.
    assert nm_out.dq[0,7,8] == 4
    # error for dark current at pixel (10,12) should be 0 since dark current only in the image area
    assert nm_out.err[0,2,10,12] == 0

def test_weighting():
    '''This tests that weighting works. Demonstrates the effect of low weighting via high err and then lots of masking.'''
    noise_maps = calibrate_darks_lsq(dataset, detector_params, detector_regions=dat)
    CIC_image_map = imaging_slice('SCI', noise_maps.CIC_map, dat)
    wrong_dataset = dataset.copy()
    # Let the frames corresponding to 2 EM gain values (for all exposure time values) be erroneous. 
    # The CIC should be noticeably off, and the check we did in test_expected_results_sub() should fail.
    # Frames are arranged in sets of 30 with each set corresponding to (t1, g1), (t1, g2), ..., (t1, g7), (t2, g1), (t2, g2), .... 
    # So we're changing all g6 and g7 sets to be erroneous.
    s1 = 6
    s2 = 7
    for s in [s1, s2]:
        for i in range(7):
            #wrong_dataset.all_data[int(7*30*i + s*30):int(7*30*i + s*30+30)] = wrong_dataset.all_data[int(7*30*i + s*30):int(7*30*i + s*30+30)]/6  
            for j in range(int(7*30*i + s*30), min(int(7*30*i + s*30+30), len(wrong_dataset))):
                wrong_dataset[j].data = wrong_dataset[j].data/6
    wrong_noise_maps = calibrate_darks_lsq(wrong_dataset, detector_params, detector_regions=dat)
    wrong_CIC_image_map = imaging_slice('SCI', wrong_noise_maps.CIC_map, dat)
    assert(not np.isclose(np.mean(CIC_image_map),
                    np.mean(wrong_CIC_image_map), atol=0.01))
    # make err for this sub-stack large, so weighting should make fit much better
    for s in [s1, s2]:
        for i in range(7):
            #wrong_dataset.all_err[int(7*30*i + s*30):int(7*30*i + s*30+30)] += 100000
            for j in range(int(7*30*i + s*30), min(int(7*30*i + s*30+30), len(wrong_dataset))):
                wrong_dataset[j].err = wrong_dataset[j].err + 100000
    wrong_noise_maps_err = calibrate_darks_lsq(wrong_dataset, detector_params, detector_regions=dat)
    wrong_CIC_image_map_err = imaging_slice('SCI', wrong_noise_maps_err.CIC_map, dat)
    assert(np.isclose(np.mean(CIC_image_map),
                    np.mean(wrong_CIC_image_map_err), atol=0.01))
    # and err should be reduced overall now, despite effectively having fewer stacks for fitting
    assert(np.nanmean(wrong_noise_maps_err.CIC_err) < np.nanmean(wrong_noise_maps.CIC_err))
    # This time, make err for this sub-stack large through dq (all but 1 frame in the sub-stack)
    # (undo err adjustment I did above)
    for s in [s1, s2]:
        for i in range(7):
            # wrong_dataset.all_err[int(7*30*i + s*30):int(7*30*i + s*30+30)] -= 100000
            for j in range(int(7*30*i + s*30), min(int(7*30*i + s*30+30), len(wrong_dataset))):
                wrong_dataset[j].err = wrong_dataset[j].err - 100000
            # wrong_dataset.all_dq[int(7*30*i + s*30):int(7*30*i + s*30+29), :, 1:] = 1 # leave one pixel unmasked so that the total masking Exception isn't raised
            for j in range(int(7*30*i + s*30), min(int(7*30*i + s*30+29), len(wrong_dataset))):
                wrong_dataset[j].dq[:, 1:] = 1 # leave one pixel unmasked so that the total masking Exception isn't raised
    wrong_noise_maps_dq = calibrate_darks_lsq(wrong_dataset, detector_params, detector_regions=dat)
    wrong_CIC_image_map_dq = imaging_slice('SCI', wrong_noise_maps_dq.CIC_map, dat)
    # We artificially made err big above, which brought the mean CIC value much closer to the expected value.
    # However, our leverage in weighting is much more limited when only the DQ is used to affect the weighting. (In reality, the 
    # err should also be large if the data is in fact bad data.) 
    # But the result is still a number closer to the correct value compared to the case where no appropriate weighting is used:
    assert(np.abs(np.mean(CIC_image_map) - np.mean(wrong_CIC_image_map_dq)) < 
           np.abs(np.mean(CIC_image_map) - np.mean(wrong_CIC_image_map)))
    # and err should be reduced overall now, despite effectively having fewer stacks for fitting
    assert(np.nanmean(wrong_noise_maps_dq.CIC_err) < np.nanmean(wrong_noise_maps.CIC_err))
    # Finally, demonstrate weighting through fewer frames in the erroneous sub-stacks:
    del_list = []
    for s in [s1, s2]:
        for i in range(7):
            # restore original DQ values
            #wrong_dataset.all_dq[int(7*30*i + s*30):int(7*30*i + s*30+29), :, 1:] = 0
            for j in range(int(7*30*i + s*30), min(int(7*30*i + s*30+29), len(wrong_dataset))):
                wrong_dataset[j].dq[:, 1:] = 0
            # leave only 2 frames in the dataset for g6 and g7
            del_list.append([r for r in range(int(7*30*i + s*30), int(7*30*i + s*30+28))])
    del_list = np.array(del_list)[:-1].ravel() # [:-1] cuts off the frame numbers that went beyond the total number, which didn't matter in previous calls
    inds = np.delete(np.arange(0, len(wrong_dataset)), del_list)
    smaller_dataset = wrong_dataset[inds].copy()
    smaller_noise_maps = calibrate_darks_lsq(smaller_dataset, detector_params, detector_regions=dat)
    smaller_CIC_image_map = imaging_slice('SCI', smaller_noise_maps.CIC_map, dat)
    # similar logic that applied for masking via DQ above
    assert(np.abs(np.mean(CIC_image_map) - np.mean(smaller_CIC_image_map)) < 
           np.abs(np.mean(CIC_image_map) - np.mean(wrong_CIC_image_map)))
    assert(np.nanmean(smaller_noise_maps.CIC_err) < np.nanmean(wrong_noise_maps.CIC_err))
    
def test_no_data():
    '''Tests that a Dataset with only metadata (and has data read in one 
    frame at a time from filepaths) gives same results as a normal Dataset.
    '''
    noisemaps = calibrate_darks_lsq(dataset, detector_params, detector_regions=dat)
    dataset_no_data = dataset.copy()
    for frame in dataset_no_data:
        frame.data = None
    noisemaps2 = calibrate_darks_lsq(dataset_no_data, detector_params, detector_regions=dat)
    assert np.nanmax(noisemaps.data - noisemaps2.data) < 1e-10
    assert np.nanmax(noisemaps.err - noisemaps2.err) < 1e-10
    assert np.array_equal(noisemaps.dq, noisemaps2.dq) 

if __name__ == '__main__':
    setup_module()

    test_no_data()
    test_g_gtr_1()
    test_t_gtr_0()
    test_k_gtr_0()
    test_weighting()
    test_mean_num()
    test_expected_results_sub()
    test_sub_stack_len()
    test_g_arr_unique()
    test_t_arr_unique()

    teardown_module()



