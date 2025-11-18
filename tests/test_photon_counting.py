import pytest
import os
import astropy.time as time
import corgidrp.mocks as mocks
import corgidrp
from astropy.io import fits
import numpy as np
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.data as data
import corgidrp.detector as detector
from corgidrp.photon_counting import get_pc_mean, photon_count, PhotonCountException
import io, contextlib

def test_negative():
    """Values at or below the 
    threshold are photon-counted as 0, including negative values."""
    fr = np.array([-3,2,0])
    pc_fr = photon_count(fr, thresh=2)
    assert (pc_fr == np.array([0,0,0])).all()
    

def test_pc():
    '''
    Tests that a pixel that is masked heavily (reducing the number of usable frames for that pixel) has a bigger err than the average pixel.

    Tests that if masking is all the way through for a certain pixel, that pixel will 
    be flagged in the DQ. 
    
    Make sure there is failure when the number of iterations niter<1.

    Tests that the use of a mask to mask out bright pixels when photon-counting, to ensure failure for bright pixels outside region of interest when no mask used
    and success when mask used to mask those bright pixels.

    Tests safemode, which makes the function issue warnings instead of crashing (useful for HOWFSC's iterative digging of dark holes).

    Checks various inputs: threshold<0 gives exception, thresh>=em_gain gives exception, providing dataset of VISTYPE='CGIVST_CAL_DRK' while also inputting a photon-counted master dark raises exception, exception raised when 'EMGAIN_C' not the same for all frames.

    Checks various metadata changes: the expected filename and history edit for the output is done (for case of dark subtraction and the case of no dark subtraction), the master dark is indicated as such via a header keyword 'PC_STAT'.

    Tests that output average value of the photon-counted frames agrees with what was simulated (for the case of dark subtraction and no dark subtraction).

    Tests that if 'ISPC' is in image header and if it is False, get_pc_mean() raises an exception.

    Tests that an exception occurs if the user input "inputmode" is inconsistent with the input dataset type.

    Tests that an exception occurs if the photon-counted master dark does not have the header 'PCTHRESH'.

    Tests that an exception occurs if the photon-counted master dark has a 'PCTHRESH' value than the one to be used on the illuminated dataset.
    '''
    np.random.seed(555)
    dataset_err, dark_dataset_err, ill_mean, dark_mean = mocks.create_photon_countable_frames(Nbrights=160, Ndarks=160, cosmic_rate=0, full_frame=False, smear=False) 
    # instead of running through walker, just do the pre-processing steps simply
    # using EM gain=5000 and kgain=7 and bias=20000 and read noise = 100 and QE=0.9 (quantum efficiency), from mocks.create_photon_countable_frames()
    for f in dataset_err.frames:
        f.data = f.data.astype(float)*7 - 20000.
    for f in dark_dataset_err.frames:
        f.data = f.data.astype(float)*7 - 20000.
    dataset_err.all_data = dataset_err.all_data.astype(float)*7 - 20000.
    dark_dataset_err.all_data = dark_dataset_err.all_data.astype(float)*7 - 20000.
    # masked for half of the bright and half of the dark frames
    dataset_err.all_dq[80:,3,3] = 1
    dark_dataset_err.all_dq[80:,3,3] = 1
    dataset_err.all_dq[:, 2, 2] = 1 #masked all the way through for (2,2)
    for f in dataset_err.frames: 
        f.ext_hdr['RN'] = 100
        f.ext_hdr['KGAINPAR'] = 7
    for f in dark_dataset_err.frames: 
        f.ext_hdr['RN'] = 100
        f.ext_hdr['KGAINPAR'] = 7
    # process the frames to make PC dark
    dark_dataset_err[0].ext_hdr['HISTORY'] = '' # define a history value since get_pc_mean() uses it
    pc_dark = get_pc_mean(dark_dataset_err, inputmode='darks')
    assert pc_dark.ext_hdr['PC_STAT'] == 'photon-counted master dark'
    # now process illuminated frames and subtract the PC dark
    dataset_err[0].ext_hdr['HISTORY'] = '' # define a history value since get_pc_mean() uses it
    pc_dataset_err = get_pc_mean(dataset_err, pc_master_dark=pc_dark)

    assert pc_dataset_err.frames[-1].filename == dataset_err[-1].filename

    history = ''
    for line in pc_dataset_err.frames[0].ext_hdr["HISTORY"]:
        history += line
    assert "Dark-subtracted with PC dark: yes" in history
    assert np.isclose(pc_dataset_err.all_data.mean(), ill_mean - dark_mean, rtol=0.01) 
    # the DQ for that pixel should be 1
    assert pc_dataset_err[0].dq[2,2] == 1
    # err for (3,3) is above the 95th percentile of error: 
    assert pc_dataset_err[0].err[0][3,3]>np.nanpercentile(pc_dataset_err[0].err,95)

    # also when niter<1, exception
    with pytest.raises(PhotonCountException):
        get_pc_mean(dataset_err, niter=0)
    # when thresh<0, exception
    with pytest.raises(PhotonCountException):
        get_pc_mean(dataset_err, T_factor=-1, niter=2)
    # when thresh>=em_gain, exception
    with pytest.raises(Exception):
        get_pc_mean(dataset_err, T_factor=50, niter=2)
    # can't provide master dark while "VISTYPE" of input dataset is 'CGIVST_CAL_DRK'
    with pytest.raises(PhotonCountException):
        get_pc_mean(dark_dataset_err, pc_master_dark=pc_dark, inputmode='darks')
    # must have same 'EMGAIN_C' and other header values throughout input dataset
    dark_dataset_err.frames[0].ext_hdr['EMGAIN_C'] = 4999
    with pytest.raises(PhotonCountException):
        get_pc_mean(dark_dataset_err, inputmode='dark')
    # change back:
    dark_dataset_err.frames[0].ext_hdr['EMGAIN_C'] = 5000

    # test to make sure PC dark's threshold matches the one used for illuminated frames 
    with pytest.raises(PhotonCountException):
        pc_dark.ext_hdr['PCTHRESH'] = 499
        get_pc_mean(dataset_err, pc_master_dark=pc_dark, inputmode='illuminated')
    # PC dark should have header 'PCTHRESH'
    with pytest.raises(PhotonCountException):
        pc_dark.ext_hdr.pop("PCTHRESH")
        get_pc_mean(dataset_err, pc_master_dark=pc_dark, inputmode='illuminated')
    # set it back:
    pc_dark.ext_hdr['PCTHRESH'] = 500
    # to trigger pc_ecount_max error
    for f in dataset_err.frames[:-2]: #all but 2 of the frames
        f.data[22:40,23:49] = np.abs(f.data.astype(float)[22:40,23:49]*3000)
    dataset_err.all_data[:-2,22:40,23:49] = np.abs(dataset_err.all_data.astype(float)[:-2,22:40,23:49]*3000)
    with pytest.raises(Exception):
        get_pc_mean(dataset_err, pc_ecount_max=1)
    # now use mask to exclude 22,23 from region of interest:
    mask = np.zeros_like(dataset_err.all_data[0])
    mask[22:40,23:49] = 1 #exclude
    thisfile_dir = os.path.dirname(__file__)
    mask_filepath = os.path.join(thisfile_dir, 'test_data', 'pc_mask.fits')
    hdr = fits.Header()
    prim = fits.PrimaryHDU(header=hdr)
    img = fits.ImageHDU(mask)
    hdul = fits.HDUList([prim,img])
    hdul.writeto(mask_filepath, overwrite=True)
    pc_dataset_err = get_pc_mean(dataset_err, pc_ecount_max=1, mask_filepath=mask_filepath)
    copy_pc = pc_dataset_err.all_data.copy()
    # mask out all the irrelevant pixels:
    copy_pc[0,22:40,23:49] = np.nan
    # didn't dark-subtract this time
    history = ''
    for line in pc_dataset_err.frames[0].ext_hdr["HISTORY"]:
        history += line
    assert "Dark-subtracted with PC dark: no" in history
    assert np.isclose(np.nanmean(copy_pc), ill_mean, rtol=0.01) 
    # test safemode=False: should get warning instead of exception
    with pytest.warns(UserWarning):
        get_pc_mean(dataset_err, safemode=False)
    # test ISPC header value
    with pytest.raises(PhotonCountException):
        dataset_err.frames[0].ext_hdr['ISPC'] = False
        get_pc_mean(dataset_err, pc_master_dark=pc_dark)
    # set to True now
    dataset_err.frames[0].ext_hdr['ISPC'] = True
    # test inputmode's compatibility with dataset type
    with pytest.raises(PhotonCountException):
        get_pc_mean(dataset_err, pc_master_dark=pc_dark, inputmode='darks')
    with pytest.raises(PhotonCountException):
        get_pc_mean(dark_dataset_err, inputmode='illuminated')

def test_pc_subsets():
    '''
    Tests that the optional binning of frames works.
    '''
    np.random.seed(555)
    dataset_bin, dark_dataset_bin, ill_mean, dark_mean = mocks.create_photon_countable_frames(Nbrights=161, Ndarks=162, cosmic_rate=0, full_frame=False, smear=False) 
    # instead of running through walker, just do the pre-processing steps simply
    # using EM gain=5000 and kgain=7 and bias=20000 and read noise = 100 and QE=0.9 (quantum efficiency), from mocks.create_photon_countable_frames()
    for f in dataset_bin.frames:
        f.data = f.data.astype(float)*7 - 20000.
    for f in dark_dataset_bin.frames:
        f.data = f.data.astype(float)*7 - 20000.
    dataset_bin.all_data = dataset_bin.all_data.astype(float)*7 - 20000.
    dark_dataset_bin.all_data = dark_dataset_bin.all_data.astype(float)*7 - 20000.
    # process darks and check NUM_FR
    dark_dataset_bin[0].ext_hdr['HISTORY'] = '' # define a history value since get_pc_mean() uses it
    pc_dark = get_pc_mean(dark_dataset_bin, inputmode='darks', bin_size=40)
    assert pc_dark.ext_hdr['NUM_FR'] == 40 # The 2 remainder frames ignored for consistent statistics among the PC-averaged output frames
    assert 'Number of subsets: 4' in pc_dark.ext_hdr['HISTORY'][-4]
    # now process illuminated frames and subtract the PC dark
    dataset_bin[0].ext_hdr['HISTORY'] = '' # define a history value since get_pc_mean() uses it
    pc_dataset = get_pc_mean(dataset_bin, pc_master_dark=pc_dark, bin_size=40)
    # bigger rtol below since we have fewer frames averaged over in each of the 161//40 = 162//40 = 4 frames in the output
    assert np.isclose(pc_dataset.all_data.mean(), ill_mean - dark_mean, rtol=0.06) 
    assert pc_dataset.frames[0].ext_hdr['NUM_FR'] == 40
    assert pc_dataset.frames[-1].ext_hdr['NUM_FR'] == 40 # The 1 remainder frame ignored for consistent statistics among the PC-averaged output frames
    assert 'Number of subsets: 4' in pc_dataset.frames[0].ext_hdr['HISTORY'][-2]
    #since number of frames in a dark subset would be less than that of a subset in illuminated, warning statement is printed
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        get_pc_mean(dataset_bin, pc_master_dark=pc_dark, bin_size=51)
    captured = buf.getvalue()
    assert "Number of frames that created the photon-counted master dark should be greater than or equal to the number of illuminated frames in order for the result to be reliable.\n" in captured
    # but this is fine:
    get_pc_mean(dataset_bin, pc_master_dark=pc_dark, bin_size=38)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        get_pc_mean(dataset_bin, pc_master_dark=pc_dark, bin_size=51)
    captured2 = buf.getvalue()
    assert captured2 == captured

def test_no_data():
    '''Tests that a Dataset with only metadata (and has data read in one 
    frame at a time from filepaths) gives same results as a normal Dataset.
    '''
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    for name in os.listdir(datadir):
            path = os.path.join(datadir, name)
            os.remove(path)
    test_dataset, _, _, _ = mocks.create_photon_countable_frames(Nbrights=3, Ndarks=1, flux=0.1)
    # instead of running through walker, just do the pre-processing steps simply
    # using EM gain=5000 and kgain=7 and bias=20000 and read noise = 100 and QE=0.9 (quantum efficiency), from mocks.create_photon_countable_frames()
    for f in test_dataset.frames:
        f.data = f.data.astype(float)*7 - 20000.
    test_dataset.all_data = test_dataset.all_data.astype(float)*7 - 20000.
    for fr in test_dataset:
        fr.ext_hdr['HISTORY'] = '' # define a history value since get_pc_mean() uses it
    test_dataset.save(datadir)
    test_dataset_no_data = test_dataset.copy()
    for frame in test_dataset_no_data:
        frame.data = None
    with_data = get_pc_mean(test_dataset)
    without_data = get_pc_mean(test_dataset_no_data)
    # output is Dataset of 1 frame
    assert np.array_equal(with_data[0].data, without_data[0].data)
    assert np.array_equal(with_data[0].err, without_data[0].err)
    assert np.array_equal(with_data[0].dq, without_data[0].dq)    

if __name__ == '__main__':
    test_no_data()
    test_pc_subsets()
    test_pc()
    test_negative()
    
    