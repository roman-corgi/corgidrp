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

def test_negative():
    """Values at or below the 
    threshold are photon-counted as 0, including negative values."""
    fr = np.array([-3,2,0])
    pc_fr = photon_count(fr, thresh=2)
    assert (pc_fr == np.array([0,0,0])).all()
    

def test_pc():
    '''Test that a pixel that is masked heavily (reducing the number of usable frames for that pixel) has a bigger err than the average pixel.
    And if masking is all the way through for a certain pixel, that pixel will 
    be flagged in the DQ. Also, make sure there is failure when niter<1 and threshold<0, and test other built-in catches.
    Test safemode. Test mask_filepath to ensure failure for bright pixels outside region of interest when no mask used 
    and success when mask used to mask those bright pixels.'''
    # exposure time too long to get reasonable photon-counted result (after photometric correction)
    np.random.seed(555)
    dataset_err, dark_dataset_err, ill_mean, dark_mean = mocks.create_photon_countable_frames(Nbrights=160, Ndarks=160, cosmic_rate=0, full_frame=False) 
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
        f.ext_hdr['KGAIN'] = 7
    for f in dark_dataset_err.frames: 
        f.ext_hdr['RN'] = 100
        f.ext_hdr['KGAIN'] = 7
    # process the frames to make PC dark
    pc_dark = get_pc_mean(dark_dataset_err)
    assert pc_dark.ext_hdr['PC_STAT'] == 'photon-counted master dark'
    # now process illuminated frames and subtract the PC dark
    pc_dataset_err = get_pc_mean(dataset_err, pc_master_dark=pc_dark)
    assert pc_dataset_err.frames[0].filename[-7:] == 'pc.fits'
    assert pc_dataset_err.frames[0].filepath[-7:] == 'pc.fits'
    history = ''
    for line in pc_dataset_err.frames[0].ext_hdr["HISTORY"]:
        history += line
    assert "Dark-subtracted: yes" in history
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
    # can't provide Dark class instance while "VISTYPE" of input dataset is 'DARK'
    with pytest.raises(PhotonCountException):
        get_pc_mean(dark_dataset_err, pc_master_dark=pc_dark)
    # must have same 'CMDGAIN' and other header values throughout input dataset
    dark_dataset_err.frames[0].ext_hdr['CMDGAIN'] = 4999
    with pytest.raises(PhotonCountException):
        get_pc_mean(dark_dataset_err, pc_master_dark=pc_dark)

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
    # didn't dark-subtract this time:
    assert pc_dataset_err.frames[0].filename.endswith('pc_no_ds.fits')
    assert pc_dataset_err.frames[0].filepath.endswith('pc_no_ds.fits')
    history = ''
    for line in pc_dataset_err.frames[0].ext_hdr["HISTORY"]:
        history += line
    assert "Dark-subtracted: no" in history
    assert np.isclose(np.nanmean(copy_pc), ill_mean, rtol=0.01) 
    # test safemode=False: should get warning instead of exception
    with pytest.warns(UserWarning):
        get_pc_mean(dataset_err, safemode=False)

if __name__ == '__main__':
    test_pc()
    test_negative()
    
    