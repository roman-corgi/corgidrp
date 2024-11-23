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

detector_params = data.DetectorParams({})

def test_expected_results():
    '''Results are as expected theoretically.  Also runs raw frames through pre-processing pipeline.'''
    dataset, ill_mean, dark_mean = mocks.create_photon_countable_frames(Nbrights=20, Ndarks=30, cosmic_rate=0)
    thisfile_dir = os.path.dirname(__file__) # this file's folder
    outputdir = os.path.join(thisfile_dir, 'simdata', 'pc_test_data')
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    # empty out directory of any previous files
    for f in os.listdir(outputdir):
        os.remove(os.path.join(outputdir,f))
    dataset.save(outputdir, ['pc_frame_{0}.fits'.format(i) for i in range(len(dataset))])
    l1_data_filelist = []
    for f in os.listdir(outputdir):
        l1_data_filelist.append(os.path.join(outputdir, f))

    this_caldb = caldb.CalDB() # connection to cal DB
    # remove other KGain calibrations that may exist in case they don't have the added header keywords
    for i in range(len(this_caldb._db['Type'])):
        if this_caldb._db['Type'][i] == 'KGain':
            this_caldb._db = this_caldb._db.drop(i)
    this_caldb.save()
    # KGain
    kgain_val = 7 # default value used in mocks.create_photon_countable_frames()
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(l1_data_filelist)
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    # add in keywords that didn't make it into mock_kgain.fits, using values used in mocks.create_photon_countable_frames()
    kgain.ext_hdr['RN'] = 100
    kgain.ext_hdr['RN_ERR'] = 0
    kgain.save(filedir=outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)

    # NoiseMap
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electrons'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=outputdir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    walker.walk_corgidrp(l1_data_filelist, '', outputdir, template="l1_to_l2b_pc.json")
    # get photon-counted frame
    for f in os.listdir(outputdir):
        if f.endswith('_pc.fits'):
            pc_filename = f
    pc_file = os.path.join(outputdir, pc_filename)
    pc_frame = fits.getdata(pc_file)
    pc_frame_err = fits.getdata(pc_file, 'ERR')
    pc_ext_hdr = fits.getheader(pc_file, 1)
    # more frames (which would take longer to run) would give an even better agreement than the 5% agreement below
    assert np.isclose(pc_frame.mean(), ill_mean - dark_mean, rtol=0.05) 
    assert 'niter=2' in pc_ext_hdr["HISTORY"][-1]
    assert 'T_factor=5' in pc_ext_hdr["HISTORY"][-1]
    assert pc_frame_err.min() >= 0
    assert pc_frame_err[0][3,3] >= 0


def test_negative():
    """Values at or below the 
    threshold are photon-counted as 0, including negative values."""
    fr = np.array([-3,2,0])
    pc_fr = photon_count(fr, thresh=2)
    assert (pc_fr == np.array([0,0,0])).all()
    

def test_masking_increases_err():
    '''Test that a pixel that is masked heavily (reducing the number of usable frames for that pixel) has a bigger err than the average pixel.
    And if masking is all the way through for a certain pixel, that pixel will 
    be flagged in the DQ. Also, make sure there is failure when niter<1.'''
    # exposure time too long to get reasonable photon-counted result (after photometric correction)
    dataset_err, _, _ = mocks.create_photon_countable_frames(Nbrights=60, Ndarks=60, cosmic_rate=0, full_frame=False) 
    # instead of running through walker, just do the pre-processing steps simply
    # using kgain=7 and bias=2000 and read noise = 100, from mocks.create_photon_countable_frames()
    dataset_err.all_data = dataset_err.all_data.astype(float)*7 - 2000.
    # masked for half of the bright and half of the dark frames
    dataset_err.all_dq[30:90,3,3] = 1
    dataset_err.all_dq[:, 2, 2] = 1 #masked all the way through for (2,2)
    for f in dataset_err.frames: 
        f.ext_hdr['RN'] = 100
        f.ext_hdr['KGAIN'] = 7
    pc_dataset_err = get_pc_mean(dataset_err, detector_params)
    # the DQ for that pixel should be 1
    assert pc_dataset_err[0].dq[2,2] == 1
    # err for (3,3) is above the 95th percentile of error: 
    assert pc_dataset_err[0].err[0][3,3]>np.nanpercentile(pc_dataset_err[0].err,95)
    # also when niter<1, exception
    with pytest.raises(PhotonCountException):
        get_pc_mean(dataset_err, detector_params, niter=0)

if __name__ == '__main__':
    test_masking_increases_err()
    test_negative()
    test_expected_results()
    
    