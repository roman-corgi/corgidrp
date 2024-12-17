import os
import pytest
import numpy as np
from astropy.io import fits
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset
import corgidrp.corethroughput as corethroughput

ct_filepath = os.path.join(os.path.dirname(__file__), 'test_data')
# Mock error
err = np.ones([1,1024,1024]) * 0.5
# Default headers
prhd, exthd = create_default_headers()

def setup_module():
    """
    Create a dataset with some representative psf responses. 
    """
    # corgidrp dataset
    global dataset_psf, dataset_psf_rev
    # arbitrary set of PSF positions to be tested in EXCAM pixels referred to (0,0)
    global psf_position_x, psf_position_y
    # Some arbitrary shifts
    psf_position_x = [512, 522, 532, 542, 552, 562, 522, 532, 542, 552, 562]
    psf_position_y = [512, 522, 532, 542, 552, 562, 502, 492, 482, 472, 462]
    # fsm positions for the off-axis psfs
    global fsm_pos
    fsm_pos = [[1,1]]*len(psf_position_x[1:])
    global idx_os11
    idx_os11 = 8
    global ct_os11
    ct_os11 = []

    data_unocc = np.zeros([1024, 1024])
    # unocculted PSF
    unocc_psf_filepath = os.path.join(ct_filepath, 'hlc_os11_no_fpm.fits')
    # os11 unocculted PSF is sampled at the same pixel pitch as EXCAM
    unocc_psf = fits.getdata(unocc_psf_filepath)
    # Insert PSF at its location
    idx_0_0 = psf_position_x[0] - unocc_psf.shape[0]
    idx_0_1 = idx_0_0 + unocc_psf.shape[0]
    idx_1_0 = psf_position_y[0] - unocc_psf.shape[1]
    idx_1_1 = idx_1_0 + unocc_psf.shape[1]
    data_unocc[idx_0_0:idx_0_1, idx_1_0:idx_1_1] = unocc_psf
    
    data_psf = [Image(data_unocc,pri_hdr = prhd, ext_hdr = exthd, err = err)]
    # oversampled os11 psfs
    occ_psf_filepath = os.path.join(ct_filepath, 'hlc_os11_psfs_oversampled.fits')
    occ_psf = fits.getdata(occ_psf_filepath)
    for i_psf, _ in enumerate(psf_position_x[1:]):
        psf_tmp = occ_psf[0, idx_os11+i_psf]
        # re-sample to EXCAM's pixel pitch: os11 off-axis psf is 5x oversampled
        psf_tmp_red = 25*block_reduce(psf_tmp, block_size=(5,5), func=np.mean)
        data_tmp = np.zeros([1024, 1024])
        idx_0_0 = psf_position_x[i_psf+1] - psf_tmp_red.shape[0] // 2
        idx_0_1 = idx_0_0 + psf_tmp_red.shape[0]
        idx_1_0 = psf_position_y[i_psf+1] - psf_tmp_red.shape[1] // 2
        idx_1_1 = idx_1_0 + psf_tmp_red.shape[1]
        data_tmp[idx_0_0:idx_0_1, idx_1_0:idx_1_1] = psf_tmp_red
        data_psf += [Image(data_tmp,pri_hdr = prhd, ext_hdr = exthd, err = err)]
        ct_os11 += [psf_tmp[psf_tmp > psf_tmp.max()/2].sum()/unocc_psf.sum()]

    dataset_psf = Dataset(data_psf)

def test_fsm_pos():
    """ Test FSM positions are a list of N pairs of values, where N is the number
        of off-axis psfs.
    """
    # do not pass if fsm_pos is not a list
    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, np.array(fsm_pos))
    # do not pass if fsm_pos has less elements
    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, fsm_pos[1:])
    # do not pass if fsm_pos has more elements
    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, fsm_pos+[[1,1]])
    # do not pass if fsm_pos is not a list of paired values
    fsm_pos_bad = [[1,1]]*(len(psf_position_x[1:]) - 1)
    fsm_pos_bad += [[1]]
    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, fsm_pos_bad)

def test_psf_pix_and_ct():
    """
    Test 1090881 - Given a core throughput dataset consisting of M clean frames
    (nominally 1024x1024) taken at different FSM positions, the CTC GSW shall
    estimate the pixel location and core throughput of each PSF.

    NOTE: the list of M clean frames may be a subset of the frames collected during
    core throughput data collection, to allow for the removal of outliers.
    """

    # do not pass if setting a method that does not exist
    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, fsm_pos, pix_method='bad')

    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, fsm_pos, ct_method='bad')

    psf_pix, ct = corethroughput.estimate_psf_pix_and_ct(dataset_psf, fsm_pos)

    # Read OS11 PSF offsets (l/D=50.19mas=2.3 EXCAM pix, 1 EXCAM pix=0.4347825 l/D, 1 EXCAM pix=21.8213 mas)
    r_off = fits.getdata(os.path.join(ct_filepath, 'hlc_os11_psfs_radial_offsets.fits'))
    r_off_pix = r_off[idx_os11:idx_os11+len(psf_pix)] * 2.3
    # Difference between expected and retrieved positions
    diff_pix_x = psf_position_x[1:] - psf_pix[:,0]
    # os11 azimuthal axis
    assert diff_pix_x == pytest.approx(0)
    # os11 radial axis
    diff_pix_y = psf_position_y[1:] + r_off_pix - psf_pix[:,1] 
    assert diff_pix_y == pytest.approx(0, abs=0.75)

    assert np.all(ct) > 0
    assert ct == pytest.approx(np.array(ct_os11), abs=0.02)
    
if __name__ == '__main__':

    test_fsm_pos()
    test_psf_pix_and_ct()


