import os
import pytest
import numpy as np
from astropy.io import fits
from scipy.signal import decimate

from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset
from corgidrp import corethroughput

ct_filepath = os.path.join(os.path.dirname(__file__), 'test_data')
# Mock error
err = np.ones([1,1024,1024]) * 0.5
# Default headers
prhd, exthd = create_default_headers()

def setup_module():
    """
    Create a dataset with some representative psf responses. 
    """
    global dataset_psf
    # arbitrary set of PSF positions to be tested in EXCAM pixels referred to (0,0)
    global psf_position_max
    global idx_os11
    idx_os11 = 8
    global ct_os11
    ct_os11 = []

    data_psf = []
    # add pupil image(s) of the unocculted source's observation
    data_pupil = fits.getdata(os.path.join(ct_filepath, 'pupil_image_0000094916.fits'))
    # Normalize to 1 since OS11 off-axis PSFs are already normalized to the
    # unocculted response
    data_pupil /= np.sum(data_pupil)
    # Add some noise (pupil images are high SNR)
    data_pupil_1 = data_pupil.copy()
    rng = np.random.default_rng(seed=0)
    data_pupil_1 += rng.normal(0, data_pupil.std()/10, data_pupil_1.shape)
    data_pupil_2 = data_pupil.copy()
    data_pupil_2 += rng.normal(0, data_pupil.std()/10, data_pupil_1.shape)
    data_psf += [Image(data_pupil_1,pri_hdr = prhd, ext_hdr = exthd, err = err)]
    data_psf += [Image(data_pupil_2,pri_hdr = prhd, ext_hdr = exthd, err = err)]
    # Total counts from the pupil images
    unocc_psf_norm = (data_pupil_1.sum()+data_pupil_2.sum())/2

    # add os11 psfs
    occ_psf_filepath = os.path.join(ct_filepath, 'hlc_os11_psfs_oversampled.fits')
    occ_psf = fits.getdata(occ_psf_filepath)
    # Some arbitrary shifts
    psf_position_x = [512, 522, 532, 542, 552, 562, 522, 532, 542, 552, 562]
    psf_position_y = [512, 522, 532, 542, 552, 562, 502, 492, 482, 472, 462]
    psf_position_max = []
    for i_psf, _ in enumerate(psf_position_x):
        psf_tmp = occ_psf[0, idx_os11+i_psf]
        # re-sample to EXCAM's pixel pitch: os11 off-axis psf is 5x oversampled
        psf_tmp_red = 25*decimate(decimate(psf_tmp, 5, axis=0), 5, axis=1)
        data_tmp = np.zeros([1024, 1024])
        idx_0_0 = psf_position_x[i_psf] - psf_tmp_red.shape[0] // 2
        idx_0_1 = idx_0_0 + psf_tmp_red.shape[0]
        idx_1_0 = psf_position_y[i_psf] - psf_tmp_red.shape[1] // 2
        idx_1_1 = idx_1_0 + psf_tmp_red.shape[1]
        data_tmp[idx_0_0:idx_0_1, idx_1_0:idx_1_1] = psf_tmp_red
        data_psf += [Image(data_tmp,pri_hdr = prhd, ext_hdr = exthd, err = err)]
        psf_position_max += [np.unravel_index(data_tmp.argmax(), data_tmp.shape)]
        ct_os11 += [data_tmp[data_tmp >= data_tmp.max()/2].sum()/unocc_psf_norm]

    dataset_psf = Dataset(data_psf)
    psf_position_max = np.array(psf_position_max)

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
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, pix_method='bad')

    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, ct_method='bad')

    psf_pix_est_max, ct_est = corethroughput.estimate_psf_pix_and_ct(dataset_psf)

    # Difference between expected and retrieved positions for the max (peak) method
    diff_pix_x = psf_position_max[:,0] - psf_pix_est_max[:,0]
    # os11 azimuthal axis
    assert diff_pix_x == pytest.approx(0)
    # os11 radial axis
    diff_pix_y = psf_position_max[:,1] - psf_pix_est_max[:,1] 
    assert diff_pix_y == pytest.approx(0)

    # core throughput in (0,1]
    assert np.all(ct_est) > 0
    assert np.all(ct_est) <= 1

    # comparison between I/O values
    assert ct_est == pytest.approx(np.array(ct_os11))

def test_fpm_pos():
    """
    Test 1090882 - Given 1) the location of the center of the FPM coronagraphic
    mask in EXCAM pixels during the coronagraphic observing sequence and 2) the
    FPAM and FSAM encoder positions during both the coronagraphic and core
    throughput observing sequences, the CTC GSW shall compute the center of the
    FPM coronagraphic mask during the core throughput observing sequence.
    """
    
    # If the input FPM, FPAM or FSAM positions are not each a 2-dimensional
    # array, the function must fail
    with pytest.raises(OSError):
        corethroughput.get_ct_fpm_center(np.array(1))   
        corethroughput.get_ct_fpm_center(np.array([1]))

    # FPM center must be within EXCAM boundaries and with enough space to
    # accommodate the HLC mask area (OWA radius <=9.7 l/D ~ 487 mas ~ 22.34
    # EXCAM pixels)
    EXCAM_center_pos_pix = np.array([512,512])
    # EXCAM's pixel pitch and theoretical values for mas/um for FPAM and FSAM
    FPAM_center_pos_um = EXCAM_center_pos_pix * 21.8 / 2.67
    FSAM_center_pos_um = EXCAM_center_pos_pix * 21.8 / 2.10
    if False:
        for row in range(2):
            for pix in [15, 1015]:
                if row:
                    center_pix = np.array([pix, EXCAM_center_pos_pix[1] ])
                else:
                    center_pix = np.array([EXCAM_center_pos_pix[0], pix])
                with pytest.raises(ValueError):
                    corethroughput.get_ct_fpm_center(center_pix,
                        fpam_pos_cor=FPAM_center_pos_um,
                        fpam_pos_ct=FPAM_center_pos_um,
                        fsam_pos_cor=FSAM_center_pos_um,
                        fsam_pos_ct=FSAM_center_pos_um)
    
        # If we parse a file for the rotation matrix that does not exist, the function
        # must fail
        with pytest.raises(OSError):
            corethroughput.get_ct_fpm_center(EXCAM_center_pos_pix,
                        fpam_pos_cor=FPAM_center_pos_um,
                        fpam_pos_ct=FPAM_center_pos_um,
                        fsam_pos_cor=FSAM_center_pos_um,
                        fsam_pos_ct=FSAM_center_pos_um,
                        fpam2excam_matrix='bad_file.fits')
        with pytest.raises(OSError):
            corethroughput.get_ct_fpm_center(EXCAM_center_pos_pix,
                        fpam_pos_cor=FPAM_center_pos_um,
                        fpam_pos_ct=FPAM_center_pos_um,
                        fsam_pos_cor=FSAM_center_pos_um,
                        fsam_pos_ct=FSAM_center_pos_um,
                        fsam2excam_matrix='bad_file.fits')
     
    # Using values within the range should return a meaningful value
    # Shifting the FPM by (+1,+1) EXCAM pixels -using approx. values
    fpm_center_ct_pix = corethroughput.get_ct_fpm_center(EXCAM_center_pos_pix,
        fpam_pos_cor=FPAM_center_pos_um,
        fpam_pos_ct=FPAM_center_pos_um + np.array([-1,1])*21.8/2.67,
        fsam_pos_cor=FSAM_center_pos_um,
        fsam_pos_ct=FSAM_center_pos_um + np.array([-1,-1])*21.8/2.10)

    assert (fpm_center_ct_pix - np.array([1,1]) ==
        pytest.approx(EXCAM_center_pos_pix, abs=0.05))

def test_ct_map():
    """ 
    Test 1090883 - Given 1) an array of PSF pixel locations and 2) the location
    of the center of the FPAM coronagraphic mask in EXCAM pixels during core
    throughput calibrations, and 3) corresponding core throughputs for each PSF,
    the CTC GSW shall compute a 2D floating-point interpolated core throughput
    map.
    """
    psf_pix = np.array(psf_position_max[:,0], psf_position_max[:,1])
    fpam_pix = np.array([513,515])
    target_pix = np.array([520, 520])

    # If FPAM position is not a 2-dimensional array, the function must fail
    with pytest.raises(TypeError):
        corethroughput.ct_map(psf_pix, fpam_pix[0], ct_os11, target_pix)

    # If FPAM position is outside a reasonable range, the function must fail
    with pytest.raises(ValueError):
        corethroughput.ct_map(psf_pix, np.array([1000,512]), ct_os11, target_pix)
    with pytest.raises(ValueError):
        corethroughput.ct_map(psf_pix, np.array([512,1000]), ct_os11, target_pix)
    with pytest.raises(ValueError):
        corethroughput.ct_map(psf_pix, np.array([1000,1000]), ct_os11, target_pix)
    with pytest.raises(ValueError):
        corethroughput.ct_map(psf_pix, np.array([100,512]), ct_os11, target_pix)
    with pytest.raises(ValueError):
        corethroughput.ct_map(psf_pix, np.array([512,100]), ct_os11, target_pix)
    with pytest.raises(ValueError):
        corethroughput.ct_map(psf_pix, np.array([100,100]), ct_os11, target_pix)    

    # If the psf positions is not a 2-dimensional array, the function must fail
    with pytest.raises(TypeError):
        corethroughput.ct_map(psf_pix[0,:], fpam_pix, ct_os11, target_pix)
    # There must be more than one PSF to be able to interpolate
    with pytest.raises(IndexError):
        corethroughput.ct_map(psf_pix[:,0], fpam_pix, ct_os11, target_pix)
    # If the number of core throughput values is different than the number of
    # PSFs, the function must fails
    with pytest.raises(ValueError):
         corethroughput.ct_map(psf_pix[:,0:-1], fpam_pix, ct_os11, target_pix)
    with pytest.raises(ValueError):
         corethroughput.ct_map(psf_pix, fpam_pix, ct_os11[0:-1], target_pix)
    # If ct is > 1 or < 0, the function must fail
    ct_os11_wrong = ct_os11.copy()
    ct_os11_wrong[0] = 1.2
    with pytest.raises(ValueError):
         corethroughput.ct_map(psf_pix, fpam_pix, ct_os11_wrong, target_pix)
    ct_os11_wrong[0] = 0
    with pytest.raises(ValueError):
         corethroughput.ct_map(psf_pix, fpam_pix, ct_os11_wrong, target_pix)
    ct_os11_wrong[0] = -0.1
    with pytest.raises(ValueError):
         corethroughput.ct_map(psf_pix, fpam_pix, ct_os11_wrong, target_pix)
    # If the target pixels are outside the range of the original data, the
    # function must fail
    target_pix_x = [531.8, 541.6, 551.4, 560, 521.4, 532, 542,
        552, 562]
    target_pix_y = [530.4, 540, 550.3, 561.2, 500.6, 492.6, 482.8,
        474, 476]
    target_pix = np.array([target_pix_x, target_pix_y])
    with pytest.raises(ValueError):
        corethroughput.ct_map(psf_pix, fpam_pix, ct_os11, target_pix)
    # If target positions are the same as the reference ones, the core throughput
    # must be the same
    target_pix = psf_pix
        

    # If all the conditions are met, the function must return a set of interpolated
    # core throughput values within (0,1]
    target_pix_x = [531.8, 541.6, 551.4, 512, 519.4, 532, 542,
        552, 562]
    target_pix_y = [530.4, 540, 550.3, 512, 512.6, 492.6, 482.8,
        474, 476]
    target_pix = np.array([target_pix_x, target_pix_y])
    ct_map = corethroughput.ct_map(psf_pix, fpam_pix, ct_os11, target_pix)
    # core throughput in (0,1]
    assert np.all(ct_map[-1]) > 0
    assert np.all(ct_map[-1]) <= 1
    # Add some numerical comparison based on expected changes of core throughput
    assert np.all(ct_map[-1] < np.mean(ct_os11) + 2*np.std(ct_os11))
    assert np.all(ct_map[-1] > np.mean(ct_os11) - 2*np.std(ct_os11))

if __name__ == '__main__':
    test_psf_pix_and_ct()
    test_fpm_pos()
    test_ct_map()


