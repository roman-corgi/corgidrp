import os
import pytest
import numpy as np
import astropy.time as time
from astropy.io import fits
from scipy.signal import decimate

import corgidrp
import corgidrp.data as data
from corgidrp.mocks import create_default_headers, create_ct_psfs
from corgidrp.data import Image, Dataset
from corgidrp import corethroughput

# Generate a calibration file with the FPAM and FSAM rotation matrices if it
# does not exist
if not os.path.exists(os.path.join(corgidrp.default_cal_dir,
    'FpamFsamRotMat_2024-02-10T00:00:00.000.fits')):
    default_rot = data.FpamFsamRotMat([], 
        date_valid=time.Time("2024-02-10 00:00:00", scale='utc'))
    default_rot.save(filedir=corgidrp.default_cal_dir)

def setup_module():
    """
    Create a dataset with some representative psf responses. 
    """
    global cfam_name
    cfam_name = '1F'
    global dataset_ct, dataset_ct_syn
    # arbitrary set of PSF locations to be tested in EXCAM pixels referred to (0,0)
    global psf_loc_in, psf_loc_syn
    global ct_in, ct_syn

    # Default headers
    prhd, exthd = create_default_headers()
    # cfam filter
    exthd['CFAMNAME'] = cfam_name
    data_ct = []
    # Add pupil image(s) of the unocculted source's observation
    pupil_image_1 = np.zeros([1024, 1024])
    # Set it to some known value for some known pixels
    pupil_image_1[510:530, 510:530]=1
    pupil_image_2 = np.zeros([1024, 1024])
    # Set it to some known value for some known pixels
    pupil_image_2[510:530, 510:530]=3
    # Add specific values for pupil images:
    # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
    exthd_pupil = exthd.copy()
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'
    # Mock error
    err = np.ones([1,1024,1024])
    data_ct += [Image(pupil_image_1,pri_hdr = prhd, ext_hdr = exthd_pupil, err = err)]
    data_ct += [Image(pupil_image_2,pri_hdr = prhd, ext_hdr = exthd_pupil, err = err)]
    # Total counts from the pupil images
    unocc_psf_norm = (pupil_image_1.sum()+pupil_image_2.sum())/2
    # Adjust for pupil vs. direct imaging lens transmission
    # This function was written and tested during TVAC tests. There's a test value
    # in test_psf_pix_and_ct() later on
    di_over_pil = corethroughput.di_over_pil_transmission(filter=exthd['CFAMNAME'])
    unocc_psf_norm *= di_over_pil

    # 100 psfs with fwhm=50 mas in band 1 (mock.py)
    data_psf, psf_loc_in, half_psf = create_ct_psfs(50, cfam_name='1F',n_psfs=100)
    # Input CT
    ct_in = half_psf/unocc_psf_norm
    # Add pupil images
    data_ct += data_psf
    dataset_ct = Dataset(data_ct)

    data_ct = []
    # Synthetic PSF for a functional test
    psf_test = np.zeros([1024, 1024])
    # Maximum value at one pixel
    psf_loc_syn = [500, 400]
    # Set of known values at selected locations
    psf_test[psf_loc_syn[1]-3:psf_loc_syn[1]+4,
        psf_loc_syn[0]-3:psf_loc_syn[0]+4] = 1
    psf_test[psf_loc_syn[1]-2:psf_loc_syn[1]+3,
        psf_loc_syn[0]-2:psf_loc_syn[0]+3] = 2
    psf_test[psf_loc_syn[1]-1:psf_loc_syn[1]+2,
        psf_loc_syn[0]-1:psf_loc_syn[0]+2] = 3
    psf_test[psf_loc_syn[1],
        psf_loc_syn[0]] = 4
    data_ct += [Image(psf_test,pri_hdr=prhd, ext_hdr=exthd, err=err)]
    # Synthetic pupil images
    data_ct += [Image(pupil_image_1,pri_hdr = prhd, ext_hdr = exthd_pupil, err = err)]
    data_ct += [Image(pupil_image_2,pri_hdr = prhd, ext_hdr = exthd_pupil, err = err)]
    # Known CT
    ct_syn = (4+3*8+2*16)/(0.5*(1+3)*400)/di_over_pil
    # Dataset
    dataset_ct_syn = Dataset(data_ct) 

def test_psf_pix_and_ct():
    """
    Test 1090881 - Given a core throughput dataset consisting of M clean frames
    (nominally 1024x1024) taken at different FSM locations, the CTC GSW shall
    estimate the pixel location and core throughput of each PSF.

    NOTE: the list of M clean frames may be a subset of the frames collected during
    core throughput data collection, to allow for the removal of outliers.
    """

    # test 1:
    # Check transmition ratio of direct imaging vs pupil lenses
    # This function was written and tested during TVAC. The ratio is known to
    # be 1.01633660... by running the TVAC function
    assert (corethroughput.di_over_pil_transmission(filter=cfam_name) ==
        pytest.approx(1.01633660))

    # test 2:
    # Check that the step function retrieves the expected location and CT of
    # a set of simulated 2D Gaussian PSFs (created in setup_module before:)
    psf_loc_est, ct_est = corethroughput.estimate_psf_pix_and_ct(dataset_ct)
    # Difference between expected and retrieved locations for the max (peak) method
    diff_psf_loc = psf_loc_in - psf_loc_est
    # Set a difference of 0.005 pixels
    assert np.all(np.abs(diff_psf_loc) <= 0.005)

    # core throughput in (0,1]
    assert np.all(ct_est) > 0
    assert np.all(ct_est) <= 1

    # comparison between I/O values (1%)
    assert np.all(np.abs(ct_est-ct_in) <= 0.01)

    # test 3:
    # Functional test with some mock data with known PSF location and CT
    # Synthetic PSF from setup_module
    psf_loc_est, ct_est = corethroughput.estimate_psf_pix_and_ct(dataset_ct_syn)
    # Exact agreement
    assert np.all(psf_loc_est == psf_loc_syn)
    assert ct_syn == ct_est

def test_fpm_pos():
    """
    Test 1090882 - Given 1) the location of the center of the FPM coronagraphic
    mask in EXCAM pixels during the coronagraphic observing sequence and 2) the
    FPAM and FSAM encoder positions during both the coronagraphic and core
    throughput observing sequences, the CTC GSW shall compute the center of the
    FPM coronagraphic mask during the core throughput observing sequence.

    Python algorithm written during TVAC for using FPAM and FSAM matrices
    
          delta_pam = np.array([[dh], [dv]]) # fill these in
          # read FPAM or FSAM matrices: 
          M = np.array([[ M00, M01], [M10, M11]], dtype=float32)
          delta_pix = M @ delta_pam
    """

    # DRP calibration files
    fpam2excam_matrix, fsam2excam_matrix = corethroughput.read_rot_matrix()
    # TVAC files
    fpam2excam_matrix_tvac = fits.getdata(os.path.join(os.getcwd(),
        'test_data', 'fpam_to_excam_modelbased.fits'))
    fsam2excam_matrix_tvac = fits.getdata(os.path.join(os.getcwd(),
        'test_data', 'fsam_to_excam_modelbased.fits'))

    # test 1:
    # Check that DRP calibration files for FPAM and FSAM agree with TVAC files
    # Some irrelevant rounding happens when defining FpamFsamRotMat() in data.py
    assert np.all(fpam2excam_matrix - fpam2excam_matrix_tvac <= 1e-9)
    assert np.all(fsam2excam_matrix - fsam2excam_matrix_tvac <= 1e-9)
   
    # test 2:
    # Using values within the range should return a meaningful value. Tested 10 times
    rng = np.random.default_rng(0)
    for _ in range(10):
        EXCAM_center_pos_pix = np.array([rng.integers(300,700),rng.integers(300,700)])
        # Irrelevant change of units for the origin since what matters is the difference
        # (see below) Written using model values for mas/um for FPAM and FSAM
        FPAM_center_pos_um = EXCAM_center_pos_pix * 21.8 / 2.67
        FSAM_center_pos_um = EXCAM_center_pos_pix * 21.8 / 2.10
        # Delta H/V in um
        delta_fpam_um = np.array([[rng.uniform(1,10)], [rng.uniform(1,10)]])
        delta_fsam_um = np.array([[rng.uniform(1,10)], [rng.uniform(1,10)]])
        # Expected shifts
        delta_fpam_pix = fpam2excam_matrix @ delta_fpam_um
        delta_fsam_pix = fsam2excam_matrix @ delta_fsam_um
        fpam_center_ct_pix_out, fsam_center_ct_pix_out = \
            corethroughput.get_ct_fpm_center(EXCAM_center_pos_pix,
            fpam_pos_cor=FPAM_center_pos_um,
            fpam_pos_ct=FPAM_center_pos_um + delta_fpam_um.transpose()[0],
            fsam_pos_cor=FSAM_center_pos_um,
            fsam_pos_ct=FSAM_center_pos_um + delta_fsam_um.transpose()[0])

        # Compare output with expected value. Some of the random tests have differences
        # of ~1e-13, while others are exactly equal
        fpam_center_ct_pix_in = EXCAM_center_pos_pix + delta_fpam_pix.transpose()[0]
        fpam_center_ct_pix_out - fpam_center_ct_pix_in
        assert np.all(fpam_center_ct_pix_out - fpam_center_ct_pix_in <= 1e-12)
        fsam_center_ct_pix_in = EXCAM_center_pos_pix + delta_fsam_pix.transpose()[0]
        fsam_center_ct_pix_out - fsam_center_ct_pix_in
        assert np.all(fsam_center_ct_pix_out - fsam_center_ct_pix_in <= 1e-12)

def test_ct_map():
    """ 
    Test 1090883 - Given 1) an array of PSF pixel locations and 2) the location
    of the center of the FPAM coronagraphic mask in EXCAM pixels during core
    throughput calibrations, and 3) corresponding core throughputs for each PSF,
    the CTC GSW shall compute a 2D floating-point interpolated core throughput
    map.
    """
    psf_pix = psf_loc_in.transpose()
    fpam_pix = np.array([513,515])
    target_pix = np.array([520, 520])

    # test 1:
    # If all the target pixels are outside the range of the original data, the
    # function must fail
    target_pix_x = [331.8, 141.6, 851.4, 560, 521.4, 532, 542,
        752, 362]
    target_pix_y = [830.4, 540, 550.3, 361.2, 210.6, 920.6, 382.8,
        474, 476]
    target_pix = np.array([target_pix_x, target_pix_y])
    with pytest.raises(ValueError):
        corethroughput.ct_map(psf_pix, fpam_pix, ct_in, target_pix)
    
    # test 2:
    # If inputs are valid, the function must return a set of interpolated
    # core throughput values within (0,1]
    target_pix_x = [531.8, 541.6, 551.4, 512, 519.4, 532, 542,
        552, 562]
    target_pix_y = [530.4, 540, 550.3, 512, 512.6, 492.6, 482.8,
        474, 476]
    target_pix = np.array([target_pix_x, target_pix_y])
    
    ct_map = corethroughput.ct_map(psf_pix, fpam_pix, ct_in, target_pix)
    # core throughput in (0,1]
    assert np.all(ct_map[-1]) > 0
    assert np.all(ct_map[-1]) <= 1
    # Add some numerical comparison based on expected changes of core throughput
    assert np.all(ct_map[-1] < np.mean(ct_in) + 2*np.std(ct_in))
    assert np.all(ct_map[-1] > np.mean(ct_in) - 2*np.std(ct_in))

if __name__ == '__main__':
    test_psf_pix_and_ct()
    test_fpm_pos()
    test_ct_map()


