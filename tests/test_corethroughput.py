import os
import pytest
import numpy as np
import astropy.time as time
from astropy.io import fits
from scipy.signal import decimate

import corgidrp
import corgidrp.data as data
from corgidrp.mocks import create_default_L3_headers, create_ct_psfs
from corgidrp.data import Image, Dataset, FpamFsamCal, CoreThroughputCalibration
from corgidrp import corethroughput, caldb

here = os.path.abspath(os.path.dirname(__file__))

def setup_module():
    """
    Create a dataset with some representative psf responses. 
    """
    global cfam_name
    cfam_name = '1F'
    # CT and coronagraphic datasets
    global dataset_ct, dataset_ct_syn, dataset_cor
    # arbitrary set of PSF locations to be tested in EXCAM pixels referred to (0,0)
    global psf_loc_in, psf_loc_syn
    global ct_in, ct_syn

    # Default headers
    prhd, exthd = create_default_L3_headers()
    # DRP
    exthd['DRPCTIME'] = time.Time.now().isot
    exthd['DRPVERSN'] = corgidrp.__version__
    # cfam filter
    exthd['CFAMNAME'] = cfam_name
    # FPAM/FSAM
    # Choose some H/V values for FPAM/FSAM  during corethroughput observations
    exthd['FPAM_H'] = 6854
    exthd['FPAM_V'] = 22524
    exthd['FSAM_H'] = 29471
    exthd['FSAM_V'] = 12120

    data_ct = []
    # Add pupil image(s) of the unocculted source's observation to test that
    # the corethroughput calibration function can handle more than one pupil image
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
    err = np.ones([1024,1024])
    # Add pupil images
    data_ct += [Image(pupil_image_1,pri_hdr = prhd, ext_hdr = exthd_pupil, err = err)]
    data_ct += [Image(pupil_image_2,pri_hdr = prhd, ext_hdr = exthd_pupil, err = err)]
    # Total counts from the pupil images
    unocc_psf_norm = (pupil_image_1.sum()+pupil_image_2.sum())/2
    # Adjust for pupil vs. direct imaging lens transmission
    # This function was written and tested during TVAC tests. There's a test value
    # in test_psf_pix_and_ct() later on
    di_over_pil = corethroughput.di_over_pil_transmission(cfam_name=exthd['CFAMNAME'])
    unocc_psf_norm *= di_over_pil

    # 100 psfs with fwhm=50 mas in band 1 (mock.py)
    data_psf, psf_loc_in, half_psf = create_ct_psfs(50, cfam_name='1F',
        n_psfs=100)
    # Input CT
    ct_in = half_psf/unocc_psf_norm
    # Add PSF images
    data_ct += data_psf
    dataset_ct = Dataset(data_ct)

    # Synthetic PSF for a functional test
    data_ct = []
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

    # Coronagraphic dataset (only headers will be used)
    # FPAM/FSAM
    # Choose some H/V values for FPAM/FSAM  during corethroughput observations
    # These values are *different* than the ones in the dataset_ct defined before
    exthd['FPAM_H'] = 6757
    exthd['FPAM_V'] = 22424
    exthd['FSAM_H'] = 29387
    exthd['FSAM_V'] = 12238
    # FPM center
    exthd['MASKLOCX'] = 509
    exthd['MASKLOCY'] = 513
    data_cor = [Image(np.zeros([1024, 1024]), pri_hdr=prhd, ext_hdr=exthd, err=err)]
    dataset_cor = Dataset(data_cor)

def test_psf_pix_and_ct():
    """
    Test 1090881 - Given a core throughput dataset consisting of M clean frames
    (nominally 1024x1024) taken at different FSM locations, the CTC GSW shall
    estimate the pixel location and core throughput of each PSF.

    NOTE: the list of M clean frames may be a subset of the frames collected during
    core throughput data collection, to allow for the removal of outliers.
    """
    # Test 1:
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
    # comparison between I/O values (<=1% due to pixelization effects vs.
    # expected analytical value)
    assert np.all(np.abs(ct_est-ct_in) <= 0.01)

    # Test 2:
    # Functional test with some mock data with known PSF location and CT
    # Synthetic PSF from setup_module
    psf_loc_est, ct_est = corethroughput.estimate_psf_pix_and_ct(dataset_ct_syn)
    # In this test, there must be an exact agreement
    assert np.all(psf_loc_est[0] == psf_loc_syn)
    assert np.abs(ct_est[0]-ct_syn) < 1e-16

    print('Tests of PSF locations and CT values passed')

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
    # FPAM/FSAM transformations in DRP
    fpam_fsam_trans = FpamFsamCal(os.path.join(corgidrp.default_cal_dir,
        'FpamFsamCal_2024-02-10T00:00:00.000.fits'))
    
    fpam2excam_matrix, fsam2excam_matrix = fpam_fsam_trans.data
    # TVAC files
    fpam2excam_matrix_tvac = fits.getdata(os.path.join(here, 'test_data',
        'fpam_to_excam_modelbased.fits'))
    fsam2excam_matrix_tvac = fits.getdata(os.path.join(here, 'test_data',
        'fsam_to_excam_modelbased.fits'))

    # test 1:
    # Check that DRP calibration files for FPAM and FSAM agree with TVAC files
    # Some irrelevant rounding happens when defining FpamFsamCal() in data.py
    assert np.all(fpam2excam_matrix - fpam2excam_matrix_tvac <= 1e-9)
    assert np.all(fsam2excam_matrix - fsam2excam_matrix_tvac <= 1e-9)
   
    # test 2:
    # Using values within the range should return a meaningful value. Tested 10 times
    rng = np.random.default_rng(0)
    for _ in range(10):
        # EXCAM_center_pos_pix, FPAM_center_pos_um, FSAM_center_pos_um have to be read from the Headers of the dataset_cor

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

    print('Tests of FPAM/FSAM to EXCAM passed')

def test_cal_file():
    """ Test creation of core throughput calibration file. """

    # Write core throughput calibration file
    ct_cal_inputs = corethroughput.generate_ct_cal(dataset_ct)
    ct_cal_file_in = CoreThroughputCalibration(
        ct_cal_inputs[0], pri_hdr=dataset_ct[0].pri_hdr, ext_hdr=ct_cal_inputs[3],
        dq=ct_cal_inputs[1], input_hdulist=ct_cal_inputs[2],
        input_dataset=dataset_ct)
    # It's fine to use a hardcoded filename for UTs
    ct_cal_file_in.save(filedir=corgidrp.default_cal_dir,
        filename='CoreThroughputCalibration_2025-02-15T00:00:00.fits')

    # This test checks that the CT cal file has the right information by making
    # sure that I=O (Note: the comparison b/w analytical predictions
    # vs. centroid/pixelized data is part of the tests on test_psf_pix_and_ct before)

    # Input values for PSF centers and corresponding CT values
    psf_loc_input, ct_input = \
        corethroughput.estimate_psf_pix_and_ct(dataset_ct)

    # Test: open calibration file
    try:
        ct_cal_file = corgidrp.data.CoreThroughputCalibration(ct_cal_file_in.filepath)
    except:
        raise IOError('CT cal file was not saved')

    # Test 1: PSF cube
    # Recover off-axis PSF cube from CT Dataset
    psf_cube_in = []
    for frame in dataset_ct:
        try:
        # Pupil images of the unocculted source satisfy:
        # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
            exthd = frame.ext_hdr
            if (exthd['DPAMNAME']=='PUPIL' and exthd['LSAMNAME']=='OPEN' and
                exthd['FSAMNAME']=='OPEN' and exthd['FPAMNAME']=='OPEN_12'):
                continue
        except:
           pass
        psf_cube_in += [frame.data]
    psf_cube_in = np.array(psf_cube_in)

    # Compare the PSF cube from the calibration file, which may have a smaller
    # extension, with the input ones
    cal_file_side_0 = ct_cal_file.data.shape[1]
    cal_file_side_1 = ct_cal_file.data.shape[2]
    for i_psf, psf in enumerate(psf_cube_in):
        loc_00 = np.argwhere(psf == ct_cal_file.data[i_psf][0][0])[0]
        assert np.all(psf[loc_00[0]:loc_00[0]+cal_file_side_0,
            loc_00[1]:loc_00[1]+cal_file_side_1] == ct_cal_file.data[i_psf])

    # Test 2: PSF positions and CT map
    # x location wrt FPM
    assert np.all(psf_loc_input[:,0] == ct_cal_file.ct_excam[0])
    # y location wrt FPM
    assert np.all(psf_loc_input[:,1] == ct_cal_file.ct_excam[1])
    # CT map
    assert np.all(ct_input == ct_cal_file.ct_excam[2])

    # Remove test CT cal file
    if os.path.exists(ct_cal_file.filepath):
        os.remove(ct_cal_file.filepath)

    print('Tests of the CT cal file passed')

if __name__ == '__main__':
    test_psf_pix_and_ct()
    test_fpm_pos()
    test_cal_file()


