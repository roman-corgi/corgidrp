import os
import shutil
import pytest
import numpy as np
import astropy.time as time
from astropy.io import fits
from scipy.signal import decimate
from astropy.modeling import models
from astropy.modeling.models import Gaussian2D
from termcolor import cprint
import corgidrp.caldb as caldb

import corgidrp
from corgidrp.mocks import (create_default_L3_headers, create_ct_psfs,
    create_ct_interp, create_ct_cal)
from corgidrp.data import (Image, Dataset, FpamFsamCal, CoreThroughputCalibration,
    CoreThroughputMap)
from corgidrp import corethroughput
from corgidrp import caldb

here = os.path.abspath(os.path.dirname(__file__))


def print_fail():
    cprint(' FAIL ', "black", "on_red")


def print_pass():
    cprint(' PASS ', "black", "on_green")


def setup_module():
    """
    Create datasets needed for the UTs
    """
    # Ensure default calibration files exist
    caldb.initialize()
    
    global FPAM_H_CT, FPAM_V_CT, FSAM_H_CT, FSAM_V_CT
    # Choose some H/V values for FPAM/FSAM  during corethroughput observations
    FPAM_H_CT, FPAM_V_CT, FSAM_H_CT, FSAM_V_CT = 6854, 22524, 29471, 12120
    global n_radii, n_azimuths, max_angle
    global cfam_name
    cfam_name = '1F'
    # CT and coronagraphic datasets
    global dataset_ct, dataset_ct_syn, dataset_ct_interp, dataset_psf_eq_rad
    global dataset_cor
    # Arbitrary set of PSF locations to be tested in EXCAM pixels referred to (0,0)
    global psf_loc_in, psf_loc_syn
    global ct_in, ct_syn
   
    # Default headers
    prhd, exthd, errhdr, dqhdr = create_default_L3_headers()
    # DRP
    exthd['DRPCTIME'] = time.Time.now().isot
    exthd['DRPVERSN'] = corgidrp.__version__
    # cfam filter
    exthd['CFAMNAME'] = cfam_name

    # FPAM/FSAM during CT observing sequence
    exthd['FPAM_H'] = FPAM_H_CT
    exthd['FPAM_V'] = FPAM_V_CT
    exthd['FSAM_H'] = FSAM_H_CT
    exthd['FSAM_V'] = FSAM_V_CT
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
    dataset_pupil = [Image(pupil_image_1,pri_hdr = prhd, ext_hdr = exthd_pupil, err = err),
        Image(pupil_image_2,pri_hdr = prhd, ext_hdr = exthd_pupil, err = err)]
    data_ct += dataset_pupil
    # Total counts from the pupil images
    unocc_psf_norm = (pupil_image_1.sum()+pupil_image_2.sum())/2
    # Adjust for pupil vs. direct imaging lens transmission
    # This function was written and tested during TVAC tests. There's a test value
    # in test_psf_pix_and_ct() later on
    di_over_pil = corethroughput.di_over_pil_transmission(cfam_name=exthd['CFAMNAME'])
    unocc_psf_norm *= di_over_pil
    # Generate 100 psfs with fwhm=50 mas in band 1 (mock.py)
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
    data_ct += dataset_pupil
    # Known CT
    ct_syn = (4+3*8+2*16)/(0.5*(1+3)*400)/di_over_pil
    # Dataset
    dataset_ct_syn = Dataset(data_ct) 

    # Coronagraphic dataset (only headers will be used)
    # FPAM/FSAM
    # Choose some H/V values for FPAM/FSAM  during coronagraphic observations
    # These values are *different* than the ones in the dataset_ct defined before
    exthd['FPAM_H'] = FPAM_H_CT - 107
    exthd['FPAM_V'] = FPAM_V_CT + 37
    exthd['FSAM_H'] = FSAM_H_CT + 97
    exthd['FSAM_V'] = FSAM_V_CT - 135
    # FPM center
    exthd['STARLOCX'] = 509
    exthd['STARLOCY'] = 513
    data_cor = [Image(np.zeros([1024, 1024]), pri_hdr=prhd, ext_hdr=exthd, err=err)]
    # Set proper filename following CGI convention
    data_cor[0].filename = "cgi_0000000000000090526_20240101t1200000_l2b.fits"
    dataset_cor = Dataset(data_cor)

    # Dataset with some CT profile defined in create_ct_interp
    # Pupil image
    pupil_image = np.zeros([1024, 1024])
    # Set it to some known value for a selected range of pixels
    pupil_image[510:530, 510:530]=1
    prhd, exthd_pupil, errhdr, dqhdr = create_default_L3_headers()
    # DRP
    exthd_pupil['DRPCTIME'] = time.Time.now().isot
    exthd_pupil['DRPVERSN'] = corgidrp.__version__
    # cfam filter
    exthd_pupil['CFAMNAME'] = cfam_name
    # Add specific values for pupil images:
    # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'
    # Choose some H/V values for FPAM/FSAM  during corethroughput observations
    exthd_pupil['FPAM_H'] = FPAM_H_CT
    exthd_pupil['FPAM_V'] = FPAM_V_CT
    exthd_pupil['FSAM_H'] = FSAM_H_CT
    exthd_pupil['FSAM_V'] = FSAM_V_CT
    # Mock error
    err = np.ones([1024,1024])
    # Collect Images
    data_ct_interp = [Image(pupil_image,pri_hdr = prhd,
        ext_hdr = exthd_pupil, err = err)]
    # Set of off-axis PSFs with a CT profile defined in create_ct_interp
    # First, we need the CT FPM center to create the CT radial profile
    # We can use a miminal dataset to get to know it
    data_ct_interp += [data_psf[0]]
    ct_cal_tmp = corethroughput.generate_ct_cal(Dataset(data_ct_interp))
    # FPAM/FSAM
    fpam_fsam_cal = FpamFsamCal(os.path.join(corgidrp.default_cal_dir,
        'FpamFsamCal_2024-02-10T00.00.00.000.fits'))
    # FPM during the CT observations (different to the coronagraphic one since
    # FPAM/FSAM H/V values are different)
    fpm_ct = ct_cal_tmp.GetCTFPMPosition(dataset_cor, fpam_fsam_cal)[0]
    # Generate the mock data for CT interpolation knowing the CT FPM
    data_ct_interp = [Image(pupil_image,pri_hdr = prhd,
        ext_hdr = exthd_pupil, err = err)]
    # Synthetic psfs with known CT values (mock.py)
    n_radii = 9
    n_azimuths = 5
    max_angle = 2/3*np.pi
    data_ct_interp += create_ct_interp(
        n_radii=n_radii,
        n_azimuths=n_azimuths,
        max_angle=max_angle,
        fpm_x=fpm_ct[0],
        fpm_y=fpm_ct[1],
        norm=pupil_image_1.sum())[0]
    dataset_ct_interp = Dataset(data_ct_interp)

    # Needed for PSF interpolation
    # Dataset with two PSFs at equal radial distance from the CT FPM's center
    data_psf_tmp = [Image(pupil_image,pri_hdr = prhd,
        ext_hdr = exthd_pupil, err = err)]
    # We need to estimate the location of the FPM's center to create the PSFs
    # at given predefined locatioins
    data_psf_tmp += [data_psf[0]]
    ct_cal_tmp2 = corethroughput.generate_ct_cal(Dataset(data_psf_tmp))
    # FPM during the CT observations (different to the coronagraphic one since
    # FPAM/FSAM H/V values are different)
    fpm_ct_2 = ct_cal_tmp2.GetCTFPMPosition(dataset_cor, fpam_fsam_cal)[0]
    # Generate the mock data for CT interpolation knowing the CT FPM
    data_psf_eq_rad = [Image(pupil_image,pri_hdr = prhd,
        ext_hdr = exthd_pupil, err = err)]
    # Band 1 FWHM: Enough approximation (and to a good extent, irrelevant)
    fwhm_mas = 50
    imshape = (219, 219)
    y, x = np.indices(imshape)
    # Following astropy documentation:
    # Generate 2 PSFs with the same radial distance to (0,0) and different A
    fpm_ct_frac_2 = fpm_ct_2 % 1
    # Choose the location of the PSFs
    aa = 100
    bb = 20
    model_params = []
    # Fill out the four quadrants to have more of a variety with PSFs at equal radii
    for idx in range(4):
        aa_w_sgn = (-1)**idx * aa
        bb_w_sgn = (-1)**(idx//2) * bb
        model_params += [
            dict(amplitude=11,
                x_mean=imshape[0]//2 + fpm_ct_frac_2[0]+aa_w_sgn,
                y_mean=imshape[1]//2 + fpm_ct_frac_2[1]+bb_w_sgn,
                x_stddev=fwhm_mas/21.8/2.335,
                y_stddev=fwhm_mas/21.8/2.335),
            dict(amplitude=23,
                x_mean=imshape[0]//2 + fpm_ct_frac_2[0]+bb_w_sgn,
                y_mean=imshape[1]//2 + fpm_ct_frac_2[1]+aa_w_sgn,
                x_stddev=fwhm_mas/21.8/2.335,
                y_stddev=fwhm_mas/21.8/2.335)]
    model_list = [models.Gaussian2D(**kwargs) for kwargs in model_params]
    # Render models to image using full evaluation
    for model in model_list:
        psf = np.zeros(imshape)
        model.bounding_box = None
        model.render(psf)
        image = np.zeros([1024, 1024])
        # Insert PSF 
        image[int(fpm_ct_2[1])-imshape[1]//2:int(fpm_ct_2[1])+imshape[1]//2+1,
            int(fpm_ct_2[0])-imshape[0]//2:int(fpm_ct_2[0])+imshape[0]//2+1] = psf
        # Build up the Dataset
        data_psf_eq_rad += [Image(image,pri_hdr=prhd, ext_hdr=exthd, err=err)]    

    dataset_psf_eq_rad = Dataset(data_psf_eq_rad)

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
    atol_psf_loc = 0.005
    atol_ct = 0.01

    test_result = np.all(np.abs(diff_psf_loc) <= atol_psf_loc)
    assert test_result
    # Print out the result
    print('\nGaussian position from estimate_psf_pix_and_ct() is correct to within %.3f pixels: ' % (atol_psf_loc), end='')
    print_pass() if test_result else print_fail()

    # comparison between I/O values (<=1% due to pixelization effects vs.
    # expected analytical value)
    test_result = np.all(np.abs(ct_est-ct_in) <= atol_ct)
    assert test_result
    # Print out the result
    print('\nGaussian CT value from estimate_psf_pix_and_ct() is correct to within %.1f%%: ' % (atol_ct*100), end='')
    print_pass() if test_result else print_fail()

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

    test_result = np.all(psf_loc_est[0] == psf_loc_syn)
    assert test_result
    # Print out the result
    print('Synthetic PSF position from estimate_psf_pix_and_ct() is exactly correct: ', end='')
    print_pass() if test_result else print_fail()

    atol_ct = 1e-16
    test_result = np.abs(ct_est[0]-ct_syn) < atol_ct
    assert test_result
    # Print out the result
    print('Synthetic PSF CT value from estimate_psf_pix_and_ct() is correct to within %.1g: ' % (atol_ct), end='')
    print_pass() if test_result else print_fail()

    print('Tests about PSF locations and CT values passed')

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
    # FPAM/FSAM transformations
    fpam_fsam_cal = FpamFsamCal(os.path.join(corgidrp.default_cal_dir,
        'FpamFsamCal_2024-02-10T00.00.00.000.fits'))
    
    fpam2excam_matrix, fsam2excam_matrix = fpam_fsam_cal.data
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
    FPM_center_pos_pix = np.array([dataset_cor[0].ext_hdr['STARLOCX'],
            dataset_cor[0].ext_hdr['STARLOCY']])
    FPAM_center_pos_um = np.array([dataset_cor[0].ext_hdr['FPAM_H'],
            dataset_cor[0].ext_hdr['FPAM_V']])
    FSAM_center_pos_um =  np.array([dataset_cor[0].ext_hdr['FSAM_H'],
            dataset_cor[0].ext_hdr['FSAM_V']])
    rng = np.random.default_rng(0)
    fpm_pos_result_list = []  # initialize
    n_trials = 10
    for ii in range(n_trials):
        # Random shift for Delta H/V in um
        delta_fpam_um = np.array([rng.uniform(1,10), rng.uniform(1,10)])
        delta_fsam_um = np.array([rng.uniform(1,10), rng.uniform(1,10)])
        # Update dataset_ct headers
        dataset_ct[0].ext_hdr['FPAM_H'] = dataset_cor[0].ext_hdr['FPAM_H'] + delta_fpam_um[0]
        dataset_ct[0].ext_hdr['FPAM_V'] = dataset_cor[0].ext_hdr['FPAM_V'] + delta_fpam_um[1]
        dataset_ct[0].ext_hdr['FSAM_H'] = dataset_cor[0].ext_hdr['FSAM_H'] + delta_fsam_um[0]
        dataset_ct[0].ext_hdr['FSAM_V'] = dataset_cor[0].ext_hdr['FSAM_V'] + delta_fsam_um[1]
        # Create CT cal file
        ct_cal_tmp = corethroughput.generate_ct_cal(dataset_ct)
        # Get CT FPM center
        fpam_ct_pix_out, fsam_ct_pix_out = \
            ct_cal_tmp.GetCTFPMPosition(
                dataset_cor,
                fpam_fsam_cal)
        # Expected shifts in EXCAM pixels
        delta_fpam_pix = fpam2excam_matrix @ delta_fpam_um
        delta_fsam_pix = fsam2excam_matrix @ delta_fsam_um
        # Compare output and input: Some of the random tests have differences
        # of ~1e-13, while others are exactly equal
        atol_fpm_pos = 1e-12
        fpam_ct_pix_in = FPM_center_pos_pix + delta_fpam_pix
        test_result_fpm_pos = np.all(fpam_ct_pix_out - fpam_ct_pix_in <= atol_fpm_pos)
        assert test_result_fpm_pos
        fpm_pos_result_list.append(test_result_fpm_pos)
        # print(f'Trial {ii}: FPM position from FpamFsamCal() is correct within {atol_fpm_pos}: ', end='')
        # print_pass() if test_result_fpm_pos else print_fail()

        atol_fs_pos = 1e-12
        fsam_ct_pix_in = FPM_center_pos_pix + delta_fsam_pix
        assert np.all(fsam_ct_pix_out - fsam_ct_pix_in <= atol_fs_pos)

    test_result_fpm_pos_all = np.all(fpm_pos_result_list)
    print(f'For all {n_trials} trials, FPM position from FpamFsamCal() is correct within {atol_fpm_pos}: ', end='')
    print_pass() if test_result_fpm_pos_all else print_fail()

    print('Tests about FPAM/FSAM to EXCAM passed')

def test_cal_file():
    """ Test creation of core throughput calibration file. """

    # Write core throughput calibration file
    ct_cal_file_in = corethroughput.generate_ct_cal(dataset_ct)
    test_dir = os.path.join(here, 'simdata')
    # Start with all clean
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(test_dir) 
    ct_cal_file_in.save(filedir=test_dir)

    # Check that the filename is what we expect
    ct_cal_filename = dataset_ct[-1].filename.replace("_l2b", "_ctp_cal")
    ct_cal_filepath = os.path.join(test_dir,ct_cal_filename)
    if os.path.exists(ct_cal_filepath) is False:
        raise IOError(f'Core throughput calibration file {ct_cal_filepath} does not exist.')
    # Load the calibration file to check it has the same data contents
    ct_cal_file_load = CoreThroughputCalibration(ct_cal_filepath)
    # Use allclose for floating point comparisons to account for bit depth differences
    assert np.allclose(ct_cal_file_load.data, ct_cal_file_in.data, rtol=1e-6, atol=1e-8)
    assert np.allclose(ct_cal_file_load.err, ct_cal_file_in.err, rtol=1e-6, atol=1e-8)
    assert np.array_equal(ct_cal_file_load.dq, ct_cal_file_in.dq)  # DQ is integer, use exact
    assert np.allclose(ct_cal_file_load.ct_excam, ct_cal_file_in.ct_excam, rtol=1e-6, atol=1e-8)
    assert np.allclose(ct_cal_file_load.ct_fpam, ct_cal_file_in.ct_fpam, rtol=1e-6, atol=1e-8)
    assert np.allclose(ct_cal_file_load.ct_fsam, ct_cal_file_in.ct_fsam, rtol=1e-6, atol=1e-8)
    
    # This test checks that the CT cal file has the right information by making
    # sure that I=O (Note: the comparison b/w analytical predictions
    # vs. centroid/pixelized data is part of the tests on test_psf_pix_and_ct before)

    # Input values for PSF centers and corresponding CT values
    psf_loc_input, ct_input = \
        corethroughput.estimate_psf_pix_and_ct(dataset_ct)
    n_pos = len(psf_loc_input)

    # Test: open calibration file
    try:
        ct_cal_file = corgidrp.data.CoreThroughputCalibration(ct_cal_file_in.filepath)
    except:
        raise IOError('CT cal file was not saved')

    test_slice_count = n_pos == ct_cal_file.data.shape[0]
    print(f'Number of slices, N = {n_pos}, in datacube from generate_ct_cal() is correct: ', end='')
    print_pass() if test_slice_count else print_fail()

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

    # Verify that the PSF images are best centered at each set of coordinates.
    test_result_psf_max_row = []  # intialize
    test_result_psf_max_col = []  # intialize
    row_expected = cal_file_side_0//2
    col_expected = cal_file_side_1//2
    for i_psf in range(n_pos):
        psf = ct_cal_file.data[i_psf, :, :]
        index_flat = np.argmax(psf)
        row_index, col_index = np.unravel_index(index_flat, psf.shape)
        test_result_psf_max_row.append(row_index == row_expected)
        test_result_psf_max_col.append(col_index == col_expected)

    test_result_recentering = np.all(test_result_psf_max_row) and np.all(test_result_psf_max_col)
    print('generate_ct_cal() returns PSFs that are best centered at each set of coordinates: ', end='')
    print_pass() if test_result_recentering else print_fail()

    # Test 2: PSF positions and CT map
    # Check EXTNAME is as expected
    if ct_cal_file.ct_excam_hdr['EXTNAME'] != 'CTEXCAM':
        raise ValueError('The extension name of the CT values on EXCAM is not correct')
    # x location wrt FPM (use allclose for float32 precision differences)
    assert np.allclose(psf_loc_input[:,0], ct_cal_file.ct_excam[0], rtol=1e-6, atol=1e-8)
    # y location wrt FPM
    assert np.allclose(psf_loc_input[:,1], ct_cal_file.ct_excam[1], rtol=1e-6, atol=1e-8)
    # CT map
    assert np.allclose(ct_input, ct_cal_file.ct_excam[2], rtol=1e-6, atol=1e-8)

    # Test 3: FPAM and FSAM positions during the CT observations are present
    # Check EXTNAME is as expected
    if ct_cal_file.ct_fpam_hdr['EXTNAME'] != 'CTFPAM':
        raise ValueError('The extension name of the FPAM values on EXCAM is not correct')
    # Use allclose for float32 precision differences
    assert np.allclose(ct_cal_file.ct_fpam, np.array([dataset_ct[0].ext_hdr['FPAM_H'],
        dataset_ct[0].ext_hdr['FPAM_V']]), rtol=1e-6, atol=1e-8)
    if ct_cal_file.ct_fsam_hdr['EXTNAME'] != 'CTFSAM':
        raise ValueError('The extension name of the FSAM values on EXCAM is not correct')
    assert np.allclose(ct_cal_file.ct_fsam, np.array([dataset_ct[0].ext_hdr['FSAM_H'],
        dataset_ct[0].ext_hdr['FSAM_V']]), rtol=1e-6, atol=1e-8)
    
    # Remove test CT cal file
    if os.path.exists(ct_cal_file.filepath):
        os.remove(ct_cal_file.filepath)

    print('Tests about the CT cal file passed')

def test_ct_interp():
    """ Tests the interpolation within the standard range by popping out data
    points and checking that the interpolation is < 5% error. The core
    throughput changes linearly across the radius.
    """

    # Generate core throughput calibration file
    ct_cal_in = corethroughput.generate_ct_cal(dataset_ct_interp)
    # Get CT FPM center
    # FPAM/FSAM
    fpam_fsam_cal = FpamFsamCal(os.path.join(corgidrp.default_cal_dir,
        'FpamFsamCal_2024-02-10T00.00.00.000.fits'))
    fpam_ct_pix = ct_cal_in.GetCTFPMPosition(dataset_cor, fpam_fsam_cal)[0]
    # Reference grid to test the interpolation: wrt CT FPM because the positions
    # used to interpolate the CT are, by definition, wrt the FPM
    x_grid = ct_cal_in.ct_excam[0,:] - fpam_ct_pix[0]
    y_grid = ct_cal_in.ct_excam[1,:] - fpam_ct_pix[1]
    core_throughput = ct_cal_in.ct_excam[2,:]
    
    # In this test, we will estimate the CT at a location that agrees with one
    # of the locations used to build the CT interpolation set by removing
    # the data point and estimating the CT with the remaining unchanged set

    # Remember that all the datasets will share the same pupil image. Only one
    # off-axis PSF is removed
    pupil_image = np.zeros([1024, 1024])
    # Set it to some known value for a selected range of pixels
    pupil_image[510:530, 510:530]=1
    prhd, exthd_pupil, errhdr, dqhdr = create_default_L3_headers()
    # DRP
    exthd_pupil['DRPCTIME'] = time.Time.now().isot
    exthd_pupil['DRPVERSN'] = corgidrp.__version__
    # cfam filter
    exthd_pupil['CFAMNAME'] = cfam_name
    # Add specific values for pupil images:
    # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'
    # Choose some H/V values for FPAM/FSAM  during corethroughput observations
    # Remember the dataset_cor used in InterpolateCT() below has different values
    # to simulate different values of the FPM center during coronagraphic and
    # core throughput observing sequences
    exthd_pupil['FPAM_H'] = FPAM_H_CT
    exthd_pupil['FPAM_V'] = FPAM_V_CT
    exthd_pupil['FSAM_H'] = FSAM_H_CT
    exthd_pupil['FSAM_V'] = FSAM_V_CT
    # Mock error
    err = np.ones([1024,1024])
    # Generate random indices between 0 and the number of radii and azimuths,
    # excluding the edge cases 
    n_random = 50
    rtol_ct = 0.05
    ct_result_list = []  # initialize
    # Set seed for reproducibility of test data
    rng = np.random.default_rng(0)
    for idx in range(n_random):
        random_index_radius = rng.choice(np.arange(1, n_radii-1), 1)
        random_index_az = rng.choice(np.arange(1, n_azimuths-1), 1)
     
        #Convert these to flattned indices
        random_indices_flat = random_index_radius + random_index_az*n_radii
        
        # Record the missing value
        missing_x = x_grid[random_indices_flat]
        missing_y = y_grid[random_indices_flat]
        missing_core_throughput = core_throughput[random_indices_flat]
        # Generate CT dataset w/o the latter (needed to call the interpolant
        # without this location)
        # Dataset for CT map interpolation: pupil images plus off-axis PSFs
        data_ct = [Image(pupil_image,pri_hdr = prhd, ext_hdr = exthd_pupil, err = err)]
        data_ct += create_ct_interp(
            n_radii=n_radii,
            n_azimuths=n_azimuths,
            max_angle=max_angle,
            norm=pupil_image.sum(),
            fpm_x=fpam_ct_pix[0],
            fpm_y=fpam_ct_pix[1],
            pop_index=random_indices_flat)[0]
        dataset_ct_tmp = Dataset(data_ct)
        # Generate core throughput calibration file
        ct_cal_tmp = corethroughput.generate_ct_cal(dataset_ct_tmp)
        # Now we can interpolate the missing values
        # Test with linear mapping of radii
        interpolated_value = ct_cal_tmp.InterpolateCT(
            missing_x, missing_y, dataset_cor, fpam_fsam_cal, logr=False)[0]
        # Good to within 5%
        test_result_ct = interpolated_value == pytest.approx(missing_core_throughput, rel=rtol_ct)
        ct_result_list.append(test_result_ct)
        print(f'Trial {idx}: Core throughput estimate is correct: {interpolated_value} +/- {100*rtol_ct}% relative: ', end='')
        print('') if test_result_ct else print_fail()
        assert test_result_ct, 'Error more than 5% (linear radii mapping)'
        # Test with radii mapped into their logarithmic values before
        # constructing the interpolant 
        interpolated_value_log = ct_cal_tmp.InterpolateCT(
            missing_x, missing_y, dataset_cor, fpam_fsam_cal, logr=True)[0]
        # Good to within 2%
        assert interpolated_value_log == pytest.approx(missing_core_throughput, rel=rtol_ct), 'Error more than 5% (logarithmic radii mapping)'

    test_result_ct_all = np.all(ct_result_list)
    print(f'All {n_random} CT estimates are correct to within {100*rtol_ct}% relative: ', end='')
    print_pass() if test_result_ct_all else print_fail()

    # Test that if the radius is out of the range then an error is thrown
    # Pick a data point that is out of the range. For instance, set y to zero
    # and x to a value that is greater than the maximum radius
    radii = np.sqrt(x_grid**2 + y_grid**2)
    with pytest.raises(ValueError):
        # Too Big
        ct_cal_tmp.InterpolateCT(radii.max()+1, 0, dataset_cor, fpam_fsam_cal) 
             
    with pytest.raises(ValueError):
        #Too small
        ct_cal_tmp.InterpolateCT(0.9*radii.min(), 0, dataset_cor, fpam_fsam_cal)

    # Test that something with an azimuth out of range returns the same result
    # as within the range
    azimuths = np.arctan2(y_grid, x_grid)
    azimuths -= azimuths.min()
    x_new_out = 0.9*np.max(radii)*np.cos(np.max(azimuths)+0.1)
    y_new_out = 0.9*np.max(radii)*np.sin(np.max(azimuths)+0.1)
    interpolated_value_out = ct_cal_tmp.InterpolateCT(
        x_new_out, y_new_out, dataset_cor, fpam_fsam_cal)[0]

    x_new_in = 0.9*np.max(radii)*np.cos(0.1)
    y_new_in = 0.9*np.max(radii)*np.sin(0.1)
    interpolated_value_in = ct_cal_tmp.InterpolateCT(
        x_new_in, y_new_in, dataset_cor, fpam_fsam_cal)[0]

    assert interpolated_value_out == pytest.approx(interpolated_value_in, abs=0.01), "Error more than 1% error"
    # Make sure it still works with a non-zero starting azimuth: min_angle below
    data_ct = [Image(pupil_image,pri_hdr = prhd, ext_hdr = exthd_pupil,
            err = err)]
    data_ct_interp = create_ct_interp(
        n_radii=n_radii,
        n_azimuths=n_azimuths,
        min_angle=-0.1,
        max_angle=max_angle,
        fpm_x=fpam_ct_pix[0],
        fpm_y=fpam_ct_pix[1],
        norm=pupil_image.sum())[0]
    data_ct += data_ct_interp
    dataset_ct_az = Dataset(data_ct)
    # Generate core throughput calibration file
    ct_cal_az = corethroughput.generate_ct_cal(dataset_ct_az)

    # Out of range of the new shifted azimuths
    x_az_out = 0.9*np.max(radii) * np.cos(max_angle + 0.2)
    y_az_out = 0.9*np.max(radii) * np.sin(max_angle + 0.2)
    interpolated_value_az_out = ct_cal_az.InterpolateCT(
        x_az_out, y_az_out, dataset_cor, fpam_fsam_cal)[0]

    # In range of the new shifted azimuths
    x_az_in = 0.9*np.max(radii) * np.cos(0.2)
    y_az_in = 0.9*np.max(radii) * np.sin(0.2)
    interpolated_value_az_in = ct_cal_az.InterpolateCT(
        x_az_in, y_az_in, dataset_cor, fpam_fsam_cal)[0]
    
    assert interpolated_value_az_out == pytest.approx(interpolated_value_az_in, abs=0.01), "Error more than 1% error"

    print('Tests about CT interpolation passed')

def test_get_1d_ct():
    """Test that corethroughput.get_1d_ct() produces an array of the correct 
    shape and returns the expected PSF for each position."""

    d = 2.36 #m
    lam = 573.8e-9 #m
    pixscale_arcsec = 0.0218
    fwhm_mas = 1.22 * lam / d * 206265 * 1000
    
    # Test where DETPIX0X/Y = (0,0)
    nx,ny = (5,5)
    cenx, ceny = (25.5,30.5)
    ctcal = create_ct_cal(fwhm_mas,
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    pri_hdr = fits.Header()
    ext_hdr = fits.Header()
    ext_hdr["STARLOCX"] = 25.
    ext_hdr["STARLOCY"] = 30.
    frame = Image(np.zeros([80,80]),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)
    
    seps = [0.,1.,1.41,2.,3.,4.]

    expected_args = [12,7,6,2,0,0]

    ct_1d = corethroughput.get_1d_ct(ctcal,frame,
                                     seps)
    
    assert ct_1d.shape == (2,len(seps))

    for i,arg in enumerate(expected_args):
        assert ct_1d[1,i] == ctcal.ct_excam[2,arg]

    # Test where DETPIX0X/Y is nonzero:

    nx,ny = (5,5)
    cenx, ceny = (45.5,35.5)
    ctcal = create_ct_cal(fwhm_mas,
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    pri_hdr = fits.Header()
    ext_hdr = fits.Header()
    ext_hdr["STARLOCX"] = 25.
    ext_hdr["STARLOCY"] = 30.
    ext_hdr["DETPIX0X"] = 20
    ext_hdr["DETPIX0Y"] = 5
    frame = Image(np.zeros([80,80]),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)
    
    seps = [0.,1.,1.41,2.,3.,4.]

    expected_args = [12,7,6,2,0,0]

    ct_1d = corethroughput.get_1d_ct(ctcal,frame,
                                     seps)
    
    assert ct_1d.shape == (2,len(seps))

    for i,arg in enumerate(expected_args):
        assert ct_1d[1,i] == ctcal.ct_excam[2,arg]

def test_ct_map():
    """ Tests the creation of a core throughput map. The method InterpolateCT()
      has its own unit test and can be considered as tested in the following. """

    # I run the test for two completely different CT datasets: 2D PSFs
    # randomly distributed (dataset_ct) and the one used for CT interpolation
    # that has a set of PSFs with pre-stablished CT and locations (dataset_ct_interp)

    for dataset in [dataset_ct_interp, dataset_ct, dataset_ct_interp]:
        # Generate core throughput calibration file
        ct_cal = corethroughput.generate_ct_cal(dataset)
    
        # FPAM/FSAM
        fpam_fsam_cal = FpamFsamCal(os.path.join(corgidrp.default_cal_dir,
            'FpamFsamCal_2024-02-10T00.00.00.000.fits'))
    
        # Create CT map for the HLC area (default)
        ct_map_def = corethroughput.create_ct_map(dataset_cor, fpam_fsam_cal,
            ct_cal)

        # CT values are within [0,1]
        assert ct_map_def.data[2].min() >= 0
        assert ct_map_def.data[2].max() <= 1
        # Verify CT values are within the range of the input CT values
        # Allow some minimum tolerance due to float64 numerical precision
        tolerance = 1e-14
        assert ct_map_def.data[2].min() >= ct_cal.ct_excam[2].min() - tolerance
        assert ct_map_def.data[2].max() <= ct_cal.ct_excam[2].max() + tolerance

        # Additional test to compate the CT map with the expected model from
        # create_ct_interp (predefined to be radial and linear). The other
        # dataset, dataset_ct is not suitable for this specific test because
        # it's made of random 2D Gaussians
        if dataset == dataset_ct_interp:
            r_def = np.sqrt(ct_map_def.data[0]**2 + ct_map_def.data[1]**2)
            ct_def = r_def/r_def.max()
            # create_ct_interp did not include the pupil lens to imaging lens ratio
            ct_def /= corethroughput.di_over_pil_transmission(cfam_name)
  
            # Differences below 1% are good
            assert ct_map_def.data[2] == pytest.approx(ct_def, abs=0.01), 'Differences are greater than 1%' 
    
        # Test the ability to parse some user-defined locations
        # If the target pixels are the same as the ones in the CT file, the
        # locations *and* CT values must agree with the ones in the CT file
        # Get FPM's center during CT observations
        ct_fpm = ct_cal.GetCTFPMPosition(dataset_cor, fpam_fsam_cal)[0]
        target_pix = [ct_cal.ct_excam[0] - ct_fpm[0],
            ct_cal.ct_excam[1] - ct_fpm[1]]
        ct_map_targ = corethroughput.create_ct_map(dataset_cor, fpam_fsam_cal,
            ct_cal, target_pix=target_pix)
    
        # All locations must have a valid CT value
        assert target_pix[0] == pytest.approx(ct_map_targ.data[0], abs=1e-14)
        assert target_pix[1] == pytest.approx(ct_map_targ.data[1], abs=1e-14)
        # CT values must be the same
        assert ct_cal.ct_excam[2] == pytest.approx(ct_map_targ.data[2], abs=1e-14)

    # Test it can be saved
    test_dir = os.path.join(here, 'simdata')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(test_dir)
    ct_map_def.save(filedir=test_dir)
    assert os.path.exists(ct_map_def.filepath), f"File not found: {ct_map_def.filepath}"

    # Add open the file and compare content
    ct_map_saved = fits.open(ct_map_def.filepath)
    # CT map values: (x, y, CT) for each location
    assert np.all(ct_map_saved[1].data == ct_map_def.data)
    # ERR
    assert np.all(ct_map_saved[2].data == ct_map_def.err)
    # DQ
    assert np.all(ct_map_saved[3].data == ct_map_def.dq)

    print('Tests about CT map passed')

def test_psf_interp():
    """
    Test the ability to recover a PSF at a given (x,y) location on HLC in a
    coronagraphic observation given a CT calibration file and the PAM
    transformation from encoder values to EXCAM pixels.
    """
    # Test 1/ Equal radial distance should retrieve the one with the nearest
    # angular distance
    # Generate core throughput calibration file
    ct_cal_eq = corethroughput.generate_ct_cal(dataset_psf_eq_rad)

    # We need to refer the PSF locations to the FPM's center
    # Get PAM cal file
    fpam_fsam_cal = FpamFsamCal(os.path.join(corgidrp.default_cal_dir,
        'FpamFsamCal_2024-02-10T00.00.00.000.fits'))
    # Get CT FPM's center
    fpam_ct_pix_eq = ct_cal_eq.GetCTFPMPosition(dataset_cor, fpam_fsam_cal)[0]
    # PSF locations with respect to the FPM's center. EXCAM pixels
    x_ct_eq = ct_cal_eq.ct_excam[0] - fpam_ct_pix_eq[0]
    y_ct_eq = ct_cal_eq.ct_excam[1] - fpam_ct_pix_eq[1]
    r_ct_eq = np.sqrt(x_ct_eq**2 + y_ct_eq**2)

    # Test with a location with the same radius and an equally (absolute) angular
    # distance from two PSFs in the input dataset: It should be the average value
    # In any interpolation beyind the nearest, there will be some
    # weighting factor that takes into account the distances
    x_mid = np.sqrt(100**2+20**2)/np.sqrt(2)
    y_mid = x_mid
    psf_interp = ct_cal_eq.GetPSF(x_mid, y_mid, dataset_cor, fpam_fsam_cal)[0]
    assert np.all(psf_interp == 0.5*(ct_cal_eq.data[0]+ct_cal_eq.data[1]))

    # Test with target location with the same radius as two PSFs in the input
    # dataset, but at a different angular location closer to one of them
    # Closer to the 1st one (arbitrary small shift, one of many possible choices)
    x_targ = r_ct_eq[0]*np.cos(np.arctan(y_ct_eq[0]/x_ct_eq[0])-0.05)
    y_targ = r_ct_eq[0]*np.sin(np.arctan(y_ct_eq[0]/x_ct_eq[0])-0.05)
    psf_interp = ct_cal_eq.GetPSF(x_targ, y_targ, dataset_cor, fpam_fsam_cal)[0]
    assert np.all(psf_interp == ct_cal_eq.data[0])
   
    # Closer to the 2nd one (arbitrary small shift, one of many possible choices)
    x_targ = r_ct_eq[1]*np.cos(np.arctan(y_ct_eq[1]/x_ct_eq[1])+0.06)
    y_targ = r_ct_eq[1]*np.sin(np.arctan(y_ct_eq[1]/x_ct_eq[1])+0.06)
    psf_interp = ct_cal_eq.GetPSF(x_targ, y_targ, dataset_cor, fpam_fsam_cal)[0]
    assert np.all(psf_interp == ct_cal_eq.data[1])
    
    # Next tests: Generate core throughput calibration file with more PSFs
    ct_cal = corethroughput.generate_ct_cal(dataset_ct_interp)

    # We need to refer the PSF locations to the FPM's center
    # Get PAM cal file
    fpam_fsam_cal = FpamFsamCal(os.path.join(corgidrp.default_cal_dir,
        'FpamFsamCal_2024-02-10T00.00.00.000.fits'))
    # Get CT FPM's center
    fpam_ct_pix = ct_cal.GetCTFPMPosition(dataset_cor, fpam_fsam_cal)[0]
    # PSF locations with respect to the FPM's center. EXCAM pixels
    x_ct = ct_cal.ct_excam[0,:] - fpam_ct_pix[0]
    y_ct = ct_cal.ct_excam[1,:] - fpam_ct_pix[1]
    # Radial distance
    r_ct = np.sqrt(x_ct**2 + y_ct**2)

    # Number of positions to test (PSF stamps are small 15x15 pixels)
    n_rand = 200
    rng = np.random.default_rng(0)

    # Test 2: Check that if the location is below or above the range of radial distances
    # in the input CT dataset used to generate the CT cal file, it fails
    # Test 2.1/ Below minimum radial distance:
    x_below = rng.uniform(0, r_ct.min()-0.2, n_rand)
    y_below = r_ct.min() - x_below 
    # All must fail
    with pytest.raises(ValueError):
        ct_cal.GetPSF(x_below, y_below, dataset_cor, fpam_fsam_cal)
    # Test 2.2/ Beyond maximum radial distance 
    x_beyond = rng.uniform(r_ct.max()+0.2, 2*r_ct.max(), n_rand)
    y_beyond = x_beyond - r_ct.max()
    # All must fail
    with pytest.raises(ValueError):
        ct_cal.GetPSF(x_beyond, y_beyond, dataset_cor, fpam_fsam_cal)

    # Test 3/ Choose some arbitrary positions and check the returned PSF is the
    # same as the nearest one in the CT cal file:
    
    # Array of random numbers, covering all quadrants with radial distances from
    # the FPM within the range of the CT cal file including some locations that
    # are invalid
    r_test = rng.uniform(r_ct.min()-1, r_ct.max()+1, n_rand)
    az_test = rng.uniform(0, 2*np.pi, n_rand)
    x_test = r_test * np.cos(az_test)
    y_test = r_test * np.sin(az_test)
    # Number of locations outside the range of valid radial distances
    n_invalid = ((r_test<r_ct.min()) + (r_test>r_ct.max())).sum()
    if n_invalid == 0:
        print('Target dataset should have some locations outside the radial range of the input dataset')

    # Interpolated PSFs
    interpolated_PSF, x_out, y_out = ct_cal.GetPSF(x_test, y_test, dataset_cor, fpam_fsam_cal)
    # Test that the number of interpolated PSFs is the same as the number of
    # valid positions
    if n_invalid + len(interpolated_PSF) != n_rand:
        raise Exception('Inconsistent number of interpolated PSFs')

    # Radial distance
    r_ct = np.sqrt(x_ct**2 + y_ct**2)
    for i_psf in range(len(x_out)):
        # Radial distance difference with the input dataset
        r_out = np.sqrt(x_out[i_psf]**2 + y_out[i_psf]**2)
        # Test agreement between interpolated PSFs and nearest one in the input set
        # Remember agreement to bin radial distances to 1/10th of a pixel in the
        # nearest-polar method
        diff_r_abs = np.round(10*np.abs(r_out - r_ct)/10)
        idx_near = np.argwhere(diff_r_abs == diff_r_abs.min())
        # If there's more than one case, check the interpolated PSF is the one
        # that has the shortest angular distance to the ones in the input dataset
        # with the same radial distance, or if there are two such locations
        # (half angle), the output should be the average of both PSFs (agreement)
        if len(idx_near) > 1:
            # Difference in angle b/w target and grid
            # We want to distinguish PSFs at different quadrants
            az_grid = np.arctan2(y_ct[idx_near], x_ct[idx_near])
            az_cor = np.arctan2(y_out[i_psf], x_out[i_psf])
            # Flatten into a 1-D array
            diff_az_abs = np.abs(az_cor - az_grid).transpose()[0]
            # Azimuth binning consistent with the binning of the radial distance
            bin_az_fac = 1/10/r_out
            diff_az_abs = bin_az_fac * np.round(diff_az_abs/bin_az_fac)
            # Closest angular location to the target location within equal radius
            idx_near_az = np.argwhere(diff_az_abs == diff_az_abs.min())
            # If there are two locations (half angle), choose the average (agreement)
            if len(idx_near_az) == 2:
                assert np.all(interpolated_PSF[i_psf] ==
                    ct_cal.data[idx_near[idx_near_az]].mean(axis=0))
            # Otherwise, this is the PSF
            elif len(idx_near_az) == 1:
                assert np.all(interpolated_PSF[i_psf] ==
                    ct_cal.data[idx_near[idx_near_az[0]]])
            else:
                raise ValueError(f'There are {len(idx_near_az):d} PSFs ',
                            'equally near the target PSF. This should not happen.')
        # Otherwise this is the interpolated PSF
        else:
            assert np.all(interpolated_PSF[i_psf] == ct_cal.data[idx_near[0]])

    print('Tests about PSF interpolation passed')

def teardown_module():
    """
    Deletes variables
    """
    global FPAM_H_CT, FPAM_V_CT, FSAM_H_CT, FSAM_V_CT
    del FPAM_H_CT, FPAM_V_CT, FSAM_H_CT, FSAM_V_CT
    global n_radii, n_azimuths, max_angle
    del n_radii, n_azimuths, max_angle
    global cfam_name
    del cfam_name
    # CT and coronagraphic datasets
    global dataset_ct, dataset_ct_syn, dataset_ct_interp, dataset_psf_eq_rad
    del dataset_ct, dataset_ct_syn, dataset_ct_interp, dataset_psf_eq_rad
    global dataset_cor
    del dataset_cor
    # Arbitrary set of PSF locations to be tested in EXCAM pixels referred to (0,0)
    global psf_loc_in, psf_loc_syn
    del psf_loc_in, psf_loc_syn
    # CT values
    global ct_in, ct_syn
    del ct_in, ct_syn

if __name__ == '__main__':
    setup_module()
    test_psf_pix_and_ct()
    test_fpm_pos()
    test_cal_file()
    test_ct_interp()
    test_get_1d_ct()
    test_ct_map()
    test_psf_interp()
