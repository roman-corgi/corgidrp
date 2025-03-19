import os
import pytest
import numpy as np
import astropy.time as time
from astropy.io import fits
from scipy.signal import decimate

import corgidrp
import corgidrp.data as data
from corgidrp.mocks import (create_default_L3_headers, create_ct_psfs,
    create_ct_interp)
from corgidrp.data import Image, Dataset, FpamFsamCal, CoreThroughputCalibration
from corgidrp import corethroughput

# If a run has crashed before being able to remove the test CT cal file,
# remove it before importing caldb, which scans for new entries in the cal folder
ct_cal_test_file = 'CoreThroughputCalibration_2025-02-15T00:00:00.fits'
if os.path.exists(os.path.join(corgidrp.default_cal_dir, ct_cal_test_file)):
    os.remove(os.path.join(corgidrp.default_cal_dir, ct_cal_test_file))
from corgidrp import caldb

here = os.path.abspath(os.path.dirname(__file__))

def setup_module():
    """
    Create datasets needed for the UTs
    """
    global FPAM_H_CT, FPAM_V_CT, FSAM_H_CT, FSAM_V_CT
    # Choose some H/V values for FPAM/FSAM  during corethroughput observations
    FPAM_H_CT, FPAM_V_CT, FSAM_H_CT, FSAM_V_CT = 6854, 22524, 29471, 12120
    global n_radii, n_azimuths, max_angle
    global cfam_name
    cfam_name = '1F'
    # CT and coronagraphic datasets
    global dataset_ct, dataset_ct_syn, dataset_ct_interp
    global dataset_cor
    # Arbitrary set of PSF locations to be tested in EXCAM pixels referred to (0,0)
    global psf_loc_in, psf_loc_syn
    global ct_in, ct_syn
   
    # Default headers
    prhd, exthd = create_default_L3_headers()
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
    dataset_cor = Dataset(data_cor)

    # Dataset with some CT profile defined in create_ct_interp
    # Pupil image
    pupil_image = np.zeros([1024, 1024])
    # Set it to some known value for a selected range of pixels
    pupil_image[510:530, 510:530]=1
    prhd, exthd_pupil = create_default_L3_headers()
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
        'FpamFsamCal_2024-02-10T00:00:00.000.fits'))
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

def test_ct_map():
    """ Tests the creation of a core throughput map. The method InterpolateCT()
      has its own unit test and can be considered as tested in the following. """

    # I run the test for two completely different CT datasets: 2D PSFs
    # randomly distributed (dataset_ct) and the one used for CT interpolation
    # that has a set of PSFs with pre-stablished CT and locations (dataset_ct_interp)

    for dataset in [dataset_ct, dataset_ct_interp]:
        # Generate core throughput calibration file
        ct_cal = corethroughput.generate_ct_cal(dataset)
    
        # FPAM/FSAM
        fpam_fsam_cal = FpamFsamCal(os.path.join(corgidrp.default_cal_dir,
            'FpamFsamCal_2024-02-10T00:00:00.000.fits'))
    
        # Create CT map for the HLC area (default)
        ct_map_def = corethroughput.CreateCTMap(dataset_cor, fpam_fsam_cal, ct_cal)

        # CT values are within [0,1]
        assert ct_map_def[2].min() >= 0
        assert ct_map_def[2].max() <= 1
        # Verify CT values are within the range of the input CT values
        # Allow some minimum tolerance due to float64 numerical precision
        tolerance = 1e-14
        assert ct_map_def[2].min() >= ct_cal.ct_excam[2].min() - tolerance
        assert ct_map_def[2].max() <= ct_cal.ct_excam[2].max() + tolerance

        # Additional test to compate the CT map with the expected model from
        # create_ct_interp (predefined to be radial and linear). The other
        # dataset, dataset_ct is not suitable for this specific test because
        # it's made of random 2D Gaussians
        if dataset == dataset_ct_interp:
            r_def = np.sqrt(ct_map_def[0]**2 + ct_map_def[1]**2)
            ct_def = r_def/r_def.max()
            # create_ct_interp did not include the pupil lens to imaging lens ratio
            ct_def /= corethroughput.di_over_pil_transmission(cfam_name)
  
            # Differences below 1% are good
            assert ct_map_def[2] == pytest.approx(ct_def, abs=0.01), 'Differences are greater than 1%' 
    
        # Test the ability to parse some user-defined locations
        # If the target pixels are the same as the ones in the CT file, the
        # locations *and* CT values must agree with the ones in the CT file
        # Get FPM's center during CT observations
        ct_fpm = ct_cal.GetCTFPMPosition(dataset_cor, fpam_fsam_cal)[0]
        target_pix = [ct_cal.ct_excam[0] - ct_fpm[0],
            ct_cal.ct_excam[1] - ct_fpm[1]]
        ct_map_targ = corethroughput.CreateCTMap(dataset_cor, fpam_fsam_cal, ct_cal,
            target_pix=target_pix)
    
        # All locations must have a valid CT value
        assert target_pix[0] == pytest.approx(ct_map_targ[0], abs=1e-14)
        assert target_pix[1] == pytest.approx(ct_map_targ[1], abs=1e-14)
        # CT values must be the same
        assert ct_cal.ct_excam[2] == pytest.approx(ct_map_targ[2], abs=1e-14)

    # Test it can be saved
    default_filepath = os.path.join(corgidrp.default_cal_dir, 'ct_map.csv')
    ct_map_def = corethroughput.CreateCTMap(dataset_cor, fpam_fsam_cal, ct_cal,
        save = True)
    assert os.path.exists(default_filepath), f"File not found: {default_filepath}"

    # Add open the file and compare content
    ct_map_saved = np.loadtxt(default_filepath, delimiter=',')
    assert np.all(ct_map_saved == ct_map_def)

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
        'FpamFsamCal_2024-02-10T00:00:00.000.fits'))
    
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
    for _ in range(10):
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
        fpam_ct_pix_in = FPM_center_pos_pix + delta_fpam_pix
        assert np.all(fpam_ct_pix_out - fpam_ct_pix_in <= 1e-12)
        fsam_ct_pix_in = FPM_center_pos_pix + delta_fsam_pix
        assert np.all(fsam_ct_pix_out - fsam_ct_pix_in <= 1e-12)

    print('Tests about FPAM/FSAM to EXCAM passed')

def test_cal_file():
    """ Test creation of core throughput calibration file. """

    # Write core throughput calibration file
    ct_cal_file_in = corethroughput.generate_ct_cal(dataset_ct)
    # It's fine to use a hardcoded filename for UTs
    ct_cal_file_in.save(filedir=corgidrp.default_cal_dir, filename=ct_cal_test_file)

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
    # Check EXTNAME is as expected
    if ct_cal_file.ct_excam_hdr['EXTNAME'] != 'CTEXCAM':
        raise ValueError('The extension name of the CT values on EXCAM is not correct')
    # x location wrt FPM
    assert np.all(psf_loc_input[:,0] == ct_cal_file.ct_excam[0])
    # y location wrt FPM
    assert np.all(psf_loc_input[:,1] == ct_cal_file.ct_excam[1])
    # CT map
    assert np.all(ct_input == ct_cal_file.ct_excam[2])

    # Test 3: FPAM and FSAM positions during the CT observations are present
    # Check EXTNAME is as expected
    if ct_cal_file.ct_fpam_hdr['EXTNAME'] != 'CTFPAM':
        raise ValueError('The extension name of the FPAM values on EXCAM is not correct')
    assert np.all(ct_cal_file.ct_fpam == np.array([dataset_ct[0].ext_hdr['FPAM_H'],
        dataset_ct[0].ext_hdr['FPAM_V']]))
    if ct_cal_file.ct_fsam_hdr['EXTNAME'] != 'CTFSAM':
        raise ValueError('The extension name of the FSAM values on EXCAM is not correct')
    assert np.all(ct_cal_file.ct_fsam == np.array([dataset_ct[0].ext_hdr['FSAM_H'],
        dataset_ct[0].ext_hdr['FSAM_V']]))
    
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
        'FpamFsamCal_2024-02-10T00:00:00.000.fits'))
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
    prhd, exthd_pupil = create_default_L3_headers()
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
        # Good to within 2% 
        assert interpolated_value == pytest.approx(missing_core_throughput, abs=0.05), 'Error more than 5% (linear radii mapping)'
        # Test with radii mapped into their logarithmic values before
        # constructing the interpolant 
        interpolated_value_log = ct_cal_tmp.InterpolateCT(
            missing_x, missing_y, dataset_cor, fpam_fsam_cal, logr=True)[0]
        # Good to within 2%
        assert interpolated_value_log == pytest.approx(missing_core_throughput, abs=0.05), 'Error more than 5% (logarithmic radii mapping)'
        
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
    global dataset_ct, dataset_ct_syn, dataset_ct_interp
    del dataset_ct, dataset_ct_syn, dataset_ct_interp
    global dataset_cor
    del dataset_cor
    # Arbitrary set of PSF locations to be tested in EXCAM pixels referred to (0,0)
    global psf_loc_in, psf_loc_syn
    del psf_loc_in, psf_loc_syn
    # CT values
    global ct_in, ct_syn
    del ct_in, ct_syn

if __name__ == '__main__':
    test_psf_pix_and_ct()
    test_fpm_pos()
    test_cal_file()
    test_ct_interp()
    test_ct_map()
