import os, glob
import numpy as np
import pandas as pd
import shutil
import warnings

from astropy.io.fits import Header

import pytest

from corgidrp.data import Dataset, Image
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.pol as pol
import corgidrp.l2b_to_l3 as l2b_to_l3
import corgidrp.l3_to_l4 as l3_to_l4
import corgidrp.l4_to_tda as l4_to_tda
from corgidrp.pol import calc_stokes_unocculted

from corgidrp import star_center

from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning
from pyklip.klip import rotate



def test_image_splitting():
    """
    Create mock L2b polarimetric images, check that it is split correctly
    """

    # test autocropping WFOV
    ## generate mock data
    image_WP1_wfov = mocks.create_mock_l2b_polarimetric_image(dpamname='POL0', observing_mode='WFOV', left_image_value=1, right_image_value=2)
    image_WP2_wfov = mocks.create_mock_l2b_polarimetric_image(dpamname='POL45', observing_mode='WFOV', left_image_value=1, right_image_value=2)
    # modify err and dq value for testing
    image_WP1_wfov.err[0, 512, 340] = 1
    image_WP1_wfov.err[0, 512, 684] = 2
    image_WP1_wfov.dq[512, 340] = 1
    image_WP1_wfov.dq[512, 684] = 2
    image_WP2_wfov.err[0, 634, 390] = 1
    image_WP2_wfov.err[0, 390, 634] = 2
    image_WP2_wfov.dq[634, 390] = 1
    image_WP2_wfov.dq[390, 634] = 2
    input_dataset_wfov = data.Dataset([image_WP1_wfov, image_WP2_wfov])

    ## leave image_size parameter blank so the function automatically determines size
    output_dataset_autocrop_wfov = l2b_to_l3.split_image_by_polarization_state(input_dataset_wfov)

    ## checks that saving and loading the image doesn't cause any issues
    ### save
    test_dir = os.path.join(os.getcwd(), 'pol_output')
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(test_dir)
    output_dataset_autocrop_wfov.save(test_dir, ['wfov_pol_img_{0}.fits'.format(i) for i in range(len(output_dataset_autocrop_wfov))])
    ### load
    autocrop_wfov_filelist = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
    output_dataset_autocrop_wfov = data.Dataset(autocrop_wfov_filelist)

    ## create what the expected output data should look like
    radius_wfov = int(round((20.1 * ((0.8255 * 1e-6) / 2.363114) * 206265) / 0.0218))
    padding = 5
    img_size_wfov = 2 * (radius_wfov + padding)
    expected_output_autocrop_wfov = np.zeros(shape=(2, img_size_wfov, img_size_wfov))
    expected_err_autocrop_wfov = np.zeros(shape=(1, 2, img_size_wfov, img_size_wfov))
    expected_dq_autocrop_wfov = np.zeros(shape=(2, img_size_wfov, img_size_wfov))
    ## fill in expected values
    img_center_wfov = radius_wfov + padding
    y_wfov, x_wfov = np.indices([img_size_wfov, img_size_wfov])
    expected_output_autocrop_wfov[0, ((x_wfov-img_center_wfov)**2) + ((y_wfov-img_center_wfov)**2) <= radius_wfov**2] = 1
    expected_output_autocrop_wfov[1, ((x_wfov-img_center_wfov)**2) + ((y_wfov-img_center_wfov)**2) <= radius_wfov**2] = 2
    expected_err_autocrop_wfov[0, 0, img_center_wfov, img_center_wfov] = 1
    expected_err_autocrop_wfov[0, 1, img_center_wfov, img_center_wfov] = 2
    expected_dq_autocrop_wfov[0, img_center_wfov, img_center_wfov] = 1
    expected_dq_autocrop_wfov[1, img_center_wfov, img_center_wfov] = 2

    ## check that actual output is as expected
    assert output_dataset_autocrop_wfov.frames[0].data == pytest.approx(expected_output_autocrop_wfov)
    assert output_dataset_autocrop_wfov.frames[1].data == pytest.approx(expected_output_autocrop_wfov)
    # test err and dq cropping
    assert (output_dataset_autocrop_wfov.frames[0].err == expected_err_autocrop_wfov).all()
    assert (output_dataset_autocrop_wfov.frames[1].err == expected_err_autocrop_wfov).all()
    assert (output_dataset_autocrop_wfov.frames[0].dq == expected_dq_autocrop_wfov).all()
    assert (output_dataset_autocrop_wfov.frames[1].dq == expected_dq_autocrop_wfov).all()

    # test autocropping NFOV
    ## generate mock data
    image_WP1_nfov = mocks.create_mock_l2b_polarimetric_image(dpamname='POL0', observing_mode='NFOV', left_image_value=1, right_image_value=2)
    image_WP2_nfov = mocks.create_mock_l2b_polarimetric_image(dpamname='POL45', observing_mode='NFOV', left_image_value=1, right_image_value=2)
    image_WP1_nfov.err[0, 512, 340] = 1
    image_WP1_nfov.err[0, 512, 684] = 2
    image_WP1_nfov.dq[512, 340] = 1
    image_WP1_nfov.dq[512, 684] = 2
    image_WP2_nfov.err[0, 634, 390] = 1
    image_WP2_nfov.err[0, 390, 634] = 2
    image_WP2_nfov.dq[634, 390] = 1
    image_WP2_nfov.dq[390, 634] = 2
    input_dataset_nfov = data.Dataset([image_WP1_nfov, image_WP2_nfov])

    ## leave image_size parameter blank so the function automatically determines size
    output_dataset_autocrop_nfov = l2b_to_l3.split_image_by_polarization_state(input_dataset_nfov)
    ## create what the expected output data should look like
    radius_nfov = int(round((9.7 * ((0.5738 * 1e-6) / 2.363114) * 206265) / 0.0218))
    img_size_nfov = 2 * (radius_nfov + padding)
    expected_output_autocrop_nfov = np.zeros(shape=(2, img_size_nfov, img_size_nfov))
    expected_err_autocrop_nfov = np.zeros(shape=(1, 2, img_size_nfov, img_size_nfov))
    expected_dq_autocrop_nfov = np.zeros(shape=(2, img_size_nfov, img_size_nfov))
    ## fill in expected values
    img_center_nfov = radius_nfov + padding
    y_nfov, x_nfov = np.indices([img_size_nfov, img_size_nfov])
    expected_output_autocrop_nfov[0, ((x_nfov-img_center_nfov)**2) + ((y_nfov-img_center_nfov)**2) <= radius_nfov**2] = 1
    expected_output_autocrop_nfov[1, ((x_nfov-img_center_nfov)**2) + ((y_nfov-img_center_nfov)**2) <= radius_nfov**2] = 2
    expected_err_autocrop_nfov[0, 0, img_center_nfov, img_center_nfov] = 1
    expected_err_autocrop_nfov[0, 1, img_center_nfov, img_center_nfov] = 2
    expected_dq_autocrop_nfov[0, img_center_nfov, img_center_nfov] = 1
    expected_dq_autocrop_nfov[1, img_center_nfov, img_center_nfov] = 2

    ## check that actual output is as expected
    assert output_dataset_autocrop_nfov.frames[0].data == pytest.approx(expected_output_autocrop_nfov)
    assert output_dataset_autocrop_nfov.frames[1].data == pytest.approx(expected_output_autocrop_nfov)
    # test err and dq cropping
    assert (output_dataset_autocrop_nfov.frames[0].err == expected_err_autocrop_nfov).all()
    assert (output_dataset_autocrop_nfov.frames[1].err == expected_err_autocrop_nfov).all()
    assert (output_dataset_autocrop_nfov.frames[0].dq == expected_dq_autocrop_nfov).all()
    assert (output_dataset_autocrop_nfov.frames[1].dq == expected_dq_autocrop_nfov).all()

    # test cropping with alignment angle input
    image_WP1_custom_angle = mocks.create_mock_l2b_polarimetric_image(dpamname='POL0', observing_mode='NFOV', left_image_value=1, right_image_value=2, alignment_angle=5)
    image_WP2_custom_angle = mocks.create_mock_l2b_polarimetric_image(dpamname='POL45', observing_mode='NFOV', left_image_value=1, right_image_value=2, alignment_angle=40)
    input_dataset_custom_angle = data.Dataset([image_WP1_custom_angle, image_WP2_custom_angle])
    output_dataset_custom_angle = l2b_to_l3.split_image_by_polarization_state(input_dataset_custom_angle, alignment_angle_WP1=5, alignment_angle_WP2=40)

    ## check that actual output is as expected, should still be the same as the previous test since mock data is in NFOV mode
    assert output_dataset_custom_angle.frames[0].data == pytest.approx(expected_output_autocrop_nfov)
    assert output_dataset_custom_angle.frames[1].data == pytest.approx(expected_output_autocrop_nfov)

    # test NaN pixels
    img_size = 400
    output_dataset_custom_crop = l2b_to_l3.split_image_by_polarization_state(input_dataset_wfov, image_size=img_size)
    ## create what the expected output data should look like
    expected_output_WP1 = np.zeros(shape=(2, img_size, img_size))
    expected_output_WP2 = np.zeros(shape=(2, img_size, img_size))
    img_center = 200
    y, x = np.indices([img_size, img_size])
    expected_output_WP1[0, ((x-img_center)**2) + ((y-img_center)**2) <= radius_wfov**2] = 1
    expected_output_WP1[1, ((x-img_center)**2) + ((y-img_center)**2) <= radius_wfov**2] = 2
    expected_output_WP2[0, ((x-img_center)**2) + ((y-img_center)**2) <= radius_wfov**2] = 1
    expected_output_WP2[1, ((x-img_center)**2) + ((y-img_center)**2) <= radius_wfov**2] = 2
    expected_output_WP1[0, x >= 372] = np.nan
    expected_output_WP1[1, x <= 28] = np.nan
    expected_output_WP2[0, y <= x - 244] = np.nan
    expected_output_WP2[1, y >= x + 244] = np.nan
    ## check that the actual output is as expected
    assert output_dataset_custom_crop.frames[0].data == pytest.approx(expected_output_WP1, nan_ok=True)
    assert output_dataset_custom_crop.frames[1].data == pytest.approx(expected_output_WP2, nan_ok=True)

    # test that an error is raised if we set the image size too big
    with pytest.raises(ValueError):
        invalid_output = l2b_to_l3.split_image_by_polarization_state(input_dataset_wfov, image_size=682)
        
def test_calc_pol_p_and_pa_image(n_sim=100, nsigma_tol=3.):
    """
    Test `calc_pol_p_and_pa_image` using mock L4 Stokes cubes.

    This test verifies that the recovered fractional polarization (p)
    and electric-vector position angle (EVPA) are statistically consistent
    with the true input values within their propagated uncertainties.

    For each simulation, we compute normalized residuals:
        chi = (measured - true) / sigma
    for both p and EVPA. If the uncertainty propagation is correct,
    the chi distribution should follow N(0, 1).

    We then check that the median of the chi means is near zero,
    and the median of the chi standard deviations is near one.

    The tolerance (`nsigma_tol`) defines the acceptable deviation from
    ideal statistics in units of standard errors. For `n_sim` simulations,
    the expected fluctuations of the median and standard deviation of chi
    are approximately 1/sqrt(n_sim) and 1/sqrt(2*(n_sim-1)), respectively.
    Multiplying by `nsigma_tol` allows for a configurable confidence
    interval, e.g., `nsigma_tol=3` corresponds roughly to a 3-sigma limit
    on expected statistical deviations.
    """
    # --- Simulation parameters ---
    p_input = 0.1 + 0.2 * np.random.rand(n_sim)
    theta_input = 10.0 + 20.0 * np.random.rand(n_sim)

    # --- Containers for chi statistics ---
    p_chi_mean, p_chi_std = [], []
    evpa_chi_mean, evpa_chi_std = [], []

    for p, theta in zip(p_input, theta_input):

        # Generate mock Stokes cube
        Image_polmock = mocks.create_mock_stokes_image_l4(
            badpixel_fraction=0.0,
            fwhm=1e2,
            I0=1e10,
            p=p,
            theta_deg=theta
        )

        # Compute polarization products
        Image_pol = l4_to_tda.calc_pol_p_and_pa_image(Image_polmock)

        p_map = Image_pol.data[1]       # fractional polarization
        evpa_map = Image_pol.data[2]    # EVPA
        p_map_err = Image_pol.err[0][1]
        evpa_map_err = Image_pol.err[0][2]

        # Compute chi statistics
        p_chi = (p_map - p) / p_map_err
        evpa_chi = (evpa_map - theta) / evpa_map_err

        p_chi_mean.append(np.nanmedian(p_chi))
        p_chi_std.append(np.nanstd(p_chi))
        evpa_chi_mean.append(np.nanmedian(evpa_chi))
        evpa_chi_std.append(np.nanstd(evpa_chi))

    #print(np.median(p_chi_mean), np.median(p_chi_std),
    #      np.median(evpa_chi_mean), np.median(evpa_chi_std))
    tol_mean = 1. / np.sqrt(n_sim) * nsigma_tol
    tol_std = 1. / np.sqrt(2.*(n_sim-1.)) * nsigma_tol
    assert np.median(p_chi_mean) == pytest.approx(0.0, abs=tol_mean)
    assert np.median(p_chi_std) == pytest.approx(1.0, abs=tol_std)
    assert np.median(evpa_chi_mean) == pytest.approx(0.0, abs=tol_mean)
    assert np.median(evpa_chi_std) == pytest.approx(1.0, abs=tol_std)

def test_align_frames():
    """
    Test that polarimetric images are align correctly on the POL 0 subdataset
    """
    # Generate mock data
    injected_position_pol0 = [(2, - 1), (-2, 1)]
    injected_position_pol45 = [(1, - 2), (2, -1)]

    image_WP1_nfov_sp = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(dpamname='POL0',
     observing_mode='NFOV',
     left_image_value=1,
     right_image_value=2,
     star_center=injected_position_pol0,
     amplitude_multiplier=1000)
    image_WP1_nfov= mocks.create_mock_l2b_polarimetric_image(dpamname='POL0',
     observing_mode='NFOV',
     left_image_value=1,
     right_image_value=2)

    image_WP2_nfov_sp = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(dpamname='POL45',
     observing_mode='NFOV',
     left_image_value=1,
     right_image_value=2,
     star_center=injected_position_pol45,
     amplitude_multiplier=1000)
    image_WP2_nfov = mocks.create_mock_l2b_polarimetric_image(dpamname='POL45',
     observing_mode='NFOV',
     left_image_value=1,
     right_image_value=2)
    input_dataset_nfov = data.Dataset([image_WP1_nfov_sp, image_WP1_nfov, image_WP2_nfov_sp, image_WP2_nfov])
    input_dataset_autocrop_nfov = l2b_to_l3.split_image_by_polarization_state(input_dataset_nfov)
    
    # Find the star
    # Checks on finding the star are done in test_find_star.py
    dataset_with_center = l3_to_l4.find_star(input_dataset_autocrop_nfov, drop_satspots_frames=False)

    starloc_pol0 = (dataset_with_center.frames[0].ext_hdr['STARLOCX'], dataset_with_center.frames[0].ext_hdr['STARLOCY'])
    starloc_pol45 = (dataset_with_center.frames[2].ext_hdr['STARLOCX'], dataset_with_center.frames[2].ext_hdr['STARLOCY'])
    
                                              
    injected_x_slice_0, injected_y_slice_0 = (dataset_with_center.frames[0].data[0].shape[0]//2 + injected_position_pol0[0][0],
                                            dataset_with_center.frames[0].data[0].shape[1]//2 + injected_position_pol0[0][1])   
    
    injected_x_slice_45, injected_y_slice_45 = (dataset_with_center.frames[1].data[0].shape[0]//2 + injected_position_pol45[0][0],
                                                dataset_with_center.frames[1].data[0].shape[1]//2 + injected_position_pol45[0][1])

    assert np.isclose(injected_x_slice_45, starloc_pol45[0], atol=0.1), \
        f"Expected {injected_x_slice_45}, got {starloc_pol45[0]}"
    assert np.isclose(injected_y_slice_45, starloc_pol45[1], atol=0.1), \
        f"Expected {injected_y_slice_45}, got {starloc_pol45[1]}"

    # Test that the difference between the measured stars is the difference between the injected positions. 
    assert np.isclose( starloc_pol0[0] - starloc_pol45[0], injected_x_slice_0 - injected_x_slice_45, atol=0.1)
    assert np.isclose( starloc_pol0[1] - starloc_pol45[1], injected_y_slice_0 - injected_y_slice_45, atol=0.1)
    # Align the pol 45 data with the pol 0 data 
    output_dataset_aligned= l3_to_l4.align_polarimetry_frames(dataset_with_center)
    
    # Check that the pol 45 frames are now aligned on the pol 0 frames
    star_xy, list_spots_xy = star_center.star_center_from_satellite_spots(
        img_ref=output_dataset_aligned.frames[3].data[0],
        img_sat_spot=output_dataset_aligned.frames[2].data[0],
        star_coordinate_guess=(output_dataset_aligned.frames[3].data[0].shape[1]//2, output_dataset_aligned.frames[3].data[0].shape[0]//2),
        thetaOffsetGuess=0,
        satellite_spot_parameters=star_center.satellite_spot_parameters_defaults['NFOV']
    )       
    assert np.isclose(star_xy[0], starloc_pol0[0], atol=0.1), \
            f" Expected {starloc_pol0[0]}, got {star_xy[0]}"
    assert np.isclose(star_xy[1], starloc_pol0[1], atol=0.1), \
            f" Expected {starloc_pol0[1]}, got {star_xy[1]}"

    # Check that every frame is aligned on the same  on STARLOC
    starloc = []

    for frame in output_dataset_aligned.frames:
        starloc.append((frame.ext_hdr['STARLOCX'], frame.ext_hdr['STARLOCY']))

    assert all(location == starloc_pol0 for location in starloc), \
        "All frames should have the same star location."
                
def test_mueller_matrix_cal():
    '''
    Tests the creation of a Mueller Matrix calibration file from a mock dataset.
    '''
    
    #Get path to this file
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    #To test this we need this catalog with the same values as the test data also in corgidrp/data/stellar_polarization_database.csv
    path_to_pol_ref_file = os.path.join(current_file_path, "test_data","stellar_polarization_database.csv")

    q_instrumental_polarization = 0.5  # in percent
    u_instrumental_polarization = -0.1  # in percent
    q_efficiency = 0.8
    u_efficiency = 0.7
    uq_cross_talk = 0.05
    qu_cross_talk = 0.03

    mock_dataset = mocks.generate_mock_polcal_dataset(path_to_pol_ref_file,
                                           q_inst=q_instrumental_polarization,
                                           u_inst=u_instrumental_polarization,
                                           q_eff=q_efficiency,
                                           u_eff=u_efficiency,
                                           uq_ct=uq_cross_talk,
                                           qu_ct=qu_cross_talk)
    mock_dataset = l2b_to_l3.divide_by_exptime(mock_dataset)
    mock_dataset = l2b_to_l3.split_image_by_polarization_state(mock_dataset)
    stokes_dataset = pol.calc_stokes_unocculted(mock_dataset)

    #Run the Mueller matrix calibration function
    mueller_matrix = pol.generate_mueller_matrix_cal(stokes_dataset, path_to_pol_ref_file=path_to_pol_ref_file)

    #Check that the measured mueller matrix is close to the input values
    assert mueller_matrix.data[1,0] == pytest.approx(q_instrumental_polarization/100.0, abs=1e-3)
    assert mueller_matrix.data[2,0] == pytest.approx(u_instrumental_polarization/100.0, abs=1e-3)
    assert mueller_matrix.data[1,1] == pytest.approx(q_efficiency, abs=1e-3)
    assert mueller_matrix.data[2,2] == pytest.approx(u_efficiency, abs=1e-3)
    assert mueller_matrix.data[1,2] == pytest.approx(uq_cross_talk, abs=1e-3)
    assert mueller_matrix.data[2,1] == pytest.approx(qu_cross_talk, abs=1e-3)

    #Check that the type of mueller_matrix is correct
    assert isinstance(mueller_matrix, pol.MuellerMatrix)

    #Put in the ND filter and make sure the type is correct. 
    for framm in mock_dataset.frames:
        framm.ext_hdr["FPAMNAME"] = "ND225"
    mueller_matrix_nd = pol.generate_mueller_matrix_cal(stokes_dataset, path_to_pol_ref_file=path_to_pol_ref_file)
    assert isinstance(mueller_matrix_nd, pol.NDMuellerMatrix)

    #Make sure that if the dataset is mixed ND and non-ND an error is raised
    mock_dataset.frames[0].ext_hdr["FPAMNAME"] = "CLEAR"
    with pytest.raises(ValueError):
        mueller_matrix_mixed = pol.generate_mueller_matrix_cal(stokes_dataset, path_to_pol_ref_file=path_to_pol_ref_file)

def test_subtract_stellar_polarization():
    """
    Test that the subtract_stellar_polarization step function separates the input dataset by target star
    and correctly subtracts off the stellar polarization when given a dataset of L3 polarimetric images 
    """

    # define mueller matrices and stokes vectors
    nd_mueller_matrix = np.array([
        [0.5, 0.1, 0, 0],
        [0.1, -0.5, 0, 0],
        [0.05, 0.05, 0.5, 0],
        [0, 0, 0, 0.5]
    ])
    system_mueller_matrix = np.array([
        [0.9, -0.02, 0, 0],
        [0.01, -0.8, 0, 0],
        [0, 0, 0.8, 0.005],
        [0, 0, -0.01, 0.9]
    ])
    star_1_pol = np.array([1, 0, 0, 0])
    star_2_pol = np.array([1, -0.01, -0.02, 0])

    # create mock images and dataset
    # use gaussians as mock star, scale by polarized intensity to construct polarimetric data
    # add background noise to star to prevent divide by zero
    star_1 = mocks.gaussian_array(amp=100) + 0.001
    star_2 = mocks.gaussian_array(amp=150) + 0.001
    # give unocculted images with ND filter a roll angle of 30, images with FPM will have a roll angle of 45
    roll_unocculted = 30
    roll = 45
    # find polarized intensities
    star_1_nd_pol = nd_mueller_matrix @ pol.rotation_mueller_matrix(roll_unocculted) @ star_1_pol
    star_2_nd_pol = nd_mueller_matrix @ pol.rotation_mueller_matrix(roll_unocculted) @ star_2_pol
    star_1_fpm_pol = system_mueller_matrix @ pol.rotation_mueller_matrix(roll) @ star_1_pol
    star_2_fpm_pol = system_mueller_matrix @ pol.rotation_mueller_matrix(roll) @ star_2_pol
    star_1_nd_I_0 = (pol.lin_polarizer_mueller_matrix(0) @ star_1_nd_pol)[0]
    star_1_nd_I_45 = (pol.lin_polarizer_mueller_matrix(45) @ star_1_nd_pol)[0]
    star_1_nd_I_90 = (pol.lin_polarizer_mueller_matrix(90) @ star_1_nd_pol)[0]
    star_1_nd_I_135 = (pol.lin_polarizer_mueller_matrix(135) @ star_1_nd_pol)[0]
    star_2_nd_I_0 = (pol.lin_polarizer_mueller_matrix(0) @ star_2_nd_pol)[0]
    star_2_nd_I_45 = (pol.lin_polarizer_mueller_matrix(45) @ star_2_nd_pol)[0]
    star_2_nd_I_90 = (pol.lin_polarizer_mueller_matrix(90) @ star_2_nd_pol)[0]
    star_2_nd_I_135 = (pol.lin_polarizer_mueller_matrix(135) @ star_2_nd_pol)[0]
    star_1_fpm_I_0 = (pol.lin_polarizer_mueller_matrix(0) @ star_1_fpm_pol)[0]
    star_1_fpm_I_45 = (pol.lin_polarizer_mueller_matrix(45) @ star_1_fpm_pol)[0]
    star_1_fpm_I_90 = (pol.lin_polarizer_mueller_matrix(90) @ star_1_fpm_pol)[0]
    star_1_fpm_I_135 = (pol.lin_polarizer_mueller_matrix(135) @ star_1_fpm_pol)[0]
    star_2_fpm_I_0 = (pol.lin_polarizer_mueller_matrix(0) @ star_2_fpm_pol)[0]
    star_2_fpm_I_45 = (pol.lin_polarizer_mueller_matrix(45) @ star_2_fpm_pol)[0]
    star_2_fpm_I_90 = (pol.lin_polarizer_mueller_matrix(90) @ star_2_fpm_pol)[0]
    star_2_fpm_I_135 = (pol.lin_polarizer_mueller_matrix(135) @ star_2_fpm_pol)[0]
    # construct polarimetric data
    star_1_nd_wp1_data = np.array([star_1_nd_I_0 * star_1, star_1_nd_I_90 * star_1])
    star_1_nd_wp2_data = np.array([star_1_nd_I_45 * star_1, star_1_nd_I_135 * star_1])
    star_2_nd_wp1_data = np.array([star_2_nd_I_0 * star_2, star_2_nd_I_90 * star_2])
    star_2_nd_wp2_data = np.array([star_2_nd_I_45 * star_2, star_2_nd_I_135 * star_2])
    # combine star and companion data for image with fpm
    star_1_fpm_wp1_data = np.array([star_1_fpm_I_0 * star_1, star_1_fpm_I_90 * star_1])
    star_1_fpm_wp2_data = np.array([star_1_fpm_I_45 * star_1, star_1_fpm_I_135 * star_1])
    star_2_fpm_wp1_data = np.array([star_2_fpm_I_0 * star_2, star_2_fpm_I_90 * star_2])
    star_2_fpm_wp2_data = np.array([star_2_fpm_I_45 * star_2, star_2_fpm_I_135 * star_2])
    # create default header
    prihdr, exthdr, errhdr, dqhdr = mocks.create_default_L3_headers()
    # create image objects using the constructed data
    star_1_nd_wp1_img = data.Image(star_1_nd_wp1_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
    star_1_nd_wp2_img = data.Image(star_1_nd_wp2_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
    star_2_nd_wp1_img = data.Image(star_2_nd_wp1_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
    star_2_nd_wp2_img = data.Image(star_2_nd_wp2_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
    star_1_fpm_wp1_img = data.Image(star_1_fpm_wp1_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
    star_1_fpm_wp2_img = data.Image(star_1_fpm_wp2_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
    star_2_fpm_wp1_img = data.Image(star_2_fpm_wp1_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
    star_2_fpm_wp2_img = data.Image(star_2_fpm_wp2_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
    input_list = [star_1_nd_wp1_img, star_1_nd_wp2_img, star_2_nd_wp1_img, star_2_nd_wp2_img, 
                     star_1_fpm_wp1_img, star_1_fpm_wp2_img, star_2_fpm_wp1_img, star_2_fpm_wp2_img]
    # update headers
    for i in range(len(input_list)):
        input_list[i].ext_hdr['DATALVL'] = 'L3'
        # even indices correspond to POL0, odd indices correspond to POL45
        if i % 2 == 0:
            input_list[i].ext_hdr['DPAMNAME'] = 'POL0'
        else:
            input_list[i].ext_hdr['DPAMNAME'] = 'POL45'
        # first four images are unocculted with roll angle of 30
        if i < 4:
            input_list[i].ext_hdr['FPAMNAME'] = 'ND225'
            input_list[i].pri_hdr['ROLL'] = roll_unocculted
        else:
            input_list[i].pri_hdr['ROLL'] = roll
        # distinguish between the two different target stars
        if i in [0, 1, 4, 5]:
            input_list[i].pri_hdr['TARGET'] = '1'
        else:
            input_list[i].pri_hdr['TARGET'] = '2'

    # construct dataset
    input_dataset = data.Dataset(input_list)
       
    # construct mueller matrix calibration objects
    mm_prihdr, mm_exthdr, mm_errhdr, mm_dqhdr = mocks.create_default_calibration_product_headers()
    system_mm_cal = data.MuellerMatrix(system_mueller_matrix, pri_hdr=mm_prihdr.copy(), ext_hdr=mm_exthdr.copy(), input_dataset=input_dataset)
    nd_mm_cal = data.MuellerMatrix(nd_mueller_matrix, pri_hdr=mm_prihdr.copy(), ext_hdr=mm_exthdr.copy(), input_dataset=input_dataset)

    # run step function
    output_dataset = l3_to_l4.subtract_stellar_polarization(input_dataset=input_dataset, 
                                                            system_mueller_matrix_cal=system_mm_cal,
                                                            nd_mueller_matrix_cal=nd_mm_cal)
    
    # length of output dataset should now be 4 with the unocculted observations removed
    assert len(output_dataset) == 4

    # check that orthogonal speckle fields now subtract out 
    zero_image = np.zeros(shape=(50, 50))
    for output_frame in output_dataset:
        assert np.allclose(output_frame.data[0] - output_frame.data[1], zero_image)

    
    # check that total intensity stayed the same before and after
    assert np.allclose(star_1_fpm_wp1_data[0] + star_1_fpm_wp1_data[1], output_dataset.frames[0].data[0] + output_dataset.frames[0].data[1])
    assert np.allclose(star_1_fpm_wp2_data[0] + star_1_fpm_wp2_data[1], output_dataset.frames[1].data[0] + output_dataset.frames[1].data[1])
    assert np.allclose(star_2_fpm_wp1_data[0] + star_2_fpm_wp1_data[1], output_dataset.frames[2].data[0] + output_dataset.frames[2].data[1])
    assert np.allclose(star_2_fpm_wp2_data[0] + star_2_fpm_wp2_data[1], output_dataset.frames[3].data[0] + output_dataset.frames[3].data[1])

def test_combine_polarization_states():
    '''
    Generate a sequence of L3 polarimetric images at different roll angles to pass into the
    combine_polarization_states() step function, checks that the output Stokes datacube matches
    with the known on-sky Stokes vector
    '''
    # define instrument mueller matrix and target Stokes vector
    system_mueller_matrix = np.array([
        [ 0.67450, 0.00623, 0.00000, 0.00000],
        [-0.00623,-0.67448, 0.00001, 0.00001],
        [ 0.00000, 0.00000, 0.67213,-0.05384],
        [ 0.00000, 0.00000,-0.05384,-0.67211]
    ])
    #system_mueller_matrix = np.identity(4)
    target_stokes_vector = np.array([1, 0.4, -0.3, 0.02])

    # construct input polarimetric images, taken with both wollastons at roll angles from 0 to 180 in 30 degree increments
    # also construct nonpolarimetric images to test PSF subs
    input_pol_frames = []
    input_psfsub_frames = []
    prihdr, exthdr, errhdr, dqhdr = mocks.create_default_L3_headers()
    # use mock gaussian as unpolarized target image
    target_total_intensity = mocks.gaussian_array(array_shape=[50, 50], sigma=2, amp=100)
    # add spot that gets rotated to test the output image is rotated northup
    spot = mocks.gaussian_array(array_shape=[10, 10], sigma=2, amp=50)
    target_total_intensity[30:40, 20:30] += spot
    # loops through roll angles 0, 30, 60, ... , 180
    roll_angle  = 0
    while roll_angle <= 180:
        # propagate on-sky target stokes vector through roll angle rotation, system mueller matrix, and wollaston to obtain polarized intensities
        output_stokes_vector = system_mueller_matrix @ pol.rotation_mueller_matrix(roll_angle) @ target_stokes_vector
        intensity_0 = (pol.lin_polarizer_mueller_matrix(0) @ output_stokes_vector)[0]
        intensity_45 = (pol.lin_polarizer_mueller_matrix(45) @ output_stokes_vector)[0]
        intensity_90 = (pol.lin_polarizer_mueller_matrix(90) @ output_stokes_vector)[0]
        intensity_135 = (pol.lin_polarizer_mueller_matrix(135) @ output_stokes_vector)[0]

        # rotate the image so it is in the right orientation with respect to the roll angle
        target_rotated = rotate(target_total_intensity, -roll_angle, [24, 24])
        # replace NaN values from rotation with 0s
        target_rotated[np.isnan(target_rotated)] = 0

        # construct POL0 image
        pol0_data = np.array([intensity_0 * target_rotated , intensity_90 * target_rotated])
        pol0_img = data.Image(pol0_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
        pol0_img.pri_hdr['ROLL'] = roll_angle
        pol0_img.ext_hdr['DPAMNAME'] = 'POL0'
        pol0_img.ext_hdr['STARLOCX'] = 24
        pol0_img.ext_hdr['STARLOCY'] = 24
        input_pol_frames.append(pol0_img)

        # construct POL45 image
        pol45_data = np.array([intensity_45 * target_rotated, intensity_135 * target_rotated])
        pol45_img = data.Image(pol45_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
        pol45_img.pri_hdr['ROLL'] = roll_angle
        pol45_img.ext_hdr['DPAMNAME'] = 'POL45'
        pol45_img.ext_hdr['STARLOCX'] = 24
        pol45_img.ext_hdr['STARLOCY'] = 24
        input_pol_frames.append(pol45_img)

        # construct total intensity image for psf sub
        psfsub_img_1 = data.Image( (intensity_0 + intensity_90) * target_rotated, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
        psfsub_img_2 = data.Image( (intensity_45 + intensity_135) * target_rotated, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
        psfsub_img_1.pri_hdr['ROLL'] = roll_angle
        psfsub_img_2.pri_hdr['ROLL'] = roll_angle
        psfsub_img_1.ext_hdr['STARLOCX'] = 24
        psfsub_img_2.ext_hdr['STARLOCX'] = 24
        psfsub_img_1.ext_hdr['STARLOCY'] = 24
        psfsub_img_2.ext_hdr['STARLOCY'] = 24
        input_psfsub_frames.append(psfsub_img_1)
        input_psfsub_frames.append(psfsub_img_2)
        

        roll_angle = roll_angle + 30

    # construct datasets
    input_pol_dataset = data.Dataset(input_pol_frames)
    input_psfsub_dataset = data.Dataset(input_psfsub_frames)
    test_dataset = data.Dataset([input_psfsub_dataset.copy()[2]])

    # construct mueller matrix calibration file
    mm_prihdr, mm_exthdr, mm_errhdr, mm_dqhdr = mocks.create_default_calibration_product_headers()
    system_mm_cal = data.MuellerMatrix(system_mueller_matrix, pri_hdr=mm_prihdr, ext_hdr=mm_exthdr, input_dataset=input_pol_dataset)

    # call combine_polarization_states to obtain stokes datacube
    with warnings.catch_warnings():
        # catch warning raised when rotating with roll angle instead of wcs
        warnings.filterwarnings('ignore', category=UserWarning)
        output_dataset = l3_to_l4.combine_polarization_states(input_pol_dataset, 
                                                            system_mueller_matrix_cal=system_mm_cal,
                                                            use_wcs=False, 
                                                            measure_klip_thrupt=False,
                                                            measure_1d_core_thrupt=False)
    stokes_datacube = output_dataset.frames[0].data
    stokes_datacube_err = output_dataset.frames[0].err
    # run PSF subtraction on total intensity dataset
    with warnings.catch_warnings():
        # suppress astropy warnings
        warnings.filterwarnings('ignore', category=VerifyWarning)
        warnings.filterwarnings('ignore', category=FITSFixedWarning)
        output_psfsub_dataset = l3_to_l4.do_psf_subtraction(input_psfsub_dataset, 
                                                            measure_klip_thrupt=False,
                                                            measure_1d_core_thrupt=False,
                                                            numbasis=1)
    psfsub_image = output_psfsub_dataset.frames[0].data[0]
    
    # check that the output dataset is the right size, and the output Stokes datacube and error is the right dimension
    assert len(output_dataset) == 1
    assert stokes_datacube.shape == (4, 50, 50)
    assert stokes_datacube_err.shape == (1, 4, 50, 50)
    # check that output Stokes I is the PSF subtracted version
    assert np.allclose(psfsub_image, stokes_datacube[0], equal_nan=True)
    # check that Stokes Q, U, and V is correctly recovered
    # replace NaNs on the outer edges of the stokes datacube with 0s so we can compare directly with the original image
    stokes_datacube[np.isnan(stokes_datacube)] = 0
    assert np.allclose(target_stokes_vector[1] * (target_total_intensity), stokes_datacube[1], atol=0.05)
    assert np.allclose(target_stokes_vector[2] * (target_total_intensity), stokes_datacube[2], atol=0.05)
    assert np.allclose(target_stokes_vector[3] * (target_total_intensity), stokes_datacube[3], atol=0.05)
    
def test_calc_stokes_unocculted(n_sim=100, nsigma_tol=3.):
    """
    Test the `calc_stokes_unocculted` function using synthetic L3 polarimetric datasets.

    Each mock dataset contains multiple images corresponding to different Wollaston
    prisms and roll angles. The test generates a variety of fractional polarization
    values and polarization angles, computes the unocculted Stokes parameters, and
    compares the recovered Q and U against the input values using their propagated
    uncertainties. The comparison is performed in units of the standard errors (chi),
    ensuring the function correctly handles multiple images per dataset and provides
    statistically consistent results.

    The tolerance (`nsigma_tol`) defines the acceptable deviation from ideal
    statistics, expressed in units of standard errors. For `n_sim` simulations, the
    expected fluctuations of the median and standard deviation of chi are approximately
    1/sqrt(n_sim) and 1/sqrt(2*(n_sim-1)), respectively. Multiplying by `nsigma_tol`
    allows for a configurable confidence interval, e.g., `nsigma_tol=3` corresponds
    roughly to a 3-sigma limit on expected statistical deviations.
    """

    # --- Simulate varying polarization fractions ---
    p_input = 0.1 + 0.2 * np.random.rand(n_sim)
    theta_input = 10.0 + 20.0 * np.random.rand(n_sim)

    Q_recovered = []
    Qerr_recovered = []
    U_recovered = []
    Uerr_recovered = []

    n_repeat = 8
    
    # prisms and rolls
    prisms = np.append(np.tile('POL0', n_repeat//2), np.tile('POL45', n_repeat//2))
    rolls = np.full(n_repeat, 0)

    for p, theta in zip(p_input, theta_input):
        # --- Generate mock L2b image ---
        dataset_polmock = mocks.create_mock_polarization_l3_dataset(
            I0=1e10,
            p=p,
            theta_deg=theta,
            roll_angles=rolls,
            prisms=prisms
        )

        # --- Compute unocculted Stokes ---
        Image_stokes_unocculted = calc_stokes_unocculted(dataset_polmock)

        Q_obs = Image_stokes_unocculted.data[1]
        U_obs = Image_stokes_unocculted.data[2]
        Q_err = Image_stokes_unocculted.err[0][1]
        U_err = Image_stokes_unocculted.err[0][2]

        Q_recovered.append(Q_obs)
        Qerr_recovered.append(Q_err)
        U_recovered.append(U_obs)
        Uerr_recovered.append(U_err)

    # --- Convert lists to arrays ---
    Q_recovered = np.array(Q_recovered)
    Qerr_recovered = np.array(Qerr_recovered)
    U_recovered = np.array(U_recovered)
    Uerr_recovered = np.array(Uerr_recovered)

    # --- Compute chi ---
    theta_rad = np.radians(theta_input)
    Q_input = p_input * np.cos(2 * theta_rad)
    Q_chi = (Q_recovered - Q_input) / Qerr_recovered
    U_input = p_input * np.sin(2 * theta_rad)
    U_chi = (U_recovered - U_input) / Uerr_recovered

    #print(np.median(Q_chi), np.std(Q_chi), np.median(U_chi), np.std(U_chi))
    # --- Assertions ---
    tol_mean = 1. / np.sqrt(n_sim) * nsigma_tol
    tol_std = 1. / np.sqrt(2.*(n_sim-1.)) * nsigma_tol
    assert np.median(Q_chi) == pytest.approx(0, abs=tol_mean)
    assert np.std(Q_chi) == pytest.approx(1, abs=tol_std)
    assert np.median(U_chi) == pytest.approx(0, abs=tol_mean)
    assert np.std(U_chi) == pytest.approx(1, abs=tol_std)

    return

def test_calc_stokes_unocculted(n_sim=100, nsigma_tol=3.):
    """
    Test the `calc_stokes_unocculted` function using synthetic L3 polarimetric datasets.

    Each mock dataset contains multiple images corresponding to different Wollaston
    prisms and roll angles. The test generates a variety of fractional polarization
    values and polarization angles, computes the unocculted Stokes parameters, and
    compares the recovered Q and U against the input values using their propagated
    uncertainties. The comparison is performed in units of the standard errors (chi),
    ensuring the function correctly handles multiple images per dataset and provides
    statistically consistent results.

    The tolerance (`nsigma_tol`) defines the acceptable deviation from ideal
    statistics, expressed in units of standard errors. For `n_sim` simulations, the
    expected fluctuations of the median and standard deviation of chi are approximately
    1/sqrt(n_sim) and 1/sqrt(2*(n_sim-1)), respectively. Multiplying by `nsigma_tol`
    allows for a configurable confidence interval, e.g., `nsigma_tol=3` corresponds
    roughly to a 3-sigma limit on expected statistical deviations.
    """

    # --- Simulate varying polarization fractions ---
    p_input = 0.1 + 0.2 * np.random.rand(n_sim)
    theta_input = 10.0 + 20.0 * np.random.rand(n_sim)

    Q_recovered = []
    Qerr_recovered = []
    U_recovered = []
    Uerr_recovered = []

    n_repeat = 8
    
    # prisms and rolls
    prisms = np.append(np.tile('POL0', n_repeat//2), np.tile('POL45', n_repeat//2))
    rolls = np.full(n_repeat, 0)

    for p, theta in zip(p_input, theta_input):
        # --- Generate mock L2b image ---
        dataset_polmock = mocks.create_mock_polarization_l3_dataset(
            I0=1e10,
            p=p,
            theta_deg=theta,
            roll_angles=rolls,
            prisms=prisms
        )

        # --- Compute unocculted Stokes ---
        Image_stokes_unocculted = calc_stokes_unocculted(dataset_polmock)[0]

        Q_obs = Image_stokes_unocculted.data[1]
        U_obs = Image_stokes_unocculted.data[2]
        Q_err = Image_stokes_unocculted.err[0][1]
        U_err = Image_stokes_unocculted.err[0][2]

        Q_recovered.append(Q_obs)
        Qerr_recovered.append(Q_err)
        U_recovered.append(U_obs)
        Uerr_recovered.append(U_err)

    # --- Convert lists to arrays ---
    Q_recovered = np.array(Q_recovered)
    Qerr_recovered = np.array(Qerr_recovered)
    U_recovered = np.array(U_recovered)
    Uerr_recovered = np.array(Uerr_recovered)

    # --- Compute chi ---
    theta_rad = np.radians(theta_input)
    Q_input = p_input * np.cos(2 * theta_rad)
    Q_chi = (Q_recovered - Q_input) / Qerr_recovered
    U_input = p_input * np.sin(2 * theta_rad)
    U_chi = (U_recovered - U_input) / Uerr_recovered

    #print(np.median(Q_chi), np.std(Q_chi), np.median(U_chi), np.std(U_chi))
    # --- Assertions ---
    tol_mean = 1. / np.sqrt(n_sim) * nsigma_tol
    tol_std = 1. / np.sqrt(2.*(n_sim-1.)) * nsigma_tol
    assert np.median(Q_chi) == pytest.approx(0, abs=tol_mean)
    assert np.std(Q_chi) == pytest.approx(1, abs=tol_std)
    assert np.median(U_chi) == pytest.approx(0, abs=tol_mean)
    assert np.std(U_chi) == pytest.approx(1, abs=tol_std)

    ## Test that passingin multiple targets in a single dataset behaves as expected. 
    # Create a dataset with two targets, each with different known polarization
    p_target1 = 0.15
    theta_target1 = 20.0
    p_target2 = 0.25
    theta_target2 = 40.0
    prisms = np.array(['POL0', 'POL45']*4)
    rolls = np.array([0,0,0,0,0,0,0,0])

    dataset1_polmock_list = mocks.create_mock_polarization_l3_dataset(
        I0=1e10,
        p=p_target1,
        theta_deg=theta_target1,
        roll_angles=rolls,
        prisms=prisms, 
        return_image_list=True
    )
    for img in dataset1_polmock_list:
        img.pri_hdr['TARGET'] = '1'

    for img in dataset1_polmock_list:
        img.pri_hdr['TARGET'] = '2'


    dataset2_polmock_list = mocks.create_mock_polarization_l3_dataset(
        I0=1e10,
        p=p_target2,
        theta_deg=theta_target2,
        roll_angles=rolls,
        prisms=prisms, 
        return_image_list=True
    )

    #concatenate the lists
    combined_image_list = dataset1_polmock_list + dataset2_polmock_list
    combined_dataset = data.Dataset(combined_image_list)

    # Compute unocculted Stokes for combined dataset
    combined_stokes_dataset = calc_stokes_unocculted(combined_dataset)
    # Separate the results for each target
    stokes_target1 = combined_stokes_dataset[0]
    stokes_target2 = combined_stokes_dataset[1]

    #assert check that we get what is expceted for target 1
    Q1_obs = stokes_target1.data[1]
    U1_obs = stokes_target1.data[2]
    Q2_obs = stokes_target2.data[1]
    U2_obs = stokes_target2.data[2]

    Q1_input = p_target1 * np.cos(2 * np.radians(theta_target1))
    U1_input = p_target1 * np.sin(2 * np.radians(theta_target1))
    Q2_input = p_target2 * np.cos(2 * np.radians(theta_target2))
    U2_input = p_target2 * np.sin(2 * np.radians(theta_target2))

    #should be at least 0.03
    assert Q1_obs == pytest.approx(Q1_input, abs=0.03)
    assert U1_obs == pytest.approx(U1_input, abs=0.03)
    assert Q2_obs == pytest.approx(Q2_input, abs=0.03)
    assert U2_obs == pytest.approx(U2_input, abs=0.03)

    return

if __name__ == "__main__":
    # test_image_splitting()
    # test_calc_pol_p_and_pa_image()
    # test_subtract_stellar_polarization()
    test_mueller_matrix_cal()
    # test_combine_polarization_states()
    # test_align_frames()
    # test_calc_stokes_unocculted()
