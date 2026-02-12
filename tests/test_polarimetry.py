import os, glob
import numpy as np
import pandas as pd
import shutil
import warnings
import logging

from astropy.io.fits import Header

import pytest
import logging

from corgidrp.data import Dataset, Image
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.pol as pol
import corgidrp.l2b_to_l3 as l2b_to_l3
import corgidrp.l3_to_l4 as l3_to_l4
import corgidrp.l4_to_tda as l4_to_tda
from corgidrp.pol import calc_stokes_unocculted
import corgidrp.corethroughput as corethroughput
import corgidrp.check as check

from corgidrp import star_center

from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning
from pyklip.klip import rotate

from corgidrp.check import (check_filename_convention, check_dimensions, 
                           verify_hdu_count, verify_header_keywords, 
                           validate_binary_table_fields, get_latest_cal_file)


# Suppress file collision warnings from mocks.rename_files_to_cgi_format
warnings.filterwarnings("ignore", message="File collision detected.*already exists")


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
        
def test_calc_pol_p_and_pa_image(n_sim=100, nsigma_tol=3., seed=0,
                                 logger=None, log_head=""):
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
    
    # ================================================================================
    # Logger Setup
    # ================================================================================
    if logger is None:
        # Create output directory for log file
        output_dir = os.path.join(os.path.dirname(__file__), 'pol_l4_to_tda_VAP_test2')
        os.makedirs(output_dir, exist_ok=True)
        
        log_file = os.path.join(output_dir, 'pol_l4_to_tda_VAP_test2.log')
        
        # Create a new logger specifically for this test
        logger = logging.getLogger('pol_l4_to_tda_VAP_test2')
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Header banner
        logger.info('='*80)
        logger.info('POLARIMETRY L4 TO TDA VAP TEST 2')
        logger.info('='*80)
        logger.info("")
        
    # ================================================================================
    # (0) Setup Input L4 Image
    # ================================================================================
    rng = np.random.default_rng(seed)

    # --- Simulation parameters ---
    p_input = 0.1 + 0.2 * rng.random(n_sim)
    theta_input = 10.0 + 20.0 * rng.random(n_sim)

    # --- Containers for chi statistics ---
    P_chi_mean, P_chi_std = [], []
    p_chi_mean, p_chi_std = [], []
    evpa_chi_mean, evpa_chi_std = [], []

    for i, p, theta in zip(range(n_sim), p_input, theta_input):

        if i != n_sim-1:
            I0 = 1e10
        else:
            # I ~ 0 scenario for a sanity check
            I0 = 0
            
        # Generate mock Stokes cube
        common_kwargs = dict(
            fwhm=1e2,
            I0=I0,
            p=p,
            theta_deg=theta,
            rng=rng,
        )   
        
        # Generate mock Stokes cube
        Image_input = mocks.create_mock_stokes_image_l4(badpixel_fraction=1e-3, **common_kwargs)
    
        Image_input_true = mocks.create_mock_stokes_image_l4(badpixel_fraction=0.0, add_noise=False, **common_kwargs)
        P_input = np.sqrt(Image_input_true.data[1]**2. + Image_input_true.data[2]**2.)
        idx = np.where( Image_input.dq[0] != 0 )
        P_input[idx] = np.nan

        if i == 0:
            # Validate Input Image and Header
            # Check/log that L4 data input complies with cgi format
            if isinstance(Image_input, Image): 
                logger.info(log_head + f"Input format: {type(Image_input).__name__}. Expected: Image. PASS")
            else: 
                logger.info(log_head + f"Input format: {type(Image_input).__name__}. Expected: Image. FAIL")
            
            ext_hdr = Image_input.ext_hdr

            # Check/log header keywords using check.verify_header_keywords
            required_keywords = {
                'DATALVL': 'L4',
                'BUNIT': 'photoelectron/s'
            }
            check.verify_header_keywords(ext_hdr, required_keywords, frame_info=log_head, logger=logger)

        # Compute polarization products
        Image_pol = l4_to_tda.calc_pol_p_and_pa_image(Image_input)

        P_map = Image_pol.data[0]       # Polarized intensity
        p_map = Image_pol.data[1]       # fractional polarization
        evpa_map = Image_pol.data[2]    # EVPA
        P_map_err = Image_pol.err[0][0]
        p_map_err = Image_pol.err[0][1]
        evpa_map_err = Image_pol.err[0][2]
        P_dq = Image_pol.dq[0]
        p_dq = Image_pol.dq[1]
        evpa_dq = Image_pol.dq[2]

        if i == 0:
            # Check/log that Output shape is [3, H, W]
            shape_expected = (3, Image_input.data.shape[1], Image_input.data.shape[2])
            shape_actual = Image_pol.data.shape
            if shape_actual == shape_expected:
                logger.info(log_head + f"Output shape: {shape_actual}. Expected shape: {shape_expected}. PASS")
            else:
                logger.info(log_head + f"Output shape: {shape_actual}. Expected shape: {shape_expected}. FAIL")

            #Test: Check/log that flags propagate from I, Q, U to P, p, polarization angle
            dq_input = np.bitwise_or(np.bitwise_or(Image_input.dq[0], Image_input.dq[1]), Image_input.dq[2])
            dq_sets = [
                (P_dq, "Polarized intensity"),
                (p_dq, "Polarized fraction"),
                (evpa_dq, "Polarization angle"),
            ]
            for dq, label in dq_sets:
                idx = np.where( dq_input != dq )[0]
                if idx.size == 0:
                    logger.info(log_head + f"Flag propagation test - {label}: DQ flags propagated correctly. Expected: DQ matches input from I, Q, U. PASS")
                else:
                    logger.info(log_head + f"Flag propagation test - {label}: {idx.size} mismatched pixels. Expected: DQ matches input from I, Q, U. FAIL")

            # Test: Check/log that polarized intensity P is computed correctly
            Q = Image_input.data[1]
            U = Image_input.data[2]
            P_qu = np.sqrt(Q**2 + U**2)
            diff = np.nanmax(np.abs(P_map - P_qu))
            if diff < 1e-10:
                logger.info(log_head + f"Polarized intensity computation test: P == sqrt(Q^2+U^2), max difference = {diff:.2e}. Expected: < 1e-10. PASS")
            else:
                logger.info(log_head + f"Polarized intensity computation test: P == sqrt(Q^2+U^2), max difference = {diff:.2e}. Expected: < 1e-10. FAIL")
        
        if i != n_sim-1:                
            # Compute chi statistics for statistical validation tests
            # These will be used to test: fractional polarization p matches input,
            # polarization position angle matches input, and error propagation
            P_chi = (P_map - P_input) / P_map_err
            p_chi = (p_map - p) / p_map_err
            evpa_chi = (evpa_map - theta) / evpa_map_err
            
            idx_P = np.where( P_dq == 0 )
            idx_p = np.where( p_dq == 0 )
            idx_evpa = np.where( evpa_dq == 0 )
            
            # Compute mean values among pixels
            P_chi_mean.append(np.nanmedian(P_chi[idx_P]))
            P_chi_std.append(np.nanstd(P_chi[idx_P]))
            p_chi_mean.append(np.nanmedian(p_chi[idx_p]))
            p_chi_std.append(np.nanstd(p_chi[idx_p]))
            evpa_chi_mean.append(np.nanmedian(evpa_chi[idx_evpa]))
            evpa_chi_std.append(np.nanstd(evpa_chi[idx_evpa]))
        else:
            if np.nanmax(abs(Image_input.data[0])) < 1e-10:
                I_max = np.nanmax(abs(Image_input.data[0]))
                logger.info(log_head + f"Sanity check I=0: max(I) = {I_max:.2e}. Expected: < 1e-10. PASS")
                map_sets = [
                    (P_map, "Polarized intensity"),
                    (p_map, "Polarized fraction"),
                    (evpa_map, "Polarization angle"),
                ]
                for map, label in map_sets:
                    has_nan = np.any(np.isnan(map))
                    has_inf = np.any(np.isinf(map))
                    if has_nan or has_inf:
                        nan_count = np.sum(np.isnan(map)) if has_nan else 0
                        inf_count = np.sum(np.isinf(map)) if has_inf else 0
                        logger.info(log_head + f"NaN/Inf check for {label}: NaN={nan_count}, Inf={inf_count}. Expected: no NaN/Inf. FAIL")
                    else:
                        logger.info(log_head + f"NaN/Inf check for {label}: no NaN/Inf values. Expected: no NaN/Inf. PASS")
                        
    # Remove the final run with I ~ 0 for a sanity check
    n_sim -= 1
    
    # Scale n_sim by number of pixels for statistical tolerance calculation
    # Each pixel in the mock images is independent, so total samples = n_sim * n_pixels
    n_sim *= P_map.size
    
    chi_sets = [
        (P_chi_mean, P_chi_std, "Polarized intensity", None),
        (p_chi_mean, p_chi_std, "Polarized fraction", "Fractional polarization p matches input"),
        (evpa_chi_mean, evpa_chi_std, "Polarization angle", "Polarization position angle matches input angle"),
    ]

    # Test: Check/log that fractional polarization p matches input (within error)
    # Test: Check/log that polarization position angle matches input angle (within error)
    # Test: Check/log that error propagation from Q, U is correct
    tol_mean = 1. / np.sqrt(n_sim) * nsigma_tol
    tol_std = 1. / np.sqrt(2.*(n_sim-1.)) * nsigma_tol
    for chi_mean, chi_std, label, test_name in chi_sets:
        median_mean = np.median(chi_mean)
        median_std = np.median(chi_std)
        mean_ok = abs(median_mean) <= tol_mean
        std_ok = abs(median_std - 1.0) <= tol_std
        assert mean_ok
        assert std_ok
        
        # Log mean check (for p and EVPA, tests if they match input within error)
        if test_name:
            if mean_ok:
                logger.info(log_head + f"{test_name} test - {label}: median(chi_mean)={median_mean:.4f}. Expected: |median(chi_mean)|<={tol_mean:.4f} (within error). PASS")
            else:
                logger.info(log_head + f"{test_name} test - {label}: median(chi_mean)={median_mean:.4f}. Expected: |median(chi_mean)|<={tol_mean:.4f} (within error). FAIL")
        
        # Log error propagation check (tests if error propagation from Q, U is correct)
        if std_ok:
            logger.info(log_head + f"Error propagation test - {label}: median(chi_std)={median_std:.4f}. Expected: |median(chi_std)-1|<={tol_std:.4f}. PASS")
        else:
            logger.info(log_head + f"Error propagation test - {label}: median(chi_std)={median_std:.4f}. Expected: |median(chi_std)-1|<={tol_std:.4f}. FAIL")

    # Footer banner
    logger.info("")
    logger.info('='*80)
    logger.info('POLARIMETRY L4 TO TDA VAP TEST 2 COMPLETE')
    logger.info('='*80)

    return

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
    # give unocculted images with ND filter a PA_APER angle of 30, images with FPM will have a PA_APER angle of 45
    pa_aper_deg_unocculted = 30
    pa_aper_deg = 45
    # find polarized intensities
    star_1_nd_pol = nd_mueller_matrix @ pol.rotation_mueller_matrix(pa_aper_deg_unocculted) @ star_1_pol
    star_2_nd_pol = nd_mueller_matrix @ pol.rotation_mueller_matrix(pa_aper_deg_unocculted) @ star_2_pol
    star_1_fpm_pol = system_mueller_matrix @ pol.rotation_mueller_matrix(pa_aper_deg) @ star_1_pol
    star_2_fpm_pol = system_mueller_matrix @ pol.rotation_mueller_matrix(pa_aper_deg) @ star_2_pol
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
        # first four images are unocculted with PA_APER angle of 30
        if i < 4:
            input_list[i].ext_hdr['FPAMNAME'] = 'ND225'
            input_list[i].pri_hdr['PA_APER'] = pa_aper_deg_unocculted
        else:
            input_list[i].pri_hdr['PA_APER'] = pa_aper_deg
        # distinguish between the two different target stars
        if i in [0, 1, 4, 5]:
            input_list[i].pri_hdr['TARGET'] = '1'
        else:
            input_list[i].pri_hdr['TARGET'] = '2'

    # construct dataset
    input_dataset = data.Dataset(input_list)
       
    # construct mueller matrix calibration objects
    mm_prihdr, mm_exthdr, mm_errhdr, mm_dqhdr = mocks.create_default_calibration_product_headers()
    system_mm_cal = data.MuellerMatrix(system_mueller_matrix, pri_hdr=mm_prihdr.copy(), ext_hdr=mm_exthdr.copy(), input_dataset=input_dataset[4:])
    nd_mm_cal = data.NDMuellerMatrix(nd_mueller_matrix, pri_hdr=mm_prihdr.copy(), ext_hdr=mm_exthdr.copy(), input_dataset=input_dataset[:4])

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
    Generate a sequence of L3 polarimetric images at different rotation angles to pass into the
    combine_polarization_states() step function, checks that the output Stokes datacube matches
    with the known on-sky Stokes vector
    '''

    ###########################
    #### Make dummy CT cal ####
    ###########################

    # Dataset with some CT profile defined in create_ct_interp
    # Pupil image
    pupil_image = np.zeros([1024, 1024])
    # Set it to some known value for a selected range of pixels
    pupil_image[510:530, 510:530]=1
    prhd, exthd_pupil, errhdr, dqhdr = mocks.create_default_L3_headers()
    # DRP
    # cfam filter
    exthd_pupil['CFAMNAME'] = '1F'
    # Add specific values for pupil images:
    # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'

    data_psf, psf_loc_in, half_psf = mocks.create_ct_psfs(50, cfam_name='1F',
    n_psfs=100)
    ct_dataset0 = data_psf[0]
    ct_dataset0.ext_hdr['FPAMNAME'] = 'HLC12_C2R1'  # set FPM to a coronagraphic one
    
    err = np.ones([1024,1024])
    data_ct_interp = [ct_dataset0, data.Image(pupil_image,pri_hdr = prhd,
        ext_hdr = exthd_pupil, err = err)]
    # Set of off-axis PSFs with a CT profile defined in create_ct_interp
    # First, we need the CT FPM center to create the CT radial profile
    # We can use a miminal dataset to get to know it
    ct_cal_tmp = corethroughput.generate_ct_cal(data.Dataset(data_ct_interp))


    # define instrument mueller matrix and target Stokes vector
    system_mueller_matrix = np.array([
        [ 0.67450, 0.00623, 0.00000, 0.00000],
        [-0.00623,-0.67448, 0.00001, 0.00001],
        [ 0.00000, 0.00000, 0.67213,-0.05384],
        [ 0.00000, 0.00000,-0.05384,-0.67211]
    ])
    #system_mueller_matrix = np.identity(4)
    target_stokes_vector = np.array([1, 0.4, -0.3, 0.02])

    # construct input polarimetric images, taken with both wollastons at rotation angles from 0 to 180 in 30 degree increments
    # also construct nonpolarimetric images to test PSF subs
    input_pol_frames = []
    input_psfsub_frames = []
    prihdr, exthdr, errhdr, dqhdr = mocks.create_default_L3_headers()
    # use mock gaussian as unpolarized target image
    target_total_intensity = mocks.gaussian_array(array_shape=[50, 50], sigma=2, amp=100)
    # add spot that gets rotated to test the output image is rotated northup
    spot = mocks.gaussian_array(array_shape=[10, 10], sigma=2, amp=50)
    target_total_intensity[30:40, 20:30] += spot
    # loops through rotation angles 0, 30, 60, ... , 180
    rotation_angle  = 0
    while rotation_angle <= 180:
        # propagate on-sky target stokes vector through telescope angle rotation, system mueller matrix, and wollaston to obtain polarized intensities
        output_stokes_vector = system_mueller_matrix @ pol.rotation_mueller_matrix(rotation_angle) @ target_stokes_vector
        intensity_0 = (pol.lin_polarizer_mueller_matrix(0) @ output_stokes_vector)[0]
        intensity_45 = (pol.lin_polarizer_mueller_matrix(45) @ output_stokes_vector)[0]
        intensity_90 = (pol.lin_polarizer_mueller_matrix(90) @ output_stokes_vector)[0]
        intensity_135 = (pol.lin_polarizer_mueller_matrix(135) @ output_stokes_vector)[0]

        # rotate the image so it is in the right orientation with respect to the rotation angle
        target_rotated = rotate(target_total_intensity, -rotation_angle, [24, 24])
        # replace NaN values from rotation with 0s
        target_rotated[np.isnan(target_rotated)] = 0

        # construct POL0 image
        pol0_data = np.array([intensity_0 * target_rotated , intensity_90 * target_rotated])
        pol0_img = data.Image(pol0_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
        pol0_img.pri_hdr['PA_APER'] = rotation_angle
        pol0_img.ext_hdr['DPAMNAME'] = 'POL0'
        pol0_img.ext_hdr['STARLOCX'] = 24
        pol0_img.ext_hdr['STARLOCY'] = 24
        input_pol_frames.append(pol0_img)

        # construct POL45 image
        pol45_data = np.array([intensity_45 * target_rotated, intensity_135 * target_rotated])
        pol45_img = data.Image(pol45_data, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
        pol45_img.pri_hdr['PA_APER'] = rotation_angle
        pol45_img.ext_hdr['DPAMNAME'] = 'POL45'
        pol45_img.ext_hdr['STARLOCX'] = 24
        pol45_img.ext_hdr['STARLOCY'] = 24
        input_pol_frames.append(pol45_img)

        # construct total intensity image for psf sub
        psfsub_img_1 = data.Image( (intensity_0 + intensity_90) * target_rotated, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
        psfsub_img_2 = data.Image( (intensity_45 + intensity_135) * target_rotated, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
        psfsub_img_1.pri_hdr['PA_APER'] = rotation_angle
        psfsub_img_2.pri_hdr['PA_APER'] = rotation_angle
        psfsub_img_1.ext_hdr['STARLOCX'] = 24
        psfsub_img_2.ext_hdr['STARLOCX'] = 24
        psfsub_img_1.ext_hdr['STARLOCY'] = 24
        psfsub_img_2.ext_hdr['STARLOCY'] = 24
        input_psfsub_frames.append(psfsub_img_1)
        input_psfsub_frames.append(psfsub_img_2)
        

        rotation_angle = rotation_angle + 30

    # construct datasets
    input_pol_dataset = data.Dataset(input_pol_frames)
    input_psfsub_dataset = data.Dataset(input_psfsub_frames)
    test_dataset = data.Dataset([input_psfsub_dataset.copy()[2]])

    # construct mueller matrix calibration file
    mm_prihdr, mm_exthdr, mm_errhdr, mm_dqhdr = mocks.create_default_calibration_product_headers()
    system_mm_cal = data.MuellerMatrix(system_mueller_matrix, pri_hdr=mm_prihdr, ext_hdr=mm_exthdr, input_dataset=input_pol_dataset)

    # call combine_polarization_states to obtain stokes datacube
    with warnings.catch_warnings():
        # catch warning raised when rotating with PA_APER angle instead of wcs
        warnings.filterwarnings('ignore', category=UserWarning)
        output_dataset = l3_to_l4.combine_polarization_states(input_pol_dataset, 
                                                            system_mm_cal,
                                                            ct_cal_tmp,
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
    
def test_calc_stokes_unocculted(n_sim=10, nsigma_tol=3.):
    """
    Test the `calc_stokes_unocculted` function using synthetic L3 polarimetric datasets.

    Each mock dataset contains multiple images corresponding to different Wollaston
    prisms and telescope rotation angles. The test generates a variety of fractional polarization
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

    # Set the random seed for reproducibility
    rng_seed = 52
    rng = np.random.default_rng(rng_seed)

    # --- Simulate varying polarization fractions ---
    p_input = 0.1 + 0.2 * rng.random(n_sim)
    theta_input = 10.0 + 20.0 * rng.random(n_sim)

    Q_recovered = []
    Qerr_recovered = []
    U_recovered = []
    Uerr_recovered = []

    n_repeat = 8
    
    # prisms and rotations
    prisms = np.append(np.tile('POL0', n_repeat//2), np.tile('POL45', n_repeat//2))
    rotation_angles = np.full(n_repeat, 0)

    for p, theta in zip(p_input, theta_input):
        new_seed = rng.integers(0, 1e6)
        # --- Generate mock L2b image ---
        dataset_polmock = mocks.create_mock_polarization_l3_dataset(
            I0=1e10,
            p=p,
            theta_deg=theta,
            pa_aper_degs=rotation_angles,
            prisms=prisms, 
            seed=new_seed
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
    rotation_angles = np.array([0,0,0,0,0,0,0,0])

    dataset1_polmock_list = mocks.create_mock_polarization_l3_dataset(
        I0=1e10,
        p=p_target1,
        theta_deg=theta_target1, 
        pa_aper_degs=rotation_angles,
        prisms=prisms, 
        return_image_list=True,
        seed= rng.integers(0, 1e6)
    )
    for img in dataset1_polmock_list:
        img.pri_hdr['TARGET'] = '1'

    for img in dataset1_polmock_list:
        img.pri_hdr['TARGET'] = '2'


    dataset2_polmock_list = mocks.create_mock_polarization_l3_dataset(
        I0=1e10,
        p=p_target2,
        theta_deg=theta_target2,
        pa_aper_degs=rotation_angles,
        prisms=prisms, 
        return_image_list=True,
        seed=rng.integers(0, 1e6)
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

def test_compute_QphiUPhi(): 
    '''
    Test that the computer_QphiUphi function behaves as expected in three scenarios:
    1) When the input image center is correct, U_phi should be approximately zero.
    2) When the input image center is incorrect, U_phi should be nonzero.
    3) The err array in the output should have the same shape as the data array.
    4) The dq array in the output should propagate as the bitwise-OR of the input dq for Q and U.   

    Includes VAP testing for 
    '''

    #######################################
    ########## Set up VAP Logger ##########
    #######################################

    current_file_path = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(current_file_path,'l4_to_tda_compute_quphi')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # set up logging
    global logger

    log_file = os.path.join(output_dir, 'l4_to_tda_compute_QphiUphi.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('l4_to_tda_compute_QphiUphi')
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)



    ################################################
    ########## TEST Dataset 1  - VAP TEST ##########
    ################################################

    # ================================================================================
    # (4.1) Setup Input Files
    # ================================================================================
    logger.info('='*80)
    logger.info('Set up input files and save to disk')
    logger.info('='*80)
    #########################################################################################################
    ########### Create input image of L4 Stokes cube [I, Q, U, V] with tangentially poalrized disk ##########
     
    pfrac = 0.1
    qu_img = mocks.create_mock_IQUV_image(pfrac=pfrac)

    # Expect at least (I, Q, U, V) planes in the input dq
    assert qu_img.dq.shape[0] >= 4, "mock image should have I,Q,U,V planes"

    # Distinct bits for Q and U (non-overlapping)
    BIT_Q = 1 << 2
    BIT_U = 1 << 5

    # Add bits to Q and U while preserving existing dq
    dq_mod = qu_img.dq.copy()
    dq_mod[1] = dq_mod[1] | BIT_Q  # Q plane
    dq_mod[2] = dq_mod[2] | BIT_U  # U plane
    qu_img.dq = dq_mod

    mocks.rename_files_to_cgi_format(list_of_fits=[qu_img], output_dir=output_dir, level_suffix="l4")

    #Check input image complies with cgi format
    logger.info('='*80)
    logger.info('Test 4.1: Input L4 Image Data format')
    logger.info('='*80)
    frame_info = "Input L4 Polarimetry Image"
    check_filename_convention(getattr(qu_img , 'filename', None), 'cgi_*_l4_.fits', frame_info,logger,data_level='l4_')
    verify_header_keywords(qu_img .ext_hdr, {'BUNIT': 'photoelectron/s'},  frame_info,logger)
    verify_header_keywords(qu_img .ext_hdr, {'DATALVL': 'L4'},  frame_info,logger)
    logger.info("")
    
    # ================================================================================
    # (4.2) Validate Output TDA Image
    # ================================================================================
    logger.info('='*80)
    logger.info('Test 4.2: Output TDA Azimuthal components test for correct center')
    logger.info('='*80)

    ### Run the compute_QphiUphi function
    qu_phi = l4_to_tda.compute_QphiUphi(qu_img)

    q_phi = qu_phi.data[4]
    u_phi = qu_phi.data[5]

    # if n_l4_files == 1:
    #     logger.info(f"L4 Output File Count: {n_l4_files}. PASS")
    # else:
    #     logger.info(f"L4 Output File Count: {n_l4_files}. Expected 1. FAIL")

    #### Check the dimensions: 
    #VAP Version
    if qu_phi.data.shape[0] == 6:
        logger.info(f"Output Slices: {qu_phi.data.shape[0]}. PASS")
    else:
        logger.info(f"Output Slices: {qu_phi.data.shape[0]}. Expected 6. FAIL")
    #pytest Version
    assert qu_phi.data.shape[0] == 6, "Output data should have 6 slices"

    #### Check that Q_phi has the expected tangential polarization pattern
    # For tangential polarization: Q_phi should be positive and follow the intensity pattern
    # The mock creates Q = -pfrac*I*cos(2*phi), U = -pfrac*I*sin(2*phi)
    # which results in Q_phi = pfrac*I (positive, tangential pattern)
    I = qu_img.data[0]
    expected_q_phi = pfrac * I
    
    #VAP Version
    if np.mean(q_phi) > 0:
        logger.info(f"Q_phi Mean Value: {np.mean(q_phi)} > 0. PASS")
    else:
        logger.info(f"Q_phi Mean Value: {np.mean(q_phi)} <= 0. FAIL")
    # Check that Q_phi matches the expected tangential pattern (within numerical precision)
    if np.allclose(q_phi, expected_q_phi, rtol=1e-5, atol=1e-8):
        logger.info("Q_phi matches expected tangential polarization pattern. PASS")
    else:
        logger.info("Q_phi does not match expected tangential polarization pattern. FAIL")
    #pytest Version
    assert np.mean(q_phi) > 0, "Q_phi should have positive mean for tangential polarization"
    np.testing.assert_allclose(
        q_phi, expected_q_phi, rtol=1e-5, atol=1e-8,
        err_msg="Q_phi should match expected tangential polarization pattern"
    )

    #### Check that U_phi is approximately zero when the input image center is correct
    #VAP Version
    if np.allclose(u_phi, 0.0, atol=1e-6):
        logger.info(f"U_phi is approximately zero for correct center. PASS")
    else:
        logger.info(f"U_phi is not approximately zero for correct center. FAIL")
    #pytest Version
    assert np.allclose(u_phi, 0.0, atol=1e-6), "U_phi should be ~0 for correct center"
        

    #### Check that err array is consistent with data
    #VAP Version
    if qu_phi.err.shape == qu_phi.data.shape:
        logger.info(f"err has the same shape as data. PASS")
    else:
        logger.info(f"err shape {qu_phi.err.shape} does not match data shape {qu_phi.data.shape}. FAIL")
    #pytest Version
    assert qu_phi.err.shape == qu_phi.data.shape, "err should have the same shape as data"

    #### Check that dq array is consistent with data
    #VAP Version
    if qu_phi.dq.shape == qu_phi.data.shape:
        logger.info(f"dq has the same shape as data. PASS")
    else:
        logger.info(f"dq shape {qu_phi.dq.shape} does not match data shape {qu_phi.data.shape}. FAIL")
    #pytest Version
    assert qu_phi.dq.shape == qu_phi.data.shape, "dq should have the same shape as data"

    expected_or = qu_img.dq[1] | qu_img.dq[2]
    #### Check that DQ propagation is as expected:
    # #### Verify that the dq of Q_phi and U_phi propagates as the bitwise-OR of
    #### the input dq for Q and U. We set distinct bits on all pixels of Q and U
    #### so the expected OR relationship holds regardless of geometry.
    #VAP Version: 
    if np.array_equal(
        qu_phi.dq[4] & (BIT_Q | BIT_U),
        expected_or & (BIT_Q | BIT_U)
    ):
        logger.info("Q_phi dq includes bits from Q and U (OR). PASS")
    else:
        logger.info("Q_phi dq does not include bits from Q and U (OR). FAIL")
    if np.array_equal(
        qu_phi.dq[5] & (BIT_Q | BIT_U),
        expected_or & (BIT_Q | BIT_U)
    ):
        logger.info("U_phi dq includes bits from Q and U (OR). PASS")
    else:
        logger.info("U_phi dq does not include bits from Q and U (OR). FAIL")
    #pytest Version:
    # Q_phi dq should include bits from Q and U (bitwise OR)
    np.testing.assert_array_equal(
        qu_phi.dq[4] & (BIT_Q | BIT_U),
        expected_or & (BIT_Q | BIT_U),
        err_msg="Q_phi dq should include bits from Q and U (OR)."
    )

    # U_phi dq should include bits from Q and U (bitwise OR)
    np.testing.assert_array_equal(
        qu_phi.dq[5] & (BIT_Q | BIT_U),
        expected_or & (BIT_Q | BIT_U),
        err_msg="U_phi dq should include bits from Q and U (OR)."
    )


    ####################################
    ########## TEST Dataset 2 ##########
    ####################################
    
    ####Verify that U_phi is nonzero when the input image center is incorrect.
    #### 5 pixel offset is chosen to ensure significant deviation from true center.

    pfrac = 0.1
    qu_img = mocks.create_mock_IQUV_image(pfrac=pfrac)
    
    # overwrite header center with wrong value
    qu_img.ext_hdr["STARLOCX"] += 5.0
    qu_img.ext_hdr["STARLOCY"] += 5.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        mocks.rename_files_to_cgi_format(list_of_fits=[qu_img], output_dir=output_dir, level_suffix="l4")

    ### Run the compute_QphiUphi function
    qu_phi = l4_to_tda.compute_QphiUphi(qu_img)
    #Extract U_phi
    u_phi = qu_phi.data[5]

    logger.info('='*80)
    logger.info('Test 4.3: Output TDA Azimuthal components test for incorrect center')
    logger.info('='*80)

    #### Check that U_phi is nonzero when the input image center is incorrect
    #VAP Version
    if not np.allclose(u_phi, 0.0, atol=1e-6):
        logger.info(f"U_phi is nonzero for incorrect center. PASS")
    else:
        logger.info(f"U_phi is approximately zero for incorrect center. FAIL")
    #pytest Version
    assert not np.allclose(u_phi, 0.0, atol=1e-6), "U_phi should be nonzero for wrong center"

    ####################################
    ########## TEST Dataset 3 ##########
    ####################################
    
    #### Test error propagation: σ_Qφ, σ_Uφ from σ_Q, σ_U
    logger.info('='*80)
    logger.info('Test 4.4: Error propagation σ_Qφ, σ_Uφ from σ_Q, σ_U')
    logger.info('='*80)
    
    pfrac = 0.1
    qu_img = mocks.create_mock_IQUV_image(pfrac=pfrac)
    
    # Set known error values for Q and U
    # compute_QphiUphi expects err to be (>=3, n, m), so ensure it's (4, n, m)
    # If err has a leading dimension of 1, remove it
    n, m = qu_img.data.shape[1:]
    if qu_img.err.ndim == 4 and qu_img.err.shape[0] == 1:
        # err has shape (1, 4, n, m) - reshape to (4, n, m)
        qu_img.err = qu_img.err[0]
    elif qu_img.err.shape[0] == 1 and qu_img.err.ndim == 3:
        # err has shape (1, n, m) - need to expand to (4, n, m)
        qu_img.err = np.broadcast_to(qu_img.err, (4, n, m)).copy()
    
    sigma_Q_val = 0.01
    sigma_U_val = 0.015
    qu_img.err[1, :, :] = sigma_Q_val  # Q error
    qu_img.err[2, :, :] = sigma_U_val  # U error
    
    # Get center for error calculation
    cx = qu_img.ext_hdr["STARLOCX"]
    cy = qu_img.ext_hdr["STARLOCY"]
    y_idx, x_idx = np.mgrid[0:n, 0:m]
    phi = np.arctan2(y_idx - cy, x_idx - cx)
    c2 = np.cos(2.0 * phi)
    s2 = np.sin(2.0 * phi)
    
    # Expected error propagation: var_Qphi = c2^2 * sigma_Q^2 + s2^2 * sigma_U^2
    # Expected error propagation: var_Uphi = s2^2 * sigma_Q^2 + c2^2 * sigma_U^2
    expected_sigma_Qphi = np.sqrt(c2**2 * sigma_Q_val**2 + s2**2 * sigma_U_val**2)
    expected_sigma_Uphi = np.sqrt(s2**2 * sigma_Q_val**2 + c2**2 * sigma_U_val**2)
    
    qu_phi = l4_to_tda.compute_QphiUphi(qu_img)
    # Output err array is 3D with shape (6, n, m) for [I, Q, U, V, Q_phi, U_phi]
    actual_sigma_Qphi = qu_phi.err[4, :, :]
    actual_sigma_Uphi = qu_phi.err[5, :, :]
    
    #VAP Version
    if np.allclose(actual_sigma_Qphi, expected_sigma_Qphi, rtol=1e-5):
        logger.info("Error propagation for σ_Qφ matches expected. PASS")
    else:
        logger.info("Error propagation for σ_Qφ does not match expected. FAIL")
    if np.allclose(actual_sigma_Uphi, expected_sigma_Uphi, rtol=1e-5):
        logger.info("Error propagation for σ_Uφ matches expected. PASS")
    else:
        logger.info("Error propagation for σ_Uφ does not match expected. FAIL")
    #pytest Version
    np.testing.assert_allclose(
        actual_sigma_Qphi, expected_sigma_Qphi, rtol=1e-5,
        err_msg="Error propagation for σ_Qφ should match expected formula"
    )
    np.testing.assert_allclose(
        actual_sigma_Uphi, expected_sigma_Uphi, rtol=1e-5,
        err_msg="Error propagation for σ_Uφ should match expected formula"
    )

    ####################################
    ########## TEST Dataset 4 ##########
    ####################################
    
    #### Test using header STARLOCX/Y vs manual center
    logger.info('='*80)
    logger.info('Test 4.5: Using header STARLOCX/Y vs manual center')
    logger.info('='*80)
    
    pfrac = 0.1
    qu_img = mocks.create_mock_IQUV_image(pfrac=pfrac)
    
    # Get the header center
    header_cx = qu_img.ext_hdr["STARLOCX"]
    header_cy = qu_img.ext_hdr["STARLOCY"]
    
    # Compute with header center (default behavior)
    qu_phi_header = l4_to_tda.compute_QphiUphi(qu_img)
    
    # Remove header center and use manual center
    del qu_img.ext_hdr["STARLOCX"]
    del qu_img.ext_hdr["STARLOCY"]
    qu_phi_manual = l4_to_tda.compute_QphiUphi(qu_img, x_center=header_cx, y_center=header_cy)
    
    # Results should be identical
    #VAP Version
    if np.allclose(qu_phi_header.data, qu_phi_manual.data, rtol=1e-10):
        logger.info("Header STARLOCX/Y and manual center produce identical results. PASS")
    else:
        logger.info("Header STARLOCX/Y and manual center produce different results. FAIL")
    if np.allclose(qu_phi_header.err, qu_phi_manual.err, rtol=1e-10):
        logger.info("Error arrays match between header and manual center. PASS")
    else:
        logger.info("Error arrays differ between header and manual center. FAIL")
    #pytest Version
    np.testing.assert_allclose(
        qu_phi_header.data, qu_phi_manual.data, rtol=1e-10,
        err_msg="Header STARLOCX/Y and manual center should produce identical results"
    )
    np.testing.assert_allclose(
        qu_phi_header.err, qu_phi_manual.err, rtol=1e-10,
        err_msg="Error arrays should match between header and manual center"
    )

    logger.info('='*80)
    logger.info('Polarimetry L4->TDA VAP Test 4: Extended Source (Disk) Azimuthal Stokes Test: Complete')
    logger.info('='*80)


if __name__ == "__main__":
    #test_image_splitting()
    # test_calc_pol_p_and_pa_image()
    #test_subtract_stellar_polarization()
    test_mueller_matrix_cal()
    #test_combine_polarization_states()
    #test_align_frames()
    # #test_calc_stokes_unocculted()
    # test_compute_QphiUPhi()
