import numpy as np

import corgidrp
from corgidrp import star_center
import corgidrp.data as data
import corgidrp.mocks as mocks
from corgidrp.l3_to_l4 import find_star
import corgidrp.l2b_to_l3 as l2b_to_l3

old_err_tracking = corgidrp.track_individual_errors

# Static parameters for testing
image_shape = (201, 201)
satellite_spot_parameters_defaults = star_center.satellite_spot_parameters_defaults

def test_find_star_offset():
    """
    Generate mock input data and pass into find_star function with an offset guess
    """
    corgidrp.track_individual_errors = True # this test uses individual error components

    # Set the star center position for injection of satellite spots

    # Add small offset and rotation in the injected data
    injected_position = (image_shape[1] // 2  + 2, image_shape[0] // 2  - 1)
    guess_position = (image_shape[1] // 2, image_shape[0] // 2)

    satellite_spot_angle_offset = 3
    guess_angle_offset = 0

    modes = ['NFOV', 'WFOV', 'SPEC660', 'SPEC730']

    for mode in modes:
        separation = satellite_spot_parameters_defaults[mode]['separation']['spotSepPix']

        # Generate test data
        input_dataset = mocks.create_satellite_spot_observing_sequence(
                n_sci_frames=3,
                n_satspot_frames=3, 
                image_shape=image_shape,
                bg_sigma=1.0,
                bg_offset=10.0,
                gaussian_fwhm=5.0,
                separation=separation,
                star_center=injected_position,
                angle_offset=satellite_spot_angle_offset,
                amplitude_multiplier=100,
                observing_mode=mode)

        # Set initial guesses for angle offset
        thetaOffsetGuess = guess_angle_offset

        dataset_with_center = find_star(
            input_dataset=input_dataset, 
            star_coordinate_guess=guess_position,
            thetaOffsetGuess=thetaOffsetGuess)

        measured_x, measured_y = (dataset_with_center.frames[0].ext_hdr['STARLOCX'],
                                dataset_with_center.frames[0].ext_hdr['STARLOCY'])
        print(measured_x, measured_y)
        assert np.isclose(injected_position[0], measured_x, atol=0.1), \
            f"{mode}. Expected {injected_position[0]}, got {measured_x}"
        assert np.isclose(injected_position[1], measured_y, atol=0.1), \
            f"{mode}. Expected {injected_position[1]}, got {measured_y}"

    corgidrp.track_individual_errors = old_err_tracking


def test_overwrite_parameters():
    """
    Generate mock input data and pass into find_star function
    """
    corgidrp.track_individual_errors = True # this test uses individual error components

    # Set the star center position for injection of satellite spots

    injected_position = (image_shape[1] // 2, image_shape[0] // 2)
    guess_position = (image_shape[1] // 2, image_shape[0] // 2)

    satellite_spot_angle_offset = 0
    guess_angle_offset = 0

    # Change some parameters
    overwrite_parameters = {"offset": {"roiRadiusPix": 4.5, "nSteps": 3, "stepSize": 1, "nIter": 6},
                            "separation": {"roiRadiusPix": 2.5}}

    modes = ['NFOV']

    for mode in modes:
        separation = satellite_spot_parameters_defaults[mode]['separation']['spotSepPix']

        # Generate test data
        input_dataset = mocks.create_satellite_spot_observing_sequence(
                n_sci_frames=3,
                n_satspot_frames=3, 
                image_shape=image_shape,
                bg_sigma=1.0,
                bg_offset=10.0,
                gaussian_fwhm=5.0,
                separation=separation,
                star_center=injected_position,
                angle_offset=satellite_spot_angle_offset,
                amplitude_multiplier=100,
                observing_mode=mode)

        # Set initial guesses for angle offset
        thetaOffsetGuess = guess_angle_offset

        _ = find_star(
            input_dataset=input_dataset, 
            star_coordinate_guess=guess_position,
            thetaOffsetGuess=thetaOffsetGuess,
            satellite_spot_parameters=overwrite_parameters)

    corgidrp.track_individual_errors = old_err_tracking

def test_find_star_polarimetry():
    """
    Generate mock polarimetric input data and pass into find_star function with an offset guess
    """
    corgidrp.track_individual_errors = True # this test uses individual error components

    # Set the star center position for injection of satellite spots

    # Add small offset and rotation in the injected data
    injected_position = [(2, - 1), (2, - 1)]

    satellite_spot_angle_offset = 3
    guess_angle_offset = 0

    modes = ['NFOV', 'WFOV']

    for mode in modes:
        separation = satellite_spot_parameters_defaults[mode]['separation']['spotSepPix']

        # Generate test data

        image_WP1_sp = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(
            dpamname='POL0', 
            observing_mode=mode, 
            left_image_value=1, 
            right_image_value=2,
            image_shape=(1024,1024),
            bg_sigma=1.0,
            bg_offset=10.0,
            gaussian_fwhm=5.0,
            separation=separation,
            star_center=injected_position,
            angle_offset=satellite_spot_angle_offset,
            amplitude_multiplier=100)

        image_WP1= mocks.create_mock_l2b_polarimetric_image(
            dpamname='POL0', 
            observing_mode=mode, 
            left_image_value=1, 
            right_image_value=2)

        image_WP2_sp = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(
            dpamname='POL45', 
            observing_mode=mode, 
            left_image_value=1, 
            right_image_value=2,
            image_shape=(1024,1024),
            bg_sigma=1.0,
            bg_offset=10.0,
            gaussian_fwhm=5.0,
            separation=separation,
            star_center=injected_position,
            angle_offset=satellite_spot_angle_offset,
            amplitude_multiplier=100)

        image_WP2 = mocks.create_mock_l2b_polarimetric_image(
            dpamname='POL45', 
            observing_mode=mode, 
            left_image_value=1, 
            right_image_value=2)

        input_dataset = data.Dataset([image_WP1_sp, image_WP1, image_WP2_sp, image_WP2])
        input_dataset_autocrop = l2b_to_l3.split_image_by_polarization_state(input_dataset)

        thetaOffsetGuess = guess_angle_offset

        dataset_with_center = find_star(
            input_dataset=input_dataset_autocrop, 
            thetaOffsetGuess=thetaOffsetGuess)

        measured_x, measured_y = (dataset_with_center.frames[0].ext_hdr['STARLOCX'],
                                dataset_with_center.frames[0].ext_hdr['STARLOCY'])

        assert np.isclose(dataset_with_center.frames[0].data[0].shape[0]//2 + injected_position[0][0], measured_x, atol=0.1), \
            f"{mode}. Expected {injected_position[0][1]}, got {measured_x}"
        assert np.isclose(dataset_with_center.frames[0].data[0].shape[1]//2 + injected_position[0][1], measured_y, atol=0.1), \
            f"{mode}. Expected {injected_position[0][0]}, got {measured_y}"

    corgidrp.track_individual_errors = old_err_tracking


if __name__ == "__main__":
    test_find_star_offset()
    test_overwrite_parameters()
    test_find_star_polarimetry()