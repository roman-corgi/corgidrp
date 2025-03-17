import numpy as np

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
from corgidrp.l3_to_l4 import find_star

old_err_tracking = corgidrp.track_individual_errors

# Static parameters for testing
image_shape = (201, 201)

# Spot separation in pixels
# sep in lambda/D and multiply by pix per lambda/D
spot_separation_nfov = 6.5*(51.46*0.575/13) # 14.79
spot_separation_spec = 6.0*(51.46*0.730/13) # 17.34
spot_separation_wfov = 13*(51.46*0.825/13) # 42.4545

satellite_spot_parameters_defaults = {
    "NFOV": {
        "offset": {
            "spotSepPix": spot_separation_nfov,
            "roiRadiusPix": 4.5,
            "probeRotVecDeg": [0, 90],
            "nSubpixels": 100,
            "nSteps": 7,
            "stepSize": 1,
            "nIter": 6,
        },
        "separation": {
            "spotSepPix": spot_separation_nfov,
            "roiRadiusPix": 1.5,
            "probeRotVecDeg": [0, 90],
            "nSubpixels": 100,
            "nSteps": 21,
            "stepSize": 0.25,
            "nIter": 5,
        }
    },
    "SPEC660": {
        "offset": {
            "spotSepPix": spot_separation_spec,
            "roiRadiusPix": 6,
            "probeRotVecDeg": [0,],
            "nSubpixels": 100,
            "nSteps": 9,
            "stepSize": 1,
            "nIter": 6,
        },
        "separation": {
            "spotSepPix": spot_separation_spec,
            "roiRadiusPix": 4,
            "probeRotVecDeg": [0,],
            "nSubpixels": 100,
            "nSteps": 21,
            "stepSize": 0.25,
            "nIter": 5,
        }
    },
    "SPEC730": {
        "offset": {
            "spotSepPix": spot_separation_spec,
            "roiRadiusPix": 6,
            "probeRotVecDeg": [0,],
            "nSubpixels": 100,
            "nSteps": 9,
            "stepSize": 1,
            "nIter": 6,
        },
        "separation": {
            "spotSepPix": spot_separation_spec,
            "roiRadiusPix": 4,
            "probeRotVecDeg": [0,],
            "nSubpixels": 100,
            "nSteps": 21,
            "stepSize": 0.25,
            "nIter": 5,
        }
    },
    "WFOV": {
        "offset": {
            "spotSepPix": spot_separation_wfov,
            "roiRadiusPix": 4.5,
            "probeRotVecDeg": [0, 90],
            "nSubpixels": 100,
            "nSteps": 7,
            "stepSize": 1,
            "nIter": 6,
        },
        "separation": {
            "spotSepPix": spot_separation_wfov,
            "roiRadiusPix": 4.5,
            "probeRotVecDeg": [0, 90],
            "nSubpixels": 100,
            "nSteps": 21,
            "stepSize": 0.25,
            "nIter": 5,
        }
    }
}

def test_find_star_exact():
    """
    Generate mock input data and pass into find_star function
    """
    corgidrp.track_individual_errors = True # this test uses individual error components

    # Set the star center position for injection of satellite spots

    injected_position = (image_shape[1] // 2, image_shape[0] // 2)
    guess_position = (image_shape[1] // 2, image_shape[0] // 2)

    satellite_spot_angle_offset = 0
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

        assert np.isclose(injected_position[0], measured_x, atol=0.1), \
            f"{mode}. Expected {injected_position[0]}, got {measured_x}"
        assert np.isclose(injected_position[1], measured_y, atol=0.1), \
            f"{mode}. Expected {injected_position[1]}, got {measured_y}"

    corgidrp.track_individual_errors = old_err_tracking


def test_find_star_offset():
    """
    Generate mock input data and pass into find_star function with an offset guess
    """
    corgidrp.track_individual_errors = True # this test uses individual error components

    # Set the star center position for injection of satellite spots

    # Offset guess position
    injected_position = (image_shape[1] // 2, image_shape[0] // 2)
    guess_position = (image_shape[1] // 2 + 2, image_shape[0] // 2 - 1)

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

        assert np.isclose(injected_position[0], measured_x, atol=0.1), \
            f"{mode}. Expected {injected_position[0]}, got {measured_x}"
        assert np.isclose(injected_position[1], measured_y, atol=0.1), \
            f"{mode}. Expected {injected_position[1]}, got {measured_y}"

    corgidrp.track_individual_errors = old_err_tracking

if __name__ == "__main__":
    test_find_star_exact()
    test_find_star_offset()
