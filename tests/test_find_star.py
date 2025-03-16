import numpy as np

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
from corgidrp import star_center
from corgidrp.l3_to_l4 import find_star

old_err_tracking = corgidrp.track_individual_errors
# use default parameters
detector_params = data.DetectorParams({})

def test_find_star():
    """
    Generate mock input data and pass into find_star function
    """
    corgidrp.track_individual_errors = True # this test uses individual error components

    # Read initial parameters for testing

    satellite_spot_parameters = {
        "NFOV": {
            "offset": {
                "spotSepPix": 14.79,
                "roiRadiusPix": 4.5,
                "probeRotVecDeg": [0, 90],
                "nSubpixels": 100,
                "nSteps": 7,
                "stepSize": 1,
                "nIter": 6,
            },
            "separation": {
                "spotSepPix": 14.79,
                "roiRadiusPix": 1.5,
                "probeRotVecDeg": [0, 90],
                "nSubpixels": 100,
                "nSteps": 21,
                "stepSize": 0.25,
                "nIter": 5,
            }
        }
    }

    separation = satellite_spot_parameters['NFOV']['separation']['spotSepPix']

    # Set the star center position for injection of satellite spots
    image_shape = (201, 201)
    injected_position = (111, 111)

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
            angle_offset=0,
            amplitude_multiplier=100)

    # Set initial guesses for angle offset
    thetaOffsetGuess = 0

    dataset_with_center = find_star(
        input_dataset=input_dataset, 
        star_coordinate_guess=injected_position,
        thetaOffsetGuess=thetaOffsetGuess)

    measured_x, measured_y = (dataset_with_center.frames[0].ext_hdr['STARLOCX'],
                            dataset_with_center.frames[0].ext_hdr['STARLOCY'])

    assert np.isclose(injected_position[0], measured_x, atol=0.1)
    assert np.isclose(injected_position[1], measured_y, atol=0.1)

    corgidrp.track_individual_errors = old_err_tracking

if __name__ == "__main__":
    test_find_star()
