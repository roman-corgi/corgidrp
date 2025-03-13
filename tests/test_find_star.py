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
    satellite_spot_parameters = star_center.satellite_spot_parameters
    separation = satellite_spot_parameters['NFOV']['separation']['spotSepPix']

    # Generate test data
    input_dataset = mocks.create_satellite_spot_observing_sequence(
            n_sci_frames=3,
            n_satspot_frames=3, 
            image_shape=(201, 201),
            bg_sigma=1.0,
            bg_offset=10.0,
            gaussian_fwhm=5.0,
            separation=separation,
            center_offset=(0, 0),
            angle_offset=0,
            amplitude_multiplier=100)

    xOffsetGuess = 0
    yOffsetGuess = 0
    thetaOffsetGuess = 0

    dataset_with_center = find_star(
        input_dataset=input_dataset, xOffsetGuess=xOffsetGuess,
        yOffsetGuess=yOffsetGuess, thetaOffsetGuess=thetaOffsetGuess)

    assert dataset_with_center.frames[0].ext_hdr['STARLOCX'] < 0.1
    assert dataset_with_center.frames[0].ext_hdr['STARLOCY'] < 0.1

    corgidrp.track_individual_errors = old_err_tracking

if __name__ == "__main__":
    test_find_star()
