import copy
import datetime

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
                n_satspot_frames=6,
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
                n_satspot_frames=6,
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
    injected_position = [(2, 1), (-2, - 1)]

    satellite_spot_angle_offset = 3
    guess_angle_offset = 0

    modes = ['NFOV', 'WFOV']

    # Ascending SCTSRT timestamps for the three satspot acquisition groups
    sctsrt_no_offset = '2024-01-01T00:00:01'
    sctsrt_pos_offset = '2024-01-01T00:00:02'
    sctsrt_neg_offset = '2024-01-01T00:00:03'

    for mode in modes:
        separation = satellite_spot_parameters_defaults[mode]['separation']['spotSepPix']

        # Build three satspot frames per polarization type:
        # no-offset (amplitude=0), positive-offset, negative-offset.

        # POL0 satspot frames
        image_WP1_sp_nooffset = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(
            dpamname='POL0', observing_mode=mode,
            left_image_value=1, right_image_value=2,
            image_shape=(1024, 1024), bg_sigma=1.0, bg_offset=10.0, gaussian_fwhm=5.0,
            separation=separation, star_center=injected_position,
            angle_offset=satellite_spot_angle_offset, amplitude_multiplier=0)
        image_WP1_sp_nooffset.ext_hdr['SCTSRT'] = sctsrt_no_offset

        image_WP1_sp = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(
            dpamname='POL0', observing_mode=mode,
            left_image_value=1, right_image_value=2,
            image_shape=(1024, 1024), bg_sigma=1.0, bg_offset=10.0, gaussian_fwhm=5.0,
            separation=separation, star_center=injected_position,
            angle_offset=satellite_spot_angle_offset, amplitude_multiplier=100)
        image_WP1_sp.ext_hdr['SCTSRT'] = sctsrt_pos_offset

        image_WP1_sp_neg = copy.deepcopy(image_WP1_sp)
        image_WP1_sp_neg.ext_hdr['SCTSRT'] = sctsrt_neg_offset

        image_WP1 = mocks.create_mock_l2b_polarimetric_image(
            dpamname='POL0', observing_mode=mode,
            left_image_value=1, right_image_value=2)

        # POL45 satspot frames
        image_WP2_sp_nooffset = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(
            dpamname='POL45', observing_mode=mode,
            left_image_value=1, right_image_value=2,
            image_shape=(1024, 1024), bg_sigma=1.0, bg_offset=10.0, gaussian_fwhm=5.0,
            separation=separation, star_center=injected_position,
            angle_offset=satellite_spot_angle_offset, amplitude_multiplier=0)
        image_WP2_sp_nooffset.ext_hdr['SCTSRT'] = sctsrt_no_offset

        image_WP2_sp = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(
            dpamname='POL45', observing_mode=mode,
            left_image_value=1, right_image_value=2,
            image_shape=(1024, 1024), bg_sigma=1.0, bg_offset=10.0, gaussian_fwhm=5.0,
            separation=separation, star_center=injected_position,
            angle_offset=satellite_spot_angle_offset, amplitude_multiplier=100)
        image_WP2_sp.ext_hdr['SCTSRT'] = sctsrt_pos_offset

        image_WP2_sp_neg = copy.deepcopy(image_WP2_sp)
        image_WP2_sp_neg.ext_hdr['SCTSRT'] = sctsrt_neg_offset

        image_WP2 = mocks.create_mock_l2b_polarimetric_image(
            dpamname='POL45', observing_mode=mode,
            left_image_value=1, right_image_value=2)

        # Input order: no-offset, +offset, -offset satspot frames, then science frame per POL.
        # After split and find_star (drop_satspots_frames=False), frame layout:
        #   frames[0..2]: POL0 satspot (no-offset, +offset, -offset)
        #   frames[3]:    POL0 science
        #   frames[4..6]: POL45 satspot (no-offset, +offset, -offset)
        #   frames[7]:    POL45 science
        input_dataset = data.Dataset([
            image_WP1_sp_nooffset, image_WP1_sp, image_WP1_sp_neg, image_WP1,
            image_WP2_sp_nooffset, image_WP2_sp, image_WP2_sp_neg, image_WP2,
        ])
        input_dataset_autocrop = l2b_to_l3.split_image_by_polarization_state(input_dataset)

        thetaOffsetGuess = guess_angle_offset

        dataset_with_center = find_star(
            input_dataset=input_dataset_autocrop,
            thetaOffsetGuess=thetaOffsetGuess,
            drop_satspots_frames=False)

        measured_x_slice_0, measured_y_slice_0 = (dataset_with_center.frames[0].ext_hdr['STARLOCX'],
                                                   dataset_with_center.frames[0].ext_hdr['STARLOCY'])

        measured_x_slice_45, measured_y_slice_45 = (dataset_with_center.frames[4].ext_hdr['STARLOCX'],
                                                     dataset_with_center.frames[4].ext_hdr['STARLOCY'])

        injected_x_slice_0 = dataset_with_center.frames[0].data[0].shape[0] // 2 + injected_position[0][0]
        injected_y_slice_0 = dataset_with_center.frames[0].data[0].shape[1] // 2 + injected_position[0][1]
        injected_x_slice_45 = dataset_with_center.frames[4].data[0].shape[0] // 2 + injected_position[0][0]
        injected_y_slice_45 = dataset_with_center.frames[4].data[0].shape[1] // 2 + injected_position[0][1]

        # Test find star on pol 0
        assert np.isclose(injected_x_slice_0, measured_x_slice_0, atol=0.1), \
            f"{mode}. Expected {injected_x_slice_0}, got {measured_x_slice_0}"
        assert np.isclose(injected_y_slice_0, measured_y_slice_0, atol=0.1), \
            f"{mode}. Expected {injected_y_slice_0}, got {measured_y_slice_0}"

        # Test that the slices are correctly aligned: find star on the second (aligned) slice
        # of the +offset satspot frame and verify it matches the first slice.
        tuningParamDict = satellite_spot_parameters_defaults[mode]

        star_xy, list_spots_xy = star_center.star_center_from_satellite_spots(
            img_ref=dataset_with_center.frames[3].data[1],      # POL0 science, second slice
            img_sat_spot=dataset_with_center.frames[1].data[1], # POL0 +offset satspot, second slice
            star_coordinate_guess=(dataset_with_center.frames[1].data[1].shape[1] // 2,
                                   dataset_with_center.frames[1].data[1].shape[0] // 2),
            thetaOffsetGuess=thetaOffsetGuess,
            satellite_spot_parameters=tuningParamDict,
        )
        assert np.isclose(star_xy[0], measured_x_slice_0, atol=0.1), \
            f"{mode}. Expected {measured_x_slice_0}, got {star_xy[0]}"
        assert np.isclose(star_xy[1], measured_y_slice_0, atol=0.1), \
            f"{mode}. Expected {measured_y_slice_0}, got {star_xy[1]}"

        # Test find star on pol 45
        assert np.isclose(injected_x_slice_45, measured_x_slice_45, atol=0.1), \
            f"{mode}. Expected {injected_x_slice_45}, got {measured_x_slice_45}"
        assert np.isclose(injected_y_slice_45, measured_y_slice_45, atol=0.1), \
            f"{mode}. Expected {injected_y_slice_45}, got {measured_y_slice_45}"

        # Test that slices are correctly aligned for POL45
        star_xy, list_spots_xy = star_center.star_center_from_satellite_spots(
            img_ref=dataset_with_center.frames[7].data[1],      # POL45 science, second slice
            img_sat_spot=dataset_with_center.frames[5].data[1], # POL45 +offset satspot, second slice
            star_coordinate_guess=(dataset_with_center.frames[5].data[1].shape[1] // 2,
                                   dataset_with_center.frames[5].data[1].shape[0] // 2),
            thetaOffsetGuess=thetaOffsetGuess,
            satellite_spot_parameters=tuningParamDict,
        )
        assert np.isclose(star_xy[0], measured_x_slice_45, atol=0.1), \
            f"{mode}. Expected {measured_x_slice_45}, got {star_xy[0]}"
        assert np.isclose(star_xy[1], measured_y_slice_45, atol=0.1), \
            f"{mode}. Expected {measured_y_slice_45}, got {star_xy[1]}"

    corgidrp.track_individual_errors = old_err_tracking

def test_find_star_dataset_split():
    '''
    We want to test that the dataset splitting is working as expected  and then that find_star is working on the split datasets using the default split parameters.
    '''

    delta_position = 1 #Between the two visits
    injected_position = [[(2, 1), (-2, - 1)],[(2, 1+delta_position), (-2, - 1)]]
    n_visits = len(injected_position)

    satellite_spot_angle_offset = 3
    guess_angle_offset = 0

    # Ascending SCTSRT timestamps for the three satspot acquisition groups
    sctsrt_no_offset = '2024-01-01T00:00:01'
    sctsrt_pos_offset = '2024-01-01T00:00:02'
    sctsrt_neg_offset = '2024-01-01T00:00:03'

    input_images = []

    #We'll have several visits each with a different injected_position
    for visit in range(n_visits):
        mode = 'NFOV'
        separation = satellite_spot_parameters_defaults[mode]['separation']['spotSepPix']

        # POL0 satspot group: no-offset, +offset, -offset
        image_WP1_sp_nooffset = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(
            dpamname='POL0', observing_mode=mode,
            left_image_value=1, right_image_value=2,
            image_shape=(1024, 1024), bg_sigma=1.0, bg_offset=10.0, gaussian_fwhm=5.0,
            separation=separation, star_center=injected_position[visit],
            angle_offset=satellite_spot_angle_offset, amplitude_multiplier=0)
        image_WP1_sp_nooffset.pri_hdr['VISITID'] = int(image_WP1_sp_nooffset.pri_hdr['VISITID']) + visit
        image_WP1_sp_nooffset.ext_hdr['SCTSRT'] = sctsrt_no_offset

        image_WP1_sp = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(
            dpamname='POL0', observing_mode=mode,
            left_image_value=1, right_image_value=2,
            image_shape=(1024, 1024), bg_sigma=1.0, bg_offset=10.0, gaussian_fwhm=5.0,
            separation=separation, star_center=injected_position[visit],
            angle_offset=satellite_spot_angle_offset, amplitude_multiplier=100)
        image_WP1_sp.pri_hdr['VISITID'] = int(image_WP1_sp.pri_hdr['VISITID']) + visit
        image_WP1_sp.ext_hdr['SCTSRT'] = sctsrt_pos_offset

        image_WP1_sp_neg = copy.deepcopy(image_WP1_sp)
        image_WP1_sp_neg.ext_hdr['SCTSRT'] = sctsrt_neg_offset

        image_WP1 = mocks.create_mock_l2b_polarimetric_image(
            dpamname='POL0', observing_mode=mode,
            left_image_value=1, right_image_value=2)
        image_WP1.pri_hdr['VISITID'] = int(image_WP1.pri_hdr['VISITID']) + visit

        # POL45 satspot group: no-offset, +offset, -offset
        image_WP2_sp_nooffset = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(
            dpamname='POL45', observing_mode=mode,
            left_image_value=1, right_image_value=2,
            image_shape=(1024, 1024), bg_sigma=1.0, bg_offset=10.0, gaussian_fwhm=5.0,
            separation=separation, star_center=injected_position[visit],
            angle_offset=satellite_spot_angle_offset, amplitude_multiplier=0)
        image_WP2_sp_nooffset.pri_hdr['VISITID'] = int(image_WP2_sp_nooffset.pri_hdr['VISITID']) + visit
        image_WP2_sp_nooffset.ext_hdr['SCTSRT'] = sctsrt_no_offset

        image_WP2_sp = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(
            dpamname='POL45', observing_mode=mode,
            left_image_value=1, right_image_value=2,
            image_shape=(1024, 1024), bg_sigma=1.0, bg_offset=10.0, gaussian_fwhm=5.0,
            separation=separation, star_center=injected_position[visit],
            angle_offset=satellite_spot_angle_offset, amplitude_multiplier=100)
        image_WP2_sp.pri_hdr['VISITID'] = int(image_WP2_sp.pri_hdr['VISITID']) + visit
        image_WP2_sp.ext_hdr['SCTSRT'] = sctsrt_pos_offset

        image_WP2_sp_neg = copy.deepcopy(image_WP2_sp)
        image_WP2_sp_neg.ext_hdr['SCTSRT'] = sctsrt_neg_offset

        image_WP2 = mocks.create_mock_l2b_polarimetric_image(
            dpamname='POL45', observing_mode=mode,
            left_image_value=1, right_image_value=2)
        image_WP2.pri_hdr['VISITID'] = int(image_WP2.pri_hdr['VISITID']) + visit

        input_images.extend([
            image_WP1_sp_nooffset, image_WP1_sp, image_WP1_sp_neg, image_WP1,
            image_WP2_sp_nooffset, image_WP2_sp, image_WP2_sp_neg, image_WP2,
        ])

    input_dataset = data.Dataset(input_images)
    input_dataset_autocrop = l2b_to_l3.split_image_by_polarization_state(input_dataset)

    dataset_with_center = find_star(
        input_dataset=input_dataset_autocrop,
        thetaOffsetGuess=guess_angle_offset,
        drop_satspots_frames=False,
        )

    #Check that the two visits have different star centers by delta_position in injected position,
    # and that the measured star centers reflect this. This tests that the dataset splitting is working as expected,
    # and that find_star is working on the split datasets using the default split parameters.
    measured_x_visit_0, measured_y_visit_0 = (dataset_with_center.frames[0].ext_hdr['STARLOCX'],
                                               dataset_with_center.frames[0].ext_hdr['STARLOCY'])
    measured_x_visit_1, measured_y_visit_1 = (dataset_with_center.frames[-1].ext_hdr['STARLOCX'],
                                               dataset_with_center.frames[-1].ext_hdr['STARLOCY'])

    assert np.isclose(measured_x_visit_0, measured_x_visit_1, atol=0.1), \
        f"Expected the same x position for both visits, got {measured_x_visit_0} and {measured_x_visit_1}"
    assert np.isclose(measured_y_visit_0, measured_y_visit_1 - delta_position, atol=0.1), \
        f"Expected y position for visit 1 to be {delta_position} pixels different from visit 0, got {measured_y_visit_0} and {measured_y_visit_1}"



def test_satspot_invalid_count():
    """
    Verify that find_star raises ValueError when the number of SATSPOTS=True frames
    in a group is not divisible by 3.
    """
    mode = 'NFOV'
    separation = satellite_spot_parameters_defaults[mode]['separation']['spotSepPix']

    # Create a valid 6-frame dataset, then drop one satspot frame to get 5 (not divisible by 3)
    valid_dataset = mocks.create_satellite_spot_observing_sequence(
        n_sci_frames=3,
        n_satspot_frames=6,
        image_shape=image_shape,
        separation=separation,
        observing_mode=mode)

    sci_frames = [f for f in valid_dataset.frames if not f.ext_hdr['SATSPOTS']]
    satspot_frames = [f for f in valid_dataset.frames if f.ext_hdr['SATSPOTS']]
    invalid_dataset = data.Dataset(sci_frames + satspot_frames[:-1])  # 5 satspot frames

    raised = False
    try:
        find_star(invalid_dataset)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError when satspot frame count is not divisible by 3"


def test_find_star_skips_unocculted_star_fpm_frames():
    """
    Verify that unocculted-star frames with OPEN/ND FPAMNAME values skip satellite-spot
    analysis and are returned without populating STARLOC headers.
    """
    frames = []
    for i, fpamname in enumerate(['OPEN_12', 'ND225']):
        prihdr, exthdr, errhdr, dqhdr = mocks.create_default_L3_headers()
        prihdr['VISITID'] = '0000000000000000001'
        exthdr['DPAMNAME'] = 'POL0'
        exthdr['FPAMNAME'] = fpamname
        exthdr['FSMPRFL'] = 'NFOV'
        if 'SATSPOTS' in exthdr:
            del exthdr['SATSPOTS']
        frame = data.Image(
            np.ones((2, 20, 30)) * i,
            pri_hdr=prihdr,
            ext_hdr=exthdr,
            err_hdr=errhdr,
            dq_hdr=dqhdr,
        )
        frames.append(frame)

    dataset_with_center = find_star(data.Dataset(frames))

    assert len(dataset_with_center.frames) == 2
    for frame in dataset_with_center.frames:
        assert 'find_star skipped' in str(frame.ext_hdr['HISTORY'])


if __name__ == "__main__":
    # test_find_star_offset()
    # test_overwrite_parameters()
    # test_find_star_polarimetry()
    test_find_star_dataset_split()
