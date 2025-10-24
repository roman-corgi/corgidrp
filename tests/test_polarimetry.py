import numpy as np
import pytest
import corgidrp.mocks as mocks
import corgidrp.l2b_to_l3 as l2b_to_l3
import corgidrp.l3_to_l4 as l3_to_l4

import corgidrp.data as data
from corgidrp import star_center

def test_image_splitting():
    """
    Create mock L2b polarimetric images, check that it is split correctly
    """

    # test autocropping WFOV
    ## generate mock data
    image_WP1_wfov = mocks.create_mock_l2b_polarimetric_image(dpamname='POL0', observing_mode='WFOV', left_image_value=1, right_image_value=2)
    image_WP2_wfov = mocks.create_mock_l2b_polarimetric_image(dpamname='POL45', observing_mode='WFOV', left_image_value=1, right_image_value=2)
    input_dataset_wfov = data.Dataset([image_WP1_wfov, image_WP2_wfov])

    ## leave image_size parameter blank so the function automatically determines size
    output_dataset_autocrop_wfov = l2b_to_l3.split_image_by_polarization_state(input_dataset_wfov)
    ## create what the expected output data should look like
    radius_wfov = int(round((20.1 * ((0.8255 * 1e-6) / 2.363114) * 206265) / 0.0218))
    padding = 5
    img_size_wfov = 2 * (radius_wfov + padding)
    expected_output_autocrop_wfov = np.zeros(shape=(2, img_size_wfov, img_size_wfov))
    ## fill in expected values
    img_center_wfov = radius_wfov + padding
    y_wfov, x_wfov = np.indices([img_size_wfov, img_size_wfov])
    expected_output_autocrop_wfov[0, ((x_wfov-img_center_wfov)**2) + ((y_wfov-img_center_wfov)**2) <= radius_wfov**2] = 1
    expected_output_autocrop_wfov[1, ((x_wfov-img_center_wfov)**2) + ((y_wfov-img_center_wfov)**2) <= radius_wfov**2] = 2

    ## check that actual output is as expected
    assert output_dataset_autocrop_wfov.frames[0].data == pytest.approx(expected_output_autocrop_wfov)
    assert output_dataset_autocrop_wfov.frames[1].data == pytest.approx(expected_output_autocrop_wfov)

    # test autocropping NFOV
    ## generate mock data
    image_WP1_nfov = mocks.create_mock_l2b_polarimetric_image(dpamname='POL0', observing_mode='NFOV', left_image_value=1, right_image_value=2)
    image_WP2_nfov = mocks.create_mock_l2b_polarimetric_image(dpamname='POL45', observing_mode='NFOV', left_image_value=1, right_image_value=2)
    input_dataset_nfov = data.Dataset([image_WP1_nfov, image_WP2_nfov])

    ## leave image_size parameter blank so the function automatically determines size
    output_dataset_autocrop_nfov = l2b_to_l3.split_image_by_polarization_state(input_dataset_nfov)
    ## create what the expected output data should look like
    radius_nfov = int(round((9.7 * ((0.5738 * 1e-6) / 2.363114) * 206265) / 0.0218))
    img_size_nfov = 2 * (radius_nfov + padding)
    expected_output_autocrop_nfov = np.zeros(shape=(2, img_size_nfov, img_size_nfov))
    ## fill in expected values
    img_center_nfov = radius_nfov + padding
    y_nfov, x_nfov = np.indices([img_size_nfov, img_size_nfov])
    expected_output_autocrop_nfov[0, ((x_nfov-img_center_nfov)**2) + ((y_nfov-img_center_nfov)**2) <= radius_nfov**2] = 1
    expected_output_autocrop_nfov[1, ((x_nfov-img_center_nfov)**2) + ((y_nfov-img_center_nfov)**2) <= radius_nfov**2] = 2

    ## check that actual output is as expected
    assert output_dataset_autocrop_nfov.frames[0].data == pytest.approx(expected_output_autocrop_nfov)
    assert output_dataset_autocrop_nfov.frames[1].data == pytest.approx(expected_output_autocrop_nfov)

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

if __name__ == "__main__":
    test_image_splitting()
    test_align_frames()