import numpy as np
import pytest
import corgidrp.mocks as mocks
import corgidrp.l2b_to_l3 as l2b_to_l3
import corgidrp.l3_to_l4 as l3_to_l4
import corgidrp.data as data
import corgidrp.pol as pol
import os
import shutil

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

if __name__ == "__main__":
    test_image_splitting()
    test_subtract_stellar_polarization()