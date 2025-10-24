import os, glob
import numpy as np
from astropy.stats import sigma_clip
import pytest
import corgidrp.mocks as mocks
import corgidrp.l2b_to_l3 as l2b_to_l3
import corgidrp.data as data
from corgidrp.pol import calc_stokes_unocculted

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
    
def test_calc_stokes_unocculted():
    """
    Test the calc_stokes_unocculted function using mock L2b polarimetric data.

    The test simulates multiple polarimetric datasets with varying fractional 
    polarization and observational errors, then compares recovered Stokes Q and U 
    against the input values in units of their propagated errors.
    """
    # --- Paths for test data ---
    mock_data_path = os.path.join(os.path.dirname(__file__), 'test_data/')
    files = glob.glob(mock_data_path + 'example_L1_input.fits')
    n_repeat = 8
    files = np.tile(files, n_repeat)
    input_dataset = data.Dataset(files)

    # --- Simulate varying polarization fractions ---
    n_sim = 100
    p_input = 0.1 + 0.2 * np.random.rand(n_sim)
    fractional_error = -5.0 + 1.0 * np.random.rand(n_sim)
    fractional_error = 10.0 ** fractional_error

    theta_deg = 22.5 # EVPA

    Q_recovered = []
    Qerr_recovered = []
    U_recovered = []
    Uerr_recovered = []

    # prisms and rolls
    # prisms = np.append(np.tile('POL0', n_repeat//2), np.tile('POL45', n_repeat//2))
    # rolls = np.append(np.tile([-15, 15], n_repeat//4), np.tile([-15, 15], n_repeat//4))
    
    # The simplest case
    prisms = np.append(np.tile('POL0', int(n_repeat / 2)), np.tile('POL45', int(n_repeat / 2)))
    rolls = np.full(n_repeat, 0.0)

    for p, fe in zip(p_input, fractional_error):
        # --- Generate mock L2b image ---
        Image_polmock = mocks.create_mock_stokes_image_l2b(
            fractional_error=fe,
            p=p,
            theta_deg=theta_deg,
            roll_angles=rolls,
            prisms=prisms
        )

        # --- Assign mock data to input_dataset ---
        for i, dataset in enumerate(input_dataset):
            dataset.pri_hdr['ROLL'] = rolls[i]
            dataset.ext_hdr['DPAMNAME'] = prisms[i]
            dataset.data = Image_polmock.data[i]
            dataset.err = Image_polmock.err[0][i]
            dataset.dq = Image_polmock.dq[i]

        # --- Compute unocculted Stokes ---
        Image_stokes_unocculted = calc_stokes_unocculted(input_dataset)

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
    theta_rad = np.radians(theta_deg)
    Q_input = p_input * np.cos(2 * theta_rad)
    Q_chi = (Q_recovered - Q_input) / Qerr_recovered
    U_input = p_input * np.sin(2 * theta_rad)
    U_chi = (U_recovered - U_input) / Uerr_recovered

    Q_chi = sigma_clip(Q_chi, sigma=5, maxiters=5)
    U_chi = sigma_clip(U_chi, sigma=5, maxiters=5)
    #print(np.median(Q_chi), np.std(Q_chi), np.median(U_chi), np.std(U_chi))
    
    # --- Assertions ---
    tol = 0.3
    assert np.median(Q_chi) == pytest.approx(0, abs=tol)
    assert np.std(Q_chi) == pytest.approx(1, abs=tol)
    assert np.median(U_chi) == pytest.approx(0, abs=tol)
    assert np.std(U_chi) == pytest.approx(1, abs=tol)

    return

if __name__ == "__main__":
    test_image_splitting()
    test_calc_stokes_unocculted()
