import os, glob
import numpy as np
from astropy.stats import sigma_clip
import pytest

from corgidrp import data
from corgidrp.mocks import create_mock_stokes_image_l2b
from corgidrp.pol import calc_stokes_unocculted

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
        Image_polmock = create_mock_stokes_image_l2b(
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

if __name__ == '__main__':
    test_calc_stokes_unocculted()
