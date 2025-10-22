import os, glob
import numpy as np
import pytest

from corgidrp import data
from corgidrp.mocks import create_mock_polimage
from corgidrp.pol import calc_stokes_unocculted


def test_calc_stokes_unocculted():
    
    # Paths for test data
    mock_data_path = os.path.join(os.path.dirname(__file__), 'test_data/')
    files = glob.glob(mock_data_path + 'example_L1_input.fits')
    n_repeat = 8
    files = np.tile(files, n_repeat)
    input_dataset = data.Dataset(files)

    # Simulate varying polarization fractions
    p_input = np.linspace(0.01, 0.3, 10)
    p_recovered = []

    for p_val in p_input:
        prisms = np.append(np.tile('POL0', n_repeat//2), np.tile('POL45', n_repeat//2))
        rolls = np.append(np.tile([-15, 15], n_repeat//4), np.tile([-15, 15], n_repeat//4))

        Image_polmock = create_mock_polimage(p=p_val, roll_angles=rolls, prisms=prisms, return_stokes=False)

        for i, dataset in enumerate(input_dataset):
            dataset.pri_hdr['ROLL'] = rolls[i]
            dataset.ext_hdr['DPAMNAME'] = prisms[i]
            dataset.data = Image_polmock.data[i]
            dataset.err = abs(Image_polmock.data[i]) * 1e-3 # 0.1 % err

        Image_stokes_unocculted = calc_stokes_unocculted(input_dataset)
        Q_obs = Image_stokes_unocculted.data[1]
        U_obs = Image_stokes_unocculted.data[2]
        p_recovered.append(np.linalg.norm([Q_obs, U_obs]))

    assert p_recovered == pytest.approx(p_input, rel=1e-1)

    return

if __name__ == '__main__':
    test_calc_stokes_unocculted()
