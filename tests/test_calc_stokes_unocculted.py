import os, glob
import numpy as np
import matplotlib.pyplot as plt

from corgidrp import data
from corgidrp.l3_to_l4 import calc_stokes_unocculted

def create_mock_polimage(image_size=256, fwhm=100.0, I0=1e3, p=0.1, theta_deg=10.0,
                         roll_angles=[-15, 15, -15, 15], prisms=['POL0', 'POL0', 'POL45', 'POL45'],
                         savefile=None):
    """
    Generate mock polarimetric images for testing.

    Args:
        image_size (int): Size of the square image in pixels.
        fwhm (float): FWHM of Gaussian source in pixels.
        I0 (float): Peak intensity.
        p (float): Fractional polarization.
        theta_deg (float): Polarization angle in degrees.
        roll_angles (list): Telescope roll angles for each prism.
        prisms (list): Prism orientations for each image pair.
        savefile (str, optional): FITS file path to save the simulated cube.
    
    Returns:
        list of ndarray: Mock polarimetric datacubes, shape (2, image_size, image_size) per prism.
    """
    y, x = np.mgrid[0:image_size, 0:image_size]
    x0 = y0 = image_size / 2.0

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    I = I0 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))

    cubes = []
    for i, prism in enumerate(prisms):
        theta_obs = np.radians(theta_deg + roll_angles[i])
        Q = I * p * np.cos(2 * theta_obs)
        U = I * p * np.sin(2 * theta_obs)

        if prism == 'POL0':
            pair_cube = [0.5*(I + Q), 0.5*(I - Q)]
        elif prism == 'POL45':
            pair_cube = [0.5*(I + U), 0.5*(I - U)]
        else:
            raise ValueError(f"Invalid prism: {prism}")

        cubes.append(np.stack(pair_cube))  # shape = (2, image_size, image_size)

    if savefile is not None:
        from astropy.io import fits
        fits.PrimaryHDU(cubes).writeto(savefile, overwrite=True)

    return cubes


if __name__ == '__main__':
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

        image_cubes = create_mock_polimage(p=p_val, roll_angles=rolls, prisms=prisms)#, savefile='')
        image_err = np.sqrt(np.abs(image_cubes))

        for i, dataset in enumerate(input_dataset):
            dataset.pri_hdr['ROLL'] = rolls[i]
            dataset.ext_hdr['DPAMNAME'] = prisms[i]
            dataset.data = image_cubes[i]
            dataset.err = image_err[i]

        Q_obs, U_obs = calc_stokes_unocculted(input_dataset)
        p_recovered.append(np.linalg.norm([Q_obs[0], U_obs[0]]))

    p_err = np.nanmean((np.array(p_recovered) - p_input) / p_input) * 1e2
    assert p_err < 1.6, f'p_err = {p_err:.2f}%, polarized fraction could not be recovered'

    #plt.scatter(p_input, p_recovered) ; plt.xlabel('p_input') ; plt.ylabel('p_recovered') ; plt.show()
