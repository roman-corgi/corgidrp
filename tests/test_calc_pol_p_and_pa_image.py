import numpy as np
from astropy.io.fits import Header
import pytest

from corgidrp.data import Image
from corgidrp.pol import calc_pol_p_and_pa_image


def create_mock_polimage(
        image_size=256,
        fwhm=100.0,
        I0=1e3,
        p=0.1,
        theta_deg=10.0,
        roll_angles=None,
        prisms=None,
        return_stokes=False,
):
    """
    Generate mock polarimetric images with controlled polarization angles.

    Args:
        image_size (int): Size of the square image (H x W).
        fwhm (float): Full width at half maximum of the Gaussian source in pixels.
        I0 (float): Peak intensity of the Gaussian source.
        p (float): Fractional polarization.
        theta_deg (float): Polarization angle in degrees.
        roll_angles (list of float, optional): Telescope roll angles for each prism.
        prisms (list of str, optional): Prism orientations ('POL0' or 'POL45').
        return_stokes (bool, optional): If True, return full Stokes cubes [I,Q,U],
            otherwise return prism pairs.

    Returns:
        Image: Image object containing either Stokes cubes or prism pair cubes
        in `Image.data` with corresponding `err` and `dq`.

    Raises:
        ValueError: If `roll_angles` and `prisms` lengths do not match, or
                    if an invalid prism string is provided.
    """
    if roll_angles is None:
        roll_angles = [-15, 15, -15, 15]
    if prisms is None:
        prisms = ['POL0', 'POL0', 'POL45', 'POL45']

    if len(roll_angles) != len(prisms):
        raise ValueError("roll_angles and prisms must have the same length")

    # Create 2D Gaussian intensity map
    y, x = np.mgrid[0:image_size, 0:image_size]
    x0 = y0 = image_size / 2.0
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    I_map = I0 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))

    cubes_stokes = []
    cubes_prism = []

    for i, prism in enumerate(prisms):
        theta_obs = np.radians(theta_deg + roll_angles[i])
        Q_map = I_map * p * np.cos(2 * theta_obs)
        U_map = I_map * p * np.sin(2 * theta_obs)

        # Stokes cube
        stokes_cube = np.stack([I_map, Q_map, U_map])
        cubes_stokes.append(stokes_cube)

        # Prism pair
        if prism == 'POL0':
            pair_cube = np.stack([0.5*(I_map+Q_map), 0.5*(I_map-Q_map)])
        elif prism == 'POL45':
            pair_cube = np.stack([0.5*(I_map+U_map), 0.5*(I_map-U_map)])
        else:
            raise ValueError(f"Invalid prism: {prism}")
        cubes_prism.append(pair_cube)

    cubes_stokes = np.array(cubes_stokes)[0]
    cubes_prism = np.array(cubes_prism)

    if return_stokes:
        # Add blank slices for V, Qphi, Uphi
        cubes_blank = np.zeros_like(cubes_stokes)
        cubes_stokes = np.concatenate([cubes_stokes, cubes_blank], axis=0)
        cubes_out = cubes_stokes
    else:
        cubes_out = cubes_prism

    err = np.sqrt(np.abs(cubes_out))
    dq = np.zeros_like(cubes_out, dtype=int)

    Image_polmock = Image(
        cubes_out,
        pri_hdr=Header(),
        ext_hdr=Header(),
        err=err,
        dq=dq,
        err_hdr=Header(),
        dq_hdr=Header()
    )
    
    return Image_polmock

    
def test_calc_pol_p_and_pa_image(p_input=0.1, theta_input=10.0):
    """Test calc_pol_p_and_pa_image using a mock Stokes cube."""

    # Generate mock Stokes cube
    Image_polmock = create_mock_polimage(
        p=p_input,
        theta_deg=theta_input,
        roll_angles=[0],
        prisms=['POL0'],
        return_stokes=True
    )

    # Compute polarization products
    image_pol = calc_pol_p_and_pa_image(Image_polmock)

    p_map = image_pol.data[0]       # fractional polarization
    evpa_map = image_pol.data[3]    # EVPA

    assert p_map == pytest.approx(p_input), "Polarization fraction recovery failed."
    assert evpa_map == pytest.approx(theta_input), "EVPA recovery failed."


if __name__ == "__main__":
    test_calc_pol_p_and_pa_image()
