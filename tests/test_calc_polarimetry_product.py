import numpy as np
from astropy.io import fits
from corgidrp import data
from corgidrp.l4_to_tda import calc_polarimetry_product


def create_mock_polimage(
        image_size=256, fwhm=100.0, I0=1e3, p=0.1, theta_deg=10.0,
        roll_angles=[-15, 15, -15, 15], prisms=['POL0', 'POL0', 'POL45', 'POL45'],
        return_stokes=False, savefile=None
):
    """Generate mock polarimetric images with controlled polarization angles.

    Args:
        image_size (int): Size of the square image in pixels (H x W).
        fwhm (float): Full width at half maximum of the Gaussian source in pixels.
        I0 (float): Peak intensity of the Gaussian source.
        p (float): Fractional polarization.
        theta_deg (float): Polarization angle in degrees.
        roll_angles (list of float): Telescope roll angles for each prism.
        prisms (list of str): Prism orientations for each image ('POL0' or 'POL45').
        return_stokes (bool): If True, return full Stokes cubes [I,Q,U], else prism pairs.
        savefile (str or None): Optional path to save the simulated datacubes as a FITS file.

    Returns:
        list of np.ndarray: Mock polarimetric datacubes.
            - If return_stokes=True: each ndarray is a Stokes cube [I, Q, U] of shape (3, H, W).
            - If return_stokes=False: each ndarray is a prism pair [I+Q/I+U, I-Q/I-U] of shape (2, H, W).

    Raises:
        ValueError: If lengths of roll_angles and prisms do not match.
    """
    if len(roll_angles) != len(prisms):
        raise ValueError("roll_angles and prisms must have the same length")

    # Create 2D Gaussian intensity map
    y, x = np.mgrid[0:image_size, 0:image_size]
    x0 = y0 = image_size / 2.0
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    I_map = I0 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))

    cubes = []
    cubes_stokes = []

    for i, prism in enumerate(prisms):
        theta_obs = np.radians(theta_deg + roll_angles[i])
        Q_map = I_map * p * np.cos(2 * theta_obs)
        U_map = I_map * p * np.sin(2 * theta_obs)

        # Generate prism pair
        if prism == 'POL0':
            pair_cube = [0.5 * (I_map + Q_map), 0.5 * (I_map - Q_map)]
        elif prism == 'POL45':
            pair_cube = [0.5 * (I_map + U_map), 0.5 * (I_map - U_map)]
        else:
            raise ValueError(f"Invalid prism: {prism}")

        cubes.append(np.stack(pair_cube))             # shape (2,H,W)
        cubes_stokes.append(np.stack([I_map, Q_map, U_map]))  # shape (3,H,W)

    # Optional FITS save
    if savefile is not None:
        fits.PrimaryHDU(cubes_stokes).writeto(savefile, overwrite=True)

    # Select output
    cubes_out = cubes_stokes if return_stokes else cubes
    return cubes_out


if __name__ == "__main__":
    # -------------------------
    # Input parameters
    # -------------------------
    p_input = 0.1        # fractional polarization
    theta_input = 10.0   # polarization angle [deg]

    # Single prism test
    prisms = ['POL0']
    roll_angles = [0]

    # -------------------------
    # Generate mock Stokes cubes
    # -------------------------
    stokes_cubes = create_mock_polimage(
        p=p_input,
        theta_deg=theta_input,
        roll_angles=roll_angles,
        prisms=prisms,
        return_stokes=True,
        savefile=None
    )

    # -------------------------
    # Prepare Image object
    # -------------------------
    Image = data.Image
    Image.data = stokes_cubes
    Image.err = np.sqrt(np.maximum(np.abs(stokes_cubes), 1e-10))

    # -------------------------
    # Compute polarization products
    # -------------------------
    p_map, psnr_map, evpa_map = calc_polarimetry_product(Image)

    # -------------------------
    # Recovery check
    # -------------------------
    p_tol_frac = 1.6 / 100      # 1.6%
    evpa_tol_deg = 1.0          # 1 deg

    idx_p_fail = np.where(np.abs(p_map[0] - p_input) / p_input > p_tol_frac)[0]
    idx_evpa_fail = np.where(np.abs(evpa_map[0] - theta_input) > evpa_tol_deg)[0]

    assert len(idx_p_fail) == 0, f"Polarization fraction recovery failed: mean error={p_map[1].mean():.3f}"
    assert len(idx_evpa_fail) == 0, f"EVPA recovery failed: mean error={evpa_map[1].mean():.3f} deg"

    print("Mock polarimetry recovery test passed!")
