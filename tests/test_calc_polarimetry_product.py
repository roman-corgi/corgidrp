import numpy as np
from corgidrp import data
from corgidrp.l4_to_tda import calc_polarimetry_product

def create_mock_polimage(image_size=256, fwhm=100.0, I0=1e3, p=0.1, theta_deg=10.0,
                         roll_angles=[-15, 15, -15, 15], prisms=['POL0', 'POL0', 'POL45', 'POL45'],
                         return_stokes=False, savefile=None):
    """
    Generate mock polarimetric images for testing with angle control.

    Parameters
    ----------
    image_size : int
        Size of the square image (pixels)
    fwhm : float
        Full width at half maximum of Gaussian source (pixels)
    I0 : float
        Peak intensity
    p : float
        Fractional polarization
    theta_deg : float
        Polarization angle in degrees
    roll_angles : list of float
        Telescope roll angles for each prism
    prisms : list of str
        Prism orientations ('POL0' or 'POL45') for each image
    return_stokes : bool
        If True, return full Stokes cubes [I,Q,U], else return prism pairs
    savefile : str or None
        Path to save FITS file (optional)

    Returns
    -------
    list of ndarray
        Mock polarimetric datacubes
    """

    if len(roll_angles) != len(prisms):
        raise ValueError("roll_angles and prisms must have the same length")

    # Create 2D Gaussian
    y, x = np.mgrid[0:image_size, 0:image_size]
    x0 = y0 = image_size / 2.0
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    I = I0 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))

    cubes = []
    cubes_stokes = []

    for i, prism in enumerate(prisms):
        theta_obs = np.radians(theta_deg + roll_angles[i])
        Q = I * p * np.cos(2 * theta_obs)
        U = I * p * np.sin(2 * theta_obs)

        # Generate prism pair
        if prism == 'POL0':
            pair_cube = [0.5*(I + Q), 0.5*(I - Q)]
        elif prism == 'POL45':
            pair_cube = [0.5*(I + U), 0.5*(I - U)]
        else:
            raise ValueError(f"Invalid prism: {prism}")

        cubes.append(np.stack(pair_cube))          # shape (2,H,W)
        cubes_stokes.append(np.stack([I, Q, U]))   # shape (3,H,W)

    # Optional FITS save
    if savefile is not None:
        fits.PrimaryHDU(cubes_stokes).writeto(savefile, overwrite=True)

    return cubes_stokes if return_stokes else cubes

if __name__ == "__main__":

    # -------------------------
    # Input parameters
    # -------------------------
    p_input = 0.1         # fractional polarization
    theta_input = 10.0    # polarization angle [deg]

    # Single prism test
    prisms = ['POL0']
    roll_angles = [0]

    # -------------------------
    # Generate mock Stokes cubes
    # -------------------------
    stokes_cubes = create_mock_polimage(
        p=p_input, theta_deg=theta_input,
        roll_angles=roll_angles, prisms=prisms,
        return_stokes=True, savefile=None
    )

    # -------------------------
    # Prepare Image object
    # -------------------------
    Image = data.Image
    Image.data = stokes_cubes
    # Use sqrt(I) as simple error estimate, avoid zeros
    Image.err = np.sqrt(np.maximum(np.abs(stokes_cubes), 1e-10))

    # -------------------------
    # Compute polarization products
    # -------------------------
    p_map, psnr_map, evpa_map = calc_polarimetry_product(Image)

    # -------------------------
    # Recovery check
    # -------------------------
    # Tolerance thresholds
    p_tol_frac = 1.6 / 100          # 1.6 %
    evpa_tol_deg = 1.0              # 1 degree

    # Find pixels where recovered values exceed tolerance
    idx_p_fail = np.where(np.abs(p_map[0] - p_input) / p_input > p_tol_frac)[0]
    idx_evpa_fail = np.where(np.abs(evpa_map[0] - theta_input) > evpa_tol_deg)[0]

    # Assertions
    assert len(idx_p_fail) == 0, f"Polarization fraction recovery failed: mean error={p_map[1].mean():.3f}"
    assert len(idx_evpa_fail) == 0, f"EVPA recovery failed: mean error={evpa_map[1].mean():.3f} deg"

    print("Mock polarimetry recovery test passed!")
