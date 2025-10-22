# A file that holds the functions to handle polarimetry data 
import numpy as np
from astropy.io.fits import Header
from photutils.aperture import CircularAperture
from corgidrp.data import Image


def calc_stokes_unocculted(dataset,
                           pos=None,
                           encircled_radius=1.,
                           method='subpixel',
                           subpixels=5):
    """
    Compute Stokes vector (I, Q/I, U/I) from L3b polarimetric datacubes using aperture photometry.

    Each channel (I0/I90 or POL0/POL90) in each slice is treated separately; roll and prism are shared.

    Args:
        dataset (list of Image-like): L3b polarimetric datacubes with .data and .err arrays.
        pos (array_like, optional): (y, x) center of aperture. Defaults to image center.
        encircled_radius (float, optional): Aperture radius in pixels. Default 1.
        method (str, optional): Aperture photometry method. Default 'subpixel'.
        subpixels (int, optional): Number of subpixels per pixel for subpixel photometry.

    Raises:
        ValueError: If `dataset` is empty or contains unknown prism types.

    Returns:
        Image: Stokes vector image (I, Q/I, U/I) with propagated errors and headers.
    """

    if not dataset:
        raise ValueError("Dataset is empty")

    # --- Aperture setup ---
    if pos is None:
        pos = np.array(dataset[0].data.shape[1:]) / 2
    aper = CircularAperture(pos, encircled_radius)

    prism_map = {'POL0': [0., 90.], 'POL45': [45., 135.]}

    fluxes, flux_errs, thetas = [], [], []

    # --- Photometry loop ---
    for ds in dataset:
        roll = ds.pri_hdr.get('ROLL', 0.0)
        prism = ds.ext_hdr.get('DPAMNAME')
        if prism not in prism_map:
            raise ValueError(f"Unknown prism: {prism}")

        for i, phi in enumerate(prism_map[prism]):
            theta = np.radians(roll + phi)
            aperture_sum, aperture_err = aper.do_photometry(
                ds.data[i], error=ds.err[i], method=method, subpixels=subpixels
            )
            fluxes.append(aperture_sum[0])
            flux_errs.append(aperture_err[0])
            thetas.append(theta)

    fluxes = np.array(fluxes)
    flux_errs = np.array(flux_errs)
    thetas = np.array(thetas)

    # Prevent division by zero
    flux_errs[flux_errs == 0] = np.min(flux_errs[flux_errs > 0])

    # --- Weighted least squares ---
    A = np.vstack([np.ones_like(thetas),
                   0.5 * np.cos(2 * thetas),
                   0.5 * np.sin(2 * thetas)]).T
    W = np.diag(1.0 / flux_errs**2)
    cov = np.linalg.inv(A.T @ W @ A)
    params = cov @ (A.T @ W @ fluxes)
    I_val, Q_val, U_val = params
    I_err, Q_err, U_err = np.sqrt(np.diag(cov))

    # Fractional polarization
    Q_frac = Q_val / I_val
    U_frac = U_val / I_val
    Q_frac_err = np.sqrt((Q_err/I_val)**2 + (Q_val*I_err/I_val**2)**2)
    U_frac_err = np.sqrt((U_err/I_val)**2 + (U_val*I_err/I_val**2)**2)

    data_out = np.array([I_val, Q_frac, U_frac])
    err_out = np.array([I_err, Q_frac_err, U_frac_err])
    dq_out = np.zeros_like(data_out, dtype=int)

    # --- Headers ---
    pri_hdr = Header()
    ext_hdr = Header()
    ext_hdr.add_history("Computed uncalibrated Stokes parameters: I, Q/I, U/I")
    err_hdr = Header()
    dq_hdr = Header()

    return Image(
        data_out,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        err=err_out,
        dq=dq_out,
        err_hdr=err_hdr,
        dq_hdr=dq_hdr
    )
