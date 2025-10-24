# A file that holds the functions to handle polarimetry data 
import numpy as np
from astropy.io.fits import Header

from corgidrp.data import Image
from corgidrp.fluxcal import aper_phot

def calc_stokes_unocculted(dataset,
                           encircled_radius=3., pos=None):
    """
    Compute the uncalibrated Stokes parameters (I, Q/I, U/I) from L2b polarimetric datacubes.

    Each dataset corresponds to a specific Wollaston prism (e.g., POL0 or POL45)
    and roll angle. Within each dataset, two channels correspond to orthogonal
    analyzer orientations (e.g., 0 deg and 90 deg for POL0).

    The function performs aperture photometry on each channel and solves for the
    Stokes vector using a weighted least-squares fit.

    Args:
        dataset (list of Image): List of L3b polarimetric datacubes. Each entry
            must have `.data`, `.err`, and `.dq` arrays, and FITS headers with
            keywords 'ROLL' and 'DPAMNAME'.
        encircled_radius (float, optional): Aperture radius in pixels. Default is 3.0.
        pos (array_like, optional): (y, x) coordinates of the aperture center.
            Defaults to the center of the first image.

    Returns:
        Image: A `corgidrp.data.Image` instance containing:
            - data (ndarray, shape=(3,)): [I, Q/I, U/I]
            - err (ndarray, shape=(3,)): propagated uncertainties
            - dq (ndarray, shape=(3,)): quality flags (zeros)
            - pri_hdr, ext_hdr, err_hdr, dq_hdr: corresponding FITS headers

    Raises:
        ValueError: If the dataset is empty or contains unknown prism types.
    """

    if not dataset:
        raise ValueError("Dataset is empty")

    # --- Aperture setup ---
    if pos is None:
        pos = np.array(dataset[0].data.shape[1:]) / 2

    prism_map = {'POL0': [0., 90.], 'POL45': [45., 135.]}

    fluxes, flux_errs, thetas = [], [], []

    # --- Photometry loop ---
    for ds in dataset:
        roll = ds.pri_hdr.get('ROLL')
        prism = ds.ext_hdr.get('DPAMNAME')
        if prism not in prism_map:
            raise ValueError(f"Unknown prism: {prism}")

        for i, phi in enumerate(prism_map[prism]):
            # Handle the 2-channel polarimetric data cube of shape [2, H, W]
            ds_copy = ds.copy()
            ds_copy.data = ds_copy.data[i]
            ds_copy.err = ds_copy.err[i].reshape(np.append([1],[ds_copy.data.shape]))
            ds_copy.dq  = ds_copy.dq[i]
            
            flux, flux_err = aper_phot(ds_copy, encircled_radius,
                                       centering_method='xy', centering_initial_guess=pos)
            
            fluxes.append(flux)
            flux_errs.append(flux_err)
            thetas.append(np.radians(roll + phi))

    fluxes = np.array(fluxes)
    flux_errs = np.array(flux_errs)
    thetas = np.array(thetas)

    # Prevent division by zero
    flux_errs[flux_errs == 0] = np.min(flux_errs[flux_errs > 0])

    # --- Weighted least squares ---
    A = np.vstack([np.ones_like(thetas),
                   np.cos(2 * thetas),
                   np.sin(2 * thetas)]).T
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
    ext_hdr.add_history("Computed uncalibrated Stokes parameters: data=[I, Q/I, U/I]")
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
