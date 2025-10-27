# A file that holds the functions to handle polarimetry data 
import numpy as np

from corgidrp.data import Image
from corgidrp.fluxcal import aper_phot

def aper_phot_pol(ds, i, encircled_radius, pos):
    """
    Perform aperture photometry on one channel of a 2-channel polarimetric image.

    Args:
        ds (Image): Polarimetric Image object with 2-channel data.
        i (int): Channel index (0 or 1).
        encircled_radius (float): Aperture radius in pixels.
        pos (array_like): (y, x) coordinates of the aperture center.

    Returns:
        tuple: (flux, flux_err) measured in the aperture.
    """
    ds_copy = ds.copy()
    ds_copy.data = ds_copy.data[i]
    ds_copy.err = ds_copy.err[0][i].reshape(np.append([1], [ds_copy.data.shape]))
    ds_copy.dq  = ds_copy.dq[i]

    return aper_phot(ds_copy, encircled_radius,
                     centering_method='xy',
                     centering_initial_guess=pos)

def calc_stokes_unocculted(dataset,
                           encircled_radius=3., pos=None,
                           onskystokes=False):
    """
    Compute the uncalibrated Stokes parameters (I, Q/I, U/I) from L3 polarimetric datacubes.

    Each dataset contains multiple images taken with particular Wollaston prisms
    (e.g., POL0 or POL45). Each image has two analyzer orientations
    (e.g., 0 and 90 degrees for POL0, 45 and 135 degrees for POL45). 
    For each image and its roll angle, the function performs aperture photometry
    on both channels and computes the uncalibrated Stokes parameters. 
    The calculation can be done either as a simple left-right difference (instrument coordinates)
    or using a weighted least-squares fit to account for sky rotation (sky coordinates).

    Args:
        dataset (list of Image): List of L polarimetric datacubes. Each entry
            must have `.data`, `.err`, and `.dq` arrays, and FITS headers with
            keywords 'ROLL' and 'DPAMNAME'.
        encircled_radius (float, optional): Aperture radius in pixels. Default is 3.0.
        pos (array_like, optional): (y, x) coordinates of the aperture center.
            Defaults to the center of the first image.
        onskystokes (bool, optional): If True, compute Stokes in sky coordinates
            using weighted least squares. If False, return instrument coordinates
            (left-right difference).

    Returns:
        Image: A `corgidrp.data.Image` instance containing:
            - data (ndarray, shape=(3,)): [I, Q/I, U/I]
            - err (ndarray, shape=(3,)): propagated uncertainties
            - dq (ndarray, shape=(3,)): quality flags (zeros)
            - pri_hdr, ext_hdr, err_hdr, dq_hdr: corresponding FITS headers

    Raises:
        ValueError: If an image has an unknown prism name.
    """

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
            flux, flux_err = aper_phot_pol(ds, i, encircled_radius, pos)
            fluxes.append(flux)
            flux_errs.append(flux_err)
            if onskystokes:
                thetas.append(np.radians(roll + phi))
            else:
                thetas.append(np.radians(phi))

    fluxes = np.array(fluxes)
    flux_errs = np.array(flux_errs)
    thetas = np.array(thetas)

    # Prevent division by zero
    flux_errs[flux_errs == 0] = np.min(flux_errs[flux_errs > 0])

    if onskystokes:
        # --- Weighted least squares ---
        A = np.vstack([np.ones_like(thetas),
                       np.cos(2 * thetas),
                       np.sin(2 * thetas)]).T
        W = np.diag(1.0 / flux_errs**2)
        cov = np.linalg.inv(A.T @ W @ A)
        params = cov @ (A.T @ W @ fluxes)
        I_val, Q_val, U_val = params
        I_err, Q_err, U_err = np.sqrt(np.diag(cov))
    else:
        # --- Instrument coordinates: left - right ---
        n_images = len(dataset)
        fluxes = fluxes.reshape([n_images, 2])
        flux_errs = flux_errs.reshape([n_images, 2])
        thetas = thetas.reshape([n_images, 2])
        I_vals = np.sum(fluxes, axis=1)
        I_errs = np.sqrt(np.sum(flux_errs**2, axis=1))
        
        QU_vals = fluxes[:,0] - fluxes[:,1]  # left - right
        
        # Weighted means across all prisms
        wI = 1.0 / I_errs**2
        I_val = np.sum(I_vals * wI) / np.sum(wI)
        I_err = 1.0 / np.sqrt(np.sum(wI))

        idx_0 = np.where(np.degrees(thetas[:,0]) == 0)[0]
        if idx_0.size > 0:
            Q_vals = QU_vals[idx_0]
            Q_val = np.sum(Q_vals * wI[idx_0]) / np.sum(wI[idx_0])
            Q_err = 1.0 / np.sqrt(np.sum(wI[idx_0]))
        else:
            Q_val = 0.
            Q_err = 0.

        idx_45 = np.where(np.degrees(thetas[:,0]) == 45)[0]
        if idx_45.size > 0:
            U_vals = QU_vals[idx_45]
            U_val = np.sum(U_vals * wI[idx_45]) / np.sum(wI[idx_45])
            U_err = 1.0 / np.sqrt(np.sum(wI[idx_45]))
        else:
            U_val = 0.
            U_err = 0.

    # Fractional polarization
    Q_frac = Q_val / I_val
    U_frac = U_val / I_val
    Q_frac_err = np.sqrt((Q_err/I_val)**2 + (Q_val*I_err/I_val**2)**2)
    U_frac_err = np.sqrt((U_err/I_val)**2 + (U_val*I_err/I_val**2)**2)

    data_out = np.array([I_val, Q_frac, U_frac])
    err_out = np.array([I_err, Q_frac_err, U_frac_err])
    dq_out = np.zeros_like(data_out, dtype=int)

    # --- Headers ---
    pri_hdr = dataset[0].pri_hdr
    ext_hdr = dataset[0].ext_hdr
    ext_hdr.add_history("Computed uncalibrated Stokes parameters: data=[I, Q/I, U/I]")
    err_hdr = dataset[0].err_hdr
    dq_hdr = dataset[0].dq_hdr

    return Image(
        data_out,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        err=err_out,
        dq=dq_out,
        err_hdr=err_hdr,
        dq_hdr=dq_hdr
    )
