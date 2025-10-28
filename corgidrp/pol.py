# A file that holds the functions to handle polarimetry data 
import numpy as np

from corgidrp.data import Image
from corgidrp.fluxcal import aper_phot

def aper_phot_pol(image, phot_kwargs):
    """
    Perform aperture photometry on both channels of a 2-channel polarimetric image.

    Each input image contains two orthogonally polarized beams (e.g., ordinary and extraordinary).
    This function measures the flux and its uncertainty in both channels using aperture photometry.

    Args:
        image (Image): Polarimetric Image object with shape (2, ny, nx).
            Must contain `data`, `err`, and `dq` attributes.
        phot_kwargs (dict): Keyword arguments passed to `aper_phot`, defining
            aperture radius, centering method, background subtraction, etc.

    Returns:
        tuple[list, list]:
            (flux, flux_err) lists of fluxes and uncertainties for both polarization channels.
    """
    flux = []
    flux_err = []
    for i in range(2):
        im_copy = image.copy()
        im_copy.data = im_copy.data[i]
        im_copy.err = im_copy.err[0][i].reshape(np.append([1], [im_copy.data.shape]))
        im_copy.dq  = im_copy.dq[i]

        f, f_e = aper_phot(im_copy, **phot_kwargs)

        flux.append(f)
        flux_err.append(f_e)

    return flux, flux_err

def calc_stokes_unocculted(dataset,
                           phot_kwargs=None,
                           image_center_x=512, image_center_y=512):
    """
    Compute uncalibrated Stokes parameters (I, Q/I, U/I) from unocculted L3 polarimetric datacubes.

    Each element in `dataset` represents a single observation taken with a specific Wollaston prism
    (e.g., POL0 or POL45), which splits the incoming light into two orthogonally polarized beams.
    This function performs aperture photometry on each beam and computes the corresponding Stokes
    parameters in the instrument frame.

    Args:
        dataset (list of Image):
            List of L3 polarimetric images, each containing `.data`, `.err`, `.dq`, and FITS headers
            with keywords 'ROLL' and 'DPAMNAME'.
        phot_kwargs (dict, optional):
            Keyword arguments passed to `aper_phot`. If not provided, a default aperture setup is used.
        image_center_x (float, optional):
            X-coordinate of the aperture center in pixels. Default is 512.
        image_center_y (float, optional):
            Y-coordinate of the aperture center in pixels. Default is 512.

    Returns:
        Image:
            A `corgidrp.data.Image` instance containing:
            - `data` (ndarray, shape=(3,)): [I, Q/I, U/I]
            - `err`  (ndarray, shape=(3,)): propagated uncertainties
            - `dq`   (ndarray, shape=(3,)): data quality flags (zeros)
            - FITS headers (pri_hdr, ext_hdr, err_hdr, dq_hdr) propagated from the first input image.

    Raises:
        ValueError:
            If an input image contains an unrecognized prism name in 'DPAMNAME'.
    """
    # Ensure xy centering method is used with estimated centers for aperture photometry
    if phot_kwargs is None:
        phot_kwargs = {
            'encircled_radius': 5,
            'frac_enc_energy': 1.0,
            'method': 'subpixel',
            'subpixels': 5,
            'background_sub': False,
            'r_in': 5,
            'r_out': 10,
            'centroid_roi_radius': 5,
            'centering_initial_guess': [image_center_x, image_center_y]
        }

    prism_map = {'POL0': [0., 90.], 'POL45': [45., 135.]}

    fluxes, flux_errs, thetas = [], [], []

    # --- Photometry loop ---
    for ds in dataset:
        prism = ds.ext_hdr.get('DPAMNAME')
        if prism not in prism_map:
            raise ValueError(f"Unknown prism: {prism}")
        
        flux, flux_err = aper_phot_pol(ds, phot_kwargs)
        fluxes.append(flux)
        flux_errs.append(flux_err)
        
        for phi in prism_map[prism]:
            thetas.append(np.radians(phi))

    fluxes = np.array(fluxes)
    flux_errs = np.array(flux_errs)
    thetas = np.array(thetas)

    # Prevent division by zero
    flux_errs[flux_errs == 0] = np.min(flux_errs[flux_errs > 0])

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
