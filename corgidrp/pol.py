# A file that holds the functions to handle polarimetry data 
import numpy as np

from corgidrp.data import Image


def calc_pol_p_and_pa_image(input_Image):
    """Compute fractional polarization, polarization SNR, and EVPA from Stokes maps.

    Supports input images with either Q/U or Qphi/Uphi slices.

    Args:
        input_Image (Image): Object containing Stokes maps and uncertainties.

    Returns:
        Image: Image object containing
            - data: stacked [p, perr, psnr, evpa, evpa_err] (shape: 5 x H x W)
            - ext_hdr: Header with updated HISTORY
            - err, dq: arrays reflecting perr and evpa_err

    Raises:
        AttributeError: If `input_Image` is missing `data` or `err` attributes.
        ValueError: If Stokes maps I, Q, U have inconsistent shapes or insufficient slices.
    """
    # --- Extract Stokes parameters ---
    try:
        I = input_Image.data[0]
        Q = input_Image.data[1]
        U = input_Image.data[2]
        # V, Qphi, Uphi = Image.data[3:6] # unused

        if len(input_Image.err.shape) == 4:
            input_Image.err = input_Image.err[0]
        Ierr = input_Image.err[0]
        Qerr = input_Image.err[1]
        Uerr = input_Image.err[2]
        # Verr, Qphierr, Uphierr = input_Image.err[3:6] # unused
    except AttributeError as e:
        raise AttributeError("Image object must have 'data' and 'err' attributes.") from e
    except IndexError as e:
        raise ValueError("Image.data and Image.err must have at least [0..2] slices.") from e

    # --- Check shapes ---
    if I.shape != Q.shape or I.shape != U.shape:
        raise ValueError("Stokes I, Q, U maps must have the same shape.")

    # --- Polarized intensity and error ---
    P = np.sqrt(Q**2 + U**2)
    Perr = np.sqrt((Q*Qerr)**2 + (U*Uerr)**2) / np.maximum(P, 1e-10)
    Perr *= P  # scale back to absolute error

    # --- Fractional polarization and its error ---
    p = P / np.maximum(I, 1e-10)
    perr = np.sqrt((Perr/np.maximum(I,1e-10))**2 + (P*Ierr/np.maximum(I,1e-10)**2)**2)

    # --- Polarization angle (EVPA) and uncertainty ---
    evpa = 0.5 * np.arctan2(U, Q)  # radians
    evpa_err = 0.5 * np.sqrt((Q*Uerr)**2 + (U*Qerr)**2) / np.maximum(Q**2 + U**2, 1e-10)
    evpa = np.degrees(evpa)
    evpa_err = np.degrees(evpa_err)

    # --- Polarization SNR ---
    psnr = p / np.maximum(perr, 1e-10)

    # --- Stack results (axis=0: 5 x H x W) ---
    data_out = np.stack([p, perr, psnr, evpa, evpa_err], axis=0)

    # --- Headers ---
    pri_hdr = input_Image.pri_hdr
    ext_hdr = input_Image.ext_hdr
    ext_hdr.add_history("Polarization maps computed; derived p, EVPA, SNR")
    err_hdr = input_Image.err_hdr
    dq_hdr = input_Image.dq_hdr

    # --- err and dq arrays ---
    err_out = np.stack([perr, perr, np.zeros_like(psnr), evpa_err, evpa_err], axis=0)
    dq_out = np.zeros(data_out.shape, dtype=int)

    # --- Construct output Image ---
    Image_out = Image(
        data_out,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        err=err_out,
        dq=dq_out,
        err_hdr=err_hdr,
        dq_hdr=dq_hdr
    )

    return Image_out
