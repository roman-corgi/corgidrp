import numpy as np
from corgidrp.spectroscopy import fit_psf_centroid
from corgidrp.data import SpectroscopyCentroidPSF
from corgidrp import mocks

def compute_psf_centroid(psf_array, initial_cent, pri_hdr=None, ext_hdr=None, input_dataset=None, output_path=None, verbose=False, halfwidth=10, halfheight=10):
    """
    Compute PSF centroids for a grid of PSFs and store them in a calibration file.

    Args:
        psf_array (np.ndarray): 3D array of PSF images with shape (N, H, W).
        initial_cent (dict): Dictionary with initial guesses for PSF centroids.
                             Must include keys 'xcent' and 'ycent', each mapping to an array of shape (N,).
                             Example:
                                 initial_cent = {
                                     'xcent': [12.0, 15.5, 10.3],
                                     'ycent': [8.2, 14.1, 9.9]
                                 }
        pri_hdr (fits.Header, optional): Primary FITS header. If None, empty header will be used.
        ext_hdr (fits.Header, optional): Extension FITS header. If None, empty header will be used.
        input_dataset (Dataset, optional): Reference dataset to record parent filenames.
        output_path (str, optional): If provided, write output to this FITS file.
        verbose (bool): Whether to print centroid positions.

    Returns:
        SpectroscopyCentroidPSF: Calibration object with fitted (x, y) centroids.
    """
    if psf_array.ndim != 3:
        raise ValueError(f"Expected 3D PSF array, got shape {psf_array.shape}")

    if not isinstance(initial_cent, dict):
        raise TypeError("initial_cent must be a dictionary with 'xcent' and 'ycent' keys.")

    xcent = np.asarray(initial_cent.get("xcent"))
    ycent = np.asarray(initial_cent.get("ycent"))

    if xcent is None or ycent is None:
        raise ValueError("initial_cent dictionary must contain 'xcent' and 'ycent' arrays.")
    if len(psf_array) != len(xcent) or len(psf_array) != len(ycent):
        raise ValueError("Mismatch between PSF array length and guess arrays.")

    centroids = np.zeros((len(psf_array), 2))  # Store (x, y) for each PSF

    for idx in range(len(psf_array)):
        psf_data = psf_array[idx]
        xguess = xcent[idx]
        yguess = ycent[idx]

        xfit, yfit, *_ = fit_psf_centroid(
            psf_data, psf_data,  # Using PSF as its own template
            xguess, yguess,
            halfwidth=halfwidth,
            halfheight=halfheight
        )

        centroids[idx, 0] = xfit
        centroids[idx, 1] = yfit

        if verbose:
            print(f"Slice {idx}: x = {xfit:.3f}, y = {yfit:.3f}")

    calibration = SpectroscopyCentroidPSF(
        centroids,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        input_dataset=input_dataset
    )

    if output_path:
        calibration.save(output_path)
        print(f"Calibration written to: {output_path}")

    return calibration
