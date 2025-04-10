import numpy as np
from corgidrp.spectroscopy import fit_psf_centroid
from corgidrp.data import SpectroscopyCentroidPSF
from corgidrp import mocks

def compute_psf_centroid(psf_array, psf_table, pri_hdr=None, ext_hdr=None, input_dataset=None, output_path=None, verbose=False):
    """
    Compute PSF centroids for a grid of PSFs and store them in a calibration file.

    Args:
        psf_array (np.ndarray): 3D array of PSF images with shape (N, H, W).
        psf_table (astropy.Table): Table with initial guess centroids ('xcent', 'ycent') for each PSF.
        pri_hdr (fits.Header, optional): Primary FITS header. If None, empty header will be used.
        ext_hdr (fits.Header, optional): Extension FITS header. If None, empty header will be used.
        input_dataset (Dataset, optional): Reference dataset to record parent filenames.
        output_path (str, optional): If provided, write output to this FITS file.
        verbose (bool): Whether to print centroid positions.

    Returns:
        PSFCentroidCalibration: Calibration object with fitted (x, y) centroids.
    """
    if psf_array.ndim != 3:
        raise ValueError(f"Expected 3D PSF array, got shape {psf_array.shape}")
    if len(psf_array) != len(psf_table):
        raise ValueError("Mismatch between PSF array and metadata table.")

    centroids = np.zeros((len(psf_array), 2))  # Store (x, y) for each PSF

    for idx in range(len(psf_array)):
        psf_data = psf_array[idx]
        # Use table values as initial guesses for fitting
        xguess = psf_table['xcent'][idx]
        yguess = psf_table['ycent'][idx]

        xfit, yfit, *_ = fit_psf_centroid(
            psf_data, psf_data,  # Using PSF as its own template
            xguess, yguess,
            halfwidth=10, halfheight=10
        )

        centroids[idx, 0] = xfit
        centroids[idx, 1] = yfit

        if verbose:
            print(f"Slice {idx}: x = {xfit:.3f}, y = {yfit:.3f}")

    # Create calibration object with provided headers
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
