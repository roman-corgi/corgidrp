"""
From requirement 1090877:

Given:
1) M sets of clean focal-plane images collected at different FSM positions on 
   an unocculted spectrophotometric standard star with an ND filter in place.
2) A single absolute flux calibration value for the same CFAM filter.

This script computes a "sweet-spot dataset," which is an Mx3 matrix containing:
   - OD (optical density or related attenuation metric) for each dither
   - EXCAM (x,y) star center positions on the detector.

It also stores the FPAM encoder position associated with this dataset.

Note: 
- This dataset captures small-scale variation of focal-plane attenuation.
- The FPAM encoder position should be stored with the dataset for future use.

Author: Julia Milton
Date: 2024-12-09
"""

import numpy as np
from astropy.io import fits
from corgidrp.data import Dataset 

def compute_centroid(image_data):
    """
    Compute the centroid of the star in the given image.

    Parameters
    ----------
    image_data : 2D np.ndarray
        Science image data array.

    Returns
    -------
    (x_center, y_center) : (float, float)
        Estimated centroid of the star in pixel coordinates.
    """
    # Placeholder centroid calculation (e.g., simple intensity-weighted centroid):
    y_indices, x_indices = np.indices(image_data.shape)
    total_flux = np.sum(image_data)
    if total_flux <= 0:
        # Handle edge case
        return np.nan, np.nan
    x_center = np.sum(x_indices * image_data) / total_flux
    y_center = np.sum(y_indices * image_data) / total_flux
    return x_center, y_center

def compute_flux_in_image(image_data, x_center, y_center, radius=5):
    """
    Compute the integrated flux in photoelectrons for the star near the given center.

    Parameters
    ----------
    image_data : 2D np.ndarray
        The image data.
    x_center, y_center : float
        The centroid coordinates of the star.
    radius : float
        Aperture radius in pixels.

    Returns
    -------
    flux : float
        The total flux in the defined aperture.
    """
    y_indices, x_indices = np.indices(image_data.shape)
    r = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
    aperture_mask = (r <= radius)
    flux = np.sum(image_data[aperture_mask])
    return flux

def compute_od(flux, flux_calibration):
    """
    Compute the OD (or attenuation factor) from the measured flux and the
    known flux calibration value.

    Parameters
    ----------
    flux : float
        Measured flux in photoelectrons.
    flux_calibration : float
        Absolute flux calibration factor.

    Returns
    -------
    od : float
        Computed optical density or attenuation metric.
    """
    # Placeholder:
    od = flux * flux_calibration
    return od

def main(dataset_path, flux_calibration, output_file):
    """
    Main routine to compute the sweet-spot dataset.

    Parameters
    ----------
    dataset_path : str
        Path to input dataset.
    flux_calibration : float
        Absolute flux calibration value.
    output_file : str
        Name of the output FITS file for the sweet-spot dataset.
    fpam_keyword : str
        FITS header keyword for FPAM encoder position.
    """

    M = len(dataset_path)
    sweet_spot_data = np.zeros((M, 3))

    for i in range(M):
        # Open the FITS file
        print(dataset_path[i])
        image_data = dataset_path[i].data
        pri_hdr = dataset_path[i].pri_hdr
        ext_hdr = dataset_path[i].ext_hdr
  
        # Retrieve FPAM encoder position from the extension header
        fpam_h = ext_hdr.get('FPAM_H')
        fpam_v = ext_hdr.get('FPAM_V')
        print(fpam_h, fpam_v)

        # Perform processing (e.g., compute centroid, flux, etc.)
        x_center, y_center = compute_centroid(image_data)  # Function to compute the centroid
        flux = compute_flux_in_image(image_data, x_center, y_center)  # Function to compute flux
        od = compute_od(flux, flux_calibration)  # Function to compute OD

        # Store results in the sweet-spot dataset
        sweet_spot_data[i, 0] = od
        sweet_spot_data[i, 1] = x_center
        sweet_spot_data[i, 2] = y_center

    # Save results as a FITS file with FPAM encoder position
    hdu = fits.PrimaryHDU(sweet_spot_data)
    hdr = hdu.header
    #hdr[fpam_keyword] = fpam_encoder_position
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_file, overwrite=True)
    print(f"Sweet-spot dataset saved to {output_file}")
