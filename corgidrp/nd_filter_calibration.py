import os
import numpy as np
from astropy.io import fits
from corgidrp.data import Dataset
from collections import defaultdict
import re

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

def compute_centroid(image_data):
    y_indices, x_indices = np.indices(image_data.shape)
    total_flux = np.sum(image_data)
    if total_flux <= 0:
        return np.nan, np.nan
    x_center = np.sum(x_indices * image_data) / total_flux
    y_center = np.sum(y_indices * image_data) / total_flux
    return x_center, y_center

def compute_flux_in_image(image_data, x_center, y_center, radius=5):
    y_indices, x_indices = np.indices(image_data.shape)
    r = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
    aperture_mask = (r <= radius)
    flux = np.sum(image_data[aperture_mask])
    return flux  # Background subtraction could be added here if needed.

def compute_expected_flux_synphot(star_name):
    """
    Placeholder function: Compute the expected flux using synphot.
    This would depend on star's known SED, etc.
    Currently returns a dummy value.
    """
    # Integrate synphot as needed.
    return 1000.0  # placeholder

def group_by_target(dataset_entries):
    """
    Group dataset files based on the 'TARGET' value in the FITS extension header.

    Parameters
    ----------
    dataset_entries : list
        List of Dataset objects, each containing pri_hdr, ext_hdr, and data attributes.

    Returns
    -------
    grouped_files : dict
        Dictionary where keys are unique TARGET values, and values are lists of dataset entries.
    """
    grouped_files = defaultdict(list)

    for entry in dataset_entries:
        try:
            target_value = entry.ext_hdr.get('TARGET', None)
            if target_value is not None:
                grouped_files[target_value].append(entry)
            else:
                print(f"Warning: 'TARGET' not found in {entry}")
        except Exception as e:
            print(f"Error processing {entry}: {e}")

    return grouped_files

def main(dim_stars_paths, bright_stars_paths, output_path, threshold=0.1):
    """
    Main routine:
    1. Compute transmission efficiency from 10 dim stars (no ND filter).
    2. Compute OD for bright stars with ND filter in a 3x3 raster.
    3. Check OD uniformity and flag if needed.
    4. Save results to FITS files.

    Parameters
    ----------
    dim_stars_paths : list of Dataset entries
        Paths (and loaded data) for the 10 dim star datasets.
    bright_stars_paths : list of Dataset entries
        Paths (and loaded data) for bright star raster datasets.
    output_path : str
        Directory where the output files will be saved.
    threshold : float
        Threshold for checking OD uniformity.
    """

    # Step 1: Compute average dim star transmission efficiency
    ratios = []
    for entry in dim_stars_paths:
        image_data = entry.data
        ext_hdr = entry.ext_hdr
        star_name = ext_hdr['TARGET']

        x_center, y_center = compute_centroid(image_data)
        measured_flux = compute_flux_in_image(image_data, x_center, y_center)
        expected_flux = compute_expected_flux_synphot(star_name)
        ratio = measured_flux / expected_flux
        ratios.append(ratio)

    avg_optical_efficiency = np.mean(ratios)

    # Step 2: Group bright star files by their target
    grouped_files = group_by_target(bright_stars_paths)
    print("Grouped bright star files:", grouped_files)

    flux_results = {}

    # Process each group (star)
    for target, files in grouped_files.items():
        print(f"Processing target: {target}")
        od_values = []
        x_values = []
        y_values = []
        fpam_h = fpam_v = None

        for entry in files:
            image_data = entry.data
            ext_hdr = entry.ext_hdr
            fpam_h = ext_hdr.get('FPAM_H', fpam_h)
            fpam_v = ext_hdr.get('FPAM_V', fpam_v)

            x_center, y_center = compute_centroid(image_data)
            if np.isnan(x_center) or np.isnan(y_center):
                print(f"Warning: Centroid could not be computed for {entry}")
                continue

            measured_flux = compute_flux_in_image(image_data, x_center, y_center)
            expected_flux = compute_expected_flux_synphot(target)
            od = measured_flux / (expected_flux * avg_optical_efficiency)
            od_values.append(od)
            x_values.append(x_center)
            y_values.append(y_center)

        od_values = np.array(od_values)
        # Check if OD variation within threshold
        star_flag = np.std(od_values) >= threshold
        average_od = np.mean(od_values)

        star_result = {
            'od_values': od_values,
            'average_od': average_od,
            'fpam_h': fpam_h,
            'fpam_v': fpam_v,
            'flag': star_flag,
            'x_values': x_values,
            'y_values': y_values
        }
        flux_results[target] = star_result

    # Step 3: Save the calibration products for each bright star
    visit_id = 'PPPPPCCAAASSSOOOVVV'
    pattern = re.compile(rf"CGI_{visit_id}_(\d+)_NDF_CAL\.fits")

    # Determine max serial number from existing files in output_path
    max_serial = 0
    for filename in os.listdir(output_path):
        match = pattern.match(filename)
        if match:
            current_serial = int(match.group(1))
            if current_serial > max_serial:
                max_serial = current_serial

    for star_name, star_data in flux_results.items():
        od_values = star_data['od_values']
        x_values = star_data['x_values']
        y_values = star_data['y_values']
        fpam_h = star_data['fpam_h']
        fpam_v = star_data['fpam_v']

        p = len(od_values)
        sweet_spot_data = np.zeros((p, 3))
        sweet_spot_data[:, 0] = od_values
        sweet_spot_data[:, 1] = x_values
        sweet_spot_data[:, 2] = y_values

        hdu = fits.PrimaryHDU(sweet_spot_data)
        hdr = hdu.header
        hdr['FPAM_H'] = fpam_h
        hdr['FPAM_V'] = fpam_v

        serial_number = f"{max_serial + 1:03d}"
        output_filename = f"CGI_{visit_id}_{serial_number}_NDF_CAL.fits"
        output_file = os.path.join(output_path, output_filename)
        fits.HDUList([hdu]).writeto(output_file, overwrite=True)

        print(f"Sweet-spot dataset saved to {output_file}")
        max_serial += 1
