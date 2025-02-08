import os
import numpy as np
from astropy.io import fits
from collections import defaultdict
import re
import corgidrp.fluxcal as fluxcal
from scipy import integrate
from corgidrp.data import NDFilterCalibration
from corgidrp.astrom import centroid

import math

"""
From requirement 1090876.
Revised to compute a flux calibration factor from dim stars (no ND filter)
and then use that factor to compute OD for bright stars observed with the
ND filter.

Author: Julia Milton
Date: 2024-12-09
"""

# CHANGED: renamed from compute_expected_irradiance -> compute_expected_band_irradiance
def compute_expected_band_irradiance(star_name, filter_name):
    """Compute the expected band-integrated irradiance (erg / (s * cm^2)) for a star.

    Uses a CALSPEC model for the star's spectral flux density and a known
    filter transmission curve to integrate over the band. The result
    is the total irradiance over the filter.

    Args:
        star_name (str): Name of the star, recognized by `fluxcal.get_calspec_file`.
        filter_name (str): Filter identifier (e.g., '3C') that corresponds to a
                           known filter curve file.

    Returns:
        float: The expected integrated irradiance (erg / (s * cm^2)) over the filter band.
    """
    # Get the CALSPEC reference file for this star
    calspec_filepath = fluxcal.get_calspec_file(star_name)

    # Find matching filter curve
    datadir = os.path.join(os.path.dirname(fluxcal.__file__), "data", "filter_curves")
    filter_files = [f for f in os.listdir(datadir)
                    if filter_name in f and f.endswith('.csv')]
    if not filter_files:
        raise ValueError(f"No filter curve available with name {filter_name}")
    filter_filename = os.path.join(datadir, filter_files[0])

    # Read filter curve
    wave, transmission = fluxcal.read_filter_curve(filter_filename)

    # Read and interpolate CALSPEC flux
    calspec_flux = fluxcal.read_cal_spec(calspec_filepath, wave)

    # Calculate integrated irradiance
    expected_irradiance = fluxcal.calculate_band_irradiance(transmission, calspec_flux, wave)
    return expected_irradiance

def group_by_target(dataset_entry):
    """
    Group dataset objects by the 'TARGET' keyword in their FITS extension headers
    using the Dataset.split_dataset method.

    Args:
        dataset_entries (list): A list of frames (or a list of dataset objects)
            that have an 'ext_hdr' attribute containing the TARGET keyword.

    Returns:
        dict: A dictionary where each key is a target name and each value is a 
              Dataset containing all frames with that target.
    """

    # Split the dataset using the 'TARGET' keyword from the extension headers.
    # Note: split_dataset returns a tuple: (list_of_sub_datasets, list_of_unique_values)
    split_datasets, unique_vals = dataset_entry.split_dataset(exthdr_keywords=['TARGET'])
    
    groups = {}
    for key, sub_ds in zip(unique_vals, split_datasets):
        # If key is a tuple (as might happen when grouping by multiple keywords),
        # extract the first element.
        target = key[0] if isinstance(key, tuple) else key
        groups[target] = sub_ds
    return groups


def compute_avg_calibration_factor(dim_stars_paths):
    """Compute the avg flux calibration factor C using dim stars (no ND filter).

    The function uses the calibrate_fluxcal_aper function from fluxcal.py to 
    calculate the calibration factor (band flux values divided by the found 
    photoelectrons) for each image. The average calibration factor is then
    computed for all dim stars which are observed with the same filter.

    Args:
        dim_stars_paths (list): Dataset objects for dim stars with known flux
                                (no ND filter).

    Returns:
        float: The average calibration factor derived from all dim star observations.
    """
    # Compute the calibration factors for each dim star and extract the fluxcal_fac attribute.
    cal_values = [
        fluxcal.calibrate_fluxcal_aper(entry, 5, frac_enc_energy=1., method='subpixel',
                                    subpixels=5, background_sub=True, r_in=5,
                                    r_out=10, centering_method='xy').fluxcal_fac
        for entry in dim_stars_paths
    ]

    # Compute the mean calibration factor.
    avg_calibration_factor = np.mean(cal_values)
    print("Calibration factor:", avg_calibration_factor)

    return avg_calibration_factor


def main(dim_stars_dataset, bright_stars_dataset, output_path=None, file_save=False, threshold=0.1):
    """Derive flux calibration factors from dim stars, then compute ND filter calibration.

    Steps
    -----
    1. Compute flux calibration factor from dim stars (no ND filter).
    2. Group bright star files by target.
    3. Compute optical density (OD) for bright stars using the calibration factor.
    4. Check OD uniformity (standard deviation >= threshold).
    5. Save ND filter calibration results to FITS files.

    Args:
        dim_stars_dataset (list): Dataset object for dim stars with known flux.
        bright_stars_dataset (list): Dataset object for bright stars observed
                                   with the ND filter.
        output_path (str): Path where output FITS files will be saved.
        threshold (float, optional): Standard deviation threshold for OD
                                     uniformity checks. Default is 0.1.
    """
    # Step 1: Compute flux calibration factor
    cal_factor = compute_avg_calibration_factor(dim_stars_dataset)

    # Step 2: Group bright star files
    grouped_files = group_by_target(bright_stars_dataset)
    flux_results = {}

    # Process each group of bright stars
    for target, files in grouped_files.items():
        print(f"Processing target: {target}")
        od_values = []
        x_values = []
        y_values = []
        fpam_h = None
        fpam_v = None

        if not files:
            continue

        first_hdr = files[0].ext_hdr
        filter_name = first_hdr['CFAMNAME']

        # CHANGED: Use the new function name for expected band-irradiance
        expected_irradiance_no_nd = compute_expected_band_irradiance(target, filter_name)

        for entry in files:
            image_data = entry.data
            ext_hdr = entry.ext_hdr
            star_name = ext_hdr['TARGET']
            fpam_h = ext_hdr['FPAM_H']
            fpam_v = ext_hdr['FPAM_V']
            exptime = ext_hdr['EXPTIME']
            filter_used = ext_hdr['CFAMNAME']
            fsm_x = ext_hdr['FSM_X']
            fsm_y = ext_hdr['FSM_Y']

            x_center, y_center = centroid(image_data)
            if np.isnan(x_center) or np.isnan(y_center):
                print(f"Warning: Centroid could not be computed for {entry}")
                continue

            # Measured flux in electrons
            aper_sum = fluxcal.aper_phot(entry, 5, frac_enc_energy=1., 
                                 method='subpixel', subpixels=5, background_sub = True, 
                                 r_in = 5 , r_out = 10, centering_method='xy')
            
            measured_electrons = aper_sum[0]
            measured_electrons_per_s = measured_electrons / exptime

            # Convert to physical irradiance
            measured_flux_physical = measured_electrons_per_s * cal_factor

            # Transmission = (flux with ND) / (flux without ND)
            transmission = measured_flux_physical / expected_irradiance_no_nd

            # OD = -log10(Transmission)
            od = -math.log10(transmission)
            od_values.append(od)
            x_values.append(x_center)
            y_values.append(y_center)

        od_values = np.array(od_values)
        star_flag = (np.std(od_values) >= threshold)
        average_od = np.mean(od_values)

        flux_results[target] = {
            'od_values': od_values,
            'average_od': average_od,
            'fpam_h': fpam_h,
            'fpam_v': fpam_v,
            'flag': star_flag,
            'x_values': x_values,
            'y_values': y_values
        }

    # Step 3: Save results
    visit_id = 'PPPPPCCAAASSSOOOVVV'  # Placeholder
    pattern = re.compile(rf"CGI_{visit_id}_(\d+)_FilterBand_.*_NDF_CAL\.fits")

    # Find current maximum serial number
    max_serial = 0
    for filename in os.listdir(output_path):
        match = pattern.match(filename)
        if match:
            current_serial = int(match.group(1))
            if current_serial > max_serial:
                max_serial = current_serial

    # Step 4: Save calibration products
    for star_name, star_data in flux_results.items():
        od_values = star_data['od_values']
        x_values = star_data['x_values']
        y_values = star_data['y_values']

        # Nx3 data array: OD, x_center, y_center
        sweet_spot_data = np.zeros((len(od_values), 3))
        sweet_spot_data[:, 0] = od_values
        sweet_spot_data[:, 1] = x_values
        sweet_spot_data[:, 2] = y_values

        # Primary header
        pri_hdr = fits.Header()
        pri_hdr['SIMPLE'] = True
        pri_hdr['BITPIX'] = 32
        pri_hdr['OBSID'] = 0000
        pri_hdr['COMMENT'] = "NDFilterCalibration primary header"

        # Extension header
        ext_hdr = fits.Header()
        ext_hdr['FPAM_H'] = star_data['fpam_h']
        ext_hdr['FPAM_V'] = star_data['fpam_v']
        ext_hdr['HISTORY'] = f"NDFilterCalibration for {star_name}"

        ndcal_product = NDFilterCalibration(
            data_or_filepath=sweet_spot_data,
            pri_hdr=pri_hdr,
            ext_hdr=ext_hdr,
            input_dataset=bright_stars_dataset
        )

        serial_number = f"{max_serial + 1:03d}"
        # Use the same filter name from the bright star header if needed:
        output_filename = f"CGI_{visit_id}_{serial_number}_FilterBand_{filter_name}_NDF_CAL.fits"

        if output_path is not None and file_save:
            ndcal_product.save(filedir=output_path, filename=output_filename)
            print(f"ND Filter Calibration product saved to {output_filename}")
            max_serial += 1
