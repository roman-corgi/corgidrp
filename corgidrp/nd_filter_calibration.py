import os
import numpy as np
from astropy.io import fits
from corgidrp.data import Dataset
from collections import defaultdict
import re
import corgidrp.fluxcal as fluxcal
from scipy import integrate

"""
From requirement 1090877.
Revised to compute a flux calibration factor from dim stars (no ND filter)
and then use that factor to compute OD for bright stars observed with the ND filter.

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

def compute_expected_flux(star_name, filter_name):
    """
    Compute the expected absolute integrated flux of a star through a given filter.

    Parameters
    ----------
    star_name : str
        Name of the star, must be known in the fluxcal.calspec_names dict.
    filter_name : str
        Filter identifier (e.g., '3C') that corresponds to a known filter curve file.

    Returns
    -------
    expected_flux : float
        The expected integrated flux (erg/(s*cm^2)) over the filter band.
    """
    # Get CALSPEC file for this star
    calspec_filepath = fluxcal.get_calspec_file(star_name)

    # Find matching filter curve
    datadir = os.path.join(os.path.dirname(fluxcal.__file__), "data", "filter_curves")
    filter_files = [f for f in os.listdir(datadir) if filter_name in f and f.endswith('.csv')]
    if not filter_files:
        raise ValueError(f"No filter curve available with name {filter_name}")
    filter_filename = os.path.join(datadir, filter_files[0])

    # Read filter curve
    wave, transmission = fluxcal.read_filter_curve(filter_filename)

    # Read CALSPEC and interpolate flux onto filter wavelengths
    calspec_flux = fluxcal.read_cal_spec(calspec_filepath, wave)

    # Calculate band irradiance (integrated flux over the band)
    # This gives erg/(s*cm^2) when integrating flux density over wavelength
    expected_flux = calculate_band_irradiance(transmission, calspec_flux, wave)

    return expected_flux

def group_by_target(dataset_entries):
    """
    Group dataset files based on the 'TARGET' value in the FITS extension header.
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

def calculate_band_irradiance(filter_curve, calspec_flux, filter_wavelength):
    """
    Calculate the integrated flux (band irradiance) over the filter band.

    This integrates calspec_flux * filter_curve over the wavelength range,
    giving total flux in erg/(s*cm²) if calspec_flux is in erg/(s*cm²*Å).

    Args:
        filter_curve (np.array): Filter transmission curve over filter_wavelength
        calspec_flux (np.array): Flux density of the CALSPEC star in erg/(s*cm²*Å)
        filter_wavelength (np.array): Wavelengths in Å

    Returns:
        float: Integrated flux (band irradiance) in erg/(s*cm²)
    """
    # Integrate over wavelength:
    # ∫ (calspec_flux(λ) * filter_curve(λ)) dλ
    irrad = integrate.simpson(calspec_flux * filter_curve, x=filter_wavelength)
    return irrad

def compute_flux_calibration_factor(dim_stars_paths):
    """
    Compute the flux calibration factor C from dim stars (no ND filter).
    C converts measured electrons per second to physical flux in erg/(s*cm^2).

    C = expected_flux / (measured_electrons_per_second)

    Parameters
    ----------
    dim_stars_paths : list of Dataset
        List of datasets for dim stars with known flux (no ND filter).

    Returns
    -------
    calibration_factor : float
        Average calibration factor derived from all dim stars.
    """
    factors = []
    for entry in dim_stars_paths:
        image_data = entry.data
        ext_hdr = entry.ext_hdr
        star_name = ext_hdr['TARGET']
        filter_name = ext_hdr['CFAMNAME']
        exptime = ext_hdr.get('EXPTIME', 1.0)  # ensure we have exposure time

        x_center, y_center = compute_centroid(image_data)
        measured_electrons = compute_flux_in_image(image_data, x_center, y_center)
        # Convert electrons to electrons per second
        measured_electrons_per_s = measured_electrons / exptime

        expected_flux = compute_expected_flux(star_name, filter_name)  # erg/(s*cm^2)

        # Compute factor: how to get from electrons/s to erg/(s*cm^2)
        C = expected_flux / measured_electrons_per_s
        factors.append(C)

    # Average the calibration factors from all dim stars
    calibration_factor = np.mean(factors)
    return calibration_factor

def main(dim_stars_paths, bright_stars_paths, output_path, threshold=0.1):
    """
    Main routine:
    1. Derive flux calibration factor from dim stars (no ND filter).
    2. Use this factor to compute OD for bright stars with ND filter.
    3. Check OD uniformity and flag if needed.
    4. Save results to FITS files.
    """

    # Step 1: Compute flux calibration factor from dim stars
    calibration_factor = compute_flux_calibration_factor(dim_stars_paths)

    # Step 2: Group bright star files by their target
    grouped_files = group_by_target(bright_stars_paths)
    flux_results = {}

    # Process each bright star group
    for target, files in grouped_files.items():
        print(f"Processing target: {target}")
        od_values = []
        x_values = []
        y_values = []
        fpam_h = fpam_v = None

        # Assume all images for this target use the same filter?
        # Perhaps need to check and handle that case.
        if not files:
            continue
        first_hdr = files[0].ext_hdr
        filter_name = first_hdr['CFAMNAME']

        # Compute the expected flux without ND filter
        expected_flux = compute_expected_flux(target, filter_name)

        for entry in files:
            image_data = entry.data
            ext_hdr = entry.ext_hdr
            fpam_h = ext_hdr.get('FPAM_H', fpam_h)
            fpam_v = ext_hdr.get('FPAM_V', fpam_v)
            exptime = ext_hdr.get('EXPTIME', 1.0)

            x_center, y_center = compute_centroid(image_data)
            if np.isnan(x_center) or np.isnan(y_center):
                print(f"Warning: Centroid could not be computed for {entry}")
                continue

            measured_electrons = compute_flux_in_image(image_data, x_center, y_center)
            measured_electrons_per_s = measured_electrons / exptime

            # Convert measured electrons/s to physical flux using calibration factor
            measured_flux_physical = measured_electrons_per_s * calibration_factor  # erg/(s*cm^2)

            # OD = ratio of the measured flux with ND to the expected flux without ND
            od = measured_flux_physical / expected_flux
            od_values.append(od)
            x_values.append(x_center)
            y_values.append(y_center)

        od_values = np.array(od_values)
        # Check OD variation within threshold
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
    # Above needs to be changed to visit_id = ext_hdr.get('VISID') when implemented in L1s
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