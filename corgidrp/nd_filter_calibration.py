import os
import numpy as np
from astropy.io import fits
from collections import defaultdict
import re
import corgidrp.fluxcal as fluxcal
from scipy import integrate
from corgidrp.data import NDFilterCalibration

"""
From requirement 1090877.
Revised to compute a flux calibration factor from dim stars (no ND filter)
and then use that factor to compute OD for bright stars observed with the
ND filter.

Author: Julia Milton
Date: 2024-12-09
"""


def compute_centroid(image_data):
    """Compute the centroid (x, y) of an image based on its flux distribution.

    Parameters:
        image_data : numpy.ndarray. A 2D array containing the image data from 
            which to compute the centroid.

    Returns:
        x_center, y_center : tuple of float. (x_center, y_center). If the total 
            flux is zero or negative, returns (np.nan, np.nan).
    """
    y_indices, x_indices = np.indices(image_data.shape)
    total_flux = np.sum(image_data)
    if total_flux <= 0:
        return np.nan, np.nan
    x_center = np.sum(x_indices * image_data) / total_flux
    y_center = np.sum(y_indices * image_data) / total_flux
    return x_center, y_center


def compute_flux_in_image(image_data, x_center, y_center, radius=5,
                          annulus_inner=7, annulus_outer=10):
    """Compute the flux of a source at (x_center, y_center) in `image_data` by
    summing pixel values in an aperture of `radius` and subtracting the local
    background measured in an annulus between `annulus_inner` and `annulus_outer`.

    Parameters:
        image_data : numpy.ndarray. A 2D array containing the image data 
            from which to measure flux.
        x_center : float. The x-coordinate of the star's centroid.
        y_center : float. The y-coordinate of the star's centroid.
        radius : float, optional. The aperture radius (in pixels). Default is 5.
        annulus_inner : float, optional. Inner radius of the background annulus
            (in pixels). Default is 7.
        annulus_outer : float, optional. Outer radius of the background annulus
            (in pixels). Default is 10.

    Returns:
        net_flux : float. The background-subtracted flux of the source in the 
            aperture. If the centroid is invalid (NaN) or out of range, returns np.nan.
    """
    if np.isnan(x_center) or np.isnan(y_center):
        return np.nan

    # Compute distances
    y_indices, x_indices = np.indices(image_data.shape)
    r = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)

    # Aperture and annulus masks
    aperture_mask = (r <= radius)
    annulus_mask = (r >= annulus_inner) & (r <= annulus_outer)

    # Background estimation
    if not np.any(annulus_mask):
        print("Warning: No valid pixels in background annulus. "
              "Skipping background subtraction.")
        background_level = 0.0
    else:
        background_level = np.median(image_data[annulus_mask])

    # Aperture sum
    aperture_sum = np.sum(image_data[aperture_mask])

    # Background subtraction
    background_total = background_level * np.count_nonzero(aperture_mask)
    net_flux = aperture_sum - background_total

    return net_flux


def compute_expected_flux(star_name, filter_name):
    """ Compute the expected absolute integrated flux of a star through a given filter.

    Parameters:
        star_name : str. Name of the star. Must be recognized by `fluxcal.get_calspec_file`.
        filter_name : str. Filter identifier (e.g., '3C') that corresponds to a 
            known filter curve file.

    Returns:
        expected_flux : float. The expected integrated flux (erg / (s * cm^2))
            over the filter band.
    """
    # Get the CALSPEC reference file for this star
    calspec_filepath = fluxcal.get_calspec_file(star_name)

    # Find matching filter curve
    datadir = os.path.join(os.path.dirname(fluxcal.__file__), "data",
                           "filter_curves")
    filter_files = [f for f in os.listdir(datadir)
                    if filter_name in f and f.endswith('.csv')]
    if not filter_files:
        raise ValueError(f"No filter curve available with name {filter_name}")
    filter_filename = os.path.join(datadir, filter_files[0])

    # Read filter curve
    wave, transmission = fluxcal.read_filter_curve(filter_filename)

    # Read and interpolate CALSPEC flux
    calspec_flux = fluxcal.read_cal_spec(calspec_filepath, wave)

    # Calculate integrated flux
    expected_flux = calculate_band_irradiance(transmission, calspec_flux, wave)
    return expected_flux


def group_by_target(dataset_entries):
    """
    Group dataset objects by the 'TARGET' keyword in their FITS extension headers.

    Parameters:
        dataset_entries : list. A list of dataset objects, each containing an
            'ext_hdr' attribute with FITS header information.

    Returns:
        grouped_files : dict. A dictionary where each key is a target name, and
            each value is a list of dataset entries that match that target.
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
    """ Calculate the integrated flux (band irradiance) over the filter band.
        This performs the numerical integration of:
        ∫(calspec_flux(λ) * filter_curve(λ)) dλ
        over the wavelength range provided in `filter_wavelength`.

    Parameters:
        filter_curve : numpy.ndarray. Filter transmission curve values.
        calspec_flux : numpy.ndarray. Flux density of the CALSPEC star
            (erg / (s * cm^2 * Å)).
        filter_wavelength : numpy.ndarray. Wavelengths corresponding to the
            filter curve in Å.

    Returns:
        irrad : float. Integrated flux (band irradiance) in erg / (s * cm^2).
    """
    irrad = integrate.simpson(calspec_flux * filter_curve, x=filter_wavelength)
    return irrad


def compute_flux_calibration_factor(dim_stars_paths):
    """Compute the flux calibration factor (C) from dim stars observed without an ND filter. 
        The factor C converts measured electrons per second into physical flux units
        (erg / (s * cm^2)) using the relation:
        C = expected_flux / (measured_electrons_per_second)

    Parameters:
        dim_stars_paths : list. List of dataset objects for dim stars with known
            flux (no ND filter).

    Returns:
        calibration_factor : float. The average calibration factor derived from all dim
            star observations.
    """
    factors = []
    for entry in dim_stars_paths:
        image_data = entry.data
        ext_hdr = entry.ext_hdr
        star_name = ext_hdr['TARGET']
        filter_name = ext_hdr['CFAMNAME']
        exptime = ext_hdr.get('EXPTIME', 1.0)

        # Measure flux in electrons
        x_center, y_center = compute_centroid(image_data)
        measured_electrons = compute_flux_in_image(image_data, x_center,
                                                   y_center)

        measured_electrons_per_s = measured_electrons / exptime
        expected_flux = compute_expected_flux(star_name, filter_name)

        # Flux calibration factor
        C = expected_flux / measured_electrons_per_s
        factors.append(C)

    calibration_factor = np.mean(factors)
    return calibration_factor


def main(dim_stars_paths, bright_stars_paths, output_path, threshold=0.1):
    """ Derive flux calibration factors from dim stars, then compute ND filter calibration.

    Steps
    -----
    1. Compute flux calibration factor from dim stars (no ND filter).
    2. Group bright star files by target.
    3. Compute optical density (OD) for bright stars using the calibration factor.
    4. Check OD uniformity and set a flag if standard deviation >= threshold.
    5. Save ND filter calibration results to FITS files.

    Parameters:
        dim_stars_paths : list
            List of dataset objects for dim stars with known flux.
        bright_stars_paths : list
            List of dataset objects for bright stars observed with the ND filter.
        output_path : str
            Directory path where the output FITS files will be saved.
        threshold : float, optional
            Standard deviation threshold for OD uniformity checks. Default is 0.1.

    """
    # Step 1: Flux calibration factor
    cal_factor = compute_flux_calibration_factor(dim_stars_paths)

    # Step 2: Group bright star files
    grouped_files = group_by_target(bright_stars_paths)
    flux_results = {}

    # Process each bright star group
    for target, files in grouped_files.items():
        print(f"Processing target: {target}")
        od_values = []
        x_values = []
        y_values = []
        fpam_h = fpam_v = None

        if not files:
            continue

        first_hdr = files[0].ext_hdr
        filter_name = first_hdr['CFAMNAME']

        # Compute expected flux without ND
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

            measured_electrons = compute_flux_in_image(
                image_data, x_center, y_center
            )
            measured_electrons_per_s = measured_electrons / exptime

            # Convert to physical flux using cal_factor
            measured_flux_physical = measured_electrons_per_s * cal_factor

            # OD = (flux with ND) / (expected flux without ND)
            od = measured_flux_physical / expected_flux
            od_values.append(od)
            x_values.append(x_center)
            y_values.append(y_center)

        od_values = np.array(od_values)
        star_flag = np.std(od_values) >= threshold
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
    pattern = re.compile(rf"CGI_{visit_id}_(\d+)_NDF_CAL\.fits")

    # Find current maximum serial number
    max_serial = 0
    for filename in os.listdir(output_path):
        match = pattern.match(filename)
        if match:
            current_serial = int(match.group(1))
            if current_serial > max_serial:
                max_serial = current_serial

    # Save calibration products
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
            input_dataset=bright_stars_paths
        )

        serial_number = f"{max_serial + 1:03d}"
        output_filename = f"CGI_{visit_id}_{serial_number}_NDF_CAL.fits"
        ndcal_product.save(filedir=output_path, filename=output_filename)

        print(f"ND Filter Calibration product saved to {output_filename}")
        max_serial += 1
