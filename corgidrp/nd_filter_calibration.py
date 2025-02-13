import os
import re
import math
import numpy as np
from astropy.io import fits
import corgidrp.fluxcal as fluxcal
from corgidrp.data import NDFilterSweetSpotDataset
from corgidrp.astrom import centroid
from scipy.interpolate import griddata


def compute_expected_band_irradiance(star_name, filter_name):
    """
    Compute the expected band-integrated irradiance (erg/(s*cm^2)) for a star.
    Uses the CALSPEC model for the star and a filter transmission curve.
    """
    calspec_filepath = fluxcal.get_calspec_file(star_name)
    datadir = os.path.join(os.path.dirname(fluxcal.__file__), "data", "filter_curves")
    filter_files = [f for f in os.listdir(datadir)
                    if filter_name in f and f.endswith('.csv')]
    if not filter_files:
        raise ValueError(f"No filter curve available with name {filter_name}")
    filter_filename = os.path.join(datadir, filter_files[0])
    wave, transmission = fluxcal.read_filter_curve(filter_filename)
    calspec_flux = fluxcal.read_cal_spec(calspec_filepath, wave)
    return fluxcal.calculate_band_irradiance(transmission, calspec_flux, wave)


def group_by_target(dataset):
    """
    Group dataset objects by the 'TARGET' keyword.
    """
    split_datasets, unique_vals = dataset.split_dataset(exthdr_keywords=['TARGET'])
    groups = {}
    for key, sub_ds in zip(unique_vals, split_datasets):
        target = key[0] if isinstance(key, tuple) else key
        groups[target] = sub_ds
    return groups


def compute_avg_calibration_factor(dim_stars_dataset, phot_method, flux_or_irr="irr", calibrate_kwargs=None):
    """
    Compute the average flux calibration factor using dim stars (no ND filter).

    Parameters:
        dim_stars_dataset (iterable): Dataset containing dim star entries.
        phot_method (str): Photometry method to use ("Aperture" or "PSF").
        flux_or_irr (str): Whether flux ('flux') or in-band irradiance ('irr') should be used.
        calibrate_kwargs (dict, optional): Dictionary of keyword arguments to pass to calibrate_fluxcal_aper.

    Returns:
        float: The average calibration factor.
    """
    # Compute calibration factors for each entry, using the provided (or default) keyword arguments.
    if phot_method == "Aperture":
        cal_values = [
            fluxcal.calibrate_fluxcal_aper(entry, flux_or_irr, phot_kwargs=calibrate_kwargs).fluxcal_fac
            for entry in dim_stars_dataset
        ]
    elif phot_method == "PSF":
        cal_values = [
            fluxcal.calibrate_fluxcal_gauss2d(entry, flux_or_irr, phot_kwargs=calibrate_kwargs).fluxcal_fac
            for entry in dim_stars_dataset
        ]
    else:
        raise ValueError(f"Photometry method must be either Aperture or PSF.")
    
    avg_factor = np.mean(cal_values)
    return avg_factor


def process_bright_target(target, files, cal_factor, threshold, phot_method="Aperture", phot_kwargs=None):
    """
    Process bright star files for one target to compute optical density (OD)
    and (x, y) centroids for each dithered observation.
    Checks that FPAM keywords are consistent across all files.
    
    Additional photometry options are passed via phot_kwargs.
    This allows users to override default settings for functions like aper_phot.
    
    Parameters:
        target (str): The target star name.
        files (list): List of dataset entries (each with data, headers, etc.).
        cal_factor (float): Calibration factor.
        threshold (float): Threshold for flagging OD variations.
        phot_method (str): Photometry method to use ("Aperture" or "PSF").
        phot_kwargs (dict, optional): Dictionary of keyword arguments to forward to the photometry function.
    
    Returns:
        dict: A dictionary containing computed OD values, centroids, and other metadata.
    """
    od_values = []
    x_values = []
    y_values = []
    
    # Use the header from the first file as the common reference.
    first_hdr = files[0].ext_hdr
    common_cfam_name = first_hdr['CFAMNAME']
    common_fpam_name = first_hdr.get('FPAMNAME')
    common_fpam_h = first_hdr.get('FPAM_H')
    common_fpam_v = first_hdr.get('FPAM_V')
    exptime = first_hdr.get('EXPTIME')
    
    expected_irradiance_no_nd = compute_expected_band_irradiance(target, common_cfam_name)
    expected_flux = expected_irradiance_no_nd * exptime * cal_factor
    
    # Process each file.
    for entry in files:
        hdr = entry.ext_hdr
        if (hdr.get('FPAMNAME') != common_fpam_name or 
            hdr.get('FPAM_H') != common_fpam_h or 
            hdr.get('FPAM_V') != common_fpam_v or
            hdr.get('CFAMNAME') != common_cfam_name):
            raise ValueError(f"Inconsistent FPAM header values in target {target} for file {entry}!")
        
        image_data = entry.data
        exptime = hdr['EXPTIME']
        
        # Compute centroid.
        x_center, y_center = centroid(image_data)
        if np.isnan(x_center) or np.isnan(y_center):
            print(f"Warning: Centroid could not be computed for {entry}")
            continue

        # Call the appropriate photometry function using the passed keyword arguments.
        if phot_method == "Aperture":
            phot_result = fluxcal.aper_phot(entry, **phot_kwargs)
        elif phot_method == "PSF":
            phot_result = fluxcal.phot_by_gauss2d_fit(entry, **phot_kwargs)
        else:
            raise ValueError("Must choose a valid photometry method: Aperture or PSF.")
        

        transmission = phot_result[0] / expected_flux
        od = -math.log10(transmission)
        
        od_values.append(od)
        x_values.append(x_center)
        y_values.append(y_center)
    
    od_array = np.array(od_values)
    star_flag = (np.std(od_array) >= threshold) if od_array.size > 0 else False
    average_od = np.mean(od_array) if od_array.size > 0 else np.nan

    return {
        'od_values': od_array,
        'average_od': average_od,
        'FPAMNAME': common_fpam_name,
        'FPAM_H': common_fpam_h,
        'FPAM_V': common_fpam_v,
        'CFAMNAME': common_cfam_name,
        'flag': star_flag,
        'x_values': x_values,
        'y_values': y_values
    }


def get_max_serial(output_path, visit_id):
    """
    Find the current maximum serial number in the output directory.
    """
    pattern = re.compile(rf"CGI_{visit_id}_(\d+)_FilterBand_.*_NDF_SWEETSPOT\.fits")

    max_serial = 0
    for filename in os.listdir(output_path):
        match = pattern.match(filename)
        if match:
            current_serial = int(match.group(1))
            if current_serial > max_serial:
                max_serial = current_serial
    return max_serial


def interpolate_od(sweet_spot_data, x_query, y_query, method='linear'):
    """
    sweet_spot_data: Nx3 array [OD, x, y]
    Returns the interpolated OD at (x_query, y_query).
    """
    # points -> array of shape (N, 2), i.e. all (x, y)
    points = sweet_spot_data[:, 1:3]
    values = sweet_spot_data[:, 0]  # OD
    od_interp = griddata(points, values, (x_query, y_query), method=method)
    return float(od_interp)


def create_nd_sweet_spot_dataset(
    aggregated_sweet_spot_data,
    common_metadata,
    visit_id,
    current_max,
    output_path,
    clean_frame_entry=None,             
    transformation_matrix=None,            # 2x2 matrix from FPAM->EXCAM
    transformation_matrix_file=None,       # path to that matrix file
    save_to_disk=True
):
    """
    Create a *single* ND Filter Sweet Spot product:
      - Start with the Nx3 sweet spot array from the ND dithers
      - If a clean frame is provided, compute OD at the new location (via
        the transformation matrix + nearest neighbor or interpolation).
      - Append that row [OD, x, y] to the sweet spot data.
      - Save the result as one NDFilterSweetSpotDataset.
    """
    # 1. Optionally compute and append the new row from the clean frame image
    final_sweet_spot_data = aggregated_sweet_spot_data

    if (clean_frame_entry is not None) and (transformation_matrix is not None):
        # (a) Compute centroid in clean frame
        x_clean, y_clean = centroid(clean_frame_entry.data)
        hdr = clean_frame_entry.ext_hdr
        clean_fpam_h = hdr.get('FPAM_H', 0.0)
        clean_fpam_v = hdr.get('FPAM_V', 0.0)

        # (b) Compare to sweet-spot FPAM reference
        sp_fpam_h = common_metadata['FPAM_H']
        sp_fpam_v = common_metadata['FPAM_V']
        fpam_offset = np.array([clean_fpam_h - sp_fpam_h, clean_fpam_v - sp_fpam_v])

        # (c) Apply the 2x2 transformation matrix to get EXCAM offset
        excam_offset = transformation_matrix @ fpam_offset

        # (d) Adjust the clean image centroid
        x_adj = x_clean + excam_offset[0]
        y_adj = y_clean + excam_offset[1]

        # (e) Interpolate (or nearest-neighbor) OD from the existing Nx3 data
        #     to get the predicted OD for that new location.
        new_od = interpolate_od(final_sweet_spot_data, x_adj, y_adj)

        # (f) Append that row to the Nx3 array
        new_row = np.array([[new_od, x_adj, y_adj]])
        final_sweet_spot_data = np.vstack([final_sweet_spot_data, new_row])

    # 2. Build up the NDFilterSweetSpotDataset
    pri_hdr = fits.Header()
    pri_hdr['SIMPLE'] = True
    pri_hdr['BITPIX'] = 32
    pri_hdr['OBSID'] = 0
    pri_hdr['COMMENT'] = "Combined ND Filter Sweet Spot Dataset primary header"

    ext_hdr = fits.Header()
    ext_hdr['FPAMNAME'] = common_metadata.get('FPAMNAME')
    ext_hdr['FPAM_H'] = common_metadata.get('FPAM_H')
    ext_hdr['FPAM_V'] = common_metadata.get('FPAM_V')
    ext_hdr['CFAMNAME'] = common_metadata.get('CFAMNAME')
    ext_hdr['HISTORY'] = "Combined sweet-spot dataset from bright star dithers"

    # Note transformation matrix in HISTORY
    if transformation_matrix_file:
        note = f"Clean frame row appended using transform: {os.path.basename(transformation_matrix_file)}"
        ext_hdr['HISTORY'] = note

    ndsweetspot_dataset = NDFilterSweetSpotDataset(
        data_or_filepath=final_sweet_spot_data,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        input_dataset=None  # or pass something if you have a known parent dataset
    )

    # 3. Optionally save the final product
    if save_to_disk:
        serial_number = f"{current_max + 1:03d}"
        output_filename = (
            f"CGI_{visit_id}_{serial_number}_FilterBand_{ext_hdr['CFAMNAME']}_NDF_SWEETSPOT.fits"
        )
        ndsweetspot_dataset.save(filedir=output_path, filename=output_filename)
        print(f"ND Filter Sweet Spot Dataset (combined) saved to {output_filename}")

    return ndsweetspot_dataset
