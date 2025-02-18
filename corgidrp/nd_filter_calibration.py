import os
import re
import math
import numpy as np
from astropy.io import fits
import corgidrp.fluxcal as fluxcal
import corgidrp.astrom as astrom
from corgidrp.data import NDFilterSweetSpotDataset
from corgidrp.astrom import centroid_with_roi
from scipy.interpolate import griddata

# =============================================================================
# Helper Functions
# =============================================================================

def load_transformation_matrix_from_fits(file_path):
    """
    Load a transformation matrix from a FITS file.
    
    Parameters:
        file_path (str): Path to the FITS file containing the transformation matrix.
    
    Returns:
        np.ndarray: The transformation matrix extracted from the FITS file.
    """
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        if data is None:
            data = hdul[1].data
        return np.array(data)


def get_max_serial(output_path):
    """
    Find the current maximum serial number in the output directory based on filenames.

    Parameters:
        output_path (str): The path to the directory containing the FITS files.

    Returns:
        int: The highest serial number found in the filenames. Returns 0 if no matching 
            files are found.
    """
    pattern = re.compile(r"CGI_[A-Za-z0-9]+_(\d+)_FilterBand_.*_NDF_SWEETSPOT\.fits")
    max_serial = 0
    for filename in os.listdir(output_path):
        match = pattern.match(filename)
        if match:
            current_serial = int(match.group(1))
            if current_serial > max_serial:
                max_serial = current_serial
    return max_serial


def group_by_target(dataset):
    """
    Split the dataset by the 'TARGET' keyword and return a dictionary {target: subset}.

    Parameters:
        dataset (Dataset): The dataset to be split based on the 'TARGET' keyword.

    Returns:
        dict: A dictionary where keys are unique target values and values are the 
            corresponding dataset subsets.
    """
    split_datasets, unique_vals = dataset.split_dataset(exthdr_keywords=['TARGET'])
    groups = {}
    for key, sub_ds in zip(unique_vals, split_datasets):
        target = key[0] if isinstance(key, tuple) else key
        groups[target] = sub_ds
    return groups


def interpolate_od(sweet_spot_data, x_query, y_query, method='linear'):
    """
    Interpolate OD from sweet_spot_data at (x_query, y_query).

    Parameters:
        sweet_spot_data (numpy.ndarray): An Nx3 array where each row contains [OD, x, y].
        x_query (float or numpy.ndarray): The x-coordinate(s) at which to interpolate.
        y_query (float or numpy.ndarray): The y-coordinate(s) at which to interpolate.
        method (str): The interpolation method to use ('linear', 'nearest', or 'cubic'). 
            Defaults to 'linear'.

    Returns:
        float or numpy.ndarray: The interpolated OD value(s). Returns a float if a single 
            query point is provided, otherwise returns an array of interpolated values.
    """

    # Points and OD values
    points = sweet_spot_data[:, 1:3]  # shape (N,2) -> (x, y)
    values = sweet_spot_data[:, 0]    # shape (N,) -> OD

    # Prepare interpolation coordinates
    if np.isscalar(x_query) and np.isscalar(y_query):
        xi = (x_query, y_query)
    else:
        xi = np.column_stack((x_query, y_query))

    od_interp = griddata(points, values, xi, method=method)
    return float(od_interp) if np.isscalar(x_query) else od_interp


# =============================================================================
# Flux Calibration Helpers
# =============================================================================

def compute_expected_band_irradiance(star_name, filter_name):
    """
    Compute the expected band-integrated irradiance (erg/(s*cm^2)) for a given star.

    Parameters:
        star_name (str): The name of the star for which to compute the irradiance.
        filter_name (str): The name of the filter used to determine the transmission curve.

    Returns:
        float: The computed band-integrated irradiance in erg/(s*cm^2).
    
    Raises:
        ValueError: If no matching filter curve file is found.
    """
    calspec_filepath = fluxcal.get_calspec_file(star_name)
    datadir = os.path.join(os.path.dirname(fluxcal.__file__), "data", "filter_curves")
    filter_files = [f for f in os.listdir(datadir) if filter_name in f and f.endswith('.csv')]
    if not filter_files:
        raise ValueError(f"No filter curve available with name {filter_name}")
    
    filter_filename = os.path.join(datadir, filter_files[0])
    wave, transmission = fluxcal.read_filter_curve(filter_filename)
    calspec_flux = fluxcal.read_cal_spec(calspec_filepath, wave)
    return fluxcal.calculate_band_irradiance(transmission, calspec_flux, wave)


def compute_avg_calibration_factor(dim_stars_dataset, phot_method, flux_or_irr="irr", phot_kwargs=None):
    """
    Compute the average flux calibration factor using dim stars (no ND filter).

    Parameters:
        dim_stars_dataset (iterable): Dataset containing dim star entries.
        phot_method (str): Photometry method to use ("Aperture" or "PSF").
        flux_or_irr (str): Whether flux ('flux') or in-band irradiance ('irr') should be used.
        phot_kwargs (dict, optional): Dictionary of keyword arguments to pass to calibrate_fluxcal_aper.

    Returns:
        float: The average calibration factor.
    """
    if phot_kwargs is None:
        phot_kwargs = {}

    if phot_method == "Aperture":
        cal_values = [
            fluxcal.calibrate_fluxcal_aper(entry, flux_or_irr, phot_kwargs).fluxcal_fac
            for entry in dim_stars_dataset
        ]
    elif phot_method == "PSF":
        cal_values = [
            fluxcal.calibrate_fluxcal_gauss2d(entry, flux_or_irr, phot_kwargs).fluxcal_fac
            for entry in dim_stars_dataset
        ]
    else:
        raise ValueError("Photometry method must be either Aperture or PSF.")

    return np.mean(cal_values)


# =============================================================================
# OD & Photometry Computations
# =============================================================================

def _compute_od_for_file(entry, target, phot_method, phot_kwargs, ref_fpam_name,
                         ref_fpam_h, ref_fpam_v, ref_cfam_name, expected_flux):
    """
    Helper subfunction to:
      1. Validate FPAM/CFAM metadata vs. the reference.
      2. Compute centroid (x, y).
      3. Perform photometry (Aperture or PSF).
      4. Compute OD from measured flux and the expected flux.
    
    Parameters:
        entry (object): The dataset entry containing image data and metadata.
        target (str): The target identifier for the dataset entry.
        phot_method (str): The photometry method to use ('Aperture' or 'PSF').
        phot_kwargs (dict): Additional keyword arguments for the photometry method.
        ref_fpam_name (str): The reference FPAM name for validation.
        ref_fpam_h (float): The reference FPAM horizontal position.
        ref_fpam_v (float): The reference FPAM vertical position.
        ref_cfam_name (str): The reference CFAM name for validation.
        expected_flux (float): The expected flux value for computing OD.

    Returns:
        tuple:
            float: The computed optical depth (OD).
            float: The x-coordinate of the centroid.
            float: The y-coordinate of the centroid.

    Raises:
        ValueError: If FPAM/CFAM metadata do not match the reference values.
        ValueError: If an invalid photometry method is specified.
    """
    hdr = entry.ext_hdr

    # Metadata checks
    if (hdr.get('FPAMNAME') != ref_fpam_name or 
        hdr.get('FPAM_H') != ref_fpam_h or 
        hdr.get('FPAM_V') != ref_fpam_v or
        hdr.get('CFAMNAME') != ref_cfam_name):
        raise ValueError(
            f"Inconsistent FPAM/CFAM header values in target {target} for file {entry}!"
        )

    # Centroid
    x_center, y_center = centroid_with_roi(entry.data)
    if np.isnan(x_center) or np.isnan(y_center):
        print(f"Warning: Centroid could not be computed for {entry}")
        return None, None, None

    # Photometry
    if phot_method == "Aperture":
        phot_result = fluxcal.aper_phot(entry, **phot_kwargs)
    elif phot_method == "PSF":
        phot_result = fluxcal.phot_by_gauss2d_fit(entry, **phot_kwargs)
    else:
        raise ValueError("phot_method must be Aperture or PSF.")

    # Compute OD
    transmission = phot_result[0] / expected_flux
    od = -math.log10(transmission)
    return od, x_center, y_center


def process_bright_target(target, files, cal_factor, od_raster_threshold,
                          phot_method="Aperture", phot_kwargs=None):
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
        od_raster_threshold (float): Threshold for flagging OD variations.
        phot_method (str): Photometry method to use ("Aperture" or "PSF").
        phot_kwargs (dict, optional): Dictionary of keyword arguments to forward to the photometry function.
    
    Returns:
        dict: A dictionary containing computed OD values, centroids, and other metadata.
    """
    if phot_kwargs is None:
        phot_kwargs = {}

    first_hdr = files[0].ext_hdr
    ref_cfam_name = first_hdr['CFAMNAME']
    common_fpam_name = first_hdr.get('FPAMNAME')
    common_fpam_h    = first_hdr.get('FPAM_H')
    common_fpam_v    = first_hdr.get('FPAM_V')
    exptime          = first_hdr.get('EXPTIME')

    # Compute expected flux
    expected_irradiance_no_nd = compute_expected_band_irradiance(target, ref_cfam_name)
    expected_flux = expected_irradiance_no_nd / cal_factor

    od_values, x_values, y_values = [], [], []

    for entry in files:
        od, x_center, y_center = _compute_od_for_file(entry, target, phot_method, 
                                                      phot_kwargs, common_fpam_name, 
                                                      common_fpam_h, common_fpam_v, 
                                                      ref_cfam_name, expected_flux)
        
        # Skip if centroid was not valid
        if od is None:
            continue

        od_values.append(od)
        x_values.append(x_center)
        y_values.append(y_center)

    od_array = np.array(od_values)
    star_flag = (np.std(od_array) >= od_raster_threshold) if od_array.size > 0 else False
    average_od = np.mean(od_array) if od_array.size > 0 else np.nan

    return {
        'od_values': od_array,
        'average_od': average_od,
        'FPAMNAME': common_fpam_name,
        'FPAM_H': common_fpam_h,
        'FPAM_V': common_fpam_v,
        'CFAMNAME': ref_cfam_name,
        'flag': star_flag,
        'x_values': x_values,
        'y_values': y_values
    }


def create_nd_sweet_spot_dataset(
    aggregated_sweet_spot_data,
    common_metadata,
    current_max,
    output_path,
    save_to_disk=True,
    clean_frame_entry=None,             
    transformation_matrix=None,
    transformation_matrix_file=None
):
    """
    Create an NDFilterSweetSpotDataset FITS file with the Nx3 sweet-spot array.
    Optionally append a row from a clean frame if both clean_frame_entry and
    transformation_matrix are provided.
    
    Parameters:
        aggregated_sweet_spot_data (numpy.ndarray): The aggregated Nx3 array containing 
            sweet-spot data in the format [OD, x, y].
        common_metadata (dict): A dictionary containing metadata such as FPAM/CFAM names 
            and offsets.
        current_max (int): The current maximum serial number for file naming.
        output_path (str): The directory where the FITS file will be saved.
        save_to_disk (bool): Whether to save the dataset to disk. Defaults to True.
        clean_frame_entry (object, optional): An entry representing a clean frame for 
            appending an additional OD row.
        transformation_matrix (numpy.ndarray, optional): A 2x2 matrix for transforming 
            FPAM offsets to EXCAM offsets.
        transformation_matrix_file (str, optional): The file path of the transformation 
            matrix used, for logging purposes.

    Returns:
        tuple:
            NDFilterSweetSpotDataset: The generated ND filter sweet spot dataset.
            float or None: The appended OD value if a clean frame entry was provided, 
                otherwise None.
    """
    final_sweet_spot_data = aggregated_sweet_spot_data.copy()
    appended_od = None

    # Optionally append OD from clean frame
    if (clean_frame_entry is not None) and (transformation_matrix is not None):
        x_clean, y_clean = centroid_with_roi(clean_frame_entry.data)
        hdr = clean_frame_entry.ext_hdr

        # Compute FPAM offset
        clean_fpam_h = hdr.get('FPAM_H', 0.0)
        clean_fpam_v = hdr.get('FPAM_V', 0.0)
        sp_fpam_h    = common_metadata['FPAM_H']
        sp_fpam_v    = common_metadata['FPAM_V']
        fpam_offset  = np.array([clean_fpam_h - sp_fpam_h, clean_fpam_v - sp_fpam_v])

        # Transform to EXCAM offset
        excam_offset = transformation_matrix @ fpam_offset
        x_adj = x_clean + excam_offset[0]
        y_adj = y_clean + excam_offset[1]

        # Interpolate OD at that new location
        appended_od = interpolate_od(final_sweet_spot_data, x_adj, y_adj)
        new_row     = np.array([[appended_od, x_adj, y_adj]])
        final_sweet_spot_data = np.vstack([final_sweet_spot_data, new_row])

    # Build the NDFilterSweetSpotDataset
    pri_hdr = fits.Header()
    pri_hdr['SIMPLE']  = True
    pri_hdr['BITPIX']  = 32
    pri_hdr['OBSNUM']   = 000
    pri_hdr['COMMENT'] = "Combined ND Filter Sweet Spot Dataset primary header"

    ext_hdr = fits.Header()
    ext_hdr['FPAMNAME'] = common_metadata.get('FPAMNAME')
    ext_hdr['FPAM_H']   = common_metadata.get('FPAM_H')
    ext_hdr['FPAM_V']   = common_metadata.get('FPAM_V')
    ext_hdr['CFAMNAME'] = common_metadata.get('CFAMNAME')
    ext_hdr['HISTORY']  = "Combined sweet-spot dataset from bright star dithers"

    if transformation_matrix_file and appended_od is not None:
        note = f"Appended clean frame row using transform: {os.path.basename(transformation_matrix_file)}"
        ext_hdr['HISTORY'] = note
        visit_id = ext_hdr.get('VISITID', "CombinedVisit")
    else:
        visit_id = ext_hdr.get('VISITID', "CombinedVisit")

    ndsweetspot_dataset = NDFilterSweetSpotDataset(
        data_or_filepath=final_sweet_spot_data,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        input_dataset=None
    )

    if save_to_disk:
        serial_number = f"{current_max + 1:03d}"
        output_filename = (
            f"CGI_{visit_id}_{serial_number}_FilterBand_{ext_hdr['CFAMNAME']}_NDF_SWEETSPOT.fits"
        )
        ndsweetspot_dataset.save(filedir=output_path, filename=output_filename)
        print(f"ND Filter Sweet Spot Dataset saved to {output_filename}")

    return ndsweetspot_dataset, appended_od


# =============================================================================
# Main Workflow Function
# =============================================================================

def create_nd_filter_cal(dim_stars_dataset,
                         bright_stars_dataset,
                         output_path,
                         file_save,
                         od_raster_threshold,
                         clean_entry=None,
                         transformation_matrix_file=None,
                         phot_method="Aperture",
                         flux_or_irr="irr",
                         phot_kwargs=None):
    """
    Main ND Filter calibration workflow:
      1. Compute avg calibration factor from dim stars.
      2. Group bright star frames by target + measure OD, centroids.
      3. Combine all sweet-spot data into a single Nx3 array.
      4. Optionally append OD from a clean frame (if provided).
      5. Save final NDFilterSweetSpotDataset (if file_save is True).
    
    Parameters:
        dim_stars_dataset (Dataset): Dataset containing dim star images.
        bright_stars_dataset (Dataset): Dataset containing bright star images.
        output_path (str, optional): Directory to save output files.
        file_save (bool): Flag to determine if output files should be written to disk.
        od_raster_threshold (float): Threshold for flagging OD variations.
        clean_entry (Any, optional): Clean image entry for computing expected OD.
        transformation_matrix_file (str, optional): File path to a FITS file containing 
            the FPAM-to-EXCAM transformation matrix.
        phot_method (str): Photometry method ("Aperture" or "PSF").
        flux_or_irr (str): Either 'flux' or 'irr' for the calibration approach.
        phot_kwargs (dict, optional): Extra arguments for the actual photometry function 
            (e.g., aper_phot).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'flux_results': Processed data per target.
            - 'combined_sweet_spot_data': Combined sweet-spot dataset.
            - 'overall_avg_od': Overall average optical density.
            - 'clean_frame_od': Computed expected OD (if applicable).
    """
    if phot_kwargs is None:
        phot_kwargs = {}

    # 1. Average calibration factor from dim stars
    cal_factor = compute_avg_calibration_factor(dim_stars_dataset,
                                                phot_method,
                                                flux_or_irr,
                                                phot_kwargs)

    # 2. Process bright star frames
    grouped_files = group_by_target(bright_stars_dataset)
    flux_results = {}
    aggregated_data_list = []
    common_metadata = {}

    for target, files in grouped_files.items():
        if not files:
            continue
        print(f"Processing bright target files: {target}")
        star_data = process_bright_target(target, files, cal_factor,
                                          od_raster_threshold, phot_method,
                                          phot_kwargs)
        flux_results[target] = star_data

        # Convert to Nx3 array [OD, x, y]
        target_sweet_spot = np.column_stack((
            star_data['od_values'],
            star_data['x_values'],
            star_data['y_values']
        ))
        aggregated_data_list.append(target_sweet_spot)

        # Initialize or validate the common metadata
        if not common_metadata:
            common_metadata = {
                'FPAMNAME': star_data['FPAMNAME'],
                'FPAM_H': star_data['FPAM_H'],
                'FPAM_V': star_data['FPAM_V'],
                'CFAMNAME': star_data['CFAMNAME']
            }
        else:
            # Basic consistency checks
            if (common_metadata['FPAMNAME'] != star_data['FPAMNAME']
                or abs(common_metadata['FPAM_H'] - star_data['FPAM_H']) >= 20
                or abs(common_metadata['FPAM_V'] - star_data['FPAM_V']) >= 20
                or common_metadata['CFAMNAME'] != star_data['CFAMNAME']):
                raise ValueError("Inconsistent FPAM or filter metadata among bright star observations.")

    # 3. Combine all sweet-spot arrays into one dataset
    combined_sweet_spot_data = (
        np.vstack(aggregated_data_list) if aggregated_data_list else np.empty((0, 3))
    )
    od_list = [res['average_od'] for res in flux_results.values()
               if res.get('average_od') is not None]
    overall_avg_od = np.mean(od_list) if od_list else None
    print(f"Average OD across bright targets: {overall_avg_od}")

    # 4. Load transform matrix if we need to append the clean frame row
    transform = None
    if clean_entry is not None and transformation_matrix_file:
        transform = load_transformation_matrix_from_fits(transformation_matrix_file)

    # 5. Create the final NDFilterSweetSpotDataset (optionally appending clean-frame OD)
    max_serial = get_max_serial(output_path)
    sweet_spot_dataset, appended_od = create_nd_sweet_spot_dataset(
        aggregated_sweet_spot_data=combined_sweet_spot_data,
        common_metadata=common_metadata,
        current_max=max_serial,
        output_path=output_path,
        save_to_disk=file_save,
        clean_frame_entry=clean_entry,
        transformation_matrix=transform,
        transformation_matrix_file=transformation_matrix_file
    )

    return {
        'flux_results': flux_results,
        'combined_sweet_spot_data': combined_sweet_spot_data,
        'overall_avg_od': overall_avg_od,
        'clean_frame_od': appended_od
    }
