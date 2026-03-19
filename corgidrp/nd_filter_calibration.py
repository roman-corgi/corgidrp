import os
import math
import numpy as np
from astropy.io import fits
import corgidrp.fluxcal as fluxcal
from corgidrp.data import (Dataset, FluxcalFactor, NDFilterSweetSpotDataset,
    FpamFsamCal)
from corgidrp.astrom import centroid_with_roi
from scipy.interpolate import griddata
import warnings

# =============================================================================
# Helper Functions
# =============================================================================

def group_by_keyword(dataset, prihdr_keyword=None, exthdr_keyword=None):
    """
    Split the dataset by either a primary header (prihdr) or extension header (exthdr) keyword
    and return a dictionary {target: subset}.

    Parameters:
        dataset (Dataset): The dataset to be split.
        prihdr_keyword (str, optional): FITS primary header keyword to split the dataset on.
        exthdr_keyword (str, optional): FITS extension header keyword to split the dataset on.

    Returns:
        dict: A dictionary where keys are unique target values and values are the 
            corresponding dataset subsets.
    
    Raises:
        ValueError: If neither keyword is provided.
    """
    if not prihdr_keyword and not exthdr_keyword:
        raise ValueError("At least one of 'prihdr_keyword' or 'exthdr_keyword' must be provided.")

    # Determine the splitting method
    if prihdr_keyword:
        split_datasets, unique_vals = dataset.split_dataset(prihdr_keywords=[prihdr_keyword])
    else:
        split_datasets, unique_vals = dataset.split_dataset(exthdr_keywords=[exthdr_keyword])

    # Construct dictionary {target: subset}
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
        star_name (str): The name of the star or file path to the (calspec) SED fits file for which to compute the irradiance.
        filter_name (str): The name of the filter used to determine the transmission curve.

    Returns:
        float: The computed band-integrated irradiance in erg/(s*cm^2).
    
    Raises:
        ValueError: If no matching filter curve file is found.
    """
    if star_name.split(".")[-1] == "fits":
        calspec_filepath = star_name
    else:
        calspec_filepath = fluxcal.get_calspec_file(star_name)[0]
    datadir = os.path.join(os.path.dirname(fluxcal.__file__), "data", "filter_curves")
    filter_files = [f for f in os.listdir(datadir) if filter_name in f and f.endswith('.csv')]
    if not filter_files:
        raise ValueError(f"No filter curve available with name {filter_name}")
    
    filter_filename = os.path.join(datadir, filter_files[0])
    wave, transmission = fluxcal.read_filter_curve(filter_filename)
    calspec_flux = fluxcal.read_cal_spec(calspec_filepath, wave)
    return fluxcal.calculate_band_irradiance(transmission, calspec_flux, wave)


def compute_avg_calibration_factor(dim_stars_dataset, phot_method, calspec_files = None, flux_or_irr="irr", phot_kwargs=None):
    """
    Compute the average flux calibration factor using dim stars (no ND filter).

    Parameters:
        dim_stars_dataset (iterable): Dataset containing dim star entries.
        phot_method (str): Photometry method to use ("Aperture" or "Gaussian").
        calspec_files (str, optional): str of one calspec file path or list of calspec filepaths
        flux_or_irr (str): Whether flux ('flux') or in-band irradiance ('irr') should be used.
        phot_kwargs (dict, optional): Dictionary of keyword arguments to pass to calibrate_fluxcal_aper.

    Returns:
        float: The average calibration factor.
    """
    if calspec_files is not None:
        one_calspec = False
        if isinstance(calspec_files, list):
            if len(calspec_files) != len(dim_stars_dataset):
                raise ValueError("wrong number of calspec filepaths")
        else:
            one_calspec = True
    if phot_kwargs is None:
        phot_kwargs = {}

    cal_values = []
    if phot_method == "Aperture":
        for i, entry in enumerate(dim_stars_dataset):
            if calspec_files is None:
                file = None
            else:
                if one_calspec:
                    file = calspec_files
                else:
                    file = calspec_files[i]
            cal_values.append(fluxcal.calibrate_fluxcal_aper(entry, calspec_file = file, flux_or_irr = flux_or_irr, phot_kwargs = phot_kwargs).fluxcal_fac)
    elif phot_method == "Gaussian":
        for i, entry in enumerate(dim_stars_dataset):
            if calspec_files is None:
                file = None
            else:
                if one_calspec:
                    file = calspec_files
                else:
                    file = calspec_files[i]
            cal_values.append(
            fluxcal.calibrate_fluxcal_gauss2d(entry, calspec_file = file, flux_or_irr = flux_or_irr, phot_kwargs = phot_kwargs).fluxcal_fac)
    else:
        raise ValueError("Photometry method must be either Aperture or Gaussian.")

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
      3. Perform photometry (Aperture or Gaussian).
      4. Compute OD from measured flux and the expected flux.
    
    Parameters:
        entry (corgidrp.Data.Image): The dataset entry containing image data and metadata.
        target (str): The target identifier for the dataset entry.
        phot_method (str): The photometry method to use ('Aperture' or 'Gaussian').
        phot_kwargs (dict): Additional keyword arguments for the photometry method.
        ref_fpam_name (str): The reference FPAM name for validation.
        ref_fpam_h (float): The reference FPAM horizontal position.
        ref_fpam_v (float): The reference FPAM vertical position.
        ref_cfam_name (str): The reference CFAM name for validation.
        expected_flux (float): The expected flux value for computing OD in erg/(s*cm^2)

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
        abs(hdr.get('FPAM_H') - ref_fpam_h) > 1.2 or    # within non-repeatability tolerance of 1.2 um
        abs(hdr.get('FPAM_V') - ref_fpam_v) > 1.2 or 
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
    elif phot_method == "Gaussian":
        phot_result = fluxcal.phot_by_gauss2d_fit(entry, **phot_kwargs)
    else:
        raise ValueError("phot_method must be Aperture or Gaussian.")

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
        target (str): The target star name or the file path to the corresponding (calspec) SED fits file.
        files (corgidrp.data.Dataset): Dataset of bright star images
        cal_factor (float or corgidrp.data.FluxcalFactor): Calibration factor.
        od_raster_threshold (float): Threshold for flagging OD variations.
        phot_method (str): Photometry method to use ("Aperture" or "Gaussian").
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

    if type(cal_factor) == FluxcalFactor:
        cal_factor_value = cal_factor.fluxcal_fac
    else:
        cal_factor_value = cal_factor

    # Compute expected flux
    expected_irradiance_no_nd = compute_expected_band_irradiance(target, ref_cfam_name)
    expected_flux = expected_irradiance_no_nd / cal_factor_value

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

    # Check for wide variation in OD values
    # Check for wide variation in OD values
    if od_array.size > 0:
        od_std = np.std(od_array)
        if od_std >= od_raster_threshold:
            warnings.warn(
                f"OD variation is high for target '{target}': "
                f"Standard deviation ({od_std:.3f}) exceeds threshold ({od_raster_threshold:.3f})."
            )
    else:
        od_std = np.nan

    average_od = np.mean(od_array) if od_array.size > 0 else np.nan

    return {
        'od_values': od_array,
        'average_od': average_od,
        'FPAMNAME': common_fpam_name,
        'FPAM_H': common_fpam_h,
        'FPAM_V': common_fpam_v,
        'CFAMNAME': ref_cfam_name,
        'flag': (od_std >= od_raster_threshold if not np.isnan(od_std) else False),
        'x_values': x_values,
        'y_values': y_values
    }


def create_nd_sweet_spot_dataset(aggregated_sweet_spot_data, common_metadata, od_var_flag, 
                                 input_dataset):
    """
    Create an NDFilterSweetSpotDataset FITS file with the Nx3 sweet-spot array.
    
    Parameters:
        aggregated_sweet_spot_data (numpy.ndarray): The aggregated Nx3 array containing 
            sweet-spot data in the format [OD, x, y].
        common_metadata (dict): A dictionary containing metadata such as FPAM/CFAM names 
            and offsets.
        od_var_flag (Bool): A flag that is passed in if the OD variance is too high among 
            rasters.
        input_dataset (corgidrp.data.Dataset): input dataset used to create the ND Filter 
            calibration

    Returns:
        tuple:
            NDFilterSweetSpotDataset: The generated ND filter sweet spot dataset.
    """
    final_sweet_spot_data = aggregated_sweet_spot_data.copy()

    # Create the NDFilterSweetSpotDataset, merge_headers is called inside __init__ 
    ndsweetspot_dataset = NDFilterSweetSpotDataset(
        data_or_filepath=final_sweet_spot_data,
        input_dataset=input_dataset
    )

    # Set ND-filter-specific metadata (want to overwrite the FPAM info with ND filter info)
    ndsweetspot_dataset.ext_hdr['BUNIT'] = ''  # dimensionless
    ndsweetspot_dataset.ext_hdr['DATALVL'] = 'CAL'
    ndsweetspot_dataset.ext_hdr['FPAMNAME'] = common_metadata.get('FPAMNAME')
    ndsweetspot_dataset.ext_hdr['FPAM_H'] = common_metadata.get('FPAM_H')
    ndsweetspot_dataset.ext_hdr['FPAM_V'] = common_metadata.get('FPAM_V')
    ndsweetspot_dataset.ext_hdr['ODFLAG'] = od_var_flag
    ndsweetspot_dataset.ext_hdr['HISTORY'] = "Combined sweet-spot dataset from bright star dithers"

    return ndsweetspot_dataset


def calculate_od_at_new_location(clean_frame_entry, fpamfsamcal, 
                                 ndsweetspot_dataset):
    """
    Use the NDFilterSweetSpot Dataset to calculate the OD at a new location for an input 
    image, using an FpamFsamCal calibration instance.
    
    Parameters:
        clean_frame_entry (corgidrp.Data.Image): A clean frame image.
        fpamfsamcal (corgidrp.data.FpamFsamCal): an instance of the
              FpamFsamCal calibration class. 
        ndsweetspot_dataset (corgidrp.Data.NDFilterSweetSpotDataset): ND Filter 
            Sweet Spot dataset

    Returns:
        interpolated_od (float): OD that is interpolated at the new star location
    """
    final_sweet_spot_data = ndsweetspot_dataset.data
    fpam2excam_matrix, _ = fpamfsamcal.data

    if (clean_frame_entry is not None) and (fpam2excam_matrix is not None):
        x_clean, y_clean = centroid_with_roi(clean_frame_entry.data)
        cframe_hdr = clean_frame_entry.ext_hdr
        sweetspot_hdr = ndsweetspot_dataset.ext_hdr

        # Compute FPAM offset
        clean_fpam_h = cframe_hdr.get('FPAM_H', 0.0)
        clean_fpam_v = cframe_hdr.get('FPAM_V', 0.0)
        sp_fpam_h    = sweetspot_hdr.get('FPAM_H', 0.0)
        sp_fpam_v    = sweetspot_hdr.get('FPAM_V', 0.0)
        fpam_offset  = np.array([clean_fpam_h - sp_fpam_h, clean_fpam_v - sp_fpam_v])

        # Transform to EXCAM offset
        excam_offset = fpam2excam_matrix @ fpam_offset
        x_adj = x_clean + excam_offset[0]
        y_adj = y_clean + excam_offset[1]

        # Interpolate OD at that new location
        interpolated_od = interpolate_od(final_sweet_spot_data, x_adj, y_adj)

    # TO DO: add in interpolated od into the header of the file and re-save? determine how the OD 
    # will be propagated

    return interpolated_od

# =============================================================================
# Main Workflow Function
# =============================================================================

def create_nd_filter_cal(stars_dataset,
                         od_raster_threshold = 0.1,
                         phot_method="Aperture",
                         flux_or_irr="irr",
                         phot_kwargs=None,
                         fluxcal_factor=None,
                         calspec_files = None):
    """
    Main ND Filter calibration workflow:
      1. Split dataset into dim and bright stars based on FPAMNAME keyword (or use cal factor input for dim)
      2. Compute avg calibration factor from dim stars.
      2. Group bright star frames by target + measure OD, centroids.
      3. Combine all sweet-spot data into a single Nx3 array.
    
    Parameters:
        stars_dataset (Dataset): Dataset containing star images. The splitting into bright and dim stars
            is performed based on the 'FPAMNAME' value in the FITS header. For example, entries with 'FPAMNAME'
            containing "dim" (case-insensitive) are considered dim stars.
        od_raster_threshold (float): Threshold for flagging OD variations.
            # TO DO: figure out what a reasonable value for this should be 
        phot_method (str): Photometry method ("Aperture" or "Gaussian").
        flux_or_irr (str): Either 'flux' or 'irr' for the calibration approach.
        phot_kwargs (dict, optional): Extra arguments for the actual photometry function 
            (e.g., aper_phot).
        fluxcal_factor (corgidrp.Data.FluxcalFactor, optional): A pre-computed flux factor calibration product to use
            if dim stars are not included as part of the input dataset
        calspec_files (list, optional): list of calspec filepaths

    Returns:
        sweet_spot_dataset (corgidrp.Data.NDFilterSweetSpotDataset): ND Filter calibration product for the dataset given
    """
    if phot_kwargs is None:
        phot_kwargs = {}

    # 1. Split the stars dataset into dim and bright stars based on FPAMNAME or FSAMNAME
    try:
        grouped_nd_files = group_by_keyword(stars_dataset, prihdr_keyword=None, exthdr_keyword='FPAMNAME')
    except:
        grouped_nd_files = group_by_keyword(stars_dataset, prihdr_keyword=None, exthdr_keyword='FSAMNAME')
        
    dim_stars_dataset = []
    bright_stars_dataset = []

    for keyword, records in grouped_nd_files.items():
        if keyword.startswith('ND'):
            # don't overwrite
            bright_stars_dataset.extend(records)
        else:
            dim_stars_dataset.extend(records)

    bright_stars_dataset = Dataset(bright_stars_dataset)
    dim_stars_dataset = Dataset(dim_stars_dataset)


    # 2. If a fluxcal factor was provided, use that for the dim stars
    if fluxcal_factor is not None:
        cal_factor = fluxcal_factor
    else:
        # Otherwise, compute the average calibration factor from dim
        # star frames
        cal_factor = compute_avg_calibration_factor(dim_stars_dataset,
                                                    phot_method,
                                                    calspec_files = calspec_files,
                                                    flux_or_irr = flux_or_irr,
                                                    phot_kwargs = phot_kwargs)

    # 3. Process bright star frames
    grouped_files = group_by_keyword(bright_stars_dataset, prihdr_keyword='TARGET', exthdr_keyword=None)
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

        od_var_flag = star_data['flag']

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
                or abs(common_metadata['FPAM_H'] - star_data['FPAM_H']) >= 1.2  # PAM non-repeatability tolerance is +/- 1.2 um
                or abs(common_metadata['FPAM_V'] - star_data['FPAM_V']) >= 1.2
                or common_metadata['CFAMNAME'] != star_data['CFAMNAME']):
                raise ValueError("Inconsistent FPAM or filter metadata among bright star observations.")

    # 4. Combine all sweet-spot arrays into one dataset
    combined_sweet_spot_data = (
        np.vstack(aggregated_data_list) if aggregated_data_list else np.empty((0, 3))
    )
    od_list = [res['average_od'] for res in flux_results.values()
               if res.get('average_od') is not None]
    overall_avg_od = np.mean(od_list) if od_list else None
    print(f"Average OD across bright targets: {overall_avg_od}")

    # 5. Create the final NDFilterSweetSpotDataset
    
    sweet_spot_dataset = create_nd_sweet_spot_dataset(
        aggregated_sweet_spot_data=combined_sweet_spot_data,
        common_metadata=common_metadata, od_var_flag = od_var_flag, input_dataset = stars_dataset
    )

    #TO DO: do we want to return flux?
    return sweet_spot_dataset
