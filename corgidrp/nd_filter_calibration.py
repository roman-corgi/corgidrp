import os
import math
import numpy as np
from astropy.io import fits
import corgidrp.fluxcal as fluxcal
from corgidrp.data import (Dataset, FluxcalFactor, NDFilterSweetSpotDataset,
    NDSpectroscopy, FpamFsamCal)
from corgidrp.astrom import centroid_with_roi
from scipy.interpolate import griddata, interp1d
import warnings
import corgidrp.spec as spec_module


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


# =============================================================================
# Spectroscopy ND Filter Calibration 
# =============================================================================
def calculate_od_spec_at_new_location(clean_spec_image, fpamfsamcal, 
                                 ndspectroscopy_dataset, wave_grid=None):
    """
    Use the NDSpectroscopy Dataset to calculate the OD at a new location for an input 
    image, using an FpamFsamCal calibration instance.
    
    Parameters:
        clean_spec_image (corgidrp.Data.Image): A clean spec image with 'SPEC' extension hdus.
        fpamfsamcal (corgidrp.data.FpamFsamCal): an instance of the
              FpamFsamCal calibration class. 
        ndspectroscopy_dataset (corgidrp.Data.NDSpectroscopy): ND Spectroscopy dataset.
        wave_grid (list of float or np.array, optional): Wavelength grid specfied
            by user. Defaults to None (wavelength grid taken from image hdu 'SPEC_WAVE').

    Returns:
        common_wave (np.array): Wavelength grid on which OD values are computed.
        interp_od (np.array): OD as a function of wavelength that is interpolated at the new star location.
    """
    fpam2excam_matrix, _ = fpamfsamcal.data

    if (clean_spec_image is not None) and (fpam2excam_matrix is not None):
        cframe_hdr = clean_spec_image.ext_hdr
        sweetspot_hdr = ndspectroscopy_dataset.ext_hdr
        x_clean = cframe_hdr.get('WV0_X',0.0)
        y_clean = cframe_hdr.get('WV0_Y',0.0)

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
    
        common_wave, interp_od = ndspectroscopy_dataset.interpolate_od(x_adj,y_adj,spec_image=clean_spec_image)
            
    # TO DO: add in interpolated od into the header of the file and re-save? determine how the OD 
    # will be propagated

    return common_wave, interp_od

def compute_od_spectrum_for_frame(frame, target, sf_cal, calspec_filepath, ref_fpam_name,
                         ref_fpam_h, ref_fpam_v, ref_dpam_name, ref_cfam_name):
    """
    Compute OD(lambda) for a single bright-star frame observed through the ND
    filter with the prism in.

    Args:
        frame (corgidrp.data.Image): L3 frame with SPEC, SPEC_WAVE, and SPEC_ERR
            extensions (in units of photoelectron/s/bin) produced by extract_spec.
        target (str): The target star name.
        sf_cal (corgidrp.data.SpecFluxCal): Spectral flux calibration C(lambda)
            taken from the dim (no-ND) star.  Units: erg/(s*cm^2*AA) / (photoelectron/s/bin).
        calspec_filepath (str): Path to the CALSPEC SED FITS file for the bright
            star being observed through the ND filter.
        ref_fpam_name (str): The reference FPAM name for validation.
        ref_fpam_h (float): The reference FPAM horizontal position.
        ref_fpam_v (float): The reference FPAM vertical position.
        ref_dpam_name (str): The reference DPAM name for validation.
        ref_cfam_name (str): The reference CFAM name for validation.

    Returns:
        tuple of arrays:
            numpy.array: The computed optical depth (OD) as a function of wavelength.
            numpy.array: The wavelength grid.
            numpy.array: The error in OD measurements across the wavelength grid.

    Raises:
        ValueError: If FPAM/DPAM/CFAM metadata do not match the reference values.
    """
    hdr = frame.ext_hdr

    # Metadata checks
    if (hdr.get('FPAMNAME') != ref_fpam_name or 
        abs(hdr.get('FPAM_H') - ref_fpam_h) > 1.2 or    # within non-repeatability tolerance of 1.2 um
        abs(hdr.get('FPAM_V') - ref_fpam_v) > 1.2 or 
        hdr.get('CFAMNAME') != ref_cfam_name or
        hdr.get('DPAMNAME') != ref_dpam_name):
        raise ValueError(
            f"Inconsistent FPAM/CFAM/DPAM header values in target {target} for file {frame}!"
        )

    # Measured spectrum (e-/s/bin) and wavelength grid (nm) from extract_spec
    counts_nd = frame.hdu_list['SPEC'].data.astype(float)
    spec_wave  = frame.hdu_list['SPEC_WAVE'].data.astype(float)
    spec_err   = frame.hdu_list['SPEC_ERR'].data.astype(float)
    if spec_err.ndim > 1:
        spec_err = spec_err[0]   # get first error 

    # Make sure spec_wave is in ascending order bc read_cal_spec and interp1d both
    # require ascending wavelength grids
    sort_idx = np.argsort(spec_wave)
    spec_wave  = spec_wave[sort_idx]
    counts_nd  = counts_nd[sort_idx]
    spec_err   = spec_err[sort_idx]

    # CALSPEC SED for the bright star at these wavelengths.
    # read_cal_spec expects wavelengths in Angstrom, spec_wave is in nm
    sed_bright = fluxcal.read_cal_spec(calspec_filepath, spec_wave * 10.0)

    # Interpolate C(lambda) from SpecFluxCal onto the bright-star wavelength grid.
    # sf_cal.wavelength may also be in descending order so sort it ascending
    sf_sort_idx = np.argsort(sf_cal.wavelength)
    c_interp_fn = interp1d(sf_cal.wavelength[sf_sort_idx],
                           sf_cal.specflux[sf_sort_idx],
                           kind='linear', fill_value='extrapolate')
    c_at_wave = c_interp_fn(spec_wave)   # erg/(s*cm^2*AA) / (e-/s/bin)

    # Expected e-/s/bin with no ND filter in beam
    expected_counts = sed_bright / c_at_wave
    
    # Transmission and OD (suppress warnings in case it encounters nan or 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        transmission = counts_nd / expected_counts
        od_spectrum  = -np.log10(transmission)

    # Propagate photon-counting uncertainty. sigma_OD = sigma_counts / (N * ln10)
    with np.errstate(divide='ignore', invalid='ignore'):
        od_err = spec_err / (np.abs(counts_nd) * np.log(10))

    return od_spectrum, spec_wave, od_err

def process_bright_target_spec(target, files, sf_cal, calspec_filepath, od_raster_threshold):
    """
    Process bright star files for one target to compute optical density (OD)
    and (x, y) centroids for each dithered observation.
    Checks that FPAM keywords are consistent across all files.
    
    Additional photometry options are passed via phot_kwargs.
    This allows users to override default settings for functions like aper_phot.
    
    Parameters:
        target (str): The target star name.
        files (corgidrp.data.Dataset): Dataset of bright star images.
        sf_cal (corgidrp.data.SpecFluxCal): Spectral flux calibration C(lambda)
            taken from the dim (no-ND) star.  Units: erg/(s*cm^2*AA) / (photoelectron/s/bin).
        calspec_filepath (str): Path to the CALSPEC SED FITS file for the bright
            star being observed through the ND filter.
        od_raster_threshold (float): Threshold for flagging OD variations.
    
    Returns:
        dict: A dictionary containing computed OD values, wavelengths, centroids, and other metadata.
    """

    first_hdr = files[0].ext_hdr
    ref_cfam_name = first_hdr['CFAMNAME']
    common_dpam_name = first_hdr.get('DPAMNAME')
    common_fpam_name = first_hdr.get('FPAMNAME')
    common_fpam_h    = first_hdr.get('FPAM_H')
    common_fpam_v    = first_hdr.get('FPAM_V')
    exptime          = first_hdr.get('EXPTIME')

    od_all_spectra, wave_grids, od_all_err, x_values, y_values, xerr_values, yerr_values = [], [], [], [], [], [], []

    target_dataset = Dataset(files)
    grouped_dataset, fsmpos = target_dataset.split_dataset(exthdr_keywords=["FSMX","FSMY"])

    for fsm in fsmpos:
        matched_index = [i for i, key in enumerate(fsmpos) if key == fsm]
        grouped_subset = grouped_dataset[int(matched_index[0])]

        od_spectra, wave_spectra, oderr_spectra = [], [], []
        for frame in grouped_subset:
            od_spectrum, wave_spectrum, oderr_spectrum = compute_od_spectrum_for_frame(frame, target, sf_cal, 
                                                    calspec_filepath, common_fpam_name, common_fpam_h, common_fpam_v, 
                                                    common_dpam_name, ref_cfam_name)        
            # Skip if centroid was not valid
            if od_spectrum is None:
                continue
            
            od_spectra.append(od_spectrum)
            wave_spectra.append(wave_spectrum)
            oderr_spectra.append(oderr_spectrum)
        
        # EXCAM star position from wavelength zeropoint, it is same for all frames in a specific dither.
        x_center = grouped_subset[0].ext_hdr['WV0_X']
        y_center = grouped_subset[0].ext_hdr['WV0_Y']
        xcen_err = grouped_subset[0].ext_hdr['WV0_XERR']
        ycen_err = grouped_subset[0].ext_hdr['WV0_YERR']

        od_spectra = np.array(od_spectra)
        wave_spectra = np.array(wave_spectra)
        oderr_spectra = np.array(oderr_spectra)
        x_center = x_center * np.ones_like(wave_spectrum)
        y_center = y_center * np.ones_like(wave_spectrum)
        xcen_err = xcen_err * np.ones_like(wave_spectrum)
        ycen_err = ycen_err * np.ones_like(wave_spectrum)

        od_all_spectra.append(np.mean(od_spectra, axis=0))
        wave_grids.append(np.mean(wave_spectra, axis=0))
        od_all_err.append(np.sqrt(np.nansum(oderr_spectra ** 2, axis=0)) / len(oderr_spectra))
        x_values.append(x_center)
        y_values.append(y_center)
        xerr_values.append(xcen_err)
        yerr_values.append(ycen_err)

    od_stack, wave_stack, err_stack = [], [], []
    for od, wave, oderr in zip(od_all_spectra[:], wave_grids[:], od_all_err[:]):
        od_stack.append(od)
        wave_stack.append(wave)
        err_stack.append(oderr)

    od_array = np.array(od_stack)
    wave_array = np.array(wave_stack)
    err_array = np.array(err_stack)

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

    average_od = np.nanmean(od_array) if od_array.size > 0 else np.nan

    return {
        'od_values': od_array,
        'wave_grid': wave_array,
        'od_err_values': err_array,
        'average_od': average_od,
        'FPAMNAME': common_fpam_name,
        'FPAM_H': common_fpam_h,
        'FPAM_V': common_fpam_v,
        'CFAMNAME': ref_cfam_name,
        'DPAMNAME': common_dpam_name,
        'flag': (od_std >= od_raster_threshold if not np.isnan(od_std) else False),
        'x_values': x_values,
        'y_values': y_values,
        'xerr_values': xerr_values,
        'yerr_values': yerr_values
    }

def create_nd_spectroscopy_dataset(aggregated_sweet_spot_data, common_metadata, od_var_flag, 
                                    input_dataset, aggregated_sweet_spot_err = None):
    """
    Create an NDSpectroscopy FITS file with the spectroscopy sweet-spot array.
    
    Parameters:
        aggregated_sweet_spot_data (numpy.ndarray): The aggregated NxN_wavex4 array containing 
            sweet-spot data as a function of wavelength in the format [wave, OD, x, y].
        common_metadata (dict): A dictionary containing metadata such as FPAM/DPAM/CFAM names 
            and offsets.
        od_var_flag (Bool): A flag that is passed in if the OD variance is too high among 
            rasters.
        input_dataset (corgidrp.data.Dataset): input dataset used to create the ND Spec 
            calibration.
        aggregated_sweet_spot_err (numpy.ndarray, optional): The aggregated NxN_wavex4 array containing 
            OD_err as a function of wavelength in the format [wave_err, OD_err, x_err, y_err].
            Defaults to None.

    Returns:
        tuple:
            NDSpectroscopy: The generated ND spectroscopy sweet spot dataset.
    """
    final_sweet_spot_data = aggregated_sweet_spot_data.copy()
    if aggregated_sweet_spot_err is not None:
        final_sweet_spot_err = aggregated_sweet_spot_err.copy()
    
    # Create the NDSpectroscopy, merge_headers is called inside __init__ 
    ndspectroscopy_dataset = NDSpectroscopy(
        data_or_filepath=final_sweet_spot_data,
        err=final_sweet_spot_err,
        input_dataset=input_dataset
    )

    # Set ND-filter-specific metadata (want to overwrite the FPAM info with ND filter info)
    ndspectroscopy_dataset.ext_hdr['BUNIT'] = ''  # dimensionless
    ndspectroscopy_dataset.ext_hdr['DATALVL'] = 'CAL'
    ndspectroscopy_dataset.ext_hdr['FPAMNAME'] = common_metadata.get('FPAMNAME')
    ndspectroscopy_dataset.ext_hdr['FPAM_H'] = common_metadata.get('FPAM_H')
    ndspectroscopy_dataset.ext_hdr['FPAM_V'] = common_metadata.get('FPAM_V')
    ndspectroscopy_dataset.ext_hdr['CFAMNAME'] = common_metadata.get('CFAMNAME')
    ndspectroscopy_dataset.ext_hdr['DPAMNAME'] = common_metadata.get('DPAMNAME')
    ndspectroscopy_dataset.ext_hdr['ODFLAG'] = od_var_flag
    ndspectroscopy_dataset.ext_hdr['HISTORY'] = "Combined ND spec sweet-spot dataset from bright star dithers"

    return ndspectroscopy_dataset

def create_nd_filter_cal_spec(stars_dataset, spec_fluxcal=None, od_raster_threshold = 0.1, calspec_files=None, outputdir=None):
    """
    Create the spectroscopy ND filter calibration product.

    Accepts a dataset of L3 frames that have already been processed through
    divide_by_exptime, determine_wave_zeropoint, add_wavelength_map, and
    extract_spec. Each frame has SPEC, SPEC_WAVE, and SPEC_ERR extensions
    with BUNIT='photoelectron/s').

    The stars_dataset must contain:
      * dim-star frames taken with no ND filter (FPAMNAME not starting with 'ND') -
        these are used to calculate the spectral flux calibration for the instrument
        C(lambda) via spec_fluxcal(), unless a pre-computed SpecFluxCal is given via
        the spec_fluxcal argument.
      * bright-star frames (FPAMNAME starting with 'ND') — the star observed through
        the ND filter whose OD(lambda) should be measured.

    If multiple bright frames are present (ie repeated images at the same position)
    their OD(lambda) spectra are remapped to a common wavelength grid and averaged
    before the calibration product is created.

    Args:
        stars_dataset (corgidrp.data.Dataset): L3 frames with SPEC extensions.
        spec_fluxcal (corgidrp.data.SpecFluxCal, optional): Pre-computed spectral
            flux calibration product.  When supplied, dim-star frames in the
            dataset are ignored.
        od_raster_threshold (float): Threshold for flagging OD variations.
        calspec_files (str or dict, optional): CALSPEC filepath(s) for the
            bright-star target(s). If str, same file is used for all bright frames. 
            If dict, TARGET should be key with the CALSPEC filepath as the corresponding
            value, so dict must have one entry per TARGET among bright frames.
            When None the TARGET primary-header keyword is used to look up each
            star automatically.
        outputdir (str, optional): Directory where the auto-generated SpecFluxCal
            is saved when it is computed from dim-star frames.  Defaults to the
            current directory.

    Returns:
        corgidrp.data.NDSpectroscopy: OD(lambda) calibration product.
    """
    # 1. Split the dataset into dim (no ND) and bright (ND) frames by FPAMNAME.
    # Fall back to FSAMNAME if only 1 FPAMNAME group
    grouped = group_by_keyword(stars_dataset, exthdr_keyword='FPAMNAME')
    if len(grouped) < 2:
        grouped = group_by_keyword(stars_dataset, exthdr_keyword='FSAMNAME')

    dim_frames    = []
    bright_frames = []
    for keyword, records in grouped.items():
        if keyword.startswith('ND'):
            bright_frames.extend(records)
        else:
            dim_frames.extend(records)

    if not bright_frames:
        raise ValueError(
            "No bright (ND-filter) frames found in the dataset. "
            "Frames with FPAMNAME (or FSAMNAME) starting with 'ND' are required."
        )

    bright_dataset = Dataset(bright_frames)

    # 2. Make sure that the images were taken in spectroscopy mode
    first_bright = bright_dataset[0]
    dpam = first_bright.ext_hdr.get('DPAMNAME', '')
    if not dpam.startswith('PRISM'):
        raise ValueError(
            f"Expected DPAMNAME starting with 'PRISM' for spectroscopy ND "
            f"calibration, got '{dpam}'."
        )

    # 3. Get the spectral flux calibration C(lambda)
    if spec_fluxcal is not None:
        sf_cal = spec_fluxcal
    else:
        if not dim_frames:
            raise ValueError(
                "No dim-star frames found and no spec_fluxcal provided. "
                "Either include dim-star frames (FPAMNAME != ND*) in the dataset "
                "or pass a pre-computed SpecFluxCal via the spec_fluxcal argument."
            )
        dim_dataset = Dataset(dim_frames)
        # Automatically look up the dim star from its TARGET header keyword.
        # (should be a known CALSPEC standard used for flux calibration)
        sf_cal = spec_module.spec_fluxcal(dim_dataset, calspec_file=None)
        if outputdir is None:
            outputdir = '.'
        sf_cal.save(filedir=outputdir)
    
    # 4. Process bright star frames
    grouped_bright_files = group_by_keyword(bright_dataset, prihdr_keyword='TARGET', exthdr_keyword=None)
    flux_results = {}
    aggregated_data_list = []
    aggregated_err_list = []
    aggregated_xy_list = []
    common_metadata = {}

    for i, (target, files) in enumerate(grouped_bright_files.items()):
        if not files:
            continue
        print(f"Processing bright target files: {target}")
        
        if calspec_files is None:
            calspec_fp = fluxcal.get_calspec_file(target)[0]
        elif isinstance(calspec_files, str):
            calspec_fp = calspec_files
        else:
            if len(calspec_files) != len(grouped_bright_files):
                raise ValueError("Number of CALSPEC files in dict do not correspond to number of unique targets, please provide one CALSPEC file per unique target.")
            calspec_fp = calspec_files[target]
        
        star_data = process_bright_target_spec(target, files, sf_cal, calspec_fp,
                                          od_raster_threshold)
        flux_results[target] = star_data

        od_var_flag = star_data['flag']

        # Convert to an NxN_wavex4 array [wave, OD, x, y], and a NxN_wavex4 array [wave_err, OD_err, x_err, y_err]
        target_sweet_spot = np.dstack((
            star_data['wave_grid'],
            star_data['od_values'],
            star_data['x_values'],
            star_data['y_values']
        ))
        aggregated_data_list.append(target_sweet_spot)

        target_sweet_spot_err = np.dstack((
            np.zeros_like(star_data['wave_grid']),
            star_data['od_err_values'],
            star_data['xerr_values'],
            star_data['yerr_values']
        ))
        aggregated_err_list.append(target_sweet_spot_err)

        # Initialize or validate the common metadata
        if not common_metadata:
            common_metadata = {
                'FPAMNAME': star_data['FPAMNAME'],
                'FPAM_H': star_data['FPAM_H'],
                'FPAM_V': star_data['FPAM_V'],
                'CFAMNAME': star_data['CFAMNAME'],
                'DPAMNAME': star_data['DPAMNAME']
            }
        else:
            # Basic consistency checks
            if (common_metadata['FPAMNAME'] != star_data['FPAMNAME']
                or abs(common_metadata['FPAM_H'] - star_data['FPAM_H']) >= 1.2  # PAM non-repeatability tolerance is +/- 1.2 um
                or abs(common_metadata['FPAM_V'] - star_data['FPAM_V']) >= 1.2
                or common_metadata['CFAMNAME'] != star_data['CFAMNAME']
                or common_metadata['DPAMNAME'] != star_data['DPAMNAME']):
                raise ValueError("Inconsistent FPAM, DPAM, or filter metadata among bright star observations.")

    # 5. Combine all sweet-spot arrays into one dataset
    combined_sweet_spot_data = (
        np.vstack(aggregated_data_list) if aggregated_data_list else np.empty((0, 2))
    )
    combined_sweet_spot_err = (
        np.vstack(aggregated_err_list) if aggregated_err_list else np.empty((0, 2))
    )

    od_list = [res['average_od'] for res in flux_results.values()
               if res.get('average_od') is not None]
    overall_avg_od = np.nanmean(od_list) if od_list else None
    print(f"Average OD across bright targets: {overall_avg_od:.4f}")


    # 6. Create the final NDSpectroscopy
    
    sweet_spot_dataset = create_nd_spectroscopy_dataset(
        aggregated_sweet_spot_data=combined_sweet_spot_data,
        aggregated_sweet_spot_err=combined_sweet_spot_err,
        common_metadata=common_metadata, od_var_flag = od_var_flag, input_dataset = stars_dataset
    )

    #TO DO: do we want to return flux?
    return sweet_spot_dataset

def apply_od_spec_correction_to_image(clean_spec_image, fpamfsamcal, ndspectroscopy_dataset, wave_grid=None):
    """
    Wrapper function that applies OD correction as a function of wavelength to the input L4 image. Invokes 
    calculate_od_spec_at_new_location() to calculate the OD at a new location for an input 
    image, using a NDSpectroscopy calibration and a FpamFsamCal calibration instance.
    
    Parameters:
        clean_spec_image (corgidrp.Data.Image): A clean L4 spec image with 'SPEC' extension hdus.
        fpamfsamcal (corgidrp.data.FpamFsamCal): an instance of the
              FpamFsamCal calibration class. 
        ndspectroscopy_dataset (corgidrp.Data.NDSpectroscopy): ND Spectroscopy dataset.
        wave_grid (list of float or np.array, optional): Wavelength grid specfied
            by user. Defaults to None.

    Returns:
        corrected_image (corgidrp.Data.Image): A spec image with the OD-corrected spectrum and spec errors.
    """

    if clean_spec_image.ext_hdr['DATALVL'] != 'L4':
        raise Exception("Please provide valid L4 image to apply OD spec correction.")

    if 'SPEC' not in clean_spec_image.hdu_list:
        raise Exception("L4 image should have hdu 'SPEC' in hdulist")

    if 'SPEC_WAVE' not in clean_spec_image.hdu_list:
        raise Exception("L4 image should have hdu 'SPEC_WAVE' in hdulist")

    if 'SPEC_ERR' not in clean_spec_image.hdu_list:
        raise Exception("L4 image should have hdu 'SPEC_ERR' in hdulist")

    #Calculate OD as function of wavelength at image zeropoint. If no wave_grid passed, common_wave = spec_wave.
    common_wave, od_spec = calculate_od_spec_at_new_location(clean_spec_image = clean_spec_image, \
            fpamfsamcal = fpamfsamcal, ndspectroscopy_dataset = ndspectroscopy_dataset, wave_grid=wave_grid)
    
    corrected_image = clean_spec_image.copy()

    spec = corrected_image.hdu_list['SPEC'].data
    spec_wave = corrected_image.hdu_list['SPEC_WAVE'].data
    spec_err = corrected_image.hdu_list['SPEC_ERR'].data
    
    # Interpolate SPEC values on new wavelength grid. This is only done if a wave_grid with wavelength points 
    # significantly different from spec_wave (delta_wave > 0.01 nm) is passed by user into calculate_od_spec_at_new_location().
    # Else common_wave = spec_wave, and no interpolation of SPEC values is performed.
    if not np.allclose(spec_wave, common_wave, atol=0.01):
        if common_wave[0] < spec_wave[0] or common_wave[-1] > spec_wave[-1]:
            print(f"WARNING: Custom wavelength grid has points outside wavelengths in input L4 image 'SPEC_WAVE' hdu. Attempting to extrapolate 'SPEC' values at these points.")
            remap_spec  = interp1d(spec_wave, spec,    kind='linear',
                            bounds_error=False, fill_value="extrapolate")
            remap_spec_err  = interp1d(spec_wave, spec_err,    kind='linear',
                            bounds_error=False, fill_value="extrapolate")
            spec    = remap_spec(common_wave)
            spec_err = remap_spec_err(common_wave)
        
        else:
            remap_spec  = interp1d(spec_wave, spec, kind='linear',
                            bounds_error=False, fill_value=np.nan)
            remap_spec_err  = interp1d(spec_wave, spec_err, kind='linear',
                            bounds_error=False, fill_value=np.nan)
            spec    = remap_spec(common_wave)
            spec_err = remap_spec_err(common_wave)       
    
    corr_spec = spec * 10**(od_spec)
    corr_spec_err = spec_err * 10**(od_spec)

    corrected_image.hdu_list['SPEC'].data = corr_spec
    corrected_image.hdu_list['SPEC_ERR'].data = corr_spec_err
    if not np.allclose(spec_wave, common_wave, atol=0.01):  #Update wavelength grid if spec_wave and common_wave are different
        corrected_image.hdu_list['SPEC_WAVE'].data = common_wave
        corrected_image.hdu_list['SPEC_WAVE_ERR'].data = np.zeros_like(common_wave)

    return corrected_image
