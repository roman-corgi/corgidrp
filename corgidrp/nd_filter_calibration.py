import os
import re
import math
import numpy as np
from astropy.io import fits
import corgidrp.fluxcal as fluxcal
from corgidrp.data import NDFilterSweetSpotDataset, NDFilterOD
from corgidrp.astrom import centroid

# ---------------------------------------------------------------------------
# Existing Helper Functions
# ---------------------------------------------------------------------------
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


def compute_avg_calibration_factor(dim_stars_dataset):
    """
    Compute the average flux calibration factor using dim stars (no ND filter).
    """
    cal_values = [
        fluxcal.calibrate_fluxcal_aper(entry, 5, frac_enc_energy=1., method='subpixel',
                                       subpixels=5, background_sub=True, r_in=5,
                                       r_out=10, centering_method='xy').fluxcal_fac
        for entry in dim_stars_dataset
    ]
    avg_factor = np.mean(cal_values)
    print("Calibration factor:", avg_factor)
    return avg_factor


def process_bright_target(target, files, cal_factor, threshold, phot_method = "Aperture"):
    """
    Process bright star files for one target to compute optical density (OD)
    and (x, y) centroids for each dithered observation.
    Checks that FPAM keywords are consistent across all files.
    """
    od_values = []
    x_values = []
    y_values = []
    
    # Use the header from the first file as the common reference.
    first_hdr = files[0].ext_hdr
    filter_name = first_hdr['CFAMNAME']
    common_fpam_name = first_hdr.get('FPAMNAME')
    common_fpam_h = first_hdr.get('FPAM_H')
    common_fpam_v = first_hdr.get('FPAM_V')
    exptime = first_hdr.get('EXPTIME')
    
    expected_irradiance_no_nd = compute_expected_band_irradiance(target, filter_name)
    expected_flux = expected_irradiance_no_nd * exptime
    print("expected flux", expected_flux)
    print("if you multiply the bright expected flux by the ND you get: ", expected_flux*10**(-2.75))
    
    for entry in files:
        hdr = entry.ext_hdr
        if (hdr.get('FPAMNAME') != common_fpam_name or 
            hdr.get('FPAM_H') != common_fpam_h or 
            hdr.get('FPAM_V') != common_fpam_v):
            raise ValueError(f"Inconsistent FPAM header values in target {target} for file {entry}!")
        
        image_data = entry.data
        exptime = hdr['EXPTIME']
        
        x_center, y_center = centroid(image_data)
        if np.isnan(x_center) or np.isnan(y_center):
            print(f"Warning: Centroid could not be computed for {entry}")
            continue

        if phot_method == "Aperture":
            phot_result = fluxcal.aper_phot(entry, encircled_radius=5, frac_enc_energy=1.,
                                            method='subpixel', subpixels=5, 
                                            background_sub=True, r_in=5,
                                            r_out=10, centering_method='xy')
        elif phot_method == "PSF":
            phot_result = fluxcal.phot_by_gauss2d_fit(entry, fwhm=3, fit_shape=None, 
                                                      background_sub=True, r_in=5, 
                                                      r_out=10, centering_method='xy')
        else:
            raise ValueError(f"Must chose valid photometry method: Aperture or PSF.")
        
        measured_electrons = phot_result[0]
        measured_electrons_per_s = measured_electrons / exptime
        measured_flux = measured_electrons_per_s * cal_factor

        print("measured bright electrons", measured_electrons, "measured bright electrons/ exposure time", measured_electrons_per_s,
              "measured_flux", measured_flux)

        transmission = measured_flux / expected_flux
        print("calculated transmission", transmission, "supposed to be transmission", 10**(-2.75))
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
        'fpamname': common_fpam_name,
        'fpam_h': common_fpam_h,
        'fpam_v': common_fpam_v,
        'flag': star_flag,
        'x_values': x_values,
        'y_values': y_values,
        'filter_name': filter_name
    }


def get_max_serial(output_path, visit_id):
    """
    Find the current maximum serial number in the output directory.
    """
    pattern = re.compile(rf"CGI_{visit_id}_(\d+)_FilterBand_.*_NDF_CAL\.fits")
    max_serial = 0
    for filename in os.listdir(output_path):
        match = pattern.match(filename)
        if match:
            current_serial = int(match.group(1))
            if current_serial > max_serial:
                max_serial = current_serial
    return max_serial


# ---------------------------------------------------------------------------
# Updated Functions Using the New Data Classes
# ---------------------------------------------------------------------------
def create_nd_sweet_spot_dataset(aggregated_sweet_spot_data, common_metadata, visit_id, 
                                 current_max, output_path, save_to_disk=True):
    """
    Create an ND filter sweet spot dataset product (an instance of NDFilterSweetSpotDataset)
    using the aggregated sweet-spot data.

    Args:
        aggregated_sweet_spot_data (np.ndarray): An N×3 array with columns [OD, x, y].
        common_metadata (dict): Common metadata (FPAM values, filter name).
        visit_id (str): Identifier used in the output filename.
        current_max (int): Current maximum serial number.
        output_path (str): Directory where the file should be saved.
        save_to_disk (bool): If True, the product is written to disk.
    
    Returns:
        NDFilterSweetSpotDataset object.
    """
    pri_hdr = fits.Header()
    pri_hdr['SIMPLE'] = True
    pri_hdr['BITPIX'] = 32
    pri_hdr['OBSID'] = 0
    pri_hdr['COMMENT'] = "Combined ND Filter Sweet Spot Dataset primary header"
    
    ext_hdr = fits.Header()
    ext_hdr['FPAMNAME'] = common_metadata.get('fpamname')
    ext_hdr['FPAM_H'] = common_metadata.get('fpam_h')
    ext_hdr['FPAM_V'] = common_metadata.get('fpam_v')
    ext_hdr['FILTER'] = common_metadata.get('filter_name')
    ext_hdr['HISTORY'] = "Combined sweet-spot dataset from bright star observations"
    
    ndsweetspot_dataset = NDFilterSweetSpotDataset(
        data_or_filepath=aggregated_sweet_spot_data,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        input_dataset=None
    )
    
    if save_to_disk:
        serial_number = f"{current_max + 1:03d}"
        output_filename = f"CGI_{visit_id}_{serial_number}_FilterBand_{common_metadata.get('filter_name')}_NDF_SWEET.fits"
        ndsweetspot_dataset.save(filedir=output_path, filename=output_filename)
        print(f"ND Filter Sweet Spot Dataset saved to {output_filename}")
    
    return ndsweetspot_dataset


def create_expected_od_calibration_product(clean_entry, sweet_spot_data, sweet_spot_metadata, 
                                           transformation_matrix, transformation_matrix_file,
                                           visit_id, current_max, output_path, save_to_disk=True):
    """
    Compute the expected ND filter optical density (OD) for a clean image using a sweet-spot dataset and a 
    transformation matrix, and create an NDFilterOD calibration product.

    The function performs the following steps:
      1. Computes the centroid of the clean image.
      2. Retrieves FPAM values from the clean image header and from the sweet-spot metadata.
      3. Computes the offset between the clean image FPAM values and the sweet-spot FPAM values.
      4. Applies the provided 2×2 transformation matrix to map the FPAM offset to an EXCAM pixel offset.
      5. Adjusts the clean image centroid using the computed EXCAM offset.
      6. Finds the nearest sweet-spot data point (with columns [OD, x, y]) to the adjusted position.
      7. Creates an NDFilterOD calibration product with the computed expected OD as the product data.
      8. Optionally saves the calibration product to disk, embedding in the header the filename of the 
         transformation matrix used.

    Parameters:
        clean_entry: Clean image entry (must have attributes `.data` and `.ext_hdr` with FPAM values).
        sweet_spot_data (np.ndarray): Aggregated sweet-spot dataset (an M×3 array with columns [OD, x, y]).
        sweet_spot_metadata (dict): Metadata from the sweet-spot product (must include keys 'fpam_h', 'fpam_v', and 'filter_name').
        transformation_matrix (np.ndarray): A 2×2 matrix mapping FPAM motions to EXCAM pixel motions.
        transformation_matrix_file (str): File path to the FITS file containing the transformation matrix.
        visit_id (str): Identifier used in the output filename.
        current_max (int): Current maximum serial number for expected OD products.
        output_path (str): Directory where the output file should be saved.
        save_to_disk (bool): If True, the calibration product is saved to disk.

    Returns:
        NDFilterOD object representing the expected OD calibration product.
    """
    # --- Compute expected OD ---
    # Compute the centroid of the clean image.
    x_clean, y_clean = centroid(clean_entry.data)
    hdr = clean_entry.ext_hdr
    clean_fpam_h = hdr.get('FPAM_H')
    clean_fpam_v = hdr.get('FPAM_V')
    
    # Retrieve sweet-spot FPAM values.
    sp_fpam_h = sweet_spot_metadata.get('fpam_h')
    sp_fpam_v = sweet_spot_metadata.get('fpam_v')
    
    # Compute the FPAM offset and transform it to EXCAM offset.
    fpam_offset = np.array([clean_fpam_h - sp_fpam_h, clean_fpam_v - sp_fpam_v])
    excam_offset = transformation_matrix @ fpam_offset
    
    # Adjust the clean image centroid.
    x_adj = x_clean + excam_offset[0]
    y_adj = y_clean + excam_offset[1]
    
    # Find the nearest sweet-spot point (columns 1 and 2 are x and y).
    positions = sweet_spot_data[:, 1:3]
    diffs = positions - np.array([x_adj, y_adj])
    distances = np.linalg.norm(diffs, axis=1)
    nearest_idx = np.argmin(distances)
    expected_od = sweet_spot_data[nearest_idx, 0]
    
    # --- Build FITS headers for the calibration product ---
    pri_hdr = fits.Header()
    pri_hdr['SIMPLE'] = True
    pri_hdr['BITPIX'] = 32
    pri_hdr['OBSID'] = 0
    pri_hdr['COMMENT'] = "Expected ND filter OD for clean image (Requirement 1090877)"
    
    ext_hdr = fits.Header()
    # Store the clean image FPAM values.
    ext_hdr['FPAM_H'] = clean_fpam_h
    ext_hdr['FPAM_V'] = clean_fpam_v
    # Store the sweet-spot FPAM values.
    ext_hdr['SPFPAM_H'] = sp_fpam_h
    ext_hdr['SPFPAM_V'] = sp_fpam_v
    # Record the filter name.
    ext_hdr['CFAMNAME'] = sweet_spot_metadata.get('filter_name')
    # Record the transformation matrix file used (basename only).
    ext_hdr['HISTORY'] = (
        f"Expected OD computed using sweet-spot dataset and transformation matrix file: "
        f"{os.path.basename(transformation_matrix_file)} (req. 1090877)"
    )
    
    # --- Create the calibration product ---
    # Ensure the expected OD is stored as a numeric (float) array.
    data = np.array([[float(expected_od)]], dtype=float)
    
    nd_expected_product = NDFilterOD(
        data_or_filepath=data,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        input_dataset=None
    )
    
    if save_to_disk:
        serial_number = f"{current_max + 1:03d}"
        output_filename = (
            f"CGI_{visit_id}_{serial_number}_ExpectedOD_{sweet_spot_metadata.get('filter_name')}_NDF_CAL.fits"
        )
        nd_expected_product.save(filedir=output_path, filename=output_filename)
        print(f"Expected OD Calibration product saved to {output_filename}")
    
    return nd_expected_product


def get_max_serial_expected_od(output_path, visit_id):
    """
    Find the current maximum serial number for expected OD products.
    """
    pattern = re.compile(rf"CGI_{visit_id}_(\d+)_ExpectedOD_.*_NDF_CAL\.fits")
    max_serial = 0
    for filename in os.listdir(output_path):
        match = pattern.match(filename)
        if match:
            current_serial = int(match.group(1))
            if current_serial > max_serial:
                max_serial = current_serial
    return max_serial
