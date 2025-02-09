import os
import re
import math
import tempfile
from typing import List, Dict, Any, Optional

import numpy as np
from astropy.io import fits
import pytest

# Import helper functions and classes from calibration module
from corgidrp.nd_filter_calibration import (
    compute_expected_band_irradiance,
    group_by_target,
    compute_avg_calibration_factor,
    process_bright_target,
    create_nd_sweet_spot_dataset,
    create_expected_od_calibration_product,
    get_max_serial,
    get_max_serial_expected_od
)
from corgidrp.data import Dataset, NDFilterSweetSpotDataset, NDFilterOD, Image
from corgidrp.mocks import create_default_headers, create_flux_image
from corgidrp.astrom import centroid

# ---------------------------------------------------------------------------
# Global variables for test star names
# ---------------------------------------------------------------------------
BRIGHT_STARS = ['109 Vir']
DIM_STARS = ['TYC 4433-1800-1']

# ---------------------------------------------------------------------------
# Utility Function: Ensure output directory exists.
# ---------------------------------------------------------------------------
def ensure_directory_exists(path: str) -> None:
    """Create the directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper Function: Load a transformation matrix from a FITS file.
# ---------------------------------------------------------------------------
def load_transformation_matrix_from_fits(file_path: str) -> np.ndarray:
    """
    Load a transformation matrix from a FITS file.

    Parameters:
        file_path (str): Path to the FITS file containing the transformation matrix.

    Returns:
        np.ndarray: The transformation matrix extracted from the FITS file.
    """
    with fits.open(file_path) as hdul:
        # First try the primary HDU.
        data = hdul[0].data
        if data is None:
            # If no data in primary, try extension 1.
            data = hdul[1].data
        return np.array(data)

# ---------------------------------------------------------------------------
# Mocks for creating test datasets
# ---------------------------------------------------------------------------
def mock_dim_dataset_files(output_path: str, dim_exptime: float, filter_used: str, cal_factor: float) -> List[Any]:
    """
    Create mock FITS files for dim stars (calibration references). Uses
    create_flux_image from Mocks.py to do this, with the expected flux
    from calspec files for that star.

    Parameters:
        output_path (str): Directory where the files will be created.
        dim_exptime (float): Exposure time for the dim stars.
        filter_used (str): Filter identifier.
        cal_factor (float): Calibration factor to apply.

    Returns:
        List[Any]: List of mock FITS image objects.
    """
    ensure_directory_exists(output_path)
    dim_star_images = []

    for star_name in DIM_STARS:
        # Compute the expected flux for the dim star using the CALSPEC model.
        star_flux = compute_expected_band_irradiance(star_name, filter_used)
        total_dim_flux = star_flux * dim_exptime
        fwhm = 3  # Full-width at half maximum for the point spread function

        flux_image = create_flux_image(
            total_dim_flux, fwhm, cal_factor, filter_used, star_name,
            fsm_x=0, fsm_y=0, exptime=dim_exptime, filedir=output_path,
            color_cor=1.0, platescale=21.8, add_gauss_noise=True, noise_scale=1.0,
            file_save=False
        )
        dim_star_images.append(flux_image)

    return dim_star_images


def mock_bright_dataset_files(output_path: str, bright_exptime: float, filter_used: str, OD: float, cal_factor: float) -> List[Any]:
    """
    Create mock FITS files for bright stars observed with an ND filter.
    Uses create_flux_image from Mocks.py to do this, with the expected flux
    from calspec files for that star.

    Parameters:
        output_path (str): Directory where the files will be created.
        bright_exptime (float): Exposure time for bright stars.
        filter_used (str): Filter identifier.
        OD (float): Optical density for the ND filter.
        cal_factor (float): Calibration factor to apply.

    Returns:
        List[Any]: List of mock FITS image objects.
    """
    ensure_directory_exists(output_path)
    ND_transmission = 10 ** (-OD)  # Calculate transmission from optical density
    # Limit to at most 4 bright stars if more are provided.
    selected_bright_stars = BRIGHT_STARS[:4]
    x_offsets = [-10, 0, 10]
    y_offsets = [-10, 0, 10]

    bright_star_images = []
    for star_name in selected_bright_stars:
        for dy in y_offsets:
            for dx in x_offsets:
                bright_star_flux = compute_expected_band_irradiance(star_name, filter_used)
                total_bright_flux = bright_star_flux * bright_exptime
                attenuated_flux = total_bright_flux * ND_transmission
                fwhm = 3

                flux_image = create_flux_image(
                    attenuated_flux, fwhm, cal_factor, filter_used, star_name,
                    fsm_x=dx, fsm_y=dy, exptime=bright_exptime, filedir=output_path,
                    color_cor=1.0, platescale=21.8, add_gauss_noise=True, noise_scale=1.0,
                    file_save=False
                )
                bright_star_images.append(flux_image)

    return bright_star_images

# ---------------------------------------------------------------------------
# Global test parameters and constants
# ---------------------------------------------------------------------------
DIM_EXPTIME = 10.0
BRIGHT_EXPTIME = 5.0
FILTER_USED = '3C'
INPUT_OD = 2.75
CAL_FACTOR = 0.2
THRESHOLD = 0.1
VISIT_ID = 'PPPPPCCAAASSSOOOVVV'  # Must match the naming string used elsewhere

# ---------------------------------------------------------------------------
# Main workflow function: ND Filter Calibration Workflow
# ---------------------------------------------------------------------------
def nd_calibration_workflow(
    dim_stars_dataset: Dataset,
    bright_stars_dataset: Dataset,
    output_path: Optional[str] = None,
    file_save: bool = False,
    threshold: float = 0.1,
    clean_entry: Optional[Any] = None,
    transformation_matrix_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Derive flux calibration factors from dim stars and compute ND filter calibration products.

    The workflow performs the following steps:
      1. Computes the average calibration factor from the dim stars dataset.
      2. Groups the bright star frames by target and processes each group to extract
         sweet-spot data.
      3. Combines the individual sweet-spot datasets into a single dataset, which is
         saved as an NDFilterSweetSpotDataset calibration product.
      4. Optionally saves the combined sweet-spot dataset to disk.
      5. If a clean image entry and a transformation matrix FITS file are provided, computes an expected OD product,
         which is saved as an NDFilterOD calibration product.
      6. Optionally saves the OD calibration product to disk.

    Parameters:
        dim_stars_dataset (Dataset): Dataset containing dim star images.
        bright_stars_dataset (Dataset): Dataset containing bright star images.
        output_path (str, optional): Directory to save output files.
        file_save (bool): Flag to determine if output files should be written to disk.
        threshold (float): Threshold for flagging OD variations.
        clean_entry (Any, optional): Clean image entry for computing expected OD.
        transformation_matrix_file (str, optional): File path to a FITS file containing the FPAM-to-EXCAM transformation matrix.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'flux_results': Processed data per target.
            - 'combined_sweet_spot_data': Combined sweet-spot dataset.
            - 'overall_avg_od': Overall average optical density.
            - 'expected_od': Computed expected OD (if applicable).
    """
    # Step 1: Compute the average calibration factor from the dim stars.
    cal_factor = compute_avg_calibration_factor(dim_stars_dataset)

    # Step 2: Group bright star frames by target.
    grouped_files = group_by_target(bright_stars_dataset)
    flux_results = {}            # To hold processed results for each target.
    aggregated_data_list = []    # To accumulate sweet-spot data arrays.
    common_metadata = {}         # To store metadata that should be consistent across targets.

    for target, files in grouped_files.items():
        if not files:
            continue

        print(f"Processing target: {target}")
        star_data = process_bright_target(target, files, cal_factor, threshold)
        flux_results[target] = star_data

        # Stack OD, x, and y values to form a sweet-spot dataset for this target.
        target_sweet_spot = np.column_stack((
            star_data['od_values'],
            star_data['x_values'],
            star_data['y_values']
        ))
        aggregated_data_list.append(target_sweet_spot)

        # On the first valid target, record the common metadata.
        if not common_metadata:
            common_metadata = {
                'fpamname': star_data['fpamname'],
                'fpam_h': star_data['fpam_h'],
                'fpam_v': star_data['fpam_v'],
                'filter_name': star_data['filter_name']
            }
        else:
            # Verify that subsequent targets share the same metadata.
            if (common_metadata.get('fpamname') != star_data['fpamname'] or
                common_metadata.get('fpam_h') != star_data['fpam_h'] or
                common_metadata.get('fpam_v') != star_data['fpam_v'] or
                common_metadata.get('filter_name') != star_data['filter_name']):
                raise ValueError("Inconsistent FPAM or filter metadata among bright star observations.")

    # Step 3: Combine all sweet-spot datasets into a single dataset.
    if aggregated_data_list:
        combined_sweet_spot_data = np.vstack(aggregated_data_list)
    else:
        combined_sweet_spot_data = np.empty((0, 3))

    # Compute overall average optical density (OD) from the targets.
    od_list = [
        data.get("average_od")
        for data in flux_results.values()
        if data.get("average_od") is not None
    ]
    overall_avg_od = np.mean(od_list) if od_list else None
    print(f"Overall Average OD across bright targets: {overall_avg_od}")

    # Step 4: Save the combined sweet-spot dataset as an NDFilterSweetSpotDataset product if required.
    if output_path is not None and file_save:
        max_serial = get_max_serial(output_path, VISIT_ID)
        create_nd_sweet_spot_dataset(
            combined_sweet_spot_data,
            common_metadata,
            VISIT_ID,
            max_serial,
            output_path,
            save_to_disk=file_save
        )

    # Step 5: If a clean image entry and a transformation matrix FITS file are provided, compute the expected OD product.
    expected_od = None
    if clean_entry is not None and transformation_matrix_file is not None:
        max_serial_exp = get_max_serial_expected_od(output_path, VISIT_ID)
        transformation_matrix = load_transformation_matrix_from_fits(transformation_matrix_file)
        expected_od_product = create_expected_od_calibration_product(clean_entry, combined_sweet_spot_data, 
                                           common_metadata, transformation_matrix, transformation_matrix_file,
                                           VISIT_ID, max_serial_exp, output_path, save_to_disk=file_save)

        expected_od = expected_od_product.data
        print(f"Expected OD for clean image: {expected_od}")

    return {
        'flux_results': flux_results,
        'combined_sweet_spot_data': combined_sweet_spot_data,
        'overall_avg_od': overall_avg_od,
        'expected_od': expected_od
    }

# ---------------------------------------------------------------------------
# Test functions using pytest
# ---------------------------------------------------------------------------
def test_average_od_within_tolerance():
    """
    Test that the average OD computed for bright stars recovers the input OD within a specified tolerance.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock datasets.
        dim_dataset_files = mock_dim_dataset_files(tmpdir, DIM_EXPTIME, FILTER_USED, CAL_FACTOR)
        bright_dataset_files = mock_bright_dataset_files(tmpdir, BRIGHT_EXPTIME, FILTER_USED, INPUT_OD, CAL_FACTOR)
        dim_dataset = Dataset(dim_dataset_files)
        bright_dataset = Dataset(bright_dataset_files)

        # Execute the calibration workflow.
        results = nd_calibration_workflow(dim_dataset, bright_dataset, output_path=tmpdir, file_save=True, threshold=THRESHOLD)

        # Validate that results are returned.
        assert isinstance(results, dict) and results, "Results should be a non-empty dictionary."

        # For each target, verify that the computed average OD is within tolerance.
        tolerance = 0.5  # Tolerance for OD variation.
        for target, data in results['flux_results'].items():
            avg_od = data['average_od']
            assert abs(avg_od - INPUT_OD) < tolerance, (
                f"Target {target}: computed avg OD {avg_od} deviates more than {tolerance} from input OD {INPUT_OD}"
            )
            od_std = np.std(data['od_values'])
            if od_std < THRESHOLD:
                assert data['flag'] is False, f"Target {target}: low OD variation but flag is set"
            else:
                assert data['flag'] is True, f"Target {target}: high OD variation but flag is not set"


def test_output_filename_convention():
    """
    Test that the output filenames adhere to the expected naming convention.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock datasets.
        dim_dataset_files = mock_dim_dataset_files(tmpdir, DIM_EXPTIME, FILTER_USED, CAL_FACTOR)
        bright_dataset_files = mock_bright_dataset_files(tmpdir, BRIGHT_EXPTIME, FILTER_USED, INPUT_OD, CAL_FACTOR)
        dim_dataset = Dataset(dim_dataset_files)
        bright_dataset = Dataset(bright_dataset_files)

        # Run the workflow to generate output files.
        nd_calibration_workflow(dim_dataset, bright_dataset, output_path=tmpdir, file_save=True, threshold=THRESHOLD)

        # Check that at least one sweet-spot dataset file exists and follows the naming pattern.
        pattern = re.compile(rf"CGI_{VISIT_ID}_(\d{{3}})_FilterBand_{FILTER_USED}_NDF_SWEET\.fits")
        filenames = os.listdir(tmpdir)
        sweet_files = [fn for fn in filenames if pattern.match(fn)]
        assert sweet_files, "No NDFilterSweetSpotDataset files found matching the naming convention."
        for fn in sweet_files:
            match = pattern.match(fn)
            serial = int(match.group(1))
            assert serial >= 1, f"Serial number in filename {fn} is invalid."


def test_nd_filter_calibration_object():
    """
    Test that the ND filter OD calibration product generated is valid and contains the expected header keywords.
    Uses a FPAM-to-EXCAM transformation matrix FITS file and one of the bright star images as the clean entry.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock datasets.
        dim_dataset_files = mock_dim_dataset_files(tmpdir, DIM_EXPTIME, FILTER_USED, CAL_FACTOR)
        bright_dataset_files = mock_bright_dataset_files(tmpdir, BRIGHT_EXPTIME, FILTER_USED, INPUT_OD, CAL_FACTOR)
        dim_dataset = Dataset(dim_dataset_files)
        bright_dataset = Dataset(bright_dataset_files)

        # Use one of the bright star dataset files as the clean entry.
        clean_entry = bright_dataset_files[0]

        # Create a dummy transformation matrix FITS file with a 2x2 identity matrix.
        transformation_matrix_file = os.path.join(tmpdir, "fpam_to_excam.fits")
        dummy_matrix = np.eye(2)  # Use 2x2 if that matches the expected dimension
        hdu = fits.PrimaryHDU(dummy_matrix)
        hdu.writeto(transformation_matrix_file)

        # Call the workflow and pass the file path:
        nd_calibration_workflow(
            dim_dataset,
            bright_dataset,
            output_path=tmpdir,
            file_save=True,
            threshold=THRESHOLD,
            clean_entry=clean_entry,
            transformation_matrix_file=transformation_matrix_file
        )

        # Open one of the generated ND filter calibration FITS files.
        nd_files = [fn for fn in os.listdir(tmpdir) if fn.endswith('_NDF_CAL.fits')]
        assert nd_files, "No NDFilterOD files were generated."
        file_path = os.path.join(tmpdir, nd_files[0])
        with fits.open(file_path) as hdul:
            primary_hdr = hdul[0].header
            ext_hdr = hdul[1].header
            assert primary_hdr.get('SIMPLE') is True, "Primary header missing or SIMPLE keyword is not True."
            assert ext_hdr.get('FPAM_H') is not None, "Extension header missing FPAM_H keyword."

# ---------------------------------------------------------------------------
# Run tests if executed directly.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    test_nd_filter_calibration_object()
    test_output_filename_convention()
    test_average_od_within_tolerance()
    print("All tests passed.")
