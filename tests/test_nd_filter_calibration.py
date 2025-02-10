import os
import re
from typing import List, Dict, Any, Optional

import numpy as np
from astropy.io import fits
import pytest

import corgidrp.nd_filter_calibration as nd_filter_calibration
from corgidrp.data import Dataset
import corgidrp.mocks as mocks

# ---------------------------------------------------------------------------
# Global variables for test star names
# ---------------------------------------------------------------------------
BRIGHT_STARS = ['109 Vir',
                'Vega',
                'Eta Uma',
                'Lam Lep']
DIM_STARS = ['TYC 4433-1800-1',
             'TYC 4205-1677-1',
             'TYC 4212-455-1',
             'TYC 4209-1396-1',
             'TYC 4413-304-1',
             'UCAC3 313-62260',
             'BPS BS 17447-0067',
             'TYC 4424-1286-1',
             'GSC 02581-02323',
             'TYC 4207-219-1']

# ---------------------------------------------------------------------------
# Global constants for saving mocks and calibration products
# ---------------------------------------------------------------------------
DEFAULT_BRIGHT_MOCKS_DIR = "/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks_bright"
DEFAULT_DIM_MOCKS_DIR = "/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks_dim"
DEFAULT_CAL_PRODUCTS_OUTPUT_DIR = "/Users/jmilton/Github/corgidrp/tests/e2e_tests/nd_filter_output"

# ---------------------------------------------------------------------------
# Global test parameters and constants
# ---------------------------------------------------------------------------
DIM_EXPTIME = 10.0
BRIGHT_EXPTIME = 5.0
FWHM = 3
FILTER_USED = '3C'
INPUT_OD = 2.75
CAL_FACTOR = 0.8
OD_RASTER_THRESHOLD = 0.1
OD_TEST_TOLERANCE = 0.2
VISIT_ID = 'PPPPPCCAAASSSOOOVVV'  # Update to pull from VISITID when available in L1s
FILESAVE = True
ADD_BACKGROUND = False
PHOT_METHOD = "Aperture"
FLUX_OR_IRR = 'irr'

if PHOT_METHOD == "Aperture":
    PHOT_ARGS = {
        "encircled_radius": 7,       # Custom aperture radius
        "frac_enc_energy": 1,        # Custom fraction of encircled energy
        "method": "subpixel",        # Method for handling subpixel sampling
        "subpixels": 10,             # Increase subpixel resolution
        "background_sub": True,      # Enable background subtraction
        "r_in": 5,                   # Custom inner annulus radius for background estimation
        "r_out": 10,                 # Custom outer annulus radius for background estimation
        "centering_method": "xy"     # Method of determining for star position ('xy' or 'wcs')
    }
elif PHOT_METHOD == "PSF":
    PHOT_ARGS = {
        "fwhm": 3,                  # Expected full width half maximum.
        "fit_shape": None,          # Fitting region shape.
        "background_sub": True,     # Enable background subtraction
        "r_in": 5,                  # Custom inner annulus radius for background estimation
        "r_out": 10,                # Custom outer annulus radius for background estimation
        "centering_method": 'xy'    # Method of determining for star position ('xy' or 'wcs')
    }

# ---------------------------------------------------------------------------
# Set-up and helper functions
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
        data = hdul[0].data
        if data is None:
            data = hdul[1].data
        return np.array(data)


def mock_dim_dataset_files(dim_exptime: float, filter_used: str, cal_factor: float, 
                           save_mocks: bool, output_path: Optional[str] = None,) -> List[Any]:
    """
    Create mock FITS files for dim stars (calibration references) using
    create_flux_image from Mocks.py.
    
    Parameters:
        dim_exptime (float): The exposure time (in seconds) for the mock dim star observations.
        filter_used (str): The CFAM filter used.
        cal_factor (float): A calibration factor used to scale the computed flux.
        save_mocks (bool): If True, the generated mock FITS files are saved to disk.
        output_path (Optional[str], optional): The directory where the generated mock FITS files 
            will be saved. If None:
                - When `save_mocks` is True, it defaults to `DEFAULT_DIM_MOCKS_DIR`.
                - When `save_mocks` is False, it defaults to the current working directory.
            Defaults to None.

    Returns:
        List[Any]: A list containing the generated flux images for each dim star. The precise type
            of each element depends on the implementation of `create_flux_image`.

    """
    if save_mocks:
        output_path = output_path or DEFAULT_DIM_MOCKS_DIR
    else:
        output_path = output_path or os.getcwd()
    os.makedirs(output_path, exist_ok=True)

    dim_star_images = []
    for star_name in DIM_STARS:
        star_flux = nd_filter_calibration.compute_expected_band_irradiance(star_name, filter_used)
        total_dim_flux = star_flux * dim_exptime
        flux_image = mocks.create_flux_image(
            total_dim_flux, FWHM, cal_factor, filter_used, star_name,
            fsm_x=0, fsm_y=0, exptime=dim_exptime, filedir=output_path,
            color_cor=1.0, platescale=21.8, add_gauss_noise=ADD_BACKGROUND, 
            noise_scale=1.0, file_save=True
        )
        dim_star_images.append(flux_image)
    return dim_star_images

def mock_bright_dataset_files(bright_exptime: float, filter_used: str, OD: float, 
                              cal_factor: float, save_mocks: bool, 
                              output_path: Optional[str] = None) -> List[Any]:
    """
    Create mock FITS files for bright stars observed with an ND filter using
    create_flux_image from Mocks.py.
    
    If save_mocks is True and no output_path is provided, output_path is set
    to DEFAULT_BRIGHT_MOCKS_DIR.

    Parameters:
        bright_exptime (float): The exposure time (in seconds) for the bright star observations.
        filter_used (str): The CFAM filter used..
        OD (float): The optical density of the ND filter. The ND transmission = 10^(-OD).
        cal_factor (float): Calibration factor used to scale the computed flux.
        save_mocks (bool): If True, the generated FITS files are saved to disk.
        output_path (Optional[str], optional): The directory where the generated FITS files will
            be saved. If None:
                - When `save_mocks` is True, it defaults to `DEFAULT_BRIGHT_MOCKS_DIR`.
                - When `save_mocks` is False, it defaults to the current working directory.
            Defaults to None.

    Returns:
        List[Any]: A list of the generated flux images for each bright star at various offsets.
            The exact type of each element depends on the implementation of `create_flux_image`.

    """
    if save_mocks:
        output_path = output_path or DEFAULT_BRIGHT_MOCKS_DIR
    else:
        output_path = output_path or os.getcwd()
    os.makedirs(output_path, exist_ok=True)

    ND_transmission = 10 ** (-OD)
    selected_bright_stars = BRIGHT_STARS
    x_offsets = [-10, 0, 10]
    y_offsets = [-10, 0, 10]

    bright_star_images = []
    for star_name in selected_bright_stars:
        for dy in y_offsets:
            for dx in x_offsets:
                bright_star_flux = nd_filter_calibration.compute_expected_band_irradiance(star_name, filter_used)
                total_bright_flux = bright_star_flux * bright_exptime
                attenuated_flux = total_bright_flux * ND_transmission
                flux_image = mocks.create_flux_image(
                    attenuated_flux, FWHM, cal_factor, filter_used, star_name,
                    dx, dy, bright_exptime, output_path,
                    color_cor=1.0, platescale=21.8, add_gauss_noise=ADD_BACKGROUND, 
                    noise_scale=1.0, file_save=True
                )
                bright_star_images.append(flux_image)
    return bright_star_images


# ---------------------------------------------------------------------------
# Main workflow function
# ---------------------------------------------------------------------------
def nd_calibration_workflow(
    dim_stars_dataset: Dataset,
    bright_stars_dataset: Dataset,
    output_path: Optional[str] = DEFAULT_CAL_PRODUCTS_OUTPUT_DIR,
    file_save: bool = FILESAVE,
    threshold: float = OD_RASTER_THRESHOLD,
    clean_entry: Optional[Any] = None,
    transformation_matrix_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Derive flux calibration factors from dim stars and compute ND filter calibration products.

    The workflow performs the following steps:
      1. Computes the average calibration factor from the dim stars dataset.
      2. Groups the bright star frames by target and processes each group to calculate
         sweet-spot data at each.
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
    print("Computing calibration factor with dim stars")
    cal_factor = nd_filter_calibration.compute_avg_calibration_factor(dim_stars_dataset, PHOT_METHOD, FLUX_OR_IRR, 
                                                calibrate_kwargs=None)
    print("Computed calibration factor:", cal_factor)

    # Step 2: Group bright star frames by target and compute flux at each raster.
    grouped_files = nd_filter_calibration.group_by_target(bright_stars_dataset)
    flux_results = {}
    aggregated_data_list = []
    common_metadata = {}

    for target, files in grouped_files.items():
        if not files:
            continue

        print(f"Processing target: {target}")
        star_data = nd_filter_calibration.process_bright_target(target, files, cal_factor, threshold, PHOT_METHOD, PHOT_ARGS)
        flux_results[target] = star_data

        target_sweet_spot = np.column_stack((
            star_data['od_values'],
            star_data['x_values'],
            star_data['y_values']
        ))
        aggregated_data_list.append(target_sweet_spot)

        if not common_metadata:
            common_metadata = {
                'FPAMNAME': star_data['FPAMNAME'],
                'FPAM_H': star_data['FPAM_H'],
                'FPAM_V': star_data['FPAM_V'],
                'CFAMNAME': star_data['CFAMNAME']
            }
        else:
            # Header info for all stars should match each other for FPAM and 
            # CFAM filters, FPAM_H and FPAM_V can be within some tolerance
            if (common_metadata.get('FPAMNAME') != star_data['FPAMNAME'] or
                abs(common_metadata.get('FPAM_H') - star_data['FPAM_H']) >= 20 or
                abs(common_metadata.get('FPAM_V') - star_data['FPAM_V']) >= 20 or
                common_metadata.get('CFAMNAME') != star_data['CFAMNAME']):
                raise ValueError("Inconsistent FPAM or filter metadata among bright star observations.")

            
    # Step 3: Combine all sweet-spot datasets into a single dataset, and
    # Step 4: Optionally saves the sweet spot dataset calibration product to disk
    combined_sweet_spot_data = np.vstack(aggregated_data_list) if aggregated_data_list else np.empty((0, 3))
    od_list = [data.get("average_od") for data in flux_results.values() if data.get("average_od") is not None]
    overall_avg_od = np.mean(od_list) if od_list else None
    print(f"Average calculated OD across bright targets: {overall_avg_od}")
    max_serial = nd_filter_calibration.get_max_serial(output_path, VISIT_ID)
    sweet_spot_cal = nd_filter_calibration.create_nd_sweet_spot_dataset(
        combined_sweet_spot_data,
        common_metadata,
        VISIT_ID,
        max_serial,
        output_path,
        save_to_disk=file_save
        )

    # Step 5: If a clean image entry and a transformation matrix FITS file are provided, computes 
    # an expected OD product, which is saved as an NDFilterOD calibration product, and
    # Step 6: Optionally saves the NDFilterOD calibration product to disk
    expected_od = None
    if clean_entry is not None and transformation_matrix_file is not None:
        max_serial_exp = nd_filter_calibration.get_max_serial_expected_od(output_path, VISIT_ID)
        transformation_matrix = load_transformation_matrix_from_fits(transformation_matrix_file)
        expected_od_product = nd_filter_calibration.create_expected_od_calibration_product(
            clean_entry,
            combined_sweet_spot_data,
            common_metadata,
            transformation_matrix,
            transformation_matrix_file,
            VISIT_ID,
            max_serial_exp,
            output_path,
            save_to_disk=file_save
        )
        expected_od = expected_od_product.data

    return {
        'flux_results': flux_results,
        'combined_sweet_spot_data': combined_sweet_spot_data,
        'overall_avg_od': overall_avg_od,
        'expected_od': expected_od
    }


# ---------------------------------------------------------------------------
# Test functions using pytest
# ---------------------------------------------------------------------------
def test_nd_filter_calibration_object(dim_dataset, bright_dataset):
    """
    Test that the ND filter OD calibration product generated is valid and contains the expected header keywords.
    Uses a FPAM-to-EXCAM transformation matrix FITS file and one of the bright star images as the clean entry.

    Parameters:
        dim_dataset (Dataset): dataset object of dim star images
        bright_dataset (Dataset): dataset object of bright star images
    """
    print("**Testing ND filter OD calibration product generation and expected headers**")

    clean_entry = bright_dataset_files[0]

    transformation_matrix_file = os.path.join(DEFAULT_CAL_PRODUCTS_OUTPUT_DIR, "fpam_to_excam.fits")
    dummy_matrix = np.eye(2)
    hdu = fits.PrimaryHDU(dummy_matrix)
    hdu.writeto(transformation_matrix_file, overwrite=True)

    nd_calibration_workflow(dim_dataset, bright_dataset,
                            output_path=DEFAULT_CAL_PRODUCTS_OUTPUT_DIR,
                            file_save=FILESAVE, threshold=OD_RASTER_THRESHOLD,
                            clean_entry=clean_entry,
                            transformation_matrix_file=transformation_matrix_file)

    nd_files = [fn for fn in os.listdir(DEFAULT_CAL_PRODUCTS_OUTPUT_DIR) if fn.endswith('_NDF_ExpectedOD.fits')]
    assert nd_files, "No NDFilterOD files were generated."
    file_path = os.path.join(DEFAULT_CAL_PRODUCTS_OUTPUT_DIR, nd_files[0])
    with fits.open(file_path) as hdul:
        primary_hdr = hdul[0].header
        ext_hdr = hdul[1].header
        assert primary_hdr.get('SIMPLE') is True, "Primary header missing or SIMPLE keyword is not True."
        assert ext_hdr.get('FPAMNAME') is not None, "Extension header missing CFAMNAME keyword."
        assert ext_hdr.get('FPAM_H') is not None, "Extension header missing FPAM_H keyword."
        assert ext_hdr.get('FPAM_V') is not None, "Extension header missing FPAM_V keyword."
        assert ext_hdr.get('CFAMNAME') is not None, "Extension header missing CFAMNAME keyword."


def test_output_filename_convention(dim_dataset, bright_dataset):
    """
    Test that the output filenames adhere to the expected naming convention.

    Parameters:
        dim_dataset (Dataset): dataset object of dim star images
        bright_dataset (Dataset): dataset object of bright star images
    """
    print("**Testing calibration product output filename naming conventions**")

    nd_calibration_workflow(dim_dataset, bright_dataset,
                            output_path=DEFAULT_CAL_PRODUCTS_OUTPUT_DIR,
                            file_save=True, threshold=OD_RASTER_THRESHOLD)

    pattern = re.compile(rf"CGI_{VISIT_ID}_(\d{{3}})_FilterBand_{FILTER_USED}_NDF_SWEETSPOT\.fits")
    filenames = os.listdir(DEFAULT_CAL_PRODUCTS_OUTPUT_DIR)
    sweet_spot_files = [fn for fn in filenames if pattern.match(fn)]
    assert sweet_spot_files, "No NDFilterSweetSpotDataset files found matching the naming convention."
    for fn in sweet_spot_files:
        match = pattern.match(fn)
        serial = int(match.group(1))
        assert serial >= 1, f"Serial number in filename {fn} is invalid."


def test_average_od_within_tolerance(dim_dataset, bright_dataset):
    """
    Test that the average OD computed for bright stars recovers the input OD within a specified tolerance.

    Parameters:
        dim_dataset (Dataset): dataset object of dim star images
        bright_dataset (Dataset): dataset object of bright star images
    """
    print("**Testing that computed OD matches input OD to within tolerance**")

    results = nd_calibration_workflow(dim_dataset, bright_dataset,
                                      output_path=DEFAULT_CAL_PRODUCTS_OUTPUT_DIR,
                                      file_save=True, threshold=OD_RASTER_THRESHOLD)
    assert isinstance(results, dict) and results, "Results should be a non-empty dictionary."

    for target, data in results['flux_results'].items():
        avg_od = data['average_od']
        assert abs(avg_od - INPUT_OD) < OD_TEST_TOLERANCE, (
            f"Target {target}: computed avg OD {avg_od} deviates more than {OD_TEST_TOLERANCE} from input OD {INPUT_OD}"
        )
        od_std = np.std(data['od_values'])
        if od_std < OD_RASTER_THRESHOLD:
            assert data['flag'] == False, f"Target {target}: low OD variation but flag is set"
        else:
            assert data['flag'] == True, f"Target {target}: high OD variation but flag is not set"


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    dim_dataset_files = mock_dim_dataset_files(dim_exptime=DIM_EXPTIME,
                                            filter_used=FILTER_USED,
                                            cal_factor=CAL_FACTOR,
                                            save_mocks=FILESAVE,
                                            output_path=DEFAULT_DIM_MOCKS_DIR)
    bright_dataset_files = mock_bright_dataset_files(bright_exptime=BRIGHT_EXPTIME,
                                                     filter_used=FILTER_USED,
                                                     OD=INPUT_OD,
                                                     cal_factor=CAL_FACTOR,
                                                     save_mocks=FILESAVE,
                                                     output_path=DEFAULT_BRIGHT_MOCKS_DIR)
    
    dim_dataset = Dataset(dim_dataset_files)
    bright_dataset = Dataset(bright_dataset_files)

    # Uncomment the tests you want to run:
    test_nd_filter_calibration_object(dim_dataset, bright_dataset)
    test_output_filename_convention(dim_dataset, bright_dataset)
    test_average_od_within_tolerance(dim_dataset, bright_dataset)
    print("All tests passed.")
