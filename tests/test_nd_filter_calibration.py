import os
import glob
import re

import numpy as np
from astropy.io import fits
import pytest

import corgidrp.nd_filter_calibration as nd_filter_calibration
import corgidrp.l2b_to_l3 as l2b_tol3
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
PHOT_METHOD = "PSF"
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
        "centering_method": "xy",    # Method of determining for star position ('xy' or 'wcs')
        "centroid_roi_radius": 5     # Half-size of box around the peak in pixels based on desired λ/D.
    }
elif PHOT_METHOD == "PSF":
    PHOT_ARGS = {
        "fwhm": 3,                  # Expected full width half maximum.
        "fit_shape": None,          # Fitting region shape.
        "background_sub": True,     # Enable background subtraction
        "r_in": 5,                  # Custom inner annulus radius for background estimation
        "r_out": 10,                # Custom outer annulus radius for background estimation
        "centering_method": 'xy',   # Method of determining for star position ('xy' or 'wcs')
        "centroid_roi_radius": 5    # Half-size of box around the peak in pixels based on desired λ/D.
    }


# ---------------------------------------------------------------------------
# Set up mocks for testing
# ---------------------------------------------------------------------------

def mock_dim_dataset_files(dim_exptime, filter_used, cal_factor, save_mocks,
                           output_path=None):
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


def mock_bright_dataset_files(bright_exptime, filter_used, OD, cal_factor,
                              save_mocks, output_path=None):
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
                bright_star_flux = nd_filter_calibration.compute_expected_band_irradiance(star_name,
                                                                                        filter_used)
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

def mock_clean_entry(bright_dataset):
    """
    Generate a mock clean entry from the given bright dataset.

    Parameters:
        bright_dataset (list or array): The dataset containing bright entries.

    Returns:
        object: A processed clean entry derived from the first entry in the dataset.
    """
    # do whatever steps are necessary in the pipeline to make this a clean image
    clean_entry = bright_dataset[0]
    return clean_entry


def mock_transformation_matrix(output_dir):
    """
    Create a mock transformation matrix and save it as a FITS file.

    Parameters:
        output_dir (str): The directory where the transformation matrix FITS file 
            will be saved.

    Returns:
        str: The file path of the saved transformation matrix FITS file.
    """
    # eventually find Sergi's transformation matrix product and use it here
    transformation_matrix_file = os.path.join(output_dir, "fpam_to_excam.fits")
    dummy_matrix = np.eye(2)
    hdu = fits.PrimaryHDU(dummy_matrix)
    hdu.writeto(transformation_matrix_file, overwrite=True)
    return transformation_matrix_file


# ---------------------------------------------------------------------------
# Test functions using pytest
# ---------------------------------------------------------------------------
def test_nd_filter_calibration_object(dim_dataset, bright_dataset, output_dir):
    """
    Test that the ND filter OD calibration product generated is valid and contains the expected 
        header keywords.
    Uses a FPAM-to-EXCAM transformation matrix FITS file and one of the bright star images as 
        the clean entry.

    Parameters:
        dim_dataset (Dataset): dataset object of dim star images
        bright_dataset (Dataset): dataset object of bright star images
        output_dir (str): Filepath of the output directory
    """
    print("**Testing ND filter OD calibration product generation and expected headers**")

    clean_entry = mock_clean_entry(bright_dataset)

    transformation_matrix_file = mock_transformation_matrix(output_dir)

    nd_filter_calibration.create_nd_filter_cal(dim_dataset, bright_dataset, output_dir, 
                                               FILESAVE, OD_RASTER_THRESHOLD, clean_entry, 
                                               transformation_matrix_file, PHOT_METHOD, FLUX_OR_IRR, 
                                               PHOT_ARGS)

    nd_files = [fn for fn in os.listdir(output_dir) if fn.endswith('_NDF_SWEETSPOT.fits')]
    assert nd_files, "No NDFilterOD files were generated."
    file_path = os.path.join(output_dir, nd_files[0])
    with fits.open(file_path) as hdul:
        primary_hdr = hdul[0].header
        ext_hdr = hdul[1].header
        assert primary_hdr.get('SIMPLE') is True, "Primary header missing or SIMPLE keyword is not True."
        assert ext_hdr.get('FPAMNAME') is not None, "Extension header missing CFAMNAME keyword."
        assert ext_hdr.get('FPAM_H') is not None, "Extension header missing FPAM_H keyword."
        assert ext_hdr.get('FPAM_V') is not None, "Extension header missing FPAM_V keyword."
        assert ext_hdr.get('CFAMNAME') is not None, "Extension header missing CFAMNAME keyword."


def test_output_filename_convention(dim_dataset, bright_dataset, output_dir):
    """
    Test that the output filenames adhere to the expected naming convention.

    Parameters:
        dim_dataset (Dataset): dataset object of dim star images
        bright_dataset (Dataset): dataset object of bright star images
        output_dir (str): Filepath of the output directory
    """
    print("**Testing calibration product output filename naming conventions**")

    clean_entry = mock_clean_entry(bright_dataset)

    transformation_matrix_file = mock_transformation_matrix(output_dir)

    nd_filter_calibration.create_nd_filter_cal(dim_dataset, bright_dataset, output_dir, 
                                               FILESAVE, OD_RASTER_THRESHOLD, clean_entry, 
                                               transformation_matrix_file, PHOT_METHOD, FLUX_OR_IRR, 
                                               PHOT_ARGS)

    pattern = re.compile(fr"CGI_[A-Za-z0-9]+_(\d{{3}})_FilterBand_{FILTER_USED}_NDF_SWEETSPOT\.fits")
    filenames = os.listdir(DEFAULT_CAL_PRODUCTS_OUTPUT_DIR)
    sweet_spot_files = [fn for fn in filenames if pattern.match(fn)]
    assert sweet_spot_files, "No NDFilterSweetSpotDataset files found matching the naming convention."
    for fn in sweet_spot_files:
        match = pattern.match(fn)
        serial = int(match.group(1))
        assert serial >= 1, f"Serial number in filename {fn} is invalid."


def test_average_od_within_tolerance(dim_dataset, bright_dataset, output_dir):
    """
    Test that the average OD computed for bright stars recovers the input OD within a 
    specified tolerance.

    Parameters:
        dim_dataset (Dataset): dataset object of dim star images
        bright_dataset (Dataset): dataset object of bright star images
        output_dir (str): Filepath of the output directory
    """
    print("**Testing that computed OD matches input OD to within tolerance**")

    clean_entry = mock_clean_entry(bright_dataset)

    transformation_matrix_file = mock_transformation_matrix(output_dir)

    results = nd_filter_calibration.create_nd_filter_cal(dim_dataset, bright_dataset, output_dir, 
                                               FILESAVE, OD_RASTER_THRESHOLD, clean_entry, 
                                               transformation_matrix_file, PHOT_METHOD, FLUX_OR_IRR, 
                                               PHOT_ARGS)
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
def dataset_files_exist(directory):
    """Check if there are already files in the specified directory."""
    return os.path.exists(directory) and bool(glob.glob(os.path.join(directory, "*")))


if __name__ == '__main__':
    DEFAULT_BRIGHT_MOCKS_DIR = "/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks_bright"
    DEFAULT_DIM_MOCKS_DIR = "/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks_dim"

    # Check if mock data already exists
    if dataset_files_exist(DEFAULT_DIM_MOCKS_DIR):
        print(f"Using existing dim dataset from {DEFAULT_DIM_MOCKS_DIR}")
        dim_dataset_files = glob.glob(os.path.join(DEFAULT_DIM_MOCKS_DIR, "*"))
    else:
        print(f"Generating new dim dataset in {DEFAULT_DIM_MOCKS_DIR}")
        dim_dataset_files = mock_dim_dataset_files(dim_exptime=DIM_EXPTIME,
                                                   filter_used=FILTER_USED,
                                                   cal_factor=CAL_FACTOR,
                                                   save_mocks=FILESAVE,
                                                   output_path=DEFAULT_DIM_MOCKS_DIR)

    if dataset_files_exist(DEFAULT_BRIGHT_MOCKS_DIR):
        print(f"Using existing bright dataset from {DEFAULT_BRIGHT_MOCKS_DIR}")
        bright_dataset_files = glob.glob(os.path.join(DEFAULT_BRIGHT_MOCKS_DIR, "*"))
    else:
        print(f"Generating new bright dataset in {DEFAULT_BRIGHT_MOCKS_DIR}")
        bright_dataset_files = mock_bright_dataset_files(bright_exptime=BRIGHT_EXPTIME,
                                                         filter_used=FILTER_USED,
                                                         OD=INPUT_OD,
                                                         cal_factor=CAL_FACTOR,
                                                         save_mocks=FILESAVE,
                                                         output_path=DEFAULT_BRIGHT_MOCKS_DIR)

    # Load datasets
    dim_dataset = Dataset(dim_dataset_files)
    bright_dataset = Dataset(bright_dataset_files)

    # Get input data to the state they are expected to be in prior to running the ND Filter 
    # Calibration step. At a minimum for now, we want to normalize for exposure time and do
    # flat-fielding. When the function to add in the WCS headers is done, we will want to add
    # that in here as well (and remove those steps from mocks.py)
    dim_dataset_l3 = l2b_tol3.divide_by_exptime(dim_dataset)
    bright_dataset = l2b_tol3.divide_by_exptime(bright_dataset)

    # Uncomment the tests you want to run:
    test_nd_filter_calibration_object(dim_dataset, bright_dataset, DEFAULT_CAL_PRODUCTS_OUTPUT_DIR)
    test_output_filename_convention(dim_dataset, bright_dataset, DEFAULT_CAL_PRODUCTS_OUTPUT_DIR)
    test_average_od_within_tolerance(dim_dataset, bright_dataset, DEFAULT_CAL_PRODUCTS_OUTPUT_DIR)

    print("All tests passed.")
