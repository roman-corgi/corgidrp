import os
import glob
from pathlib import Path
import numbers
import numpy as np
from astropy.io import fits
import pytest
import re
import copy
from termcolor import cprint

import corgidrp
import corgidrp.fluxcal as fluxcal
import corgidrp.nd_filter_calibration as nd_filter_calibration
import corgidrp.l2b_to_l3 as l2b_tol3
import corgidrp.data as data 
from corgidrp.data import (Image, Dataset, FluxcalFactor,
    NDFilterSweetSpotDataset, FpamFsamCal)
import corgidrp.mocks as mocks

here = os.path.abspath(os.path.dirname(__file__))

def print_fail():
    cprint(' FAIL ', "black", "on_red")


def print_pass():
    cprint(' PASS ', "black", "on_green")


# ---------------------------------------------------------------------------
# Global variables and constants
# ---------------------------------------------------------------------------
BRIGHT_STARS = ['Vega']
DIM_STARS = ['TYC 4424-1286-1',
             'GSC 02581-02323']

# takes a long time to run with all stars
#BRIGHT_STARS = ['109 Vir', 'Vega', 'Eta Uma', 'Lam Lep']
#DIM_STARS = ['TYC 4433-1800-1', 'TYC 4205-1677-1', 'TYC 4212-455-1', 'TYC 4209-1396-1',
#            'TYC 4413-304-1', 'UCAC3 313-62260', 'BPS BS 17447-0067', 'TYC 4424-1286-1',
#             'GSC 02581-02323', 'TYC 4207-219-1']
calspec_filepath = os.path.join(os.path.dirname(__file__), "test_data", "alpha_lyr_stis_011.fits")

DIM_EXPTIME = 10.0
BRIGHT_EXPTIME = 5.0
FWHM = 3
FILTER_USED = '3C'
INPUT_OD = 2.25
CAL_FACTOR = 1e-7
OD_RASTER_THRESHOLD = 0.1
OD_TEST_TOLERANCE = 0.2
FILESAVE = True
ADD_GAUSS = False
BACKGROUND = 3
PHOT_METHOD = "Aperture"
FLUX_OR_IRR = 'irr'

if PHOT_METHOD == "Aperture":
    PHOT_ARGS = {
        "encircled_radius": 7,
        "frac_enc_energy": 1,
        "method": "subpixel",
        "subpixels": 10,
        "background_sub": True,
        "r_in": 5,
        "r_out": 10,
        "centering_method": "xy",
        "centroid_roi_radius": 5
    }
elif PHOT_METHOD == "Gaussian":
    PHOT_ARGS = {
        "fwhm": 3,
        "fit_shape": None,
        "background_sub": True,
        "r_in": 5,
        "r_out": 10,
        "centering_method": 'xy',
        "centroid_roi_radius": 5
    }


def is_real_positive_scalar(var):
    """
    Checks whether an object is a real positive scalar.

    Parameters:
        var (float): variable to check

    Returns:
        result (bool): Whether the check passes or not.

    """
    result = True
    if not isinstance(var, numbers.Number):
        result = False
    if not np.isrealobj(var):
        result = False
    if var <= 0:
        result = False

    return result


# ---------------------------------------------------------------------------
# Functions to generate mocks
# ---------------------------------------------------------------------------
def mock_dim_dataset_files(dim_exptime, filter_used, cal_factor, save_mocks, output_path=None, 
                           background_val=0, add_gauss_noise_val=False):
    """
    Generate and save mock dim dataset files for specified exposure time and filter.

    Parameters:
        dim_exptime (float): Exposure time for the simulated images.
        filter_used (str): Filter used for the observations.
        cal_factor (float): Calibration factor applied to the images.
        save_mocks (bool): Whether to save the generated mock images.
        output_path (str, optional): Directory path to save the images. Defaults to the current working directory.
        background_val (int, optional): Background value to be added to the images. Defaults to 0.
        add_gauss_noise_val (bool, optional): Whether to add Gaussian noise to the images. Defaults to False.

    Returns:
        list: A list of generated flux images for the dim stars.
    """
    if save_mocks:
        output_path = output_path or os.getcwd()
    else:
        output_path = output_path or os.getcwd()
    os.makedirs(output_path, exist_ok=True)
    dim_star_images = []
    for star_name in DIM_STARS:
        dim_star_flux = nd_filter_calibration.compute_expected_band_irradiance(star_name, filter_used)
        flux_image = mocks.create_flux_image(
            dim_star_flux, FWHM, cal_factor, filter=filter_used, fpamname="HOLE", target_name=star_name,
            fsm_x=0, fsm_y=0, exptime=dim_exptime, filedir=output_path,
            platescale=21.8,
            background=background_val,
            add_gauss_noise=add_gauss_noise_val,
            noise_scale=1.0, file_save=True
        )
        # ND filter calibration requires photoelectron/s units (L3 data)
        flux_image.ext_hdr['BUNIT'] = 'photoelectron/s'
        flux_image.ext_hdr['DATALVL'] = 'L3'
        dim_star_images.append(flux_image)
    return dim_star_images


def mock_bright_dataset_files(bright_exptime, filter_used, OD, cal_factor, save_mocks, output_path=None, 
                              background_val=0, add_gauss_noise_val=False):
    """
    Generate and save mock bright dataset files for specified exposure time and filter.

    Parameters:
        bright_exptime (float): Exposure time for the simulated images.
        filter_used (str): Filter used for the observations.
        OD (float): The OD used for the observations.
        cal_factor (float): Calibration factor applied to the images.
        save_mocks (bool): Whether to save the generated mock images.
        output_path (str, optional): Directory path to save the images. Defaults to the current working directory.
        background_val (int, optional): Background value to be added to the images. Defaults to 0.
        add_gauss_noise_val (bool, optional): Whether to add Gaussian noise to the images. Defaults to False.

    Returns:
        list: A list of generated flux images for the dim stars.
    """
    if save_mocks:
        output_path = output_path or os.getcwd()
    else:
        output_path = output_path or os.getcwd()
    os.makedirs(output_path, exist_ok=True)
    ND_transmission = 10 ** (-OD)
    bright_star_images = []
    for star_name in BRIGHT_STARS:
        bright_star_flux = nd_filter_calibration.compute_expected_band_irradiance(star_name, filter_used)
        attenuated_flux = bright_star_flux * ND_transmission
        for dy in [-10, 0, 10]:
            for dx in [-10, 0, 10]:
                flux_image = mocks.create_flux_image(
                    attenuated_flux, FWHM, cal_factor, filter=filter_used, fpamname="ND225", target_name=star_name,
                    fsm_x=dx, fsm_y=dy, exptime=bright_exptime, filedir=output_path,
                    platescale=21.8,
                    background=background_val,
                    add_gauss_noise=add_gauss_noise_val,
                    noise_scale=1.0, file_save=True
                )
                # ND filter calibration requires photoelectron/s units (L3 data)
                flux_image.ext_hdr['BUNIT'] = 'photoelectron/s'
                flux_image.ext_hdr['DATALVL'] = 'L3'
                bright_star_images.append(flux_image)
    return bright_star_images


def mock_clean_entry(bright_dataset):
    # TO DO: eventually add other processing steps that would produce the 
    # appropriate level input file
    return bright_dataset[0]


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------
# These fixtures generate and cache the mocks (which is time consuming) for the
# entire module
@pytest.fixture(scope="module", params=[(0, False), (BACKGROUND, ADD_GAUSS)])
def bg_settings(request):
    return request.param

@pytest.fixture(scope="module")
def dim_files_cached(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("dim_dataset")
    print(f"Generating cached dim dataset in {tmp_dir}")
    files = mock_dim_dataset_files(DIM_EXPTIME, FILTER_USED, CAL_FACTOR, save_mocks=False, output_path=str(tmp_dir))
    return files

@pytest.fixture(scope="module")
def bright_files_cached(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("bright_dataset")
    print(f"Generating cached bright dataset in {tmp_dir}")
    files = mock_bright_dataset_files(BRIGHT_EXPTIME, FILTER_USED, INPUT_OD, CAL_FACTOR, save_mocks=False, output_path=str(tmp_dir))
    return files

@pytest.fixture(scope="module")
def stars_dataset_cached(bright_files_cached, dim_files_cached):
    combined_files = bright_files_cached + dim_files_cached
    # TO DO: May eventually need to add other processing steps to get the mocks
    # to a representative input state
    return Dataset(combined_files)

@pytest.fixture(scope="module")
def stars_dataset_cached_bright_count(bright_files_cached, dim_files_cached):
    combined_files = bright_files_cached + dim_files_cached
    n_bright = len(bright_files_cached)
    # TO DO: May eventually need to add other processing steps to get the mocks
    # to a representative input state
    return Dataset(combined_files), n_bright

@pytest.fixture(scope="module")
def dim_dir(tmp_path_factory):
    """
    Creates a temporary directory, populates it with the mock dim FITS files,
    then returns the directory path.
    """
    tmp_dir = tmp_path_factory.mktemp("dim_dir_fixture")
    mock_dim_dataset_files(
        DIM_EXPTIME,
        FILTER_USED,
        CAL_FACTOR,
        save_mocks=True,
        output_path=str(tmp_dir),
        background_val=0,
        add_gauss_noise_val=False
    )
    return str(tmp_dir)

@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    print(f"Created temporary output directory: {out}")
    return str(out)

# ---------------------------------------------------------------------------
# Test functions using pytest
# ---------------------------------------------------------------------------
def test_compute_exp_irrad():
    print("**Testing calculation of same irradiance with star name and calspec file**")
    name_irr = nd_filter_calibration.compute_expected_band_irradiance('Vega', '3C')
    file_irr = nd_filter_calibration.compute_expected_band_irradiance(calspec_filepath, '3C')
    assert file_irr == name_irr

def test_nd_filter_calibration_object_with_calspec(bright_files_cached):
    print("**Testing ND filter calibration object generation and expected headers with calspec file input**")
    # don't want the datasets to get overwritten for subsequent tests
    ds_copy = copy.deepcopy(Dataset([bright_files_cached[0], bright_files_cached[1]]))
    ds_copy[1].ext_hdr["FPAMNAME"] = 'OPEN_12'
    results = nd_filter_calibration.create_nd_filter_cal(
        ds_copy, OD_RASTER_THRESHOLD, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS, 
        fluxcal_factor = None, calspec_files = [calspec_filepath])
    
    test_output_dir = os.path.join(os.path.dirname(__file__), "testcalib")
    os.makedirs(test_output_dir, exist_ok=True)
    
    results.save(filedir=test_output_dir)

    nd_files = [fn for fn in os.listdir(test_output_dir) if fn.endswith('_ndf_cal.fits')]
    assert nd_files, "No NDFilterOD files were generated."
    with fits.open(os.path.join(test_output_dir, nd_files[0])) as hdul:
        primary_hdr = hdul[0].header
        ext_hdr = hdul[1].header
        assert primary_hdr.get('SIMPLE') is True, "Primary header missing or SIMPLE not True."
        assert ext_hdr.get('FPAMNAME') is not None, "Missing FPAMNAME keyword."
        assert ext_hdr.get('FPAM_H') is not None, "Missing FPAM_H keyword."
        assert ext_hdr.get('FPAM_V') is not None, "Missing FPAM_V keyword."
        assert ext_hdr.get('CFAMNAME') is not None, "Missing CFAMNAME keyword."

def test_nd_filter_calibration_object(stars_dataset_cached):
    print("**Testing ND filter calibration object generation and expected headers**")
    # don't want the datasets to get overwritten for subsequent tests
    ds_copy = copy.deepcopy(stars_dataset_cached)
    results = nd_filter_calibration.create_nd_filter_cal(
        ds_copy, OD_RASTER_THRESHOLD, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS, 
        fluxcal_factor = None)
    
    test_output_dir = os.path.join(os.path.dirname(__file__), "testcalib")
    os.makedirs(test_output_dir, exist_ok=True)
    
    results.save(filedir=test_output_dir)

    nd_files = [fn for fn in os.listdir(test_output_dir) if fn.endswith('_ndf_cal.fits')]
    assert nd_files, "No NDFilterOD files were generated."
    with fits.open(os.path.join(test_output_dir, nd_files[0])) as hdul:
        primary_hdr = hdul[0].header
        ext_hdr = hdul[1].header
        assert primary_hdr.get('SIMPLE') is True, "Primary header missing or SIMPLE not True."
        assert ext_hdr.get('FPAMNAME') is not None, "Missing FPAMNAME keyword."
        assert ext_hdr.get('FPAM_H') is not None, "Missing FPAM_H keyword."
        assert ext_hdr.get('FPAM_V') is not None, "Missing FPAM_V keyword."
        assert ext_hdr.get('CFAMNAME') is not None, "Missing CFAMNAME keyword."


def test_output_filename_convention(stars_dataset_cached):
    print("**Testing output filename naming conventions**")
    
    # Make a copy of the dataset and retrieve expected values.
    ds_copy = copy.deepcopy(stars_dataset_cached)

    # Create test output directory
    test_output_dir = os.path.join(os.path.dirname(__file__), "testcalib")
    os.makedirs(test_output_dir, exist_ok=True)

    # Construct the expected filename from the last input dataset filename.
    expected_filename = re.sub('_l[0-9].', '_ndf_cal', stars_dataset_cached[-1].filename)
    full_expected_path = os.path.join(test_output_dir, expected_filename)

    # Create the calibration product
    results = nd_filter_calibration.create_nd_filter_cal(
        ds_copy, OD_RASTER_THRESHOLD, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS,
        fluxcal_factor=None
    )
    results.save(filedir=test_output_dir)
    
    assert os.path.exists(full_expected_path), (
        f"Expected file {expected_filename} not found in {test_output_dir}."
    )
    print("The nd_filter_calibration product file exists and meets the expected naming convention.")


def test_average_od_within_tolerance(stars_dataset_cached_bright_count):
    print("**Testing computed OD within tolerance**")
    ds_copy, n_bright = copy.deepcopy(stars_dataset_cached_bright_count)
    results = nd_filter_calibration.create_nd_filter_cal(
        ds_copy, OD_RASTER_THRESHOLD, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS, 
        fluxcal_factor = None)
    ods = results.data
    avg_od = np.mean(ods[:, 0])
    std_od = np.std(ods[:,0])
    results_hdr = results.ext_hdr
    od_flag = results_hdr.get('ODFLAG')

    assert abs(avg_od - INPUT_OD) < OD_TEST_TOLERANCE, (
        f"Avg OD {avg_od} deviates more than {OD_TEST_TOLERANCE}"
    )
    if std_od < OD_RASTER_THRESHOLD:
        assert not od_flag, f"Low OD variation but flag is set"

    else:
        assert od_flag, f"High OD variation but flag is not set"

    output_vec = ods[:, 0]
    input_vec = INPUT_OD * np.ones_like(output_vec)
    test_result = np.all(np.isclose(input_vec, output_vec, atol=OD_RASTER_THRESHOLD))

    test_result = ((n_bright, 3) == ods.shape)
    print(f'Shape test: has dimensions Mx3, where number of bright frames = M = {n_bright}: ', end='')
    print_pass() if test_result else print_fail()

    # Print out the results
    print('OD values are correct: %.2f +/- %.2f: ' % (INPUT_OD, OD_RASTER_THRESHOLD), end='')
    print_pass() if test_result else print_fail()

    test_result_x = np.all(is_real_positive_scalar(val) for val in ods[:, 1])
    test_result_y = np.all(is_real_positive_scalar(val) for val in ods[:, 2])

    print('All PSF x values are real positive scalars: ', end='')
    print_pass() if test_result_x else print_fail()

    print('All PSF y values are real positive scalars: ', end='')
    print_pass() if test_result_y else print_fail()

    avg_x_expected = 512.00
    avg_y_expected = 512.00
    avg_x = np.mean(ods[:, 1])
    avg_y = np.mean(ods[:, 2])
    test_result_x = np.isclose(avg_x_expected, avg_x, atol=0.01)
    test_result_y = np.isclose(avg_y_expected, avg_y, atol=0.01)
    print('Mean x value is %.2f pixel: ' % avg_x_expected, end='')
    print_pass() if test_result_x else print_fail()
    print('Mean y value is %.2f pixels: ' % avg_x_expected, end='')
    print_pass() if test_result_y else print_fail()


@pytest.mark.parametrize("phot_method", ["Aperture", "Gaussian"])
def test_nd_filter_calibration_phot_methods(stars_dataset_cached, phot_method):
    if phot_method == "Aperture":
        phot_args = {
            "encircled_radius": 7,
            "frac_enc_energy": 1,
            "method": "subpixel",
            "subpixels": 10,
            "background_sub": True,
            "r_in": 5,
            "r_out": 10,
            "centering_method": "xy",
            "centroid_roi_radius": 5
        }
    else:
        phot_args = {
            "fwhm": 3,
            "fit_shape": None,
            "background_sub": True,
            "r_in": 5,
            "r_out": 10,
            "centering_method": "xy",
            "centroid_roi_radius": 5
        }
    print(f"**Testing ND calibration with photometry method: {phot_method}**")
    ds_copy = copy.deepcopy(stars_dataset_cached)
    results = nd_filter_calibration.create_nd_filter_cal(
        ds_copy, OD_RASTER_THRESHOLD, phot_method, FLUX_OR_IRR, phot_args, 
        fluxcal_factor = None)
    ods = results.data
    avg_od = np.mean(ods[:, 0])
    assert abs(avg_od - INPUT_OD) < OD_TEST_TOLERANCE, (
        f"Method {phot_method}: OD mismatch"
    )


@pytest.mark.parametrize("test_od", [1.0, 3.0])
def test_multiple_nd_levels(dim_dir, output_dir, test_od):
    print(f"**Testing multiple ND levels with input OD = {test_od}**")
    bright_mocks_dir = os.path.join(output_dir, f"mock_OD{test_od}")
    bright_images = mock_bright_dataset_files(
        BRIGHT_EXPTIME, FILTER_USED, test_od, CAL_FACTOR, save_mocks=True, output_path=bright_mocks_dir
    )

    dim_filepaths = glob.glob(os.path.join(dim_dir, "*")) # use cached dim images
    dim_images = [Image(path) for path in dim_filepaths]
    # Update BUNIT for ND filter calibration requirements
    for img in dim_images:
        img.ext_hdr['BUNIT'] = 'photoelectron/s'
        img.ext_hdr['DATALVL'] = 'L3'
    combined_files = bright_images + dim_images
    combined_dataset = Dataset(combined_files)

    results = nd_filter_calibration.create_nd_filter_cal(
        combined_dataset, OD_RASTER_THRESHOLD, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS, 
        fluxcal_factor = None)
    ods = results.data
    avg_od = np.mean(ods[:, 0])
    assert abs(avg_od - test_od) < 0.2, (
        f"test_od={test_od}, got {avg_od} for target."
    )

@pytest.mark.parametrize("phot_method", ["Aperture", "Gaussian"])
def test_nd_filter_calibration_with_fluxcal(dim_dir, stars_dataset_cached, phot_method):
    """
    1) Takes a star image from stars_dataset_cached
    2) Derives a flux calibration factor using calibrate_fluxcal_aper or calibrate_fluxcal_gauss2d
    3) Passes that flux calibration factor to create_nd_filter_cal
    4) Checks the resulting ND filter calibration to ensure it ran as expected
    """
    print(f"**Testing with FluxcalFactor input using method {phot_method}**")
    # Pick images without the ND filter in
    dim_filepaths = glob.glob(os.path.join(dim_dir, "*.fits"))
    assert len(dim_filepaths) > 0, f"No FITS files found in {dim_dir}"
    
    dim_images = [Image(path) for path in dim_filepaths]
    # Update BUNIT for ND filter calibration requirements
    for img in dim_images:
        img.ext_hdr['BUNIT'] = 'photoelectron/s'
        img.ext_hdr['DATALVL'] = 'L3'

    # Convert list of Image objects into a Dataset
    dim_dataset = Dataset(dim_images)

    # 1) Generate a flux calibration object from the single image
    if phot_method == "Aperture":
        phot_kwargs = {
            "encircled_radius": 7,
            "frac_enc_energy": 1.0,
            "method": "subpixel",
            "subpixels": 10,
            "background_sub": True,
            "r_in": 5,
            "r_out": 10,
            "centering_method": "xy",
            "centroid_roi_radius": 5
        }
        fluxcal_obj = fluxcal.calibrate_fluxcal_aper(
            dim_dataset, 
            flux_or_irr="irr", 
            phot_kwargs=phot_kwargs
        )
    else:
        phot_kwargs = {
            "fwhm": 3,
            "fit_shape": None,
            "background_sub": True,
            "r_in": 5,
            "r_out": 10,
            "centering_method": "xy",
            "centroid_roi_radius": 5
        }
        fluxcal_obj = fluxcal.calibrate_fluxcal_gauss2d(
            dim_dataset, 
            flux_or_irr="irr", 
            phot_kwargs=phot_kwargs
        )

    results = nd_filter_calibration.create_nd_filter_cal(
        stars_dataset_cached,
        OD_RASTER_THRESHOLD,
        PHOT_METHOD,
        FLUX_OR_IRR,
        PHOT_ARGS,
        fluxcal_factor=fluxcal_obj
    )

    # 3) Check that the calibration worked
    ods = results.data
    assert ods.shape[0] > 0, "No data returned from ND calibration."
    avg_od = np.mean(ods[:, 0])

    # Check nominal OD
    assert abs(avg_od - INPUT_OD) < 0.3, (
        f"Expected OD near {INPUT_OD}, got {avg_od}"
    )

    print(f"ND filter calibration with fluxcal completed. OD={avg_od} (mean).")


@pytest.mark.parametrize("aper_radius", [5, 10])
def test_aperture_radius_sensitivity(stars_dataset_cached, aper_radius):
    print(f"**Testing aperture radius sensitivity: radius = {aper_radius}**")
    phot_args = {
        "encircled_radius": aper_radius,
        "frac_enc_energy": 1,
        "method": "subpixel",
        "subpixels": 5,
        "background_sub": True,
        "r_in": 5,
        "r_out": 10,
        "centering_method": "xy",
        "centroid_roi_radius": 5
    }
    ds_copy = copy.deepcopy(stars_dataset_cached)
    results = nd_filter_calibration.create_nd_filter_cal(
        ds_copy, OD_RASTER_THRESHOLD, "Aperture", "irr", phot_args, 
        fluxcal_factor = None)
    ods = results.data
    avg_od = np.mean(ods[:, 0])
    assert abs(avg_od - INPUT_OD) < 0.3, (
        f"AperRadius={aper_radius}: OD mismatch for target."
    )


def test_background_effect(tmp_path):
    """
    Generate two sets of mocks (one without background, one with background)
    and compare the calibration results. We expect that the overall OD values
    remain similar while the error increases when background is added.
    """
    # Create dim star mocks for two modes.
    dim_dir_no = tmp_path / "dim_no"
    dim_dir_bg = tmp_path / "dim_bg"
    dim_dir_no.mkdir(exist_ok=True)
    dim_dir_bg.mkdir(exist_ok=True)
    dim_files_no = mock_dim_dataset_files(DIM_EXPTIME, FILTER_USED, CAL_FACTOR, save_mocks=False,
                                      output_path=str(dim_dir_no), background_val=0, add_gauss_noise_val=False)
    dim_files_bg = mock_dim_dataset_files(DIM_EXPTIME, FILTER_USED, CAL_FACTOR, save_mocks=False,
                                      output_path=str(dim_dir_bg), background_val=BACKGROUND, add_gauss_noise_val=ADD_GAUSS)

    # Create bright star mocks for two modes.
    bright_dir_no = tmp_path / "bright_no"
    bright_dir_bg = tmp_path / "bright_bg"
    bright_dir_no.mkdir(exist_ok=True)
    bright_dir_bg.mkdir(exist_ok=True)
    bright_files_no = mock_bright_dataset_files(BRIGHT_EXPTIME, FILTER_USED, INPUT_OD, CAL_FACTOR, save_mocks=False,
                                                output_path=str(bright_dir_no), background_val=0, add_gauss_noise_val=False)
    bright_files_bg = mock_bright_dataset_files(BRIGHT_EXPTIME, FILTER_USED, INPUT_OD, CAL_FACTOR, save_mocks=False,
                                                output_path=str(bright_dir_bg), background_val=BACKGROUND, add_gauss_noise_val=ADD_GAUSS)
    
    combined_no = dim_files_no + bright_files_no
    combined_bg = dim_files_bg + bright_files_bg
    ds_no = Dataset(combined_no)
    ds_bg = Dataset(combined_bg)
    
    output_directory = str(tmp_path / "output")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    results_no = nd_filter_calibration.create_nd_filter_cal(
        ds_no, OD_RASTER_THRESHOLD, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS, 
        fluxcal_factor = None)
    results_bg = nd_filter_calibration.create_nd_filter_cal(
        ds_bg, OD_RASTER_THRESHOLD, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS,
        fluxcal_factor = None
    )

    ods_no = results_no.data
    avg_od_no = np.mean(ods_no[:, 0])
    stdev_od_no = np.std(ods_no[:,0])
    ods_bg = results_bg.data
    avg_od_bg = np.mean(ods_bg[:, 0])
    stdev_od_bg = np.std(ods_bg[:,0])
    
    print(f"Avg OD no-bg = {avg_od_no}, stdev no-bg = {stdev_od_no}; Avg OD bg = {avg_od_bg}, stdev bg = {stdev_od_bg}")
    # The overall OD should be similar, within a small tolerance.
    assert abs(avg_od_no - avg_od_bg) < 0.1, f"OD should not differ drastically between background subtraction and no background subtraction modes."


def test_calculate_od_at_new_location(output_dir):
    """
    Test calculate_od_at_new_location with:
      1) A real NDFilterSweetSpotDataset containing Nx3 data = [OD, x, y]
      2) A mock clean_frame_entry with a star centroid at a known location
      3) Known FPAM offsets in headers
      4) An identity transformation matrix file
    """

    # Create a small Nx3 sweet spot array, each row is [OD, x, y].
    # Interpolate at the center (5,5).
    # The OD values at corners are 2.0, 3.0, 4.0, 5.0 => a bilinear interpolation at (5,5) => 3.5.
    sweetspot_data = np.array([
        [2.0,  0.0,  0.0],   # OD=2.0 at (x=0,y=0)
        [3.0,  0.0, 10.0],   # OD=3.0 at (x=0,y=10)
        [4.0, 10.0,  0.0],   # OD=4.0 at (x=10,y=0)
        [5.0, 10.0, 10.0]    # OD=5.0 at (x=10,y=10)
    ], dtype=float)

    # Create a fake input dataset to set the filename
    input_prihdr, input_exthdr, errhdr, dqhdr, biashdr = mocks.create_default_L2b_headers()
    fake_input_image = Image(sweetspot_data, pri_hdr=input_prihdr, ext_hdr=input_exthdr)
    fake_input_image.filename = f"cgi_{input_prihdr['VISITID']}_{data.format_ftimeutc(input_exthdr['FTIMEUTC'])}_l2b.fits".replace(":", ".")
    fake_input_dataset = Dataset(frames_or_filepaths=[fake_input_image, fake_input_image])

    # Build the NDFilterSweetSpotDataset
    ndcal_prihdr, ndcal_exthdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ndcal_exthdr["FPAM_H"] = 0.0
    ndcal_exthdr["FPAM_V"] = 0.0
    nd_sweetspot_dataset = NDFilterSweetSpotDataset(data_or_filepath=sweetspot_data, pri_hdr=ndcal_prihdr, ext_hdr=ndcal_exthdr,
                                                    input_dataset=fake_input_dataset)
 
    # Make a 5x5 mock 'clean_frame_entry' with a star at (2,2) => centroid (2,2)
    # Shift it by (3,3) => final location (5,5).
    clean_image_data = np.zeros((5, 5), dtype=float)
    clean_image_data[2, 2] = 100.0  # star pixel
    cframe_prihdr, cframe_exthdr, errhdr, dqhdr, biashdr= mocks.create_default_L2b_headers()
    # Choosing some values that will help predict the expected value of the OD
    # when using the bilinear OD interpolation in nd_filter_calibration.interpolate_od()
    # These values ensure that the shift in EXCAM pixels is (3,3)
    cframe_exthdr["FPAM_H"] = -24.42
    cframe_exthdr["FPAM_V"] = 24.42
    clean_frame_entry = Image(data_or_filepath=clean_image_data, pri_hdr=cframe_prihdr, 
                              ext_hdr=cframe_exthdr)

    # Default FPAM/FSAM transformations (use mock instead of loading from file which
    # seems to be inconsistent)
    fpamfsamcal = mocks.create_mock_fpamfsam_cal(save_file=False)    

    # Call the function under test
    interpolated_od = nd_filter_calibration.calculate_od_at_new_location(
        clean_frame_entry=clean_frame_entry,
        fpamfsamcal=fpamfsamcal,
        ndsweetspot_dataset=nd_sweetspot_dataset
    )

    # Expect the final location = (2+3, 2+3) = (5,5).
    fpam2excam_matrix = fits.getdata(os.path.join(here, 'test_data',
        'fpam_to_excam_modelbased.fits'))
    # Check final position is (5,5)
    final_excam_pos = (np.array([2,2]) + fpam2excam_matrix @
        np.array([cframe_exthdr["FPAM_H"],cframe_exthdr["FPAM_V"]]))
    # Single precision because the FPAM_H/V values were set to be close to
    # produce a change of 3 EXCAM pixels within single precision
    assert np.all(np.abs(final_excam_pos - np.array([5,5])) < 1e-7)

    # Bilinear interpolation of corners: (2,3,4,5) at center => 3.5  
    expected_value = 3.5

    atol_nd = 1e-6
    test_result_od_accuracy = abs(interpolated_od - expected_value) < atol_nd
    print(f'calculate_od_at_new_location() estimates OD as {expected_value} +/- {atol_nd}: ', end='')
    print_pass() if test_result_od_accuracy else print_fail()

    assert test_result_od_accuracy, (
        f"Expected OD={expected_value}, got {interpolated_od}"
    )
    print('')
    print(
        f"test_calculate_od_at_new_location PASSED: "
        f"estimated OD = {interpolated_od}, expected OD = {expected_value}"
    )

'''
BRIGHT_CACHE_DIR = "/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks/bright"
DIM_CACHE_DIR = "/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks/dim"

def main():
    """
    this is for julia testing
    """

    # Make sure the directories exist
    os.makedirs(BRIGHT_CACHE_DIR, exist_ok=True)
    os.makedirs(DIM_CACHE_DIR, exist_ok=True)

    # 1) See if any .fits exist for bright stars
    bright_files = glob.glob(os.path.join(BRIGHT_CACHE_DIR, "*.fits"))
    if bright_files:
        print(f"[main()] Using cached bright .fits from:\n  {BRIGHT_CACHE_DIR}")
    else:
        print(f"[main()] No cached bright .fits found. Generating new mocks in:\n  {BRIGHT_CACHE_DIR}")
        bright_files = mock_bright_dataset_files(
            BRIGHT_EXPTIME,
            FILTER_USED,
            INPUT_OD,
            CAL_FACTOR,
            save_mocks=False,         # no real saving beyond memory?
            output_path=BRIGHT_CACHE_DIR,
        )

    # 2) See if any .fits exist for dim stars
    dim_files = glob.glob(os.path.join(DIM_CACHE_DIR, "*.fits"))
    if dim_files:
        print(f"[main()] Using cached dim .fits from:\n  {DIM_CACHE_DIR}")
    else:
        print(f"[main()] No cached dim .fits found. Generating new mocks in:\n  {DIM_CACHE_DIR}")
        dim_files = mock_dim_dataset_files(
            DIM_EXPTIME,
            FILTER_USED,
            CAL_FACTOR,
            save_mocks=False,
            output_path=DIM_CACHE_DIR,
        )

    # 3) Combine them into stars_dataset_cached
    combined_files = bright_files + dim_files
    stars_dataset_cached = Dataset(combined_files)

    # 4) Create an "output" subdirectory for storing calibration products, etc.
    output_dir = os.path.join(os.path.dirname(DIM_CACHE_DIR), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Need a "tmp" directory for test_background_effect (which expects tmp_path).
    background_tmp_dir = Path(os.path.join(os.path.dirname(DIM_CACHE_DIR), "tmp"))
    background_tmp_dir.mkdir(exist_ok=True)

    def run_test(test_func, *args, **kwargs):
        try:
            test_func(*args, **kwargs)
            print(f"{test_func.__name__} PASSED")
        except Exception as exc:
            print(f"{test_func.__name__} FAILED: {exc}")
            raise  

    print("\n========== BEGIN TESTS ==========")

    run_test(test_compute_exp_irrad)
    run_test(test_nd_filter_calibration_object_with_calspec, bright_files_cached)
    run_test(test_nd_filter_calibration_object, stars_dataset_cached)
    run_test(test_output_filename_convention, stars_dataset_cached)
    run_test(test_average_od_within_tolerance, stars_dataset_cached)

    for method in ["Aperture", "Gaussian"]:
        run_test(test_nd_filter_calibration_phot_methods, stars_dataset_cached, method)

    for test_od in [1.0, 3.0]:
        run_test(test_multiple_nd_levels, DIM_CACHE_DIR, output_dir, test_od)

    for aper_radius in [5, 10]:
        run_test(test_aperture_radius_sensitivity, stars_dataset_cached, aper_radius)

    run_test(test_background_effect, background_tmp_dir)

    run_test(test_nd_filter_calibration_with_fluxcal, DIM_CACHE_DIR, stars_dataset_cached, "Gaussian")

    test_calculate_od_at_new_location(output_dir)

    print("All tests PASSED")


if __name__ == "__main__":
    main()
'''
