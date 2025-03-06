import os
import glob
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
from astropy.io import fits
import pytest

import corgidrp.fluxcal as fluxcal
import corgidrp.nd_filter_calibration as nd_filter_calibration
import corgidrp.l2b_to_l3 as l2b_tol3
from corgidrp.data import Dataset
from corgidrp.data import Image
import corgidrp.mocks as mocks

# ---------------------------------------------------------------------------
# Global variables and constants
# ---------------------------------------------------------------------------
BRIGHT_STARS = ['109 Vir', 'Vega', 'Eta Uma', 'Lam Lep']
DIM_STARS = ['TYC 4433-1800-1', 'TYC 4205-1677-1', 'TYC 4212-455-1', 'TYC 4209-1396-1',
             'TYC 4413-304-1', 'UCAC3 313-62260', 'BPS BS 17447-0067', 'TYC 4424-1286-1',
             'GSC 02581-02323', 'TYC 4207-219-1']

DIM_EXPTIME = 10.0
BRIGHT_EXPTIME = 5.0
FWHM = 3
FILTER_USED = '3C'
INPUT_OD = 2.25
CAL_FACTOR = 0.8
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
            dim_star_flux, FWHM, cal_factor, filter_used, "HOLE", star_name,
            fsm_x=0, fsm_y=0, exptime=dim_exptime, filedir=output_path,
            color_cor=1.0, platescale=21.8,
            background=background_val,
            add_gauss_noise=add_gauss_noise_val,
            noise_scale=1.0, file_save=True
        )
        dim_star_images.append(flux_image)
    return dim_star_images


def mock_bright_dataset_files(bright_exptime, filter_used, OD, cal_factor, save_mocks, output_path=None, 
                              background_val=0, add_gauss_noise_val=False):
    """
    Generate and save mock bright dataset files for specified exposure time and filter.

    Parameters:
        bright_exptime (float): Exposure time for the simulated images.
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
    ND_transmission = 10 ** (-OD)
    bright_star_images = []
    for star_name in BRIGHT_STARS:
        for dy in [-10, 0, 10]:
            for dx in [-10, 0, 10]:
                bright_star_flux = nd_filter_calibration.compute_expected_band_irradiance(star_name, 
                                                                                          filter_used)
                attenuated_flux = bright_star_flux * ND_transmission
                flux_image = mocks.create_flux_image(
                    attenuated_flux, FWHM, cal_factor, filter_used, "ND225", star_name,
                    dx, dy, bright_exptime, output_path,
                    color_cor=1.0, platescale=21.8,
                    background=background_val,
                    add_gauss_noise=add_gauss_noise_val,
                    noise_scale=1.0, file_save=True
                )
                bright_star_images.append(flux_image)
    return bright_star_images


def mock_clean_entry(bright_dataset):
    # TO DO: eventually add other processing steps that would produce the 
    # appropriate level input file
    return bright_dataset[0]


def mock_transformation_matrix(output_dir):
    # TO DO: use Sergi's FPAM to EXCAM transformation matrix product once it is
    # ready
    transformation_matrix_file = os.path.join(output_dir, "fpam_to_excam.fits")
    dummy_matrix = np.eye(2)
    hdu = fits.PrimaryHDU(dummy_matrix)
    hdu.writeto(transformation_matrix_file, overwrite=True)
    return transformation_matrix_file

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
    return l2b_tol3.divide_by_exptime(Dataset(combined_files))

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
def test_nd_filter_calibration_object(stars_dataset_cached, output_dir):
    print("**Testing ND filter calibration object generation and expected headers**")
    results = nd_filter_calibration.create_nd_filter_cal(
        stars_dataset_cached, OD_RASTER_THRESHOLD, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS, 
        fluxcal_factor = None)
    
    # TO DO: update this when file name conventions are finalized
    results.save(output_dir, "CGI_NDF_CAL.fits")

    nd_files = [fn for fn in os.listdir(output_dir) if fn.endswith('_NDF_CAL.fits')]
    assert nd_files, "No NDFilterOD files were generated."
    with fits.open(os.path.join(output_dir, nd_files[0])) as hdul:
        primary_hdr = hdul[0].header
        ext_hdr = hdul[1].header
        assert primary_hdr.get('SIMPLE') is True, "Primary header missing or SIMPLE not True."
        assert ext_hdr.get('FPAMNAME') is not None, "Missing FPAMNAME keyword."
        assert ext_hdr.get('FPAM_H') is not None, "Missing FPAM_H keyword."
        assert ext_hdr.get('FPAM_V') is not None, "Missing FPAM_V keyword."
        assert ext_hdr.get('CFAMNAME') is not None, "Missing CFAMNAME keyword."


def test_output_filename_convention(stars_dataset_cached, output_dir):
    print("**Testing output filename naming conventions**")
    results = nd_filter_calibration.create_nd_filter_cal(
        stars_dataset_cached, OD_RASTER_THRESHOLD, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS, 
        fluxcal_factor = None)
    # TO DO: update this when file name conventions are decided
    # TO DO: update this when file name conventions are finalized
    results.save(output_dir, "CGI_NDF_CAL.fits")
    nd_files = [fn for fn in os.listdir(output_dir) if fn.endswith('_NDF_CAL.fits')]
    assert nd_files, "No files found matching naming convention."


def test_average_od_within_tolerance(stars_dataset_cached):
    print("**Testing computed OD within tolerance**")
    results = nd_filter_calibration.create_nd_filter_cal(
        stars_dataset_cached, OD_RASTER_THRESHOLD, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS, 
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
    results = nd_filter_calibration.create_nd_filter_cal(
        stars_dataset_cached, OD_RASTER_THRESHOLD, phot_method, FLUX_OR_IRR, phot_args, 
        fluxcal_factor = None)
    ods = results.data
    avg_od = np.mean(ods[:, 0])
    assert abs(avg_od - INPUT_OD) < OD_TEST_TOLERANCE, (
        f"Method {phot_method}: OD mismatch"
    )


@pytest.mark.parametrize("test_od", [1.0, 2.0, 3.0])
def test_multiple_nd_levels(dim_dir, output_dir, test_od):
    print(f"**Testing multiple ND levels with input OD = {test_od}**")
    bright_mocks_dir = os.path.join(output_dir, f"mock_OD{test_od}")
    bright_images = mock_bright_dataset_files(
        BRIGHT_EXPTIME, FILTER_USED, test_od, CAL_FACTOR, save_mocks=True, output_path=bright_mocks_dir
    )

    dim_filepaths = glob.glob(os.path.join(dim_dir, "*")) # use cached dim images
    dim_images = [Image(path) for path in dim_filepaths]
    combined_files = bright_images + dim_images
    combined_dataset = l2b_tol3.divide_by_exptime(Dataset(combined_files))

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

    # Convert list of Image objects into a Dataset
    dim_dataset = l2b_tol3.divide_by_exptime(Dataset(dim_images))

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


@pytest.mark.parametrize("aper_radius", [3, 5, 7, 10])
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
    results = nd_filter_calibration.create_nd_filter_cal(
        stars_dataset_cached, OD_RASTER_THRESHOLD, "Aperture", "irr", phot_args, 
        fluxcal_factor = None)
    ods = results.data
    avg_od = np.mean(ods[:, 0])
    assert abs(avg_od - INPUT_OD) < 0.3, (
        f"AperRadius={aper_radius}: OD mismatch for target."
    )


def test_od_stability(stars_dataset_cached):
    # TO DO: move this out of the test code and into the calibration product generation 
    print("**Testing OD stability across multiple dithers**")
    results = nd_filter_calibration.create_nd_filter_cal(
        stars_dataset_cached, OD_RASTER_THRESHOLD, "Aperture", "irr", PHOT_ARGS, 
        fluxcal_factor = None)
    ods = results.data
    std_od = np.std(ods[:, 0])
    allowed_scatter = 0.05
    assert std_od < allowed_scatter, (
        f"OD dithers for target have std={np.std(ods)}, expected < {allowed_scatter}"
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
    dim_dir_no.mkdir()
    dim_dir_bg.mkdir()
    dim_files_no = mock_dim_dataset_files(DIM_EXPTIME, FILTER_USED, CAL_FACTOR, save_mocks=False,
                                      output_path=str(dim_dir_no), background_val=0, add_gauss_noise_val=False)
    dim_files_bg = mock_dim_dataset_files(DIM_EXPTIME, FILTER_USED, CAL_FACTOR, save_mocks=False,
                                      output_path=str(dim_dir_bg), background_val=BACKGROUND, add_gauss_noise_val=ADD_GAUSS)

    # Create bright star mocks for two modes.
    bright_dir_no = tmp_path / "bright_no"
    bright_dir_bg = tmp_path / "bright_bg"
    bright_dir_no.mkdir()
    bright_dir_bg.mkdir()
    bright_files_no = mock_bright_dataset_files(BRIGHT_EXPTIME, FILTER_USED, INPUT_OD, CAL_FACTOR, save_mocks=False,
                                                output_path=str(bright_dir_no), background_val=0, add_gauss_noise_val=False)
    bright_files_bg = mock_bright_dataset_files(BRIGHT_EXPTIME, FILTER_USED, INPUT_OD, CAL_FACTOR, save_mocks=False,
                                                output_path=str(bright_dir_bg), background_val=BACKGROUND, add_gauss_noise_val=ADD_GAUSS)
    
    combined_no = dim_files_no + bright_files_no
    combined_bg = dim_files_bg + bright_files_bg
    ds_no = l2b_tol3.divide_by_exptime(Dataset(combined_no))
    ds_bg = l2b_tol3.divide_by_exptime(Dataset(combined_bg))
    
    output_directory = str(tmp_path / "output")
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
    stars_dataset_cached = l2b_tol3.divide_by_exptime(Dataset(combined_files))

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

    #run_test(test_nd_filter_calibration_object, stars_dataset_cached, output_dir)
    #run_test(test_output_filename_convention, stars_dataset_cached, output_dir)
    run_test(test_average_od_within_tolerance, stars_dataset_cached)

    #for method in ["Aperture", "Gaussian"]:
    #    run_test(test_nd_filter_calibration_phot_methods, stars_dataset_cached, method)

    #for test_od in [1.0, 2.0, 3.0]:
    #    run_test(test_multiple_nd_levels, DIM_CACHE_DIR, output_dir, test_od)

    #for aper_radius in [3, 5, 7, 10]:
    #    run_test(test_aperture_radius_sensitivity, stars_dataset_cached, aper_radius)

    #run_test(test_od_stability, stars_dataset_cached)

    #run_test(test_background_effect, background_tmp_dir)

    run_test(test_nd_filter_calibration_with_fluxcal, DIM_CACHE_DIR, stars_dataset_cached, "Gaussian")

    print("All tests PASSED")

if __name__ == "__main__":
    main()

'''
