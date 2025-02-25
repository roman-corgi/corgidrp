import os
import re
import glob
import numpy as np
from astropy.io import fits
import pytest

import corgidrp.nd_filter_calibration as nd_filter_calibration
import corgidrp.l2b_to_l3 as l2b_tol3
from corgidrp.data import Dataset
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
INPUT_OD = 2.75
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
elif PHOT_METHOD == "PSF":
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
    if save_mocks:
        output_path = output_path or os.getcwd()
    else:
        output_path = output_path or os.getcwd()
    os.makedirs(output_path, exist_ok=True)
    dim_star_images = []
    for star_name in DIM_STARS:
        dim_star_flux = nd_filter_calibration.compute_expected_band_irradiance(star_name, filter_used)
        flux_image = mocks.create_flux_image(
            dim_star_flux, FWHM, cal_factor, filter_used, star_name,
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
                    attenuated_flux, FWHM, cal_factor, filter_used, star_name,
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
def dim_dataset_cached(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("dim_dataset")
    print(f"Generating cached dim dataset in {tmp_dir}")
    files = mock_dim_dataset_files(DIM_EXPTIME, FILTER_USED, CAL_FACTOR, save_mocks=False, output_path=str(tmp_dir))
    ds = Dataset(files)
    # Divide by exposure time
    # TO DO: May eventually need to add other processing steps to get the mocks
    # to a representative input state
    return l2b_tol3.divide_by_exptime(ds)

@pytest.fixture(scope="module")
def bright_dataset_cached(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("bright_dataset")
    print(f"Generating cached bright dataset in {tmp_dir}")
    files = mock_bright_dataset_files(BRIGHT_EXPTIME, FILTER_USED, INPUT_OD, CAL_FACTOR, save_mocks=False, output_path=str(tmp_dir))
    ds = Dataset(files)
    # Divide by exposure time
    # TO DO: May eventually need to add other processing steps to get the mocks
    # to a representative input state
    return l2b_tol3.divide_by_exptime(ds)

# The output directory can still be function-scoped
@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    print(f"Created temporary output directory: {out}")
    return str(out)

# ---------------------------------------------------------------------------
# Test functions using pytest
# ---------------------------------------------------------------------------
def test_nd_filter_calibration_object(dim_dataset_cached, bright_dataset_cached, output_dir):
    print("**Testing ND filter calibration object generation and expected headers**")
    clean_entry = mock_clean_entry(bright_dataset_cached)
    transformation_matrix_file = mock_transformation_matrix(output_dir)
    nd_filter_calibration.create_nd_filter_cal(
        dim_dataset_cached, bright_dataset_cached, output_dir,
        FILESAVE, OD_RASTER_THRESHOLD, clean_entry,
        transformation_matrix_file, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS
    )
    nd_files = [fn for fn in os.listdir(output_dir) if fn.endswith('_NDF_SWEETSPOT.fits')]
    assert nd_files, "No NDFilterOD files were generated."
    with fits.open(os.path.join(output_dir, nd_files[0])) as hdul:
        primary_hdr = hdul[0].header
        ext_hdr = hdul[1].header
        assert primary_hdr.get('SIMPLE') is True, "Primary header missing or SIMPLE not True."
        assert ext_hdr.get('FPAMNAME') is not None, "Missing FPAMNAME keyword."
        assert ext_hdr.get('FPAM_H') is not None, "Missing FPAM_H keyword."
        assert ext_hdr.get('FPAM_V') is not None, "Missing FPAM_V keyword."
        assert ext_hdr.get('CFAMNAME') is not None, "Missing CFAMNAME keyword."


def test_output_filename_convention(dim_dataset_cached, bright_dataset_cached, output_dir):
    print("**Testing output filename naming conventions**")
    clean_entry = mock_clean_entry(bright_dataset_cached)
    transformation_matrix_file = mock_transformation_matrix(output_dir)
    nd_filter_calibration.create_nd_filter_cal(
        dim_dataset_cached, bright_dataset_cached, output_dir,
        FILESAVE, OD_RASTER_THRESHOLD, clean_entry,
        transformation_matrix_file, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS
    )
    pattern = re.compile(fr"CGI_[A-Za-z0-9]+_(\d{{3}})_FilterBand_{FILTER_USED}_NDF_SWEETSPOT\.fits")
    filenames = os.listdir(output_dir)
    sweet_spot_files = [fn for fn in filenames if pattern.match(fn)]
    assert sweet_spot_files, "No files found matching naming convention."
    for fn in sweet_spot_files:
        match = pattern.match(fn)
        serial = int(match.group(1))
        assert serial >= 1, f"Invalid serial number in {fn}"


def test_average_od_within_tolerance(dim_dataset_cached, bright_dataset_cached, output_dir):
    print("**Testing computed OD within tolerance**")
    clean_entry = mock_clean_entry(bright_dataset_cached)
    transformation_matrix_file = mock_transformation_matrix(output_dir)
    results = nd_filter_calibration.create_nd_filter_cal(
        dim_dataset_cached, bright_dataset_cached, output_dir,
        FILESAVE, OD_RASTER_THRESHOLD, clean_entry,
        transformation_matrix_file, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS
    )
    assert isinstance(results, dict) and results, "Results should be a non-empty dictionary."
    for target, data in results['flux_results'].items():
        avg_od = data['average_od']
        assert abs(avg_od - INPUT_OD) < OD_TEST_TOLERANCE, (
            f"Target {target}: avg OD {avg_od} deviates more than {OD_TEST_TOLERANCE}"
        )
        od_std = np.std(data['od_values'])
        if od_std < OD_RASTER_THRESHOLD:
            assert not data['flag'], f"Target {target}: low OD variation but flag is set"

        else:
            assert data['flag'], f"Target {target}: high OD variation but flag is not set"


@pytest.mark.parametrize("phot_method", ["Aperture", "PSF"])
def test_nd_filter_calibration_phot_methods(dim_dataset_cached, bright_dataset_cached, output_dir, phot_method):
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
    clean_entry = mock_clean_entry(bright_dataset_cached)
    transformation_matrix_file = mock_transformation_matrix(output_dir)
    results = nd_filter_calibration.create_nd_filter_cal(
        dim_dataset_cached, bright_dataset_cached, output_dir,
        file_save=True, od_raster_threshold=OD_RASTER_THRESHOLD,
        clean_entry=clean_entry, transformation_matrix_file=transformation_matrix_file,
        phot_method=phot_method, flux_or_irr="irr", phot_kwargs=phot_args
    )
    assert "flux_results" in results, "No flux results returned"
    for target, data in results['flux_results'].items():
        avg_od = data['average_od']
        assert abs(avg_od - INPUT_OD) < OD_TEST_TOLERANCE, (
            f"Method {phot_method}, target {target}: OD mismatch"
        )


@pytest.mark.parametrize("test_od", [1.0, 2.0, 3.0])
def test_multiple_nd_levels(dim_dataset_cached, output_dir, test_od):
    print(f"**Testing multiple ND levels with input OD = {test_od}**")
    bright_mocks_dir = os.path.join(output_dir, f"mock_brighter_OD{test_od}")
    bright_files = mock_bright_dataset_files(
        BRIGHT_EXPTIME, FILTER_USED, test_od, CAL_FACTOR, save_mocks=True, output_path=bright_mocks_dir
    )
    bright_ds = Dataset(bright_files)
    bright_ds_l3 = l2b_tol3.divide_by_exptime(bright_ds)
    clean_entry = mock_clean_entry(bright_ds_l3)
    transformation_matrix_file = mock_transformation_matrix(output_dir)
    results = nd_filter_calibration.create_nd_filter_cal(
        dim_dataset_cached, bright_ds_l3, output_dir,
        file_save=True, od_raster_threshold=OD_RASTER_THRESHOLD,
        clean_entry=clean_entry, transformation_matrix_file=transformation_matrix_file,
        phot_method="Aperture", flux_or_irr="irr", phot_kwargs=PHOT_ARGS
    )
    for target, data in results['flux_results'].items():
        avg_od = data['average_od']
        assert abs(avg_od - test_od) < 0.2, (
            f"test_od={test_od}, got {avg_od} for target {target}"
        )


@pytest.mark.parametrize("aper_radius", [3, 5, 7, 10])
def test_aperture_radius_sensitivity(dim_dataset_cached, bright_dataset_cached, output_dir, aper_radius):
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
    clean_entry = mock_clean_entry(bright_dataset_cached)
    transformation_matrix_file = mock_transformation_matrix(output_dir)
    results = nd_filter_calibration.create_nd_filter_cal(
        dim_dataset_cached, bright_dataset_cached, output_dir,
        file_save=False, od_raster_threshold=OD_RASTER_THRESHOLD,
        clean_entry=clean_entry, transformation_matrix_file=transformation_matrix_file,
        phot_method="Aperture", flux_or_irr="irr", phot_kwargs=phot_args
    )
    for target, data in results["flux_results"].items():
        avg_od = data["average_od"]
        assert abs(avg_od - INPUT_OD) < 0.3, (
            f"AperRadius={aper_radius}: OD mismatch for {target}"
        )


def test_od_stability(dim_dataset_cached, bright_dataset_cached, output_dir):
    print("**Testing OD stability across multiple dithers**")
    clean_entry = mock_clean_entry(bright_dataset_cached)
    transformation_matrix_file = mock_transformation_matrix(output_dir)
    results = nd_filter_calibration.create_nd_filter_cal(
        dim_dataset_cached, bright_dataset_cached, output_dir,
        file_save=True, od_raster_threshold=OD_RASTER_THRESHOLD,
        clean_entry=clean_entry, transformation_matrix_file=transformation_matrix_file,
        phot_method="Aperture", flux_or_irr="irr", phot_kwargs=PHOT_ARGS
    )
    for target, data in results['flux_results'].items():
        od_values = data['od_values']
        allowed_scatter = 0.05
        assert np.std(od_values) < allowed_scatter, (
            f"OD dithers for {target} have std={np.std(od_values)}, expected < {allowed_scatter}"
        )

def test_background_effect(tmp_path):
    """
    Generate two sets of mocks (one without background, one with background)
    and compare the calibration results. We expect that the overall OD values
    remain similar while the error (scatter) increases when background is added.
    """
    # Create dim star mocks for two modes.
    dim_dir_no = tmp_path / "dim_no"
    dim_dir_bg = tmp_path / "dim_bg"
    dim_dir_no.mkdir()
    dim_dir_bg.mkdir()
    files_no = mock_dim_dataset_files(DIM_EXPTIME, FILTER_USED, CAL_FACTOR, save_mocks=False,
                                      output_path=str(dim_dir_no), background_val=0, add_gauss_noise_val=False)
    files_bg = mock_dim_dataset_files(DIM_EXPTIME, FILTER_USED, CAL_FACTOR, save_mocks=False,
                                      output_path=str(dim_dir_bg), background_val=BACKGROUND, add_gauss_noise_val=ADD_GAUSS)
    ds_no = l2b_tol3.divide_by_exptime(Dataset(files_no))
    ds_bg = l2b_tol3.divide_by_exptime(Dataset(files_bg))
    
    # Create bright star mocks for two modes.
    bright_dir_no = tmp_path / "bright_no"
    bright_dir_bg = tmp_path / "bright_bg"
    bright_dir_no.mkdir()
    bright_dir_bg.mkdir()
    bright_files_no = mock_bright_dataset_files(BRIGHT_EXPTIME, FILTER_USED, INPUT_OD, CAL_FACTOR, save_mocks=False,
                                                output_path=str(bright_dir_no), background_val=0, add_gauss_noise_val=False)
    bright_files_bg = mock_bright_dataset_files(BRIGHT_EXPTIME, FILTER_USED, INPUT_OD, CAL_FACTOR, save_mocks=False,
                                                output_path=str(bright_dir_bg), background_val=BACKGROUND, add_gauss_noise_val=ADD_GAUSS)
    ds_bright_no = l2b_tol3.divide_by_exptime(Dataset(bright_files_no))
    ds_bright_bg = l2b_tol3.divide_by_exptime(Dataset(bright_files_bg))
    
    output_directory = str(tmp_path / "output")
    os.mkdir(output_directory)
    
    clean_entry_no = ds_bright_no[0]
    clean_entry_bg = ds_bright_bg[0]
    transformation_matrix_file = mock_transformation_matrix(output_directory)
    
    results_no = nd_filter_calibration.create_nd_filter_cal(
        ds_no, ds_bright_no, output_directory,
        FILESAVE, OD_RASTER_THRESHOLD, clean_entry_no,
        transformation_matrix_file, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS
    )
    results_bg = nd_filter_calibration.create_nd_filter_cal(
        ds_bg, ds_bright_bg, output_directory,
        FILESAVE, OD_RASTER_THRESHOLD, clean_entry_bg,
        transformation_matrix_file, PHOT_METHOD, FLUX_OR_IRR, PHOT_ARGS
    )
    
    for target in results_no['flux_results'].keys():
        od_no = results_no['flux_results'][target]['average_od']
        od_bg = results_bg['flux_results'][target]['average_od']
        err_no = np.std(results_no['flux_results'][target]['od_values'])
        err_bg = np.std(results_bg['flux_results'][target]['od_values'])
        print(f"Target {target}: OD no-bg = {od_no}, error no-bg = {err_no}; OD bg = {od_bg}, error bg = {err_bg}")
        # The overall OD should be similar, within a small tolerance.
        assert abs(od_no - od_bg) < 0.1, f"OD should not differ drastically between modes for {target}"
