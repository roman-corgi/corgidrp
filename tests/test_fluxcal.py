import pytest
import warnings
import os
import shutil
import time
import numpy as np
import logging
from pathlib import Path
import corgidrp
from astropy.io import fits

# Suppress all warnings for tests in this file
warnings.filterwarnings("ignore")
from corgidrp.mocks import (
    create_default_L3_headers,
    create_flux_image,
    create_pol_flux_image,
    create_mock_stokes_i_image,
    gaussian_array,
    create_ct_cal,
    create_mock_fpamfsam_cal,
    make_mock_fluxcal_factor,
    rename_files_to_cgi_format,
)
from corgidrp.data import Image, Dataset, FluxcalFactor, get_stokes_intensity_image
from corgidrp.check import verify_header_keywords
import corgidrp.fluxcal as fluxcal
import corgidrp.l4_to_tda as l4_to_tda
from astropy.modeling.models import BlackBody
import astropy.units as u
from termcolor import cprint
import numpy as np

# Suppress all warnings for all tests in this file
# warnings.filterwarnings("ignore")


data = np.ones([1024,1024]) * 2 
err = np.ones([1,1024,1024]) * 0.5
prhd, exthd, errhdr, dqhdr = create_default_L3_headers()
exthd["CFAMNAME"] = '3C'
exthd["FPAMNAME"] = 'ND475'
prhd["TARGET"] = 'VEGA'
image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err)
image2 = image1.copy()
image1.filename = "cgi_0000000000000090526_20240101t1200000_l4_.fits"
image2.filename = "cgi_0000000000000090527_20240101t1201000_l4_.fits"
dataset=Dataset([image1, image2])
calspec_filepath = os.path.join(os.path.dirname(__file__), "test_data", "alpha_lyr_stis_011.fits")


def print_fail():
    cprint(' FAIL ', "black", "on_red")


def print_pass():
    cprint(' PASS ', "black", "on_green")


def test_get_filter_name():
    """
    test that the correct filter curve file is selected
    """
    global wave
    global transmission
    filepath = fluxcal.get_filter_name(image1)
    assert filepath.split("/")[-1] == 'transmission_ID-21_3C_v0.csv'
    
    wave, transmission = fluxcal.read_filter_curve(filepath)
    
    assert np.any(wave>=7130)
    assert np.any(transmission < 1.)
    
    #test a wrong filter name
    image3 = image1.copy()
    image3.ext_hdr["CFAMNAME"] = '5C'
    with pytest.raises(ValueError):
        filepath = fluxcal.get_filter_name(image3)
        pass
    

def test_flux_calc():
    """
    test that the calspec data is read correctly
    """
    global band_flux
    calspec_flux = fluxcal.read_cal_spec(calspec_filepath, wave)
    assert calspec_flux[0] == pytest.approx(1.6121e-09, 1e-15) 
    
    band_flux = fluxcal.calculate_band_flux(transmission, calspec_flux, wave)
    eff_lambda = fluxcal.calculate_effective_lambda(transmission, calspec_flux, wave)
    assert eff_lambda == pytest.approx((wave[0]+wave[-1])/2., 3)
    
def test_colorcor():
    """
    test that the pivot reference wavelengths is close to the center of the bandpass
    and the color correction is calculated correctly
    """
    
    lambda_piv = fluxcal.calculate_pivot_lambda(transmission, wave)
    assert lambda_piv == pytest.approx((wave[0]+wave[-1])/2., 0.3)
    
    calspec_flux = fluxcal.read_cal_spec(calspec_filepath, wave)
    ## BB of an O5 star
    bbscale = 1.e-21 * u.erg/(u.s * u.cm**2 * u.AA * u.steradian)
    flux_source = BlackBody(scale = bbscale, temperature=54000.0 * u.K)
    K_bb = fluxcal.compute_color_cor(transmission, wave, calspec_flux, lambda_piv, flux_source(wave))
    assert K_bb == pytest.approx(1., 0.01)
    
    flux_source = BlackBody(scale = bbscale, temperature=100. * u.K)
    K_bb = fluxcal.compute_color_cor(transmission, wave, calspec_flux, lambda_piv, flux_source(wave))
    assert K_bb > 2#weakest star to be detected
    # sanity check
    K = fluxcal.compute_color_cor(transmission, wave, calspec_flux, lambda_piv, calspec_flux)
    assert K == 1 

    # test the corresponding pipeline step
    output_dataset = l4_to_tda.determine_color_cor(dataset, calspec_filepath, calspec_filepath)
    assert output_dataset[0].ext_hdr['LAM_REF'] == lambda_piv
    assert output_dataset[0].ext_hdr['COL_COR'] == K
    # test it with star names
    calspec_name = 'Vega'
    source_name = 'TYC 4424-1286-1'
    output_dataset = l4_to_tda.determine_color_cor(dataset, calspec_name, source_name)
    assert "1732526_stisnic_009.fits" in str(output_dataset[0].ext_hdr['HISTORY'])
    assert output_dataset[0].ext_hdr['LAM_REF'] == lambda_piv
    assert output_dataset[0].ext_hdr['COL_COR'] == pytest.approx(1,1e-2) 
    
def test_calspec_download():
    """
    test the download of a calspec fits file
    """
    
    filepath, filename = fluxcal.get_calspec_file('TYC 4424-1286-1')
    assert os.path.exists(filepath)
    assert filename == '1732526_stisnic_009.fits'
    os.remove(filepath)
    
    filepath, filename = fluxcal.get_calspec_file('Vega')
    assert os.path.exists(filepath)
    assert filename == 'alpha_lyr_stis_011.fits'
    
    calspec_dir = os.path.join(os.path.dirname(corgidrp.config_filepath), "calspec_data")
    names_file = os.path.join(calspec_dir, "calspec_names.json")
    fits_file = os.path.join(calspec_dir, 'alpha_lyr_stis_011.fits')
    assert os.path.exists(names_file)
    assert os.path.exists(fits_file)
    assert fits_file == filepath
    
    # test the priority of the fits file path to .corgidrp
    test_calspec_url = fluxcal.calspec_url
    fluxcal.calspec_url = "wrong"
    
    filepath, filename = fluxcal.get_calspec_file('Vega')
    assert os.path.exists(filepath)
    assert filename == 'alpha_lyr_stis_011.fits'
    os.remove(filepath)
    
    with pytest.raises(ValueError):
        filepath, filename = fluxcal.get_calspec_file('TYC 4424-1286-1')
    
    fluxcal.calspec_url = test_calspec_url
    
    with pytest.raises(ValueError):
        filepath, filename = fluxcal.get_calspec_file('Todesstern')

def test_app_mag():
    """
    test the calculation of the apparent Vega magnitude
    """
    # test the corresponding pipeline step
    # sanity check
    output_dataset = l4_to_tda.determine_app_mag(dataset, 'Vega')
    assert output_dataset[0].ext_hdr['APP_MAG'] == 0.
    output_dataset = l4_to_tda.determine_app_mag(dataset, calspec_filepath)
    assert output_dataset[0].ext_hdr['APP_MAG'] == pytest.approx(0.0, 0.03) 
    output_dataset = l4_to_tda.determine_app_mag(dataset, calspec_filepath, scale_factor = 0.5)
    assert output_dataset[0].ext_hdr['APP_MAG'] == pytest.approx(0.+-2.5*np.log10(0.5), 0.03)
    output_dataset = l4_to_tda.determine_app_mag(dataset, '109 Vir')
    assert output_dataset[0].ext_hdr['APP_MAG'] == pytest.approx(3.72, 0.05)
    assert 'alpha_lyr_stis_011.fits' in str(output_dataset[0].ext_hdr['HISTORY'])
    assert '109vir_stis_005.fits' in str(output_dataset[0].ext_hdr['HISTORY'])
    
def test_fluxcal_file():
    """ 
    Generate a mock fluxcal factor cal object and test the content and functionality.
    """
    fluxcal_factor = 2e-12
    fluxcal_factor_error = 1e-14
    fluxcal_fac = FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, input_dataset = dataset)
    assert fluxcal_fac.filter == '3C'
    assert fluxcal_fac.fluxcal_fac == fluxcal_factor
    assert fluxcal_fac.fluxcal_err == fluxcal_factor_error
    assert(fluxcal_fac.filename.endswith("_abf_cal.fits"))
    
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    filename = fluxcal_fac.filename
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    fluxcal_fac.save(filedir=calibdir, filename=filename)        
        
    fluxcal_filepath = os.path.join(calibdir, filename)

    fluxcal_fac_file = FluxcalFactor(fluxcal_filepath)
    assert fluxcal_fac_file.filter == '3C'
    assert fluxcal_fac_file.fluxcal_fac == fluxcal_factor
    assert fluxcal_fac_file.fluxcal_err == fluxcal_factor_error
    # JM: I moved this out of the fluxcal class and into fluxcal.py because, depending on the method you use to 
    # make the fluxcal factor, the BUNIT will vary. Doing a mock without running fluxcal methods won't update BUNIT
    #assert fluxcal_fac_file.ext_hdr["BUNIT"] == 'erg/(s * cm^2 * AA)/(electron/s)'









def _setup_vap_logger(test_name):
    """Create a logger that saves PASS/FAIL records to pol_tda_companion_phot_output."""
    log_dir = Path(__file__).resolve().parent / "pol_tda_companion_phot_output"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(test_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_dir / f"{test_name}.log", mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, log_dir



def make_1d_spec_image(spec_values, spec_err, spec_wave, roll=None, exp_time=None, col_cor=None):
    """Create a mock L4 file with 1-D spectroscopy extensions.

    Args:
        spec_values (ndarray): flux values (photoelectron/s) for `SPEC`.
        spec_err (ndarray): uncertainty array matching `SPEC` shape.
        spec_wave (ndarray): wavelength grid in nm for `SPEC_WAVE`.
        roll (str, optional): telescope roll angle
        exp_time (float, optional): exposure time in seconds
        col_cor (float, optional): color-correction factor to record.

    Returns:
        corgidrp.data.Image: image with `SPEC`, `SPEC_ERR`, `SPEC_DQ`,
        `SPEC_WAVE`, and `SPEC_WAVE_ERR` extensions populated.
    """
    data = np.zeros((10, 10))
    err = np.ones((1, 10, 10))
    dq = np.zeros((10, 10), dtype=int)
    pri_hdr, ext_hdr, err_hdr, dq_hdr = create_default_L3_headers()
    ext_hdr['BUNIT'] = 'photoelectron/s'
    ext_hdr['WV0_X'] = 0.0
    ext_hdr['WV0_Y'] = 0.0
    pri_hdr['ROLL'] = roll
    pri_hdr['EXP_TIME'] = exp_time
    if col_cor is not None:
        ext_hdr['COL_COR'] = col_cor
    img = Image(data, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err=err, dq=dq,
                err_hdr=err_hdr, dq_hdr=dq_hdr)

    spec_hdr = fits.Header()
    spec_hdr['BUNIT'] = 'photoelectron/s/bin'
    img.add_extension_hdu('SPEC', data=spec_values, header=spec_hdr)
    img.add_extension_hdu('SPEC_ERR', data=spec_err, header=spec_hdr.copy())
    img.add_extension_hdu('SPEC_DQ', data=np.zeros_like(spec_values, dtype=int))

    wave_hdr = fits.Header()
    wave_hdr['BUNIT'] = 'nm'
    img.add_extension_hdu('SPEC_WAVE', data=spec_wave, header=wave_hdr)
    img.add_extension_hdu('SPEC_WAVE_ERR', data=np.zeros_like(spec_wave), header=wave_hdr.copy())
    return img


def test_convert_spec_to_flux_basic():
    """Validate convert_spec_to_flux with slit correction and COL_COR applied."""
    spec_vals = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    spec_err = np.array([[0.5, 0.6, 0.7, 0.8, 0.9]])
    wave = np.linspace(700, 800, len(spec_vals))
    slit_factor = 0.85
    slit_tuple = (np.array([np.full_like(spec_vals, slit_factor)]), np.array([0.0]), np.array([0.0]))

    image = make_1d_spec_image(spec_vals, spec_err, wave, col_cor=2.0)
    image.hdu_list['SPEC'].header['CTCOR'] = True  # Core throughput correction already applied
    dataset = Dataset([image])
    fluxcal_factor = make_mock_fluxcal_factor(2.0, err=0.2)

    calibrated = l4_to_tda.convert_spec_to_flux(dataset, fluxcal_factor, slit_transmission=slit_tuple)
    frame = calibrated[0]
    spec_out = frame.hdu_list['SPEC'].data
    err_out = frame.hdu_list['SPEC_ERR'].data

    expected_spec = (spec_vals / slit_factor) * (fluxcal_factor.fluxcal_fac / 2.0)
    expected_err = np.sqrt(((spec_err[0] / slit_factor) * (fluxcal_factor.fluxcal_fac / 2.0))**2 +
                           (((spec_vals / slit_factor) * fluxcal_factor.fluxcal_err / 2.0))**2)

    result = np.allclose(spec_out, expected_spec) and np.allclose(err_out[0], expected_err)
    print('\nconvert_spec_to_flux basic case: ', end='')
    print_pass() if result else print_fail()

    assert result
    assert frame.hdu_list['SPEC'].header['BUNIT'] == "erg/(s*cm^2*AA)"
    assert frame.hdu_list['SPEC_ERR'].header['BUNIT'] == "erg/(s*cm^2*AA)"
    assert frame.hdu_list['SPEC'].header['SLITCOR'] is True
    assert frame.ext_hdr['SPECUNIT'] == "erg/(s*cm^2*AA)"


def test_convert_spec_to_flux_no_slit():
    """Validate convert_spec_to_flux when no slit transmission vector is supplied."""
    spec_vals = np.array([5.0, 6.0, 7.0])
    spec_err = np.array([[0.2, 0.3, 0.4]])
    wave = np.linspace(600, 650, len(spec_vals))

    image = make_1d_spec_image(spec_vals, spec_err, wave)
    image.hdu_list['SPEC'].header['CTCOR'] = True  # core throughput correction already applied
    dataset = Dataset([image])
    fluxcal_factor = make_mock_fluxcal_factor(1.5, err=0.1)

    calibrated = l4_to_tda.convert_spec_to_flux(dataset, fluxcal_factor)
    frame = calibrated[0]

    expected_spec = spec_vals * fluxcal_factor.fluxcal_fac
    expected_err = np.sqrt((spec_err[0] * fluxcal_factor.fluxcal_fac) ** 2 +
                           (spec_vals * fluxcal_factor.fluxcal_err) ** 2)

    result = np.allclose(frame.hdu_list['SPEC'].data, expected_spec) and \
             np.allclose(frame.hdu_list['SPEC_ERR'].data[0], expected_err)
    print('\nconvert_spec_to_flux without slit: ', end='')
    print_pass() if result else print_fail()

    assert result
    assert frame.hdu_list['SPEC'].header['SLITCOR'] is False
    assert frame.hdu_list['SPEC'].header['BUNIT'] == "erg/(s*cm^2*AA)"
    assert frame.ext_hdr['FLUXFAC'] == fluxcal_factor.fluxcal_fac


def test_convert_spec_to_flux_slit_scalar_map():
    """Tuple slit transmission should produce a wavelength-dependent correction."""
    spec_vals = np.array([10.0, 12.0, 14.0, 16.0])
    spec_err = np.full((1, spec_vals.size), 0.5)
    wave = np.linspace(700, 730, spec_vals.size)

    image = make_1d_spec_image(spec_vals, spec_err, wave)
    image.ext_hdr['WV0_X'] = 25.0
    image.ext_hdr['WV0_Y'] = 0.0
    image.hdu_list['SPEC'].header['CTCOR'] = True  # Core throughput correction already applied
    dataset = Dataset([image])
    fluxcal_factor = make_mock_fluxcal_factor(2.0, err=0.2)

    # Build a slit map where the nearest position to WV0_X has a flat 0.45 curve
    slit_row_nearest = np.full_like(spec_vals, 0.45, dtype=float)
    slit_map = np.array([
        np.full_like(spec_vals, 0.5, dtype=float),
        slit_row_nearest,
        np.full_like(spec_vals, 0.3, dtype=float),
    ])
    slit_x = np.array([0.0, 20.0, 100.0])
    slit_y = np.zeros_like(slit_x)

    calibrated = l4_to_tda.convert_spec_to_flux(
        dataset,
        fluxcal_factor,
        slit_transmission=(slit_map, slit_x, slit_y),
    )
    frame = calibrated[0]
    slit_factor = 0.45
    expected_spec = (spec_vals / slit_row_nearest) * fluxcal_factor.fluxcal_fac
    expected_err = np.sqrt(
        ((spec_err[0] / slit_factor) * fluxcal_factor.fluxcal_fac) ** 2 +
        ((spec_vals / slit_factor) * fluxcal_factor.fluxcal_err) ** 2
    )

    result = (
        np.allclose(frame.hdu_list['SPEC'].data, expected_spec) and
        np.allclose(frame.hdu_list['SPEC_ERR'].data[0], expected_err) and
        frame.hdu_list['SPEC'].header['SLITCOR'] is True and
        np.isclose(frame.hdu_list['SPEC'].header['SLITFAC'], slit_factor)
    )
    print('\nconvert_spec_to_flux slit tuple: ', end='')
    print_pass() if result else print_fail()

    assert result


def test_apply_core_throughput_correction():
    # Build a mock spectrum and put WV0 inside the CT calibration grid
    spec_vals = np.array([10.0, 15.0, 20.0])
    original_spec = spec_vals.copy()
    spec_err = np.array([[0.5, 0.6, 0.7]])
    original_err = spec_err.copy()
    wave = np.linspace(700, 760, spec_vals.size)
    frame = make_1d_spec_image(spec_vals, spec_err, wave)
    frame.ext_hdr['WV0_X'] = 70.0
    frame.ext_hdr['WV0_Y'] = 0.0

    # Use CT calibration + FPAM/FSAM calibration
    ct_cal = create_ct_cal(fwhm_mas=50)
    fpam_fsam_cal = create_mock_fpamfsam_cal()

    frame.ext_hdr.setdefault('STARLOCX', 0.0)
    frame.ext_hdr.setdefault('STARLOCY', 0.0)

    # Convert WV0_X/Y (absolute EXCAM pixels) to FPM-relative coordinates
    # STARLOCX/Y is the FPM center during the coronagraphic observation
    fpm_center_x = frame.ext_hdr['STARLOCX']
    fpm_center_y = frame.ext_hdr['STARLOCY']
    wv0_x_relative = frame.ext_hdr['WV0_X'] - fpm_center_x
    wv0_y_relative = frame.ext_hdr['WV0_Y'] - fpm_center_y

    # Get the interpolated factor for this location to compare with the applied correction.
    # InterpolateCT expects coordinates relative to the FPM center
    ct_values = ct_cal.InterpolateCT(
        wv0_x_relative,
        wv0_y_relative,
        Dataset([frame.copy()]),
        fpam_fsam_cal,
    )
    ct_factor = np.asarray(ct_values).ravel()[0]

    # Apply correction
    applied_ct, corrected_frame = l4_to_tda.apply_core_throughput_correction(frame, ct_cal, fpam_fsam_cal)

    spec_ok = np.allclose(corrected_frame.hdu_list['SPEC'].data, original_spec / ct_factor)
    err_ok = np.allclose(corrected_frame.hdu_list['SPEC_ERR'].data[0], original_err[0] / ct_factor)
    applied_ok = np.isclose(applied_ct, ct_factor)

    print('\napply_core_throughput_correction: ', end='')
    if applied_ok and spec_ok and err_ok:
        print_pass()
    else:
        print_fail()

    assert applied_ok
    assert spec_ok
    assert err_ok
    assert frame.hdu_list['SPEC'].header['CTCOR'] is True
    assert np.isclose(frame.hdu_list['SPEC'].header['CTFAC'], ct_factor)


def test_compute_spec_flux_ratio_single_roll():
    """Flux ratio for one roll."""
    host_spec = np.array([10.0, 12.0, 14.0, 16.0])
    comp_spec = np.array([5.0, 6.0, 7.0, 8.0])
    spec_err = np.full((1, host_spec.size), 0.2)
    wave = np.linspace(700, 760, host_spec.size)

    host_ds = make_1d_spec_image(host_spec, spec_err, wave, roll='ROLL_A',
                                 exp_time=10.0, col_cor=True)
    comp_ds = make_1d_spec_image(comp_spec, spec_err, wave, roll='ROLL_B',
                                 exp_time=10.0, col_cor=True)

    host_ds.ext_hdr['FSMLOS'] = 0
    comp_ds.ext_hdr['FSMLOS'] = 1

    # Place the companion at a valid WV0 location 
    comp_ds.ext_hdr.setdefault('STARLOCX', 0.0)
    comp_ds.ext_hdr.setdefault('STARLOCY', 0.0)
    comp_ds.ext_hdr['WV0_X'] = 70.0
    comp_ds.ext_hdr['WV0_Y'] = 0.0
    fluxcal_factor = make_mock_fluxcal_factor(2.5, err=0.1)

    # Apply CT correction to comp
    ct_cal = create_ct_cal(fwhm_mas=50)
    fpam_fsam_cal = create_mock_fpamfsam_cal()
    applied, _ = l4_to_tda.apply_core_throughput_correction(comp_ds, ct_cal, fpam_fsam_cal)

    host_cal = l4_to_tda.convert_spec_to_flux(Dataset([host_ds]), fluxcal_factor)
    comp_cal = l4_to_tda.convert_spec_to_flux(Dataset([comp_ds]), fluxcal_factor)
    host_spec_flux = np.array(host_cal[0].hdu_list['SPEC'].data, dtype=float)
    comp_spec_flux = np.array(comp_cal[0].hdu_list['SPEC'].data, dtype=float)
    host_err_flux = np.squeeze(np.array(host_cal[0].hdu_list['SPEC_ERR'].data, dtype=float))
    comp_err_flux = np.squeeze(np.array(comp_cal[0].hdu_list['SPEC_ERR'].data, dtype=float))
    # Expected ratio uncertainty using the same propagation as compute_spec_flux_ratio
    ratio_err_expected = np.sqrt(
        (comp_err_flux / host_spec_flux) ** 2 +
        ((comp_spec_flux * host_err_flux) / (host_spec_flux ** 2)) ** 2
    )

    ratio, wavelength, metadata = l4_to_tda.compute_spec_flux_ratio(host_ds, comp_ds, fluxcal_factor)
    expected = comp_spec / host_spec

    result = (
        np.allclose(ratio, expected) and
        np.array_equal(wavelength, wave) and
        metadata['roll'] == 'ROLL_A' and
        metadata['companion_roll'] == 'ROLL_B' and
        np.allclose(metadata['ratio_err'], ratio_err_expected, equal_nan=True)
    )
    print('\ncompute_spec_flux_ratio single roll: ', end='')
    print_pass() if result else print_fail()

    assert result
    assert np.allclose(metadata['ratio_err'], ratio_err_expected, equal_nan=True)


def test_compute_spec_flux_ratio_weighted():
    """Combine spectra from multiple rolls, then compute a single flux ratio."""
    host_a = np.array([12.0, 14.0, 16.0, 18.0])
    comp_a = host_a * 0.5
    err_a = np.full((1, host_a.size), 0.3)
    wave_a = np.array([700.0, 710.0, 720.0, 730.0])

    host_ds_a = make_1d_spec_image(host_a, err_a, wave_a, roll='ROLL_A', exp_time=5.0, col_cor=True)
    comp_ds_a = make_1d_spec_image(comp_a, err_a, wave_a, roll='ROLL_A', exp_time=5.0, col_cor=True)

    host_b = np.array([8.0, 6.0, 4.0, 2.0])
    comp_b = np.array([1.0, 2.0, 3.0, 4.0])
    err_b = np.full((1, host_b.size), 0.4)
    wave_b_host = np.array([800.0, 780.0, 760.0, 740.0])
    wave_b_comp = np.array([790.0, 770.0, 750.0, 730.0])

    host_ds_b = make_1d_spec_image(host_b, err_b, wave_b_host, roll='ROLL_B', exp_time=15.0, col_cor=True)
    comp_ds_b = make_1d_spec_image(comp_b, err_b, wave_b_comp, roll='ROLL_B', exp_time=15.0, col_cor=True)

    fluxcal_factor = make_mock_fluxcal_factor(1.8, err=0.05)

    # Combine host spectra from rolls A and B (raw units)
    host_comb_spec, host_comb_wave, host_comb_err, host_rolls = l4_to_tda.combine_spectra(
        Dataset([host_ds_a, host_ds_b])
    )

    # Combine companion spectra from rolls A and B (raw units)
    comp_comb_spec, comp_comb_wave, comp_comb_err, comp_rolls = l4_to_tda.combine_spectra(
        Dataset([comp_ds_a, comp_ds_b])
    )

    # Build combined host and companion Images in raw units
    host_comb_image = make_1d_spec_image(
        host_comb_spec,
        host_comb_err.reshape(1, -1),
        host_comb_wave,
    )
    comp_comb_image = make_1d_spec_image(
        comp_comb_spec,
        comp_comb_err.reshape(1, -1),
        comp_comb_wave,
    )

    host_comb_image.ext_hdr['FSMLOS'] = 0
    comp_comb_image.ext_hdr['FSMLOS'] = 1

    # Apply core-throughput correction to the combined companion spectrum
    ct_cal = create_ct_cal(fwhm_mas=50)
    fpam_fsam_cal = create_mock_fpamfsam_cal()
    comp_comb_image.ext_hdr.setdefault('STARLOCX', 0.0)
    comp_comb_image.ext_hdr.setdefault('STARLOCY', 0.0)
    comp_comb_image.ext_hdr['WV0_X'] = 70.0
    comp_comb_image.ext_hdr['WV0_Y'] = 0.0
    _, _ = l4_to_tda.apply_core_throughput_correction(comp_comb_image, ct_cal, fpam_fsam_cal)

    # Compute flux-calibrated combined spectra to build the expected ratio and error
    host_cal = l4_to_tda.convert_spec_to_flux(Dataset([host_comb_image]), fluxcal_factor)
    comp_cal = l4_to_tda.convert_spec_to_flux(Dataset([comp_comb_image]), fluxcal_factor)

    host_flux = np.array(host_cal[0].hdu_list['SPEC'].data, dtype=float)
    comp_flux = np.array(comp_cal[0].hdu_list['SPEC'].data, dtype=float)
    host_err_flux = np.squeeze(np.array(host_cal[0].hdu_list['SPEC_ERR'].data, dtype=float))
    comp_err_flux = np.squeeze(np.array(comp_cal[0].hdu_list['SPEC_ERR'].data, dtype=float))

    # Compute flux ratio using the combined spectra (production path)
    ratio, wavelength, metadata = l4_to_tda.compute_spec_flux_ratio(
        host_comb_image, comp_comb_image, fluxcal_factor
    )

    # Expected ratio and uncertainty in flux units
    expected_ratio = comp_flux / host_flux
    expected_ratio_err = np.sqrt(
        (comp_err_flux / host_flux) ** 2
        + ((comp_flux * host_err_flux) / (host_flux ** 2)) ** 2
    )

    result = (
        np.allclose(ratio, expected_ratio, equal_nan=True)
        and np.array_equal(wavelength, host_comb_wave)
        and np.allclose(metadata['ratio_err'], expected_ratio_err, equal_nan=True)
    )
    print('\ncompute_spec_flux_ratio weighted rolls: ', end='')
    print_pass() if result else print_fail()

    assert result

def test_abs_fluxcal():
    """ 
    Generate a simulated image and test the flux calibration computation.
    
    """
    rel_tol_flux = 0.05

    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not
    datadir = os.path.join(os.path.dirname(__file__), "test_data", "sim_fluxcal")
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    # check that the results folder exists and create if not
    resdir = os.path.join(os.path.dirname(__file__), "test_data", "results")
    if not os.path.exists(resdir):
        os.mkdir(resdir)

    fwhm = 3
    flux_ratio = 200
    cal_factor = band_flux/flux_ratio
    # create a simulated mock image with a central point source + noise that
    # has a flux band_flux and a flux calibration factor band_flux/200
    # that results in a total extracted count of 200 photo electrons
    flux_image = create_flux_image(
        band_flux, fwhm, cal_factor, filter='3C', target_name='Vega',
        fsm_x=0.0, fsm_y=0.0, exptime=1.0, filedir=datadir, platescale=21.8,
        background=0, add_gauss_noise=True, noise_scale=1., file_save=True)
    # bunit needs to be photoelectron/s for later tests, so set that now
    flux_image.ext_hdr['BUNIT'] = 'photoelectron/s'
    assert isinstance(flux_image, Image)
    sigma = fwhm/(2.*np.sqrt(2*np.log(2)))
    radius = 3.* sigma
    
    #Test the aperture photometry
    #The error of one pixel is 1, so the error of the aperture sum should be: 
    error_sum = np.sqrt(np.pi * radius * radius)
    [flux_el_ap, flux_err_ap] = fluxcal.aper_phot(flux_image, radius, frac_enc_energy=0.997, method='subpixel', subpixels=5,
              background_sub=False, r_in=5, r_out=10, centering_method='xy', centroid_roi_radius=5)
    #200 is the input count of photo electrons
    assert flux_el_ap == pytest.approx(200, rel = 0.05)
    assert flux_err_ap == pytest.approx(error_sum, rel = 0.05)
    
    
    #Test the 2D Gaussian fit photometry
    #The error of one pixel is 1, so the error of a circular aperture with radius of 2 sigma should be about:
    error_gauss = np.sqrt(np.pi * 4 * sigma * sigma)
    flux_el_gauss, flux_err_gauss = fluxcal.phot_by_gauss2d_fit(flux_image, fwhm, fit_shape = 41)
    assert flux_el_gauss == pytest.approx(200, rel = 0.05)
    assert flux_err_gauss == pytest.approx(error_gauss, rel = 0.05)
    
    flux_el_gauss, flux_err_gauss = fluxcal.phot_by_gauss2d_fit(flux_image, fwhm)
    assert flux_el_gauss == pytest.approx(200, rel = 0.05)
    assert flux_err_gauss == pytest.approx(error_gauss, rel = 0.05)
    #test the generation of the flux cal factors cal file
    dataset = Dataset([flux_image])
    fluxcal_factor = fluxcal.calibrate_fluxcal_aper(dataset, flux_or_irr = 'flux', phot_kwargs=None)
    assert fluxcal_factor.filter == '3C'
    # band_flux/200 was the input calibration factor cal_factor of the
    # simulated mock image "alpha_lyr_stis_011.fits"
    test_result = fluxcal_factor.fluxcal_fac == pytest.approx(cal_factor, rel=rel_tol_flux)
    assert test_result
    # Print out the result
    print('\nFlux from fluxcal.calibrate_fluxcal_aper() is correct to within %.2f%% ***: ' % (rel_tol_flux*100), end='')
    print_pass() if test_result else print_fail()

    # divisive error propagation of the aperture phot error
    err_fluxcal_ap = band_flux/flux_el_ap**2*flux_err_ap
    assert fluxcal_factor.fluxcal_err == pytest.approx(err_fluxcal_ap)
    # TO DO: add this test back in when filename conventions are settled
    # assert fluxcal_factor.filename == 'mock_flux_image_Vega_0.0_0.0_FluxcalFactor_3C_ND475.fits'
    fluxcal_factor_gauss = fluxcal.calibrate_fluxcal_gauss2d(
        dataset, flux_or_irr='flux', phot_kwargs=None)
    assert fluxcal_factor_gauss.filter == '3C'
    test_result = fluxcal_factor_gauss.fluxcal_fac == pytest.approx(
        cal_factor, rel=rel_tol_flux)
    assert test_result
    # Print out the result
    print('\nFlux from fluxcal.calibrate_fluxcal_gauss2d() is correct to within %.2f%% ***: ' % (rel_tol_flux*100), end='')
    print_pass() if test_result else print_fail()

    # divisive error propagation of the 2D Gaussian fit phot error
    err_fluxcal_gauss = band_flux/flux_el_gauss**2*flux_err_gauss
    assert fluxcal_factor_gauss.fluxcal_err == pytest.approx(err_fluxcal_gauss)
    
    #test the flux conversion computation.
    old_ind = corgidrp.track_individual_errors
    corgidrp.track_individual_errors = True
    flux_dataset = Dataset([flux_image, flux_image])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # catch "there is no COL_COR keyword" from l4_to_tda
        output_dataset = l4_to_tda.convert_to_flux(flux_dataset, fluxcal_factor)
    assert len(output_dataset) == 2
    assert output_dataset[0].ext_hdr['BUNIT'] == "erg/(s*cm^2*AA)"
    assert output_dataset[0].ext_hdr['FLUXFAC'] == fluxcal_factor.fluxcal_fac
    assert "fluxcal_factor" in str(output_dataset[0].ext_hdr['HISTORY'])
    
    #this is the amplitude of the simulated 2D Gaussian mock image in the center
    #it should be close to the value in the output image center
    el_cen = flux_dataset[0].data[512,512]
    amplitude = el_cen * fluxcal_factor.fluxcal_fac
    data_cen = output_dataset[0].data[512, 512]
    assert data_cen == pytest.approx(amplitude)
    assert output_dataset[0].err_hdr["LAYER_2"] == "fluxcal_error"
    #this is the error propagation of the multiplication with the flux calibration factor
    #and should agree with the value of the output image
    flux_err_cen = flux_dataset[0].err[0,512,512]
    err_layer2 = fluxcal_factor.fluxcal_err * el_cen
    err_comb = np.sqrt((flux_err_cen * fluxcal_factor.fluxcal_fac)**2 + err_layer2**2)
    assert output_dataset[0].err[0,512, 512] == pytest.approx(err_comb)
    assert output_dataset[0].err[1,512, 512] == pytest.approx(err_layer2)
    
    #Test the optional background subtraction#
    flux_image_back = create_flux_image(band_flux, fwhm, cal_factor, filter='3C', target_name='Vega', fsm_x=0.0, 
                      fsm_y=0.0, exptime=1.0, filedir=datadir, platescale=21.8, background=3,
                      add_gauss_noise=True, noise_scale=1., file_save=True)
    # bunit needs to be photoelectron/s for later tests, so set that now
    flux_image_back.ext_hdr['BUNIT'] = 'photoelectron/s'
    
    [flux_back, flux_err_back, back] = fluxcal.aper_phot(flux_image_back, radius, frac_enc_energy=0.997, method='subpixel', subpixels=5,
              background_sub=True, r_in=5, r_out=10, centering_method='xy', centroid_roi_radius=5)

    #calculated median background should be close to the input
    assert back == pytest.approx(3, abs = 0.03)
    #the found values should be close to the ones without background subtraction
    assert flux_back == pytest.approx(flux_el_ap, abs = 1)
    assert flux_err_back == pytest.approx(flux_err_ap, abs = 0.03)

    [flux_back, flux_err_back, back] = fluxcal.phot_by_gauss2d_fit(flux_image_back, fwhm, fit_shape=None, background_sub=True, r_in=5,
                        r_out=10, centering_method='xy', centroid_roi_radius=5)
    #calculated median background should be close to the input
    assert back == pytest.approx(3, abs = 0.03)
    #the found values should be close to the ones without background subtraction
    assert flux_back == pytest.approx(flux_el_gauss, abs = 1)
    assert flux_err_back == pytest.approx(flux_err_gauss, abs = 0.03)
    
    #Also test again the generation of the cal file now with a background subtraction
    aper_kwargs = {
        "encircled_radius": radius,
        "frac_enc_energy": 0.997,
        "method": "subpixel",
        "subpixels": 10,
        "background_sub": True,
        "r_in": 5,
        "r_out": 10,
        "centering_method": "xy",
        "centroid_roi_radius": 5
    }
    gauss_kwargs = {
        'fwhm': fwhm,                  
        'fit_shape': None,            
        'background_sub': True,        
        'r_in': 5.0,                   
        'r_out': 10.0,                
        'centering_method': 'xy',     
        'centroid_roi_radius': 5       
    }
    fluxcal_factor_back = fluxcal.calibrate_fluxcal_aper(flux_image_back, flux_or_irr = 'flux', phot_kwargs=aper_kwargs)
    assert fluxcal_factor_back.fluxcal_fac == pytest.approx(fluxcal_factor.fluxcal_fac)
    assert fluxcal_factor_back.ext_hdr["LOCBACK"] == back
    assert 'alpha_lyr_stis_011.fits' in str (fluxcal_factor_back.ext_hdr['HISTORY'])
    fluxcal_factor_back_gauss = fluxcal.calibrate_fluxcal_gauss2d(flux_image_back, flux_or_irr = 'flux', phot_kwargs=gauss_kwargs)
    assert fluxcal_factor_back_gauss.fluxcal_fac == pytest.approx(fluxcal_factor_gauss.fluxcal_fac)
    assert fluxcal_factor_back_gauss.ext_hdr["LOCBACK"] == back
    assert 'alpha_lyr_stis_011.fits' in str (fluxcal_factor_back_gauss.ext_hdr['HISTORY'])
    
    #test the direct input of the calspec fits file
    fluxcal_factor_back = fluxcal.calibrate_fluxcal_aper(flux_image_back, calspec_file = calspec_filepath, flux_or_irr = 'flux', phot_kwargs=aper_kwargs)
    assert fluxcal_factor_back.fluxcal_fac == pytest.approx(fluxcal_factor.fluxcal_fac)
    assert 'alpha_lyr_stis_011.fits' in str (fluxcal_factor_back.ext_hdr['HISTORY'])
    fluxcal_factor_back_gauss = fluxcal.calibrate_fluxcal_gauss2d(flux_image_back, calspec_file = calspec_filepath, flux_or_irr = 'flux', phot_kwargs=gauss_kwargs)
    assert fluxcal_factor_back_gauss.fluxcal_fac == pytest.approx(fluxcal_factor_gauss.fluxcal_fac)
    assert 'alpha_lyr_stis_011.fits' in str (fluxcal_factor_back_gauss.ext_hdr['HISTORY'])
    
    # test l4_to_tda.determine_flux
    input_dataset = Dataset([flux_image_back, flux_image_back])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # catch "there is no COL_COR keyword" from l4_to_tda
        output_dataset = l4_to_tda.determine_flux(input_dataset, fluxcal_factor_back,  photo = "aperture", phot_kwargs = aper_kwargs)
    assert output_dataset[0].ext_hdr["FLUX"] == pytest.approx(band_flux)
    assert output_dataset[0].ext_hdr["LOCBACK"] == pytest.approx(3, abs = 0.03)
    #sanity check: vega is input source, so app mag 0
    assert output_dataset[0].ext_hdr["APP_MAG"] == pytest.approx(0.0)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # catch "there is no COL_COR keyword" from l4_to_tda
        output_dataset = l4_to_tda.determine_flux(input_dataset, fluxcal_factor_back_gauss,  photo = "2dgauss", phot_kwargs = gauss_kwargs)
    assert output_dataset[0].ext_hdr["FLUX"] == pytest.approx(band_flux)
    assert output_dataset[0].ext_hdr["LOCBACK"] == pytest.approx(3, abs = 0.03)
    #sanity check: Vega is input source, so app mag 0
    assert output_dataset[0].ext_hdr["APP_MAG"] == pytest.approx(0.0)
    
    #estimate of the error propagated to the final flux
    flux_err_ap = np.sqrt(error_sum**2 * fluxcal_factor.fluxcal_fac**2 + fluxcal_factor.fluxcal_err**2 * 200**2)
    flux_err_gauss = np.sqrt(error_gauss**2 * fluxcal_factor_gauss.fluxcal_fac**2 + fluxcal_factor_gauss.fluxcal_err**2 * 200**2)
    
    input_dataset = Dataset([flux_image, flux_image])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # catch "there is no COL_COR keyword" from l4_to_tda
        output_dataset = l4_to_tda.determine_flux(input_dataset, fluxcal_factor,  photo = "aperture", phot_kwargs = None)
    assert output_dataset[0].ext_hdr["FLUX"] == pytest.approx(band_flux)
    assert output_dataset[0].ext_hdr["FLUXERR"] == pytest.approx(flux_err_ap, rel = 0.1)
    assert output_dataset[0].ext_hdr["LOCBACK"] == 0
    mag_err_ap = 2.5/np.log(10) * flux_err_ap/band_flux
    
    assert output_dataset[0].ext_hdr["MAGERR"] == pytest.approx(mag_err_ap, rel = 0.1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # catch "there is no COL_COR keyword" from l4_to_tda
        output_dataset = l4_to_tda.determine_flux(input_dataset, fluxcal_factor_gauss,  photo = "2dgauss", phot_kwargs = None)
    assert output_dataset[0].ext_hdr["FLUX"] == pytest.approx(band_flux)
    assert output_dataset[0].ext_hdr["FLUXERR"] == pytest.approx(flux_err_gauss, rel = 0.1)
    assert output_dataset[0].ext_hdr["LOCBACK"] == 0
    mag_err_gauss = 2.5/np.log(10) * flux_err_gauss/band_flux
    
    assert output_dataset[0].ext_hdr["MAGERR"] == pytest.approx(mag_err_gauss, rel = 0.1)
    
    corgidrp.track_individual_errors = old_ind

def test_pol_abs_fluxcal():
    """ 
    Generate a simulated polarimetric image and test the flux calibration computation.
    Adapted from test_abs_fluxcal()
    
    """
    rel_tol_flux = 0.05

    # create a simulated polarimetric flux image
    # check that the simulated image folder exists and create if not
    datadir = os.path.join(os.path.dirname(__file__), "test_data", "sim_fluxcal")
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    # check that the results folder exists and create if not
    resdir = os.path.join(os.path.dirname(__file__), "test_data", "results")
    if not os.path.exists(resdir):
        os.mkdir(resdir)
    
    fwhm = 3
    flux_ratio = 400
    cal_factor = band_flux/flux_ratio
    #split flux by polarization
    band_flux_left = 0.6 * band_flux
    band_flux_right = 0.4 * band_flux
    # create a simulated mock images for WP1 and WP2
    #left PSF should have count of 240 photo electrons
    #right PSF should have count of 160 photo electons
    flux_image_WP1 = create_pol_flux_image(
        band_flux_left, band_flux_right, fwhm, cal_factor, filter='3C', dpamname='POL0', target_name='Vega',
        fsm_x=0.0, fsm_y=0.0, exptime=1.0, filedir=datadir, platescale=21.8,
        background=0, add_gauss_noise=True, noise_scale=1., file_save=True)
    flux_image_WP2 = create_pol_flux_image(
        band_flux_left, band_flux_right, fwhm, cal_factor, filter='3C', dpamname='POL45', target_name='Vega',
        fsm_x=0.0, fsm_y=0.0, exptime=1.0, filedir=datadir, platescale=21.8,
        background=0, add_gauss_noise=True, noise_scale=1., file_save=True)
    # bunit needs to be photoelectron/s for later tests, so set that now
    flux_image_WP1.ext_hdr['BUNIT'] = 'photoelectron/s'
    flux_image_WP2.ext_hdr['BUNIT'] = 'photoelectron/s'
    assert isinstance(flux_image_WP1, Image)
    assert isinstance(flux_image_WP2, Image)
    sigma = fwhm/(2.*np.sqrt(2*np.log(2)))
    radius = 3.* sigma

    #Test the aperture photometry for each polarization state
    #The error of one pixel is 1, so the error of the aperture sum should be: 
    error_sum = np.sqrt(np.pi * radius * radius)
    [flux_el_pol0, flux_err_pol0] = fluxcal.aper_phot(flux_image_WP1, radius, frac_enc_energy=0.997, method='subpixel', subpixels=5,
              background_sub=False, r_in=5, r_out=10, centering_method='xy', centroid_roi_radius=5, centering_initial_guess=(340, 512))
    [flux_el_pol90, flux_err_pol90] = fluxcal.aper_phot(flux_image_WP1, radius, frac_enc_energy=0.997, method='subpixel', subpixels=5,
              background_sub=False, r_in=5, r_out=10, centering_method='xy', centroid_roi_radius=5, centering_initial_guess=(684, 512))
    [flux_el_pol45, flux_err_pol45] = fluxcal.aper_phot(flux_image_WP2, radius, frac_enc_energy=0.997, method='subpixel', subpixels=5,
              background_sub=False, r_in=5, r_out=10, centering_method='xy', centroid_roi_radius=5, centering_initial_guess=(390, 634))
    [flux_el_pol135, flux_err_pol135] = fluxcal.aper_phot(flux_image_WP2, radius, frac_enc_energy=0.997, method='subpixel', subpixels=5,
              background_sub=False, r_in=5, r_out=10, centering_method='xy', centroid_roi_radius=5, centering_initial_guess=(634, 390))
    #240 is the input count of photo electrons for 0 and 45
    #160 is the input count of phot electrons for 90 and 135
    assert flux_el_pol0== pytest.approx(240, rel = 0.05)
    assert flux_el_pol45== pytest.approx(240, rel = 0.05)
    assert flux_el_pol90== pytest.approx(160, rel = 0.05)
    assert flux_el_pol135== pytest.approx(160, rel = 0.05)
    assert flux_err_pol0 == pytest.approx(error_sum, rel = 0.05)
    assert flux_err_pol45 == pytest.approx(error_sum, rel = 0.05)
    assert flux_err_pol90 == pytest.approx(error_sum, rel = 0.05)
    assert flux_err_pol135 == pytest.approx(error_sum, rel = 0.05)

    dataset_WP1 = Dataset([flux_image_WP1])
    dataset_WP2 = Dataset([flux_image_WP2])
    fluxcal_factor_WP1 = fluxcal.calibrate_pol_fluxcal_aper(dataset_WP1, 512, 512, flux_or_irr = 'flux', phot_kwargs=None)
    fluxcal_factor_WP2 = fluxcal.calibrate_pol_fluxcal_aper(dataset_WP2, 512, 512, flux_or_irr = 'flux', phot_kwargs=None)
    assert fluxcal_factor_WP1.filter == '3C'
    assert fluxcal_factor_WP2.filter == '3C'
    # band_flux/400 was the input calibration factor cal_factor of the
    # simulated mock image "alpha_lyr_stis_011.fits"
    test_result_WP1 = fluxcal_factor_WP1.fluxcal_fac == pytest.approx(cal_factor, rel=rel_tol_flux)
    test_result_WP2 = fluxcal_factor_WP2.fluxcal_fac == pytest.approx(cal_factor, rel=rel_tol_flux)
    assert test_result_WP1
    assert test_result_WP2

    print('\nWP1 polarimetric flux from fluxcal.calibrate_pol_fluxcal_aper() is correct to within %.2f%% ***: ' % (rel_tol_flux*100), end='')
    print_pass() if test_result_WP1 else print_fail()

    print('\nWP2 polarimetric flux from fluxcal.calibrate_pol_fluxcal_aper() is correct to within %.2f%% ***: ' % (rel_tol_flux*100), end='')
    print_pass() if test_result_WP2 else print_fail()

    # divisive error propagation of the aperture phot error
    err_fluxcal_WP1 = band_flux/(flux_el_pol0 + flux_el_pol90)**2*np.sqrt((flux_err_pol0)**2 + (flux_err_pol90)**2)
    err_fluxcal_WP2 = band_flux/(flux_el_pol45 + flux_el_pol135)**2*np.sqrt((flux_err_pol45)**2 + (flux_err_pol135)**2)
    assert fluxcal_factor_WP1.fluxcal_err == pytest.approx(err_fluxcal_WP1)
    assert fluxcal_factor_WP2.fluxcal_err == pytest.approx(err_fluxcal_WP2)

    #Test the optional background subtraction#
    flux_image_back_WP1 = create_pol_flux_image(
                      band_flux_left, band_flux_right, fwhm, cal_factor, filter='3C', dpamname='POL0', target_name='Vega', fsm_x=0.0, 
                      fsm_y=0.0, exptime=1.0, filedir=datadir, platescale=21.8, background=3,
                      add_gauss_noise=True, noise_scale=1., file_save=True)
    flux_image_back_WP2 = create_pol_flux_image(
                      band_flux_left, band_flux_right, fwhm, cal_factor, filter='3C', dpamname='POL45', target_name='Vega', fsm_x=0.0, 
                      fsm_y=0.0, exptime=1.0, filedir=datadir, platescale=21.8, background=3,
                      add_gauss_noise=True, noise_scale=1., file_save=True)
    # bunit needs to be photoelectron/s for later tests, so set that now
    flux_image_back_WP1.ext_hdr['BUNIT'] = 'photoelectron/s'
    flux_image_back_WP2.ext_hdr['BUNIT'] = 'photoelectron/s'
    [flux_back_pol0, flux_err_back_pol0, back_pol0] = fluxcal.aper_phot(flux_image_back_WP1, radius, frac_enc_energy=0.997, method='subpixel', subpixels=5,
              background_sub=True, r_in=5, r_out=10, centering_method='xy', centroid_roi_radius=5, centering_initial_guess=(340, 512))
    [flux_back_pol90, flux_err_back_pol90, back_pol90] = fluxcal.aper_phot(flux_image_back_WP1, radius, frac_enc_energy=0.997, method='subpixel', subpixels=5,
              background_sub=True, r_in=5, r_out=10, centering_method='xy', centroid_roi_radius=5, centering_initial_guess=(684, 512))
    [flux_back_pol45, flux_err_back_pol45, back_pol45] = fluxcal.aper_phot(flux_image_back_WP2, radius, frac_enc_energy=0.997, method='subpixel', subpixels=5,
              background_sub=True, r_in=5, r_out=10, centering_method='xy', centroid_roi_radius=5, centering_initial_guess=(390, 634))
    [flux_back_pol135, flux_err_back_pol135, back_pol135] = fluxcal.aper_phot(flux_image_back_WP2, radius, frac_enc_energy=0.997, method='subpixel', subpixels=5,
              background_sub=True, r_in=5, r_out=10, centering_method='xy', centroid_roi_radius=5, centering_initial_guess=(634, 390))
    #calculated median background should be close to the input
    assert back_pol0 == pytest.approx(3, rel = 0.05)
    assert back_pol45 == pytest.approx(3, rel = 0.05)
    assert back_pol90 == pytest.approx(3, rel = 0.05)
    assert back_pol135 == pytest.approx(3, rel = 0.05)
    #the found values should be close to the ones without background subtraction
    assert flux_back_pol0 == pytest.approx(flux_el_pol0, rel = 0.05)
    assert flux_back_pol45 == pytest.approx(flux_el_pol45, rel = 0.05)
    assert flux_back_pol90 == pytest.approx(flux_el_pol90, rel = 0.05)
    assert flux_back_pol135 == pytest.approx(flux_el_pol135, rel = 0.05)
    assert flux_err_back_pol0 == pytest.approx(flux_err_pol0, rel = 0.05)
    assert flux_err_back_pol45 == pytest.approx(flux_err_pol45, rel = 0.05)
    assert flux_err_back_pol90 == pytest.approx(flux_err_pol90, rel = 0.05)
    assert flux_err_back_pol135 == pytest.approx(flux_err_pol135, rel = 0.05)

    #Also test again the generation of the cal file now with a background subtraction
    aper_kwargs = {
        "encircled_radius": radius,
        "frac_enc_energy": 0.997,
        "method": "subpixel",
        "subpixels": 10,
        "background_sub": True,
        "r_in": 5,
        "r_out": 10,
        "centering_method": "xy",
        "centroid_roi_radius": 5
    }
    fluxcal_factor_back_WP1 = fluxcal.calibrate_pol_fluxcal_aper(flux_image_back_WP1, 512, 512, flux_or_irr = 'flux', phot_kwargs=aper_kwargs)
    fluxcal_factor_back_WP2 = fluxcal.calibrate_pol_fluxcal_aper(flux_image_back_WP2, 512, 512, flux_or_irr = 'flux', phot_kwargs=aper_kwargs)
    assert fluxcal_factor_back_WP1.fluxcal_fac == pytest.approx(fluxcal_factor_WP1.fluxcal_fac)
    assert fluxcal_factor_back_WP2.fluxcal_fac == pytest.approx(fluxcal_factor_WP2.fluxcal_fac)
    assert fluxcal_factor_back_WP1.ext_hdr["LOCBACK"] == 0.5 * (back_pol0 + back_pol90)
    assert fluxcal_factor_back_WP2.ext_hdr["LOCBACK"] == 0.5 * (back_pol45 + back_pol135)
    assert 'alpha_lyr_stis_011.fits' in str (fluxcal_factor_back_WP1.ext_hdr['HISTORY'])
    assert 'alpha_lyr_stis_011.fits' in str (fluxcal_factor_back_WP2.ext_hdr['HISTORY'])

    #test the direct input of the calspec fits file
    fluxcal_factor_back_WP1 = fluxcal.calibrate_pol_fluxcal_aper(flux_image_back_WP1, 512, 512, calspec_file = calspec_filepath, flux_or_irr = 'flux', phot_kwargs=aper_kwargs)
    fluxcal_factor_back_WP2 = fluxcal.calibrate_pol_fluxcal_aper(flux_image_back_WP2, 512, 512, calspec_file = calspec_filepath, flux_or_irr = 'flux', phot_kwargs=aper_kwargs)
    assert fluxcal_factor_back_WP1.fluxcal_fac == pytest.approx(fluxcal_factor_WP1.fluxcal_fac)
    assert fluxcal_factor_back_WP2.fluxcal_fac == pytest.approx(fluxcal_factor_WP2.fluxcal_fac)
    assert 'alpha_lyr_stis_011.fits' in str (fluxcal_factor_back_WP1.ext_hdr['HISTORY'])
    assert 'alpha_lyr_stis_011.fits' in str (fluxcal_factor_back_WP2.ext_hdr['HISTORY'])
    

def test_l4_companion_photometry():
    """VAP Test 3: Companion photometry + apparent magnitude validation."""
    output_dir = Path(__file__).resolve().parent / "pol_tda_companion_phot_output"
    
    # Clear output directory at the start of the test
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    
    # Set up logger
    logger, output_dir = _setup_vap_logger('test_l4_companion_photometry')
    
    logger.info('=' * 80)
    logger.info('Polarimetry L4-> TDA VAP Test 3: Companion Photometry / Apparent Magnitude')
    logger.info('=' * 80)
    phot_kwargs = {
        'encircled_radius': 5,
        'frac_enc_energy': 1.0,
        'method': 'subpixel',
        'subpixels': 5,
        'background_sub': False,
        'centering_method': 'xy',
        'centroid_roi_radius': 5,
    }

    host_counts = 2.5e5
    companion_counts = 5.0e4
    col_cor = 1.2
    host_image = create_mock_stokes_i_image(host_counts, 'HOST', seed=1, is_coronagraphic=False)
    time.sleep(1)  # Wait 1 second to get different filenames
    # Place companion off-center so it can be detected, but within CT calibration range (< 7 pixels from center)
    companion_image = create_mock_stokes_i_image(companion_counts, 'COMP', col_cor=col_cor, seed=2, is_coronagraphic=True, xoffset=5.0, yoffset=3.0)

    # Save input images
    logger.info('Saving input host and companion images to output directory')
    rename_files_to_cgi_format(list_of_fits=[host_image], output_dir=str(output_dir), level_suffix="l4")
    rename_files_to_cgi_format(list_of_fits=[companion_image], output_dir=str(output_dir), level_suffix="l4")

    host_intensity_ds = Dataset([get_stokes_intensity_image(host_image)])
    companion_intensity_image = get_stokes_intensity_image(companion_image)
    companion_intensity_ds = Dataset([companion_intensity_image])

    logger.info(f"Finding companion location in image")
    # Source is created with sigma=3.0, so FWHM = 2.355 * 3.0 = 7.07 pixels
    companion_i_image = l4_to_tda.find_source(companion_intensity_image, fwhm=7.0, nsigma_threshold=3.0)
    companion_info = companion_i_image.ext_hdr.get('snyx000', None)
    if companion_info:
        snr, companion_y, companion_x = map(float, companion_info.split(','))
        logger.info(f"Companion detected: SNR={snr:.1f}, location (x,y)=({companion_x:.1f},{companion_y:.1f}). PASS.")
    else:
        logger.warning("No companion detected by find_source. FAIL.")
        companion_x = None
        companion_y = None

    host_dataset = Dataset([host_image])
    companion_dataset = Dataset([companion_image])
    
    # Get FPM center position first (needed to create CT calibration at correct location)
    fpamfsam_cal = create_mock_fpamfsam_cal()
    # Create temporary CT cal to get FPM center
    temp_ct_cal = create_ct_cal(fwhm_mas=50, cfam_name='3C', cenx=0.0, ceny=0.0, nx=11, ny=11)
    fpm_center, fpm_center_fsam = temp_ct_cal.GetCTFPMPosition(companion_dataset, fpamfsam_cal)
    
    # Create CT calibration centered at the FPM center location
    ct_cal = create_ct_cal(fwhm_mas=50, cfam_name='3C', cenx=fpm_center[0], ceny=fpm_center[1], nx=11, ny=11)
    
    # Try to apply core throughput correction
    # InterpolateCT expects coordinates relative to FPM center, not absolute pixel coordinates 
    # (which is what companion_x and companion_y are)
    comp_x_relative = companion_x - fpm_center[0]
    comp_y_relative = companion_y - fpm_center[1]
    
    logger.info(f"Applying core throughput correction at companion location (x,y)=({comp_x_relative:.1f},{comp_y_relative:.1f}) relative to FPM center")
    try:
        ct_factor = np.asarray(
            ct_cal.InterpolateCT(comp_x_relative, comp_y_relative, companion_dataset, fpamfsam_cal)
        ).ravel()[0]
        if not np.isfinite(ct_factor) or ct_factor <= 0:
            logger.warning("Interpolated core throughput factor must be positive and finite. FAIL.")
        companion_image.data = companion_image.data / ct_factor
        companion_image.err = companion_image.err / ct_factor
        companion_image.ext_hdr['CTCOR'] = True
        companion_image.ext_hdr['CTFACT'] = ct_factor
        logger.info(f"Core throughput correction applied: CT factor = {ct_factor:.4f}. PASS.")
    except Exception as e:
        logger.warning(f"Could not apply core throughput correction: {e}. FAIL.")
        companion_image.ext_hdr['CTCOR'] = False
        ct_factor = 1.0  # Use 1 if correction fails
    fluxcal_factor = make_mock_fluxcal_factor(2.0, err=0.05)

    checks = []
    ext_hdr = companion_image.ext_hdr
    cgi_keys = ['CFAMNAME', 'DPAMNAME', 'LSAMNAME']
    checks.append(verify_header_keywords(ext_hdr, cgi_keys, frame_info="Companion header", logger=logger))
    checks.append(verify_header_keywords(ext_hdr, {'DATALVL': 'L4'}, frame_info="Companion header", logger=logger))
    checks.append(verify_header_keywords(ext_hdr, {'BUNIT': 'photoelectron/s'}, frame_info="Companion header", logger=logger))
    checks.append(verify_header_keywords(ext_hdr, {'CTCOR': True}, frame_info="Companion header", logger=logger))
    checks.append(verify_header_keywords(ext_hdr, {'CTFACT': ct_factor}, frame_info="Companion header", logger=logger))

    comp_intensity = companion_intensity_ds[0]
    host_intensity = host_intensity_ds[0]

    comp_ap, comp_ap_err = fluxcal.aper_phot(comp_intensity, **phot_kwargs)
    host_ap, host_ap_err = fluxcal.aper_phot(host_intensity, **phot_kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # catch "there is no COL_COR keyword" from l4_to_tda
        host_flux_ds = l4_to_tda.determine_flux(host_intensity_ds, fluxcal_factor, phot_kwargs=phot_kwargs)
        comp_flux_ds = l4_to_tda.determine_flux(companion_intensity_ds, fluxcal_factor, phot_kwargs=phot_kwargs)

    host_flux = host_flux_ds[0].ext_hdr['FLUX']
    comp_flux = comp_flux_ds[0].ext_hdr['FLUX']
    comp_flux_err = comp_flux_ds[0].ext_hdr['FLUXERR']
    flux_logged = comp_flux_ds[0].ext_hdr.get('FLUX')
    flux_details = f"FLUX={flux_logged:.3f}" if flux_logged is not None else "FLUX missing"
    message = "Companion flux recorded in header"
    condition = 'FLUX' in comp_flux_ds[0].ext_hdr
    logger.info(f"{message} | {flux_details}: {'PASS' if condition else 'FAIL'}")
    checks.append(condition)

    # Confirm the fluxcal factor (with color correction) predicts the measured companion flux
    factor = fluxcal_factor.fluxcal_fac / col_cor
    factor_err = fluxcal_factor.fluxcal_err / col_cor
    expected_comp_flux = comp_ap * factor
    expected_comp_flux_err = np.sqrt((comp_ap_err * factor) ** 2 + (comp_ap * factor_err) ** 2)
    message = "Companion flux matches expected scaling"
    condition = np.isclose(comp_flux, expected_comp_flux, rtol=5e-3)
    details = f"measured={comp_flux:.2f}, expected={expected_comp_flux:.2f}"
    logger.info(f"{message} | {details}: {'PASS' if condition else 'FAIL'}")
    checks.append(condition)

    # Confirm that enabling COL_COR actually reduces the measured flux relative to the aperture sum
    message = "Color correction applied when COL_COR present"
    condition = comp_flux < comp_ap * fluxcal_factor.fluxcal_fac
    details = f"flux_with_col_cor={comp_flux:.2f}, no_col_cor={comp_ap * fluxcal_factor.fluxcal_fac:.2f}"
    logger.info(f"{message} | {details}: {'PASS' if condition else 'FAIL'}")
    checks.append(condition)

    # Confirm that the flux uncertainty matches what is propagated from aperture and calibration errors
    message = "Flux uncertainty propagated from aperture sum"
    condition = np.isclose(comp_flux_err, expected_comp_flux_err, rtol=5e-3)
    details = f"measured={comp_flux_err:.2f}, expected={expected_comp_flux_err:.2f}"
    logger.info(f"{message} | {details}: {'PASS' if condition else 'FAIL'}")
    checks.append(condition)

    # Check that companion/host flux ratio matches the injected counts after color correction
    ratio_measured = comp_flux / host_flux
    ratio_expected = (comp_ap / col_cor) / host_ap
    ratio_tolerance = max(expected_comp_flux_err / comp_flux, 0.05)
    message = "Flux ratio matches expected value"
    condition = np.isclose(ratio_measured, ratio_expected, rtol=ratio_tolerance)
    details = f"measured={ratio_measured:.3f}, expected={ratio_expected:.3f}"
    logger.info(f"{message} | {details}: {'PASS' if condition else 'FAIL'}")
    checks.append(condition)

    filter_file = fluxcal.get_filter_name(comp_flux_ds[0])
    companion_mag = fluxcal.calculate_vega_mag(comp_flux, filter_file)
    companion_mag_err = 2.5 / np.log(10) * comp_flux_err / comp_flux
    expected_mag = fluxcal.calculate_vega_mag(expected_comp_flux, filter_file)
    expected_mag_err = 2.5 / np.log(10) * expected_comp_flux_err / expected_comp_flux

    # Convert the measured companion flux into a Vega magnitude and compare against expectation
    message = "Apparent magnitude derived from measured companion flux"
    condition = np.isclose(companion_mag, expected_mag, rtol=5e-3)
    details = f"measured={companion_mag:.3f}, expected={expected_mag:.3f}"
    logger.info(f"{message} | {details}: {'PASS' if condition else 'FAIL'}")
    checks.append(condition)

    # Check that magnitude uncertainty is the flux-error propagation scaled into magnitudes
    message = "Magnitude uncertainty propagated from flux error"
    condition = np.isclose(companion_mag_err, expected_mag_err, rtol=5e-3)
    details = f"measured={companion_mag_err:.3f}, expected={expected_mag_err:.3f}"
    logger.info(f"{message} | {details}: {'PASS' if condition else 'FAIL'}")
    checks.append(condition)

    result = all(checks)
    if result:
        logger.info('test_l4_companion_photometry overall: PASS')
    else:
        logger.info('test_l4_companion_photometry overall: FAIL')
    logger.info('=' * 80)
    logger.info('End of Polarimetry L4->TDA VAP Test 3')
    logger.info('=' * 80)
    print_pass() if result else print_fail()
    assert result

if __name__ == '__main__':
    test_get_filter_name()
    test_flux_calc()
    test_colorcor()
    test_calspec_download()
    test_app_mag()
    test_fluxcal_file()
    test_abs_fluxcal()
    test_pol_abs_fluxcal()
    test_convert_spec_to_flux_basic()
    test_convert_spec_to_flux_no_slit()
    test_convert_spec_to_flux_slit_scalar_map()
    test_apply_core_throughput_correction()
    test_compute_spec_flux_ratio_single_roll()
    test_compute_spec_flux_ratio_weighted()
    test_l4_companion_photometry()