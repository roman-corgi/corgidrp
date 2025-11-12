import pytest
import warnings
import os
import numpy as np
import corgidrp
from astropy.io import fits
from corgidrp.mocks import create_default_L3_headers
from corgidrp.mocks import create_flux_image
from corgidrp.mocks import create_pol_flux_image
from corgidrp.data import Image, Dataset, FluxcalFactor
import corgidrp.fluxcal as fluxcal
import corgidrp.l4_to_tda as l4_to_tda
from astropy.modeling.models import BlackBody
import astropy.units as u
from termcolor import cprint


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


def make_1d_spec_image(spec_values, spec_err, spec_wave, col_cor=None):
    """Create a mock L4 file with 1-D spectroscopy extensions.

    Args:
        spec_values (ndarray): flux values (photoelectron/s/bin) for `SPEC`.
        spec_err (ndarray): uncertainty array matching `SPEC` shape.
        spec_wave (ndarray): wavelength grid in nm for `SPEC_WAVE`.
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


def make_mock_fluxcal_factor(value, err=0.0):
    """Build a FluxcalFactor with minimal metadata for testing.

    Args:
        value (float): absolute flux calibration factor.
        err (float, optional): uncertainty on the calibration factor.

    Returns:
        corgidrp.data.FluxcalFactor: calibration object referencing a dummy
        dataset for history bookkeeping.
    """
    pri_hdr, ext_hdr, err_hdr, dq_hdr = create_default_L3_headers()
    ext_hdr['CFAMNAME'] = '3D'
    ext_hdr['DPAMNAME'] = 'PRISM3'
    ext_hdr['FSAMNAME'] = 'R1C2'
    dummy_data = np.zeros((2, 2))
    dummy_err = np.zeros((1, 2, 2))
    dummy_dq = np.zeros((2, 2), dtype=int)
    dummy_img = Image(dummy_data, pri_hdr=pri_hdr.copy(), ext_hdr=ext_hdr.copy(), err=dummy_err,
                      dq=dummy_dq, err_hdr=err_hdr.copy(), dq_hdr=dq_hdr.copy())
    return FluxcalFactor(value, err=err, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                          input_dataset=Dataset([dummy_img]))


def test_convert_spec_to_flux_basic():
    """Validate convert_spec_to_flux with slit correction and COL_COR applied."""
    spec_vals = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    spec_err = np.array([[0.5, 0.6, 0.7, 0.8, 0.9]])
    wave = np.linspace(700, 800, len(spec_vals))
    slit = np.array([0.8, 0.9, 1.0, 0.85, 0.95])

    image = make_1d_spec_image(spec_vals, spec_err, wave, col_cor=2.0)
    dataset = Dataset([image])
    fluxcal_factor = make_mock_fluxcal_factor(2.0, err=0.2)

    calibrated = l4_to_tda.convert_spec_to_flux(dataset, fluxcal_factor, slit_transmission=slit)
    frame = calibrated[0]
    spec_out = frame.hdu_list['SPEC'].data
    err_out = frame.hdu_list['SPEC_ERR'].data

    expected_spec = (spec_vals / slit) * (fluxcal_factor.fluxcal_fac / 2.0)
    expected_err = np.sqrt(((spec_err[0] / slit) * (fluxcal_factor.fluxcal_fac / 2.0))**2 +
                           (((spec_vals / slit) * fluxcal_factor.fluxcal_err / 2.0))**2)

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



