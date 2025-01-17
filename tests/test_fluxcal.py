import pytest
import os
import numpy as np
import corgidrp
from corgidrp.mocks import create_default_headers
from corgidrp.mocks import create_flux_image
from corgidrp.data import Image, Dataset, FluxcalFactor
import corgidrp.fluxcal as fluxcal
import corgidrp.l4_to_tda as l4_to_tda
from astropy.modeling.models import BlackBody
import astropy.units as u

data = np.ones([1024,1024]) * 2 
err = np.ones([1,1024,1024]) * 0.5
prhd, exthd = create_default_headers()
exthd["CFAMNAME"] = '3C'
exthd["TARGET"] = 'VEGA'
image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err)
image2 = image1.copy()
dataset=Dataset([image1, image2])
calspec_filepath = os.path.join(os.path.dirname(__file__), "test_data", "alpha_lyr_stis_011.fits")

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
    dataset2 = Dataset([image3, image3])
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
    assert output_dataset[0].ext_hdr['LAM_REF'] == lambda_piv
    assert output_dataset[0].ext_hdr['COL_COR'] == pytest.approx(1,1e-2) 
    
def test_calspec_download():
    """
    test the download of a calspec fits file
    """
    filepath = fluxcal.get_calspec_file('Vega')
    assert os.path.exists(filepath)
    os.remove(filepath)
    filepath = fluxcal.get_calspec_file('TYC 4424-1286-1')
    assert os.path.exists(filepath)
    os.remove(filepath)
    
    with pytest.raises(ValueError):
        filepath = fluxcal.get_calspec_file('Todesstern')

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

def test_fluxcal_file():
    """ 
    Generate a mock fluxcal factor cal object and test the content and functionality.
    """
    fluxcal_factor = np.array([[2e-12]])
    fluxcal_factor_error = np.array([[[1e-14]]])
    fluxcal_fac = FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, input_dataset = dataset)
    assert fluxcal_fac.filter == '3C'
    assert fluxcal_fac.fluxcal_fac == fluxcal_factor[0,0]
    assert fluxcal_fac.fluxcal_err == fluxcal_factor_error[0,0,0]
    
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    filename = fluxcal_fac.filename
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    fluxcal_fac.save(filedir=calibdir, filename=filename)        
        
    fluxcal_filepath = os.path.join(calibdir, filename)

    fluxcal_fac_file = FluxcalFactor(fluxcal_filepath)
    assert fluxcal_fac_file.filter == '3C'
    assert fluxcal_fac_file.fluxcal_fac == fluxcal_factor[0,0]
    assert fluxcal_fac_file.fluxcal_err == fluxcal_factor_error[0,0,0]
    assert fluxcal_fac_file.ext_hdr["BUNIT"] == 'erg/(s * cm^2 * AA)/electron'

def test_abs_fluxcal():
    """ 
    Generate a simulated image and test the flux calibration computation.
    
    """
    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not
    datadir = os.path.join(os.path.dirname(__file__), "test_data", "sim_fluxcal")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    
    fwhm = 3
    color_cor = 0.9
    flux_image = create_flux_image(band_flux, fwhm, band_flux/200, color_cor = color_cor, filedir=datadir, file_save=True)
    assert type(flux_image) == Image
    sigma = fwhm/(2.*np.sqrt(2*np.log(2)))
    radius = 3.* sigma
        
    flux_el_ap, flux_err_ap = fluxcal.aper_phot(flux_image, radius, 0.997)
    assert flux_el_ap == pytest.approx(200, abs = 6)
    assert flux_err_ap == pytest.approx(8,1)
    
    flux_el_gauss, flux_err_gauss = fluxcal.phot_by_gauss2d_fit(flux_image, fwhm, fit_shape = 41)
    assert flux_el_gauss == pytest.approx(200, abs = 6)
    assert flux_err_gauss == pytest.approx(1, abs = 0.3)
    
    flux_el_gauss, flux_err_gauss = fluxcal.phot_by_gauss2d_fit(flux_image, fwhm)
    assert flux_el_gauss == pytest.approx(200, abs = 6)
    assert flux_err_gauss == pytest.approx(1, abs = 0.3)
    
    #test the generation of the flux cal factors cal file
    fluxcal_factor = fluxcal.calibrate_fluxcal_aper(flux_image, radius, 0.997)
    assert fluxcal_factor.filter == '3C'
    assert fluxcal_factor.fluxcal_fac == pytest.approx(band_flux/200, abs = 0.3e-12)
    assert fluxcal_factor.fluxcal_err == pytest.approx(3.1e-13, abs = 0.1e-13)
    assert fluxcal_factor.filename == 'sim_fluxcal_FluxcalFactor_3C.fits'
    fluxcal_factor = fluxcal.calibrate_fluxcal_gauss2d(flux_image, fwhm)
    assert fluxcal_factor.filter == '3C'
    assert fluxcal_factor.fluxcal_fac == pytest.approx(band_flux/200, abs =  0.3e-12)
    assert fluxcal_factor.fluxcal_err == pytest.approx(4.5e-14, abs = 0.1e-14)
    
    #test the flux conversion computation.
    corgidrp.track_individual_errors = True
    flux_dataset = Dataset([flux_image, flux_image])
    output_dataset = l4_to_tda.convert_to_flux(flux_dataset, fluxcal_factor)
    assert len(output_dataset) == 2
    assert output_dataset[0].ext_hdr['BUNIT'] == "erg/(s*cm^2*AA)"
    assert output_dataset[0].ext_hdr['FLUXFAC'] == fluxcal_factor.fluxcal_fac
    assert "fluxcal_factor" in str(output_dataset[0].ext_hdr['HISTORY'])
    output_dataset[0].save(filename = "test.fits")
    
    el_cen = flux_dataset[0].data[512,512]
    amplitude = band_flux/(2. * np.pi * sigma**2)
    data_cen = output_dataset[0].data[512, 512]
    assert data_cen == pytest.approx(amplitude, rel = 0.04)
    assert output_dataset[0].err_hdr["LAYER_2"] == "fluxcal_error"
    flux_err_cen = flux_dataset[0].err[0,512,512]
    err_layer2 = fluxcal_factor.fluxcal_err/color_cor * el_cen
    err_comb = np.sqrt((flux_err_cen/color_cor * fluxcal_factor.fluxcal_fac)**2 + err_layer2**2)
    assert output_dataset[0].err[0,512, 512] == err_comb
    assert output_dataset[0].err[1,512, 512] == err_layer2
    
    corgidrp.track_individual_errors = False

if __name__ == '__main__':
    test_get_filter_name()
    test_flux_calc()
    test_colorcor()
    test_calspec_download()
    test_app_mag()
    test_fluxcal_file()
    test_abs_fluxcal()



