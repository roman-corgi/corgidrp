import pytest
import os
import numpy as np
from corgidrp.mocks import create_default_headers
from corgidrp.mocks import create_flux_image
from corgidrp.data import Image, Dataset
import corgidrp.fluxcal as fluxcal
import corgidrp.l4_to_tda as l4_to_tda
from astropy.modeling.models import BlackBody
import astropy.units as u
from astropy import wcs
from astropy.io.fits import file
from photutils.aperture import CircularAperture
from photutils.centroids import centroid_2dg
from photutils import psf
from astropy.coordinates import SkyCoord

data = np.ones([1024,1024]) * 2 
err = np.ones([1,1024,1024]) * 0.5
prhd, exthd = create_default_headers()
exthd["CFAMNAME"] = '3C'
image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err)
image2 = image1.copy()
dataset=Dataset([image1, image2])
calspec_filepath = os.path.join(os.path.dirname(__file__), "test_data", "bd_75d325_stis_006.fits")

def test_get_filter_name():
    """
    test that the correct filter curve file is selected
    """
    global wave
    global transmission
    filepath = fluxcal.get_filter_name(dataset)
    assert filepath.split("/")[-1] == 'transmission_ID-21_3C_v0.csv'
    
    wave, transmission = fluxcal.read_filter_curve(filepath)
    
    assert np.any(wave>=7130)
    assert np.any(transmission < 1.)
    
    #test a wrong filter name
    image3 = image1.copy()
    image3.ext_hdr["CFAMNAME"] = '5C'
    dataset2 = Dataset([image3, image3])
    with pytest.raises(Exception):
        filepath = pytest.fluxcal.get_filter_name(dataset2)
        pass
    

def test_flux_calc():
    """
    test that the calspec data is read correctly
    """
    calspec_flux = fluxcal.read_cal_spec(calspec_filepath, wave)
    assert calspec_flux[0] == pytest.approx(2e-13, 1e-15) 
    
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
    assert K_bb > 2
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
    

def test_abs_fluxcal():
    """ 
    Generate a simulated image and test the flux calibration computation.
    
    """
    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not
    datadir = os.path.join(os.path.dirname(__file__), "test_data", "sim_fluxcal")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    #weakest star to be detected
    flux = (30. * u.STmag).to(u.erg/u.s/u.cm**2/u.AA) 
    fwhm = 3
    #readnoise about 10
    flux_image = create_flux_image(flux.value * 10, fwhm, flux.value/20, filedir=datadir, file_save=True)
    assert type(flux_image) == Image
    sigma = fwhm/(2.*np.sqrt(2*np.log(2)))
    radius = 3.* sigma
        
    flux, flux_err = fluxcal.aper_phot(flux_image, radius, 0.997)
    assert flux == pytest.approx(200, abs = 6)
    assert flux_err == pytest.approx(8,1)
    
    flux, flux_err = fluxcal.phot_by_gauss2d_fit(flux_image, fwhm, fit_shape = 41)
    assert flux == pytest.approx(200, abs = 6)
    assert flux_err == pytest.approx(1, abs = 0.3)
    
    flux, flux_err = fluxcal.phot_by_gauss2d_fit(flux_image, fwhm)
    assert flux == pytest.approx(200, abs = 6)
    assert flux_err == pytest.approx(1, abs = 0.3)
    
if __name__ == '__main__':
    test_get_filter_name()
    test_flux_calc()
    test_colorcor()
    test_calspec_download()
    test_abs_fluxcal()




