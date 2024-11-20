import pytest
import os
import numpy as np
from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset
import corgidrp.fluxcal as fluxcal
from astropy.modeling.models import BlackBody
import astropy.units as u

data = np.ones([1024,1024]) * 2 
err = np.ones([1,1024,1024]) * 0.5
prhd, exthd = create_default_headers()
exthd["CFAMNAME"] = '3C'
image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err)
image2 = image1.copy()
dataset=Dataset([image1, image2])
calspec_filepath = os.path.join(os.path.dirname(__file__), "test_data", "bd_75d325_stis_006.fits")

def test_get_filter_name():
    """test that the correct filter curve file is selected"""
    global wave
    global transmission
    filepath = fluxcal.get_filter_name(dataset)
    assert filepath.split("/")[-1] == 'transmission_ID-21_3C_v0.csv'
    
    wave, transmission = fluxcal.read_filter_curve(filepath)
    
    assert np.any(wave>=713)
    assert np.any(transmission < 1.)

def test_flux_calc():
    """test that the calspec data is read correctly"""
    calspec_flux = fluxcal.read_cal_spec(calspec_filepath, wave)
    print(wave)
    print(calspec_flux)
    
    assert calspec_flux[0] == pytest.approx(2e-15, 1e-16) 
    
    band_flux = fluxcal.calculate_band_flux(transmission, calspec_flux, wave)
    print(band_flux)
    eff_lambda = fluxcal.calculate_effective_lambda(transmission, calspec_flux, wave)
    print(eff_lambda)
    assert eff_lambda == pytest.approx((wave[0]+wave[-1])/2., 0.3)
    
def test_colorcor():
    """test that the pivot reference wavelengths is close to the center of the bandpass"""
    lambda_piv = fluxcal.calculate_pivot_lambda(transmission, wave)
    print(lambda_piv)
    print((wave[0]+wave[-1])/2.)
    assert lambda_piv == pytest.approx((wave[0]+wave[-1])/2., 0.3)
    
    calspec_flux = fluxcal.read_cal_spec(calspec_filepath, wave)
    ## BB of an O5 star
    bbscale = 1e-9 * u.Watt / (u.m**2 * u.nm * u.steradian)
    flux_source = BlackBody(scale=bbscale, temperature=54000.0 * u.K)
    print (flux_source(wave))
    K = fluxcal.compute_colorcor(transmission, wave, calspec_flux, lambda_piv, flux_source(wave))
    print (K)
    assert K == pytest.approx(1., 0.01)
    #sanity check
    K = fluxcal.compute_colorcor(transmission, wave, calspec_flux, lambda_piv, 10 * calspec_flux)
    assert K == 1 

if __name__ == '__main__':
    test_get_filter_name()
    test_flux_calc()
    test_colorcor()




