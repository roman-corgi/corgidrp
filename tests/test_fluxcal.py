import pytest
import os
import numpy as np
from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset
import corgidrp.fluxcal as fluxcal
import corgidrp.l4_to_tda as l4_to_tda
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
    output_dataset = l4_to_tda.determine_color_cor(dataset, calspec_name, calspec_name)
    assert output_dataset[0].ext_hdr['LAM_REF'] == lambda_piv
    assert output_dataset[0].ext_hdr['COL_COR'] == K
    
def test_calspec_download():
    """
    test the download of a calspec fits file
    """
    filepath = fluxcal.get_calspec_file('Vega')
    assert os.path.exists(filepath)
    os.remove(filepath)
    with pytest.raises(ValueError):
        filepath = fluxcal.get_calspec_file('Todesstern')
    

if __name__ == '__main__':
    test_get_filter_name()
    test_flux_calc()
    test_colorcor()
    test_calspec_download()




