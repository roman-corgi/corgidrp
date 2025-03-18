from corgidrp.mocks import create_psfsub_dataset,create_ct_cal
from corgidrp.klip_fm import meas_klip_thrupt, get_closest_psf
from corgidrp.l3_to_l4 import do_psf_subtraction
import pytest
import numpy as np

## Helper functions/quantities

iwa_lod = 3.
owa_lod = 9.7
d = 2.36 #m
lam = 573.8e-9 #m
pixscale_arcsec = 0.0218
fwhm_mas = 1.22 * lam / d * 206265 * 1000

iwa_pix = iwa_lod * lam / d * 206265 / pixscale_arcsec
owa_pix = owa_lod * lam / d * 206265 / pixscale_arcsec

st_amp = 100.
noise_amp=1e-11
pl_contrast=1e-4

# Mock CT calibration

ct_cal = create_ct_cal(fwhm_mas)
outdir = 'klipcal_output'
fileprefix = 'FAKE'
annuli = 1
subsections = 1
movement = 1
calibrate_flux = False
mode = 'ADI+RDI'

st_amp = 100.
noise_amp = 1e-6
pl_contrast = 1e-4
rolls = [0,90,0,0]
numbasis = [1,2]
inject_snr = 5
klip_params = {
            'outputdir':outdir,'fileprefix':fileprefix,
            'annuli':annuli, 'subsections':subsections, 
            'movement':movement, 'numbasis':numbasis,
            'mode':mode,'calibrate_flux':calibrate_flux}
        
mock_sci,mock_ref = create_psfsub_dataset(2,2,rolls,
                                        st_amp=st_amp,
                                        noise_amp=noise_amp,
                                        pl_contrast=pl_contrast)

psfsub_dataset = do_psf_subtraction(mock_sci,mock_ref,
                            numbasis=numbasis,
                            fileprefix='test_ADI+RDI',
                            do_crop=False,
                            measure_klip_thrupt=False)

## pyKLIP data class tests

def test_create_ct_cal():
    pass

def test_get_closest_psf():
    
    # Should get the first PSF
    cenx = 50.5
    ceny = 40.5
    goal_xy = (-100,-100)
    psf = get_closest_psf(ct_cal,cenx,ceny,*goal_xy)
    print(np.sum(psf))

    # Should get the last PSF
    goal_xy = (100,100)
    psf = get_closest_psf(ct_cal,cenx,ceny,*goal_xy)
    print(np.sum(psf))

def test_inject_psf():
    pass

def test_measure_noise():
    pass

def test_meas_klip_RDI():
    pass

def test_meas_klip_ADIRDI():

    

    meas_klip_thrupt(mock_sci, mock_ref, # pre-psf-subtracted dataset
                     psfsub_dataset, # post-subtraction dataset
                     ct_cal,
                     inject_snr,
                     numbasis,
                     seps=[20.], # in pixels from mask center
                     pas=[30.], # Degrees
                     )

    # See if it runs
    pass

if __name__ == '__main__':  
    test_get_closest_psf()
    test_meas_klip_ADIRDI()
    pass