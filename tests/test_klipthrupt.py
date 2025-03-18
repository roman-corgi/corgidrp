from corgidrp.mocks import create_psfsub_dataset,create_ct_cal
from corgidrp.klip_fm import meas_klip_thrupt, get_closest_psf
from corgidrp.l3_to_l4 import do_psf_subtraction
from corgidrp.astrom import centroid
import pytest
import numpy as np
import os

## Helper functions/quantities

iwa_lod = 3.
owa_lod = 9.7
d = 2.36 #m
lam = 573.8e-9 #m
pixscale_arcsec = 0.0218
fwhm_mas = 1.22 * lam / d * 206265 * 1000
fwhm_pix = fwhm_mas * 0.001 * pixscale_arcsec
iwa_pix = iwa_lod * lam / d * 206265 / pixscale_arcsec
owa_pix = owa_lod * lam / d * 206265 / pixscale_arcsec

st_amp = 100.
noise_amp=1e-11
pl_contrast=1e-4

# Mock CT calibration

outdir = 'klipcal_output'
fileprefix = 'FAKE'
annuli = 1
subsections = 1
movement = 1
calibrate_flux = False
mode = 'ADI+RDI'

if not os.path.exists(outdir):
    os.mkdir(outdir)

st_amp = 100.
noise_amp = 1e-6
pl_contrast = 1e-4
rolls = [0,90,0,0]
numbasis = [1,2]
inject_snr = 5
klip_params = {
            'outdir':outdir,'fileprefix':fileprefix,
            'annuli':annuli, 'subsections':subsections, 
            'movement':movement, 'numbasis':numbasis,
            'mode':mode,'calibrate_flux':calibrate_flux}
        

## pyKLIP data class tests

def test_create_ct_cal():

    nx,ny = (3,5)
    
    cenx,ceny = (10.5,20.5)

    n_psfs = nx * ny

    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    # Check that the correct number of psfs are created
    assert ctcal.ct_excam.shape == (3,n_psfs)

    # Check that each psf is the correct amplitude 
    # (scales with i for debugging purposes)
    for i in range(1,n_psfs+1):
        psf = ctcal.data[i-1]
        assert np.sum(psf) == pytest.approx(i,rel=0.01)

        # Check that psf is odd shape and is centered
        assert np.all(np.array(psf.shape) % 2 == 1)
        assert np.array(centroid(psf)) == pytest.approx(np.array(psf.shape)/2.-0.5)


def test_get_closest_psf():
    
    # Should get the first PSF
    nx,ny = (3,5)
    cenx,ceny = (10.5,20.5)
    n_psfs = nx * ny
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    goal_dxdy = (-100,-100)
    psf = get_closest_psf(ctcal,cenx,ceny,*goal_dxdy)
    assert psf == pytest.approx(ctcal.data[0])

    # Should get the last PSF
    goal_dxdy = (100,100)
    psf = get_closest_psf(ctcal,cenx,ceny,*goal_dxdy)
    assert psf == pytest.approx(ctcal.data[-1])

    # Should get the center PSF (at index 7)
    goal_dxdy = (0.,0.)
    psf = get_closest_psf(ctcal,cenx,ceny,*goal_dxdy)
    assert psf == pytest.approx(ctcal.data[7])

    # Should get the PSF at (9.5,21.5) (at index 3)
    goal_dxdy = (-1.2,1.1)
    psf = get_closest_psf(ctcal,cenx,ceny,*goal_dxdy)
    assert psf == pytest.approx(ctcal.data[3])


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
                     klip_params,
                     inject_snr,
                     seps=[15.], # in pixels from mask center
                     pas=[30.], # Degrees
                     )

    # See if it runs
    pass

if __name__ == '__main__':  
    test_create_ct_cal()
    
    test_get_closest_psf()
    
    # mock_sci,mock_ref = create_psfsub_dataset(2,2,rolls,
    #                                         st_amp=st_amp,
    #                                         noise_amp=noise_amp,
    #                                         pl_contrast=pl_contrast)

    # psfsub_dataset = do_psf_subtraction(mock_sci,
    #                             reference_star_dataset=mock_ref,
    #                             numbasis=numbasis,
    #                             fileprefix='test_ADI+RDI',
    #                             do_crop=False,
    #                             measure_klip_thrupt=False)
    # test_meas_klip_ADIRDI()
    pass