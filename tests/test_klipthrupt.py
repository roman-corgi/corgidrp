from corgidrp.mocks import create_psfsub_dataset,create_ct_cal
from corgidrp.klip_fm import meas_klip_thrupt, get_closest_psf, inject_psf, measure_noise
from corgidrp.l3_to_l4 import do_psf_subtraction
from corgidrp.astrom import centroid
from corgidrp.data import Image
from astropy.io import fits
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
fwhm_pix = fwhm_mas * 0.001 / pixscale_arcsec
iwa_pix = iwa_lod * lam / d * 206265 / pixscale_arcsec
owa_pix = owa_lod * lam / d * 206265 / pixscale_arcsec

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
noise_amp = 1e-3
pl_contrast = 1e-2
rolls = [0,15.,0,0]
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
    ct_cenx,ct_ceny = (10.5,20.5)
    cenx,ceny = (10.0,20.0)
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=ct_cenx,ceny=ct_ceny,
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

    # Test injecting psf on edge of array
    
    # Set up frame of zeros
    # Need .ext_hdr['ROLL'] and .ext_hdr['STARLOCX/Y']

    nx,ny = (21,21)
    cenx, ceny = (25.,30.)
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    inj_flux = 10.


    # Test 0 separation to make sure we can scale and center
    sep = 0.0
    pa = 0.0
    roll = 0.0
    frame_shape = (50,60)
    expected_peak = (ceny,cenx)

    ext_hdr = fits.Header()
    ext_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape),
                  pri_hdr=fits.Header(),
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)
    
    assert np.sum(frame_out.data) == pytest.approx(inj_flux)

    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak
    
    # Test separation of 1 pixel
    sep = 1.0
    pa = 0.0
    roll = 0.0
    frame_shape = (50,60)
    expected_peak = (ceny+1,cenx)

    ext_hdr = fits.Header()
    ext_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape),
                  pri_hdr=fits.Header(),
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)
    
    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak

    # Test PA
    sep = 1.0
    pa = 90.0
    roll = 0.0
    frame_shape = (50,60)
    expected_peak = (ceny,cenx-1)

    ext_hdr = fits.Header()
    ext_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape),
                  pri_hdr=fits.Header(),
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)
    
    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak

    # Test Roll
    sep = 5.0
    pa = 0.0
    roll = 90.0
    frame_shape = (50,60)
    expected_peak = (ceny,cenx+5)

    ext_hdr = fits.Header()
    ext_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape),
                  pri_hdr=fits.Header(),
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)
    
    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak

    pass


def test_measure_noise():

    cenx,ceny = (120.,130.)
    frame_shape = (200,200)
    seps = np.arange(50.,71.,5.)
    border_sep = 62.5
    
    fwhm = 2. # pix

    ext_hdr = fits.Header()
    ext_hdr['MASKLOCX'] = cenx
    ext_hdr['MASKLOCY'] = ceny
    
    image = np.zeros((2,*frame_shape))

    frame = Image(image,
                  pri_hdr=fits.Header(),
                  ext_hdr=ext_hdr)

    # No KL mode specified
    noise_profile = measure_noise(frame, seps, fwhm)

    # Check data shape
    assert noise_profile.shape == (5,2)
    assert noise_profile == pytest.approx(0.)

    # Different noise values for each KL mode 
    # and inner/outer region

    std1 = 1.
    std2 = 2.
    std3 = 3.
    std4 = 4.

    image = np.zeros((2,*frame_shape))
    frame = Image(image,
                  pri_hdr=fits.Header(),
                  ext_hdr=ext_hdr)

    y, x = np.indices(frame.data.shape[1:])
    sep_map = np.sqrt((y-ceny)**2 + (x-cenx)**2)
    

    rng = np.random.default_rng(0)
    frame.data[0,:,:] = np.where(sep_map>border_sep,rng.normal(0.,std1,frame_shape),rng.normal(0.,std2,frame_shape))
    frame.data[1,:,:] = np.where(sep_map>border_sep,rng.normal(0.,std3,frame_shape),rng.normal(0.,std4,frame_shape))

    # No KL mode specified
    noise_profile = measure_noise(frame, seps, fwhm)

    # Check data shape
    assert noise_profile.shape == (5,2)

    # Check that std measurement is correct
    assert noise_profile[:3,0] == pytest.approx(std2,rel=0.05)
    assert noise_profile[3:,0] == pytest.approx(std1,rel=0.05)
    assert noise_profile[:3,1] == pytest.approx(std4,rel=0.05)
    assert noise_profile[3:,1] == pytest.approx(std3,rel=0.05)
    
    # Specify a particular KL mode
    noise_profile = measure_noise(frame, seps, fwhm,klmode_index=0)

    # Check data shape
    assert noise_profile.shape == (5,)
    assert noise_profile[:3] == pytest.approx(std2,rel=0.05)
    assert noise_profile[3:] == pytest.approx(std1,rel=0.05)

    # Specify a different particular KL mode
    noise_profile = measure_noise(frame, seps, fwhm,klmode_index=1)

    # Check data shape
    assert noise_profile.shape == (5,)
    assert noise_profile[:3] == pytest.approx(std4,rel=0.05)
    assert noise_profile[3:] == pytest.approx(std3,rel=0.05)

    pass


def test_meas_klip_RDI():
    pass


def test_meas_klip_ADIRDI():

    nx,ny = (21,21)
    cenx, ceny = (25.,30.)
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    mock_sci,mock_ref = create_psfsub_dataset(2,2,rolls,
                                            fwhm_pix=fwhm_pix,
                                            st_amp=st_amp,
                                            noise_amp=noise_amp,
                                            pl_contrast=pl_contrast)

    psfsub_dataset = do_psf_subtraction(mock_sci,
                                reference_star_dataset=mock_ref,
                                numbasis=numbasis,
                                fileprefix='test_ADI+RDI',
                                do_crop=False,
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False)

    meas_klip_thrupt(mock_sci, mock_ref, # pre-psf-subtracted dataset
                     psfsub_dataset, # post-subtraction dataset
                     ctcal,
                     klip_params,
                     inject_snr,
                     seps=[15.,25.], # in pixels from mask center
                     pas=[0.,60.,120.,180.,240.,300.], # Degrees
                     cand_locs=[(15.,0.)])

    # See if it runs
    pass

if __name__ == '__main__':  
    # test_create_ct_cal()
    
    # test_get_closest_psf()
    
    # test_inject_psf()

    # test_measure_noise()

    test_meas_klip_ADIRDI()

    pass