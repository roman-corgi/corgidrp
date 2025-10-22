from corgidrp.mocks import create_psfsub_dataset,create_ct_cal, gaussian_array
from corgidrp.klip_fm import meas_klip_thrupt, get_closest_psf, inject_psf, measure_noise
from corgidrp.l3_to_l4 import do_psf_subtraction, crop
from corgidrp.astrom import centroid, seppa2xy, create_circular_mask
from corgidrp.data import Image, Dataset
from astropy.io import fits
from scipy.ndimage import shift, rotate
from pyklip.fakes import gaussfit2d


import pytest
import numpy as np
import os

## Helper functions/quantities

lam = 573.8e-9 #m
d = 2.36 #m  
pixscale_mas = 0.0218 * 1000 # mas
fwhm_mas = 1.22 * lam / d * 206265. * 1000.
fwhm_pix = fwhm_mas / pixscale_mas  
res_elem = 3 * fwhm_pix # pix

owa_mas = 450. 
owa_pix = owa_mas / pixscale_mas   

iwa_mas = 140. 
iwa_pix = iwa_mas / pixscale_mas  

# Overriding separations for testing purposes
seps = np.array([10.,15.,20.,25.,30.])

# Mock CT calibration settings

nx,ny = (21,21)
cenx, ceny = (25.,30.)

# Mock PSF subtraction data settings
st_amp = 100.
noise_amp = 1e-3
pl_contrast = 0.0
rolls = [0,10.,0,0]

# Injection test settings
inj_flux = 10.

# KLIP throughput calculation settings
inject_snr = 10.
outdir = 'klipcal_output'
fileprefix = 'FAKE'
annuli = 1
subsections = 1
movement = 1
numbasis = [1]
calibrate_flux = False

# max_thrupt_tolerance = 1 + (noise_amp * (2*fwhm_pix)**2)
max_thrupt_tolerance = 1.05

if not os.path.exists(outdir):
    os.mkdir(outdir)
   
## pyKLIP data class tests

def test_create_ct_cal():
    """Test that mocks.create_ct_cal() generates the correct number of PSFs, 
    each with the correct shape, and that each PSF is a predictable amplitude 
    for debugging purposes.
    """

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
        assert np.max(psf) == pytest.approx(ctcal.ct_excam[2][i-1],rel=0.01)

        # Check that psf is odd shape and is centered
        assert np.all(np.array(psf.shape) % 2 == 1)
        assert np.array(centroid(psf)) == pytest.approx(np.array(psf.shape)/2.-0.5)


def test_get_closest_psf():
    """Test that the correct PSF is grabbed from the CT Calibration 
    object for each position.
    """
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
    """Test that PSFs are injected with the correct amplitude and peak pixel location
    """
    # Mock CT cal object
    nx,ny = (21,21)
    cenx, ceny = (25.,30.)
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny,
                  psfsize=5)

    # Test 0 separation to make sure we can scale and center
    sep = 0.0
    pa = 0.0
    roll = 0.0
    frame_shape_yx = (50,60)
    expected_peak = (ceny,cenx)

    pri_hdr=fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)
    
    assert np.max(frame_out.data) == pytest.approx(inj_flux)

    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak
    
    # Test separation of exactly 1 pixel
    sep = 1.0
    pa = 0.0
    roll = 0.0
    frame_shape_yx = (50,60)
    expected_peak = (ceny+1,cenx)

    pri_hdr=fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)
    
    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak

    # Test PA
    sep = 1.0
    pa = 90.0
    roll = 0.0
    frame_shape_yx = (50,60)
    expected_peak = (ceny,cenx-1)

    pri_hdr=fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)
    
    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak

    # Test Roll
    sep = 5.0
    pa = 0.0
    roll = 90.0
    frame_shape_yx = (50,60)
    expected_peak = (ceny,cenx+5)

    pri_hdr = fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)
    
    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak

    # Test injecting a psf over the left edge of the array
    sep = 9.0
    pa = 90.0
    roll = 0.0
    frame_shape_yx = (11,21)
    ceny,cenx = (5,10)
    expected_peak_yx = (5,1)

    pri_hdr=fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)

    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak_yx

    # Test injecting a psf over the right edge of the array
    sep = 9.0
    pa = -90.0
    roll = 0.0
    frame_shape_yx = (11,21)
    ceny,cenx = (5,10)
    expected_peak_yx = (5,19)

    pri_hdr=fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)

    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak_yx

    
    # Test injecting a psf over the top edge of the array
    sep = 4.0
    pa = 0.0
    roll = 0.0
    frame_shape_yx = (11,21)
    ceny,cenx = (5,10)
    expected_peak_yx = (9,10)

    pri_hdr=fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)

    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak_yx


    # Test injecting a psf over the bottom edge of the array
    sep = 4.0
    pa = 180.0
    roll = 0.0
    frame_shape_yx = (11,21)
    ceny,cenx = (5,10)
    expected_peak_yx = (1,10)

    pri_hdr=fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)

    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak_yx


    # Test injecting a psf over the bottom left corner of the array
    sep = 4.0 * np.sqrt(2)
    pa = 135.0
    roll = 0.0
    frame_shape_yx = (11,11)
    ceny,cenx = (5,5)
    expected_peak_yx = (1,1)

    pri_hdr=fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)

    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak_yx


    # Test injecting a psf over the bottom right corner of the array
    sep = 4.0 * np.sqrt(2)
    pa = -135.0
    roll = 0.0
    frame_shape_yx = (11,11)
    ceny,cenx = (5,5)
    expected_peak_yx = (1,9)

    pri_hdr=fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)

    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak_yx

    # Test injecting a psf over the top left corner of the array
    sep = 4.0 * np.sqrt(2)
    pa = 45.0
    roll = 0.0
    frame_shape_yx = (11,11)
    ceny,cenx = (5,5)
    expected_peak_yx = (9,1)

    pri_hdr=fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)

    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak_yx
    
    # Test injecting a psf over the top right corner of the array
    sep = 4.0 * np.sqrt(2)
    pa = -45.0
    roll = 0.0
    frame_shape_yx = (11,11)
    ceny,cenx = (5,5)
    expected_peak_yx = (9,9)

    pri_hdr=fits.Header()
    ext_hdr = fits.Header()
    pri_hdr['ROLL'] = roll
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    frame = Image(np.zeros(frame_shape_yx),
                  pri_hdr=pri_hdr,
                  ext_hdr=ext_hdr)

    frame_out, psf_model, psf_cenxy = inject_psf(frame, ctcal,inj_flux, sep,pa)

    assert np.unravel_index(np.argmax(frame_out.data),frame_out.data.shape) == expected_peak_yx


def test_measure_noise():
    """Check that annular noise profile measurement produces arrays with 
    the correct shape and values for a case with zero noise, and for a case 
    with different noise levels within and without a given radius of pixels.
    Also checks that klip_fm.measure_noise() can return results for only a 
    specific KL mode truncation if desired.
    """

    cenx,ceny = (120.,130.)
    frame_shape_yx = (200,200)
    seps = np.arange(50.,71.,5.)
    border_sep = 62.5
    
    fwhm = 2. # pix

    ext_hdr = fits.Header()
    ext_hdr['STARLOCX'] = cenx
    ext_hdr['STARLOCY'] = ceny
    
    image = np.zeros((2,*frame_shape_yx))

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

    image = np.zeros((2,*frame_shape_yx))
    frame = Image(image,
                  pri_hdr=fits.Header(),
                  ext_hdr=ext_hdr)

    y, x = np.indices(frame.data.shape[1:])
    sep_map = np.sqrt((y-ceny)**2 + (x-cenx)**2)
    

    rng = np.random.default_rng(0)
    frame.data[0,:,:] = np.where(sep_map>border_sep,rng.normal(0.,std1,frame_shape_yx),rng.normal(0.,std2,frame_shape_yx))
    frame.data[1,:,:] = np.where(sep_map>border_sep,rng.normal(0.,std3,frame_shape_yx),rng.normal(0.,std4,frame_shape_yx))

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


def test_meas_klip_ADI():
    """Checks that KLIP throughput measurement for ADI is always between 0 and 1 
    (within 5%) and that throughput increases with separation from the mask.
    """
    global kt_adi 

    mode = 'ADI'
    nsci, nref = (2,0)

    nx,ny = (21,21)
    cenx, ceny = (25.,30.)
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)

    klip_params = {
                'outdir':outdir,'fileprefix':fileprefix,
                'annuli':annuli, 'subsections':subsections, 
                'movement':movement, 'numbasis':numbasis,
                'mode':mode,'calibrate_flux':calibrate_flux}
    
    mock_sci,mock_ref = create_psfsub_dataset(nsci,nref,rolls,
                                            fwhm_pix=fwhm_pix,
                                            st_amp=st_amp,
                                            noise_amp=noise_amp,
                                            pl_contrast=pl_contrast)

    # # Plot input dataset
    # import matplotlib.pyplot as plt
    # dataset_frames = mock_sci if mock_ref is None else (*mock_sci,*mock_ref)
    # fig,axes = plt.subplots(1,len(dataset_frames),sharey=True,layout='constrained',figsize=(3*len(dataset_frames),3))
    # for f,frame in enumerate(dataset_frames):
    #     im0 = axes[f].imshow(frame.data,origin='lower')
    #     plt.colorbar(im0,ax=axes[f],shrink=0.8)
    #     axes[f].scatter(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'],s=1)
    #     name = 'Sci' if f < len(mock_sci) else 'Ref'
    #     n_frame = f if f < len(mock_sci) else f-len(mock_sci)
    #     axes[f].set_title(f'{name} Input Frame {n_frame}')
    # plt.show()
    klip_kwargs={
                'numbasis' : numbasis,
                'mode' : mode}
    psfsub_dataset = do_psf_subtraction(mock_sci,
                                reference_star_dataset=mock_ref,
                                fileprefix='test_KL_THRU',
                                do_crop=False,
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs)
    
    # # Plot Psf subtraction result
    # if psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'RDI':
    #     analytical_result = rotate(mock_sci[0].data - mock_ref[0].data,-rolls[0],reshape=False,cval=np.nan)
    # elif psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'ADI':
    #     analytical_result = shift((rotate(mock_sci[0].data - mock_sci[1].data,-rolls[0],reshape=False,cval=0) + rotate(mock_sci[1].data - mock_sci[0].data,-rolls[1],reshape=False,cval=0)) / 2,
    #                     [0.5,0.5],
    #                     cval=np.nan)
    # elif psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'ADI+RDI':
    #     analytical_result = (rotate(mock_sci[0].data - (mock_sci[1].data/2+mock_ref[0].data/2),-rolls[0],reshape=False,cval=0) + rotate(mock_sci[1].data - (mock_sci[0].data/2+mock_ref[0].data/2),-rolls[1],reshape=False,cval=0)) / 2
    # import matplotlib.pyplot as plt
    # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
    # im0 = axes[0].imshow(psfsub_dataset[0].data[0],origin='lower')
    # plt.colorbar(im0,ax=axes[0],shrink=0.8)
    # axes[0].set_title(f'Output data')
    # im1 = axes[1].imshow(analytical_result,origin='lower')
    # plt.colorbar(im1,ax=axes[1],shrink=0.8)
    # axes[1].set_title('Analytical result')
    # diff = psfsub_dataset[0].data[0] - analytical_result
    # im2 = axes[2].imshow(diff,origin='lower')
    # plt.colorbar(im2,ax=axes[2],shrink=0.8)
    # axes[2].set_title('Difference')
    # plt.suptitle(f'PSF Subtraction {psfsub_dataset[0].pri_hdr["KLIP_ALG"]} ({psfsub_dataset[0].ext_hdr["KLMODE0"]} KL Modes)')
    # plt.show()   

    klip_params['mode'] = mode
    kt_adi = meas_klip_thrupt(mock_sci, mock_ref, # pre-psf-subtracted dataset
                     psfsub_dataset, # post-subtraction dataset
                     ctcal,
                     klip_params,
                     inject_snr,
                     seps = seps, # in pixels from mask center
                     cand_locs=[])

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(figsize=(6,4))
    # plt.plot(kt_adi[0],kt_adi[1],label='ADI')
    # plt.title('KLIP throughput')
    # plt.legend()
    # plt.xlabel('separation (pixels)')
    # plt.show()

    # Check KL thrupt is <= 1 within noise tolerance
    assert np.all(kt_adi[1:,:,0] < max_thrupt_tolerance)

    # Check KL thrupt is > 0
    assert np.all(kt_adi[1:,:,0] > 0.)

    # Check KL thrupt increases with separation
    for i in range(1,len(kt_adi[0])):
        assert np.all(kt_adi[1:,i,0] > kt_adi[1:,i-1,0] - 0.05) # add a fudge factor of 5%

    # Check recovered FWHM increases with separation
    for i in range(1,len(kt_adi[0])):
        assert np.all(kt_adi[1:,i,1] > kt_adi[1:,i-1,1] - 0.1) # add fudge factor of 0.1 pixels


def test_meas_klip_RDI():
    """Checks that KLIP throughput measurement for RDI is always between 0.8 and 1 
    (within 5%).
    """
    global kt_rdi
    mode = 'RDI'
    nsci, nref = (1,1)

    nx,ny = (21,21)
    cenx, ceny = (25.,30.)
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)

    klip_params = {
                'outdir':outdir,'fileprefix':fileprefix,
                'annuli':annuli, 'subsections':subsections, 
                'movement':movement, 'numbasis':numbasis,
                'mode':mode,'calibrate_flux':calibrate_flux}
    
    mock_sci,mock_ref = create_psfsub_dataset(nsci,nref,rolls,
                                            fwhm_pix=fwhm_pix,
                                            st_amp=st_amp,
                                            noise_amp=noise_amp,
                                            pl_contrast=pl_contrast)

    # # Plot input dataset
    # import matplotlib.pyplot as plt
    # dataset_frames = mock_sci if mock_ref is None else (*mock_sci,*mock_ref)
    # fig,axes = plt.subplots(1,len(dataset_frames),sharey=True,layout='constrained',figsize=(3*len(dataset_frames),3))
    # for f,frame in enumerate(dataset_frames):
    #     im0 = axes[f].imshow(frame.data,origin='lower')
    #     plt.colorbar(im0,ax=axes[f],shrink=0.8)
    #     axes[f].scatter(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'],s=1)
    #     name = 'Sci' if f < len(mock_sci) else 'Ref'
    #     n_frame = f if f < len(mock_sci) else f-len(mock_sci)
    #     axes[f].set_title(f'{name} Input Frame {n_frame}')
    # plt.show()
    klip_kwargs={
                'numbasis' : numbasis,
                'mode' : mode}
    psfsub_dataset = do_psf_subtraction(mock_sci,
                                reference_star_dataset=mock_ref,
                                fileprefix='test_KL_THRU',
                                do_crop=False,
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs)

    klip_params['mode'] = mode
    kt_rdi = meas_klip_thrupt(mock_sci, mock_ref, # pre-psf-subtracted dataset
                     psfsub_dataset, # post-subtraction dataset
                     ctcal,
                     klip_params,
                     inject_snr,
                     seps=seps)

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(figsize=(6,4))
    # plt.plot(kt_rdi[0],kt_rdi[1],label='RDI')
    # plt.title('KLIP throughput')
    # plt.legend()
    # plt.xlabel('separation (pixels)')
    # plt.show()

    # Check KL thrupt is <= 1 within noise tolerance
    assert np.all(kt_rdi[1:,:,0] < max_thrupt_tolerance)

    # Check KL thrupt > 0.8
    assert np.all(kt_rdi[1:,:,0] > 0.8)


def test_meas_klip_ADIRDI():
    """Checks that KLIP throughput measurement for ADI+RDI is always between 0 and 1.
    """
    global kt_adirdi
    
    mode = 'ADI+RDI'
    nsci, nref = (2,1)

    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    klip_params = {
                'outdir':outdir,'fileprefix':fileprefix,
                'annuli':annuli, 'subsections':subsections, 
                'movement':movement, 'numbasis':numbasis,
                'mode':mode,'calibrate_flux':calibrate_flux}
    
    mock_sci,mock_ref = create_psfsub_dataset(nsci,nref,rolls,
                                            fwhm_pix=fwhm_pix,
                                            st_amp=st_amp,
                                            noise_amp=noise_amp,
                                            pl_contrast=pl_contrast)

    klip_kwargs={
        'numbasis' : numbasis,
        'mode' : mode}
    psfsub_dataset = do_psf_subtraction(mock_sci,
                                reference_star_dataset=mock_ref,
                                fileprefix='test_KL_THRU',
                                do_crop=False,
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs)

    kt_adirdi = meas_klip_thrupt(mock_sci, mock_ref, # pre-psf-subtracted dataset
                     psfsub_dataset, # post-subtraction dataset
                     ctcal,
                     klip_params,
                     inject_snr,
                     seps=seps)

    # Check KL thrupt is <= 1 within noise tolerance
    assert np.all(kt_adirdi[1:,:,0] < max_thrupt_tolerance)

    # Check KL thrupt is > 0
    assert np.all(kt_adirdi[1:,:,0] > 0.)


def test_compare_RDI_ADI():
    """Check that mean ADI throughput is lower than mean RDI throughput.
    """

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots()
    # ax.plot(kt_adi[0],kt_adi[1],label='ADI')
    # ax.plot(kt_rdi[0],kt_adirdi[1],label='ADI+RDI')
    # ax.plot(kt_rdi[0],kt_rdi[1],label='RDI')
    # plt.legend()
    # ax.set_ylim(-0.1,1.1)
    # plt.title('KLIP Throughput')
    # plt.xlabel('Separation (pixels)')
    # plt.show()

    # Check that ADI thrupt < RDI thrupt
    mean_adi = np.mean(kt_adi[1:,:,0])
    mean_rdi = np.mean(kt_rdi[1:,:,0])

    assert mean_adi < mean_rdi


def test_psfsub_withklipandctmeas_adi():
    """Check that KLIP throughput and CT calibration can be run as part 
    of the PSF subtraction step function. Check that KLIP throughput and 
    core throughput sample the same separations by default. Check that an 
    input planet flux (after CT) can be recovered via the KLIP throughput 
    within 1% error after ADI.
    """
    nsci, nref = (2,0) # Mode ADI
 
    st_amp = 100.
    noise_amp = 1e-3
    pl_contrast = 3e-4 
    pl_amp = st_amp * pl_contrast
    pl_loc = (20.,0.)
    est_pl_snr = pl_amp / noise_amp
    mock_sci,mock_ref = create_psfsub_dataset(nsci,nref,rolls,
                                            fwhm_pix=fwhm_pix,
                                            st_amp=st_amp,
                                            noise_amp=noise_amp,
                                            pl_contrast=pl_contrast,
                                            pl_sep=pl_loc[0],
                                )

    nx,ny = (21,21)
    cenx, ceny = (25.,30.)
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    klip_kwargs={
                'numbasis' : numbasis}
                                
    psfsub_dataset = do_psf_subtraction(mock_sci,ctcal,
                                reference_star_dataset=mock_ref,
                                fileprefix='test_KL_THRU',
                                do_crop=False,
                                measure_klip_thrupt=True,
                                measure_1d_core_thrupt=True,
                                cand_locs= [pl_loc],
                                kt_seps=[pl_loc[0]],
                                **klip_kwargs)
    

    # Plot Psf subtraction result
    # if psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'RDI':
    #     analytical_result = rotate(mock_sci[0].data - mock_ref[0].data,-rolls[0],reshape=False,cval=np.nan)
    # elif psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'ADI':
    #     analytical_result = shift((rotate(mock_sci[0].data - mock_sci[1].data,-rolls[0],reshape=False,cval=0) + rotate(mock_sci[1].data - mock_sci[0].data,-rolls[1],reshape=False,cval=0)) / 2,
    #                     [0.5,0.5],
    #                     cval=np.nan)
    # elif psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'ADI+RDI':
    #     analytical_result = (rotate(mock_sci[0].data - (mock_sci[1].data/2+mock_ref[0].data/2),-rolls[0],reshape=False,cval=0) + rotate(mock_sci[1].data - (mock_sci[0].data/2+mock_ref[0].data/2),-rolls[1],reshape=False,cval=0)) / 2
    # import matplotlib.pyplot as plt
    # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
    # im0 = axes[0].imshow(psfsub_dataset[0].data[0],origin='lower')
    # plt.colorbar(im0,ax=axes[0],shrink=0.8)
    # axes[0].set_title(f'Output data')
    # im1 = axes[1].imshow(analytical_result,origin='lower')
    # plt.colorbar(im1,ax=axes[1],shrink=0.8)
    # axes[1].set_title('Analytical result')
    # diff = psfsub_dataset[0].data[0] - analytical_result
    # im2 = axes[2].imshow(diff,origin='lower')
    # plt.colorbar(im2,ax=axes[2],shrink=0.8)
    # axes[2].set_title('Difference')
    # plt.suptitle(f'PSF Subtraction {psfsub_dataset[0].pri_hdr["KLIP_ALG"]} ({psfsub_dataset[0].ext_hdr["KLMODE0"]} KL Modes)')
    # plt.show()    

    # Check that klip and ct separations are the same
    kt = psfsub_dataset[0].hdu_list['KL_THRU'].data
    kt_seps = kt[0,:,0]

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(figsize=(6,4))
    # plt.scatter(kt[0],kt[1],label=psfsub_dataset[0].pri_hdr["KLIP_ALG"])
    # plt.title('KLIP throughput')
    # plt.legend()
    # plt.xlabel('separation (pixels)')
    # plt.show()

    ct = psfsub_dataset[0].hdu_list['CT_THRU'].data
    ct_seps = ct[0]

    assert np.all(kt_seps == ct_seps)

    # Fit 2d gaussian to planet location:

    # Crop data around location to be same as psf_model cutout
    locxy = seppa2xy(*pl_loc,psfsub_dataset[0].ext_hdr['STARLOCX'],psfsub_dataset[0].ext_hdr['STARLOCY'])

    # Measure background via sigma clip 
    n_loops = 5
    masked_data = psfsub_dataset[0].data.copy()[0]
    for n in range(n_loops):
        std = np.nanstd(masked_data)
        med = np.nanmedian(masked_data)
        clip_thresh = 3 * std
        masked_data = np.where(np.abs(masked_data-med)>clip_thresh,np.nan,masked_data)
    
    # Subtract median
    bg_level = np.nanmedian(masked_data)
    medsubtracted_data = psfsub_dataset[0].data[0] - bg_level

    # Crop the data, pad with nans if we're cropping over the edge
    cutout_shape = np.array([21,21])
    cutout = np.zeros(cutout_shape)
    cutoutcenyx = cutout_shape/2. - 0.5
    cutout[:] = np.nan
    cutout_starty, cutout_startx = (0,0)
    cutout_endy, cutout_endx = cutout.shape

    data_shape = medsubtracted_data.shape
    data_center_indyx = np.array([locxy[1],locxy[0]]).astype(int)
    data_start_indyx = (data_center_indyx - cutout_shape//2)
    data_end_indyx = (data_start_indyx + cutout_shape)
    data_starty,data_startx = data_start_indyx
    data_endy,data_endx = data_end_indyx
    
    if data_starty < 0:
        cutout_starty = -data_starty
        data_starty = 0
    
    if data_startx < 0:
        cutout_startx = -data_startx
        data_startx = 0
    
    if data_endy >= data_shape[0]:
        y_overhang = data_endy - medsubtracted_data.shape[0]
        cutout_endy = cutout_shape[0] - y_overhang
        data_endy = data_shape[0]

    if data_endx >= data_shape[1]:
        x_overhang = data_endx - medsubtracted_data.shape[1]
        cutout_endx = cutout_shape[1] - x_overhang
        data_endx = data_shape[1]

    cutout[cutout_starty:cutout_endy,
                cutout_startx:cutout_endx] = medsubtracted_data[data_starty:data_endy,
                                                    data_startx:data_endx]
    
    postklip_peak, post_fwhm, post_xfit, post_yfit = gaussfit2d(
                cutout, 
                cutoutcenyx[1], 
                cutoutcenyx[0], 
                searchrad=5, 
                guessfwhm=fwhm_pix, 
                guesspeak=pl_amp, 
                refinefit=True) 

    # import matplotlib.pyplot as plt
    # post_sigma = post_fwhm / (2 * np.sqrt(2. * np.log(2.)))
    # final_model = gaussian_array(array_shape=cutout_shape,
    #                              sigma=post_sigma,
    #                              amp=postklip_peak,
    #                              xoffset=post_xfit-10.,
    #                              yoffset=post_yfit-10.)
    # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
    # im0 = axes[0].imshow(cutout,origin='lower')
    # plt.colorbar(im0,ax=axes[0],shrink=0.8)
    # axes[0].set_title(f'Data')
    # im1 = axes[1].imshow(final_model,origin='lower')
    # plt.colorbar(im1,ax=axes[1],shrink=0.8)
    # axes[1].set_title('Model')
    # diff = cutout-final_model
    # im2 = axes[2].imshow(diff,origin='lower')
    # plt.colorbar(im2,ax=axes[2],shrink=0.8)
    # axes[2].set_title('Residuals')
    # plt.suptitle(f'Final PSF Fit')
    # plt.show()    

    
    pl_kt = kt[1,np.argmin(np.abs(kt_seps-pl_loc[0])),0]
    
    pl_counts = np.pi * pl_amp * fwhm_pix**2 / 4. / np.log(2.)
    recovered_pl_counts = np.pi * postklip_peak * post_fwhm**2 / 4. / np.log(2.)
    recovered_pl_counts_ktcorrected = recovered_pl_counts / pl_kt


    assert pl_counts == pytest.approx(recovered_pl_counts_ktcorrected,rel = 0.01) 


def test_psfsub_withklipandctmeas_rdi():
    """Check that KLIP throughput and CT calibration can be run as part 
    of the PSF subtraction step function. Check that KLIP throughput and 
    core throughput sample the same separations by default. Check that an 
    input planet flux (after CT) can be recovered via the KLIP throughput 
    within 5% error after RDI.
    """
    nsci, nref = (1,1) # Mode RDI
 
    st_amp = 100.
    noise_amp = 1e-3
    pl_contrast = 3e-3 
    pl_amp = st_amp * pl_contrast
    pl_loc = (20.,0.)
    est_pl_snr = pl_amp / noise_amp
    mock_sci,mock_ref = create_psfsub_dataset(nsci,nref,rolls,
                                            fwhm_pix=fwhm_pix,
                                            st_amp=st_amp,
                                            noise_amp=noise_amp,
                                            pl_contrast=pl_contrast,
                                            pl_sep=pl_loc[0],
                                )

    nx,ny = (21,21)
    cenx, ceny = (25.,30.)
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    klip_kwargs={
                'numbasis' : numbasis}
                                
    psfsub_dataset = do_psf_subtraction(mock_sci,ctcal,
                                reference_star_dataset=mock_ref,
                                fileprefix='test_KL_THRU',
                                do_crop=False,
                                measure_klip_thrupt=True,
                                measure_1d_core_thrupt=True,
                                cand_locs= [pl_loc],
                                kt_seps=[pl_loc[0]],
                                **klip_kwargs)
    

    # Plot Psf subtraction result
    
    if psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'RDI':
        analytical_result = rotate(mock_sci[0].data - mock_ref[0].data,-rolls[0],reshape=False,cval=np.nan)
    elif psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'ADI':
        analytical_result = shift((rotate(mock_sci[0].data - mock_sci[1].data,-rolls[0],reshape=False,cval=0) + rotate(mock_sci[1].data - mock_sci[0].data,-rolls[1],reshape=False,cval=0)) / 2,
                        [0.5,0.5],
                        cval=np.nan)
    elif psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'ADI+RDI':
        analytical_result = (rotate(mock_sci[0].data - (mock_sci[1].data/2+mock_ref[0].data/2),-rolls[0],reshape=False,cval=0) + rotate(mock_sci[1].data - (mock_sci[0].data/2+mock_ref[0].data/2),-rolls[1],reshape=False,cval=0)) / 2

    mask = create_circular_mask(analytical_result.shape[-2:],
                                r=3*fwhm_pix,
                                center=(psfsub_dataset[0].ext_hdr['STARLOCX'],
                                        psfsub_dataset[0].ext_hdr['STARLOCY']))
    masked_analytical_result = np.where(mask,np.nan,analytical_result)


    # Check that klip and ct separations are the same
    kt = psfsub_dataset[0].hdu_list['KL_THRU'].data
    kt_seps = kt[0,:,0]

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(figsize=(6,4))
    # plt.scatter(kt[0],kt[1],label=psfsub_dataset[0].pri_hdr["KLIP_ALG"])
    # plt.title('KLIP throughput')
    # plt.legend()
    # plt.xlabel('separation (pixels)')
    # plt.show()

    ct = psfsub_dataset[0].hdu_list['CT_THRU'].data
    ct_seps = ct[0]

    assert np.all(kt_seps == ct_seps)

    # Fit 2d gaussian to planet location:

    # Crop data around location to be same as psf_model cutout
    locxy = seppa2xy(*pl_loc,psfsub_dataset[0].ext_hdr['STARLOCX'],psfsub_dataset[0].ext_hdr['STARLOCY'])

    # Measure background via sigma clip 
    n_loops = 5
    masked_data = psfsub_dataset[0].data.copy()[0]
    for n in range(n_loops):
        std = np.nanstd(masked_data)
        med = np.nanmedian(masked_data)
        clip_thresh = 3 * std
        masked_data = np.where(np.abs(masked_data-med)>clip_thresh,np.nan,masked_data)
    
    # Subtract median
    bg_level = np.nanmedian(masked_data)
    medsubtracted_data = psfsub_dataset[0].data[0] #- bg_level

    # import matplotlib.pyplot as plt
    # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
    # im0 = axes[0].imshow(medsubtracted_data,origin='lower')
    # plt.colorbar(im0,ax=axes[0],shrink=0.8)
    # axes[0].set_title(f'BG-subtracted Output data')
    # im1 = axes[1].imshow(analytical_result,origin='lower')
    # plt.colorbar(im1,ax=axes[1],shrink=0.8)
    # axes[1].set_title('Analytical result')
    # diff = medsubtracted_data - masked_analytical_result
    # im2 = axes[2].imshow(diff,origin='lower')
    # plt.colorbar(im2,ax=axes[2],shrink=0.8)
    # axes[2].set_title('Difference')
    # plt.suptitle(f'PSF Subtraction {psfsub_dataset[0].pri_hdr["KLIP_ALG"]} ({psfsub_dataset[0].ext_hdr["KLMODE0"]} KL Modes)')
    # plt.show()    


    # Crop the data, pad with nans if we're cropping over the edge
    cutout_shape = np.array([21,21])
    cutout = np.zeros(cutout_shape)
    cutoutcenyx = cutout_shape/2. - 0.5
    cutout[:] = np.nan
    cutout_starty, cutout_startx = (0,0)
    cutout_endy, cutout_endx = cutout.shape

    data_shape = medsubtracted_data.shape
    data_center_indyx = np.array([locxy[1],locxy[0]]).astype(int)
    data_start_indyx = (data_center_indyx - cutout_shape//2)
    data_end_indyx = (data_start_indyx + cutout_shape)
    data_starty,data_startx = data_start_indyx
    data_endy,data_endx = data_end_indyx
    
    if data_starty < 0:
        cutout_starty = -data_starty
        data_starty = 0
    
    if data_startx < 0:
        cutout_startx = -data_startx
        data_startx = 0
    
    if data_endy >= data_shape[0]:
        y_overhang = data_endy - medsubtracted_data.shape[0]
        cutout_endy = cutout_shape[0] - y_overhang
        data_endy = data_shape[0]

    if data_endx >= data_shape[1]:
        x_overhang = data_endx - medsubtracted_data.shape[1]
        cutout_endx = cutout_shape[1] - x_overhang
        data_endx = data_shape[1]

    cutout[cutout_starty:cutout_endy,
                cutout_startx:cutout_endx] = medsubtracted_data[data_starty:data_endy,
                                                    data_startx:data_endx]
    

    postklip_peak, post_fwhm, post_xfit, post_yfit = gaussfit2d(
                cutout, 
                cutoutcenyx[1], 
                cutoutcenyx[0], 
                searchrad=5, 
                guessfwhm=fwhm_pix, 
                guesspeak=pl_amp, 
                refinefit=True) 
    post_sigma = post_fwhm / (2 * np.sqrt(2. * np.log(2.)))
    final_model = gaussian_array(array_shape=cutout_shape,
                                 sigma=post_sigma,
                                 amp=postklip_peak,
                                 xoffset=post_xfit-10.,
                                 yoffset=post_yfit-10.)
    

    diff = cutout-final_model

    # import matplotlib.pyplot as plt
    # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
    # im0 = axes[0].imshow(cutout,origin='lower')
    # plt.colorbar(im0,ax=axes[0],shrink=0.8)
    # axes[0].set_title(f'Data')
    # im1 = axes[1].imshow(final_model,origin='lower')
    # plt.colorbar(im1,ax=axes[1],shrink=0.8)
    # axes[1].set_title('Model')
    # im2 = axes[2].imshow(diff,origin='lower')
    # plt.colorbar(im2,ax=axes[2],shrink=0.8)
    # axes[2].set_title('Residuals')
    # plt.suptitle(f'Final PSF Fit')
    # plt.show()    
    
    pl_kt = kt[1,np.argmin(np.abs(kt_seps-pl_loc[0])),0]
    
    pl_counts = np.pi * pl_amp * fwhm_pix**2 / 4. / np.log(2.)
    recovered_pl_counts = np.pi * postklip_peak * post_fwhm**2 / 4. / np.log(2.)
    recovered_pl_counts_ktcorrected = recovered_pl_counts / pl_kt

    assert pl_counts == pytest.approx(recovered_pl_counts_ktcorrected,rel = 0.05) 


def test_psfsub_withKTandCTandCrop_adi():
    """Check that KLIP throughput, CT calibration, and crop step can be run as part 
    of the PSF subtraction step function. Check that KLIP throughput and 
    core throughput sample the same separations by default. Check that an 
    input planet flux (after CT) can be recovered via the KLIP throughput 
    within 1% error after ADI.
    """    
    nsci, nref = (2,0) # Mode ADI
 
    st_amp = 100.
    noise_amp = 1e-3
    pl_contrast = 3e-4 
    pl_amp = st_amp * pl_contrast
    pl_loc = (20.,0.)
    mock_sci,mock_ref = create_psfsub_dataset(nsci,nref,rolls,
                                            fwhm_pix=fwhm_pix,
                                            st_amp=st_amp,
                                            noise_amp=noise_amp,
                                            pl_contrast=pl_contrast,
                                            pl_sep=pl_loc[0],
                                )

    nx,ny = (21,21)
    cenx, ceny = (25.,30.)
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    klip_kwargs={'numbasis' : numbasis}
    psfsub_dataset = do_psf_subtraction(mock_sci,ctcal,
                                reference_star_dataset=mock_ref,                                
                                fileprefix='test_KL_THRU',
                                do_crop=True,
                                measure_klip_thrupt=True,
                                measure_1d_core_thrupt=True,
                                cand_locs= [pl_loc],
                                kt_seps=[pl_loc[0]],
                                **klip_kwargs)

    # # Plot Psf subtraction result
    # import matplotlib.pyplot as plt
    # if psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'RDI':
    #     analytical_result = rotate(mock_sci[0].data - mock_ref[0].data,-rolls[0],reshape=False,cval=np.nan)
    # elif psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'ADI':
    #     analytical_result = shift((rotate(mock_sci[0].data - mock_sci[1].data,-rolls[0],reshape=False,cval=0) + rotate(mock_sci[1].data - mock_sci[0].data,-rolls[1],reshape=False,cval=0)) / 2,
    #                     [0.5,0.5],
    #                     cval=np.nan)
    # elif psfsub_dataset[0].pri_hdr['KLIP_ALG'] == 'ADI+RDI':
    #     analytical_result = (rotate(mock_sci[0].data - (mock_sci[1].data/2+mock_ref[0].data/2),-rolls[0],reshape=False,cval=0) + rotate(mock_sci[1].data - (mock_sci[0].data/2+mock_ref[0].data/2),-rolls[1],reshape=False,cval=0)) / 2
    # prihdr = fits.Header()
    # exthdr = fits.Header()
    # exthdr["STARLOCX"] = mock_sci[0].ext_hdr["STARLOCX"]
    # exthdr["STARLOCY"] = mock_sci[0].ext_hdr["STARLOCY"]
    # exthdr["STARLOCX"] = mock_sci[0].ext_hdr["STARLOCX"]
    # exthdr["STARLOCY"] = mock_sci[0].ext_hdr["STARLOCY"]
    # exthdr.set("LSAMNAME",'NFOV')
    # ar_image = Image(analytical_result,
    #                  prihdr,
    #                  exthdr) 
    # ar_dataset_cropped = crop(Dataset([ar_image]))
    # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
    # im0 = axes[0].imshow(psfsub_dataset[0].data[0],origin='lower')
    # plt.colorbar(im0,ax=axes[0],shrink=0.8)
    # axes[0].set_title(f'Output data')
    # im1 = axes[1].imshow(ar_dataset_cropped[0].data,origin='lower')
    # plt.colorbar(im1,ax=axes[1],shrink=0.8)
    # axes[1].set_title('Analytical result')
    # diff = psfsub_dataset[0].data[0] - ar_dataset_cropped[0].data
    # im2 = axes[2].imshow(diff,origin='lower')
    # plt.colorbar(im2,ax=axes[2],shrink=0.8)
    # axes[2].set_title('Difference')
    # plt.suptitle(f'PSF Subtraction {psfsub_dataset[0].pri_hdr["KLIP_ALG"]} ({psfsub_dataset[0].ext_hdr["KLMODE0"]} KL Modes)')
    # plt.show()    

    # Check that klip and ct separations are the same
    kt = psfsub_dataset[0].hdu_list['KL_THRU'].data
    kt_seps = kt[0,:,0]

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(figsize=(6,4))
    # plt.scatter(kt[0],kt[1],label=psfsub_dataset[0].pri_hdr["KLIP_ALG"])
    # plt.title('KLIP throughput')
    # plt.legend()
    # plt.xlabel('separation (pixels)')
    # plt.show()

    ct = psfsub_dataset[0].hdu_list['CT_THRU'].data
    ct_seps = ct[0]

    assert np.all(kt_seps == ct_seps)

    # Fit 2d gaussian to planet location:

    # Crop data around location to be same as psf_model cutout
    locxy = seppa2xy(*pl_loc,psfsub_dataset[0].ext_hdr['STARLOCX'],psfsub_dataset[0].ext_hdr['STARLOCY'])

    # Measure background via sigma clip 
    n_loops = 5
    masked_data = psfsub_dataset[0].data.copy()[0]
    for n in range(n_loops):
        std = np.nanstd(masked_data)
        med = np.nanmedian(masked_data)
        clip_thresh = 3 * std
        masked_data = np.where(np.abs(masked_data-med)>clip_thresh,np.nan,masked_data)
    
    # Subtract median
    bg_level = np.nanmedian(masked_data)
    medsubtracted_data = psfsub_dataset[0].data[0] - bg_level

    # Crop the data, pad with nans if we're cropping over the edge
    cutout_shape = np.array([21,21])
    cutout = np.zeros(cutout_shape)
    cutoutcenyx = cutout_shape/2. - 0.5
    cutout[:] = np.nan
    cutout_starty, cutout_startx = (0,0)
    cutout_endy, cutout_endx = cutout.shape

    data_shape = medsubtracted_data.shape
    data_center_indyx = np.array([locxy[1],locxy[0]]).astype(int)
    data_start_indyx = (data_center_indyx - cutout_shape//2)
    data_end_indyx = (data_start_indyx + cutout_shape)
    data_starty,data_startx = data_start_indyx
    data_endy,data_endx = data_end_indyx
    
    if data_starty < 0:
        cutout_starty = -data_starty
        data_starty = 0
    
    if data_startx < 0:
        cutout_startx = -data_startx
        data_startx = 0
    
    if data_endy >= data_shape[0]:
        y_overhang = data_endy - medsubtracted_data.shape[0]
        cutout_endy = cutout_shape[0] - y_overhang
        data_endy = data_shape[0]

    if data_endx >= data_shape[1]:
        x_overhang = data_endx - medsubtracted_data.shape[1]
        cutout_endx = cutout_shape[1] - x_overhang
        data_endx = data_shape[1]

    cutout[cutout_starty:cutout_endy,
                cutout_startx:cutout_endx] = medsubtracted_data[data_starty:data_endy,
                                                    data_startx:data_endx]
    

    postklip_peak, post_fwhm, post_xfit, post_yfit = gaussfit2d(
                cutout, 
                cutoutcenyx[1], 
                cutoutcenyx[0], 
                searchrad=5, 
                guessfwhm=fwhm_pix, 
                guesspeak=pl_amp, 
                refinefit=True) 

    # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
    # post_sigma = post_fwhm / (2 * np.sqrt(2. * np.log(2.)))
    # final_model = gaussian_array(array_shape=cutout_shape,
    #                              sigma=post_sigma,
    #                              amp=postklip_peak,
    #                              xoffset=post_xfit-10.,
    #                              yoffset=post_yfit-10.)
    # im0 = axes[0].imshow(cutout,origin='lower')
    # plt.colorbar(im0,ax=axes[0],shrink=0.8)
    # axes[0].set_title(f'Data')
    # im1 = axes[1].imshow(final_model,origin='lower')
    # plt.colorbar(im1,ax=axes[1],shrink=0.8)
    # axes[1].set_title('Model')
    # diff = cutout-final_model
    # im2 = axes[2].imshow(diff,origin='lower')
    # plt.colorbar(im2,ax=axes[2],shrink=0.8)
    # axes[2].set_title('Residuals')
    # plt.suptitle(f'Final PSF Fit')
    # plt.show()    

    pl_kt = kt[1,np.argmin(np.abs(kt_seps-pl_loc[0])),0]
    
    pl_counts = np.pi * pl_amp * fwhm_pix**2 / 4. / np.log(2.)
    recovered_pl_counts = np.pi * postklip_peak * post_fwhm**2 / 4. / np.log(2.)
    recovered_pl_counts_ktcorrected = recovered_pl_counts / pl_kt

    assert pl_counts == pytest.approx(recovered_pl_counts_ktcorrected,rel=0.01)


if __name__ == '__main__':  
    test_create_ct_cal()
    test_get_closest_psf()
    test_inject_psf()
    test_measure_noise()

    test_meas_klip_ADI()
    test_meas_klip_RDI()
    test_meas_klip_ADIRDI()
    test_compare_RDI_ADI()

    test_psfsub_withklipandctmeas_adi()
    test_psfsub_withklipandctmeas_rdi()
    test_psfsub_withKTandCTandCrop_adi()
    
