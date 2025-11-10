from corgidrp.mocks import create_psfsub_dataset,create_ct_cal, create_default_L4_headers
from corgidrp.klip_fm import measure_noise
from corgidrp.l3_to_l4 import do_psf_subtraction
from corgidrp.l4_to_tda import compute_flux_ratio_noise
import corgidrp.data as data
import corgidrp.mocks as mocks

import pytest
import warnings
import numpy as np

## Helper functions/quantities

iwa_lod = 3.
owa_lod = 9.7
d = 2.36 #m
lam = 573.8e-9 #m
pixscale_arcsec = 0.0218
fwhm_mas = 1.22 * lam / d * 206265 * 1000
fwhm_pix = fwhm_mas * 0.001 / pixscale_arcsec
sig_pix = fwhm_pix / (2 * np.sqrt(2. * np.log(2.)))
iwa_pix = iwa_lod * lam / d * 206265 / pixscale_arcsec
owa_pix = owa_lod * lam / d * 206265 / pixscale_arcsec

# Mock CT calibration

annuli = 1
subsections = 1
movement = 1
calibrate_flux = False

st_amp = 100.
noise_amp = 1e-3
pl_contrast = 0.0
rolls = [0,15.,0,0]
numbasis = [1]

max_thrupt_tolerance = 1 + (noise_amp * (2*fwhm_pix)**2)


def test_expected_flux_ratio_noise():

    # RDI
    mode = 'RDI'
    nsci, nref = (1,1)
 
    st_amp = 100.
    noise_amp = 1e-3
    pl_contrast = 0. # No planet
    rolls = [0,15.,0,0]
    numbasis = [1,2]
    data_shape=(101,101)
    mock_sci_rdi,mock_ref_rdi = create_psfsub_dataset(nsci,nref,rolls,
                                            fwhm_pix=fwhm_pix,
                                            st_amp=st_amp,
                                            noise_amp=noise_amp,
                                            pl_contrast=pl_contrast,
                                            data_shape=data_shape)

    nx,ny = (21,21)
    cenx, ceny = (25.,30.)
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    klip_kwargs={'numbasis' : numbasis}
    psfsub_dataset_rdi = do_psf_subtraction(mock_sci_rdi,ctcal,
                                reference_star_dataset=mock_ref_rdi,
                                fileprefix='test_KL_THRU',
                                measure_klip_thrupt=True,
                                measure_1d_core_thrupt=True,
                                **klip_kwargs)
    
    # make unocculted star 
    x = np.arange(psfsub_dataset_rdi[0].data.shape[-1])
    y = np.arange(psfsub_dataset_rdi[0].data.shape[-2])
    X,Y = np.meshgrid(x,y)
    XY = np.vstack([X.ravel(),Y.ravel()])
    def gauss_spot(xy, A, x0, y0, sx, sy):
        (x, y) = xy
        return A*np.e**(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2)))
    star_amp = 100
    x0 = 15
    y0 = 17
    sig_x = 4
    sig_y = 4
    FWHM_star = 2*np.sqrt(2*np.log(2))*sig_x
    # expected flux of star, same for each frame of input dataset to compute_flux_ratio_noise:
    # integral under Gaussian times ND transmission
    Fs_expected = np.pi*star_amp*FWHM_star**2/(4*np.log(2)) * 1e-2
    star_PSF = np.reshape(gauss_spot(XY,star_amp,x0,y0,sig_x,sig_y), X.shape)
    # add some noise to the star 
    np.random.seed(987)
    star_PSF += np.random.poisson(lam=star_PSF.mean(), size=star_PSF.shape)
    prihdr, exthdr, errhdr, dqhdr = create_default_L4_headers()
    star_image = data.Image(star_PSF, prihdr, exthdr)
    star_dataset = data.Dataset([star_image for i in range(len(psfsub_dataset_rdi))])
    # fake an ND calibration:
    nd_x, nd_y = np.meshgrid(np.linspace(0, data_shape[1], 5), np.linspace(0, data_shape[0], 5))
    nd_x = nd_x.ravel()
    nd_y = nd_y.ravel()
    nd_od = np.ones(nd_y.shape) * 1e-2
    pri_hdr, ext_hdr, errhdr, dqhdr, biashdr = mocks.create_default_L2b_headers()
    nd_cal = data.NDFilterSweetSpotDataset(np.array([nd_od, nd_x, nd_y]).T, pri_hdr=pri_hdr,
                                      ext_hdr=ext_hdr)
    
    # now see what the step function gives, with and without a supplied star location guess:
    frn_dataset_nostarloc = compute_flux_ratio_noise(psfsub_dataset_rdi, nd_cal, star_dataset, halfwidth=3)
    frn_dataset_starloc = compute_flux_ratio_noise(psfsub_dataset_rdi, nd_cal, star_dataset, unocculted_star_loc=np.array([[17],[15]]), halfwidth=3)

    for frn_dataset in [frn_dataset_nostarloc, frn_dataset_starloc]:
        for frame in frn_dataset:
            flux_ratio_noise = frame.hdu_list['FRN_CRV'].data
            frn_seps = flux_ratio_noise[0]
            klip = frame.hdu_list['KL_THRU'].data
            separations = klip[0,:,0]
            klip_tp = klip[1:,:,0]
            klip_fwhms = frame.hdu_list['KL_THRU'].data[1:,:,1]
            core_tp = frame.hdu_list['CT_THRU'].data[1]
            annular_noise = measure_noise(frame, separations, hw=3)
            noise_amp = annular_noise.T
            # expected planet flux: Guassian integral
            Fp_expected = np.pi*noise_amp*klip_fwhms**2/(4*np.log(2))
            # expected flux ratio noise, adjusting for KLIP and core throughputs
            frn_expected = (Fp_expected/Fs_expected)/(core_tp*klip_tp)
            flux_ratio_noise = frame.hdu_list['FRN_CRV'].data
            frn_seps = flux_ratio_noise[0]
            assert np.array_equal(frn_seps, separations)
            assert np.allclose(flux_ratio_noise[2:], frn_expected, rtol=0.05)
            assert 'FRN_CRV' in frn_dataset[0].ext_hdr["HISTORY"][-1]
            assert frn_dataset[0].hdu_list['FRN_CRV'].header['BUNIT'] == 'Fp/Fs'

    # last 2 separations below close to what they are in the KLIP extension, and increased the length from 2 separations in KLIP extension to 3 here to make sure that can be interpolated and handled
    requested_separations = np.array([26.5, 6.4, 14.8])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # catch Not all requested_separations from l4_to_tda
        frn_dataset_rseps = compute_flux_ratio_noise(psfsub_dataset_rdi, nd_cal, star_dataset, requested_separations=requested_separations, halfwidth=3)
    for frame in frn_dataset_rseps:
        flux_ratio_noise = frame.hdu_list['FRN_CRV'].data
        frn_seps = flux_ratio_noise[0]
        klip = frame.hdu_list['KL_THRU'].data
        klip_tp = klip[1:,:,0]
        klip_fwhms = frame.hdu_list['KL_THRU'].data[1:,:,1]
        core_tp = frame.hdu_list['CT_THRU'].data[1]
        annular_noise = measure_noise(frame, requested_separations, hw=3)
        # the last 2 separation values should be close to what is expected from KLIP non-interpolated values
        noise_amp = annular_noise.T[:,1:]
        # expected planet flux: Guassian integral
        Fp_expected = np.pi*noise_amp*klip_fwhms**2/(4*np.log(2))
        # expected flux ratio noise, adjusting for KLIP and core throughputs
        frn_expected = (Fp_expected/Fs_expected)/(core_tp*klip_tp)
        flux_ratio_noise = frame.hdu_list['FRN_CRV'].data
        frn_seps = flux_ratio_noise[0]
        assert np.array_equal(frn_seps, requested_separations)
        # the last 2 separation values should be close to what is expected from KLIP non-interpolated values
        assert np.allclose(flux_ratio_noise[2:,1:], frn_expected, rtol=0.05)

    # halfwidth from the KLIP extension data's 2 separations used below
    frn_dataset_hw = compute_flux_ratio_noise(psfsub_dataset_rdi, nd_cal, star_dataset, halfwidth=None)
    frn_dataset_true_hw = compute_flux_ratio_noise(psfsub_dataset_rdi, nd_cal, star_dataset, halfwidth=8.41973985/2)
    frn_hw = frn_dataset_hw.frames[0].hdu_list['FRN_CRV'].data
    frn_true_hw = frn_dataset_true_hw.frames[0].hdu_list['FRN_CRV'].data
    assert np.array_equal(frn_hw, frn_true_hw)
    
    #test inputs
    # number of frames in input_dataset and unocculted_star_dataset should be equal
    bad_star_dataset = data.Dataset([star_image for i in range(len(psfsub_dataset_rdi)+2)])
    with pytest.raises(ValueError):
        compute_flux_ratio_noise(psfsub_dataset_rdi, nd_cal, bad_star_dataset)
    # requested_separations includes at least 1 value outside the range covered by the KLIP extension header, so extrapolation is used
    with pytest.warns(UserWarning):
        compute_flux_ratio_noise(psfsub_dataset_rdi, nd_cal, star_dataset, requested_separations=np.array([4, 13, 17]))
    # bad halfwidth input value
    with pytest.raises(ValueError):
        compute_flux_ratio_noise(psfsub_dataset_rdi, nd_cal, star_dataset, halfwidth=0)
    # Halfwidth is wider than half the minimum spacing between separation values
    with pytest.warns(UserWarning):
        compute_flux_ratio_noise(psfsub_dataset_rdi, nd_cal, star_dataset, halfwidth=100)
    # star location guess outside array bounds
    with pytest.raises(IndexError):
        compute_flux_ratio_noise(psfsub_dataset_rdi, nd_cal, star_dataset, unocculted_star_loc=np.array([[50],[101]]))
    

if __name__ == '__main__':  
    test_expected_flux_ratio_noise()

    pass