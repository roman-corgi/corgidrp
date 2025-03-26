from corgidrp.mocks import create_psfsub_dataset,create_ct_cal
from corgidrp.klip_fm import measure_noise
from corgidrp.l3_to_l4 import do_psf_subtraction
from corgidrp.l4_to_tda import compute_flux_ratio_noise

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
sig_pix = fwhm_pix / (2 * np.sqrt(2. * np.log(2.)))
iwa_pix = iwa_lod * lam / d * 206265 / pixscale_arcsec
owa_pix = owa_lod * lam / d * 206265 / pixscale_arcsec

# Mock CT calibration

outdir = 'klipcal_output'
fileprefix = 'FAKE'
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

if not os.path.exists(outdir):
    os.mkdir(outdir)
   
## pyKLIP data class tests

def test_expected_contrast():

    # RDI
    mode = 'RDI'
    nsci, nref = (1,1)
 
    st_amp = 100.
    noise_amp = 1e-3
    pl_contrast = 0. # No planet
    rolls = [0,15.,0,0]
    numbasis = [1,2]
    mock_sci_rdi,mock_ref_rdi = create_psfsub_dataset(nsci,nref,rolls,
                                            fwhm_pix=fwhm_pix,
                                            st_amp=st_amp,
                                            noise_amp=noise_amp,
                                            pl_contrast=pl_contrast)

    

    nx,ny = (21,21)
    cenx, ceny = (25.,30.)
    ctcal = create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx=cenx,ceny=ceny,
                  nx=nx,ny=ny)
    
    
    psfsub_dataset_rdi = do_psf_subtraction(mock_sci_rdi,ctcal,
                                reference_star_dataset=mock_ref_rdi,
                                numbasis=numbasis,
                                fileprefix='test_KL_THRU',
                                mode=None,
                                do_crop=False,
                                measure_klip_thrupt=True,
                                measure_1d_core_thrupt=True)
    
    # getting what the contrast curve for the first frame should be:
    frame0 = psfsub_dataset_rdi[0]
    klip = frame0.hdu_list['KL_THRU'].data
    separations = klip[0]
    klip_tp = klip[1:]
    core_tp = frame0.hdu_list['CT_THRU'].data[1]
    annular_noise = measure_noise(frame0, separations, hw=3)
    contrast_curve_expected = annular_noise.T/(klip_tp*core_tp)

    # now see what the step function gives:
    contrast_dataset = compute_flux_ratio_noise(psfsub_dataset_rdi, halfwidth=3)
    contrast = contrast_dataset[0].hdu_list['CON_CRV'].data
    contrast_seps = contrast[0]

    assert np.array_equal(contrast_seps, separations)
    assert np.array_equal(contrast[1:], contrast_curve_expected)
    assert 'CON_CRV' in contrast_dataset[0].ext_hdr["HISTORY"][-1]
    assert contrast_dataset[0].hdu_list['CON_CRV'].header['BUNIT'] == "erg/(s*cm^2*AA)"
    
    #test inputs
    with pytest.raises(ValueError):
        compute_flux_ratio_noise(psfsub_dataset_rdi, halfwidth=0)
    with pytest.warns(UserWarning):
        compute_flux_ratio_noise(psfsub_dataset_rdi, halfwidth=100)

if __name__ == '__main__':  
    test_expected_contrast()

    pass