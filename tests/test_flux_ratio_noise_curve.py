from corgidrp.mocks import create_psfsub_dataset,create_ct_cal, create_default_L4_headers
from corgidrp.klip_fm import measure_noise
from corgidrp.l3_to_l4 import do_psf_subtraction
from corgidrp.l4_to_tda import compute_flux_ratio_noise
import corgidrp.data as data
import corgidrp.mocks as mocks

import pytest
import warnings
import numpy as np

import os
import logging
import shutil

from corgidrp.check import (check_filename_convention, 
                           verify_header_keywords, 
                           )

## Helper functions/quantities

# iwa_lod = 3.
# owa_lod = 9.7
iwa_mas = 140 #default in klip_fm.meas_klip_thrupt()
owa_mas = 450 #default in klip_fm.meas_klip_thrupt()
d = 2.36 #m
lam = 573.8e-9 #m
pixscale_arcsec = 0.0218
fwhm_mas = 1.22 * lam / d * 206265 * 1000
fwhm_pix = fwhm_mas * 0.001 / pixscale_arcsec
sig_pix = fwhm_pix / (2 * np.sqrt(2. * np.log(2.)))
# iwa_pix = iwa_lod * lam / d * 206265 / pixscale_arcsec
# owa_pix = owa_lod * lam / d * 206265 / pixscale_arcsec
iwa_pix = iwa_mas/1000/pixscale_arcsec
owa_pix = owa_mas/1000/pixscale_arcsec

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
    

def test_expected_flux_ratio_noise_pol():
    '''
    Test compute_flux_ratio_noise for POL mode

    Including the VAP testing infrastructure. 
    '''

    #######################################
    ########## Set up VAP Logger ##########
    #######################################

    current_file_path = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(current_file_path, 'test_data','l4_to_tda_compute_quphi')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # set up logging
    global logger

    log_file = os.path.join(output_dir, 'l4_to_tda_pol_flux_ratio_noise.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('l4_to_tda_pol_flux_ratio_noise')
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


    ##########################################################################
    ### Generate all the input data (copy and pasted from the on-pol case) ###
    ################################
    logger.info('='*80)
    logger.info('Polarimetry L4->TDA VAP Test 1: Flux Ratio Noise vs Separation')
    logger.info('='*80)
    
    logger.info('='*80)
    logger.info('Pre-test: set up input files and save to disk')
    logger.info('='*80)
        

    # RDI
    mode = 'RDI'
    nsci, nref = (1,1)
 
    st_amp = 100.
    noise_amp = 1e-3
    pl_contrast = 0. # No planet
    rolls = [0,15.,0,0]
    numbasis = [2] #Can only select 1 for pol mode
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
    # Use fwhm_pix to calculate sigma for consistency with the PSF subtraction dataset
    # FWHM = 2*sqrt(2*ln(2)) * sigma, so sigma = FWHM / (2*sqrt(2*ln(2)))
    sig_x = fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
    sig_y = sig_x 
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
    

    ####################################
    ## Create a mock L4 pol dataset ####
    ####################################

    l4_image = mocks.create_mock_stokes_image_l4(image_size = psfsub_dataset_rdi[0].data.shape[0])

    l4_image.data = psfsub_dataset_rdi[0].data
    l4_image.err = psfsub_dataset_rdi[0].err
    l4_image.dq = psfsub_dataset_rdi[0].dq
    l4_image.hdu_list['KL_THRU'] = psfsub_dataset_rdi[0].hdu_list['KL_THRU']
    l4_image.hdu_list['CT_THRU'] = psfsub_dataset_rdi[0].hdu_list['CT_THRU']

    mocks.rename_files_to_cgi_format(list_of_fits=[l4_image], output_dir=output_dir, level_suffix="l4")

    # ================================================================================
    # (2) Validate Input Images
    # ================================================================================
    
    logger.info('='*80)
    logger.info('Test Case 1: Input L4 Stokes Image Data Format and Content')
    logger.info('='*80)

    #Check input image complies with cgi format
    logger.info('='*80)
    logger.info('Test 4.1: Input L4 Image Data format')
    logger.info('='*80)
    frame_info = "Input L4 Polarimetry Image"
    check_filename_convention(getattr(l4_image , 'filename', None), 'cgi_*_l4_.fits', frame_info,logger,data_level='l4_')
    verify_header_keywords(l4_image .ext_hdr, {'BUNIT': 'photoelectron/s'},  frame_info,logger)
    verify_header_keywords(l4_image .ext_hdr, {'DATALVL': 'L4'},  frame_info,logger)
    logger.info("")


    ### Extract Stokes I frames from the L4 POL dataset to fake it as an L4 data product ###
    stokes_I = data.get_stokes_intensity_image(l4_image)
    stokes_I.data = stokes_I.data[None,:]
    stokes_I.err = stokes_I.err[None,:]
    stokes_I.dq = stokes_I.dq[None,:]
    
    # Set STARLOCX and STARLOCY to match the star position in the image (needed for the flux ratio noise curve)
    # Star is at x0, y0 in the test setup
    stokes_I.ext_hdr['STARLOCX'] = x0
    stokes_I.ext_hdr['STARLOCY'] = y0

    ############################################
    ### Now run the flux ratio noise curve ###
    ############################################
    stokes_I_dataset = data.Dataset([stokes_I])
    frn_dataset_pol = compute_flux_ratio_noise(stokes_I_dataset, nd_cal, star_dataset, unocculted_star_loc=np.array([[17],[15]]), halfwidth=3)

    # ================================================================================
    # (4) Validate Output Calibration Product
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Flux Ratio Noise Data Format and Content')
    logger.info('='*80)

    #  Check/log that FRN_CRV has correct shape [2+M, N] where:
    # - Row 0: separations in pixels
    # - Row 1: separations in mas
    # - Rows 2+: FRN values for M KL mode truncations
    logger.info('='*80)
    logger.info('Test 2.1: FRN_CRV Data Format')
    logger.info('='*80)
    frame_info = "Flux Ratio Noise Curve Extension"
    for frame in frn_dataset_pol:
        # Check that FRN_CRV extension exists
        if 'FRN_CRV' not in frame.hdu_list:
            logger.error(f"{frame_info}: FRN_CRV extension header not found. FAIL")
            assert False, f"{frame_info}: FRN_CRV extension header not found."
        else:
            logger.info(f"{frame_info}: FRN_CRV extension header exists. PASS")
        
        frn_crv_data = frame.hdu_list['FRN_CRV'].data
        n_separations = frn_crv_data.shape[1]
        n_klip_modes = 1 #Always 1 for pol mode. 
        expected_shape = (2 + n_klip_modes, n_separations)
        #VAP Version
        if frn_crv_data.shape != expected_shape:
            logger.error(f"{frame_info}: FRN_CRV data has shape {frn_crv_data.shape}, expected {expected_shape}. FAIL")
        else:
            logger.info(f"{frame_info}: FRN_CRV data has correct shape {frn_crv_data.shape}. PASS")
        #pytest version:
        assert frn_crv_data.shape == expected_shape, f"{frame_info}: FRN_CRV data has shape {frn_crv_data.shape}, expected {expected_shape}."
        
        # Verify row 0 contains separations in pixels
        separations_pix = frn_crv_data[0, :]
        if not np.all(separations_pix > 0):
            logger.error(f"{frame_info}: Row 0 (separations in pixels) contains non-positive values. FAIL")
        else:
            logger.info(f"{frame_info}: Row 0 correctly contains separations in pixels (all positive). PASS")
        assert np.all(separations_pix > 0), f"{frame_info}: Row 0 (separations in pixels) contains non-positive values."
        
        # Verify row 1 contains separations in mas
        separations_mas = frn_crv_data[1, :]
        if not np.all(separations_mas > 0):
            logger.error(f"{frame_info}: Row 1 (separations in mas) contains non-positive values. FAIL")
        else:
            logger.info(f"{frame_info}: Row 1 correctly contains separations in mas (all positive). PASS")
        assert np.all(separations_mas > 0), f"{frame_info}: Row 1 (separations in mas) contains non-positive values."
        
        # Verify rows 2+ contain FRN values
        frn_values = frn_crv_data[2:, :]
        if frn_values.shape[0] != n_klip_modes:
            logger.error(f"{frame_info}: Rows 2+ have shape {frn_values.shape[0]}, expected {n_klip_modes} KL mode truncations. FAIL")
        else:
            logger.info(f"{frame_info}: Rows 2+ correctly contain FRN values for {n_klip_modes} KL mode truncation(s). PASS")
        assert frn_values.shape[0] == n_klip_modes, f"{frame_info}: Rows 2+ have shape {frn_values.shape[0]}, expected {n_klip_modes} KL mode truncations."

    logger.info("")

    # Check/log that separations are within IWA-OWA range
    logger.info('='*80)
    logger.info('Test 2.2: FRN_CRV Separation Range Check')
    logger.info('='*80)
    for frame in frn_dataset_pol:
        frn_crv_data = frame.hdu_list['FRN_CRV'].data
        separations_pix = frn_crv_data[0,:]
        #Check if approximately small or large than IWA/OWA limits
        tolerance = 1e-3
        if np.any(separations_pix < iwa_pix - tolerance) or np.any(separations_pix > owa_pix + tolerance):
            logger.error(f"{frame_info}: FRN_CRV separations {separations_pix} exceed IWA/OWA limits ({iwa_pix}, {owa_pix}). FAIL")
        else:
            logger.info(f"{frame_info}: All FRN_CRV separations are within IWA/OWA limits ({iwa_pix}, {owa_pix}). PASS")
        #Pytest version:
        assert np.all(separations_pix >= iwa_pix - tolerance) and np.all(separations_pix <= owa_pix + tolerance), f"{frame_info}: FRN_CRV separations {separations_pix} exceed IWA/OWA limits ({iwa_pix}, {owa_pix})."

        #Check if separations in mas are within IWA/OWA limits
        iwa_mas_check = iwa_mas
        owa_mas_check = owa_mas
        if np.any(separations_mas < iwa_mas_check - tolerance) or np.any(separations_mas > owa_mas_check + tolerance):
            logger.error(f"{frame_info}: FRN_CRV separations in mas {separations_mas} exceed IWA/OWA limits ({iwa_mas_check}, {owa_mas_check}). FAIL")
        else:
            logger.info(f"{frame_info}: All FRN_CRV separations in mas are within IWA/OWA limits ({iwa_mas_check}, {owa_mas_check}). PASS")
        #Pytest version:
        assert np.all(separations_mas >= iwa_mas_check - tolerance) and np.all(separations_mas <= owa_mas_check + tolerance), f"{frame_info}: FRN_CRV separations in mas {separations_mas} exceed IWA/OWA limits ({iwa_mas_check}, {owa_mas_check})."
    logger.info("")

    # Check/log that FRN values are positive
    logger.info('='*80)
    logger.info('Test 2.3: FRN_CRV Positive Values Check')
    logger.info('='*80)
    for frame in frn_dataset_pol:
        frn_crv_data = frame.hdu_list['FRN_CRV'].data
        # FRN values are in rows 2+ (rows 0 and 1 are separations in pixels and mas)
        frn_values = frn_crv_data[2:, :]
        
        # Check for finite values (not NaN or inf)
        if not np.all(np.isfinite(frn_values)):
            logger.error(f"{frame_info}: FRN_CRV contains non-finite values (NaN or inf). FAIL")
            assert False, f"{frame_info}: FRN_CRV contains non-finite values (NaN or inf)."
        
        # Check that all values are positive (using same condition for logger and assert)
        all_positive = np.all(frn_values > 0)
        if not all_positive:
            non_positive_count = np.sum(frn_values <= 0)
            min_value = np.nanmin(frn_values)
            logger.error(f"{frame_info}: FRN_CRV contains {non_positive_count} non-positive values (min value: {min_value}). FAIL")
        else:
            logger.info(f"{frame_info}: All FRN_CRV values are positive. PASS")
        #Pytest version:
        assert all_positive, f"{frame_info}: FRN_CRV contains non-positive values."
    logger.info("")

    # Check/log that interpolation works for custom separation grid
    logger.info('='*80)
    logger.info('Test 2.4: FRN_CRV Interpolation Check for Custom Separation Grid')
    logger.info('='*80)
    requested_separations = np.array([26.5, 6.4, 14.8])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # catch Not all requested_separations from l4_to_tda
        frn_dataset_rseps_pol = compute_flux_ratio_noise(stokes_I_dataset, nd_cal, star_dataset, requested_separations=requested_separations, halfwidth=3)
    
    for frame in frn_dataset_rseps_pol:
        frn_crv_data = frame.hdu_list['FRN_CRV'].data
        frn_seps = frn_crv_data[0]
        if not np.array_equal(frn_seps, requested_separations):
            logger.error(f"{frame_info}: FRN_CRV separations {frn_seps} do not match requested separations {requested_separations}. FAIL")
        else:
            logger.info(f"{frame_info}: FRN_CRV separations match requested separations {requested_separations}. PASS")
        #Pytest version:
        assert np.array_equal(frn_seps, requested_separations), f"{frame_info}: FRN_CRV separations {frn_seps} do not match requested separations {requested_separations}."

    logger.info('='*80)
    logger.info('End of Polarimetry L4->TDA VAP Test 1: Flux Ratio Noise vs Separation')
    logger.info('='*80)
    # ================================================================================

    


if __name__ == '__main__':  
    # test_expected_flux_ratio_noise()
    test_expected_flux_ratio_noise_pol()