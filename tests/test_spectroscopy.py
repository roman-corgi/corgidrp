import os, sys
import numpy as np
import pytest
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.spectroscopy as spectroscopy
from astropy.io import fits
import itertools
import pandas as pd
from astropy.table import Table, Column

def test_fit_psf_centroid(errortol_pix = 0.01, verbose = False):
    """
    Test the accuracy of the PSF centroid fitting function with an array of simulated 
    noiseless SPC prism EXCAM images computed for a grid of offsets.

    Args:
        errortol_pix (float): Tolerance on centroid errors on each axis, in EXCAM pixels
        verbose (bool): If verbose=True, print the results from each individual test
    """
    print('Testing the PSF centroid fitting function used by the spectroscopy wavecal recipe...')

    test_data_path = os.path.join(os.path.dirname(corgidrp.__path__[0]), "tests", "test_data", "spectroscopy")
    spc_offset_psf_array_fname = os.path.join(test_data_path, 
            'g0v_vmag6_spc-spec_band3_unocc_CFAM3d_NOSLIT_PRISM3_offset_array.fits')
    psf_array = fits.getdata(spc_offset_psf_array_fname, ext=0)
    psf_truth_table = fits.getdata(spc_offset_psf_array_fname , ext=1)

    offset_inds = range(psf_array.shape[0])
    xerr_list = list()
    yerr_list = list()
    xerr_gauss_list = list()
    yerr_gauss_list = list()

    for (template_idx, data_idx) in itertools.combinations(offset_inds, 2):
        psf_template = psf_array[template_idx]
        psf_data = psf_array[data_idx]
        (true_xcent_template, true_ycent_template) = (psf_truth_table['xcent'][template_idx], 
                                                      psf_truth_table['ycent'][template_idx])
        (true_xcent_data, true_ycent_data) = (psf_truth_table['xcent'][data_idx], 
                                              psf_truth_table['ycent'][data_idx])

        (xfit, yfit,
         xfit_gauss, yfit_gauss, 
         peak_pix_snr,
         x_precis, y_precis) = spectroscopy.fit_psf_centroid(psf_data, psf_template,
                                                            true_xcent_template, true_ycent_template,
                                                            halfwidth = 10, halfheight = 10)

        err_xfit, err_yfit = (xfit - true_xcent_data, 
                              yfit - true_ycent_data)
        err_xfit_gauss, err_yfit_gauss = (xfit_gauss - true_xcent_data, 
                                          yfit_gauss - true_ycent_data)
        xerr_list.append(err_xfit)
        yerr_list.append(err_yfit)
        xerr_gauss_list.append(err_xfit_gauss)
        yerr_gauss_list.append(err_yfit_gauss)

        if verbose:
            print(f"True source (x,y) position in test PSF image (slice {data_idx})\n" + 
                  f"{true_xcent_data:.3f}, {true_ycent_data:.3f}")
            print(f"True source (x,y) position in PSF template (slice {template_idx})\n" + 
                  f"{true_xcent_template:.3f}, {true_ycent_template:.3f}")
            print(f"True source offset from template PSF to test PSF: \n" + 
                  f"{true_xcent_data - true_xcent_template:.3f}, {true_ycent_data - true_ycent_template:.3f}")
            print("Centroid (x,y) estimate from template fit for test PSF image:\n" + 
                    f"{xfit:.3f}, {yfit:.3f}")
            print(f"Gaussian profile fit centroid (x,y) estimate for PSF test image:\n" + 
                    f"{xfit_gauss:.3f}, {yfit_gauss:.3f}")
            print("Centroid (x,y) errors:\n" + 
                  f"{err_xfit:.3f}, {err_yfit:.3f} (PSF template fit)\n" + 
                  f"{err_xfit_gauss:.3f}, {err_yfit_gauss:.3f} (2D Gaussian profile fit)\n") 

        assert err_xfit == pytest.approx(0, abs=errortol_pix), \
                f"Accuracy failure: x-axis centroid error {err_xfit:.1E} pixels versus {errortol_pix:.1E} pixel tolerance"
        assert err_yfit == pytest.approx(0, abs=errortol_pix), \
                f"Accuracy failure: y-axis centroid error {err_yfit:.1E} pixels versus {errortol_pix:.1E} pixel tolerance"

    print(f"Std dev of (x,y) centroid errors from {len(xerr_list)} PSF template fit tests:\n" + 
          f"{np.std(xerr_list):.2E}, {np.std(yerr_list):.2E} pixels")
    print(f"Std dev of (x,y) centroid errors from {len(xerr_list)} Gaussian profile fit tests:\n" + 
          f"{np.std(xerr_gauss_list):.2E}, {np.std(yerr_gauss_list):.2E} pixels")
    print("PSF centroid fit accuracy test passed.")

def test_dispersion_fit(errortol_nm = 0.5, prism = 'PRISM3'):
    """ 
    Test the function that fits the spectral dispersion profile of the CGI ZOD prism.
    """
    test_data_path = os.path.join(os.path.dirname(corgidrp.__path__[0]), 
                                  "tests", "test_data", "spectroscopy")
    assert prism in ['PRISM2', 'PRISM3']
    # load inputs and instrument data
    if prism == 'PRISM2':
        subband_list = ['2A', '2B', '2C']
        tvac_dispersion_params_fname = os.path.join(test_data_path, 
                                                    "TVAC_PRISM2_dispersion_profile.npz")
        ref_wavlen = 660.0
        ref_cfam = '2C'
        bandpass = [610, 710]
    else:
        subband_list = ['3A', '3B', '3C', '3D', '3E', '3G']
        tvac_dispersion_params_fname = os.path.join(test_data_path, 
                                                    "TVAC_PRISM3_dispersion_profile.npz")
        ref_wavlen = 730.0
        ref_cfam = '3C'
        bandpass = [675, 785]
    tvac_dispersion = np.load(tvac_dispersion_params_fname)
    pixel_pitch_um = 13.0 # EXCAM pixel pitch (microns)

    # One version of the filter sweep simulation array has no filter-to-filter image offsets; 
    # those PSFs will serve as the centroid fitting templates
    prism_filtersweep_nooffsets_sim_fname = os.path.join(test_data_path,
            'g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep.fits')
    # One version includes filter-to-filter image offsets, which are recorded in an extension table.
    prism_filtersweep_offsets_sim_fname = os.path.join(test_data_path,
            'g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep_withoffsets.fits')
    
    bandpass_center_table_fname = os.path.join(test_data_path, "CGI_bandpass_centers.csv")
    bandpass_center_table = pd.read_csv(bandpass_center_table_fname, index_col=0)
    tvac_centwav_exists = [centwav > 0 for centwav in bandpass_center_table['TVAC TV-40b center wavelength (nm)']]
    tvac_filters = bandpass_center_table.index.values[tvac_centwav_exists]
    
    template_array = fits.getdata(prism_filtersweep_nooffsets_sim_fname, ext=0)
    template_pos_table = Table(fits.getdata(prism_filtersweep_nooffsets_sim_fname, ext=1))
    data_array = fits.getdata(prism_filtersweep_offsets_sim_fname, ext=0)
    filtersweep_table = Table(fits.getdata(prism_filtersweep_offsets_sim_fname, ext=1))
    filtersweep_hdr = fits.getheader(prism_filtersweep_offsets_sim_fname, ext=0)
    true_clocking_angle = filtersweep_hdr['PRISMANG']

    # Add random noise to the filter sweep template images to serve as fake data
    np.random.seed(5)
    read_noise = 200
    noisy_data_array = (np.random.poisson(np.abs(data_array) / 2) + 
                       np.random.normal(loc=0, scale=read_noise, size=data_array.shape))

    template_pos_table.add_index('CFAM')
    filtersweep_table.add_index('CFAM')
    template_pos_table['CFAM'] = [cfam.upper().strip() for cfam in template_pos_table['CFAM']]
    filtersweep_table['CFAM'] = [cfam.upper().strip() for cfam in filtersweep_table['CFAM']]
    filtersweep_table.add_column(Column(name='center wavel (nm)', data=[0.]*len(filtersweep_table)), index=1)
    filtersweep_table.rename_column('xoffset', 'CFAM x offset')
    filtersweep_table.rename_column('yoffset', 'CFAM y offset')
    filtersweep_table.rename_column('xcent', 'true x_cent')
    filtersweep_table.rename_column('ycent', 'true y_cent')
    filtersweep_table['x_cent'] = 0.
    filtersweep_table['y_cent'] = 0.
    filtersweep_table['x_cent offset corr'] = 0.
    filtersweep_table['y_cent offset corr'] = 0.
    filtersweep_table['x_cent gauss'] = 0.
    filtersweep_table['y_cent gauss'] = 0.
    filtersweep_table['peak pix SNR'] = 0.
    filtersweep_table['x_err est'] = 0.
    filtersweep_table['y_err est'] = 0.

    for i, cfam in enumerate(filtersweep_table['CFAM']):
        if cfam == '3': # larger fitting stamp needed for broadband filter 
            halfheight = 30
        else:
            halfheight = 10
    
        psf_data = noisy_data_array[i]
        psf_template = template_array[i]
        xcent_template, ycent_template = (template_pos_table['xcent'][i], template_pos_table['ycent'][i])
        (xcent_fit, ycent_fit, 
         xcent_gauss, ycent_gauss,
         psf_peakpix_snr,
         xprecis, yprecis) = spectroscopy.fit_psf_centroid(psf_data, psf_template, 
                                          xcent_template=template_pos_table.loc[cfam]['xcent'],
                                          ycent_template=template_pos_table.loc[cfam]['ycent'],
                                          halfheight=halfheight)
    
        filtersweep_table['x_cent'][i] = xcent_fit
        filtersweep_table['y_cent'][i] = ycent_fit
        filtersweep_table['x_cent gauss'][i] = xcent_gauss
        filtersweep_table['y_cent gauss'][i] = ycent_gauss
        filtersweep_table['peak pix SNR'][i] = psf_peakpix_snr
        filtersweep_table['x_err est'][i] = xprecis
        filtersweep_table['y_err est'][i] = yprecis
    
        if cfam in tvac_filters:
            filtersweep_table.loc[cfam]['center wavel (nm)'] = bandpass_center_table.loc[cfam,'TVAC TV-40b center wavelength (nm)']
        else:
            filtersweep_table.loc[cfam]['center wavel (nm)'] = bandpass_center_table.loc[cfam,'Phase C center wavelength (nm)']
    # Subtract the 'known' CFAM image offsets from the centroid estimates.
    filtersweep_table['CFAM x offset'] = filtersweep_table['CFAM x offset'] - filtersweep_table.loc[ref_cfam]['CFAM x offset']
    filtersweep_table['CFAM y offset'] = filtersweep_table['CFAM y offset'] - filtersweep_table.loc[ref_cfam]['CFAM y offset']
    filtersweep_table['x_cent offset corr'] = filtersweep_table['x_cent'] - filtersweep_table['CFAM x offset']
    filtersweep_table['y_cent offset corr'] = filtersweep_table['y_cent'] - filtersweep_table['CFAM y offset']
    # For testing purposes, compare the fitted positions to the actual positions if they were provided.
    if 'true x_cent' in filtersweep_table.colnames and 'true y_cent' in filtersweep_table.colnames:
        filtersweep_table['x_err true'] = filtersweep_table['x_cent'] - filtersweep_table['true x_cent']
        filtersweep_table['y_err true'] = filtersweep_table['y_cent'] - filtersweep_table['true y_cent']
        filtersweep_table['x_err gauss true'] = filtersweep_table['x_cent gauss'] - filtersweep_table['true x_cent']
        filtersweep_table['y_err gauss true'] = filtersweep_table['y_cent gauss'] - filtersweep_table['true y_cent']

    subband_mask = [cfam in subband_list for cfam in filtersweep_table['CFAM']]
    meas = filtersweep_table[subband_mask]
    assert len(meas) >= 4, 'Need images taken in at least four sub-band filters to model the dispersion'

    (clocking_angle,
     clocking_angle_uncertainty) = spectroscopy.estimate_dispersion_clocking_angle(
            meas['x_cent offset corr'], meas['y_cent offset corr'], weights = 1. / meas['y_err est']
     )
    assert clocking_angle == pytest.approx(true_clocking_angle, abs=0.1)
    print("Estimated clocking angle = {:.2f} +/- {:.2f} deg; input model angle is {:.2f} deg".format(
            clocking_angle, clocking_angle_uncertainty, true_clocking_angle))

    (pfit_pos_vs_wavlen, cov_pos_vs_wavlen,
     pfit_wavlen_vs_pos, cov_wavlen_vs_pos) = spectroscopy.fit_dispersion_polynomials(
            meas['center wavel (nm)'],meas['x_cent offset corr'], meas['y_cent offset corr'], 
            meas['y_err est'], clocking_angle, ref_wavlen, pixel_pitch_um 
     )
    pos_func_wavlen = np.poly1d(pfit_pos_vs_wavlen)
    wavlen_func_pos = np.poly1d(pfit_wavlen_vs_pos)

    #### Load TVAC dispersion profile to compare and test the agreement in the wavelength vs position.
    tvac_wavlen_func_pos = np.poly1d(tvac_dispersion['wavelen_vs_position_polycoeff'])
    tvac_pos_func_wavlen = np.poly1d(tvac_dispersion['position_vs_wavelen_polycoeff'])    

    (xtest_min, xtest_max) = (tvac_pos_func_wavlen((bandpass[0] - ref_wavlen)/ref_wavlen),
                              tvac_pos_func_wavlen((bandpass[1] - ref_wavlen)/ref_wavlen))

    xtest = np.linspace(xtest_min, xtest_max, 1000)
    tvac_model_wavlens = tvac_wavlen_func_pos(xtest)
    corgi_model_wavlens = wavlen_func_pos(xtest)
    wavlen_model_error = corgi_model_wavlens - tvac_model_wavlens
    worst_case_wavlen_error = np.abs(wavlen_model_error).max()
    print(f"Worst case wavelength disagreement from the test input model: {worst_case_wavlen_error:.2f} nm")
    assert worst_case_wavlen_error == pytest.approx(0, abs=errortol_nm)
    print("Dispersion profile fit test passed.")

if __name__ == "__main__":
    # The test applied to spectroscopy.fit_psf_centroid() loads 
    # a simulation file containing an array of PSFs computed for 
    # a 2-D grid of sub-pixel offsets.
    test_fit_psf_centroid(errortol_pix = 1E-2, verbose=False)

    # Test the dispersion profile fitting function with an array
    # of simulated PSF images for a set of sub-band color filters.
    test_dispersion_fit(errortol_nm = 0.5)
