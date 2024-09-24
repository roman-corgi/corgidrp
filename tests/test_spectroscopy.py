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
from pathlib import Path
from astropy.table import Table, Column

from scipy.stats import binned_statistic
from scipy.interpolate import interp1d

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

def test_dispersion_fit(errortol_nm = 0.5, prism = 'PRISM3', test_product_path = None):
    """ 
    Test the function that fits the spectral dispersion profile of the CGI ZOD prism.

    For test inputs, load an array of slitless prism images of an unocculted
    star, simulated for a set of sub-band calibration filters, using the
    TVAC-derived dispersion profile. Test the accuracy of the dispersion fit
    based on the estimation of the input dispersion profile.

    Args:
    
    errortol_nm (numpy.float): Tolerance on the worst-case wavelength
    disagreement between the input (TVAC) dispersion model and the estimated dispersion
    profile, compared across the bandpass.

    prism (str): Label for the selected DPAM zero-deviation prism; must be
    either 'PRISM3' or 'PRISM2'

    test_product_path (str): Location to store the dispersion profile estimated
    during the test.

    """
    test_data_path = os.path.join(os.path.dirname(corgidrp.__path__[0]), 
            "tests", "test_data", "spectroscopy")
    if test_product_path == None:
        test_product_path = os.path.join(test_data_path, "test_products")
    assert prism in ['PRISM2', 'PRISM3']
    # load inputs and instrument data
    if prism == 'PRISM2':
        subband_list = ['2A', '2B', '2C']
        tvac_dispersion_params_fname = os.path.join(
                  test_data_path, "TVAC_PRISM2_dispersion_profile.npz")
        ref_wavlen = 660.0
        ref_cfam = '2'
        bandpass = [610, 710]
    else:
        subband_list = ['3A', '3B', '3C', '3D', '3E', '3G']
        tvac_dispersion_params_fname = os.path.join(
                  test_data_path, "TVAC_PRISM3_dispersion_profile.npz")
        ref_wavlen = 730.0
        ref_cfam = '3'
        bandpass = [675, 785]
    tvac_dispersion = spectroscopy.DispersionModel(tvac_dispersion_params_fname)
    pixel_pitch_um = 13.0 # EXCAM pixel pitch (microns)

    # One version of the filter sweep simulation array has no filter-to-filter image offsets; 
    # those PSFs will serve as the centroid fitting templates
    prism_filtersweep_nooffsets_sim_fname = os.path.join(test_data_path,
            'g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep.fits')
    # One version includes filter-to-filter image offsets, which are recorded in an extension table.
    prism_filtersweep_offsets_sim_fname = os.path.join(test_data_path,
            'g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep_withoffsets.fits')
    # File name of table to store the test centroid fit results
    prism_filtersweep_centroid_table_fname = os.path.join(
            test_product_path,
            Path(prism_filtersweep_offsets_sim_fname).stem + '_centroids.ecsv')
    
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
    tvac_wavlen_func_pos = np.poly1d(tvac_dispersion.wavlen_vs_pos_polycoeff)
    tvac_pos_func_wavlen = np.poly1d(tvac_dispersion.pos_vs_wavlen_polycoeff)    

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

    if not os.path.exists(test_product_path):
        os.mkdir(test_product_path)
    
    dispersion_profile_npz_fname = "corgidrp_{:s}_dispersion_profile.npz".format(prism)
    corgi_dispersion_profile = spectroscopy.DispersionModel(
        clocking_angle = clocking_angle,
        clocking_angle_uncertainty = clocking_angle_uncertainty,
        pos_vs_wavlen_polycoeff = pfit_pos_vs_wavlen,
        pos_vs_wavlen_cov = cov_pos_vs_wavlen,
        wavlen_vs_pos_polycoeff = pfit_wavlen_vs_pos,
        wavlen_vs_pos_cov = cov_wavlen_vs_pos
    )

    filtersweep_table.write(prism_filtersweep_centroid_table_fname, overwrite=True)
    print(f"Stored the centroid estimate table to {prism_filtersweep_centroid_table_fname}")

    corgi_dispersion_profile.save(test_product_path, dispersion_profile_npz_fname)
    print(f"Stored the dispersion profile fit results to {corgi_dispersion_profile.filepath}")

def test_wave_cal(errortol_nm = 1.0, prism = 'PRISM3', zeropt = None, test_product_path = None):
    """

    Test the wave_cal_map() step function that computes the wavelength
    calibration map and lookup table.

    In addition to a dispersion profile data structure, the wave_cal_map()
    function requires a wavelength zero-point position on the EXCAM array. As a
    test input, use the PSF centroid of one of the simulated sub-band filter
    images.

    Args:
    
    errortol_nm (numpy.float): Tolerance on the worst-case wavelength
    disagreement between the input (TVAC) dispersion model and the estimated dispersion
    profile, compared across the bandpass.

    prism (str): Label for the selected DPAM zero-deviation prism; must be
    either 'PRISM3' or 'PRISM2'

    zeropt (spectroscopy.WavelengthZeropoint): Wavelength zero point data
    object; the class is defined in the corgidrp spectroscopy module.

    test_product_path (str): Location to store the dispersion profile estimated
    during the test.

    """

    test_data_path = os.path.join(os.path.dirname(corgidrp.__path__[0]), "tests", "test_data", "spectroscopy")
    test_product_path = os.path.join(test_data_path, "test_products")

    # Use an array of simulated filter sweep 
    prism_filtersweep_sim_fname = prism_filtersweep_offsets_sim_fname = os.path.join(test_data_path,
        'g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep_withoffsets.fits')
    # File name of table of test centroid fit results
    prism_filtersweep_centroid_table_fname = os.path.join(
        test_product_path,
        Path(prism_filtersweep_sim_fname).stem + '_centroids.ecsv')
    # File name of wavelength calibration map
    wavecal_map_fname = os.path.join(
        test_product_path,
        Path(prism_filtersweep_sim_fname).stem + '_wavecal.fits')   

    prism_filtersweep_array = fits.getdata(prism_filtersweep_sim_fname, ext=0)
    filtersweep_table = Table.read(prism_filtersweep_centroid_table_fname)
    filtersweep_table.add_index('CFAM')
    bandpass_center_table_fname = os.path.join(test_data_path, "CGI_bandpass_centers.csv")
    bandpass_center_table = pd.read_csv(bandpass_center_table_fname, index_col=0)
    
    dispersion_params_fname = os.path.join(test_product_path, 'corgidrp_PRISM3_dispersion_profile.npz')
    disp_params = spectroscopy.DispersionModel(dispersion_params_fname)

    pixel_pitch_um = 13.0 # EXCAM pixel pitch (microns)
    if prism == 'PRISM2':
        ref_wavlen = 660.0
        ref_cfam = '2'
        zeropt_cfam = '2C'
    else:
        ref_wavlen = 730.0
        ref_cfam = '3'
        zeropt_cfam = '3D'

    if zeropt is None:
        zeropt = spectroscopy.WavelengthZeropoint(
            prism,
            bandpass_center_table.loc[zeropt_cfam]['TVAC TV-40b center wavelength (nm)'],
            filtersweep_table.loc[zeropt_cfam]['x_cent offset corr'], 
            filtersweep_table.loc[zeropt_cfam]['y_cent offset corr'],
            filtersweep_table.loc[zeropt_cfam]['x_err est'], 
            filtersweep_table.loc[zeropt_cfam]['y_err est'],
            prism_filtersweep_array[0].shape 
        )

    (wavlen_cal_map, 
     wavlen_uncertainty_map,
     pos_lookup_table) = spectroscopy.create_wave_cal_map(disp_params, zeropt, ref_wavlen)

    worst_case_wavlen_uncertainty = np.nanmax(wavlen_uncertainty_map)
    print(f"Worst case wavelength uncertainty after Monte Carlo error propagation: {worst_case_wavlen_uncertainty:.2f} nm")
    assert worst_case_wavlen_uncertainty == pytest.approx(0, abs=errortol_nm)

    # Store the wavelength calibration map, wavelength uncertainty map, and
    # wavelength-position lookup table to a FITS file.
    primary_hdu = fits.PrimaryHDU(data = wavlen_cal_map)
    # write zeropoint to FITS header
    primary_hdu.header['prism'] = zeropt.prism
    primary_hdu.header['wavlen0'] = zeropt.wavlen
    primary_hdu.header['x0'] = zeropt.x
    primary_hdu.header['x0_err'] = zeropt.x_err
    primary_hdu.header['y0'] = zeropt.x
    primary_hdu.header['y0_err'] = zeropt.y_err
    
    error_hdu = fits.ImageHDU(data = wavlen_uncertainty_map)
    
    lookup_table_hdu = fits.TableHDU.from_columns((
        fits.Column(name='wavlen', array=pos_lookup_table['Wavelength (nm)'], format='F'),
        fits.Column(name='x', array=pos_lookup_table['x (column)'], format='F'),
        fits.Column(name='x_err', array=pos_lookup_table['x uncertainty'], format='F'),
        fits.Column(name='y', array=pos_lookup_table['y (row)'], format='F'),
        fits.Column(name='y_err', array=pos_lookup_table['y uncertainty'], format='F')
    ))
    hdulist = fits.HDUList([primary_hdu, error_hdu, lookup_table_hdu])
    hdulist.writeto(wavecal_map_fname, overwrite=True)
    print(f"Wrote wavelength calibration map to {wavecal_map_fname}")

if __name__ == "__main__":
    # The test applied to spectroscopy.fit_psf_centroid() loads 
    # a simulation file containing an array of PSFs computed for 
    # a 2-D grid of sub-pixel offsets.
    test_fit_psf_centroid(errortol_pix = 1E-2, verbose=False)

    # Test the dispersion profile fitting function with an array
    # of simulated PSF images for a set of sub-band color filters.
    test_dispersion_fit(errortol_nm = 1.0)

    # Test the wavelength calibration map function with the dispersion profile
    # computed above and a wavelength zero-point test input.
    test_wave_cal(errortol_nm = 1.0)


    