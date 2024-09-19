import os, sys
import numpy as np
import pytest
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.spectroscopy as spectroscopy
from astropy.io import fits
import itertools

def test_fit_psf_centroid(errortol_pix = 0.01, verbose = False):
    """
    Test the accuracy of the PSF centroid fitting function with an array of simulated 
    noiseless SPC prism EXCAM images computed for a grid of offsets.

    Args:
        errortol_pix (float): Tolerance on centroid errors on each axis, in EXCAM pixels
        verbose (bool): If verbose=True, print the results from each individual test
    """
    print('Testing the PSF centroid fitting function used by the spectroscopy wavecal recipe...')

    test_data_path = os.path.join(os.path.dirname(__file__), "test_data")
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

if __name__ == "__main__":
    # The test applied to spectroscopy.fit_psf_centroid() loads 
    # a simulation file containing an array of PSFs computed for 
    # a 2-D grid of sub-pixel offsets.
    test_fit_psf_centroid(errortol_pix=1E-2, verbose = False)
