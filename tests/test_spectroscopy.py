import os, sys
import numpy as np
import pytest
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.spectroscopy as spectroscopy
from astropy.io import fits
import itertools
import matplotlib.pyplot as plt

def test_spectroscopy():
    print('Testing spectroscopy utility and step functions')

    print('-------------------------')
    print('Centroid fit test')
    # load simulated data files
    test_data_path = os.path.join(os.path.dirname(__file__), "test_data")
    slitprism_g0v_offset_fname = os.path.join(test_data_path, 
            'g0v_vmag6_spc-spec_band3_unocc_CFAM3d_NOSLIT_PRISM3_offset_array.fits')
    #slitprism_a0v_offset_fname = os.path.join(test_data_path, 
    #        'a0v_vmag6_spc-spec_band3_unocc_CFAM3d_R3C1SLIT_PRISM3_offset_array.fits')

    slitprism_g0v_offset_array = fits.getdata(slitprism_g0v_offset_fname, ext=0)
    slitprism_g0v_offset_table = fits.getdata(slitprism_g0v_offset_fname, ext=1)

    offset_inds = range(slitprism_g0v_offset_array.shape[0])
    xerr_list = list()
    yerr_list = list()
    xerr_gauss_list = list()
    yerr_gauss_list = list()

    for (template_idx, data_idx) in set(itertools.product(offset_inds, offset_inds)):
        psf_template = slitprism_g0v_offset_array[template_idx]
        psf_data = slitprism_g0v_offset_array[data_idx]
        (true_xcent_template, true_ycent_template) = (slitprism_g0v_offset_table['xcent'][template_idx], 
                                                      slitprism_g0v_offset_table['ycent'][template_idx])
        (true_xcent_data, true_ycent_data) = (slitprism_g0v_offset_table['xcent'][data_idx], 
                                              slitprism_g0v_offset_table['ycent'][data_idx])

        (xfit, yfit,
         xfit_gauss, yfit_gauss,
         x_precis, y_precis) = spectroscopy.fit_psf_centroid(psf_template, psf_data,
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

        print(f"True source (x,y) position in test PSF image (slice {data_idx} from \n" + 
              f"{os.path.basename(slitprism_g0v_offset_fname)}):\n" + 
              f"{true_xcent_data:.3f}, {true_ycent_data:.3f}")
        print(f"True source (x,y) position in PSF template (slice {template_idx} from \n" + 
              f"{os.path.basename(slitprism_g0v_offset_fname)}):\n" + 
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

        assert err_xfit == pytest.approx(0, abs=0.1)
        assert err_yfit == pytest.approx(0, abs=0.1)
        assert xfit_gauss == pytest.approx(xfit, abs=0.5)
        assert yfit_gauss == pytest.approx(yfit, abs=0.5)

    print(f"Std dev of (x,y) centroid errors from {len(xerr_list)} PSF template fit tests:\n" + 
          f"{np.std(xerr_list):.2E}, {np.std(yerr_list):.2E} pixels")
    print(f"Std dev of (x,y) centroid errors from {len(xerr_list)} Gaussian profile fit tests:\n" + 
          f"{np.std(xerr_gauss_list):.2E}, {np.std(yerr_gauss_list):.2E} pixels")

if __name__ == "__main__":
    test_spectroscopy()