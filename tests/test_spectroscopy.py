import os, sys
import numpy as np
import pytest
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.spectroscopy as spectroscopy
from astropy.io import fits


def test_spectroscopy():
    print('Testing spectroscopy utility and step functions')

    print('-------------------------')
    print('Centroid fit test')
    # load simulated data files
    test_data_path = os.path.join(os.path.dirname(__file__), "test_data")
    slitprism_g0v_offset_fname = os.path.join(test_data_path, 
            'g0v_vmag6_spc-spec_band3_unocc_CFAM3d_R3C1SLIT_PRISM3_offset_array.fits')
    #slitprism_a0v_offset_fname = os.path.join(test_data_path, 
    #        'a0v_vmag6_spc-spec_band3_unocc_CFAM3d_R3C1SLIT_PRISM3_offset_array.fits')

    slitprism_g0v_offset_array = fits.getdata(slitprism_g0v_offset_fname, ext=0)
    slitprism_g0v_offset_table = fits.getdata(slitprism_g0v_offset_fname, ext=1)

    # Use first and last slices of cube as "template" and "data" images
    template_idx = 0
    data_idx = slitprism_g0v_offset_array.shape[0] - 1 # last slice
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

    print(f"True source (x,y) position in test PSF image (slice {data_idx} from \n" + 
          f"{os.path.basename(slitprism_g0v_offset_fname)}):\n" + 
          f"{true_xcent_data:.3f}, {true_ycent_data:.3f}")
    print(f"True source (x,y) position in PSF template (slice {template_idx} from \n" + 
          f"{os.path.basename(slitprism_g0v_offset_fname)}):\n" + 
          f"{true_xcent_template:.3f}, {true_ycent_template:.3f}")
    print("Centroid (x,y) estimate from template fit for test PSF image:\n" + 
            f"{xfit:.3f}, {yfit:.3f}")
    print(f"Gaussian profile fit centroid (x,y) estimate for PSF test image:\n" + 
            f"{xfit:.3f}, {yfit:.3f}")
    print("Centroid (x,y) errors:\n" + 
          f"{err_xfit:.3f}, {err_yfit:.3f} (PSF template fit)\n" + 
          f"{err_xfit_gauss:.3f}, {err_yfit_gauss:.3f} (2D Gaussian profile fit)\n") 

    assert err_xfit == pytest.approx(0, abs=0.2)
    assert err_yfit == pytest.approx(0, abs=0.2)
    assert xfit_gauss == pytest.approx(xfit, abs=0.5)
    assert yfit_gauss == pytest.approx(yfit, abs=0.5)

if __name__ == "__main__":
    test_spectroscopy()