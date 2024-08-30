import os, sys
import numpy as np
import pytest
sys.path.append('/home/mwoodlan/corgidrp/')
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.spectroscopy as spectroscopy
from astropy.io import fits


def test_spectroscopy():
    print('testing spectroscopy')

    test_data_path = '~/corgidrp/tests/test_data/'
    psf_template_file = os.path.join(test_data_path, 'a0_spec_sim_unocc_g0v_3a.fits')
    psf_template_hdulist = fits.open(psf_template_file)
    psf_template = psf_template_hdulist[0].data


    psf_data_file = os.path.join(test_data_path, 'a0_spec_sim_unocc_a0v_3a.fits')
    psf_data_hdulist = fits.open(psf_data_file)
    psf_data = psf_data_hdulist[0].data


    x = spectroscopy(psf_template, psf_data)

# load simulated data files
# psf template - here is g0v star

# "data" - here is a0v star




if __name__ == "__main__":
    test_spectroscopy()