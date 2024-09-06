import os, sys
import numpy as np
import pytest
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.spectroscopy as spectroscopy
from astropy.io import fits


def test_spectroscopy():
    print('testing spectroscopy utility and step functions')

    # load simulated data files
    # psf template - here is g0v star
    test_data_path = os.path.join(os.path.dirname(__file__), "test_data")
    psf_template_file = os.path.join(test_data_path, 'g0_spec_sim_unocc_3a.fits')
    psf_template_hdulist = fits.open(psf_template_file)
    psf_template = psf_template_hdulist[0].data

    # "data" - here is a0v star
    psf_data_file = os.path.join(test_data_path, 'a0_spec_sim_unocc_3a.fits')
    # psf_data_file = os.path.join(test_data_path, 'g0_spec_sim_unocc_3a.fits')
    psf_data_hdulist = fits.open(psf_data_file)
    psf_data = psf_data_hdulist[0].data

    x = spectroscopy.fit_centroid(psf_template, psf_data)
    print(x, 'binary offset')


if __name__ == "__main__":
    test_spectroscopy()