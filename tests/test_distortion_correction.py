from corgidrp import data
from corgidrp.l3_to_l4 import distortion_correction

import numpy as np
import os

def test_distortion_correction():
    """
    Unit test of the distortion correction function
    
    """

    # read in mock dataset (assume it has already had distortion coefficients computed)
    datadir = os.path.join(os.path.dirname(__file__), "simastrom")
    image_path = os.path.join(datadir, 'mock_distortion.fits')
    dataset = data.Dataset([image_path])

    # load in a reference dataset that has zero distortion
    ref_path = os.path.join(datadir, 'mock_distortion_reference.fits')
    ref_dataset = data.Dataset([ref_path])

    # read in the AstrometricCalibration file with the distortion coefficients
    astrom_path = os.path.join(datadir, 'mock_distortion_astrom.fits')
    astrom_dataset = data.Dataset([astrom_path])

    # use the distortion correction function
    undistorted_dataset = distortion_correction(dataset, astrom_dataset)

    # check that the corrected image is the same as the reference
    