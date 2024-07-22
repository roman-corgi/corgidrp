import os
import numpy as np
import pytest

import corgidrp
from corgidrp import data, mocks, astrom
import astropy
import astropy.io.ascii as ascii

def test_astrom():
    """ 
    Generate a simulated image and test the astrometric calibration computation.
    
    """
    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not
    datadir = os.path.join(os.path.dirname(__file__), "test_data", "simastrom")
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    field_path = os.path.join(os.path.dirname(__file__), "test_data", "JWST_CALFIELD2020.csv")

    corgidrp.mocks.create_astrom_data(field_path=field_path, filedir=datadir)

    image_path = os.path.join(datadir, 'simcal_astrom.fits')
    # guess_path = os.path.join(datadir, 'guesses.csv')
    # target_path = os.path.join(datadir, 'target_guess.csv')

    # open the image
    dataset = corgidrp.data.Dataset([image_path])
    assert len(dataset) == 1
    assert type(dataset[0]) == corgidrp.data.Image

    # perform the astrometric calibration
    astrom_cal = corgidrp.astrom.boresight_calibration(input_dataset=dataset, field_path=field_path)
    assert len(astrom_cal.data) == 4

    # the data was generated to have the following image properties
    expected_platescale = 21.8
    expected_northangle = 45

    # check orientation is correct within 0.3 [deg]
    # and plate scale is correct within 0.5 [mas] (arbitrary)
    assert astrom_cal.platescale == pytest.approx(expected_platescale, abs=0.5)
    assert astrom_cal.northangle == pytest.approx(expected_northangle, abs=0.3)

    # check that the center is correct within 30 [mas]
    # the simulated image should have no shift from the target
    target = dataset[0].pri_hdr['RA'], dataset[0].pri_hdr['DEC']
    ra, dec = astrom_cal.boresight[0], astrom_cal.boresight[1]
    assert ra == pytest.approx(target[0], abs=0.0000083)
    assert dec == pytest.approx(target[1], abs=0.0000083)

if __name__ == "__main__":
    test_astrom()