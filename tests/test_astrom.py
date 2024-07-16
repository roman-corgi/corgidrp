import os
import numpy as np

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
    guess_path = os.path.join(datadir, 'guesses.csv')
    # target_path = os.path.join(datadir, 'target_guess.csv')

    # open the image
    dataset = corgidrp.data.Dataset([image_path])
    assert len(dataset) == 1
    assert type(dataset[0]) == corgidrp.data.Image

    # perform the astrometric calibration
    astrom_cal = corgidrp.astrom.boresight_calibration(input_dataset=dataset, guesses=guess_path)
    assert len(astrom_cal.data) == 4

    # the data was generated to have the following image properties
    expected_platescale = 21.8
    expected_northangle = 45

    # check orientation is correct within 0.3 [deg]
    assert np.abs(expected_platescale - astrom_cal.platescale) < 0.5  # not sure how accurate this needs to be
    assert np.abs(expected_northangle - astrom_cal.northangle) < 0.3

    # check that derived center coordinate falls within the field limits/ skycoords used
    guesses = ascii.read(guess_path)
    ra, dec = astrom_cal.boresight[0], astrom_cal.boresight[1]
    assert ra >= np.min(guesses['RA'])
    assert ra <= np.max(guesses['RA'])
    assert dec >= np.min(guesses['DEC'])
    assert dec <= np.max(guesses['DEC'])

    # the image was generated to have no offset, so check that boresight center is close to target
    expected_ra, expected_dec = dataset[0].ext_hdr['CRVAL1'], dataset[0].ext_hdr['CRVAL2']
    error_window = (30 * astropy.units.mas).to(astropy.units.deg).value
    assert np.abs(expected_ra - ra) < error_window
    assert np.abs(expected_dec - dec) < error_window

if __name__ == "__main__":
    test_astrom()