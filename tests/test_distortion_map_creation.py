import os
import pickle
import numpy as np
import pytest

import corgidrp
import corgidrp.mocks as mocks
import corgidrp.astrom as astrom
import corgidrp.data as data

import astropy
from astropy.io import ascii

def test_distortion():
    """ 
    Generate a simulated image and test the distortion map creation as part of the boresight calibration.
    
    """
    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not
    datadir = os.path.join(os.path.dirname(__file__), "test_data", "simastrom")
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    field_path = os.path.join(os.path.dirname(__file__), "test_data", "JWST_CALFIELD2020.csv")
    distortion_coeffs_path = os.path.join(os.path.dirname(__file__), "test_data", "distortion_expected_coeffs.csv")

    mocks.create_astrom_data(field_path=field_path, filedir=datadir, distortion_coeffs_path=distortion_coeffs_path)

    image_path = os.path.join(datadir, 'simcal_astrom.fits')
    source_match_path = os.path.join(datadir, 'guesses.csv')
    matches = ascii.read(source_match_path)

    # open the image
    dataset = data.Dataset([image_path])

    # perform the astrometric calibration
    # feed in the correct matches and use only ~100 randomly selected stars
    rng = np.random.default_rng(seed=17)
    select_stars = rng.choice(len(matches), size=150, replace=False)

    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, field_matches=[matches[select_stars]], find_distortion=True)

    ## check that the distortion map does not create offsets greater than 4[mas]
        # compute the distortion maps created from the best fit coeffs
    coeffs = astrom_cal.distortion_coeffs
    expected_coeffs = np.genfromtxt(distortion_coeffs_path)

        # note the image shape and center around the image center
    image_shape = np.shape(dataset[0].data)
    yorig, xorig = np.indices(image_shape)
    y0, x0 = image_shape[0]//2, image_shape[1]//2
    yorig -= y0
    xorig -= x0

        # get the number of fitting params from the order
    fitorder = int(coeffs[1])
    fitparams = (fitorder + 1)**2
    true_fitorder = int(expected_coeffs[-1])
    true_fitparams = (true_fitorder + 1)**2

        # reshape the coeff arrays
    best_params_x = coeffs[0][:fitparams]
    best_params_x = best_params_x.reshape(fitorder+1, fitorder+1)
    total_orders = np.arange(fitorder+1)[:,None] + np.arange(fitorder+1)[None, :]
    best_params_x = best_params_x / 500**(total_orders)

    true_params_x = expected_coeffs[:-1][:true_fitparams]
    true_params_x = true_params_x.reshape(true_fitorder+1, true_fitorder+1)
    true_total_orders = np.arange(true_fitorder+1)[:,None] + np.arange(true_fitorder+1)[None, :]
    true_params_x = true_params_x / 500**(true_total_orders)

        # evaluate the polynomial at all pixel positions
    x_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), best_params_x)
    x_corr = x_corr.reshape(xorig.shape)
    x_diff = x_corr - xorig

    true_x_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), true_params_x)
    true_x_corr = true_x_corr.reshape(xorig.shape)
    true_x_diff = true_x_corr - xorig

        # reshape and evaluate the same for y
    best_params_y = coeffs[0][fitparams:]
    best_params_y = best_params_y.reshape(fitorder+1, fitorder+1)
    best_params_y = best_params_y / 500**(total_orders)

    true_params_y = expected_coeffs[:-1][true_fitparams:]
    true_params_y = true_params_y.reshape(true_fitorder+1, true_fitorder+1)
    true_params_y = true_params_y / 500**(true_total_orders)
    
        # evaluate the polynomial at all pixel positions
    y_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), best_params_y)
    y_corr = y_corr.reshape(yorig.shape)
    y_diff = y_corr - yorig

    true_y_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), true_params_y)
    true_y_corr = true_y_corr.reshape(yorig.shape)
    true_y_diff = true_y_corr - yorig

    # check the distortion maps are less than the maximum injected distortion (~3 pixels)
    assert np.all(np.abs(x_diff)) < np.max(np.abs(true_x_diff))
    assert np.all(np.abs(y_diff)) < np.max(np.abs(true_y_diff))

    # check they can be pickled (for CTC operations)
    pickled = pickle.dumps(astrom_cal)
    pickled_astrom = pickle.loads(pickled)
    assert np.all((astrom_cal.data == pickled_astrom.data))

    # save and check it can be pickled after save
    astrom_cal.save(filedir=datadir, filename="astrom_cal_output.fits")
    astrom_cal_2 = data.AstrometricCalibration(os.path.join(datadir, "astrom_cal_output.fits"))

    # check they can be pickled (for CTC operations)
    pickled = pickle.dumps(astrom_cal_2)
    pickled_astrom = pickle.loads(pickled)
    assert np.all((astrom_cal.data == pickled_astrom.data)) # check it is the same as the original

if __name__ == "__main__":
    test_distortion()