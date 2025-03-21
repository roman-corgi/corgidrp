import os
import pickle
import numpy as np
import pytest
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.astrom as astrom
import corgidrp.data as data
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

    mocks.create_astrom_data(field_path=field_path, filedir=datadir)

    image_path = os.path.join(datadir, 'simcal_astrom.fits')

    # open the image
    dataset = data.Dataset([image_path])
    assert len(dataset) == 1
    assert type(dataset[0]) == data.Image

    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, find_threshold=5)

    # the data was generated to have the following image properties
    expected_platescale = 21.8
    expected_northangle = 45

    # check orientation is correct within 0.05 [deg]
    # and plate scale is correct within 0.5 [mas] (arbitrary)
    assert astrom_cal.platescale == pytest.approx(expected_platescale, abs=0.5)
    assert astrom_cal.northangle == pytest.approx(expected_northangle, abs=0.05)

    # check that the center is correct within 3 [mas]
    # the simulated image should have no shift from the target
    target = dataset[0].pri_hdr['RA'], dataset[0].pri_hdr['DEC']
    ra, dec = astrom_cal.boresight[0], astrom_cal.boresight[1]
    assert ra == pytest.approx(target[0], abs=8.333e-7)
    assert dec == pytest.approx(target[1], abs=8.333e-7)

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
    expected_coeffs = np.genfromtxt(distortion_coeffs_path)

    mocks.create_astrom_data(field_path=field_path, filedir=datadir, rotation=20, distortion_coeffs_path=distortion_coeffs_path)

    image_path = os.path.join(datadir, 'simcal_astrom.fits')
    source_match_path = os.path.join(datadir, 'guesses.csv')
    matches = ascii.read(source_match_path)

    # open the image
    dataset = data.Dataset([image_path])

    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, field_matches=[matches], find_threshold=400, comparison_threshold=75, find_distortion=True, fitorder=3, position_error=0.5)
    #, initial_dist_guess=expected_coeffs[:-1]

    ## check that the distortion map does not create offsets greater than 4[mas]
        # compute the distortion maps created from the best fit coeffs
    coeffs = astrom_cal.distortion_coeffs[:-1]

        # note the image shape and center around the image center
    image_shape = np.shape(dataset[0].data)
    yorig, xorig = np.indices(image_shape)
    y0, x0 = image_shape[0]//2, image_shape[1]//2
    yorig -= y0
    xorig -= x0

        # get the number of fitting params from the order
    fitorder = int(astrom_cal.distortion_coeffs[-1])
    fitparams = (fitorder + 1)**2
    true_fitorder = int(expected_coeffs[-1])
    true_fitparams = (true_fitorder + 1)**2

        # reshape the coeff arrays for the best fit and true coeff params
    best_params_x = coeffs[:fitparams]
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
    best_params_y = coeffs[fitparams:]
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
    # assert np.all(np.abs(x_diff) < np.max(np.abs(true_x_diff)))
    # assert np.all(np.abs(y_diff) < np.max(np.abs(true_y_diff)))

    # check that the distortion error in the central 1" x 1" region (center ~45 x 45 pixels) 
    # has distortion error < 4 [mas] (~0.1835 [pixel])
    lower_lim, upper_lim = int((1024//2) - ((1000/21.8)//2)), int((1024//2) + ((1000/21.8)//2))
    central_1arcsec_x = x_diff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]
    central_1arcsec_y = y_diff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]
    true_1arcsec_x = true_x_diff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]
    true_1arcsec_y = true_y_diff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]

    assert np.all(np.abs(central_1arcsec_x - true_1arcsec_x) < 0.1835)
    assert np.all(np.abs(central_1arcsec_y - true_1arcsec_y) < 0.1835)

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


def test_seppa2dxdy():

    seps = np.array([10.0,15.0,20,10,10,10,10])
    pas = np.array([0.,90.,-90,45,-45,135,-135])

    expect_dx = np.array([0.,-15.0,20.,-10./np.sqrt(2.),10./np.sqrt(2.),-10./np.sqrt(2.),10./np.sqrt(2.)])
    expect_dy = np.array([10.,0,0,10./np.sqrt(2.),10./np.sqrt(2.),-10./np.sqrt(2.),-10./np.sqrt(2.)])

    expect_dxdy = np.array([expect_dx,expect_dy])

    dxdy = astrom.seppa2dxdy(seps,pas)

    assert dxdy == pytest.approx(expect_dxdy)

def test_seppa2xy():

    seps = np.array([10.0,15.0,20.,10,10,10,10])
    pas = np.array([0.,90.,-90.,45,-45,135,-135])
    cenx = 25.
    ceny = 35.

    expect_x = np.array([25.,10.0,45.,cenx-10./np.sqrt(2.),cenx+10./np.sqrt(2.),cenx-10./np.sqrt(2.),cenx+10./np.sqrt(2.)])
    expect_y = np.array([45.,35.,35.,ceny+10./np.sqrt(2.),ceny+10./np.sqrt(2.),ceny-10./np.sqrt(2.),ceny-10./np.sqrt(2.)])

    expect_xy = np.array([expect_x,expect_y])

    dxdy = astrom.seppa2xy(seps,pas,cenx,ceny)

    assert dxdy == pytest.approx(expect_xy)

if __name__ == "__main__":
    # test_astrom()
    # test_distortion()
    test_seppa2dxdy()
    test_seppa2xy()