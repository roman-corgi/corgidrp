import os
import pickle
import numpy as np
import pytest
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.astrom as astrom
import corgidrp.data as data
import astropy.io.ascii as ascii
from termcolor import cprint


def print_fail():
    cprint(' FAIL ', "black", "on_red")


def print_pass():
    cprint(' PASS ', "black", "on_green")


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
    
    # create a dataset with dithers
    # dataset = mocks.create_astrom_data(field_path=field_path, filedir=datadir, rotation=20, dither_pointings=4)
    dataset = mocks.create_astrom_data(field_path=field_path, rotation=20, dither_pointings=4)

    # image_path = os.path.join(datadir, 'simcal_astrom.fits')
    # open the image
    # dataset = data.Dataset([image_path])

    # check the dataset format
    assert len(dataset) == 5  # one pointing + 4 dithers
    assert isinstance(dataset[0], data.Image)

    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, find_threshold=200)

    # the data was generated to have the following image properties
    expected_platescale = 21.8
    atol_platescale = 0.5

    # check orientation is correct within 0.05 [deg]
    # and plate scale is correct within 0.5 [mas] (arbitrary)
    expected_northangle = 20
    atol_northangle = 0.05
    test_result_platescale = (astrom_cal.northangle == pytest.approx(expected_northangle, abs=atol_northangle))
    print(f'\nPlate scale estimate from boresight_calibration() is accurate: {expected_platescale} +/- {atol_platescale}: ', end='')
    print_pass() if test_result_platescale else print_fail()
    assert test_result_platescale

    test_result_northangle = (astrom_cal.northangle == pytest.approx(expected_northangle, abs=atol_northangle))
    assert test_result_northangle

    # check that the center is correct within 3 [mas]
    # the simulated image should have zero offset
    target = dataset[0].pri_hdr['RA'], dataset[0].pri_hdr['DEC']
    ra, dec = astrom_cal.boresight
    assert ra == pytest.approx(target[0], abs=8.333e-7)     # reported as ra offset
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

    # create dithered dataset 
    # mocks.create_astrom_data(field_path=field_path, filedir=datadir, rotation=20, distortion_coeffs_path=distortion_coeffs_path, dither_pointings=4)
    dataset = mocks.create_astrom_data(field_path=field_path, rotation=20, distortion_coeffs_path=distortion_coeffs_path, dither_pointings=4)

    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, find_threshold=400, find_distortion=True, fitorder=3, position_error=0.5)

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

    # check that the distortion error in the central 1" x 1" region (center ~45 x 45 pixels) 
    # has distortion error < 4 [mas] (~0.1835 [pixel])
    atol_dist_mas = 4
    mas_per_pix = 21.8
    mas_across = 1000
    atol_dist_pix = atol_dist_mas/mas_per_pix
    lower_lim, upper_lim = int((1024//2) - ((mas_across/mas_per_pix)//2)), int((1024//2) + ((mas_across/mas_per_pix)//2))

    central_1arcsec_x = x_diff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]
    central_1arcsec_y = y_diff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]
    
    true_1arcsec_x = true_x_diff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]
    true_1arcsec_y = true_y_diff[lower_lim: upper_lim+1,lower_lim: upper_lim+1]

    test_result_distortion_x = np.all(np.abs(central_1arcsec_x - true_1arcsec_x) < atol_dist_pix)
    print(f'\nDistortion map in x is accurate within {atol_dist_mas} mas in central {mas_across} mas x {mas_across} mas: ', end='')
    print_pass() if test_result_distortion_x else print_fail()
    assert test_result_distortion_x

    test_result_distortion_y = np.all(np.abs(central_1arcsec_y - true_1arcsec_y) < atol_dist_pix)
    print(f'\nDistortion map in y is accurate within {atol_dist_mas} mas in central {mas_across} mas x {mas_across} mas: ', end='')
    print_pass() if test_result_distortion_y else print_fail()
    assert test_result_distortion_y

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
    """Test that conversion from separation/position angle to delta x/y 
    produces the expected result for varying input separations and angles."""

    seps = np.array([10.0,15.0,20,10,10,10,10])
    pas = np.array([0.,90.,-90,45,-45,135,-135])

    expect_dx = np.array([0.,-15.0,20.,-10./np.sqrt(2.),10./np.sqrt(2.),-10./np.sqrt(2.),10./np.sqrt(2.)])
    expect_dy = np.array([10.,0,0,10./np.sqrt(2.),10./np.sqrt(2.),-10./np.sqrt(2.),-10./np.sqrt(2.)])

    expect_dxdy = np.array([expect_dx,expect_dy])

    dxdy = astrom.seppa2dxdy(seps,pas)

    assert dxdy == pytest.approx(expect_dxdy)


def test_seppa2xy():
    """Test that conversion from separation/position angle to detector x/y coordinates
    produces the expected result for varying input separations and angles."""

    seps = np.array([10.0,15.0,20.,10,10,10,10])
    pas = np.array([0.,90.,-90.,45,-45,135,-135])
    cenx = 25.
    ceny = 35.

    expect_x = np.array([25.,10.0,45.,cenx-10./np.sqrt(2.),cenx+10./np.sqrt(2.),cenx-10./np.sqrt(2.),cenx+10./np.sqrt(2.)])
    expect_y = np.array([45.,35.,35.,ceny+10./np.sqrt(2.),ceny+10./np.sqrt(2.),ceny-10./np.sqrt(2.),ceny-10./np.sqrt(2.)])

    expect_xy = np.array([expect_x,expect_y])

    dxdy = astrom.seppa2xy(seps,pas,cenx,ceny)

    assert dxdy == pytest.approx(expect_xy)

def test_create_circular_mask():
    """Test that astrom.create_circular_mask() calculates the center 
    of an image correctly and produces a mask."""

    img = np.zeros((10,10))
    r = 2

    mask1 = astrom.create_circular_mask(img.shape, center=None, r=r)
    mask2 = astrom.create_circular_mask(img.shape, center=(4.5,4.5), r=r)

    # Make sure automatic centering works
    assert mask1 == pytest.approx(mask2)

    # Make sure some pixels have been masked
    assert mask1.size - np.count_nonzero(mask1) > 0


def test_get_polar_dist():
    """Test that astrom.get_polar_dist() calculates distances correctly 
    in varying directions."""
    
    # Test vertical line
    seppa1 = (10,0)
    seppa2 = (10,180)
    dist = 20.

    assert astrom.get_polar_dist(seppa1,seppa2) == dist

    # Test horizontal line
    seppa1 = (10,90)
    seppa2 = (10,270)
    dist = 20.

    assert astrom.get_polar_dist(seppa1,seppa2) == dist

    # Test 45 degree line
    seppa1 = (10,0)
    seppa2 = (10,90)
    dist = 10. * np.sqrt(2.)

    assert astrom.get_polar_dist(seppa1,seppa2) == dist

    pass

def test_transform_coeff_to_distortion_map():
    """Test that astrom.transform_coeff_to_map() produces the correct distortion map from given
    legendre coefficients."""

    im_shape = np.array([1024, 1024])
    fit_order = 3

    # Test coeffs corresponding to no distortion
    zero_coeffs = np.array([  0,   0,   0,   0, 500,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0, 500,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0])

    z_xdiff, z_ydiff = astrom.transform_coeff_to_map(zero_coeffs, fit_order, im_shape)

    # Check that the computed distortion map is zero everywhere
    assert np.all(z_xdiff == 0)
    assert np.all(z_ydiff == 0)

if __name__ == "__main__":
    test_astrom()
    test_distortion()
    test_seppa2dxdy()
    test_seppa2xy()
    test_create_circular_mask()
    test_get_polar_dist()
    test_transform_coeff_to_distortion_map()