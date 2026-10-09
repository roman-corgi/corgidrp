import os
import pickle
import numpy as np
import pytest
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.astrom as astrom
import corgidrp.data as data
import astropy
import astropy.io.ascii as ascii
from termcolor import cprint
from astropy.coordinates import SkyCoord

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
    dataset = mocks.create_astrom_data(field_path=field_path, rotation=20, dither_pointings=4, vignette_radius=None)

    # check the dataset format
    assert len(dataset) == 5  # one pointing + 4 dithers
    assert isinstance(dataset[0], data.Image)

    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, find_threshold=25)

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
    true_boresight_skycoord = SkyCoord(ra=target[0], dec=target[1], unit='deg')
    ra, dec = astrom_cal.boresight
    actual_boresight_skycoord = SkyCoord(ra=ra, dec=dec, unit='deg')

    ra_error, dec_error = actual_boresight_skycoord.spherical_offsets_to(true_boresight_skycoord)
    assert ra_error.deg == pytest.approx(0, abs=8.333e-7)     # reported as ra offset
    assert dec_error.deg == pytest.approx(0, abs=8.333e-7)

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

def test_astrom_ref_pixel():
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
    dataset = mocks.create_astrom_data(field_path=field_path, rotation=20, dither_pointings=4, vignette_radius=None)

    # check the dataset format
    assert len(dataset) == 5  # one pointing + 4 dithers
    assert isinstance(dataset[0], data.Image)

    # perform the astrometric calibration
    # * with respect to a new reference pixel
    reference_pixel = (500., 450.)
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, find_threshold=25, reference_pixel=reference_pixel)

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
    ###*** Use SkyCoord here to translate position difference correctly ***###
    vert_ang = np.radians(expected_northangle)
    pc = np.array([[-np.cos(vert_ang), np.sin(vert_ang)], [np.sin(vert_ang), np.cos(vert_ang)]])
    cdmatrix = pc * (expected_platescale * 0.001) / 3600.

    new_hdr = {}
    new_hdr['CD1_1'] = cdmatrix[0,0]
    new_hdr['CD1_2'] = cdmatrix[0,1]
    new_hdr['CD2_1'] = cdmatrix[1,0]
    new_hdr['CD2_2'] = cdmatrix[1,1]
    new_hdr['CRPIX1'] = 512.
    new_hdr['CRPIX2'] = 512.
    new_hdr['CTYPE1'] = 'RA---TAN'
    new_hdr['CTYPE2'] = 'DEC--TAN'
    new_hdr['CDELT1'] = (expected_platescale * 0.001) / 3600.
    new_hdr['CDELT2'] = (expected_platescale * 0.001) / 3600.
    new_hdr['CRVAL1'] = target[0]       # the simulated image should have no shift from the target at 512., 512.
    new_hdr['CRVAL2'] = target[1]
    w = astropy.wcs.WCS(new_hdr)

    # use astropy wcs to find the true coordinate value of the reference pixel
    # assume an arbitrary reference pixel location [500., 450.] which is specified in the recipe
    expected_center_skycoord = astropy.wcs.utils.pixel_to_skycoord(reference_pixel[0], reference_pixel[1], wcs=w, origin=1)

    ra, dec = astrom_cal.boresight
    actual_boresight_skycoord = SkyCoord(ra=ra, dec=dec, unit='deg')
    ra_error, dec_error = actual_boresight_skycoord.spherical_offsets_to(expected_center_skycoord)
    
    test_result_ra_error = (ra_error.deg == pytest.approx(0, abs=8.333e-7))
    print(f'\nBoresight RA estimate from boresight_calibration() is accurate: {expected_center_skycoord.ra.value} +/- {8.333e-7 * (3_600_000):.4f} [mas]: ', end='')
    print_pass() if test_result_ra_error else print_fail()
    assert test_result_ra_error     # reported as ra offset

    test_result_dec_error = (dec_error.deg == pytest.approx(0, abs=8.333e-7))
    print(f'\nBoresight Dec estimate from boresight_calibration() is accurate: {expected_center_skycoord.dec.value} +/- {8.333e-7 * (3_600_000):.4f} [mas]: ', end='')
    print_pass() if test_result_dec_error else print_fail()
    assert test_result_dec_error    

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



def test_astrom_vignette(vignette_radius=3_460):
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
    dataset = mocks.create_astrom_data(field_path=field_path, rotation=20, dither_pointings=4, vignette_radius=vignette_radius)

    # check the dataset format
    assert len(dataset) == 5  # one pointing + 4 dithers
    assert isinstance(dataset[0], data.Image)

    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, find_threshold=25)

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
    true_boresight_skycoord = SkyCoord(ra=target[0], dec=target[1], unit='deg')
    ra, dec = astrom_cal.boresight
    actual_boresight_skycoord = SkyCoord(ra=ra, dec=dec, unit='deg')

    ra_error, dec_error = actual_boresight_skycoord.spherical_offsets_to(true_boresight_skycoord)
    assert ra_error.deg == pytest.approx(0, abs=8.333e-7)     # reported as ra offset
    assert dec_error.deg == pytest.approx(0, abs=8.333e-7)

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
    dataset = mocks.create_astrom_data(field_path=field_path, rotation=20, distortion_coeffs_path=distortion_coeffs_path, vignette_radius=3_460, dither_pointings=4)

    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, find_threshold=25, find_distortion=True, fitorder=3, position_error=0.5)

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

@pytest.mark.parametrize("num_pointings", [1, 2, 3])
@pytest.mark.parametrize("position_angles", [(0., 0., 0.), (0., 90., 0.)])
@pytest.mark.filterwarnings("ignore:Keyword (RA_APER|PA_APER) not identical across frames:RuntimeWarning")
def test_boresight_combining_preserves_pointing_order(monkeypatch, num_pointings,
                                                    position_angles):
    """Keep the first input pointing as the boresight after header grouping.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for isolating source measurements.
        num_pointings (int): Number of distinct pointings to combine.
        position_angles (tuple): Position angle for each pointing.
    """
    frames = []
    targets = ["Z reference", "A dither", "M dither"]
    base_primary_header, base_extension_header, _, _, _ = mocks.create_default_L2b_headers()
    for repeat in range(2):
        for pointing in range(num_pointings):
            primary_header = base_primary_header.copy()
            extension_header = base_extension_header.copy()
            primary_header["TARGET"] = targets[pointing]
            primary_header["RA"] = 80. + pointing * 0.01
            primary_header["DEC"] = -69.
            primary_header["RA_APER"] = primary_header["RA"]
            primary_header["DEC_APER"] = primary_header["DEC"]
            primary_header["PA_APER"] = position_angles[pointing]
            frame = data.Image(np.full((4, 4), pointing + 10 * repeat, dtype=float),
                               pri_hdr=primary_header, ext_hdr=extension_header)
            frame.filename = f"pointing_{pointing}_repeat_{repeat}.fits"
            frames.append(frame)
    dataset = data.Dataset(frames)

    measured_images = []

    def find_sources(image, **keywords):
        """Record the combined image used for each measurement.

        Args:
            image (numpy.ndarray): Combined image.
            keywords (dict): Source-finding parameters.

        Returns:
            None: Placeholder for the isolated source matching.
        """
        measured_images.append(image.copy())
        return None

    monkeypatch.setattr(astrom, "find_source_locations", find_sources)
    monkeypatch.setattr(astrom, "match_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(astrom, "compute_platescale_and_northangle",
                        lambda *args, **kwargs: (21.8, -45.))
    monkeypatch.setattr(astrom, "compute_boresight", lambda *args, **kwargs: (0., 0.))

    calibration = astrom.boresight_calibration(dataset, frames_to_combine=True)

    np.testing.assert_allclose(calibration.boresight, [80., -69.])
    for pointing, image in enumerate(measured_images):
        np.testing.assert_array_equal(image, np.full((4, 4), pointing + 5.))
        assert calibration.ext_hdr[f"F{pointing}POS"] == pytest.approx(80. + pointing * 0.01)
    assert len(measured_images) == num_pointings


if __name__ == "__main__":
    test_astrom()
    test_astrom_ref_pixel()
    test_astrom_vignette()
    test_distortion()
    test_seppa2dxdy()
    test_seppa2xy()
    test_create_circular_mask()
    test_get_polar_dist()
    test_transform_coeff_to_distortion_map()