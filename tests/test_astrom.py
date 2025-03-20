import os
import pickle
import numpy as np
import pytest
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.astrom as astrom
import corgidrp.data as data

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
    assert len(astrom_cal.data) == 4

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

def test_seppa2dxdy():

    seps = np.array([10.0,15.0])
    pas = np.array([0.,90.])

    expect_dx = np.array([0.,-15.0])
    expect_dy = np.array([10.,0])

    expect_dxdy = np.array([expect_dx,expect_dy])

    dxdy = astrom.seppa2dxdy(seps,pas)

    assert dxdy == pytest.approx(expect_dxdy)

def test_seppa2xy():

    seps = np.array([10.0,15.0])
    pas = np.array([0.,90.])
    cenx = 25.
    ceny = 35.

    expect_x = np.array([25.,10.0])
    expect_y = np.array([45.,35.])

    expect_xy = np.array([expect_x,expect_y])

    dxdy = astrom.seppa2xy(seps,pas,cenx,ceny)

    assert dxdy == pytest.approx(expect_xy)

if __name__ == "__main__":
    #test_astrom()
    test_seppa2dxdy()
    test_seppa2xy()