import glob
import os

import corgidrp.data as data
import corgidrp.detector as detector
import corgidrp.mocks as mocks
import numpy as np
import pytest


def test_prescan_sub():
    """
    Generate mock input data and pass into dark subtraction function
    """
    ###### create simulated data
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    # dataset = data.Dataset(['example_L1_input.fits'])

    obstype = 'ENG'
    mocks.create_prescan_files(filedir=datadir, obstype=obstype)

    ####### test data architecture
    filenames = glob.glob(os.path.join(datadir, f"sim_prescan_{obstype}*.fits"))

    dataset = data.Dataset(filenames)

    assert len(dataset) == 2
    
    # check that data is consistently modified
    dataset.all_data[0, 0, 0] = 0
    assert dataset[0].data[0, 0] == 0

    ###### create input data
    # input_frame = mocks.create_prescan_files(dataset)

    output_frame = detector.prescan_biassub_v2(dataset)

    # check the level of dark current is approximately correct
    # assert np.mean(dark_frame.data) == pytest.approx(150, abs=1e-2)

    # save dark
    # calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    # dark_filename = "sim_dark_calib.fits"
    # if not os.path.exists(calibdir):
    #     os.mkdir(calibdir)
    # dark_frame.save(filedir=calibdir, filename=dark_filename)

    ###### perform dark subtraction
    # load in the dark
    # dark_filepath = os.path.join(calibdir, dark_filename)
    # new_dark = data.Dark(dark_filepath)
    # subtract darks from itself
    # darkest_dataset = detector.dark_subtraction(dark_dataset, new_dark)

    # check the level of the dataset is now approximately 0 
    # assert np.mean(darkest_dataset.all_data) == pytest.approx(0, abs=1e-2)
    # print(np.mean(darkest_dataset.all_data))
    # print(darkest_dataset[0].ext_hdr)
    

if __name__ == "__main__":
    test_prescan_sub()