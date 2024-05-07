import os
import glob
import pytest
import numpy as np
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.detector as detector
import corgidrp.l2a_to_l2b as l2a_to_l2b

old_err_tracking = corgidrp.track_individual_errors

def test_dark_sub():
    """
    Generate mock input data and pass into dark subtraction function
    """
    corgidrp.track_individual_errors = True # this test uses individual error components

    ###### create simulated data
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    
    mocks.create_dark_calib_files(filedir=datadir)

    ####### test data architecture
    dark_filenames = glob.glob(os.path.join(datadir, "simcal_dark*.fits"))

    dark_dataset = data.Dataset(dark_filenames)

    assert len(dark_dataset) == 10
    
    # check that data is consistently modified
    dark_dataset.all_data[0,0,0] = 0
    assert dark_dataset[0].data[0,0] == 0
    
    dark_dataset[0].data[0,0] = 1
    assert dark_dataset.all_data[0,0,0] == 1

    ###### create dark
    dark_frame = detector.create_dark_calib(dark_dataset)

    # check the level of dark current is approximately correct
    assert np.mean(dark_frame.data) == pytest.approx(150, abs=1e-2)
    
    # check that the error is determined correctly
    assert np.array_equal(np.std(dark_dataset.all_data, axis = 0)/np.sqrt(len(dark_dataset)), dark_frame.err[0])
    
    # save dark
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    dark_filename = "sim_dark_calib.fits"
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    dark_frame.save(filedir=calibdir, filename=dark_filename)

    ###### perform dark subtraction
    # load in the dark
    dark_filepath = os.path.join(calibdir, dark_filename)
    new_dark = data.Dark(dark_filepath)
    # subtract darks from itself
    darkest_dataset = l2a_to_l2b.dark_subtraction(dark_dataset, new_dark)
    assert(dark_filename in str(darkest_dataset[0].ext_hdr["HISTORY"]))

    # check the level of the dataset is now approximately 0 
    assert np.mean(darkest_dataset.all_data) == pytest.approx(0, abs=1e-2)
    # check the propagated errors
    assert darkest_dataset[0].err_hdr["Layer_2"] == "dark_error"
    assert(np.mean(darkest_dataset.all_err) == pytest.approx(np.mean(dark_frame.err), abs = 1e-2))
    print("mean of all data:", np.mean(darkest_dataset.all_data))
    print("mean of all errors:", np.mean(darkest_dataset.all_err))
    print(darkest_dataset[0].ext_hdr)

    corgidrp.track_individual_errors = old_err_tracking
    

if __name__ == "__main__":
    test_dark_sub()