import os
import glob
import pytest
import numpy as np
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.detector as detector
import corgidrp.l2a_to_l2b as l2a_to_l2b

def test_flat_div(): 
    
    """
    Generate mock input data and pass into flat division function
    """
        ###### create simulated data
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simflatdata")
    if not os.path.exists(datadir):
    os.mkdir(datadir)
    
    mocks.create_flat_calib_files(filedir=datadir)
    
    flat_filenames = glob.glob(os.path.join(datadir, "simcal_flat*.fits"))

    flat_dataset = data.Dataset(flat_filenames)

    assert len(flat_dataset) == 10
   
    
    
    ###### create master flat
    flat_frame = detector.create_master_flat(flat_dataset)
    
    
    # check that the error is determined correctly
    assert np.array_equal(np.std(flat_dataset.all_data, axis = 0)/np.sqrt(len(flat_dataset)), flat_frame.err)
    
    # save flat
    calibdir = os.path.join(os.path.dirname('test'), "testcalib")
    flat_filename = "sim_flat_calib.fits"
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
        flat_frame.save(filedir=calibdir, filename=flat_filename)  
    
    ###### perform flat division
    # load in the masterflat
    flat_filepath = os.path.join(calibdir, flat_filename)
    new_masterflat = data.Masterflat(flat_filepath)
    # divide sim_data from masterflat
    flat_dataset = l2a_to_l2b.flat_division(flat_dataset, new_masterflat)
    #assert(flat_filename in str(flat_dataset[0].ext_hdr["HISTORY"]))

    # check the level of the dataset is now approximately 0 
    assert np.mean(flat_dataset.all_data) == pytest.approx(1, abs=1e-2)
    # check the propagated errors
    assert flat_dataset[0].err_hdr["Layer_2"] == "masterflat_error"
    assert(np.mean(flat_dataset.all_err) == pytest.approx(np.mean(flat_frame.err), abs = 1e-2))
    print("mean of all data:", np.mean(flat_dataset.all_data))
    print("mean of all errors:", np.mean(flat_dataset.all_err))
    print(flat_dataset[0].ext_hdr)
    
    
if __name__ == "__main__":
    test_flat_div()