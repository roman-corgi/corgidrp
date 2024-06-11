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

def test_flat_div():
    """
    Generate mock input data and pass into flat division function
    """
    corgidrp.track_individual_errors = True # this test uses individual error components

    ###### create simulated data
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir) 
    mocks.create_simflat_dataset(filedir=datadir)
    
    # simulated images to be checked in flat division
    simdata_filenames=glob.glob(os.path.join(datadir, "sim_flat*.fits"))
    simflat_dataset=data.Dataset(simdata_filenames)
     
    # creat one dummy flat field perform flat division
    mocks.create_flatfield_dummy(filedir=datadir)
	#test data architecture
    flat_filenames = glob.glob(os.path.join(datadir, "flat_field*.fits"))
    flat_dataset = data.Dataset(flat_filenames)
    #check that data is consistently modified
    flat_dataset.all_data[0,0,0] = 1
    assert flat_dataset[0].data[0,0] == 1
    flat_dataset[0].data[0,0] = 1
    assert flat_dataset.all_data[0,0,0] == 1
    
    ###### create flatfield
    flat_frame = detector.create_flatfield(flat_dataset)
    # check the level of counts in flatfield is approximately correct
    assert np.mean(flat_frame.data) == pytest.approx(1, abs=1e-2)
    # check that the error is determined correctly
    assert np.array_equal(np.std(flat_dataset.all_data, axis = 0)/np.sqrt(len(flat_dataset)), flat_frame.err[0])
	# save flatfield
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    flat_filename = "sim_flat_calib.fits"
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    flat_frame.save(filedir=calibdir, filename=flat_filename)
    
    ###### perform flat division
    # load in the flatfield
    flat_filepath = os.path.join(calibdir, flat_filename)
    flatfield = data.FlatField(flat_filepath)
    # divide the simulated dataset with the dummy flatfield
    flatdivided_dataset = l2a_to_l2b.flat_division(simflat_dataset, flatfield)
    
    # perform checks after the flat divison
    assert(flat_filename in str(flatdivided_dataset[0].ext_hdr["HISTORY"]))
    # check the level of the dataset is now approximately 100
    assert np.mean(flatdivided_dataset.all_data) == pytest.approx(150, abs=1e-2)
    # check the propagated errors
    assert flatdivided_dataset[0].err_hdr["Layer_2"] == "FlatField error"
    print("mean of all simulated data",np.mean(simflat_dataset.all_data))
    print("mean of all simulated data error",np.mean(simflat_dataset.all_err) )
    print("mean of all flat divided data:", np.mean(flatdivided_dataset.all_data))
    print("mean of all flat divided data errors:", np.mean(flatdivided_dataset.all_err))
    print("mean of flatfield:", np.mean(flatfield.data))
    print("mean of flatfield err:", np.mean(flatfield.err))
    
    err_flatdiv=np.mean(flatdivided_dataset.all_err)
    err_estimated=np.sqrt(((np.mean(flatfield.data))**2)*(np.mean(simflat_dataset.all_err))**2+((np.mean(simflat_dataset.all_data))**2)*(np.mean(flatfield.err))**2)
    print("mean of all flat divided data errors:",err_flatdiv)
    print("Error estimated:",err_estimated)
    assert(err_flatdiv == pytest.approx(err_estimated, abs = 1e-0))
    
    print(flatdivided_dataset[0].ext_hdr)
    corgidrp.track_individual_errors = old_err_tracking
    
    
if __name__ == "__main__":
    test_flat_div()