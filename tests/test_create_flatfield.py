import os
import glob
import pytest
import numpy as np
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.detector as detector
import corgidrp.l2a_to_l2b as l2a_to_l2b
import photutils.centroids as centr


old_err_tracking = corgidrp.track_individual_errors


def test_create_flatfield():
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
    
    ###### create simulated raster scanned data
    # check that simulated raster scanned data folder exists, and create if not
    file_dir = os.path.join(os.path.dirname(__file__), "simdata_rasterscan")
    data_dir = os.path.join(os.path.dirname(__file__),"test_data/")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir) 
    print(data_dir)
    print(file_dir)
    mocks.create_onsky_rasterscans(datadir=data_dir,filedir=file_dir)
    
    ###### create flat field
    flat_filenames = glob.glob(os.path.join(file_dir, "*.fits"))
    flat_dataset = data.Dataset(flat_filenames)
    onskyflat_frame = detector.create_onsky_flatfield(flat_dataset)
    neptune_band1_flatfield=onskyflat_frame[0]
    neptune_band4_flatfield=onskyflat_frame[1]
    uranus_band1_flatfield=onskyflat_frame[2]
    uranus_band4_flatfield=onskyflat_frame[3]
    assert np.nanmean(neptune_band1_flatfield.data) == pytest.approx(1, abs=1e-2)
    assert np.nanmean(neptune_band4_flatfield.data) == pytest.approx(1, abs=1e-2)
    assert np.nanmean(uranus_band1_flatfield.data) == pytest.approx(1, abs=1e-2)
    assert np.nanmean(uranus_band4_flatfield.data) == pytest.approx(1, abs=1e-2)
    
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    
    flat_filename_1 = "sim_flat_calib_neptune_band1.fits"
    flat_filename_2 = "sim_flat_calib_neptune_band4.fits"
    flat_filename_3 = "sim_flat_calib_uranus_band1.fits"
    flat_filename_4 = "sim_flat_calib_uranus_band4.fits"
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    neptune_band1_flatfield.save(filedir=calibdir, filename=flat_filename_1)
    neptune_band4_flatfield.save(filedir=calibdir, filename=flat_filename_2)
    uranus_band1_flatfield.save(filedir=calibdir, filename=flat_filename_3)
    uranus_band4_flatfield.save(filedir=calibdir, filename=flat_filename_4)
    
    ###### perform flat division
    # load in the flatfield
    flat_filepath = os.path.join(calibdir, flat_filename_1)
    neptune_band1_flatfield = data.FlatField(flat_filepath)
    flat_filepath = os.path.join(calibdir, flat_filename_2)
    neptune_band4_flatfield = data.FlatField(flat_filepath)
    flat_filepath = os.path.join(calibdir, flat_filename_3)
    uranus_band1_flatfield = data.FlatField(flat_filepath)
    flat_filepath = os.path.join(calibdir, flat_filename_4)
    uranus_band4_flatfield = data.FlatField(flat_filepath)
    
    flatdivided_dataset_1 = l2a_to_l2b.flat_division(simflat_dataset, neptune_band1_flatfield)
    flatdivided_dataset_2 = l2a_to_l2b.flat_division(simflat_dataset, neptune_band4_flatfield)
    flatdivided_dataset_3 = l2a_to_l2b.flat_division(simflat_dataset, uranus_band1_flatfield)
    flatdivided_dataset_4 = l2a_to_l2b.flat_division(simflat_dataset, uranus_band4_flatfield)
    
	# perform checks after the flat divison for one of the dataset
    assert(flat_filename_1 in str(flatdivided_dataset_1[0].ext_hdr["HISTORY"]))
	
	# check the propagated errors for one of the dataset
    assert flatdivided_dataset_1[0].err_hdr["Layer_2"] == "FlatField_error"
    print("mean of all simulated data",np.mean(simflat_dataset.all_data))
    print("mean of all simulated data error",np.nanmean(simflat_dataset.all_err) )
    print("mean of all flat divided data:", np.nanmean(flatdivided_dataset_1.all_data))
    print("mean of flatfield:", np.nanmean(neptune_band1_flatfield.data))
    print("mean of flatfield err:", np.nanmean(neptune_band1_flatfield.err))
    
    err_flatdiv=np.nanmean(flatdivided_dataset_1.all_err)
    err_estimated=np.sqrt(((np.nanmean(neptune_band1_flatfield.data))**2)*(np.nanmean(simflat_dataset.all_err))**2+((np.nanmean(simflat_dataset.all_data))**2)*(np.nanmean(neptune_band1_flatfield.err))**2)
    print("mean of all flat divided data errors:",err_flatdiv)
    print("Error estimated:",err_estimated)
    
    print(flatdivided_dataset_1[0].ext_hdr)
    corgidrp.track_individual_errors = old_err_tracking


if __name__ == "__main__":
    test_create_flatfield()