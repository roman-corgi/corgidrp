import os
import glob
import pickle
import pytest
import numpy as np
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.detector as detector
import corgidrp.flat as flat
import corgidrp.l2a_to_l2b as l2a_to_l2b
import photutils.centroids as centr


old_err_tracking = corgidrp.track_individual_errors

def test_create_flatfield_neptune():
    """
    Generate mock input data and pass into flat division function
    """
    corgidrp.track_individual_errors = True # this test uses individual error components

    ###### create simulated data
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir) 
    
    # simulated images to be checked in flat division
    simflat_dataset = mocks.create_simflat_dataset(filedir=datadir)
    
    ###### create simulated raster scanned data
    # check that simulated raster scanned data folder exists, and create if not
    file_dir = os.path.join(os.path.dirname(__file__), "simdata_rasterscan")
    data_dir = os.path.join(os.path.dirname(__file__),"test_data/")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir) 
    filenames = glob.glob(os.path.join(data_dir, "med*.fits"))
    data_set = data.Dataset(filenames)
    # creating flatfield for neptune for band 1
    planet='neptune'; band='1'
    flat_dataset = mocks.create_onsky_rasterscans(data_set,planet='neptune',band='1',im_size=1024,d=45, n_dith=3,radius=50,snr=250,snr_constant=4.55)
    
    ####### create flat field 
    onskyflat_field = flat.create_onsky_flatfield(flat_dataset, planet='neptune',band='1',up_radius=55, im_size=1024, N=2, rad_mask=1.26,  planet_rad=50, n_pix=300, n_pad=0,image_center_x=512,image_center_y=512)

    assert np.nanmean(onskyflat_field.data) == pytest.approx(1, abs=1e-2)
    assert np.size(np.where(np.isnan(onskyflat_field.data))) == 0 # no bad pixels
    
    # check the flat can be pickled (for CTC operations)
    pickled = pickle.dumps(onskyflat_field)
    pickled_flat = pickle.loads(pickled)
    assert np.all(onskyflat_field.data == pickled_flat.data)

    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    onskyflat_field.save(filedir=calibdir)
    
    ###### perform flat division
    # load in the flatfield
    # check that the filename is what we expect
    flat_filename = flat_dataset[-1].filename.replace("_l2a", "_flt_cal")
    flat_filepath = os.path.join(calibdir, flat_filename)
    onsky_flatfield = data.FlatField(flat_filepath)

    # check the flat can be pickled (for CTC operations)
    pickled = pickle.dumps(onskyflat_field)
    pickled_flat = pickle.loads(pickled)
    assert np.all(onskyflat_field.data == pickled_flat.data)
    
    flatdivided_dataset = l2a_to_l2b.flat_division(simflat_dataset,onsky_flatfield)
    
    
    # perform checks after the flat divison for one of the dataset
    assert(flat_filename in "".join(flatdivided_dataset[0].ext_hdr["HISTORY"]))


    
    # check the propagated errors for one of the dataset
    assert flatdivided_dataset[0].err_hdr["Layer_2"] == "FlatField_error"
    print("mean of all simulated data",np.mean(simflat_dataset.all_data))
    print("mean of all simulated data error",np.nanmean(simflat_dataset.all_err) )
    print("mean of all flat divided data:", np.nanmean(flatdivided_dataset.all_data))
    print("mean of flatfield:", np.nanmean(onsky_flatfield.data))
    print("mean of flatfield err:", np.nanmean(onsky_flatfield.err))
    
    err_flatdiv=np.nanmean(flatdivided_dataset.all_err)
    err_estimated=np.sqrt(((np.nanmean(onsky_flatfield.data))**2)*(np.nanmean(simflat_dataset.all_err))**2+((np.nanmean(simflat_dataset.all_data))**2)*(np.nanmean(onsky_flatfield.err))**2)
    print("mean of all flat divided data errors:",err_flatdiv)
    print("Error estimated:",err_estimated)
    assert(err_flatdiv == pytest.approx(err_estimated, abs = 1e-1))

    corgidrp.track_individual_errors = old_err_tracking

    return
    
     #creating flatfield using uranus for band4 
    
def test_create_flatfield_uranus():

    corgidrp.track_individual_errors = True
    ###### create simulated data
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir) 
    
    # simulated images to be checked in flat division
    simflat_dataset = mocks.create_simflat_dataset(filedir=datadir)
    
    ###### create simulated raster scanned data
    # check that simulated raster scanned data folder exists, and create if not
    file_dir = os.path.join(os.path.dirname(__file__), "simdata_rasterscan")
    data_dir = os.path.join(os.path.dirname(__file__),"test_data/")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir) 
    filenames = glob.glob(os.path.join(data_dir, "med*.fits"))
    data_set = data.Dataset(filenames)
    planet='uranus'; band='4'
    flat_dataset = mocks.create_onsky_rasterscans(data_set,planet='uranus',band='4',im_size=1024,d=50, n_dith=3,radius=90,snr=250,snr_constant=9.66)
    
    ####### create flat field
    onskyflat_field = flat.create_onsky_flatfield(flat_dataset, planet='uranus',band='1',up_radius=55, im_size=1024, N=2, rad_mask=1.26,  planet_rad=50, n_pix=320, n_pad=0,image_center_x=512,image_center_y=512)

    assert np.nanmean(onskyflat_field.data) == pytest.approx(1, abs=1e-2)
    assert np.size(np.where(np.isnan(onskyflat_field.data))) == 0 # no bad pixels
    
    
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    onskyflat_field.save(filedir=calibdir)
    
    ###### perform flat division
    # load in the flatfield
    # check that the filename is what we expect
    flat_filename = flat_dataset[-1].filename.replace("_l2a", "_flt_cal")
    flat_filepath = os.path.join(calibdir, flat_filename)
    onsky_flatfield = data.FlatField(flat_filepath)

    
    flatdivided_dataset = l2a_to_l2b.flat_division(simflat_dataset,onsky_flatfield)
    
    
    # perform checks after the flat divison for one of the dataset
    assert(flat_filename in "".join(flatdivided_dataset[0].ext_hdr["HISTORY"]))
    
    # check the propagated errors for one of the dataset
    assert flatdivided_dataset[0].err_hdr["Layer_2"] == "FlatField_error"
    print("mean of all simulated data",np.mean(simflat_dataset.all_data))
    print("mean of all simulated data error",np.nanmean(simflat_dataset.all_err) )
    print("mean of all flat divided data:", np.nanmean(flatdivided_dataset.all_data))
    print("mean of flatfield:", np.nanmean(onsky_flatfield.data))
    print("mean of flatfield err:", np.nanmean(onsky_flatfield.err))
    
    err_flatdiv=np.nanmean(flatdivided_dataset.all_err)
    err_estimated=np.sqrt(((np.nanmean(onsky_flatfield.data))**2)*(np.nanmean(simflat_dataset.all_err))**2+((np.nanmean(simflat_dataset.all_data))**2)*(np.nanmean(onsky_flatfield.err))**2)
    print("mean of all flat divided data errors:",err_flatdiv)
    print("Error estimated:",err_estimated)
    assert(err_flatdiv == pytest.approx(err_estimated, abs = 1e-1))
    
    corgidrp.track_individual_errors = old_err_tracking

    return

if __name__ == "__main__":
    test_create_flatfield_uranus()
    test_create_flatfield_neptune()
    