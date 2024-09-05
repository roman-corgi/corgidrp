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
    filenames = glob.glob(os.path.join(data_dir, "med*.fits"))
    data_set = data.Dataset(filenames)
    # creating flatfield for neptune for band 1
    planet='neptune'; band='1'
    mocks.create_onsky_rasterscans(data_set,filedir=file_dir,planet='neptune',band='1',im_size=1024,d=50, n_dith=3,numfiles=36,radius=54,snr=250,snr_constant=4.55)
    
    ####### create flat field 
    flat_dataset=[]
    flat_filenames = glob.glob(os.path.join(file_dir, "neptune*.fits"))
    flat_dataset_all = data.Dataset(flat_filenames)
    for i in range(len(flat_dataset_all)):
        target=flat_dataset_all[i].pri_hdr['TARGET']
        filter=flat_dataset_all[i].pri_hdr['FILTER']
        if planet==target and band==filter: 
            flat_dataset.append(flat_dataset_all[i])
    onskyflat_field = detector.create_onsky_flatfield(flat_dataset, planet='neptune',band='1',up_radius=55, im_size=1024, N=3, rad_mask=1.26,  planet_rad=50, n_pix=165, n_pad=0)

    assert np.nanmean(onskyflat_field.data) == pytest.approx(1, abs=1e-2)
    
    
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    
    flat_filename = "sim_flatfield_"+str(planet)+"_"+str(band)+".fits"
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    onskyflat_field.save(filedir=calibdir, filename=flat_filename)
    
    ###### perform flat division
    # load in the flatfield
    flat_filepath = os.path.join(calibdir, flat_filename)
    onsky_flatfield = data.FlatField(flat_filepath)

    
    flatdivided_dataset = l2a_to_l2b.flat_division(simflat_dataset,onsky_flatfield)
    print(flatdivided_dataset[0].ext_hdr["HISTORY"])
    
    
    # perform checks after the flat divison for one of the dataset
    assert(flat_filename in str(flatdivided_dataset[0].ext_hdr["HISTORY"]))


    
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
    
    print(flatdivided_dataset[0].ext_hdr)

    return
    
     #creating flatfield using uranus for band4 
    
def test_create_flatfield_uranus():
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
    filenames = glob.glob(os.path.join(data_dir, "med*.fits"))
    data_set = data.Dataset(filenames)
    planet='uranus'; band='4'
    mocks.create_onsky_rasterscans(data_set,filedir=file_dir,planet='uranus',band='4',im_size=1024,d=65, n_dith=2,numfiles=36,radius=90,snr=250,snr_constant=9.66)
    
    ####### create flat field
    flat_dataset=[]
    flat_filenames = glob.glob(os.path.join(file_dir, "uranus*.fits"))
    flat_dataset_all = data.Dataset(flat_filenames)
    for i in range(len(flat_dataset_all)):
        target=flat_dataset_all[i].pri_hdr['TARGET']
        filter=flat_dataset_all[i].pri_hdr['FILTER']
        if planet==target and band==filter: 
            flat_dataset.append(flat_dataset_all[i])
    onskyflat_field = detector.create_onsky_flatfield(flat_dataset, planet='uranus',band='4',up_radius=55, im_size=1024, N=3, rad_mask=1.75,  planet_rad=65, n_pix=165)

    assert np.nanmean(onskyflat_field.data) == pytest.approx(1, abs=1e-2)
    
    
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    
    flat_filename = "sim_flatfield_"+str(planet)+"_"+str(band)+".fits"
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    onskyflat_field.save(filedir=calibdir, filename=flat_filename)
    
    ###### perform flat division
    # load in the flatfield
    flat_filepath = os.path.join(calibdir, flat_filename)
    onsky_flatfield = data.FlatField(flat_filepath)

    
    flatdivided_dataset = l2a_to_l2b.flat_division(simflat_dataset,onsky_flatfield)
    print(flatdivided_dataset[0].ext_hdr["HISTORY"])
    
    
    # perform checks after the flat divison for one of the dataset
    assert(flat_filename in str(flatdivided_dataset[0].ext_hdr["HISTORY"]))
    
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
    
    print(flatdivided_dataset[0].ext_hdr)
    corgidrp.track_individual_errors = old_err_tracking

    return

if __name__ == "__main__":
    test_create_flatfield_neptune()
    test_create_flatfield_uranus()