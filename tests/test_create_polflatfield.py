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

def test_create_polflatfield_pol0_neptune():
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
    file_dir = os.path.join(os.path.dirname(__file__), "simdata_rasterscan_pol")
    data_dir = os.path.join(os.path.dirname(__file__),"test_data/")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir) 
    filenames = glob.glob(os.path.join(data_dir, "med*.fits"))
    data_set = data.Dataset(filenames)

    # creating flatfield for neptune for band 1
    planet='neptune'; band='1'
    #creates a planet image with spatial variation of polarization for POL0 
    pol_image=mocks.create_spatial_pol(data_set,filedir=None,nr=60,pfov_size=140,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,planet='neptune',band='1',dpamname='POL0')
    #creates raster scanned images for POL0 
    polraster_dataset = mocks.create_onsky_rasterscans(pol_image,filedir=file_dir,planet='neptune',band='1',im_size=800,d=40, n_dith=2,radius=55,snr=250,snr_constant=4.55,flat_map=None, raster_radius=40, raster_subexps=1)
     #creates flatfield for POL0 
    polflatfield_pol0=flat.create_onsky_pol_flatfield(polraster_dataset,planet='neptune',band='1',up_radius=55,im_size=1024,N=1,rad_mask=1.26, planet_rad=50, n_pix=174, observing_mode='NFOV', n_pad=0,fwhm_guess=20, sky_annulus_rin=2, sky_annulus_rout=4,plate_scale=0.0218,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,dpamname='POL0')

    assert np.nanmean(polflatfield_pol0.data) == pytest.approx(1, abs=1e-2)
    assert np.size(np.where(np.isnan(polflatfield_pol0.data))) == 0 # no bad pixels
    
    # check the flat can be pickled (for CTC operations)
    pickled = pickle.dumps(polflatfield_pol0)
    pickled_flat = pickle.loads(pickled)
    assert np.all(polflatfield_pol0.data == pickled_flat.data)

    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    polflatfield_pol0.save(filedir=calibdir)
    
    ###### perform flat division
    # load in the flatfield
    # check that the filename is what we expect
    flat_filename = polraster_dataset[-1].filename.replace("_l2a", "_flt_cal")
    flat_filepath = os.path.join(calibdir, flat_filename)
    polflatfield_pol0 = data.FlatField(flat_filepath)

    # check the flat can be pickled (for CTC operations)
    pickled = pickle.dumps(polflatfield_pol0)
    pickled_flat = pickle.loads(pickled)
    assert np.all(polflatfield_pol0.data == pickled_flat.data)
    
    polflatdivided_dataset = l2a_to_l2b.flat_division(simflat_dataset,polflatfield_pol0)
    
    
    # perform checks after the flat divison for one of the dataset
    assert(flat_filename in "".join(polflatdivided_dataset[0].ext_hdr["HISTORY"]))


    
    # check the propagated errors for one of the dataset
    assert polflatdivided_dataset[0].err_hdr["Layer_2"] == "FlatField_error"
    print("mean of all simulated data",np.mean(simflat_dataset.all_data))
    print("mean of all simulated data error",np.nanmean(simflat_dataset.all_err) )
    print("mean of all flat divided data:", np.nanmean(polflatdivided_dataset.all_data))
    print("mean of flatfield:", np.nanmean(polflatfield_pol0.data))
    print("mean of flatfield err:", np.nanmean(polflatfield_pol0.err))
    
    err_flatdiv=np.nanmean(polflatdivided_dataset.all_err)
    err_estimated=np.sqrt(((np.nanmean(polflatfield_pol0.data))**2)*(np.nanmean(simflat_dataset.all_err))**2+((np.nanmean(simflat_dataset.all_data))**2)*(np.nanmean(polflatfield_pol0.err))**2)
    print("mean of all flat divided data errors:",err_flatdiv)
    print("Error estimated:",err_estimated)
    assert(err_flatdiv == pytest.approx(err_estimated, abs = 1e-1))

    corgidrp.track_individual_errors = old_err_tracking

    return

def test_create_polflatfield_pol45_neptune():
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
    file_dir = os.path.join(os.path.dirname(__file__), "simdata_rasterscan_pol")
    data_dir = os.path.join(os.path.dirname(__file__),"test_data/")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir) 
    filenames = glob.glob(os.path.join(data_dir, "med*.fits"))
    data_set = data.Dataset(filenames)

   # creating flatfield for neptune for band 1
    planet='neptune'; band='1'
    #creates a planet image with spatial variation of polarization for POL45
    pol_image=mocks.create_spatial_pol(data_set,nr=60,filedir=None,pfov_size=140,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,planet='neptune',band='1',dpamname='POL45')
    #creates raster scanned images for POL45
    polraster_dataset = mocks.create_onsky_rasterscans(pol_image,filedir=file_dir,planet='neptune',band='1',im_size=800,d=50, n_dith=2,radius=55,snr=250,snr_constant=4.55,flat_map=None, raster_radius=40, raster_subexps=1)
     #creates flatfield for  POL45
    polflatfield_pol45=flat.create_onsky_pol_flatfield(polraster_dataset,planet='neptune',band='1',up_radius=55,im_size=1024,N=1,rad_mask=1.26, planet_rad=50, n_pix=174,observing_mode='NFOV', n_pad=0,fwhm_guess=20, sky_annulus_rin=2, sky_annulus_rout=4,plate_scale=0.0218,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,dpamname='POL45')
    assert np.nanmean(polflatfield_pol45.data) == pytest.approx(1, abs=1e-2)
    assert np.size(np.where(np.isnan(polflatfield_pol45.data))) == 0 # no bad pixels
    
    # check the flat can be pickled (for CTC operations)
    pickled = pickle.dumps(polflatfield_pol45)
    pickled_flat = pickle.loads(pickled)
    assert np.all(polflatfield_pol45.data == pickled_flat.data)

    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    polflatfield_pol45.save(filedir=calibdir)
    
    ###### perform flat division
    # load in the flatfield
    # check that the filename is what we expect
    flat_filename = polraster_dataset[-1].filename.replace("_l2a", "_flt_cal")
    flat_filepath = os.path.join(calibdir, flat_filename)
    polflatfield_pol45 = data.FlatField(flat_filepath)

    # check the flat can be pickled (for CTC operations)
    pickled = pickle.dumps(polflatfield_pol45)
    pickled_flat = pickle.loads(pickled)
    assert np.all(polflatfield_pol45.data == pickled_flat.data)
    
    polflatdivided_dataset = l2a_to_l2b.flat_division(simflat_dataset,polflatfield_pol45)
    
    
    # perform checks after the flat divison for one of the dataset
    assert(flat_filename in "".join(polflatdivided_dataset[0].ext_hdr["HISTORY"]))
    
    # check the propagated errors for one of the dataset
    assert polflatdivided_dataset[0].err_hdr["Layer_2"] == "FlatField_error"
    print("mean of all simulated data",np.mean(simflat_dataset.all_data))
    print("mean of all simulated data error",np.nanmean(simflat_dataset.all_err) )
    print("mean of all flat divided data:", np.nanmean(polflatdivided_dataset.all_data))
    print("mean of flatfield:", np.nanmean(polflatfield_pol45.data))
    print("mean of flatfield err:", np.nanmean(polflatfield_pol45.err))
    
    err_flatdiv=np.nanmean(polflatdivided_dataset.all_err)
    err_estimated=np.sqrt(((np.nanmean(polflatfield_pol45.data))**2)*(np.nanmean(simflat_dataset.all_err))**2+((np.nanmean(simflat_dataset.all_data))**2)*(np.nanmean(polflatfield_pol45.err))**2)
    print("mean of all flat divided data errors:",err_flatdiv)
    print("Error estimated:",err_estimated)
    assert(err_flatdiv == pytest.approx(err_estimated, abs = 1e-1))

    corgidrp.track_individual_errors = old_err_tracking

    return


def test_create_polflatfield_pol0_uranus():

    corgidrp.track_individual_errors = True
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
    planet='uranus'; band='1'
    pol_image=mocks.create_spatial_pol(data_set,filedir=None,nr=90,pfov_size=200,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,planet='uranus',band='1',dpamname='POL0')
    polraster_dataset = mocks.create_onsky_rasterscans(pol_image,filedir=file_dir,planet='uranus',band='1',im_size=900,d=65, n_dith=2,radius=90,snr=250,snr_constant=9.66,flat_map=None, raster_radius=40, raster_subexps=1)
    polflatfield_pol0=flat.create_onsky_pol_flatfield(polraster_dataset,planet='uranus',band='1',up_radius=55,im_size=1024,N=1,rad_mask=1.26, planet_rad=50, n_pix=174,observing_mode='NFOV', n_pad=0, fwhm_guess=25,sky_annulus_rin=2, sky_annulus_rout=4,plate_scale=0.0218,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,dpamname='POL0')

    assert np.nanmean(polflatfield_pol0.data) == pytest.approx(1, abs=1e-2)
    assert np.size(np.where(np.isnan(polflatfield_pol0.data))) == 0 # no bad pixels
    
    # check the flat can be pickled (for CTC operations)
    pickled = pickle.dumps(polflatfield_pol0)
    pickled_flat = pickle.loads(pickled)
    assert np.all(polflatfield_pol0.data == pickled_flat.data)

    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    polflatfield_pol0.save(filedir=calibdir)
    
    ###### perform flat division
    # load in the flatfield
    # check that the filename is what we expect
    flat_filename = polraster_dataset[-1].filename.replace("_l2a", "_flt_cal")
    flat_filepath = os.path.join(calibdir, flat_filename)
    polflatfield_pol0 = data.FlatField(flat_filepath)

    # check the flat can be pickled (for CTC operations)
    pickled = pickle.dumps(polflatfield_pol0)
    pickled_flat = pickle.loads(pickled)
    assert np.all(polflatfield_pol0.data == pickled_flat.data)
    
    polflatdivided_dataset = l2a_to_l2b.flat_division(simflat_dataset,polflatfield_pol0)
    
    
    # perform checks after the flat divison for one of the dataset
    assert(flat_filename in "".join(polflatdivided_dataset[0].ext_hdr["HISTORY"]))


    
    # check the propagated errors for one of the dataset
    assert polflatdivided_dataset[0].err_hdr["Layer_2"] == "FlatField_error"
    print("mean of all simulated data",np.mean(simflat_dataset.all_data))
    print("mean of all simulated data error",np.nanmean(simflat_dataset.all_err) )
    print("mean of all flat divided data:", np.nanmean(polflatdivided_dataset.all_data))
    print("mean of flatfield:", np.nanmean(polflatfield_pol0.data))
    print("mean of flatfield err:", np.nanmean(polflatfield_pol0.err))
    
    err_flatdiv=np.nanmean(polflatdivided_dataset.all_err)
    err_estimated=np.sqrt(((np.nanmean(polflatfield_pol0.data))**2)*(np.nanmean(simflat_dataset.all_err))**2+((np.nanmean(simflat_dataset.all_data))**2)*(np.nanmean(polflatfield_pol0.err))**2)
    print("mean of all flat divided data errors:",err_flatdiv)
    print("Error estimated:",err_estimated)
    assert(err_flatdiv == pytest.approx(err_estimated, abs = 1e-1))

    corgidrp.track_individual_errors = old_err_tracking


    return

def test_create_polflatfield_pol45_uranus():

    corgidrp.track_individual_errors = True
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
    planet='uranus'; band='1'
    pol_image=mocks.create_spatial_pol(data_set,filedir=None,nr=90,pfov_size=200,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,planet='uranus',band='1',dpamname='POL45')
    polraster_dataset = mocks.create_onsky_rasterscans(pol_image,filedir=file_dir,planet='uranus',band='1',im_size=900,d=65, n_dith=2,radius=90,snr=250,snr_constant=9.66,flat_map=None, raster_radius=40, raster_subexps=1)
    polflatfield_pol45=flat.create_onsky_pol_flatfield(polraster_dataset,planet='uranus',band='1',up_radius=55,im_size=1024,N=1,rad_mask=1.26, planet_rad=50, n_pix=174,observing_mode='NFOV', n_pad=0,fwhm_guess=25, sky_annulus_rin=2, sky_annulus_rout=4,plate_scale=0.021,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,dpamname='POL45')

    assert np.nanmean(polflatfield_pol45.data) == pytest.approx(1, abs=1e-2)
    assert np.size(np.where(np.isnan(polflatfield_pol45.data))) == 0 # no bad pixels
    
    # check the flat can be pickled (for CTC operations)
    pickled = pickle.dumps(polflatfield_pol45)
    pickled_flat = pickle.loads(pickled)
    assert np.all(polflatfield_pol45.data == pickled_flat.data)

    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    polflatfield_pol45.save(filedir=calibdir)
    
    ###### perform flat division
    # load in the flatfield
    # check that the filename is what we expect
    flat_filename = polraster_dataset[-1].filename.replace("_l2a", "_flt_cal")
    flat_filepath = os.path.join(calibdir, flat_filename)
    polflatfield_pol45 = data.FlatField(flat_filepath)

    # check the flat can be pickled (for CTC operations)
    pickled = pickle.dumps(polflatfield_pol45)
    pickled_flat = pickle.loads(pickled)
    assert np.all(polflatfield_pol45.data == pickled_flat.data)
    
    polflatdivided_dataset = l2a_to_l2b.flat_division(simflat_dataset,polflatfield_pol45)
    
    
    # perform checks after the flat divison for one of the dataset
    assert(flat_filename in "".join(polflatdivided_dataset[0].ext_hdr["HISTORY"]))


    
    # check the propagated errors for one of the dataset
    assert polflatdivided_dataset[0].err_hdr["Layer_2"] == "FlatField_error"
    print("mean of all simulated data",np.mean(simflat_dataset.all_data))
    print("mean of all simulated data error",np.nanmean(simflat_dataset.all_err) )
    print("mean of all flat divided data:", np.nanmean(polflatdivided_dataset.all_data))
    print("mean of flatfield:", np.nanmean(polflatfield_pol45.data))
    print("mean of flatfield err:", np.nanmean(polflatfield_pol45.err))
    
    err_flatdiv=np.nanmean(polflatdivided_dataset.all_err)
    err_estimated=np.sqrt(((np.nanmean(polflatfield_pol45.data))**2)*(np.nanmean(simflat_dataset.all_err))**2+((np.nanmean(simflat_dataset.all_data))**2)*(np.nanmean(polflatfield_pol45.err))**2)
    print("mean of all flat divided data errors:",err_flatdiv)
    print("Error estimated:",err_estimated)
    assert(err_flatdiv == pytest.approx(err_estimated, abs = 1e-1))

    corgidrp.track_individual_errors = old_err_tracking


    return

if __name__ == "__main__":
    
    test_create_polflatfield_pol0_neptune()
    test_create_polflatfield_pol45_neptune()
    test_create_polflatfield_pol0_uranus()
    test_create_polflatfield_pol45_uranus()


