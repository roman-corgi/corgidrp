import os
from pathlib import Path
import numpy as np
import scipy.ndimage
import pandas as pd
import astropy.io.fits as fits
from astropy.time import Time
import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord
import astropy.wcs as wcs
from astropy.table import Table
from astropy.convolution import convolve_fft
import photutils.centroids as centr
import corgidrp.data as data
from corgidrp.data import Image
import corgidrp.detector as detector
from corgidrp.detector import imaging_area_geom, unpack_geom
from corgidrp.pump_trap_calibration import (P1, P1_P1, P1_P2, P2, P2_P2, P3, P2_P3, P3_P3, tau_temp)

from emccd_detect.emccd_detect import EMCCDDetect
from emccd_detect.util.read_metadata_wrapper import MetadataWrapper

detector_areas_test= {
'SCI' : { #used for unit tests; enables smaller memory usage with frames of scaled-down comparable geometry
        'frame_rows': 120, 
        'frame_cols': 220,
        'image': {
            'rows': 104,
            'cols': 105,
            'r0c0': [2, 108]
            },
        'prescan_reliable': {
            'rows': 120,
            'cols': 108,
            'r0c0': [0, 0]
        },        
        'prescan': {
            'rows': 120,
            'cols': 108,
            'r0c0': [0, 0],
            'col_start': 0, #10
            'col_end': 108, #100
        }, 
        'serial_overscan': {
            'rows': 120,
            'cols': 5,
            'r0c0': [0, 215]
        },
        'parallel_overscan': {
            'rows': 14,
            'cols': 107,
            'r0c0': [106, 108]
        }
        },
'ENG' : { #used for unit tests; enables smaller memory usage with frames of scaled-down comparable geometry
        'frame_rows' : 220,
        'frame_cols' : 220,
        'image' : {
            'rows': 102,
            'cols': 102,
            'r0c0': [13, 108]
            },
        'prescan' : {
            'rows': 220,
            'cols': 108,
            'r0c0': [0, 0],
            'col_start': 0, #10
            'col_end': 108, #100
            },
        'prescan_reliable' : {
            'rows': 220,
            'cols': 20,
            'r0c0': [0, 80]
            },
        'parallel_overscan' : {
            'rows': 116,
            'cols': 105,
            'r0c0': [104, 108]
            },
        'serial_overscan' : {
            'rows': 220,
            'cols': 5,
            'r0c0': [0, 215]
            },
        }
}

def create_noise_maps(FPN_map, FPN_map_err, FPN_map_dq, CIC_map, CIC_map_err, CIC_map_dq, DC_map, DC_map_err, DC_map_dq):
    '''
    Create simulated noise maps for test_masterdark_from_noisemaps.py.

    Arguments:
        FPN_map: 2D np.array for fixed-pattern noise (FPN) data array
        FPN_map_err: 2D np.array for FPN err array
        FPN_map_dq: 2D np.array for FPN DQ array
        CIC_map: 2D np.array for clock-induced charge (CIC) data array
        CIC_map_err: 2D np.array for CIC err array
        CIC_map_dq: 2D np.array for CIC DQ array
        DC_map: 2D np.array for dark current data array
        DC_map_err: 2D np.array for dark current err array
        DC_map_dq: 2D np.array for dark current DQ array

    Returns:
        corgidrp.data.DetectorNoiseMaps instance
    '''

    prihdr, exthdr = create_default_headers()
    # taken from end of calibrate_darks_lsq()
    exthdr['EXPTIME'] = None
    if 'EMGAIN_M' in exthdr.keys():
        exthdr['EMGAIN_M'] = None
    exthdr['CMDGAIN'] = None
    exthdr['KGAIN'] = None
    exthdr['BUNIT'] = 'detected electrons'
    exthdr['HIERARCH DATA_LEVEL'] = None
    # simulate raw data filenames
    exthdr['DRPNFILE'] = 2
    exthdr['FILE0'] = '0.fits'
    exthdr['FILE1'] = '1.fits'
    exthdr['B_O'] = 0.01
    exthdr['B_O_UNIT'] = 'DN'
    exthdr['B_O_ERR'] = 0.001

    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electrons'
    exthdr['DATATYPE'] = 'DetectorNoiseMaps'
    input_data = np.stack([FPN_map, CIC_map, DC_map])
    err = np.stack([[FPN_map_err, CIC_map_err, DC_map_err]])
    dq = np.stack([FPN_map_dq, CIC_map_dq, DC_map_dq])
    noise_maps = data.DetectorNoiseMaps(input_data, pri_hdr=prihdr, ext_hdr=exthdr, err=err,
                              dq=dq, err_hdr=err_hdr)
    return noise_maps

def create_synthesized_master_dark_calib(detector_areas):
    '''
    Create simulated data specifically for test_calibrate_darks_lsq.py.

    Args:
        detector_areas: dict
        a dictionary of detector geometry properties.  Keys should be as found
        in detector_areas in detector.py.


    Returns:
        dataset: corgidrp.data.Dataset instances
    The simulated dataset
    '''

    dark_current = 8.33e-4 #e-/pix/s
    cic=0.02  # e-/pix/frame
    read_noise=100 # e-/pix/frame
    bias=2000 # e-
    eperdn = 7 # e-/DN conversion; used in this example for all stacks
    EMgain_picks = (np.linspace(2, 5000, 7))
    exptime_picks = (np.linspace(2, 100, 7))
    grid = np.meshgrid(EMgain_picks, exptime_picks)
    EMgain_arr = grid[0].ravel()
    exptime_arr = grid[1].ravel()
    #added in after emccd_detect makes the frames (see below)
    # The mean FPN that will be found is eperdn*(FPN//eperdn)
    # due to how I simulate it and then convert the frame to uint16
    FPN = 21 # e
    # the bigger N is, the better the adjusted R^2 per pixel becomes
    N = 30 #Use N=600 for results with better fits (higher values for adjusted
    # R^2 per pixel)
    # image area, including "shielded" rows and cols:
    imrows, imcols, imr0c0 = imaging_area_geom('SCI', detector_areas)
    prerows, precols, prer0c0 = unpack_geom('SCI', 'prescan', detector_areas)

    frame_list = []
    for i in range(len(EMgain_arr)):
        for l in range(N): #number of frames to produce
            # Simulate full dark frame (image area + the rest)
            frame_rows = detector_areas['SCI']['frame_rows']
            frame_cols = detector_areas['SCI']['frame_cols']
            frame_dn_dark = np.zeros((frame_rows, frame_cols))
            im = np.random.poisson(cic*EMgain_arr[i]+
                                exptime_arr[i]*EMgain_arr[i]*dark_current,
                                size=(frame_rows, frame_cols))
            frame_dn_dark = im
            # prescan has no dark current
            pre = np.random.poisson(cic*EMgain_arr[i],
                                    size=(prerows, precols))
            frame_dn_dark[prer0c0[0]:prer0c0[0]+prerows,
                            prer0c0[1]:prer0c0[1]+precols] = pre
            rn = np.random.normal(0, read_noise,
                                    size=(frame_rows, frame_cols))
            with_rn = frame_dn_dark + rn + bias

            frame_dn_dark = with_rn/eperdn
            # simulate a constant FPN in image area (not in prescan
            # so that it isn't removed when bias is removed)
            frame_dn_dark[imr0c0[0]:imr0c0[0]+imrows,imr0c0[1]:
            imr0c0[1]+imcols] += FPN/eperdn # in DN
            # simulate telemetry rows, with the last 5 column entries with high counts
            frame_dn_dark[-1,-5:] = 100000 #DN
            # take raw frames and process them to what is needed for input
            # No simulated pre-processing bad pixels or cosmic rays, so just subtract bias
            # and multiply by k gain
            frame_dn_dark -= bias/eperdn
            frame_dn_dark *= eperdn

            # Now make this into a bunch of corgidrp.Dataset stacks
            prihdr, exthdr = create_default_headers()
            frame = data.Image(frame_dn_dark, pri_hdr=prihdr,
                            ext_hdr=exthdr)
            frame.ext_hdr['CMDGAIN'] = EMgain_arr[i]
            frame.ext_hdr['EXPTIME'] = exptime_arr[i]
            frame.ext_hdr['KGAIN'] = eperdn
            frame_list.append(frame)
    dataset = data.Dataset(frame_list)

    return dataset

def create_dark_calib_files(filedir=None, numfiles=10):
    """
    Create simulated data to create a master dark.
    Assume these have already undergone L1 processing and are L2a level products

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 10.

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset
    """
    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    filepattern = "simcal_dark_{0:04d}.fits"
    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_headers()
        exthdr['KGAIN'] = 7
        np.random.seed(456+i); sim_data = np.random.poisson(lam=150., size=(1200, 2200)).astype(np.float64)
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))
        frames.append(frame)
    dataset = data.Dataset(frames)
    return dataset

def create_simflat_dataset(filedir=None, numfiles=10):
    """
    Create simulated data to check the flat division

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 10.

    Returns:
        corgidrp.data.Dataset:
        The simulated dataset
    """
    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    filepattern = "sim_flat_{0:04d}.fits"
    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_headers()
        # generate images in normal distribution with mean 1 and std 0.01
        np.random.seed(456+i); sim_data = np.random.poisson(lam=150., size=(1024, 1024)).astype(np.float64)
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))
        frames.append(frame)
    dataset = data.Dataset(frames)
    return dataset

def create_raster(mask,data,dither_sizex=None,dither_sizey=None,row_cent = None,col_cent = None,n_dith=None,mask_size=420,snr=250,planet=None, band=None, radius=None, snr_constant=None):
    """Performs raster scan of Neptune or Uranus images
    
    Args:
        mask (int): (Required)  Mask used for the image. (Size of the HST images, 420 X 420 pixels with random values mean=1, std=0.03)
        data (float):(Required) Data in array npixels*npixels format to be raster scanned
        dither_sizex (int):(Required) Size of the dither in X axis in pixels (number of pixels across the planet (neptune=50 and uranus=65))
        dither_sizey (int):(Required) Size of the dither in X axis in pixels (number of pixels across the planet (neptune=50 and uranus=65))
        row_cent (int): (Required)  X coordinate of the centroid
        col_cent (int): (Required)  Y coordinate of the centroid
        n_dith (int): number of dithers required
        mask_size (int): Size of the mask in pixels  (Size of the HST images, 420 X 420 pixels with random values mean=1, std=0.03)
        snr (int): Required SNR in the planet images (=250 in the HST images)
        planet (str): neptune or uranus
        band (str): 1 or 4
        radius (int): radius of the planet in pixels (radius=54 for neptune, radius=90)
        snr_constant (int): constant for snr reference  (4.95 for band1 and 9.66 for band4)
        
	Returns:
    	dither_stack_norm (np.array): stacked dithers of the planet images
    	cent (np.array): centroid of images 
    	
        
    """  
 
    cents = []
    
    data_display = data.copy()
    col_max = int(col_cent) + int(mask_size/2)
    col_min = int(col_cent) - int(mask_size/2)
    row_max = int(row_cent) + int(mask_size/2)
    row_min = int(row_cent) - int(mask_size/2)
    dithers = []
    
    if dither_sizey == None:
        dither_sizey = dither_sizex

    
    for i in np.arange(-n_dith,n_dith):
        for j in np.arange(-n_dith,n_dith):
            mask_data = data.copy()
            new_image_row_coords = np.arange(row_min + (dither_sizey * j), row_max + (dither_sizey * j))
            new_image_col_coords = np.arange(col_min + (dither_sizex * i), col_max + (dither_sizex * i))
            new_image_col_coords, new_image_row_coords = np.meshgrid(new_image_col_coords, new_image_row_coords)
            image_data = scipy.ndimage.map_coordinates(mask_data, [new_image_row_coords, new_image_col_coords], mode="constant", cval=0)
            # image_data = mask_data[row_min + (dither_sizey * j):row_max + (dither_sizey * j), col_min + (dither_sizex * i):col_max + (dither_sizex * i)]
            cents.append(((mask_size/2) + (row_cent - int(row_cent)) - (dither_sizey//2) - (dither_sizey * j), (mask_size/2) + (col_cent - int(col_cent)) - (dither_sizex//2) - (dither_sizex * i)))
            # try:
            new_image_data = image_data * mask
            
            snr_ref = snr/np.sqrt(snr_constant)

            u_centroid = centr.centroid_1dg(new_image_data)
            uxc = int(u_centroid[0])
            uyc = int(u_centroid[1])

            modified_data = new_image_data

            nx = np.arange(0,modified_data.shape[1])
            ny = np.arange(0,modified_data.shape[0])
            nxx,nyy = np.meshgrid(nx,ny)
            nrr = np.sqrt((nxx-uxc)**2 + (nyy-uyc)**2)

            planmed = np.median(modified_data[nrr<radius])
            modified_data[nrr<=radius] = np.random.normal(modified_data[nrr<=radius], (planmed/snr_ref) * np.abs(modified_data[nrr<=radius]/planmed))
            
            new_image_data_snr = modified_data
            # except ValueError:
            #     print(image_data.shape)
            #     print(mask.shape)
            dithers.append(new_image_data_snr)

    dither_stack_norm = []
    for dither in dithers:
        dither_stack_norm.append(dither) 
    dither_stack = None 
    
    median_dithers = None 
    final = None 
    full_mask = mask 
    
    return dither_stack_norm,cents
    
def create_onsky_rasterscans(dataset,filedir=None,planet=None,band=None, im_size=420, d=None, n_dith=3, radius=None, snr=250, snr_constant=None, flat_map=None, raster_radius=40, raster_subexps=1):
    """
    Create simulated data to check the flat division
    
    Args:
       dataset (corgidrp.data.Dataset): dataset of HST images of neptune and uranus
       filedir (str): Full path to directory to save the raster scanned images.
       planet (str): neptune or uranus
       band (str): 1 or 4
       im_size (int): x-dimension of the planet image (in pixels= 420 for the HST images)
       d (int): number of pixels across the planet (neptune=50 and uranus=65)
       n_dith (int): Number of dithers required (Default is 3)
       radius (int): radius of the planet in pixels (radius=54 for neptune, radius=90 in HST images)
       snr (int): SNR required for the planet image (default is 250 for the HST images)
       snr_constant (int): constant for snr reference  (4.95 for band1 and 9.66 for band4)
       flat_map (np.array): a user specified flat map. Must have shape (im_size, im_size). Default: None; assumes each pixel drawn from a normal distribution with 3% rms scatter
       raster_radius (float): radius of circular raster done to smear out image during observation, in pixels
       raster_subexps (int): number of subexposures that consist of a singular raster. Currently just duplicates images and does not simulate partial rasters
        
    Returns: 
    	corgidrp.data.Dataset:
        The simulated dataset of raster scanned images of planets uranus or neptune
    """
    n = im_size

    if flat_map is None:
        qe_prnu_fsm_raster = np.random.normal(1,.03,(n,n))
    else:
        qe_prnu_fsm_raster = flat_map

    pred_cents=[]
    planet_rot_images=[]
    
    for i in range(len(dataset)):
        target=dataset[i].pri_hdr['TARGET']
        filter=dataset[i].pri_hdr['FILTER']
        if planet==target and band==filter: 
            planet_image=dataset[i].data
            centroid=centr.centroid_com(planet_image)
            xc=centroid[0]
            yc=centroid[1]
            planet_image = convolve_fft(planet_image, detector.raster_kernel(raster_radius, planet_image))
            if planet == 'neptune':
                planetrad=radius; snrcon=snr_constant
                planet_repoint_current = create_raster(qe_prnu_fsm_raster,planet_image,row_cent=yc+(d//2),col_cent=xc+(d//2), dither_sizex=d, dither_sizey=d,n_dith=n_dith,mask_size=n,snr=snr,planet=target,band=filter,radius=planetrad, snr_constant=snrcon)
            elif planet == 'uranus':
                planetrad=radius; snrcon=snr_constant     
                planet_repoint_current = create_raster(qe_prnu_fsm_raster,planet_image,row_cent=yc,col_cent=xc, dither_sizex=d, dither_sizey=d,n_dith=n_dith,mask_size=n,snr=snr,planet=target,band=filter,radius=planetrad, snr_constant=snrcon)
    
    numfiles = len(planet_repoint_current[0])
    for j in np.arange(numfiles):
        for k in range(raster_subexps):
            # don't know how to simualate partial rasters, so we just append the same image multiple times
            # it's ok to append the same noise as well because we simulated the full raster to reach the SNR after combining subexps
            planet_rot_images.append(planet_repoint_current[0][j])
            pred_cents.append(planet_repoint_current[1][j])

    filepattern= planet+'_'+band+"_"+"raster_scan_{0:01d}.fits"
    frames=[]
    for i in range(numfiles*raster_subexps):
        prihdr, exthdr = create_default_headers()
        sim_data=planet_rot_images[i]
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        pl=planet
        band=band
        frame.pri_hdr.append(('TARGET', pl), end=True)
        frame.pri_hdr.append(('FILTER', band), end=True)
        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))
        frames.append(frame)
    raster_dataset = data.Dataset(frames)
    return raster_dataset

def create_flatfield_dummy(filedir=None, numfiles=2):

    """
    Turn this flat field dataset of image frames that were taken for performing the flat calibration and
    to make one master flat image

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 1 to create the dummy flat can be changed to any number

    Returns:
        corgidrp.data.Dataset:
        a set of flat field images
    """
    ## Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    filepattern= "flat_field_{0:01d}.fits"
    frames=[]
    for i in range(numfiles):
        prihdr, exthdr = create_default_headers()
        np.random.seed(456+i); sim_data = np.random.normal(loc=1.0, scale=0.01, size=(1024, 1024))
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))
        frames.append(frame)
    flatfield = data.Dataset(frames)
    return flatfield

def create_nonlinear_dataset(nonlin_filepath, filedir=None, numfiles=2,em_gain=2000):
    """
    Create simulated data to non-linear data to test non-linearity correction.

    Args:
        nonlin_filepath (str): path to FITS file containing nonlinear calibration data (e.g., tests/test_data/nonlin_sample.fits)
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 2 (not creating the cal here, just testing the function)
        em_gain (int): The EM gain to use for the simulated data.  Defaults to 2000.

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset
    """

    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    filepattern = "simcal_nonlin_{0:04d}.fits"
    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_headers()
        #Add the CMDGAIN to the headers
        exthdr['CMDGAIN'] = em_gain
        # Create a default
        size = 1024
        sim_data = np.zeros([size,size])
        data_range = np.linspace(800,65536,size)
        # Generate data for each row, where the mean increase from 10 to 65536
        for x in range(size):
            np.random.seed(120+x); sim_data[:, x] = np.random.poisson(data_range[x], size).astype(np.float64)

        non_linearity_correction = data.NonLinearityCalibration(nonlin_filepath)

        #Apply the non-linearity to the data. When we correct we multiple, here when we simulate we divide
        #This is a bit tricky because when we correct the get_relgains function takes the current state of 
        # the data as input, which when actually used will be the non-linear data. Here we try to get close 
        # to that by calculating the relative gains after applying the relative gains one time. This won't be 
        # perfect, but it'll be closer than just dividing by the straight simulated data. 

        sim_data_tmp = sim_data/detector.get_relgains(sim_data,em_gain,non_linearity_correction)

        sim_data /= detector.get_relgains(sim_data_tmp,em_gain,non_linearity_correction)

        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))
        frames.append(frame)
    dataset = data.Dataset(frames)
    return dataset

def create_cr_dataset(nonlin_filepath, filedir=None, datetime=None, numfiles=2, em_gain=500, numCRs=5, plateau_length=10):
    """
    Create simulated non-linear data with cosmic rays to test CR detection.

    Args:
        nonlin_filepath (str): path to FITS file containing nonlinear calibration data (e.g., tests/test_data/nonlin_sample.fits)
        filedir (str): (Optional) Full path to directory to save to.
        datetime (astropy.time.Time): (Optional) Date and time of the observations to simulate.
        numfiles (int): Number of files in dataset.  Defaults to 2 (not creating the cal here, just testing the function)
        em_gain (int): The EM gain to use for the simulated data.  Defaults to 2000.
        numCRs (int): The number of CR hits to inject. Defaults to 5.
        plateau_length (int): The minimum length of a CR plateau that will be flagged by the filter.

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset.
    """

    if datetime is None:
        datetime = Time('2024-01-01T11:00:00.000Z')

    detector_params = data.DetectorParams({}, date_valid=Time("2023-11-01 00:00:00"))

    kgain = detector_params.params['kgain']
    fwc_em_dn = detector_params.params['fwc_em'] / kgain
    fwc_pp_dn = detector_params.params['fwc_pp'] / kgain
    fwc = np.min([fwc_em_dn,em_gain*fwc_pp_dn])
    dataset = create_nonlinear_dataset(nonlin_filepath, filedir=None, numfiles=numfiles,em_gain=em_gain)

    im_width = dataset.all_data.shape[-1]

    # Overwrite dataset with a poisson distribution
    np.random.seed(123)
    dataset.all_data[:,:,:] = np.random.poisson(lam=150,size=dataset.all_data.shape).astype(np.float64)

    # Loop over images in dataset
    for i in range(len(dataset.all_data)):

        # Save the date
        dataset[i].ext_hdr['DATETIME'] = str(datetime)

        # Pick random locations to add a cosmic ray
        for x in range(numCRs):
            np.random.seed(123+x)
            loc = np.round(np.random.uniform(0,im_width-1, size=2)).astype(int)

            # Add the CR plateau
            tail_start = np.min([loc[1]+plateau_length,im_width])
            dataset.all_data[i,loc[0],loc[1]:tail_start] += fwc

            if tail_start < im_width-1:
                tail_len = im_width-tail_start
                cr_tail = [fwc/(j+1) for j in range(tail_len)]
                dataset.all_data[i,loc[0],tail_start:] += cr_tail

        # Save frame if desired
        if filedir is not None:
            filepattern = "simcal_cosmics_{0:04d}.fits"
            dataset[i].save(filedir=filedir, filename=filepattern.format(i))

    return dataset

def create_prescan_files(filedir=None, numfiles=2, arrtype="SCI"):
    """
    Create simulated raw data.

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 2.
        arrtype (str): Observation type. Defaults to "SCI".

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset
    """
    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    if arrtype == "SCI":
        size = (1200, 2200)
    elif arrtype == "ENG":
        size = (2200, 2200)
    elif arrtype == "CAL":
        size = (2200,2200)
    else:
        raise ValueError(f'Arrtype {arrtype} not in ["SCI","ENG","CAL"]')


    filepattern = f"sim_prescan_{arrtype}"
    filepattern = filepattern+"{0:04d}.fits"

    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_headers(arrtype=arrtype)
        sim_data = np.random.poisson(lam=150., size=size).astype(np.float64)
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)

        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))

        frames.append(frame)

    dataset = data.Dataset(frames)

    return dataset

def create_default_headers(arrtype="SCI", vistype="TDEMO"):
    """
    Creates an empty primary header and an Image extension header with some possible keywords

    Args:
        arrtype (str): Array type (SCI or ENG). Defaults to "SCI". 
        vistype (str): Visit type. Defaults to "TDEMO"

    Returns:
        tuple:
            prihdr (fits.Header): Primary FITS Header
            exthdr (fits.Header): Extension FITS Header

    """
    prihdr = fits.Header()
    exthdr = fits.Header()

    if arrtype != "SCI":
        NAXIS1 = 2200
        NAXIS2 = 1200
    else:
        NAXIS1 = 2200
        NAXIS2 = 2200

    # fill in prihdr
    prihdr['AUXFILE'] = 'mock_auxfile.fits'
    prihdr['OBSID'] = 0
    prihdr['BUILD'] = 0
    # prihdr['OBSTYPE'] = arrtype
    prihdr['VISTYPE'] = vistype
    prihdr['MOCK'] = True
    prihdr['TELESCOP'] = 'ROMAN'
    prihdr['INSTRUME'] = 'CGI'
    prihdr['OBSNAME'] = 'MOCK'
    prihdr['TARGET'] = 'MOCK'
    prihdr['OBSNUM'] = '000'
    prihdr['CAMPAIGN'] = '000'
    prihdr['PROGNUM'] = '00000'
    prihdr['SEGMENT'] = '000'
    prihdr['VISNUM'] = '000'
    prihdr['EXECNUM'] = '00'
    prihdr['VISITID'] = prihdr['PROGNUM'] + prihdr['EXECNUM'] + prihdr['CAMPAIGN'] + prihdr['SEGMENT'] + prihdr['OBSNUM'] + prihdr['VISNUM']
    prihdr['PSFREF'] = False
    prihdr['SIMPLE'] = True
    prihdr['NAXIS'] = 0
        

    # fill in exthdr
    exthdr['NAXIS'] = 2
    exthdr['NAXIS1'] = NAXIS1
    exthdr['NAXIS2'] = NAXIS2
    exthdr['PCOUNT'] = 0
    exthdr['GCOUNT'] = 1
    exthdr['BSCALE'] = 1
    exthdr['BZERO'] = 32768
    exthdr['ARRTYPE'] = arrtype 
    exthdr['SCTSRT'] = '2024-01-01T12:00:00.000Z'
    exthdr['SCTEND'] = '2024-01-01T20:00:00.000Z'
    exthdr['STATUS'] = 0
    exthdr['HVCBIAS'] = 1
    exthdr['OPMODE'] = ""
    exthdr['EXPTIME'] = 60.0
    exthdr['CMDGAIN'] = 1.0
    exthdr['CYCLES'] = 100000000000
    exthdr['LASTEXP'] = 1000000
    exthdr['BLNKTIME'] = 10
    exthdr['EXPCYC'] = 100
    exthdr['OVEREXP'] = 0
    exthdr['NOVEREXP'] = 0
    exthdr['EXCAMT'] = 40.0
    exthdr['FCMLOOP'] = ""
    exthdr['FSMINNER'] = ""
    exthdr['FSMLOS'] = ""
    exthdr['FSM_X'] = 50.0
    exthdr['FSM_Y'] = 50.0
    exthdr['DMZLOOP'] = ""
    exthdr['SPAM_H'] = 1.0
    exthdr['SPAM_V'] = 1.0
    exthdr['FPAM_H'] = 1.0
    exthdr['FPAM_V'] = 1.0
    exthdr['LSAM_H'] = 1.0
    exthdr['LSAM_V'] = 1.0
    exthdr['FSAM_H'] = 1.0
    exthdr['FSAM_V'] = 1.0
    exthdr['CFAM_H'] = 1.0
    exthdr['CFAM_V'] = 1.0
    exthdr['DPAM_H'] = 1.0
    exthdr['DPAM_V'] = 1.0
    exthdr['DATETIME'] = '2024-01-01T11:00:00.000Z'
    exthdr['HIERARCH DATA_LEVEL'] = "L1"
    exthdr['MISSING'] = False

    return prihdr, exthdr

def create_badpixelmap_files(filedir=None, col_bp=None, row_bp=None):
    """
    Create simulated bad pixel map data. Code value is 4.

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        col_bp (array): (Optional) Array of column indices where bad detector
            pixels are found.
        row_bp (array): (Optional) Array of row indices where bad detector
            pixels are found.

    Returns:
        corgidrp.data.BadPixelMap:
            The simulated dataset
    """
    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    prihdr, exthdr = create_default_headers()
    sim_data = np.zeros([1024,1024], dtype = np.uint16)
    if col_bp is not None and row_bp is not None:
        for i_col in col_bp:
            for i_row in row_bp:
                sim_data[i_col, i_row] += 4
    frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)

    if filedir is not None:
        frame.save(filedir=filedir, filename= "sim_bad_pixel.fits")

    badpixelmap = data.Dataset([frame])

    return badpixelmap

def nonlin_coefs(filename,EMgain,order):
    """
    Reads TVAC nonlinearity table from location specified by ‘filename’.
    The column in the table closest to the ‘EMgain’ value is selected and fits
    a polynomial of order ‘order’. The coefficients of the fit are adjusted so
    that the polynomial function equals unity at 3000 DN. Outputs array polynomial
    coefficients, array of DN values from the TVAC table, and an array of the
    polynomial function values for all the DN values.

    Args:
      filename (string): file name
      EMgain (int): em gain value
      order (int): polynomial order

    Returns:
      np.array: fit coefficients
      np.array: DN values
      np.array: fit values
    """
    # filename is the name of the csv text file containing the TVAC nonlin table
    # EM gain selects the closest column in the table
    # Load the specified file
    bigArray = pd.read_csv(filename, header=None).values
    EMgains = bigArray[0, 1:]
    DNs = bigArray[1:, 0]

    # Find the closest EM gain available to what was requested
    iG = (np.abs(EMgains - EMgain)).argmin()

    # Fit the nonlinearity numbers to a polynomial
    vals = bigArray[1:, iG + 1]
    coeffs = np.polyfit(DNs, vals, order)

    # shift so that function passes through unity at 3000 DN for these tests
    fitVals0 = np.polyval(coeffs, DNs)
    ind = np.where(DNs == 3000)
    unity_val = fitVals0[ind][0]
    coeffs[3] = coeffs[3] - (unity_val-1.0)
    fitVals = np.polyval(coeffs,DNs)

    return coeffs, DNs, fitVals

def nonlin_factor(coeffs,DN):
    """ 
    Takes array of nonlinearity coefficients (from nonlin_coefs function)
    and an array of DN values and returns the nonlinearity values array. If the
    DN value is less 800 DN, then the nonlinearity value at 800 DN is returned.
    If the DN value is greater than 10000 DN, then the nonlinearity value at
    10000 DN is returned.
    
    Args:
       coeffs (np.array): nonlinearity coefficients
       DN (int): DN value
       
    Returns:
       float: nonlinearity value
    """
    # input coeffs from nonlin_ceofs and a DN value and return the
    # nonlinearity factor
    min_value = 800.0
    max_value = 10000.0
    f_nonlin = np.polyval(coeffs, DN)
    # Control values outside the min/max range
    f_nonlin = np.where(DN < min_value, np.polyval(coeffs, min_value), f_nonlin)
    f_nonlin = np.where(DN > max_value, np.polyval(coeffs, max_value), f_nonlin)

    return f_nonlin

def make_fluxmap_image(
        f_map,
        bias,
        kgain,
        rn,
        emgain, 
        time,
        coeffs,
        nonlin_flag=False,
        divide_em=False,
        ):
    """ 
    This function makes a SCI-sized frame with simulated noise and a fluxmap. It
    also performs bias-subtraction and division by EM gain if required. It is used
    in the unit tests test_nonlin.py and test_kgain_cal.py

    Args:
        f_map (np.array): fluxmap in e/s/px. Its size is 1024x1024 pixels.
        bias (float): bias value in electrons.
        kgain (float): value of K-Gain in electrons per DN.
        rn (float): read noise in electrons.
        emgain (float): calue of EM gain. 
        time (float):  exposure time in sec.
        coeffs (np.array): array of cubic polynomial coefficients from nonlin_coefs.
        nonlin_flag (bool): (Optional) if nonlin_flag is True, then nonlinearity is applied.
        divide_em (bool): if divide_em is True, then the emgain is divided
        
    Returns:
        corgidrp.data.Image
    """
    # Generate random values of rn in electrons from a Gaussian distribution
    random_array = np.random.normal(0, rn, (1200, 2200)) # e-
    # Generate random values from fluxmap from a Poisson distribution
    Poiss_noise_arr = emgain*np.random.poisson(time*f_map) # e-
    signal_arr = np.zeros((1200,2200))
    start_row = 10
    start_col = 1100
    signal_arr[start_row:start_row + Poiss_noise_arr.shape[0],
                start_col:start_col + Poiss_noise_arr.shape[1]] = Poiss_noise_arr
    temp = random_array + signal_arr # e-
    if nonlin_flag:
        temp2 = nonlin_factor(coeffs, signal_arr/kgain)
        frame = np.round((bias + random_array + signal_arr/temp2)/kgain) # DN
    else:
        frame = np.round((bias+temp)/kgain) # DN

    # Subtract bias and divide by EM gain if required. TODO: substitute by
    # prescan_biassub step function in l1_to_l2a.py and the em_gain_division
    # step function in l2a_to_l2b.py    
    offset_colroi1 = 799
    offset_colroi2 = 1000
    offset_colroi = slice(offset_colroi1,offset_colroi2)
    row_meds = np.median(frame[:,offset_colroi], axis=1)
    row_meds = row_meds[:, np.newaxis]
    frame -= row_meds
    if divide_em:
        frame = frame/emgain

    prhd, exthd = create_default_headers()
    # Record actual commanded EM
    exthd['CMDGAIN'] = emgain
    # Record actual exposure time
    exthd['EXPTIME'] = time
    # Mock error maps
    err = np.ones([1200,2200]) * 0.5
    dq = np.zeros([1200,2200], dtype = np.uint16)
    image = Image(frame, pri_hdr = prhd, ext_hdr = exthd, err = err,
        dq = dq)
    return image

def create_astrom_data(field_path, filedir=None, subfield_radius=0.02, platescale=21.8, rotation=45, add_gauss_noise=True):
    """
    Create simulated data for astrometric calibration.

    Args:
        field_path (str): Full path to directory with test field data (ra, dec, vmag, etc.)
        filedir (str): (Optional) Full path to directory to save to.
        subfield_radius (float): The radius [deg] around the target coordinate for creating a subfield to produce the image from
        platescale (float): The plate scale of the created image data (default: 21.8 [mas/pixel])
        rotation (float): The north angle of the created image data (default: 45 [deg])
        add_gauss_noise (boolean): Argument to determine if gaussian noise should be added to the data (default: True)

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset

    """
    if type(field_path) != str:
        raise TypeError('field_path must be a str')

    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)
    
    # hard coded image properties
    size = (1024, 1024)
    sim_data = np.zeros(size)
    ny, nx = size
    center = [nx //2, ny //2]
    target = (80.553428801, -69.514096821)
    fwhm = 3
    subfield_radius = 0.02 #[deg]
    
    # load in the field data and restrict to 0.02 [deg] radius around target
    cal_field = ascii.read(field_path)
    subfield = cal_field[((cal_field['RA'] >= target[0] - subfield_radius) & (cal_field['RA'] <= target[0] + subfield_radius) & (cal_field['DEC'] >= target[1] - subfield_radius) & (cal_field['DEC'] <= target[1] + subfield_radius))]

    cal_SkyCoords = SkyCoord(ra= subfield['RA'], dec= subfield['DEC'], unit='deg', frame='icrs')  # save these subfield skycoords somewhere

    # create the simulated image header
    vert_ang = np.radians(rotation)
    pc = np.array([[-np.cos(vert_ang), np.sin(vert_ang)], [np.sin(vert_ang), np.cos(vert_ang)]])
    cdmatrix = pc * (platescale * 0.001) / 3600.

    new_hdr = {}
    new_hdr['CD1_1'] = cdmatrix[0,0]
    new_hdr['CD1_2'] = cdmatrix[0,1]
    new_hdr['CD2_1'] = cdmatrix[1,0]
    new_hdr['CD2_2'] = cdmatrix[1,1]

    new_hdr['CRPIX1'] = center[0]
    new_hdr['CRPIX2'] = center[1]

    new_hdr['CTYPE1'] = 'RA---TAN'
    new_hdr['CTYPE2'] = 'DEC--TAN'

    new_hdr['CDELT1'] = (platescale * 0.001) / 3600
    new_hdr['CDELT2'] = (platescale * 0.001) / 3600

    new_hdr['CRVAL1'] = target[0]
    new_hdr['CRVAL2'] = target[1]

    w = wcs.WCS(new_hdr)

    # create the image data
    xpix, ypix = wcs.utils.skycoord_to_pixel(cal_SkyCoords, wcs=w)
    pix_inds = np.where((xpix >= 0) & (xpix <= 1024) & (ypix >= 0) & (ypix <= 1024))[0]

    xpix = xpix[pix_inds]
    ypix = ypix[pix_inds]

    amplitudes = np.power(10, ((subfield['VMAG'][pix_inds] - 22.5) / (-2.5))) * 10  

    # inject gaussian psf stars
    for xpos, ypos, amplitude in zip(xpix, ypix, amplitudes):  
        stampsize = int(np.ceil(3 * fwhm))
        sigma = fwhm/ (2.*np.sqrt(2*np.log(2)))
        
        # coordinate system
        y, x = np.indices([stampsize, stampsize])
        y -= stampsize // 2
        x -= stampsize // 2
        
        # find nearest pixel
        x_int = int(round(xpos))
        y_int = int(round(ypos))
        x += x_int
        y += y_int
        
        xmin = x[0][0]
        xmax = x[-1][-1]
        ymin = y[0][0]
        ymax = y[-1][-1]
        
        psf = amplitude * np.exp(-((x - xpos)**2. + (y - ypos)**2.) / (2. * sigma**2))

        # crop the edge of the injection at the edge of the image
        if xmin <= 0:
            psf = psf[:, -xmin:]
            xmin = 0
        if ymin <= 0:
            psf = psf[-ymin:, :]
            ymin = 0
        if xmax >= nx:
            psf = psf[:, :-(xmax-nx + 1)]
            xmax = nx - 1
        if ymax >= ny:
            psf = psf[:-(ymax-ny + 1), :]
            ymax = ny - 1

        # inject the stars into the image
        sim_data[ymin:ymax + 1, xmin:xmax + 1] += psf

    if add_gauss_noise:
        # add Gaussian random noise
        noise_rng = np.random.default_rng(10)
        gain = 1
        ref_flux = 10
        noise = noise_rng.normal(scale= ref_flux/gain * 0.1, size= size)
        sim_data = sim_data + noise

    # load as an image object
    frames = []
    prihdr, exthdr = create_default_headers()
    prihdr['VISTYPE'] = 'BORESITE'
    prihdr['RA'] = target[0]
    prihdr['DEC'] = target[1]

    newhdr = fits.Header(new_hdr)
    frame = data.Image(sim_data, pri_hdr= prihdr, ext_hdr= newhdr)
    filename = "simcal_astrom.fits"
    if filedir is not None:
        # save source SkyCoord locations and pixel location estimates
        guess = Table()
        guess['x'] = [int(x) for x in xpix]
        guess['y'] = [int(y) for y in ypix]
        guess['RA'] = cal_SkyCoords[pix_inds].ra
        guess['DEC'] = cal_SkyCoords[pix_inds].dec
        ascii.write(guess, filedir+'/guesses.csv', overwrite=True)

        frame.save(filedir=filedir, filename=filename)

    frames.append(frame)
    dataset = data.Dataset(frames)

    return dataset

def generate_mock_pump_trap_data(output_dir,meta_path, EMgain=10, 
                                 read_noise = 100, eperdn = 6, e2emode=False, 
                                 nonlin_path=None, arrtype='SCI'):
    """
    Generate mock pump trap data, save it to the output_directory
    
    Args:
        output_dir (str): output directory
        meta_path (str): metadata path
        EMgain (float): desired EM gain for frames
        read_noise (float): desired read noise for frames
        eperdn (float):  desired k gain (e-/DN conversion factor)
        e2emode (bool):  If True, e2e simulated data made instead of data for the unit test.  
            Difference b/w the two: 
            This e2emode data differs from the data generated when e2emode is False in the following ways:
            -The bright pixel of each trap is simulated in a more realistic way (i.e., at every phase time frame).
            -Simulated readout is more realistic (read noise, EM gain, k gain, nonlinearity, bias invoked after traps simulated).  
            In the other dataset (when e2emode is False), readout was simulated before traps were added, and no nonlinearity was applied.  
            Also, the number of electrons in the dark pixels of the dipoles can no longer be negative, and this condition is enforced.
            -The number of pumps and injected charge are much higher in these frames so that traps stand out above the read noise.  
            This was not an issue in the other dataset since read noise was added to frames that were EM-gained before charge was injected, which suppressed the effective read noise.
            -The EM gain used is 1.5.  For a large injected charge amount, the EM gain cannot be very high because of the risk of saturation.  
            -The number of phase times is 10 per scheme, to reduce the dataset size (compared to 100 when e2emode is False).
            -The frame format is ENG, as real trap-pump data is.
        nonlin_path (str): Path of nonlinearity correction file to use.  
            The inverse is applied, implementing rather than correcting nonlinearity.  
            If None, no nonlinearity is applied.  Defaults to None.
        arrtype (str): array type (for this function, choice of 'SCI' or 'ENG')
    """

    #If output_dir doesn't exist then make it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # here = os.path.abspath(os.path.dirname(__file__))
    # meta_path = Path(here, '..', 'util', 'metadata_test.yaml')
    #meta_path = Path(here, '..', 'util', 'metadata.yaml')
    meta = MetadataWrapper(meta_path)
    num_pumps = 10000
    multiple = 1
    #nrows, ncols, _ = meta._imaging_area_geom()
    # the way emccd_detect works is that it takes an input for the selected
    # image area within the viable CCD pixels, so my input here must be that
    # smaller size (below) as opposed to the full useable CCD pixel size
    # (commented out above)
    nrows, ncols, _ = meta._unpack_geom('image')
    #EM gain
    g = EMgain
    cic = 200  
    rn = read_noise 
    dc = {180: 0.163, 190: 0.243, 200: 0.323, 210: 0.403,
          220: 0.483}
    # dc = {180: 0, 190: 0, 195: 0, 200: 0, 210: 0, 220: 0}
    bias = 1000 
    inj_charge = 500 # 0
    full_well_image=50000.  # e-
    full_well_serial=50000.
    # trap-pumping done when CGI is secondary instrument (i.e., dark):
    fluxmap = np.zeros((nrows, ncols))
    # frametime for pumped frames: 1000ms, or 1 s
    frametime = 1
    # set these to have no effect, then use these with their input values at the end
    later_eperdn = eperdn
    if e2emode: 
        eperdn = 1
        cic = 0.02
        num_pumps = 50000 #120000#90000#15000#5000
        inj_charge = 27000 #31000#70000#45000#8000 #num_pumps/2 # more than num_pumps/4, so no mean_field input needed
        multiple = 1
        g = 1
        rn = 0
        bias = 0
        full_well_image=105000.  # e-
        full_well_serial=105000.
        phase_times = 10
    bias_dn = bias/eperdn
    nbits = 14 #1
    
    def _ENF(g, Nem):
        """
        Returns the ENF.

        Args:
            g (float): gain
            Nem (int): Nem

        Returns:
            float: ENF

        """
        return np.sqrt(2*(g-1)*g**(-(Nem+1)/Nem) + 1/g)
    # std dev in e-, before gain division
    std_dev = np.sqrt(100**2 + _ENF(g,604)**2*g**2*(cic+ 1*dc[220]))
    fit_thresh = 3 #standard deviations above mean for trap detection
    #Offset ensures detection.  Physically, shouldn't have to add offset to
    #frame to meet threshold for detection, but in a small-sized frame, the
    # addition of traps increases the std dev a lot
    # (gain divided, w/o offset: from 22 e- before traps to 73 e- after adding)
    # If I run code with lower threshold, though, I can do an offset of 0.
    # For regular full-sized, definitely shouldn't have to add in offset.
    # Also, a trap can't capture more than mean per pixel e-, which is 200e-
    # in this case.  So max amp P1 trap will not be 2500e- but rather the
    #mean e- per pixel!  But this discrepancy doesn't affect validity of tests.

    offset_u = 0
    # offset_u = (bias_dn + ((cic+1*dc[220])*g + fit_thresh*std_dev/g)/eperdn+\
    #    inj_charge/eperdn)
    # #offset_l = bias_dn + ((cic+1*dc[220])*g - fit_thresh*std_dev/g)/eperdn
    # gives these 0 offset in the function (which gives e-), then add it in
    # by hand and convert to DN
    # and I increase dark current with temp linearly (even though it should
    # be exponential, but dc really doesn't affect anything here)
    emccd = {}
    # leaving out 170K
    #170K: gain of 10-20; gives g*CIC ~ 2000 e-
    # emccd[170] = EMCCDDetect(
    #         em_gain=1,#10,
    #         full_well_image=50000.,  # e-
    #         full_well_serial=50000.,  # e-
    #         dark_current=0.083,  # e-/pix/s
    #         cic=200, # e-/pix/frame; lots of CIC from all the prep clocking
    #         read_noise=100.,  # e-/pix/frame
    #         bias=bias,  # e-
    #         qe=0.9,
    #         cr_rate=0.,  # hits/cm^2/s
    #         pixel_pitch=13e-6,  # m
    #         eperdn=7.,
    #         nbits=14,
    #         numel_gain_register=604,
    #         meta_path=meta_path
    #    )
    #180K: gain of 10-20
    emccd[180] = EMCCDDetect(
            em_gain=g,#10,
            full_well_image=full_well_image,  # e-
            full_well_serial=full_well_serial,  # e-
            dark_current=dc[180], #0.163,  # e-/pix/s
            cic=cic, # e-/pix/frame; lots of CIC from all the prep clocking
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #190K: gain of 10-20
    emccd[190] = EMCCDDetect(
            em_gain=g,#10,
            full_well_image=full_well_image,  # e-
            full_well_serial=full_well_serial,  # e-
            dark_current= dc[190],#0.243,  # e-/pix/s
            cic=cic, # e-/pix/frame
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #195K: gain of 10-20
    # emccd[195] = EMCCDDetect(
    #         em_gain=g,#10,
    #         full_well_image=50000.,  # e-
    #         full_well_serial=50000.,  # e-
    #         dark_current= dc[195],#0.263,  # e-/pix/s
    #         cic=cic, # e-/pix/frame
    #         read_noise=rn,  # e-/pix/frame
    #         bias=bias,  # e-
    #         qe=0.9,
    #         cr_rate=0.,  # hits/cm^2/s
    #         pixel_pitch=13e-6,  # m
    #         eperdn=eperdn,
    #         nbits=nbits,
    #         numel_gain_register=604,
    #         meta_path=meta_path
    #     )
    #200K: gain of 10-20
    emccd[200] = EMCCDDetect(
            em_gain=g,#10,
            full_well_image=full_well_image,  # e-
            full_well_serial=full_well_serial,  # e-
            dark_current=dc[200], #0.323,  # e-/pix/s
            cic=cic, # e-/pix/frame
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #210K: gain of 10-20
    emccd[210] = EMCCDDetect(
            em_gain=g, #10,
            full_well_image=full_well_image,  # e-
            full_well_serial=full_well_serial,  # e-
            dark_current=dc[210], #0.403,  # e-/pix/s
            cic=cic, # e-/pix/frame
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #220K: gain of 10-20
    emccd[220] = EMCCDDetect(
            em_gain=g, #10,
            full_well_image=full_well_image,  # e-
            full_well_serial=full_well_serial,  # e-
            dark_current=dc[220], #0.483,  # e-/pix/s
            cic=cic, # e-/pix/frame; divide by 15 to get the same 1000
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )

    #when tauc is 3e-3, that gives a mean e- field of 2090 e-
    tauc = 1e-8 #3e-3
    tauc2 = 1.2e-8 # 3e-3
    tauc3 = 1e-8 # 3e-3
    # tried for mean field test, but gave low amps that got lost in noise
    tauc4 = 1e-3 #constant Pc over time not a great approximation in theory
    #In order of amplitudes overall (given comparable tau and tau2):
    # P1 biggest, then P3, then P2
    # E,E3 and cs,cs3 params below chosen to ensure a P1 trap found at its
    # peak amp for good eperdn determination
    # E3,cs3: will give tau outside of 1e-6,1e-2
    # for all temps except 220K; we'll just make sure it's present in all
    # scheme 1 stacks for all temps to ensure good eperdn for all temps;
    # E, cs: will give tau outside of 1e-6, 1e-2
    # for just 170K, which I took out of temp_data
    # E2, cs2: fine for all temps
    E = 0.32 #eV
    E2 = 0.28 #0.24 # eV
    E3 = 0.4 #eV
    # tried mean field test (gets tau = 1e-4 for 180K)
    E4 = 0.266 #eV
    cs = 2 #in 1e-19 m^2
    cs2 = 12 #3 #8 # in 1e-19 m^2
    cs3 = 2 # in 1e-19 m^2
    # for mean field test
    cs4 = 4 # in 1e-19 m^2
    #temp_data = np.array([170, 180, 190, 200, 210, 220])
    temp_data = np.array([180, 190, 195, 200, 210, 220])
    #temp_data = np.array([180])
    taus = {}
    taus2 = {}
    taus3 = {}
    taus4 = {}
    for i in temp_data:
        taus[i] = tau_temp(i, E, cs)
        taus2[i] = tau_temp(i, E2, cs2)
        taus3[i] = tau_temp(i, E3, cs3)
        taus4[i] = tau_temp(i, E4, cs4)
    #tau = 7.5e-3
    #tau2 = 8.8e-3
    if e2emode:
        time_data = (np.logspace(-6, -2, phase_times))*10**6 # in us 
    else:
        time_data = (np.logspace(-6, -2, 100))*10**6 # in us 
    #time_data = (np.linspace(1e-6, 1e-2, 50))*10**6 # in us
    time_data = time_data.astype(float)
    # make one phase time a repitition
    time_data[-1] = time_data[-2]
    time_data = np.array(time_data.tolist()*multiple)
    time_data_s = time_data/10**6 # in s
    # half the # of frames for length limit
    length_limit = 5 #int(np.ceil((len(time_data)/2)))
    # mean of these frames will be a bit more than 2000e-, which is gain*CIC
    # std dev: sqrt(rn^2 + ENF^2 * g^2(e- signal))

    # with offset_u non-zero in below, I expect to get eperdn 4.7 w/ the code
    amps1 = {}; amps2 = {}; amps3 = {}
    amps1_k = {}; amps1_tau2 = {}; amps3_tau2 = {}; amps1_mean_field = {}
    amps2_mean_field = {}
    amps11 = {}; amps12 = {}; amps22 = {}; amps23 = {}; amps33 = {}; amps21 ={}
    for i in temp_data:
        amps1[i] = offset_u + g*P1(time_data_s, 0, tauc, taus[i], num_pumps)/eperdn
        amps11[i] = offset_u + g*P1_P1(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_pumps)/eperdn
        amps2[i] = offset_u + g*P2(time_data_s, 0, tauc, taus[i], num_pumps)/eperdn
        amps12[i] = offset_u + g*P1_P2(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_pumps)/eperdn
        amps22[i] = offset_u + g*P2_P2(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_pumps)/eperdn
        amps3[i] = offset_u + g*P3(time_data_s, 0, tauc, taus[i], num_pumps)/eperdn
        amps33[i] = offset_u + g*P3_P3(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_pumps)/eperdn
        amps23[i] = offset_u + g*P2_P3(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_pumps)/eperdn
        # just for (98,33)
        amps21[i] =  offset_u + g*P1_P2(time_data_s, 0, tauc2, taus2[i],
            tauc, taus[i], num_pumps)/eperdn
        # now a special amps just for ensuring good eperdn determination
        # actually, doesn't usually meet trap_id thresh, but no harm
        # including it
        amps1_k[i] = offset_u + g*P1(time_data_s, 0, tauc3, taus3[i], num_pumps)/eperdn
        # for the case of (89,2) with a single trap with tau2
        amps1_tau2[i] = offset_u + g*P1(time_data_s, 0, tauc2, taus2[i], num_pumps)/eperdn
        # for the case of (77,90) with a single trap with tau2
        amps3_tau2[i] = offset_u + g*P3(time_data_s, 0, tauc2, taus2[i], num_pumps)/eperdn
        #amps1_k[i] = g*2500/eperdn
        # make a trap for the mean_field test (when mean field=400e- < 2500e-)
        #this trap peaks at 250 e-
        amps1_mean_field[i] = offset_u + \
            g*P1(time_data_s,0,tauc4,taus4[i], num_pumps)/eperdn
        amps2_mean_field[i] = offset_u + \
            g*P2(time_data_s,0,tauc4,taus4[i], num_pumps)/eperdn
    amps_1_trap = {1: amps1, 2: amps2, 3: amps3, 'sp': amps1_k,
            '1b': amps1_tau2, '3b': amps3_tau2, 'mf1': amps1_mean_field,
            'mf2': amps2_mean_field}
    amps_2_trap = {11: amps11, 12: amps12, 21: amps21, 22: amps22, 23: amps23,
        33: amps33}

    #r0c0[0]: starting row for imaging area (physical CCD pixels)
    #r0c0[1]: starting col for imaging area (physical CCD pixels)
    _, _, r0c0 = meta._imaging_area_geom()

    def add_1_dipole(img_stack, row, col, ori, prob, start, end, temp):
        """Adds a dipole to an image stack img_stack at the location of the
        bright pixel given by row and col (relative to image area coordinates)
        that is of orientation 'above' or
        'below' (specified by ori) for a number of unique phase times
        going from start to end (inclusive; don't use -1 for end; 0 for start
        means first frame, length of time array means last frame), and the
        dipole is of the probability function prob (which can be 1, 2, 3,
        'sp', '1b', '3b', 'mf1', or 'mf2').
        The temperature is specified by temp (in K).
        
        When e2emode is True, the amount subtracted from the dark pixel and added to the bright 
        pixel of a given dipole is constrained so that a pixel is not left with a negative number of electrons. 
        See doc string of generate_mock_pump_trap_data for full e2emode details.

        Args: 
            img_stack (np.array): image stack
            row (int): row
            col (int): col
            ori (str): orientation
            prob (int): probability
            start (int): start
            end (int): end
            temp (int): temperature

        Returns:
            np.array: image stack
        """
        # length limit controlled by how 'long' deficit pixel is since
        #threshold should be met for all frames for bright pixel
        if ori == 'above':
            #img_stack[start:end,r0c0[0]+row+1,r0c0[1]+col] = offset_l
            region = img_stack[start:end,r0c0[0]+row+1,r0c0[1]+col]
            region_c = img_stack[start:end,r0c0[0]+row+1,r0c0[1]+col].copy()
        if ori == 'below':
            #img_stack[start:end,r0c0[0]+row-1,r0c0[1]+col] = offset_l
            region = img_stack[start:end,r0c0[0]+row-1,r0c0[1]+col]
            region_c = img_stack[start:end,r0c0[0]+row-1,r0c0[1]+col].copy()
        region -= amps_1_trap[prob][temp][start:end]
        if e2emode:
            # can't draw more e- than what's there
            neg_inds = np.where(region < 0)
            good_inds = np.where(region >= 0)
            if neg_inds[0].size > 0:
                print(neg_inds[0].size)
                pass
            region[neg_inds[0]] = 0
            img_stack[start:end,r0c0[0]+row,r0c0[1]+col][good_inds[0]] += amps_1_trap[prob][temp][start:end][good_inds[0]]
            img_stack[start:end,r0c0[0]+row,r0c0[1]+col][neg_inds[0]] += region_c[neg_inds[0]]
        else:
            img_stack[: ,r0c0[0]+row,r0c0[1]+col] += amps_1_trap[prob][temp][:]

        return img_stack

    def add_2_dipole(img_stack, row, col, ori1, ori2, prob, start1, end1,
        start2, end2, temp):
        """Adds a 2-dipole to an image stack img_stack at the location of the
        bright pixel given by row and col (relative to image area coordinates)
        that is of orientation 'above' or
        'below' (specified by ori1 and ori2).  The 1st dipole is for a number
        of unique phase times going from start1 to end1, and
        the 2nd dipole starts from start2 and ends at end2 (inclusive; don't
        use -1 for end; 0 for start means first frame, length of time array
        means last frame). The 2-dipole is of probability function
        prob.  Valid values for prob are 11, 12, 22, 23, and 33.
        The temperature is specified by temp (in K).

        When e2emode is True, the amount subtracted from the dark pixel and added to the bright 
        pixel of a given dipole is constrained so that a pixel is not left with a negative number of electrons. 
        Also, start2:end2 should not overlap with start1:end1, and the ranges should 
        cover the whole 0:10 frames.  This condition allows for the simulation of the probability 
        distribution across all phase times.
        See doc string of generate_mock_pump_trap_data for full e2emode details.
        
        Args:
            img_stack (np.array): image stack
            row (int): row
            col (int): col
            ori1 (str): orientation 1
            ori2 (str): orientation 2
            prob (int): probability
            start1 (int): start 1
            end1 (int): end 1
            start2 (int): start 2
            end2 (int): end 2  
            temp (int): temperature

        Returns:
            np.array: image stack    
        """
        # length limit controlled by how 'long' deficit pixel is since
        #threshold should be met for all frames for bright pixel
        if ori1 == 'above':
            region1 = img_stack[start1:end1,r0c0[0]+row+1,r0c0[1]+col]
            region1_c = img_stack[start1:end1,r0c0[0]+row+1,r0c0[1]+col].copy()
            #img_stack[start1:end1,r0c0[0]+row+1,r0c0[1]+col] = offset_l
        if ori1 == 'below':
            #img_stack[start1:end1,r0c0[0]+row-1,r0c0[1]+col] = offset_l
            region1 = img_stack[start1:end1,r0c0[0]+row-1,r0c0[1]+col] 
            region1_c = img_stack[start1:end1,r0c0[0]+row-1,r0c0[1]+col].copy()
        if ori2 == 'above':
            #img_stack[start2:end2,r0c0[0]+row+1,r0c0[1]+col] = offset_l
            region2 = img_stack[start2:end2,r0c0[0]+row+1,r0c0[1]+col]
            region2_c = img_stack[start2:end2,r0c0[0]+row+1,r0c0[1]+col].copy()
        if ori2 == 'below':
            region2 = img_stack[start2:end2,r0c0[0]+row-1,r0c0[1]+col]
            region2_c = img_stack[start2:end2,r0c0[0]+row-1,r0c0[1]+col].copy()
        # technically, should subtract 1 prob distribution at at time (amps_1_trap), but I'm just subtracting 
        # a bit more than I'm supposed to, and doesn't matter too much since these 
        # are the deficit pixels (or pixel) next to the bright pixel, which is what counts for doing fits
        region1 -= amps_2_trap[prob][temp][start1:end1]
        region2 -= amps_2_trap[prob][temp][start2:end2]
        if e2emode:
            # can't draw more e- than what's there
            neg_inds1 = np.where(region1 < 0)
            if neg_inds1[0].size > 0:
                print(neg_inds1[0].size)
                pass
            good_inds1 = np.where(region1 >= 0)
            region1[neg_inds1] = 0
            img_stack[start1:end1,r0c0[0]+row,r0c0[1]+col][good_inds1[0]] += amps_2_trap[prob][temp][start1:end1][good_inds1[0]]
            img_stack[start1:end1,r0c0[0]+row,r0c0[1]+col][neg_inds1[0]] += region1_c[neg_inds1[0]]
        
            # can't draw more e- than what's there
            neg_inds2 = np.where(region2 < 0)
            if neg_inds2[0].size > 0:
                print(neg_inds2[0].size)
                pass
            good_inds2 = np.where(region2 >= 0)
            region2[neg_inds2] = 0
            img_stack[start2:end2,r0c0[0]+row,r0c0[1]+col][good_inds2[0]] += amps_2_trap[prob][temp][start2:end2][good_inds2[0]]
            img_stack[start2:end2,r0c0[0]+row,r0c0[1]+col][neg_inds2[0]] += region2_c[neg_inds2[0]]
        
        else:
            img_stack[:,r0c0[0]+row,r0c0[1]+col] += amps_2_trap[prob][temp][:]
        # technically, if there is overlap b/w start1:end1 and start2:end2,
        # then you are physically causing too big of a deficit since you're
        # saying more emitted than the amount captured in bright pixel, so
        # avoid this
        return img_stack

    def make_scheme_frames(emccd_inst, phase_times = time_data,
        inj_charge = inj_charge ):
        """Makes a series of frames according to the emccd_detect instance
        emccd_inst, one for each element in the array phase_times (assumed to
        be in s).

        Args:
            emccd_inst (EMCCDDetect): emccd instance
            phase_times (np.array): phase times
            inj_charge (int): injection charge

        Returns:
            np.array: full frames
        """
        full_frames = []
        for i in range(len(phase_times)):
            full = (emccd_inst.sim_full_frame(fluxmap,frametime)).astype(float)
            full_frames.append(full)
        # inj charge is before gain, but since it has no variance,
        # g*0 = no noise from this
        full_frames = np.stack(full_frames)
        # lazy and not putting in the last image row and col, but doesn't
        #matter since I only use prescan and image areas
        # add to just image area so that it isn't wiped with bias subtraction
        full_frames[:,r0c0[0]:,r0c0[1]:] += inj_charge
        return full_frames

    def add_defect(sch_imgs, prob, ori, temp):
        """Adds to all frames of an image stack sch_imgs a defect area with
        local mean above image-area mean such that a
        dipole in that area that isn't detectable unless ill_corr is True.
        The dipole is a single trap with orientation
        ori ('above' or 'below') and is of probability function prob
        (can be 1, 2, or 3).  The temperature is specified by temp (in K).

        Note: If a defect region is arbitrarily small (e.g., a 2x2 region of
        very bright pixels hiding a trap dipole), that trap simply will not
        be found since the illumination correction bin size is not allowed to
        be less than 5.  In v2.0, a moving median subtraction can be
        implemented that would be more likely to catch cases similar to that.
        However, physically, a defect region of such a small number of rows is
        improbable; even a cosmic ray hit, which could have this signature for
        perhaps 1 phase time, is very unlikely to hit the same region while
        data for each phase time is being taken.
        
        When e2emode is True, the amount subtracted from the dark pixel and added to the bright 
        pixel of a given dipole is constrained so that a pixel is not left with a negative number of electrons. 
        This condition allows for the simulation of the probability 
        distribution across all phase times.
        See doc string of generate_mock_pump_trap_data for full e2emode details.

        Args: 
            sch_imgs (np.array): scheme images
            prob (int): probability
            ori (str): orientation
            temp (int): temperature

            
        Returns:
            np.array: scheme images
            
        """
        # area with defect (high above mean),
        # but no dipole that stands out enough without ill_corr = True
        amount = 9000
        if e2emode:
            amount = inj_charge*2
        sch_imgs[:,r0c0[0]+12:r0c0[0]+22,r0c0[1]+17:r0c0[1]+27]=g*amount/eperdn
        # now a dipole that meets threshold around local mean doesn't meet
        # threshold around frame mean; would be detected only after
        # illumination correction
        if ori == 'above':
            region = sch_imgs[:,r0c0[0]+13+1, r0c0[1]+21] 
            region_c = region.copy()
        if ori == 'below':
            region = sch_imgs[:,r0c0[0]+13-1, r0c0[1]+21] 
            region_c = region.copy()
                # 2*offset_u - fit_thresh*std_dev/eperdn
        region -= amps_1_trap[prob][temp][:]
        if e2emode: # realistic handling:  can't trap more charge than what's there in a pixel
            neg_inds = np.where(region < 0)
            if neg_inds[0].size > 0:
                print(neg_inds[0].size)
            good_inds = np.where(region >= 0)
            region[neg_inds[0]] = 0
            sch_imgs[good_inds[0],r0c0[0]+13, r0c0[1]+21] += amps_1_trap[prob][temp][good_inds[0]]
            sch_imgs[neg_inds[0],r0c0[0]+13,r0c0[1]+21] += region_c[neg_inds[0]]
        else:
            sch_imgs[:,r0c0[0]+13, r0c0[1]+21] += amps_1_trap[prob][temp][:]

        return sch_imgs
    
    #initializing
    sch = {1: None, 2: None, 3: None, 4: None}
    #temps = {170: sch, 180: sch, 190: sch, 200: sch, 210: sch, 220: sch}
    # change from last iteration: make copies of sch below b/c make_scheme_frames() below was changing sch present in 
    # EVERY temp for every iteration in the temps for loop; however, no actual change in the output since 
    # the output .fits files were saved before the next iteration's make_scheme_frames() is called. So, Max's
    # unit test is unchanged. 
    temps = {180: sch, 190: sch.copy(), 200: sch.copy(), 210: sch.copy(), 220: sch.copy()}
    #temps = {180: sch}

    # first, get rid of files already existing in the folders where I'll put
    # the simulated data
    # for temp in temps.keys():
    #     for sch in [1,2,3,4]:
    #         curr_sch_dir = Path(here, 'test_data_sub_frame_noise', str(temp)+'K',
    #             'Scheme_'+str(sch))
    #         for file in os.listdir(curr_sch_dir):
    #             os.remove(Path(curr_sch_dir, file))

    for temp in temps.keys():
        for sc in [1,2,3,4]:
            temps[temp][sc] = make_scheme_frames(emccd[temp])
        # 14 total traps (15 with the (13,19) defect trap); at least 1 in every
        # possible sub-electrode location
        # careful not to add traps in defect region; do that with add_defect()
        # careful not to add, e.g., bright pixel of one trap in the deficit
        # pixel of another trap since that would negate the original trap

        # add in 'LHSel1' trap in midst of defect for all phase times
        # (only detectable with ill_corr)
        add_defect(temps[temp][1], 1, 'below', temp)
        add_defect(temps[temp][3], 3, 'below', temp)
        #this defect was used for k_prob=2 case instead of the 2 lines above
        # 'LHSel2':
    #    add_defect(temps[temp][1], 2, 'above', temp)
    #    add_defect(temps[temp][2], 1, 'below', temp)
    #    add_defect(temps[temp][4], 3, 'above', temp)
        # add in 'special' max amp trap for good eperdn determination
        # has tau value outside of 1e-6 to 1e-2, but provides a peak trap
        # actually, doesn't meet threshold usually to count as trap, but
        #no harm leaving it in
        if not e2emode:
            add_1_dipole(temps[temp][1], 33, 77, 'below', 'sp', 0, 100, temp)
            # add in 'CENel1' trap for all phase times
        #    add_1_dipole(temps[temp][3], 26, 28, 'below', 'mf2', 0, 100, temp)
        #    add_1_dipole(temps[temp][4], 26, 28, 'above', 'mf2', 0, 100, temp)
            add_1_dipole(temps[temp][3], 26, 28, 'below', 2, 0, 100, temp)
            add_1_dipole(temps[temp][4], 26, 28, 'above', 2, 0, 100, temp)
            # add in 'RHSel1' trap for more than length limit (but diff lengths)
            #unused sch2 in this same pixel that is compatible with another trap
            add_1_dipole(temps[temp][1], 50, 50, 'above', 1, 0, 100, temp)
            add_1_dipole(temps[temp][4], 50, 50, 'above', 3, 3, 98, temp)
            add_1_dipole(temps[temp][2], 50, 50, 'below', 1, 2, 99, temp)
            # FALSE TRAPS: 'LHSel2' trap that doesn't meet length limit of unique
            # phase times even though the actual length is met for first 2
            # (and/or doesn't pass trap_id(), but I've already tested this case in
            # its unit test file)
            # (3rd will be 'unused')
            add_1_dipole(temps[temp][1], 71, 84, 'above', 2, 95, 100, temp)
            add_1_dipole(temps[temp][2], 71, 84, 'below', 1, 95, 100, temp)
            add_1_dipole(temps[temp][4], 71, 84, 'above', 3, 9, 20, temp)
            # 'LHSel2' trap
            add_1_dipole(temps[temp][1], 60, 80, 'above', 2, 1, 100, temp)
            add_1_dipole(temps[temp][2], 60, 80, 'below', 1, 1, 100, temp)
            add_1_dipole(temps[temp][4], 60, 80, 'above', 3, 1, 100, temp)
            # 'CENel2' trap
            add_1_dipole(temps[temp][1], 68, 67, 'above', 1, 0, 100, temp)
            add_1_dipole(temps[temp][2], 68, 67, 'below', 1, 0, 100, temp)
        #    add_1_dipole(temps[temp][1], 68, 67, 'above', 'mf1', 0, 100, temp)
        #    add_1_dipole(temps[temp][2], 68, 67, 'below', 'mf1', 0, 100, temp)
            # 'RHSel2' and 'LHSel3' traps in same pixel (could overlap phase time),
            # but good detectability means separation of peaks
            add_1_dipole(temps[temp][1], 98, 33, 'above', 1, 0, 100, temp)
            add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
                60, 100, 0, 40, temp) #80, 100, 0, 20, temp)
            add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
                60, 100, 0, 40, temp)
            # old:
            # add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
            #     50, 100, 0, 50, temp) #80, 100, 0, 20, temp)
            # add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
            #     50, 100, 0, 50, temp)
            # 'CENel3' trap (where sch3 has a 2-trap where one goes unused)
            add_2_dipole(temps[temp][3], 41, 15, 'above', 'above', 23,
            30, 100, 0, 30, temp)
            add_1_dipole(temps[temp][4], 41, 15, 'below', 2, 30, 100, temp)
            # 'RHSel3' and 'LHSel4'
            add_1_dipole(temps[temp][1], 89, 2, 'below', '1b', 0, 100, temp)
            add_2_dipole(temps[temp][2], 89, 2, 'above', 'above', 12,
                60, 100, 0, 30, temp) #30 was 40 in the past
            add_2_dipole(temps[temp][3], 89, 2, 'above', 'above', 33,
                60, 100, 0, 40, temp)
            # 2 'LHSel4' traps; whether the '0' or '1' trap gets assigned tau2 is
            # somewhat random; if one has an earlier starting temp than the other,
            # it would get assigned tau
            add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
                0, 40, 63, 100, temp)
            add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
                0, 40, 63, 100, temp)
            add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
                0, 40, 63, 100, temp) #30, 60, 100
            # old:
            # add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
            #     0, 40, 50, 100, temp)
            # 'CENel4' trap
            add_1_dipole(temps[temp][1], 56, 56, 'below', 1, 1, 100, temp)
            add_1_dipole(temps[temp][2], 56, 56, 'above', 1, 3, 99, temp)
            #'RHSel4' and 'CENel2' trap (tests 'a' and 'b' splitting in trap_fit_*)
            add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
                60, 100, 0, 40, temp)
            add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
                60, 100, 0, 40, temp)
            add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, 40, temp)
            # old:
            # add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
            #     30, 100, 0, 30, temp)
            # add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
            #     53, 100, 0, 53, temp)
            # add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, 30, temp)

        if e2emode: # full range should be covered if trap present
            add_1_dipole(temps[temp][1], 33, 77, 'below', 'sp', 0, phase_times, temp)
            # add in 'CENel1' trap for all phase times
        #    add_1_dipole(temps[temp][3], 26, 28, 'below', 'mf2', 0, 100, temp)
        #    add_1_dipole(temps[temp][4], 26, 28, 'above', 'mf2', 0, 100, temp)
            add_1_dipole(temps[temp][3], 26, 28, 'below', 2, 0, phase_times, temp)
            add_1_dipole(temps[temp][4], 26, 28, 'above', 2, 0, phase_times, temp)
            # add in 'RHSel1' trap for more than length limit (but diff lengths)
            #unused sch2 in this same pixel that is compatible with another trap
            add_1_dipole(temps[temp][1], 50, 50, 'above', 1, 0, phase_times, temp)
            add_1_dipole(temps[temp][4], 50, 50, 'above', 3, 3, phase_times, temp)
            add_1_dipole(temps[temp][2], 50, 50, 'below', 1, 2, phase_times, temp)
            # FALSE TRAPS: 'LHSel2' trap that doesn't meet length limit of unique
            # phase times even though the actual length is met for first 2
            # (and/or doesn't pass trap_id(), but I've already tested this case in
            # its unit test file)
            # (3rd will be 'unused')
            add_1_dipole(temps[temp][1], 71, 84, 'above', 2, 95, phase_times, temp)
            add_1_dipole(temps[temp][2], 71, 84, 'below', 1, 95, phase_times, temp)
            add_1_dipole(temps[temp][4], 71, 84, 'above', 3, 9, phase_times, temp)
            # 'LHSel2' trap
            add_1_dipole(temps[temp][1], 60, 80, 'above', 2, 1, phase_times, temp)
            add_1_dipole(temps[temp][2], 60, 80, 'below', 1, 1, phase_times, temp)
            add_1_dipole(temps[temp][4], 60, 80, 'above', 3, 1, phase_times, temp)
            # 'CENel2' trap
            add_1_dipole(temps[temp][1], 68, 67, 'above', 1, 0, phase_times, temp)
            add_1_dipole(temps[temp][2], 68, 67, 'below', 1, 0, phase_times, temp)
        #    add_1_dipole(temps[temp][1], 68, 67, 'above', 'mf1', 0, 100, temp)
        #    add_1_dipole(temps[temp][2], 68, 67, 'below', 'mf1', 0, 100, temp)
            # 'RHSel2' and 'LHSel3' traps in same pixel (could overlap phase time),
            # but good detectability means separation of peaks
            add_1_dipole(temps[temp][1], 98, 33, 'above', 1, 0, phase_times, temp)
            add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp) #80, 100, 0, 20, temp)
            add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            # old:
            # add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
            #     50, 100, 0, 50, temp) #80, 100, 0, 20, temp)
            # add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
            #     50, 100, 0, 50, temp)
            # 'CENel3' trap (where sch3 has a 2-trap where one goes unused)
            add_2_dipole(temps[temp][3], 41, 15, 'above', 'above', 23,
            int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            add_1_dipole(temps[temp][4], 41, 15, 'below', 2, 0, phase_times, temp)
            # 'RHSel3' and 'LHSel4'
            add_1_dipole(temps[temp][1], 89, 2, 'below', '1b', 0, phase_times, temp)
            add_2_dipole(temps[temp][2], 89, 2, 'above', 'above', 12,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp) #30 was 40 in the past
            add_2_dipole(temps[temp][3], 89, 2, 'above', 'above', 33,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            # 2 'LHSel4' traps; whether the '0' or '1' trap gets assigned tau2 is
            # somewhat random; if one has an earlier starting temp than the other,
            # it would get assigned tau
            add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
                0, int(phase_times/2), int(phase_times/2), phase_times, temp)
            add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
                0, int(phase_times/2), int(phase_times/2), phase_times, temp)
            add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
                0, int(phase_times/2), int(phase_times/2), phase_times, temp) #30, 60, 100
            # old:
            # add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
            #     0, 40, 50, 100, temp)
            # 'CENel4' trap
            add_1_dipole(temps[temp][1], 56, 56, 'below', 1, 1, phase_times, temp)
            add_1_dipole(temps[temp][2], 56, 56, 'above', 1, 3, phase_times, temp)
            #'RHSel4' and 'CENel2' trap (tests 'a' and 'b' splitting in trap_fit_*)
            add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, phase_times, temp)
            # old:
            # add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
            #     30, 100, 0, 30, temp)
            # add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
            #     53, 100, 0, 53, temp)
            # add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, 30, temp)
        pass
        if e2emode:
            readout_emccd = EMCCDDetect(
                em_gain=EMgain, #10,
                full_well_image=full_well_image,  # e-
                full_well_serial=full_well_serial,  # e-
                dark_current=0,  # e-/pix/s
                cic=0, # e-/pix/frame
                read_noise=read_noise,  # e-/pix/frame
                bias=1000,  # e-
                qe=1, # no QE hit here; just simulating readout
                cr_rate=0.,  # hits/cm^2/s
                pixel_pitch=13e-6,  # m
                eperdn=later_eperdn,
                nbits=nbits,
                numel_gain_register=604,
                meta_path=meta_path,
                nonlin_path=nonlin_path
                )
        # save to FITS files
        for sc in [1,2,3,4]:
            for i in range(len(temps[temp][sc])):
                if e2emode:
                    if temps[temp][sc][i].any() >= full_well_image:
                        raise Exception('Saturated before EM gain applied.')
                    # Now apply readout things for e2e mode 
                    gain_counts = np.reshape(readout_emccd._gain_register_elements(temps[temp][sc][i].ravel()),temps[temp][sc][i].shape)
                    if gain_counts.any() >= full_well_serial:
                        raise Exception('Saturated after EM gain applied.')
                    output_dn = readout_emccd.readout(gain_counts)
                else:
                    output_dn = temps[temp][sc][i]
                prihdr, exthdr = create_default_headers(arrtype)
                prim = fits.PrimaryHDU(header = prihdr)
                hdr_img = fits.ImageHDU(output_dn, header=exthdr)
                hdul = fits.HDUList([prim, hdr_img])
                ## Fill in the headers that matter to corgidrp
                hdul[1].header['EXCAMT']  = temp
                hdul[1].header['CMDGAIN'] = EMgain
                hdul[1].header['ARRTYPE'] = arrtype
                for j in range(1, 5):
                    if sc == j:
                        hdul[1].header['TPSCHEM' + str(j)] = num_pumps
                    else:
                        hdul[1].header['TPSCHEM' + str(j)] = 0
                hdul[1].header['TPTAU'] = time_data[i]
                
                t = time_data[i]
                # curr_sch_dir = Path(here, 'test_data_sub_frame_noise', str(temp)+'K',
                # 'Scheme_'+str(sch))

                # if os.path.isfile(Path(output_dir,
                # str(temp)+'K'+'Scheme_'+str(sch)+'TPUMP_Npumps_10000_gain'+str(g)+'_phasetime'+str(t)+'.fits')):
                #     hdul.writeto(Path(output_dir,
                #     str(temp)+'K'+'Scheme_'+str(sch)+'TPUMP_Npumps_10000_gain'+str(g)+'_phasetime'+
                #     str(t)+'_2.fits'), overwrite = True)
                # else: 
                mult_counter = 0
                filename = Path(output_dir,
                str(temp)+'K'+'Scheme_'+str(sc)+'TPUMP_Npumps'+str(int(num_pumps))+'_gain'+str(EMgain)+'_phasetime'+
                str(t)+'.fits')
                if multiple > 1:
                    if not os.path.exists(filename):
                        hdul.writeto(filename, overwrite = True)
                    else:
                        mult_counter += 1
                        hdul.writeto(str(filename)[:-4]+'_'+str(mult_counter)+'.fits', overwrite = True)
                else:
                    hdul.writeto(filename, overwrite = True)

default_wcs_string = """WCSAXES =                    2 / Number of coordinate axes                      
CRPIX1  =                  0.0 / Pixel coordinate of reference point            
CRPIX2  =                  0.0 / Pixel coordinate of reference point            
CDELT1  =                  1.0 / Coordinate increment at reference point        
CDELT2  =                  1.0 / Coordinate increment at reference point        
CRVAL1  =                  0.0 / Coordinate value at reference point            
CRVAL2  =                  0.0 / Coordinate value at reference point            
LATPOLE =                 90.0 / [deg] Native latitude of celestial pole        
MJDREF  =                  0.0 / [d] MJD of fiducial time
"""

def gaussian_array(array_shape=[50,50],sigma=2.5,amp=1.,xoffset=0.,yoffset=0.):
    """Generate a 2D square array with a centered gaussian surface (for mock PSF data).

    Args:
        array_shape (int, optional): Shape of desired array in pixels. Defaults to [50,50].
        sigma (float, optional): Standard deviation of the gaussian curve, in pixels. Defaults to 5.
        amp (float,optional): Amplitude of gaussian curve. Defaults to 1.
        xoffset (float,optional): x offset of gaussian from array center. Defaults to 0.
        yoffset (float,optional): y offset of gaussian from array center. Defaults to 0.
        
    Returns:
        np.array: 2D array of a gaussian surface.
    """
    x, y = np.meshgrid(np.linspace(-array_shape[0]/2, array_shape[0]/2, array_shape[0]),
                        np.linspace(-array_shape[1]/2, array_shape[1]/2, array_shape[1]))
    dst = np.sqrt((x-xoffset)**2+(y-yoffset)**2)

    # lower normal part of gaussian
    normal = 1/(2.0 * np.pi * sigma**2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst)**2 / (2.0 * sigma**2))) * normal * amp
    
    return gauss

def create_psfsub_dataset(n_sci,n_ref,roll_angles,darkhole_scifiles=None,darkhole_reffiles=None,
                          wcs_header = None,
                          data_shape = [60,60],
                          centerxy = None,
                          outdir = None):
    """Generate a mock science and reference dataset ready for the PSF subtraction step.
    TODO: reference a central pixscale number, rather than hard code.

    Args:
        n_sci (int): number of science frames, must be >= 1.
        n_ref (int): nummber of reference frames, must be >= 0.
        roll_angles (list-like): list of the roll angles of each science and reference 
            frame, with the science frames listed first. 
        darkhole_scifiles (list of str, optional): Filepaths to the darkhole science frames. If not provided, 
            a noisy 2D gaussian will be used instead. Defaults to None.
        darkhole_reffiles (list of str, optional): Filepaths to the darkhole reference frames. If not provided, 
            a noisy 2D gaussian will be used instead. Defaults to None.
        wcs_header (astropy.fits.Header, optional): Fits header object containing WCS information. If not provided, 
            a mock header will be created. Defaults to None.
        data_shape (list-like): desired shape of data array. Must have length 2. Defaults to [1024,1024].
        outdir (str, optional): Desired output directory. If not provided, data will not be saved. Defaults to None.

    Returns:
        tuple: corgiDRP science Dataset object and reference Dataset object.
    """

    assert len(data_shape) == 2
    
    if roll_angles is None:
        roll_angles = [0.] * (n_sci+n_ref)

    # mask_center = np.array(data_shape)/2
    # star_pos = mask_center
    pixscale = 0.0218 # arcsec

    # Build each science/reference frame
    sci_frames = []
    ref_frames = []
    for i in range(n_sci+n_ref):

        # Create default headers
        prihdr, exthdr = create_default_headers()
        
        # Read in darkhole data, if provided
        if i<n_sci and not darkhole_scifiles is None:
            fpath = darkhole_scifiles[i]
            _,fname = os.path.split(fpath)
            darkhole = fits.getdata(fpath)
            
            fill_value = np.nanmin(darkhole)
            img_data = np.full(data_shape,fill_value)

            # Overwrite center of array with the darkhole data
            cr_psf_pix = np.array(darkhole.shape) / 2 - 0.5
            if centerxy is None:
                full_arr_center = np.array(img_data.shape) // 2 
            else:
                full_arr_center = (centerxy[1],centerxy[0])
            start_psf_ind = full_arr_center - np.array(darkhole.shape) // 2
            img_data[start_psf_ind[0]:start_psf_ind[0]+darkhole.shape[0],start_psf_ind[1]:start_psf_ind[1]+darkhole.shape[1]] = darkhole
            psfcenty, psfcentx = cr_psf_pix + start_psf_ind
        
        elif i>=n_sci and not darkhole_reffiles is None:
            fpath = darkhole_reffiles[i-n_sci]
            _,fname = os.path.split(fpath)
            darkhole = fits.getdata(fpath)
            fill_value = np.nanmin(darkhole)
            img_data = np.full(data_shape,fill_value)

            # Overwrite center of array with the darkhole data
            cr_psf_pix = np.array(darkhole.shape) / 2 - 0.5
            if centerxy is None:
                full_arr_center = np.array(img_data.shape) // 2 
            else:
                full_arr_center = (centerxy[1],centerxy[0])
            start_psf_ind = full_arr_center - np.array(darkhole.shape) // 2
            img_data[start_psf_ind[0]:start_psf_ind[0]+darkhole.shape[0],start_psf_ind[1]:start_psf_ind[1]+darkhole.shape[1]] = darkhole
            psfcenty, psfcentx = cr_psf_pix + start_psf_ind

        # Otherwise generate a 2D gaussian for a fake PSF
        else:
            label = 'ref' if i>= n_sci else 'sci'
            fname = f'MOCK_{label}_roll{roll_angles[i]}.fits'
            arr_center = np.array(data_shape) / 2 - 0.5
            if centerxy is None:
                psfcenty,psfcentx = arr_center
            else:
                psfcentx,psfcenty = centerxy
            
            psf_off_xy = (psfcentx-arr_center[1],psfcenty-arr_center[0])
            img_data = gaussian_array(array_shape=data_shape,
                                      xoffset=psf_off_xy[0],
                                      yoffset=psf_off_xy[1])
            
            # Add some noise
            rng = np.random.default_rng(seed=None)
            noise = rng.normal(0,1e-11,img_data.shape)
            img_data += noise

            # Add fake planet to sci files
            if i<n_sci:
                pa_deg = -roll_angles[i]
                sep_pix = 10
                xoff,yoff = sep_pix * np.array([-np.sin(np.radians(pa_deg)),np.cos(np.radians(pa_deg))])
                planet_psf = gaussian_array(array_shape=data_shape,
                                            amp=1e-6,
                                            xoffset=xoff+psf_off_xy[0],
                                            yoffset=yoff+psf_off_xy[1])
                img_data += planet_psf
        

        # Add necessary header keys
        prihdr['TELESCOP'] = 'ROMAN'
        prihdr['INSTRUME'] = 'CGI'
        prihdr['XOFFSET'] = 0.0
        prihdr['YOFFSET'] = 0.0
        prihdr['FILENAME'] = fname
        prihdr["MODE"] = 'HLC'
        prihdr["BAND"] = 1

        exthdr['BUNIT'] = 'MJy/sr'
        exthdr['MASKLOCX'] = psfcentx
        exthdr['MASKLOCY'] = psfcenty
        exthdr['STARLOCX'] = psfcentx
        exthdr['STARLOCY'] = psfcenty
        exthdr['PIXSCALE'] = pixscale
        exthdr["ROLL"] = roll_angles[i]
        exthdr["HIERARCH DATA_LEVEL"] = 'L3'
        #exthdr["HISTORY"] = "" # This line keeps triggering an "illegal value" error

        # Add WCS header info, if provided
        if wcs_header is None:
            wcs_header = fits.header.Header.fromstring(default_wcs_string,sep='\n')
            # wcs_header._cards = wcs_header._cards[-1]
        exthdr.extend(wcs_header)

        # Make a corgiDRP Image frame
        frame = data.Image(img_data, pri_hdr=prihdr, ext_hdr=exthdr)

        # Add it to the correct dataset
        if i < n_sci:
            sci_frames.append(frame)
        else:
            ref_frames.append(frame)

    sci_dataset = data.Dataset(sci_frames)

    if len(ref_frames) > 0:
        ref_dataset = data.Dataset(ref_frames)
    else:
        ref_dataset = None

    # Save datasets if outdir was provided
    if not outdir is None:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            
        sci_dataset.save(filedir=outdir, filenames=['mock_psfsub_L2b_sci_input_dataset.fits'])
        if len(ref_frames) > 0:
            ref_dataset.save(filedir=outdir, filenames=['mock_psfsub_L2b_ref_input_dataset.fits'])

    return sci_dataset,ref_dataset