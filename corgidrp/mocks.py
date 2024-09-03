import os
import numpy as np
import pandas as pd
import astropy.io.fits as fits
from astropy.time import Time
import numpy as np
import corgidrp.data as data
from corgidrp.data import Image
import corgidrp.detector as detector
import photutils.centroids as centr
from corgidrp.detector import imaging_area_geom, unpack_geom

import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord
import astropy.wcs as wcs
from astropy.table import Table

detector_areas_test= {
'SCI' : { #used for unit tests; enables smaller memory usage with frames of scaled-down comparable geometry
        'frame_rows' : 120,
        'frame_cols' : 220,
        'image' : {
            'rows': 104,
            'cols': 105,
            'r0c0': [2, 108]
            },
        'prescan' : {
            'rows': 120,
            'cols': 108,
            'r0c0': [0, 0]
            },
        'prescan_reliable' : {
            'rows': 120,
            'cols': 108,
            'r0c0': [0, 0]
            },
        'parallel_overscan' : {
            'rows': 14,
            'cols': 107,
            'r0c0': [106, 108]
            },
        'serial_overscan' : {
            'rows': 120,
            'cols': 5,
            'r0c0': [0, 215]
            },
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
            'rows': 120,
            'cols': 108,
            'r0c0': [0, 0]
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
    exthdr['BUNIT'] = 'detected EM electrons'
    exthdr['HIERARCH DATA_LEVEL'] = None
    # simulate raw data filenames
    exthdr['DRPNFILE'] = 2
    exthdr['FILE0'] = '0.fits'
    exthdr['FILE1'] = '1.fits'
    exthdr['B_O'] = 0.01
    exthdr['B_O_UNIT'] = 'DN'
    exthdr['B_O_ERR'] = 0.001

    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected EM electrons'
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
        n_dith (int): number of dithers required (n_dith=3 for Neptune and n_dith=2 for uranus)
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
    if planet == 'neptune':
        dith_end = n_dith
    elif planet == 'uranus':
        dith_end = n_dith+1
    
    for i in np.arange(-n_dith,dith_end):
        for j in np.arange(-n_dith,dith_end):
            mask_data = data.copy()
            image_data = mask_data[row_min + (dither_sizey * j):row_max + (dither_sizey * j), col_min + (dither_sizex * i):col_max + (dither_sizex * i)]
            cents.append(((mask_size/2) + (row_cent - int(row_cent)) - (dither_sizey//2) - (dither_sizey * j), (mask_size/2) + (col_cent - int(col_cent)) - (dither_sizex//2) - (dither_sizex * i)))
            # try:
            new_image_data = image_data * mask
            
            if planet == 'neptune' and band=='1' or planet == 'uranus' and band=='1':
                snr_ref = snr/np.sqrt(snr_constant)
            elif planet == 'neptune' and band=='4' or planet == 'uranus' and band=='4' :
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
    
def create_onsky_rasterscans(dataset,filedir=None,planet=None,band=None, im_size=420, d=None, n_dith=None, numfiles=36, radius=None, snr=250, snr_constant=None):
    """
    Create simulated data to check the flat division
    
    Args:
       dataset (corgidrp.data.Dataset): dataset of HST images of neptune and uranus
       filedir (str): Full path to directory to save the raster scanned images.
       planet (str): neptune or uranus
       band (str): 1 or 4
       im_size (int): x-dimension of the planet image (in pixels= 420 for the HST images)
       d (int): number of pixels across the planet (neptune=50 and uranus=65)
       n_dith (int): Number of dithers required (n_dith=3 for neptune and n_dith=2 for Uranus)
       numfiles (int): total number of raster images (default 36) 
       radius (int): radius of the planet in pixels (radius=54 for neptune, radius=90 in HST images)
       snr (int): SNR required for the planet image (default is 250 for the HST images)
       snr_constant (int): constant for snr reference  (4.95 for band1 and 9.66 for band4)
        
    Returns: 
    	corgidrp.data.Dataset:
        The simulated dataset of raster scanned images of planets uranus or neptune
    """
    n = im_size
    qe_prnu_fsm_raster = np.random.normal(1,.03,(n,n))
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
            if planet == 'neptune':
                planetrad=radius; snrcon=snr_constant
                planet_repoint_current = create_raster(qe_prnu_fsm_raster,planet_image,row_cent=yc+(d//2),col_cent=xc+(d//2), dither_sizex=d, dither_sizey=d,n_dith=n_dith,mask_size=n,snr=snr,planet=target,band=filter,radius=planetrad, snr_constant=snrcon)
            elif planet == 'uranus':
                planetrad=radius; snrcon=snr_constant     
                planet_repoint_current = create_raster(qe_prnu_fsm_raster,planet_image,row_cent=yc,col_cent=xc, dither_sizex=d, dither_sizey=d,n_dith=n_dith,mask_size=n,snr=snr,planet=target,band=filter,radius=planetrad, snr_constant=snrcon)
    
    for j in np.arange(len(planet_repoint_current[0])):
        for j in np.arange(len(planet_repoint_current[0])):
            planet_rot_images.append(planet_repoint_current[0][j])
            pred_cents.append(planet_repoint_current[1][j])
    filepattern= planet+'_'+band+"_"+"raster_scan_{0:01d}.fits"
    frames=[]
    for i in range(numfiles):
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
        data_range = np.linspace(10,65536,size)
        # Generate data for each row, where the mean increase from 10 to 65536
        for x in range(size):
            np.random.seed(123+x); sim_data[:, x] = np.random.poisson(data_range[x], size).astype(np.float64)

        non_linearity_correction = data.NonLinearityCalibration(nonlin_filepath)

        #Apply the non-linearity to the data. When we correct we multiple, here when we simulate we divide
        sim_data /= detector.get_relgains(sim_data,em_gain,non_linearity_correction)

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

def create_prescan_files(filedir=None, numfiles=2, obstype="SCI"):
    """
    Create simulated raw data.

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 2.
        obstype (str): Observation type. Defaults to "SCI".

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset
    """
    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    if obstype == "SCI":
        size = (1200, 2200)
    elif obstype == "ENG":
        size = (2200, 2200)
    elif obstype == "CAL":
        size = (2200,2200)
    else:
        raise ValueError(f'Obstype {obstype} not in ["SCI","ENG","CAL"]')


    filepattern = f"sim_prescan_{obstype}"
    filepattern = filepattern+"{0:04d}.fits"

    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_headers(obstype=obstype)
        sim_data = np.random.poisson(lam=150., size=size).astype(np.float64)
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)

        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))

        frames.append(frame)

    dataset = data.Dataset(frames)

    return dataset

def create_default_headers(obstype="SCI"):
    """
    Creates an empty primary header and an Image extension header with some possible keywords

    Args:
        obstype (str): Observation type. Defaults to "SCI".

    Returns:
        tuple:
            prihdr (fits.Header): Primary FITS Header
            exthdr (fits.Header): Extension FITS Header

    """
    prihdr = fits.Header()
    exthdr = fits.Header()

    if obstype != "SCI":
        NAXIS1 = 2200
        NAXIS2 = 1200
    else:
        NAXIS1 = 2200
        NAXIS2 = 2200

    # fill in prihdr
    prihdr['OBSID'] = 0
    prihdr['BUILD'] = 0
    prihdr['OBSTYPE'] = obstype
    prihdr['MOCK'] = True

    # fill in exthdr
    exthdr['NAXIS'] = 2
    exthdr['NAXIS1'] = NAXIS1
    exthdr['NAXIS2'] = NAXIS2
    exthdr['PCOUNT'] = 0
    exthdr['GCOUNT'] = 1
    exthdr['BSCALE'] = 1
    exthdr['BZERO'] = 32768
    exthdr['ARRTYPE'] = obstype # seems to be the same as OBSTYPE
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
    # Generate random values of rn in elecrons from a Gaussian distribution
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

def create_astrom_data(field_path, filedir=None, subfield_radius=0.02):
    """
    Create simulated data for astrometric calibration.

    Args:
        field_path (str): Full path to directory with test field data (ra, dec, vmag, etc.)
        filedir (str): (Optional) Full path to directory to save to.
        subfield_radius (float): The radius [deg] around the target coordinate for creating a subfield to produce the image from

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
    platescale = 21.8   #[mas]
    rotation = 45       #[deg]
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

    # add Gaussian random noise
    noise_rng = np.random.default_rng(10)
    gain = 1
    ref_flux = 10
    noise = noise_rng.normal(scale= ref_flux/gain * 0.1, size= size)
    sim_data = sim_data + noise

    # load as an image object
    frames = []
    prihdr, exthdr = create_default_headers()
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