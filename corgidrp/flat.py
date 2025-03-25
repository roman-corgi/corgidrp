# Place to put flatfield-related utility functions

import numpy as np
from scipy import interpolate
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter as gauss
from scipy import ndimage
from scipy.signal import convolve2d
import astropy.io.fits as fits
from astropy.convolution import convolve_fft
import photutils.centroids as centr
from photutils.aperture import CircularAperture

import corgidrp.data as data

def create_flatfield(flat_dataset):

    """
    Turn this dataset of image frames that were taken for performing the flat calibration and
    to make one master flat image
    this is currently a placeholder, until the final flat fielding calibration package is completed.

    Args:
       flat_dataset (corgidrp.data.Dataset): a dataset of Image frames (L2a-level)

    Returns:
        flat_field (corgidrp.data.FlatField): a master flat for flat calibration
    """


    combined_frame = np.nanmean(flat_dataset.all_data, axis=0)

    flat_field = data.FlatField(combined_frame, pri_hdr=flat_dataset[0].pri_hdr.copy(),
                         ext_hdr=flat_dataset[0].ext_hdr.copy(), input_dataset=flat_dataset)

    #determine the standard error of the mean: stddev/sqrt(n_frames)
    flat_field.err = np.nanstd(flat_dataset.all_data, axis=0)/np.sqrt(len(flat_dataset))
    flat_field.err=flat_field.err.reshape((1,)+flat_field.err.shape) # Get it into the right dimension


    return flat_field

def raster_kernel(width, image, hard=True):
    """
    Convolution kernel to create flat field circular raster pattern

    Args:
        width (float): radius of circular raster in pixels
        image (np.array): 2-D image to specify the full size of the kernel needed
        hard (bool): if true, use hard edge kernel, otherwise, use Gaussian tapered kernel
    
    Returns:
        np.array: smoothing kernel value at each pixel
    """

    kernel_width = width
    im = image

    # define coordinate grid
    x = np.arange(0,im.shape[1] + 1) - im.shape[1]/2 - 1
    y = np.arange(0,im.shape[0] + 1) - im.shape[0]/2 - 1
    xx,yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    rr_hard = np.sqrt(xx**2 + yy**2)

    # Define the convolution kernel
    if hard == True:
        rr_hard[rr_hard<=kernel_width/2] = 1
        rr_hard[rr_hard>kernel_width/2] = 0
        kernel = rr_hard # option 1: hard-edge kernel
    else:
        kernel = np.exp(-rr**2/(2*kernel_width**2)) # option 2: Gaussian kernel

    return kernel

def flatfield_residuals(images, N=None):
    """Turn this dataset of image frames of neptune or uranus and create matched filters and estimate residuals after 
     dividing from matched filters

     Todo: Propoagate the errors: incorporate the errors in the raster frames of neptune and jupiter and the errors in the matched filter

     Args:
    	images (np.array): 2D array of cropped neptune or uranus image frames
        N (int): number of images to be grouped. defaults to 3 for both Neptune and uranus. (If we use different number of dithers for neptune and uranus, option is provided to change N)

     Returns:
    	matched_residuals (np.array): residual image frames of neptune or uranus divided by matched filter
	"""
    raster_images = np.array(images)
    images_split = np.array(np.split(np.array(raster_images),N))
    matched_filters = np.array([np.nanmedian(np.stack(images_split[i],2),2) for i in np.arange(0,len(images_split))])
    matched_filters_smooth = [gauss(matched_filters[i],3) for i in range(len(matched_filters))] 
    
    matched_residuals=[];
    matched_residuals_err=[]
    for j in range(len(raster_images)):
        n_images=int(np.floor(j//(len(raster_images)//len(matched_filters_smooth))))
        matched_residuals.append(raster_images[j]/matched_filters_smooth[n_images])
    return(matched_residuals)
	    
def combine_flatfield_rasters(residual_images,cent=None,planet=None,band=None,im_size=420,rad_mask=None,  planet_rad=None, n_pix=165, n_pad=302):
    """combine the dataset of residual image frames of neptune or uranus and create flat field 
    	and associated error

    	Args:
        	residual_images (np.array): Residual images frames divided by the mnatched filter of neptune and uranus
            cent (np.array): centroid of the image frames
        	planet (str):   name of the planet neptune or uranus
        	band (str):  band of the observation band1 or band4
            im_size (int): x-dimension of the planet image (in pixels= 420 for the HST images)
            rad_mask (float): radius in pixels used for creating a mask for band (band 1 =1.25, band 4=1.75)
            planet_rad (int): radius of the planet in pixels (planet_rad=54 for neptune, planet_rad=65)
            n_pix (int): Number of pixels in radius covering the Roman CGI imaging FOV defaults to 165 pixels
            n_pad (int): Number of pixels padded with '1s'  to generate the image size 1024X1024 pixels around imaging FOV (defaults to 302 pixels)

        	
    	Returns:
        	full_residuals (np.array): flat field created from uranus or neptune images
        	err_residuals (np.array):  Error in the flatfield estimated using the ideal flat field
    """
    n = im_size
    
    full_residuals = np.zeros((n,n))
    err_residuals= np.zeros((n,n))
    if planet_rad is None:
        if planet.lower() == 'neptune':
             planet_rad = 50
        elif planet.lower() == 'uranus':
             planet_rad = 65
    
    if rad_mask is None:
         if band[0] == "1":
            rad_mask = 1.25
         elif band[0] == "4":
            rad_mask = 1.75
    
    aperture = CircularAperture((np.ceil(rad_mask), np.ceil(rad_mask)), r=rad_mask)
    mask= aperture.to_mask().data
    rad = planet_rad
    for i in np.arange(len(residual_images)):
        nx = np.arange(0,residual_images[i].shape[1])
        ny = np.arange(0,residual_images[i].shape[0])
        nxx,nyy = np.meshgrid(nx,ny)
        nrr = np.sqrt((nxx-rad-5)**2 + (nyy-rad-5)**2)
        nrr_copy = nrr.copy();  nrr_err_copy=nrr.copy()
        nrr_copy[nrr<rad] = residual_images[i][nrr<rad]
        nrr_copy[nrr>=rad] = None
        ymin = int(cent[i][0])
        ymax = int(cent[i][1])
        xmin = int(cent[i][2])
        xmax = int(cent[i][3])
        
        bool_innotzero = np.logical_and(nrr<rad,full_residuals[ymin:ymax,xmin:xmax]!=0)
        bool_iniszero = np.logical_and(nrr<rad,full_residuals[ymin:ymax,xmin:xmax]==0)
        bool_outisnotzero = np.logical_and(nrr>=rad,full_residuals[ymin:ymax,xmin:xmax]!=0)
        
        full_residuals[ymin:ymax,xmin:xmax][bool_innotzero] = (nrr_copy[bool_innotzero] + full_residuals[ymin:ymax,xmin:xmax][bool_innotzero])/2
        full_residuals[ymin:ymax,xmin:xmax][bool_iniszero] = nrr_copy[bool_iniszero]
        full_residuals[ymin:ymax,xmin:xmax][bool_outisnotzero] = full_residuals[ymin:ymax,xmin:xmax][bool_outisnotzero]

    
        full_residuals_resel = ndimage.convolve(full_residuals,mask)
        
        
    
    full_residuals[full_residuals==0] = None
    nx = np.arange(0,full_residuals_resel.shape[1])
    ny = np.arange(0,full_residuals_resel.shape[0])
    nxx,nyy = np.meshgrid(ny,nx)
    nrr = np.sqrt((nxx-n/2)**2 + (nyy-n/2)**2)
    full_residuals[nrr>n_pix]= 1
    
    
    full_residuals=np.pad(full_residuals, ((n_pad,n_pad),(n_pad,n_pad)), mode='constant',constant_values=(1))
    err_residuals=np.pad(err_residuals, ((n_pad,n_pad),(n_pad,n_pad)), mode='constant',constant_values=(0))
    
    return (full_residuals,err_residuals)
    
    
def create_onsky_flatfield(dataset, planet=None,band=None,up_radius=55,im_size=None,N=1,rad_mask=None, planet_rad=None, n_pix=44, n_pad=None, sky_annulus_rin=2, sky_annulus_rout=4):
    """Turn this dataset of image frames of uranus or neptune raster scannned that were taken for performing the flat calibration and create one master flat image. 
    The input image frames are L2b image frames that have been dark subtracted, divided by k-gain, divided by EM gain, desmeared. 

    
        Args:
            dataset (corgidrp.data.Dataset): a dataset of image frames that are raster scanned (L2a-level)
            planet (str): neptune or uranus
            band (str): 1 or 4
            up_radius (int): Number of pixels on either side of centroided planet images (=55 pixels for Neptune and uranus)
            im_size (int): x-dimension of the input image (in pixels; default is size of input dataset; = 420 for the HST images)
            N (int): Number of images to be combined for creating a matched filter (defaults to 1, may not work for N>1 right now)
            rad_mask (float): radius in pixels used for creating a mask for band (band1=1.25, band4=1.75)
            planet_rad (int): radius of the planet in pixels (planet_rad=50 for neptune, planet_rad=65)
            n_pix (int): Number of pixels in radius covering the Roman CGI imaging FOV (defaults to 44 pix for Band1 HLC; 165 pixels for full shaped pupil FOV).
            n_pad (int): Number of pixels padded with '1s'  to generate the image size 1024X1024 pixels around imaging FOV (defaults to None; rest of the FOV to reach 1024)
            sky_annulus_rin (float): Inner radius of annulus to use for sky subtraction. In units of planet_rad. 
                                     If both sky_annulus_rin and sky_annulus_rout = None, skips sky subtraciton.
            sky_annulus_rout (float): Outer radius of annulus to use for sky subtraction. In units of planet_rad. 
            
    	Returns:
    		data.FlatField (corgidrp.data.FlatField): a master flat for flat calibration using on sky images of planet in band specified
    		
	"""
    if im_size is None:
        # assume square images
        im_size = dataset[0].data.shape[0]

    if n_pad is None:
        n_pad = 1024 - im_size
        if n_pad < 0:
            n_pad = 0 

    if planet is None:
         planet=dataset[0].pri_hdr['TARGET']
    if band is None:
         band=dataset[0].ext_hdr['CFAMNAME']
    
    if planet_rad is None:
        if planet.lower() =='neptune':
             planet_rad = 50
        elif planet.lower() == 'uranus':
             planet_rad = 65
    
    if rad_mask is None:
         if band[0] == "1":
            rad_mask = 1.25
         elif band[0] == "4":
            rad_mask = 1.75

    smooth_images=[]; raster_images_cent=[]; cent=[]; act_cents=[]; frames=[];
    for j in range(len(dataset)):
        planet_image=dataset[j].data
        prihdr=dataset[j].pri_hdr
        exthdr=dataset[j].ext_hdr
        image_size=np.shape(planet_image)
        centroid = centr.centroid_com(planet_image)
        centroid[np.isnan(centroid)]=0
        act_cents.append((centroid[1],centroid[0]))
        xc =int( centroid[0])
        yc = int(centroid[1])

        # sky subtraction if needed
        if sky_annulus_rin is not None and sky_annulus_rout is not None:
            ycoords, xcoords = np.indices(planet_image.shape)
            dist_from_planet = np.sqrt((xcoords - xc)**2 + (ycoords - yc)**2)
            sky_annulus = np.where((dist_from_planet >= sky_annulus_rin*planet_rad) & (dist_from_planet < sky_annulus_rout*planet_rad))
            planet_image -= np.nanmedian(planet_image[sky_annulus])

        smooth_images.append(planet_image)
        # cropping the raster scanned images
        raster_images_cent.append(smooth_images[j][yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius])
        #centroid of the cropped images
        cent.append((yc-up_radius,yc+up_radius,xc-up_radius,xc+up_radius))

    resi_images=flatfield_residuals(raster_images_cent,N=N)
    raster_com=combine_flatfield_rasters(resi_images,planet=planet,band=band,cent=cent, im_size=im_size, rad_mask=rad_mask,planet_rad=planet_rad,n_pix=n_pix, n_pad=n_pad)
    onskyflat=raster_com[0]
    onsky_flatfield = data.FlatField(onskyflat, pri_hdr=prihdr, ext_hdr=exthdr, input_dataset=dataset)
    onsky_flatfield.err=raster_com[1]
    
    return(onsky_flatfield)
