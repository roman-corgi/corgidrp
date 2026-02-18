# Place to put flatfield-related utility functions
import warnings
import numpy as np
from scipy import interpolate
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter as gauss
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.ndimage import zoom
import astropy.io.fits as fits
from astropy.convolution import convolve_fft
from astropy.utils.exceptions import AstropyUserWarning
import photutils.centroids as centr
from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder
import corgidrp.data as data
import corgidrp.mocks as mocks


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

def flatfield_residuals(dataset, N=None):
    """Turn this dataset of image frames of neptune or uranus and create matched filters and estimate residuals after 
     dividing from matched filters


     Args:
    	dataset (corgidrp.data.Dataset): dataset of cropped neptune or uranus image frames
        N (int): number of images to be grouped. defaults to 3 for both Neptune and uranus. (If we use different number of dithers for neptune and uranus, option is provided to change N)

     Returns:
    	matched_residuals_dataset (corgidrp.data.Dataset) : dataset of neptune or uranus images divided by matched filter
	"""
    raster_images = np.array(dataset.all_data)
    # M is the number of images in each group used to calculate a filter.
    M = len(raster_images) // N
    images_split = np.array(np.split(np.array(raster_images),N))
    matched_filters = np.array([np.nanmedian(np.stack(images_split[i],2),2) for i in np.arange(0,len(images_split))])
    matched_filters_smooth = [gauss(matched_filters[i],3) for i in range(len(matched_filters))] 

    #Estimate the initial uncertainty (standard error of the mean/median) for each filter group.
    sigma_matched_filters = np.array([np.nanstd(images_split[i], axis=0) / np.sqrt(M) for i in np.arange(N)])
    # Apply the same smoothing (Gauss(sigma=3)) to the error maps as was applied to the filters.
    sigma_matched_filters_smooth = [gauss(sigma_matched_filters[i], 3) for i in range(N)]


    planet_matched_residuals=[]; planet_matched_residuals_err=[]; matched_residuals=[];matched_residuals_dataset=[]
    for j in range(len(dataset)):
        n_images=int(np.floor(j//(len(raster_images)//len(matched_filters_smooth))))
    
        planet_image = dataset[j].data
        planet_image_err = dataset[j].err
        prihdr=dataset[j].pri_hdr
        exthdr=dataset[j].ext_hdr

        planet_matched_filter= matched_filters_smooth[n_images]
        #calculate the residual
        planet_residual = planet_image / planet_matched_filter
        planet_matched_residuals.append(planet_residual)

        #error propagation for division (Z = X/Y) ---
        # Sigma_Z = |Z| * sqrt( (Sigma_X/X)^2 + (Sigma_Y/Y)^2 )
    
    
        planet_matched_filter_err = sigma_matched_filters_smooth[n_images]
        relative_variance_image = (planet_image_err / np.maximum(planet_image, 1e-6))**2
        #relative variance of the filter: (Sigma_Y/Y)^2
        relative_variance_filter = (planet_matched_filter_err / np.maximum(planet_matched_filter, 1e-6))**2
        #final Error Calculation
        planet_residual_error = planet_residual * np.sqrt(relative_variance_image + relative_variance_filter)
        planet_matched_residuals_err.append(planet_residual_error)
    
        matched_residuals_frames = mocks.Image(planet_residual, pri_hdr = prihdr, ext_hdr = exthdr, err = planet_residual_error)
    
        matched_residuals.append(matched_residuals_frames)
        matched_residuals_dataset=data.Dataset(matched_residuals)
        
    
    return(matched_residuals_dataset)
	    
def combine_flatfield_rasters(resi_images_dataset,cent=None,planet=None,band=None,im_size=420,rad_mask=None,  planet_rad=None, n_pix=165, n_pad=302,image_center_x=512,image_center_y=512):
    """combine the dataset of residual image frames of neptune or uranus and create flat field 
    	and associated error

    	Args:
        	resi_images_dataset (corgidrp.data.Dataset): dataset of residual images of uranus or Neptune
            cent (np.array): centroid of residual images of uranus or Neptune
        	planet (str):   name of the planet neptune or uranus
        	band (str):  band of the observation band1 or band4
            im_size (int): x-dimension of the planet image (in pixels= 420 for the HST images)
            rad_mask (float): radius in pixels used for creating a mask for band (band 1 =1.25, band 4=1.75)
            planet_rad (int): radius of the planet in pixels (planet_rad=54 for neptune, planet_rad=65)
            n_pix (int): Number of pixels in radius covering the Roman CGI imaging FOV defaults to 165 pixels
            n_pad (int): Number of pixels padded with '1s'  to generate the image size 1024X1024 pixels around imaging FOV (defaults to 302 pixels)
            image_center_x (int): x coordinate of the center pixel of the final image 
            image_center_y (int): y coordinate of the center pixel of the final image 

        	
    	Returns:
        	full_residuals (np.array): flat field created from uranus or neptune images
        	err_residuals (np.array):  Error in the flatfield estimated using the ideal flat field
            cent_n (np.array): centroid of the combined image
    """
    n = im_size
    
    full_residuals = np.zeros((n,n))
    p_flat=full_residuals.copy()
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
    for i  in range(len(resi_images_dataset)):
        nx = np.arange(0,resi_images_dataset[i].data.shape[1])
        ny = np.arange(0,resi_images_dataset[i].data.shape[0])
        nxx,nyy = np.meshgrid(nx,ny)
        nrr = np.sqrt((nxx-rad-5)**2 + (nyy-rad-5)**2)
       
        nrr_copy = nrr.copy();  
        nrr_err_copy=nrr.copy()
    
        nrr_copy[nrr<rad] = resi_images_dataset[i].data[nrr<rad]
        nrr_err_copy[nrr<rad] = resi_images_dataset[i].err[0][nrr<rad]
    
        nrr_copy[nrr>=rad] = None
        nrr_err_copy[nrr>=rad] = None
       
        ymin = int(cent[i][0])
        ymax = int(cent[i][1])
        xmin = int(cent[i][2])
        xmax = int(cent[i][3])
        
        bool_innotzero = np.logical_and(nrr<rad,full_residuals[ymin:ymax,xmin:xmax]!=0)
        bool_iniszero = np.logical_and(nrr<rad,full_residuals[ymin:ymax,xmin:xmax]==0)
        bool_outisnotzero = np.logical_and(nrr>=rad,full_residuals[ymin:ymax,xmin:xmax]!=0)

        bool_innotzero_err = np.logical_and(nrr<rad,err_residuals[ymin:ymax,xmin:xmax]!=0)
        bool_iniszero_err = np.logical_and(nrr<rad,err_residuals[ymin:ymax,xmin:xmax]==0)
        bool_outisnotzero_err = np.logical_and(nrr>=rad,err_residuals[ymin:ymax,xmin:xmax]!=0)
        
        full_residuals[ymin:ymax,xmin:xmax][bool_innotzero] = (nrr_copy[bool_innotzero] + full_residuals[ymin:ymax,xmin:xmax][bool_innotzero])/2
        full_residuals[ymin:ymax,xmin:xmax][bool_iniszero] = nrr_copy[bool_iniszero]
        full_residuals[ymin:ymax,xmin:xmax][bool_outisnotzero] = full_residuals[ymin:ymax,xmin:xmax][bool_outisnotzero]

        err_residuals[ymin:ymax,xmin:xmax][bool_innotzero_err] = (nrr_err_copy[bool_innotzero_err] + err_residuals[ymin:ymax,xmin:xmax][bool_innotzero_err])/2
        err_residuals[ymin:ymax,xmin:xmax][bool_iniszero_err] = nrr_err_copy[bool_iniszero_err]
        err_residuals[ymin:ymax,xmin:xmax][bool_outisnotzero_err] = err_residuals[ymin:ymax,xmin:xmax][bool_outisnotzero_err]
    
        full_residuals_resel = ndimage.convolve(full_residuals,mask)
        
    full_residuals[full_residuals==0] = None
    resid_mask = ~np.isnan(full_residuals)
    cent_rmask=centr.centroid_com(resid_mask)
    p_flat=np.roll(full_residuals, (image_center_x-int(cent_rmask[1]),image_center_y-int(cent_rmask[0])), axis=(0,1))
    p_flat_err=np.roll(err_residuals, (image_center_x-int(cent_rmask[1]),image_center_y-int(cent_rmask[0])), axis=(0,1))
    nx = np.arange(0,full_residuals_resel.shape[1])
    ny = np.arange(0,full_residuals_resel.shape[0])
    nxx,nyy = np.meshgrid(ny,nx)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AstropyUserWarning) 
        cent_n=centr.centroid_com(p_flat)
    nrr = np.sqrt((nxx-cent_n[0])**2 + (nyy-cent_n[1])**2)
    p_flat[nrr>n_pix//2]= 1
    p_flat_err[nrr>n_pix//2]=0
    
    full_residuals=np.pad(p_flat, ((n_pad,n_pad),(n_pad,n_pad)), mode='constant',constant_values=(1))
    err_residuals=np.pad(p_flat_err, ((n_pad,n_pad),(n_pad,n_pad)), mode='constant',constant_values=(0))
    
    return (full_residuals,err_residuals,cent_n)
    
    
def create_onsky_flatfield(dataset, planet=None,band=None,up_radius=55,im_size=None,N=1,rad_mask=None, planet_rad=None, n_pix=44, n_pad=None, sky_annulus_rin=2, sky_annulus_rout=4,image_center_x=512,image_center_y=512):
    """Turn this dataset of image frames of uranus or neptune raster scannned that were taken for performing the flat calibration and create one master flat image. 
    The input image frames are L2b image frames that have been dark subtracted, divided by k-gain, divided by EM gain, desmeared. 

    
        Args:
            dataset (corgidrp.data.Dataset): a dataset of image frames of uranus or neptune that are raster scanned (L2a-level)
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
            image_center_x (int): x coordinate of the center pixel of the final image 
            image_center_y (int): y coordinate of the center pixel of the final image
            
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

    raster_frames=[]; cent=[]; act_cents=[]; frames=[];
    for j in range(len(dataset)):
        planet_image=dataset[j].data
        planet_image_err=dataset[j].err[0]
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

        # cropping the raster scanned images
        raster_images=planet_image[yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius]
        raster_images_err=planet_image_err[yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius]
        #centroid of the cropped images
        cent.append((yc-up_radius,yc+up_radius,xc-up_radius,xc+up_radius))

        r_frames= mocks.Image(raster_images, pri_hdr = prihdr, ext_hdr = exthdr, err = raster_images_err)
        raster_frames.append(r_frames)
    
    raster_images_dataset=data.Dataset(raster_frames)
    resi_images_dataset=flatfield_residuals(raster_images_dataset,N=N)
    raster_com=combine_flatfield_rasters(resi_images_dataset,planet=planet,band=band,cent=cent, im_size=im_size, rad_mask=rad_mask,planet_rad=planet_rad,n_pix=n_pix, n_pad=n_pad,image_center_x=image_center_x,image_center_y=image_center_y)
    onskyflat=raster_com[0]
    onsky_flatfield = data.FlatField(onskyflat, pri_hdr=prihdr, ext_hdr=exthdr, input_dataset=dataset)
    onsky_flatfield.err=raster_com[1]
    
    return(onsky_flatfield)

def create_onsky_pol_flatfield(dataset, planet=None,band=None,up_radius=55,im_size=None,N=1,rad_mask=None, planet_rad=None, n_pix=None, observing_mode='NFOV', n_pad=None, fwhm_guess=20,sky_annulus_rin=2, sky_annulus_rout=4,plate_scale=0.0218,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,dpamname=None):
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
            observing_mode (string): observing mode of the coronagraph
            n_pad (int): Number of pixels padded with '1s'  to generate the image size 1024X1024 pixels around imaging FOV (defaults to None; rest of the FOV to reach 1024)
            fwhm_guess (int):FWHM guess for the planet image which is downsampled by a factor of 8
            sky_annulus_rin (float): Inner radius of annulus to use for sky subtraction. In units of planet_rad. 
                                     If both sky_annulus_rin and sky_annulus_rout = None, skips sky subtraciton.
            sky_annulus_rout (float): Outer radius of annulus to use for sky subtraction. In units of planet_rad. 
            plate_scale (float): platescale estimated from calibration (0.0218)
            image_center_x (int): x coordinate of the center pixel of the final image with two orthogonal pol components
            image_center_y (int): y coordinate of the center pixel of the final image with two orthogonal pol components
            separation_diameter_arcsec (float): separation between the two orthogonal pol components in arcsecs
            alignment_angle_WP1 (int): wollaston prism angle for Pol0 - 0
            alignment_angle_WP2 (int): wollaston prism angle for Pol45- 45
            dpamname (str): wollaston prism POL0 or POL45. Defaults to DPAMNAME in the header.
            
    	Returns:
    		data.FlatField (corgidrp.data.FlatField): a master flat corresponding to Pol0 or Pol 45 for flat calibration using on sky images of planet in band specified
    		
	"""
    
    if im_size is None:
        # assume square images
        im_size = dataset[0].data.shape[0]
    n=im_size
    if n_pad is None:
        n_pad = 1024 - im_size
        if n_pad < 0:
            n_pad = 0 

    if planet is None:
         planet=dataset[0].pri_hdr['TARGET']
    if band is None:
         band=dataset[0].ext_hdr['CFAMNAME']
    if dpamname is None:
        dpamname=dataset[0].ext_hdr['DPAMNAME']

    
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
    if n_pix is None:
        if observing_mode=='NFOV':
            n_pix=44
        elif observing_mode=='WFOV':
            n_pix=174
    
    # the planet images with two pol components are downsampled by a factor of 8 for finding the centroids using daostarfinder
    raster_frames_pol1=[];raster_frames_pol2=[]; cent_pol1=[]; cent_pol2=[]; act_cents_1=[]; act_cents_2=[];
    for j in range(len(dataset)):
        planet_image=dataset[j].data
        planet_image_err=dataset[j].err[0]
        prihdr=dataset[j].pri_hdr
        exthdr=dataset[j].ext_hdr
        planet_image_downsampled = zoom(planet_image, 1/8)
        threshold_value = np.max(planet_image)
        daofind = DAOStarFinder(fwhm=fwhm_guess, threshold=threshold_value, min_separation=10.0)
        sources = daofind(planet_image_downsampled)

        if sources is not None:

            x_centroids = sources['xcentroid']*8
            y_centroids = sources['ycentroid']*8

            pol1_x = int(x_centroids[0])
            pol1_y = int(y_centroids[0])
            pol2_x = int(x_centroids[1])
            pol2_y = int(y_centroids[1])

            x_centroids[np.isnan(x_centroids)]=0
            y_centroids[np.isnan(y_centroids)]=0
            act_cents_1.append((pol1_x,pol1_y))
            act_cents_2.append((pol2_x,pol2_y))


        # sky subtraction if needed
        xc=np.mean([pol1_x,pol2_x])
        yc=np.mean([pol1_y,pol2_y])
        if sky_annulus_rin is not None and sky_annulus_rout is not None:
            ycoords, xcoords = np.indices(planet_image.shape)
            dist_from_planet = np.sqrt((xcoords - xc)**2 + (ycoords - yc)**2)
            sky_annulus = np.where((dist_from_planet >= sky_annulus_rin*planet_rad) & (dist_from_planet < sky_annulus_rout*planet_rad))
            planet_image -= np.nanmedian(planet_image[sky_annulus])

        
        # cropping the raster scanned images
        raster_images_pol1=planet_image[pol1_y-up_radius:pol1_y+up_radius,pol1_x-up_radius:pol1_x+up_radius]
        raster_images_pol2=planet_image[pol2_y-up_radius:pol2_y+up_radius,pol2_x-up_radius:pol2_x+up_radius]
        
        raster_images_pol1_err=planet_image_err[pol1_y-up_radius:pol1_y+up_radius,pol1_x-up_radius:pol1_x+up_radius]
        raster_images_pol2_err=planet_image_err[pol2_y-up_radius:pol2_y+up_radius,pol2_x-up_radius:pol2_x+up_radius]
        
        #centroid of the cropped images
        cent_pol1.append((pol1_y-up_radius,pol1_y+up_radius,pol1_x-up_radius,pol1_x+up_radius))
        cent_pol2.append((pol2_y-up_radius,pol2_y+up_radius,pol2_x-up_radius,pol2_x+up_radius))

        r_frames_pol1 = mocks.Image(raster_images_pol1, pri_hdr = prihdr, ext_hdr = exthdr, err = raster_images_pol1_err)
        r_frames_pol2 = mocks.Image(raster_images_pol2, pri_hdr = prihdr, ext_hdr = exthdr, err = raster_images_pol2_err)

        raster_frames_pol1.append(r_frames_pol1)
        raster_frames_pol2.append(r_frames_pol2)
    
    raster_images_pol1_dataset=data.Dataset(raster_frames_pol1)
    raster_images_pol2_dataset=data.Dataset(raster_frames_pol2)


    resi_images_pol1_dataset=flatfield_residuals(raster_images_pol1_dataset,N=N)
    resi_images_pol2_dataset=flatfield_residuals(raster_images_pol2_dataset,N=N)

    combined_rasters_pol1=combine_flatfield_rasters(resi_images_pol1_dataset,planet=planet,band=band,cent=cent_pol1, im_size=im_size, rad_mask=rad_mask,planet_rad=planet_rad,n_pix=n_pix, n_pad=n_pad,image_center_x=512,image_center_y=512)
    combined_rasters_pol2=combine_flatfield_rasters(resi_images_pol2_dataset,planet=planet,band=band,cent=cent_pol2, im_size=im_size, rad_mask=rad_mask,planet_rad=planet_rad,n_pix=n_pix, n_pad=n_pad,image_center_x=512,image_center_y=512)
    
    if dpamname == 'POL0':
    #place image according to specified angle
        angle_rad = (alignment_angle_WP1 * np.pi) / 180
    else:
        angle_rad = (alignment_angle_WP2 * np.pi) / 180
     
    
    displacement_x = int(round((separation_diameter_arcsec * np.cos(angle_rad)) / (2 * plate_scale)))
    displacement_y = int(round((separation_diameter_arcsec * np.sin(angle_rad)) / (2 * plate_scale)))
    center_left = (image_center_x - displacement_x, image_center_y + displacement_y)
    center_right = (image_center_x + displacement_x, image_center_y - displacement_y)


    image_radius = n_pix // 2
    start_left = (center_left[0] - image_radius, center_left[1] - image_radius)
    start_right = (center_right[0] - image_radius, center_right[1] - image_radius)
    
    cent_n_pol1=combined_rasters_pol1[2]
    cent_n_pol2=combined_rasters_pol2[2]
    
    onsky_polflat=np.ones((n,n))
    onsky_polflat_error=np.zeros((n,n))

    onsky_polflat[start_left[1]:start_left[1]+n_pix, start_left[0]:start_left[0]+n_pix]=combined_rasters_pol1[0][int(cent_n_pol1[1]) - n_pix//2:int(cent_n_pol1[1]) + n_pix//2,int(cent_n_pol1[0]) - n_pix//2:int(cent_n_pol1[0]) + n_pix//2]
    onsky_polflat[start_right[1]:start_right[1]+n_pix, start_right[0]:start_right[0]+n_pix]=combined_rasters_pol2[0][int(cent_n_pol2[1])- n_pix//2:int(cent_n_pol2[1]) + n_pix//2,int(cent_n_pol2[0]) - n_pix//2:int(cent_n_pol2[0]) + n_pix//2]

    onsky_polflat_error[start_left[1]:start_left[1]+n_pix, start_left[0]:start_left[0]+n_pix]=combined_rasters_pol1[1][int(cent_n_pol1[1]) - n_pix//2:int(cent_n_pol1[1]) + n_pix//2,int(cent_n_pol1[0]) - n_pix//2:int(cent_n_pol1[0]) + n_pix//2]
    onsky_polflat_error[start_right[1]:start_right[1]+n_pix, start_right[0]:start_right[0]+n_pix]=combined_rasters_pol2[1][int(cent_n_pol2[1]) - n_pix//2:int(cent_n_pol2[1]) + n_pix//2,int(cent_n_pol2[0]) - n_pix//2:int(cent_n_pol2[0]) + n_pix//2]   
    
    
    onsky_pol_flatfield = data.FlatField(onsky_polflat, pri_hdr=prihdr, ext_hdr=exthdr, input_dataset=dataset)
    onsky_pol_flatfield.err=onsky_polflat_error
    
    return(onsky_pol_flatfield)
