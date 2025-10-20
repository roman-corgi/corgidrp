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
import corgidrp.mocks as mocks
import corgidrp.data as data
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import corgidrp.flat as flat
from skimage.transform import downscale_local_mean

                                     
def create_spatial_pol(dataset,nr=60,pfov_size=174,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,planet=None,band=None,prism='POL0'):
    """Turns a dataset of neptune or uranus images with single planet images into the images observed through the wollaston prisms also incorporates the spatial variation of polarization on the 
        surface of the planet

    
        Args:
            dataset (corgidrp.data.Dataset): a dataset of image frames that are raster scanned (L2a-level)
            nr (int): planet radius
            pfov_size (int): size of the image created for the polarization variation, typically ~ 2 x nr
            image_center_x (int): x coordinate of the center pixel of the final image with two orthogonal pol components
            image_center_y (int): y coordinate of the center pixel of the final image with two orthogonal pol components
            separation_diameter_arcsec (float): separation between the two orthogonal pol components in arcsecs
            alignment_angle_WP1 (int): wollaston prism angle for Pol0 - 0
            alignment_angle_WP2 (int): wollaston prism angle for Pol45- 45
            planet (str): neptune or uranus
            band (str): 1 or 4
            prism (str): wollaston prism pol0 or pol45
            
    	Returns:
    		data.Dataset: dataset of uranus or neptune with spatial variation of polarization corresponding to specific wollaston prism
    		
	"""
    pfov = pfov_size
    polar_fov = np.ones((pfov,pfov))

     # default to unvignetted polarimetry FOV diameter of 3.8"
    radius_arcsec = 3.8 / 2
    # convert to pixel: 0.0218" = 1 pixel
    radius_pix = int(round(radius_arcsec / 0.0218))

    x = np.arange(0,pfov)
    y = np.arange(0,pfov)
    xx, yy = np.meshgrid(x,y)
    nrr = np.sqrt((xx-(pfov//2))**2 + (yy-(pfov//2))**2)

    polar_filter = np.logical_and(yy<.99*xx,yy<.99*(pfov-xx)) + np.logical_and(yy>.99*xx,yy>.99*(pfov-xx))
    
    for i in range(len(dataset)):
        target=dataset[i].pri_hdr['TARGET']
        filter=dataset[i].pri_hdr['FILTER']
        if planet==target and band==filter: 
            planet_image=dataset[i].data
    
    if planet == 'uranus' and band=="1":
        polar_fov[polar_filter==True] = .0115
        polar_fov[polar_filter==False] = -0.0115
    elif planet == 'uranus' and band=="4" or planet == 'neptune' and band=="4":
        polar_fov[polar_filter==True] = .005
        polar_fov[polar_filter==False] = -0.005
    elif planet == 'neptune' and band=="1":
        polar_fov[polar_filter==True] = .006
        polar_fov[polar_filter==False] = -0.006
    r_xy = polar_fov
    
    n_rad=nr
    if n_rad is None:
        if planet.lower() =='neptune':
             n_rad = 60
        elif planet.lower() == 'uranus':
             n_rad = 95
    
    r_xy[nrr>=n_rad] = 0
    
   
    u_data=planet_image
    I_1 = u_data.copy()
    I_2 = u_data.copy()
    centroid_init = centr.centroid_1dg(u_data)
    xc_init=int(centroid_init[0])
    yc_init=int(centroid_init[1])


    if prism == 'POL0':
    #place image according to specified angle
        angle_rad = (alignment_angle_WP1 * np.pi) / 180
    else:
        angle_rad = (alignment_angle_WP2 * np.pi) / 180
     
    
    displacement_x = int(round((separation_diameter_arcsec * np.cos(angle_rad)) / (2 * 0.0218)))
    displacement_y = int(round((separation_diameter_arcsec * np.sin(angle_rad)) / (2 * 0.0218)))
    center_left = (image_center_x - displacement_x, image_center_y + displacement_y)
    center_right = (image_center_x + displacement_x, image_center_y - displacement_y)

    WP_image=np.random.poisson(lam=0.199, size=(1024, 1024)).astype(np.float64)
    
    image_radius = pfov_size // 2
    start_left = (center_left[0] - image_radius, center_left[1] - image_radius)
    start_right = (center_right[0] - image_radius, center_right[1] - image_radius)

    y, x = np.indices([np.shape(WP_image)[0], np.shape(WP_image)[1]])
    x_left = x + start_left[0]
    y_left = y + start_left[1]
    x_right = x + start_right[0]
    y_right = y + start_right[1]
    
    WP_pol=WP_image.copy() 
    WP_pol[start_left[1]:start_left[1]+pfov, start_left[0]:start_left[0]+pfov]=I_1[yc_init - (pfov//2):yc_init + (pfov//2),xc_init - (pfov//2):xc_init + (pfov//2)]* 0.5 * (1+(2*r_xy)) 
    WP_pol[start_right[1]:start_right[1]+pfov, start_right[0]:start_right[0]+pfov]=I_2[yc_init - (pfov//2):yc_init + (pfov//2),xc_init - (pfov//2):xc_init + (pfov//2)]* 0.5 * (1-(2*r_xy))


    prihdr, exthdr = mocks.create_default_L1_headers()
    frame = data.Image(WP_pol, pri_hdr=prihdr, ext_hdr=exthdr)
    frame.pri_hdr.set('TARGET', planet)
    frame.pri_hdr.set('FILTER',band)
    polmap = data.Dataset([frame])
        
    return (polmap)  
               
def combine_pol_flatfield_rasters(residual_images,residual_images_err,cent=None,planet=None,band=None,im_size=420,rad_mask=None, planet_rad=None, n_pix=174, n_pad=302,image_center_x=512,image_center_y=512):
    """combine the dataset of residual image frames of neptune or uranus and create flat field 
    	and associated error

    	Args:
        	residual_images (np.array): Residual images frames divided by the mnatched filter of neptune and uranus
            residual_images_err (np.array): Error from the Residual images frames divided by the mnatched filter of neptune and uranus
            cent (np.array): centroid of the image frames
        	planet (str):   name of the planet neptune or uranus
        	band (str):  band of the observation band1 or band4
            im_size (int): x-dimension of the planet image 
            rad_mask (float): radius in pixels used for creating a mask for band (band 1 =1.25, band 4=1.75)
            planet_rad (int): radius of the planet in pixels (planet_rad=54 for neptune, planet_rad=65)
            n_pix (int): Number of pixels in radius covering the Roman CGI imaging FOV defaults to 165 pixels
            n_pad (int): Number of pixels padded with '1s'  to generate the image size 1024X1024 pixels around imaging FOV (defaults to 302 pixels)
            image_center_x (int): x coordinate of the center pixel of the final image with two orthogonal pol components
            image_center_y (int): y coordinate of the center pixel of the final image with two orthogonal pol components
	
    	Returns:
        	full_residuals (np.array): flat field created from uranus or neptune images
        	err_residuals (np.array):  Error in the flatfield estimated using the ideal flat field
            cent_n (np.array): centroids of the residual images
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
    for i in np.arange(len(residual_images)):
        nx = np.arange(0,residual_images[i].shape[1])
        ny = np.arange(0,residual_images[i].shape[0])
        nxx,nyy = np.meshgrid(nx,ny)
        nrr = np.sqrt((nxx-rad-5)**2 + (nyy-rad-5)**2)
       
        nrr_copy = nrr.copy();  
        nrr_err_copy=nrr.copy()
    
        nrr_copy[nrr<rad] = residual_images[i][nrr<rad]
        nrr_err_copy[nrr<rad] = residual_images_err[i][nrr<rad]
    
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
    cent_n=centr.centroid_com(p_flat)
    nrr = np.sqrt((nxx-cent_n[0])**2 + (nyy-cent_n[1])**2)
    p_flat[nrr>n_pix//2]= 1
    p_flat_err[nrr>n_pix//2]=0
    
    full_residuals=np.pad(p_flat, ((n_pad,n_pad),(n_pad,n_pad)), mode='constant',constant_values=(1))
    err_residuals=np.pad(p_flat_err, ((n_pad,n_pad),(n_pad,n_pad)), mode='constant',constant_values=(0))
    
    return (full_residuals,err_residuals,cent_n)


def create_onsky_pol_flatfield(dataset, planet=None,band=None,up_radius=55,im_size=None,N=1,rad_mask=None, planet_rad=None, n_pix=174, n_pad=None, sky_annulus_rin=2, sky_annulus_rout=4,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,prism='POL0'):
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
            image_center_x (int): x coordinate of the center pixel of the final image with two orthogonal pol components
            image_center_y (int): y coordinate of the center pixel of the final image with two orthogonal pol components
            separation_diameter_arcsec (float): separation between the two orthogonal pol components in arcsecs
            alignment_angle_WP1 (int): wollaston prism angle for Pol0 - 0
            alignment_angle_WP2 (int): wollaston prism angle for Pol45- 45
            prism (str): wollaston prism pol0 or pol45
            
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

    smooth_images=[]; raster_images_cent_pol1=[]; cent_pol1=[]; act_cents_1=[]; act_cents_2=[];raster_images_cent_pol2=[];cent_pol2=[];
    for j in range(len(dataset)):
        planet_image=dataset[j].data
        prihdr=dataset[j].pri_hdr
        exthdr=dataset[j].ext_hdr
        planet_image_downsampled = downscale_local_mean(planet_image, (10,10))
        fwhm_guess = 10.0
        threshold_value = np.max(planet_image)
        daofind = DAOStarFinder(fwhm=fwhm_guess, threshold=threshold_value, min_separation=10.0)
        sources = daofind(planet_image_downsampled)

        if sources is not None:

            x_centroids = sources['xcentroid']*10
            y_centroids = sources['ycentroid']*10

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

        smooth_images.append(planet_image)
        # cropping the raster scanned images
        raster_images_cent_pol1.append(smooth_images[j][pol1_y-up_radius:pol1_y+up_radius,pol1_x-up_radius:pol1_x+up_radius])
        raster_images_cent_pol2.append(smooth_images[j][pol2_y-up_radius:pol2_y+up_radius,pol2_x-up_radius:pol2_x+up_radius])

        #centroid of the cropped images
        cent_pol1.append((pol1_y-up_radius,pol1_y+up_radius,pol1_x-up_radius,pol1_x+up_radius))
        cent_pol2.append((pol2_y-up_radius,pol2_y+up_radius,pol2_x-up_radius,pol2_x+up_radius))

    resi_images_pol1=flat.flatfield_residuals(raster_images_cent_pol1,N=N)[0]
    resi_images_pol2=flat.flatfield_residuals(raster_images_cent_pol2,N=N)[0]

    resi_images_pol1_err=flat.flatfield_residuals(raster_images_cent_pol1,N=N)[1]
    resi_images_pol2_err=flat.flatfield_residuals(raster_images_cent_pol2,N=N)[1]

    raster_com_pol1=combine_pol_flatfield_rasters(resi_images_pol1,resi_images_pol1_err,planet=planet,band=band,cent=cent_pol1, im_size=im_size, rad_mask=rad_mask,planet_rad=planet_rad,n_pix=n_pix, n_pad=n_pad,image_center_x=512,image_center_y=512)
    raster_com_pol2=combine_pol_flatfield_rasters(resi_images_pol2,resi_images_pol2_err,planet=planet,band=band,cent=cent_pol2, im_size=im_size, rad_mask=rad_mask,planet_rad=planet_rad,n_pix=n_pix, n_pad=n_pad,image_center_x=512,image_center_y=512)
    
    if prism == 'POL0':
    #place image according to specified angle
        angle_rad = (alignment_angle_WP1 * np.pi) / 180
    else:
        angle_rad = (alignment_angle_WP2 * np.pi) / 180
     
    
    displacement_x = int(round((separation_diameter_arcsec * np.cos(angle_rad)) / (2 * 0.0218)))
    displacement_y = int(round((separation_diameter_arcsec * np.sin(angle_rad)) / (2 * 0.0218)))
    center_left = (image_center_x - displacement_x, image_center_y + displacement_y)
    center_right = (image_center_x + displacement_x, image_center_y - displacement_y)


    image_radius = n_pix // 2
    start_left = (center_left[0] - image_radius, center_left[1] - image_radius)
    start_right = (center_right[0] - image_radius, center_right[1] - image_radius)
    
    cent_n_pol1=raster_com_pol1[2]
    cent_n_pol2=raster_com_pol2[2]
    
    onsky_polflat=np.ones((n,n))
    onsky_polflat_error=np.zeros((n,n))

    onsky_polflat[start_left[1]:start_left[1]+n_pix, start_left[0]:start_left[0]+n_pix]=raster_com_pol1[0][int(cent_n_pol1[1]) - n_pix//2:int(cent_n_pol1[1]) + n_pix//2,int(cent_n_pol1[0]) - n_pix//2:int(cent_n_pol1[0]) + n_pix//2]
    onsky_polflat[start_right[1]:start_right[1]+n_pix, start_right[0]:start_right[0]+n_pix]=raster_com_pol2[0][int(cent_n_pol2[1])- n_pix//2:int(cent_n_pol2[1]) + n_pix//2,int(cent_n_pol2[0]) - n_pix//2:int(cent_n_pol2[0]) + n_pix//2]

    onsky_polflat_error[start_left[1]:start_left[1]+n_pix, start_left[0]:start_left[0]+n_pix]=raster_com_pol1[1][int(cent_n_pol1[1]) - n_pix//2:int(cent_n_pol1[1]) + n_pix//2,int(cent_n_pol1[0]) - n_pix//2:int(cent_n_pol1[0]) + n_pix//2]
    onsky_polflat_error[start_right[1]:start_right[1]+n_pix, start_right[0]:start_right[0]+n_pix]=raster_com_pol2[1][int(cent_n_pol2[1]) - n_pix//2:int(cent_n_pol2[1]) + n_pix//2,int(cent_n_pol2[0]) - n_pix//2:int(cent_n_pol2[0]) + n_pix//2]   
    
    
    onsky_pol_flatfield = data.FlatField(onsky_polflat, pri_hdr=prihdr, ext_hdr=exthdr, input_dataset=dataset)
    onsky_pol_flatfield.err=onsky_polflat_error
    
    return(onsky_pol_flatfield)
       
    
    
