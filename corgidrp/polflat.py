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
from scipy.ndimage import zoom
               
def create_onsky_pol_flatfield(dataset, planet=None,band=None,up_radius=55,im_size=None,N=1,rad_mask=None, planet_rad=None, n_pix=174, n_pad=None, fwhm_guess=20,sky_annulus_rin=2, sky_annulus_rout=4,plate_scale=0.0218,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,dpamname='POL0'):
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
            dpamname (str): wollaston prism pol0 or pol45
            
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


    resi_images_pol1_dataset=flat.flatfield_residuals(raster_images_pol1_dataset,N=N)
    resi_images_pol2_dataset=flat.flatfield_residuals(raster_images_pol2_dataset,N=N)

    combined_rasters_pol1=flat.combine_flatfield_rasters(resi_images_pol1_dataset,planet=planet,band=band,cent=cent_pol1, im_size=im_size, rad_mask=rad_mask,planet_rad=planet_rad,n_pix=n_pix, n_pad=n_pad,image_center_x=512,image_center_y=512)
    combined_rasters_pol2=flat.combine_flatfield_rasters(resi_images_pol2_dataset,planet=planet,band=band,cent=cent_pol2, im_size=im_size, rad_mask=rad_mask,planet_rad=planet_rad,n_pix=n_pix, n_pad=n_pad,image_center_x=512,image_center_y=512)
    
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
       
    
    
