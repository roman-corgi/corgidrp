# A file that holds the functions that transmogrify l3 data to l4 data 

from pyklip.klip import rotate
import scipy.ndimage
from corgidrp import data
from scipy.ndimage import rotate as rotate_scipy # to avoid duplicated name
from scipy.ndimage import shift
import numpy as np
import glob

def distortion_correction(input_dataset, astrom_calibration):
    """
    
    Apply the distortion correction to the dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)
        astrom_calibration (corgidrp.data.AstrometricCalibration): an AstrometricCalibration calibration file to model the distortion

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the distortion correction applied
    """
    undistorted_dataset = input_dataset.copy()
    distortion_coeffs = astrom_calibration.distortion_coeffs[0]
    distortion_order = int(astrom_calibration.distortion_coeffs[1])

    undistorted_ims = []

    # apply the distortion correction to each image in the dataset
    for undistorted_data in undistorted_dataset:

        im_data = undistorted_data.data
        imgsizeX, imgsizeY = im_data.shape

        # set image size to the largest axis if not square imagea
        if (imgsizeX >= imgsizeY): imgsize = imgsizeX
        else: imgsize = imgsizeY

        yorig, xorig = np.indices(im_data.shape)
        y0, x0 = imgsize//2, imgsize//2
        yorig -= y0
        xorig -= x0

        ### compute the distortion map based on the calibration file passed in
        fitparams = (distortion_order + 1)**2

            # reshape the coefficient arrays
        x_params = distortion_coeffs[:fitparams]
        x_params = x_params.reshape(distortion_order+1, distortion_order+1)

        total_orders = np.arange(distortion_order+1)[:,None] + np.arange(distortion_order+1)[None,:]
        x_params = x_params / 500**(total_orders)

            # evaluate the legendre polynomial at all pixel positions
        x_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), x_params)
        x_corr = x_corr.reshape(xorig.shape)

        distmapX = x_corr - xorig

            # reshape and evaluate the same way for the y coordinates
        y_params = distortion_coeffs[fitparams:]
        y_params = y_params.reshape(distortion_order+1, distortion_order+1)
        y_params = y_params /500**(total_orders)

        y_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), y_params)
        y_corr = y_corr.reshape(yorig.shape)
        distmapY = y_corr - yorig

        # apply the distortion grid to the image indeces and map the image
        gridx, gridy = np.meshgrid(np.arange(imgsize), np.arange(imgsize))
        gridx = gridx - distmapX
        gridy = gridy - distmapY

        undistorted_image = scipy.ndimage.map_coordinates(im_data, [gridy, gridx])

        undistorted_ims.append(undistorted_image)

    history_msg = 'Distortion correction completed'

    undistorted_dataset.update_after_processing_step(history_msg, new_all_data=np.array(undistorted_ims))

    return undistorted_dataset

def find_star(input_dataset):
    """
    
    Find the star position in each Image in the dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the stars identified
    """

    return input_dataset.copy()

def do_psf_subtraction(input_dataset, reference_star_dataset=None):
    """
    
    Perform PSF subtraction on the dataset. Optionally using a reference star dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)
        reference_star_dataset (corgidrp.data.Dataset): a dataset of Images of the reference star [optional]

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the PSF subtraction applied
    """

    return input_dataset.copy()

def northup(input_dataset,correct_wcs=False):
    """
    Derotate the Image, ERR, and DQ data by the angle offset to make the FoV up to North. 
    Now tentatively assuming 'ROLL' in the primary header incorporates all the angle offset, and the center of the FoV is the star position.
    WCS correction is not yet implemented - TBD.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)
	correct_wcs: if you want to correct WCS solutions after rotation, set True. Now hardcoded with not using astr_hdr.

    Returns:
        corgidrp.data.Dataset: North is up, East is left
    
    """
    
    # make a copy 
    processed_dataset = input_dataset.copy()

    new_all_data = []; new_all_err = []; new_all_dq = []
    for processed_data in processed_dataset:
        # read the roll angle parameter, assuming this info is recorded in the primary header as requested
        roll_angle = processed_data.pri_hdr['ROLL']

        ## image extension ##
        im_hd = processed_data.ext_hdr
        im_data = processed_data.data
        ylen, xlen = im_data.shape

        # define the center for rotation
        try: 
            xcen, ycen = im_hd['PSFCENTX'], im_hd['PSFCENTY'] # TBU, after concluding the header keyword
        except KeyError:
            xcen, ycen = xlen/2, ylen/2
    
        # look for WCS solutions
        if correct_wcs is False: 
            astr_hdr = None 
        else:
            astr_hdr = None # hardcoded now, no WCS information in the header

        # derotate
        im_derot = rotate(im_data,-roll_angle,(xcen,ycen),astr_hdr=astr_hdr)
        new_all_data.append(im_derot)
        ##############

        ## HDU ERR ##
        err_data = processed_data.err
        err_derot = np.expand_dims(rotate(err_data[0],-roll_angle,(xcen,ycen)), axis=0) # err data shape is 1x1024x1024
        new_all_err.append(err_derot)
        #############

        ## HDU DQ ##
	# all DQ pixels must have integers, use scipy.ndimage.rotate with order=0 instead of klip.rotate (rotating the other way)
        dq_data = processed_data.dq
        if xcen != xlen/2 or ycen != ylen/2: 
                # padding, shifting (rot center to image center), rotating, re-shift (image center to rot center), and cropping
                # calculate shift values
                xshift = xcen-xlen/2; yshift = ycen-ylen/2
		
                # pad and shift
                pad_x = int(np.ceil(abs(xshift))); pad_y = int(np.ceil(abs(yshift)))
                dq_data_padded = np.pad(dq_data,pad_width=((pad_y, pad_y), (pad_x, pad_x)),mode='constant',constant_values=np.nan)
                dq_data_padded_shifted = shift(dq_data_padded,(-yshift,-xshift),order=0,mode='constant',cval=np.nan)

                # define slices for cropping
                crop_x = slice(pad_x,pad_x+xlen); crop_y = slice(pad_y,pad_y+ylen)

                # rotate, re-shift, and crop
                dq_derot = shift(rotate_scipy(dq_data_padded_shifted, roll_angle, order=0, mode='constant', reshape=False, cval=np.nan),\
                 (yshift,xshift),order=0,mode='constant',cval=np.nan)[crop_y,crop_x]
        else: 
                # simply rotate 
                dq_derot = rotate_scipy(dq_data, roll_angle, order=0, mode='constant', reshape=False, cval=np.nan)
        	
        new_all_dq.append(dq_derot)
        ############

    hisotry_msg = f'FoV rotated by {-roll_angle}deg counterclockwise at a roll center {xcen, ycen}'
    
    processed_dataset.update_after_processing_step(hisotry_msg, new_all_data=np.array(new_all_data), new_all_err=np.array(new_all_err),\
                                                   new_all_dq=np.array(new_all_dq))
    
    return processed_dataset

def update_to_l4(input_dataset):
    """
    Updates the data level to L4. Only works on L3 data.

    Currently only checks that data is at the L3 level

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)

    Returns:
        corgidrp.data.Dataset: same dataset now at L4-level
    """
    # check that we are running this on L1 data
    for orig_frame in input_dataset:
        if orig_frame.ext_hdr['DATA_LEVEL'] != "L3":
            err_msg = "{0} needs to be L3 data, but it is {1} data instead".format(orig_frame.filename, orig_frame.ext_hdr['DATA_LEVEL'])
            raise ValueError(err_msg)

    # we aren't altering the data
    updated_dataset = input_dataset.copy(copy_data=False)

    for frame in updated_dataset:
        # update header
        frame.ext_hdr['DATA_LEVEL'] = "L4"
        # update filename convention. The file convention should be
        # "CGI_[dataleel_*]" so we should be same just replacing the just instance of L1
        frame.filename = frame.filename.replace("_L3_", "_L4_", 1)

    history_msg = "Updated Data Level to L4"
    updated_dataset.update_after_processing_step(history_msg)

    return updated_dataset
