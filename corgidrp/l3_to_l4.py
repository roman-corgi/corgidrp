# A file that holds the functions that transmogrify l3 data to l4 data 

from pyklip.klip import rotate
from corgidrp import data
from scipy.ndimage import rotate as rotate_scipy # to avoid duplicated name
from scipy.ndimage import shift
import warnings
import numpy as np
import glob

def distortion_correction(input_dataset, distortion_calibration):
    """
    
    Apply the distortion correction to the dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)
        distortion_calibration (corgidrp.data.DistortionCalibration): a DistortionCalibration calibration file to model the distortion

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the distortion correction applied
    """

    return input_dataset.copy()

def find_star(input_dataset):
    """
    
    Find the star position in each Image in the dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the stars identified
    """

    return input_dataset.copy()

def crop(input_dataset,sizexy=60,centerxy=None):
    """
    
    Crop the Images in a Dataset to a desired field of view. Default behavior is to 
    crop the image to the dark hole region, centered on the pixel intersection nearest 
    to the star location. Assumes 3D Image data is a stack of 2D data arrays, so only 
    crops the last two indices.

    TODO: 
        - Pad with nans if you try to crop outside the array (handle err & DQ too)
        - Option to crop to an odd data array and center on a pixel?

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (any level)
        sizexy (int or array of int): desired frame size, if only one number is provided the 
            desired shape is assumed to be square, otherwise xy order. Defaults to 60.
        centerxy (float or array of float): desired center (xy order), should be a pixel intersection (a.k.a 
            half-integer). Defaults to the "STARLOCX/Y" header values.

    Returns:
        corgidrp.data.Dataset: a version of the input dataset cropped to the desired FOV.
    """

    # Copy input dataset
    dataset = input_dataset.copy()

    # Require even data shape
    if not np.all(np.array(sizexy)%2==0):
        raise UserWarning('Even sizexy is required.')
    
    # Need to loop over frames and reinit dataset because array sizes change
    frames_out = []

    for frame in dataset:
        prihdr = frame.pri_hdr
        exthdr = frame.ext_hdr
        dqhdr = frame.dq_hdr
        errhdr = frame.err_hdr

        # Assign new array sizes and center location
        frame_shape = frame.data.shape
        if isinstance(sizexy,int):
            sizexy = [sizexy]*2
        if isinstance(centerxy,float):
            centerxy = [centerxy] * 2
        elif centerxy is None:
            if ("STARLOCX" in exthdr.keys()) and ("STARLOCY" in exthdr.keys()):
                centerxy = np.array([exthdr["STARLOCX"],exthdr["STARLOCY"]])
            else: raise ValueError('centerxy not provided but STARLOCX/Y are missing from image extension header.')
        # Round to centerxy to nearest half-pixel
        centerxy = np.array(centerxy)
        if not np.all((centerxy-0.5)%1 == 0):
            old_centerxy = centerxy.copy()
            centerxy = np.round(old_centerxy-0.5)+0.5
            warnings.warn(f'Desired center {old_centerxy} is not at the intersection of 4 pixels. Centering on the nearest intersection {centerxy}')
            
        # Crop the data
        start_ind = (centerxy + 0.5 - np.array(sizexy)/2).astype(int)
        end_ind = (centerxy + 0.5 + np.array(sizexy)/2).astype(int)
        x1,y1 = start_ind
        x2,y2 = end_ind

        # Check if cropping outside the FOV
        xleft_pad = -x1 if (x1<0) else 0
        xrright_pad = x2-frame_shape[-1]+1 if (x2 > frame_shape[-1]) else 0
        ybelow_pad = -y1 if (y1<0) else 0
        yabove_pad = y2-frame_shape[-2]+1 if (y2 > frame_shape[-2]) else 0
        
        if np.any(np.array([xleft_pad,xrright_pad,ybelow_pad,yabove_pad])> 0) :
            raise ValueError("Trying to crop to a region outside the input data array. Not yet configured.")

        if frame.data.ndim == 2:
            cropped_frame_data = frame.data[y1:y2,x1:x2]
            cropped_frame_err = frame.err[:,y1:y2,x1:x2]
            cropped_frame_dq = frame.dq[y1:y2,x1:x2]
        elif frame.data.ndim == 3:
            cropped_frame_data = frame.data[:,y1:y2,x1:x2]
            cropped_frame_err = frame.err[:,:,y1:y2,x1:x2]
            cropped_frame_dq = frame.dq[:,y1:y2,x1:x2]
        else:
            raise ValueError('Crop function only supports 2D or 3D frame data.')

        # Update headers
        exthdr["NAXIS1"] = sizexy[0]
        exthdr["NAXIS2"] = sizexy[1]
        dqhdr["NAXIS1"] = sizexy[0]
        dqhdr["NAXIS2"] = sizexy[1]
        errhdr["NAXIS1"] = sizexy[0]
        errhdr["NAXIS2"] = sizexy[1]
        errhdr["NAXIS3"] = cropped_frame_err.shape[0]
        if ("STARLOCX" in exthdr.keys()):
            exthdr["STARLOCX"] -= x1
            exthdr["STARLOCY"] -= y1
        if ("MASKLOCX" in exthdr.keys()):
            exthdr["MASKLOCX"] -= x1
            exthdr["MASKLOCY"] -= y1
        if ("CRPIX1" in prihdr.keys()):
            prihdr["CRPIX1"] -= x1
            prihdr["CRPIX2"] -= y1
        if ("PSFCENTX" in prihdr.keys()):
            prihdr["PSFCENTX"] -= x1
            prihdr["PSFCENTY"] -= y1

        new_frame = data.Image(cropped_frame_data,prihdr,exthdr,cropped_frame_err,cropped_frame_dq,frame.err_hdr,frame.dq_hdr)
        frames_out.append(new_frame)

    output_dataset = data.Dataset(frames_out)
    
    return output_dataset

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
