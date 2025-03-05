# A file that holds the functions that transmogrify l3 data to l4 data 

from pyklip.klip import rotate
from corgidrp import data
from corgidrp.detector import flag_nans,nan_flags
from scipy.ndimage import rotate as rotate_scipy # to avoid duplicated name
from scipy.ndimage import shift
import warnings
import numpy as np
import glob
import pyklip.rdi
import os
from astropy.io import fits
import warnings

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
            in ext_hdr["STARLOCX/Y"]
    """

    return input_dataset.copy()

def crop(input_dataset,sizexy=None,centerxy=None):
    """
    
    Crop the Images in a Dataset to a desired field of view. Default behavior is to 
    crop the image to the dark hole region, centered on the pixel intersection nearest 
    to the star location. Assumes 3D Image data is a stack of 2D data arrays, so only 
    crops the last two indices. Currently only configured for HLC mode.

    TODO: 
        - Pad with nans if you try to crop outside the array (handle err & DQ too)
        - Option to crop to an odd data array and center on a pixel?

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (any level)
        sizexy (int or array of int): desired frame size, if only one number is provided the 
            desired shape is assumed to be square, otherwise xy order. If not provided, 
            defaults to 60 for NFOV (narrow field-of-view) observations. Defaults to None.
        centerxy (float or array of float): desired center (xy order), should be a pixel intersection (a.k.a 
            half-integer) otherwise the function rounds to the nearest intersection. Defaults to the 
            "STARLOCX/Y" header values.

    Returns:
        corgidrp.data.Dataset: a version of the input dataset cropped to the desired FOV.
    """

    # Copy input dataset
    dataset = input_dataset.copy()

    # Require even data shape
    if not sizexy is None and not np.all(np.array(sizexy)%2==0):
        raise UserWarning('Even sizexy is required.')
       
    # Need to loop over frames and reinit dataset because array sizes change
    frames_out = []

    for frame in dataset:
        prihdr = frame.pri_hdr
        exthdr = frame.ext_hdr
        dqhdr = frame.dq_hdr
        errhdr = frame.err_hdr

        # Pick default crop size based on the size of the effective field of view
        if sizexy is None:
            if exthdr['LSAMNAME'] == 'NFOV':
                sizexy = 60
            else:
                raise UserWarning('Crop function is currently only configured for NFOV (narrow field-of-view) observations if sizexy is not provided.')

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
        errhdr["NAXIS3"] = cropped_frame_err.shape[-3]
        if frame.data.ndim == 3:
            exthdr["NAXIS3"] = frame.data.shape[0]
            dqhdr["NAXIS3"] = frame.dq.shape[0]
            errhdr["NAXIS4"] = frame.err.shape[0]
        
        updated_hdrs = []
        if ("STARLOCX" in exthdr.keys()):
            exthdr["STARLOCX"] -= x1
            exthdr["STARLOCY"] -= y1
            updated_hdrs.append('STARLOCX/Y')
        if ("MASKLOCX" in exthdr.keys()):
            exthdr["MASKLOCX"] -= x1
            exthdr["MASKLOCY"] -= y1
            updated_hdrs.append('MASKLOCX/Y')
        if ("CRPIX1" in prihdr.keys()):
            prihdr["CRPIX1"] -= x1
            prihdr["CRPIX2"] -= y1
            updated_hdrs.append('CRPIX1/2')
        new_frame = data.Image(cropped_frame_data,prihdr,exthdr,cropped_frame_err,cropped_frame_dq,frame.err_hdr,frame.dq_hdr)
        frames_out.append(new_frame)

    output_dataset = data.Dataset(frames_out)

    history_msg1 = f"""Frames cropped to new shape {list(output_dataset[0].data.shape)} on center {list(centerxy)}. Updated header kws: {", ".join(updated_hdrs)}."""
    output_dataset.update_after_processing_step(history_msg1)
    
    return output_dataset

def do_psf_subtraction(input_dataset, reference_star_dataset=None,
                       mode=None, annuli=1,subsections=1,movement=1,
                       numbasis=[1,4,8,16],outdir='KLIP_SUB',fileprefix="",
                       do_crop=True,
                       crop_sizexy=None
                       ):
    """
    
    Perform PSF subtraction on the dataset. Optionally using a reference star dataset.
    TODO: 
        Handle nans & propagate DQ array
        What info is missing from output dataset headers?
        Add comments to new ext header cards
        
    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)
        reference_star_dataset (corgidrp.data.Dataset, optional): a dataset of Images of the reference 
            star [optional]
        mode (str, optional): pyKLIP PSF subraction mode, e.g. ADI/RDI/ADI+RDI. Mode will be chosen autonomously 
            if not specified.
        annuli (int, optional): number of concentric annuli to run separate subtractions on. Defaults to 1.
        subsections (int, optional): number of angular subsections to run separate subtractions on. Defaults to 1.
        movement (int, optional): KLIP movement parameter. Defaults to 1.
        numbasis (int or list of int, optional): number of KLIP modes to retain. Defaults to [1,4,8,16].
        outdir (str or path, optional): path to output directory. Defaults to "KLIP_SUB".
        fileprefix (str, optional): prefix of saved output files. Defaults to "".
        do_crop (bool): whether to crop data before PSF subtraction. Defaults to True.
        crop_sizexy (list of int, optional): Desired size to crop the images to before PSF subtraction. Defaults to 
            None, which results in the step choosing a crop size based on the imaging mode. 

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the PSF subtraction applied (L4-level)

    """

    sci_dataset = input_dataset.copy()
    
    # Use input reference dataset if provided
    if not reference_star_dataset is None:
        ref_dataset = reference_star_dataset.copy()

    # Try getting PSF references via the "PSFREF" header kw
    else:
        split_datasets, unique_vals = sci_dataset.split_dataset(prihdr_keywords=["PSFREF"])
        unique_vals = np.array(unique_vals)

        if 0. in unique_vals:
            sci_dataset = split_datasets[int(np.nonzero(np.array(unique_vals) == 0)[0])]
        else:
            raise UserWarning('No science files found in input dataset.')

        if 1. in unique_vals:
            ref_dataset = split_datasets[int(np.nonzero(np.array(unique_vals) == 1)[0])]
        else:
            ref_dataset = None

    assert len(sci_dataset) > 0, "Science dataset has no data."

    # Choose PSF subtraction mode if unspecified
    if mode is None:
        
        if not ref_dataset is None and len(sci_dataset)==1:
            mode = 'RDI' 
        elif not ref_dataset is None:
            mode = 'ADI+RDI'
        else:
            mode = 'ADI' 

    else: assert mode in ['RDI','ADI+RDI','ADI'], f"Mode {mode} is not configured."

    # Format numbases
    if isinstance(numbasis,int):
        numbasis = [numbasis]

    # Set up outdir
    outdir = os.path.join(outdir,mode)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Crop data
    if do_crop:
        sci_dataset = crop(sci_dataset,sizexy=crop_sizexy)
        ref_dataset = None if ref_dataset is None else crop(ref_dataset,sizexy=crop_sizexy) 

    # Mask data where DQ > 0, let pyklip deal with the nans
    sci_dataset_masked = nan_flags(sci_dataset)
    ref_dataset_masked = None if ref_dataset is None else nan_flags(ref_dataset)

    # Run pyklip
    pyklip_dataset = data.PyKLIPDataset(sci_dataset_masked,psflib_dataset=ref_dataset_masked)
    pyklip.parallelized.klip_dataset(pyklip_dataset, outputdir=outdir,
                              annuli=annuli, subsections=subsections, movement=movement, numbasis=numbasis,
                              calibrate_flux=False, mode=mode,psf_library=pyklip_dataset._psflib,
                              fileprefix=fileprefix)
    
    # Construct corgiDRP dataset from pyKLIP result
    result_fpath = os.path.join(outdir,f'{fileprefix}-KLmodes-all.fits')   
    pyklip_data = fits.getdata(result_fpath)
    pyklip_hdr = fits.getheader(result_fpath)

    # TODO: Handle DQ & errors correctly
    err = np.zeros([1,*pyklip_data.shape])
    dq = np.zeros_like(pyklip_data) # This will get filled out later

    # Clean up primary header
    pri_hdr = pyklip_hdr.copy()
    naxis1 = pri_hdr['NAXIS1']
    naxis2 = pri_hdr['NAXIS2']
    naxis3 = pri_hdr['NAXIS3']
    pri_hdr['NAXIS'] = 0

    remove_kws = ['NAXIS1','NAXIS2','NAXIS3',
                    'CRPIX1','CRPIX2']
    for kw in remove_kws:
        del pri_hdr[kw]

    # Add observation info from input dataset
    pri_hdr_keys = ['TELESCOP','INSTRUME']
    for kw in pri_hdr_keys:
        pri_hdr[kw] = sci_dataset[0].pri_hdr[kw]
    
    # Make extension header
    ext_hdr = fits.Header()
    ext_hdr['NAXIS'] = 2
    ext_hdr['NAXIS1'] = naxis1
    ext_hdr['NAXIS2'] = naxis2
    ext_hdr['NAXIS3'] = naxis2

    # Add info from input dataset
    ext_hdr_keys = ['BUNIT','PLTSCALE','CFAMNAME',
                    'DPAMNAME','FPAMNAME','FSAMNAME',
                    'LSAMNAME','SPAMNAME']
    for kw in ext_hdr_keys:
        ext_hdr[kw] = sci_dataset[0].ext_hdr[kw]

    # Add info from pyklip
    ext_hdr['KLIP_ALG'] = mode
    #ext_hdr['KLMODES'] = pyklip_hdr[f'KLMODE{i}']
    ext_hdr['STARLOCX'] = pyklip_hdr['PSFCENTX']
    ext_hdr['STARLOCY'] = pyklip_hdr['PSFCENTY']
    if "HISTORY" in sci_dataset[0].ext_hdr.keys():
        history_str = str(sci_dataset[0].ext_hdr['HISTORY'])
        ext_hdr['HISTORY'] = ''.join(history_str.split('\n'))
    
    # Construct Image object and add to list
    frame = data.Image(pyklip_data,
                        pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                        err=err, dq=dq)
    
    dataset_out = data.Dataset([frame])

    # Flag nans in the dq array and then add nans to the error array
    dataset_out = flag_nans(dataset_out,flag_val=1)
    dataset_out = nan_flags(dataset_out,threshold=1)
    
    history_msg = f'PSF subtracted via pyKLIP {mode}.'
    
    dataset_out.update_after_processing_step(history_msg)
    
    return dataset_out

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
            xcen, ycen = im_hd['STARLOCX'], im_hd['STARLOCY'] 
        except KeyError:
            warnings.warn('"STARLOCX/Y" missing from ext_hdr. Rotating about center of array.')
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
