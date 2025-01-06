# A file that holds the functions that transmogrify l3 data to l4 data 

from pyklip.klip import rotate
from corgidrp import data
from corgidrp.detector import flag_nans,nan_flags
from scipy.ndimage import rotate as rotate_scipy # to avoid duplicated name
from scipy.ndimage import shift
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

def do_psf_subtraction(input_dataset, reference_star_dataset=None,
                       mode=None, annuli=1,subsections=1,movement=1,
                       numbasis = [1,4,8,16],outdir='KLIP_SUB',fileprefix=""
                       ):
    """
    
    Perform PSF subtraction on the dataset. Optionally using a reference star dataset.
    TODO: 
        Handle nans & propagate DQ array
        Crop data to darkhole size ~(60x60) centered on nearest pixel (waiting on crop step function PR)
        Rotate north at the end
        Do frame combine before PSF subtraction?
        What info is missing from output dataset headers?
        Add comments to new ext header cards
        Figure out output roll angle.
        How to populate HISTORY header kw?

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)
        reference_star_dataset (corgidrp.data.Dataset): a dataset of Images of the reference star [optional]
        mode (str): pyKLIP PSF subraction mode, e.g. ADI/RDI/ADI+RDI. Mode will be chosen autonomously if not specified.
        annuli (int): number of concentric annuli to run separate subtractions on. Defaults to 1.
        subsections (int): number of angular subsections to run separate subtractions on. Defaults to 1.
        movement (int): KLIP movement parameter. Defaults to 1.
        numbasis (int or list of int): number of KLIP modes to retain. Defaults to [1,4,8,16].
        outdir (str or path): path to output directory. Defaults to "KLIP_SUB".
        fileprefix (str): prefix of saved output files. Defaults to "".

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the PSF subtraction applied (L4-level)

    """

    sci_dataset = input_dataset.copy()
    if not reference_star_dataset is None:
        ref_dataset = reference_star_dataset.copy()
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

    # TODO: Crop data (make sure psf center is updated)

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

    frames = []
    for i,frame_data in enumerate(pyklip_data):

        # TODO: Handle DQ & errors correctly
        err = np.zeros_like(frame_data)
        dq = np.zeros_like(frame_data) # This will get filled out later

        # Clean up primary header
        pri_hdr = pyklip_hdr.copy()
        naxis1 = pri_hdr['NAXIS1']
        naxis2 = pri_hdr['NAXIS2']
        del pri_hdr['NAXIS1']
        del pri_hdr['NAXIS2']
        del pri_hdr['NAXIS3']
        pri_hdr['NAXIS'] = 0

        # Add observation info from input dataset
        pri_hdr['TELESCOP'] = sci_dataset[0].pri_hdr['TELESCOP']
        pri_hdr['INSTRUME'] = sci_dataset[0].pri_hdr['INSTRUME']
        pri_hdr['MODE'] = sci_dataset[0].pri_hdr['MODE']
        pri_hdr['BAND'] = sci_dataset[0].pri_hdr['BAND']
        
        # Make extension header
        ext_hdr = fits.Header()
        ext_hdr['NAXIS'] = 2
        ext_hdr['NAXIS1'] = naxis1
        ext_hdr['NAXIS2'] = naxis2
        ext_hdr['BUNIT'] = sci_dataset[0].ext_hdr['BUNIT']
        ext_hdr['PIXSCALE'] = sci_dataset[0].ext_hdr['PIXSCALE']
        ext_hdr['KLIP_ALG'] = mode
        ext_hdr['KLMODES'] = pyklip_hdr[f'KLMODE{i}']
        ext_hdr['STARLOCX'] = pyklip_hdr['PSFCENTX']
        ext_hdr['STARLOCY'] = pyklip_hdr['PSFCENTY']
        ext_hdr['PSFCENTX'] = pyklip_hdr['PSFCENTX']
        ext_hdr['PSFCENTY'] = pyklip_hdr['PSFCENTY']
        if "HISTORY" in sci_dataset[0].ext_hdr.keys():
            ext_hdr['HISTORY'] = sci_dataset[0].ext_hdr['HISTORY']
        
        frame = data.Image(frame_data,
                                    pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                                    err=err, dq=dq)
        
        frames.append(frame)
    
    dataset_out = data.Dataset(frames)

    # Flag nans in the dq array and then add nans to the error array
    dataset_out = flag_nans(dataset_out,flag_val=1)
    dataset_out = nan_flags(dataset_out,threshold=1)
    
    history_msg = f'PSF subtracted via pyKLIP {mode}.'
    
    dataset_out.update_after_processing_step(history_msg)
    
    # TODO: Update DQ to 1 where there are nans

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
