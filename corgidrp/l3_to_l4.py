# A file that holds the functions that transmogrify l3 data to l4 data 

from pyklip.klip import rotate
from corgidrp import data
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

        # define the center for rotation
        try: 
            xcen, ycen = im_hd['PSFCENTX'], im_hd['PSFCENTY']
        except KeyError:
            xcen, ycen = im_data.shape[1]/2, im_data.shape[0]/2
    
        # look for WCS solutions
        if correct_wcs is False: # hardcoded now, no WCS information
            astr_hdr = None 
        #else:
        #   astr_hdr = None

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
        dq_data = processed_data.dq
        dq_derot = rotate(dq_data,-roll_angle,(xcen,ycen))
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
