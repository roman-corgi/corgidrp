# A file that holds the functions that transmogrify l2a data to l2b data 
import numpy as np

def add_photon_noise(input_dataset):
    """
    Propagate the photon noise determined from the image signal to the error map.
    The image values must already be in units of photons 

    Args:
       input_dataset (corgidrp.data.Dataset): a dataset of Images with values in photons (L2a-level)
    
    Returns:
        corgidrp.data.Dataset: photon noise propagated to the image error extensions of the input dataset
    """
    # you should make a copy the dataset to start
    phot_noise_dataset = input_dataset.copy() # necessary at all?
    
    for i, frame in enumerate(phot_noise_dataset.frames):
        frame.add_error_term(np.sqrt(frame.data), "photnoise_error")
    
    new_all_err = np.array([frame.err for frame in phot_noise_dataset.frames])        
                
    history_msg = "photon noise propagated to error map"
    # update the output dataset
    phot_noise_dataset.update_after_processing_step(history_msg, new_all_err = new_all_err)
    
    return phot_noise_dataset


def dark_subtraction(input_dataset, dark_frame):
    """
    
    Perform dark current subtraction of a dataset using the corresponding dark frame

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need dark subtraction (L2a-level)
        dark_frame (corgidrp.data.Dark): a Dark frame to model the dark current

    Returns:
        corgidrp.data.Dataset: a dark subtracted version of the input dataset including error propagation
    """
    # you should make a copy the dataset to start
    darksub_dataset = input_dataset.copy()

    darksub_cube = darksub_dataset.all_data - dark_frame.data
    
    # propagate the error of the dark frame
    if hasattr(dark_frame, "err"):
        darksub_dataset.add_error_term(dark_frame.err[0], "dark_error")   
    else:
        raise Warning("no error attribute in the dark frame")
    
    #darksub_dataset.all_err = np.array([frame.err for frame in darksub_dataset.frames])
    history_msg = "Dark current subtracted using dark {0}".format(dark_frame.filename)

    # update the output dataset with this new dark subtracted data and update the history
    darksub_dataset.update_after_processing_step(history_msg, new_all_data=darksub_cube, header_entries = {"BUNIT":"photoelectrons"})

    return darksub_dataset

def frame_select(input_dataset):
    """
    
    Selects the frames that we want to use for further processing. 
    TODO: Decide on frame selection criteria

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with only the frames we want to use
    """
    return None

def convert_to_electrons(input_dataset): 
    """
    
    Convert the data from ADU to electrons. 
    TODO: Establish the interaction with the CalDB for obtaining gain calibration 
    TODO: Make sure to update the headers to reflect the change in units

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in electrons
    """

    return None 

def em_gain_division(input_dataset):
    """
    
    Convert the data from detected EM electrons to detected electrons by dividing the commanded em_gain. 
    Update the change in units in the header [detected electrons].

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in units "detected electrons"
    """
    
    # you should make a copy the dataset to start
    emgain_dataset = input_dataset.copy()
    emgain_cube = emgain_dataset.all_data
    emgain_error = emgain_dataset.all_err
    
    unique = True
    emgain = emgain_dataset[0].ext_hdr["CMDGAIN"]
    for i in range(len(emgain_dataset)): 
        if emgain != emgain_dataset[i].ext_hdr["CMDGAIN"]:
            unique = False
            emgain = emgain_dataset[i].ext_hdr["CMDGAIN"] 
        emgain_cube[i] /= emgain
        emgain_error[i] /= emgain
    
    if unique:
        history_msg = "data divided by em_gain {0}".format(str(emgain))
    else:
        history_msg = "data divided by non-unique em_gain"

    # update the output dataset with this em_gain divided data and update the history
    emgain_dataset.update_after_processing_step(history_msg, new_all_data=emgain_cube, new_all_err=emgain_error, header_entries = {"BUNIT":"detected electrons"})

    return emgain_dataset
    
    

def cti_correction(input_dataset):
    """
    
    Apply the CTI correction to the dataset.
    TODO: Establish the interaction with the CalDB for obtaining CTI correction calibrations

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)
        
    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the CTI correction applied
    """

    return None

def flat_division(input_dataset, master_flat):
    """
    
    Divide the dataset by the master flat field. 

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)
        master_flat (corgidrp.data.Flat): a master flat field to divide by

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the flat field divided out
    """

    return None

def correct_bad_pixels(input_dataset, bp_mask):
    """
    
    Correct for bad pixels: Bad pixels are identified as part of the data
        calibration. This function replaces bad pixels by NaN values. It also
        updates its DQ storing the type of bad pixel at each bad pixel location,
        and it records the fact that the pixel has been replaced by NaN.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)
        bp_mask (corgidrp.data.BadPixelMap): Bad-pixel mask built from the bad
        pixel calibration file.


    Returns:
        corgidrp.data.Dataset: a version of the input dataset with bad detector
        pixels and cosmic rays replaced by NaNs
 
    """

    data = input_dataset.copy()
    data_cube = data.all_data
    dq_cube = data.all_dq.astype(np.uint8)

    for i in range(data_cube.shape[0]):
        # combine DQ and BP masks
        bp_dq_mask = np.bitwise_or(dq_cube[i],bp_mask[0].data.astype(np.uint8))
        # mask affected pixels with NaN
        bp = np.where(bp_dq_mask != 0)
        data_cube[i, bp[0], bp[1]] = np.nan
        # Update DQ to keep track of replaced bad pixel values
        bp_dq_mask[bp[0], bp[1]]=np.bitwise_or(bp_dq_mask[bp[0], bp[1]], 2)
        dq_cube[i] = bp_dq_mask

    history_msg = "removed pixels affected by bad pixels"
    data.update_after_processing_step(history_msg, new_all_data=data_cube,
        new_all_dq=dq_cube)

    return data
