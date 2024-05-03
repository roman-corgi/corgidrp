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
        darksub_dataset.add_error_term(dark_frame.err, "dark_error")   
    else:
        raise Warning("no error attribute in the dark frame")
    
    #darksub_dataset.all_err = np.array([frame.err for frame in darksub_dataset.frames])
    history_msg = "Dark current subtracted using dark {0}".format(dark_frame.filename)

    # update the output dataset with this new dark subtracted data and update the history
    darksub_dataset.update_after_processing_step(history_msg, new_all_data=darksub_cube)

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

def convert_to_electrons(input_dataset, k_gain): 
    """
    
    Convert the data from ADU to electrons. 
    TODO: Establish the interaction with the CalDB for obtaining gain calibration 

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)
        k_gain(corgidrp.data.kgain: kgain calibration file

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in electrons
    """
   # you should make a copy the dataset to start
    kgain_dataset = input_dataset.copy()
    kgain_cube = kgain_dataset.all_data
    kgain_error = kgain_dataset.all_err
    
    kgain = k_gain.value #extract from caldb
    kgain_cube *= kgain
    kgain_error *= kgain
    
    history_msg = "data converted to detected EM electrons by kgain {0}".format(str(kgain))

    # update the output dataset with this converted data and update the history
    kgain_dataset.update_after_processing_step(history_msg, new_all_data=kgain_cube, new_all_err=kgain_error, header_entries = {"BUNIT":"detected EM electrons", "KGAIN":kgain})

    return kgain_dataset

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

def correct_bad_pixels(input_dataset):
    """
    
    Compute bad pixel map and correct for bad pixels. 

    MMB Notes: 
        - We'll likely want to be able to accept an external bad pixel map, either 
        from the CalDB or input by a user. 
        - We may want to accept just a list of bad pixels from a user too, thus 
        saving them the trouble of actually making their own map. 
        - We may want flags to decide which pixels in the DQ we correct. We may 
        not want to correct everything in the DQ extension
        - Different bad pixels in the DQ may be corrected differently.


    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with bad pixels corrected
    """

    return None

