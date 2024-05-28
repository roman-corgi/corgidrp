# A file that holds the functions that transmogrify l2b data to l3 data 
import numpy as np

def create_wcs(input_dataset):
    """
    
    Create the WCS headers for the dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the WCS headers added
    """

    return None

def divide_by_exptime(input_dataset):
    """
    
    Divide the data by the exposure time to get the units in electrons/s

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in units of electrons/s
    """

    data = input_dataset.copy()

    all_data_new = np.zeros(data.all_data.shape)
    all_err_new = np.zeros(data.all_err.shape)

    for i in range(len(data.frames)):
        exposure_time = float(data.frames[i].ext_hdr['EXPTIME'])

        data.frames[i].data = data.frames[i].data / exposure_time
        data.frames[i].err = data.frames[i].err / exposure_time

        all_data_new[i] = data.frames[i].data
        all_err_new[i] = data.frames[i].err

        data.frames[i].ext_hdr.set('UNITS', 'electrons/s' )


    history_msg = "divided by the exposure time"
    data.update_after_processing_step(history_msg, new_all_data=all_data_new, new_all_err = all_err_new)

    return data
