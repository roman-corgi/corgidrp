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

    return input_dataset.copy()

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
        # data.frames[i].err = data.frames[i].err / exposure_time
        scale_factor = (1/exposure_time) * np.ones([data.frames[i].err.shape[1], data.frames[i].err.shape[2]])
        # print(data.frames[i].err.shape)
        # print(data.frames[i].err[0])
        # print(scale_factor.shape)
        #data.frames[i].rescale_error(data.frames[i].err, 'normalized by the exposure time')
        data.frames[i].rescale_error(scale_factor, 'normalized by the exposure time')

        all_data_new[i] = data.frames[i].data
        all_err_new[i] = data.frames[i].err

        data.frames[i].ext_hdr.set('UNITS', 'electrons/s')
    
    history_msg = 'divided by the exposure time'
    data.update_after_processing_step(history_msg, new_all_data = all_data_new, new_all_err = all_err_new)


    return data

def update_to_l3(input_dataset):
    """
    Updates the data level to L3. Only works on L2b data.

    Currently only checks that data is at the L2b level

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)

    Returns:
        corgidrp.data.Dataset: same dataset now at L3-level
    """
    # check that we are running this on L1 data
    for orig_frame in input_dataset:
        if orig_frame.ext_hdr['DATA_LEVEL'] != "L2b":
            err_msg = "{0} needs to be L2b data, but it is {1} data instead".format(orig_frame.filename, orig_frame.ext_hdr['DATA_LEVEL'])
            raise ValueError(err_msg)

    # we aren't altering the data
    updated_dataset = input_dataset.copy(copy_data=False)

    for frame in updated_dataset:
        # update header
        frame.ext_hdr['DATA_LEVEL'] = "L3"
        # update filename convention. The file convention should be
        # "CGI_[dataleel_*]" so we should be same just replacing the just instance of L1
        frame.filename = frame.filename.replace("_L2b_", "_L3_", 1)

    history_msg = "Updated Data Level to L3"
    updated_dataset.update_after_processing_step(history_msg)

    return updated_dataset