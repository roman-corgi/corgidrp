import numpy as np
import corgidrp.data as data

def create_dark_calib(dark_dataset):
    """
    Turn this dataset of image frames that were taken to measure 
    the dark current into a dark calibration frame

    Args:
        dark_dataset (corgidrp.data.Dataset): a dataset of Image frames (L2a-level)
    Returns:
        data.Dark: a dark calibration frame
    """
    combined_frame = np.nanmean(dark_dataset.all_data, axis=0)

    new_dark = data.Dark(combined_frame, pri_hdr=dark_dataset[0].pri_hdr.copy(), 
                         ext_hdr=dark_dataset[0].ext_hdr.copy(), input_dataset=dark_dataset)
    
    return new_dark


def dark_subtraction(input_dataset, dark_frame):
    """
    Perform dark current subtraction of a dataset using the corresponding dark frame

    Args:
        intpu_dataset (corgidrp.data.Dataset): a dataset of Images that need dark subtraction (L2a-level)
        dark_frame (corgidrp.data.Dark): a Dark frame to model the dark current
    Returns:
        corgidrp.data.Dataset: a dark subtracted version of the input dataset
    """
    darksub_cube = input_dataset.all_data - dark_frame.data

    history_msg = "Dark current subtracted using dark {0}".format(dark_frame.filename)

    # note that current implementation this points to the same dataset, just with updated data
    # THIS COULD CHANGE DEPENDING ON ARCHITECTURE 
    # please use this syntax currently to ensure design can be flexible
    darksub_dataset = input_dataset.update_after_processing_step(history_msg, darksub_cube)

    return darksub_dataset
