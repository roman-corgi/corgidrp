# A file that holds the functions that transmogrify l2a data to l2b data 

# A file that holds the functions that transmogrify l1 data to l2a data 


def dark_subtraction(input_dataset, dark_frame):
    """
    
    Perform dark current subtraction of a dataset using the corresponding dark frame

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need dark subtraction (L2a-level)
        dark_frame (corgidrp.data.Dark): a Dark frame to model the dark current

    Returns:
        corgidrp.data.Dataset: a dark subtracted version of the input dataset
    """
    # you should make a copy the dataset to start
    darksub_dataset = input_dataset.copy()

    darksub_cube = darksub_dataset.all_data - dark_frame.data

    history_msg = "Dark current subtracted using dark {0}".format(dark_frame.filename)

    # update the output dataset with this new dark subtracted data and update the history
    darksub_dataset.update_after_processing_step(history_msg, new_all_data=darksub_cube)

    return darksub_dataset