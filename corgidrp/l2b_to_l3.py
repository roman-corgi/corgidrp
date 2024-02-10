# A file that holds the functions that transmogrify l2b data to l3 data 

def create_wcs(input_dataset):
    """
    
    Create the WCS headers for the dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the WCS headers added
    """

    pass

def divide_by_exptime(input_dataset):
    """
    
    Divide the data by the exposure time to get the units in electrons/s

    TODO: Make sure to update the headers to reflect the change in units

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in units of electrons/s
    """

    pass
