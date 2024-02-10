# A file that holds the functions that transmogrify l1 data to l2a data 


def prescan_process(input_dataset): 
    """
    
    Crops the data arrays if appropriate and subtracts the prescan-bias

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need prescan processing (L1-level)

    Returns:
        corgidrp.data.Dataset: a prescan-processed version of the input dataset
    """

    pass



def detect_cosmic_rays(input_dataset):
    """
    
    Detects cosmis rays in a given images. Updates the DQ to reflect the pixels that are affected. 
    TODO: Decide if we want this step to optionally compensate for them, or if that's a different step. 

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need cosmic ray identification (L1-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset of the input dataset where the cosmic rays have been identified. 
    """

    pass