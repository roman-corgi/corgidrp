# A file that holds the functions that transmogrify l1 data to l2a data 


def prescan_process(input_dataset): 
    """
    
    Crops the data arrays if appropriate and subtracts the prescan-bias

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need prescan processing (L1-level)

    Returns:
        corgidrp.data.Dataset: a prescan-processed version of the input dataset
    """

    return None



def detect_cosmic_rays(input_dataset):
    """
    
    Detects cosmis rays in a given images. Updates the DQ to reflect the pixels that are affected. 
    TODO: Decide if we want this step to optionally compensate for them, or if that's a different step. 

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need cosmic ray identification (L1-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset of the input dataset where the cosmic rays have been identified. 
    """

    return None

def correct_nonlinearity(input_dataset, non_lin_correction):
    """
    Perform non-linearity correction of a dataset using the corresponding non-linearity correction

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need non-linearity correction (L2a-level)
        non_lin_correction (corgidrp.data.NonLinearityCorrection): a NonLinearityCorrection calibration file to model the non-linearity

    Returns:
        corgidrp.data.Dataset: a non-linearity corrected version of the input dataset
    """

    return None

def prescan_biassub(input_dataset, bias_offset=0., return_full_frame=False):
    """
    Perform pre-scan bias subtraction of a dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L1a-level)
        bias_offset (float): an offset value to be subtracted from the bias
        return_full_frame (bool): flag indicating whether to return the full frame or only the bias-subtracted image area
    Returns:
        corgidrp.data.Dataset: a pre-scan bias subtracted version of the input dataset
    """
    return None