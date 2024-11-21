# A file that holds the functions that transmogrify l2b data to l3 data 
import corgidrp.fluxcal as fluxcal


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

    TODO: Make sure to update the headers to reflect the change in units

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in units of electrons/s
    """

    return input_dataset.copy()

def determine_color_cor(input_dataset, calspec_filepath, source_sed):
    """
    determine the color correction factor of the observed source
    at the reference wavelength of the used filter band and put it into the header
    
    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)
        calspec_filepath (str): file name of the known reference flux (usually CALSPEC)
        source_sed (np.array): flux model of the observed source in CALSPEC units (erg/(s * cm^2 * AA)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with updated header including 
        the reference wavelength and the color correction factor
    """
    color_dataset = input_dataset.copy()
    # get the filter name from the header keyword 'CFAMNAME'
    filter_name = fluxcal.get_filter_name(color_dataset)
    # read the transmission curve from the color filter file
    wave, filter_trans = fluxcal.read_filter_curve(filter_name)
    # calculate the reference wavelength of the color filter
    lambda_ref = fluxcal.calculate_pivot_lambda(filter_trans, wave)
    # calculate the flux from the user given CALSPEC file
    flux_ref = fluxcal.read_cal_spec(calspec_filepath, wave)
    #Calculate the color correction factor
    k = fluxcal.compute_color_cor(filter_trans, wave, flux_ref, lambda_ref, source_sed)
    
    # write the reference wavelength and the color correction factor to the header (keyword names tbd)
    history_msg = "the color correction is calculated and added to the header {0}".format(str(k))
    # update the header of the output dataset and update the history
    color_dataset.update_after_processing_step(history_msg, header_entries = {"LAM_REF": lambda_ref, "COL_COR": k})
    return color_dataset

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