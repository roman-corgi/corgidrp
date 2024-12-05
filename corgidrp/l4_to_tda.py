# A file that holds the functions that transmogrify l4 data to TDA (Technical Demo Analysis) data 
import corgidrp.fluxcal as fluxcal

def determine_color_cor(input_dataset, ref_star, source_star):
    """
    determine the color correction factor of the observed source
    at the reference wavelength of the used filter band and put it into the header.
    We assume that each frame in the dataset was observed with the same color filter.
    
    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)
        ref_star (str): either the fits file path of the known reference flux (usually CALSPEC),
                        or the (SIMBAD) name of a CALSPEC star
        source_star (str): either the fits file path of the flux model of the observed source in 
                           CALSPEC units (erg/(s * cm^2 * AA) and format or the (SIMBAD) name of a CALSPEC star
    
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
    
    # ref_star/source_star is either the star name or the file path to fits file
    if ref_star.split(".")[-1] == "fits":
        calspec_filepath = ref_star
    else:
        calspec_filepath = fluxcal.get_calspec_file(ref_star)
    if source_star.split(".")[-1] == "fits":
        source_filepath = source_star
    else:
        source_filepath = fluxcal.get_calspec_file(source_star)
    
    # calculate the flux from the user given CALSPEC file binned on the wavelength grid of the filter
    flux_ref = fluxcal.read_cal_spec(calspec_filepath, wave)
    # we assume that the source spectrum is a calspec standard or its 
    # model data is in a file with the same format and unit as the calspec data
    source_sed = fluxcal.read_cal_spec(source_filepath, wave)
    #Calculate the color correction factor
    k = fluxcal.compute_color_cor(filter_trans, wave, flux_ref, lambda_ref, source_sed)
    
    # write the reference wavelength and the color correction factor to the header (keyword names tbd)
    history_msg = "the color correction is calculated and added to the header {0}".format(str(k))
    # update the header of the output dataset and update the history
    color_dataset.update_after_processing_step(history_msg, header_entries = {"LAM_REF": lambda_ref, "COL_COR": k})
    
    return color_dataset

def update_to_tda(input_dataset):
    """
    Updates the data level to TDA (Technical Demo Analysis). Only works on L4 data.

    Currently only checks that data is at the L4 level

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L4-level)

    Returns:
        corgidrp.data.Dataset: same dataset now at TDA-level
    """
    # check that we are running this on L1 data
    for orig_frame in input_dataset:
        if orig_frame.ext_hdr['DATA_LEVEL'] != "L4":
            err_msg = "{0} needs to be L4 data, but it is {1} data instead".format(orig_frame.filename, orig_frame.ext_hdr['DATA_LEVEL'])
            raise ValueError(err_msg)

    # we aren't altering the data
    updated_dataset = input_dataset.copy(copy_data=False)

    for frame in updated_dataset:
        # update header
        frame.ext_hdr['DATA_LEVEL'] = "TDA"
        # update filename convention. The file convention should be
        # "CGI_[datalevel_*]" so we should be same just replacing the just instance of L1
        frame.filename = frame.filename.replace("_L4_", "_TDA_", 1)

    history_msg = "Updated Data Level to TDA"
    updated_dataset.update_after_processing_step(history_msg)

    return updated_dataset