# A file that holds the functions that transmogrify l4 data to TDA (Technical Demo Analysis) data 
import corgidrp.fluxcal as fluxcal
import numpy as np
import warnings

def determine_app_mag(input_dataset, source_star, scale_factor = 1.):
    """
    determine the apparent Vega magnitude of the observed source
    in the used filter band and put it into the header.
    We assume that each frame in the dataset was observed with the same color filter.
    
    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)
        source_star (str): either the fits file path of the flux model of the observed source in 
                           CALSPEC units (erg/(s * cm^2 * AA) and format or the (SIMBAD) name of a CALSPEC star
        scale_factor (float): factor applied to the flux of the calspec standard source, so that you can apply it 
                              if you have a different source with similiar spectral type, but no calspec standard.
                              Defaults to 1.
    
    Returns:
        corgidrp.data.Dataset: a version of the input dataset with updated header including 
                               the apparent magnitude
    """
    mag_dataset = input_dataset.copy()
    # get the filter name from the header keyword 'CFAMNAME'
    filter_name = fluxcal.get_filter_name(mag_dataset[0])
    # read the transmission curve from the color filter file
    wave, filter_trans = fluxcal.read_filter_curve(filter_name)

    if source_star.split(".")[-1] == "fits":
        source_filepath = source_star
    else:
        source_filepath = fluxcal.get_calspec_file(source_star)
    
    vega_filepath = fluxcal.get_calspec_file('Vega')
    
    # calculate the flux of VEGA and the source star from the user given CALSPEC file binned on the wavelength grid of the filter
    vega_sed = fluxcal.read_cal_spec(vega_filepath, wave)
    source_sed = fluxcal.read_cal_spec(source_filepath, wave) * scale_factor
    #Calculate the irradiance of vega and the source star in the filter band
    vega_irr = fluxcal.calculate_band_irradiance(filter_trans, vega_sed, wave)
    source_irr = fluxcal.calculate_band_irradiance(filter_trans, source_sed, wave)
    #calculate apparent magnitude
    app_mag = -2.5 * np.log10(source_irr/vega_irr)
    # write the reference wavelength and the color correction factor to the header (keyword names tbd)
    history_msg = "the apparent Vega magnitude is calculated and added to the header {0}".format(str(app_mag))
    # update the header of the output dataset and update the history
    mag_dataset.update_after_processing_step(history_msg, header_entries = {"APP_MAG": app_mag})
    
    return mag_dataset


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
    filter_name = fluxcal.get_filter_name(color_dataset[0])
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


def convert_to_flux(input_dataset, fluxcal_factor):
    """

    Convert the data from electron unit to flux unit erg/(s * cm^2 * AA).

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images
        fluxcal_factor (corgidrp.data.FluxcalFactor): flux calibration file

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in flux units
    """
   # you should make a copy the dataset to start
    flux_dataset = input_dataset.copy()
    flux_cube = flux_dataset.all_data
    flux_error = flux_dataset.all_err
    if "COL_COR" in flux_dataset[0].ext_hdr:
        color_cor_fac = flux_dataset[0].ext_hdr['COL_COR']
    else: 
        warnings.warn("There is no COL_COR keyword in the header, color correction was not done, it is set to 1")
        color_cor_fac = 1
    factor = fluxcal_factor.fluxcal_fac/color_cor_fac
    factor_error = fluxcal_factor.fluxcal_err/color_cor_fac
    error_frame = flux_cube * factor_error
    flux_cube *= factor
    
    #scale also the old error with the flux_factor and propagate the error 
    # err = sqrt(err_flux^2 * flux_fac^2 + fluxfac_err^2 * flux^2)
    factor_2d = np.ones(np.shape(flux_dataset[0].data)) * factor
    flux_dataset.rescale_error(factor_2d, "fluxcal_factor")
    flux_dataset.add_error_term(error_frame, "fluxcal_error")

    history_msg = "data converted to flux unit erg/(s * cm^2 * AA) by fluxcal_factor {0} plus color correction".format(fluxcal_factor.fluxcal_fac)

    # update the output dataset with this converted data and update the history
    flux_dataset.update_after_processing_step(history_msg, new_all_data=flux_cube, header_entries = {"BUNIT":"erg/(s*cm^2*AA)", "FLUXFAC":fluxcal_factor.fluxcal_fac})
    return flux_dataset

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