# A file that holds the functions that transmogrify l4 data to TDA (Technical Demo Analysis) data 
import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d, LinearNDInterpolator
import warnings
from photutils.psf import fit_2dgaussian
from corgidrp.data import Dataset, Image, FluxcalFactor
import corgidrp.fluxcal as fluxcal
from corgidrp import check
from corgidrp.klip_fm import measure_noise
from corgidrp.find_source import make_snmap, psf_scalesub
from scipy.stats import gaussian_kde

def determine_app_mag(input_data, source_star, scale_factor = 1.):
    """
    Determine the apparent Vega magnitude by comparing CALSPEC SEDs.

    This function integrates the CALSPEC spectrum given by source_star through the
    filter bandpass of the input image(s), integrates the Vega CALSPEC spectrum through
    the same bandpass, and writes APP_MAG = -2.5 log10(source_irr / vega_irr) into
    the header. The dataset is only inspected to ensure all frames share the same
    filter and target. This function does not use the measured photometry/flux in input_data. 
    To convert a measured flux into a Vega magnitude, use fluxcal.calculate_vega_mag or determine_flux.

    Args:
        input_data (corgidrp.data.Dataset or corgidrp.data.Image): 
            A dataset of Images (L2b-level) or a single Image. Must be all of the same source with same filter.
        source_star (str): either the fits file path of the flux model of the observed source in 
                           CALSPEC units erg/(s * cm^2 * AA) and format or the (SIMBAD) name of a CALSPEC star
        scale_factor (float): factor applied to the flux of the calspec standard source, so that you can apply it 
                              if you have a different source with similar spectral type, but no calspec standard.
                              Defaults to 1.
    
    Returns:
        mag_data (corgidrp.data.Dataset): A version of the input with an updated header including the apparent 
            magnitude.
    """
    # If input is a dataset, process each image
    if isinstance(input_data, Dataset):
        mag_data = input_data.copy()

        # Make sure all frames in dataset have the same filter and target
        filter_name = fluxcal.get_filter_name(mag_data[0])
        target_name = mag_data[0].pri_hdr["TARGET"]
        
        for img in mag_data:
            img_filter = fluxcal.get_filter_name(img)
            img_target = img.pri_hdr["TARGET"]

            if img_filter != filter_name:
                raise ValueError(f"All images in dataset must be taken with the same CFAMNAME for calculating"
                                 f"apparent magnitude. Found {img_filter}, expected {filter_name}."
                                 )
            if img_target != target_name:
                raise ValueError(f"All images in dataset must be taken of the same TARGET for calculating"
                                 f"apparent magnitude. Found {img_target}, expected {target_name}."
                                 )

    elif isinstance(input_data, Image):
        mag_data = Dataset([input_data.copy()])
        filter_name = fluxcal.get_filter_name(mag_data[0]) 

    # read the transmission curve from the color filter file
    wave, filter_trans = fluxcal.read_filter_curve(filter_name)
    
    if source_star.split(".")[-1] == "fits":
        source_filepath = source_star
        source_filename = os.path.basename(source_star)
    else:
        source_filepath, source_filename = fluxcal.get_calspec_file(source_star)
    
    vega_filepath, vega_filename = fluxcal.get_calspec_file('Vega')
    # calculate the flux of VEGA and the source star from the user given CALSPEC file binned on the wavelength grid of the filter
    vega_sed = fluxcal.read_cal_spec(vega_filepath, wave)
    source_sed = fluxcal.read_cal_spec(source_filepath, wave) * scale_factor
    #Calculate the irradiance of vega and the source star in the filter band
    vega_irr = fluxcal.calculate_band_irradiance(filter_trans, vega_sed, wave)
    source_irr = fluxcal.calculate_band_irradiance(filter_trans, source_sed, wave)

    #calculate apparent magnitude
    app_mag = -2.5 * np.log10(source_irr/vega_irr)

    # write the reference wavelength and the color correction factor to the header (keyword names tbd)
    history_msg = "the apparent Vega magnitude is calculated and added to the header {0} applying source SED file {1} and VEGA SED file {2}".format(str(app_mag), source_filename, vega_filename)
    # update the header of the output dataset and update the history
    mag_data.update_after_processing_step(history_msg, header_entries = {"APP_MAG": app_mag})
    
    return mag_data


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
                           CALSPEC units erg/(s * cm^2 * AA) and format or the (SIMBAD) name of a CALSPEC star
    
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
        calspec_ref_name = os.path.basename(ref_star)
    else:
        calspec_filepath, calspec_ref_name = fluxcal.get_calspec_file(ref_star)
    if source_star.split(".")[-1] == "fits":
        source_filepath = source_star
        source_filename = os.path.basename(source_star)
    else:
        source_filepath, source_filename = fluxcal.get_calspec_file(source_star)
    
    # calculate the flux from the user given CALSPEC file binned on the wavelength grid of the filter
    flux_ref = fluxcal.read_cal_spec(calspec_filepath, wave)
    # we assume that the source spectrum is a calspec standard or its 
    # model data is in a file with the same format and unit as the calspec data
    source_sed = fluxcal.read_cal_spec(source_filepath, wave)
    #Calculate the color correction factor
    k = fluxcal.compute_color_cor(filter_trans, wave, flux_ref, lambda_ref, source_sed)
    
    # write the reference wavelength and the color correction factor to the header (keyword names tbd)
    history_msg = "the color correction is calculated and added to the header: {0}, source SED file: {1}, reference SED file: {2}".format(str(k), source_filename, calspec_ref_name)
    # update the header of the output dataset and update the history
    color_dataset.update_after_processing_step(history_msg, header_entries = {"LAM_REF": lambda_ref, "COL_COR": k})
    
    return color_dataset


def convert_spec_to_flux(input_dataset, fluxcal_factor, slit_transmission=None):
    """
    Flux calibrate 1-D spectroscopy spectra stored in the L4 SPEC extension.
    The function applies COL_COR when present, propagates calibration
    uncertainties, and applies slit transmission correction if requested. Requires
    the input dataset to have already been core-throughput corrected
    (ie SPEC header contains CTCOR=True).

    Args:
        input_dataset (corgidrp.data.Dataset): L4 dataset containing SPEC,
            SPEC_ERR, SPEC_DQ, SPEC_WAVE, and SPEC_WAVE_ERR extensions.
        fluxcal_factor (corgidrp.data.FluxcalFactor): absolute flux calibration
            product used to scale the spectrum.
        slit_transmission (tuple or list of tuples, optional): slit throughput
            information from spec.slit_transmission(). Provide either a single
            (slit_map, slit_x, slit_y) tuple applied to every frame or a list
            containing one tuple per frame in input_dataset.

    Returns:
        corgidrp.data.Dataset: copy of the input dataset with the
        SPEC/SPEC_ERR data converted to erg/(s*cm^2*Å) and
        headers/history updated.
        
    """
    if not isinstance(fluxcal_factor, FluxcalFactor):
        raise TypeError("fluxcal_factor must be a corgidrp.data.FluxcalFactor instance.")

    spec_dataset = input_dataset.copy()

    # Normalize slit transmission input to per-frame list
    if slit_transmission is None:
        slit_per_frame = [None] * len(spec_dataset)
    elif isinstance(slit_transmission, tuple):
        if len(slit_transmission) != 3:
            raise TypeError("slit_transmission tuples must be (slit_map, slit_x, slit_y).")
        slit_per_frame = [slit_transmission] * len(spec_dataset)
    elif isinstance(slit_transmission, list):
        if len(slit_transmission) not in (1, len(spec_dataset)):
            raise ValueError("slit_transmission must have length 1 or match the dataset length.")
        slit_per_frame = list(slit_transmission)
        for tup in slit_per_frame:
            if not (isinstance(tup, tuple) and len(tup) == 3):
                raise TypeError("Each slit_transmission entry must be a (slit_map, slit_x, slit_y) tuple.")
        if len(slit_per_frame) == 1 and len(spec_dataset) > 1:
            slit_per_frame = slit_per_frame * len(spec_dataset)
    else:
        raise TypeError("slit_transmission must be None, a (slit_map, slit_x, slit_y) tuple, or a list of such tuples.")

    history_messages = []

    for idx, frame in enumerate(spec_dataset):
        if 'SPEC' not in frame.hdu_list:
            raise ValueError("Input dataset does not contain a 'SPEC' extension.")

        is_coron = frame.ext_hdr.get('FSMLOS') == 1 # using FSMLOS=1 to check if the image is coronagraphic
        if is_coron:
            if not frame.hdu_list['SPEC'].header.get('CTCOR', False):
                raise ValueError("Core throughput correction must be applied before convert_spec_to_flux for coronagraphic images (missing CTCOR flag).")

        spec = frame.hdu_list['SPEC'].data.astype(float, copy=True)
        spec_header = frame.hdu_list['SPEC'].header
        spec_err = frame.hdu_list['SPEC_ERR'].data.astype(float, copy=True)

        if spec_header.get('BUNIT', '').strip().lower() != "photoelectron/s/bin":
            raise ValueError("SPEC extension must have BUNIT 'photoelectron/s/bin' before flux calibration.")

        # Apply algorithm throughput correction (ALGO_THRU) if present (PSF-subtracted frames)
        if 'ALGO_THRU' in frame.hdu_list:
            algo_thru = frame.hdu_list['ALGO_THRU'].data.astype(float)
            if algo_thru.shape != spec.shape:
                raise ValueError(
                    f"ALGO_THRU shape {algo_thru.shape} must match SPEC shape {spec.shape}."
                )
            # Divide by algorithm throughput, accounting for zeros/non-finite
            valid = np.isfinite(algo_thru) & (algo_thru != 0)
            spec = np.divide(spec, algo_thru, out=np.full_like(spec, np.nan), where=valid)
            spec_err = np.divide(spec_err, algo_thru, out=np.full_like(spec_err, np.nan), where=valid)
            spec_header['ALGOCOR'] = True
            history_messages.append("Applied algorithm throughput correction (ALGO_THRU).")

        # Apply slit transmission correction
        slit_vals = slit_per_frame[idx]
        slit_applied = False
        slit_curve = None
        if slit_vals is not None:
            slit_applied = True
            slit_curve = np.asarray(select_slit_transmission_curve(frame, slit_vals), dtype=float)
            if slit_curve.shape != spec.shape:
                raise ValueError(
                    f"slit_transmission curve shape {slit_curve.shape} must match SPEC shape {spec.shape}."
                )
            # Divide by wavelength-dependent slit transmission, accounting for zeros/non-finite
            valid = np.isfinite(slit_curve) & (slit_curve != 0)
            spec = np.divide(spec, slit_curve, out=np.full_like(spec, np.nan), where=valid)
            spec_err = np.divide(spec_err, slit_curve, out=np.full_like(spec_err, np.nan), where=valid)
            spec_header['SLITFAC'] = float(np.nanmean(slit_curve))
            spec_header['SLITCOR'] = True
        else:
            spec_header['SLITCOR'] = False
            if 'SLITFAC' in spec_header:
                del spec_header['SLITFAC']

        # Apply flux calibration factor and color correction
        color_cor_fac = frame.ext_hdr.get('COL_COR', 1.0)
        factor = fluxcal_factor.fluxcal_fac / color_cor_fac
        factor_error = fluxcal_factor.fluxcal_err / color_cor_fac

        # Convert to flux units and propagate uncertainties
        spec_flux = spec * factor
        spec_flux_err = np.sqrt((spec_err * factor) ** 2 + (spec * factor_error) ** 2)

        frame.hdu_list['SPEC'].data[:] = spec_flux
        frame.hdu_list['SPEC_ERR'].data[:] = spec_flux_err
        spec_header['BUNIT'] = "erg/(s*cm^2*AA)"
        if 'SPEC_ERR' in frame.hdu_list:
            frame.hdu_list['SPEC_ERR'].header['BUNIT'] = "erg/(s*cm^2*AA)"

        history_messages.append(
            f"Calibrated 1D spectrum with fluxcal_factor={fluxcal_factor.fluxcal_fac}, "
            f"COL_COR={color_cor_fac}, slit_correction={slit_applied}."
        )

    if history_messages:
        spec_dataset.update_after_processing_step(
            " ".join(history_messages),
            header_entries={"SPECUNIT": "erg/(s*cm^2*AA)", "FLUXFAC": fluxcal_factor.fluxcal_fac}
        )

    return spec_dataset


def apply_core_throughput_correction(frame,
                                     core_throughput_cal,
                                     fpam_fsam_cal,
                                     logr=False):
    """
    Apply a core-throughput correction to a single L4 spectroscopy frame.

    Args:
        frame (corgidrp.data.Image): L4 spectroscopy frame containing SPEC/SPEC_ERR HDUs.
        core_throughput_cal (corgidrp.data.CoreThroughputCalibration): calibration product
            providing InterpolateCT().
        fpam_fsam_cal (corgidrp.data.FpamFsamCal): calibration relating FPAM/FSAM motions
            to EXCAM coordinates.
        logr (bool): passed through to InterpolateCT (logarithmic radii interpolation).

    Returns:
        tuple: (ct_value, corrected_frame) where:
            - ct_value (float): core-throughput factor applied to the frame.
            - corrected_frame (corgidrp.data.Image): the corrected frame (same object as input, modified in place).
    """
    if fpam_fsam_cal is None:
        raise ValueError("fpam_fsam_cal is required for core throughput correction.")

    try:
        wv0_x = float(frame.ext_hdr['WV0_X'])
        wv0_y = float(frame.ext_hdr['WV0_Y'])
    except KeyError as exc:
        raise ValueError("Frame is missing WV0_X/WV0_Y required for core throughput correction.") from exc

    # Convert WV0_X/Y (absolute EXCAM pixels) to FPM-relative coordinates
    # STARLOCX/Y is the FPM center during the coronagraphic observation
    try:
        fpm_center_x = float(frame.ext_hdr['STARLOCX'])
        fpm_center_y = float(frame.ext_hdr['STARLOCY'])
    except KeyError as exc:
        raise ValueError("Frame is missing STARLOCX/STARLOCY required for core throughput correction.") from exc

    wv0_x_relative = wv0_x - fpm_center_x
    wv0_y_relative = wv0_y - fpm_center_y

    # InterpolateCT expects coordinates relative to the FPM center
    ct_values = core_throughput_cal.InterpolateCT(
        wv0_x_relative,
        wv0_y_relative,
        Dataset([frame]),
        fpam_fsam_cal,
        logr=logr,
    )
    ct_value = float(np.atleast_1d(ct_values).ravel()[0])
    if not np.isfinite(ct_value) or ct_value <= 0:
        raise ValueError(f"Invalid core throughput value {ct_value}.")

    frame.hdu_list['SPEC'].data[:] /= ct_value
    if 'SPEC_ERR' in frame.hdu_list:
        frame.hdu_list['SPEC_ERR'].data[:] /= ct_value
        frame.hdu_list['SPEC_ERR'].header['CTCOR'] = True
        frame.hdu_list['SPEC_ERR'].header['CTFAC'] = ct_value
    frame.hdu_list['SPEC'].header['CTCOR'] = True
    frame.hdu_list['SPEC'].header['CTFAC'] = ct_value
    
    return ct_value, frame


def select_slit_transmission_curve(frame, slit_tuple):
    """
    Select the slit-transmission curve for the frame from the tuple returned by
    spec.slit_transmission.

    Args:
        frame (corgidrp.data.Image): L4 spectroscopy frame whose WV0_X/WV0_Y
            coordinates identify where the slit correction should be evaluated.
        slit_tuple (tuple): Output from spec.slit_transmission containing
            (slit_map, slit_x, slit_y) arrays, where slit_map has shape
            (N_positions, N_wavelengths) and slit_x/slit_y are 1-D arrays of
            length N_positions giving the EXCAM coordinates of each position.

    Returns:
        numpy.ndarray: 1-D slit throughput curve sampled on the frame's SPEC
        wavelength grid.
    """
    slit_map, slit_x, slit_y = slit_tuple
    slit_map = np.asarray(slit_map, dtype=float)
    slit_x = np.asarray(slit_x, dtype=float)
    slit_y = np.asarray(slit_y, dtype=float)
    try:
        wv0_x = float(frame.ext_hdr['WV0_X'])
        wv0_y = float(frame.ext_hdr['WV0_Y'])
    except KeyError as exc:
        raise ValueError("Frame must contain WV0_X and WV0_Y for slit correction.") from exc

    # Slit map should be (N_positions, N_wave) or already 1-D in wavelength
    if slit_map.ndim == 1:
        slit_curve = slit_map
    elif slit_map.ndim == 2:
        if slit_map.shape[0] != slit_x.size or slit_x.size != slit_y.size:
            raise ValueError("slit_map first dimension must match slit_x and slit_y length.")
        # Find the closest sampled slit position to the spectrum's WV0 location (not interpolating,
        # just doing nearest neighbor lookup)
        idx = np.argmin(np.hypot(slit_x - wv0_x, slit_y - wv0_y))
        slit_curve = slit_map[idx]
    else:
        raise ValueError("slit_transmission map must be 1-D or 2-D.")

    slit_curve = np.asarray(slit_curve, dtype=float).ravel()

    # Require that the slit transmission is defined on the same size wavelength grid as SPEC
    # note: should spec.slit_transmission() also return a wavelength array to make sure it's
    # the same wavelength grid?
    spec_wave = frame.hdu_list['SPEC_WAVE'].data
    if slit_curve.size != spec_wave.size:
        raise ValueError(
            f"slit_transmission wavelength axis (len={slit_curve.size}) must match "
            f"SPEC_WAVE length (len={spec_wave.size})."
        )

    return slit_curve


def combine_spectra(input_dataset):
    """
    Combine multiple 1-D spectra in a Dataset into a single spectrum.

    This function takes a Dataset of images that each contain SPEC, 
    SPEC_ERR and SPEC_WAVE extensions, interpolates all spectra onto
    a common wavelength grid, and combines them using inverse-variance
    weighting (1/σ^2).

    Args:
        input_dataset (corgidrp.data.Dataset): Dataset of Images with 1-D
            spectra and SPEC/SPEC_ERR/SPEC_WAVE extensions.

    Returns:
        tuple: (combined_spec, wavelength, combined_err, rolls) where:
            - combined_spec (ndarray): weighted spectrum on the reference grid
            - wavelength (ndarray): reference wavelength grid
            - combined_err (ndarray): 1σ uncertainty of the combined spectrum
            - rolls (list): list of roll angles 
    """

    # Collect per-frame spectra, uncertainties, and wavelength grids
    spec_list = []
    err_list = []
    wave_list = []
    rolls = []

    for img in input_dataset:
        spec = np.array(img.hdu_list['SPEC'].data, dtype=float)
        spec_err = np.array(img.hdu_list['SPEC_ERR'].data, dtype=float)
        spec_err = np.squeeze(spec_err)
        wave = np.array(img.hdu_list['SPEC_WAVE'].data, dtype=float)

        if spec.shape != wave.shape:
            raise ValueError(f"SPEC shape {spec.shape} must match SPEC_WAVE shape {wave.shape}.")
        if spec_err.shape != spec.shape:
            raise ValueError(f"SPEC_ERR shape {spec_err.shape} must match SPEC shape {spec.shape}.")

        spec_list.append(spec)
        err_list.append(spec_err)
        wave_list.append(wave)
        rolls.append(img.pri_hdr.get('ROLL'))

    reference_wave = wave_list[0]
    ref_decreasing = reference_wave[0] > reference_wave[-1]
    reference_wave_ordered = reference_wave[::-1] if ref_decreasing else reference_wave

    spectra_per_frame = {}
    errs_per_frame = {}
    weighted_numer = np.zeros_like(reference_wave, dtype=float)
    weighted_denom = np.zeros_like(reference_wave, dtype=float)

    for idx in range(len(spec_list)):
        spec = spec_list[idx]
        spec_err = err_list[idx]
        wavelength = wave_list[idx]
        # Align each spectrum/err onto the reference grid if needed
        if np.allclose(wavelength, reference_wave):
            aligned_spec = spec
            aligned_err = spec_err
        else:
            # Regrid spectrum onto the reference wavelength grid
            src_decreasing = wavelength[0] > wavelength[-1]
            wave_ordered = wavelength[::-1] if src_decreasing else wavelength
            spec_ordered = spec[::-1] if src_decreasing else spec
            interp = np.interp(
                reference_wave_ordered,
                wave_ordered,
                spec_ordered,
                left=spec_ordered[0],
                right=spec_ordered[-1],
            )
            aligned_spec = interp[::-1] if ref_decreasing else interp

            # Regrid uncertainties in the same way
            err_ordered = spec_err[::-1] if src_decreasing else spec_err
            err_interp = np.interp(
                reference_wave_ordered,
                wave_ordered,
                err_ordered,
                left=err_ordered[0],
                right=err_ordered[-1],
            )
            aligned_err = err_interp[::-1] if ref_decreasing else err_interp

        spectra_per_frame[idx] = aligned_spec
        errs_per_frame[idx] = aligned_err

        # Inverse-variance weighting
        weights = 1.0 / (aligned_err ** 2)
        weighted_numer += aligned_spec * weights
        weighted_denom += weights

    # Final combined spectrum and uncertainty on the reference grid
    combined_spec = weighted_numer / weighted_denom
    combined_err = np.sqrt(1.0 / weighted_denom)

    return combined_spec, reference_wave, combined_err, rolls


def compute_spec_flux_ratio(host_image, companion_image, fluxcal_factor,
                            slit_transmission=None):
    """
    Compute the flux ratio of a single companion image relative to a single
    host image.

    Args:
        host_image (corgidrp.data.Image): L4 image containing the host spectrum (can be a combined/
        weighted spectrum).
        companion_image (corgidrp.data.Image): L4 image containing the companion spectrum (can be 
        a combined/ weighted spectrum).
        fluxcal_factor (corgidrp.data.FluxcalFactor): absolute flux calibration product.
        slit_transmission (tuple, optional): slit throughput tuple
            (slit_map, slit_x, slit_y) to apply during flux calibration.

    Returns:
        tuple: (flux_ratio, wavelength, metadata) where:
            - flux_ratio (numpy.ndarray): companion/host spectrum R(λ).
            - wavelength (numpy.ndarray): wavelength array in nm.
            - metadata (dict): contains:
                - 'roll': host roll angle
                - 'companion_roll': companion roll angle 
                - 'ratio_err': 1σ uncertainty on the flux ratio R(λ).
    """

    # Flux calibrate both spectra so the ratio is computed in physical units.
    host_ds = Dataset([host_image])
    comp_ds = Dataset([companion_image])
    host_cal = convert_spec_to_flux(host_ds, fluxcal_factor, slit_transmission=slit_transmission)
    comp_cal = convert_spec_to_flux(comp_ds, fluxcal_factor, slit_transmission=slit_transmission)

    host_spec = np.array(host_cal[0].hdu_list['SPEC'].data, dtype=float)
    comp_spec = np.array(comp_cal[0].hdu_list['SPEC'].data, dtype=float)
    host_err = np.array(host_cal[0].hdu_list['SPEC_ERR'].data, dtype=float)
    comp_err = np.array(comp_cal[0].hdu_list['SPEC_ERR'].data, dtype=float)
    host_err = np.squeeze(host_err)
    comp_err = np.squeeze(comp_err)
    host_wave = host_cal[0].hdu_list['SPEC_WAVE'].data
    comp_wave = comp_cal[0].hdu_list['SPEC_WAVE'].data

    # Align wavelength grids if the host/companion spectra were sampled in opposite directions
    # (np.interp requires increasing x-coordinates).
    if not np.allclose(host_wave, comp_wave):
        host_decreasing = host_wave[0] > host_wave[-1]
        comp_decreasing = comp_wave[0] > comp_wave[-1]

        host_wave_interp = host_wave[::-1] if host_decreasing else host_wave
        if comp_decreasing:
            comp_wave_interp = comp_wave[::-1]
            comp_spec_interp = comp_spec[::-1]
            comp_err_interp = comp_err[::-1]
        else:
            comp_wave_interp = comp_wave
            comp_spec_interp = comp_spec
            comp_err_interp = comp_err

        comp_spec_aligned = np.interp(
            host_wave_interp,
            comp_wave_interp,
            comp_spec_interp,
            left=comp_spec_interp[0],
            right=comp_spec_interp[-1]
        )
        comp_err_aligned = np.interp(
            host_wave_interp,
            comp_wave_interp,
            comp_err_interp,
            left=comp_err_interp[0],
            right=comp_err_interp[-1]
        )
        comp_spec = comp_spec_aligned[::-1] if host_decreasing else comp_spec_aligned
        comp_err = comp_err_aligned[::-1] if host_decreasing else comp_err_aligned

    # Ratio calculation - invalid host/companion values become NaN rather than raising warnings.
    with np.errstate(divide='ignore', invalid='ignore'):
        flux_ratio = comp_spec / host_spec
        invalid_mask = (host_spec == 0) | ~np.isfinite(host_spec) | ~np.isfinite(comp_spec)
        flux_ratio[invalid_mask] = np.nan

    # Propagate uncertainties for comp/host division:
    # σ_R^2 = (σ_C / H)^2 + (C * σ_H / H^2)^2
    ratio_unc = np.full_like(flux_ratio, np.nan, dtype=float)
    valid_unc = (
        (host_spec != 0) &
        np.isfinite(host_spec) &
        np.isfinite(comp_spec) &
        np.isfinite(host_err) &
        np.isfinite(comp_err)
    )
    if np.any(valid_unc):
        comp_term = np.zeros_like(flux_ratio, dtype=float)
        host_term = np.zeros_like(flux_ratio, dtype=float)
        comp_term[valid_unc] = (comp_err[valid_unc] / host_spec[valid_unc]) ** 2
        host_term[valid_unc] = (
            (comp_spec[valid_unc] * host_err[valid_unc]) / (host_spec[valid_unc] ** 2)
        ) ** 2
        variance = comp_term + host_term
        ratio_unc[valid_unc] = np.sqrt(variance[valid_unc])

    metadata = {
        'roll': host_image.pri_hdr.get('ROLL'),
        'companion_roll': companion_image.pri_hdr.get('ROLL'),
        'ratio_err': ratio_unc,
    }

    return flux_ratio, host_wave, metadata


def convert_to_flux(input_dataset, fluxcal_factor):
    """

    Convert the data from photoelectron unit to flux unit erg/(s * cm^2 * AA).

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images
        fluxcal_factor (corgidrp.data.FluxcalFactor): flux calibration file

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in flux units
    """
    # you should make a copy the dataset to start
    if input_dataset[0].ext_hdr['BUNIT'] != "photoelectron/s":
        raise ValueError("input dataset must have unit photoelectron/s for the conversion, not {0}".format(input_dataset[0].ext_hdr['BUNIT']))
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
    flux_dataset.rescale_error(factor, "fluxcal_factor")
    flux_dataset.add_error_term(error_frame, "fluxcal_error")

    history_msg = "data converted to flux unit erg/(s * cm^2 * AA) by fluxcal_factor {0} plus color correction".format(fluxcal_factor.fluxcal_fac)

    # update the output dataset with this converted data and update the history
    flux_dataset.update_after_processing_step(history_msg, new_all_data=flux_cube, header_entries = {"BUNIT":"erg/(s*cm^2*AA)", "FLUXFAC":fluxcal_factor.fluxcal_fac})
    return flux_dataset


def compute_flux_ratio_noise(input_dataset, NDcalibration, unocculted_star_dataset, unocculted_star_loc=None, requested_separations=None, halfwidth=None):
    '''
    Uses the PSF-subtracted frame and its algorithm throughput vs separation to 
    produce a calibrated 1-sigma flux ratio "contrast" curve (or "noise curve" since contrast curve is typically 5-sigma), also accounting for the throughput of the coronagraph.
    It calculates flux ratio noise curve value for each radial separation from the subtracted star location, interpolating KLIP and core throughput values at these input separations.
    It uses a dataset of unocculted stars and ND transmission to determine the integrated flux of the Gaussian-fit star (where each frame in the dataset is assumed to correspond to the frames 
    in the input_dataset), and an estimate of planet flux per frame of input_dataset is made by calculating the integrated flux of a Gaussian with amplitude equal to 
    the annular noise and FWHM equal to that used for KLIP algorithm througput for each radial separation.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of PSF-subtracted Images
        NDcalibration (corgidrp.data.NDFilterSweetSpotDataset): ND filter calibration
        unocculted_star_dataset (corgidrp.data.Dataset): a dataset of unocculted star Images corresponding to the Images in input_dataset.   Should have the same number of frames as input_dataset (1-to-1 correspondence).
        unocculted_star_loc (2-D float array, optional): array of coordinates of the unocculted stars according to the order given in the unocculted_star_dataset. 
            The first row of the array is for row position, and the second row is for column position. 
            If None, the peak pixel location is used for each frame.  Defaults to None. 
        requested_separations (float array, optional): separations at which to compute the flux ratio noise curve.  If None, the separations used for 
            the core throughput are used (e.g., no interpolation needed).  Defaults to None.
        halfwidth (float, optional): halfwidth of the annulus to use for noise calculation.  If None, half 
            of the minimum spacing between separation distances (if it isn't uniform spacing) is used.  Defaults to None.

    Returns:
        corgidrp.data.Dataset: input dataset with an additional extension header 'FRN_CRV' for every frame, containing the 
            calibrated flux ratio noise curve as a function of radial separation.  The data in that extension for a given frame is a (2+M)xN array,
            where:
            --the first row contains the separation radii in pixels 
            --the second row containts the separation radii in milli-arcseconds (mas) 
            --and the M rows contain the corresponding flux ratio noise curve values for the M KL mode truncations (maintaining the KL index ordering).
            TODO:  Add uncertainty to flux ratio noise curve based on uncertainties in core throughput and algorithm throughput if those are implemented in the future.
    '''
    output_dataset = input_dataset.copy()
    if len(input_dataset) != len(unocculted_star_dataset):
        raise ValueError('The number of frames in input_dataset and unocculted_star_dataset must be the same.')
    for i, frame in enumerate(output_dataset.frames):
        pixscale_mas = frame.ext_hdr['PLTSCALE']  
        klip_tp = frame.hdu_list['KL_THRU'].data[1:,:,0]
        core_tp = frame.hdu_list['CT_THRU'].data[1]
        klip_seps = frame.hdu_list['KL_THRU'].data[0,:,0]
        ct_seps = frame.hdu_list['CT_THRU'].data[0]
        klip_fwhms = frame.hdu_list['KL_THRU'].data[1:,:,1]
        min_sep = np.max([np.min(klip_seps), np.min(ct_seps)])
        max_sep = np.min([np.max(klip_seps), np.max(ct_seps)])
        if requested_separations is None:
            requested_separations = klip_seps
        if np.any(requested_separations < min_sep) or np.any(requested_separations > max_sep):
            warnings.warn('Not all requested_separations are within the range of the separations used for the KLIP and core throughputs.  Extrapolation will be used.')
        ct_spacings = ct_seps - np.roll(ct_seps, 1)
        # ignore the meaningless first entry (b/c of looping around with np.roll)
        min_ct_spacing = np.min(ct_spacings[1:])
        klip_spacings = klip_seps - np.roll(klip_seps, 1)
        # ignore the meaningless first entry (b/c of looping around with np.roll)
        min_klip_spacing = np.min(klip_spacings[1:])
        min_spacing = np.min([min_ct_spacing, min_klip_spacing])
        if halfwidth is None:
            halfwidth = min_spacing/2
        check.real_positive_scalar(halfwidth, 'halfwidth', ValueError)
        if halfwidth > min_spacing/2:
            warnings.warn('Halfwidth is wider than half the minimum spacing between separation values.')
        annular_noise = measure_noise(frame, requested_separations, halfwidth) # in photoelectrons/s
        # now need to get Fp/Fs
        # For star flux, Fs:  integrated flux of star modeled as analytic formula for volume under 2-D Gaussian defined 
        # by amplitude and FWHM used for KLIP throughput calculation.  Amplitude found by doing Gaussian fit.
        star_fr = unocculted_star_dataset.frames[i]
        if unocculted_star_loc is None:
            peak_row, peak_col = np.where(star_fr.data == star_fr.data.max())
            pos = (peak_row[0], peak_col[0])
        else:
            pos = (unocculted_star_loc[0][i], unocculted_star_loc[1][i])
        if pos[0] > star_fr.data.shape[0] or pos[1] > star_fr.data.shape[1]:
            raise ValueError('The guess centroid pixel location for the unocculted star is outside the image bounds.')
        # fit_shape inupt below must have odd numbers:
        if np.mod(star_fr.data.shape[0], 2) == 0:
            row_shape  = star_fr.data.shape[0] - 1
        else:
            row_shape = star_fr.data.shape[0]
        if np.mod(star_fr.data.shape[1], 2) == 0:
            col_shape  = star_fr.data.shape[1] - 1
        else:
            col_shape = star_fr.data.shape[1]
        fit_shape = (row_shape, col_shape)
        data = star_fr.data[:row_shape, :col_shape]
        mask = star_fr.dq.astype(bool)[:row_shape, :col_shape]
        guess_row, guess_col = pos
        # Get the value at the max_row and max_col position
        half_value = data[guess_row, guess_col] / 2
        # Calculate the absolute difference from half_value for all pixels
        abs_diff = np.abs(data - half_value)
        # Get the indices of the 20 smallest differences
        closest_indices = np.unravel_index(np.argsort(abs_diff.ravel())[:20], data.shape)
        # to estimate a guess FWHM over a large frame (to ensure a decent fit),
        # Find the highest-density location among the 20 pixels closest to half the guess star amplitude
        positions = np.vstack([closest_indices[0], closest_indices[1]])
        # Perform kernel density estimation
        kde = gaussian_kde(positions)
        density = kde(positions)
        # Find the index of the highest density
        highest_density_index = np.argmax(density)
        # Get the row and column of the highest-density location
        median_row = closest_indices[0][highest_density_index]
        median_col = closest_indices[1][highest_density_index]
        fwhm_guess = 2*np.sqrt((median_row-guess_row)**2 + (median_col-guess_col)**2)

        psf_phot = fit_2dgaussian(data, xypos=pos, fit_shape=fit_shape, fwhm=fwhm_guess, fix_fwhm=False,
                                mask=mask, error=None)
        star_xs = psf_phot.results['x_fit']
        star_ys = psf_phot.results['y_fit']
        # in case more than 1 PSF found:
        star_ind = np.argmin(np.sqrt((star_xs-guess_col)**2+(star_ys-guess_row)**2))
        star_x = star_xs[star_ind]
        star_y = star_ys[star_ind]
        #TODO perhaps incorporate into ERR in future, and incorporate error in psf_phot argument above (must be non-zero, though)
        star_err = psf_phot.results['flux_err'][star_ind] 
        # transmission through ND filter:
        ND_transmission = NDcalibration.interpolate_od(star_x, star_y)
        # integral under the fitted 2-D Gaussian for the unocculted star:
        Fs = ND_transmission * psf_phot.results['flux_fit'][star_ind] 
        # For planet flux, Fp:  treat the annular noise value as the amplitude of a 2-D Gaussian and use the 
        # same FWHM used for KLIP throughput calculation.  The analytic formula for volume under the Gaussian is the integrated flux.
        noise_amp = annular_noise.T
        # interpolate FWHMs to use based on requested_separations
        interp_fwhms = np.zeros((len(klip_fwhms), len(requested_separations)))
        for j in range(len(klip_fwhms)):
            fwhms_func = interp1d(klip_seps, klip_fwhms[j], kind='linear', fill_value='extrapolate')
            interp_fwhms[j] = fwhms_func(requested_separations)

        Fp = np.pi*noise_amp*interp_fwhms**2/(4*np.log(2)) #integral of 2-D Gaussian
        # Interpolate/extrapolate the algorithm and core throughputs at the desired separations
        klip_interp_func = interp1d(klip_seps, klip_tp, kind='linear', fill_value='extrapolate')
        klip_tp = klip_interp_func(requested_separations)
        ct_interp_func = interp1d(ct_seps, core_tp, kind='linear', fill_value='extrapolate')
        core_tp = ct_interp_func(requested_separations)
        frn_vals = (Fp/Fs)/(klip_tp*core_tp)
        # include row for separations in milli-arcseconds (mas)
        requested_mas = requested_separations * pixscale_mas
        flux_ratio_noise_curve = np.vstack([requested_separations, requested_mas, frn_vals])
        hdr = fits.Header()
        hdr['BUNIT'] = "Fp/Fs"
        hdr['COMMENT'] = "Flux ratio noise curve as a function of radial separation.  First row:  separation radii in pixels.  Second row:  separation radii in mas.  Remaining rows:  flux ratio noise curve values for KL mode truncations."
        frame.add_extension_hdu('FRN_CRV', data = flux_ratio_noise_curve, header=hdr)
        history_msg = 'Calibrated flux ratio noise curve added to extension header FRN_CRV.'
    output_dataset.update_after_processing_step(history_msg)
    return output_dataset

def determine_flux(input_dataset, fluxcal_factor,  photo = "aperture", phot_kwargs = None):
    """
    Calculates the total number of photoelectrons/s of a point source and convert them to the flux in erg/(s * cm^2 * AA).
    Write the flux and corresponding error in the header. Convert the flux to Vega magnitude and write it in the header.
    We assume that the source is the brightest point source in the field close to the center.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images with the source
        fluxcal_factor (corgidrp.data.FluxcalFactor): flux calibration file
        photo (String): do either aperture photometry ("aperture") or 2DGaussian fit of the point source ("2dgauss")
        phot_kwargs (dict): parameters of the photometry method, for details see fluxcal.aper_phot and fluxcal.phot_by_gauss_2dfit

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in flux units
    """
    # you should make a copy the dataset to start
    if input_dataset[0].ext_hdr['BUNIT'] != "photoelectron/s":
        raise ValueError("input dataset must have unit photoelectron/s for the flux determination, not {0}".format(input_dataset[0].ext_hdr['BUNIT']))
    flux_dataset = input_dataset.copy()
    if "COL_COR" in flux_dataset[0].ext_hdr:
        color_cor_fac = flux_dataset[0].ext_hdr['COL_COR']
    else: 
        warnings.warn("There is no COL_COR keyword in the header, color correction was not done, it is set to 1")
        color_cor_fac = 1
    factor = fluxcal_factor.fluxcal_fac/color_cor_fac
    factor_error = fluxcal_factor.fluxcal_err/color_cor_fac
    if photo == "aperture":
        if phot_kwargs == None:
            #set default values
            phot_kwargs = {
              'encircled_radius': 5,
              'frac_enc_energy': 1.0,
              'method': 'subpixel',
              'subpixels': 5,
              'background_sub': False,
              'r_in': 5,
              'r_out': 10,
              'centering_method': 'xy',
              'centroid_roi_radius': 5
            }
        phot_values = [fluxcal.aper_phot(image, **phot_kwargs) for image in flux_dataset]
    elif photo == "2dgauss":
        if phot_kwargs == None:
            #set default values
            phot_kwargs = {
              'fwhm': 3,
              'fit_shape': None,
              'background_sub': False,
              'r_in': 5,
              'r_out': 10,
              'centering_method': 'xy',
              'centroid_roi_radius': 5
              }
        phot_values = [fluxcal.phot_by_gauss2d_fit(image, **phot_kwargs) for image in flux_dataset]
    else:
        raise ValueError(photo + " is not a valid photo parameter, choose aperture or 2dgauss")
    
    if phot_kwargs.get('background_sub', False):
        ap_sum, ap_sum_err, back = np.mean(np.array(phot_values),0)
    else:
        ap_sum, ap_sum_err = np.mean(np.array(phot_values),0)
        back = 0
    
    flux = ap_sum * factor
    flux_err = np.sqrt(ap_sum_err**2 * factor**2 + factor_error**2 *ap_sum**2)
    #Also determine the apparent Vega magnitude
    filter_file = fluxcal.get_filter_name(flux_dataset[0])
    vega_mag = fluxcal.calculate_vega_mag(flux, filter_file)
    
    #calculate the magnitude error from the flux error and put it in MAGERR header
    vega_mag_err = 2.5/np.log(10) * flux_err/flux
    
    history_msg = "star {0} flux calculated as {1} erg/(s * cm^2 * AA) corresponding to {2} vega magnitude".format(flux_dataset[0].pri_hdr["TARGET"],flux, vega_mag)

    # update the output dataset with this converted data and update the history
    flux_dataset.update_after_processing_step(history_msg, header_entries = {"FLUXFAC": fluxcal_factor.fluxcal_fac, "LOCBACK": back, "FLUX": flux, "FLUXERR": flux_err, "APP_MAG": vega_mag, "MAGERR": vega_mag_err})
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
        if orig_frame.ext_hdr['DATALVL'] != "L4":
            err_msg = "{0} needs to be L4 data, but it is {1} data instead".format(orig_frame.filename, orig_frame.ext_hdr['DATALVL'])
            raise ValueError(err_msg)

    # we aren't altering the data
    updated_dataset = input_dataset.copy(copy_data=False)

    for frame in updated_dataset:
        # update header
        frame.ext_hdr['DATALVL'] = "TDA"
        # update filename convention. The file convention should be
        # "CGI_[datalevel_*]" so we should be same just replacing the just instance of L1
        frame.filename = frame.filename.replace("_l4_", "_tda_", 1)

    history_msg = "Updated Data Level to TDA"
    updated_dataset.update_after_processing_step(history_msg)

    return updated_dataset


def find_source(input_image, psf=None, fwhm=2.8, nsigma_threshold=5.0,
                image_without_planet=None):
    """
    Detects sources in an image based on a specified SNR threshold and save their approximate pixel locations and SNRs into the header.
    
    Args:
        input_image (corgidrp.data.Image): The input image to search for sources (L4-level).
        psf (ndarray, optional): The PSF used for detection. If None, a Gaussian approximation is created.
        fwhm (float, optional): Full-width at half-maximum of the PSF in pixels.
        nsigma_threshold (float, optional): The SNR threshold for source detection.
        image_without_planet (ndarray, optional): An image without any sources (~noise map) to make snmap more accurate.

    Returns:
        corgidrp.data.Image: A copy of the input image with the detected sources and their SNRs saved in the header.
        
    """

    new_image = input_image.copy()
    
    # Ensure an odd-sized box for PSF convolution
    boxsize = int(fwhm * 3)
    boxsize += 1 if boxsize % 2 == 0 else 0 # Ensure an odd box size
    
    # Create coordinate grids
    y, x = np.indices((boxsize, boxsize))
    y -= boxsize // 2 ; x -= boxsize // 2
    
    if psf is None:
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2))) # Convert FWHM to sigma
        psf = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

    # Generate a binary mask for PSF convolution   
    distance_map = np.sqrt(x**2 + y**2)  
    idx = np.where( (distance_map <= fwhm*0.5) )   
    psf_binarymask = np.zeros_like(psf) ; psf_binarymask[idx] = 1

    # Compute the SNR map using cross-correlation
    image_residual = np.zeros_like(new_image.data) + new_image.data
    image_snmap = make_snmap(image_residual, psf_binarymask, image_without_planet=image_without_planet)
    
    sn_source, xy_source = [], []

    # Iteratively detect sources above the SNR threshold
    while np.nanmax(image_snmap) >= nsigma_threshold:

        sn = np.nanmax(image_snmap)
        xy = np.unravel_index(np.nanargmax(image_snmap), image_snmap.shape)
        
        if sn > nsigma_threshold:
            sn_source.append(sn)
            xy_source.append(xy)

            # Scale and subtract the detected PSF from the image
            image_residual = psf_scalesub(image_residual, xy, psf, fwhm)
                
            # Update the SNR map after source removal
            image_snmap = make_snmap(image_residual, psf_binarymask, image_without_planet=image_without_planet)
        
    # Store detected sources in FITS header
    for i in range(len(sn_source)):
        new_image.ext_hdr[f'snyx{i:03d}'] = f'{sn_source[i]:5.1f},{xy_source[i][0]:4d},{xy_source[i][1]:4d}'        
    # names of the header keywords are tentative
    
    return new_image

def calculate_zero_point(image, star_name, encircled_radius, phot_kwargs=None):
    """
    Calculate the photometric zero point for a given star image.

    Computes the zero point by comparing the measured photon flux from the image with the apparent magnitude 
    determined for the star relative to Vega. If no photometry keyword arguments are provided, default values 
    are used for the aperture photometry parameters.

    Args:
        image (corgidrp.data.Image): An image containing the star.
        star_name (str): The name of the star, used to determine its apparent magnitude.
        encircled_radius (float): The radius within which to sum the counts for aperture photometry.
        phot_kwargs (dict, optional): A dictionary of keyword arguments for photometry, including parameters 
            like 'frac_enc_energy', 'method', 'subpixels', 'background_sub', 'r_in', 'r_out', 
            'centering_method', and 'centroid_roi_radius'. Defaults to a predefined set of parameters if None.

    Returns:
        zp (float): The computed zero point based on the apparent magnitude and the measured counts sum.
    """
    if phot_kwargs is None:
        phot_kwargs = {
            'frac_enc_energy': 1.0,
            'method': 'subpixel',
            'subpixels': 5,
            'background_sub': False,
            'r_in': 5,
            'r_out': 10,
            'centering_method': 'xy',
            'centroid_roi_radius': 5
        }

    # Compute apparent magnitude of this star compared to Vega in this band
    mag_dataset = determine_app_mag(image, star_name)
    
    # Get the apparent magnitude from the updated dataset
    app_mag = mag_dataset[0].ext_hdr["APP_MAG"]

    # Compute zero point using real measured photons
    ap_sum = fluxcal.aper_phot(image, encircled_radius, **phot_kwargs, centering_initial_guess=None)
    zp = app_mag + 2.5 * np.log10(ap_sum)

    return zp

def calc_pol_p_and_pa_image(input_Image):
    """Compute polarization intensity, fractional polarization, and EVPA from Stokes maps.

    Args:
        input_Image (Image): Object containing Stokes maps and uncertainties.

    Returns:
        Image: Image object containing
            - data: stacked [P, p, evpa] (shape: 3 x H x W)
            - ext_hdr: Header with updated HISTORY
            - err, dq: arrays reflecting Perr, perr and evpa_err

    Raises:
        AttributeError: If `input_Image` is missing `data` or `err` attributes.
        ValueError: If Stokes maps I, Q, U have inconsistent shapes or insufficient slices.
    """
    # --- Extract Stokes parameters ---
    try:
        I, Q, U = input_Image.data[0:3]
        Ierr, Qerr, Uerr = input_Image.err[0][0:3]
        Idq, Qdq, Udq = input_Image.dq[0:3]
        # V, Qphi, Uphi = Image.data[3:6] # unused
    except AttributeError as e:
        raise AttributeError("Image object must have 'data' and 'err' attributes.") from e
    except IndexError as e:
        raise ValueError("Image.data and Image.err must have at least [0..2] slices.") from e

    # --- Polarized intensity and error ---
    P = np.sqrt(Q**2 + U**2)
    Perr = np.sqrt((Q * Qerr)**2 + (U * Uerr)**2) / np.maximum(P, 1e-10)

    # --- Fractional polarization and its error ---
    p = P / np.maximum(I, 1e-10)
    perr = np.sqrt((Perr / np.maximum(I, 1e-10))**2 +
                   (P * Ierr / np.maximum(I, 1e-10)**2)**2)

    # --- Polarization angle (EVPA) and uncertainty ---
    evpa = 0.5 * np.arctan2(U, Q)  # radians
    evpa_err = 0.5 * np.sqrt((U * Qerr)**2 + (Q * Uerr)**2) / np.maximum(Q**2 + U**2, 1e-10)
    evpa = np.degrees(evpa)
    evpa_err = np.degrees(evpa_err)

    # --- Data quality propagation ---
    dq = np.bitwise_or(np.bitwise_or(Idq, Qdq), Udq)

    # --- Stack results ---
    data_out = np.stack([P, p, evpa], axis=0)
    err_out = np.stack([Perr, perr, evpa_err], axis=0)
    dq_out = np.stack([dq, dq, dq], axis=0)

    # --- Headers ---
    pri_hdr = input_Image.pri_hdr
    ext_hdr = input_Image.ext_hdr
    err_hdr = input_Image.err_hdr
    dq_hdr = input_Image.dq_hdr

    ext_hdr.add_history(
        "Derived polarization products: data=[P, p, EVPA]; "
        "err=[Perr, perr, EVPA_err]; dq propagated from I,Q,U."
    )

    # --- Construct output Image ---
    Image_out = Image(
        data_out,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        err=err_out,
        dq=dq_out,
        err_hdr=err_hdr,
        dq_hdr=dq_hdr
    )

    return Image_out

def compute_QphiUphi(image, x_center=None, y_center=None):
    """
    Compute Q_phi and U_phi from Stokes Q and U, returning an Image with shape [6, n, m]:
    [I, Q, U, V, Q_phi, U_phi].

    Args:
        image: Image
            Input image whose `data` is shaped [4, n, m] as [I, Q, U, V]. If the extension
            header contains 'STARLOCX' and 'STARLOCY', these are used as the stellar center.
        x_center: float or None
            Optional override for the x-coordinate of the center. Used only when the header
            does not provide 'STARLOCX'. If both header and this argument are missing, the
            image center ( (m-1)/2 ) is used.
        y_center: float or None
            Optional override for the y-coordinate of the center. Used only when the header
            does not provide 'STARLOCY'. If both header and this argument are missing, the
            image center ( (n-1)/2 ) is used.

    Returns:
        Image: A copy of the input image with data expanded to [6, n, m] as
            [I, Q, U, V, Q_phi, U_phi]. The `err` array is expanded to match and the
            uncertainties for Q_phi/U_phi are propagated from Q and U (assuming no covariance).
            The `dq` planes are expanded to match; if I/Q/U/V have identical dq, that mask is
            copied to both Q_phi and U_phi, otherwise Q_phi inherits Q’s dq and U_phi inherits
            U’s dq.
    """

    # copy of the input image
    out = image.copy(copy_data=True)

    data = out.data
    I, Q, U, V = data
    n, m = I.shape

    ext_hdr = getattr(out, "ext_hdr", None)

    # Determine center coordinates: header > input args > image center (print which is used)
    if ext_hdr is not None and ("STARLOCX" in ext_hdr) and ("STARLOCY" in ext_hdr):
        cx = float(ext_hdr["STARLOCX"])
        cy = float(ext_hdr["STARLOCY"])
    elif (x_center is not None) and (y_center is not None):
        cx = float(x_center)
        cy = float(y_center)
    else:
        cx = (m - 1) * 0.5
        cy = (n - 1) * 0.5

    # Polar angle φ and rotation (use float64 to reduce numerical errors)
    y_idx, x_idx = np.mgrid[0:n, 0:m]
    phi = np.arctan2(y_idx - cy, x_idx - cx)

    c2 = np.cos(2.0 * phi, dtype=np.float64)
    s2 = np.sin(2.0 * phi, dtype=np.float64)

    Qf = Q.astype(np.float64, copy=False)
    Uf = U.astype(np.float64, copy=False)

    Q_phi = -Qf * c2 - Uf * s2
    U_phi =  Qf * s2 - Uf * c2

    # Cast back to the same dtype as input data
    Q_phi = Q_phi.astype(data.dtype, copy=False)
    U_phi = U_phi.astype(data.dtype, copy=False)

    # Expand data to [6,n,m]
    out.data = np.concatenate([data, Q_phi[None, ...], U_phi[None, ...]], axis=0)

    # Ensure err / dq match the data (create if missing, or add 2 new planes)
    nplanes = out.data.shape[0]

    # --- err propagation ---
    if out.err is None:
        out.err = np.zeros((nplanes, n, m), dtype=out.data.dtype)
    else:
        if out.err.shape[0] >= 3 and out.err.shape[1:] == (n, m):
            sigma_Q = out.err[1].astype(np.float64)
            sigma_U = out.err[2].astype(np.float64)
            var_Qphi = (c2**2) * sigma_Q**2 + (s2**2) * sigma_U**2
            var_Uphi = (s2**2) * sigma_Q**2 + (c2**2) * sigma_U**2

            err_Qphi = np.sqrt(var_Qphi).astype(out.data.dtype)
            err_Uphi = np.sqrt(var_Uphi).astype(out.data.dtype)

            out.err = np.concatenate([out.err,
                                    err_Qphi[None, ...],
                                    err_Uphi[None, ...]], axis=0)
        else:
            out.err = np.zeros((nplanes, n, m), dtype=out.data.dtype)

    # --- dq alignment & inheritance ---
    if getattr(out, "dq", None) is None:
        # dq is nothing -> create zeros with correct shape
        out.dq = np.zeros((nplanes, n, m), dtype=np.uint16)
    elif out.dq.ndim != 3 or out.dq.shape[1:] != (n, m):
        # shape mismatch -> reset
        out.dq = np.zeros((nplanes, n, m), dtype=np.uint16)
    else:
        # if we still have only I,Q,U,V planes, append two new planes
        if out.dq.shape[0] == nplanes - 2 and out.dq.shape[0] >= 4:
            dq_Q = out.dq[1].astype(np.uint16, copy=False)
            dq_U = out.dq[2].astype(np.uint16, copy=False)
            dq_or = (dq_Q | dq_U).astype(np.uint16)
            out.dq = np.concatenate([out.dq, dq_or[None, ...], dq_or[None, ...]], axis=0)
    
        # if already 6 planes, (re)compute Q_phi/U_phi dq for correctness
        elif out.dq.shape[0] == nplanes:
            out.dq[4] = (out.dq[1].astype(np.uint16) | out.dq[2].astype(np.uint16))
            out.dq[5] = (out.dq[1].astype(np.uint16) | out.dq[2].astype(np.uint16))
    
        else:
            # anything else -> reset
            out.dq = np.zeros((nplanes, n, m), dtype=np.uint16)

    # Add HISTORY record
    msg = f"Computed Q_phi/U_phi with center=({cx:.6f},{cy:.6f}); output data shape {out.data.shape}."

    if ext_hdr is not None:
        try:
            ext_hdr['HISTORY'] = msg
        except Exception:
            pass

    return out
