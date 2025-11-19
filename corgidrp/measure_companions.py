import os
import numpy as np
from astropy.table import Table
import corgidrp.fluxcal as fluxcal
import corgidrp.klip_fm as klip_fm
from corgidrp.data import Image, Dataset
import corgidrp.l4_to_tda as l4_to_tda
import corgidrp.l3_to_l4 as l3_to_l4
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize
import pyklip.fakes


def measure_companions(
    host_star_image,
    psf_sub_image,
    ct_cal,
    fpam_fsam_cal,
    nd_cal=None,
    phot_method='aperture',
    photometry_kwargs=None,
    fluxcal_factor=None,
    host_star_in_calspec=True,
    thrp_corr="L4",
    coronagraphic_dataset=None,
    refstar_dataset=None,
    klip_fm_kwargs=None,
    verbose=True,
    kl_mode_idx=-1,
    cand_locs=None
):
    """
    Measure companion properties in a coronagraphic image and return a table with companion position, flux ratio, and apparent magnitude.
    
    Args:
        host_star_image (corgidrp.data.Image): Unocculted 2-D image of the host star (potentially corrected for ND transmission)
        psf_sub_image (corgidrp.data.Image): PSF-subtracted image with companions.
        ct_cal (corgidrp.data.CoreThroughputCalibration): Core throughput calibration data.
        fpam_fsam_cal (corgidrp.data.FpamFsamCal): Transformation calibration data.
        nd_cal (corgidrp.data.NDFilterSweetSpotDataset): ND calibration if host_star_image has not been corrected for ND. If None,
            no correction will be done.
        phot_method (str): Photometry method to use ('aperture' or 'gauss2d').
        photometry_kwargs (dict): Dictionary of keyword arguments for photometry.
        fluxcal_factor (corgidrp.Data.FluxcalFactor): Flux calibration factor object.
        host_star_in_calspec (bool): Flag indicating whether to use host star magnitude from calspec.
        thrp_corr (str): How to do the algorithm throughput estimation to measure the flux. Options are:
                         1) "L4", which uses the algorithm throughput calculated from L3 -> L4 processing
                         2) "None", which ignorees algorithm throughput
                         3) "KLIP-FM", which forward models the PSF through KLIP (requires passing in 
                         coronagraphic_dataset and refstar_dataset, if RDI). [not implemented]
        coronagraphic_dataset (corgidrp.data.Dataset): Dataset containing coronagraphic images.
        refstar_dataset (corgidrp.data.Dataset): RDI reference star dataset for PSF subtraction
        klip_fm_kwargs (dict): keyword arguments to pass into forward_model_psf()
        verbose (bool): Flag to enable verbose output.
        kl_mode_idx (int): Index of the KL mode to use (must match the one used for PSF subtraction).
            Defaults to the last KL mode.
        cand_locs (list of tuples, optional): Locations of known off-axis sources to measure flux. 
            Each tuple should be of the format (sep_pix, pa_degrees). If not, the function
            will look in the header for detections from l4_to_tda.find_source 
    
    Returns:
        result_table (astropy.table.Table): Table containing companion measurements.
    """
    # Measure counts of the host star image
    guess_index = np.unravel_index(np.nanargmax(host_star_image.data), host_star_image.data.shape)
    host_star_peakflux, host_star_fwhm, x_host, y_host = pyklip.fakes.gaussfit2d(host_star_image.data, guess_index[1], guess_index[0], searchrad=7, guessfwhm=3, 
                                                        guesspeak=np.nanmax(host_star_image.data), refinefit=True)
    # host_star_counts, _ = measure_counts(ref_star_dataset[0], phot_method, None, **photometry_kwargs)
    
    # correct host star counts for  for ND if ND cal is provided
    if nd_cal is not None:
        od_val = nd_cal.interpolate_od(x_host, y_host)
        host_star_peakflux /= od_val
    

    # measure peak flux in CT dataset and use as CT=1
    _, _, ct = ct_cal.ct_excam
    max_index = np.argmax(ct)
    ct_max_frame = ct_cal.data[int(max_index)]
    guess_index = np.unravel_index(np.nanargmax(ct_max_frame), ct_max_frame.shape)
    ct_psf_peakflux, ct_psf_fwhm, _, _ = pyklip.fakes.gaussfit2d(ct_max_frame, guess_index[1], guess_index[0], searchrad=7, guessfwhm=3, 
                                                        guesspeak=np.nanmax(ct_max_frame), refinefit=True)
    # reference_psf_counts, _ = measure_counts(ref_psf_min_mask_effect, phot_method, None, **photometry_kwargs)
    
    # flux ratio between host star and CT star to scale CT PSFs
    host_to_ct_psf_ratio = (host_star_peakflux * host_star_fwhm**2) / (ct_psf_peakflux * ct_psf_fwhm**2)

    # Get the star and mask location from the PSF-subtracted image header.
    x_star = psf_sub_image.ext_hdr.get('STARLOCX')
    y_star = psf_sub_image.ext_hdr.get('STARLOCY')


    # determine x/y location of candidates
    if cand_locs is None:
        # Extract companion positions from the PSF-subtracted image and coronagraphic image headers.
        companions = parse_companions(psf_sub_image.ext_hdr)
    else:
        companions = []
        for i, cand in enumerate(cand_locs):
            x_comp = x_star + cand[0] * np.cos(np.radians(cand[1] + 90))
            y_comp = y_star + cand[0] * np.sin(np.radians(cand[1] + 90)) 
            companions.append({"id": i, "y": y_comp, "x": x_comp})

    
    # Create a for the single psf subtraction for API purposes
    psf_sub_dataset = Dataset([psf_sub_image])

    # Measure the flux of each companion
    results = []
    i = 0
    for comp_sub in companions:
        x_psf, y_psf = comp_sub['x'], comp_sub['y']
        x_psf_int = int(np.round(x_psf))
        y_psf_int = int(np.round(y_psf))

        # Compute separation and position angle relative to the star.
        dx, dy = x_psf - x_star, y_psf - y_star
        guesssep = np.hypot(dx, dy)
        guesspa = -np.degrees(np.arctan2(dx, dy)) % 360

        # Get the off-axis PSF at the planet location using the CT calibration file.
        interp_psfs, _, _ = ct_cal.GetPSF(x_psf - x_star, y_psf - y_star, psf_sub_dataset, fpam_fsam_cal)
        nearest_psf = interp_psfs[0]
        scaled_host_psf_at_planet_location = nearest_psf * host_to_ct_psf_ratio
        # measure flux of the star if it was at planet separation from mask (used in measuring flux ratio)
        guess_index = np.unravel_index(np.nanargmax(scaled_host_psf_at_planet_location), scaled_host_psf_at_planet_location.shape)
        host_planet_loc_peakflux, host_planet_loc_fwhm, _, _ = pyklip.fakes.gaussfit2d(scaled_host_psf_at_planet_location, 
                                                        guess_index[1], guess_index[0], searchrad=7, guessfwhm=ct_psf_fwhm, 
                                                        guesspeak=np.nanmax(scaled_host_psf_at_planet_location), refinefit=True)

        # Which approach to use for KLIP throughput correction.
        if thrp_corr == "L4":
            # use the klip throughput stored from PSF subtraction
            # pro: already calculated
            # con: maybe for planets of a different brightness, also must use same photonetry routine as was done for L4 KLIP throughput

            # measure flux at plaent position
            psf_sub_frame = psf_sub_image.data[kl_mode_idx]
            comp_peakflux, comp_fwhm, x_comp, y_comp = pyklip.fakes.gaussfit2d(psf_sub_frame, x_psf, y_psf, searchrad=7, 
                                                                               guessfwhm=host_planet_loc_fwhm, 
                                                                               guesspeak=psf_sub_frame[y_psf_int, x_psf_int], 
                                                                               refinefit=True)
            meas_sep_pix = np.sqrt((x_comp - x_star)**2 + (y_comp - y_star)**2)
            psf_sub_counts = np.pi * comp_peakflux * comp_fwhm**2 / 4. / np.log(2.) # total integrated flux for saving

            # interpolate algorithm throhgput from L4 data product 
            algo_thrp_data = psf_sub_image.hdu_list['KL_THRU'].data
            algo_thrp_seps = algo_thrp_data[0,:,0]
            algo_thrp = algo_thrp_data[1:][kl_mode_idx, :, 0]
            thrp_fwhms = algo_thrp_data[1:][kl_mode_idx, :, 1]

            this_thrp = np.interp(meas_sep_pix, algo_thrp_seps, algo_thrp)
            this_fwhm = np.interp(meas_sep_pix, algo_thrp_seps, thrp_fwhms)

            # correct for KLIP throughput and calculate flux ratio
            companion_host_ratio = (comp_peakflux / this_thrp) / host_planet_loc_peakflux 
        elif thrp_corr == "KLIP-FM":
            # KLIP-FM PSF forward moeling
            # Pro: PSF fit exactly for planet, exact for RDI
            # Con: linear approximation for ADI, not the best for bright planets ad ADI
            raise NotImplementedError("KLIP-FM not yet implemented")
            # #Forward model the off-axis image of the host star if it was at the planet location through the PSF subtraction process
            # kl_value, ct_value, modeled_image = forward_model_psf(
            #     coronagraphic_dataset, ref_star_dataset, ct_cal, scaled_star_psf,
            #     guesssep, guesspa, numbasis=numbasis, nwalkers=nwalkers, nburn=nburn,
            #     nsteps=nsteps, numthreads=numthreads, outputdir=output_dir, plot_results=plot_results,
            #     kl_mode_idx=kl_mode_idx
            # )
            # fm_counts_uncorrected, _ = measure_counts(modeled_image, phot_method, None, **photometry_kwargs)
            # # correct for algorithmic efficiency
            # model_counts = fm_counts_uncorrected / kl_value
            # if verbose == True:
            #     print("Host star if it was at companion ", i, " location forward modeled counts uncorrected: ", fm_counts_uncorrected)
            #     print("Host star if it was at companion ", i, " location forward modeled counts corrected: ", model_counts)
            #     print("Recovered companion ", i, " PSF sub efficiency: ", kl_value)
        elif thrp_corr == "None":
            # Don't do FM
            # Pro: simple, also can use photometry routine of choice
            # Con: doesn't account for KLIP throughput

            # Set default photometry keyword arguments if none are provided.
            photometry_kwargs = photometry_kwargs or get_photometry_kwargs(phot_method)

            # Measure counts in the PSF-subtracted image at the companion location.
            # select the frame from the KL cube
            psf_sub_frame = psf_sub_image.copy()
            psf_sub_frame.data = psf_sub_image.data[kl_mode_idx]
            psf_sub_frame.dq = psf_sub_image.dq[kl_mode_idx]
            psf_sub_frame.err = psf_sub_image.err[:,kl_mode_idx]
            psf_sub_counts, _ = measure_counts(psf_sub_frame, phot_method, (x_psf, y_psf), **photometry_kwargs)
            if verbose == True:
                print("Companion ", i, " coronagraphic, PSF-subtracted counts: ", psf_sub_counts)

            # modeled_image = simplified_psf_sub(scaled_star_psf, ct_cal, guesssep, psf_sub_efficiency)
            # measure the host star brightness. need to stick into an image class
            scaled_host_star_img = host_star_image.copy()
            scaled_host_star_img.data = scaled_host_psf_at_planet_location
            scaled_host_star_img.dq = np.zeros(scaled_host_psf_at_planet_location.shape)
            scaled_host_star_img.err = np.zeros((1,) + scaled_host_psf_at_planet_location.shape)
            host_star_counts, _ = measure_counts(scaled_host_star_img, phot_method, None, **photometry_kwargs)
            if verbose == True:
                print("Host star ", i, " simplified model counts corrected: ", host_star_counts)
        
            companion_host_ratio = psf_sub_counts / host_star_counts
        else:
            raise ValueError("{0} is not a valid option for thrp_corr".format(thrp_corr))

        # Calculate the apparent magnitude based on the host star magnitude or flux calibration.
        if host_star_in_calspec:
            apmag_data = l4_to_tda.determine_app_mag(host_star_image, host_star_image.pri_hdr['TARGET'])
            host_star_apmag = apmag_data[0].ext_hdr['APP_MAG']
            companion_mag = host_star_apmag - 2.5 * np.log10(companion_host_ratio)
        else:
            if fluxcal_factor is None:
                raise ValueError("Provide fluxcal_factor or set host_star_in_calspec=True.")
            zero_point = fluxcal_factor.ext_hdr.get('ZP')
            companion_mag = -2.5 * np.log10(psf_sub_counts) + zero_point

        # Append the results for this companion.
        results.append((
            comp_sub['id'], x_psf, y_psf, psf_sub_counts, companion_host_ratio, companion_mag
        ))
        i+=1

    # Return the results as an Astropy Table.
    return Table(
        rows=results,
        names=[
            'id', 'x', 'y',
            'measured companion counts',
            'counts_ratio',
            'companion estimated mag'
        ]
    )


def parse_companions(ext_hdr):
    """
    Parse companion positions from an image extension header.
    
    Args:
        ext_hdr (dict): Dictionary representing the image extension header.
    
    Returns:
        companions (list): List of dictionaries with keys 'id', 'x', and 'y' for each companion.
    """
    companions = []
    # Iterate over header keys to find those starting with 'SNYX'.
    for key, val in ext_hdr.items():
        if key.startswith('SNYX'):
            parts = val.split(',')
            # Ensure there are at least three parts: id, x, and y.
            if len(parts) >= 3:
                companions.append({"id": key, "y": float(parts[1]), "x": float(parts[2])})
    return companions



def measure_counts(image, phot_method, initial_xy, **kwargs):
    """
    Measure the flux counts in an image using a specified photometry method.
    
    Args:
        image (corgidrp.data.Image): Input image for photometry.
        phot_method (str): Photometry method to use ('aperture' or 'gauss2d').
        initial_xy (tuple or None): Initial (x, y) guess for centroiding.
        kwargs (dict): Arbitrary keyword arguments passed directly to the photometry method
            (e.g., fluxcal.phot_by_gauss2d_fit or fluxcal.aper_phot).
    
    Returns:
        flux (float): Measured flux in the image.
        flux_err (float): Estimated error in the measured flux.
    """
    if phot_method == 'gauss2d':
        flux, flux_err, *_ = fluxcal.phot_by_gauss2d_fit(image, centering_initial_guess=initial_xy, **kwargs)
    elif phot_method == 'aperture':
        flux, flux_err, *_ = fluxcal.aper_phot(image, centering_initial_guess=initial_xy, **kwargs)
    else:
        raise ValueError(f"Invalid photometry method: {phot_method}")
    return flux, flux_err


def get_photometry_kwargs(phot_method):
    """
    Retrieve default photometry keyword arguments based on the specified method.
    
    Args:
        phot_method (str): Photometry method to use ('aperture' or 'gauss2d').
    
    Returns:
        options[phot_method] (dict): Dictionary of default photometry keyword arguments.
    """
    common = {'centering_method': 'xy', 'centroid_roi_radius': 5}
    options = {
        'gauss2d': {'fwhm': 4, 'background_sub': True, 'r_in': 5, 'r_out': 10, **common},
        'aperture': {'encircled_radius': 7, 'frac_enc_energy': 1.0, 'subpixels': 10,
                     'background_sub': True, 'r_in': 5, 'r_out': 10, **common}
    }
    if phot_method not in options:
        raise ValueError(f"Invalid photometry method '{phot_method}'.")
    return options[phot_method]


def simplified_psf_sub(psf_frame, ct_cal, guesssep, efficiency):
    """
    TO DO: add in core throughput efficiencies?
    Perform a simplified PSF subtraction by scaling the PSF frame using a provided efficiency.
    
    Args:
        psf_frame (corgidrp.data.Image): Input PSF image.
        ct_cal (corgidrp.data.CoreThroughputCalibration): Core throughput calibration data.
        guesssep (float): Estimated separation used to determine throughput.
        efficiency (float): PSF subtraction efficiency factor.
    
    Returns:
        (corgidrp.data.Image): Scaled PSF-subtracted image.
        None (NoneType): Placeholder value.
    """
    sub_frame = psf_frame.data * efficiency
    return Image(sub_frame, pri_hdr=psf_frame.pri_hdr, ext_hdr=psf_frame.ext_hdr, err=getattr(psf_frame, 'err', None))



def forward_model_psf(
    coronagraphic_dataset,
    reference_star_dataset,
    ct_calibration,
    scaled_star_psf,
    guesssep,
    guesspa,
    outputdir=".",
    fileprefix="companion_fm",
    numbasis=[1, 2],
    mode="ADI",
    annuli=1,
    subsections=1,
    movement=1,
    inject_norm="sum",
    method="mcmc",
    stamp_size=30,
    fwhm_guess=3.0,
    nwalkers=10,
    nburn=5,
    nsteps=20,
    numthreads=1,
    star_center_err=0.05,
    platescale=21.8,
    platescale_err=0.02,
    pa_offset=0.0,
    pa_uncertainty=0.1,
    plot_results=False,
    kl_mode_idx=-1  # default to the last KL mode
):
    """
    Forward model the PSF for a companion by injecting a normalized PSF into each frame and performing KLIP subtraction.
    
    TODO: this is not fully implemented yet.

    Args:
        coronagraphic_dataset (corgidrp.data.Dataset): Dataset containing coronagraphic images.
        reference_star_dataset (corgidrp.data.Dataset): Dataset containing reference star images.
        ct_calibration (corgidrp.data.CoreThroughputCalibration): Core throughput calibration data.
        scaled_star_psf (corgidrp.data.Image): PSF subtracted image at the location of the companion,
            scaled to the host star's brightness. 
        guesssep (float): Estimated separation of the companion.
        guesspa (float): Estimated position angle (in degrees) of the companion.
        outputdir (str): Directory path to save output files.
        fileprefix (str): File prefix for output files.
        numbasis (list): List of KLIP modes to retain.
        mode (str): PSF subtraction mode (e.g., "ADI").
        annuli (int): Number of annuli to use for subtraction.
        subsections (int): Number of subsections for subtraction.
        movement (int): Movement parameter for subtraction.
        inject_norm (str): Normalization method for PSF injection.
        method (str): Fitting method (e.g., "mcmc").
        stamp_size (int): Size of the stamp used in forward modeling.
        fwhm_guess (float): Initial guess for the FWHM of the PSF.
        nwalkers (int): Number of MCMC walkers.
        nburn (int): Number of burn-in steps for MCMC.
        nsteps (int): Number of MCMC steps.
        numthreads (int): Number of threads for computation.
        star_center_err (float): Error in the star center location.
        platescale (float): Plate scale (e.g., mas/pixel).
        platescale_err (float): Error in the plate scale.
        pa_offset (float): Position angle offset.
        pa_uncertainty (float): Uncertainty in the position angle.
        plot_results (bool): Flag to enable plotting of forward modeling results.
        kl_mode_idx (int): Index of the KL mode to use (must match the one used for PSF subtraction).
    
    Returns:
        kl_value (float): KLIP throughput value extracted from the subtraction.
        ct_value (float): Core throughput value extracted from the subtraction.
        klip_image (corgidrp.data.Image): Final PSF-subtracted image after forward modeling.
    """
    amp = np.nanmax(scaled_star_psf.data)

    if plot_results == True:
        plot_dataset(coronagraphic_dataset, 'Coronagraph dataset', cmap='plasma')

    fm_dataset = coronagraphic_dataset.copy()

    # Inject the normalized PSF into each frame.
    for idx, frame in enumerate(fm_dataset):
        # TO DO: look into why this only works if I do negative guesspa
        injected_frame, _, _ = klip_fm.inject_psf(frame, ct_calibration, amp, 
                                                  guesssep, guesspa)
        fm_dataset[idx].data = injected_frame.data
    
    if plot_results == True:
        plot_dataset(fm_dataset, 'Injected PSF (fm_dataset)', cmap='plasma')

    # Perform PSF subtraction using the l3_to_l4 pipeline.
    fm_psfsub = l3_to_l4.do_psf_subtraction(
        input_dataset=fm_dataset,
        ct_calibration=ct_calibration,
        reference_star_dataset=reference_star_dataset,
        mode=mode,
        annuli=annuli,
        subsections=subsections,
        movement=movement,
        numbasis=numbasis,
        outdir=outputdir,
        fileprefix=fileprefix,
        do_crop=True,
        crop_sizexy=(100, 100),
        measure_klip_thrupt=True,
        measure_1d_core_thrupt=True
    )
    # Get the final frame from the subtraction.
    klip_data = fm_psfsub[0].data[kl_mode_idx]
    klip_thru_hdu = fm_psfsub[0].hdu_list['KL_THRU']
    klip_thru_data = klip_thru_hdu.data
    print("KL throughput", klip_thru_data)
    # Find the index of the closest separation value
    closest_idx = np.abs(klip_thru_data[0] - guesssep).argmin()

    # Get the corresponding algorithm throughput, returned data is
    # (N, n_seps, 2) where:
    # N is 1 plus the number of KL mode truncation choices (the extra “1” is for the reference separations).
    # n_seps is the number of separations sampled.
    # The last dimension (of size 2) holds the pair: (throughput, output FWHM) for each separation.
    separations = klip_thru_data[0, :, 0]  # extract separation values from the header row
    # Find the index of the separation closest to desired_sep:
    closest_idx = np.argmin(np.abs(separations - guesssep))
    kl_throughput_value = klip_thru_data[kl_mode_idx, closest_idx, 0]

    # TO DO: figure out what to do about core throughput, if anything
    ct_value = 1                    # Placeholder value for core throughput.
    ct_data = fm_psfsub[0].data[kl_mode_idx]
    ct_hdu = fm_psfsub[0].hdu_list['CT_THRU']
    ct_data = ct_hdu.data
    print("CT throughput", ct_data)
    klip_image = Image(klip_data, pri_hdr=fm_psfsub[0].pri_hdr, ext_hdr=fm_psfsub[0].ext_hdr)

    if plot_results == True:
        plot_dataset(klip_image, 'PSF-Subtracted (fm_psfsub)', cmap='plasma')

    #TO DO: don't hardcode this
    comp_keywords = [key for key in fm_psfsub[0].ext_hdr if key.startswith("SNYX")]
    for key in comp_keywords:
        klip_image = update_companion_location_in_cropped_image(klip_image, key, (512, 512), (50, 50))

    return kl_throughput_value, ct_value, klip_image



def update_companion_location_in_cropped_image(image, comp_keyword, old_host, new_host):
    """
    Update the companion location in an image header after cropping, based on new host coordinates.
    
    Args:
        image (corgidrp.data.Image): Image containing companion location in its header.
        comp_keyword (str): Header keyword that holds the companion location.
        old_host (tuple): Original (x, y) coordinates of the host star.
        new_host (tuple): New (x, y) coordinates of the host star in the cropped image.
    
    Returns:
        image (corgidrp.data.Image): Image with updated companion location in the header.
    """
    ext_hdr = image.ext_hdr
    if comp_keyword not in ext_hdr:
        raise KeyError(f"Keyword {comp_keyword} not found in image header.")
    parts = ext_hdr[comp_keyword].split(',')
    if len(parts) < 3:
        raise ValueError(f"Unexpected format for companion location in {comp_keyword}: {ext_hdr[comp_keyword]}")
    # Since header format is "SNR,y,x", parse accordingly.
    sn_val = float(parts[0])
    old_comp_y = float(parts[1])
    old_comp_x = float(parts[2])
    # Update: host coordinates are (x,y) so use old_host[0] for x and old_host[1] for y.
    new_comp_y = int(round(old_comp_y - old_host[1] + new_host[1]))
    new_comp_x = int(round(old_comp_x - old_host[0] + new_host[0]))
    ext_hdr[comp_keyword] = f"{sn_val:5.1f},{new_comp_y:4d},{new_comp_x:4d}"
    return image



def extract_single_frame(image, frame_index=0):
    """
    Extract a single frame from a multi-frame image, including associated error and DQ arrays.
    
    Args:
        image (corgidrp.data.Image): Multi-frame image object.
        frame_index (int): Index of the frame to extract.
    
    Returns:
        new_image (corgidrp.data.Image): Extracted single frame image with error and DQ information.
    """
    data_frame = image.data[frame_index, :, :]
    # Handle error array extraction based on its dimensions.
    if image.err is not None:
        err_frame = image.err[0, frame_index, :, :] if image.err.ndim == 4 else image.err[frame_index, :, :]
    else:
        err_frame = None
    dq_frame = image.dq[frame_index, :, :] if image.dq is not None else None
    new_image = Image(data_frame, pri_hdr=image.pri_hdr, ext_hdr=image.ext_hdr, err=err_frame, dq=dq_frame)
    # Preserve specific header units if present.
    if hasattr(image, 'hdu_list'):
        new_image.hdu_list = {key: image.hdu_list[key] for key in image.hdu_list if key in ['KL_THRU', 'CT_THRU']}
    return new_image


def plot_dataset(dataset, title_prefix, cmap='plasma'):
    """
    Plots frames from a dataset in a grid layout using ZScale normalization.
    
    Function accepts either:
      - A Dataset object with a 'frames' attribute,
      - A list or tuple of frame objects (each having a 'data' attribute), or
      - A single frame object.
    
    Args:
        dataset (Dataset, list, tuple, or object): The input dataset or frame(s).
        title_prefix (str): Title prefix for each subplot.
        cmap (str): Colormap to use for the image display.
    """
    # If dataset is a Dataset object (assumed to have a 'frames' attribute), extract the frames.
    if hasattr(dataset, 'frames'):
        frames = dataset.frames
    # If dataset is not a list/tuple (i.e. a single frame), wrap it in a list.
    elif not isinstance(dataset, (list, tuple)):
        frames = [dataset]
    else:
        frames = dataset

    num_frames = len(frames)
    ncols = int(np.ceil(np.sqrt(num_frames)))
    nrows = int(np.ceil(num_frames / ncols))
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = np.array(axs).flatten()
    
    for i, frame in enumerate(frames):
        data = frame.data
        data_max = data.max()
        # Use ZScaleInterval to compute the limits.
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(data)
        
        im = axs[i].imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i].set_title(f'{title_prefix} Frame {i} (max: {data_max:.2e})')
        plt.colorbar(im, ax=axs[i])
    
    # Turn off any unused subplots.
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    
    fig.suptitle(f'{title_prefix} Frames', fontsize=16)
    plt.tight_layout()
    plt.show()