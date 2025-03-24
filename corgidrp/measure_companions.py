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


def measure_companions(
    coronagraphic_dataset,
    ref_star_dataset,
    psf_sub_image,
    ref_psf_min_mask_effect,
    ct_cal,
    FpamFsamCal,
    phot_method='aperture',
    photometry_kwargs=None,
    fluxcal_factor=None,
    host_star_in_calspec=True,
    forward_model=False,
    numbasis=[1, 2],
    nwalkers=10,
    nburn=5,
    nsteps=20,
    numthreads=1,
    output_dir=".",
    verbose=True
):
    """
    Measure companion properties in a coronagraphic image and return a table with companion position, flux ratio, and apparent magnitude.
    
    Args:
        coronagraphic_dataset (corgidrp.data.Dataset): Dataset containing coronagraphic images.
        ref_star_dataset (corgidrp.data.Dataset): Dataset containing reference star images.
        psf_sub_image (corgidrp.data.Image): PSF-subtracted image with companions.
        ref_psf_min_mask_effect (corgidrp.data.Image): Reference PSF image with minimal mask effect.
        ct_cal (corgidrp.data.CoreThroughputCalibration): Core throughput calibration data.
        FpamFsamCal (corgidrp.Data.FpamFsamCal): Transformation calibration data.
        phot_method (str): Photometry method to use ('aperture' or 'gauss2d').
        photometry_kwargs (dict): Dictionary of keyword arguments for photometry.
        fluxcal_factor (corgidrp.Data.FluxcalFactor): Flux calibration factor object.
        host_star_in_calspec (bool): Flag indicating whether to use host star magnitude from calspec.
        forward_model (bool): Flag to enable forward-modeling for flux estimation.
        numbasis (list): List of KLIP modes to retain.
        nwalkers (int): Number of MCMC walkers for forward-modeling.
        nburn (int): Number of burn-in steps for MCMC.
        nsteps (int): Number of MCMC steps.
        numthreads (int): Number of threads for MCMC computation.
        output_dir (str): Output directory path for forward-modeling results.
        verbose (bool): Flag to enable verbose output.
    
    Returns:
        result_table (astropy.table.Table): Table containing companion measurements.
    """
    # Set default photometry keyword arguments if none are provided.
    photometry_kwargs = photometry_kwargs or get_photometry_kwargs(phot_method)

    # Measure counts of the host star and reference PSF from the provided images.
    # TO DO: correct for ND filter here
    host_star_counts, _ = measure_counts(ref_star_dataset[0], phot_method, None, **photometry_kwargs)
    reference_psf_counts, _ = measure_counts(ref_psf_min_mask_effect, phot_method, None, **photometry_kwargs)
    host_to_ref_psf_ratio = host_star_counts / reference_psf_counts

    # Extract companion positions from the PSF-subtracted image and coronagraphic image headers.
    companions = parse_companions(psf_sub_image.ext_hdr)
    companions_coron = parse_companions(coronagraphic_dataset[0].ext_hdr)
    
    # Get the star location from the PSF-subtracted image header.
    x_star = psf_sub_image.ext_hdr.get('STARLOCX')
    y_star = psf_sub_image.ext_hdr.get('STARLOCY')
    
    # Create a dataset of calibration images from the core throughput calibration.
    ct_dataset = ct_dataset_from_cal(ct_cal)
    results = []

    # Process each companion by comparing positions in the PSF-subtracted and coronagraphic images.
    for comp_sub, comp_coron in zip(companions, companions_coron):
        x_psf, y_psf = comp_sub['x'], comp_sub['y']
        x_coron, y_coron = comp_coron['x'], comp_coron['y']
        
        # Compute separation and position angle relative to the star.
        dx, dy = x_psf - x_star, y_psf - y_star
        guesssep = np.hypot(dx, dy)
        guesspa = np.degrees(np.arctan2(dx, dy)) % 360

        # Get the calibration frame based on the companion's position.
        _, nearest_frame = measure_core_throughput_at_location(x_psf, y_psf, x_star, y_star, ct_cal, ct_dataset)
        
        # Scale the calibration PSF to the host star flux level.
        scaled_star_psf = Image(
            data_or_filepath=nearest_frame.data * host_to_ref_psf_ratio,
            pri_hdr=nearest_frame.pri_hdr,
            ext_hdr=nearest_frame.ext_hdr,
            err=getattr(nearest_frame, 'err', None)
        )

        # Measure counts in the PSF-subtracted image at the companion location.
        psf_sub_counts, _ = measure_counts(psf_sub_image, phot_method, (x_psf, y_psf), **photometry_kwargs)
        # Measure counts in the original coronagraphic image at the companion location.
        companion_pre_sub_counts, _ = measure_counts(coronagraphic_dataset[0], phot_method, (x_coron, y_coron), **photometry_kwargs)
        psf_sub_efficiency = psf_sub_counts / companion_pre_sub_counts

        # Use forward-modeling or a simplified subtraction approach to model the PSF and do PSF-subtraction.
        if forward_model:
            #Forward model the off-axis image of the host star if it was at the planet location through the PSF subtraction process
            kl_value, ct_value, modeled_image = forward_model_psf(
                coronagraphic_dataset, ref_star_dataset, ct_cal, scaled_star_psf,
                guesssep, guesspa, numbasis=numbasis, nwalkers=nwalkers, nburn=nburn,
                nsteps=nsteps, numthreads=numthreads, outputdir=output_dir
            )
            fm_counts_uncorrected, _ = measure_counts(modeled_image, phot_method, None, **photometry_kwargs)
            # correct for algorithmic efficiency
            fm_counts = fm_counts_uncorrected / kl_value
        else:
            modeled_image = simplified_psf_sub(scaled_star_psf, ct_cal, guesssep, psf_sub_efficiency)
            fm_counts, _ = measure_counts(modeled_image, phot_method, None, **photometry_kwargs)
        
        companion_host_ratio = psf_sub_counts / fm_counts

        # Calculate the apparent magnitude based on the host star magnitude or flux calibration.
        if host_star_in_calspec:
            apmag_data = l4_to_tda.determine_app_mag(ref_star_dataset[0], ref_star_dataset[0].pri_hdr['TARGET'])
            host_star_apmag = apmag_data[0].ext_hdr['APP_MAG']
            companion_mag = host_star_apmag - 2.5 * np.log10(companion_host_ratio)
        else:
            if fluxcal_factor is None:
                raise ValueError("Provide fluxcal_factor or set host_star_in_calspec=True.")
            zero_point = fluxcal_factor.ext_hdr.get('ZP')
            companion_mag = -2.5 * np.log10(psf_sub_counts) + zero_point

        # Append the results for this companion.
        results.append((
            comp_sub['id'], x_psf, y_psf,
            psf_sub_counts, fm_counts, companion_host_ratio, companion_mag
        ))

    # Return the results as an Astropy Table.
    return Table(
        rows=results,
        names=[
            'id', 'x', 'y',
            'measured companion counts',
            'simulated host star counts',
            'counts_ratio',
            'mag'
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
                companions.append({"id": key, "x": float(parts[1]), "y": float(parts[2])})
    return companions


def ct_dataset_from_cal(ct_cal):
    """
    Create a dataset from a core throughput calibration object.
    
    Args:
        ct_cal (corgidrp.data.CoreThroughputCalibration): Core throughput calibration data 
            containing a PSF cube.
    
    Returns:
        dataset (corgidrp.data.Dataset): Dataset consisting of individual calibration PSF images.
    """
    psf_cube = ct_cal.data
    # Create an Image object for each slice of the PSF cube.
    psf_images = [
        Image(psf_cube[i], pri_hdr=ct_cal.pri_hdr, ext_hdr=ct_cal.ext_hdr)
        for i in range(psf_cube.shape[0])
    ]
    return Dataset(psf_images)


def measure_counts(image, phot_method, initial_xy, **kwargs):
    """
    Measure the flux counts in an image using a specified photometry method.
    
    Args:
        image (corgidrp.data.Image): Input image for photometry.
        phot_method (str): Photometry method to use ('aperture' or 'gauss2d').
        initial_xy (tuple or None): Initial (x, y) guess for centroiding.
        kwargs(dict): Arbitrary keyword arguments passed directly to the photometry method
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


def lookup_core_throughput(ct_cal, desired_sep):
    """
    Lookup the core throughput value closest to the desired separation.
    
    Args:
        ct_cal (corgidrp.data.CoreThroughputCalibration): Core throughput calibration data 
            containing ct_excam and header info.
        desired_sep (float): Desired separation in pixels.
    
    Returns:
        closest_sep (float): Separation value closest to desired_sep.
        idx (int): Index of the closest separation value.
        throughput (float): Core throughput value corresponding to the closest separation.
    """
    x, y, ct = ct_cal.ct_excam
    mask_x = ct_cal.ext_hdr['MASKLOCX']
    mask_y = ct_cal.ext_hdr['MASKLOCY']
    # Compute separations from the mask center.
    separations = np.sqrt((x - mask_x)**2 + (y - mask_y)**2)
    idx = np.argmin(np.abs(separations - desired_sep))
    return separations[idx], idx, ct[idx]


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
    plot_results=False
):
    """
    Forward model the PSF for a companion by injecting a normalized PSF into each frame and performing KLIP subtraction.
    
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
    
    Returns:
        kl_value (float): KLIP throughput value extracted from the subtraction.
        ct_value (float): Core throughput value extracted from the subtraction.
        klip_image (corgidrp.data.Image): Final PSF-subtracted image after forward modeling.
    """
    amp = np.nanmax(scaled_star_psf.data)

    # Debugging plotting:
    #plot_dataset(coronagraphic_dataset, 'Coronagraph dataset', cmap='plasma')

    fm_dataset = coronagraphic_dataset.copy()

    # Inject the normalized PSF into each frame.
    for idx, frame in enumerate(fm_dataset):
        # TO DO: look into why this only works if I do negative guesspa
        injected_frame, _, _ = klip_fm.inject_psf(frame, ct_calibration, amp, 
                                                  guesssep, -guesspa)
        fm_dataset[idx].data = injected_frame.data
    
    # Debugging plotting:
    #plot_dataset(fm_dataset, 'Injected PSF (fm_dataset)', cmap='plasma')

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
    klip_data = fm_psfsub[0].data[-1]
    klip_thru_hdu = fm_psfsub[0].hdu_list['KL_THRU']
    klip_thru_data = klip_thru_hdu.data
    # Find the index of the closest separation value
    closest_idx = np.abs(klip_thru_data[0] - guesssep).argmin()

    # Get the corresponding throughput from the last row
    kl_throughput_value = klip_thru_data[-1, closest_idx]
    # TO DO: figure out what to do about core throughput, if anything
    ct_value = 1                    # Placeholder value for core throughput.
    klip_image = Image(klip_data, pri_hdr=fm_psfsub[0].pri_hdr, ext_hdr=fm_psfsub[0].ext_hdr)
    # Update companion location in the cropped image header.
    comp_keyword = next(key for key in fm_psfsub[0].ext_hdr if key.startswith("SNYX"))

    # Debugging plotting: Plot the PSF-subtracted dataset (fm_psfsub)
    #plot_dataset(klip_image, 'PSF-Subtracted (fm_psfsub)', cmap='plasma')

    #TO DO: don't hardcode this, ideally you can use masklocx and y
    klip_image = update_companion_location_in_cropped_image(klip_image, comp_keyword, (512, 512), (50, 50))
    return kl_throughput_value, ct_value, klip_image


def measure_core_throughput_at_location(x_c, y_c, x_star, y_star, ct_cal, ct_dataset):
    """
    Measure the core throughput at the location of the companion relative to the star.
    
    Args:
        x_c (float): x-coordinate of the companion.
        y_c (float): y-coordinate of the companion.
        x_star (float): x-coordinate of the star.
        y_star (float): y-coordinate of the star.
        ct_cal (corgidrp.data.CoreThroughputCalibration): Core throughput calibration data.
        ct_dataset (corgidrp.data.Dataset): Dataset of calibration PSF images.
    
    Returns:
        throughput (float): Core throughput value at the companion location.
        corresponding_frame (corgidrp.data.Image): Calibration frame corresponding to the throughput.
    """
    dx, dy = x_c - x_star, y_c - y_star
    guesssep = np.hypot(dx, dy)
    _, idx, throughput = lookup_core_throughput(ct_cal, guesssep)
    corresponding_frame = list(ct_dataset)[idx]
    return throughput, corresponding_frame


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
    sn_val, old_comp_x, old_comp_y = float(parts[0]), float(parts[1]), float(parts[2])
    new_comp_x = int(round(old_comp_x - old_host[0] + new_host[0]))
    new_comp_y = int(round(old_comp_y - old_host[1] + new_host[1]))
    ext_hdr[comp_keyword] = f"{sn_val:5.1f},{new_comp_x:4d},{new_comp_y:4d}"
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
    
    For each frame, the ZScale algorithm determines suitable limits (vmin and vmax)
    to enhance the contrast in the image, similar to ds9's zscale.
    
    This function accepts either:
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