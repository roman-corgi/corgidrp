import os
import numpy as np
from astropy.table import Table
import corgidrp.fluxcal as fluxcal
import pyklip.fm as fm
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fitpsf as fitpsf
from corgidrp.data import PyKLIPDataset, Image
import numpy as np
import corgidrp.fluxcal as fluxcal
import corgidrp.l4_to_tda as l4_to_tda    
import astropy.io.fits as fits


def measure_companions(
    direct_star_image,
    psf_sub_image,
    reference_psf,
    ct_dataset,
    ct_cal,
    phot_method='aperture',
    coronagraphic_dataset=None,
    fluxcal_factor=None,
    FpamFsamCal = None,
    host_star_in_calspec=True,
    forward_model=False,
    output_dir = ".",
    verbose=True
):
    """
    Measure companion properties in a final (or intermediate) coronagraphic image,
    returning position, counts ratio (companion/starlight), and apparent magnitude.

    This function does the following:
      1) Measure host star counts in a direct image.
      2) Measure reference PSF counts (from off-axis calibration).
      3) Compute ratio for scaling the off-axis PSF to represent the star at companion's location.
      4) (Optional) Forward-model off-axis star image through PSF subtraction if method= 'forward_model'.
      5) Measure the companion count ratio & apparent magnitude.

    Args:
        direct_star_image (corgidrp.data.Image): A direct (unocculted) image of the host star (in e-). 
        psf_sub_image (corgidrp.data.Image): The final (or intermediate) PSF-subtracted image with companions
            in e-, with .data, .hdr, .ext_hdr.
        reference_psf (corgidrp.data.Image): An off-axis calibration PSF of a star at some known 
            separation where mask effects are negligible (it is near 6 lam/D).
        ct_dataset (corgidrp.data.Dataset, optional): A dataset containing some coronagraphic observations.
        ct_cal (corgidrp.data.CoreThroughputCalibration, optional): A Core Throughput calibration file containing 
            PSFs at different distances from the mask and their corresponding positions and throughputs.
        phot_method ({'aperture', 'gauss2d'}, optional): Photometry method for measuring 
            companion counts on the (already) PSF-subtracted image.
        coronagraphic_dataset (corgidrp.data.Dataset, optional): Coronagraphic images with companions. 
        fluxcal_factor (corgidrp.Data.FluxcalFactor, optional): Fluxcal factor with .ext_hdr['ZP'].
        FpamFsamCal (corgidrp.data.FpamFsamCal, optional): FPAM to EXCAM transformation matrix.
        host_star_in_calspec (bool, optional):
        forward_model (bool, optional):
        output_dir (str, optional): Output directory path.
        verbose (bool, optional): If True, print progress messages.

    Returns:
        result_table (astropy.table.Table):A table with the following columns (for checking now, replace later
            with regular returns):
                - id : companion label
                - x, y : location in pixels
                - counts_raw : companion counts in e- (post-subtraction)
                - counts_err : estimated counts uncertainty
                - counts_corr : counts after throughput correction
                - counts_ratio : counts_corr / host_star_counts (if host_star_counts is given)
                - mag : apparent magnitude (computed via host_star_apmag or zero point or fluxcal_factor)
    """
    phot_method_kwargs = get_photometry_kwargs(phot_method)

    # i. Select reference PSF: assume off-axis PSF with highest CT has negligible effect from the mask 
    # (it is near 6 lam/D see Figs. 9 and 11 in John Krist's paper), or use simulated PSF w/o FPM masks
   
    # ii. Measure host star counts from direct image 
    host_star_counts, _ = measure_counts(direct_star_image, phot_method, None, **phot_method_kwargs)

    # iii. Measure reference PSF counts. An off-axis calibration PSF of a star at some known 
    #    separation where mask effects are negligible
    reference_psf_counts, _ = measure_counts(reference_psf, phot_method, None, **phot_method_kwargs)

    # iv. Compute the ratio of photometry_host_star/photometry_reference_PSF
    host_psf_ratio = host_star_counts/reference_psf_counts

    ### Parse companion positions from headers
    companions = [{"id": key, "x": float(x), "y": float(y)}
                  for key, val in psf_sub_image.ext_hdr.items() if key.startswith('SNYX')
                  for _, x, y in [val.split(',')]]

    if verbose: print(f"Found {len(companions)} companion(s) in header.")

    ### Loop over companions 
    results = []
    for comp in companions:
        x_c, y_c = comp['x'], comp['y']

        # v. Get/interpolate the off-axis PSF at the separation of the companion
        throughput_factor, nearest_frame = measure_core_throughput_at_location(x_c, y_c, ct_cal, 
                                                                               ct_dataset, FpamFsamCal)
        
        # vi. Scale the off-axis PSF by the computed ratio above to get the off-axis PSF of the host 
        # star if it was at the same position as the companion
        scaled_psf_data = nearest_frame.data * host_psf_ratio  

        ## Create a new Image object with the same headers and error maps (if applicable)
        scaled_psf_at_companion_loc = Image(
            data_or_filepath=scaled_psf_data, 
            pri_hdr=nearest_frame.pri_hdr, 
            ext_hdr=nearest_frame.ext_hdr, 
            err=nearest_frame.err if hasattr(nearest_frame, 'err') else None
            )

        # vii. Do PSF subtraction for the scaled_psf_at_companion_location, either though forward modeling
        # or through classical PSF subtraction
        
        if forward_model == True:
            # Get star location from headers
            x_star = scaled_psf_at_companion_loc.ext_hdr.get('STARLOCX', None)
            y_star = scaled_psf_at_companion_loc.ext_hdr.get('STARLOCY', None)
            guesssep = np.sqrt((x_c - x_star) ** 2 + (y_c - y_star) ** 2)
            guesspa = np.degrees(np.arctan2(x_c - x_star, y_star - y_c)) % 360  # Ensures PA is [0, 360] degrees
            psf_guess_counts, _ = measure_counts(scaled_psf_at_companion_loc, phot_method, None, 
                                                                 **phot_method_kwargs)

            print("checking star loc", x_star, y_star)

            fit = forward_model_psf(coronagraphic_dataset, scaled_psf_at_companion_loc, guesssep, guesspa,
                                    psf_guess_counts, outputdir=output_dir)
            print("forward model fit", fit)
        else:
            psf_sub_star_at_companion_loc, efficiency = classical_psf_sub(scaled_psf_at_companion_loc)

        # viii. Measure flux ratio of companion to host star
        psf_sub_star_at_companion_loc_counts, _ = measure_counts(psf_sub_star_at_companion_loc, 
                                                                 phot_method, None, **phot_method_kwargs)
        psf_sub_companion_counts, _ = measure_counts(psf_sub_image, phot_method, (x_c, y_c), 
                                                                 **phot_method_kwargs)
        
        companion_host_ratio = psf_sub_companion_counts / psf_sub_star_at_companion_loc_counts

        # ix. Calculate companion's apparent magnitude
        if host_star_in_calspec == True:
            apmag_data = l4_to_tda.determine_app_mag(direct_star_image, direct_star_image.pri_hdr['TARGET'])
            host_star_apmag = apmag_data[0].ext_hdr['APP_MAG']
            companion_mag = host_star_apmag - 2.5 * np.log10(companion_host_ratio)
        else:
            zero_point = fluxcal_factor.ext_hdr.get('ZP') if fluxcal_factor else None
            counts_corr = psf_sub_companion_counts # TO DO: look at Commit 3de7490 for how to correct these counts
            companion_mag = -2.5 * np.log10(counts_corr) + zero_point

        results.append((comp['id'], x_c, y_c, psf_sub_companion_counts, psf_sub_star_at_companion_loc_counts, companion_host_ratio, companion_mag))

    return Table(rows=results, names=['id', 'x', 'y', 'psf_sub_companion_counts', 'psf_sub_host_star_at_companion_loc_counts',
                                      'counts_ratio', 'mag'])


# ---------------- Helper functions -------------------------------------- #

def measure_counts(input_image_or_dataset, phot_method, initial_xy_guess, **kwargs):
    """
    Measure counts in an image using the specified method.

    Args:
        input_image_or_dataset (corgidrp.data.Image or corgidrp.data.Dataset):
            The image or dataset where the companion exists.
        phot_method (str): The measurement method ('forward_model', 'psf_fit', or 'aperture').
        initial_xy_guess (tuple): Initial (x, y) guess for the companion location.
        kwargs (dict): Additional method-specific parameters.

    Returns:
        counts_val (float): Estimated counts of the companion.
        counts_err (float): Estimated counts uncertainty.
    """

    if phot_method == 'gauss2d':
        return fluxcal.phot_by_gauss2d_fit(input_image_or_dataset, centering_initial_guess=initial_xy_guess, **kwargs)[:2]

    elif phot_method == 'aperture':
        return fluxcal.aper_phot(input_image_or_dataset, centering_initial_guess=initial_xy_guess, **kwargs)[:2]

    raise ValueError(f"Invalid photometry method: {phot_method}")


def get_photometry_kwargs(phot_method):
    """
    Returns method-specific keyword arguments for different photometry methods.

    Args:
        phot_method (str): The photometry method to be used. Options are:
            - 'gauss2d': Uses a 2D Gaussian fit.
            - 'aperture': Uses aperture photometry.

    Returns:
        photometry_options[phot_method] (dict): A dictionary containing the appropriate 
            keyword arguments for the chosen photometry method.
    """
    common_kwargs = {'centering_method': 'xy', 'centroid_roi_radius': 5}

    photometry_options = {
        'gauss2d': {'fwhm': 3, 'background_sub': True, 'r_in': 5, 'r_out': 10, **common_kwargs},
        'aperture': {'encircled_radius': 4, 'frac_enc_energy': 1, 'subpixels': 5, 'background_sub': True,
                     'r_in': 6, 'r_out': 12, **common_kwargs}
    }

    if phot_method not in photometry_options:
        raise ValueError(f"Invalid photometry method '{phot_method}'. Choose from {list(photometry_options.keys())}.")

    return photometry_options[phot_method]


def forward_model_psf(
    coronagraphic_dataset,
    off_axis_psf,
    guesssep,
    guesspa,
    guessflux,
    outputdir=".",
    fileprefix="companion-fm",
    numbasis=[1, 7, 100],
    method="mcmc",
    annulus_halfwidth=15,   # half-width (in pixels) of annulus centered on guesssep
    movement=4,             # exclusion criterion in KLIP
    stamp_size=13,          # size of fitting stamp
    corr_len_guess=3.0,     # initial guess for correlation length
    x_range=1.5,            # prior range for X offset
    y_range=1.5,            # prior range for Y offset
    flux_range=20.0,         # prior range for flux
    corr_len_range=1.0,     # log10 prior range for correlation length
    # If method == "mcmc":
    nwalkers=100,
    nburn=200,
    nsteps=800,
    numthreads=1,
    # Calibration error propagation (placeholders)
    star_center_err=0.05,   # in pixels
    platescale=21.8,        # mas/pixel 
    platescale_err=0.02,    # mas/pixel
    pa_offset=0.0,          # known absolute offset in degrees
    pa_uncertainty=0.1      # PA calibration uncertainty in degrees
):
    """
    Forward model an off-axis PSF through KLIP-FM to extract astrometry/photometry 
    of a known companion in a coronagraphic dataset.

    Args:
        coronagraphic_dataset (corgidrp.Dataset): The dataset containing coronagraphic images 
            (with known companion(s)). 
        off_axis_psf (corgidrp.Image): The off-axis PSF image (scaled to approximate the companion 
            flux). This will serve as the 'template' for the forward-modeled PSF.
        guesssep (float): Initial guess at companion separation (in pixels).
        guesspa (float): Initial guess at companion position angle (in degrees).
        guessflux (float): Initial guess at companion flux scaling (dimension depends on how
            off_axis_psf has been normalized).
        outputdir (str): Directory to store intermediate KLIP-FM outputs (optional).
        fileprefix (str): Prefix for output files.
        numbasis (list of int): List of KL basis sizes to use.
        method ({"mcmc", "maxl"}): “mcmc” = Bayesian (emcee) MCMC fit,
            “maxl” = frequentist maximum-likelihood fit
        annulus_halfwidth (float): Half-width of the annulus (in pixels) around `guesssep` in 
            which KLIP will be performed.
        movement (float): Angular exclusion criterion for PSF subtraction in pyKLIP.
        stamp_size (int): Size of the extracted stamp (in pixels) around the companion for 
            fitting.
        corr_len_guess (float): Initial guess of the Gaussian Process correlation length.
        x_range (float): Bounds for x/y offsets in linear space (uniform prior).
        y_range (float): Bounds for x/y offsets in linear space (uniform prior).
        flux_range(float): Bounds for log10 priors on flux length.
        corr_len_range (float): Bounds for log10 priors on correlation length.
        nwalkers (int): MCMC hyperparameters if method="mcmc".
        nburn (int): MCMC hyperparameters if method="mcmc".
        nsteps (int): MCMC hyperparameters if method="mcmc".
        numthreads (int): Parallel threads for MCMC.
        star_center_err (float): Uncertainty on star center (in pixels), for post-fit error 
            propagation.
        platescale (float): Plate scale in mas/pixel, for converting pixel offsets to mas.
        platescale_err (float): Plate scale uncertainty in mas/pixel.
        pa_offset (float): Known absolute offset in instrument position angle (in degrees).
        pa_uncertainty (float): Additional systematic uncertainty in position angle (in degrees).

    Returns:
        fit (pyklip.fitpsf.FMAstrometry): The fitting object containing best-fit parameters, 
            chains (if MCMC),and raw/final astrometry with uncertainties.
    """
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # 1) Wrap coronagraphic_dataset in PyKLIP dataset
    coronagraphic_dataset = PyKLIPDataset(coronagraphic_dataset, highpass=False)

    # 2) Prepare the 'psfs' argument for FMPlanetPSF
    #    off_axis_psf might be 2D or 3D (if multi-wavelength)?
    psf_array = off_axis_psf.data 
    dataset_shape = coronagraphic_dataset.input.shape 
    dataset_shape = coronagraphic_dataset.input.shape[-2:]

    # May (or may not) need multiple wavelength channels:
    # For now assume single-channel data:
    # Otherwise, replace wvs=[some_array_of_wavelengths].
    wvs = [1.0]  # placeholder single wavelength

    if psf_array.data.ndim == 2:
        psf_4d = psf_array[np.newaxis, np.newaxis, :, :]
    else:
        psf_4d = psf_array

    # 3) Initialize the Forward Model Planet PSF class
    fm_class = fmpsf.FMPlanetPSF(dataset_shape, np.array(numbasis), guesssep,
        guesspa, guessflux, psf_4d, wvs, star_spt=None, spectrallib=None)

    # 4) Run KLIP-FM on the coronagraphic dataset
    annuli = [[guesssep - annulus_halfwidth, guesssep + annulus_halfwidth]]

    fm.klip_dataset(coronagraphic_dataset, fm_class, outputdir=outputdir,
        fileprefix=fileprefix, numbasis=numbasis, annuli=annuli, subsections=1,
        padding=0, movement=movement)

    # 5) Read the forward-modeled results and the PSF-subtracted data
    fm_filename = os.path.join(outputdir, f"{fileprefix}-fmpsf-KLmodes-all.fits")
    klip_filename = os.path.join(outputdir, f"{fileprefix}-klipped-KLmodes-all.fits")

    kl_index = 1

    fm_hdu = fits.open(fm_filename)
    data_hdu = fits.open(klip_filename)

    fm_frame = fm_hdu[0].data[kl_index]
    fm_centx = fm_hdu[0].header['PSFCENTX']
    fm_centy = fm_hdu[0].header['PSFCENTY']

    data_frame = data_hdu[0].data[kl_index]
    data_centx = data_hdu[0].header["PSFCENTX"]
    data_centy = data_hdu[0].header["PSFCENTY"]

    # these are the guessed values from the FM header
    guesssep = fm_hdu[0].header['FM_SEP']
    guesspa  = fm_hdu[0].header['FM_PA']

    fm_hdu.close()
    data_hdu.close()

    # 6) Build the FMAstrometry fitting object
    #fit = fitpsf.FMAstrometry(guesssep, guesspa, stamp_size, method=method)

    fit = fitpsf.FMAstrometry(guesssep, guesspa, stamp_size, method=method)

    # Generate FM stamp (the forward-modeled PSF for the companion)
    fit.generate_fm_stamp(fm_frame, [fm_centx, fm_centy], padding=5)

    # Generate data stamp (the PSF-subtracted image of the companion)
    fit.generate_data_stamp(data_frame, [data_centx, data_centy], 
                            dr=4,               # radial annulus for local noise estimate
                            exclusion_radius=10 # exclude actual companion location
    )

    # 7) Setup a Gaussian Process kernel
    fit.set_kernel("matern32", [corr_len_guess], [r"$l$"])

    # 8) Set up prior bounds (or parameter bounds for max-likelihood)
    fit.set_bounds(x_range, y_range, flux_range, [corr_len_range])

    # 9) Perform the fit
    if method == "mcmc":
        fit.fit_astrometry(nwalkers=nwalkers, nburn=nburn, nsteps=nsteps, numthreads=numthreads)
    else:
        fit.fit_astrometry()


    # 10) Propagate calibration uncertainties (star center, platescale, etc.)
    fit.propogate_errs(
        star_center_err=star_center_err,
        platescale=platescale,
        platescale_err=platescale_err,
        pa_offset=pa_offset,
        pa_uncertainty=pa_uncertainty
    )

    return fit


def measure_core_throughput_at_location(
    x_c, y_c,
    ct_cal,
    ct_dataset,
    FpamFsamCal
):
    """
    Get the core throughput factor from a loaded CoreThroughputCalibration object
    at the companion location (x_c, y_c) and return the corresponding frame.

    Args:
        x_c (float): The companion's pixel x location in the coronagraphic image.
        y_c (float): The companion's pixel y location in the coronagraphic image.
        ct_dataset (corgidrp.data.Dataset): The dataset used for the coronagraphic 
            observation, or at least the first frame's header info.
        ct_cal (corgidrp.data.CoreThroughputCalibration): The loaded calibration file with 
            PSF basis and arrays of throughput.
        FpamFsamCal (corgidrp.data.FpamFsamCal): The object to help map FPAM changes to 
            EXCAM pixel offsets.

    Returns:
        throughput_factor (float): The core throughput factor at the specified location 
            (dimensionless).
        nearest_frame (corgidrp.data.Image): The frame from the dataset corresponding 
            to the nearest neighbor PSF location.
    """
    # Get the star center position (from calibration - only use the first returned center)
    star_center = ct_cal.GetCTFPMPosition(ct_dataset, FpamFsamCal)[0]
    dx, dy = x_c - star_center[0], y_c - star_center[1]
    
    # Extract PSF coordinates and throughput values from the calibration file.
    # ct_cal.ct_excam is expected to have shape (3, N) where N is the number of PSF images.
    xyvals = ct_cal.ct_excam[:2]    # first two rows: x and y coordinates
    throughvals = ct_cal.ct_excam[2]  # third row: throughput measurements
    
    # Find index of the nearest PSF location in the calibration measurements.
    idx_best = int(np.argmin(np.hypot(xyvals[0] - dx, xyvals[1] - dy)))
    throughput_factor = max(throughvals[idx_best], 1.0)
    
    # Assume ct_dataset is already a Dataset of PSF images in the same order as ct_cal.ct_excam.
    nearest_frame = ct_dataset[idx_best]
    
    return throughput_factor, nearest_frame


def classical_psf_sub(psf_frame):
    """
    Placeholder for a PSF-subtraction efficiency factor function.
    
    Args:
        psf_frame (corgidrp.data.Image): The input PSF image from which the 
            companion signal is to be subtracted.
    
    Returns:
    psf_sub_image (corgidrp.data.Image): A new Image object containing the 
        PSF-subtracted data.
    efficiency (float): The efficiency factor applied to the PSF data (e.g., 0.7).
    """
    # e.g. near the star, the companion is more heavily subtracted, etc.
    # Return a factor in (0,1].
    efficiency = 0.7
    psf_sub_frame = psf_frame.data * efficiency

    psf_sub_image = Image(
        data_or_filepath=psf_sub_frame, 
        pri_hdr=psf_frame.pri_hdr, 
        ext_hdr=psf_frame.ext_hdr, 
        err=psf_frame.err if hasattr(psf_frame, 'err') else None
        )
    return psf_sub_image, efficiency
