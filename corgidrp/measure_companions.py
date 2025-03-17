import numpy as np
from astropy.table import Table
import corgidrp.fluxcal as fluxcal
import pyklip.klip as klip
from pyklip.parallelized import klip_dataset
import pyklip.fakes as fakes
from corgidrp.data import PyKLIPDataset, Dataset, Image, CoreThroughputCalibration
import numpy as np
import corgidrp.fluxcal as fluxcal


def measure_companions(
    psf_image,
    coronagraphic_dataset=None,
    phot_method='aperture',
    apply_throughput=False,
    apply_fluxcal=False,
    fluxcal_factor=None,
    ct_cal = None,
    ct_dataset = None,
    FpamFsamCal = None,
    apply_psf_sub_eff=False,
    host_star_counts=None,
    host_star_apmag=None,
    direct_star_image=None,
    reference_psf=None,
    out_dir = None,
    verbose=True
):
    """
    Measure companion properties in a final (or intermediate) coronagraphic image,
    returning position, counts ratio (companion/starlight), and apparent magnitude.

    This version follows the procedure:
      1) Measure host star counts in a direct image (if provided).
      2) Measure reference PSF counts (e.g. from off-axis calibration).
      3) Compute ratio for scaling the off-axis PSF to represent the star at companion's location.
      4) (Optional) Forward-model off-axis star image through PSF subtraction if method= 'forward_model'.
      5) Measure the companion count ratio & apparent magnitude (via aperture, PSF fit, or forward model).

    Parameters
    ----------
    psf_image (corgidrp.data.Image): The final (or intermediate) PSF-subtracted image in e-/s/pixel, with 
        .data, .hdr, .ext_hdr.
    phot_method ({'aperture', 'psf_fit', 'forward_model'}, optional): Photometry method for measuring 
        companion counts on the (already) PSF-subtracted image.
    apply_throughput (bool, optional): If True, apply a radial throughput correction or a 
        user-supplied method.
    apply_fluxcal (bool, optional): If True, convert from e- to magnitudes using fluxcal_factor 
        (which may have ext_hdr['ZP']). TO DO
    ct_cal (corgidrp.data.CoreThroughputCalibration): A Core Throughput calibration file containing 
        PSFs at different distances from the mask and their corresponding positions and throughputs.
    fluxcal_factor (corgidrp.Data.FluxcalFactor, optional): Fluxcal factor with .ext_hdr['ZP'].
    ct_dataset (corgidrp.data.Dataset, optional): A dataset containing some coronagraphic observations.
    FpamFsamCal (corgidrp.data.FpamFsamCal): FPAM to EXCAM transformation matrix.
    apply_psf_sub_eff (bool, optional): If True, apply an additional correction factor for 
        PSF-subtraction efficiency.
    psf_fwhm (float, optional): Approximate PSF FWHM in pixels. Used for psf-fit initialization or 
        aperture defaults.
    aperture_radius (float, optional): Aperture radius (in pixels) if using aperture photometry.
    host_star_counts (float, optional): The star's measured counts in e-. Used for count_ratios.
    host_star_apmag (float, optional): The star's apparent magnitude. If provided (with host_star_counts), 
        we can compute the companion's apparent mag.
    direct_star_image (corgidrp.data.Image, optional): A direct (unocculted) image of the host star (in e-). 
    reference_psf (corgidrp.data.Image, optional): An off-axis calibration PSF of a star at some known 
        separation where mask effects are negligible.
    out_dir (str):
    verbose (bool, optional): If True, print progress messages.

    Returns
    -------
    result_table (astropy.table.Table):
        Columns:
            - id : companion label
            - x, y : location in pixels
            - counts_raw : companion counts in e- (post-subtraction)
            - counts_err : estimated counts uncertainty
            - counts_corr : counts after throughput correction
            - counts_ratio : counts_corr / host_star_counts (if host_star_counts is given)
            - mag : apparent magnitude (computed via host_star_apmag or zero point or fluxcal_factor)
    """

   # Measure host star counts if not provided
    if host_star_counts is None and direct_star_image is not None:
        if verbose: print("Measuring total star counts from direct star image...")
        host_star_counts, _ = measure_counts(direct_star_image, "aperture", None)

    # Measure reference PSF counts if provided
    reference_psf_counts = None
    if reference_psf:
        reference_psf_counts, _ = measure_counts(reference_psf, "aperture", None)
        if verbose: print(f"Measured reference PSF counts: {reference_psf_counts:.2f} e-/s")

    # Parse companion positions from headers
    star_center = (psf_image.ext_hdr.get('STARLOCX', psf_image.data.shape[1] / 2),
                   psf_image.ext_hdr.get('STARLOCY', psf_image.data.shape[0] / 2))
    
    companions = [{"id": key, "x": float(x), "y": float(y)}
                  for key, val in psf_image.ext_hdr.items() if key.startswith('SNYX')
                  for _, x, y in [val.split(',')]]

    if verbose: print(f"Found {len(companions)} companion(s) in header.")

    # Loop over companions and measure counts
    results = []
    for comp in companions:
        x_c, y_c = comp['x'], comp['y']
        input_data = coronagraphic_dataset if phot_method == 'forward_model' else psf_image

        method_kwargs = get_photometry_kwargs(phot_method, out_dir)
        counts_raw, counts_err = measure_counts(input_data, phot_method, (x_c, y_c), **method_kwargs)

        counts_corr = apply_throughput_correction(x_c, y_c, counts_raw, star_center, ct_cal, ct_dataset,
                                                  FpamFsamCal, apply_psf_sub_eff, apply_throughput)

        counts_ratio = counts_corr / host_star_counts if host_star_counts else None
        companion_mag = compute_companion_magnitude(counts_corr, counts_ratio, host_star_apmag, fluxcal_factor)

        results.append((comp['id'], x_c, y_c, counts_raw, counts_err, counts_corr, counts_ratio, companion_mag))

    return Table(rows=results, names=['id', 'x', 'y', 'counts_raw', 'counts_err',
                                      'counts_corr', 'counts_ratio', 'mag'])


# ---------------- Helper functions -------------------------------------- #

def measure_counts(input_image_or_dataset, phot_method, initial_xy_guess, **kwargs):
    """
    Measure counts in an image using the specified method.

    Parameters:
    input_image_or_dataset (corgidrp.data.Image or corgidrp.data.Dataset):
        The image or dataset where the companion exists.
    method (str): The measurement method ('forward_model', 'psf_fit', or 'aperture').
    initial_xy_guess (tuple): Initial (x, y) guess for the companion location.
    **kwargs (dict): Additional method-specific parameters.

    Returns:
    counts_val (float): Estimated counts of the companion.
    counts_err (float): Estimated counts uncertainty.
    """

    if phot_method == 'forward_model':
        return forward_model_counts(input_image_or_dataset, initial_xy_guess, **kwargs)

    elif phot_method == 'psf_fit':
        return fluxcal.phot_by_gauss2d_fit(input_image_or_dataset, initial_guess=initial_xy_guess, **kwargs)[:2]

    elif phot_method == 'aperture':
        return fluxcal.aper_phot(input_image_or_dataset, centering_initial_guess=initial_xy_guess, **kwargs)[:2]

    raise ValueError(f"Invalid photometry method: {phot_method}")


def get_photometry_kwargs(phot_method, out_dir):
    """
    Returns method-specific keyword arguments for different photometry methods.

    Parameters:
    phot_method (str): The photometry method to be used. Options are:
        - 'forward_model': Uses KLIP-based forward modeling.
        - 'psf_fit': Uses a 2D Gaussian fit.
        - 'aperture': Uses aperture photometry.
    out_dir (str): The directory where outputs should be saved (only applicable for 'forward_model').

    Returns:
    photometry_options[phot_method] (dict): A dictionary containing the appropriate keyword arguments 
        for the chosen photometry method.
    """
    common_kwargs = {'centering_method': 'xy', 'centroid_roi_radius': 5}

    photometry_options = {
        'forward_model': {'out_dir': out_dir, 'do_klip': True, 'kl_modes': (5, 10)},
        'psf_fit': {'psf_fwhm': 3, 'background_sub': True, 'r_in': 5, 'r_out': 10, **common_kwargs},
        'aperture': {'encircled_radius': 4, 'frac_enc_energy': 1, 'subpixels': 5, 'background_sub': True,
                     'r_in': 6, 'r_out': 12, **common_kwargs}
    }

    if phot_method not in photometry_options:
        raise ValueError(f"Invalid photometry method '{phot_method}'. Choose from {list(photometry_options.keys())}.")

    return photometry_options[phot_method]


def forward_model_counts(
    image_or_dataset,
    companion_xy,
    out_dir = None,
    reference_psf_counts=None,
    host_star_counts=None,
    fwhm=3.5,  # Default PSF FWHM in pixels
    do_klip=True,
    kl_modes=(5,),
    verbose=True
):
    """
    Forward-model a companion at (x_c, y_c) by injecting a synthetic companion
    (off-axis star) into the raw frames, running KLIP, and measuring the counts
    in the final residual.

    Parameters
    ----------
    image_or_dataset (corgidrp.data.Image or corgidrp.data.Dataset): A coronagraphic 
        dataset containing one or more science frames where the companions exist.
    companion_xy (tuple (x, y)): Pixel coordinates in the science image of the 
        companion to forward-model.
    out_dir (str): Output directory path for pyklip.
    reference_psf_counts (float or None): If you have measured counts from a reference 
        PSF that is known to be effectively off-axis, can scale by (host_star_counts / 
        reference_psf_counts).
    host_star_counts (float or None): The unocculted, direct star counts of the host star 
        from a calibration image, before any coronagraph is applied and before any PSF 
        subtraction. Counts should be in e- (or e-/s, but use the convention consistently). 
        Used to scale the off-axis star injection.
    fwhm (float): Full Width at Half Maximum (FWHM) of the PSF, in pixels.
    do_klip (bool): Whether to run KLIP. If False, skip the actual subtraction step 
        (useful for debugging).
    kl_modes (tuple): The KL modes used in klip.klip_dataset.
    verbose (bool): Print debug messages.

    Returns:
    counts_val (float): Estimated counts of the forward-modeled companion in e- (or e-/s) 
        after the KLIP subtraction.
    counts_err (float): Estimated counts uncertainty.
    """
    x_c, y_c = companion_xy
    science_dataset = image_or_dataset if isinstance(image_or_dataset, Dataset) else Dataset([image_or_dataset])
    pk_data = PyKLIPDataset(science_dataset, highpass=False)

    dx, dy = x_c - pk_data.centers[0][0], y_c - pk_data.centers[0][1]
    separation, position_angle = np.hypot(dx, dy), np.degrees(np.arctan2(dx, -dy)) % 360

    injected_counts = (host_star_counts / reference_psf_counts) if reference_psf_counts else 100.0

    fakes.inject_planet(pk_data.input, pk_data.centers, injected_counts, pk_data.wcs, separation, position_angle,
                        fwhm=fwhm)

    if verbose:
        print(f"[ForwardModel] Injected planet at {separation:.2f} px, PA={position_angle:.2f}, counts={injected_counts:.3g}")

    if do_klip:
        klip_dataset(pk_data, outputdir=out_dir, fileprefix="fwdmodel_injected", numbasis=kl_modes)
        sub_img = Image(pk_data.output[-1])
    else:
        sub_img = Image(pk_data.input[0])

    return fluxcal.aper_phot(sub_img, encircled_radius=fwhm, r_in=fwhm*1.5, r_out=fwhm*2.5,
                             centering_initial_guess=(x_c, y_c), background_sub=True)[:2]


def apply_throughput_correction(x, y, counts, star_center, ct_cal, ct_dataset, FpamFsamCal,
                                apply_psf_sub_eff, apply_throughput):
    """
    Applies throughput and PSF-subtraction efficiency corrections to measured counts.

    Parameters:
    x (float): X-coordinate of the companion in the image (pixels).
    y (float): Y-coordinate of the companion in the image (pixels).
    counts (float): Measured companion counts (e- or e-/s) before correction.
    star_center (tuple (float, float)): Coordinates of the star center in the 
        image (pixels).
    ct_cal (corgidrp.data.CoreThroughputCalibration): Core throughput calibration 
        object, containing PSF throughput information.
    ct_dataset (corgidrp.data.Dataset): Dataset used for generating the throughput 
        calibration.
    FpamFsamCal (corgidrp.data.FpamFsamCal): Calibration object used for mapping 
        FPAM to EXCAM coordinates.
    apply_psf_sub_eff (bool): Whether to apply a correction for PSF-subtraction 
        efficiency.
    apply_throughput (bool): Whether to apply a radial throughput correction using 
        `ct_cal`.

    Returns:
    counts (float): Corrected companion counts (e- or e-/s) after applying throughput 
        and PSF-sub efficiency.

    """
    dx, dy = x - star_center[0], y - star_center[1]
    sep_pix = np.hypot(dx, dy)

    if apply_psf_sub_eff:
        counts /= get_psf_sub_eff(sep_pix)

    if apply_throughput:
        counts /= measure_core_throughput_at_location(x, y, ct_cal, ct_dataset, FpamFsamCal)

    return counts


def measure_core_throughput_at_location(
    x_c, y_c,
    ct_cal,
    ct_dataset,
    FpamFsamCal
):
    """
    Apply the core throughput factor from a loaded CoreThroughputCalibration object
    at the companion location (x_c, y_c).

    Parameters:
    x_c, y_c (float): The companion's pixel location in the coronagraphic image.
    ct_dataset (corgidrp.data.Dataset): The same dataset used for the coronagraphic 
        observation, or at least the first frame's header info.
    ct_cal (corgidrp.data.CoreThroughputCalibration): The loaded calibration file with 
        PSF basis and arrays of throughput.
    FpamFsamCal (corgidrp.data.FpamFsamCal): The object to help map FPAM changes to 
        EXCAM pixel offsets.
    counts_raw (float): The raw measured companion counts (in e-/s) before throughput 
        correction.

    Returns:
    throughput_factor (float): The core throughput factor at the specified location 
        (dimensionless).
    """
    star_center = ct_cal.GetCTFPMPosition(ct_dataset, FpamFsamCal)[0]
    dx, dy = x_c - star_center[0], y_c - star_center[1]

    xyvals, throughvals = ct_cal.ct_excam[:2], ct_cal.ct_excam[2]
    idx_best = np.argmin(np.hypot(xyvals[0] - dx, xyvals[1] - dy))

    return max(throughvals[idx_best], 1.0)


def get_psf_sub_eff(rad_pix):
    """
    Placeholder for a PSF-subtraction efficiency factor function.
    """
    # e.g. near the star, the companion is more heavily subtracted, etc.
    # Return a factor in (0,1].
    efficiency = 0.7
    return efficiency


def compute_companion_magnitude(counts_corr, counts_ratio, host_star_apmag, fluxcal_factor):
    """
    Computes the apparent magnitude of a companion based on either the host star's magnitude 
    and counts ratio or a flux calibration zero point.

    Parameters:
    counts_corr (float): Corrected companion counts (e- or e-/s) after applying throughput 
        and PSF-sub efficiency.
    counts_ratio (float or None): Ratio of the companion's corrected counts to the host star's 
        counts (counts_corr / host_star_counts).
    host_star_apmag (float or None): Apparent magnitude of the host star, used for computing 
        the companion magnitude if counts_ratio is provided.
    fluxcal_factor (corgidrp.data.FluxcalFactor or None): Flux calibration factor object 
        containing the zero point (ext_hdr['ZP']) for absolute magnitude conversion.

    Returns:
    (float): Apparent magnitude of the companion.
    """
    zero_point = fluxcal_factor.ext_hdr.get('ZP') if fluxcal_factor else None

    if host_star_apmag and counts_ratio:
        return host_star_apmag - 2.5 * np.log10(counts_ratio)

    if zero_point and counts_corr > 0:
        return -2.5 * np.log10(counts_corr) + zero_point

    raise ValueError("Cannot compute magnitude; missing required parameters.")