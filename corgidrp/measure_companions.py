import numpy as np
from astropy.table import Table
import pyklip.klip as klip
import corgidrp.fluxcal as fluxcal
from corgidrp.data import CoreThroughputCalibration


def measure_companions(
    image,
    method='aperture',
    apply_throughput=False,
    apply_fluxcal=False,
    fluxcal_factor=None,
    ct_cal = None,
    cor_dataset = None,
    FpamFsamCal = None,
    apply_psf_sub_eff=False,
    psf_fwhm=3.0,
    aperture_radius=5.0,
    star_flux_e_s=None,
    star_mag=None,
    # Arguments to handle off-axis star modeling
    direct_star_image=None,
    reference_psf=None,
    do_forward_model=False,
    verbose=True
):
    """
    Measure companion properties in a final (or intermediate) coronagraphic image,
    returning position, flux ratio (companion/starlight), and apparent magnitude.

    This version follows the procedure:
      1) Measure host star flux in a direct image (if provided).
      2) Measure reference PSF flux (e.g. from off-axis calibration).
      3) Compute ratio for scaling the off-axis PSF to represent the star at companion's location.
      4) (Optional) Forward-model that off-axis star image through PSF subtraction if do_forward_model=True.
      5) Measure the companion flux ratio & apparent magnitude (via aperture, PSF fit, or forward model).
    
    If a zero point (ZP) is stored in fluxcal_factor.ext_hdr['ZP'], will use:
        mag_comp = -2.5 * log10(flux_companion_e_s) + ZP

    Parameters
    ----------
    image : corgidrp.data.Image
        The final (or intermediate) PSF-subtracted image in e-/s/pixel, with .data, .hdr, .ext_hdr.
    method : {'aperture', 'psf_fit', 'forward_model'}, optional
        Photometry method for measuring companion flux on the (already) PSF-subtracted image.
    apply_throughput : bool, optional
        If True, apply a radial throughput correction or a user-supplied method.
    apply_fluxcal : bool, optional
        If True, convert from e-/s to magnitudes using fluxcal_factor (which may have ext_hdr['ZP']). 
    ct_cal (corgidrp.data.CoreThroughputCalibration): A Core Throughput calibration file containing 
        PSFs at different distances from the mask and their corresponding positions and throughputs.
    fluxcal_factor : object, optional
        E.g. corgidrp.Data.FluxcalFactor. Might have .ext_hdr['ZP'] or a .to_magnitude() method.
    cor_dataset (corgidrp.data.Dataset): a dataset containing some coronagraphic observations.
    FpamFsamCal (corgidrp.data.FpamFsamCal): FPAM to EXCAM transformation matrix.
    apply_psf_sub_eff : bool, optional
        If True, apply an additional correction factor for PSF-subtraction efficiency.
    psf_fwhm : float, optional
        Approximate PSF FWHM in pixels. Used for psf-fit initialization or aperture defaults.
    aperture_radius : float, optional
        Aperture radius (in pixels) if using aperture photometry.
    star_flux_e_s : float, optional
        The star's flux in e-/s (integrated over the same band). Used for flux_ratios.
    star_mag : float, optional
        The star's apparent magnitude. If provided (with star_flux_e_s), we can compute the companion's apparent mag.
    direct_star_image : corgidrp.data.Image, optional
        A direct (unocculted) image of the host star (in e-/s/pixel). If you want to measure star flux consistently.
    reference_psf : corgidrp.data.Image, optional
        An off-axis calibration PSF of a star at some known separation where mask effects are negligible.
    cor_throughput_class : object, optional
        An object with a method like .GetPSF(x, y, dataset, FpamFsamCal) that returns an off-axis PSF at (x,y).
    do_forward_model : bool, optional
        If True, forward-model the off-axis star image through the same PSF-sub or KLIP steps and fit to data.
    verbose : bool, optional
        If True, print progress messages.

    Returns
    -------
    result_table : astropy.table.Table
        Columns:
            - id : companion label
            - x, y : location in pixels
            - flux_raw : companion flux in e-/s (post-subtraction)
            - flux_err : estimated flux uncertainty
            - flux_corr : flux after throughput correction
            - flux_ratio : flux_corr / star_flux_e_s (if star_flux_e_s is given)
            - mag : apparent magnitude (computed via star_mag or zero point or fluxcal_factor)
    """

    # 1) Possibly measure star_flux_e_s from direct_star_image if not given:
    if star_flux_e_s is None and direct_star_image is not None:
        if verbose:
            print("Measuring total star flux from direct star image...")
        # Example: Aperture or other method on direct_star_image
        star_flux_e_s = measure_star_flux(direct_star_image)

    # 2) Possibly measure flux from reference_psf if not given. 
    #    This is used to figure out how to scale an off-axis PSF.
    reference_psf_flux = None
    if reference_psf is not None:
        reference_psf_flux = measure_reference_psf_flux(reference_psf)
        if verbose:
            print(f"Measured reference PSF flux = {reference_psf_flux:.2f} e-/s")

    # 3) Create or retrieve the off-axis star image at each companion’s location (scaled properly).
    #    This off-axis star represents "how bright the host star would look" if placed at that separation.
    #    Do this inside the forward-model step or in the measurement routine below.

    # 4) Parse out companion positions from the image header
    img_data = image.data
    ext_hdr = image.ext_hdr
    if 'STARLOCX' in ext_hdr and 'STARLOCY' in ext_hdr:
        star_center = (ext_hdr['STARLOCX'], ext_hdr['STARLOCY'])
    else:
        ny, nx = img_data.shape
        star_center = (nx / 2, ny / 2)
        if verbose:
            print("STARLOCX/STARLOCY not found; defaulting to image center.")

    # Gather companion info from SNYX### keywords
    companion_list = []
    for key in ext_hdr.keys():
        if key.startswith('SNYX'):
            # Format: "SNR,x,y"
            snr_str, x_str, y_str = ext_hdr[key].split(',')
            companion_list.append({
                'id': key,
                'snr': float(snr_str),
                'x_init': float(x_str),
                'y_init': float(y_str)
            })

    if verbose:
        print(f"Found {len(companion_list)} companion(s) in header keywords.")

    # (Optional for now) refine companion center positions
    for comp in companion_list:
        comp['x'] = comp['x_init']
        comp['y'] = comp['y_init']
        # 2D centroid or fit here to refine positions?
        # comp['x'], comp['y'] = refine_position(img_data, comp['x_init'], comp['y_init'], ...)

    # Prepare lists for final results
    comp_ids, xs, ys = [], [], []
    flux_raws, flux_errs, flux_corrs = [], [], []
    flux_ratios, mags = [], []

    # 5) Loop over companions, measure flux by chosen method
    for comp in companion_list:
        cid = comp['id']
        x_c, y_c = comp['x'], comp['y']

        # (A) If do_forward_model = True, build a forward model of the star at (x_c, y_c),
        #     run it through the same PSF-subtraction pipeline, and do a fit. 
        if method == 'forward_model' or do_forward_model:
            flux_val, flux_err = forward_model_flux(
                image,
                x_c, y_c,
                reference_psf_flux=reference_psf_flux,
                star_flux_e_s=star_flux_e_s,
                do_klip=True, 
                verbose=verbose
            )

        elif method == 'psf_fit':
            # A simpler 2D Gaussian fit on the post-subtraction image
            flux_val, flux_err, *_ = fluxcal.phot_by_gauss2d_fit(
                image,
                psf_fwhm=psf_fwhm,
                fit_shape=None,
                background_sub=True,
                r_in=5,
                r_out=10,
                centering_method='xy',
                centroid_roi_radius=aperture_radius,
                initial_guess=(x_c, y_c)
            )

        else:  # 'aperture' by default
            flux_val, flux_err, *_ = fluxcal.aper_phot(
                image,
                encircled_radius=aperture_radius,
                frac_enc_energy=1.0,
                method='subpixel',
                subpixels=5,
                background_sub=True,
                r_in=5,
                r_out=10,
                centering_method='xy',
                centroid_roi_radius=5,
                centering_initial_guess=(x_c, y_c)
            )

        # Store raw flux
        comp_ids.append(cid)
        xs.append(x_c)
        ys.append(y_c)
        flux_raws.append(flux_val)
        flux_errs.append(flux_err)

    # 6) Optionally apply throughput or PSF-sub efficiency corrections
    for i, comp in enumerate(companion_list):
        flux_corr = flux_raws[i]
        dx = xs[i] - star_center[0]
        dy = ys[i] - star_center[1]
        sep_pix = np.hypot(dx, dy)

        if apply_throughput:
            flux_corr = apply_core_throughput_correction(xs[i], ys[i], ct_cal, cor_dataset, 
                                                         FpamFsamCal, flux_raws[i])

        if apply_psf_sub_eff:
            # apply an additional correction factor for PSF-subtraction efficiency?
            psf_sub_eff = get_psf_sub_eff(sep_pix)
            if psf_sub_eff <= 0:
                raise ValueError("Invalid PSF-subtraction efficiency")
            flux_corr /= psf_sub_eff

        flux_corrs.append(flux_corr)

    # 7) Compute flux ratio & companion magnitude
    zero_point = None
    if apply_fluxcal and fluxcal_factor is not None:
        zero_point = fluxcal_factor.ext_hdr.get('ZP', None)

    for i in range(len(companion_list)):
        f_c = flux_corrs[i]
        # flux ratio if star_flux_e_s is given
        fratio = None
        if star_flux_e_s:
            fratio = f_c / star_flux_e_s
        flux_ratios.append(fratio)

        # magnitude from zero_point if we have it
        mag_comp = None
        if zero_point is not None and f_c > 0:
            mag_comp = -2.5 * np.log10(f_c) + zero_point

        # or from star_mag if we have star_flux_e_s
        if mag_comp is None and (star_mag is not None) and (fratio is not None) and (fratio > 0):
            mag_comp = star_mag - 2.5 * np.log10(fratio)
        mags.append(mag_comp)

    # 8) Build results table
    t = Table()
    t['id'] = comp_ids
    t['x'] = xs
    t['y'] = ys
    t['flux_raw'] = flux_raws
    t['flux_err'] = flux_errs
    t['flux_corr'] = flux_corrs
    t['flux_ratio'] = flux_ratios
    t['mag'] = mags

    if verbose:
        print("Measurement complete. Returning result table.")

    return t


# -------------------------------------------------------------------
# Below are placeholders for the additional steps and routines used above
# Replace these with actual measurement or forward-modeling code.
# -------------------------------------------------------------------

def measure_star_flux(direct_star_image):
    """
    Example routine to measure total star flux from a direct (unocculted) image.
    Could be an aperture capturing the entire star, a PSF fit, etc.
    """
    # Real code might do:
    # flux_val, flux_err, *_ = fluxcal.aper_phot(...)
    # return flux_val
    return 1.0e6  # placeholder


def measure_reference_psf_flux(reference_psf):
    """
    Example routine to measure flux from a reference PSF (e-/s).
    """
    # Real code might do an aperture or encircled energy measurement:
    # flux_val, flux_err, *_ = fluxcal.aper_phot(...)
    return 1.0e5  # placeholder


def forward_model_flux(
    dataset,
    x_c, y_c,
    FpamFsamCal = None,
    reference_psf_flux=None,
    star_flux_e_s=None,
    do_klip=True,
    verbose=True
):
    """
    Place an off-axis PSF at (x_c, y_c), run it through the same
    PSF-subtraction or KLIP process, and fit for the best match
    to the actual companion in the image.

    Returns the flux_val (e-/s) and flux_err for the companion.
    """
    # 1) Interpolate the off-axis PSF for location (x_c, y_c)
    # Retrieve the star PSF at this separation
    offaxis_psf = CoreThroughputCalibration.GetPSF(
        x_cor=x_c,
        y_cor=y_c,
        cor_dataset=dataset,
        FpamFsamCal=FpamFsamCal     # FPAM to Excam transformation matrix
    )

    # 2) Scale off-axis PSF by ratio of (star_flux / reference_psf_flux)
    #    so it represents how bright the star would look if placed at (x_c, y_c)
    if reference_psf_flux and reference_psf_flux != 0 and star_flux_e_s is not None:
        ratio = star_flux_e_s / reference_psf_flux
        offaxis_psf_scaled = offaxis_psf * ratio
    else:
        offaxis_psf_scaled = offaxis_psf

    # 3) Forward-model: inject offaxis_psf_scaled into empty data or reference frames,
    #    then run the same PSF-sub algorithm. Use PyKLIP:
    if do_klip:
        # Pass injected frames to pyKLIP,
        # run klip.klip_dataset to produce a processed image,
        # and measure the extracted flux. 
        pass

    # 4) Fit or measure how well the forward-modeled companion matches the actual companion
    #    Minimizing residuals or using a small region near (x_c, y_c)?
    flux_estimate = 150.0
    flux_err_est = 15.0

    return flux_estimate, flux_err_est


def apply_core_throughput_correction(
    x_c, y_c,
    ct_cal,
    cor_dataset,
    FpamFsamCal,
    flux_raw
):
    """
    Apply the core throughput factor from a loaded CoreThroughputCalibration object
    at the companion location (x_c, y_c).

    Parameters
    ----------
    x_c, y_c : float
        The companion's pixel location in the coronagraphic image.
    cor_dataset : corgidrp.data.Dataset
        The same dataset used for the coronagraphic observation, or at least
        the first frame's header info.
    ct_cal : corgidrp.data.CoreThroughputCalibration
        The loaded calibration file with PSF basis and arrays of throughput.
    FpamFsamCal : corgidrp.data.FpamFsamCal
        The object to help map FPAM changes to EXCAM pixel offsets.
    flux_raw : float
        The raw measured companion flux (in e-/s) before throughput correction.

    Returns
    -------
    flux_corrected : float
        The flux after applying the throughput correction factor.
    """

    # Get the FPM center arrays from the calibration vs. the cor_dataset
    # Use this to find the closest value in the measured throughput set
    # to the companion's x, y position. Do nearest neighbor here (does
    # Sergi's new code cover this?)
    star_center = ct_cal.GetCTFPMPosition(cor_dataset, FpamFsamCal)[0]  # returns (x,y), ignoring FSAM for example
    dx = x_c - star_center[0]
    dy = y_c - star_center[1]

    # ct_excam is shape (3, N) => [xvals, yvals, throughput].
    # Find the throughput of the PSF that is nearest to (dx, dy):
    xyvals = ct_cal.ct_excam[:2]  # shape (2, N)
    throughvals = ct_cal.ct_excam[2]  # shape (N,)

    diffs = xyvals.T - np.array([dx, dy])
    rr = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    idx_best = np.argmin(rr)
    throughput_factor = throughvals[idx_best]
    if throughput_factor <= 0:
        throughput_factor = 1.0  # fallback

    flux_corrected = flux_raw / throughput_factor
    return flux_corrected


def get_psf_sub_eff(rad_pix):
    """
    Example placeholder for a PSF-subtraction efficiency factor, if you have a known
    curve of how much flux is attenuated by the PSF subtraction at each separation.
    """
    # e.g. near the star, the companion is more heavily subtracted, etc.
    # Return a factor in (0,1].
    efficiency = 1.0 - 0.1*(rad_pix / 50.)
    return max(efficiency, 0.05)
