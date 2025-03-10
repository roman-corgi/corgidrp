import numpy as np
from astropy.table import Table
import corgidrp.fluxcal as fluxcal

# ultimately want to move all this into l4_to_tda.py, this is just for testing now

def measure_companions(
    image,
    method='aperture',
    apply_throughput=False,
    apply_fluxcal=False,
    fluxcal_factor=None,
    apply_psf_sub_eff=False,
    psf_fwhm=3.0,
    aperture_radius=5.0,
    star_flux_e_s=None,
    star_mag=None,
    verbose=True
):
    """
    Measure companion properties in a final PSF-subtracted image,
    returning position, flux ratio (companion/starlight), and apparent magnitude.

    If a zero point (ZP) is stored in fluxcal_factor.ext_hdr['ZP'], will use:
        mag_comp = -2.5 * log10(flux_companion_e_s) + ZP

    Parameters
    ----------
    image : corgidrp.data.Image
        The final PSF-subtracted image in photoelectrons/second/pixel (and its header).
    method : {'aperture', 'psf_fit', 'forward_model'}, optional
        Photometry method for measuring companion flux.
    apply_throughput : bool, optional
        If True, apply a throughput correction (requires known throughput vs. separation).
    apply_fluxcal : bool, optional
        If True, apply flux calibration to convert from e-/s to magnitudes using fluxcal_factor.
    fluxcal_factor : object, optional
        e.g. corgidrp.Data.FluxcalFactor. 
        - Possibly has ext_hdr['ZP'] storing the zero point.
        - Alternatively, might have a .to_magnitude() method.
    psf_fwhm : float, optional
        Approximate PSF FWHM in pixels. Used for psf-fit initialization or aperture default.
    aperture_radius : float, optional
        Aperture radius (pixels) if using aperture photometry.
    star_flux_e_s : float, optional
        The star's flux in e-/s (integrated over the same band).
        If provided, compute flux_ratio = flux_comp / star_flux_e_s.
    star_mag : float, optional
        The star's apparent magnitude. If also providing star_flux_e_s, can compute
        companion's apparent magnitude as m_comp = star_mag - 2.5*log10(flux_ratio).
    verbose : bool, optional
        If True, print messages.

    Returns
    -------
    result_table : astropy.table.Table
        Columns:
            - id : companion label
            - x, y : location in pixels
            - flux_raw : companion flux in e-/s (post-subtraction)
            - flux_err : estimated flux uncertainty
            - flux_corr : flux after throughput correction (if applied)
            - flux_ratio : flux_corr / star_flux_e_s (if star_flux_e_s is given)
            - mag : apparent magnitude (computed via star_mag or zero point or fluxcal_factor)
    """

    # 1) Load data & header
    img_data = image.data
    ext_hdr = image.ext_hdr

    # 2) Determine star center
    if 'STARLOCX' in ext_hdr and 'STARLOCY' in ext_hdr:
        star_center = (ext_hdr['STARLOCX'], ext_hdr['STARLOCY'])
    else:
        if verbose:
            print("STARLOCX/STARLOCY not found in header; defaulting to image center.")
        ny, nx = img_data.shape
        star_center = (nx / 2, ny / 2)

    # 3) Gather companion info from SNYX### keywords
    companion_list = []
    for key in ext_hdr.keys():
        if key.startswith('SNYX'):
            snr_str, x_str, y_str = ext_hdr[key].split(',')
            snr_val = float(snr_str)
            x_val   = float(x_str)
            y_val   = float(y_str)
            companion_list.append({
                'id': key,      # e.g. 'SNYX001'
                'snr': snr_val,
                'x_init': x_val,
                'y_init': y_val
            })

    if verbose:
        print(f"Found {len(companion_list)} companion(s) in header keywords.")

    # 4) Optionally refine positions
    for comp in companion_list:
        comp['x'] = comp['x_init']
        comp['y'] = comp['y_init']
        # could do a 2D centroid or local PSF fit if needed.

    # 5) Measure flux for each companion
    comp_ids, xs, ys = [], [], []
    flux_raws, flux_errs = [], []

    for comp in companion_list:
        cid = comp['id']
        x_c = comp['x']
        y_c = comp['y']

        if method == 'aperture':
            print("starting flux measure")
            flux_result = fluxcal.aper_phot(
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
        elif method == 'psf_fit':
            flux_result = fluxcal.phot_by_gauss2d_fit(
                image,
                psf_fwhm=psf_fwhm,
                fit_shape=None,
                background_sub=True,
                r_in=5,
                r_out=10,
                centering_method='xy',
                centroid_roi_radius=aperture_radius
            )
        elif method == 'forward_model':
            flux_val, flux_err = _forward_model_flux(img_data, x_c, y_c)
        else:
            raise ValueError(f"Unknown method '{method}'")
        
        flux_val, flux_err, *back = flux_result 

        comp_ids.append(cid)
        xs.append(x_c)
        ys.append(y_c)
        flux_raws.append(flux_val)
        flux_errs.append(flux_err)

    # 6) Throughput correction
    flux_corrs = []
    for i, comp in enumerate(companion_list):
        f_corr = flux_raws[i]
        
        # (A) If applying a radial throughput correction (e.g. from standard star)
        if apply_throughput:
            dx = xs[i] - star_center[0]
            dy = ys[i] - star_center[1]
            sep_pix = np.hypot(dx, dy)
            throughput_factor = get_throughput_factor(sep_pix)
            f_corr /= throughput_factor

        # (B) If applying the PSF-subtraction efficiency from forward modeling
        #if apply_psf_sub_eff:
            # Option 1: read from a function like get_psf_sub_eff(sep_pix)
            #psf_sub_eff = get_psf_sub_eff(sep_pix)

            # Option 2: or might pass in an array of per-companion factors
            # psf_sub_eff = psf_sub_eff_array[i]

            #if psf_sub_eff <= 0:
            #    raise ValueError("Invalid PSF-subtraction efficiency")
            #f_corr /= psf_sub_eff

        flux_corrs.append(f_corr)

    # 7) Compute flux ratio & magnitudes
    flux_ratios = [None] * len(companion_list)
    mags        = [None] * len(companion_list)

    # Attempt to retrieve ZP from fluxcal_factor if it exists
    zero_point = None
    if apply_fluxcal and fluxcal_factor is not None:
        zero_point = fluxcal_factor.ext_hdr.get('ZP', None) 
        print("zero point", zero_point)

    for i in range(len(companion_list)):
        f_c = flux_corrs[i]  # e-/s

        # (A) flux ratio if star_flux_e_s is provided
        if star_flux_e_s is not None:
            flux_ratios[i] = f_c / star_flux_e_s

        # If we haven't assigned mag yet, check if zero_point is available
        if (mags[i] is None) and (zero_point is not None):
            # m = -2.5*log10(f_c) + zero_point
            if f_c > 0:
                mags[i] = -2.5 * np.log10(f_c) + zero_point
            else:
                mags[i] = None

        # If still no magnitude, try star_mag if star_flux_e_s is known
        if (mags[i] is None) and (star_mag is not None) and (flux_ratios[i] is not None):
            # m_comp = star_mag - 2.5 * log10(flux_ratio)
            mags[i] = star_mag - 2.5 * np.log10(flux_ratios[i])

    # 8) Build result table for checking now, ultimately want to just write this to headers
    t = Table()
    t['id']         = comp_ids
    t['x']          = xs
    t['y']          = ys
    t['flux_raw']   = flux_raws
    t['flux_err']   = flux_errs
    t['flux_corr']  = flux_corrs
    t['flux_ratio'] = flux_ratios
    t['mag']        = mags

    if verbose:
        print("Measurement complete. Returning result table with position, flux ratio, and magnitude.")
    return t

# ------------- Helpers -------------
def _forward_model_flux(img_data, x_c, y_c):
    """
    Placeholder for local forward-modeling approach.
    """
    flux_estimate = 100.0
    flux_err_est  = 10.0
    return flux_estimate, flux_err_est

def get_throughput_factor(rad_pix):
    """
    Placeholder for returning the throughput factor (0<factor<1) based on
    separation in pixels from prior calibrations (PSF throughput).
    """
    factor = 0.7 + 0.2*(rad_pix / 100.)
    return min(max(factor, 0.1), 1.0)
