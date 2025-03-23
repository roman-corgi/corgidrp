import os
import numpy as np
import math
from skimage.transform import resize
from astropy.table import Table
import corgidrp.fluxcal as fluxcal
import corgidrp.klip_fm as klip_fm
import pyklip
import pyklip.fm as fm
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fitpsf as fitpsf
import corgidrp.mocks as mocks
import pyklip.rdi as rdi
from corgidrp.data import PyKLIPDataset, Image, Dataset
import corgidrp.l4_to_tda as l4_to_tda   
import corgidrp.l3_to_l4 as l3_to_l4 
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from pyklip.fitpsf import FMAstrometry
from astropy.visualization import ZScaleInterval, ImageNormalize, PercentileInterval


def measure_companions(
    coronagraphic_dataset,
    ref_star_dataset,
    psf_sub_image,
    ref_psf_min_mask_effect,
    ct_cal,
    FpamFsamCal,
    phot_method='aperture',
    fluxcal_factor=None,
    host_star_in_calspec=True,
    forward_model=False,
    nwalkers=10,
    nburn=5,
    nsteps=20,
    numthreads=1,
    output_dir=".",
    verbose=True
):
    """
    Measure companion properties in a final (or intermediate) coronagraphic image,
    returning position, counts ratio (companion/starlight), and apparent magnitude.

    This function does the following:
      1) Measure host star counts in a direct (unocculted) image.
      2) Measure reference PSF counts (from an off-axis calibration).
      3) Compute a ratio for scaling that PSF to represent the star at companion's location.
      4) (Optional) Forward-model that PSF with pyklip's FMAstrometry, or do a simpler 
         "classical" approach.
      5) Measure each companion's flux ratio & apparent magnitude, return in a table.

    Args:
    coronagraphic_dataset (corgidrp.data.Dataset): A coronagraphic dataset (not PSF-subtracted)
        with a star and a companion (in e-). 
    ref_star_dataset (corgidrp.data.Dataset): Multiple frames of a reference star taken at different
        roll angles, with no coronagraph in place and no companions (in e-).
    psf_sub_image (corgidrp.data.Image): A PSF-subtracted frame of star behind the coronagraph, with 
        companions
    ref_psf_min_mask_effect (corgidrp.data.Image): An off-axis calibration PSF of a star at some known 
        separation where mask effects are negligible (it is near 6 lam/D).
    ct_dataset (corgidrp.data.Dataset): The Core Throughput calibration dataset.
    ct_cal (corgidrp.data.CoreThroughputCalibration): A Core Throughput calibration file containing 
        PSFs at different distances from the mask and their corresponding positions and throughputs.
    phot_method ({'aperture', 'gauss2d'}, optional): Photometry method for measuring 
        companion counts on the (already) PSF-subtracted image.
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

    # (A) Get default photometry parameters for aperture or gauss2d
    phot_method_kwargs = get_photometry_kwargs(phot_method)

    # i. Measure host star counts from direct/unocculted ref_star_dataset
    host_star_counts, _ = measure_counts(ref_star_dataset[0], phot_method, None, **phot_method_kwargs)

    # ii. Measure reference PSF counts (off-axis calibration)
    reference_psf_counts, _ = measure_counts(ref_psf_min_mask_effect, phot_method, None, **phot_method_kwargs)

    # iii. The ratio of host star flux to off-axis reference PSF flux
    host_psf_ratio = host_star_counts / reference_psf_counts

    # Parse companion positions from the PSF-subtracted image header
    companions = []
    for key, val in psf_sub_image.ext_hdr.items():
        if key.startswith('SNYX'):
            parts = val.split(',')
            x_val = float(parts[1])
            y_val = float(parts[2])
            companions.append({"id": key, "x": x_val, "y": y_val})

    if verbose:
        print(f"Found {len(companions)} companion(s) in header.")
    
    # get ct_dataset from ct_cal
    ct_dataset = ct_dataset_from_cal(ct_cal)
    
    results = []

    # Loop over each companion
    for comp in companions:
        x_c, y_c = comp['x'], comp['y']

        # (B) Get the off-axis PSF from calibration at that radial distance
        throughput_factor, nearest_frame = measure_core_throughput_at_location(
            x_c, y_c, ct_cal, ct_dataset, FpamFsamCal
        )
        # Scale the nearest_frame to the star's flux if the star were located at (x_c, y_c)
        scaled_psf_data = nearest_frame.data * host_psf_ratio

        scaled_psf_at_companion_loc = Image(
            data_or_filepath=scaled_psf_data, 
            pri_hdr=nearest_frame.pri_hdr, 
            ext_hdr=nearest_frame.ext_hdr, 
            err=nearest_frame.err if hasattr(nearest_frame, 'err') else None
        )

        # Store two key fluxes:
        #  - psf_sub_companion_counts (the measured companion flux)
        #  - psf_sub_star_at_companion_loc_counts (the flux we'd measure for the star 
        #    at that location). Then the ratio is companion/star_at_loc.

        if forward_model:
            # (C1) Forward-model approach with PyKLIP
            x_star = coronagraphic_dataset[0].ext_hdr.get('STARLOCX', None)
            y_star = coronagraphic_dataset[0].ext_hdr.get('STARLOCY', None)
            dx = x_c - x_star
            dy = y_c - y_star
            guesssep = np.hypot(dx, dy)
            guesspa = np.degrees(np.arctan2(dx, dy)) % 360

            # The amplitude guess: measure flux of the scaled PSF
            psf_guess_counts, _ = measure_counts(
                scaled_psf_at_companion_loc, phot_method, None, **phot_method_kwargs
            )
            # Pass that as guessflux
            fit = forward_model_psf(
                coronagraphic_dataset, ref_star_dataset, ct_cal,
                guesssep, guesspa, guessflux=psf_guess_counts,
                nwalkers=nwalkers, nburn=nburn, nsteps=nsteps, numthreads=numthreads,
                outputdir=output_dir
            )

            # 1) "raw_flux" is a ParamRange with .bestfit, .error, etc.
            companion_flux_scale = fit.raw_flux.bestfit
            companion_flux_scale_err = fit.raw_flux.error
            print("Best-fit flux scale:", companion_flux_scale)
            print("Flux scale uncertainty:", companion_flux_scale_err)

            # 2) The final companion flux is scale * guessflux
            companion_flux = companion_flux_scale * psf_guess_counts

            # This is our "measured companion counts"
            psf_sub_companion_counts = companion_flux

            # The "star flux at that location" is the overall host_star_counts 
            # (does this ignore any local throughput? if it does, to incorporate 
            #  local mask throughput, could do star_flux_at_loc = host_star_counts * (some factor). 
            psf_sub_star_at_companion_loc_counts = host_star_counts

        else:
            # (C2) Classical approach:
            # Subtract star at that location
            psf_sub_star_at_companion_loc, efficiency = classical_psf_sub(scaled_psf_at_companion_loc)

            # Then measure flux in that subbed star image
            psf_sub_star_at_companion_loc_counts, _ = measure_counts(
                psf_sub_star_at_companion_loc, phot_method, None, **phot_method_kwargs
            )

            # Also measure the companion flux in the final PSF-subtracted image
            psf_sub_companion_counts, _ = measure_counts(
                psf_sub_image, phot_method, (x_c, y_c), **phot_method_kwargs
            )

        # (D) If forward_model, already have `psf_sub_companion_counts` & 
        #     `psf_sub_star_at_companion_loc_counts`.
        #     Otherwise, define them from the lines above.
        if not forward_model:
            companion_host_ratio = (
                psf_sub_companion_counts / psf_sub_star_at_companion_loc_counts
            )
        else:
            companion_host_ratio = psf_sub_companion_counts / psf_sub_star_at_companion_loc_counts

        # (E) Convert flux ratio to magnitude
        if host_star_in_calspec:
            # The star's apparent magnitude in this band
            apmag_data = l4_to_tda.determine_app_mag(ref_star_dataset[0], ref_star_dataset[0].pri_hdr['TARGET'])
            host_star_apmag = apmag_data[0].ext_hdr['APP_MAG']
            companion_mag = host_star_apmag - 2.5 * np.log10(companion_host_ratio)
        else:
            # Use zero point from fluxcal_factor
            if fluxcal_factor is None:
                raise ValueError("No fluxcal_factor provided, but host star is not in calspec. Provide fluxcal_factor.")
            zero_point = fluxcal_factor.ext_hdr.get('ZP')
            # The companion flux is "psf_sub_companion_counts"
            counts_corr = psf_sub_companion_counts
            companion_mag = -2.5 * np.log10(counts_corr) + zero_point

        # (F) Append to results
        if forward_model:
            results.append((
                comp['id'],
                x_c,
                y_c,
                psf_sub_companion_counts,          # final companion flux
                psf_sub_star_at_companion_loc_counts,
                companion_host_ratio,
                companion_mag
            ))
        else:
            results.append((
                comp['id'],
                x_c,
                y_c,
                psf_sub_companion_counts,
                psf_sub_star_at_companion_loc_counts,
                companion_host_ratio,
                companion_mag
            ))

    # Build a table with consistent columns
    result_table = Table(
        rows=results,
        names=[
            'id', 'x', 'y',
            'psf_sub_companion_counts',
            'psf_sub_star_at_companion_loc_counts',
            'counts_ratio',
            'mag'
        ]
    )
    return result_table


# ---------------- Helper functions -------------------------------------- #
def ct_dataset_from_cal(ct_cal):
    psf_cube = ct_cal.data  # shape (N, 19, 19)
    psf_images = []
    for i in range(psf_cube.shape[0]):
        psf_img = Image(psf_cube[i], pri_hdr=ct_cal.pri_hdr, ext_hdr=ct_cal.ext_hdr)
        psf_images.append(psf_img)
    dataset_ct = Dataset(psf_images)
    return dataset_ct

def measure_counts(input_image_or_dataset, phot_method, initial_xy_guess, **kwargs):
    """
    Measure counts in an image using the specified phot_method 
    ('aperture' or 'gauss2d'), returning (flux, flux_err).
    """
    if phot_method == 'gauss2d':
        return fluxcal.phot_by_gauss2d_fit(
            input_image_or_dataset, centering_initial_guess=initial_xy_guess, **kwargs
        )[:2]
    elif phot_method == 'aperture':
        return fluxcal.aper_phot(
            input_image_or_dataset, centering_initial_guess=initial_xy_guess, **kwargs
        )[:2]
    else:
        raise ValueError(f"Invalid photometry method: {phot_method}")


def get_photometry_kwargs(phot_method):
    """
    Returns method-specific default keyword args for aperture or gauss2d photometry.
    """
    common_kwargs = {'centering_method': 'xy', 'centroid_roi_radius': 5}

    photometry_options = {
        'gauss2d': {
            'fwhm': 3, 'background_sub': True, 'r_in': 5, 'r_out': 10, **common_kwargs
        },
        'aperture': {
            'encircled_radius': 4, 'frac_enc_energy': 1, 'subpixels': 5, 
            'background_sub': True, 'r_in': 6, 'r_out': 12, **common_kwargs
        }
    }

    if phot_method not in photometry_options:
        raise ValueError(
            f"Invalid photometry method '{phot_method}'. "
            f"Choose from {list(photometry_options.keys())}."
        )
    return photometry_options[phot_method]


def classical_psf_sub(psf_frame):
    """
    Placeholder function for classical PSF subtraction.
    Returns a new image with the star's flux scaled by an efficiency factor,
    plus the factor itself.
    """
    efficiency = 0.7
    psf_sub_frame = psf_frame.data * efficiency
    psf_sub_image = Image(
        data_or_filepath=psf_sub_frame, 
        pri_hdr=psf_frame.pri_hdr, 
        ext_hdr=psf_frame.ext_hdr, 
        err=psf_frame.err if hasattr(psf_frame, 'err') else None
    )
    return psf_sub_image, efficiency


def forward_model_psf(
    coronagraphic_dataset,
    reference_star_dataset,
    ct_calibration,
    guesssep,
    guesspa,
    guessflux,
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
    plot_results=True
):
    """
    Example forward-model routine using klip_fm.inject_psf + PyKLIP's FMAstrometry.
    Returns the fitted FMAstrometry object.
    """
    fm_dataset = coronagraphic_dataset.copy()
    injected_images = []

    # 1) Inject a fake planet into each science frame
    for idx, frame in enumerate(fm_dataset):
        injected_frame, psf_model, _ = klip_fm.inject_psf(
            frame_in=frame,
            ct_calibration=ct_calibration,
            amp=guessflux,     # total flux (e-) to inject
            sep_pix=guesssep,
            pa_deg=guesspa,
            norm=inject_norm
        )
        fm_dataset[idx].data = injected_frame.data
        injected_images.append(injected_frame.data)

    # Optional plotting
    if plot_results:
        n_frames = len(injected_images)
        cols = int(np.ceil(np.sqrt(n_frames)))
        rows = int(np.ceil(n_frames / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        if n_frames == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for i, img in enumerate(injected_images):
            norm = ImageNormalize(img, interval=ZScaleInterval())
            axes[i].imshow(img, origin='lower', cmap='inferno', norm=norm)
            axes[i].set_title(f"Injected Frame {i}")
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()

    # 2) Run KLIP to subtract the stellar PSF
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
        do_crop=False,
        measure_klip_thrupt=True,
        measure_1d_core_thrupt=True
    )
    klip_data = fm_psfsub[0].data[-1]  # final KL mode

    if plot_results:
        fig2, ax2 = plt.subplots(figsize=(6,6))
        norm2 = ImageNormalize(klip_data, interval=ZScaleInterval())
        im2 = ax2.imshow(klip_data, origin='lower', cmap='inferno', norm=norm2)
        fig2.colorbar(im2, ax=ax2, label="Intensity")
        ax2.set_title("Final PSF-sub image")
        plt.show()

    # 3) FMAstrometry
    fit = FMAstrometry(guesssep, -guesspa, stamp_size, method=method)

    # Load forward-modelled frame from disk
    filename = os.path.join(outputdir, "ADI", fileprefix + "-KLmodes-all.fits")
    fm_hdu = fits.open(filename)
    fm_frame = fm_hdu[0].data[1]
    fm_centx = fm_hdu[0].header['PSFCENTX']
    fm_centy = fm_hdu[0].header['PSFCENTY']
    fm_hdu.close()

    # Load data stamp frame from separate file
    data_filename = os.path.join(outputdir, "ADI", "FAKE_1KLMODES-KLmodes-all.fits")
    data_hdu = fits.open(data_filename)
    data_frame = data_hdu[0].data[0]
    data_centx = data_hdu[0].header["PSFCENTX"]
    data_centy = data_hdu[0].header["PSFCENTY"]
    data_hdu.close()

    fit.generate_fm_stamp(fm_frame, [fm_centx, fm_centy], padding=5)
    fit.generate_data_stamp(data_frame, [data_centx, data_centy], dr=4, exclusion_radius=10)

    # Possibly plot these stamps, etc.
    fit.set_kernel("matern32", [3.], [r"$l$"])
    fit.set_bounds(1.5, 1.5, 1.,[1.]) # x_range, y_range, flux_range, hyperparams_range
    fit.fit_astrometry(nwalkers=nwalkers, nburn=nburn, nsteps=nsteps, numthreads=numthreads)

    return fit


def measure_core_throughput_at_location(
    x_c, y_c,
    ct_cal,
    ct_dataset,
    FpamFsamCal
):
    """
    Given a companion location x_c, y_c, 
    find the nearest precomputed calibration PSF in ct_cal and return it.
    """
    star_center = ct_cal.GetCTFPMPosition(ct_dataset, FpamFsamCal)[0]
    dx, dy = x_c - star_center[0], y_c - star_center[1]

    # ct_cal.ct_excam => shape (3, N). The third row is throughput.
    xyvals = ct_cal.ct_excam[:2]
    throughvals = ct_cal.ct_excam[2]

    # find nearest index
    idx_best = int(np.argmin(np.hypot(xyvals[0] - dx, xyvals[1] - dy)))
    throughput_factor = max(throughvals[idx_best], 1.0)

    # assume ct_dataset is aligned with ct_cal.ct_excam
    nearest_frame = ct_dataset[idx_best]
    
    return throughput_factor, nearest_frame
