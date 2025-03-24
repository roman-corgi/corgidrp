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
import corgidrp.corethroughput as corethroughput
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
    photometry_kwargs=None,
    fluxcal_factor=None,
    host_star_in_calspec=True,
    forward_model=False,
    numbasis = [1, 2],
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
      4) Forward-model that PSF with pyklip's FMAstrometry, or do a simpler approach to determine
            what the throughput is and account for losses incurred by PSF subtraction.
      5) Measure each companion's flux ratio & apparent magnitude, return in a table.

    Args:
        coronagraphic_dataset (corgidrp.data.Dataset): A coronagraphic dataset (not PSF-subtracted)
            with a star and a companion (in e-). 
        ref_star_dataset (corgidrp.data.Dataset): Multiple frames of a reference star taken at different
            roll angles, with no coronagraph in place and no companions (in e-).
        psf_sub_image (corgidrp.data.Image): A PSF-subtracted frame of star behind the coronagraph, 
            with companions.
        ref_psf_min_mask_effect (corgidrp.data.Image): An off-axis calibration PSF of a star at some 
            known separation where mask effects are negligible.
        ct_cal (corgidrp.data.CoreThroughputCalibration): A Core Throughput calibration file containing 
            PSFs at different distances from the mask and their positions/throughputs.
        FpamFsamCal (corgidrp.data.FpamFsamCal): FPAM/FSAM to EXCAM transformation matrix.
        phot_method ({'aperture', 'gauss2d'}, optional): Photometry method for measuring 
            companion counts on the (already) PSF-subtracted image.
        photometry_kwargs (dict, optional): A dictionary of keyword arguments to pass to the 
            photometry routines. If None, defaults will be used.
        fluxcal_factor (corgidrp.Data.FluxcalFactor, optional): Fluxcal factor with .ext_hdr['ZP'].
        host_star_in_calspec (bool, optional): If True, fetch the star's magnitude from calspec. 
            Otherwise, use zero_point from fluxcal_factor.
        forward_model (bool, optional): If True, run PyKLIP-based forward modeling to get flux. 
            Otherwise do classical photometry.
        numbasis (list, optional): Number of KLIP modes to retain.
        nwalkers (int, optional): Number of MCMC walkers if forward_model=True.
        nburn (int, optional): Number of MCMC burn steps if forward_model=True.
        nsteps (int, optional): Number of MCMC steps if forward_model=True.
        numthreads (int, optional): Threads for MCMC.
        output_dir (str, optional): Output directory path.
        verbose (bool, optional): If True, print progress messages.

    Returns:
        result_table (astropy.table.Table): A table with columns:
            - id: companion label
            - x, y: location in pixels
            - measured_flux: measured flux in e- (either final companion flux from FM or direct photometry)
            - counts_ratio: companion flux / host star flux
            - mag: apparent magnitude (computed from ratio or flux)
    """
    # Use provided photometry parameters or fallback to defaults.
    if photometry_kwargs is None:
        photometry_kwargs = get_photometry_kwargs(phot_method)

    # i. Measure host star counts from direct/unocculted ref_star_dataset
    host_star_counts, _ = measure_counts(ref_star_dataset[0], phot_method, None, **photometry_kwargs)

    # ii. Measure reference PSF counts (off-axis calibration)
    reference_psf_counts, _ = measure_counts(ref_psf_min_mask_effect, phot_method, None, **photometry_kwargs)

    print("checking input counts", host_star_counts, reference_psf_counts)

    # iii. The ratio of host star flux to off-axis reference PSF flux
    host_to_ref_psf_ratio = host_star_counts / reference_psf_counts

    # Parse companion positions from the PSF-subtracted image header, this image may be smaller than coronagraphic
    #   dataset or reference images
    companions = []
    for key, val in psf_sub_image.ext_hdr.items():
        if key.startswith('SNYX'):
            parts = val.split(',')
            x_val = float(parts[1])
            y_val = float(parts[2])
            companions.append({"id": key, "x": x_val, "y": y_val})

    x_star_in_psf_sub_frame = psf_sub_image.ext_hdr.get('STARLOCX', None)
    y_star_in_psf_sub_frame = psf_sub_image.ext_hdr.get('STARLOCY', None)

    # Get companion locations in coronagraphic dataset
    companions_coron = []
    for key, val in coronagraphic_dataset[0].ext_hdr.items():
        if key.startswith('SNYX'):
            parts = val.split(',')
            x_val = float(parts[1])
            y_val = float(parts[2])
            companions_coron.append({"id": key, "x": x_val, "y": y_val})
    
    # get ct_dataset from ct_cal
    ct_dataset = ct_dataset_from_cal(ct_cal)

    results = []

    # Loop over each companion
    for comp_sub, comp_coron in zip(companions, companions_coron):
        # comp_sub and comp_coron are dictionaries corresponding to the same companion
        x_comp_in_psf_sub_frame, y_comp_in_psf_sub_frame = comp_sub['x'], comp_sub['y']
        x_comp_in_coron_frame, y_comp_in_coron_frame = comp_coron['x'], comp_coron['y']
        
        dx = x_comp_in_psf_sub_frame - x_star_in_psf_sub_frame
        dy = y_comp_in_psf_sub_frame - y_star_in_psf_sub_frame
        guesssep = np.hypot(dx, dy)
        guesspa = np.degrees(np.arctan2(dx, dy)) % 360

        # (B) Get the off-axis PSF from calibration at that radial distance
        throughput_factor, nearest_frame = measure_core_throughput_at_location(
            x_comp_in_psf_sub_frame, y_comp_in_psf_sub_frame, x_star_in_psf_sub_frame, 
            y_star_in_psf_sub_frame, ct_cal, ct_dataset
        )

        # Measure the counts of the off-axis PSF from calibration at that radial distance for
        # core throughput/ mask scaling factor
        #psf_at_location_counts, _ = measure_counts(nearest_frame, phot_method, None, **photometry_kwargs)
        
        #mask_throughput_ratio = psf_at_location_counts / reference_psf_counts

        # Scale that PSF to the star’s flux if it were at (x_c, y_c)
        scaled_star_psf_data = nearest_frame.data * host_to_ref_psf_ratio
        scaled_star_psf_at_companion_loc = Image(
            data_or_filepath=scaled_star_psf_data, 
            pri_hdr=nearest_frame.pri_hdr, 
            ext_hdr=nearest_frame.ext_hdr, 
            err=nearest_frame.err if hasattr(nearest_frame, 'err') else None
        )

        scaled_star_psf_at_companion_loc_counts, _ = measure_counts(
            scaled_star_psf_at_companion_loc, phot_method, 
            None, **photometry_kwargs
        )
        print("star if it was at the companion's location counts:", scaled_star_psf_at_companion_loc_counts)

        # (C) Measure the companion flux in the final PSF-subtracted image 
        psf_sub_companion_counts, _ = measure_counts(
            psf_sub_image, phot_method, (x_comp_in_psf_sub_frame, y_comp_in_psf_sub_frame), 
            **photometry_kwargs
        )

        print("actual companion psf sub counts", psf_sub_companion_counts)

        companion_star_counts_pre_sub, _ = measure_counts(
            coronagraphic_dataset[0], phot_method, (x_comp_in_coron_frame, y_comp_in_coron_frame), 
            **photometry_kwargs
        )

        psf_sub_efficiency = psf_sub_companion_counts /companion_star_counts_pre_sub

        print("PSF sub efficiency", psf_sub_efficiency)

        # Forward model vs. simplified approach:
        if forward_model:
            # (C1) Forward-model approach with PyKLIP

            kl_throughput, ct_throughput, psf_sub_star_at_companion_loc = forward_model_psf(
                coronagraphic_dataset, ref_star_dataset, ct_cal,
                guesssep, guesspa, guessflux=1,  # using a normalized injection
                numbasis=numbasis, nwalkers=nwalkers, nburn=nburn, nsteps=nsteps,
                numthreads=numthreads, outputdir=output_dir
            )

            '''
            comp_keywords = [key for key in psf_sub_star_at_companion_loc.ext_hdr.keys() if key.startswith("SNYX")]
            comp_keyword = comp_keywords[0]
            comp_locations_in_fm = psf_sub_star_at_companion_loc.ext_hdr[comp_keyword]
            print("printing comp location", comp_locations_in_fm)
            parts_in_fm = comp_locations_in_fm.split(',')
            x_val_in_fm = float(parts_in_fm[1])
            y_val_in_fm = float(parts_in_fm[2])
            psf_sub_star_at_companion_loc_counts, _ = measure_counts(
                psf_sub_star_at_companion_loc, phot_method, (x_val_in_fm, y_val_in_fm), 
                **photometry_kwargs
            )

            plt.figure(figsize=(8, 8))
            plt.imshow(psf_sub_star_at_companion_loc.data, origin='lower', cmap='inferno')
            plt.scatter([x_val_in_fm], [y_val_in_fm],
                        s=100, edgecolor='cyan', facecolor='none', linewidth=2,
                        label='Companion Location')
            plt.title('PSF Sub Star at Companion Location')
            plt.legend()
            plt.colorbar(label='Intensity (e-)')
            plt.show()
            '''

            # Using the computed throughput from PyKLIP which forward models the off-axis image of the 
            # host star through the PSF subtraction process, scale the counts
            corrected_psf_sub_companion_counts = psf_sub_companion_counts
            companion_host_ratio = psf_sub_companion_counts  / psf_sub_star_at_companion_loc_counts
        else:
            # (C2) Simplified approach
            psf_sub_star_at_companion_loc, efficiency = simplified_psf_sub(scaled_star_psf_at_companion_loc,
                                                                           ct_cal, guesssep, psf_sub_efficiency)
            
            psf_sub_star_at_companion_loc_counts, _ = measure_counts(
                psf_sub_star_at_companion_loc, phot_method, None, **photometry_kwargs
            )

            corrected_psf_sub_companion_counts = psf_sub_companion_counts
            companion_host_ratio = psf_sub_companion_counts / psf_sub_star_at_companion_loc_counts

        # Plotting for debugging
        psf_sub_star_at_companion_loc_counts, _ = measure_counts(
            psf_sub_star_at_companion_loc, phot_method, None, 
            **photometry_kwargs
        )

        plt.figure(figsize=(8, 8))
        plt.imshow(psf_sub_star_at_companion_loc.data, origin='lower', cmap='inferno')
        plt.title('PSF Sub Star at Companion Location')
        plt.legend()
        plt.colorbar(label='Intensity (e-)')
        plt.show()

        # (D) Compute the companion magnitude
        if host_star_in_calspec:
            apmag_data = l4_to_tda.determine_app_mag(ref_star_dataset[0], ref_star_dataset[0].pri_hdr['TARGET'])
            host_star_apmag = apmag_data[0].ext_hdr['APP_MAG']
            companion_mag = host_star_apmag - 2.5 * np.log10(companion_host_ratio)
        else:
            if fluxcal_factor is None:
                raise ValueError(
                    "No fluxcal_factor provided, but host_star_in_calspec is False. "
                    "Provide fluxcal_factor or set host_star_in_calspec=True."
                )
            zero_point = fluxcal_factor.ext_hdr.get('ZP')
            companion_mag = -2.5 * np.log10(psf_sub_companion_counts) + zero_point

        results.append((
            comp_sub['id'],
            x_comp_in_psf_sub_frame,
            y_comp_in_psf_sub_frame,
            corrected_psf_sub_companion_counts,
            psf_sub_star_at_companion_loc_counts,
            companion_host_ratio,
            companion_mag
        ))

    result_table = Table(
        rows=results,
        names=[
            'id', 'x', 'y',
            'measured companion counts',  # e- (either final companion flux from FM or direct photometry)
            'simulated host star at companion location counts',
            'counts_ratio',
            'mag'
        ]
    )
    return result_table


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def ct_dataset_from_cal(ct_cal):
    psf_cube = ct_cal.data  # shape (N, height, width)
    psf_images = []
    for i in range(psf_cube.shape[0]):
        psf_img = Image(psf_cube[i], pri_hdr=ct_cal.pri_hdr, ext_hdr=ct_cal.ext_hdr)
        psf_images.append(psf_img)
    dataset_ct = Dataset(psf_images)
    return dataset_ct


def measure_counts(input_image_or_dataset, phot_method, initial_xy_guess, **kwargs):
    if phot_method == 'gauss2d':
        flux, flux_err, *_ = fluxcal.phot_by_gauss2d_fit(
            input_image_or_dataset, centering_initial_guess=initial_xy_guess, **kwargs
        )
        return flux, flux_err
    elif phot_method == 'aperture':
        flux, flux_err, *_ = fluxcal.aper_phot(
            input_image_or_dataset, centering_initial_guess=initial_xy_guess, **kwargs
        )
        return flux, flux_err
    else:
        raise ValueError(f"Invalid photometry method: {phot_method}")


def get_photometry_kwargs(phot_method):
    """
    (Optional) Default photometry kwargs if none are provided.
    """
    common_kwargs = {'centering_method': 'xy', 'centroid_roi_radius': 5}
    photometry_options = {
        'gauss2d': {
            'fwhm': 4,
            'background_sub': True,
            'r_in': 5,
            'r_out': 10,
            **common_kwargs
        },
        'aperture': {
            'encircled_radius': 7,
            'frac_enc_energy': 1.0,
            'subpixels': 10,
            'background_sub': True,
            'r_in': 5,
            'r_out': 10,
            **common_kwargs
        }
    }
    if phot_method not in photometry_options:
        raise ValueError(
            f"Invalid photometry method '{phot_method}'. Choose from {list(photometry_options.keys())}."
        )
    return photometry_options[phot_method]


def simplified_psf_sub(psf_frame, ct_cal, guesssep, psf_sub_efficiency):
    # TO DO: something with the ct_throughput value?
    closest_sep, idx, ct_throughput = lookup_core_throughput(ct_cal, guesssep)
    print("guesssep", guesssep, ct_throughput)
    psf_sub_frame = psf_frame.data * psf_sub_efficiency
    psf_sub_image = Image(
        data_or_filepath=psf_sub_frame,
        pri_hdr=psf_frame.pri_hdr,
        ext_hdr=psf_frame.ext_hdr,
        err=psf_frame.err if hasattr(psf_frame, 'err') else None
    )
    return psf_sub_image, ct_throughput


def lookup_core_throughput(ct_cal, desired_sep):
    """
    Look up the core throughput value closest to a given pixel separation.
    
    Args:
        ct_table (numpy.ndarray): A 2D array of shape (N, 3) where each row contains
            [x, y, core_throughput] for a calibration PSF.
        starloc (tuple): The (x, y) position of the star in the cropped image,
            e.g., (STARLOCX, STARLOCY).
        desired_sep (float): The desired pixel separation from the star center.
    
    Returns:
        tuple: A tuple (closest_sep, throughput) where:
            - closest_sep (float): The separation value (in pixels) from the ct_table
              that is closest to desired_sep.
            - throughput (float): The corresponding core throughput value.
    """
    # Existing lookup methods weren't working so i'm writing this one for now. 
    x, y, ct = ct_cal.ct_excam
    masklocx = ct_cal.ext_hdr['MASKLOCX']
    masklocy = ct_cal.ext_hdr['MASKLOCY']

    # Compute separation for each PSF relative to the star location.
    separations = np.sqrt((x - masklocx)**2 + (y - masklocy)**2)

    # DEBUGGING ct values 
    '''
    import pandas as pd
    debug_df = pd.DataFrame({
        'separation': separations,
        'ct': ct
    })
    print("Debug table of separation and CT values:")
    print(debug_df)
    '''
    
    # Find the index where the separation is closest to desired_sep.
    idx = np.argmin(np.abs(separations - desired_sep))
    
    closest_sep = separations[idx]
    throughput = ct[idx]
    
    return closest_sep, idx, throughput


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
    fm_dataset = coronagraphic_dataset.copy()
    injected_images = []
    for idx, frame in enumerate(fm_dataset):
        injected_frame, psf_model, _ = klip_fm.inject_psf(
            frame_in=frame,
            ct_calibration=ct_calibration,
            amp=guessflux,
            sep_pix=guesssep,
            pa_deg=guesspa,
            norm=inject_norm
        )
        fm_dataset[idx].data = injected_frame.data
        injected_images.append(injected_frame.data)

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
        crop_sizexy=(100,100),
        measure_klip_thrupt=True,
        measure_1d_core_thrupt=True
    )
    klip_data = fm_psfsub[0].data[-1]
    if plot_results:
        fig2, ax2 = plt.subplots(figsize=(6,6))
        norm2 = ImageNormalize(klip_data, interval=ZScaleInterval())
        im2 = ax2.imshow(klip_data, origin='lower', cmap='inferno', norm=norm2)
        fig2.colorbar(im2, ax=ax2, label="Intensity (e-)")
        ax2.set_title("Final PSF-sub image")
        plt.show()


    # DEBUGGING
    kl_throughput = fm_psfsub[0].hdu_list['KL_THRU'].data
    ct_throughput = fm_psfsub[0].hdu_list['CT_THRU'].data

    #print("KL Throughput:", kl_throughput)
    #print("CT Throughput:", ct_throughput)

    # For the throughput tables:
    # The top row (index 0) is the separations.
    # For KL throughput, use the last row (index -1) corresponding to the last KL mode.
    kl_idx = np.argmin(np.abs(kl_throughput[0] - guesssep))
    kl_value = kl_throughput[-1, kl_idx]
    
    # For CT throughput, there are two rows:
    # the top row is separations, the second row (index 1) is the CT throughput.
    ct_idx = np.argmin(np.abs(ct_throughput[0] - guesssep))
    ct_value = ct_throughput[1, ct_idx]

    '''
    # DEBUGGING
    print("Computed companion separation (pixels):", guesssep)
    print("Closest separation in KL throughput table:", kl_throughput[0, kl_idx])
    print("Selected KL throughput (last KL mode):", kl_value)
    print("Closest separation in CT throughput table:", ct_throughput[0, ct_idx])
    print("Selected CT throughput:", ct_value)
    '''

    # Return the klip_data frame to do aperture photometry, but set the negative numbers to 0
    klip_data[klip_data < 0] = 0
    klip_image = Image(data_or_filepath=klip_data,
                      pri_hdr=fm_psfsub[0].pri_hdr,
                      ext_hdr=fm_psfsub[0].ext_hdr)
    
    comp_keywords = [key for key in fm_psfsub[0].ext_hdr.keys() if key.startswith("SNYX")]
    comp_keyword = comp_keywords[0]
    # TO DO: don't hard code this
    klip_image = update_companion_location_in_cropped_image(klip_image, comp_keyword, (512,512), 
                                                            (50,50))
    # Not sure how to use what the below code returns. It is in the KLIP documentation for 
    # doing astrometry but the documentation isn't clear to me on what "raw_flux" scaling 
    # factor is or how to use it. It doesn't look correct to me. I'll go with Ell's method 
    # for now.
    '''
    fit = FMAstrometry(guesssep, -guesspa, stamp_size, method=method)
    filename = os.path.join(outputdir, "ADI", fileprefix + "-KLmodes-all.fits")
    fm_hdu = fits.open(filename)
    fm_frame = fm_hdu[0].data[1]
    fm_centx = fm_hdu[0].header['PSFCENTX']
    fm_centy = fm_hdu[0].header['PSFCENTY']
    fm_hdu.close()

    data_filename = os.path.join(outputdir, "ADI", "FAKE_1KLMODES-KLmodes-all.fits")
    if os.path.exists(data_filename):
        data_hdu = fits.open(data_filename)
        data_frame = data_hdu[0].data[0]
        data_centx = data_hdu[0].header["PSFCENTX"]
        data_centy = data_hdu[0].header["PSFCENTY"]
        data_hdu.close()
    else:
        data_frame = fm_frame
        data_centx = fm_centx
        data_centy = fm_centy

    fit.generate_fm_stamp(fm_frame, [fm_centx, fm_centy], padding=5)
    fit.generate_data_stamp(data_frame, [data_centx, data_centy], dr=4, exclusion_radius=10)
    fit.set_kernel("matern32", [3.], [r"$l$"])
    fit.set_bounds(1.5, 1.5, 1., [1.])
    fit.fit_astrometry(nwalkers=nwalkers, nburn=nburn, nsteps=nsteps, numthreads=numthreads)'
    '''
    return kl_value, ct_value, klip_image


def measure_core_throughput_at_location(x_c, y_c, x_star, y_star, ct_cal, ct_dataset):
    dx = x_c - x_star
    dy = y_c - y_star
    guesssep = np.hypot(dx, dy)
    closest_sep, idx, throughput = lookup_core_throughput(ct_cal, guesssep)
    
    # Convert ct_dataset (a Dataset) to a list of frames and pick the one at index idx.
    corresponding_frame = list(ct_dataset)[idx]
    
    return throughput, corresponding_frame


def update_companion_location_in_cropped_image(image, comp_keyword, old_host, new_host):
    """
    Update the companion location in the cropped image and return the updated image.

    Args:
        image (corgidrp.data.Image): The image object whose header will be updated.
        comp_keyword (str): The header keyword for the companion location (e.g., "SNYX001").
        old_host (tuple): The (x, y) position of the host star in the original image.
        new_host (tuple): The (x, y) position of the host star in the cropped image.

    Returns:
        image (corgidrp.data.Image): The image object with the updated companion location in its header.

    Raises:
        KeyError: If comp_keyword is not found in the image header.
        ValueError: If the companion location format is not as expected or cannot be parsed.
    """
    # Assume the companion information is stored in the extension header.
    ext_hdr = image.ext_hdr

    if comp_keyword not in ext_hdr:
        raise KeyError(f"Keyword {comp_keyword} not found in image header.")

    # Parse the companion value; expected format: "sn_value,x,y"
    parts = ext_hdr[comp_keyword].split(',')
    if len(parts) < 3:
        raise ValueError(f"Unexpected format for companion location in {comp_keyword}: {ext_hdr[comp_keyword]}")
    
    try:
        sn_val = float(parts[0])
        old_comp_x = float(parts[1])
        old_comp_y = float(parts[2])
    except Exception as e:
        raise ValueError(f"Error parsing {comp_keyword} in image header: {e}")

    # Compute new companion location in the cropped image.
    new_comp_x = old_comp_x - old_host[0] + new_host[0]
    new_comp_y = old_comp_y - old_host[1] + new_host[1]

    # Convert to integer pixel coordinates.
    new_comp_x = int(round(new_comp_x))
    new_comp_y = int(round(new_comp_y))

    # Update the companion location in the header.
    ext_hdr[comp_keyword] = f"{sn_val:5.1f},{new_comp_x:4d},{new_comp_y:4d}"

    return image


def extract_single_frame(image, frame_index=0):
    """
    Extract a single frame from a multi-frame corgidrp.data.Image object along with its 
    corresponding error and DQ arrays, and preserve the KL_THRU and CT_THRU HDUs.

    Args:
        image (corgidrp.data.Image): The multi-frame image object.
        frame_index (int, optional): The index of the frame to extract (default is 0).

    Returns:
        corgidrp.data.Image: A new Image object consisting of the selected data frame,
        its corresponding error and DQ arrays, and with the KL_THRU and CT_THRU HDUs assigned.
    """
    # Extract the data frame. Assume image.data is a NumPy array of shape (n_frames, height, width)
    data_frame = image.data[frame_index, :, :]

    # Extract the error array corresponding to this frame.
    if image.err is not None:
        # Adjust slicing according to the structure of image.err.
        # If image.err has shape (n_err, n_frames, height, width), we pick index 0 for error type.
        if image.err.ndim == 4:
            err_frame = image.err[0, frame_index, :, :]
        else:
            err_frame = image.err[frame_index, :, :]
    else:
        err_frame = None

    # Extract the DQ array corresponding to this frame.
    dq_frame = image.dq[frame_index, :, :] if image.dq is not None else None

    # Build a new Image object with the extracted components.
    # Note: We do not pass hdu_list as a keyword argument.
    new_image = Image(data_frame,
                      pri_hdr=image.pri_hdr,
                      ext_hdr=image.ext_hdr,
                      err=err_frame,
                      dq=dq_frame)
    
    # Now assign the HDU list attribute from the original image.
    if hasattr(image, 'hdu_list'):
        new_image.hdu_list = {key: image.hdu_list[key] for key in image.hdu_list if key in ['KL_THRU', 'CT_THRU']}
    
    return new_image
