import os
import numpy as np
from astropy.io import fits
import corgidrp.mocks as mocks
import corgidrp.l3_to_l4 as l3_to_l4
import corgidrp.l4_to_tda as l4_to_tda
from corgidrp.data import Image
import corgidrp.measure_companions as measure_companions
import corgidrp.fluxcal as fluxcal
import corgidrp.nd_filter_calibration as nd_filter_calibration


# Global Constants
HOST_STAR = 'TYC 4433-1800-1'
CFAM = '1F'
FWHM = 3
INPUT_EFFICIENCY_FACTOR = 1
PHOT_METHOD = "aperture"
FLUX_OR_IRR = 'irr'
NUM_IMAGES = 10
ROLL_ANGLES = np.linspace(0, 45 , NUM_IMAGES)
PSF_SUB_SCALE = 0.7
NUMBASIS = [1, 2]
OUT_DIR = os.path.join('corgidrp', 'data', 'L4TestInput')
FULL_SIZE_IMAGE = (1024,1024)
CROPPED_IMAGE_SIZE = (200,200)

PHOT_ARGS = {
    "aperture": {
        "encircled_radius": 7,
        "frac_enc_energy": 1.0,
        "method": "subpixel",
        "subpixels": 10,
        "background_sub": True,
        "r_in": 5,
        "r_out": 10,
        "centering_method": "xy",
        "centroid_roi_radius": 5,
    },
    "gauss2d": {
        "fwhm": 4,
        "fit_shape": None,
        "background_sub": True,
        "r_in": 5,
        "r_out": 10,
        "centering_method": 'xy',
        "centroid_roi_radius": 5
    }
}[PHOT_METHOD]


def get_fluxcal_factor(image, method, phot_args, flux_or_irr):
    """
    Compute flux calibration factor.

    Args:
        image (corgidrp.data.Image): The direct star image.
        method (str): Photometry method ('aperture' or 'gauss2d').
        phot_args (dict): Arguments for the photometry method.
        flux_or_irr (str): 'flux' or 'irr'.

    Returns:
        fluxcal_factor (corgidrp.fluxcal.FluxcalFactor): Flux calibration factor.
    """
    if method == "aperture":
        return fluxcal.calibrate_fluxcal_aper(image, flux_or_irr=flux_or_irr, phot_kwargs=phot_args)
    else:
        return fluxcal.calibrate_fluxcal_gauss2d(image, flux_or_irr=flux_or_irr, phot_kwargs=phot_args)


def generate_test_data(out_dir):
    """
    Generate mock data: direct star images, coronagraphic frames, and a PSF-subtracted frame.

    Returns:
        ref_star_dataset, host_star_counts, fluxcal_factor, zero_point, dataset_ct,
        ct_cal, FpamFsamCal, psf_sub_frame, coron_data
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Generate core throughput calibration dataset
    dataset_ct, ct_cal, dataset_ct_nomask = mocks.create_mock_ct_dataset_and_cal_file(
        fwhm=50, n_psfs=20, cfam_name=CFAM, save_cal_file=True, image_shape=FULL_SIZE_IMAGE
    )

    scaling_factor, all_scaling_factors = measure_companions.compute_occ_to_unocc_scaling(dataset_ct, 
                                                                    dataset_ct_nomask, 
                                                                    ct_cal, PHOT_ARGS)
    print("scaling factor", all_scaling_factors)
    FpamFsamCal = mocks.create_mock_fpamfsam_cal(save_file=False)

    # 2) Generate reference star dataset (ND filter, etc.)
    ref_star_flux = nd_filter_calibration.compute_expected_band_irradiance(HOST_STAR, CFAM)
    ref_star_dataset = mocks.generate_reference_star_dataset_with_flux(
        n_frames=NUM_IMAGES,
        roll_angles=ROLL_ANGLES,
        flux_erg_s_cm2=ref_star_flux,
        fwhm=FWHM,
        cal_factor=1e10,
        optical_throughput=INPUT_EFFICIENCY_FACTOR,
        color_cor=1.0,
        filter=CFAM,
        fpamname='ND475',
        target_name=HOST_STAR,
        fsm_x=0.0,
        fsm_y=0.0,
        exptime=1.0,
        platescale=21.8,  # mas/pixel
        background=0,
        add_gauss_noise=True,
        noise_scale=1.,
        filedir=OUT_DIR,
        file_save=True,
        shape=CROPPED_IMAGE_SIZE
    )

    # 3) Measure the host star’s total counts & zero point
    host_star_counts, _, _ = fluxcal.aper_phot(
        ref_star_dataset[0], encircled_radius=10, frac_enc_energy=1.0,
        method='subpixel', subpixels=5, background_sub=True, r_in=12, r_out=20,
        centering_method='xy', centroid_roi_radius=10
    )
    host_star_mag = l4_to_tda.determine_app_mag(ref_star_dataset[0], HOST_STAR)

    print(f"\nMeasured host star counts: {host_star_counts:.5f} photoelectrons")
    print("Host star mag:", host_star_mag[0].ext_hdr['APP_MAG'])

    fluxcal_factor = get_fluxcal_factor(ref_star_dataset[0], PHOT_METHOD, PHOT_ARGS, FLUX_OR_IRR)
    zero_point = fluxcal_factor.ext_hdr.get('ZP', None)

    # 4) Generate coronagraphic frames (not PSF-subtracted)
    host_star_center = tuple(x // 2 for x in FULL_SIZE_IMAGE)
    coron_data = mocks.generate_coron_dataset_with_companions(
        n_frames=NUM_IMAGES,
        shape=FULL_SIZE_IMAGE,
        host_star_center=host_star_center,
        host_star_counts=host_star_counts,
        roll_angles=ROLL_ANGLES,
        companion_xy=(host_star_center[0] + 20, host_star_center[1] - 20),  
        companion_counts=host_star_counts / 2,
        filter='1F',
        platescale=0.0218,
        add_noise=True,
        noise_std=1.0e-2,
        outdir=OUT_DIR,
        darkhole_file=dataset_ct[0],
        apply_coron_mask=True,
        coron_mask_radius=20,
        ct_cal=ct_cal,
        use_ct_cal=True,
        cor_dataset=dataset_ct,
        FpamFsamCal=FpamFsamCal,
        scaling_factors = all_scaling_factors
    )

    print("Companion 1 AP Mag:", (-2.5 * np.log10(host_star_counts / 2) + zero_point))

    # 5) Make a “PSF-subtracted” frame with that companion
    psf_sub_dataset = l3_to_l4.do_psf_subtraction(coron_data, ct_calibration=ct_cal,
                                                  numbasis=NUMBASIS,
                                                  do_crop=True, crop_sizexy=CROPPED_IMAGE_SIZE)

    # Assuming psf_sub_dataset[0] is a multi-frame Image:
    psf_sub_image = measure_companions.extract_single_frame(psf_sub_dataset[0], frame_index=0)

    # update companion location in SNYX header due to frame cropping
    comp_keywords = [key for key in psf_sub_image.ext_hdr.keys() if key.startswith("SNYX")]
    comp_keyword = comp_keywords[0]
    final_psf_sub_image = measure_companions.update_companion_location_in_cropped_image(psf_sub_image, comp_keyword, 
                                                                                  tuple(x // 2 for x in FULL_SIZE_IMAGE), 
                                                                                  tuple(x // 2 for x in CROPPED_IMAGE_SIZE))
    output_filename = "final_psf_sub_image.fits"
    final_psf_sub_image.save(filedir=OUT_DIR, filename=output_filename)
    print(f"Saved final PSF-subtracted image to: {os.path.join(OUT_DIR, output_filename)}")

    return ref_star_dataset, host_star_counts, fluxcal_factor, zero_point, \
           dataset_ct, ct_cal, FpamFsamCal, final_psf_sub_image, coron_data, all_scaling_factors


def test_measure_companions_wcs(out_dir):
    """
    Run the full test: measure companion with classical + forward modeling.
    """
    (ref_star_dataset, host_star_counts, fluxcal_factor, zero_point,
     dataset_ct, ct_cal, FpamFsamCal, psf_sub_frame, coron_data, all_scaling_factors) = generate_test_data(out_dir)
    
    # Identify a reference PSF at large separation where mask effects are negligible
    throughput_values = ct_cal.ct_excam[2]  # row 2 contains the core throughput values
    max_index = np.argmax(throughput_values)
    reference_psf = ct_cal.data[max_index]

    # Debugging:
    data = ct_cal.ct_excam.T  # Transpose so that each row is one PSF measurement
    import pandas as pd
    df = pd.DataFrame(data, columns=['x', 'y', 'core_throughput'])
    print("Core Throughput Table:")
    print(df)

    pri_hdr = ct_cal.pri_hdr
    ext_hdr = ct_cal.ext_hdr
    pri_hdr['FILENAME'] = 'MockReferencePSF.fits'

    # Set WCS info for demonstration
    ext_hdr['PLTSCALE'] = 0.0218
    ext_hdr['WCSAXES'] = 2
    ext_hdr['CTYPE1'] = 'RA---TAN'
    ext_hdr['CTYPE2'] = 'DEC--TAN'
    ext_hdr['CRPIX1'] = 100
    ext_hdr['CRPIX2'] = 100
    ext_hdr['CRVAL1'] = 0.0
    ext_hdr['CRVAL2'] = 0.0
    ext_hdr['CDELT1'] = -0.00027778
    ext_hdr['CDELT2'] = 0.00027778
    ext_hdr['PC1_1'] = 1.0
    ext_hdr['PC1_2'] = 0.0
    ext_hdr['PC2_1'] = 0.0
    ext_hdr['PC2_2'] = 1.0

    reference_psf = Image(data_or_filepath=reference_psf, pri_hdr=pri_hdr, ext_hdr=ext_hdr)
    checking_psf = Image(data_or_filepath=ct_cal.data[16], pri_hdr=pri_hdr, ext_hdr=ext_hdr)

    ref_psf_counts, _ = measure_companions.measure_counts(reference_psf, PHOT_METHOD, None, **PHOT_ARGS)
    far_psf_counts, _ = measure_companions.measure_counts(checking_psf, PHOT_METHOD, None, **PHOT_ARGS)
    print("checking counts", ref_psf_counts, far_psf_counts)

    # -- Run companion measurement with forward_model = True
    result_table = measure_companions.measure_companions(
        coron_data, ref_star_dataset, psf_sub_frame,
        ref_psf_min_mask_effect=reference_psf,
        ct_cal=ct_cal, FpamFsamCal=FpamFsamCal,
        phot_method='aperture',
        photometry_kwargs=PHOT_ARGS, 
        fluxcal_factor=fluxcal_factor,
        forward_model=False,
        scaling_factors=all_scaling_factors,
        numbasis = NUMBASIS,
        output_dir=out_dir,
        verbose=True
    )


    print("\nResult Table:\n", result_table)
    # Confirm we get at least 1 companion
    # assert len(result_table) == 1, "Expected 1 measured companion"


if __name__ == "__main__":
    test_measure_companions_wcs(OUT_DIR)
