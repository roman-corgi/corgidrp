import os
import numpy as np
import corgidrp.mocks as mocks
import corgidrp.measure_companions as measure_companions
import corgidrp.fluxcal as fluxcal
import corgidrp.nd_filter_calibration as nd_filter_calibration


# Global Constants
INPUT_STARS = ['109 Vir']
FWHM = 3
CAL_FACTOR = 0.8
PHOT_METHOD = "Gaussian"
FLUX_OR_IRR = 'irr'

PHOT_ARGS = {
    "Aperture": {
        "encircled_radius": 7,
        "frac_enc_energy": 1.0,
        "method": "subpixel",
        "subpixels": 10,
        "background_sub": True,
        "r_in": 5,
        "r_out": 10,
        "centering_method": "xy",
        "centroid_roi_radius": 5,
        "centering_initial_guess": None
    },
    "Gaussian": {
        "fwhm": 4,
        "fit_shape": None,
        "background_sub": True,
        "r_in": 5,
        "r_out": 10,
        "centering_method": 'xy',
        "centroid_roi_radius": 5,
        "centering_initial_guess": None
    }
}[PHOT_METHOD]


def get_fluxcal_factor(image, method, phot_args, flux_or_irr):
    """
    Compute flux calibration factor.

    Parameters:
    image (corgidrp.data.Image): The direct star image.
    method (str): Photometry method ('Aperture' or 'Gaussian').
    phot_args (dict): Arguments for the photometry method.
    flux_or_irr (str): 'flux' or 'irr'.

    Returns:
    fluxcal_factor (corgidrp.fluxcal.FluxcalFactor): Flux calibration
        factor.
    """
    if method == "Aperture":
        return fluxcal.calibrate_fluxcal_aper(image, flux_or_irr=flux_or_irr, phot_kwargs=phot_args)
    return fluxcal.calibrate_fluxcal_gauss2d(image, flux_or_irr=flux_or_irr, phot_kwargs=phot_args)


def generate_test_data(out_dir):
    """
    Generate mock test data: direct star image and PSF-subtracted frame.

    Parameters:
    out_dir (str): Output directory for saved FITS files.

    Returns:
    direct_star_image (corgidrp.data.Image):
    host_star_counts (float):
    zero_point (float):
    ct_cal (corgidrp.data.CoreThroughputCalibration):
    FpamFsamCal (corgidrp.data.FpamFsamCal):
    psf_sub_frame (corgidrp.data.Image):
    """
    os.makedirs(out_dir, exist_ok=True)

    # Generate Direct Star Image
    star_flux = nd_filter_calibration.compute_expected_band_irradiance(INPUT_STARS[0], "3C")
    direct_star_image = mocks.create_flux_image(
        star_flux, FWHM, CAL_FACTOR, "3C", "HOLE", INPUT_STARS[0],
        fsm_x=0, fsm_y=0, exptime=5, filedir=out_dir,
        color_cor=1.0, platescale=21.8, background=0,
        add_gauss_noise=False, noise_scale=1.0, file_save=True
    )

    # Compute Host Star Counts
    host_star_counts, _, _ = fluxcal.aper_phot(
        direct_star_image, encircled_radius=10, frac_enc_energy=1.0, method='subpixel',
        subpixels=5, background_sub=True, r_in=12, r_out=20, centering_method='xy',
        centroid_roi_radius=10
    )

    print(f"\nMeasured host star counts: {host_star_counts:.5f} photoelectrons")

    # Compute Zero Point
    fluxcal_factor = get_fluxcal_factor(direct_star_image, PHOT_METHOD, PHOT_ARGS, FLUX_OR_IRR)
    zero_point = fluxcal_factor.ext_hdr.get('ZP', None)

    # Generate Calibration Data
    dataset_ct, ct_cal = mocks.create_mock_ct_dataset_and_cal_file(fwhm=50, n_psfs=20, cfam_name='3C', save_cal_file=True)
    FpamFsamCal = mocks.create_mock_fpamfsam_cal(save_file=False)

    # Generate PSF-subtracted Frame
    psf_sub_frame = mocks.generate_psfsub_image_with_companions(
        nx=200, ny=200, host_star_center=None, host_star_counts=host_star_counts,
        psf_sub_scale=0.7, companion_xy=[(120, 80), (90, 130)],
        companion_counts=[host_star_counts / 2, host_star_counts / 3],
        companion_mags=[(-2.5 * np.log10(host_star_counts / 2) + zero_point),
                        (-2.5 * np.log10(host_star_counts / 3) + zero_point)],
        zero_point=zero_point, ct_cal=ct_cal, use_ct_cal=True, cor_dataset=dataset_ct,
        FpamFsamCal=FpamFsamCal, blur_sigma=0.5, noise_std=1e-8, outdir=out_dir, platescale=0.0218
    )

    coron_data = mocks.generate_coron_dataset_with_companions(
        n_frames=1, shape=(200, 200), companion_xy=[(120, 80), (90, 130)], companion_counts=[host_star_counts / 2, host_star_counts / 3],
        host_star_counts=host_star_counts, roll_angles=[10.0], add_noise=True, noise_std=1e-8, outdir=out_dir
    )

    return direct_star_image, host_star_counts, fluxcal_factor, zero_point, dataset_ct, ct_cal, FpamFsamCal, psf_sub_frame, coron_data


def test_measure_companions_wcs(out_dir):
    """
    Run the full test case for measuring companions.
    """
    direct_star_image, host_star_counts, fluxcal_factor, zero_point, dataset_ct, ct_cal, FpamFsamCal, \
        psf_sub_frame, coron_data = generate_test_data(out_dir)

    # Run Measurement
    result_table = measure_companions.measure_companions(
        psf_sub_image=psf_sub_frame, coronagraphic_dataset=coron_data,
        phot_method='aperture',
        ct_cal=ct_cal, ct_dataset=dataset_ct, FpamFsamCal=FpamFsamCal,
        fluxcal_factor=fluxcal_factor, host_star_counts=host_star_counts,
        forward_model=False, direct_star_image= direct_star_image,
        reference_psf=dataset_ct[0], output_dir = out_dir, verbose=True
    )

    print("\nResult Table:\n", result_table)
    assert len(result_table) == 2, "Expected 2 measured companions"


if __name__ == "__main__":
    out_dir = os.path.join('corgidrp', 'data', 'L4TestInput')
    test_measure_companions_wcs(out_dir)