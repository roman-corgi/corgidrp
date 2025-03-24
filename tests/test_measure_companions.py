import os
import numpy as np
from astropy.table import Table
import corgidrp.fluxcal as fluxcal
import corgidrp.klip_fm as klip_fm
from corgidrp.data import Image, Dataset
import corgidrp.mocks as mocks
import corgidrp.l4_to_tda as l4_to_tda
import corgidrp.l3_to_l4 as l3_to_l4
import corgidrp.measure_companions as measure_companions
import corgidrp.nd_filter_calibration as nd_filter_calibration
from corgidrp.astrom import seppa2dxdy
import tempfile
import pytest
import pickle

# Global Constants
HOST_STAR = 'TYC 4433-1800-1'
CFAM = '1F'
FWHM = 3
INPUT_EFFICIENCY_FACTOR = 1
PHOT_METHOD = "aperture"
FLUX_OR_IRR = 'irr'
NUM_IMAGES = 10
ROLL_ANGLES = np.linspace(0, 45, NUM_IMAGES)
NUMBASIS = [1, 4, 8]
FULL_SIZE_IMAGE = (1024, 1024)
CROPPED_IMAGE_SIZE = (200, 200)
COMP_REL_COUNTS_SCALE = 1 / 3
COMP_SEP_PIX = 28                   # Number of pixels companion is separated from star center
COMP_PA = 45                        # Companion degree position (counterclockwise from north-up)
# Use a relative path for OUT_DIR
OUT_DIR = os.path.join("tests/test_data", "L4_to_TDA_Inputs")
os.makedirs(OUT_DIR, exist_ok=True)  # Ensure the folder exists

# Flag to control whether to load mocks from disk (if available)
LOAD_FROM_DISK = True  # Set to True to load files rather than re-generating them

# Reusable photometry parameters for flux calibration measurements.
PHOT_KWARGS_COMMON = {
    "encircled_radius": 10,
    "frac_enc_energy": 1.0,
    "method": "subpixel",
    "subpixels": 5,
    "background_sub": True,
    "r_in": 12,
    "r_out": 20,
    "centering_method": "xy",
    "centroid_roi_radius": 10
}

# Photometry arguments for companion measurements.
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
    
    Args:
        out_dir (str): Output directory for saving data.
    
    Returns:
        ref_star_dataset, host_star_counts, fluxcal_factor, zero_point, 
        dataset_ct, ct_cal, FpamFsamCal, final_psf_sub_image, coron_data
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Generate core throughput calibration dataset.
    dataset_ct, ct_cal, dataset_ct_nomask = mocks.create_mock_ct_dataset_and_cal_file(
        fwhm=50, n_psfs=20, cfam_name=CFAM, save_cal_file=True, image_shape=FULL_SIZE_IMAGE,
        total_counts=100
    )
    ct_cal_counts_ref, _, _ = fluxcal.aper_phot(dataset_ct[3], **PHOT_KWARGS_COMMON)
    ct_cal_counts_ref_no_mask, _, _ = fluxcal.aper_phot(dataset_ct_nomask[3], **PHOT_KWARGS_COMMON)
    mask_throughput_ratio = ct_cal_counts_ref / ct_cal_counts_ref_no_mask

    ct_cal_counts_ref_mask_far, _, _ = fluxcal.aper_phot(dataset_ct[16], **PHOT_KWARGS_COMMON)
    ct_cal_counts_ref_mask_close, _, _ = fluxcal.aper_phot(dataset_ct[8], **PHOT_KWARGS_COMMON)
    location_throughput_ratio = ct_cal_counts_ref_mask_close / ct_cal_counts_ref_mask_far

    FpamFsamCal = mocks.create_mock_fpamfsam_cal(save_file=False)

    # 2) Generate reference star dataset.
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
        platescale=21.8,
        background=0,
        add_gauss_noise=True,
        noise_scale=1.0,
        filedir=out_dir,
        file_save=True,
        shape=CROPPED_IMAGE_SIZE
    )

    # 3) Measure host star counts and determine zero point.
    host_star_counts, _, _ = fluxcal.aper_phot(ref_star_dataset[0], **PHOT_KWARGS_COMMON)
    host_star_mag = l4_to_tda.determine_app_mag(ref_star_dataset[0], HOST_STAR)
    fluxcal_factor = get_fluxcal_factor(ref_star_dataset[0], PHOT_METHOD, PHOT_ARGS, FLUX_OR_IRR)
    zero_point = fluxcal_factor.ext_hdr.get('ZP', None)

    # 4) Generate coronagraphic frames.
    host_star_center = tuple(x // 2 for x in FULL_SIZE_IMAGE)
    coron_data = mocks.generate_coron_dataset_with_companions(
        n_frames=NUM_IMAGES,
        shape=FULL_SIZE_IMAGE,
        host_star_center=host_star_center,
        host_star_counts=host_star_counts,
        roll_angles=ROLL_ANGLES,
        companion_sep_pix=COMP_SEP_PIX,
        companion_pa_deg=COMP_PA,
        companion_counts=host_star_counts * COMP_REL_COUNTS_SCALE,
        filter='1F',
        platescale=0.0218,
        add_noise=True,
        noise_std=1.0e-2,
        outdir=out_dir,
        darkhole_file=dataset_ct[0],
        apply_coron_mask=True,
        coron_mask_radius=20,
        throughput_factor=location_throughput_ratio
    )

    # 5) Create a PSF-subtracted frame.
    psf_sub_dataset = l3_to_l4.do_psf_subtraction(
        coron_data, ct_calibration=ct_cal,
        numbasis=NUMBASIS,
        do_crop=True, crop_sizexy=CROPPED_IMAGE_SIZE
    )
    psf_sub_image = measure_companions.extract_single_frame(psf_sub_dataset[0], frame_index=0)
    psf_sub_image.data[psf_sub_image.data < 0] = 0
    psf_sub_image = Image(data_or_filepath=psf_sub_image.data,
                          pri_hdr=psf_sub_image.pri_hdr,
                          ext_hdr=psf_sub_image.ext_hdr)
    # Update companion location after cropping using fixed update.
    comp_keyword = next(key for key in psf_sub_image.ext_hdr if key.startswith("SNYX"))
    final_psf_sub_image = measure_companions.update_companion_location_in_cropped_image(
        psf_sub_image, comp_keyword,
        tuple(x // 2 for x in FULL_SIZE_IMAGE),
        tuple(x // 2 for x in CROPPED_IMAGE_SIZE)
    )
    output_filename = "final_psf_sub_image.fits"
    final_psf_sub_image.save(filedir=out_dir, filename=output_filename)

    return (ref_star_dataset, host_star_counts, fluxcal_factor, zero_point,
            dataset_ct, ct_cal, FpamFsamCal, final_psf_sub_image, coron_data)


def generate_or_load_test_data(out_dir, load_from_disk=False):
    """
    Generate or load mock data.
    
    If load_from_disk is True and the expected saved files exist in out_dir,
    then load the mocks from disk. Otherwise, generate the mocks and save them.
    """
    # Check for one of the key files that indicates mocks have been saved.
    final_psf_file = os.path.join(out_dir, "final_psf_sub_image.fits")
    if load_from_disk and os.path.exists(final_psf_file):
        print("Loading mocks from disk...")
        # Load final PSF-subtracted image using the Image class.
        final_psf_sub_image = Image(data_or_filepath=final_psf_file)
        # Load other objects from pickle files.
        with open(os.path.join(out_dir, "ct_data.pkl"), "rb") as f:
            dataset_ct, ct_cal = pickle.load(f)
        with open(os.path.join(out_dir, "fluxcal_data.pkl"), "rb") as f:
            host_star_counts, fluxcal_factor, zero_point = pickle.load(f)
        with open(os.path.join(out_dir, "ref_star_dataset.pkl"), "rb") as f:
            ref_star_dataset = pickle.load(f)
        with open(os.path.join(out_dir, "FpamFsamCal.pkl"), "rb") as f:
            FpamFsamCal = pickle.load(f)
        with open(os.path.join(out_dir, "coron_data.pkl"), "rb") as f:
            coron_data = pickle.load(f)
        return (ref_star_dataset, host_star_counts, fluxcal_factor, zero_point,
                dataset_ct, ct_cal, FpamFsamCal, final_psf_sub_image, coron_data)
    else:
        print("Generating mocks...")
        data = generate_test_data(out_dir)
        # Save key groups to disk for faster future loading.
        with open(os.path.join(out_dir, "ct_data.pkl"), "wb") as f:
            pickle.dump((data[4], data[5]), f)
        with open(os.path.join(out_dir, "fluxcal_data.pkl"), "wb") as f:
            pickle.dump((data[1], data[2], data[3]), f)
        with open(os.path.join(out_dir, "ref_star_dataset.pkl"), "wb") as f:
            pickle.dump(data[0], f)
        with open(os.path.join(out_dir, "FpamFsamCal.pkl"), "wb") as f:
            pickle.dump(data[6], f)
        with open(os.path.join(out_dir, "coron_data.pkl"), "wb") as f:
            pickle.dump(data[8], f)
        # final_psf_sub_image is already saved as a FITS file.
        return data


def _common_measure_companions_test(forward_model_flag):
    """
    Helper function to run measure_companions with the given forward_model flag,
    and check that exactly one companion is detected with expected coordinates and magnitude.
    """

    (ref_star_dataset, host_star_counts, fluxcal_factor, zero_point,
        dataset_ct, ct_cal, FpamFsamCal, psf_sub_frame, coron_data) = generate_or_load_test_data(OUT_DIR, load_from_disk=LOAD_FROM_DISK)
    
    result_table = measure_companions.measure_companions(
        coron_data, ref_star_dataset, psf_sub_frame,
        ref_psf_min_mask_effect=Image(data_or_filepath=ct_cal.data[16],
                                        pri_hdr=ct_cal.pri_hdr,
                                        ext_hdr=ct_cal.ext_hdr),
        ct_cal=ct_cal, FpamFsamCal=FpamFsamCal,
        phot_method=PHOT_METHOD,
        photometry_kwargs=PHOT_ARGS,
        fluxcal_factor=fluxcal_factor,
        forward_model=forward_model_flag,
        numbasis=NUMBASIS,
        output_dir=OUT_DIR,
        verbose=False
    )
    
    # Assert exactly one companion is detected.
    assert len(result_table) == 1, "Expected exactly one companion in the results."
    
    # Calculate expected companion coordinates.   
    # TO DO: don't hard code the 100, use starloc or maskloc if available
    dx, dy = seppa2dxdy(COMP_SEP_PIX, COMP_PA)
    expected_x = 100 + dx
    expected_y = 100 + dy

    print(result_table)
    
    measured_x = result_table['x'][0]
    measured_y = result_table['y'][0]
    assert abs(measured_x - expected_x) < 5, f"Companion x-coordinate off: expected {expected_x}, got {measured_x}"
    assert abs(measured_y - expected_y) < 5, f"Companion y-coordinate off: expected {expected_y}, got {measured_y}"
    
    # Calculate expected companion magnitude.
    apmag_data = l4_to_tda.determine_app_mag(ref_star_dataset[0], ref_star_dataset[0].pri_hdr['TARGET'])
    host_star_apmag = float(apmag_data[0].ext_hdr['APP_MAG'])
    expected_comp_mag = host_star_apmag - 2.5 * np.log10(COMP_REL_COUNTS_SCALE)
    measured_mag = result_table['mag'][0]
    assert abs(measured_mag - expected_comp_mag) < 0.1, f"Companion magnitude off: expected {expected_comp_mag}, got {measured_mag}"


def test_measure_companions_forward_modeling():
    """
    Test measure_companions using forward modeling.
    """
    _common_measure_companions_test(forward_model_flag=True)


def test_measure_companions_non_forward_modeling():
    """
    Test measure_companions using the simplified (non-forward modeling) approach.
    """
    _common_measure_companions_test(forward_model_flag=False)


if __name__ == "__main__":
    # Run tests when executing the file directly.
    # Uncomment the test you want to run.
    print("Running test: non-forward modeling")
    test_measure_companions_non_forward_modeling()
    print("Non-forward modeling test passed.")

    print("Running test: forward modeling")
    test_measure_companions_forward_modeling()
    print("Forward modeling test passed.")
    
    print("All tests passed successfully.")
