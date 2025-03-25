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
from astropy.io.fits import Header

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
PLOT_RESULTS = True
LOAD_FROM_DISK = False  # Flag to control whether to load mocks from disk (if available)

# Define a list of companions.
# Each dictionary defines the companion's sep (in pixels), position angle (degrees counter-
# clockwise from north), and a scaling factor on the host star counts. Make sep between IWA 
# (~6.5 pix) and OWA (~20.6 pix)
COMPANION_PARAMS = [
    {"sep_pix": 40, "pa": 45, "counts_scale": 1/3},
    {"sep_pix": 50, "pa": 120, "counts_scale": 1/4}  
]

# Use a relative path for OUT_DIR
OUT_DIR = os.path.join("tests/test_data", "L4_to_TDA_Inputs")
os.makedirs(OUT_DIR, exist_ok=True)  # Ensure the folder exists

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
    Compute flux calibration factor based on the specified calibration method.
    
    Args:
        image (object): Input image data to be calibrated.
        method (str): Calibration method to apply. Use "aperture" for aperture photometry or any other value for 2D Gaussian calibration.
        phot_args (dict): Additional keyword arguments for the photometry functions.
        flux_or_irr (str): Indicates whether to calibrate flux or irradiance.
    
    Returns:
        float: The computed flux calibration factor.
    """
    if method == "aperture":
        return fluxcal.calibrate_fluxcal_aper(image, flux_or_irr=flux_or_irr, phot_kwargs=phot_args)
    else:
        return fluxcal.calibrate_fluxcal_gauss2d(image, flux_or_irr=flux_or_irr, phot_kwargs=phot_args)


def generate_test_data(out_dir):
    """
    Generate mock data including direct star images, coronagraphic frames with multiple injected companions,
    and a PSF-subtracted frame.
    
    This function does the following:
      1) Creates a core throughput calibration dataset and associated calibration file.
      2) Generates a reference star dataset with flux calibration.
      3) Measures host star counts and computes the flux calibration factor and apparent magnitude.
      4) Creates coronagraphic frames with multiple companions using the host star counts and 
         companion parameters (separation, position angle, and count scaling).
      5) Performs PSF subtraction on the coronagraphic dataset to create a final PSF-subtracted image,
         and updates companion locations after cropping.
    
    Args:
        out_dir (str): Directory path where the generated test data files will be saved. 
                       The directory will be created if it does not already exist.
    
    Returns:
        tuple: A tuple containing the following:
            ref_star_dataset (list): A list of reference star images with associated flux calibration.
            host_star_counts (float): Measured host star counts from the reference star dataset.
            fluxcal_factor (float): Computed flux calibration factor for the reference star image.
            host_star_mag (float): Determined apparent magnitude of the host star.
            dataset_ct (object): Core throughput calibration dataset (raw calibration images).
            ct_cal (object): Core throughput calibration data including throughput values and header info.
            FpamFsamCal (object): Calibration data for FPAM-FSAM.
            psf_sub_image (Image): Final PSF-subtracted image after processing and companion location updates.
            coron_data (object): Generated coronagraphic dataset containing frames with multiple injected companions.
    """
    os.makedirs(out_dir, exist_ok=True)

    host_star_center = tuple(x // 2 for x in FULL_SIZE_IMAGE)

    # 1) Generate core throughput calibration dataset.
    dataset_ct, ct_cal, dataset_ct_nomask = mocks.create_mock_ct_dataset_and_cal_file(
        fwhm=50, n_psfs=20, cfam_name=CFAM, save_cal_file=True, image_shape=FULL_SIZE_IMAGE,
        total_counts=100
    )
    
    # get the index/ image of the PSF with maximum core throughput for reference
    x, y, ct = ct_cal.ct_excam
    max_index = np.argmax(ct)
    ct_cal_counts_ref_mask_far, _, _ = fluxcal.aper_phot(dataset_ct[int(max_index)], **PHOT_KWARGS_COMMON)

    companion_throughput_ratios = []
    for i, comp in enumerate(COMPANION_PARAMS):
        separation, idx, throughput = measure_companions.lookup_core_throughput(ct_cal, comp["sep_pix"])
        ct_cal_counts_ref_mask_close, _, _ = fluxcal.aper_phot(dataset_ct[int(idx)], **PHOT_KWARGS_COMMON)

        location_throughput_ratio = ct_cal_counts_ref_mask_close / ct_cal_counts_ref_mask_far
        companion_throughput_ratios.append(location_throughput_ratio)

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
        pltscale_mas= 21.8,
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

    # 4) Generate coronagraphic frames with multiple companions.
    # For multiple companions, pass lists for sep, PA and counts.
    coron_data = mocks.generate_coron_dataset_with_companions(
        n_frames=NUM_IMAGES,
        shape=FULL_SIZE_IMAGE,
        host_star_center=host_star_center,
        host_star_counts=host_star_counts,
        roll_angles=ROLL_ANGLES,
        companion_sep_pix=[cp["sep_pix"] for cp in COMPANION_PARAMS],
        companion_pa_deg=[cp["pa"] for cp in COMPANION_PARAMS],
        companion_counts=[host_star_counts * cp["counts_scale"] for cp in COMPANION_PARAMS],
        filter='1F',
        pltscale_as=0.0218,
        add_noise=True,
        noise_std=1.0e-2,
        outdir=out_dir,
        darkhole_file=dataset_ct[0],
        apply_coron_mask=True,
        coron_mask_radius=20,
        throughput_factors=companion_throughput_ratios
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
    
    # Update companion locations after cropping.
    comp_keywords = [key for key in psf_sub_image.ext_hdr if key.startswith("SNYX")]

    # Loop through each companion key and update its location.
    for key in comp_keywords:
        psf_sub_image = measure_companions.update_companion_location_in_cropped_image(
            psf_sub_image, key,
            tuple(x // 2 for x in FULL_SIZE_IMAGE),
            tuple(x // 2 for x in CROPPED_IMAGE_SIZE)
        )

    output_filename = "final_psf_sub_image.fits"
    psf_sub_image.save(filedir=out_dir, filename=output_filename)

    return (ref_star_dataset, host_star_counts, fluxcal_factor, host_star_mag,
            dataset_ct, ct_cal, FpamFsamCal, psf_sub_image, coron_data)


def generate_or_load_test_data(out_dir, load_from_disk=False):
    """
    Generate or load mock datasets for testing.
    
    This function checks whether previously generated mock datasets exist on disk 
    in the specified output directory. If load_from_disk is True and the final PSF-subtracted 
    image file is found, it loads all necessary datasets (including core throughput calibration data, 
    flux calibration data, reference star dataset, FPAM-FSAM calibration data, and coronagraphic data) 
    from disk. Otherwise, it generates new mock datasets using generate_test_data, saves them to disk, 
    and returns the generated data.
    
    Args:
        out_dir (str): Directory path where the test data files are stored or will be generated.
        load_from_disk (bool): Flag indicating whether to load existing mock data from disk 
                               (if True) or generate new mock data (if False).
    
    Returns:
        tuple: A tuple containing the following elements:
            ref_star_dataset (list): A list of reference star images with associated flux calibration.
            host_star_counts (float): Measured host star counts from the reference star dataset.
            fluxcal_factor (float): Computed flux calibration factor for the reference star image.
            zero_point (float): Determined zero point for the host star photometry.
            dataset_ct (object): Core throughput calibration dataset (raw calibration images).
            ct_cal (object): Core throughput calibration data including throughput values and header info.
            FpamFsamCal (object): Calibration data for FPAM-FSAM.
            final_psf_sub_image (Image): Final PSF-subtracted image generated or loaded from disk.
            coron_data (object): Generated coronagraphic dataset containing frames with injected companions.
    """

    final_psf_file = os.path.join(out_dir, "final_psf_sub_image.fits")
    if load_from_disk and os.path.exists(final_psf_file):
        print("Loading mocks from disk...")
        final_psf_sub_image = Image(data_or_filepath=final_psf_file)
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
        return data
    

def _common_measure_companions_test(forward_model_flag):
    """
    Helper function to test the `measure_companions` function using the specified forward modeling flag.
    
    This function performs the following steps:
      1. Loads or generates mock test datasets, including reference star images, flux calibration data,
         core throughput calibration data, a PSF-subtracted image, and coronagraphic frames.
      2. Prints the host star magnitude for diagnostic purposes.
      3. Calls the `measure_companions` function to detect companions in the coronagraphic data.
      4. Verifies that the number of detected companions matches the number of injected companions.
      5. For each detected companion, calculates the expected position in the cropped image based on 
         the known companion separation and position angle, and asserts that the measured position 
         is within an acceptable tolerance.
      6. Computes the expected companion magnitude from the host star magnitude and companion flux scaling, 
         and asserts that the measured magnitude is within a defined tolerance.
      7. Prints the resulting table of measured companion properties.
    
    Args:
        forward_model_flag (bool): Flag indicating whether to use forward modeling in the companion measurement.
        
    """
    (ref_star_dataset, host_star_counts, fluxcal_factor, host_star_mag, dataset_ct, ct_cal, FpamFsamCal, 
     psf_sub_image, coron_data) = generate_or_load_test_data(OUT_DIR, load_from_disk=LOAD_FROM_DISK)
    
    print(f"Host Star Magnitude: {host_star_mag[0].ext_hdr['APP_MAG']}")

    result_table = measure_companions.measure_companions(
        coron_data, ref_star_dataset, psf_sub_image,
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
        verbose=False,
        plot_results=PLOT_RESULTS
    )
    
    # Expect the number of detected companions to equal the number injected.
    expected_n = len(COMPANION_PARAMS)
    assert len(result_table) == expected_n, f"Expected {expected_n} companions, but found {len(result_table)}."
    
    # Determine the expected companion positions in the cropped image.
    # The cropped star center is at (CROPPED_IMAGE_SIZE[0]//2, CROPPED_IMAGE_SIZE[1]//2)
    star_loc_cropped = (CROPPED_IMAGE_SIZE[0] // 2, CROPPED_IMAGE_SIZE[1] // 2)
    apmag_data = l4_to_tda.determine_app_mag(ref_star_dataset[0], ref_star_dataset[0].pri_hdr['TARGET'])
    host_star_apmag = float(apmag_data[0].ext_hdr['APP_MAG'])
    
    for i, comp in enumerate(COMPANION_PARAMS):
        dx, dy = seppa2dxdy(comp["sep_pix"], comp["pa"])
        expected_x = star_loc_cropped[0] + dx
        expected_y = star_loc_cropped[1] + dy
        measured_x = result_table['x'][i]
        measured_y = result_table['y'][i]
        print("locs", expected_x, expected_y, measured_x, measured_y)
        assert abs(measured_x - expected_x) < 5, f"Companion {i} x-coordinate off: expected {expected_x}, got {measured_x}"
        assert abs(measured_y - expected_y) < 5, f"Companion {i} y-coordinate off: expected {expected_y}, got {measured_y}"
        
        # Calculate expected companion magnitude.
        expected_comp_mag = host_star_apmag - 2.5 * np.log10(comp["counts_scale"])
        measured_mag = result_table['companion estimated mag'][i]
        print(f"Companion {i} Magnitude: {measured_mag}")
        assert abs(measured_mag - expected_comp_mag) < 0.15, f"Companion {i} magnitude off: expected {expected_comp_mag}, got {measured_mag}"
    
    print(result_table)


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


def test_update_companion_location():
    """
    Test that the companion location update function correctly adjusts the coordinates
    after cropping.
    
    The header is assumed to have the format 'SNR,y,x' under the key 'SNYX001'.
    With a full image of size (1024,1024), the full image center is (512,512),
    and for a cropped image of size (200,200), assume its host star is at (100,100).
    
    For an original header value of '5.0,522,502' (signal, y, x):
      new_y = 522 - 512 + 100 = 110,
      new_x = 502 - 512 + 100 = 90,
    expected header becomes "5.0,110,90" (ignoring any formatting spaces).
    """
    # Create a dummy header using astropy.io.fits.Header
    dummy_hdr = Header()
    dummy_hdr['SNYX001'] = "5.0,522,502"  # signal, y, x
    
    dummy_data = np.zeros(FULL_SIZE_IMAGE)
    img = Image(data_or_filepath=dummy_data, pri_hdr={}, ext_hdr=dummy_hdr.copy())
    
    # Define host centers:
    old_host = (512, 512)   # full image center
    new_host = (100, 100)   # new host position in the cropped image
    
    # Update the companion location.
    updated_img = measure_companions.update_companion_location_in_cropped_image(
        img, "SNYX001", old_host, new_host
    )
    
    new_val = updated_img.ext_hdr.get("SNYX001")
    # Parse the returned string to compare values.
    parts = new_val.split(',')
    sn_val = float(parts[0])
    new_y = int(parts[1])
    new_x = int(parts[2])
    
    assert abs(sn_val - 5.0) < 1e-3, f"Expected SNR 5.0, got {sn_val}"
    assert new_y == 110, f"Expected new companion y-coordinate 110, got {new_y}"
    assert new_x == 90, f"Expected new companion x-coordinate 90, got {new_x}"


def test_robustness_high_noise():
    """
    Test the companion measurement pipeline under high noise conditions to check robustness.
    """
    # Generate a minimal core throughput calibration dataset.
    dataset_ct, ct_cal, _ = mocks.create_mock_ct_dataset_and_cal_file(
        fwhm=50, n_psfs=20, cfam_name=CFAM, save_cal_file=False, image_shape=FULL_SIZE_IMAGE,
        total_counts=100
    )
    x, y, ct = ct_cal.ct_excam
    max_index = np.argmax(ct)
    ct_cal_counts_ref_mask_far, _, _ = fluxcal.aper_phot(dataset_ct[int(max_index)], **PHOT_KWARGS_COMMON)
    
    companion_throughput_ratios = []
    for i, comp in enumerate(COMPANION_PARAMS):
        separation, idx, throughput = measure_companions.lookup_core_throughput(ct_cal, comp["sep_pix"])
        ct_cal_counts_ref_mask_close, _, _ = fluxcal.aper_phot(dataset_ct[int(idx)], **PHOT_KWARGS_COMMON)
        ratio = ct_cal_counts_ref_mask_close / ct_cal_counts_ref_mask_far
        companion_throughput_ratios.append(ratio)
        
    host_star_center = (FULL_SIZE_IMAGE[0] // 2, FULL_SIZE_IMAGE[1] // 2)
    host_star_counts = 1e5  # Use a fixed value for testing.
    
    # Generate coronagraphic data with elevated noise.
    high_noise_std = 0.1
    coron_data = mocks.generate_coron_dataset_with_companions(
        n_frames=NUM_IMAGES,
        shape=FULL_SIZE_IMAGE,
        host_star_center=host_star_center,
        host_star_counts=host_star_counts,
        roll_angles=ROLL_ANGLES,
        companion_sep_pix=[cp["sep_pix"] for cp in COMPANION_PARAMS],
        companion_pa_deg=[cp["pa"] for cp in COMPANION_PARAMS],
        companion_counts=[host_star_counts * cp["counts_scale"] for cp in COMPANION_PARAMS],
        filter='1F',
        pltscale_as=0.0218,
        add_noise=True,
        noise_std=high_noise_std,
        outdir=OUT_DIR,
        darkhole_file=dataset_ct[0],
        apply_coron_mask=True,
        coron_mask_radius=20,
        throughput_factors=companion_throughput_ratios
    )
    
    # Generate a minimal reference star dataset.
    ref_star_dataset = mocks.generate_reference_star_dataset_with_flux(
        n_frames=NUM_IMAGES,
        roll_angles=ROLL_ANGLES,
        flux_erg_s_cm2=1e-15,
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
        pltscale_mas=21.8,
        background=0,
        add_gauss_noise=True,
        noise_scale=1.0,
        filedir=OUT_DIR,
        file_save=False,
        shape=CROPPED_IMAGE_SIZE
    )
    fluxcal_factor = get_fluxcal_factor(ref_star_dataset[0], PHOT_METHOD, PHOT_ARGS, FLUX_OR_IRR)
    
    # Use the first frame of coron_data as a proxy for the psf_sub_image.
    psf_sub_image = coron_data[0]
    
    result_table = measure_companions.measure_companions(
        coron_data, ref_star_dataset, psf_sub_image,
        ref_psf_min_mask_effect=Image(data_or_filepath=ct_cal.data[16],
                                      pri_hdr=ct_cal.pri_hdr,
                                      ext_hdr=ct_cal.ext_hdr),
        ct_cal=ct_cal, FpamFsamCal=mocks.create_mock_fpamfsam_cal(save_file=False),
        phot_method=PHOT_METHOD,
        photometry_kwargs=PHOT_ARGS,
        fluxcal_factor=fluxcal_factor,
        forward_model=False,
        numbasis=NUMBASIS,
        output_dir=OUT_DIR,
        verbose=False
    )
    
    expected_n = len(COMPANION_PARAMS)
    assert len(result_table) == expected_n, f"Expected {expected_n} companions under high noise, but found {len(result_table)}."


if __name__ == "__main__":
    # Run tests when executing the file directly.
    print("Running test: non-forward modeling")
    #test_measure_companions_non_forward_modeling()
    print("Non-forward modeling test passed.")

    print("Running test: forward modeling")
    test_measure_companions_forward_modeling()
    print("Forward modeling test passed.")
    
    print("Running companion location test.")
    test_update_companion_location()
    print("Companion location update test passed.")

    print("Running high noise robustness test.")
    test_robustness_high_noise()
    print("High noise robustness test passed.")
    
    print("All tests passed successfully.")
