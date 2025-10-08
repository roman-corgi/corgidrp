import os
import numpy as np
import corgidrp.fluxcal as fluxcal
from corgidrp.data import Image, Dataset, NDFilterSweetSpotDataset
import corgidrp.mocks as mocks
import corgidrp.l4_to_tda as l4_to_tda
import corgidrp.l3_to_l4 as l3_to_l4
from corgidrp.l2b_to_l3 import crop
import corgidrp.measure_companions as measure_companions
import corgidrp.nd_filter_calibration as nd_filter_calibration
from corgidrp.astrom import seppa2dxdy
import pickle
from astropy.io.fits import Header
from pyklip.fakes import gaussfit2d, inject_planet

# Global Constants
HOST_STAR = 'TYC 4433-1800-1'
CFAM = '1F'
FWHM = 3
INPUT_EFFICIENCY_FACTOR = 1
PHOT_METHOD = "aperture"
FLUX_OR_IRR = 'irr'
NUM_IMAGES = 10
ROLL_ANGLES = np.zeros(NUM_IMAGES)
ROLL_ANGLES[NUM_IMAGES//2:] = 45
NUMBASIS = [1, 4, 8]
FULL_SIZE_IMAGE = (1024, 1024)
CROPPED_IMAGE_SIZE = (200, 200) # Practically this will be an odd shape so we should test that
PLOT_RESULTS = False
LOAD_FROM_DISK = False  # Flag to control whether to load mocks from disk (if available)
KL_MODE = -1            # Use the last KL_MODE
VERBOSE = True

# Define a list of companions.
# Each dictionary defines the companion's sep (in pixels), position angle (degrees counter-
# clockwise from north), and a scaling factor on the host star counts. Make sep between IWA 
# (~6.5 pix) and OWA (~20.6 pix)
COMPANION_PARAMS = [
    {"sep_pix": 13, "pa": 45, "counts_scale": 1e-3},
    {"sep_pix": 18, "pa": 120, "counts_scale": 8e-4}  
]

# Use a relative path for OUT_DIR
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_data", "L4_to_TDA_Inputs")
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
    # 1) Generate reference star dataset.
    host_star_flux = nd_filter_calibration.compute_expected_band_irradiance(HOST_STAR, CFAM)
    host_star_image = mocks.create_flux_image( host_star_flux, FWHM, 1e-16,  
        # dimensionless color correction factor
        filter='1F', 
        fpamname='HOLE',
        target_name=HOST_STAR
    )
    # bunit needs to be photoelectron/s for later tests, so set that now
    host_star_image.ext_hdr['BUNIT'] = 'photoelectron/s'

    # 2) Measure host star counts and determine zero point.
    host_star_counts, _, _, _ = gaussfit2d(host_star_image.data, host_star_image.data.shape[1]//2, host_star_image.data.shape[0]//2, searchrad=5, guessfwhm=3, 
                                                        guesspeak=1, refinefit=True) 
    # host_star_counts, _, _ = fluxcal.aper_phot(ref_star_dataset[0], **PHOT_KWARGS_COMMON)
    host_star_mag = l4_to_tda.determine_app_mag(host_star_image, HOST_STAR)
    fluxcal_factor = get_fluxcal_factor(host_star_image, PHOT_METHOD, PHOT_ARGS, FLUX_OR_IRR)
    if VERBOSE == True:
        print("Host star unocculted counts: ", host_star_counts)
        i = 0
        for companion in COMPANION_PARAMS:
            print("Companion ", i, " unocculted counts: ", host_star_counts * companion["counts_scale"])
            i+=1

    # 3) Attenuate host star frame by ND
    nd_x, nd_y = np.meshgrid(np.linspace(300, 700, 5), np.linspace(300, 700, 5))
    nd_x = nd_x.ravel()
    nd_y = nd_y.ravel()
    nd_od = np.ones(nd_y.shape) * 1e-2
    pri_hdr, ext_hdr, errhdr, dqhdr, biashdr = mocks.create_default_L2b_headers()
    nd_cal = NDFilterSweetSpotDataset(np.array([nd_od, nd_x, nd_y]).T, pri_hdr=pri_hdr,
                                      ext_hdr=ext_hdr)
    host_star_image.data *= nd_cal.interpolate_od(512, 512)

    # 4) Generate coronagraphic frames (just star, no companions) and RDI reference star dataset
    coron_data, ref_data = mocks.create_psfsub_dataset(NUM_IMAGES, NUM_IMAGES, np.append(ROLL_ANGLES, ROLL_ANGLES), 
                                                        data_shape = FULL_SIZE_IMAGE,
                                                        centerxy = host_star_center,
                                                        st_amp = host_star_counts * 0.01,
                                                        noise_amp = 1.,
                                                        fwhm_pix = 2.5,
                                                        ref_psf_spread=1. ,
                                                        pl_contrast=0,
                                                        )

    # 5) Generate core throughput calibration dataset.
    # assume 50 mas PSF
    ct_cal = mocks.create_ct_cal(50, cfam_name='1F', cenx=CROPPED_IMAGE_SIZE[0]/2, ceny=CROPPED_IMAGE_SIZE[1]/2, nx=41, ny=41)
    ct_cal_full_frame = mocks.create_ct_cal(50, cfam_name='1F', cenx=FULL_SIZE_IMAGE[0]/2, ceny=FULL_SIZE_IMAGE[1]/2, nx=41, ny=41)
    FpamFsamCal = mocks.create_mock_fpamfsam_cal(save_file=False)

    # get the index/ image of the PSF with maximum core throughput for reference
    x, y, ct = ct_cal.ct_excam
    max_index = np.argmax(ct)
    ct_cal_counts_ref_mask_far, _, _, _ = gaussfit2d(ct_cal.data[int(max_index)], 10, 10, searchrad=5, guessfwhm=3, 
                                                        guesspeak=1, refinefit=True) 
    # ct_cal_counts_ref_mask_far, _, _ = fluxcal.aper_phot(ct_cal.data[int(max_index)], **PHOT_KWARGS_COMMON)
    if VERBOSE == True:
        print("Reference PSF with maximum core throughput counts: ", ct_cal_counts_ref_mask_far)

    # 6) Use CT to generate off-axis PSFs of the planets for injection
    companion_throughput_ratios = []
    companion_unscaled_psfs = []
    for i, comp in enumerate(COMPANION_PARAMS):
        xoffset = comp['sep_pix'] * np.cos(np.degrees(comp['pa'] + 90))
        yoffset = comp['sep_pix'] * np.sin(np.degrees(comp['pa'] + 90))
        interp_psfs, _, _ = ct_cal_full_frame.GetPSF(xoffset, yoffset, coron_data, FpamFsamCal)
        nearest_psf = interp_psfs[0]

        ct_cal_counts_ref_mask_close, _, _, _ = gaussfit2d(nearest_psf, 10, 10, searchrad=5, guessfwhm=3, 
                                                        guesspeak=1, refinefit=True) 
        # ct_cal_counts_ref_mask_close, _, _ = fluxcal.aper_phot(ct_cal.data[int(idx)], **PHOT_KWARGS_COMMON)
        if VERBOSE == True:
            print("Reference PSF nearest companion ", i, " location counts: ", ct_cal_counts_ref_mask_close)

        location_throughput_ratio = ct_cal_counts_ref_mask_close / ct_cal_counts_ref_mask_far
        companion_throughput_ratios.append(location_throughput_ratio)

        companion_unscaled_psfs.append(nearest_psf)

    # 7) Scale planet PSFs to the appropriate flux and inject into the data
    companion_psfs = []
    for i, comp in enumerate(COMPANION_PARAMS):
        unscaled_psf = companion_unscaled_psfs[i]
        star_psf_at_same_loc = unscaled_psf * host_star_counts / ct_cal_counts_ref_mask_far
        planet_psf = star_psf_at_same_loc * comp["counts_scale"]
        companion_psfs.append(planet_psf)

    
    # inject planets into coronagrpahic dataset
    rolls = np.array([frame.pri_hdr['ROLL'] for frame in coron_data])
    for i, comp in enumerate(COMPANION_PARAMS):
        planet_psf = companion_psfs[i]
        inject_planet(coron_data.all_data, [host_star_center for _ in coron_data], [planet_psf for _ in coron_data],
                      [None for _ in coron_data], comp['sep_pix'], 0, thetas=90 + comp['pa'] - rolls)


    # 8) Create the PSF-subtracted frame using the dataset with planets and the RDI reference star dtaset
    cand_locs = []
    for i, comp in enumerate(COMPANION_PARAMS):
        cand_locs.append((comp['sep_pix'], comp['pa']))
    
    klip_kwargs={'numbasis' : NUMBASIS,
                     'mode' : 'RDI'}
        
    cropped_dataset = crop(coron_data,sizexy=CROPPED_IMAGE_SIZE)
    cropped_ref_data = crop(ref_data,sizexy=CROPPED_IMAGE_SIZE)
    
    # RDI
    psf_sub_dataset = l3_to_l4.do_psf_subtraction(
        cropped_dataset, 
        reference_star_dataset=cropped_ref_data,
        ct_calibration=ct_cal,
        #do_crop=True, crop_sizexy=CROPPED_IMAGE_SIZE,
        cand_locs=cand_locs,
        num_processes=1,
        kt_seps=[9,17],
        kt_pas=[150, 210, 270, 330],
        **klip_kwargs
    )


    output_filename = "final_psf_sub_image.fits"
    psf_sub_dataset[0].save(filedir=out_dir, filename=output_filename)

    return (host_star_image, host_star_counts, fluxcal_factor, host_star_mag, ct_cal, FpamFsamCal, nd_cal, psf_sub_dataset[0], coron_data, ref_data)


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
            ct_cal = pickle.load(f)
        with open(os.path.join(out_dir, "fluxcal_data.pkl"), "rb") as f:
            host_star_counts, fluxcal_factor, zero_point = pickle.load(f)
        with open(os.path.join(out_dir, "host_star_dataset.pkl"), "rb") as f:
            host_star_dataset = pickle.load(f)
        with open(os.path.join(out_dir, "FpamFsamCal.pkl"), "rb") as f:
            FpamFsamCal = pickle.load(f)
        with open(os.path.join(out_dir, "nd_cal.pkl"), "rb") as f:
            nd_cal = pickle.load(f)
        with open(os.path.join(out_dir, "coron_data.pkl"), "rb") as f:
            coron_data = pickle.load(f)
        with open(os.path.join(out_dir, "ref_data.pkl"), "rb") as f:
            ref_data = pickle.load(f)
        return (host_star_dataset, host_star_counts, fluxcal_factor, zero_point,
                ct_cal, FpamFsamCal, nd_cal, final_psf_sub_image, coron_data, ref_data)
    else:
        print("Generating mocks...")
        data = generate_test_data(out_dir)
        with open(os.path.join(out_dir, "ct_data.pkl"), "wb") as f:
            pickle.dump((data[4]), f)
        with open(os.path.join(out_dir, "fluxcal_data.pkl"), "wb") as f:
            pickle.dump((data[1], data[2], data[3]), f)
        with open(os.path.join(out_dir, "host_star_dataset.pkl"), "wb") as f:
            pickle.dump(data[0], f)
        with open(os.path.join(out_dir, "FpamFsamCal.pkl"), "wb") as f:
            pickle.dump(data[5], f)
        with open(os.path.join(out_dir, "nd_cal.pkl"), "wb") as f:
            pickle.dump(data[6], f)
        with open(os.path.join(out_dir, "coron_data.pkl"), "wb") as f:
            pickle.dump(data[8], f)
        with open(os.path.join(out_dir, "ref_data.pkl"), "wb") as f:
            pickle.dump(data[9], f)
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
        forward_model_flag (str): flag indicating which forward modeling technique to use
        
    """
    (host_star_image, host_star_counts, fluxcal_factor, host_star_mag, ct_cal, FpamFsamCal, nd_cal,
     psf_sub_image, coron_data, ref_data) = generate_or_load_test_data(OUT_DIR, load_from_disk=LOAD_FROM_DISK)
    
    print(f"Host Star Magnitude: {host_star_mag[0].ext_hdr['APP_MAG']}")

    cand_locs = []
    for i, comp in enumerate(COMPANION_PARAMS):
        cand_locs.append((comp['sep_pix'], comp['pa']))
        
    result_table = measure_companions.measure_companions(
        host_star_image, psf_sub_image,
        ct_cal=ct_cal, fpam_fsam_cal=FpamFsamCal,
        nd_cal=nd_cal,
        phot_method=PHOT_METHOD,
        photometry_kwargs=PHOT_ARGS,
        fluxcal_factor=fluxcal_factor,
        thrp_corr=forward_model_flag,
        verbose=VERBOSE,
        kl_mode_idx = KL_MODE,
        cand_locs = cand_locs
    )
    
    # Expect the number of detected companions to equal the number injected.
    expected_n = len(COMPANION_PARAMS)
    assert len(result_table) == expected_n, f"Expected {expected_n} companions, but found {len(result_table)}."
    
    # Determine the expected companion positions in the cropped image.
    # The cropped star center is at (CROPPED_IMAGE_SIZE[0]//2, CROPPED_IMAGE_SIZE[1]//2)
    star_loc_cropped = (CROPPED_IMAGE_SIZE[0] // 2, CROPPED_IMAGE_SIZE[1] // 2)
    apmag_data = l4_to_tda.determine_app_mag(host_star_image, host_star_image.pri_hdr['TARGET'])
    host_star_apmag = float(apmag_data[0].ext_hdr['APP_MAG'])
    
    for i, comp in enumerate(COMPANION_PARAMS):
        dx, dy = seppa2dxdy(comp["sep_pix"], comp["pa"])
        expected_x = star_loc_cropped[0] + dx
        expected_y = star_loc_cropped[1] + dy
        measured_x = result_table['x'][i]
        measured_y = result_table['y'][i]
        assert abs(measured_x - expected_x) < 5, f"Companion {i} x-coordinate off: expected {expected_x}, got {measured_x}"
        assert abs(measured_y - expected_y) < 5, f"Companion {i} y-coordinate off: expected {expected_y}, got {measured_y}"
        
        # Calculate expected companion magnitude.
        expected_comp_mag = host_star_apmag - 2.5 * np.log10(comp["counts_scale"])
        measured_mag = result_table['companion estimated mag'][i]
        print(f"Companion {i} Magnitude: {measured_mag}")
        assert abs(measured_mag - expected_comp_mag) < 0.15, f"Companion {i} magnitude off: expected {expected_comp_mag}, got {measured_mag}"
    
    print(result_table)


def test_measure_companions_using_L4_klipthroughput():
    """
    Test measure_companions using forward modeling.
    """
    _common_measure_companions_test(forward_model_flag="L4")


def test_measure_companions_non_forward_modeling():
    """
    Test measure_companions using the simplified (non-forward modeling) approach.
    """
    _common_measure_companions_test(forward_model_flag="None")


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



if __name__ == "__main__":
    # Run tests when executing the file directly.
    print("Running test: non-forward modeling")
    test_measure_companions_non_forward_modeling()
    print("Non-forward modeling test passed.")

    print("Running test: using saved L4 KLIP throughput")
    test_measure_companions_using_L4_klipthroughput()
    print("L4 KLIP throughput test passed.")
    
    print("Running companion location test.")
    test_update_companion_location()
    print("Companion location update test passed.")

    print("All tests passed successfully.")
