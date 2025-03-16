import os
import sys
import numpy as np
import astropy.time as time
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter

from pyklip.instruments.utils.wcsgen import generate_wcs

import corgidrp
import corgidrp.mocks as mocks
import corgidrp.l2b_to_l3 as l2b_to_l3
import corgidrp.measure_companions as measure_companions
import corgidrp.fluxcal as fluxcal
import corgidrp.nd_filter_calibration as nd_filter_calibration
from corgidrp.data import Image, Dataset, FpamFsamCal, CoreThroughputCalibration
from corgidrp.mocks import create_default_L3_headers, create_ct_psfs
from corgidrp import corethroughput

INPUT_STARS = ['109 Vir']
FWHM = 3
CAL_FACTOR = 0.8
PHOT_METHOD = "Aperture"
FLUX_OR_IRR = 'irr'

if PHOT_METHOD == "Aperture":
    PHOT_ARGS = {
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
    }
elif PHOT_METHOD == "Gaussian":
    PHOT_ARGS = {
        "fwhm": 3,
        "fit_shape": None,
        "background_sub": True,
        "r_in": 5,
        "r_out": 10,
        "centering_method": 'xy',
        "centroid_roi_radius": 5,
        "centering_initial_guess": None
    }

#### CREATE MOCKS - move these to mocks.py most likely ####
def create_mock_psfsub_image_wcs(
        ny=200, nx=200,
        star_center=None,
        roll_angle=0.0,
        platescale=0.0218,
        injected_companions=None,
        add_noise=False,
        noise_std=0.01,
        background_level=0.0
    ):
    """
    Create a mock PSF-subtracted image with WCS, error, and DQ HDUs.

    Parameters
    ----------
    ny, nx : int
        Image dimensions.
    star_center : (y, x) or None
        Location of star center. Default = center of array.
    roll_angle : float
        Roll angle in degrees for the WCS.
    platescale : float
        Plate scale in arcsec/pixel. (21.8 mas/px = 0.0218 arcsec/px)
    injected_companions : list of dict or None
        List of dictionaries defining injected sources. 
        Example: [{'x':120, 'y':80, 'flux':0.2}, ...]
    add_noise : bool, optional
        Whether to add Gaussian noise to the image (default: True). Haven't really figured 
        out a good way to do this yet so i don't recommend using this option.
    noise_std : float, optional
        Standard deviation of the Gaussian noise (default: 0.01).

    Returns
    -------
    hdul : fits.HDUList
        [HDU0: Primary (no data),
         HDU1: Science data,
         HDU2: Error array,
         HDU3: Data-quality array]
    """
    # Default star center
    if star_center is None:
        star_center = (ny // 2, nx // 2)

    # 1) Build a mock science data array
    data = np.full((ny, nx), background_level, dtype=np.float32)

    # Create an artificial hole at the star center
    rr, cc = np.indices(data.shape)
    rr_off = rr - star_center[0]
    cc_off = cc - star_center[1]
    star_hole = np.exp(-(rr_off**2 + cc_off**2) / (2 * (5.0**2)))
    data -= 0.5 * star_hole.astype(np.float32)

    # Inject faint companions with Gaussian profiles
    if injected_companions:
        for comp in injected_companions:
            yy, xx = comp['y'], comp['x']
            flux = comp['flux']
            r2 = (rr - yy)**2 + (cc - xx)**2
            
            # Normalize Gaussian PSF so that total flux sums to desired value
            gaus = (flux / (2 * np.pi * (1.5**2))) * np.exp(-r2 / (2 * (1.5**2)))
            data += gaus.astype(np.float32)

    # Apply a slight blur to simulate realistic conditions
    data = gaussian_filter(data, sigma=0.5)

    # Add Gaussian noise if enabled
    if add_noise:
        noise = np.random.normal(0., noise_std, data.shape)
        data += noise.astype(np.float32)

    # 2) Create L4 headers
    prihdr, exthdr = mocks.create_default_L4_headers()
    exthdr.update({
        'NAXIS': 2,
        'NAXIS1': nx,
        'NAXIS2': ny,
        'DATALVL': 'L4',
        'BUNIT': 'ELECTRONS/S',
        'STARLOCX': star_center[1],
        'STARLOCY': star_center[0]
    })

    # 3) Generate WCS from pyKLIP
    px_center = [star_center[1], star_center[0]]  # (x, y)
    wcs_obj = generate_wcs(roll_angle, px_center, platescale=platescale)
    wcs_header = wcs_obj.to_header()

    # Mapping of WCS header keys to exthdr keys
    wcs_mapping = {
        "WCSAXES": "WCSAXES",
        "CRPIX1": "CRPIX1",
        "CRPIX2": "CRPIX2",
        "PC1_1": "CD1_1",  # Rename from PC1_1 to CD1_1?
        "PC1_2": "CD1_2",  # Rename from PC1_2 to CD1_2?
        "PC2_1": "CD2_1",  # Rename from PC2_1 to CD2_1?
        "PC2_2": "CD2_2",  # Rename from PC2_2 to CD2_2?
        "CDELT1": "CDELT1",
        "CDELT2": "CDELT2",
        "CUNIT1": "CUNIT1",
        "CUNIT2": "CUNIT2",
        "CTYPE1": "CTYPE1",
        "CTYPE2": "CTYPE2",
        "CRVAL1": "CRVAL1",
        "CRVAL2": "CRVAL2",
        "LONPOLE": "LONPOLE",
        "LATPOLE": "LATPOLE",
        "MJDREF": "MJDREF",
        "RADESYS": "RADESYS"
    }

    # Copy only existing keys from wcs_header to exthdr, applying renaming where needed
    exthdr.update({exthdr_key: wcs_header[wcs_key] for wcs_key, exthdr_key in 
                   wcs_mapping.items() if wcs_key in wcs_header})

    # Add custom platescale value
    exthdr["PLTSCALE"] = platescale

    # 4) Insert detected companion metadata
    if injected_companions:
        for i, dcomp in enumerate(injected_companions, start=1):
            snr = dcomp.get('snr', 5.0)
            exthdr[f'SNYX{i:03d}'] = f"{snr:.1f},{dcomp['x']},{dcomp['y']}"

    # 5) Create placeholder error & DQ arrays
    err_data = np.full_like(data, noise_std if add_noise else 0.01, dtype=np.float32)
    dq_data  = np.zeros_like(data, dtype=np.uint16)

    # Construct FITS HDU list
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=prihdr),
        fits.ImageHDU(data=data, header=exthdr, name="SCI"),
        fits.ImageHDU(data=err_data, header=fits.Header({'EXTNAME': 'ERR'})),
        fits.ImageHDU(data=dq_data, header=fits.Header({'EXTNAME': 'DQ'}))
    ])

    return hdul


def mock_flux_image(exptime, filter_used, cal_factor, save_mocks, output_path=None, 
                           background_val=0, add_gauss_noise_val=False):
    """
    Generate and save mock dataset files for specified exposure time and filter.

    Parameters:
        dim_exptime (float): Exposure time for the simulated images.
        filter_used (str): Filter used for the observations.
        cal_factor (float): Calibration factor applied to the images.
        save_mocks (bool): Whether to save the generated mock images.
        output_path (str, optional): Directory path to save the images. Defaults to the current working directory.
        background_val (int, optional): Background value to be added to the images. Defaults to 0.
        add_gauss_noise_val (bool, optional): Whether to add Gaussian noise to the images. Defaults to False.

    Returns:
        list: A list of generated flux images for the dim stars.
    """
    if save_mocks:
        output_path = output_path or os.getcwd()
    else:
        output_path = output_path or os.getcwd()
    os.makedirs(output_path, exist_ok=True)
    flux_star_images = []
    for star_name in INPUT_STARS:
        star_flux = nd_filter_calibration.compute_expected_band_irradiance(star_name, filter_used)
        flux_image = mocks.create_flux_image(
            star_flux, FWHM, cal_factor, filter_used, "HOLE", star_name,
            fsm_x=0, fsm_y=0, exptime=exptime, filedir=output_path,
            color_cor=1.0, platescale=21.8,
            background=background_val,
            add_gauss_noise=add_gauss_noise_val,
            noise_scale=1.0, file_save=True
        )
        flux_star_images.append(flux_image)
    return flux_star_images


def create_mock_ct_dataset_and_cal_file(
    fwhm=50,
    n_psfs=100,
    cfam_name='1F',
    pupil_value_1=1,
    pupil_value_2=3,
    seed=None,
    save_cal_file=False,
    cal_filename=None
):
    """
    Create a mock dataset suitable for generating a Core Throughput calibration file,
    then generate and return that calibration file in-memory.

    Parameters
    ----------
    fwhm : float, optional
        The FWHM (in mas) for the mock off-axis PSFs (used by create_ct_psfs).
    n_psfs : int, optional
        Number of off-axis PSFs to generate.
    cfam_name : str, optional
        CFAM filter name to store in the header.
    pupil_value_1 : float, optional
        A value to fill in the first pupil image (used to simulate unocculted frames).
    pupil_value_2 : float, optional
        A value to fill in the second pupil image.
    seed : int, optional
        Random seed for reproducibility (used if create_ct_psfs has random offsets).
    save_cal_file : bool, optional
        Whether to save the generated calibration file to disk.
    cal_filename : str, optional
        Filename to use if saving the calibration file. If None, a default is generated.

    Returns
    -------
    dataset_ct : corgidrp.data.Dataset
        The constructed dataset containing pupil frames + off-axis PSFs.
    ct_cal : corgidrp.data.CoreThroughputCalibration
        The generated core throughput calibration object (in-memory).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # ----------------------------
    # A) Create the base headers
    # ----------------------------
    prhd, exthd = create_default_L3_headers()
    exthd['DRPCTIME'] = time.Time.now().isot
    exthd['DRPVERSN'] = corgidrp.__version__
    exthd['CFAMNAME'] = cfam_name

    # For example, choose some FPAM/FSAM positions during CT observations
    # (just arbitrary or from real test data)
    exthd['FPAM_H'] = 6854
    exthd['FPAM_V'] = 22524
    exthd['FSAM_H'] = 29471
    exthd['FSAM_V'] = 12120

    # Make a pupil header so we can mark these frames as unocculted
    exthd_pupil = exthd.copy()
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'

    # ----------------------------
    # B) Create the unocculted/pupil frames
    # ----------------------------
    # So 1024x1024 arrays with uniform “patches”
    shape = (1024, 1024)
    pupil_image_1 = np.zeros(shape)
    pupil_image_2 = np.zeros(shape)
    # fill some patch with pupil_value_1
    pupil_image_1[510:530, 510:530] = pupil_value_1
    pupil_image_2[510:530, 510:530] = pupil_value_2
    err = np.ones(shape)

    # Build Images
    im_pupil1 = Image(pupil_image_1, pri_hdr=prhd, ext_hdr=exthd_pupil, err=err)
    im_pupil2 = Image(pupil_image_2, pri_hdr=prhd, ext_hdr=exthd_pupil, err=err)

    # ----------------------------
    # C) Create a set of off-axis PSFs
    # ----------------------------
    data_psf, psf_locs, half_psf = create_ct_psfs(
        fwhm_mas=fwhm,
        cfam_name=cfam_name,
        n_psfs=n_psfs
    )

    # Combine all frames into a single Dataset
    data_ct = [im_pupil1, im_pupil2] + data_psf
    dataset_ct = Dataset(data_ct)

    # ----------------------------
    # D) Generate the CT cal file
    # ----------------------------
    ct_cal_tmp = corethroughput.generate_ct_cal(dataset_ct)

    # Optionally save it to disk
    if save_cal_file:
        if not cal_filename:
            # e.g. "CoreThroughputCalibration_<ISOTIME>.fits"
            cal_filename = f"CoreThroughputCalibration_{time.Time.now().isot}.fits"
        cal_filepath = os.path.join(corgidrp.default_cal_dir, cal_filename)
        ct_cal_tmp.save(filedir=corgidrp.default_cal_dir, filename=cal_filename)
        print(f"Saved CT cal file to: {cal_filepath}")

    return dataset_ct, ct_cal_tmp


def create_mock_fpamfsam_cal(
    fpam_matrix=None,
    fsam_matrix=None,
    date_valid=None,
    save_file=False,
    output_dir=None,
    filename=None
):
    """
    Create and optionally save a mock FpamFsamCal object.

    Parameters
    ----------
    fpam_matrix : np.ndarray of shape (2,2) or None
        The custom transformation matrix from FPAM to EXCAM. 
        If None, defaults to FpamFsamCal.fpam_to_excam_modelbased.
    fsam_matrix : np.ndarray of shape (2,2) or None
        The custom transformation matrix from FSAM to EXCAM.
        If None, defaults to FpamFsamCal.fsam_to_excam_modelbased.
    date_valid : astropy.time.Time or None
        Date/time from which this calibration is valid.
        If None, defaults to the current time.
    save_file : bool, optional
        If True, save the generated calibration file to disk.
    output_dir : str, optional
        Directory in which to save the file if save_file=True. Defaults to current dir.
    filename : str, optional
        Filename to use if saving to disk. If None, a default name is generated.

    Returns
    -------
    FpamFsamCal
        The newly-created FpamFsamCal object (in memory).
    """
    if fpam_matrix is None:
        fpam_matrix = FpamFsamCal.fpam_to_excam_modelbased
    if fsam_matrix is None:
        fsam_matrix = FpamFsamCal.fsam_to_excam_modelbased

    # Ensure the final shape is (2, 2, 2):
    # [ [fpam_matrix], [fsam_matrix] ]
    combined_array = np.array([fpam_matrix, fsam_matrix])  # shape (2,2,2)

    # Create the calibration object in-memory
    fpamfsam_cal = FpamFsamCal(data_or_filepath=combined_array, date_valid=date_valid)

    if save_file:
        # By default, use the filename from the object's .filename unless overridden
        if not filename:
            filename = fpamfsam_cal.filename  # e.g. "FpamFsamCal_<ISOTIME>.fits"

        if not output_dir:
            output_dir = '.'

        # Save the calibration file
        filepath = os.path.join(output_dir, filename)
        fpamfsam_cal.save(filedir=output_dir, filename=filename)
        print(f"Saved FpamFsamCal to {filepath}")

    return fpamfsam_cal


def inject_companion_mag(x, y, mag, test_zero_point=20):
    """
    Helper to compute a companion flux (in e-/s) for a given 'test zero point' scenario.

    Args:
        x, y (float): Pixel location of the companion.
        mag (float): Desired apparent magnitude.
        test_zero_point (float): Arbitrary zero point used for flux injection.

    Returns:
        dict: e.g. {'x': 120, 'y': 80, 'flux': ...}
    """
    # flux_e_s = 10^[-0.4 * (m - ZP_TEST)]
    flux_e_s = 10**(-0.4 * (mag - test_zero_point))
    return {'x': x, 'y': y, 'flux': flux_e_s}


def test_measure_companions_wcs():
    """
    1) Generate a mock 'direct star image' (flux_image) to measure star flux.
    2) Inject 2 companions in the coronagraphic (or PSF-sub) image.
    3) Define them as "detected_companions" so SNYX### keywords get created.
    4) Save multi-extension FITS with ERR + DQ HDUs.
    5) Call measure_companions and compute flux ratios + magnitudes.
    """
    out_dir = os.path.join('corgidrp', 'data', 'L4TestInput')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mock_l4_companions.fits')


    # A) Generate/measure a "direct star image"
    # mock_flux_image(...) should return an unocculted star image.
    flux_image_list = mock_flux_image(
        exptime=5,
        filter_used="3C",
        cal_factor=CAL_FACTOR,
        save_mocks=True,
        output_path='/Users/jmilton/Github/corgidrp/corgidrp/data/L4TestInput',
        background_val=0,
        add_gauss_noise_val=False
    )
    # Convert from total counts to count rate if needed
    flux_image_list = l2b_to_l3.divide_by_exptime(Dataset(flux_image_list))
    direct_star_image = flux_image_list[0]  # Just take the first for star flux

    # Do a measurement of some kind to get star_flux_e_s in e-/s, maybe this will
    # just known flux / flux cal, not sure yet
    star_flux, star_flux_err, _ = fluxcal.aper_phot(
        direct_star_image,
        encircled_radius=10,
        frac_enc_energy=1.0,
        method='subpixel',
        subpixels=5,
        background_sub=True,
        r_in=12,
        r_out=20,
        centering_method='xy',  
        centroid_roi_radius=10
    )
    print(f"\nMeasured host star flux from direct image: {star_flux:.5f} e-/s")

    # Also get a fluxcal_factor, which might tbd return a zero point:
    if PHOT_METHOD == "Aperture":
        fluxcal_factor = fluxcal.calibrate_fluxcal_aper(
            direct_star_image,
            flux_or_irr=FLUX_OR_IRR,
            phot_kwargs=PHOT_ARGS
        )
    else:
        fluxcal_factor = fluxcal.calibrate_fluxcal_gauss2d(
            direct_star_image,
            flux_or_irr=FLUX_OR_IRR,
            phot_kwargs=PHOT_ARGS
        )

    zero_point = fluxcal_factor.ext_hdr.get('ZP', None)
    print(f"Derived fluxcal_factor: {fluxcal_factor.data} | ZP={zero_point}")

    # B) Create the coronagraphic/PSF-sub image with companion(s) injected

    # create_mock_psfsub_image_wcs returns a mock post-sub image
    # with negative hole near the star center and faint Gaussian companions.
    injected = [
        inject_companion_mag(120, 80, mag=3,  test_zero_point=zero_point),
        inject_companion_mag(90, 130, mag=1, test_zero_point=zero_point),
    ]

    hdul = create_mock_psfsub_image_wcs(
        ny=200, nx=200,
        star_center=None,   # default = image center
        roll_angle=10.0,
        platescale=0.0218,
        injected_companions=injected,
        add_noise=False,
        noise_std=0.01,
        background_level=0.0
    )
    hdul.writeto(out_path, overwrite=True)
    print(f"Mock post-sub file saved to {out_path}")

    # Wrap in corgidrp.data.Image
    psf_sub_image = Image(out_path)

    # Create core throughput cal product
    dataset_ct, ct_cal = create_mock_ct_dataset_and_cal_file(
        fwhm=50,
        n_psfs=20,
        cfam_name='3C',
        save_cal_file=False
    )

    # Create mock fpam to excam cal 
    FpamFsamCal = create_mock_fpamfsam_cal(save_file=False)

    # C) Run measure_companions
    # Pass the measured star_flux_e_s so that measure_companions can compute flux_ratio
    # and (if zero_point is present) an apparent magnitude as well.
    result_table = measure_companions.measure_companions(
        image=psf_sub_image,
        method='aperture',           # or 'psf_fit' or 'forward_model'
        apply_throughput=True,
        apply_fluxcal=True,
        ct_cal = ct_cal,
        cor_dataset = dataset_ct,
        FpamFsamCal = FpamFsamCal,
        fluxcal_factor=fluxcal_factor,
        star_flux_e_s=star_flux,    
        # star_mag=...               # If want to do companion mag = star_mag - 2.5log10(ratio)
        verbose=True
    )

    print("\nResult Table:\n", result_table)
    assert len(result_table) == 2, "Expected 2 measured companions"

    # Inspect results
    for row in result_table:
        print(f"ID={row['id']} | (x,y)=({row['x']:.1f},{row['y']:.1f}) "
              f"| flux_raw={row['flux_raw']:.3f} e-/s | ratio={row['flux_ratio']:.3e} "
              f"| mag={row['mag']}")

    print("Test completed successfully")


'''
@pytest.mark.parametrize("known_mag, test_zp", [
    (1, 10),  # Very bright source, should be accurate
    (5, 10),  # Mid-range magnitude
    (10, 15),  # Dimmer source
    (12, 15),  # Near noise limit
    (14, 15),  # Likely to fail if noise is added to image
])'
'''
def test_validate_zero_point(known_mag, test_zp):
    """
    Validates if the computed zero point matches expectations for various 
    known magnitudes and test zero points.

    1. Compute expected flux for a source with known magnitude.
    2. Inject a single source into a simulated image.
    3. Measure the source flux using aperture photometry.
    4. Compute the measured zero point and compare it to the expected value.
    5. Ensure the measured zero point is within a reasonable tolerance (0.05).

    Parameters:
        known_mag (float): Apparent magnitude of the injected test source.
        test_zp (float): Arbitrary zero point used to compute the expected flux.

    """
    print(f"\n *** Validating known zero points: known_mag={known_mag}, test_zp={test_zp} ***")

    # Output directory for the test FITS file
    out_path = os.path.join('corgidrp', 'data', 'L4TestInput', f'mock_l4_zp_{test_zp}_mag_{known_mag}.fits')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Compute expected flux based on magnitude and zero point
    known_flux = 10**(-0.4 * (known_mag - test_zp))

    # Inject a single test source
    injected = [inject_companion_mag(120, 80, mag=known_mag, test_zero_point=test_zp)]
    print(f"Injected source at (120, 80) with flux {injected[0]['flux']:.6f} e-/s")

    # Create a mock image with injected source (no noise for this test)
    hdul = create_mock_psfsub_image_wcs(
        ny=200, nx=200,
        star_center=None,
        roll_angle=10.0,
        platescale=0.0218,
        injected_companions=injected,
        add_noise=False,
        noise_std=0.01,
        background_level=0.0
    )
    hdul.writeto(out_path, overwrite=True)
    input_image = Image(out_path)

    # Measure the flux using aperture photometry
    flux, flux_err, _ = fluxcal.aper_phot(
        input_image,
        encircled_radius=5,
        frac_enc_energy=1.0,
        method='subpixel',
        subpixels=5,
        background_sub=True,
        r_in=5,
        r_out=10,
        centering_method='xy',
        centroid_roi_radius=5
    )

    print(f"Measured flux: {flux:.6f} ± {flux_err:.6f} e-/s")

    # Compute expected and measured zero points
    expected_zp = known_mag + 2.5 * np.log10(known_flux)
    measured_zp = known_mag + 2.5 * np.log10(flux)

    print(f"Expected ZP: {expected_zp:.3f}")
    print(f"Measured ZP: {measured_zp:.3f}")

    # Assertion: Allow up to 0.05 deviation. for now. until errors are tracked better.
    tolerance = 0.05
    assert abs(measured_zp - expected_zp) <= tolerance, (
        f"Zero point mismatch for known_mag={known_mag}, test_zp={test_zp}:\n"
        f"Expected ZP: {expected_zp:.3f}\n"
        f"Measured ZP: {measured_zp:.3f}\n"
        f"Difference: {abs(measured_zp - expected_zp):.3f} (exceeds {tolerance})"
    )

    print("Test passed\n")

if __name__ == "__main__":
    test_measure_companions_wcs()
    #test_validate_zero_point(14,15)