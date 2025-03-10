import os
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter

from pyklip.instruments.utils.wcsgen import generate_wcs

import corgidrp.mocks as mocks
import corgidrp.l2b_to_l3 as l2b_to_l3
import corgidrp.measure_companions as measure_companions
import corgidrp.fluxcal as fluxcal
import corgidrp.nd_filter_calibration as nd_filter_calibration
from corgidrp.data import Image, Dataset

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
      1) Inject 2 companions in the image.
      2) Also define them as "detected_companions" with SNR + location,
         so the SNYX### keywords get created.
      3) Save multi-extension FITS with ERR + DQ HDUs.
      5) Call measure_companions.
    """
    out_dir = os.path.join('corgidrp', 'data', 'L4TestInput')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mock_l4_companions.fits')

    # Get a fluxcal factor calibration product
    flux_image = mock_flux_image(5, "3C", CAL_FACTOR, True, output_path='/Users/jmilton/Github/corgidrp/corgidrp/data/L4TestInput', 
                           background_val=0, add_gauss_noise_val=False)
    
    flux_image = l2b_to_l3.divide_by_exptime(Dataset(flux_image))

    if PHOT_METHOD == "Aperture":
        fluxcal_factor = fluxcal.calibrate_fluxcal_aper(flux_image[0], flux_or_irr = FLUX_OR_IRR, phot_kwargs=PHOT_ARGS)
    else:
        fluxcal_factor = fluxcal.calibrate_fluxcal_gauss2d(flux_image[0], flux_or_irr = FLUX_OR_IRR, phot_kwargs=PHOT_ARGS)

    print("fluxcal factor", fluxcal_factor.data, fluxcal_factor.ext_hdr['ZP'])

    # Get the zero point
    zero_point = fluxcal_factor.ext_hdr['ZP']

    injected = [
        inject_companion_mag(120, 80, mag=3, test_zero_point=zero_point),
        inject_companion_mag(90, 130, mag=1, test_zero_point=zero_point),
    ]

    # Generate the HDUList
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

    # Write to disk
    hdul.writeto(out_path, overwrite=True)
    print(f"Mock file saved to {out_path}")

    input_image = Image(out_path)

    # Run measure_companions
    result_table = measure_companions.measure_companions(
        image=input_image,
        method='aperture',
        apply_throughput=True,
        apply_fluxcal=True,
        fluxcal_factor = fluxcal_factor,
        verbose=True
    )
    print("\nResult Table:\n", result_table)

    assert len(result_table) == 2, "Expected 2 measured companions"

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