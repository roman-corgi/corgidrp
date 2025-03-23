import os
import numpy as np
import corgidrp.mocks as mocks
import corgidrp.l4_to_tda as l4_to_tda
from corgidrp.data import Image
import corgidrp.measure_companions as measure_companions
import corgidrp.fluxcal as fluxcal
import corgidrp.nd_filter_calibration as nd_filter_calibration


# Global Constants
HOST_STAR = 'TYC 4433-1800-1'
CFAM = '1F'
FWHM = 3
INPUT_EFFICIENCY_FACTOR = 0.8
PHOT_METHOD = "Aperture"
FLUX_OR_IRR = 'irr'
NUM_IMAGES = 10
ROLL_ANGLES = np.linspace(0, 45 , NUM_IMAGES)
PSF_SUB_SCALE = 0.7
OUT_DIR = os.path.join('corgidrp', 'data', 'L4TestInput')

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

    Args:
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

    Args:
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

    # 1) Generate core throughput calibration dataset and FPAM/ FSAM to EXCAM rot matrix
    dataset_ct, ct_cal = mocks.create_mock_ct_dataset_and_cal_file(fwhm=50, n_psfs=20, cfam_name=CFAM, save_cal_file=True)
    FpamFsamCal = mocks.create_mock_fpamfsam_cal(save_file=False)


    # 2) Generate reference star dataset with no companions, not PSF-subtracted
    #       ND Filter should be in place and attenuate signal. <- TO DO
    #       Units of image should be in photoelectrons
    
    ref_star_flux = nd_filter_calibration.compute_expected_band_irradiance(HOST_STAR, "1F")
    ref_star_dataset = mocks.generate_reference_star_dataset_with_flux(
        n_frames=NUM_IMAGES,
        roll_angles=ROLL_ANGLES,       
        flux_erg_s_cm2=ref_star_flux,
        fwhm=FWHM,
        cal_factor = 1e10,
        optical_throughput=INPUT_EFFICIENCY_FACTOR,     
        color_cor=1.0,
        filter=CFAM,
        fpamname='ND475',
        target_name=HOST_STAR,
        fsm_x=0.0,
        fsm_y=0.0,
        exptime=1.0,
        platescale=21.8,     # mas/pixel
        background=0,
        add_gauss_noise=True,
        noise_scale=1.,
        filedir=OUT_DIR,
        file_save=True,
        shape = (200,200)
    )


    # 3) Figure out the counts and zero point of the host star, using aperture photometry
    # TO DO: color correction here
    host_star_counts, _, _ = fluxcal.aper_phot(
        ref_star_dataset[0], encircled_radius=10, frac_enc_energy=1.0, method='subpixel',
        subpixels=5, background_sub=True, r_in=12, r_out=20, centering_method='xy',
        centroid_roi_radius=10
    )

    host_star_mag = l4_to_tda.determine_app_mag(ref_star_dataset[0], HOST_STAR)

    print(f"\nMeasured host star counts: {host_star_counts:.5f} photoelectrons")
    print("Host star mag", host_star_mag[0].ext_hdr['APP_MAG'])

    fluxcal_factor = get_fluxcal_factor(ref_star_dataset[0], PHOT_METHOD, PHOT_ARGS, FLUX_OR_IRR)
    zero_point = fluxcal_factor.ext_hdr.get('ZP', None)


    # 4) Generate coronagraphic dataset - star behind coronagraph with companions, 
    #   not PSF-subtracted
    #       Units of image should be in photoelectrons
    #       TO DO: make this more aligned with the reference star dataset? aka use
    #       filter, etc?
    # TO DO: add more companions
    coron_data = mocks.generate_coron_dataset_with_companions(
        n_frames=NUM_IMAGES, shape=(200, 200), 
        companion_xy = (120,80), 
        companion_counts= host_star_counts / 2,                                #[host_star_counts / 2, host_star_counts / 3],
        host_star_counts=host_star_counts, roll_angles=ROLL_ANGLES, add_noise=True, noise_std=1e-2, darkhole_file=dataset_ct[0],
        outdir=OUT_DIR
    )

    print("Companion 1 AP Mag:", (-2.5 * np.log10(host_star_counts / 2) + zero_point))

    
    # 5) Generate PSF-subtracted frame(s?) of star behind the coronagraph, with companions
    #       Units should be in photoelectrons
    psf_sub_frame = mocks.generate_psfsub_image_with_companions(
        nx=200, ny=200, host_star_center=None, host_star_counts=host_star_counts,
        psf_sub_scale=PSF_SUB_SCALE, companion_xy=[(120,80)],                   #[(120, 80), (90, 130)],
        companion_counts=[host_star_counts / 2],                                #[host_star_counts / 2, host_star_counts / 3],
        companion_mags=[(-2.5 * np.log10(host_star_counts / 2) + zero_point)],  #[(-2.5 * np.log10(host_star_counts / 2) + zero_point), (-2.5 * np.log10(host_star_counts / 3) + zero_point)],
        zero_point=zero_point, ct_cal=ct_cal, cor_dataset = dataset_ct, use_ct_cal=True,
        FpamFsamCal=FpamFsamCal, blur_sigma=0.5, noise_std=1e-8, outdir=out_dir, platescale=0.0218
    )


    return ref_star_dataset, host_star_counts, fluxcal_factor, zero_point, dataset_ct, ct_cal, FpamFsamCal, psf_sub_frame, coron_data


def test_measure_companions_wcs(out_dir):
    """
    Run the full test case for measuring companions.
    """
    ref_star_dataset, host_star_counts, fluxcal_factor, zero_point, dataset_ct, ct_cal, FpamFsamCal, \
        psf_sub_frame, coron_data = generate_test_data(out_dir)
    
    # Get reference psf (it is near 6 lam/D), assume off-axis PSF with highest CT has negligible 
    # effect from the mask. For now I am identifying it here and passing it in, but maybe we can 
    # just pass in the whole dataset and pick it out using distance or highest CT.
    reference_psf = ct_cal.data[0]
    pri_hdr=ct_cal.pri_hdr
    ext_hdr=ct_cal.ext_hdr
    pri_hdr['FILENAME'] = 'MockReferencePSF.fits'
    ext_hdr['PLTSCALE'] = 0.0218
    ext_hdr['WCSAXES'] = 2
    ext_hdr['CTYPE1']  = 'RA---TAN'
    ext_hdr['CTYPE2']  = 'DEC--TAN'
    ext_hdr['CRPIX1']  = 100       # Set the reference pixel at the center
    ext_hdr['CRPIX2']  = 100
    ext_hdr['CRVAL1']  = 0.0       # Reference coordinate (e.g., 0 deg)
    ext_hdr['CRVAL2']  = 0.0
    ext_hdr['CDELT1']  = -0.00027778  # Pixel scale in degrees (approx 1 arcsec/pixel; negative for RA)
    ext_hdr['CDELT2']  = 0.00027778   # Pixel scale in degrees
    ext_hdr['PC1_1']   = 1.0
    ext_hdr['PC1_2']   = 0.0
    ext_hdr['PC2_1']   = 0.0
    ext_hdr['PC2_2']   = 1.0
    reference_psf = Image(data_or_filepath=reference_psf, pri_hdr=pri_hdr, ext_hdr=ext_hdr)

    # Run Measurement
    result_table = measure_companions.measure_companions(coron_data, ref_star_dataset, psf_sub_frame, reference_psf,
                                                         ct_cal, FpamFsamCal, phot_method='aperture',
                                                         fluxcal_factor=fluxcal_factor, forward_model=True, output_dir = out_dir, 
                                                         verbose=True
    )

    print("\nResult Table:\n", result_table)
    #assert len(result_table) == 2, "Expected 2 measured companions"


if __name__ == "__main__":
    test_measure_companions_wcs(OUT_DIR)