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
import corgidrp.l4_to_tda as l4_to_tda
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

def generate_coron_dataset_with_companions(
    n_frames=1,
    shape=(200, 200),
    star_center=None,
    companion_xy=None,
    companion_flux=100.0,
    star_flux=1e5,
    roll_angles=None,
    platescale=0.0218,
    add_noise=False,
    noise_std=1.0e-2,
    outdir=None
):
    """
    Create a mock "coronagraphic" dataset with a star behind a coronagraph, plus one or more companions.

    - If the plan is to do forward modeling, pass this dataset to the
      PSF-subtraction pipeline (pyKLIP) to see how the companion is subtracted.
    - If you only have one frame, that’s not ideal for ADI, but you can still do RDI if you have references.

    Parameters
    ----------
    n_frames : int
        Number of frames (images) to create. If >1, you can vary roll angles or
        do ADI-like analysis.
    shape : (ny, nx)
        Size of each frame in pixels.
    star_center : (x, y) or None
        Pixel coordinates of the star center. If None, defaults to the image center.
    companion_xy : list of (x, y) or None
        One or more companion coordinates. E.g. [(120, 80), (90, 130)].
        If None, no companion is injected.
    companion_flux : float or list of float
        Flux for each companion. If multiple companions, pass a list with same length.
    star_flux : float
        Flux of the star. (Used to create a Gaussian approximation or something simpler.)
    roll_angles : list of float or None
        If n_frames>1, pass a list of roll angles. If None, defaults to all 0.
    platescale : float
        Plate scale in arcsec/pixel.
    add_noise : bool
        Whether to add random noise.
    noise_std : float
        Stddev of the noise if add_noise=True.
    outdir : str or None
        If not None, saves the resulting frames to disk in outdir. If None, does not save.

    Returns
    -------
    Dataset
        A corgidrp.data.Dataset with n_frames of coronagraphic images, each with a star and optional companions.
    """
    ny, nx = shape
    if star_center is None:
        star_center = (nx / 2, ny / 2)  # (x, y)

    # If companion_xy is a list of coordinates:
    if companion_xy is None:
        # No companion by default
        companion_xy = []
    # If companion_flux is a single float but multiple companions exist, unify
    if isinstance(companion_flux, (int, float)):
        companion_flux = [companion_flux] * len(companion_xy)

    if roll_angles is None:
        roll_angles = [0.0]*n_frames
    elif len(roll_angles) != n_frames:
        raise ValueError("roll_angles must be length n_frames or None.")

    frames = []
    for i in range(n_frames):
        # Build a data array
        data_arr = np.zeros((ny, nx), dtype=np.float32)

        # (A) Insert a star + coronagraph
        # For simplicity, do a 2D Gaussian for the star, or a "hole" in the center, etc. 
        xgrid, ygrid = np.meshgrid(np.arange(nx), np.arange(ny))
        # standard deviation for star? e.g. FWHM=3 => sigma=3/(2sqrt(2ln2)) ~ 1.27
        sigma = 1.2
        r2 = (xgrid - star_center[0])**2 + (ygrid - star_center[1])**2
        star_gaus = (star_flux / (2*np.pi*sigma**2)) * np.exp(-0.5*r2/sigma**2)
        # reduce star flux at the center. doing a partial mask for demonstration:
        # if you want a hole radius=5:
        hole_mask = r2 < 5**2
        star_gaus[hole_mask] *= 0.01  # 99% blocked in the center
        data_arr += star_gaus.astype(np.float32)

        # (B) Insert companion(s)
        for j, (cx, cy) in enumerate(companion_xy):
            flux_c = companion_flux[j]
            # simple Gaussian companion
            sigma_c = 1.0
            r2c = (xgrid - cx)**2 + (ygrid - cy)**2
            comp_gaus = (flux_c / (2*np.pi*sigma_c**2)) * np.exp(-0.5*r2c/sigma_c**2)
            data_arr += comp_gaus.astype(np.float32)

        # (C) Add noise if requested
        if add_noise:
            noise = np.random.normal(0., noise_std, data_arr.shape)
            data_arr += noise.astype(np.float32)

        # (D) Build primary & extension headers
        prihdr, exthdr = mocks.create_default_L3_headers()
        # Minimal keywords
        prihdr["FILENAME"] = f"mock_coron_{i:03d}.fits"
        exthdr["ROLL"] = roll_angles[i]
        exthdr["PLTSCALE"] = platescale
        exthdr["STARLOCX"] = star_center[0]
        exthdr["STARLOCY"] = star_center[1]
        exthdr["CFAMNAME"] = "1F"  # example
        exthdr["DATALVL"] = "L3"  # or L2, depending on your pipeline convention

        # optional WCS
        wcs_obj = generate_wcs(
            roll_angles[i],
            [star_center[0], star_center[1]],
            platescale=platescale
        )
        wcs_header = wcs_obj.to_header()
        exthdr.update(wcs_header)

        # Make a corgidrp Image
        frame = Image(data_arr, pri_hdr=prihdr, ext_hdr=exthdr)
        frames.append(frame)

    dataset = Dataset(frames)

    # Optionally save
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        # you can either save each frame individually or combine them in a single FITS
        file_list = [f"mock_coron_{i:03d}.fits" for i in range(n_frames)]
        dataset.save(filedir=outdir, filenames=file_list)
        print(f"Saved {n_frames} coronagraphic frames to {outdir}")

    return dataset


def generate_psfsub_image_with_companions(
    nx=200, ny=200,
    star_center=None,
    star_flux=1e5,          # Arbitrary star flux
    star_sub_scale=0.7,
    # Companion position/flux vs. magnitude
    companion_xy=None,
    companion_flux=None,
    companion_mags=None,
    zero_point=25.0,
    # If you have star flux/mag & want F_comp = F_star * 10^-0.4(m_c - m_s),
    # you can do that here. For simplicity, we do a fixed zero_point approach.
    
    ct_cal=None,            # Optional CoreThroughputCalibration
    use_ct_cal=False,       # Whether to apply throughput from ct_cal
    cor_dataset = None,     # Dataset used to make the CoreThroughputCalibration
    FpamFsamCal = None,
    blur_sigma=0.5,
    noise_std=1e-7,
    outdir=None,
    roll_angle=0.0,
    platescale=0.0218,
    hole_radius=None,
):
    """
    Create a single mock PSF-subtracted residual image with star mostly removed 
    and faint companions. We simulate the star + companions prior to subtraction,
    then apply a global factor star_sub_scale to mimic the star's partial removal
    (thus also removing some fraction of the companions).

    Optionally:
      - Convert companion magnitudes to flux using a zero point.
      - Apply a throughput factor from a CoreThroughputCalibration object.

    Parameters
    ----------
    nx, ny : int
        Image size in pixels.
    star_center : (x, y) or None
        If None, defaults to image center.
    star_flux : float
        Total flux of the star prior to PSF subtraction.
    star_sub_scale : float
        Fraction of the star’s flux (and companion flux) that remains after subtraction.
        E.g., 0.7 => 70% remains => partial under-subtraction. 
        If <0, we get negative residual (over-subtraction).
    companion_xy : list of (x, y) or None
        Pixel coords of each companion.
    companion_flux : float or list of float, optional
        Direct flux for each companion (pre-throughput).
    companion_mags : float or list of float, optional
        Apparent magnitudes for each companion, which we’ll convert to flux via zero_point.
        If present, we ignore companion_flux (unless you want to combine them, but typically no).
    zero_point : float
        Photometric zero point used if companion_mags is specified. 
        e.g. F = 10^-0.4 (mag - ZP)
    ct_cal : CoreThroughputCalibration or None
        If provided and use_ct_cal=True, apply throughput factor at each companion location.
    use_ct_cal : bool
        If True, apply the throughput factor from ct_cal for each companion location.
    blur_sigma : float
        Gaussian blur at the end to mimic real instrumentation.
    noise_std : float
        Standard deviation of random noise to add.
    outdir : str or None
        If a directory is given, we save the final image there.
    roll_angle : float
        For WCS generation in degrees.
    platescale : float
        Arcsec per pixel.
    hole_radius : float or None
        If set, reduce flux in a circular region by some factor (e.g. 0.1).

    Returns
    -------
    frame : corgidrp.data.Image
        The final post-sub image in a corgidrp Image object (with SCI/ERR/DQ).
    """

    # 1) Grid for the image
    if star_center is None:
        star_center = (nx / 2, ny / 2)
    xgrid, ygrid = np.meshgrid(np.arange(nx), np.arange(ny))
    r2 = (xgrid - star_center[0])**2 + (ygrid - star_center[1])**2

    # 2) Create star PSF
    sigma_star = 2.0
    star_psf = star_flux / (2 * np.pi * sigma_star**2) * np.exp(-0.5 * r2 / sigma_star**2)

    # 3) Initialize companion arrays
    if companion_xy is None:
        companion_xy = []
    # Convert single float => list
    if isinstance(companion_flux, (int, float)):
        companion_flux = [companion_flux] * len(companion_xy)
    if isinstance(companion_mags, (int, float)):
        companion_mags = [companion_mags] * len(companion_xy)

    # If user provided magnitudes, convert them to flux
    # ignoring companion_flux if both are provided:
    comp_flux_list = []
    if companion_mags is not None:
        # Convert each mag => flux via zero point
        for mag in companion_mags:
            flux_c = 10**(-0.4*(mag - zero_point))  # e-/s, for example
            comp_flux_list.append(flux_c)
    elif companion_flux is not None:
        comp_flux_list = companion_flux[:]
    else:
        # No companions
        comp_flux_list = []

    # 4) If desired, apply a core throughput factor for each companion
    #    That is, the pre-sub flux is scaled by the coronagraph's throughput at that offset

    # 5) Add companions to the star image (pre-sub)
    for (cx, cy), flux_c in zip(companion_xy, comp_flux_list):
        # optional throughput factor
        if not use_ct_cal or ct_cal is None:
            throughput_factor = 1.0
        else:
            throughput_factor = measure_companions.measure_core_throughput_at_location(
                cx, cy,
                ct_cal,
                cor_dataset,
                FpamFsamCal,
                flux_c
            )
        print("throughput factor for making mocks", throughput_factor)
        flux_c_actual = flux_c * throughput_factor

        print(f"Companion at ({cx:.2f}, {cy:.2f}) - Input Flux Before CT: {flux_c:.6g} e-")
        print(f"Companion at ({cx:.2f}, {cy:.2f}) - Flux After CT: {flux_c_actual:.6g} e-")

        # simple 2D Gaussian
        sigma_c = 1.2
        rr2c = (xgrid - cx)**2 + (ygrid - cy)**2
        comp_gaus = flux_c_actual / (2 * np.pi * sigma_c ** 2) * np.exp(-0.5 * rr2c / sigma_c ** 2)
        star_psf += comp_gaus  

    # 6) "PSF subtraction": we scale the star+companions by star_sub_scale
    residual_image = star_sub_scale * star_psf
    print(f"Companion at ({cx:.2f}, {cy:.2f}) - Flux After CT and PSF sub: {flux_c_actual*star_sub_scale:.6g} e-")

    # 7) Optional hole in center
    if hole_radius is not None and hole_radius > 0:
        hole_mask = (r2 <= hole_radius**2)
        residual_image[hole_mask] *= 0.1  # arbitrary fraction

    # 8) Final blur + noise
    from scipy.ndimage import gaussian_filter
    residual_image = gaussian_filter(residual_image, sigma=blur_sigma).astype(np.float32)
    residual_image += np.random.normal(0., noise_std, residual_image.shape).astype(np.float32)

    # 9) Build corgidrp Image

    prihdr, exthdr = mocks.create_default_L4_headers()
    exthdr["STARLOCX"] = star_center[0]
    exthdr["STARLOCY"] = star_center[1]
    wcs_obj = generate_wcs(roll_angle, [star_center[0], star_center[1]], platescale=platescale)
    wcs_header = wcs_obj.to_header()
    exthdr.update(wcs_header)

    # Record companion location in the header
    for i, (cx, cy) in enumerate(companion_xy, start=1):
        exthdr[f"SNYX{i:03d}"] = f"5.0,{cx},{cy}"

    err_data = np.full_like(residual_image, noise_std, dtype=np.float32)
    dq_data  = np.zeros_like(residual_image, dtype=np.uint16)

    frame = Image(residual_image, pri_hdr=prihdr, ext_hdr=exthdr, err=err_data, dq=dq_data)

    # 10) Save if requested
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        frame.save(filedir=outdir, filename=f"mock_psfsub_{i:03d}.fits")
        print(f"Saved PSF-subtracted frame to", os.path.join(outdir, f"mock_psfsub_{i:03d}.fits"))

    return frame


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

    # A) Generate a direct unocculted star image (from known standard stars for now)
    flux_image_list = mock_flux_image(
        exptime=5,
        filter_used="3C",
        cal_factor=CAL_FACTOR,
        save_mocks=True,
        output_path=out_dir,
        background_val=0,
        add_gauss_noise_val=False
    )

    # Convert from total counts to count rate if needed
    flux_image_list = l2b_to_l3.divide_by_exptime(Dataset(flux_image_list))
    direct_star_image = flux_image_list[0]  # Just take the first for star flux

    # Calculate star AP_MAG
    # TO DO: what if the unocculted star image isn't a standard star? use the zero point idea maybe
    image_with_mag = l4_to_tda.determine_app_mag(direct_star_image, direct_star_image.pri_hdr["TARGET"])
    host_star_ap_mag = image_with_mag[0].ext_hdr['APP_MAG']
    print("Host star ap mag,", host_star_ap_mag)

    # Do a measurement of some kind to get star_flux_e in e-, maybe this will
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
    print(f"\nMeasured host star flux from direct image photometry: {star_flux:.5f} photoelectrons")

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

    # Create core throughput cal product
    dataset_ct, ct_cal = create_mock_ct_dataset_and_cal_file(
        fwhm=50,
        n_psfs=20,
        cfam_name='3C',
        save_cal_file=False
    )

    # Create mock fpam to excam cal 
    FpamFsamCal = create_mock_fpamfsam_cal(save_file=False)

    print("Star mag at (120,80): ", (-2.5 * np.log10(star_flux/2) + zero_point))
    print("Star mag at (90,130): ", (-2.5 * np.log10(star_flux/3) + zero_point))

    # B) Create the coronagraphic/PSF-sub image with companion(s) injected
    psf_sub_frame = generate_psfsub_image_with_companions(
        nx=200, ny=200,
        star_center=None,
        star_flux=star_flux,        
        star_sub_scale=0.7,
        # Companion position/flux vs. magnitude
        companion_xy=[(120,80), (90,130)],
        companion_flux=[star_flux/2, star_flux/3],
        companion_mags=[(-2.5 * np.log10(star_flux/2) + zero_point), (-2.5 * np.log10(star_flux/3) + zero_point)],
        zero_point=zero_point,
        ct_cal=ct_cal,            # Optional CoreThroughputCalibration
        use_ct_cal=True,       # Whether to apply throughput from ct_cal
        cor_dataset = dataset_ct,     # Dataset used to make the CoreThroughputCalibration
        FpamFsamCal = FpamFsamCal,
        blur_sigma=0.5,
        noise_std=1e-8,
        outdir=out_dir,
        roll_angle=0.0,
        platescale=0.0218,
        hole_radius=None,
    )

    coron_data = generate_coron_dataset_with_companions(
        n_frames=1,
        shape=(200,200),
        companion_xy=[(120,80)],
        companion_flux=2000.,
        star_flux=1e5,
        roll_angles=[10.0],
        add_noise=True,
        noise_std=5.,
        outdir=out_dir
    )

    # C) Run measure_companions
    # Pass the measured star_flux_e so that measure_companions can compute flux_ratio
    # and (if zero_point is present) an apparent magnitude as well.
    result_table = measure_companions.measure_companions(
        image=psf_sub_frame,
        method='aperture',           # or 'psf_fit' or 'forward_model'
        apply_throughput=True,
        apply_fluxcal=True,
        ct_cal = ct_cal,
        cor_dataset = dataset_ct,
        FpamFsamCal = FpamFsamCal,
        fluxcal_factor=fluxcal_factor,
        star_flux_e=star_flux,    
        apply_psf_sub_eff= True,
        # star_mag=...               # If want to do companion mag = star_mag - 2.5log10(ratio)
        verbose=True
    )

    print("\nResult Table:\n", result_table)
    assert len(result_table) == 2, "Expected 2 measured companions"



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
    out_dir = os.path.join('corgidrp', 'data', 'L4TestInput')
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    # Compute expected flux based on magnitude and zero point
    known_flux = 10**(-0.4 * (known_mag - test_zp))

    # Create a mock image with injected source
    coron_data = generate_coron_dataset_with_companions(
        n_frames=1,
        shape=(200,200),
        companion_xy=[(120,80)],
        companion_flux=2000.,
        star_flux=1e5,
        roll_angles=[10.0],
        add_noise=True,
        noise_std=5.,
        outdir=out_dir
    )

    # Measure the flux using aperture photometry
    flux, flux_err, _ = fluxcal.aper_phot(
        coron_data,
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