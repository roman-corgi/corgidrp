# A file that holds the functions to handle polarimetry data 
import os
import numpy as np
import pandas as pd

from corgidrp.data import Image, NDMuellerMatrix, MuellerMatrix, Dataset
from corgidrp.fluxcal import aper_phot, measure_aper_flux_pol

def aper_phot_pol(image, phot_kwargs):
    """
    Perform aperture photometry on both channels of a 2-channel polarimetric image.

    Each input image contains two orthogonally polarized beams (e.g., ordinary and extraordinary).
    This function measures the flux and its uncertainty in both channels using aperture photometry.

    Args:
        image (Image): Polarimetric Image object with shape (2, ny, nx).
            Must contain `data`, `err`, and `dq` attributes.
        phot_kwargs (dict): Keyword arguments passed to `aper_phot`, defining
            aperture radius, centering method, background subtraction, etc.

    Returns:
        tuple[list, list]:
            (flux, flux_err) lists of fluxes and uncertainties for both polarization channels.
    """
    flux = []
    flux_err = []
    for i in range(2):
        im_copy = image.copy()
        im_copy.data = im_copy.data[i]
        im_copy.err = im_copy.err[0][i].reshape(np.append([1], [im_copy.data.shape]))
        im_copy.dq  = im_copy.dq[i]

        f, f_e = aper_phot(im_copy, **phot_kwargs)

        flux.append(f)
        flux_err.append(f_e)

    return flux, flux_err

def calc_stokes_unocculted(input_dataset,
                           phot_kwargs=None,
                           image_center_x=None, 
                           image_center_y=None,
                           split_pa_states=True,
                           pa_tolerance=0.1):
    """
    Compute uncalibrated Stokes parameters (I, Q/I, U/I) from unocculted L3 polarimetric datacubes.

    Each element in `dataset` represents a single observation taken with a specific Wollaston prism
    (e.g., POL0 or POL45), which splits the incoming light into two orthogonally polarized beams.
    This function performs aperture photometry on each beam and computes the corresponding Stokes
    parameters in the instrument frame.

    Args:
        input_dataset (corgidrp.data.Dataset):
            A corgidrp dataset of L3 polarimetric images
        phot_kwargs (dict, optional):
            Keyword arguments passed to `aper_phot`. If not provided, a default aperture setup is used.
        image_center_x (float, optional):
            X-coordinate of the aperture center in pixels. Default is None. 
            If None, assume the center of the array is a good guess. 
        image_center_y (float, optional):
            Y-coordinate of the aperture center in pixels. Default is None.
            If None, assume the center of the array is a good guess. 
        split_pa_states (bool, optional):
            If True, split the input dataset by both target and PA_APER. If False, split only by target.
            Default is True.
        pa_tolerance (float, optional):
            Maximum allowed difference in PA_APER (deg) to group frames together when split_pa_states is True.
            Default is 0.1.

    Returns:
        Image:
            A `corgidrp.data.Image` instance containing:
            - `data` (ndarray, shape=(3,)): [I, Q/I, U/I]
            - `err`  (ndarray, shape=(3,)): propagated uncertainties
            - `dq`   (ndarray, shape=(3,)): data quality flags (zeros)
            - FITS headers (pri_hdr, ext_hdr, err_hdr, dq_hdr) propagated from the first input image.

    Raises:
        ValueError:
            If an input image contains an unrecognized prism name in 'DPAMNAME'.
    """
    # Ensure xy centering method is used with estimated centers for aperture photometry
    if phot_kwargs is None:
        phot_kwargs = {
            'encircled_radius': 5,
            'frac_enc_energy': 1.0,
            'method': 'subpixel',
            'subpixels': 5,
            'background_sub': False,
            'r_in': 5,
            'r_out': 10,
            'centroid_roi_radius': 5,
            'centering_initial_guess': [image_center_x, image_center_y]
        }

    prism_map = {'POL0': [0., 90.], 'POL45': [45., 135.]}

    # split datasets by target if there are multiple targets
    if split_pa_states:
        datasets = []
        target_datasets, _ = input_dataset.split_dataset(prihdr_keywords=["TARGET"])
        for target_dataset in target_datasets:
            # Assign frames to a cluster based on the nearest PA_APER
            clusters = []
            for frame in target_dataset.frames:
                pa = frame.pri_hdr["PA_APER"] % 360.0 # in case PA_APER can be negative..
                pa_rad = np.deg2rad(pa)
                pa_sin = np.sin(pa_rad)
                pa_cos = np.cos(pa_rad)
                best_idx = None
                best_diff = None
                for idx, cluster in enumerate(clusters):
                    pa_diff = abs(((pa - cluster["pa_center"] + 180.0) % 360.0) - 180.0)
                    if pa_diff <= pa_tolerance and (best_diff is None or pa_diff < best_diff):
                        best_idx = idx
                        best_diff = pa_diff
                if best_idx is None:
                    clusters.append({
                        "pa_center": pa,
                        "sum_sin": pa_sin,
                        "sum_cos": pa_cos,
                        "frames": [frame],
                    })
                else:
                    cluster = clusters[best_idx]
                    cluster["frames"].append(frame)
                    cluster["sum_sin"] += pa_sin
                    cluster["sum_cos"] += pa_cos
                    cluster["pa_center"] = np.degrees(
                        np.arctan2(cluster["sum_sin"], cluster["sum_cos"])
                    ) % 360.0
            for cluster in clusters:
                datasets.append(Dataset(cluster["frames"]))
    else:
        datasets, _ = input_dataset.split_dataset(prihdr_keywords=["TARGET"])

    stokes_vectors = []

    for dataset in datasets:
        fluxes, flux_errs, thetas = [], [], []
        # --- Photometry loop ---
        for ds in dataset:
            prism = ds.ext_hdr.get('DPAMNAME')
            if prism not in prism_map:
                raise ValueError(f"Unknown prism: {prism}")
            
            flux, flux_err = aper_phot_pol(ds, phot_kwargs)
            fluxes.append(flux)
            flux_errs.append(flux_err)
            
            for phi in prism_map[prism]:
                thetas.append(np.radians(phi))

        fluxes = np.array(fluxes)
        flux_errs = np.array(flux_errs)
        thetas = np.array(thetas)

        # Prevent division by zero
        if np.any(flux_errs == 0):
            flux_errs[flux_errs == 0] = np.min(flux_errs[flux_errs > 0])

        # --- Instrument coordinates: left - right ---
        n_images = len(dataset)
        fluxes = fluxes.reshape([n_images, 2])
        flux_errs = flux_errs.reshape([n_images, 2])
        thetas = thetas.reshape([n_images, 2])
        I_vals = np.sum(fluxes, axis=1)
        I_errs = np.sqrt(np.sum(flux_errs**2, axis=1))
            
        QU_vals = fluxes[:,0] - fluxes[:,1]  # left - right
            
        # Weighted means across all prisms
        wI = 1.0 / I_errs**2
        I_val = np.sum(I_vals * wI) / np.sum(wI)
        I_err = 1.0 / np.sqrt(np.sum(wI))

        idx_0 = np.where(np.degrees(thetas[:,0]) == 0)[0]
        if idx_0.size > 0:
            Q_vals = QU_vals[idx_0]
            Q_val = np.sum(Q_vals * wI[idx_0]) / np.sum(wI[idx_0])
            Q_err = 1.0 / np.sqrt(np.sum(wI[idx_0]))
        else:
            Q_val = 0.
            Q_err = 0.

        idx_45 = np.where(np.degrees(thetas[:,0]) == 45)[0]
        if idx_45.size > 0:
            U_vals = QU_vals[idx_45]
            U_val = np.sum(U_vals * wI[idx_45]) / np.sum(wI[idx_45])
            U_err = 1.0 / np.sqrt(np.sum(wI[idx_45]))
        else:
            U_val = 0.
            U_err = 0.

        # Fractional polarization
        Q_frac = Q_val / I_val
        U_frac = U_val / I_val
        Q_frac_err = np.sqrt((Q_err/I_val)**2 + (Q_val*I_err/I_val**2)**2)
        U_frac_err = np.sqrt((U_err/I_val)**2 + (U_val*I_err/I_val**2)**2)

        data_out = np.array([I_val, Q_frac, U_frac, 0.])
        err_out = np.array([I_err, Q_frac_err, U_frac_err, np.inf])
        dq_out = np.zeros_like(data_out, dtype=int)

        # --- Headers ---
        pri_hdr = dataset[0].pri_hdr
        ext_hdr = dataset[0].ext_hdr
        ext_hdr.add_history("Computed uncalibrated Stokes parameters: data=[I, Q/I, U/I]")
        err_hdr = dataset[0].err_hdr
        dq_hdr = dataset[0].dq_hdr

        stokes_vector = Image(
            data_out,
            pri_hdr=pri_hdr,
            ext_hdr=ext_hdr,
            err=err_out,
            dq=dq_out,
            err_hdr=err_hdr,
            dq_hdr=dq_hdr
        )
        stokes_vector.filename = os.path.basename(dataset[0].filename).replace("l3", "stokes")

        stokes_vectors.append(stokes_vector)

    stokes_dataset = Dataset(stokes_vectors)

    return stokes_dataset

def calc_stokes_unocculted_dithered(input_dataset,
                                    phot_kwargs=None,
                                    image_center_x=None, 
                                    image_center_y=None,
                                    split_pa_states=True,
                                    pa_tolerance=0.1):

    """
    Compute uncalibrated Stokes parameters (I, Q/I, U/I) from unocculted L3 polarimetric datacubes.

    This is a dithered unocculted polarization-calibration variant of
    `calc_stokes_unocculted`. It is intended for calibrators with acquisition
    residuals that place their PSFs at slightly different detector positions.
    The dither pattern should keep every PSF within one fixed detector aperture
    and make the unaligned, dither-summed PSF illumination footprints overlap
    sufficiently between calibrators. This reduces calibrator-dependent
    weighting of spatial detector response; a shared aperture alone does not.

    Unlike `calc_stokes_unocculted`, which centers aperture photometry
    independently for each input frame, this function derives one aperture
    center from all input frames and keeps it fixed in detector coordinates.
    It assumes the required dither pattern and overlap are present but does not
    create or verify them.

    Each TARGET and PA_APER group must contain both POL0 and POL45 frames. The
    function sums the dithered left and right beam fluxes before forming Q/I or
    U/I; for example, POL0 uses Q/I = sum(F_left - F_right) /
    sum(F_left + F_right). With the fixed aperture, this is equivalent to
    combining the images in detector coordinates before photometry. A single
    absolute I is not defined across dither positions, so I is output as NaN
    while the POL0 and POL45 summed intensities are retained.

    Each element in `dataset` represents a single observation taken with a specific Wollaston prism
    (e.g., POL0 or POL45), which splits the incoming light into two orthogonally polarized beams.
    This function performs aperture photometry on each beam and computes the corresponding Stokes
    parameters in the instrument frame.

    Args:
        input_dataset (corgidrp.data.Dataset):
            A corgidrp dataset of L3 polarimetric images
        phot_kwargs (dict, optional):
            Keyword arguments passed to `aper_phot`. If not provided, a default aperture setup is used.
        image_center_x (float, optional):
            X-coordinate of the aperture center in pixels. Default is None. 
            If None, assume the center of the array is a good guess. 
        image_center_y (float, optional):
            Y-coordinate of the aperture center in pixels. Default is None.
            If None, assume the center of the array is a good guess. 
        split_pa_states (bool, optional):
            If True, split the input dataset by both target and PA_APER. If False, split only by target.
            Default is True.
        pa_tolerance (float, optional):
            Maximum allowed difference in PA_APER (deg) to group frames together when split_pa_states is True.
            Default is 0.1.

    Returns:
        Image:
            A `corgidrp.data.Image` instance containing:
            - `data` (ndarray, shape=(6,)): [I, Q/I, U/I, V, I_POL0, I_POL45]
              where I is NaN because no common total intensity is defined for
              unregistered dithered frames.
            - `err`  (ndarray, shape=(6,)): propagated uncertainties
            - `dq`   (ndarray, shape=(6,)): data quality flags (zeros)
            - FITS headers (pri_hdr, ext_hdr, err_hdr, dq_hdr) propagated from the first input image.

    Raises:
        ValueError:
            If an input image contains an unrecognized prism name in 'DPAMNAME'.
    """
    if image_center_x is None or image_center_y is None:
        peak_positions = []
        for frame in input_dataset:
            for beam in frame.data:
                if not np.any(np.isfinite(beam)):
                    continue
                peak_y, peak_x = np.unravel_index(
                    np.nanargmax(beam), beam.shape)
                peak_positions.append((peak_x, peak_y))
        if not peak_positions:
            raise ValueError("Cannot determine a fixed aperture center from empty frames")

        mean_peak_x, mean_peak_y = np.mean(peak_positions, axis=0)
        if image_center_x is None:
            image_center_x = float(mean_peak_x)
        if image_center_y is None:
            image_center_y = float(mean_peak_y)

    # Ensure xy centering method is used with estimated centers for aperture photometry
    if phot_kwargs is None:
        phot_kwargs = {
            'encircled_radius': 5,
            'frac_enc_energy': 1.0,
            'method': 'subpixel',
            'subpixels': 5,
            'background_sub': False,
            'centroid_roi_radius': 0,
            'centering_initial_guess': [image_center_x, image_center_y]
        }

    prism_map = {'POL0': [0., 90.], 'POL45': [45., 135.]}

    # split datasets by target if there are multiple targets
    if split_pa_states:
        datasets = []
        target_datasets, _ = input_dataset.split_dataset(prihdr_keywords=["TARGET"])
        for target_dataset in target_datasets:
            # Assign frames to a cluster based on the nearest PA_APER
            clusters = []
            for frame in target_dataset.frames:
                pa = frame.pri_hdr["PA_APER"] % 360.0 # in case PA_APER can be negative..
                pa_rad = np.deg2rad(pa)
                pa_sin = np.sin(pa_rad)
                pa_cos = np.cos(pa_rad)
                best_idx = None
                best_diff = None
                for idx, cluster in enumerate(clusters):
                    pa_diff = abs(((pa - cluster["pa_center"] + 180.0) % 360.0) - 180.0)
                    if pa_diff <= pa_tolerance and (best_diff is None or pa_diff < best_diff):
                        best_idx = idx
                        best_diff = pa_diff
                if best_idx is None:
                    clusters.append({
                        "pa_center": pa,
                        "sum_sin": pa_sin,
                        "sum_cos": pa_cos,
                        "frames": [frame],
                    })
                else:
                    cluster = clusters[best_idx]
                    cluster["frames"].append(frame)
                    cluster["sum_sin"] += pa_sin
                    cluster["sum_cos"] += pa_cos
                    cluster["pa_center"] = np.degrees(
                        np.arctan2(cluster["sum_sin"], cluster["sum_cos"])
                    ) % 360.0
            for cluster in clusters:
                datasets.append(Dataset(cluster["frames"]))
    else:
        datasets, _ = input_dataset.split_dataset(prihdr_keywords=["TARGET"])

    stokes_vectors = []

    for dataset in datasets:
        fluxes, flux_errs, thetas = [], [], []
        # --- Photometry loop ---
        for ds in dataset:
            prism = ds.ext_hdr.get('DPAMNAME')
            if prism not in prism_map:
                raise ValueError(f"Unknown prism: {prism}")
            
            flux, flux_err = aper_phot_pol(ds, phot_kwargs)
            fluxes.append(flux)
            flux_errs.append(flux_err)
            
            for phi in prism_map[prism]:
                thetas.append(np.radians(phi))

        fluxes = np.array(fluxes)
        flux_errs = np.array(flux_errs)
        thetas = np.array(thetas)

        # Prevent division by zero
        if np.any(flux_errs == 0):
            flux_errs[flux_errs == 0] = np.min(flux_errs[flux_errs > 0])

        # --- Instrument coordinates: left - right ---
        n_images = len(dataset)
        fluxes = fluxes.reshape([n_images, 2])
        flux_errs = flux_errs.reshape([n_images, 2])
        thetas = thetas.reshape([n_images, 2])
  
        I_vals = np.sum(fluxes, axis=1)
        I_errs = np.sqrt(np.sum(flux_errs**2, axis=1))
        QU_vals = fluxes[:, 0] - fluxes[:, 1]
        QU_errs = I_errs
        if np.any(I_vals == 0):
            raise ValueError("Cannot normalize Stokes parameters with zero intensity")

        idx_0 = np.where(np.degrees(thetas[:, 0]) == 0)[0]
        idx_45 = np.where(np.degrees(thetas[:, 0]) == 45)[0]
        if idx_0.size == 0 or idx_45.size == 0:
            raise ValueError("Both POL0 and POL45 frames are required")

        # Preserve total intensity separately for each Wollaston setting.
        i_pol0_sum = np.sum(I_vals[idx_0])
        i_pol0_err = np.sqrt(np.sum(I_errs[idx_0]**2))
        i_pol45_sum = np.sum(I_vals[idx_45])
        i_pol45_err = np.sqrt(np.sum(I_errs[idx_45]**2))

        # Combine dithered fluxes in detector coordinates before forming Q/I and U/I.
        q_sum = np.sum(QU_vals[idx_0])
        q_sum_err = np.sqrt(np.sum(QU_errs[idx_0]**2))
        u_sum = np.sum(QU_vals[idx_45])
        u_sum_err = np.sqrt(np.sum(QU_errs[idx_45]**2))
        Q_frac = q_sum / i_pol0_sum
        U_frac = u_sum / i_pol45_sum
        Q_frac_err = np.sqrt(
            (q_sum_err / i_pol0_sum)**2
            + (q_sum * i_pol0_err / i_pol0_sum**2)**2
        )
        U_frac_err = np.sqrt(
            (u_sum_err / i_pol45_sum)**2
            + (u_sum * i_pol45_err / i_pol45_sum**2)**2
        )

        # Keep the first four entries compatible with calc_stokes_unocculted.
        # A common I is undefined without registering the dithered frames.
        data_out = np.array([np.nan, Q_frac, U_frac, 0., i_pol0_sum, i_pol45_sum])
        err_out = np.array([np.inf, Q_frac_err, U_frac_err, np.inf,
                            i_pol0_err, i_pol45_err])
        dq_out = np.zeros_like(data_out, dtype=int)

        # --- Headers ---
        pri_hdr = dataset[0].pri_hdr
        ext_hdr = dataset[0].ext_hdr
        ext_hdr.add_history(
            "Computed uncalibrated Stokes parameters: "
            "data=[I, Q/I, U/I, V, I_POL0, I_POL45]; I is undefined"
        )
        err_hdr = dataset[0].err_hdr
        dq_hdr = dataset[0].dq_hdr

        stokes_vector = Image(
            data_out,
            pri_hdr=pri_hdr,
            ext_hdr=ext_hdr,
            err=err_out,
            dq=dq_out,
            err_hdr=err_hdr,
            dq_hdr=dq_hdr
        )
        stokes_vector.filename = os.path.basename(dataset[0].filename).replace("l3", "stokes")

        stokes_vectors.append(stokes_vector)

    stokes_dataset = Dataset(stokes_vectors)

    return stokes_dataset

def generate_mueller_matrix_cal(input_dataset, 
                                path_to_pol_ref_file=None,
                                svd_threshold=1e-5):
    '''
    Calculates the Mueller Matrix calibration for a given dataset of polarimetric observations.
    The expected input is a dataset of stokes vectors measured from known polarized standard stars, separated by 
    target and PA_APER angle. The function reads in a polarization reference file containing the known polarization 
    properties of the targets, and uses these to calculate the Mueller Matrix elements via SVD inversion.
    When present, the trailing `I_POL0` and `I_POL45` entries are additionally used to
    measure the relative Q/U-to-I terms, M01/M00 and M02/M00, from their variation with
    roll angle.
    
    The pol reference file should contain the known polarization properties of the targets in the dataset.
    It should be a csv file with the following columns:
    TARGET, [CFAM,] P, P_err, PA, PA_err
    where TARGET is the name of the target, CFAM is the optional color-filter name, P is the degree of
    polarization in percent, P_err is the error in the degree of polarization in percent, PA is the
    polarization angle in degrees, and PA_err is the error in the polarization angle in degrees. When
    present, CFAM is matched to CFAMNAME in each input frame header; otherwise TARGET must be unique.

    The error calculation propagates both the photometric measurement noise on the observed Stokes vectors
    and the uncertainties in the reference star polarization fraction and angle (P_err, PA_err from the
    reference file), combined in quadrature.

    Args: 
        input_dataset (corgidrp.data.Dataset): A CorgiDRP dataset consisting of stokes vectors.
            This data should be either all ND datasets or all non-ND datasets.
        path_to_pol_ref_file (str): The path to the polarization reference file. 
            Default is "./data/stellar_polarization_database.csv".
        svd_threshold (float, optional): The threshold for singular values in the SVD inversion. Defaults to 1e-5 (semi-arbitrary).
    
    Returns:
        mueller_matrix_obj (MuellerMatrix or NDMuellerMatrix): The generated Mueller Matrix object.
    '''

    dataset = input_dataset.copy()

    if path_to_pol_ref_file is None:
        path_to_pol_ref_file = os.path.join(
            os.path.dirname(__file__),
            "data",
            "stellar_polarization_database.csv",
        )

    # check that all the data in the dataset is either ND or non-ND, by looking for ND in the FPAMNAME keyword
    nd_flags = [("ND" in data.ext_hdr["FPAMNAME"]) for data in dataset]
    if all(nd_flags):
        is_nd = True
    elif not any(nd_flags):
        is_nd = False
    else:
        raise ValueError("All datasets in the input dataset must be either ND or non-ND.")

    # Read in the polarization reference file
    pol_ref = pd.read_csv(path_to_pol_ref_file, skipinitialspace=True)
    pol_ref.columns = pol_ref.columns.str.strip()
    pol_ref["TARGET"] = pol_ref["TARGET"].str.strip()
    has_cfam = "CFAM" in pol_ref.columns
    if has_cfam:
        pol_ref["CFAM"] = pol_ref["CFAM"].str.strip()

    # split the datasets into different targets
    # Original behavior: this returns one target name per unique TARGET and drops roll states.
    # _, targets = dataset.split_dataset(prihdr_keywords=["TARGET"])
    # Keep one reference target per Stokes measurement, including each roll state.
    targets = [image.pri_hdr["TARGET"] for image in dataset]
    cfam_names = (
        [image.ext_hdr["CFAMNAME"] for image in dataset]
        if has_cfam else [None] * len(dataset)
    )

    n_targets = np.unique(targets).shape[0]
    # Select one polarization reference for every Stokes measurement.
    pol_rows_by_measurement = []
    for target, cfam_name in zip(targets, cfam_names):
        if has_cfam:
            pol_rows = pol_ref[
                (pol_ref["TARGET"] == target)
                & (pol_ref["CFAM"] == cfam_name)
            ]
            ref_label = f"TARGET={target}, CFAM={cfam_name}"
        else:
            pol_rows = pol_ref[pol_ref["TARGET"] == target]
            ref_label = f"TARGET={target}"
        if len(pol_rows) != 1:
            raise ValueError(
                f"Expected one polarization reference for {ref_label}; "
                f"found {len(pol_rows)}."
            )
        pol_rows_by_measurement.append(pol_rows)
    
    has_intensity_vectors = all(
        image.data.size >= 6 and image.err[0].size >= 6
        for image in dataset
    )
    # Measure normalized Q and U from the original Stokes-vector positions.
    stokes_vectors = []
    stokes_vector_errs = []
    if has_intensity_vectors:
        I_vectors = []
        I_vector_errs = []
    rotation_angles = []
    for image in dataset:
        stokes_vectors.append(image.data[1:3]) # Grab just Q and U
        stokes_vector_errs.append(image.err[0][1:3]) # Grab just Q and U errors
        if has_intensity_vectors:
            I_vectors.append(image.data[4:6]) # Grab trailing I_POL0 and I_POL45
            I_vector_errs.append(image.err[0][4:6])
        # PA_APER is the on-sky position angle east of north for the CGI aperture
        rotation_angles.append(image.pri_hdr["PA_APER"])
    stokes_vectors = np.append(stokes_vectors[0], stokes_vectors[1:])
    stokes_vector_errs = np.append(stokes_vector_errs[0], stokes_vector_errs[1:])
    if has_intensity_vectors:
        I_vectors = np.asarray(I_vectors)
        I_vector_errs = np.asarray(I_vector_errs)

    # generate the matrix of meausurements six columns [1 q_star, u_star, 0,0,0] for q_measured
    # and [0,0,0, 1, q_star, u_star] for u_measured. Transform Q/U from sky to detector frame using PA_APER.
    Q_ref_err_sq = np.zeros(len(targets))
    U_ref_err_sq = np.zeros(len(targets))
    cov_QU_ref = np.zeros(len(targets))
    Q_refs = np.zeros(len(targets))
    U_refs = np.zeros(len(targets))
    stokes_matrix = np.zeros((2*len(dataset), 6))
    for i, target in enumerate(targets):
        pol_row = pol_rows_by_measurement[i]
        P = pol_row["P"].values[0] / 100.0 # convert from percent to fraction
        PA = pol_row["PA"].values[0] - rotation_angles[i] # in degrees
        P_err = pol_row["P_err"].values[0] / 100.0 # convert from percent to fraction
        PA_err_rad = pol_row["PA_err"].values[0] * np.pi / 180.0 # convert deg to rad
        PA_rad = np.radians(PA)

        # calculate the Stokes parameters Q and U from P and PA
        Q, U = get_qu_from_p_theta(P, PA)
        Q_refs[i] = Q
        U_refs[i] = U

        stokes_matrix[2*i,:] = [1, Q, U, 0, 0, 0]
        stokes_matrix[2*i+1,:] = [0, 0, 0, 1, Q, U]

        # Store reference star polarization uncertainties for post-SVD error propagation.
        # Q = P*cos(2*PA), U = P*sin(2*PA), so by first-order error propagation:
        #   Var(Q) = (dQ/dP)^2 * sigma_P^2 + (dQ/dPA)^2 * sigma_PA^2
        #          = cos^2(2*PA) * sigma_P^2 + (2*P*sin(2*PA))^2 * sigma_PA^2
        #   Var(U) = sin^2(2*PA) * sigma_P^2 + (2*P*cos(2*PA))^2 * sigma_PA^2
        # Q and U from the same star are correlated through shared P and PA:
        #   Cov(Q,U) = (dQ/dP)(dU/dP) * sigma_P^2 + (dQ/dPA)(dU/dPA) * sigma_PA^2
        #            = cos(2*PA)*sin(2*PA) * (sigma_P^2 - 4*P^2*sigma_PA^2)
        Q_ref_err_sq[i] = (np.cos(2*PA_rad) * P_err)**2 + (2 * P * np.sin(2*PA_rad) * PA_err_rad)**2
        U_ref_err_sq[i] = (np.sin(2*PA_rad) * P_err)**2 + (2 * P * np.cos(2*PA_rad) * PA_err_rad)**2
        cov_QU_ref[i] = np.cos(2*PA_rad) * np.sin(2*PA_rad) * (P_err**2 - 4 * P**2 * PA_err_rad**2)

    # invert the stokes matrix using SVD and multiply the the normalized differences to get the mueller matrix elements
    u,s,v=np.linalg.svd(stokes_matrix)
    #SVD of non-square matrices needs array re-shaping
    rank = s.size
    u = u[:, :rank]
    v = v[:rank, :]
    # limit the singular values to improve the conditioning of the inversion
    s[s < svd_threshold] = svd_threshold
    stokes_matrix_inv=np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))
    mueller_elements = np.dot(stokes_matrix_inv, np.array(stokes_vectors))
    mueller_elements_covar = np.matmul(stokes_matrix_inv,stokes_matrix_inv.T)
    mueller_elements_covar[mueller_elements_covar <0] = 0

    # Propagate reference star pol uncertainties through design matrix A.
    # Note: this must run after mueller_elements is computed above, since ref_var depends on m.
    # The model is m = A^+ b, where A is built from reference Q,U and b is observed Q,U.
    # For a perturbation dA, the first-order shift is dm = -A^+(dA)m (Golub & Van Loan).
    # A perturbation dQ_ref on star i affects A[2i,1] and A[2i+1,4] (both equal Q_ref),
    # so the sensitivity of MM element k to dQ_ref is:
    #   dm_k/dQ_ref = -(A^+[k,2i]*m[1] + A^+[k,2i+1]*m[4])
    # and similarly for dU_ref (columns 2 and 5, elements m[2] and m[5]).
    # The variance contribution is then:
    #   Var(m_k) += c_Q^2 * Var(Q_ref) + 2*c_Q*c_U * Cov(Q_ref,U_ref) + c_U^2 * Var(U_ref)
    # where c_Q = A^+[k,2i]*m[1] + A^+[k,2i+1]*m[4], c_U = A^+[k,2i]*m[2] + A^+[k,2i+1]*m[5].
    ref_var = np.zeros(6)
    for i in range(len(targets)):
        c_Q = (stokes_matrix_inv[:, 2*i]   * mueller_elements[1] +
               stokes_matrix_inv[:, 2*i+1] * mueller_elements[4])
        c_U = (stokes_matrix_inv[:, 2*i]   * mueller_elements[2] +
               stokes_matrix_inv[:, 2*i+1] * mueller_elements[5])
        ref_var += (c_Q**2 * Q_ref_err_sq[i]
                    + 2 * c_Q * c_U * cov_QU_ref[i]
                    + c_U**2 * U_ref_err_sq[i])

    # combine measurement noise and reference star uncertainty
    meas_var = np.diag(stokes_matrix_inv @ np.diag(stokes_vector_errs**2) @ stokes_matrix_inv.T)
    mueller_elements_err = np.sqrt(meas_var + ref_var)

    # Fit M01/M00 and M02/M00 from intensity changes between roll states. Each
    # target and Wollaston setting has its own unknown source brightness, which
    # cancels when two rolls of that target/setting are compared.
    if has_intensity_vectors:
        I_rows = []
        I_obs = []
        I_obs_errs = []
        target_arr = np.asarray(targets)
        for wol_index in range(2):
            for target in np.unique(target_arr):
                target_idx = np.flatnonzero(target_arr == target)
                if len(target_idx) < 2:
                    continue

                ref_idx = target_idx[0]
                I_ref = I_vectors[ref_idx, wol_index]
                I_ref_err = I_vector_errs[ref_idx, wol_index]
                for curr_idx in target_idx[1:]:
                    I_curr = I_vectors[curr_idx, wol_index]
                    I_curr_err = I_vector_errs[curr_idx, wol_index]
                    I_sum = I_ref + I_curr
                    if I_sum == 0:
                        raise ValueError("Cannot fit Q/U-to-I terms with zero total intensity")

                    # I = source_flux * (M00 + M01*Q + M02*U). Dividing this
                    # two-roll difference by their intensity sum removes source_flux and M00.
                    I_rows.append([
                        (I_ref * Q_refs[curr_idx] - I_curr * Q_refs[ref_idx]) / I_sum,
                        (I_ref * U_refs[curr_idx] - I_curr * U_refs[ref_idx]) / I_sum,
                    ])
                    I_obs.append(
                        (I_curr - I_ref) / I_sum
                    )
                    I_obs_errs.append(np.sqrt(
                        (-2 * I_curr / I_sum**2 * I_ref_err)**2
                        + (2 * I_ref / I_sum**2 * I_curr_err)**2
                    ))

        if len(I_rows) < 2:
            raise ValueError(
                "At least two independent roll differences are required to fit Q/U-to-I terms"
            )

        I_matrix = np.asarray(I_rows)
        I_obs = np.asarray(I_obs)
        I_obs_errs = np.asarray(I_obs_errs)
        I_u, I_s, I_v = np.linalg.svd(I_matrix, full_matrices=False)
        if I_s[-1] < svd_threshold:
            raise ValueError("Insufficient roll and polarization diversity to fit Q/U-to-I terms")
        I_matrix_inv = np.dot(
            I_v.transpose(),
            np.dot(np.diag(I_s**-1), I_u.transpose()),
        )
        I_elements = np.dot(I_matrix_inv, I_obs)
        I_meas_var = np.diag(
            I_matrix_inv
            @ np.diag(I_obs_errs**2)
            @ I_matrix_inv.T
        )
        I_elements_err = np.sqrt(I_meas_var)

    # Fill in the Mueller matrix.
    mueller_matrix = np.eye(4)
    mueller_matrix[1:3, :3] = mueller_elements.reshape(2, 3)

    mueller_matrix_err = np.full((4, 4), np.nan)
    mueller_matrix_err[1:3, :3] = mueller_elements_err.reshape(2, 3)
    if has_intensity_vectors:
        mueller_matrix[0, 1:3] = I_elements
        mueller_matrix_err[0, 1:3] = I_elements_err

    if is_nd:
        mueller_matrix_obj = NDMuellerMatrix(mueller_matrix,pri_hdr=dataset[0].pri_hdr.copy(),
                         ext_hdr=dataset[0].ext_hdr.copy(), input_dataset=dataset,
                         err=mueller_matrix_err)
        cal_suffix = "_ndm_cal.fits"
    else:
        mueller_matrix_obj = MuellerMatrix(mueller_matrix,pri_hdr=dataset[0].pri_hdr.copy(),
                         ext_hdr=dataset[0].ext_hdr.copy(), input_dataset=dataset,
                         err=mueller_matrix_err)
        cal_suffix = "_mmx_cal.fits"

    # Convert the in-memory Stokes product name to the calibration product name.
    mueller_matrix_obj.filename = mueller_matrix_obj.filename.replace(
        "_stokes.fits", cal_suffix
    )
    mueller_matrix_obj.pri_hdr["FILENAME"] = mueller_matrix_obj.filename

    return mueller_matrix_obj

def get_qu_from_p_theta(p, theta):
    '''

    Convert either the degree of polarization and polarization angle to normalized Stokes q (Q/I) and u (U/I), 
    or the polarized intensity (P) and angle (PA) into Stokes Q and U intensities. 

    Convert polarization and angle into Stokes Q and U components.

    This function can operate in two distinct modes depending on the nature of p:

    1. Normalized Stokes:
       - p represents the degree of polarization (fractional, 0–1, not percent).
       - Returns the normalized Stokes parameters (q = Q/I, u = U/I), i.e. unitless ratios.

    2. Absolute mode:
       - p represents the polarized intensity P (same units as total intensity I).
       - Returns the absolute Stokes intensities Q and U (in the same units as p).

    Args:
        p (float): Either the fractional degree of polarization (0–1) or polarized intensity P. 
        theta (float): Polarization angle in degrees.

    Returns:
        tuple: (Q, U) Stokes parameters, either normalized (q,u) or absolute (Q,U) depending on the meaning of p.

    Example:
        >>> get_qu_from_p_theta(0.05, 30)   # 5% polarization (as fraction) 
        >>> get_qu_from_p_theta(5, 30)      # polarized intensity = 5 (arbitrary intensity units)

    '''
    Q = p * np.cos(2 * np.radians(theta))
    U = p * np.sin(2 * np.radians(theta))
    return Q, U

def measure_normalized_difference_L2b(input_pol_Image,
                                    image_center_x=512,image_center_y=512,
                                    separation_diameter_arcsec=7.5, alignment_angle=None,
                                    phot_kwargs=None):
    '''
    Measure the normalized difference for a single CorgiDRP pol Image.
    The normalized difference is defined as (I0 - I90) / (I0 + I90) for Q,
    and (I45 - I135) / (I45 + I135) for U.

    Args:
        input_pol_Image (CorgiDRP Image): A CorgiDRP Image object that has been processed through the pol pipeline.
            It should have the FPAMNAME keyword in the header to identify the polarization angle.
        image_center_x (int, optional): The x-coordinate of the image center. Defaults to 512.
        image_center_y (int, optional): The y-coordinate of the image center. Defaults to 512.
        separation_diameter_arcsec (float, optional): The separation in arcseconds between the center of the two FOVs. 
            Defaults to 7.5 arcseconds.
        alignment_angle (float, optional): The alignment angle of the Wollaston prism in degrees.
            This is used to determine which polarization state is being measured (e.g., 0 for POL0, 45 for POL45).
            If None, the function will attempt to determine the angle from the DPAMNAME keyword in the header.
        phot_kwargs (dict): A dictionary of keyword arguments to pass to the aperture photometry function.
            See the documentation for the fluxcal.calibrate_fluxcal_aper function for more details.

    Returns:
        normalized_difference (float): The measured normalized difference.
        error (float): The error in the normalized difference.
    '''

    pol_Image = input_pol_Image.copy()

    aper_flux_1, aper_flux_2 =  measure_aper_flux_pol(pol_Image, 
                                                      image_center_x = image_center_x, 
                                                        image_center_y = image_center_y,
                                                        separation_diameter_arcsec = separation_diameter_arcsec,
                                                        alignment_angle = alignment_angle,
                                                      phot_kwargs=phot_kwargs)   
    
    difference = aper_flux_1[0] - aper_flux_2[0]
    sum_ = aper_flux_1[0] + aper_flux_2[0]
    normalized_difference = difference / sum_
    sum_diff_err = (aper_flux_1[1]**2 + aper_flux_2[1]**2)**0.5

    #if F=A/B, then dF = F*sqrt((dA/A)^2 + (dB/B)^2)
    error = np.abs(normalized_difference)*np.sqrt(sum_diff_err**2/difference**2 + sum_diff_err**2/sum_**2)

    return normalized_difference, error 

def rotation_mueller_matrix(angle):
    '''

    constructs a rotation matrix from a given angle

    Args:
        angle (float): the angle of rotation in degrees
        
    Returns:
        rotation_matrix (np.array) The 4x4 mueller matrix for rotation at the given angle
    '''
    theta = angle * (np.pi / 180)
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(2*theta), np.sin(2*theta), 0],
        [0,-np.sin(2*theta), np.cos(2*theta), 0],
        [0, 0, 0, 1]
    ])
    return rotation_matrix

def lin_polarizer_mueller_matrix(angle):
    '''
    constructs a linear polarizer matrix from a given angle

    Args:
        angle (float): the polarization angle of the polarizer
        
    Returns:
        pol_matrix (np.array) The 4x4 mueller matrix for a linear polarizer at the given angle
    '''
    # convert degree to rad
    theta = angle * (np.pi / 180)
    cos = np.cos(2 * theta)
    sin = np.sin(2 * theta)
    pol_matrix = 0.5 * np.array([
        [1, cos, sin, 0],
        [cos, cos**2, cos * sin, 0],
        [sin, cos * sin, sin**2, 0],
        [0, 0, 0, 0]
    ])
    return pol_matrix
