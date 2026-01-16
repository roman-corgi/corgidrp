# A file that holds the functions to handle polarimetry data 
import os
import numpy as np
import pandas as pd

import corgidrp.check as check
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
                           split_rolls=True):
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
        split_rolls (bool, optional):
            If True, split the input dataset by both target and roll angle. If False, split only by target.
            Default is True.

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

    #split datasets by target if there are multiple targets
    # targets = []
    if split_rolls:   
        datasets, _ = input_dataset.split_dataset(prihdr_keywords=["TARGET", "ROLL"])
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

def generate_mueller_matrix_cal(input_dataset, 
                                path_to_pol_ref_file=None,
                                svd_threshold=1e-5):
    '''
    Calculates the Mueller Matrix calibration for a given dataset of polarimetric observations.
    The expected input is a dataset of stokes vectors measured from known polarized standard stars, separated by 
    target and roll angle. The function reads in a polarization reference file containing the known polarization 
    properties of the targets, and uses these to calculate the Mueller Matrix elements via SVD inversion.
    
    The pol reference file should contain the known polarization properties of the targets in the dataset.
    It should be a csv file with the following columns:
    TARGET, P, P_err, PA_err
    where TARGET is the name of the target, P is the degree of polarization in percent, P_err is the error 
    in the degree of polarization in percent, and PA_err is the error in the polarization angle in degrees

    The current error calculation takes into consideration the errors on the normalized differences only, 
    and does not propagate the errors from the reference polarization values. This could be improved in future versions.

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
        path_to_pol_ref_file = os.path.join(os.path.dirname(__file__), "data", "stellar_polarization_database.csv")

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
    # extract the target names
    pol_ref_targets = pol_ref["TARGET"].tolist()

    # split the datasets into different targets
    _, targets = dataset.split_dataset(prihdr_keywords=["TARGET"])

    n_targets = np.unique(targets).shape[0]
    # check that all the targets from the dataset are in the pol reference file
    for target in targets:
        if target not in pol_ref_targets:
            raise ValueError(f"Target {target} not found in polarization reference file.")
    
    # measure the normalized difference for each dataset
    stokes_vectors = []
    stokes_vector_errs = []
    roll_angles = []
    for image in dataset:
        stokes_vectors.append(image.data[1:3]) #Grab just Q and U
        stokes_vector_errs.append(image.err[0][1:3]) #Grab just Q and U errors
        roll_angles.append(image.pri_hdr["ROLL"])
    stokes_vectors = np.append(stokes_vectors[0], stokes_vectors[1:])
    stokes_vector_errs = np.append(stokes_vector_errs[0], stokes_vector_errs[1:])

    # generate the matrix of meausurements six columns [1 q_star, u_star, 0,0,0] for q_measured
    # and [0,0,0, 1, q_star, u_star] for u_measured #Where Q and U have been rotated by the roll angle: 
    stokes_matrix = np.zeros((2*len(dataset), 6))
    for i, target in enumerate(targets):
        pol_row = pol_ref[pol_ref["TARGET"] == target]
        P = pol_row["P"].values[0] / 100.0 # convert from percent to fraction
        PA = pol_row["PA"].values[0] + roll_angles[i] # in degrees

        # calculate the Stokes parameters Q and U from P and PA
        Q, U = get_qu_from_p_theta(P, PA)

        stokes_matrix[2*i,:] = [1, Q, U, 0, 0, 0]
        stokes_matrix[2*i+1,:] = [0, 0, 0, 1, Q, U] 

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
    mueller_elements_err = np.diag(np.matmul(stokes_matrix_inv.T,stokes_matrix_inv)*(stokes_vector_errs**2))**0.5

    #Fill in the mueller matrix
    mueller_matrix = np.zeros((4,4))
    mueller_matrix[0,0] = 1
    mueller_matrix[1,0] = mueller_elements[0]
    mueller_matrix[1,1] = mueller_elements[1]
    mueller_matrix[1,2] = mueller_elements[2]
    mueller_matrix[2,0] = mueller_elements[3]
    mueller_matrix[2,1] = mueller_elements[4]
    mueller_matrix[2,2] = mueller_elements[5]
    mueller_matrix[3,3] = 1

    mueller_matrix_err = np.zeros((4,4))*np.nan
    mueller_matrix_err[1,0] = mueller_elements_err[0]
    mueller_matrix_err[1,1] = mueller_elements_err[1]
    mueller_matrix_err[1,2] = mueller_elements_err[2]
    mueller_matrix_err[2,0] = mueller_elements_err[3]
    mueller_matrix_err[2,1] = mueller_elements_err[4]
    mueller_matrix_err[2,2] = mueller_elements_err[5]

    # Merge headers for combined frame
    pri_hdr, ext_hdr, err_hdr, dq_hdr = check.merge_headers_for_combined_frame(dataset)
    
    if is_nd:
        mueller_matrix_obj = NDMuellerMatrix(mueller_matrix, pri_hdr=pri_hdr,
                         ext_hdr=ext_hdr, err_hdr=err_hdr, input_dataset=dataset)
    else:
        mueller_matrix_obj = MuellerMatrix(mueller_matrix, pri_hdr=pri_hdr,
                         ext_hdr=ext_hdr, err_hdr=err_hdr, input_dataset=dataset)

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
