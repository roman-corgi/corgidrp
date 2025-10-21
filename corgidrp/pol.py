import pandas as pd
from corgidrp.fluxcal import measure_aper_flux_pol
import numpy as np
from corgidrp.data import NDMuellerMatrix, MuellerMatrix

def generate_mueller_matrix_cal(input_dataset, image_center_x=512, image_center_y=512, separation_diameter_arcsec=7.5, 
                                alignment_angles=[0,45], phot_kwargs=None,
                                path_to_pol_ref_file="./data/stellar_polarization_database.csv",
                                svd_threshold=1e-5):
    '''
    The pol reference file should contain the known polarization properties of the targets in the dataset.
    It should be a csv file with the following columns:
    TARGET, P, P_err, PA_err
    where TARGET is the name of the target, P is the degree of polarization in percent, P_err is the error 
    in the degree of polarization in percent, and PA_err is the error in the polarization angle in degrees

    The current error calculation takes into consideration the errors on the normalized differences only, 
    and does not propagate the errors from the reference polarization values. This could be improved in future versions.

    Args: 
        input_dataset (list): A list of CorgiDRP data objects that will be used to generate the Mueller Matrix.
            This should be a list of either all ND datasets or all non-ND datasets.
        image_center_x (int, optional): The x-coordinate of the image center. Defaults to 512.
        image_center_y (int, optional): The y-coordinate of the image center. Defaults to 512.
        separation_diameter_arcsec (float, optional): The separations in arcseconds between the center of the two FOVs. 
            Defaults to 7.5 arcseconds.
        alignment_angles (list, optional): A list of two angles in degrees that represent the alignment angles 
            of the Wollaston prism for the two polarization states (e.g., [0, 45] for POL0 and POL45). Defaults to [0, 45].
        phot_kwargs (dict, optional): A dictionary of keyword arguments to pass to the aperture photometry function.
            See the documentation for the fluxcal.calibrate_fluxcal_aper function for more details.
        path_to_pol_ref_file (str): The path to the polarization reference file. 
            Default is "./data/stellar_polarization_database.csv".
        svd_threshold (float, optional): The threshold for singular values in the SVD inversion. Defaults to 1e-5 (semi-arbitrary).
    
    Returns:
        mueller_matrix_obj (MuellerMatrix or NDMuellerMatrix): The generated Mueller Matrix object.
    '''

    dataset = input_dataset.copy()

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
    normalized_differences = []
    roll_angles = []
    for image in dataset:
        
        #check dpam for POL0 or POL45, pick angle accordingly, 
        if "POL0" in image.ext_hdr["DPAMNAME"]:
            alignment_angle = alignment_angles[0]
            wollaston = 'POL0'
        elif "POL45" in image.ext_hdr["DPAMNAME"]:
            alignment_angle = alignment_angles[1]
            wollaston = 'POL45' 
        else:
            raise ValueError("DPAMNAME keyword must contain either POL0 or POL45 to determine polarization angle.")
        norm_diff, norm_diff_err = measure_normalized_difference_L3(
            image,
            image_center_x=image_center_x,
            image_center_y=image_center_y,
            separation_diameter_arcsec=separation_diameter_arcsec,
            alignment_angle=alignment_angle,
            phot_kwargs=phot_kwargs,
        )
        normalized_differences.append((norm_diff, norm_diff_err))
        roll_angles.append(image.pri_hdr["ROLL"])

    # generate the matrix of meausurements six columns [1 Q, U, 0,0,0] for POL0
    # and [1,0,0, 1, Q, U] for POL45 #Where Q and U have been rotated by the roll angle: 
    stokes_matrix = np.zeros((len(dataset), 6))
    for i, (norm_diff, norm_diff_err) in enumerate(normalized_differences):
        target = dataset[i].pri_hdr["TARGET"]
        pol_row = pol_ref[pol_ref["TARGET"] == target]
        P = pol_row["P"].values[0] / 100.0 # convert from percent to fraction
        PA = pol_row["PA"].values[0] + roll_angles[i] # in degrees

        # calculate the Stokes parameters Q and U from P and PA
        Q, U = get_qu_from_p_theta(P, PA)

        if "POL0" in dataset[i].ext_hdr["DPAMNAME"]:
            stokes_matrix[i, :] = [1, Q, U, 0, 0, 0]
        elif "POL45" in dataset[i].ext_hdr["DPAMNAME"]:
            stokes_matrix[i, :] = [0, 0, 0, 1, Q, U]


    # invert the stokes matrix using SVD and multiply the the normalized differences to get the mueller matrix elements
    u,s,v=np.linalg.svd(stokes_matrix)
    # limit the singular values to improve the conditioning of the inversion
    s[s < svd_threshold] = svd_threshold
    stokes_matrix_inv=np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))
    mueller_elements = np.dot(stokes_matrix_inv, np.array(normalized_differences)[:,0])
    mueller_elements_covar = np.matmul(stokes_matrix_inv,stokes_matrix_inv.T)
    mueller_elements_covar[mueller_elements_covar <0] = 0
    mueller_elements_err = np.diag(np.matmul(stokes_matrix_inv,stokes_matrix_inv.T)*(np.array(normalized_differences)[:,1]**2))**0.5

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

    if is_nd:
        mueller_matrix_obj = NDMuellerMatrix(mueller_matrix,pri_hdr=dataset[0].pri_hdr.copy(),
                         ext_hdr=dataset[0].ext_hdr.copy(), input_dataset=dataset)
    else:
        mueller_matrix_obj = MuellerMatrix(mueller_matrix,pri_hdr=dataset[0].pri_hdr.copy(),
                         ext_hdr=dataset[0].ext_hdr.copy(), input_dataset=dataset)

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

def measure_normalized_difference_L3(input_pol_Image,
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
