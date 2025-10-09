import pandas as pd
from fluxcal import measure_aper_flux_pol
import numpy as np




def generate_mueller_matrix_cal(input_dataset, path_to_pol_ref_file="./data/stellar_polarization_database.csv"):
    '''
    A step function that generates a MuellerMatrix calibration file from a dataset.
    If the dataset is an ND dataset, then it generates an ND MuellerMatrix calibration file.

    The pol reference file should contain the known polarization properties of the targets in the dataset.
    It should be a csv file with the following columns:
    TARGET, P, P_err, PA_err
    where TARGET is the name of the target, P is the degree of polarization in percent, P_err is the error 
    in the degree of polarization in percent, and PA_err is the error in the polarization angle in degrees

    Args: 
        input_dataset (list): A list of CorgiDRP data objects that will be used to generate the Mueller Matrix.
            This should be a list of either all ND datasets or all non-ND datasets.
        path_to_pol_ref_file (str): The path to the polarization reference file. 
            Default is "./data/stellar_polarization_database.csv".
    Returns:
        mueller_matrix (MuellerMatrix or NDMuellerMatrix): The generated Mueller Matrix calibration file.
    '''

    dataset = input_dataset.copy()

    # check that all the data in the dataset is either ND or non-ND, by looking for ND in the FPAMNAME keyword
    nd_flags = [("ND" in data.header["FPAMNAME"]) for data in dataset]
    if all(nd_flags):
        is_nd = True
    elif not any(nd_flags):
        is_nd = False
    else:
        raise ValueError("All datasets in the input dataset must be either ND or non-ND.")

    # Read in the polarization reference file
    pol_ref = pd.read_csv(path_to_pol_ref_file)
    # extract the target names
    pol_ref_targets = pol_ref["TARGET"].tolist()

    # split the datasets into different targets
    datasets, targets = dataset.split_by_target(prihdr_keywords="TARGET")

    # check that all the targets from the dataset are in the pol reference file
    for target in targets:
        if target not in pol_ref_targets:
            raise ValueError(f"Target {target} not found in polarization reference file.")
    
    # measure the normalized difference for each dataset
    normalized_differences = []
    for image in dataset:
        


    # generate the mueller matrix from the stokes vectors and the known properties

    # propagate errors

    # create the mueller matrix object

    # return the mueller matrix object (nd or non-nd)
    pass


def measure_normalized_difference_L3(input_pol_Image,
                                    image_center_x=512,image_center_y=512,
                                    separation_diameter_arcsec=7.5, alignment_angle=None,
                                    flux_or_irr = 'flux',phot_kwargs=None):
    '''
    Measure the normalized difference for a single CorgiDRP pol Image.
    The normalized difference is defined as (I0 - I90) / (I0 + I90) for Q,
    and (I45 - I135) / (I45 + I135) for U.

    Args:
        pol_Image (CorgiDRP Image): A CorgiDRP Image object that has been processed through the pol pipeline.
            It should have the FPAMNAME keyword in the header to identify the polarization angle.
        phot_kwargs (dict): A dictionary of keyword arguments to pass to the aperture photometry function.
            See the documentation for the fluxcal.calibrate_fluxcal_aper function for more details.

    Returns:
        normalized_difference (float): The measured normalized difference.
        error (float): The error in the normalized difference.
    '''

    pol_Image = input_pol_Image.copy()

    # check that the image has the FPAMNAME keyword
    if "FPAMNAME" not in pol_Image.header:
        raise ValueError("The input image must have the FPAMNAME keyword in the header.")
    
    aper_flux_1, aper_flux_2 =  measure_aper_flux_pol(pol_Image, 
                                                      image_center_x = image_center_x, 
                                                        image_center_y = image_center_y,
                                                        separation_diameter_arcsec = separation_diameter_arcsec,
                                                        alignment_angle = alignment_angle,
                                                        flux_or_irr = flux_or_irr,
                                                      phot_kwargs=phot_kwargs)   
    
    difference = aper_flux_1[0] - aper_flux_2[0]
    sum_ = aper_flux_1[0] + aper_flux_2[0]
    normalized_difference = difference / sum_
    sum_diff_err = (aper_flux_1[1]**2 + aper_flux_2[1]**2)**0.5

    error = normalized_difference*np.sqrt(sum_diff_err**2/difference**2 + sum_diff_err**2/sum_**2)

    return normalized_difference, error 