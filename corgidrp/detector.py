import numpy as np
from scipy import interpolate
import corgidrp.data as data


def create_dark_calib(dark_dataset):
    """
    Turn this dataset of image frames that were taken to measure
    the dark current into a dark calibration frame

    Args:
        dark_dataset (corgidrp.data.Dataset): a dataset of Image frames (L2a-level)

    Returns:
        data.Dark: a dark calibration frame
    """
    combined_frame = np.nanmean(dark_dataset.all_data, axis=0)

    new_dark = data.Dark(combined_frame, pri_hdr=dark_dataset[0].pri_hdr.copy(),
                         ext_hdr=dark_dataset[0].ext_hdr.copy(), input_dataset=dark_dataset)

    return new_dark


def dark_subtraction(input_dataset, dark_frame):
    """
    Perform dark current subtraction of a dataset using the corresponding dark frame

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need dark subtraction (L2a-level)
        dark_frame (corgidrp.data.Dark): a Dark frame to model the dark current

    Returns:
        corgidrp.data.Dataset: a dark subtracted version of the input dataset
    """
    # you should make a copy the dataset to start
    darksub_dataset = input_dataset.copy()

    darksub_cube = darksub_dataset.all_data - dark_frame.data

    history_msg = "Dark current subtracted using dark {0}".format(dark_frame.filename)

    # update the output dataset with this new dark subtracted data and update the history
    darksub_dataset.update_after_processing_step(history_msg, new_all_data=darksub_cube)

    return darksub_dataset

def correct_nonlinearity(input_dataset, non_lin_correction):
    """
    Perform non-linearity correction of a dataset using the corresponding non-linearity correction

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need non-linearity correction (L2a-level)
        non_lin_correction (corgidrp.data.NonLinearityCorrection): a NonLinearityCorrection calibration file to model the non-linearity

    Returns:
        corgidrp.data.Dataset: a non-linearity corrected version of the input dataset
    """
    #Copy the dataset to start
    linearized_dataset = input_dataset.copy()

    #Apply the non-linearity correction to the data
    linearized_cube = linearized_dataset.all_data
    #Check to see if EM gain is in the header, if not, raise an error
    if "EMGAIN" not in linearized_dataset[0].ext_hdr.keys():
        raise ValueError("EM gain not found in header of input dataset. Non-linearity correction requires EM gain to be in header.")

    em_gain = linearized_dataset[0].ext_hdr["EMGAIN"] #NOTE THIS REQUIRES THAT THE EM GAIN IS MEASURED ALREADY

    for i in range(linearized_cube.shape[0]):
        linearized_cube[i] *= get_relgains(linearized_cube[i], em_gain, non_lin_correction)

    history_msg = "Data corrected for non-linearity with {0}".format(non_lin_correction.filename)

    linearized_cube.update_after_processing_step(history_msg, new_all_data=linearized_cube)

    return linearized_cube

def get_relgains(frame, em_gain, non_lin_correction):
    """
    For a given bias subtracted frame of dn counts, return a same sized
    array of relative gain values.

    This algorithm contains two interpolations:

    - A 2d interpolation to find the relative gain curve for a given EM gain
    - A 1d interpolation to find a relative gain value for each given dn
      count value.

    Both of these interpolations are linear, and both use their edge values as
    constant extrapolations for out of bounds values.

    Parameters:
        frame (array_like): Array of dn count values.
        em_gain (float): Detector EM gain.
        non_lin_correction (corgi.drp.NonLinearityCorrection): A NonLinearityCorrection calibration file.

    Returns:
        array_like: Array of relative gain values.
    """

    # Column headers are gains, row headers are dn counts
    gain_ax = non_lin_correction.data[0, 1:]
    count_ax = non_lin_correction.data[1:, 0]
    # Array is relative gain values at a given dn count and gain
    relgains = non_lin_correction.data[1:, 1:]

    #MMB Note: This check is maybe better placed in the code that is saving the non-linearity correction file? 
    # Check for increasing axes
    if np.any(np.diff(gain_ax) <= 0):
        raise ValueError('Gain axis (column headers) must be increasing')
    if np.any(np.diff(count_ax) <= 0):
        raise ValueError('Counts axis (row headers) must be increasing')
    # Check that curves (data in columns) contain or straddle 1.0
    if (np.min(relgains, axis=0) > 1).any() or \
       (np.max(relgains, axis=0) < 1).any():
        raise ValueError('Gain curves (array columns) must contain or '
                              'straddle a relative gain of 1.0')
    
    # Create interpolation for em gain (x), counts (y), and relative gain (z).
    # Note that this defaults to using the edge values as fill_value for
    # out of bounds values (same as specified below in interp1d)
    f = interpolate.RectBivariateSpline(gain_ax,
                                    count_ax,
                                    relgains.T,
                                    kx=1,
                                    ky=1,
    )
    # Get the relative gain curve for the given gain value
    relgain_curve = f(em_gain, count_ax)[0]

    # Create interpolation for dn counts (x) and relative gains (y). For
    # out of bounds values use edge values
    ff = interpolate.interp1d(count_ax, relgain_curve, kind='linear',
                              bounds_error=False,
                              fill_value=(relgain_curve[0], relgain_curve[-1]))
    # For each dn count, find the relative gain
    counts_flat = ff(frame.ravel())

    return counts_flat.reshape(frame.shape)