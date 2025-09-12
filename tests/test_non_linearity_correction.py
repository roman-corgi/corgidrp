#A file to test the non-linearity correction, including a comparison with the II&T pipeline
import os
import glob
import pickle
import numpy as np
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.l1_to_l2a as l1_to_l2a
from scipy import interpolate
import pytest

class NonlinException(Exception):
    """Exception class for nonlin module."""

def _parse_file(nonlin_path):
    """

    ** This function has been copied directly from the II&T pipeline 
    and is only used here to validate the new DRP vs. the old pipeline **

    Get data from nonlin file.
    
    
    Args:
        nonlin_path (str): (Optional) Full path to the non-linearity calibrationf ile.

    Returns:
        tuple:
            gain_ax (numpy.array): The gain axis
            count_ax (numpy.array): The count axis
            relgains (numpy.array): The relative gains
    """
    # Read nonlin csv
    nonlin_raw = np.genfromtxt(nonlin_path, delimiter=',')

    # File format checks
    if nonlin_raw.ndim < 2 or nonlin_raw.shape[0] < 2 or \
       nonlin_raw.shape[1] < 2:
        raise NonlinException('Nonlin array must be at least 2x2 (room for x '
                              'and y axes and one data point)')
    if not np.isnan(nonlin_raw[0, 0]):
        raise NonlinException('First value of csv (upper left) must be set to '
                              '"nan"')

    # Column headers are gains, row headers are dn counts
    gain_ax = nonlin_raw[0, 1:]
    count_ax = nonlin_raw[1:, 0]
    # Array is relative gain values at a given dn count and gain
    relgains = nonlin_raw[1:, 1:]

    # Check for increasing axes
    if np.any(np.diff(gain_ax) <= 0):
        raise NonlinException('Gain axis (column headers) must be increasing')
    if np.any(np.diff(count_ax) <= 0):
        raise NonlinException('Counts axis (row headers) must be increasing')
    # Check that curves (data in columns) contain or straddle 1.0
    if (np.min(relgains, axis=0) > 1).any() or \
       (np.max(relgains, axis=0) < 1).any():
        raise NonlinException('Gain curves (array columns) must contain or '
                              'straddle a relative gain of 1.0')

    return gain_ax, count_ax, relgains


def get_relgains(frame, em_gain, nonlin_path):
    """

    ** This function has been copied directly from the II&T pipeline 
    and is only used here to validate the new DRP vs. the old pipeline 
    It is different from the function in detector.py**

    For a given bias subtracted frame of dn counts, return a same sized
    array of relative gain values.

    Args:
        frame (np.array): Array of dn count values.
        em_gain (float): Detector EM gain.
        nonlin_path (str): Full path of nonlinearity calibration csv.

    Returns:
        array_like : Array of relative gain values.
            

    Notes:
        This algorithm contains two interpolations:

         - A 2d interpolation to find the relative gain curve for a given EM gain
         - A 1d interpolation to find a relative gain value for each given dn
         count value.

        Both of these interpolations are linear, and both use their edge values as
        constant extrapolations for out of bounds values.
    """

    # Get file data
    gain_ax, count_ax, relgains = _parse_file(nonlin_path)

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

def test_non_linearity_correction():
    """
    Generate a non-linearity correction calibration and test the correction 
    
    Ported from II&T Pipeline
    """

    #Create a mock dataset because it is a required input when creating a NonLinearityCalibration
    dummy_dataset = mocks.create_prescan_files()

    # Make a non-linearity correction calibration file
    input_non_linearity_filename = "nonlin_table_TVAC.txt"
    input_non_linearity_path = os.path.join(os.path.dirname(__file__), "test_data", input_non_linearity_filename)
    test_non_linearity_filename = input_non_linearity_filename.split(".")[0] + ".fits"
    test_non_linearity_path = os.path.join(os.path.dirname(__file__), "test_data", test_non_linearity_filename)
    tvac_nonlin_data = np.genfromtxt(input_non_linearity_path, delimiter=",")


    pri_hdr, ext_hdr = mocks.create_default_L1_headers()
    non_linearity_correction = data.NonLinearityCalibration(tvac_nonlin_data,pri_hdr=pri_hdr,ext_hdr=ext_hdr,input_dataset = dummy_dataset)
    non_linearity_correction.save(filename = test_non_linearity_path)

    # check the nonlin can be pickled (for CTC operations)
    pickled = pickle.dumps(non_linearity_correction)
    pickled_nonlin = pickle.loads(pickled)
    assert np.all((non_linearity_correction.data == pickled_nonlin.data) | np.isnan(non_linearity_correction.data))

    # import IPython; IPython.embed()

    ###### create a simulated dataset that is non-linear
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    emgain = 2000
    mocks.create_nonlinear_dataset(test_non_linearity_path, filedir=datadir,em_gain=emgain)

    ####### open up the files
    sim_data_filenames = glob.glob(os.path.join(datadir, "simcal_nonlin*.fits"))
    nonlinear_dataset = data.Dataset(sim_data_filenames)
    assert len(nonlinear_dataset) == 2

    ######## perform non-linearity correction
    non_linearity_correction = data.NonLinearityCalibration(test_non_linearity_path)

    # check the nonlin can be pickled (for CTC operations)
    pickled = pickle.dumps(non_linearity_correction)
    pickled_nonlin = pickle.loads(pickled)
    assert np.all((non_linearity_correction.data == pickled_nonlin.data) | np.isnan(non_linearity_correction.data))

    # set up values for testing flags and the value of the flag
    non_linear_flag = 64
    non_linear_pixel_value = 1e6

    # seperate dataset to test flagging
    flagged_dataset = nonlinear_dataset.copy()

    # Inject a non-linear pixel and check that if it is flagged
    flagged_dataset.all_data[0,0,0] = non_linear_pixel_value

    flagged_dataset = l1_to_l2a.correct_nonlinearity(flagged_dataset, non_linearity_correction,threshold=non_linear_pixel_value-1)

    assert flagged_dataset.all_dq[0,0,0] >= non_linear_flag # flagged_dataset.all_data[0,0,0] should be flagged
    flagged_dataset.all_dq[0,0,0] = 0 # reset the flag

    assert np.all(flagged_dataset.all_dq < non_linear_flag) # all other pixels should not be flagged 

    linear_dataset = l1_to_l2a.correct_nonlinearity(nonlinear_dataset, non_linearity_correction)

    #The data was generated with a ramp in the x-direction going from 10 to 65536
    expected_ramp = np.linspace(800,65536,1024)
    #Let's collapse the data and see if there's a ramp. 
    collapsed_data = np.mean(linear_dataset.all_data, axis=(0,1))

    #Relative correction
    relative_correction = (collapsed_data-expected_ramp)/collapsed_data

    #We are happy if the relative correction is less than 1% [TBC]
    assert np.all(relative_correction < 1e-2)


    #Let's test that this returns the same thing as the II&T pipeline
    linear_data_iit = nonlinear_dataset.all_data*get_relgains(nonlinear_dataset.all_data,emgain,input_non_linearity_path)
    #We want the difference between the II&T version and ours to be zero. 
    assert np.all(np.abs(linear_dataset.all_data[0]-linear_data_iit[0]) < 1e-2)
    # assert np.mean(linear_dataset.all_data - linear_data_iit) == pytest.approx(0, abs=1e-2)

if __name__ == "__main__":
    test_non_linearity_correction()