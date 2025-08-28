"""
Unit test suite for the calibrate_kgain module.

Frames for unit tests are simulated SCI-size frames made with Gaussian and Poisson 
noises included. The assumed flux map is a realistic pupil image made from 
TVAC frames.
"""

import os
import pytest
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import test_check
from corgidrp import check
from corgidrp.data import Image, Dataset
from corgidrp.mocks import (make_fluxmap_image, nonlin_coefs)
from corgidrp.calibrate_kgain import (calibrate_kgain, CalKgainException, kgain_params_default)



######################## function definitions ###############################

def count_contiguous_repeats(arr):
    """
    This function returns the count of the contiguous repeated exposure times.
    If input array is empty, returns empty array.

    Args:
        arr (np.array): 1d array of exposure times.

    Returns:
        counts (list): count of the contiguous repeated exposure times
    """

    if isinstance(arr, (np.ndarray, list)) and len(arr) == 0:
        return []
    counts = []
    current_count = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            current_count += 1
        else:
            counts.append(current_count)
            current_count = 1
    counts.append(current_count)  # append the count of the last element
    return counts

def setup_module():
    """
    Sets up module
    """
    global n_cal, n_mean, kgain_in
    global dataset_kg
    ############### make stacks with simulated frames ##################
    np.random.seed(8585)

    # path to nonlin table made from running calibrate_nonlin.py on TVAC frames
    # table used only to choose parameters to make analytic nonlin functions
    here = os.path.abspath(os.path.dirname(__file__))
    nonlin_table_path = Path(here,'test_data','nonlin_table_TVAC.txt')
    nonlin_flag = False # True adds nonlinearity to simulated frames

    # Load the arrays needed for calibrate_nonlin function from the .npz file
    loaded = np.load(Path(here,'test_data','nonlin_arrays_ut.npz'))
    # Access the arrays needed for calibrate_nonlin function
    exp_time_stack_arr0 = loaded['array1']
    time_stack_arr0 = loaded['array2']
    len_list0 = loaded['array3']
    # Reducing the number of frames used in unit tests (each has 5 substacks).
    # Set to False to run the same test as in the IIT pipeline
    rn_in = 130 # read noise in e-
    if True:
        n_cal = 3
        exp_time_stack_arr0 = np.delete(exp_time_stack_arr0, np.s_[n_cal*5:])
        time_stack_arr0 = np.delete(time_stack_arr0, np.s_[n_cal*5:])
        # Update len_list0
        len_list0[0] = n_cal*5
        # Usual number of frames to deal with real rn values is ~200
        rn_in = 130/np.sqrt(200/n_cal) # read noise in e-

    # Load the flux map
    fluxmap_init =  np.load(Path(here,'test_data','FluxMap1024.npy'))
    fluxmap_init[fluxmap_init < 50] = 0 # cleanup flux map a bit
    fluxMap = 0.8*fluxmap_init # e/s/px, for G = 1

    # assumed detector parameters
    kgain_in = 8.7 # e-/DN
    bias = 2000 # e-
    actual_gain = 1.0

    # cubic function nonlinearity for emgain of 1
    if nonlin_flag:
        coeffs_1, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)
    else:
        coeffs_1 = [0.0, 0.0, 0.0, 1.0]
        _, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)

    frame_list = []
    # make some uniform frames with emgain = 1 (must be unity) P.S. IIT would use ~30
    n_mean = 3
    for j in range(n_mean):
        image_sim = make_fluxmap_image(fluxMap,bias,kgain_in,rn_in, 1, 7.0,coeffs_1,
            nonlin_flag=nonlin_flag)
        # Datetime cannot be duplicated
        image_sim.ext_hdr['DATETIME'] = time_stack_arr0[j]
        # Temporary keyword value. Mean frame is TBD
        image_sim.pri_hdr['OBSNAME'] = 'MNFRAME'
        if j == 1:
            #to test the correct handling of cosmic ray flags
            image_sim.dq[500, 1200:1400] = 128
        frame_list.append(image_sim)

    index = 0
    iG = 0 # doing only the em gain = 1 case
    g = actual_gain # Note: Same value for all frames used to calibrate K-gain
    exp_time_loop = exp_time_stack_arr0[index:index+len_list0[iG]]
    index = index + len_list0[iG]
    if nonlin_flag:
        coeffs, _, vals = nonlin_coefs(nonlin_table_path,g,3)
    else:
        coeffs = [0.0, 0.0, 0.0, 1.0]
        vals = np.ones(len(DNs))

    exp_repeat_counts = count_contiguous_repeats(exp_time_loop)
    for j in range(len(exp_repeat_counts)):
        for t in range(exp_repeat_counts[j]):
            # Simulate full frame
            exp_time = exp_time_loop[t+j*exp_repeat_counts[j]]
            image_sim = make_fluxmap_image(fluxMap,bias,kgain_in,rn_in,g,
                                exp_time,coeffs,nonlin_flag=nonlin_flag,
                                divide_em=True)
            image_sim.ext_hdr['DATETIME'] = time_stack_arr0[t+j*exp_repeat_counts[j]]
            # OBSNAME has no KGAIN value, but NONLIN
            image_sim.pri_hdr['OBSNAME'] = 'NONLIN'
            #to test the correct handling of cosmic ray flags
            if t == 1:
                image_sim.dq[500, 200:300] = 128
            frame_list.append(image_sim)
    dataset_kg = Dataset(frame_list)


def teardown_module():
    """
    Run at end of tests. Deletes variables
    """
    global n_cal, n_mean, kgain_in
    global dataset_kg

    del n_cal, n_mean, kgain_in
    del dataset_kg

# set input parameters for calibrate_kgain function
min_val = 800
max_val = 3000
binwidth = 68

def test_expected_results_sub():
    """Outputs are as expected, for imported frames."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning) # dof <= 0, and invalid value in scalar div
        warnings.filterwarnings("ignore", category=UserWarning) #catch expected Number of sub-stacks in cal_list warning from calibrate_kgain, because in this test n_cal = 3
        kgain = calibrate_kgain(dataset_kg, n_cal, n_mean, min_val, max_val, binwidth)
        
    signal_bins_N = kgain_params_default['signal_bins_N']
    # kgain - should be close to the assumed value
    assert np.isclose(round(kgain.value,1), kgain_in, atol=0.5)
    assert np.all(np.equal(kgain.ptc.shape, (signal_bins_N,2)))

    # test bad input for kgain_params
    kgain_params_bad = kgain_params_default.copy()
    kgain_params_bad['colroi2'] = 'foo'
    with pytest.raises(TypeError):
        calibrate_kgain(dataset_kg, n_cal, n_mean, min_val, max_val, binwidth,
                        kgain_params=kgain_params_bad)

def test_psi():
    """These three below must be positive scalar integers."""
    check_list = test_check.psilist
    # min_val
    for perr in check_list:
        with pytest.raises(TypeError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning) #catch expected Number of sub-stacks in cal_list warning from calibrate_kgain, because in this test n_cal = 3
                calibrate_kgain(dataset_kg, n_cal, n_mean, perr, max_val, binwidth)
    # max_val
    for perr in check_list:
        with pytest.raises(TypeError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning) #catch expected Number of sub-stacks in cal_list warning from calibrate_kgain, because in this test n_cal = 3
                calibrate_kgain(dataset_kg, n_cal, n_mean, min_val, perr, binwidth)

    # binwidth
    for perr in check_list:
        with pytest.raises(TypeError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning) #catch expected Number of sub-stacks in cal_list warning from calibrate_kgain, because in this test n_cal = 3
                calibrate_kgain(dataset_kg, n_cal, n_mean, min_val, max_val, perr)
      
def test_binwidth():
    """binwidth must be >= 10."""
    with pytest.raises(CalKgainException):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning) #catch expected Number of sub-stacks in cal_list warning from calibrate_kgain
            calibrate_kgain(dataset_kg, n_cal, n_mean, min_val, max_val, 9)
 
if __name__ == '__main__':
    setup_module()
    print('Running test_expected_results_sub')
    test_expected_results_sub()
    print('Running test_psi')
    test_psi()
    print('Running test_binwidth')
    test_binwidth()
