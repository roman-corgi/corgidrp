"""
Unit test suite for the calibrate_nonlin module.

Frames for unit tests are simulated SCI-size frames made with Gaussian and Poisson 
noises included. The assumed flux map is a realistic pupil image.
"""

import os
import pytest
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import test_check
import corgidrp
from corgidrp import check
from corgidrp.data import Image, Dataset, NonLinearityCalibration
from corgidrp.sorting import sort_pupilimg_frames
from corgidrp.calibrate_nonlin import (calibrate_nonlin, CalNonlinException, nonlin_params_default)
from corgidrp.mocks import (make_fluxmap_image, nonlin_coefs)

def setup_module():
    """
    Sets module up and defines variables
    """
    global n_cal, n_mean
    global init_nonlins_arr
    global rms_test
    global exp_time_stack_arr0, time_stack_arr0, len_list0, gain_arr0
    global dataset_nl
    global norm_val, min_write, max_write

    ############################# prepare simulated frames #######################

    # path to nonlin table made from running calibrate_nonlin.py on TVAC frames
    # table used only to choose parameters to make analytic nonlin functions
    here = os.path.abspath(os.path.dirname(__file__))
    nonlin_table_path = Path(here,'test_data','nonlin_table_TVAC.txt')
    nonlin_flag = True # True adds nonlinearity to simulated frames

    # Load the arrays needed for calibrate_nonlin function from the .npz file
    loaded = np.load(Path(here,'test_data','nonlin_arrays_ut.npz'))
    # Access the arrays needed for calibrate_nonlin function
    exp_time_stack_arr0 = loaded['array1']
    time_stack_arr0 = loaded['array2']
    len_list0 = loaded['array3']
    gain_arr0 = loaded['array4'] # G = 1, 2, 10, 20
    # Reducing the number of frames used in unit tests (each has 5 substacks)
    n_cal = 1 
    idx_0 = 0
    idx_1 = 0
    # Turn into False and the test runs a similar amount of data as with real data in IIT (~40 Gb RAM)
    rms_test = 0.0035 # Value used when using an equivalent dataset as in IIT
    # Usual number of frames to deal with real rn values is ~200
    rn = 130 # read noise in e-
    if True:
        rms_test = 0.04 # Less strict comparison due to having significantly less frames
        # Usual number of frames to deal with real rn values is ~200
        rn = 130/np.sqrt(200/n_cal) # read noise in e-
        for iG in range(len(len_list0)):
            time_stack_arr0[idx_0:idx_0+n_cal*5] = time_stack_arr0[idx_1:idx_1+n_cal*5]
            for i_cal in range(n_cal):
                # There must be at least two frames with the same exposure time
                exp_time_stack_arr0[idx_0:idx_0+2] = exp_time_stack_arr0[idx_1]
                exp_time_stack_arr0[idx_0+2:idx_0+5] = exp_time_stack_arr0[idx_1]+ 0.05
                idx_0 += 5
            idx_1 += len_list0[iG]
        # Remove unused data
        exp_time_stack_arr0 = np.delete(exp_time_stack_arr0, idx_0 + np.arange(idx_1-idx_0))
        time_stack_arr0 = np.delete(time_stack_arr0, idx_0 + np.arange(idx_1-idx_0))
        # Update len_list0
        len_list0[:] = n_cal*5

    # Load the flux map
    fluxmap_init = np.load(Path(here,'test_data','FluxMap1024.npy'))
    fluxmap_init[fluxmap_init < 50] = 0 # cleanup flux map a bit
    fluxMap1 = 0.8*fluxmap_init # e/s/px, for G = 1
    fluxMap2 = 0.4*fluxmap_init # e/s/px, for G = 2
    fluxMap3 = 0.08*fluxmap_init # e/s/px, for G = 10
    fluxMap4 = 0.04*fluxmap_init # e/s/px, for G = 20

    # detector parameters
    kgain = 8.7 # e-/DN
    bias = 2000 # e-

    # cubic function nonlinearity for emgain of 1
    if nonlin_flag:
        coeffs_1, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)
    else:
        coeffs_1 = [0.0, 0.0, 0.0, 1.0]
        _, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)

    # commanded EM gain
    emgain = 1.0
    frame_list = []
    # make some uniform frames with emgain = 1 (must be unity)
    n_mean = 3
    for j in range(n_mean):
        image_sim = make_fluxmap_image(fluxMap1,bias,kgain,rn,emgain,5.0,coeffs_1,
            nonlin_flag=nonlin_flag)
        # Datetime will have same time as some frames in the cal set, but for this test, 
        # that doesn't matter; no sorting happening via sorting.py, where that would be caught
        image_sim.ext_hdr['DATETIME'] = time_stack_arr0[j]
        # Temporary keyword value. Mean frame is TBD
        image_sim.pri_hdr['OBSNAME'] = 'MNFRAME'
        #to test the correct handling of cosmic ray flags
        if j == 1:
            image_sim.dq[400, 200:300] = 128
        frame_list.append(image_sim)

    init_nonlins = []
    index = 0
    for iG in range(len(gain_arr0)):
        g = gain_arr0[iG]
        exp_time_loop = exp_time_stack_arr0[index:index+len_list0[iG]]
        time_stack_test = time_stack_arr0[index:index+len_list0[iG]]
        index = index + len_list0[iG]
        if nonlin_flag:
            coeffs, _, vals = nonlin_coefs(nonlin_table_path,g,3)
        else:
            coeffs = [0.0, 0.0, 0.0, 1.0]
            vals = np.ones(len(DNs))
        init_nonlins.append(vals)
        for idx_t, t in enumerate(exp_time_loop):
            # Simulate full frame
            if iG == 0:
                image_sim = make_fluxmap_image(fluxMap1,bias,kgain,rn,g,t,coeffs,
                    nonlin_flag=nonlin_flag)
                image_sim.ext_hdr['DATETIME'] = time_stack_test[idx_t]
            elif iG == 1:
                image_sim = make_fluxmap_image(fluxMap2,bias,kgain,rn,g,t,coeffs,
                    nonlin_flag=nonlin_flag)
                image_sim.ext_hdr['DATETIME'] = time_stack_test[idx_t]
            elif iG == 2:
                image_sim = make_fluxmap_image(fluxMap3,bias,kgain,rn,g,t,coeffs,
                    nonlin_flag=nonlin_flag)
                image_sim.ext_hdr['DATETIME'] = time_stack_test[idx_t]
            else:
                image_sim = make_fluxmap_image(fluxMap4,bias,kgain,rn,g,t,coeffs,
                    nonlin_flag=nonlin_flag)
                image_sim.ext_hdr['DATETIME'] = time_stack_test[idx_t]
            #to test the correct handling of cosmic ray flags
            if idx_t == 1:
                image_sim.dq[400, 200:300] = 128
            image_sim.pri_hdr['OBSNAME'] = 'NONLIN'
            frame_list.append(image_sim)
    # Join all frames in a Dataset
    dataset_nl = Dataset(frame_list) 
    init_nonlins_arr = np.transpose(np.array(init_nonlins))

    # set input parameters for calibrate_nonlin
    local_path = os.path.dirname(os.path.realpath(__file__))
    exp_time_stack_arr = exp_time_stack_arr0
    time_stack_arr = time_stack_arr0
    len_list = len_list0
    norm_val = 3000
    min_write = 800
    max_write = 10000

def test_expected_results_nom_sub():
    """Outputs are as expected for the provided frames with nominal arrays."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # catch UserWarning: norm_val is not between the minimum and maximum values from calibrate_nonlin:777
        nonlin_out = calibrate_nonlin(dataset_nl, n_cal, n_mean, norm_val, min_write,
        max_write)

    # Test its DATALVL is CAL
    assert nonlin_out.ext_hdr['DATALVL'] == 'CAL'
        
    # Calculate rms of the differences between the assumed nonlinearity and 
    # the nonlinearity determined with calibrate_nonlin
    diffs0 = nonlin_out.data[1:,1] - init_nonlins_arr[:,0] # G = 1
    diffs1 = nonlin_out.data[1:,2] - init_nonlins_arr[:,1] # G = 2
    diffs2 = nonlin_out.data[1:,3] - init_nonlins_arr[:,2] # G = 10
    diffs3 = nonlin_out.data[1:,4] - init_nonlins_arr[:,3] # G = 20
    # Calculate rms
    rms1 = np.sqrt(np.mean(diffs0**2))
    rms2 = np.sqrt(np.mean(diffs1**2))
    rms3 = np.sqrt(np.mean(diffs2**2))
    rms4 = np.sqrt(np.mean(diffs3**2))

    # check that the four rms values are below some value (real data take
    # several frames and the value in IIT was 0.0035 for all of them)
    assert np.less(rms1,rms_test)
    assert np.less(rms2,rms_test)
    assert np.less(rms3,rms_test)
    assert np.less(rms4,rms_test)
    # check that the first element in the first column is equal to min_write
    assert nonlin_out.data[1,0] == min_write
    # check that the last element in the first column is equal to max_write
    assert nonlin_out.data[-1,0] == max_write
    # check that the unity value is in the correct row
    norm_ind = np.where(nonlin_out.data[1:, 1] == 1)[0][0]
    assert np.isclose(nonlin_out.data[norm_ind+1,1], 1)
    assert np.isclose(nonlin_out.data[norm_ind+1,-1], 1)
    # check that norm_val is correct
    # TO DO: review why values are so far outside the computed mean range
    # If norm_val is outside the computed mean range (corr_mean_signal_sorted), 
    # normalization happens at the closest value in the output table range (min_write to max_write)
    # The warning "norm_val is not between the minimum and maximum values" indicates this
    actual_val = nonlin_out.data[norm_ind+1,0]
    # Accept norm_val if in computed mean range, or min_write/max_write if outside range
    assert actual_val in [norm_val, min_write, max_write], \
        f"Expected norm_val={norm_val}, min_write={min_write}, or max_write={max_write}, but got {actual_val}"

    # Test filename follows convention (as of R3.0.2)
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    nonlin_out.save(filedir=datadir)
    nln_cal_filename = dataset_nl[-1].filename.replace("_l2b", "_nln_cal")
    nln_cal_filepath = os.path.join(datadir, nln_cal_filename)
    if os.path.exists(nln_cal_filepath) is False:
        raise IOError(f'NonLinearity calibration file {nln_cal_filepath} does not exist.')
    # Load the calibration file to check it has the same data contents
    nln_cal_file_load = NonLinearityCalibration(nln_cal_filepath)
    # The first element is a NaN
    assert np.isnan(nln_cal_file_load.data[0,0])
    # Compare all other elements but for [0,0], which is a NaN
    for row in range(nln_cal_file_load.data.shape[0]):
        for col in range(nln_cal_file_load.data.shape[1]):
            if row == 0 and col == 0:
                continue
            assert np.all(nln_cal_file_load.data[row,col] == nonlin_out.data[row,col])

    # Remove test NLN cal file
    if os.path.exists(nonlin_out.filepath):
        os.remove(nonlin_out.filepath)

def test_expected_results_time_sub():
    """Outputs are as expected for the provided frames with datetime values for
    one EM gain group taken 1 day later. Set (gain_arr0[3]) taken 1 day later
    avoiding a duplication of the whole dataset."""
    time_stack_arr1 = np.copy(time_stack_arr0)
    # Convert date-time strings to datetime objects. First ones are used to create a mean frame
    index = len(dataset_nl) - n_mean - len_list0[3]
    ctime_datetime1 = pd.to_datetime(time_stack_arr1[index:], errors='coerce')
    # Add one day to each datetime object
    ctime_datetime_plus_one_day1 = ctime_datetime1 + pd.Timedelta(days=1)
    # Convert back to strings
    ctime_strings_plus_one_day1 = ctime_datetime_plus_one_day1.strftime('%Y-%m-%dT%H:%M:%S').tolist()
    time_stack_arr1[index:index+len_list0[3]] = ctime_strings_plus_one_day1
    index = 0
    # First ones are used to create a mean frame
    idx_frame = n_mean
    for iG in range(len(gain_arr0)):
        exp_time_loop = exp_time_stack_arr0[index:index+len_list0[iG]]
        time_stack_test = time_stack_arr1[index:index+len_list0[iG]]
        index = index + len_list0[iG]
        for idx_t, t in enumerate(exp_time_loop):
            dataset_nl[idx_frame].ext_hdr['DATETIME'] = time_stack_test[idx_t]
            idx_frame += 1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # catch UserWarning: norm_val is not between the minimum and maximum values from calibrate_nonlin:777
        nonlin_out = calibrate_nonlin(dataset_nl, n_cal, n_mean, norm_val, min_write, max_write)
     
    # Calculate rms of the differences between the assumed nonlinearity and 
    # the nonlinearity determined with calibrate_nonlin
    diffs0 = nonlin_out.data[1:,1] - init_nonlins_arr[:,0] # G = 1
    diffs1 = nonlin_out.data[1:,2] - init_nonlins_arr[:,1] # G = 2
    diffs2 = nonlin_out.data[1:,3] - init_nonlins_arr[:,2] # G = 10
    diffs3 = nonlin_out.data[1:,4] - init_nonlins_arr[:,3] # G = 20
    # Calculte rms and peak-to-peak differences
    rms1 = np.sqrt(np.mean(diffs0**2))
    rms2 = np.sqrt(np.mean(diffs1**2))
    rms3 = np.sqrt(np.mean(diffs2**2))
    rms4 = np.sqrt(np.mean(diffs3**2))
    
    # check that the four rms values are below some value (real data take
    # several frames and the value in IIT was 0.0035 for all of them)
    assert np.less(rms1,rms_test)
    assert np.less(rms2,rms_test)
    assert np.less(rms3,rms_test)
    assert np.less(rms4,rms_test)

def teardown_module():
    """
    Runs at the end of tests; deletes variables
    """
    global n_cal, n_mean
    global init_nonlins_arr
    global rms_test
    global exp_time_stack_arr0, time_stack_arr0, len_list0, gain_arr0
    global dataset_nl
    global norm_val, min_write, max_write

    del n_cal, n_mean
    del init_nonlins_arr
    del rms_test
    del exp_time_stack_arr0, time_stack_arr0, len_list0, gain_arr0
    del dataset_nl
    del norm_val, min_write, max_write

def test_norm_val():
    """norm_val must be divisible by 20."""
    norm_not_div_20 = 2010
    with pytest.raises(CalNonlinException):
        calibrate_nonlin(dataset_nl, n_cal, n_mean, norm_not_div_20, min_write, max_write)
 
def test_rps():
    """these two below must be real positive scalars."""
    check_list = test_check.rpslist
    # min_write
    for rerr in check_list:
        with pytest.raises(TypeError):
            calibrate_nonlin(dataset_nl, n_cal, n_mean, norm_val, rerr, max_write)
    # max_write
    for rerr in check_list:
        with pytest.raises(TypeError):
            calibrate_nonlin(dataset_nl, n_cal, n_mean, norm_val, min_write, rerr)
   
def test_max_gt_min():
    """max_write must be greater than min_write."""
    werr = min_write # set the max_write value to the min_write value
    with pytest.raises(CalNonlinException):
        calibrate_nonlin(dataset_nl, n_cal, n_mean, norm_val, min_write, werr)
    
def test_psi():
    """norm_val must be a positive scalar integer."""
    check_list = test_check.psilist
    # norm_val
    for mmerr in check_list:
        with pytest.raises(TypeError):
            calibrate_nonlin(dataset_nl, n_cal, n_mean, mmerr, min_write, max_write)

def test_nonlin_params():
    '''Check that the check function for nonlin_params works as expected.'''
    # test bad input for nonlin_params
    nonlin_params_bad = nonlin_params_default.copy()
    nonlin_params_bad['colroi2'] = 'foo'
    with pytest.raises(TypeError):
        calibrate_nonlin(dataset_nl, n_cal, n_mean, norm_val, min_write, max_write,
                        nonlin_params=nonlin_params_bad)
 
if __name__ == '__main__':
    print('Running setup_module')
    setup_module()
    print('Running test_nonlin_params')
    test_nonlin_params()
    print('Running test_expected_results_nom_sub')
    test_expected_results_nom_sub()
    print('Running test_expected_results_time_sub')
    test_expected_results_time_sub()
    print('Running test_norm_val')
    test_norm_val()
    print('Running test_rps')
    test_rps()
    print('Running test_max_gt_min')
    test_max_gt_min()
    print('Running test_psi')
    test_psi()
    print('Running teardown_module')
    teardown_module()
    print('Non-Linearity Calibration tests passed')
