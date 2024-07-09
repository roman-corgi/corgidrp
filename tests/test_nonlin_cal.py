"""
Unit test suite for the calibrate_nonlin module.

Frames for unit tests are simulated SCI-size frames made with Gaussian and Poisson 
noises included. The assumed flux map is a realistic pupil image.
"""

import os
import pandas as pd
import unittest
import warnings
import numpy as np
from astropy.io import fits
from pathlib import Path

import test_check
from corgidrp import check
from corgidrp.data import Image, Dataset
from corgidrp.mocks import (create_default_headers, make_fluxmap_frame)
from corgidrp.calibrate_nonlin import (calibrate_nonlin, CalNonlinException)

# function definitions
def nonlin_coefs(filename,EMgain,order):
    """ Reads TVAC nonlinearity table from location specified by ‘filename’.
    The column in the table closest to the ‘EMgain’ value is selected and fits
    a polynomial of order ‘order’. The coefficients of the fit are adjusted so
    that the polynomial function equals unity at 3000 DN. Outputs array polynomial
    coefficients, array of DN values from the TVAC table, and an array of the
    polynomial function values for all the DN values.
    """
    # filename is the name of the csv text file containing the TVAC nonlin table
    # EM gain selects the closest column in the table
    # Load the specified file
    bigArray = pd.read_csv(filename, header=None).values
    EMgains = bigArray[0, 1:]
    DNs = bigArray[1:, 0]
    
    # Find the closest EM gain available to what was requested
    iG = (np.abs(EMgains - EMgain)).argmin()
    
    # Fit the nonlinearity numbers to a polynomial
    vals = bigArray[1:, iG + 1]
    coeffs = np.polyfit(DNs, vals, order)
    
    # shift so that function passes through unity at 3000 DN for these tests
    fitVals0 = np.polyval(coeffs, DNs)
    ind = np.where(DNs == 3000)
    unity_val = fitVals0[ind][0]
    coeffs[3] = coeffs[3] - (unity_val-1.0)
    fitVals = np.polyval(coeffs,DNs)
    
    return coeffs, DNs, fitVals

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

# Load the flux map
hdul =  fits.open(Path(here,'test_data','FluxMap1024.fits'))
fluxmap_init = hdul[0].data
hdul.close()
fluxmap_init[fluxmap_init < 50] = 0 # cleanup flux map a bit
fluxMap1 = 0.8*fluxmap_init # e/s/px, for G = 1
fluxMap2 = 0.4*fluxmap_init # e/s/px, for G = 2
fluxMap3 = 0.08*fluxmap_init # e/s/px, for G = 10
fluxMap4 = 0.04*fluxmap_init # e/s/px, for G = 20

# detector parameters
kgain = 8.7 # e-/DN
rn = 130 # read noise in e-
bias = 2000 # e-

# cubic function nonlinearity for emgain of 1
if nonlin_flag:
    coeffs_1, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)
else:
    coeffs_1 = [0.0, 0.0, 0.0, 1.0]
    _, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)

emgain = 1.0
frame_list2 = []
# make 30 uniform frames with emgain = 1
for j in range(30):
    frame2 = make_fluxmap_frame(fluxMap1,bias,kgain,rn,emgain,5.0,coeffs_1,
        nonlin_flag=nonlin_flag)
    frame_list2.append(frame2)
stack_arr2 = np.stack(frame_list2)

init_nonlins = []
stack_list = []
index = 0
for iG in range(len(gain_arr0)):
    g = gain_arr0[iG]
    exp_time_loop = exp_time_stack_arr0[index:index+len_list0[iG]]
    time_stack_test = time_stack_arr0[index:index+len_list0[iG]]
    index = index + len_list0[iG]
    frame_list = [] # initialize frame stack
    if nonlin_flag:
        coeffs, _, vals = nonlin_coefs(nonlin_table_path,g,3)
    else:
        coeffs = [0.0, 0.0, 0.0, 1.0]
        vals = np.ones(len(DNs))
    init_nonlins.append(vals)
    for t in exp_time_loop:
        # Simulate full frame
        if iG == 0:
            frame_sim = make_fluxmap_frame(fluxMap1,bias,kgain,rn,g,t,coeffs,
                nonlin_flag=nonlin_flag)
        elif iG == 1:
            frame_sim = make_fluxmap_frame(fluxMap2,bias,kgain,rn,g,t,coeffs,
                nonlin_flag=nonlin_flag)
        elif iG == 2:
            frame_sim = make_fluxmap_frame(fluxMap3,bias,kgain,rn,g,t,coeffs,
                nonlin_flag=nonlin_flag)
        else:
            frame_sim = make_fluxmap_frame(fluxMap4,bias,kgain,rn,g,t,coeffs,
                nonlin_flag=nonlin_flag)
        frame_list.append(frame_sim)
    frame_stack = np.stack(frame_list)
    stack_list.append(frame_stack)
stack_arr = np.vstack(stack_list)
init_nonlins_arr = np.transpose(np.array(init_nonlins))

# prepare a second version of time_stack_arr that has all the frames for the 
# fourth set (gain_arr0[3]) taken 1 day later
time_stack_arr1 = np.copy(time_stack_arr0)
# Convert date-time strings to datetime objects
index = index - len_list0[3]
ctime_datetime1 = pd.to_datetime(time_stack_test, errors='coerce')
# Add one day to each datetime object
ctime_datetime_plus_one_day1 = ctime_datetime1 + pd.Timedelta(days=1)
# Convert back to strings
ctime_strings_plus_one_day1 = ctime_datetime_plus_one_day1.strftime('%Y-%m-%dT%H:%M:%S').tolist()
time_stack_arr1[index:index+len_list0[3]] = ctime_strings_plus_one_day1

# set input parameters for calibrate_nonlin
local_path = os.path.dirname(os.path.realpath(__file__))
norm_val = 3000
min_write = 800
max_write = 10000

class TestCalibrateNonlin(unittest.TestCase):
    """Unit tests for calibrate_nonlin method."""

    def setUp(self):

        self.exp_time_stack_arr = exp_time_stack_arr0
        self.time_stack_arr = time_stack_arr0
        self.len_list = len_list0
        self.actual_gain_arr = gain_arr0
        self.min_write = min_write
        self.max_write = max_write
        self.norm_val = norm_val

        # filter out expected warnings
        warnings.filterwarnings('ignore', category=UserWarning,
            module='nonlinearity.calibrate_nonlin')

    def test_expected_results_nom_sub(self):
        """Outputs are as expected for the provided frames with nominal arrays."""
        (headers, nonlin_arr, csv_lines, means_min_max) = calibrate_nonlin(stack_arr, 
                            self.exp_time_stack_arr, self.time_stack_arr, 
                            self.len_list, stack_arr2, self.actual_gain_arr, 
                            self.norm_val, self.min_write, self.max_write)
        
        # Calculate rms of the differences between the assumed nonlinearity and 
        # the nonlinearity determined with calibrate_nonlin
        diffs0 = nonlin_arr[:,1] - init_nonlins_arr[:,0] # G = 1
        diffs1 = nonlin_arr[:,2] - init_nonlins_arr[:,1] # G = 2
        diffs2 = nonlin_arr[:,3] - init_nonlins_arr[:,2] # G = 10
        diffs3 = nonlin_arr[:,4] - init_nonlins_arr[:,3] # G = 20
        # Calculate rms
        rms1 = np.sqrt(np.mean(diffs0**2))
        rms2 = np.sqrt(np.mean(diffs1**2))
        rms3 = np.sqrt(np.mean(diffs2**2))
        rms4 = np.sqrt(np.mean(diffs3**2))
        
        # check that the four rms values are below the max value
        self.assertTrue(np.less(rms1,0.0035))
        self.assertTrue(np.less(rms2,0.0035))
        self.assertTrue(np.less(rms3,0.0035))
        self.assertTrue(np.less(rms4,0.0035))
        # check that the first element in the first column is equal to min_write
        self.assertTrue(np.equal(nonlin_arr[0,0], self.min_write))
        # check that the last element in the first column is equal to max_write
        self.assertTrue(np.equal(nonlin_arr[-1,0], self.max_write))
        # check that the unity value is in the correct row
        norm_ind = np.where(nonlin_arr[:, 1] == 1)[0]
        self.assertTrue(np.equal(nonlin_arr[norm_ind,1], 1))
        self.assertTrue(np.equal(nonlin_arr[norm_ind,-1], 1))
        # check that norm_val is correct
        self.assertTrue(np.equal(nonlin_arr[norm_ind,0], self.norm_val))
        # check one of the header values
        self.assertTrue(np.equal(headers[1].astype(float), 1))
        self.assertTrue(np.equal(len(means_min_max),len(self.actual_gain_arr)))
        
    def test_expected_results_time_stack_sub(self):
        """Outputs are as expected for the provided frames with 
        time_stack_arr values for one EM gain group taken 1 day later."""
        (headers, nonlin_arr, csv_lines, means_min_max) = calibrate_nonlin(stack_arr, 
                            self.exp_time_stack_arr, time_stack_arr1, 
                            self.len_list, stack_arr2, self.actual_gain_arr, 
                            self.norm_val, self.min_write, self.max_write)
        
        # Calculate rms of the differences between the assumed nonlinearity and 
        # the nonlinearity determined with calibrate_nonlin
        diffs0 = nonlin_arr[:,1] - init_nonlins_arr[:,0] # G = 1
        diffs1 = nonlin_arr[:,2] - init_nonlins_arr[:,1] # G = 2
        diffs2 = nonlin_arr[:,3] - init_nonlins_arr[:,2] # G = 10
        diffs3 = nonlin_arr[:,4] - init_nonlins_arr[:,3] # G = 20
        # Calculte rms and peak-to-peak differences
        rms1 = np.sqrt(np.mean(diffs0**2))
        rms2 = np.sqrt(np.mean(diffs1**2))
        rms3 = np.sqrt(np.mean(diffs2**2))
        rms4 = np.sqrt(np.mean(diffs3**2))
        
        # check that the four rms values are below the max value
        self.assertTrue(np.less(rms1,0.0035))
        self.assertTrue(np.less(rms2,0.0035))
        self.assertTrue(np.less(rms3,0.0035))
        self.assertTrue(np.less(rms4,0.0035))
    
    def test_3D_1(self):
        """stack_arr must be 3-D."""
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr[0], self.exp_time_stack_arr, 
                                self.time_stack_arr, self.len_list, stack_arr2, 
                                self.actual_gain_arr, self.norm_val, self.min_write, 
                                self.max_write)

    def test_sub_stack_len_1(self):
        """Number of sub-stacks in stack_arr must '
                'equal the sum of the elements in len_list."""
        sum_len_list = np.sum(self.len_list)-1
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr[0:sum_len_list], self.exp_time_stack_arr, 
                                self.time_stack_arr, self.len_list, stack_arr2, 
                                self.actual_gain_arr, self.norm_val, self.min_write, 
                                self.max_write)

    def test_3D_2(self):
        """stack_arr2 must be 3-D."""
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr, self.exp_time_stack_arr, 
                                self.time_stack_arr, self.len_list, stack_arr2[0], 
                                self.actual_gain_arr, self.norm_val, self.min_write, 
                                self.max_write)
    
    def test_sub_stack2_len(self):
        """stack_arr2 should have at least 30 sub-stacks."""
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr, self.exp_time_stack_arr, 
                                self.time_stack_arr, self.len_list, stack_arr2[0:28], 
                                self.actual_gain_arr, self.norm_val, self.min_write, 
                                self.max_write)
    
    def test_exp_time_stack_arr(self):
        """exp_time_stack_arr must be a 1-D, real array."""
        for terr0 in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                calibrate_nonlin(stack_arr, terr0, 
                                    self.time_stack_arr, self.len_list, stack_arr2, 
                                    self.actual_gain_arr, self.norm_val, 
                                    self.min_write, self.max_write)
        for terr0 in ut_check.rarraylist:
            with self.assertRaises(TypeError):
                calibrate_nonlin(stack_arr, terr0, 
                                    self.time_stack_arr, self.len_list, stack_arr2, 
                                    self.actual_gain_arr, self.norm_val, 
                                    self.min_write, self.max_write)

    def test_exp_time_stack_arr_gt0(self):
        """exp_time_stack_arr elements must all be greater than 0."""
        #same length as exp_time_stack_arr, but contains 0
        exp_arr = np.arange(0,len(self.exp_time_stack_arr))
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr, exp_arr, 
                                self.time_stack_arr, self.len_list, stack_arr2, 
                                self.actual_gain_arr, self.norm_val, self.min_write, 
                                self.max_write)

    def test_exp_time_stack_arr_rept(self):
        """each substack of stack_arr must have a group of frames '
            'with a repeated exposure time."""
        # make an array with the number of elements equal to the length of 
        # exp_time_stack_arr but without a group of frames with repeated 
        # exposure time.
        no_repeat_arr = np.copy(self.exp_time_stack_arr)
        # replace the last 5 exposure times (repeated) in first subgroup of 
        # frames with different exp times
        no_repeat_arr[self.len_list[0]-5:self.len_list[0]] = \
            1 + no_repeat_arr[self.len_list[0]-10:self.len_list[0]-5]
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr, no_repeat_arr, 
                                self.time_stack_arr, self.len_list, stack_arr2, 
                                self.actual_gain_arr, self.norm_val, self.min_write, 
                                self.max_write)
        
    def test_unique_time_stack_arr(self):
        """All elements of time_stack_arr must be unique."""
        terr = self.time_stack_arr
        terr[1] = terr[0] # set second element equal to the first
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr, self.exp_time_stack_arr, 
                                terr, self.len_list, stack_arr2, 
                                self.actual_gain_arr, self.norm_val, self.min_write, 
                                self.max_write)
    
    def test_len_list(self):
        """len_list must have at least one element."""
        # make an empty list
        empty_list = []
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr, self.exp_time_stack_arr, 
                                self.time_stack_arr, empty_list, stack_arr2, 
                                self.actual_gain_arr, self.norm_val, self.min_write, 
                                self.max_write)
    
    def test_actual_gain_arr_len(self):
        """Length of actual_gain_arr must be equal to length of len_list."""
        # make array with fewer elements than len_list
        act_err1 = np.arange(1,len(self.len_list)-1)
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr, self.exp_time_stack_arr, 
                                self.time_stack_arr, self.len_list, stack_arr2, 
                                act_err1, self.norm_val, self.min_write, 
                                self.max_write)
    
    def test_actual_gain_arr_1(self):
        """Every element of actual_gain_arr must be >= 1."""
        # make array with an element less than 1
        act_err2 = np.arange(1,len(self.actual_gain_arr))
        act_err2[0] = 0.5
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr, self.exp_time_stack_arr, 
                                self.time_stack_arr, self.len_list, stack_arr2, 
                                act_err2, self.norm_val, self.min_write, 
                                self.max_write)
    
    def test_norm_val(self):
        """norm_val must be divisible by 20."""
        norm_not_div_20 = 2010
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr, self.exp_time_stack_arr, 
                                self.time_stack_arr, self.len_list, stack_arr2, 
                                self.actual_gain_arr, norm_not_div_20, self.min_write, 
                                self.max_write)
    
    def test_rps(self):
        """these two below must be real positive scalars."""
        check_list = ut_check.rpslist
        # min_write
        for rerr in check_list:
            with self.assertRaises(TypeError):
                calibrate_nonlin(stack_arr, self.exp_time_stack_arr, 
                                    self.time_stack_arr, self.len_list, stack_arr2, 
                                    self.actual_gain_arr, self.norm_val, rerr, 
                                    self.max_write)
        # max_write
        for rerr in check_list:
            with self.assertRaises(TypeError):
                calibrate_nonlin(stack_arr, self.exp_time_stack_arr, 
                                    self.time_stack_arr, self.len_list, stack_arr2, 
                                    self.actual_gain_arr, self.norm_val, 
                                    self.min_write, rerr)
    
    def test_max_gt_min(self):
        """max_write must be greater than min_write."""
        werr = self.min_write # set the max_write value to the min_write value
        with self.assertRaises(CalNonlinException):
            calibrate_nonlin(stack_arr, self.exp_time_stack_arr, 
                                self.time_stack_arr, self.len_list, stack_arr2, 
                                self.actual_gain_arr, self.norm_val, 
                                self.min_write, werr)
    
    def test_psi(self):
        """norm_val must be a positive scalar integer."""
        check_list = ut_check.psilist
        # norm_val
        for mmerr in check_list:
            with self.assertRaises(TypeError):
                calibrate_nonlin(stack_arr, self.exp_time_stack_arr, 
                                    self.time_stack_arr, self.len_list, stack_arr2, 
                                    self.actual_gain_arr, mmerr, 
                                    self.min_write, self.max_write)
    
if __name__ == '__main__':
    unittest.main()

