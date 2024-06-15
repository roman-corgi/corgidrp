"""
Unit test suite for the calibrate_kgain module.

Frames for unit tests are simulated SCI-size frames made with Gaussian and Poisson 
noises included. The assumed flux map is a realistic pupil image made from 
TVAC frames.
"""

import os
import pandas as pd
from pathlib import Path
import unittest
import warnings
import numpy as np
from astropy.io import fits

from corgidrp.calibrate_kgain import (calibrate_kgain, CalKgainException, check)

######################## function definitions ###############################

def nonlin_coefs(filename,EMgain,order):
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

def nonlin_factor(coeffs,DN):
    # input ceoffs from nonlin_ceofs and a DN value and return the 
    # nonlinearity factor
    min_value = 800.0
    max_value = 10000.0
    f_nonlin = np.polyval(coeffs, DN)
    # Control values outside the min/max range
    f_nonlin = np.where(DN < min_value, np.polyval(coeffs, min_value), f_nonlin)
    f_nonlin = np.where(DN > max_value, np.polyval(coeffs, max_value), f_nonlin)
    
    return f_nonlin

def make_frame(f_map, bias, kgain, rn, emgain, time, coeffs, nonlin_flag):
    # makes a SCI-sized frame with simulated noise and a fluxmap
    # f_map is the fluxmap in e/s/px and is 1024x1024 pixels in size
    # rn is read noise in electrons
    # bias is in electrons
    # time is exposure time in sec
    # coeffs is the array of cubic polynomial coefficients from nonlin_coefs
    # if nonlin_flag is True, then nonlinearity is applied
    
    # Generate random values of rn in elecrons from a Gaussian distribution
    random_array = np.random.normal(0, rn, (1200, 2200)) # e-
    # Generate random values from fluxmap from a Poisson distribution
    Poiss_noise_arr = emgain*np.random.poisson(time*f_map) # e-
    signal_arr = np.zeros((1200,2200))
    start_row = 10
    start_col = 1100
    signal_arr[start_row:start_row + Poiss_noise_arr.shape[0], 
                start_col:start_col + Poiss_noise_arr.shape[1]] = Poiss_noise_arr
    temp = random_array + signal_arr # e-
    if nonlin_flag:
        temp2 = nonlin_factor(coeffs, signal_arr/kgain)
        frame = np.round((bias + random_array + signal_arr/temp2)/kgain) # DN
    else:    
        frame = np.round((bias+temp)/kgain) # DN
        
    return frame

def count_contiguous_repeats(arr):
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

############### make stacks with simulated frames ##################

# path to nonlin table made from running calibrate_nonlin.py on TVAC frames
# table used only to choose parameters to make analytic nonlin functions
here = os.path.abspath(os.path.dirname(__file__))
nonlin_table_path = Path(here,'test_data','nonlin_table_TVAC.txt')
nonlin_flag = False # True adds nonlinearity to simulated frames

# Load the arrays needed for calibrate_nonlin function from the .npz file
loaded = np.load(Path(here,'test_data','nonlin_arrays_ut.npz'))
# Access the arrays needed for calibrate_nonlin function
exp_time_stack_arr0 = loaded['array1']
len_list0 = loaded['array3']

# Load the flux map
hdul =  fits.open(Path(here,'test_data','FluxMap1024.fits'))
fluxmap_init = hdul[0].data
hdul.close()
fluxmap_init[fluxmap_init < 50] = 0 # cleanup flux map a bit
fluxMap = 0.8*fluxmap_init # e/s/px, for G = 1

# assumed detector parameters
kgain_in = 8.7 # e-/DN
rn_in = 130 # read noise in e-
bias = 2000 # e-
emgain = 1.0

# cubic function nonlinearity for emgain of 1
if nonlin_flag:
    coeffs_1, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)
else:
    coeffs_1 = [0.0, 0.0, 0.0, 1.0]
    _, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)

frame_list2 = []
# make 30 uniform frames with emgain = 1
for j in range(30):
    frame2 = make_frame(fluxMap,bias,kgain_in,rn_in,emgain,7.0,coeffs_1,nonlin_flag)
    frame_list2.append(frame2)
stack_arr2 = np.stack(frame_list2)

index = 0
iG = 0 # doing only the em gain = 1 case
g = emgain
exp_time_loop = exp_time_stack_arr0[index:index+len_list0[iG]]
index = index + len_list0[iG]
frame_list = [] # initialize frame stack
if nonlin_flag:
    coeffs, _, vals = nonlin_coefs(nonlin_table_path,g,3)
else:
    coeffs = [0.0, 0.0, 0.0, 1.0]
    vals = np.ones(len(DNs))

stack_list = []
exp_repeat_counts = count_contiguous_repeats(exp_time_loop)
for j in range(len(exp_repeat_counts)):
    frame_list = []
    for t in range(exp_repeat_counts[j]):
        # Simulate full frame
        exp_time = exp_time_loop[t+j*exp_repeat_counts[j]]
        frame_sim = make_frame(fluxMap,bias,kgain_in,rn_in,g,
                               exp_time,coeffs,nonlin_flag)
        frame_list.append(frame_sim)
    frame_stack = np.stack(frame_list)
    stack_list.append(frame_stack)
stack_arr = np.stack(stack_list)

# set input parameters for calibrate_kgain function
min_val = 800
max_val = 3000
binwidth = 68

################### define class for tests ######################

class TestCalibrateKgain(unittest.TestCase):
    """Unit tests for calibrate_kgain method."""
    
    # sort out paths
    local_path = os.path.dirname(os.path.realpath(__file__))

    # example config file
    config_file = os.path.join(os.path.join(local_path, 'config_files'),
                               'kgain_parms.yaml')
    
    def setUp(self):

        self.emgain = emgain
        self.min_val = min_val
        self.max_val = max_val
        self.binwidth = binwidth

        # filter out expected warnings
        warnings.filterwarnings('ignore', category=UserWarning,
            module='kgain.calibrate_kgain')

    def test_expected_results_sub(self):
        """Outputs are as expected, for imported frames."""
        (kgain, read_noise_gauss, read_noise_stdev, ptc) = \
        calibrate_kgain(stack_arr, stack_arr2, self.emgain,
            self.min_val, self.max_val, self.binwidth, self.config_file)
        
        from corgidrp.detector import kgain_params as constants_config
        signal_bins_N = constants_config['signal_bins_N']
        # kgain - should be close to the assumed value
        self.assertTrue(np.isclose(round(kgain,1), kgain_in, atol=0.5))
        # read noises. these are not requirements, but nice to check
        self.assertTrue(np.isclose(round(read_noise_gauss,1), rn_in, atol=8))
        self.assertTrue(np.isclose(round(read_noise_stdev,1), rn_in, atol=8))
        # check that the ptc output is the correct size
        self.assertTrue(np.all(np.equal(ptc.shape, (signal_bins_N,2))))

    def test_ndarray(self):
        """stack_arr and stack_arr2 must be ndarrays."""
        array1 = np.array([1, 2, 3])  # Shape (3,)
        array2 = np.array([[4, 5, 6], [7, 8, 9]])  # Shape (2, 3)
        array3 = np.array([[10], [11]])  # Shape (2, 1)
        object_arr = np.array([array1, array2, array3], dtype=object)
        # stack_arr
        with self.assertRaises(CalKgainException):
            calibrate_kgain(object_arr, stack_arr2, self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)
        # stack_arr2
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, object_arr, self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)
    
    def test_4D(self):
        """stack_arr should be 4-D."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr[0], stack_arr2, self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)

    def test_sub_stack_len(self):
        """stack_arr must have at least 10 sub-stacks."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr[0:8], stack_arr2, self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)

    def test_sub_sub_stack_len(self):
        """Each sub-stack of stack_arr must have 5 sub-stacks."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr[:,0:3,:,:], stack_arr2, self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)
    
    def test_3D(self):
        """stack_arr2 must be 3-D."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2[0], self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)
    
    def test_sub_stack2_len(self):
        """stack_arr2 must have at least 30 sub-stacks."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2[0:28], self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)

    def test_psi(self):
        """These three below must be positive scalar integers."""
        check_list = ut_check.psilist
        # min_val
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                    perr, self.max_val, self.binwidth, self.config_file)

        # max_val
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                    self.min_val, perr, self.binwidth, self.config_file)

        # binwidth
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                    self.min_val, self.max_val, perr, self.config_file)
        
    def test_binwidth(self):
        """binwidth must be >= 10."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                self.min_val, self.max_val, 9, self.config_file)
    
    def test_rps(self):
        """emgain must be a real positive scalar."""
        check_list = ut_check.rpslist
        # min_write
        for rerr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, rerr, 
                    self.min_val, self.max_val, self.binwidth, self.config_file)
    
    def test_emgain(self):
        """emgain must be >= 1."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2, 0.5, 
                self.min_val, self.max_val, self.binwidth, self.config_file)

"""
Class to hold input-checking functions to minimize repetition
Note: This module, used by test_kgain.py and test_nonlin.py is included
here because at this moment it is not clear if the functions in the module are
of general utility for corgidrp
"""

"""Unit tests for check.py."""
class ut_check:

    import unittest
    
    import numpy as np
    
    from corgidrp.calibrate_kgain import check
    
    # Invalid values
    
    # string
    strlist = [1j, None, (1.,), [5, 5], -1, 0, 1.0]
    # real scalar
    rslist = [1j, None, (1.,), [5, 5], 'txt']
    # real nonnegative scalar
    rnslist = [1j, None, (1.,), [5, 5], 'txt', -1]
    # real positive scalar
    rpslist = [1j, None, (1.,), [5, 5], 'txt', -1, 0]
    # real scalar integer
    rsilist = [1j, None, (1.,), [5, 5], 'txt', 1.0]
    # nonnegative scalar integer
    nsilist = [1j, None, (1.,), [5, 5], 'txt', -1, 1.0]
    # positive scalar integer
    psilist = [1j, None, (1.,), [5, 5], 'txt', -1, 0, 1.0]
    # real array
    rarraylist = [1j*np.ones((5, 4)), (1+1j)*np.ones((5, 5, 5)), 'foo',
                  np.array([[1, 2], [3, 4], [5, 'a']])]
    # 1D array
    oneDlist = [np.ones((5, 4)), np.ones((5, 5, 5)), 'foo']
    # 2D array
    twoDlist = [np.ones((5,)), np.ones((5, 5, 5)), [], 'foo']
    # 2D square array
    twoDsquarelist = [np.ones((5,)), np.ones((5, 5, 5)), np.ones((5, 4)),
                      [], 'foo']
    # 3D array
    threeDlist = [np.ones((5,)), np.ones((5, 5)), np.ones((2, 2, 2, 2)), [], 'foo']
    
    
    class TestCheckException(Exception):
        pass
    
    
    class TestCheck(unittest.TestCase):
        """
        For each check, test with valid and invalid inputs for all three inputs.
    
        Test valid here as well since most other functions rely on these for
        error checking
        """
    
        # real_positive_scalar
        def test_real_positive_scalar_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: real positive scalar
            """
            try:
                check.real_positive_scalar(1, 'rps', TestCheckException)
            except check.CheckException:
                self.fail('real_positive_scalar failed on valid input')
            pass
    
        def test_real_positive_scalar_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: real positive scalar
            """
            for v0 in rpslist:
                with self.assertRaises(TestCheckException):
                    check.real_positive_scalar(v0, 'rps', TestCheckException)
                    pass
                pass
            pass
    
        def test_real_positive_scalar_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.real_positive_scalar(1, (1,), TestCheckException)
                pass
            pass
    
        def test_real_positive_scalar_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.real_positive_scalar(1, 'rps', 'TestCheckException')
                pass
            pass
    
        # real_nonnegative_scalar
        def test_real_nonnegative_scalar_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: real nonnegative scalar
            """
            try:
                check.real_nonnegative_scalar(0, 'rps', TestCheckException)
            except check.CheckException:
                self.fail('real_nonnegative_scalar failed on valid input')
            pass
    
        def test_real_nonnegative_scalar_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: real nonnegative scalar
            """
            for v0 in rnslist:
                with self.assertRaises(TestCheckException):
                    check.real_nonnegative_scalar(v0, 'rps', TestCheckException)
                    pass
                pass
            pass
    
        def test_real_nonnegative_scalar_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.real_nonnegative_scalar(0, (1,), TestCheckException)
                pass
            pass
    
        def test_real_nonnegative_scalar_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.real_nonnegative_scalar(0, 'rps', 'TestCheckException')
                pass
            pass
    
        # real_array
        def test_real_array_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: real array
            """
            try:
                check.real_array(np.ones((5, 5)), 'real', TestCheckException)
            except check.CheckException:
                self.fail('real_array failed on valid input')
            pass
    
        def test_real_array_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: real array
            """
            for v0 in rarraylist:
                with self.assertRaises(TestCheckException):
                    check.real_array(v0, '1D', TestCheckException)
                    pass
                pass
            pass
    
        def test_real_array_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.real_array(np.ones((5, 5)), (1,), TestCheckException)
                pass
            pass
    
        def test_real_array_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.real_array(np.ones((5, )), 'rps', 'TestCheckException')
                pass
            pass
    
        # oneD_array
        def test_oneD_array_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: 1D array
            """
            try:
                check.oneD_array(np.ones((5, )), '1D', TestCheckException)
            except check.CheckException:
                self.fail('oneD_array failed on valid input')
            pass
    
        def test_oneD_array_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: 1D array
            """
            for v0 in oneDlist:
                with self.assertRaises(TestCheckException):
                    check.oneD_array(v0, '1D', TestCheckException)
                    pass
                pass
            pass
    
        def test_oneD_array_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.oneD_array(np.ones((5, )), (1,), TestCheckException)
                pass
            pass
    
        def test_oneD_array_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.oneD_array(np.ones((5, )), 'rps', 'TestCheckException')
                pass
            pass
    
        # twoD_array
        def test_twoD_array_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: 2D array
            """
            try:
                check.twoD_array(np.ones((5, 5)), '2d', TestCheckException)
            except check.CheckException:
                self.fail('twoD_array failed on valid input')
            pass
    
        def test_twoD_array_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: 2D array
            """
            for v0 in twoDlist:
                with self.assertRaises(TestCheckException):
                    check.twoD_array(v0, '2d', TestCheckException)
                    pass
                pass
            pass
    
        def test_twoD_array_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.twoD_array(np.ones((5, 5)), (1,), TestCheckException)
                pass
            pass
    
        def test_twoD_array_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.twoD_array(np.ones((5, 5)), 'rps', 'TestCheckException')
                pass
            pass
    
        # twoD_square_array
        def test_twoD_square_array_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: 2D array
            """
            try:
                check.twoD_array(np.ones((5, 5)), '2d', TestCheckException)
            except check.CheckException:
                self.fail('twoD_square_array failed on valid input')
            pass
    
        def test_twoD_square_array_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: 2D array
            """
            for v0 in twoDsquarelist:
                with self.assertRaises(TestCheckException):
                    check.twoD_square_array(v0, '2d', TestCheckException)
                    pass
                pass
            pass
    
        def test_twoD_square_array_bad_var_shape(self):
            """
            Fail on invalid variable type.
    
            Type: 2D square array
            """
            for v0 in [np.ones((5, 4)), np.ones((4, 6))]:
                with self.assertRaises(TestCheckException):
                    check.twoD_square_array(v0, '2d', TestCheckException)
                    pass
                pass
            pass
    
        def test_twoD_square_array_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.twoD_square_array(np.ones((5, 5)), (1,), TestCheckException)
                pass
            pass
    
        def test_twoD_square_array_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.twoD_square_array(np.ones((5, 5)), 'rps',
                                        'TestCheckException')
                pass
            pass
    
        # threeD_array
        def test_threeD_array_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: 3D array
            """
            try:
                check.threeD_array(np.ones((5, 5, 2)), '3d', TestCheckException)
            except check.CheckException:
                self.fail('threeD_array failed on valid input')
            pass
    
        def test_threeD_array_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: 3D array
            """
            for v0 in threeDlist:
                with self.assertRaises(TestCheckException):
                    check.threeD_array(v0, '3d', TestCheckException)
                    pass
                pass
            pass
    
        def test_threeD_array_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.threeD_array(np.ones((5, 5, 2)), (1,), TestCheckException)
                pass
            pass
    
        def test_threeD_array_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.threeD_array(np.ones((5, 5, 2)), 'rps', 'TestCheckException')
                pass
            pass
    
        # real_scalar
        def test_real_scalar_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: real scalar
            """
            try:
                check.real_scalar(1, 'rs', TestCheckException)
            except check.CheckException:
                self.fail('real_scalar failed on valid input')
            pass
    
        def test_real_scalar_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: real scalar
            """
            for v0 in rslist:
                with self.assertRaises(TestCheckException):
                    check.real_scalar(v0, 'rs', TestCheckException)
                    pass
                pass
            pass
    
        def test_real_scalar_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.real_scalar(1, (1,), TestCheckException)
                pass
            pass
    
        def test_real_scalar_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.real_scalar(1, 'rs', 'TestCheckException')
                pass
            pass
    
        # positive_scalar_integer
        def test_positive_scalar_integer_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: positive scalar integer
            """
            try:
                check.positive_scalar_integer(1, 'psi', TestCheckException)
            except check.CheckException:
                self.fail('positive_scalar_integer failed on valid input')
            pass
    
        def test_positive_scalar_integer_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: positive scalar integer
            """
            for v0 in psilist:
                with self.assertRaises(TestCheckException):
                    check.positive_scalar_integer(v0, 'psi', TestCheckException)
                    pass
                pass
            pass
    
        def test_positive_scalar_integer_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.positive_scalar_integer(1, (1,), TestCheckException)
                pass
            pass
    
        def test_positive_scalar_integer_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.positive_scalar_integer(1, 'psi', 'TestCheckException')
                pass
            pass
    
        # nonnegative_scalar_integer
        def test_nonnegative_scalar_integer_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: nonnegative scalar integer
            """
            for j in [0, 1, 2]:
                try:
                    check.nonnegative_scalar_integer(j, 'nsi', TestCheckException)
                except check.CheckException:
                    self.fail('nonnegative_scalar_integer failed on valid input')
                pass
            pass
    
        def test_nonnegative_scalar_integer_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: nonnegative scalar integer
            """
            for v0 in nsilist:
                with self.assertRaises(TestCheckException):
                    check.nonnegative_scalar_integer(v0, 'nsi', TestCheckException)
                    pass
                pass
            pass
    
        def test_nonnegative_scalar_integer_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.nonnegative_scalar_integer(1, (1,), TestCheckException)
                pass
            pass
    
        def test_nonnegative_scalar_integer_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.nonnegative_scalar_integer(1, 'nsi', 'TestCheckException')
                pass
            pass
    
        # scalar_integer
        def test_scalar_integer_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: scalar integer
            """
            for j in [-2, -1, 0, 1, 2]:
                try:
                    check.scalar_integer(j, 'si', TestCheckException)
                except check.CheckException:
                    self.fail('scalar_integer failed on valid input')
                pass
            pass
    
        def test_scalar_integer_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: scalar integer
            """
            for v0 in rsilist:
                with self.assertRaises(TestCheckException):
                    check.scalar_integer(v0, 'si', TestCheckException)
                    pass
                pass
            pass
    
        def test_scalar_integer_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.scalar_integer(1, (1,), TestCheckException)
                pass
            pass
    
        def test_scalar_integer_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.scalar_integer(1, 'si', 'TestCheckException')
                pass
            pass
    
        def test_string_good(self):
            """
            Verify checker works correctly for valid input.
    
            Type: string
            """
            for j in ['a', '1', '.']:
                try:
                    check.string(j, 'string', TestCheckException)
                except check.CheckException:
                    self.fail('string failed on valid input')
                pass
            pass
    
        def test_string_bad_var(self):
            """
            Fail on invalid variable type.
    
            Type: string
            """
            for v0 in strlist:
                with self.assertRaises(TestCheckException):
                    check.string(v0, 'string', TestCheckException)
                    pass
                pass
            pass
    
        def test_string_bad_vname(self):
            """Fail on invalid input name for user output."""
            with self.assertRaises(check.CheckException):
                check.string('a', ('a',), TestCheckException)
                pass
            pass
    
        def test_string_bad_vexc(self):
            """Fail on input vexc not an Exception."""
            with self.assertRaises(check.CheckException):
                check.scalar_integer('a', 'string', 'TestCheckException')
                pass
            pass


if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()

