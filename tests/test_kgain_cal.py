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

import test_check
from corgidrp import check
from corgidrp.data import Image, Dataset
from corgidrp.mocks import (create_default_headers, make_fluxmap_frame)
from corgidrp.calibrate_kgain import (calibrate_kgain, CalKgainException, kgain_params)

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
## DATA: make_fluxmap_frame
for j in range(30):
    frame2 = make_fluxmap_frame(fluxMap,bias,kgain_in,rn_in,emgain,7.0,coeffs_1,
        nonlin_flag=nonlin_flag)
    frame_list2.append(frame2)
# Remove singleton dimensions
stack_arr2 = np.squeeze(np.stack(frame_list2))

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
        frame_sim = make_fluxmap_frame(fluxMap,bias,kgain_in,rn_in,g,
                               exp_time,coeffs,nonlin_flag=nonlin_flag)
        frame_list.append(frame_sim)
    frame_stack = np.stack(frame_list)
    stack_list.append(frame_stack)
# Remove singleton dimensions 
stack_arr = np.squeeze(np.stack(stack_list))

# set input parameters for calibrate_kgain function
min_val = 800
max_val = 3000
binwidth = 68

################### define class for tests ######################

class TestCalibrateKgain(unittest.TestCase):
    """Unit tests for calibrate_kgain method."""
    
    # sort out paths
    local_path = os.path.dirname(os.path.realpath(__file__))

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
            self.min_val, self.max_val, self.binwidth)
        
        from corgidrp.detector import kgain_params as constants_config
        signal_bins_N = constants_config['signal_bins_N']
        # kgain - should be close to the assumed value
        self.assertTrue(np.isclose(round(kgain.value,1), kgain_in, atol=0.5))
        # read noises. these are not requirements, but nice to check
        self.assertTrue(np.isclose(round(read_noise_gauss,1), rn_in, atol=8))
        self.assertTrue(np.isclose(round(read_noise_stdev,1), rn_in, atol=8))
        # check that the ptc output is the correct size
        self.assertTrue(np.all(np.equal(ptc.shape, (signal_bins_N,2))))

    def test_4D(self):
        """stack_arr should be 4-D."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr[0], stack_arr2, self.emgain, 
                self.min_val, self.max_val, self.binwidth)

    def test_sub_stack_len(self):
        """stack_arr must have at least 10 sub-stacks."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr[0:8], stack_arr2, self.emgain, 
                self.min_val, self.max_val, self.binwidth)

    def test_sub_sub_stack_len(self):
        """Each sub-stack of stack_arr must have 5 sub-stacks."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr[:,0:3], stack_arr2, self.emgain, 
                self.min_val, self.max_val, self.binwidth)
    
    def test_3D(self):
        """stack_arr2 must be 3-D."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2[0], self.emgain, 
                self.min_val, self.max_val, self.binwidth)
    
    def test_sub_stack2_len(self):
        """stack_arr2 must have at least 30 sub-stacks."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2[0:28], self.emgain, 
                self.min_val, self.max_val, self.binwidth)

    def test_psi(self):
        """These three below must be positive scalar integers."""
        check_list = test_check.psilist
        # min_val
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                    perr, self.max_val, self.binwidth)

        # max_val
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                    self.min_val, perr, self.binwidth)

        # binwidth
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                    self.min_val, self.max_val, perr)
        
    def test_binwidth(self):
        """binwidth must be >= 10."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                self.min_val, self.max_val, 9)
    
    def test_rps(self):
        """emgain must be a real positive scalar."""
        check_list = test_check.rpslist
        # min_write
        for rerr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, rerr, 
                    self.min_val, self.max_val, self.binwidth)
    
    def test_emgain(self):
        """emgain must be >= 1."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2, 0.5, 
                self.min_val, self.max_val, self.binwidth)

if __name__ == '__main__':
    unittest.main()
