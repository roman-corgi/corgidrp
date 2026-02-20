"""Analysis of trap-pumped images for calibration.  Assumes trap-pumped
full frames as input.  Some of the trap-finding and parameter-fitting code
adapted from code from Nathan Bush, and his code was used for his paper, the
basis for this code:
Nathan Bush, David Hall, and Andrew Holland,
J. of Astronomical Telescopes, Instruments, and Systems, 7(1), 016003 (2021).
https://doi.org/10.1117/1.JATIS.7.1.016003
- Kevin Ludwick, UAH, 6/2022

- MMB Changes for corgi-drp: 
    - Removed everything meta and replaced with functions from detector.py
    - Replaced II&T get_relgains function with corgidrp function -> Now 
    requires a NonLinearityCorrection object to work. Defaults to None
    - Removed image slicing
    - Removed non-linearity correction
    - Input dataset is now a Dataset object from corgidrp.data
    - output is now a corgi.drp.TrapCalibration object
"""
import warnings
import numpy as np

from corgidrp.detector import *
from scipy.optimize import curve_fit

import corgidrp.check as check
from corgidrp.data import TrapCalibration, typical_bool_keywords, typical_cal_invalid_keywords

class TPumpAnException(Exception):
    """Exception class for tpumpanalysis."""

def P1(time_data, offset, tauc, tau, num_pumps=2000):
    """Probability function 1, one trap.

    Args:
        time_data (array): Phase times in seconds.
        offset (float): Offset in the fitting of data for amplitude vs phase time.
        tauc (float): Capture time constant. Units: e-.
        tau (float): Release time constant. Units: seconds.
        num_pumps (int): Number of cycles for trap pumping. Must be greater than 0.

    Returns:
        array: Expected amplitude values for the given phase times.
    """
    pc = 1 - np.exp(-time_data/tauc)
    return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-2*time_data/tau)))

def P1_P1(time_data, offset, tauc, tau, tauc2, tau2, num_pumps = 2000):
    """Probability function 1, two traps.

    Args:
        time_data (array): Phase times in seconds.
        offset (float): Offset in the fitting of data for amplitude vs phase time.
        tauc (float): Capture time constant for the first trap. Units: e-.
        tau (float): Release time constant for the first trap. Units: seconds.
        tauc2 (float): Capture time constant for the second trap. Units: e-.
        tau2 (float): Release time constant for the second trap. Units: seconds.
        num_pumps (int): Number of cycles for trap pumping. Must be greater than 0.

    Returns:
        array: Expected amplitude values for the given phase times.
    """
    pc = 1 - np.exp(-time_data/tauc)
    pc2 = 1 - np.exp(-time_data/tauc2)
    return offset+num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-2*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-2*time_data/tau2))


def P2(time_data, offset, tauc, tau, num_pumps = 2000):
    """Probability function 2, one trap.

    Args:
        time_data (array): Phase times in seconds.
        offset (float): Offset in the fitting of data for amplitude vs phase time.
        tauc (float): Capture time constant. Units: e-.
        tau (float): Release time constant. Units: seconds.
        num_pumps (int): Number of cycles for trap pumping. Must be greater than 0.

    Returns:
        array: Expected amplitude values for the given phase times.
    """
    pc = 1 - np.exp(-time_data/tauc)
    return offset+(num_pumps*pc*(np.exp(-2*time_data/tau)-
        np.exp(-3*time_data/tau)))

def P1_P2(time_data, offset, tauc, tau, tauc2, tau2, num_pumps = 2000):
    """One trap for probability function 1, one for probability function 2.

    Args:
        time_data (array): Phase times in seconds.
        offset (float): Offset in the fitting of data for amplitude vs phase time.
        tauc (float): Capture time constant for the first trap. Units: e-.
        tau (float): Release time constant for the first trap. Units: seconds.
        tauc2 (float): Capture time constant for the second trap. Units: e-.
        tau2 (float): Release time constant for the second trap. Units: seconds.
        num_pumps (int): Number of cycles for trap pumping. Must be greater than 0.

    Returns:
        array: Expected amplitude values for the given phase times.
    """
    pc = 1 - np.exp(-time_data/tauc)
    pc2 = 1 - np.exp(-time_data/tauc2)
    return offset+num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-2*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

def P2_P2(time_data, offset, tauc, tau, tauc2, tau2, num_pumps = 2000):
    """Probability function 2, two traps.

    Args:
        time_data (array): Phase times in seconds.
        offset (float): Offset in the fitting of data for amplitude vs phase time.
        tauc (float): Capture time constant for the first trap. Units: e-.
        tau (float): Release time constant for the first trap. Units: seconds.
        tauc2 (float): Capture time constant for the second trap. Units: e-.
        tau2 (float): Release time constant for the second trap. Units: seconds.
        num_pumps (int): Number of cycles for trap pumping. Must be greater than 0.

    Returns:
        array: Expected amplitude values for the given phase times.s
    """
    pc = 1 - np.exp(-time_data/tauc)
    pc2 = 1 - np.exp(-time_data/tauc2)
    return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
        np.exp(-3*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

def P3(time_data, offset, tauc, tau, num_pumps = 2000):
    """Probability function 3, one trap.

    Args:
        time_data (array): Phase times in seconds.
        offset (float): Offset in the fitting of data for amplitude vs phase time.
        tauc (float): Capture time constant. Units: e-.
        tau (float): Release time constant. Units: seconds.
        num_pumps (int): Number of cycles for trap pumping. Must be greater than 0.

    Returns:
        array: Expected amplitude values for the given phase times.
    """
    pc = 1 - np.exp(-time_data/tauc)
    return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-4*time_data/tau)))

def P3_P3(time_data, offset, tauc, tau, tauc2, tau2, num_pumps = 2000):
    """Probability function 3, two traps.

    Args:
        time_data (array): Phase times in seconds.
        offset (float): Offset in the fitting of data for amplitude vs phase time.
        tauc (float): Capture time constant for the first trap. Units: e-.
        tau (float): Release time constant for the first trap. Units: seconds.
        tauc2 (float): Capture time constant for the second trap. Units: e-.
        tau2 (float): Release time constant for the second trap. Units: seconds.
        num_pumps (int): Number of cycles for trap pumping. Must be greater than 0.

    Returns:
        array: Expected amplitude values for the given phase times.
    """
    pc = 1 - np.exp(-time_data/tauc)
    pc2 = 1 - np.exp(-time_data/tauc2)
    return offset+num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-4*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

def P2_P3(time_data, offset, tauc, tau, tauc2, tau2, num_pumps = 2000):
    """One trap for probability function 2, one for probability function 3.

    Args:
        time_data (array): Phase times in seconds.
        offset (float): Offset in the fitting of data for amplitude vs phase time.
        tauc (float): Capture time constant for the first trap. Units: e-.
        tau (float): Release time constant for the first trap. Units: seconds.
        tauc2 (float): Capture time constant for the second trap. Units: e-.
        tau2 (float): Release time constant for the second trap. Units: seconds.
        num_pumps (int): Number of cycles for trap pumping. Must be greater than 0.

    Returns:
        array: Expected amplitude values for the given phase times.
    """
    pc = 1 - np.exp(-time_data/tauc)
    pc2 = 1 - np.exp(-time_data/tauc2)
    return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
        np.exp(-3*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))
def illumination_correction(img, binsize, ill_corr):
    """Performs non-uniform illumination correction by taking sections of the
    image and performing local illumination subtraction.

    This function was copied and pasted from trip_id.py in the II&T pipeline. 
    
    Args:
        img (2-D array): Image to be corrected.
        binsize (int): Number of pixels over which to average for subtraction. If None, 
            acts as if ill_corr is False.
        ill_corr (bool): If True, subtracts the local median of the square region of 
            side length equal to binsize from each pixel. If False, simply subtracts 
            from each pixel the median of the whole image region.
    
    Returns:
        tuple: A tuple containing:
            corrected_img (2-D array): Corrected image.
            local_ill (2-D array): Frame with pixel values equal to the amount 
            that was subtracted from each.
    """

    # TODO v2:  use a sliding bin to correct (takes care of linear differences
    # b/w bins)

    # check inputs
    try:
        img = np.array(img).astype(float)
    except:
        raise TypeError("img elements should be real numbers")
    check.twoD_array(img, 'img', TypeError)
    if binsize is not None:
        check.positive_scalar_integer(binsize, 'binsize', TypeError)
    if type(ill_corr) != bool:
        raise TypeError('ill_corr must be True or False')

    if (not ill_corr) or (binsize is None):
        loc_ill = np.median(img)*np.ones_like(img).astype(float)
        corrected_img = img - loc_ill

    if ill_corr and (binsize is not None):
        # ensures there is a bin that runs all the way to the end
        if np.mod(len(img), binsize) == 0:
            row_bins = np.arange(0, len(img)+1, binsize)
        else:
            row_bins = np.arange(0, len(img), binsize)
        # If len(img) not divisible by binsize, this makes last bin of size
        # binsize + the remainder
        row_bins[-1] = len(img)

        # same thing for columns now
        if np.mod(len(img[0]), binsize) == 0:
            col_bins = np.arange(0, len(img[0])+1, binsize)
        else:
            col_bins = np.arange(0, len(img[0]), binsize)
        col_bins[-1] = len(img[0])

        # initializing
        loc_ill = (np.zeros([len(row_bins)-1, len(col_bins)-1])).astype(float)

        corrected_img = (np.zeros([len(img), len(img[0])])).astype(float)

        for i in range(len(row_bins)-1):
            for j in range(len(col_bins)-1):
                loc_ill[i,j]=np.median(img[int(row_bins[i]):int(row_bins[i+1]),
                                    int(col_bins[j]):int(col_bins[j+1])])
                corrected_img[int(row_bins[i]):int(row_bins[i+1]),
                    int(col_bins[j]):int(col_bins[j+1])] = \
                    img[int(row_bins[i]):int(row_bins[i+1]),
                    int(col_bins[j]):int(col_bins[j+1])] - loc_ill[i,j]

    #getting per pixel local_ill:
    local_ill = img - corrected_img

    return corrected_img, local_ill

def tau_temp(temp_data, E, cs):
    """Calculates the release time constant (tau) based on the input temperature, 
    energy level, and capture cross section for holes.

    This function was copied and pasted from the trip_fitting.py in the II&T pipeline.
    
    This function computes the release time constant (tau, in seconds) using the 
    input temperature data, energy level, and capture cross section for holes. 
    For more details, refer to the Appendix in "2020 Bush et al.pdf".
    
    Args:
        temp_data (float): Temperature in Kelvin.
        E (float): Energy level in electron volts (eV).
        cs (float): Capture cross section for holes, either in units of 
            1e-19 m^2 or 1e-15 cm^2.
    
    Returns:
        float: The release time constant (tau) in seconds.
    """

    k = 8.6173e-5 # eV/K
    kb = 1.381e-23 # mks units
    hconst = 6.626e-34 # mks units
    Eg = 1.1692 - (4.9e-4)*temp_data**2/(temp_data+655) #eV
    me = 9.109e-31 # kg, rest mass of electron
    mlstar = 0.1963 * me #kg
    mtstar = 0.1905 * 1.1692 * me / Eg #kg
    mstardc = 6**(2/3) * (mtstar**2*mlstar)**(1/3) #kg
    vth = np.sqrt(3*kb*temp_data/mstardc) # m/s
    Nc = 2*(2*np.pi*mstardc*kb*temp_data/(hconst**2))**1.5 # 1/m^3
    # added a factor of 1e-19 so that curve_fit step size reasonable
    return np.e**(E/(k*temp_data))/(cs*Nc*vth*1e-19)

def sig_tau_temp(temp_data, E, cs, sig_E, sig_cs):
    """Calculates the standard deviation of the release time constant via error propagation.
    
    This function computes the standard deviation of the release time constant (sig_tau, in seconds) 
    through error propagation, based on the input temperature, energy level, capture cross section 
    for holes, and their respective standard deviations. For more details, refer to the Appendix in 
    "2020 Bush et al.pdf".

    This function was copied and pasted from the trip_fitting.py in the II&T pipeline.
    
    Args:
        temp_data (float): Temperature in Kelvin.
        E (float): Energy level in electron volts (eV).
        cs (float): Capture cross section for holes, either in units of 
            1e-19 m^2 or 1e-15 cm^2.
        sig_E (float): Standard deviation of the energy level (eV).
        sig_cs (float): Standard deviation of the capture cross section for holes.
    
    Returns:
        float: The standard deviation of the release time constant (sig_tau) in seconds.
    """

    k = 8.6173e-5 # eV/K
    kb = 1.381e-23 # mks units
    hconst = 6.626e-34 # mks units
    Eg = 1.1692 - (4.9e-4)*temp_data**2/(temp_data+655)
    me = 9.109e-31 # kg, rest mass of electron
    mlstar = 0.1963 * me #kg
    mtstar = 0.1905 * 1.1692 * me / Eg #kg
    mstardc = 6**(2/3) * (mtstar**2*mlstar)**(1/3) #kg
    vth = np.sqrt(3*kb*temp_data/mstardc) # m/s
    Nc = 2*(2*np.pi*mstardc*kb*temp_data/(hconst**2))**1.5 # 1/m^3
    # added a factor of 1e-19 so that curve_fit step size reasonable
    tau = np.e**(E/(k*temp_data))/(cs*Nc*vth*1e-19)
    dtau_dcs = - tau/(cs*1e-19)
    dtau_dE = tau/(k*temp_data)
    sig_tau = np.sqrt((dtau_dcs*sig_cs*1e-19)**2 + (dtau_dE*sig_E)**2)
    return sig_tau

def trap_id(cor_img_stack, ill_corr_min, ill_corr_max, timings, thresh_factor,
            length_limit):
    """
    Identifies dipoles in an image stack based on threshold amplitude and categorizes them.
    
    This function analyzes a stack of trap-pumped images taken at different phase times 
    to identify dipoles by detecting adjacent pixels that exceed a threshold amplitude 
    above and below the mean. The threshold must be met at a sufficient number of phase 
    times. The function then categorizes the bright pixel of each dipole into one of three 
    categories: 'above', 'below', or 'both'.

    The function was copied and pasted from trap_id.py in the II&T pipeline.
    
    Args:
        cor_img_stack (3-D array): Stack of trap-pumped images taken at different phase 
            times. Each frame should have the same dimensions. Units can be in electrons 
            (e-) or digital numbers (DN), but they are input as electrons when this function 
            is called in `tpump_analysis()`.
        ill_corr_min (2-D array): Frame with pixel values equal to the minimum median 
            taken over phase times that was subtracted during `illumination_correction()`. 
            If `ill_corr` was False, the median was global over the whole image region. 
            If `ill_corr` was True, the median was over a local square of side length 
            `binsize` pixels in `illumination_correction()`.
        ill_corr_max (2-D array): Frame with pixel values equal to the maximum median 
            taken over phase times that was subtracted during `illumination_correction()`. 
            The conditions are the same as described for `ill_corr_min`.
        timings (array-like): Array of the phase times corresponding to the ordering of 
            the frames in `cor_img_stack`. Units are in seconds.
        thresh_factor (float): Number of standard deviations from the mean that a dipole 
            must exceed to be considered for a trap. If too high, dipoles with increasing 
            amplitude over time are identified, which is not characteristic of an actual trap. 
            If too low, the resulting dipoles may have amplitudes that are too noisy or low.
        length_limit (int): Minimum number of frames in which a dipole must meet the 
            threshold to be considered a true trap.
    
    Returns:
        rc_above (dict): A dictionary with keys for each bright pixel of an 'above' dipole, 
            formatted as:
            {
                (row, col): {
                    'amps_above': array([amp1, amp2, ...]),
                    'loc_med_min': float,
                    'loc_med_max': float
                },
                ...
            }
            'amps_above' is an array of amplitudes in the same order as the phase time 
            order in `timings`. 'loc_med_min' and 'loc_med_max' are the minimum and maximum 
            bias values over all phase times, respectively.
        
        rc_below (dict): A dictionary with keys for each bright pixel of a 'below' dipole, 
            formatted similarly to `rc_above`.
        
        rc_both (dict): A dictionary with keys for each bright pixel that is both an 'above' 
            and 'below' dipole, formatted as:
            {
                (row, col): {
                    'amps_both': array([amp1, amp2, ...]),
                    'loc_med_min': float,
                    'loc_med_max': float,
                    'above': {
                        'amp': array([amp1a, amp2a, ...]),
                        't': array([t1a, t2a, ...])
                    },
                    'below': {
                        'amp': array([amp1b, amp2b, ...]),
                        't': array([t1b, t2b, ...])
                    }
                },
                ...
            }
            'amps' and 't' under 'above' and 'below' are arrays identified specifically with 
            their respective dipoles. 'amps_both' contains all amplitudes for that pixel 
            in the same order as `timings`.
    """

    # Check inputs
    try:
        #checks whether elements are good and whether each frame has same dims
        cor_img_stack = np.stack(cor_img_stack).astype(float)
    except:
        raise TypeError("cor_img_stack elements should be real numbers, and "
        "each frame should have the same dimensions.")
        # and complex numbers can't be cast as float
    check.threeD_array(cor_img_stack, 'cor_img_stack', TypeError)
    try:
        ill_corr_min = np.array(ill_corr_min).astype(float)
    except:
        raise TypeError("ill_corr_min elements should be real numbers")
        # and complex numbers can't be cast as float
    check.twoD_array(ill_corr_min, 'ill_corr_min', TypeError)
    if np.shape(ill_corr_min) != np.shape(cor_img_stack[0]):
        raise ValueError('The dimensions of ill_corr_min should match '
        'those of each frame in cor_img_stack.')
    try:
        ill_corr_max = np.array(ill_corr_max).astype(float)
    except:
        raise TypeError("ill_corr_max elements should be real numbers")
        # and complex numbers can't be cast as float
    check.twoD_array(ill_corr_max, 'ill_corr_max', TypeError)
    if np.shape(ill_corr_max) != np.shape(cor_img_stack[0]):
        raise ValueError('The dimensions of ill_corr_max should match '
        'those of each frame in cor_img_stack.')
    try:
        timings = np.array(timings).astype(float)
    except:
        raise TypeError("timings elements should be real numbers")
    check.oneD_array(timings, 'timings', TypeError)
    if len(timings) != len(cor_img_stack):
        raise ValueError('timings must have length equal to number of frames '
        'in cor_img_stack')
    check.real_positive_scalar(thresh_factor, 'thresh_factor', TypeError)
    check.positive_scalar_integer(length_limit, 'length_limit', TypeError)
    if length_limit > len(cor_img_stack):
        raise ValueError('length_limit cannot be longer than the number of '
        'frames in cor_img_stack')
    # This also ensures that cor_img_stack is not 0 frames

    IS_DIPOLE_UPPER = []
    IS_DIPOLE_LOWER = []
    for frame in cor_img_stack:
        IS_DIPOLE_UPPER.append((frame > (np.median(frame) +
                np.std(frame)*thresh_factor)).astype(int))
        IS_DIPOLE_LOWER.append((frame < (np.median(frame) -
                np.std(frame)*thresh_factor)).astype(int))  
        # IS_DIPOLE_UPPER.append((frame > (np.percentile(frame, 100-thresh_factor))).astype(int))
        # IS_DIPOLE_LOWER.append((frame < (np.percentile(frame, thresh_factor))).astype(int))
    IS_DIPOLE_UPPER = np.stack(IS_DIPOLE_UPPER)
    IS_DIPOLE_LOWER = np.stack(IS_DIPOLE_LOWER)

    # Dipole_above means the mean bright pixel of dipole is above
    Dipole_above = IS_DIPOLE_UPPER + np.roll(IS_DIPOLE_LOWER, -1, axis=1)
    Dipole_below = IS_DIPOLE_UPPER + np.roll(IS_DIPOLE_LOWER, 1, axis=1)
    # assign edge rows as 0 to eliminate false positive from "rolling"
    Dipole_above[:,-1,:] = 0
    Dipole_below[:,0,:] = 0

    # will have a 2 if both criteria met.  Turn that into a 1 for every dipole.
    # must be float to allow for fractional values in next step
    dipoles_above_count = (Dipole_above > 1).astype(float)
    dipoles_below_count = (Dipole_below > 1).astype(float)

    for t in range(len(timings)):
        #count how many frames there are for a given phase time
        # timings and dipoles_*_count have same length
        # divide by number of counts for each non-unique phase time
        pt_count = list(timings).count(list(timings)[t])
        dipoles_above_count[t] = dipoles_above_count[t]/pt_count
        dipoles_below_count[t] = dipoles_below_count[t]/pt_count

    # if at least one of the phase times with multiple frames has a dipole
    # in a given pixel, then ceil will count that phase time as a '1' in sum
    ALL_DIPOLES_sum_above = np.ceil(np.sum(dipoles_above_count, axis = 0))
    ALL_DIPOLES_sum_below = np.ceil(np.sum(dipoles_below_count, axis = 0))

    # there should be dipoles present in the locations for enough phase times
    dipoles_above_thresh = (ALL_DIPOLES_sum_above >= length_limit).astype(int)
    dipoles_below_thresh = (ALL_DIPOLES_sum_below >= length_limit).astype(int)
    [row_coords_above, col_coords_above] = np.where(dipoles_above_thresh)
    [row_coords_below, col_coords_below] = np.where(dipoles_below_thresh)
    above_rc = list(zip(row_coords_above, col_coords_above))
    below_rc = list(zip(row_coords_below, col_coords_below))
    # now remove the coordinates shared between both, and collect common
    # coordinates into both_rc
    set_above_rc = set(above_rc) - set(below_rc)
    set_below_rc = set(below_rc) - set(above_rc)
    both_rc = list(set(above_rc) - set_above_rc)
    above_rc = list(set_above_rc)
    below_rc = list(set_below_rc)
    rc_both = {}
    amps_both = np.zeros([len(both_rc), len(cor_img_stack)])

    for i in range(len(both_rc)):
        amps_both[i] = cor_img_stack[:, both_rc[i][0], both_rc[i][1]].copy()
        rc_both[both_rc[i]] = {'amps_both': amps_both[i],
            'loc_med_min': ill_corr_min[both_rc[i][0], both_rc[i][1]].copy(),
            'loc_med_max': ill_corr_max[both_rc[i][0], both_rc[i][1]].copy()}
        # getting times that corresponded to above and below dipoles
        rc_both[both_rc[i]]['above'] = {'amp': [], 't': []}
        rc_both[both_rc[i]]['below'] = {'amp': [], 't': []}
        for z in range(len(dipoles_above_count)):
            if dipoles_above_count[z][both_rc[i]] > 0:
                rc_both[both_rc[i]]['above']['amp'].append(
                            cor_img_stack[z][both_rc[i]])
                rc_both[both_rc[i]]['above']['t'].append(timings[z])
        for z in range(len(dipoles_below_count)):
            if dipoles_below_count[z][both_rc[i]] > 0:
                rc_both[both_rc[i]]['below']['amp'].append(
                        cor_img_stack[z][both_rc[i]])
                rc_both[both_rc[i]]['below']['t'].append(timings[z])
        rc_both[both_rc[i]]['above']['amp'] = np.array(
                        rc_both[both_rc[i]]['above']['amp'])
        rc_both[both_rc[i]]['above']['t'] = np.array(
                        rc_both[both_rc[i]]['above']['t'])
        rc_both[both_rc[i]]['below']['amp'] = np.array(
                        rc_both[both_rc[i]]['below']['amp'])
        rc_both[both_rc[i]]['below']['t'] = np.array(
                        rc_both[both_rc[i]]['below']['t'])

    rc_above = {}
    amps_above = np.zeros([len(above_rc),len(cor_img_stack)])
    for i in range(len(above_rc)):
        amps_above[i] = cor_img_stack[:, above_rc[i][0], above_rc[i][1]].copy()
        rc_above[above_rc[i]] = {'amps_above': amps_above[i],
            'loc_med_min': ill_corr_min[above_rc[i][0], above_rc[i][1]].copy(),
            'loc_med_max': ill_corr_max[above_rc[i][0], above_rc[i][1]].copy()}

    # same thing as above, but now for the "below" dipoles
    rc_below = {}
    amps_below = np.zeros([len(below_rc),len(cor_img_stack)])
    for i in range(len(below_rc)):
        amps_below[i] = cor_img_stack[:, below_rc[i][0], below_rc[i][1]].copy()
        rc_below[below_rc[i]] = {'amps_below': amps_below[i],
            'loc_med_min': ill_corr_min[below_rc[i][0], below_rc[i][1]].copy(),
            'loc_med_max': ill_corr_max[below_rc[i][0], below_rc[i][1]].copy()}

    return rc_above, rc_below, rc_both

def trap_fit(scheme, amps, times, num_pumps, fit_thresh, tau_min, tau_max,
             tauc_min, tauc_max, offset_min, offset_max, both_a=None):
    """
    For a given temperature, scheme, and pixel, this function examines data
    for amplitude vs phase time and fits for release time constant (tau) and
    the probability for capture (pc). It tries fitting for a single trap in
    the pixel, and if the goodness of fit is not high enough (if less than
    fit_thresh), then the function attempts to fit for two traps in the pixel.
    The exception is the case where a 'both' type pixel is input; then only a
    two-trap fit is attempted. The function assumes the full time-dependent
    form for capture probability.

    The function was copied and pasted from trap_fitting.py in the II&T pipeline.

    Args:
        scheme (int): The scheme under consideration. Only certain probability 
            functions are valid for different schemes. Values: {1, 2, 3, 4}.
        amps (array): Amplitudes of bright pixel of the dipole. Units: e-.
        times (array): Phase times in the same order as amps. Units: seconds.
        num_pumps (int): Number of cycles for trap pumping. Must be greater than 0.
        fit_thresh (float): Minimum value required for adjusted coefficient of 
            determination (adjusted R^2) for curve fitting for the release time 
            constant (tau) using data for dipole amplitude vs phase time. The closer 
            to 1, the better the fit. Must be between 0 and 1.
        tau_min (float): Lower bound value for tau (release time constant) for 
            curve fitting. Units: seconds. Must be greater than or equal to 0.
        tau_max (float): Upper bound value for tau (release time constant) for 
            curve fitting. Units: seconds. Must be greater than tau_min.
        tauc_min (float): Lower bound value for tauc (capture time constant) for 
            curve fitting. Units: e-. Must be greater than or equal to 0.
        tauc_max (float): Upper bound value for tauc (capture time constant) for 
            curve fitting. Units: e-. Must be greater than tauc_min.
        offset_min (float): Lower bound value for the offset in the fitting of data 
            for amplitude vs phase time. Acts as a nuisance parameter. Units: e-.
        offset_max (float): Upper bound value for the offset in the fitting of data 
            for amplitude vs phase time. Acts as a nuisance parameter. Units: e-. 
            Must be greater than offset_min.
        both_a (dict, optional): Use None if fitting for a dipole that is of the 
            'above' or 'below' kind. Use the dictionary corresponding to 
            rc_both[(row,col)]['above'] if fitting for a dipole that is of the 
            'both' kind. Defaults to None.

    Returns:
        dict: Dictionary of fit data. Contains:
            - prob: {1, 2, 3}; denotes probability function that gave the best fit
            - pc: Constant capture probability, best-fit value
            - pc_err: Standard deviation of pc
            - tau: Release time constant, best-fit value
            - tau_err: Standard deviation of tau

        The structure of the returned dictionary varies based on `both_a`:
        - If `both_a` is None and one trap is the best fit:
            {prob: [[pc, pc_err, tau, tau_err]]}
        - If `both_a` is None and two traps are the best fit:
            {prob1: [[pc, pc_err, tau, tau_err]],
             prob2: [[pc2, pc2_err, tau2, tau2_err]]}
        - If both traps are of prob1 type:
            {prob1: [[pc, pc_err, tau, tau_err], [pc2, pc2_err, tau2, tau2_err]]}
        - If `both_a` is not None:
            {type1: {1: [[pc, pc_err, tau, tau_err]]},
             type2: {2: [[pc2, pc2_err, tau2, tau2_err]]}}
            where type1 is either 'a' for above or 'b' for below, and type2 is
            whichever of those type1 is not.
    """


    # TODO time-dep pc and pc2:  assume constant charge packet throughout the
    # different phase times across all num_pumps, when in reality, the charge
    # packet changes a little with each pump (a la 'express=num_pumps' in
    # arcticpy).  The fit function would technically be a sum of num_pumps
    # terms in which tauc is a slightly different value in each.  So the value
    # of tauc may not be that useful; could compare its uncertainty to that of
    # the values in the literature.  But it's generally good for large charge
    # packet values (perhaps it nominally corresponds to the average between
    # the starting charge packet value for trap pumping and the ending value
    # read off the highest-amp phase time frame.

    # check inputs
    if scheme != 1 and scheme != 2 and scheme != 3 and scheme != 4:
        raise TypeError('scheme must be 1, 2, 3, or 4.')
    try:
        amps = np.array(amps).astype(float)
    except:
        raise TypeError("amps elements should be real numbers")
    check.oneD_array(amps, 'amps', TypeError)
    try:
        times = np.array(times).astype(float)
    except:
        raise TypeError("times elements should be real numbers")
    check.oneD_array(times, 'times', TypeError)
    if len(np.unique(times)) < 6:
        raise IndexError('times must have a number of unique phase times '
        'longer than the number of fitted parameters.')
    if len(amps) != len(times):
        raise ValueError("times and amps should have same number of elements")
    check.positive_scalar_integer(num_pumps, 'num_pumps', TypeError)
    check.real_nonnegative_scalar(fit_thresh, 'fit_thresh', TypeError)
    if fit_thresh > 1:
        raise ValueError('fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(tau_min, 'tau_min', TypeError)
    check.real_nonnegative_scalar(tau_max, 'tau_max', TypeError)
    if tau_max <= tau_min:
        raise ValueError('tau_max must be > tau_min')
    check.real_nonnegative_scalar(tauc_min, 'tauc_min', TypeError)
    check.real_nonnegative_scalar(tauc_max, 'tauc_max', TypeError)
    if tauc_max <= tauc_min:
        raise ValueError('tauc_max must be > tauc_min')
    check.real_scalar(offset_min, 'offset_min', TypeError)
    check.real_scalar(offset_max, 'offset_max', TypeError)
    if offset_max <= offset_min:
        raise ValueError('offset_max must be > offset_min')
    if (type(both_a) is not dict) and (both_a is not None):
        raise TypeError('both_a must be a rc_both[(row,col)][\'above\'] '
        'dictionary or None.  See trap_id() doc string for more details.')
    if both_a is not None:
        try:
            x = len(both_a['amp'])
            y = len(both_a['t'])
        except:
            raise KeyError('both_a must contain \'amp\' and \'t\' keys.')
        if x != y:
            raise ValueError('both_a[\'amp\'] and both_a[\'t\'] must have the '
            'same number of elements')
    # if input for both_a doesn't conform to expected dictionary structure,
    # exceptions will be raised

    #upper bound for tauc: 1*eperdn, but our knowledge of eperdn may have
    # error.

    # Makes sense that you wouldn't find traps at times far away from time
    # constant, so to avoid false good fits, restrict bounds between 10^-6 and
    # 10^-2
    # in order, for one trap:  offset, tauc, tau
    l_bounds_one = [offset_min, tauc_min, tau_min]
    u_bounds_one = [offset_max, tauc_max, tau_max]
    # avoid initial guess of 0
    offset0 = (offset_min+offset_max)/2
    if offset0 == 0:
        offset0 = min(1 + (offset_min+offset_max)/2, offset_max)
    offset0l = offset_min
    if offset0l == 0:
        offset0l = min(offset0l+1, offset_max)
    offset0u = offset_max
    if offset0u == 0:
        offset0u = max(offset0u-1, offset_min)
    # tauc0 = 1e-9
    # if tauc0 < tauc_min or tauc0 > tauc_max:
    tauc0 = (tauc_min+tauc_max)/2
    tauc0l = tauc_min
    if tauc0l == 0:
        tauc0l = min(tauc0l+1e-12, tauc_max)
    tauc0u = tauc_max
    tau0 = (tau_min + tau_max)/2
    tau0l = tau_min
    if tau0l == 0:
        tau0l = min(tau0l+1e-7, tau_max)
    tau0u = tau_max
    # tau0 = np.median(times)
    # if tau_min > tau0 or tau_max < tau0:
    #     tau0 = (tau_min+tau_max)/2
    # start search from biggest time since these data points more
    # spread out if phase times are taken evenly spaced in log space; don't
    # want to give too much weight to the bunched-up early times and lock in
    # an answer too early in the curve search, so start at other end.
    # Try search from earliest time and biggest time, and see which gives a
    # bigger adj R^2 value
    p01l = [offset0, tauc0, tau0l]
    p01u = [offset0, tauc0, tau0u]
    #l_bounds_one = [-100000, 0, 0]
    #u_bounds_one = [100000, 1, 1]
    bounds_one = (l_bounds_one, u_bounds_one)
    # in order, for two traps:  offset, tauc, tau, tauc2, tau2
    l_bounds_two = [offset_min, tauc_min, tau_min, tauc_min, tau_min]
    u_bounds_two = [offset_max, tauc_max, tau_max, tauc_max, tau_max]
    # p02l = [offset0l, k_min, tauc_min, tau0l, tauc_min, tau0l]
    # p02u = [offset0u, k_max, tauc_max, tau0u, tauc_max, tau0u]
    p02l = [offset0, tauc0u, tau0l, tauc0l, tau0u]
    p02u = [offset0, tauc0l, tau0u, tauc0u, tau0l]
    # p02l = [offset0, tauc0, tau0l, tauc0, tau0u]
    # p02u = [offset0, tauc0, tau0u, tauc0, tau0l]
    #l_bounds_two = [-100000, 0, 0, 0, 0]
    #u_bounds_two = [100000, 1, 1, 1, 1]
    bounds_two = (l_bounds_two, u_bounds_two)

    # same for every curve fit (sum of the squared difference total)
    sstot = np.sum((amps - np.mean(amps))**2)

    if scheme == 1 or scheme == 2:

        if both_a == None:
            # attempt all possible probability functions
            try:
                popt1l, pcov1l = curve_fit(P1, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf,kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
                popt1u, pcov1u = curve_fit(P1, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf,kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit1l = P1(times, popt1l[0], popt1l[1], popt1l[2],num_pumps=num_pumps)
            fit1u = P1(times, popt1u[0], popt1u[1], popt1u[2],num_pumps=num_pumps)
            ssres1l = np.sum((fit1l - amps)**2)
            ssres1u = np.sum((fit1u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value1l = 1 - (ssres1l/sstot)*(len(times) - 1)/(len(times) -
                len(popt1l))
            R_value1u = 1 - (ssres1u/sstot)*(len(times) - 1)/(len(times) -
                len(popt1u))
            R_value1 = max(R_value1l, R_value1u)
            if R_value1 == R_value1l:
                popt1 = popt1l; pcov1 = pcov1l
            if R_value1 == R_value1u:
                popt1 = popt1u; pcov1 = pcov1u

            try:
                popt2l, pcov2l = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
                popt2u, pcov2u = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit2l = P2(times, popt2l[0], popt2l[1], popt2l[2], num_pumps=num_pumps)
            fit2u = P2(times, popt2u[0], popt2u[1], popt2u[2], num_pumps=num_pumps)
            ssres2l = np.sum((fit2l - amps)**2)
            ssres2u = np.sum((fit2u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value2l = 1 - (ssres2l/sstot)*(len(times) - 1)/(len(times) -
                len(popt2l))
            R_value2u = 1 - (ssres2u/sstot)*(len(times) - 1)/(len(times) -
                len(popt2u))
            R_value2 = max(R_value2l, R_value2u)
            if R_value2 == R_value2l:
                popt2 = popt2l; pcov2 = pcov2l
            if R_value2 == R_value2u:
                popt2 = popt2u; pcov2 = pcov2u

            # accept the best fit and require threshold met
            maxR1 = max(R_value1, R_value2)

            if maxR1 >= fit_thresh and maxR1 == R_value1:
                tauc = popt1[1]
                tau = popt1[2]
                _, tauc_err, tau_err  = np.sqrt(np.diag(pcov1))
                return {1: [[tauc, tauc_err, tau, tau_err]]}
            if maxR1 >= fit_thresh and maxR1 == R_value2:
                tauc = popt2[1]
                tau = popt2[2]
                _, tauc_err, tau_err  = np.sqrt(np.diag(pcov2))
                return {2: [[tauc, tauc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt11l, pcov11l = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
            popt11u, pcov11u = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit11l = P1_P1(times, popt11l[0], popt11l[1], popt11l[2], popt11l[3],
        popt11l[4], num_pumps=num_pumps)
        fit11u = P1_P1(times, popt11u[0], popt11u[1], popt11u[2], popt11u[3],
        popt11u[4], num_pumps=num_pumps)
        ssres11l = np.sum((fit11l - amps)**2)
        ssres11u = np.sum((fit11u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value11l = 1 - (ssres11l/sstot)*(len(times) - 1)/(len(times) -
            len(popt11l))
        R_value11u = 1 - (ssres11u/sstot)*(len(times) - 1)/(len(times) -
            len(popt11u))
        R_value11 = max(R_value11l, R_value11u)
        if R_value11 == R_value11l:
            popt11 = popt11l; pcov11 = pcov11l
        if R_value11 == R_value11u:
            popt11 = popt11u; pcov11 = pcov11u

        try:
            popt12l, pcov12l = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
            popt12u, pcov12u = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit12l = P1_P2(times, popt12l[0], popt12l[1], popt12l[2], popt12l[3],
        popt12l[4], num_pumps=num_pumps)
        fit12u = P1_P2(times, popt12u[0], popt12u[1], popt12u[2], popt12u[3],
        popt12u[4], num_pumps=num_pumps)
        ssres12l = np.sum((fit12l - amps)**2)
        ssres12u = np.sum((fit12u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value12l = 1 - (ssres12l/sstot)*(len(times) - 1)/(len(times) -
            len(popt12l))
        R_value12u = 1 - (ssres12u/sstot)*(len(times) - 1)/(len(times) -
            len(popt12u))
        R_value12 = max(R_value12l, R_value12u)
        if R_value12 == R_value12l:
            popt12 = popt12l; pcov12 = pcov12l
        if R_value12 == R_value12u:
            popt12 = popt12u; pcov12 = pcov12u

        try:
            popt22l, pcov22l = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
            popt22u, pcov22u = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22l = P2_P2(times, popt22l[0], popt22l[1], popt22l[2], popt22l[3],
        popt22l[4], num_pumps=num_pumps)
        fit22u = P2_P2(times, popt22u[0], popt22u[1], popt22u[2], popt22u[3],
        popt22u[4], num_pumps=num_pumps)
        ssres22l = np.sum((fit22l - amps)**2)
        ssres22u = np.sum((fit22u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22l = 1 - (ssres22l/sstot)*(len(times) - 1)/(len(times) -
            len(popt22l))
        R_value22u = 1 - (ssres22u/sstot)*(len(times) - 1)/(len(times) -
            len(popt22u))
        R_value22 = max(R_value22l, R_value22u)
        if R_value22 == R_value22l:
            popt22 = popt22l; pcov22 = pcov22l
        if R_value22 == R_value22u:
            popt22 = popt22u; pcov22 = pcov22u

        maxR2 = max(R_value11, R_value12, R_value22)
        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value11:
            off = popt11[0]
            tauc = popt11[1]
            tau = popt11[2]
            tauc2 = popt11[3]
            tau2 = popt11[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov11))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                # TODO v2, for trap_fit and trap_fit_const:
                # if 'amp' list for 'above' is identical to 'amp'
                #list for 'below', then earmark for assignment whenever
                # finding optimal matchings of schemes for sub-el loc; if
                #no particular assignment is better at that point, then the
                #assignment doesn't matter
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P1 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(2)
                #P1 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{1: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{1: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {1: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}

        if maxR2 == R_value12:
            off = popt12[0]
            tauc = popt12[1]
            tau = popt12[2]
            tauc2 = popt12[3]
            tau2 = popt12[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov12))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P1 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {1: [[tauc, tauc_err, tau, tau_err]],
                2: [[tauc2, tauc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            tauc = popt22[1]
            tau = popt22[2]
            tauc2 = popt22[3]
            tau2 = popt22[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}

    if scheme == 3 or scheme == 4:

        if both_a == None:
            #attempt both probability functions
            try:
                popt3l, pcov3l = curve_fit(P3, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
                popt3u, pcov3u = curve_fit(P3, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit3l = P3(times, popt3l[0], popt3l[1], popt3l[2], num_pumps=num_pumps)
            fit3u = P3(times, popt3u[0], popt3u[1], popt3u[2], num_pumps=num_pumps)
            ssres3l = np.sum((fit3l - amps)**2)
            ssres3u = np.sum((fit3u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value3l = 1 - (ssres3l/sstot)*(len(times) - 1)/(len(times) -
                len(popt3l))
            R_value3u = 1 - (ssres3u/sstot)*(len(times) - 1)/(len(times) -
                len(popt3u))
            R_value3 = max(R_value3l, R_value3u)
            if R_value3 == R_value3l:
                popt3 = popt3l; pcov3 = pcov3l
            if R_value3 == R_value3u:
                popt3 = popt3u; pcov3 = pcov3u

            try:
                popt2l, pcov2l = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
                popt2u, pcov2u = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit2l = P2(times, popt2l[0], popt2l[1], popt2l[2], num_pumps=num_pumps)
            fit2u = P2(times, popt2u[0], popt2u[1], popt2u[2], num_pumps=num_pumps)
            ssres2l = np.sum((fit2l - amps)**2)
            ssres2u = np.sum((fit2u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value2l = 1 - (ssres2l/sstot)*(len(times) - 1)/(len(times) -
                len(popt2l))
            R_value2u = 1 - (ssres2u/sstot)*(len(times) - 1)/(len(times) -
                len(popt2u))
            R_value2 = max(R_value2l, R_value2u)
            if R_value2 == R_value2l:
                popt2 = popt2l; pcov2 = pcov2l
            if R_value2 == R_value2u:
                popt2 = popt2u; pcov2 = pcov2u

            # accept the best fit and require threshold met
            maxR1 = max(R_value3, R_value2)

            if maxR1 >= fit_thresh and maxR1 == R_value3:
                tauc = popt3[1]
                tau = popt3[2]
                _, tauc_err, tau_err  = np.sqrt(np.diag(pcov3))
                return {3: [[tauc, tauc_err, tau, tau_err]]}
            if maxR1 >= fit_thresh and maxR1 == R_value2:
                tauc = popt2[1]
                tau = popt2[2]
                _, tauc_err, tau_err  = np.sqrt(np.diag(pcov2))
                return {2: [[tauc, tauc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt33l, pcov33l = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
            popt33u, pcov33u = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit33l = P3_P3(times, popt33l[0], popt33l[1], popt33l[2], popt33l[3],
        popt33l[4], num_pumps=num_pumps)
        fit33u = P3_P3(times, popt33u[0], popt33u[1], popt33u[2], popt33u[3],
        popt33u[4], num_pumps=num_pumps)
        ssres33l = np.sum((fit33l - amps)**2)
        ssres33u = np.sum((fit33u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value33l = 1 - (ssres33l/sstot)*(len(times) - 1)/(len(times) -
            len(popt33l))
        R_value33u = 1 - (ssres33u/sstot)*(len(times) - 1)/(len(times) -
            len(popt33u))
        R_value33 = max(R_value33l, R_value33u)
        if R_value33 == R_value33l:
            popt33 = popt33l; pcov33 = pcov33l
        if R_value33 == R_value33u:
            popt33 = popt33u; pcov33 = pcov33u

        try:
            popt23l, pcov23l = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
            popt23u, pcov23u = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit23l = P2_P3(times, popt23l[0], popt23l[1], popt23l[2], popt23l[3],
        popt23l[4], num_pumps=num_pumps)
        fit23u = P2_P3(times, popt23u[0], popt23u[1], popt23u[2], popt23u[3],
        popt23u[4], num_pumps=num_pumps)
        ssres23l = np.sum((fit23l - amps)**2)
        ssres23u = np.sum((fit23u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value23l = 1 - (ssres23l/sstot)*(len(times) - 1)/(len(times) -
            len(popt23l))
        R_value23u = 1 - (ssres23u/sstot)*(len(times) - 1)/(len(times) -
            len(popt23u))
        R_value23 = max(R_value23l, R_value23u)
        if R_value23 == R_value23l:
            popt23 = popt23l; pcov23 = pcov23l
        if R_value23 == R_value23u:
            popt23 = popt23u; pcov23 = pcov23u

        try:
            popt22l, pcov22l = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
            popt22u, pcov22u = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf, kwargs={"num_pumps" : num_pumps})#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22l = P2_P2(times, popt22l[0], popt22l[1], popt22l[2], popt22l[3],
        popt22l[4], num_pumps=num_pumps)
        fit22u = P2_P2(times, popt22u[0], popt22u[1], popt22u[2], popt22u[3],
        popt22u[4], num_pumps=num_pumps)
        ssres22l = np.sum((fit22l - amps)**2)
        ssres22u = np.sum((fit22u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22l = 1 - (ssres22l/sstot)*(len(times) - 1)/(len(times) -
            len(popt22l))
        R_value22u = 1 - (ssres22u/sstot)*(len(times) - 1)/(len(times) -
            len(popt22u))
        R_value22 = max(R_value22l, R_value22u)
        if R_value22 == R_value22l:
            popt22 = popt22l; pcov22 = pcov22l
        if R_value22 == R_value22u:
            popt22 = popt22u; pcov22 = pcov22u

        maxR2 = max(R_value33, R_value23, R_value22)

        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value33:
            off = popt33[0]
            tauc = popt33[1]
            tau = popt33[2]
            tauc2 = popt33[3]
            tau2 = popt33[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov33))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P3 for tau
                a_tau = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                #P3 for tau2
                a_tau2 = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{3: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{3: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {3: [[tauc, tauc_err, tau, tau_err], [tauc2, tauc2_err,
                tau2, tau2_err]]}

        if maxR2 == R_value23:
            off = popt23[0]
            tauc = popt23[1]
            tau = popt23[2]
            tauc2 = popt23[3]
            tau2 = popt23[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov23))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P3 for tau2
                a_tau2 = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{3: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err]],
                3: [[tauc2, tauc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            tauc = popt22[1]
            tau = popt22[2]
            tauc2 = popt22[3]
            tau2 = popt22[4]
            _, tauc_err, tau_err, tauc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'b':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[tauc, tauc_err, tau, tau_err]]},
                        'a':{2: [[tauc2, tauc2_err, tau2, tau2_err]]}}
            return {2: [[tauc, tauc_err, tau, tau_err],
                [tauc2, tauc2_err, tau2, tau2_err]]}

def trap_fit_const(scheme, amps, times, num_pumps, fit_thresh, tau_min,
                   tau_max, pc_min, pc_max, offset_min, offset_max, both_a=None):
    """
    For a given temperature, scheme, and pixel, this function examines data
    for amplitude vs phase time and fits for release time constant (tau) and
    the probability for capture (pc). It tries fitting for a single trap in
    the pixel, and if the goodness of fit is not high enough (if less than
    fit_thresh), then the function attempts to fit for two traps in the pixel.
    The exception is the case where a 'both' type pixel is input; then only a
    two-trap fit is attempted. The function assumes a constant for pc rather
    than its actual time-dependent form.

    The function was copied and pasted from trap_fitting.py in the II&T pipeline.

    Args:
        scheme (int): The scheme under consideration. Only certain probability 
            functions are valid for different schemes. Values: {1, 2, 3, 4}.
        amps (array): Amplitudes of bright pixel of the dipole. Units: e-.
        times (array): Phase times in the same order as amps. Units: seconds.
        num_pumps (int): Number of cycles for trap pumping. Must be greater than 0.
        fit_thresh (float): Minimum value required for adjusted coefficient of 
            determination (adjusted R^2) for curve fitting for the release time 
            constant (tau) using data for dipole amplitude vs phase time. The closer 
            to 1, the better the fit. Must be between 0 and 1.
        tau_min (float): Lower bound value for tau (release time constant) for 
            curve fitting. Units: seconds. Must be greater than or equal to 0.
        tau_max (float): Upper bound value for tau (release time constant) for 
            curve fitting. Units: seconds. Must be greater than tau_min.
        pc_min (float): Lower bound value for pc (capture probability) for 
            curve fitting. Units: e-. Must be greater than or equal to 0.
        pc_max (float): Upper bound value for pc (capture probability) for 
            curve fitting. Units: e-. Must be greater than pc_min.
        offset_min (float): Lower bound value for the offset in the fitting of 
            data for amplitude vs phase time. Acts as a nuisance parameter. Units: e-.
        offset_max (float): Upper bound value for the offset in the fitting of 
            data for amplitude vs phase time. Acts as a nuisance parameter. Units: e-. 
            Must be greater than offset_min.
        both_a (dict, optional): Use None if fitting for a dipole that is of the 
            'above' or 'below' kind. Use the dictionary corresponding to 
            rc_both[(row,col)]['above'] if fitting for a dipole that is of the 
            'both' kind. Defaults to None.

    Returns:
        dict: Dictionary of fit data. Contains:
            - prob: {1, 2, 3}; denotes probability function that gave the best fit
            - pc: Constant capture probability, best-fit value
            - pc_err: Standard deviation of pc
            - tau: Release time constant, best-fit value
            - tau_err: Standard deviation of tau

        The structure of the returned dictionary varies based on `both_a`:
        - If `both_a` is None and one trap is the best fit:
            {prob: [[pc, pc_err, tau, tau_err]]}
        - If `both_a` is None and two traps are the best fit:
            {prob1: [[pc, pc_err, tau, tau_err]],
             prob2: [[pc2, pc2_err, tau2, tau2_err]]}
        - If both traps are of prob1 type:
            {prob1: [[pc, pc_err, tau, tau_err], [pc2, pc2_err, tau2, tau2_err]]}
        - If `both_a` is not None:
            {type1: {1: [[pc, pc_err, tau, tau_err]]},
             type2: {2: [[pc2, pc2_err, tau2, tau2_err]]}}
            where type1 is either 'a' for above or 'b' for below, and type2 is
            whichever of those type1 is not.
    """

    # TODO time-dep pc and pc2:  assume constant charge packet throughout the
    # different phase times across all num_pumps, when in reality, the charge
    # packet changes a little with each pump (a la 'express=num_pumps' in
    # arcticpy).  The fit function would technically be a sum of num_pumps
    # terms in which tauc is a slightly different value in each.  So the value
    # of tauc may not be that useful; could compare its uncertainty to that of
    # the values in the literature.  But it's generally good for large charge
    # packet values (perhaps it nominally corresponds to the average between
    # the starting charge packet value for trap pumping and the ending value
    # read off the highest-amp phase time frame.

    # check inputs
    if scheme != 1 and scheme != 2 and scheme != 3 and scheme != 4:
        raise TypeError('scheme must be 1, 2, 3, or 4.')
    try:
        amps = np.array(amps).astype(float)
    except:
        raise TypeError("amps elements should be real numbers")
    check.oneD_array(amps, 'amps', TypeError)
    try:
        times = np.array(times).astype(float)
    except:
        raise TypeError("times elements should be real numbers")
    check.oneD_array(times, 'times', TypeError)
    if len(np.unique(times)) < 6:
        raise IndexError('times must have a number of unique phase times '
        'longer than the number of fitted parameters.')
    if len(amps) != len(times):
        raise ValueError("times and amps should have same number of elements")
    check.positive_scalar_integer(num_pumps, 'num_pumps', TypeError)
    check.real_nonnegative_scalar(fit_thresh, 'fit_thresh', TypeError)
    if fit_thresh > 1:
        raise ValueError('fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(tau_min, 'tau_min', TypeError)
    check.real_nonnegative_scalar(tau_max, 'tau_max', TypeError)
    if tau_max <= tau_min:
        raise ValueError('tau_max must be > tau_min')
    check.real_nonnegative_scalar(pc_min, 'pc_min', TypeError)
    check.real_nonnegative_scalar(pc_max, 'pc_max', TypeError)
    if pc_max <= pc_min:
        raise ValueError('pc_max must be > pc_min')
    check.real_scalar(offset_min, 'offset_min', TypeError)
    check.real_scalar(offset_max, 'offset_max', TypeError)
    if offset_max <= offset_min:
        raise ValueError('offset_max must be > offset_min')
    if (type(both_a) is not dict) and (both_a is not None):
        raise TypeError('both_a must be a rc_both[(row,col)][\'above\'] '
        'dictionary.  See trap_id() doc string for more details.')
    if both_a is not None:
        try:
            x = len(both_a['amp'])
            y = len(both_a['t'])
        except:
            raise KeyError('both_a must contain \'amp\' and \'t\' keys.')
        if x != y:
            raise ValueError('both_a[\'amp\'] and both_a[\'t\'] must have the '
            'same number of elements')
    # if both_a doesn't have expected keys, exceptions will be raised

    # define all these inside this function because they depend also on
    # num_pumps, and I can't specify an unfitted parameter in the function
    # definition if I want to use curve_fit
    def P1(time_data, offset, pc, tau):
        """Probability function 1, one trap.

        Args:
            time_data (array): Phase times in seconds.
            offset (float): Offset in the fitting of data for amplitude vs phase time.
                Acts as a nuisance parameter. Units: e-.
            pc (float): Capture probability. Units: e-.
            tau (float): Release time constant. Units: seconds.

        Returns:
            array: Amplitude vs phase time for a single trap.
        """
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau)))

    def P1_P1(time_data, offset, pc, tau, pc2, tau2):
        """Probability function 1, two traps.

        Args:
            time_data (array): Phase times in seconds.
            offset (float): Offset in the fitting of data for amplitude vs phase time.
                Acts as a nuisance parameter. Units: e-.
            pc (float): Capture probability for the first trap. Units: e-.
            tau (float): Release time constant for the first trap. Units: seconds.
            pc2 (float): Capture probability for the second trap. Units: e-.
            tau2 (float): Release time constant for the second trap. Units: seconds.

        Returns:
            array: Amplitude vs phase time for two traps.
        """
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-2*time_data/tau2))

    def P2(time_data, offset, pc, tau):
        """Probability function 2, one trap.

        Args:
            time_data (array): Phase times in seconds.
            offset (float): Offset in the fitting of data for amplitude vs phase time.
                Acts as a nuisance parameter. Units: e-.
            pc (float): Capture probability. Units: e-.
            tau (float): Release time constant. Units: seconds.

        Returns:
            array: Amplitude vs phase time for a single trap.
        """
        return offset+(num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau)))

    def P1_P2(time_data, offset, pc, tau, pc2, tau2):
        """One trap for probability function 1, one for probability function 2.

        Args:
            time_data (array): Phase times in seconds.
            offset (float): Offset in the fitting of data for amplitude vs phase time.
                Acts as a nuisance parameter. Units: e-.
            pc (float): Capture probability for the first trap. Units: e-.
            tau (float): Release time constant for the first trap. Units: seconds.
            pc2 (float): Capture probability for the second trap. Units: e-.
            tau2 (float): Release time constant for the second trap. Units: seconds.

        Returns:
            array: Amplitude vs phase time for two traps.
        """
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P2_P2(time_data, offset, pc, tau, pc2, tau2):
        """Probability function 2, two traps.

        Args:
            time_data (array): Phase times in seconds.
            offset (float): Offset in the fitting of data for amplitude vs phase time.
                Acts as a nuisance parameter. Units: e-.
            pc (float): Capture probability for the first trap. Units: e-.
            tau (float): Release time constant for the first trap. Units: seconds.
            pc2 (float): Capture probability for the second trap. Units: e-.
            tau2 (float): Release time constant for the second trap. Units: seconds.

        Returns:
            array: Amplitude vs phase time for two traps.
        """
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

    def P3(time_data, offset, pc, tau):
        """Probability function 3, one trap.

        Args:
            time_data (array): Phase times in seconds.
            offset (float): Offset in the fitting of data for amplitude vs phase time.
                Acts as a nuisance parameter. Units: e-.
            pc (float): Capture probability. Units: e-.
            tau (float): Release time constant. Units: seconds.

        Returns:
            array: Amplitude vs phase time for a single trap
        """
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau)))

    def P3_P3(time_data, offset, pc, tau, pc2, tau2):
        """Probability function 3, two traps.

        Args:
            time_data (array): Phase times in seconds.
            offset (float): Offset in the fitting of data for amplitude vs phase time.
                Acts as a nuisance parameter. Units: e-.
            pc (float): Capture probability for the first trap. Units: e-.
            tau (float): Release time constant for the first trap. Units: seconds.
            pc2 (float): Capture probability for the second trap. Units: e-.
            tau2 (float): Release time constant for the second trap. Units: seconds.

        Returns:
            array: Amplitude vs phase time for two traps.
        """
        return offset+num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-4*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    def P2_P3(time_data, offset, pc, tau, pc2, tau2):
        """One trap for probability function 2, one for probability function 3.

        Args:
            time_data (array): Phase times in seconds.
            offset (float): Offset in the fitting of data for amplitude vs phase time.
                Acts as a nuisance parameter. Units: e-.
            pc (float): Capture probability for the first trap. Units: e-.
            tau (float): Release time constant for the first trap. Units: seconds.
            pc2 (float): Capture probability for the second trap. Units: e-.
            tau2 (float): Release time constant for the second trap. Units: seconds.

        Returns:
            array: Amplitude vs phase time for two traps.
        """
        return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
            np.exp(-3*time_data/tau))+ \
            num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

    #upper bound for pc: 1*eperdn, but our knowledge of eperdn may have error.

    # Makes sense that you wouldn't find traps at times far away from time
    # constant, so to avoid false good fits, restrict bounds between 10^-6 and
    # 10^-2
    # in order, for one trap:  offset, pc, tau
    l_bounds_one = [offset_min, pc_min, tau_min]
    u_bounds_one = [offset_max, pc_max, tau_max]
    # avoid initial guess of 0
    offset0 = (offset_min+offset_max)/2
    if (offset_min+offset_max)/2 == 0:
        offset0 = min(1 + (offset_min+offset_max)/2, offset_max)
    offset0l = offset_min
    if offset0l == 0:
        offset0l = min(offset0l+1, offset_max)
    offset0u = offset_max
    if offset0u == 0:
        offset0u = max(offset0u-1, offset_min)
    pc0 = 1
    if 1 < pc_min or 1 > pc_max:
        pc0 = pc_min
    tau0 = (tau_min + tau_max)/2
    tau0l = tau_min
    if tau0l == 0:
        tau0l = min(tau0l+1e-7, tau_max)
    tau0u = tau_max
    # tau0 = np.median(times)
    # if tau_min > tau0 or tau_max < tau0:
    #     tau0 = (tau_min+tau_max)/2
    # start search from biggest time since these data points more
    # spread out if phase times are taken evenly spaced in log space; don't
    # want to give too much weight to the bunched-up early times and lock in
    # an answer too early in the curve search, so start at other end.
    # Try search from earliest time and biggest time, and see which gives a
    # bigger adj R^2 value
    p01l = [offset0, pc0, tau0l]
    p01u = [offset0, pc0, tau0u]
    #l_bounds_one = [-100000, 0, 0]
    #u_bounds_one = [100000, 1, 1]
    bounds_one = (l_bounds_one, u_bounds_one)
    # in order, for two traps:  offset, tauc, tau, tauc2, tau2
    l_bounds_two = [offset_min, pc_min, tau_min, pc_min, tau_min]
    u_bounds_two = [offset_max, pc_max, tau_max, pc_max, tau_max]
    # p02l = [offset0l, k_min, tauc_min, tau0l, tauc_min, tau0l]
    # p02u = [offset0u, k_max, tauc_max, tau0u, tauc_max, tau0u]
    p02l = [offset0, pc0, tau0l, pc0, tau0u]
    p02u = [offset0, pc0, tau0u, pc0, tau0l]
    #l_bounds_two = [-100000, 0, 0, 0, 0]
    #u_bounds_two = [100000, 1, 1, 1, 1]
    bounds_two = (l_bounds_two, u_bounds_two)

    # same for every curve fit (sum of the squared difference total)
    sstot = np.sum((amps - np.mean(amps))**2)

    if scheme == 1 or scheme == 2:

        if both_a == None:
            # attempt all possible probability functions
            try:
                popt1l, pcov1l = curve_fit(P1, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt1u, pcov1u = curve_fit(P1, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit1l = P1(times, popt1l[0], popt1l[1], popt1l[2])
            fit1u = P1(times, popt1u[0], popt1u[1], popt1u[2])
            ssres1l = np.sum((fit1l - amps)**2)
            ssres1u = np.sum((fit1u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value1l = 1 - (ssres1l/sstot)*(len(times) - 1)/(len(times) -
                len(popt1l))
            R_value1u = 1 - (ssres1u/sstot)*(len(times) - 1)/(len(times) -
                len(popt1u))
            R_value1 = max(R_value1l, R_value1u)
            if R_value1 == R_value1l:
                popt1 = popt1l; pcov1 = pcov1l
            if R_value1 == R_value1u:
                popt1 = popt1u; pcov1 = pcov1u

            try:
                popt2l, pcov2l = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt2u, pcov2u = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit2l = P2(times, popt2l[0], popt2l[1], popt2l[2])
            fit2u = P2(times, popt2u[0], popt2u[1], popt2u[2])
            ssres2l = np.sum((fit2l - amps)**2)
            ssres2u = np.sum((fit2u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value2l = 1 - (ssres2l/sstot)*(len(times) - 1)/(len(times) -
                len(popt2l))
            R_value2u = 1 - (ssres2u/sstot)*(len(times) - 1)/(len(times) -
                len(popt2u))
            R_value2 = max(R_value2l, R_value2u)
            if R_value2 == R_value2l:
                popt2 = popt2l; pcov2 = pcov2l
            if R_value2 == R_value2u:
                popt2 = popt2u; pcov2 = pcov2u

            # accept the best fit and require threshold met
            maxR1 = max(R_value1, R_value2)

            if maxR1 >= fit_thresh and maxR1 == R_value1:
                pc = popt1[1]
                tau = popt1[2]
                _, pc_err, tau_err  = np.sqrt(np.diag(pcov1))
                return {1: [[pc, pc_err, tau, tau_err]]}
            if maxR1 >= fit_thresh and maxR1 == R_value2:
                pc = popt2[1]
                tau = popt2[2]
                _, pc_err, tau_err  = np.sqrt(np.diag(pcov2))
                return {2: [[pc, pc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt11l, pcov11l = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt11u, pcov11u = curve_fit(P1_P1, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit11l = P1_P1(times, popt11l[0], popt11l[1], popt11l[2], popt11l[3],
        popt11l[4])
        fit11u = P1_P1(times, popt11u[0], popt11u[1], popt11u[2], popt11u[3],
        popt11u[4])
        ssres11l = np.sum((fit11l - amps)**2)
        ssres11u = np.sum((fit11u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value11l = 1 - (ssres11l/sstot)*(len(times) - 1)/(len(times) -
            len(popt11l))
        R_value11u = 1 - (ssres11u/sstot)*(len(times) - 1)/(len(times) -
            len(popt11u))
        R_value11 = max(R_value11l, R_value11u)
        if R_value11 == R_value11l:
            popt11 = popt11l; pcov11 = pcov11l
        if R_value11 == R_value11u:
            popt11 = popt11u; pcov11 = pcov11u

        try:
            popt12l, pcov12l = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt12u, pcov12u = curve_fit(P1_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit12l = P1_P2(times, popt12l[0], popt12l[1], popt12l[2], popt12l[3],
        popt12l[4])
        fit12u = P1_P2(times, popt12u[0], popt12u[1], popt12u[2], popt12u[3],
        popt12u[4])
        ssres12l = np.sum((fit12l - amps)**2)
        ssres12u = np.sum((fit12u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value12l = 1 - (ssres12l/sstot)*(len(times) - 1)/(len(times) -
            len(popt12l))
        R_value12u = 1 - (ssres12u/sstot)*(len(times) - 1)/(len(times) -
            len(popt12u))
        R_value12 = max(R_value12l, R_value12u)
        if R_value12 == R_value12l:
            popt12 = popt12l; pcov12 = pcov12l
        if R_value12 == R_value12u:
            popt12 = popt12u; pcov12 = pcov12u

        try:
            popt22l, pcov22l = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt22u, pcov22u = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22l = P2_P2(times, popt22l[0], popt22l[1], popt22l[2], popt22l[3],
        popt22l[4])
        fit22u = P2_P2(times, popt22u[0], popt22u[1], popt22u[2], popt22u[3],
        popt22u[4])
        ssres22l = np.sum((fit22l - amps)**2)
        ssres22u = np.sum((fit22u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22l = 1 - (ssres22l/sstot)*(len(times) - 1)/(len(times) -
            len(popt22l))
        R_value22u = 1 - (ssres22u/sstot)*(len(times) - 1)/(len(times) -
            len(popt22u))
        R_value22 = max(R_value22l, R_value22u)
        if R_value22 == R_value22l:
            popt22 = popt22l; pcov22 = pcov22l
        if R_value22 == R_value22u:
            popt22 = popt22u; pcov22 = pcov22u

        maxR2 = max(R_value11, R_value12, R_value22)
        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value11:
            off = popt11[0]
            pc = popt11[1]
            tau = popt11[2]
            pc2 = popt11[3]
            tau2 = popt11[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov11))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                # TODO v2, for trap_fit and trap_fit_const:
                # if 'amp' list for 'above' is identical to 'amp'
                #list for 'below', then earmark for assignment whenever
                # finding optimal matchings of schemes for sub-el loc; if
                #no particular assignment is better at that point, then the
                #assignment doesn't matter
                #TODO v2, for trap_fit and trap_fit_const:
                # if picking peak amp phase time to match with the closer tau
                # value isn't good enough (not decipherable if not enough data
                # points to make it out), then I could look at the
                # complementary probability functions in the corresponding
                #deficit pixels and curve fit those
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P1 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(2)
                #P1 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{1: [[pc, pc_err, tau, tau_err]]},
                        'b':{1: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[pc, pc_err, tau, tau_err]]},
                        'a':{1: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {1: [[pc, pc_err, tau, tau_err],
                [pc2, pc2_err, tau2, tau2_err]]}

        if maxR2 == R_value12:
            off = popt12[0]
            pc = popt12[1]
            tau = popt12[2]
            pc2 = popt12[3]
            tau2 = popt12[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov12))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P1 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{1: [[pc, pc_err, tau, tau_err]]},
                        'b':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{1: [[pc, pc_err, tau, tau_err]]},
                        'a':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {1: [[pc, pc_err, tau, tau_err]],
                2: [[pc2, pc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            pc = popt22[1]
            tau = popt22[2]
            pc2 = popt22[3]
            tau2 = popt22[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[pc, pc_err, tau, tau_err]]},
                        'b':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[pc, pc_err, tau, tau_err]]},
                        'a':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {2: [[pc, pc_err, tau, tau_err],
                [pc2, pc2_err, tau2, tau2_err]]}

    if scheme == 3 or scheme == 4:

        if both_a == None:
            #attempt both probability functions
            try:
                popt3l, pcov3l = curve_fit(P3, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt3u, pcov3u = curve_fit(P3, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit3l = P3(times, popt3l[0], popt3l[1], popt3l[2])
            fit3u = P3(times, popt3u[0], popt3u[1], popt3u[2])
            ssres3l = np.sum((fit3l - amps)**2)
            ssres3u = np.sum((fit3u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value3l = 1 - (ssres3l/sstot)*(len(times) - 1)/(len(times) -
                len(popt3l))
            R_value3u = 1 - (ssres3u/sstot)*(len(times) - 1)/(len(times) -
                len(popt3u))
            R_value3 = max(R_value3l, R_value3u)
            if R_value3 == R_value3l:
                popt3 = popt3l; pcov3 = pcov3l
            if R_value3 == R_value3u:
                popt3 = popt3u; pcov3 = pcov3u

            try:
                popt2l, pcov2l = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01l, maxfev = np.inf)#, sigma = 0.1*amps)
                popt2u, pcov2u = curve_fit(P2, times, amps, bounds=bounds_one,
                p0 = p01u, maxfev = np.inf)#, sigma = 0.1*amps)
            except:
                warnings.warn('curve_fit failed')
                return None
            fit2l = P2(times, popt2l[0], popt2l[1], popt2l[2])
            fit2u = P2(times, popt2u[0], popt2u[1], popt2u[2])
            ssres2l = np.sum((fit2l - amps)**2)
            ssres2u = np.sum((fit2u - amps)**2)
            # coefficient of determination, adjusted R^2:
            # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
            R_value2l = 1 - (ssres2l/sstot)*(len(times) - 1)/(len(times) -
                len(popt2l))
            R_value2u = 1 - (ssres2u/sstot)*(len(times) - 1)/(len(times) -
                len(popt2u))
            R_value2 = max(R_value2l, R_value2u)
            if R_value2 == R_value2l:
                popt2 = popt2l; pcov2 = pcov2l
            if R_value2 == R_value2u:
                popt2 = popt2u; pcov2 = pcov2u

            # accept the best fit and require threshold met
            maxR1 = max(R_value3, R_value2)

            if maxR1 >= fit_thresh and maxR1 == R_value3:
                pc = popt3[1]
                tau = popt3[2]
                _, pc_err, tau_err  = np.sqrt(np.diag(pcov3))
                return {3: [[pc, pc_err, tau, tau_err]]}
            if maxR1 >= fit_thresh and maxR1 == R_value2:
                pc = popt2[1]
                tau = popt2[2]
                _, pc_err, tau_err  = np.sqrt(np.diag(pcov2))
                return {2: [[pc, pc_err, tau, tau_err]]}

        # maxR1 must have been below fit_thresh.  Now try 2 traps

        try:
            popt33l, pcov33l = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt33u, pcov33u = curve_fit(P3_P3, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit33l = P3_P3(times, popt33l[0], popt33l[1], popt33l[2], popt33l[3],
        popt33l[4])
        fit33u = P3_P3(times, popt33u[0], popt33u[1], popt33u[2], popt33u[3],
        popt33u[4])
        ssres33l = np.sum((fit33l - amps)**2)
        ssres33u = np.sum((fit33u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value33l = 1 - (ssres33l/sstot)*(len(times) - 1)/(len(times) -
            len(popt33l))
        R_value33u = 1 - (ssres33u/sstot)*(len(times) - 1)/(len(times) -
            len(popt33u))
        R_value33 = max(R_value33l, R_value33u)
        if R_value33 == R_value33l:
            popt33 = popt33l; pcov33 = pcov33l
        if R_value33 == R_value33u:
            popt33 = popt33u; pcov33 = pcov33u

        try:
            popt23l, pcov23l = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt23u, pcov23u = curve_fit(P2_P3, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit23l = P2_P3(times, popt23l[0], popt23l[1], popt23l[2], popt23l[3],
        popt23l[4])
        fit23u = P2_P3(times, popt23u[0], popt23u[1], popt23u[2], popt23u[3],
        popt23u[4])
        ssres23l = np.sum((fit23l - amps)**2)
        ssres23u = np.sum((fit23u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value23l = 1 - (ssres23l/sstot)*(len(times) - 1)/(len(times) -
            len(popt23l))
        R_value23u = 1 - (ssres23u/sstot)*(len(times) - 1)/(len(times) -
            len(popt23u))
        R_value23 = max(R_value23l, R_value23u)
        if R_value23 == R_value23l:
            popt23 = popt23l; pcov23 = pcov23l
        if R_value23 == R_value23u:
            popt23 = popt23u; pcov23 = pcov23u

        try:
            popt22l, pcov22l = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02l, maxfev = np.inf)#, sigma = 0.1*amps)
            popt22u, pcov22u = curve_fit(P2_P2, times, amps, bounds=bounds_two,
            p0 = p02u, maxfev = np.inf)#, sigma = 0.1*amps)
        except:
            warnings.warn('curve_fit failed')
            return None
        fit22l = P2_P2(times, popt22l[0], popt22l[1], popt22l[2], popt22l[3],
        popt22l[4])
        fit22u = P2_P2(times, popt22u[0], popt22u[1], popt22u[2], popt22u[3],
        popt22u[4])
        ssres22l = np.sum((fit22l - amps)**2)
        ssres22u = np.sum((fit22u - amps)**2)
        # coefficient of determination, adjusted R^2:
        # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
        R_value22l = 1 - (ssres22l/sstot)*(len(times) - 1)/(len(times) -
            len(popt22l))
        R_value22u = 1 - (ssres22u/sstot)*(len(times) - 1)/(len(times) -
            len(popt22u))
        R_value22 = max(R_value22l, R_value22u)
        if R_value22 == R_value22l:
            popt22 = popt22l; pcov22 = pcov22l
        if R_value22 == R_value22u:
            popt22 = popt22u; pcov22 = pcov22u

        maxR2 = max(R_value33, R_value23, R_value22)

        if maxR2 < fit_thresh:
            warnings.warn('No curve fit gave adjusted R^2 value above '
            'fit_thresh')
            return None

        if maxR2 == R_value33:
            off = popt33[0]
            pc = popt33[1]
            tau = popt33[2]
            pc2 = popt33[3]
            tau2 = popt33[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov33))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P3 for tau
                a_tau = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                #P3 for tau2
                a_tau2 = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{3: [[pc, pc_err, tau, tau_err]]},
                        'b':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{3: [[pc, pc_err, tau, tau_err]]},
                        'a':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {3: [[pc, pc_err, tau, tau_err], [pc2, pc2_err,
                tau2, tau2_err]]}

        if maxR2 == R_value23:
            off = popt23[0]
            pc = popt23[1]
            tau = popt23[2]
            pc2 = popt23[3]
            tau2 = popt23[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov23))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P3 for tau2
                a_tau2 = t_a[max_a_ind[0]]/(2*np.log(2)/3)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[pc, pc_err, tau, tau_err]]},
                        'b':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[pc, pc_err, tau, tau_err]]},
                        'a':{3: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {2: [[pc, pc_err, tau, tau_err]],
                3: [[pc2, pc2_err, tau2, tau2_err]]}

        if maxR2 == R_value22:
            off = popt22[0]
            pc = popt22[1]
            tau = popt22[2]
            pc2 = popt22[3]
            tau2 = popt22[4]
            _, pc_err, tau_err, pc2_err, tau2_err  = \
                np.sqrt(np.diag(pcov22))
            if both_a != None:
                amp_a = both_a['amp']
                t_a = both_a['t']
                max_a_ind = np.where(amp_a == np.max(amp_a))[0]
                #P2 for tau
                a_tau = t_a[max_a_ind[0]]/np.log(3/2)
                #P2 for tau2
                a_tau2 = t_a[max_a_ind[0]]/np.log(3/2)
                if np.abs(a_tau - tau) <= np.abs(a_tau2 - tau2):
                    return {'a':{2: [[pc, pc_err, tau, tau_err]]},
                        'b':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
                else:
                    return {'b':{2: [[pc, pc_err, tau, tau_err]]},
                        'a':{2: [[pc2, pc2_err, tau2, tau2_err]]}}
            return {2: [[pc, pc_err, tau, tau_err],
                [pc2, pc2_err, tau2, tau2_err]]}

def fit_cs(taus, tau_errs, temps, cs_fit_thresh, E_min, E_max, cs_min, cs_max,
           input_T):
    """
    This function fits the cross section for holes (cs) for a given trap
    by curve-fitting release time constant (tau) vs temperature. Returns fit
    parameters and the release time constant at the desired input temperature.

    Args:
        taus (array): Array of tau values (in seconds).
        tau_errs (array): Array of tau uncertainty values (in seconds), with elements 
            in the same order as that of taus.
        temps (array): Array of temperatures (in Kelvin), with elements in the same 
            order as that of taus.
        cs_fit_thresh (float): The minimum value required for adjusted coefficient of 
            determination (adjusted R^2) for curve fitting for the capture cross section 
            for holes (cs) using data for tau vs temperature. The closer to 1, the better 
            the fit. Must be between 0 and 1.
        E_min (float): Lower bound for E (energy level in release time constant) for curve 
            fitting, in eV. Must be greater than or equal to 0.
        E_max (float): Upper bound for E (energy level in release time constant) for curve 
            fitting, in eV. Must be greater than E_min.
        cs_min (float): Lower bound for cs (capture cross section for holes in release time 
            constant) for curve fitting, in 1e-19 m^2. Must be greater than or equal to 0.
        cs_max (float): Upper bound for cs (capture cross section for holes in release time 
            constant) for curve fitting, in 1e-19 m^2. Must be greater than cs_min.
        input_T (float): Temperature of Roman EMCCD at which to calculate the release time 
            constant (in units of Kelvin). Must be greater than 0.

    Returns:
        E (float): Energy level (in eV).
        sig_E (float): Standard deviation error of energy level, in eV.
        cs (float): Cross section for holes, in cm^2.
        sig_cs (float): Standard deviation error of cross section for holes, in cm^2.
        Rsq (float): Adjusted R^2 for the tau vs temperature fit that was done to obtain cs.
        tau_input_T (float): Tau evaluated at desired temperature of Roman EMCCD, input_T, 
            in seconds.
        sig_tau_input_T (float): Standard deviation error of tau at desired temperature of 
            Roman EMCCD, input_T. Found by propagating error by utilizing sig_cs and sig_E, 
            in seconds.
    """

    # input checks
    try:
        taus = np.array(taus).astype(float)
    except:
        raise TypeError("taus elements should be real numbers")
    check.oneD_array(taus, 'taus', TypeError)
    try:
        temps = np.array(temps).astype(float)
    except:
        raise TypeError("temps elements should be real numbers")
    check.oneD_array(temps, 'temps', TypeError)
    if len(temps) != len(taus):
        raise ValueError('temps and taus must have the same length')
    try:
        tau_errs = np.array(tau_errs).astype(float)
    except:
        raise TypeError("tau_errs elements should be real numbers")
    check.oneD_array(tau_errs, 'tau_errs', TypeError)
    if len(tau_errs) != len(taus):
        raise ValueError('tau_errs and taus must have the same length')
    check.real_nonnegative_scalar(cs_fit_thresh, 'cs_fit_thresh', TypeError)
    if cs_fit_thresh > 1:
        raise ValueError('cs_fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(E_min, 'E_min', TypeError)
    check.real_nonnegative_scalar(E_max, 'E_max', TypeError)
    if E_max <= E_min:
        raise ValueError('E_max must be > E_min')
    check.real_nonnegative_scalar(cs_min, 'cs_min', TypeError)
    check.real_nonnegative_scalar(cs_max, 'cs_max', TypeError)
    if cs_max <= cs_min:
        raise ValueError('cs_max must be > cs_min')
    check.real_positive_scalar(input_T, 'input_T', TypeError)
    if len(np.unique(temps)) < 3:
        print('temps did not have a unique number of temperatures '
        'longer than the number of fitted parameters.')
        return None, None, None, None, None, None, None

    l_bounds = [E_min, cs_min]
    # typically, E is no more than 0.5 eV, and eV = 1.6e-19 J
    # cs usually 1e-15 to 1e-14 cm^2, or 1e-19 to 1e-18 m^2
    u_bounds = [E_max, cs_max]  # E in eV, cs in 1e-19 m^2
    bounds = (l_bounds, u_bounds)
    p0 = [(E_min + E_max)/2, (cs_min + cs_max)/2]
    # in case you have a perfect tau_err of 0 (highly unlikely), set it equal
    # to machine precision eps to prevent the curve_fit from crashing
    tau_errs[tau_errs==0] = np.finfo(float).eps

    try:
        # We consider absolute_sigma=True since we likely won't have many data
        # points, so sample variance of residuals may be low.  When it's False,
        # the tau_errs are scaled to match sample variance of residuals after
        # the fit, which would  most likely be a scaling down.  So when it's
        # True, the resulting std dev from pcov would likely be bigger and
        # maybe more accurate.  If we do have enough data points, the opposite
        # would be true.  So best thing to do is to take the case where the
        # error is bigger:
        poptt, pcovt = curve_fit(tau_temp, temps, taus, bounds = bounds, p0=p0,
                sigma = tau_errs, absolute_sigma=True, maxfev = np.inf)
        poptf, pcovf = curve_fit(tau_temp, temps, taus, bounds = bounds, p0=p0,
                sigma = tau_errs, absolute_sigma=False, maxfev = np.inf)
        if np.sum(np.sqrt(np.diag(pcovt))) >= np.sum(np.sqrt(np.diag(pcovf))):
            popt = poptt
            pcov = pcovt
        else:
            popt = poptf
            pcov = pcovf
    except:
            warnings.warn('curve_fit failed')
            return None, None, None, None, None, None, None
    E = popt[0] # in eV
    sig_E = np.sqrt(np.diag(pcov))[0]
    cs = popt[1]*1e-15 # in cm^2
    sig_cs = np.sqrt(np.diag(pcov))[1]*1e-15 # in cm^2
    tau_input_T = tau_temp(input_T, E, cs*1e15) # in s
    sig_tau_input_T = sig_tau_temp(input_T, E, cs*1e15, sig_E, sig_cs*1e15) #s

    fit = tau_temp(temps, popt[0], popt[1])
    ssres = np.sum((fit - taus)**2)
    sstot = np.sum((taus - np.mean(taus))**2)
    # coefficient of determination, adjusted R^2:
    # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
    Rsq = 1 - (ssres/sstot)*(len(temps) - 1)/(len(temps) - len(popt))

    if Rsq < cs_fit_thresh:
        print('Fitting of tau vs temperature has an adjusted R^2 '
        'value < cs_fit_thresh')

    return (E, sig_E, cs, sig_cs, Rsq, tau_input_T, sig_tau_input_T)


def tpump_analysis(input_dataset, time_head = 'TPTAU', 
                   mean_field = None, length_lim = 5,
    thresh_factor = 1.5, k_prob = 1, ill_corr = True, tfit_const = True,
    tau_fit_thresh = 0.8, tau_min = 0.7e-6, tau_max = 1.3e-2, tauc_min = 0,
    tauc_max = 1e-5, pc_min = 0, pc_max = 2, offset_min = 10,
    offset_max = 10,
    cs_fit_thresh = 0.8, E_min = 0, E_max = 1, cs_min = 0,
    cs_max = 50, bins_E = 100, bins_cs = 10, input_T = 180,
    sample_data = False,
    verbose=False, bin_size=10):
    """This function analyzes trap-pumped frames and outputs the location of
    each radiation trap (pixel and sub-electrode location within the pixel),
    everything needed to determine the release time constant at any temperature
    (the capture cross section for holes and the energy level), trap densities
    (i.e., how many traps per pixel for each kind of trap found), and
    information about the capture time constant (for potential future
    analysis).  This function only works as intended for the EMCCD with its
    electrodes' paticular electric potential shape that will
    be used for trap pumping on the Roman telescope.  It can find up to two
    traps per sub-electrode location per pixel.  The function has an option to
    save a preliminary output that takes a long time to create (the dictionary
    called 'temps') and an option to load that in and start the function at
    that point.

    The frames in the input_dataset should be SCI full frames that:
    - have had their bias subtracted 
    - have been corrected for nonlinearity
    - have been divided by EMGAIN

    The following parameters from the II&T Trap pumping code are stored in the
    object calibration file: 
    trap_densities : list
        A list of lists, where a list is provided for each type of trap.
        The trap density for a trap type is the # of traps in a given 2-D bin
        of E and tau divided by the total number of pixels in the image area.
        The binning by default is fine enough to distinguish all trap types
        found in the literarture. Only the bins that contain non-zero entries
        are returned here.  Each trap-type list is of the following format:
        [trap density, E, cs].
        E is the central value of the E bin, and cs is the central value of the
        cs bin. Stored in a hdu extention named 'trap_densities'
    bad_fit_counter : int
        Number of times trap_fit() provided fits of amplitude vs phase time
        that were below fit_thresh over all schemes and temperature (which
        would include any preliminary trap identifications that were rejected
        because they weren't consistent across all schemes for sub-electrode
        locations).
    pre_sub_el_count : int
        Number of times a trap was identified before filtering them through
        sub-electrode location.  The counter is over all schemes and
        temperatures.
    unused_fit_data : int or None
        Number of times traps were identified that were not matched up for
        sub-electrode location determination. 
    unused_temp_fit_data : int
        Number of times traps were identified that did not get used in
        identifying release time constant values across all temperatures
    two_or_less_count : int
        Number of traps that only appeared at 2 or fewer temperatures.
    noncontinuous_count : int
        Number of traps that appeared at a noncontinuous series of
        temperatures.

    Args:
        input_dataset (corgi.drp.Dataset): The input dataset to be analyzed. The dataset should be a stack of trap-pumped frames.
        time_head (str): Keyword corresponding to phase time for each FITS file. The keyword value is assumed to be a float (units of microseconds). Defaults to 'TPTAU'.
        mean_field (float, optional): The mean electron level that was present in each pixel before trap pumping was performed (excluding EM gain). Only useful if the mean level is less than num_pumps/4 e-. If num_pumps/4 e- or higher, use None.
        length_lim (int, optional): Minimum number of frames for which a dipole needs to meet the threshold to be considered a true trap. Defaults to 5.
        thresh_factor (float, optional): Number of standard deviations from the mean a dipole should stand out to be considered for a trap. Defaults to 1.5.
        k_prob (int, optional): The probability function used for finding the e-/DN factor. If the code fails with an exception, re-run the code with 2.  Defaults to 1.
        ill_corr (bool, optional): Whether to run local illumination correction on each trap-pumped frame. Defaults to True.
        tfit_const (bool, optional): Whether to use trap_fit_const() for curve fitting, treating the capture probability as constant. Defaults to True.
        tau_fit_thresh (float, optional): Minimum adjusted R^2 value required for curve fitting for the release time constant (tau). Defaults to 0.8.
        tau_min (float, optional): Lower bound value for tau (release time constant) for curve fitting, in seconds. Defaults to 0.7e-6.
        tau_max (float, optional): Upper bound value for tau (release time constant) for curve fitting, in seconds. Defaults to 1.3e-2.
        tauc_min (float, optional): Lower bound value for tauc (capture time constant) for curve fitting, in seconds. Only used if tfit_const = False. Defaults to 0.
        tauc_max (float, optional): Upper bound value for tauc (capture time constant) for curve fitting, in seconds. Only used if tfit_const = False. Defaults to 1e-5.
        pc_min (float, optional): Lower bound value for pc (capture probability) for curve fitting, in e-. Only used if tfit_const = True. Defaults to 0.
        pc_max (float, optional): Upper bound value for pc (capture probability) for curve fitting, in e-. Only used if tfit_const = True. Defaults to 2.
        offset_min (float, optional): Lower bound for the offset in the curve fit relative to 0 for fitting amplitude vs phase time. Defaults to 10.
        offset_max (float, optional): Upper bound for the offset in the curve fit relative to 0 for fitting amplitude vs phase time. Defaults to 10.
        cs_fit_thresh (float, optional): Minimum adjusted R^2 value required for curve fitting for the capture cross section for holes (cs) using data for tau vs temperature. Defaults to 0.8.
        E_min (float, optional): Lower bound for E (energy level in release time constant) for curve fitting, in eV. Defaults to 0.
        E_max (float, optional): Upper bound for E (energy level in release time constant) for curve fitting, in eV. Defaults to 1.
        cs_min (float, optional): Lower bound for cs (capture cross section for holes in release time constant) for curve fitting, in 1e-19 m^2. Defaults to 0.
        cs_max (float, optional): Upper bound for cs (capture cross section for holes in release time constant) for curve fitting, in 1e-19 m^2. Defaults to 50.
        bins_E (int, optional): Number of bins used for energy level in categorizing traps into 2-D bins of energy level and cross section. Defaults to 100.
        bins_cs (int, optional): Number of bins used for cross section in categorizing traps into 2-D bins of energy level and cross section. Defaults to 10.
        input_T (float, optional): Temperature of Roman EMCCD at which to calculate the release time constant (in units of Kelvin). Defaults to 180.
        sample_data (bool, optional): Whether to run the sample data on Alfresco. Defaults to False.
        verbose (bool, optional): Whether to print out additional information. Defaults to False.
        bin_size (int, optional): Side length of the square of pixels to consider for binning in illumination_correction(). If None, the square root of the smaller dimension (the smaller of the number of rows and number of cols) is used. Defaults to 10. 
            If a value bigger than the smaller dimension is input, then the size of the smaller dimension is used instead.  The optimal value for bin_size depends on the trap density, which is unknown, so in principle, 
            this function could be run several times with decreasing bin size until the maximum number of traps have been detected.
    
    Returns:
        corgi.drp.TrapCalibration: An object containing the results of the trap calibration. The trap densities are appended as an extension HDU, and several other parameters are stored as header keywords in the ext_hdr header.
    """
    
    # Make a copy of the input dataset to operate on
    working_dataset = input_dataset.copy()

    if type(sample_data) != bool:
        raise TypeError('sample_data should be True or False')
    if not sample_data: # don't need these for the Alfresco sample data
        if type(time_head) != str:
            raise TypeError('time_head must be a string')
    check.real_positive_scalar(thresh_factor, 'thresh_factor', TypeError)
    if mean_field is not None:
        check.real_positive_scalar(mean_field, 'mean_field', TypeError)
    check.positive_scalar_integer(length_lim, 'length_lim', TypeError)
    if k_prob != 1 and k_prob != 2:
        raise TypeError('k_prob must be either 1 or 2')
    if type(ill_corr) != bool:
        raise TypeError('ill_corr must be True or False')
    if type(tfit_const) != bool:
        raise TypeError('ill_corr must be True or False')
    check.real_nonnegative_scalar(tau_fit_thresh, 'tau_fit_thresh', TypeError)
    if tau_fit_thresh > 1:
        raise ValueError('tau_fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(tau_min, 'tau_min', TypeError)
    check.real_nonnegative_scalar(tau_max, 'tau_max', TypeError)
    if tau_max <= tau_min:
        raise ValueError('tau_max must be > tau_min')
    check.real_nonnegative_scalar(tauc_min, 'tauc_min', TypeError)
    check.real_nonnegative_scalar(tauc_max, 'tauc_max', TypeError)
    if tauc_max <= tauc_min:
        raise ValueError('tauc_max must be > tauc_min')
    check.real_nonnegative_scalar(pc_min, 'pc_min', TypeError)
    check.real_nonnegative_scalar(pc_max, 'pc_max', TypeError)
    if pc_max <= pc_min:
        raise ValueError('pc_max must be > pc_min')
    check.real_nonnegative_scalar(offset_min, 'offset_min', TypeError)
    check.real_nonnegative_scalar(offset_max, 'offset_max', TypeError)
    check.real_nonnegative_scalar(cs_fit_thresh, 'cs_fit_thresh', TypeError)
    if cs_fit_thresh > 1:
        raise ValueError('cs_fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(E_min, 'E_min', TypeError)
    check.real_nonnegative_scalar(E_max, 'E_max', TypeError)
    if E_max <= E_min:
        raise ValueError('E_max must be > E_min')
    check.real_nonnegative_scalar(cs_min, 'cs_min', TypeError)
    check.real_nonnegative_scalar(cs_max, 'cs_max', TypeError)
    if cs_max <= cs_min:
        raise ValueError('cs_max must be > cs_min')
    check.positive_scalar_integer(bins_E, 'bins_E', TypeError)
    check.positive_scalar_integer(bins_cs, 'bins_cs', TypeError)
    check.real_positive_scalar(input_T, 'input_T', TypeError)

    # to count how many apparent dipoles that
    #didn't meet the fit threshold
    bad_fit_counter = 0
    # to count how many fit attempts occured
    num_attempted_fits = 0
    # counter for number of traps before filtering them for sub-el location
    # (over all schemes and temperatures)
    pre_sub_el_count = 0
    unused_fit_data = None
    # count # times Pc is bigger than one (relevant when trap_fit_const() used)
    pc_bigger_1 = 0
    pc_biggest = 0
    # count # times emit time constant > 1e-2 or < 1e-6. Relevant only if
    # tau_min and tau_max are outside of 1e-6 to 1e-2.
    tau_outside = 0
    
    temps = {}

    #First we'll sort the input dataset by temperature
    dataset_list, dataset_temperatures = working_dataset.split_dataset(exthdr_keywords = ['EXCAMT'])
    
    # for dir in os.listdir(base_dir):
    for i,dataset in enumerate(dataset_list):
  
        curr_temp = dataset_temperatures[i]
        schemes = {}
        # initializing eperdn here; used if scheme is 1
        eperdn = 1

        scheme_header_keywords = ['TPSCHEM1','TPSCHEM2', 'TPSCHEM3','TPSCHEM4']
        scheme_datasets, scheme_list = dataset.split_dataset(exthdr_keywords = scheme_header_keywords)
        
        sch_list = []
        scheme_num_pumps = [] #the number of pumps for each scheme

        for i in range(len(scheme_datasets)):
            #Grab the first file's extension header from each dataset
            header0 = scheme_datasets[i][0].ext_hdr

            #Find the TPSCHEM keyword that is non-zero
            this_num_pumps = [header0[x] for x in scheme_header_keywords]
            this_num_pumps = np.array(scheme_list[i] )
            this_scheme = int(np.where(this_num_pumps != 0)[0].item()) + 1
            sch_list.append(this_scheme)

            #Grab the number of pumps from the first dataset. 
            scheme_num_pumps.append(this_num_pumps[this_scheme-1])

        #Sort the things so that Scheme 1 is first
        sch_order = np.argsort(sch_list)
        sch_list = np.array(sch_list)[sch_order]
        # Reorder the list of datasets using list comprehension instead of numpy array
        # (datasets can have different lengths, so can't convert to numpy array)
        scheme_datasets = [scheme_datasets[i] for i in sch_order]
        scheme_num_pumps = np.array(scheme_num_pumps)[sch_order]

        #Make a check to make sure that one of the schemes is Scheme 1
        if 1 not in sch_list:
            raise TPumpAnException('Scheme 1 files must run first for'
                    ' an accurate eperdn estimation')

        for curr_sch in sch_list: 
            # scheme_path = os.path.abspath(Path(sch_dir_path, sch_dir))
            # if os.path.isfile(scheme_path): # just want directories
                # continue
            # curr_sch = int(sch_dir[-1])
            # if Scheme_1 present, the following shouldn't happen
            if schemes == {} and curr_sch != 1:
                raise TPumpAnException('Scheme 1 files must run first for'
                    ' an accurate eperdn estimation')
            frames = []
            cor_frames = []
            timings = []

            #Grab the number of pumps associated with this scheme. 
            num_pumps = scheme_num_pumps[(sch_list == curr_sch)][0]
            
            # Get the dataset for this scheme (scheme_datasets is now a list, not numpy array)
            scheme_idx = np.where(sch_list == curr_sch)[0][0]
            for frame in scheme_datasets[scheme_idx]:

                #Get the data and the phase time - convert from us to seconds
                phase_time = float(frame.ext_hdr[time_head])/10**6

                timings.append(phase_time)   
                frames.append(frame.data)

            # no need for cosmic ray removal since we do ill. correction
            # plus cosmics wouldn't look same as it would on regular frame,
            # and they shouldn't affect the detection of dipoles (or very low chance)
            # This can also be mitigated by taking multiple frames per
            # phase time.
            # no need for flat-fielding since trap pumping will be done
            # while dark (i.e., a flat field of dark)
            # if frames == []:
            #     raise TPumpAnException('Scheme folder had no data '
            #     'files in it.')
            # if curr_sch is 1, then this simply multiplies by 1
            frames = np.stack(frames)*eperdn
            timings = np.array(timings)
            # min number of frames for a dipole should meet threshold so
            # that it is
            # considered a potential true trap, unless the number of
            # available frames is smaller
            length_limit = min(length_lim, len(np.unique(timings)))
            #To accurately determine eperdn, the median level needs to be
            #subtracted off if one is to compare, for P1,
            # 2500e- sitting on top of nothing; if it were sitting on
            # top of the mean field (i.e.,
            #whatever injected charge is there after bias subtraction), it
            #would be more than 2500e-.
            # illumination correction only subtracts and doesn't divide,
            # so e- units unaffected; the offset introduced taken care of
            # by nuisance parameter 'offset' in curve fitting.

            #And every time trap_id() called when local_ill_max
            # and local_ill_min is not
            # None, illumination_correction() has already been performed,
            # and eperdn has already been applied, so the units are e-.
            local_ill_frs = []
            # reasonable binsize
            nrows = frames[0].shape[0]
            ncols = frames[0].shape[1]
            small = min(nrows, ncols)
            if bin_size is not None:
                binsize = min(small, bin_size)
            else: 
                binsize = int(np.sqrt(small))
            for frame in frames:
                img, local_ill = illumination_correction(frame,
                    binsize = binsize, ill_corr=ill_corr)
                cor_frames.append(img)
                local_ill_frs.append(local_ill)
            local_ill_max = np.max(np.stack(local_ill_frs), axis = 0)
            local_ill_min = np.min(np.stack(local_ill_frs), axis = 0)
            cor_frames = np.stack(cor_frames)

            rc_above, rc_below, rc_both = trap_id(cor_frames,
                local_ill_min, local_ill_max, timings,
                thresh_factor = thresh_factor, length_limit=length_limit)

            # no minimum binsize b/c with each trap, there's a deficit and
            # a surplus of equal amounts about the flat field, which
            # doesn't change the median

            # could fit eperdn as a nuisance param k multiplying,
            # like k*Num_pumps*Pc*Pe, but some of k could bleed into
            # Pc (and possibly Pe, but Pe can only range from 0 to 1/4 or
            # 4/27 for scheme 1 or 2; so Pc should absorb it).
            # So for accurate fitting, need to find k first.
            # TODO v2: Make list of traps sorted by amp, and try
            # trap_fit_const() on trap with highest amp.  If a 1-trap is
            # good fit, assign k to be the factor by
            # which Pc is bigger than 1.  If instead a 2-trap fits, then
            # go to the next brightests trap, and repeat until you get a
            # 1-trap fit.  Then I need to change doc string to indicate
            # pc_max and pc_min still relevant even if tfit_const = False.
            if curr_sch == 1:
                max_a = [] # initialize
                for i in rc_above:
                    max_a.append(np.max(rc_above[i]['amps_above']))
                max_b = []
                for i in rc_below:
                    max_b.append(np.max(rc_below[i]['amps_below']))
                max_bo = []
                for i in rc_both:
                    max_bo.append(np.max(rc_both[i]['amps_both']))
                #peak_trap = max(max(max_a), max(max_b), max(max_bo))
                # to account for 2-traps, which are fewer and may not be
                # simply double the max trap amplitude, take median
                max_all = max_a + max_b + max_bo
                max_all = np.array(max_all)
                max_max_all = 0
                if len(max_all) != 0:
                    max_max_all = max(max_all)
                if len(max_all) == 0 or max_max_all <= 0:
                    raise TPumpAnException('No traps were found in '
                    'scheme {} at temperature {} K.'.format(curr_sch,
                    curr_temp))
                peak_trap = np.median(max_all)
                # eperdn applicable to all frames, a value for each temp
                # (which is expected to be the same for each temp)
                if k_prob == 1:
                    prob_factor_eperdn = 1/4
                if k_prob == 2:
                    prob_factor_eperdn = 4/27
                # if mean e- per pixel is lower than num_pumps/4, then max amp
                # is that mean e- per pixel amount
                if mean_field is not None:
                    max_e = min(num_pumps*1*prob_factor_eperdn, mean_field)
                else:
                    max_e = num_pumps*1*prob_factor_eperdn
                eperdn = max_e/peak_trap
                #convert to e-
                cor_frames *= eperdn

                for i in rc_above:
                    rc_above[i]['amps_above'] *= eperdn
                    rc_above[i]['loc_med_min'] *= eperdn
                    rc_above[i]['loc_med_max'] *= eperdn
                for i in rc_below:
                    rc_below[i]['amps_below'] *= eperdn
                    rc_below[i]['loc_med_min'] *= eperdn
                    rc_below[i]['loc_med_max'] *= eperdn
                for i in rc_both:
                    rc_both[i]['amps_both'] *= eperdn
                    rc_both[i]['loc_med_min'] *= eperdn
                    rc_both[i]['loc_med_max'] *= eperdn
            # below: short for no prob fit for scheme 1 above, below, both
            no_prob1a = 0
            no_prob1b = 0
            no_prob1bo = 0
            # delete coords that have None for trap_fit return
            del_a_list = []
            del_b_list = []
            del_bo_list = []
            # two- and one-trap counter over 'above', 'below', and 'both'
            #(for internal testing)
            two_trap_count = 0
            one_trap_count = 0
            # curve-fit 'above' traps
            fit_a_count = 0
            # of attempted fits that will be done
            num_attempted_fits += len(rc_above)+len(rc_below)+len(rc_both)
            for i in rc_above:
                loc_med_min = rc_above[i]['loc_med_min']
                loc_med_max = rc_above[i]['loc_med_max']
                # set average offset to expected level after
                #illumination_correction(), which is 0 (b/c no negative
                # counts), by subtracting
                loc_avg = (loc_med_min+loc_med_max)/2
                off_min = 0 - max(offset_min,np.abs(loc_avg - loc_med_max))
                off_max = 0 + max(offset_max,np.abs(loc_med_min - loc_avg))
                if not tfit_const:
                    fd = trap_fit(curr_sch, rc_above[i]['amps_above'],
                        timings, num_pumps, tau_fit_thresh, tau_min,
                        tau_max, tauc_min, tauc_max, off_min,
                        off_max) #, k_min, k_max)
                if tfit_const:
                    fd = trap_fit_const(curr_sch,
                        rc_above[i]['amps_above'], timings, num_pumps,
                        tau_fit_thresh, tau_min, tau_max, pc_min, pc_max,
                        off_min, off_max)
                if fd is None:
                    bad_fit_counter += 1
                    print('bad fit above: ', i)
                    del_a_list.append(i)
                if fd is not None:
                    fit_a_count += 1
                    # find out how many pixels were fitted for 2 traps
                    #(for internal testing)
                    temp_2_count = 0
                    for key in fd.keys():
                        # only meaningful if trap_fit_const() used
                        #(for internal testing)
                        # Pc - Pc_err > 1:
                        if fd[key][0][0]-fd[key][0][1] > 1:
                            pc_bigger_1 += 1
                            if fd[key][0][0]+fd[key][0][1] > pc_biggest:
                                pc_biggest = fd[key][0][0]+fd[key][0][1]
                        # only meaningful if tau bounds outside 1e-6, 1e-2
                        #(for internal testing)
                        #tau + tau_err < 1e-6, tau - tau_err < 1e-2:
                        if fd[key][0][2]+fd[key][0][3] < 1e-6 or \
                            fd[key][0][2]-fd[key][0][3] > 1e-2:
                            tau_outside += 1
                        if len(fd[key]) == 2:
                            if fd[key][1][0]-fd[key][1][1] > 1:
                                pc_bigger_1 += 1
                                if fd[key][1][0]+fd[key][1][1] >pc_biggest:
                                    pc_biggest =fd[key][1][0]+fd[key][1][1]
                            if fd[key][1][2]+fd[key][1][3] < 1e-6 or \
                                fd[key][1][2]-fd[key][1][3] > 1e-2:
                                tau_outside += 1
                        temp_2_count += len(fd[key])
                    if temp_2_count == 2:
                        two_trap_count += 1
                        # overall trap count, before sub-el location
                        pre_sub_el_count += 2
                    if temp_2_count == 1:
                        one_trap_count += 1
                        # overall trap count, before sub-el location
                        pre_sub_el_count += 1
                    # check if no prob func 1 fits found
                    if curr_sch == 1 and k_prob not in fd:
                        no_prob1a += 1
                    rc_above[i]['fit_data_above'] = fd
            for no_fit_coord in del_a_list:
                rc_above.__delitem__(no_fit_coord)
            # curve-fit 'below' traps
            fit_b_count = 0
            for i in rc_below:
                loc_med_min = rc_below[i]['loc_med_min']
                loc_med_max = rc_below[i]['loc_med_max']
                # set average offset to expected level after
                #illumination_correction(), which is 0 (b/c no negative
                # counts), by subtracting
                loc_avg = (loc_med_min+loc_med_max)/2
                off_min = 0 - max(offset_min,np.abs(loc_avg - loc_med_max))
                off_max = 0 + max(offset_max,np.abs(loc_med_min - loc_avg))
                if not tfit_const:
                    fd = trap_fit(curr_sch, rc_below[i]['amps_below'],
                        timings, num_pumps, tau_fit_thresh, tau_min,
                        tau_max, tauc_min, tauc_max, off_min,
                        off_max) #, k_min, k_max)
                if tfit_const:
                    fd = trap_fit_const(curr_sch,
                        rc_below[i]['amps_below'], timings, num_pumps,
                        tau_fit_thresh, tau_min, tau_max, pc_min, pc_max,
                        off_min, off_max)
                if fd is None:
                    bad_fit_counter += 1
                    print('bad fit below: ', i)
                    del_b_list.append(i)
                if fd is not None:
                    fit_b_count += 1
                    # find out how many pixels were fitted for 2 traps
                    temp_2_count = 0
                    for key in fd.keys():
                        # only meaningful if trap_fit_const() used
                        #(for internal testing)
                        #Pc - Pc_err >1:
                        if fd[key][0][0]-fd[key][0][1] > 1:
                            pc_bigger_1 += 1
                            if fd[key][0][0]+fd[key][0][1] > pc_biggest:
                                pc_biggest = fd[key][0][0]+fd[key][0][1]
                        # only meaningful if tau bounds outside 1e-6, 1e-2
                        #(for internal testing)
                        #tau + tau_err < 1e-6, tau - tau_err < 1e-2
                        if fd[key][0][2]+fd[key][0][3] < 1e-6 or \
                            fd[key][0][2]-fd[key][0][3] > 1e-2:
                            tau_outside += 1
                        if len(fd[key]) == 2:
                            if fd[key][1][0]-fd[key][1][1] > 1:
                                pc_bigger_1 += 1
                                if fd[key][1][0]+fd[key][1][1] >pc_biggest:
                                    pc_biggest =fd[key][1][0]+fd[key][1][1]
                            if fd[key][1][2]+fd[key][1][3] < 1e-6 or \
                                fd[key][1][2]-fd[key][1][3] > 1e-2:
                                tau_outside += 1
                        temp_2_count += len(fd[key])
                    if temp_2_count == 2:
                        two_trap_count += 1
                        # overall trap count, before sub-el location
                        pre_sub_el_count += 2
                    if temp_2_count == 1:
                        one_trap_count += 1
                        # overall trap count, before sub-el location
                        pre_sub_el_count += 1
                    # check if no prob func 1 fits found
                    if curr_sch == 1 and k_prob not in fd:
                        no_prob1b += 1
                    rc_below[i]['fit_data_below'] = fd
            for no_fit_coord in del_b_list:
                rc_below.__delitem__(no_fit_coord)
            # curve-fit 'both' traps
            fit_bo_count = 0
            for i in rc_both:
                loc_med_min = rc_both[i]['loc_med_min']
                loc_med_max = rc_both[i]['loc_med_max']
                # set average offset to expected level after
                #illumination_correction(), which is 0 (b/c no negative
                # counts), by subtracting
                loc_avg = (loc_med_min+loc_med_max)/2
                off_min = 0 - max(offset_min,np.abs(loc_avg - loc_med_max))
                off_max = 0 + max(offset_max,np.abs(loc_med_min - loc_avg))
                if not tfit_const:
                    fd = trap_fit(curr_sch, rc_both[i]['amps_both'],
                        timings, num_pumps, tau_fit_thresh, tau_min,
                        tau_max, tauc_min, tauc_max, off_min,
                        off_max, #k_min, k_max,
                        both_a = rc_both[i]['above'])
                if tfit_const:
                    fd = trap_fit_const(curr_sch, rc_both[i]['amps_both'],
                        timings, num_pumps, tau_fit_thresh, tau_min,
                        tau_max, pc_min, pc_max, off_min,
                        off_max, both_a = rc_both[i]['above'])
                if fd is None:
                    bad_fit_counter += 1
                    print('bad fit both: ', i)
                    del_bo_list.append(i)
                if fd is not None:
                    fit_bo_count += 1
                    # all pixels in 'both' are 2-traps
                    two_trap_count += 1
                    # overall trap count, before sub-el location
                    pre_sub_el_count += 2
                    temp_2_count = 0
                    for val in fd.values():
                        for pval in val.values():
                        # only meaningful if trap_fit_const() used
                        #(for internal testing)
                        #Pc - Pc_err > 1:
                            if pval[0][0]-pval[0][1] > 1:
                                pc_bigger_1 += 1
                                if pval[0][0]+pval[0][1] > pc_biggest:
                                    pc_biggest = pval[0][0]+pval[0][1]
                            # meaningful if tau bounds outside 1e-6, 1e-2
                            #(for internal testing)
                            #tau + tau_err < 1e-6, tau - tau_err < 1e-2:
                            if pval[0][2]+pval[0][3] < 1e-6 or \
                                pval[0][2]-pval[0][3] > 1e-2:
                                tau_outside += 1
                    # check if no prob func 1 fits found
                    if curr_sch == 1 and (k_prob not in fd['a']) and \
                        (k_prob not in fd['b']):
                        no_prob1bo += 1
                    rc_both[i]['fit_data_both'] = fd
                    # make new dictionary for this coord in 'above' since
                    # by construction it doesn't exist as an 'above'
                    # entry yet
                    rc_above[i] = {'fit_data_above': fd['a'],
                        'amps_above': rc_both[i]['above']['amp']}
                    # similarly for 'below'
                    rc_below[i] = {'fit_data_below': fd['b'],
                        'amps_below': rc_both[i]['below']['amp']}
            for no_fit_coord in del_bo_list:
                rc_both.__delitem__(no_fit_coord)

            if verbose: 
                print('temperature: ', curr_temp, ', scheme: ', curr_sch,
                    ', number of two-trap pixels (before sub-electrode '
                    'location): ', two_trap_count, ', number of one-trap '
                    'pixels (before sub-electrode location): ',
                    one_trap_count, ', eperdn: ', eperdn)
                print('above traps found: ', rc_above.keys())
                print('below traps found: ', rc_below.keys())
                print('both traps found: ', rc_both.keys())
                print('bad fit counter: ', bad_fit_counter, '_____________')
                
            if curr_sch == 1 and \
                (fit_a_count + fit_b_count + fit_bo_count -
                (no_prob1a + no_prob1b + no_prob1bo)) == 0:
                #so no traps for prob 1 in scheme 1, so eperdn should be
                # set according to prob 2 in scheme 1
                tot = fit_a_count + fit_b_count + fit_bo_count
                # could still be traps in barrier electrodes 1 and 3, but
                # realistically none if this fails with both choices of
                # k_prob
                raise TPumpAnException('No traps for probability function '
                '1 found under scheme 1. Re-run with prob_factor_eperdn '
                '= 4/27 and k_prob = 2. If that failed, no traps '
                'present in frame. Total # of trap pixels found for '
                'scheme 1 at current temperature: ', tot)
            #TODO v2.0: COULD happen that you happen traps in schemes 2, 3,
            #  and 4 but none in sch 1.  Could account for this, but
            # wouldn't have very good statistics in schemes 3 and 4 for
            # traps that approach Pc ~ 1.  In any case, no traps in sch 1
            # would be very rare.
            # collecting in convenient dictionary for further manipulation
            schemes[curr_sch] = {'rc_above': rc_above,
                'rc_below': rc_below, 'rc_both': rc_both,
                'timings': timings}
        # for a given scheme, the bright pixel coords for "above" CAN
        # be the same as those for "below" (for different ranges of phase
        # time or overlapping phase times)
        traps = {}
        # rc_both has been split up into rc_above and rc_below
        # For a given scheme, rc_below, rc_above, rc_both: each has
        # unique coord keys, but across dictionaries, there is overlap now
        # for the 'both' coords

        # getting coords that are shared across schemes
        def _el_loc_coords2(sch1, or1, sch2, or2):
            """Gets coordinates of dipoles shared across 2 schemes (sch1
            and sch2) with orientations or1 and or2 at a
            given temperature.
            For the purpose of sub-electrode location.
            
            Args:
                sch1 (int): The first scheme number.
                or1 (str): The orientation of the first scheme.
                sch2 (int): The second scheme number.
                or2 (str): The orientation of the second scheme.
                
                Returns:
                    list: The coordinates of the dipoles shared across the two schemes.
            """
            coords = list(set(schemes[sch1]['rc_'+or1]) -
                (set(schemes[sch1]['rc_'+or1]) -
                set(schemes[sch2]['rc_'+or2])))
            return coords

        def _el_loc_coords3(sch1, or1, sch2, or2, sch3, or3):
            """Gets coordinates of dipoles shared across 3 schemes (sch1,
            sch2, and sch3) with orientations or1 and or2 and or3 at a
            given temperature.
            For the purpose of sub-electrode location.
            
            Args:
                sch1 (int): The first scheme number.
                or1 (str): The orientation of the first scheme.
                sch2 (int): The second scheme number.
                or2 (str): The orientation of the second scheme.
                sch3 (int): The third scheme number.
                or3 (str): The orientation of the third scheme.
                
                Returns:
                    list: The coordinates of the dipoles shared across the three schemes.
            """
            coords12 = _el_loc_coords2(sch1, or1, sch2, or2)
            coords23 = _el_loc_coords2(sch2, or2, sch3, or3)
            coords123 = list(set(coords12) - (set(coords12) -
                set(coords23)))
            return coords123

        def _el_loc_2(sch1, or1, prob1, sch2, or2, prob2, i, subel_loc):
            """At a given temperature, for coordinate i shared across 2
            schemes (sch1 and sch2), match up dipoles that meet the
            orientation (or1 and or2) and probability function
            specifications (prob1 and prob2) for a sub-electrode location
            (subel_loc) with release time constant (tau) values and
            uncertainties such that uncertainty window across both schemes
            is minimized (thus identifying these data with the same
            physical trap).   It also outputs capture probability info
            which may be useful for future analysis.  It appends all this
            to the traps dictionary and deletes these dipole entries from
            the schemes dictionary so that they aren't used for matching up
            again.

            Args:
                sch1 (int): The first scheme number.
                or1 (str): The orientation of the first scheme.
                prob1 (str): The probability function of the first scheme.
                sch2 (int): The second scheme number.
                or2 (str): The orientation of the second scheme.
                prob2 (str): The probability function of the second scheme.
                i (int): The coordinate index.
                subel_loc (int): The sub-electrode location.
            """
            max_amp1 = max(schemes[sch1]['rc_'+or1][i]['amps_'+or1])
            max_amp2 = max(schemes[sch2]['rc_'+or2][i]['amps_'+or2])
            fd1 = schemes[sch1]['rc_'+or1][i]['fit_data_'+or1]
            fd2 = schemes[sch2]['rc_'+or2][i]['fit_data_'+or2]
            if prob1 in fd1 and prob2 in fd2:
                # to account for the case of 2 same sub-el traps for a
                # given pixel:
                iterations = min(len(fd1[prob1]), len(fd2[prob2]))
                for iteration in range(iterations):
                    # initialize minimum difference b/w lower and upper
                    # bounds on tau
                    min_range = np.inf
                    j0 = None
                    k0 = None
                    tau_avg = None
                    tau_up = None
                    for j in fd1[prob1]:
                        for k in fd2[prob2]:
                            # find pair that minimizes overall uncertainty
                            # window; it's okay if individual tau windows
                            # don't overlap since systematic errors may
                            # have affecte them
                            #index 2: corresponds to tau
                            #index 3: corresponds to tau_err
                            up = max(j[2] + j[3], k[2] + k[3])
                            low = min(j[2] - j[3], k[2] - k[3])
                            if (up-low) < min_range:
                                min_range = up - low
                                j0 = j
                                k0 = k
                                tau_avg = (up + low)/2
                                tau_up = up
                    # TODO v2.0:  seek absolute best pairings by
                    # across all traps in a given pixel by
                    # earmarking the fit data that have been
                    # selected above, and after full matching-
                    # up process, compare the earmarked ones
                    # to determine optimal sets so that the reported
                    # uncertainties for each trap's tau is overall
                    # minimized.
                    # Could append to
                    # each selected fit data list the j and k values of the
                    # other fit data set(s) it was matched with.
                    if (j0 is not None) and (k0 is not None) and (tau_avg
                        is not None) and (tau_up is not None):
                        #  Need amps in traps output? Sure, max amp value,
                        # so that the starting charge packet and the max
                        # amp after all the pumps gives a range of charge
                        # packet value corresponding to tau values
                        # (even though the fitting itself assumed constant
                        # charge packet value).
                        cap1 = j0[0]
                        cap1_err = j0[1]
                        cap2 = k0[0]
                        cap2_err = k0[1]
                        # pixel could have more than one trap,
                        #so trap[i] could already exist at given iteration
                        if i not in traps:
                            traps[i] = []
                        traps[i].append([subel_loc, tau_avg,
                                        tau_up - tau_avg,
                                        [cap1, cap1_err, max_amp1, cap2,
                                        cap2_err, max_amp2], iteration])
                        # now remove these trap data b/c they've been
                        #  matched
                        fd1[prob1].remove(j0)
                        fd2[prob2].remove(k0)

        def _el_loc_3(sch1, or1, prob1, sch2, or2, prob2, sch3, or3, prob3,
            i, subel_loc):
            """At a given temperature, for coordinate i shared across 3
            schemes (sch1, sch2, and sch3), match up dipoles that meet the
            orientation (or1,  or2, and or3) and probability function
            specifications (prob1, prob2, and prob3) for a sub-electrode
            location (subel_loc) with release time constant (tau) values
            and uncertainties such that uncertainty window across the
            schemes is minimized (thus identifying these data with the same
            physical trap).   It also outputs capture probability info
            which may be useful for future analysis.  It appends all this
            to the traps dictionary and deletes these dipole entries from
            the schemes dictionary so that they aren't used for matching up
            again.

            Args:
                sch1 (int): The first scheme number.
                or1 (str): The orientation of the first scheme.
                prob1 (str): The probability function of the first scheme.
                sch2 (int): The second scheme number.
                or2 (str): The orientation of the second scheme.
                prob2 (str): The probability function of the second scheme.
                sch3 (int): The third scheme number.
                or3 (str): The orientation of the third scheme.
                prob3 (str): The probability function of the third scheme.
                i (int): The coordinate index.
                subel_loc (int): The sub-electrode location.
            """
            max_amp1 = max(schemes[sch1]['rc_'+or1][i]['amps_'+or1])
            max_amp2 = max(schemes[sch2]['rc_'+or2][i]['amps_'+or2])
            max_amp3 = max(schemes[sch3]['rc_'+or3][i]['amps_'+or3])
            fd1 = schemes[sch1]['rc_'+or1][i]['fit_data_'+or1]
            fd2 = schemes[sch2]['rc_'+or2][i]['fit_data_'+or2]
            fd3 = schemes[sch3]['rc_'+or3][i]['fit_data_'+or3]
            if prob1 in fd1 and prob2 in fd2 and prob3 in fd3:
                # to account for the case of 2 same sub-el traps for a
                # given pixel:
                iterations = min(len(fd1[prob1]), len(fd2[prob2]))
                for iteration in range(iterations):
                    # initialize minimum difference b/w central values
                    min_range = np.inf
                    j0 = None
                    k0 = None
                    l0 = None
                    tau_avg = None
                    tau_up = None
                    for j in fd1[prob1]:
                        for k in fd2[prob2]:
                            for l in fd3[prob3]:
                                # find pair that minimizes overall
                                # uncertainty window; it's okay if
                                # individual tau windows
                                #  don't overlap since systematic errors
                                # may have affected them
                                #index 2: corresponds to tau
                                #index 3: corresponds to tau_err
                                up = max(j[2] + j[3], k[2] + k[3],
                                    l[2] + l[3])
                                low= min(j[2] - j[3], k[2] - k[3],
                                    l[2] - l[3])
                                if(up - low) < min_range:
                                    min_range = up - low
                                    j0 = j
                                    k0 = k
                                    l0 = l
                                    tau_avg = (up + low)/2
                                    tau_up = up
                    # TODO v2.0:  seek absolute best pairings by
                    # across all traps in a given pixel by
                    # earmarking the fit data that have been
                    # selected above, and after full matching
                    # up process, compare the earmarked ones
                    # (along with any fit data that didn't get
                    # chosen for traps) to determine optimal
                    # pairings so that the least amount of fit
                    # data goes unused for trap locations.  Could append to
                    # each selected fit data list the j and k values of the
                    # other fit data set(s) it was matched with.
                    if (j0 is not None) and (k0 is not None) and (l0
                        is not None) and (tau_avg is not None) and (tau_up
                        is not None):

                        cap1 = j0[0]
                        cap1_err = j0[1]
                        cap2 = k0[0]
                        cap2_err = k0[1]
                        cap3 = l0[0]
                        cap3_err = l0[1]
                        # pixel could have more than one trap,
                        #so trap[i] could already exist at given iteration
                        if i not in traps:
                            traps[i] = []
                        traps[i].append([subel_loc, tau_avg,
                                        tau_up - tau_avg,
                                        [cap1, cap1_err, max_amp1, cap2,
                                        cap2_err, max_amp2, cap3, cap3_err,
                                        max_amp3], iteration])
                        # now remove these trap data b/c they've been
                        # matched
                        fd1[prob1].remove(j0)
                        fd2[prob2].remove(k0)
                        fd3[prob3].remove(l0)

        # go through all 3-scheme sub-electrode locations first, since
        # there could be scenarios where a pixel has 2-traps in different
        # schemes that can match a 2-scheme ID and rob from a true 3-scheme
        #only do these operations if all 4 schemes present
        if set([1,2,3,4]) == set(schemes.keys()):
            # LHS of electrode 2: sch 1 'above', sch 2 'below', sch 4 'above'
            LHSel2_coords = _el_loc_coords3(1, 'above', 2, 'below',
                4, 'above')
            # RHS of electrode 2
            RHSel2_coords = _el_loc_coords3(1, 'above', 2, 'below',
                4, 'below')
            # LHS of electrode 4
            LHSel4_coords = _el_loc_coords3(1, 'below', 2, 'above',
                3, 'above')
            # RHS of electrode 4
            RHSel4_coords = _el_loc_coords3(1, 'below', 2, 'above',
                3, 'below')
            for i in LHSel2_coords:
                # sch1 'above' prob2, sch2 'below' prob1, sch4 'above'prob3
                _el_loc_3(1, 'above', 2, 2, 'below', 1, 4, 'above', 3,
                    i, 'LHSel2')
            for i in RHSel2_coords:
                _el_loc_3(1, 'above', 1, 2, 'below', 2, 4, 'below', 3,
                    i, 'RHSel2')
            for i in LHSel4_coords:
                _el_loc_3(1, 'below', 1, 2, 'above', 2, 3, 'above', 3,
                    i, 'LHSel4')
            for i in RHSel4_coords:
                _el_loc_3(1, 'below', 2, 2, 'above', 1, 3, 'below', 3,
                    i, 'RHSel4')
            # now go through all 2-scheme sub-electrode locations
            # LHS of electrode 1: scheme 1 'below' and scheme 3 'below'
            LHSel1_coords = _el_loc_coords2(1, 'below', 3, 'below')
            # center of electrode 1
            CENel1_coords = _el_loc_coords2(3, 'below', 4, 'above')
            # RHS of electrode 1
            RHSel1_coords = _el_loc_coords2(1, 'above', 4, 'above')
            # center of electrode 2
            CENel2_coords = _el_loc_coords2(1, 'above', 2, 'below')
            # LHS of electrode 3
            LHSel3_coords = _el_loc_coords2(2, 'below', 4, 'below')
            # center of electrode 3
            CENel3_coords = _el_loc_coords2(3, 'above', 4, 'below')
            # RHS of electrode 3
            RHSel3_coords = _el_loc_coords2(2, 'above', 3, 'above')
            # center of electrode 4
            CENel4_coords = _el_loc_coords2(1, 'below', 2, 'above')
            for i in LHSel1_coords:
                # scheme 1 'below' prob 1, scheme 3 'below' prob 3
                _el_loc_2(1, 'below', 1, 3, 'below', 3, i, 'LHSel1')
            for i in CENel1_coords:
                _el_loc_2(3, 'below', 2, 4, 'above', 2, i, 'CENel1')
            for i in RHSel1_coords:
                _el_loc_2(1, 'above', 1, 4, 'above', 3, i, 'RHSel1')
            for i in CENel2_coords:
                _el_loc_2(1, 'above', 1, 2, 'below', 1, i, 'CENel2')
            for i in LHSel3_coords:
                _el_loc_2(2, 'below', 1, 4, 'below', 3, i, 'LHSel3')
            for i in CENel3_coords:
                _el_loc_2(3, 'above', 2, 4, 'below', 2, i, 'CENel3')
            for i in RHSel3_coords:
                _el_loc_2(2, 'above', 1, 3, 'above', 3, i, 'RHSel3')
            for i in CENel4_coords:
                _el_loc_2(1, 'below', 1, 2, 'above', 1, i, 'CENel4')


        # see how many dipoles were not matched up in sub-electrode
        # location
        unused_fit_data = 0
        for sch in schemes.values():
            for coord in sch['rc_above'].values():
                unused_fit_data += len(coord['fit_data_above'])
            for coord in sch['rc_below'].values():
                unused_fit_data += len(coord['fit_data_below'])

        temps[curr_temp] = traps
 
    # initializing
    trap_list = []
    for temp in temps.values():
        for pix, trs in temp.items():
            for tr in trs:
                # including tr[-1], which is 'iteration' since there could be
                #more than 1 trap in same sub-el location of a given pixel
                if (pix, tr[0], tr[-1]) not in trap_list:
                    trap_list.append((pix, tr[0], tr[-1]))
    # structural formats below:
    # trap_list = [((row,col), 'sub_el', iteration), ...]
    #temps = {T1: {(row,col): [['sub_el', tau, sigma_tau,
    # [cap info], iteration],..],..},...}
    # gives an list of temperatures in ascending order so that the selection of
    # tau values to match up in a given pixel and sub-el location possible
    sorted_temps = sorted(temps)
    #trap_dict = {((row,col),'sub_el',iteration): {'T': [temps],
    # 'tau': [corresponding taus], 'sigma_tau': [simga_taus],
    # 'cap':[[cap info at T1], [cap info at T2],..]},..}
    # initializing
    trap_dict = {}
    # pick out traps in same pixel and sub-el location, and for each next temp
    # step, the tau value that's closest to previous (i.e., that falls along
    # same tau vs temp curve) since there could be multiple traps in the same
    # sub-el location with different tau values
    for pix_el in trap_list:
        for temp in sorted_temps:
            if pix_el[0] in temps[temp]: #if coord is present in this temp
                dist_tau = np.inf # initialize
                tr0 = 0 # initialize
                tr1 = 0 # initialize
                for tr in temps[temp][pix_el[0]]:
                    if tr[0] == pix_el[1]: # if same sub-el location
                        if pix_el not in trap_dict:
                            trap_dict[pix_el] = {'T': [temp], 'tau': [tr[1]],
                            'sigma_tau': [tr[2]], 'cap': [tr[3]]}
                            tr0 = tr
                            break
                            #Euclidean distance to next tau in tau-temp space
                        elif ((trap_dict[pix_el]['T'][-1] - temp)**2 +
                            (trap_dict[pix_el]['tau'][-1] - tr[1])**2) < \
                                dist_tau:
                            dist_tau = ((trap_dict[pix_el]['T'][-1] - temp)**2
                            + (trap_dict[pix_el]['tau'][-1] - tr[1])**2)
                            tr1 = tr
                if tr0 != 0:
                    temps[temp][pix_el[0]].remove(tr0)
                if tr1 != 0:
                    trap_dict[pix_el]['T'].append(temp)
                    trap_dict[pix_el]['tau'].append(tr1[1])
                    trap_dict[pix_el]['sigma_tau'].append(tr1[2])
                    trap_dict[pix_el]['cap'].append(tr1[3])
                    temps[temp][pix_el[0]].remove(tr1)
    #see how many unused sets of fit data left when getting taus for
    # each temp (physically, should be 0 if all were associated with traps)
    unused_temp_fit_data = 0
    for temp in temps.values():
        for rc in temp.values():
            unused_temp_fit_data += len(rc)
    # count how many traps only have data for 2 or fewer temperatures
    two_or_less_count = 0
    # count how many traps have gaps in the temperatures (a trap should show
    # up over a continuous range of temps)
    noncontinuous_count = 0
    # lists for making histogram to categorize for outputting trap densities
    E_vals = []
    cs_vals = []
    for pix_el in trap_dict.values():
        if len(pix_el['T']) <= 2:
            two_or_less_count += 1
        T1 = pix_el['T'][0]
        Tf = pix_el['T'][-1]
        if len(pix_el['T']) - 1 != (sorted_temps.index(Tf) -
            sorted_temps.index(T1)):
            noncontinuous_count += 1

        taus = np.array(pix_el['tau'])
        tau_errs = np.array(pix_el['sigma_tau'])
        temperatures = np.array(pix_el['T'])
        (E, sig_E, cs, sig_cs, Rsq, tau_input_T,
         sig_tau_input_T) = fit_cs(taus, tau_errs,
            temperatures, cs_fit_thresh, E_min, E_max, cs_min, cs_max, input_T)
        pix_el['E'] = E # in eV
        pix_el['sig_E'] = sig_E # in eV
        pix_el['cs'] = cs # in cm^2
        pix_el['sig_cs'] = sig_cs # in cm^2
        pix_el['Rsq'] = Rsq
        pix_el['tau at input T'] = tau_input_T # in s
        pix_el['sig_tau at input T'] = sig_tau_input_T # in s
        if Rsq is not None:
            if Rsq >= cs_fit_thresh:
                E_vals.append(E)
                cs_vals.append(cs)
    E_vals = np.array(E_vals)
    cs_vals = np.array(cs_vals)
    #originally used for default: bins = [int(1e-19/1e-21) , int(1e-17/1e-20)]
    bins = [bins_E, bins_cs]

    trap_densities = []
    if len(E_vals) != 0 and len(cs_vals) != 0:
        H, E_edges, cs_edges = np.histogram2d(E_vals, cs_vals, bins = bins,
            range = [[0, max(1, E_vals.max()) + 0.5/bins[0]],
                [0, max(1e-14, cs_vals.max()) + 1e-14*0.5/bins[1]]])

        nrows, ncols, _ = imaging_area_geom(dataset[0].ext_hdr['ARRTYPE'])
        for i in range(bins[0]):
            for j in range(bins[1]):
                # [percentage of traps out of total # pixels, avg E bin value,
                # avg cs bin value]
                if H[i, j] != 0:
                    trap_densities.append([H[i, j]/(nrows*ncols),
                        (E_edges[i+1] + E_edges[i])/2,
                        (cs_edges[j+1] + cs_edges[j])/2])

    
    trapcal = create_TrapCalibration_from_trap_dict(trap_dict, input_dataset)

    #Add all of the old default outputs as header keywords or extensions
    trapcal.add_extension_hdu('trap_densities', data = np.array(trap_densities))
    trapcal.ext_hdr['badfitct'] = (bad_fit_counter , 'bad_fit_counter')
    trapcal.ext_hdr['prsbelct'] = (pre_sub_el_count, 'pre_sub_el_count')
    trapcal.ext_hdr['unfitdat'] = (unused_fit_data, 'unused_fit_data')
    trapcal.ext_hdr['untempfd'] = (unused_temp_fit_data, 'unused_temp_fit_data')
    trapcal.ext_hdr['twoorles'] = (two_or_less_count, 'two_or_less_count')
    trapcal.ext_hdr['noncontc'] = (noncontinuous_count, 'noncontinuous_count')
    
    return trapcal

    # return (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
    #     unused_fit_data, unused_temp_fit_data, two_or_less_count,
    #     noncontinuous_count)


def create_TrapCalibration_from_trap_dict(trap_dict,input_dataset):
    '''
    A function that converts a trap dictionary into a corgidrp.data.TrapCalibration
    file. 

    The trap dictionary is defined as follows: A dictionary with a key for each trap for which acceptable fits for
    the release time constant (tau) for all schemes could be made.  It has
    the following format, and an example entry for a
    pixel at location (row, col) on the right-hand side (RHS) of
    electrode 2 containg 1 of possibly two traps is shown.  The 0 in the
    key denotes that this is the first trap found at this pixel and
    sub-electrode location.  If a 2nd trap was found at this same location,
    another trap with a 1 in the key would be present in trap_dict.
    trap_dict = {
        ((row, col), 'RHSel2', 0): {'T': [160, 162, 164, 166, 168],
        'tau': [1.51e-6, 1.49e-6, 1.53e-6, 1.50e-6, 1.52e-6],
        'sigma_tau': [2e-7, 1e-7, 1e-7, 2e-7, 1e-7],
        'cap': [[cap1, cap1_err, max_amp1, cap2, cap2_err, max_amp2], ...],
        'E': 0.23,
        'sig_E': 0.02,
        'cs': 2.6e-15,
        'sig_cs': 0.3e-15,
        'Rsq': 0.96,
        'tau at input T': 1.61e-6,
        'sig_tau at input T': 2.02e-6},
        ...}
    'T': temperatures for which values of tau were successfully fit. In K.
        'tau': the tau values corresponding to these temperatures in the same
        order.  In seconds.
        'sigma_tau': overall uncetainty in tau taken from the errors in the
        fits from all the schemes that were used to specify trap location
        (corresponding to the same order as 'T' and 'tau'). In seconds.
        'cap': cap1 is either the probability of capture
        (if trap_fit_const() used) or tauc (capture time constant) if
        trap_fit() used. cap1_err is the error from fitting.  max_amp1 is the
        maximum amplitude of the dipole from the curve fit for that pixel.
        Similarly for cap2, cap2_err, and max_amp2, and there can be a 3rd set
        of parameters if a trap sub-electrode location was determined using
        3 schemes.  This data may be useful for future analysis.
        'E': energy level.  In eV.
        'sig_E': standard deviation error of energy level.  In eV.
        'cs': cross section for holes.  In cm^2.
        'sig_cs': standard deviation error of cross section for holes.
        In cm^2.
        'Rsq': adjusted R^2 for the tau vs temperature fit that was done to
        obtain cs.
        'tau at input T': tau (in seconds) evaluated at desired temperature of
        Roman EMCCD, input_T.
        'sig_tau at input T': standard deviation error of tau at desired
        temperature of Roman EMCCD, input_T.  Found by propagating error by
        utilizing 'sig_cs' and 'sig_E'.

    We will recode the string parts of that dictionary specifying the 
    sub-electrode location for a given pixel to a number code. An example of such 
    a string is 'RHSel2', which denotes the right-hand side of electrode 2. It can 
    be 'RHS', 'LHS', or 'CEN' for electrodes 1 through 4. So that can be 
    converted to a number code (1: 'LHS', 2: 'CEN', 3: 'RHS'), so that 'RHSel2' 
    would be 32. 

    Args: 
        trap_dict (dict): A dictionary output by tpump_analysis
        input_dataset (list): A list of corgidrp.data.TPumpData objects.
    
    Returns: 
        trap_cal (corgidrp.data.TrapCalibration): A trap calibration file. 
    '''
    
    n_traps = len(trap_dict)

    electrode_dict = {"LHS": 10, 
                      "CEN": 20,
                      "RHS": 30}

    #Create the main array of traps and their properties, and fill it in
    trap_cal_array = np.zeros([n_traps,10])
    for i,key in enumerate(trap_dict.keys()): 
        key_list = list(key)

        #The first two elements should be the row and column of the trap
        trap_cal_array[i,0] = key_list[0][0]
        trap_cal_array[i,1] = key_list[0][1]

        #Convert the electrode_string into a number (described in the docstring)
        split_electrode_str = key_list[1].split("el")
        trap_cal_array[i,2] = electrode_dict[split_electrode_str[0]] + \
                                int(split_electrode_str[1])

        #Which trap is this at this location? 
        trap_cal_array[i,3] = key_list[2]
        
        #Now extract the dictionary entry: 
        dict_entry = trap_dict[key]

        #Grab the first time constant 
        #TODO: It seems like there could be three of these. How do we choose which one? 
        trap_cal_array[i,4] = dict_entry['cap'][0][0]

        #The maximum amplitude of the dipole
        #TODO: There could also be three of these
        trap_cal_array[i,5] = dict_entry['cap'][0][2]

        #The energy level of the hole
        trap_cal_array[i,6] = dict_entry['E']

        #The cross section for holes
        trap_cal_array[i,7] = dict_entry['cs']

        #R^2 value of fit
        trap_cal_array[i,8] = dict_entry['Rsq']

        #Release time constant at desired Temperature
        trap_cal_array[i,9] = dict_entry['tau at input T']

    invalid_tpu_keywords = typical_cal_invalid_keywords + ['EXPTIME', 'EMGAIN_C', 'EMGAIN_A', 'KGAINPAR', 'HVCBIAS', 'KGAIN_ER', 'TPTAU', 'TPSCHEM1', 'TPSCHEM2', 'TPSCHEM3', 'TPSCHEM4']
    # Remove specific keywords
    for key in ['PROGNUM', 'EXECNUM', 'CAMPAIGN', 'SEGMENT', 'VISNUM', 'OBSNUM', 'CPGSFILE', 'VISITID']:
        if key in invalid_tpu_keywords:
            invalid_tpu_keywords.remove(key)
    prhdr, exthdr, errhdr, dqhdr = check.merge_headers(input_dataset, any_true_keywords=typical_bool_keywords, invalid_keywords=invalid_tpu_keywords)
    exthdr['BUNIT'] = ""

    trapcal = TrapCalibration(trap_cal_array, pri_hdr = prhdr, 
                    ext_hdr = exthdr, 
                    input_dataset=input_dataset, err_hdr=errhdr, dq_hdr=dqhdr)
    
    return trapcal

def rebuild_dict(trap_pump_array):
        '''
        Partially rebuild the trap_dictionary from the trap_pump_array to help with testing

        Args:
            trap_pump_array: array of trap_pump objects
        
        Returns:
            trap_dict: dictionary of trap_pump objects

        '''
        trap_dict = {}

        electrode_dict = {"LHS": 10, 
                        "CEN": 20,
                        "RHS": 30}
        
        electrode_dict_inverse = {10: "LHS",
                                20: "CEN",
                                30: "RHS"}
        
        
        for trap_pump in trap_pump_array:
            electrode_key = int(((trap_pump[2] // 10) % 10)*10)
            electrode_number = int(trap_pump[2] % 10)
            electrode_string = electrode_dict_inverse[electrode_key]+"el"+str(electrode_number)
            key = ((trap_pump[0],trap_pump[1]), electrode_string, int(trap_pump[3]))
            
            trap_dict[key] = {}
            trap_dict[key]['cap']  = [trap_pump[4], 0, trap_pump[5]] #Add the error in to keep the expected dimensions
            trap_dict[key]['E'] = trap_pump[6]
            trap_dict[key]['cs'] = trap_pump[7]
            trap_dict[key]['Rsq'] = trap_pump[8]
            trap_dict[key]['tau at input T'] = trap_pump[9]

        return trap_dict