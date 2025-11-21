# calibrate nonlin

import io
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
try:
    from numpy.exceptions import RankWarning
except:
    from numpy import RankWarning

from corgidrp import check
import corgidrp.data as data

# Dictionary with constant non-linearity calibration parameters
nonlin_params_default = {
    # ROI constants
    'rowroi1': 305,
    'rowroi2': 736,
    'colroi1': 1385,
    'colroi2': 1846,
     
    # background ROIs
    'rowback11': 20,
    'rowback12': 301,
    'rowback21': 740,
    'rowback22': 1001,
    'colback11': 1200,
    'colback12': 2001,
    'colback21': 1200,
    'colback22': 2001,
     
    # minimum exposure time, s
    'min_exp_time': 0,

    # histogram bin parameters; min_bin is in DN
    'num_bins': 50,
    'min_bin': 200,
     
    # factor to mutiply bin_edge values when making mask
    'min_mask_factor': 1.1,
    }
 
def check_nonlin_params(nonlin_params):
    """ Checks integrity of kgain parameters in the dictionary nonlin_params. 

    Args:
        nonlin_params (dict):  Dictionary of parameters used for calibrating nonlinearity.
    """
    if 'rowroi1' not in nonlin_params:
        raise ValueError('Missing parameter:  rowroi1.')
    if 'rowroi2' not in nonlin_params:
        raise ValueError('Missing parameter:  rowroi2.')
    if 'colroi1' not in nonlin_params:
        raise ValueError('Missing parameter:  colroi1.')
    if 'colroi2' not in nonlin_params:
        raise ValueError('Missing parameter:  colroi2.')
    if 'rowback11' not in nonlin_params:
        raise ValueError('Missing parameter:  rowback11.')
    if 'rowback12' not in nonlin_params:
        raise ValueError('Missing parameter:  rowback12.')
    if 'rowback21' not in nonlin_params:
        raise ValueError('Missing parameter:  rowback21.')
    if 'rowback22' not in nonlin_params:
        raise ValueError('Missing parameter:  rowback22.')
    if 'colback11' not in nonlin_params:
        raise ValueError('Missing parameter:  colback11.')
    if 'colback12' not in nonlin_params:
        raise ValueError('Missing parameter:  colback12.')
    if 'colback21' not in nonlin_params:
        raise ValueError('Missing parameter:  colback21.')
    if 'colback22' not in nonlin_params:
        raise ValueError('Missing parameter:  colback22.')
    if 'min_exp_time' not in nonlin_params:
        raise ValueError('Missing parameter:  min_exp_time.')
    if 'num_bins' not in nonlin_params:
        raise ValueError('Missing parameter:  num_bins.')
    if 'min_bin' not in nonlin_params:
        raise ValueError('Missing parameter:  min_bin.')
    if 'min_mask_factor' not in nonlin_params:
        raise ValueError('Missing parameter:  min_mask_factor.')
    
    if not isinstance(nonlin_params['rowroi1'], (float, int)):
        raise TypeError('rowroi1 is not a number')
    if not isinstance(nonlin_params['rowroi2'], (float, int)):
        raise TypeError('rowroi2 is not a number')
    if not isinstance(nonlin_params['colroi1'], (float, int)):
        raise TypeError('colroi1 is not a number')
    if not isinstance(nonlin_params['colroi2'], (float, int)):
        raise TypeError('colroi2 is not a number')
    if not isinstance(nonlin_params['rowback11'], (float, int)):
        raise TypeError('rowback11 is not a number')
    if not isinstance(nonlin_params['rowback12'], (float, int)):
        raise TypeError('rowback12 is not a number')
    if not isinstance(nonlin_params['rowback21'], (float, int)):
        raise TypeError('rowback21 is not a number')
    if not isinstance(nonlin_params['rowback22'], (float, int)):
        raise TypeError('rowback22 is not a number')
    if not isinstance(nonlin_params['colback11'], (float, int)):
        raise TypeError('colback11 is not a number')
    if not isinstance(nonlin_params['colback12'], (float, int)):
        raise TypeError('colback12 is not a number')
    if not isinstance(nonlin_params['colback21'], (float, int)):
        raise TypeError('colback21 is not a number')
    if not isinstance(nonlin_params['colback22'], (float, int)):
        raise TypeError('colback22 is not a number')
    if not isinstance(nonlin_params['min_exp_time'], (float, int)):
        raise TypeError('min_exp_time is not a number')
    if not isinstance(nonlin_params['num_bins'], (float, int)):
        raise TypeError('num_bins is not a number')
    if not isinstance(nonlin_params['min_bin'], (float, int)):
        raise TypeError('min_bin is not a number')
    if not isinstance(nonlin_params['min_mask_factor'], (float, int)):
        raise TypeError('min_mask_factor is not a number')
    

class CalNonlinException(Exception):
    """Exception class for calibrate_nonlin."""

def calibrate_nonlin(dataset_nl,
                     n_cal=20, n_mean=30, norm_val = 2500, min_write = 800.0,
                     max_write = 10000.0,
                     lowess_frac = 0.1, rms_low_limit = 0.004, rms_upp_limit = 0.2,
                     pfit_upp_cutoff1 = -2, pfit_upp_cutoff2 = -3,
                     pfit_low_cutoff1 = 2, pfit_low_cutoff2 = 1,
                     make_plot=False, plot_outdir='figures', show_plot=False,
                     verbose=False, nonlin_params=None, apply_dq = True):
    """
    Function that derives the non-linearity calibration table for a set of DN
    and EM values.

    Args:
      dataset_nl (corgidrp.Dataset): The frames in the dataset are
        bias-subtracted. The dataset contains frames belonging to two different
        sets -- Mean frame, a large array of unity gain frames, and set with
        non-unity gain frames.
        Mean frame -- Unity gain frames with constant exposure time. These frames
        are used to create a mean pupil image. The mean frame is used to select
        pixels in each frame of the large array of unity gain frames (see next)
        to calculate its mean signal. In general, it is expected that at least
        30 frames or more will be taken for this set. In TVAC, 30 frames, each
        with an exposure time of 5.0 sec were taken.
        Large array of unity gain frames: Set of unity gain frames with subsets
        of equal exposure times. Data for each subset should be taken sequentially:
        Each subset must have at least 5 frames. All frames for a subset are taken
        before moving to the next subset. Two of the subsets have the same (repeated)
        exposure time. These two subsets are not contiguous: The first subset is
        taken near the start of the data collection and the second one is taken
        at the end of the data collection (see TVAC example below). The mean
        signal of these two subsets is used to correct for illumination
        brightness/sensor sensitivity drifts for all the frames in the whole set,
        depending on when the frames were taken. There should be no other repeated
        exposure time among the subsets. In TVAC, a total of 110 frames were taken
        within this category. The 110 frames consisted of 22 subsets, each with
        5 frames. All 5 frames had the same exposure time. The exposure times in
        TVAC in seconds were, each repeated 5 times to collect 5 frames in each
        subset -- 0.077, 0.770, 1.538, 2.308, 3.077, 3.846, 4.615, 5.385, 6.154,
        6.923, 7.692, 8.462, 9.231, 10.000, 11.538, 10.769, 12.308, 13.077,
        13.846, 14.615, 15.385, and 1.538 (again).
        Set with non-unity gain frames: a set of subsets of frames. All frames
        in each subset have a unique, non-unity EM gain. For instance, in TVAC,
        11 subsets were considered with EM values (EMGAIN_C): 1.65, 5.24, 8.60,
        16.70, 27.50, 45.26, 87.50, 144.10, 237.26, 458.70 and 584.40. These
        correspond to a range of actual EM gains from about 2 to 7000. Each subset
        collects the same number of frames, which is at least 20 frames. In TVAC,
        each non-unity EM value had 22 frames. In each subset, there are two
        repeated exposure times: one near the start of the data collection and
        one at the very end. The exposure times of the frames in each EM subset
        do not need to be the same. For EM=1.65, the values of the exposure times
        in seconds were: 0.076, 0.758, 1.515, 2.273, 3.031, 3.789, 4.546, 5.304,
        6.062, 6.820, 7.577, 8.335, 9.093, 9.851, 10.608, 11.366, 12.124, 12.881,
        13.639, 14.397, 15.155, and 1.515 (repeated). And for EM=5.24, the 22
        values of the exposure times in seconds were: 0.070, 0.704, 1.408, 2.112,
        2.816, 3.520, 4.225, 4.929, 5.633, 6.337, 7.041, 7.745, 8.449, 9.153,
        9.857, 10.561, 11.265, 11.969, 12.674, 13.378, 14.082, and 1.408 (repeated).
      n_cal (int):
        Minimum number of frames per sub-stack used to calibrate Non-Linearity. The default
        value is 20.
      n_mean (int):
        Minimum number of frames used to generate the mean frame. The default value
        is 30.
      norm_val (int): (Optional) Value in DN to normalize the nonlinearity values to.
        Must be greater than 0 and must be divisible by 20 without remainder.
        (1500 to 3000 recommended).
      min_write (float): (Optional) Minimum mean value in DN to output in
        nonlin. (800.0 recommended)
      max_write (float): (Optional) Maximum mean value in DN to output in
        nonlin. (10000.0 recommended)
      lowess_frac (float): (Optional) factor to use in lowess smoothing function,
        larger is smoother
      rms_low_limit (float): (Optional) rms relative error selection limits for
        linear fit. Lower limit.
      rms_upp_limit (float): (Optional) rms relative error selection limits for
        linear fit. Upper limit. rms_upp_limit must be greater than rms_low_limit.
      pfit_upp_cutoff1 (int): (Optional) polyfit upper cutoff. The following limits were
        determined with simulated frames. If rms_low_limit < rms_y_rel_err < rms_upp_limit,
        this is the upper value applied to select the data to be fitted.
      pfit_upp_cutoff2 (int): (Optional) polyfit upper cutoff. The following limits were
        determined with simulated frames. If rms_y_rel_err >= rms_upp_limit,
        this is the upper value applied to select the data to be fitted.
      pfit_low_cutoff1 (int): (Optional) polyfit upper cutoff. The following limits were
        determined with simulated frames. If rms_low_limit < rms_y_rel_err < rms_upp_limit,
        this is the lower value applied to select the data to be fitted.
      pfit_low_cutoff2 (int): (Optional) polyfit upper cutoff. The following limits were
        determined with simulated frames. If rms_y_rel_err >= rms_upp_limit,
        this is the lower value applied to select the data to be fitted.
      make_plot (bool): (Optional) generate and store plots. Default is True.
      plot_outdir (str): (Optional) Output directory to store figues. Default is
        'figures'. The default directory is not tracked by git.
      show_plot (bool): (Optional) display the plots. Default is False.
      verbose (bool): (Optional) display various diagnostic print messages.
        Default is False.
      nonlin_params (dict): (Optional) Dictionary of row and col specifications
        for the region of interest (indicated by 'roi') where the frame is illuminated and for 
        two background regions (indicated by 'back1' and 'back2') where the frame is not illuminated.  
        Must contain 'rowroi1','rowroi2','colroi1','colroi2','rowback11','rowback12',
        'rowback21','rowback22','colback11','colback12','colback21',and 'colback22'.
        The 'roi' needs one square region specified, and 'back' needs two square regions, 
        where a '1' ending indicates the smaller of two values, and a '2' ending indicates the larger 
        of two values.  The coordinates of each square are specified by matching 
        up as follows: (rowroi1, colroi1), (rowroi1, colroi2), (rowback11, colback11), 
        (rowback11, colback12), etc. Defaults to nonlin_params_default specified in this file.
      apply_dq (bool): consider the dq mask (from cosmic ray detection) or not
      
    Returns:
      nonlin_arr (NonLinearityCalibration): 2-D array with nonlinearity values
        for input signal level (DN) in rows and EM gain values in columns. The
        input signal in DN is the first column. Signal values start with min_write
        and run through max_write in steps of 20 DN.
    """
    if nonlin_params is None:
        nonlin_params = nonlin_params_default
        
    check_nonlin_params(nonlin_params)

    # dataset_nl.all_data must be 3-D 
    if np.ndim(dataset_nl.all_data) != 3:
        raise Exception('dataset_nl.all_data must be 3-D')
    # cast dataset objects into np arrays and retrieve aux information
    cal_list, mean_frame_list, exp_arr, datetime_arr, len_list, actual_gain_arr, datetimes_sort_inds, _ = \
        nonlin_kgain_dataset_2_stack(dataset_nl, apply_dq = apply_dq, cal_type='nonlin')
    cal_arr = np.vstack(cal_list)[datetimes_sort_inds]
    mean_frame_arr = np.stack(mean_frame_list)
    # Get relevant constants
    rowroi1 = nonlin_params['rowroi1']
    rowroi2 = nonlin_params['rowroi2']
    colroi1 = nonlin_params['colroi1']
    colroi2 = nonlin_params['colroi2']
    rowback11 = nonlin_params['rowback11']
    rowback12 = nonlin_params['rowback12']
    rowback21 = nonlin_params['rowback21']
    rowback22 = nonlin_params['rowback22']
    colback11 = nonlin_params['colback11']
    colback12 = nonlin_params['colback12']
    colback21 = nonlin_params['colback21']
    colback22 = nonlin_params['colback22']
    min_exp_time = nonlin_params['min_exp_time']
    num_bins = nonlin_params['num_bins']
    min_bin = nonlin_params['min_bin']
    min_mask_factor = nonlin_params['min_mask_factor']

    if type(cal_arr) != np.ndarray:
        raise TypeError('cal_arr must be an ndarray.')
    if np.ndim(cal_arr) != 3:
        raise CalNonlinException('cal_arr must be 3-D')
    if len(len_list) < 1:
        raise CalNonlinException('Number of elements in len_list must '
                'be greater than or equal to 1.')
    if np.sum(len_list) != len(cal_arr):
        raise CalNonlinException('Number of sub-stacks in cal_arr must '
                'equal the sum of the elements in len_list')
    # cal_arr must have at least 20 frames for each EM gain
    if np.any(np.array(len_list) < n_cal):
        raise Exception(f'cal_arr must have at least {n_cal} frames for each EM value')
    if len(np.unique(datetime_arr)) != len(datetime_arr):
        raise CalNonlinException('All elements of datetime_arr must be unique.')
    for g_index in range(len(len_list)):
        # Define the start and stop indices
        start_index = int(np.sum(len_list[0:g_index]))
        stop_index = start_index + len_list[g_index]
        # Convert camera times to datetime objects
        ctim_strings = datetime_arr[start_index:stop_index]
        ctim_datetime = pd.to_datetime(ctim_strings, errors='coerce')
        # Check if the array is time-ordered in increasing order
        is_increasing = np.all(ctim_datetime[:-1] <= ctim_datetime[1:])
        if not is_increasing:
            raise CalNonlinException('Elements of datetime_arr must be '
                    'in increasing time order for each EM gain value.')
    if type(mean_frame_arr) != np.ndarray:
        raise TypeError('mean_frame_arr must be an ndarray.')
    if np.ndim(mean_frame_arr) != 3:
        raise CalNonlinException('mean_frame_arr must be 3-D (i.e., a stack of '
                '2-D sub-stacks')
    # mean_frame_arr must have at least 30 frames
    if len(mean_frame_arr) < n_mean:
        raise CalNonlinException(f'Number of frames in mean_frame_arr must '
                'be at least {n_mean}.')
    
    check.real_array(exp_arr, 'exp_arr', TypeError)
    check.oneD_array(exp_arr, 'exp_arr', TypeError)
    if (exp_arr <= min_exp_time).any():
        raise CalNonlinException('Each element of exp_arr must be '
            ' greater than min_exp_time.')
    # check to see if there is at least one set of exposure times with length different from that of the others
    index = 0
    r_flag = True
    for x in range(len(len_list)):
        temp = np.copy(exp_arr[index:index+len_list[x]])
        # Unique counts of exposure times
        _, u_counts = np.unique(temp, return_counts=True)
        # Check if all elements are the same
        all_elements_same = np.all(u_counts == u_counts[0])
        if all_elements_same == True:
            r_flag = False
        index = index + len_list[x]
    # check to see that there is a repeated exposure time (e.g., at least one set in between 2 sets of the same exposure time)
    index = 0
    repeated_lens = [] # to be used later, to know how to split up the repeated sets
    for x in range(len(len_list)):
        temp = np.copy(exp_arr[index:index+len_list[x]])
        # first condition below: frames are already time-ordered, so if there is non-monotonicity, there is repitition, which we want
        # r_flag condition:  merely a set with length longer than the others (which TVAC code has); this gives a "way out" if no repeated set after other sets
        if np.all(np.diff(temp) >= 0) and not r_flag:
            raise CalNonlinException('Each substack of cal_arr must have a '
            'group of frames with a repeated exposure time.')
        if np.all(np.diff(temp) < 0):
            repeat_ind = np.where(np.diff(temp) < 0)[0][0]
            ending = np.where(np.diff(temp)[repeat_ind+1:] != 0)[0]
            if len(ending) == 0: # repeated set is last one in time
                end_ind = None
            else:
                end_ind = ending[0]+1
            repeated_lens.append(len(np.diff(temp)[repeat_ind:end_ind]))
        
        index = index + len_list[x]
    if len(len_list) != len(actual_gain_arr):
        raise CalNonlinException('Length of actual_gain_arr be the same as the '
                                 'length of len_list.')
    if sum(1 for number in actual_gain_arr if number < 1) != 0:
        raise CalNonlinException('Each element of actual_gain_arr must be greater '
            'than or equal to 1.')
    check.real_array(actual_gain_arr, 'actual_gain_arr', TypeError)
    check.oneD_array(actual_gain_arr, 'actual_gain_arr', TypeError)
    check.positive_scalar_integer(norm_val, 'norm_val', TypeError)
    if np.mod(norm_val, 20) !=0:
        raise CalNonlinException('norm_val must be divisible by 20.')
    check.real_positive_scalar(min_write, 'min_write', TypeError)
    check.real_positive_scalar(max_write, 'max_write', TypeError)
    if min_write >= max_write:
        raise CalNonlinException('max_write must be greater than min_write')
    if (norm_val < min_write) or (norm_val > max_write):
        raise CalNonlinException('norm_val must be between min_write and '
                                 'max_write.')
    check.real_nonnegative_scalar(rms_low_limit, 'rms_low_limit', TypeError)
    check.real_nonnegative_scalar(rms_upp_limit, 'rms_upp_limit', TypeError)
    if rms_low_limit >= rms_upp_limit:
        raise CalNonlinException('rms_upp_limit must be greater than rms_low_limit')

    if not isinstance(lowess_frac, (float, int)):
        raise TypeError('lowess_frac is not a number')
    if not isinstance(rms_low_limit, (float, int)):
        raise TypeError('rms_low_limit is not a number')
    if not isinstance(rms_upp_limit, (float, int)):
        raise TypeError('rms_upp_limit is not a number')
    if not isinstance(pfit_upp_cutoff1, (float, int)):
        raise TypeError('pfit_upp_cutoff1 is not a number')
    if not isinstance(pfit_upp_cutoff2, (float, int)):
        raise TypeError('pfit_upp_cutoff2 is not a number')
    if not isinstance(pfit_low_cutoff1, (float, int)):
        raise TypeError('pfit_low_cutoff1 is not a number')
    if not isinstance(pfit_low_cutoff2, (float, int)):
        raise TypeError('pfit_low_cutoff2 is not a number')

    if make_plot is True:
        # Avoid issues with importing matplotlib on headless servers without GUI
        # support without proper configuration
        import matplotlib.pyplot as plt
        # Output directory
        if os.path.exists(plot_outdir) is False:
            os.mkdir(plot_outdir)
            if verbose:
                print('Output directory for figures created in ', os.getcwd())
    
    ######################### start of main code #############################
    
    # Define pixel ROIs
    rowroi = list(range(rowroi1, rowroi2))
    colroi = list(range(colroi1, colroi2))
    
    # Background subtraction regions
    rowback1 = list(range(rowback11, rowback12))
    rowback2 = list(range(rowback21, rowback22))
    colback1 = list(range(colback11, colback12))
    colback2 = list(range(colback21, colback22))
    
    ####################### create good_mean_frame ###################
    
    nrow = len(mean_frame_arr[0])
    ncol = len(mean_frame_arr[0][0])
    
    good_mean_frame = np.zeros((nrow, ncol))
    nFrames = len(mean_frame_arr)

    good_mean_frame = good_mean_frame / nFrames
    
    mean_frame_index = 0
    # Loop over the mean_frame_arr frames
    for i in range(nFrames):
        frame = mean_frame_arr[i]
    
        # Add this frame to the cumulative good_mean_frame
        good_mean_frame += frame
        mean_frame_index += 1

    # Calculate the average of the frames if required
    if mean_frame_index > 0:
        good_mean_frame /= mean_frame_index 
    
    # plot, if requested
    if make_plot:
        fname = 'non_lin_good_frame'
        # Slice the good_mean_frame array
        frame_slice = good_mean_frame[np.ix_(rowroi, colroi)]
        # Create a figure and plot the sliced frame
        plt.figure()
        # 'viridis' is a common colormap
        plt.imshow(frame_slice, aspect='equal', cmap='viridis')
        plt.colorbar()
        plt.title('Good quality mean frame')
        plt.savefig(f'{plot_outdir}/{fname}')
        if verbose:
            print(f'Figure {fname} stored in {plot_outdir}')
        if show_plot:
            plt.show()
        plt.close()
    
    # Convert to numpy arrays if they are not already
    rowroi = np.array(rowroi)
    colroi = np.array(colroi)
    
    if make_plot:
        fname = 'non_lin_mean_frame_histogram'
        # Plot a histogram of the values within the specified ROI
        roi_values = good_mean_frame[rowroi[:, None], colroi]
        plt.figure()
        # 'auto' lets matplotlib decide the number of bins
        plt.hist(roi_values.flatten(), bins='auto', log=True)
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.title('Histogram of Mean Frame in ROI')
        plt.savefig(f'{plot_outdir}/{fname}')
        if verbose:
            print(f'Figure {fname} stored in {plot_outdir}')
        if show_plot:
            plt.show()
        plt.close()
    
    # find minimum in histogram
    # 1000-1500 DN recommended when the peak of histogram of  
    # "good_mean_frame" is between 2000 and 4000 DN)
    roi_values = good_mean_frame[rowroi[:, None], colroi]
    hst_counts, hist_edges = np.histogram(roi_values.flatten(),bins=num_bins)
    # range above some value
    above_range = (hist_edges[:-1] >= min_bin)
    # Filter the counts and bin_edges arrays
    filtered_counts_above = hst_counts[above_range]
    filtered_bin_edges_above = hist_edges[:-1][above_range]
    # Find the index of the maximum count within the filtered range
    max_count_index_above_range = np.argmax(filtered_counts_above)
    # Get the corresponding bin edge
    max_edge_value = filtered_bin_edges_above[max_count_index_above_range]
    # Find the indices of the bins that fall within the specified range
    within_range = (hist_edges[:-1] >= min_bin) & (hist_edges[:-1] <= max_edge_value)
    # Filter the counts and bin_edges arrays
    filtered_counts = hst_counts[within_range]
    filtered_bin_edges = hist_edges[:-1][within_range]
    # Find the index of the minimum count within the filtered range
    min_count_index_within_range = np.argmin(filtered_counts)
    # Get the corresponding bin edge value and increase by min_mask_factor
    min_mask = min_mask_factor*filtered_bin_edges[min_count_index_within_range]
    # Create the mask
    mask = np.where(good_mean_frame < min_mask, 0, 1)
    
    # plot, if requested
    if make_plot:
        fname = 'non_lin_mask'
        # Plot the mask
        plt.figure()
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.colorbar()
        plt.savefig(f'{plot_outdir}/{fname}')
        if verbose:
            print(f'Figure {fname} stored in {plot_outdir}')
        if show_plot:
            plt.show()
        plt.close()
        
        fname = 'non_lin_mean_frame'
        # Plot the mean frame
        plt.figure()
        # 'viridis' is a good default color map
        plt.imshow(good_mean_frame, cmap='viridis')
        plt.title('Mean Frame')
        plt.colorbar()
        plt.close()
    
    # initialize arrays for nonlin results table
    nonlin = []
    
    ######################## loop over em gain values #########################
    for gain_index in range(len(len_list)):
        
        start_index = int(np.sum(len_list[0:gain_index]))
        stop_index = start_index + len_list[gain_index]
        # Convert camera times to datetime objects
        ctime_strings = datetime_arr[start_index:stop_index]
        ctime_datetime = pd.to_datetime(ctime_strings, errors='coerce')
        
        # Select exp times for this em gain
        exp_em = exp_arr[start_index:stop_index]
        
        # select frames for this em gain
        full_flst = cal_arr[start_index:stop_index]
        
        # Unique exposure times and their counts
        exposure_strings_list, counts = np.unique(exp_em, return_counts=True)
        
        # Grouping exposures and finding the max count
        max_count_index = np.argmax(counts)
        repeat_exp = exposure_strings_list[max_count_index]  # Exposure time of repeated frames
        
        # Calculate mean time differences as aid in illumination drift corrections
        group_mean_time = []
        first_flag = False
        
        for t0 in exposure_strings_list:
            idx = np.where(exp_em == t0)[0]
            if t0 != repeat_exp:
                del_s = (ctime_datetime[idx] - ctime_datetime[0]).total_seconds()
                group_mean_time.append(np.mean(del_s))
            elif t0 == repeat_exp and not first_flag:
                # NOTE works fine for same number of frames per exposure time for a given EM gain (which is the observation plan, and this 
                # is what the TVAC code has), but for more general case, use repeated_lens[gain_index] instead for the number of frames in the 2nd (repeated) set
                idx_2 = len(idx) // 2 
                start_time_repeated = ctime_datetime[idx[0]]
                end_time_repeated = ctime_datetime[idx[idx_2]]
                del_s2 = (ctime_datetime[idx[:idx_2]] - ctime_datetime[0]).total_seconds()
                group_mean_time.append(np.mean(del_s2))
                first_flag = True
        
        if verbose is True:
            print(group_mean_time)
            print('Time between repeated exposure frames for EM gain = ', actual_gain_arr[gain_index],': ',
                  (end_time_repeated - start_time_repeated).total_seconds(), 'seconds')
        
        # Additional setup
        mean_signal = []
        repeat_flag = 0
        filtered_exposure_times = []
        
        for jj in range(len(exposure_strings_list)):
            current_exposure_time = exposure_strings_list[jj]
        
            if current_exposure_time >= min_exp_time:
                if current_exposure_time == repeat_exp:
                    repeat_flag = 1
        
                # Filtering frames based on the current exposure time
                selected_files = [
                    full_flst[idx] for idx, exp_time in enumerate(exp_em) if exp_time == current_exposure_time
                ]

                filtered_exposure_times.append(current_exposure_time)
        
                # Initialize for processing of files
                mean_frame_index = 0
                frame_count = []
                frame_mean = []
                if not repeat_flag:
                    for iframe in range(len(selected_files)):
                        
                        frame_1 = selected_files[iframe]
                        frame_1 = frame_1.astype(np.float64)
        
                        # Subtract background
                        frame_1_back1 = np.nanmean(frame_1[rowback1[0]:rowback1[-1]+1, 
                                                        colback1[0]:colback1[-1]+1])
                        frame_1_back2 = np.nanmean(frame_1[rowback2[0]:rowback2[-1]+1, 
                                                        colback2[0]:colback2[-1]+1])
                        frame_back = (frame_1_back1 + frame_1_back2) / 2
        
                        # Calculate counts and mean in the ROI after background subtraction
                        roi_frame = frame_1[rowroi[0]:rowroi[-1]+1, colroi[0]:colroi[-1]+1] - frame_back
                        frame_count0 = np.sum(roi_frame)
                        frame_mean0 = frame_1 - frame_back
        
                        # Apply mask and calculate the positive mean
                        frame_mean0 *= mask
                        positive_means = frame_mean0[frame_mean0 > 0]
                        frame_mean1 = np.nanmean(positive_means) if positive_means.size > 0 else np.nan
        
                        frame_count.append(frame_count0)
                        frame_mean.append(frame_mean1)
                        
                        mean_frame_index += 1
                    mean_signal.append(np.nanmean(frame_mean))
                elif repeat_flag:
                    # for repeated exposure frames, split into the first half/set
                    # and the second half/set
                    # NOTE works fine for same number of frames per exposure time for a given EM gain (which is the observation plan, and this 
                    # is what the TVAC code has), but for more general case, use repeated_lens[gain_index] instead for the number of frames in the 2nd (repeated) set
                    first_half = len(selected_files) // 2
                    for i in range(first_half):

                        frame_1 = selected_files[i]
                        frame_1 = frame_1.astype(np.float64)
        
                        # Subtract background
                        frame_1_back1 = np.nanmean(frame_1[rowback1[0]:rowback1[-1]+1, 
                                                        colback1[0]:colback1[-1]+1])
                        frame_1_back2 = np.nanmean(frame_1[rowback2[0]:rowback2[-1]+1, 
                                                        colback2[0]:colback2[-1]+1])
                        frame_back = (frame_1_back1 + frame_1_back2) / 2
        
                        # Calculate counts and mean in the ROI after background subtraction
                        roi_frame = frame_1[rowroi[0]:rowroi[-1]+1, 
                                            colroi[0]:colroi[-1]+1] - frame_back
                        frame_count0 = np.sum(roi_frame)
                        frame_mean0 = frame_1 - frame_back
        
                        # Apply mask and calculate the positive mean
                        frame_mean0 *= mask
                        positive_means = frame_mean0[frame_mean0 > 0]
                        frame_mean1 = np.nanmean(positive_means) if positive_means.size > 0 else np.nan
                        
                        frame_count.append(frame_count0)
                        frame_mean.append(frame_mean1)
                        
                        mean_frame_index += 1
                    mean_signal.append(np.nanmean(frame_mean))
                    repeat1_mean_signal = np.nanmean(frame_mean)
                    
                    second_half = len(selected_files)
                    for i in range(first_half + 1, second_half):
                       
                        frame_1 = selected_files[i]
                        frame_1 = frame_1.astype(np.float64)
        
                        # Subtract background
                        frame_1_back1 = np.nanmean(frame_1[rowback1[0]:rowback1[-1]+1, colback1[0]:colback1[-1]+1])
                        frame_1_back2 = np.nanmean(frame_1[rowback2[0]:rowback2[-1]+1, colback2[0]:colback2[-1]+1])
                        frame_back = (frame_1_back1 + frame_1_back2) / 2
        
                        # Calculate counts and mean
                        roi_frame = frame_1[rowroi[0]:rowroi[-1]+1, colroi[0]:colroi[-1]+1] - frame_back
                        frame_count0 = np.sum(roi_frame)
                        frame_mean0 = frame_1 - frame_back
                        frame_mean0 *= mask
                        positive_means = frame_mean0[frame_mean0 > 0]
                        frame_mean1 = np.nanmean(positive_means) if positive_means.size > 0 else np.nan
        
                        frame_count.append(frame_count0)
                        frame_mean.append(frame_mean1)
        
                        mean_frame_index += 1
                    # Calculate the mean signal from the second half of the processing
                    repeat2_mean_signal = np.nanmean(frame_mean)
                    repeat_flag = 0  # Reset flag

        # Calculate the time deltas in seconds from the first frame
        delta_ctimes_s = (ctime_datetime - ctime_datetime[0]).total_seconds()
        
        # Make sure delta_ctimes_s is a pandas Series with numeric values
        delta_ctimes_s = pd.Series(delta_ctimes_s, index=ctime_datetime)
        
        # Calculate the difference in signals
        delta_signal = repeat2_mean_signal - repeat1_mean_signal
        
        # Assuming all_exposure_strings and repeat_exp are already defined
        
        # Find indices of the frames where the exposure time matches repeat_exp
        repeat_times_idx = np.where(exp_em == repeat_exp)[0]  # np.where returns a tuple, extract first element
        
        # Calculate the mean times for the first and second halves of these indices
        # NOTE works fine for same number of frames per exposure time for a given EM gain (which is the observation plan, and this 
        # is what the TVAC code has), but for more general case, use repeated_lens[gain_index] instead for the number of frames in the 2nd (repeated) set
        first_half = len(repeat_times_idx) // 2
        first_half_mean_time = delta_ctimes_s.iloc[repeat_times_idx[:first_half]].mean()
        
        second_half = len(repeat_times_idx)
        second_half_mean_time = delta_ctimes_s.iloc[repeat_times_idx[first_half:second_half]].mean()
        
        if verbose is True:
            print("First half mean time:", first_half_mean_time)
            print("Second half mean time:", second_half_mean_time)
        
        # Calculate DN/s
        illum_slope = delta_signal / (second_half_mean_time - first_half_mean_time)
        
        # Calculate DN
        illum_inter = repeat1_mean_signal - illum_slope * first_half_mean_time
        
        # Adjust observations based on calculated slope and intercept
        illum_obs = (group_mean_time - group_mean_time[0]) * illum_slope + illum_inter
        
        # Correct the illumination observations
        illum_corr = illum_obs / illum_obs[0]
        
        # Correct the mean signal
        #illum_cor = np.ones(len(illum_corr))
        corr_mean_signal = mean_signal / illum_corr
        
        # Sort arrays by exposure time
        filt_exp_times_sorted, I = np.sort(filtered_exposure_times), np.argsort(filtered_exposure_times)
        corr_mean_signal_sorted = np.array(corr_mean_signal)[I]
        
        if make_plot:
            fname = 'non_lin_signal_vs_exp'
            # Plotting the corrected mean signal against sorted exposure times
            plt.figure()
            plt.plot(filt_exp_times_sorted, corr_mean_signal_sorted, 'o', label='Data Points')
            plt.title('Signal versus exposure time')
            plt.xlabel('Exposure time (s)')
            plt.ylabel('Signal (DN)')
        
        # Fit a polynomial to selected points (excluding some points)
        p0 = np.polyfit(filt_exp_times_sorted, corr_mean_signal_sorted, 1)
        y0 = np.polyval(p0, filt_exp_times_sorted)
        y_rel_err = np.abs((corr_mean_signal_sorted - y0)/corr_mean_signal_sorted)
        rms_y_rel_err = np.sqrt(np.mean(y_rel_err**2))
        # NOTE: the following limits were determined with simulated frames
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RankWarning)
            if rms_y_rel_err < rms_low_limit:
                p1 = np.polyfit(filt_exp_times_sorted, corr_mean_signal_sorted, 1)
            elif (rms_y_rel_err >= rms_low_limit) and (rms_y_rel_err < rms_upp_limit):
                p1 = np.polyfit(filt_exp_times_sorted[pfit_low_cutoff1:pfit_upp_cutoff1], 
                                corr_mean_signal_sorted[pfit_low_cutoff1:pfit_upp_cutoff1], 1)
            else:
                p1 = np.polyfit(filt_exp_times_sorted[pfit_low_cutoff2:pfit_upp_cutoff2], 
                                corr_mean_signal_sorted[pfit_low_cutoff2:pfit_upp_cutoff2], 1)
            y1 = np.polyval(p1, filt_exp_times_sorted)
        
        if make_plot:
            fname = 'non_lin_fit'
            # Plot the fitted line
            plt.plot(filt_exp_times_sorted, y1, label='Fitted Line')
            
            # Show the plot with legend
            plt.legend()
            plt.savefig(f'{plot_outdir}/{fname}')
            if verbose:
                print(f'Figure {fname} stored in {plot_outdir}')
            if show_plot:
                plt.show()
            plt.close()
        
        # Calculating relative gain
        rel_gain = corr_mean_signal_sorted / y1
        
        # Smoothing the relative gain data; larger 'lowess_frac' gives smoother curve
        rel_gain_smoothed = lowess(rel_gain, 
                            corr_mean_signal_sorted, frac=lowess_frac)[:, 1]
        
        # find the min/max values of corrected measured means and append array
        temp_min = np.nanmin(corr_mean_signal_sorted)
        temp_max = np.nanmax(corr_mean_signal_sorted)
        
        if make_plot:
            # Plotting Signal vs. Relative Gain
            plt.figure()
            plt.plot(corr_mean_signal_sorted, rel_gain, 'o', label='Original Data')
            plt.ylim([0.95, 1.05])
            plt.xlim([1, 14000])
            plt.axhline(1.0, linestyle='--', color='k', linewidth=1)  # horizontal line at 1.0
            
            plt.title('Signal/fit versus Signal')
            plt.xlabel('Signal (DN)')
            plt.ylabel('Relative gain')
            
            # Plot the smoothed data
            plt.plot(corr_mean_signal_sorted, rel_gain_smoothed, 'r-', label='Smoothed Data')
            
            # Show legend and plot
            plt.legend()
            plt.savefig(f'{plot_outdir}/{fname}')
            if verbose:
                print(f'Figure {fname} stored in {plot_outdir}')
            if show_plot:
                plt.show()
            plt.close()
        
        # Generate evenly spaced values between 20 and 14000
        mean_linspace = np.linspace(20, 14000, 1+int((14000-20)/20))
        
        # Interpolate/extrapolate the relative gain values
        interp_func = interp1d(corr_mean_signal_sorted, 
                        rel_gain_smoothed, kind='linear', fill_value='extrapolate')
        rel_gain_interp = interp_func(mean_linspace)
        
        # Normalize the relative gain to the value at norm_val DN
        # First, find the index for norm_val DN in mean_linspace
        idxnorm = np.where(mean_linspace == norm_val)[0][0]
        normconst = rel_gain_interp[idxnorm]
        rel_gain_interp /= normconst
        if (norm_val < temp_min) or (norm_val > temp_max):
            print('norm_val is not between the minimum and maximum values '
                          'of the means for the current EM gain. Extrapolation '
                          'will be used for norm_val.')
        
        if make_plot:
            fname = 'non_lin_fit_norm_dn'
            # Plotting Signal vs. Relative Gain normalized at norm_val DN
            plt.figure()
            plt.plot(corr_mean_signal_sorted, rel_gain / normconst, 'o', label='Original Data')
            plt.ylim([0.95, 1.05])
            plt.xlim([1, 14000])
            plt.axhline(1.0, linestyle='--', color='k', linewidth=1)  # horizontal line at 1.0
            
            plt.title(f'Signal/fit versus Signal (norm @ {norm_val} DN)')
            plt.xlabel('Signal (DN)')
            plt.ylabel('Relative gain')
            
            # Plot the interpolated data
            plt.plot(mean_linspace, rel_gain_interp, 'r-', label='Interpolated Data')
            plt.legend()
            plt.savefig(f'{plot_outdir}/{fname}')
            if verbose:
                print(f'Figure {fname} stored in {plot_outdir}')
            if show_plot:
                plt.show()
            plt.close()
        
        # NOTE: nonlinearity is equal to 1/rel_gain
        # multiply raw data by 1/rel_gain to correct for nonlinearity
        temp = 1/rel_gain_interp
        nonlin.append(temp)
    
    # prepare nonlin array
    nonlin_arr0 = np.transpose(np.array(nonlin))
    # insert new column at the start of nonlin_arr
    nonlin_arr1 = np.insert(nonlin_arr0, 0, mean_linspace, axis=1)
    # select rows that satisfy min/max limits
    nonlin_arr2 = nonlin_arr1[nonlin_arr1[:, 0] >= min_write]
    nonlin_arr3 = nonlin_arr2[nonlin_arr2[:, 0] <= max_write]
    # See data.NonLinearityCalibration doc string for more details:
    # [0, 1:]: Gain axis values
    # [1:, 0]: "count" axis value
    actual_gain_arr = np.insert(actual_gain_arr, 0, np.nan)
    n_col = len(nonlin_arr3) + 1
    n_row = len(actual_gain_arr)
    nonlin_data=np.insert(nonlin_arr3, 0, actual_gain_arr).reshape(n_col,n_row)
    
    # Return NonLinearity instance
    prhd = dataset_nl.frames[0].pri_hdr
    exthd = dataset_nl.frames[0].ext_hdr
    exthd['HISTORY'] = f"Non-linearity calibration derived from a set of frames on {exthd['DATETIME']}"
    # Just for the purpose of getting the instance created
    nonlin = data.NonLinearityCalibration(nonlin_data,
        pri_hdr = prhd, ext_hdr = exthd, input_dataset=dataset_nl)
    
    # === BEGIN DEBUG BLOCK ===
    print("NONLIN-DEBUG: calibrate_nonlin summary")
    print("  Input args:")
    print("    norm_val  =", norm_val)
    print("    min_write =", min_write)
    print("    max_write =", max_write)
    print("    temp_min  =", temp_min)
    print("    temp_max  =", temp_max)

    data_arr = nonlin.data
    print("  Output data shape:", getattr(data_arr, "shape", None))

    # Find unity row using the same logic as test_nonlin_cal.py
    ones_col = data_arr[1:, 1]
    idx = np.where(ones_col == 1)[0]

    if idx.size:
        norm_ind = int(idx[0])
        row = data_arr[norm_ind + 1, :]
        print("NONLIN-DEBUG: unity row index (norm_ind+1) =", norm_ind + 1)
        print("  unity row        =", row)
        print("  x_at_unity       =", float(row[0]))
        print("  unity col1       =", float(row[1]))
        print("  unity last col   =", float(row[-1]))
        print("  would-be assert  :", float(row[0]), "==", norm_val)
    else:
        print("NONLIN-DEBUG WARNING: no unity row (data_arr[1:,1] == 1) found")

    print("-" * 60)
    # === END DEBUG BLOCK ===
    
    return nonlin

def nonlin_kgain_dataset_2_stack(dataset, apply_dq = True, cal_type='nonlin'):
    """
    Casts the CORGIDRP Dataset object for non-linearity calibration into a stack
    of numpy arrays sharing the same commanded gain value. It also returns the list of
    unique EM values and set of exposure times used with each EM. Note: it also
    performs a set of tests about the integrity of the data type and values in
    the dataset.

    Args:
        dataset (corgidrp.Dataset): Dataset with a set of of EXCAM illuminated
        pupil L1 SCI frames (counts in DN)
        apply_dq (bool): consider the dq mask (from cosmic ray detection) or not
        cal_type (str): If 'kgain', then sets of frames with the same exposure time for a given EM gain 
            are truncated so that each has the same number of frames.  Otherwise (for the 'nonlin' case), there is no truncation.

    Returns:
        list of data arrays associated with each frame
        list of mean frames
        array of exposure times associated with each frame
        array of datetimes associated with each frame
        list with the number of frames with same EM gain
        List of actual EM gains
        array of indices for timestamp ordering
        number of frames each set of frames with same exposure time truncated to for EM gain = 1 frames

    """
    # Split Dataset
    dataset_cp = dataset.copy()
    split = dataset_cp.split_dataset(exthdr_keywords=['EMGAIN_C'])
    
    # Calibration data
    stack = []
    # Mean frame data
    mean_frame_stack = []
    record_exp_time = True
    # Exposure times
    exp_times = []
    # Datetimes
    datetimes = []
    # Size of each sub stack
    len_sstack = []
    # Record measured gain of each substack of calibration frames
    gains = []
    smallest_set_len = None
    for idx_set, data_set in enumerate(split[0]):
        obsname_dsets, obsname_vals = data_set.split_dataset(prihdr_keywords=['OBSNAME'])
        cal_dsets = []
        mnframe_ind = None
        for i, v in enumerate(obsname_vals):
            if v.upper()=='MNFRAME':
                mnframe_ind = i
            else:
                cal_dsets.append(obsname_dsets[i])
        # First layer (array of unique EM values)
        stack_cp = []
        len_cal_frames = 0
        record_gain = True 
        if mnframe_ind is not None:
            for frame in obsname_dsets[mnframe_ind]:
                if apply_dq:
                    bad = np.where(frame.dq > 0)
                    frame.data[bad] = np.nan
                if record_exp_time:
                    exp_time_mean_frame = frame.ext_hdr['EXPTIME'] 
                    record_exp_time = False
                if frame.ext_hdr['EXPTIME'] != exp_time_mean_frame:
                    raise Exception('Frames used to build the mean frame must have the same exposure time')
                if frame.ext_hdr['EMGAIN_C'] != 1:
                    raise Exception('The commanded gain used to build the mean frame must be unity')
                mean_frame_stack.append(frame.data)
        for cal_dset in cal_dsets:
            # each of dsets has just one frame in it
            dsets, vals = cal_dset.split_dataset(exthdr_keywords=['DATETIME','EXPTIME'])
            smallest_set_length = np.inf
            sub = []
            start_val = float(vals[0][1])
            start_val_ind = 0
            exptime_dsets = []
            for val in vals:
                if vals.index(val) == 0:
                    continue
                if float(val[1]) == start_val:
                    pass
                else:
                    exptime_dsets.append(dsets[start_val_ind:vals.index(val)])
                    start_val = float(val[1])
                    start_val_ind = vals.index(val)
            # ending set not covered
            exptime_dsets.append(dsets[start_val_ind:])
            for i in exptime_dsets:
                if len(i) < smallest_set_length:
                    smallest_set_length = len(i)
            if cal_type == 'nonlin':
                smallest_set_length = None # for nonlin, don't need to truncate exptime sets to be same length for a given EM gain
            for i, exptime_dset_list in enumerate(exptime_dsets):
                sub = np.stack([dset.frames[0].data for dset in exptime_dset_list[:smallest_set_length]])
                for exptime_dset in exptime_dset_list[:smallest_set_length]:
                    frame = exptime_dset.frames[0]
                    if not (frame.pri_hdr['OBSNAME'] == 'KGAIN' or 
                    frame.pri_hdr['OBSNAME'] == 'NONLIN'):
                        raise Exception('OBSNAME can only be MNFRAME or NONLIN in non-linearity')
                    datetime = frame.ext_hdr['DATETIME']                
                    if isinstance(datetime, str) is False:
                        raise Exception('DATETIME must be a string')
                    datetimes.append(datetime)
                    exp_time = frame.ext_hdr['EXPTIME']
                    if isinstance(exp_time, float) is False:
                        raise Exception('Exposure times must be float')
                    if exp_time <=0:
                        raise Exception('Exposure times must be positive')
                    exp_times.append(exp_time)
                    if record_gain:
                        try: # if EM gain measured directly from frame
                            gains.append(frame.ext_hdr['EMGAIN_M'])
                        except:
                            if frame.ext_hdr['EMGAIN_A'] > 0: # use applied EM gain if available
                                gains.append(frame.ext_hdr['EMGAIN_A'])
                            else: # use commanded gain otherwise
                                gains.append(frame.ext_hdr['EMGAIN_C'])
                            record_gain = False
                    if gains[-1] == 1:
                        smallest_set_len = smallest_set_length
                stack_cp.append(sub)
                len_cal_frames += len(sub)
            # Length of substack must be at least 1
            if len(stack_cp) == 0:
                raise Exception('Substacks must have at least one element')
            else:
                stack.append(np.vstack(stack_cp))
                len_sstack.append(len_cal_frames)
    
    # All elements of datetimes must be unique
    if len(datetimes) != len(set(datetimes)):
        raise Exception('DATETIMEs cannot be duplicated')
    # Every EM gain must be greater than or equal to 1
    if np.any(np.array(split[1]) < 1):
        raise Exception('Each set of frames categorized by commanded EM gains must be have 1 or more frames')
    if np.any(np.array(gains) < 1):
        raise Exception('Actual EM gains must be greater than or equal to 1')
    
    # sort frames by time stamp for drift correction later
    datetimes_sort_inds = np.argsort(datetimes)
    return (stack, mean_frame_stack, np.array(exp_times)[datetimes_sort_inds],
        np.array(datetimes)[datetimes_sort_inds], len_sstack, np.array(gains), datetimes_sort_inds, smallest_set_len)
