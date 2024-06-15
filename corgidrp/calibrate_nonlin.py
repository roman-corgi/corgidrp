# calibrate nonlin

import os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from scipy.interpolate import interp1d
import io
import matplotlib.pyplot as plt
#from statsmodels.nonparametric.smoothers_lowess import lowess

from corgidrp.calibrate_kgain import check

class CalNonlinException(Exception):
    """Exception class for calibrate_nonlin."""

def calibrate_nonlin(stack_arr, exp_time_stack_arr, time_stack_arr, len_list, 
                     stack_arr2, actual_gain_arr, norm_val = 2500, 
                     min_write = 800.0, max_write = 10000.0,
                     mkplot=None, verbose=None):
    """Given a large array of stacks with 1 or more EM gains, and sub-stacks of 
    frames ranging over exposure time, each sub-stack having at least 1 illuminated 
    pupil SCI-sized L1 frame for each exposure time, this function processes the 
    frames to create a nonlinearity table. A mean pupil array is created from a 
    separate stack of frames of constant exposure time and used to make a mask; 
    the mask is used to select pixels in each frame in the large array of stacks 
    in order to calculate its mean signal.
    
    Two sub-stacks/groups of frames at each EM gain value contain noncontiguous 
    frames with the same (repeated) exposure time, taken near the start and end 
    of the frame sequence. Their mean signals are computed and used to correct for 
    illumination brightness/sensor sensitivity drifts for all the frames for a 
    given EM gain, depending on when the frames were taken. The repeated exposure 
    time frames should only be repeated once (as opposed to 3 times, etc) and 
    other sets of exposure times for each EM gain should not be repeated.
    
    Note, it is assumed that the frames for the large array of stacks are 
    collected in a systematic way, such that frames having the same exposure 
    time for a given EM gain are collected contiguously (with the exception of 
    the repeated group of frames noted above). The frames within each EM gain 
    group must also be time ordered. For best results, the mean signal in the 
    pupil region for the longest exposure time at each em gain setting should 
    be between 8000 and 10000 DN.
    
    A linear fit is applied to the corrected mean signals versus exposure time. 
    Relative gain values are calculated from the ratio of the mean signals 
    to the linear fit. Nonlinearity is then calculated from the inverse of
    the relative gain and output as an array. The nonlinearity values, along with 
    the actual EM gain for each column and mean counts in DN for each row, are 
    returned as two arrays. One array contains the column headers with 
    actual/measured EM gain, and the other array contains the means in DN and the 
    nonlinearity values. The mean values start with min_write and run through 
    max_write.
    
    Parameters
    ----------
    stack_arr : array-like
        stack_arr is a stack of frames of dimention 3, which is implicitly 
        subdivided into smaller ranges of grouped frames. The frames are EXCAM 
        illuminated pupil L1 SCI frames. There must be one or more unique EM 
        gain values and at least 20 unique exposure times for each EM gain. The 
        number of frames for each EM gain can vary. The size of stack_arr is: 
        Sum(N_t[g]) x 1200 x 2200, where N_t[g] is the number of frames having 
        EM gain value g, and the sum is over g.
    
    exp_time_stack_arr : array-like
        exp_time_stack_arr is an array of dimension 1 of exposure times (in s) 
        corresponding to the frames in stack_arr in the order found there. The 
        length of exp_time_stack_arr must equal the number of frames used to 
        construct stack_arr. There must be at least 20 unique exposure times at 
        each EM gain. The values must be greater than 0.
    
    time_stack_arr : array-like
        time_stack_arr is an array of dimension 1 of date-time strings 
        corresponding to the frames in stack_arr in the same order found there. 
        The length of time_stack_arr must equal the number of frames in 
        stack_arr. All the elements must be unique. The frames in a given group 
        of constant EM gain must be time-ordered.
        
    len_list : list
        len_list is a list of the number of frames in each em gain group of 
        frames in stack_arr in the same order. The sum of elements of len_list 
        must equal to the number of sub-stacks in stack_arr. The number of 
        elements (= the number of unique em gain values) in len_list must be 
        one or greater.
    
    stack_arr2 : array-like
        stack_arr2 is a stack array of EXCAM unity EM gain illuminated pupil L1 
        SCI frames. stack_arr2 contains a stack of frames of uniform exp time, 
        such that the mean signal in the pupil regions is a few thousand DN.
    
    actual_gain_arr : array-like
        The array of actual EM gain values (as opposed to commanded EM gain) 
        corresponding to the number of EM gain values used to construct 
        stack_arr and in the same order. Note: calibrate_nonlin does not 
        calculate actual EM gain values -- they must be provided in this array. 
        The length of actual_gain_arr must equal the length of len_list. Values 
        must be >= 1.0.
        
    norm_val : int
        Value in DN to normalize the nonlinearity values to. Must be greater than 
        0 and must be divisible by 20 without remainder. (1500 to 3000 recommended).
    
    min_write : float
        Minimum mean value in DN to output in nonlin_arr and csv_lines. 
        (800.0 recommended)
    
    max_write : float
        Maximum mean value in DN to output in nonlin_arr and csv_lines. 
        (10000.0 recommended)
    
    mkplot : boolean
        Option to display plots. Default is None. If mkplot is anything other 
        than None, then this option is chosen.
        
    verbose : boolean
        Option to display various diagnostic print messages. Default is None. 
        If verbose is anything other than None, then this option is chosen.
    
    Returns
    -------
    headers: array-like
        1-D array of headers used to build csv-lines. The length is equal to 
        the number of columns in 'nonlin_arr' and is one greater than the 
        length of 'actual_gain_arr'.
    
    nonlin_arr: array-like
        2-D array with nonlinearity values for input signal level (DN) in rows 
        and EM gain values in columns. The input signal in DN is the first column. 
        Signal values start with min_write and run through max_write in steps 
        of 20 DN.
    
    csv_lines : list
        List of strings containing the contents of 'headers' and 'nonlin_arr'.
    
    means_min_max : array-like
        minima and maxima of mean values (in DN) used the fit each for EM gain. 
        The size of means_min_max is N x 2, where N is the length of actual_gain_arr.
    """
    
    # input checks
    # load in default parameters
    from corgidrp.detector import nonlin_params as master_files

    # check pointer yaml file
    if 'offset_colroi1' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'offset_colroi2' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowroi1' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowroi2' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colroi1' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colroi2' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowback11' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowback12' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowback21' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowback22' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colback11' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colback12' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colback21' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colback22' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'min_exp' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'num_bins' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'min_bin' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'min_mask_factor' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'lowess_frac' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rms_low_limit' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rms_upp_limit' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'fit_upp_cutoff1' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'fit_upp_cutoff2' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'fit_low_cutoff1' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'fit_low_cutoff2' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    
    if not isinstance(master_files['offset_colroi1'], (float, int)):
        raise TypeError('offset_colroi1 is not a number')
    if not isinstance(master_files['offset_colroi2'], (float, int)):
        raise TypeError('offset_colroi2 is not a number')
    if not isinstance(master_files['rowroi1'], (float, int)):
        raise TypeError('rowroi1 is not a number')
    if not isinstance(master_files['rowroi2'], (float, int)):
        raise TypeError('rowroi2 is not a number')
    if not isinstance(master_files['colroi1'], (float, int)):
        raise TypeError('colroi1 is not a number')
    if not isinstance(master_files['colroi2'], (float, int)):
        raise TypeError('colroi2 is not a number')
    if not isinstance(master_files['rowback11'], (float, int)):
        raise TypeError('rowback11 is not a number')
    if not isinstance(master_files['rowback12'], (float, int)):
        raise TypeError('rowback12 is not a number')
    if not isinstance(master_files['rowback21'], (float, int)):
        raise TypeError('rowback21 is not a number')
    if not isinstance(master_files['rowback22'], (float, int)):
        raise TypeError('rowback22 is not a number')
    if not isinstance(master_files['colback11'], (float, int)):
        raise TypeError('colback11 is not a number')
    if not isinstance(master_files['colback12'], (float, int)):
        raise TypeError('colback12 is not a number')
    if not isinstance(master_files['colback21'], (float, int)):
        raise TypeError('colback21 is not a number')
    if not isinstance(master_files['colback22'], (float, int)):
        raise TypeError('colback22 is not a number')
    if not isinstance(master_files['min_exp'], (float, int)):
        raise TypeError('min_exp is not a number')
    if not isinstance(master_files['num_bins'], (float, int)):
        raise TypeError('num_bins is not a number')
    if not isinstance(master_files['min_bin'], (float, int)):
        raise TypeError('min_bin is not a number')
    if not isinstance(master_files['min_mask_factor'], (float, int)):
        raise TypeError('min_mask_factor is not a number')
    if not isinstance(master_files['lowess_frac'], (float, int)):
        raise TypeError('lowess_frac is not a number')
    if not isinstance(master_files['rms_low_limit'], (float, int)):
        raise TypeError('rms_low_limit is not a number')
    if not isinstance(master_files['rms_upp_limit'], (float, int)):
        raise TypeError('rms_upp_limit is not a number')
    if not isinstance(master_files['fit_upp_cutoff1'], (float, int)):
        raise TypeError('fit_upp_cutoff1 is not a number')
    if not isinstance(master_files['fit_upp_cutoff2'], (float, int)):
        raise TypeError('fit_upp_cutoff2 is not a number')
    if not isinstance(master_files['fit_low_cutoff1'], (float, int)):
        raise TypeError('fit_low_cutoff1 is not a number')
    if not isinstance(master_files['fit_low_cutoff2'], (float, int)):
        raise TypeError('fit_low_cutoff2 is not a number')
    
    # get relevant constants
    from corgidrp.detector import nonlin_params as constants_config
    offset_colroi1 = constants_config['offset_colroi1']
    offset_colroi2 = constants_config['offset_colroi2']
    rowroi1 = constants_config['rowroi1']
    rowroi2 = constants_config['rowroi2']
    colroi1 = constants_config['colroi1']
    colroi2 = constants_config['colroi2']
    rowback11 = constants_config['rowback11']
    rowback12 = constants_config['rowback12']
    rowback21 = constants_config['rowback21']
    rowback22 = constants_config['rowback22']
    colback11 = constants_config['colback11']
    colback12 = constants_config['colback12']
    colback21 = constants_config['colback21']
    colback22 = constants_config['colback22']
    min_exp_time = constants_config['min_exp']
    num_bins = constants_config['num_bins']
    min_bin = constants_config['min_bin']
    min_mask_factor = constants_config['min_mask_factor']
    lowess_frac = constants_config['lowess_frac']
    rms_low_limit = constants_config['rms_low_limit']
    rms_upp_limit = constants_config['rms_upp_limit']
    fit_upp_cutoff1 = constants_config['fit_upp_cutoff1']
    fit_upp_cutoff2 = constants_config['fit_upp_cutoff2']
    fit_low_cutoff1 = constants_config['fit_low_cutoff1']
    fit_low_cutoff2 = constants_config['fit_low_cutoff2']
    
    if type(stack_arr) != np.ndarray:
        raise TypeError('stack_arr must be an ndarray.')
    if np.ndim(stack_arr) != 3:
        raise CalNonlinException('stack_arr must be 3-D')
    if np.sum(len_list) != len(stack_arr):
        raise CalNonlinException('Number of sub-stacks in stack_arr must '
                'equal the sum of the elements in len_list')
    if len(len_list) < 1:
        raise CalNonlinException('Number of elements in len_list must '
                'be greater than or equal to 1.')
    if len(np.unique(time_stack_arr)) != len(time_stack_arr):
        raise CalNonlinException('All elements of time_stack_arr must be unique.')
    for g_index in range(len(len_list)):
        # Define the start and stop indices
        start_index = int(np.sum(len_list[0:g_index]))
        stop_index = start_index + len_list[g_index]
        # Convert camera times to datetime objects
        ctim_strings = time_stack_arr[start_index:stop_index]
        ctim_datetime = pd.to_datetime(ctim_strings, errors='coerce')
        # Check if the array is time-ordered in increasing order
        is_increasing = np.all(ctim_datetime[:-1] <= ctim_datetime[1:])
        if not is_increasing:
            raise CalNonlinException('Elements of time_stack_arr must be '
                    'in increasing time order for each EM gain value.')
    if type(stack_arr2) != np.ndarray:
        raise TypeError('stack_arr2 must be an ndarray.')
    if np.ndim(stack_arr2) != 3:
        raise CalNonlinException('stack_arr2 must be 3-D (i.e., a stack of '
                '2-D sub-stacks')
    if len(stack_arr2) < 30:
        raise CalNonlinException('Number of frames in stack_arr2 must '
                'be at least 30.')
    check.real_array(exp_time_stack_arr, 'exp_time_stack_arr', TypeError)
    check.oneD_array(exp_time_stack_arr, 'exp_time_stack_arr', TypeError)
    if (exp_time_stack_arr <= min_exp_time).any():
        raise CalNonlinException('Each element of exp_time_stack_arr must be '
            ' greater than min_exp_time.')
    index = 0
    r_flag = True
    for x in range(len(len_list)):
        temp = np.copy(exp_time_stack_arr[index:index+len_list[x]])
        # Unique counts of exposure times
        _, u_counts = np.unique(temp, return_counts=True)
        # Check if all elements are the same
        all_elements_same = np.all(u_counts == u_counts[0])
        if all_elements_same == True:
            r_flag = False
        index = index + len_list[x]
    if not r_flag:
        raise CalNonlinException('each substack of stack_arr must have a '
            'group of frames with a repeated exposure time.')   
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
    
    ######################### start of main code #############################
    
    # Define offset columns ROI range
    offset_colroi = slice(offset_colroi1,offset_colroi2)
    
    # Define pixel ROIs
    rowroi = list(range(rowroi1, rowroi2))
    colroi = list(range(colroi1, colroi2))
    
    # Background subtraction regions
    rowback1 = list(range(rowback11, rowback12))
    rowback2 = list(range(rowback21, rowback22))
    colback1 = list(range(colback11, colback12))
    colback2 = list(range(colback21, colback22))
    
    ####################### create good_mean_frame ###################
    
    nrow = len(stack_arr2[0])
    ncol = len(stack_arr2[0][0])
    
    good_mean_frame = np.zeros((nrow, ncol))
    nFrames = len(stack_arr2)
    
    mean_frame_index = 0
    # Loop over the stack_arr2 frames
    for i in range(nFrames):
        frame = stack_arr2[i]
        frame = frame.astype(np.float64)
    
        # Subtract row-wise medians
        row_meds = np.median(frame[:, offset_colroi], axis=1)
        frame -= row_meds[:, np.newaxis]
    
        # Add this frame to the cumulative good_mean_frame
        good_mean_frame += frame
        mean_frame_index += 1

    # Calculate the average of the frames if required
    if mean_frame_index > 0:
        good_mean_frame /= mean_frame_index 
    
    # plot, if requested
    if mkplot is not None:
        # Slice the good_mean_frame array
        frame_slice = good_mean_frame[np.ix_(rowroi, colroi)]
        # Create a figure and plot the sliced frame
        plt.figure()
        # 'viridis' is a common colormap
        plt.imshow(frame_slice, aspect='equal', cmap='viridis')
        plt.colorbar()
        plt.title('Good quality mean frame')
        plt.show()
    
    # Convert to numpy arrays if they are not already
    rowroi = np.array(rowroi)
    colroi = np.array(colroi)
    
    if mkplot is not None:
        # Plot a histogram of the values within the specified ROI
        roi_values = good_mean_frame[rowroi[:, None], colroi]
        plt.figure()
        # 'auto' lets matplotlib decide the number of bins
        plt.hist(roi_values.flatten(), bins='auto', log=True)
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.title('Histogram of Mean Frame in ROI')
        plt.show()
    
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
    if mkplot is not None:
        # Plot the mask
        plt.figure()
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.colorbar()
        plt.show()
        
        # Plot the mean frame
        plt.figure()
        # 'viridis' is a good default color map
        plt.imshow(good_mean_frame, cmap='viridis')
        plt.title('Mean Frame')
        plt.colorbar()
        plt.show()
    
    # initialize arrays for nonlin results table
    nonlin = []
    means_min_max = []
    
    ######################## loop over em gain values #########################
    for gain_index in range(len(len_list)):
        
        start_index = int(np.sum(len_list[0:gain_index]))
        stop_index = start_index + len_list[gain_index]
        # Convert camera times to datetime objects
        ctime_strings = time_stack_arr[start_index:stop_index]
        ctime_datetime = pd.to_datetime(ctime_strings, errors='coerce')
        
        # Select exp times for this em gain
        exp_time_arr = exp_time_stack_arr[start_index:stop_index]
        
        # select frames for this em gain
        full_flst = stack_arr[start_index:stop_index]
        
        # Unique exposure times and their counts
        exposure_strings_list, counts = np.unique(exp_time_arr, return_counts=True)
        
        # Grouping exposures and finding the max count
        max_count_index = np.argmax(counts)
        repeat_exp = exposure_strings_list[max_count_index]  # Exposure time of repeated frames
        
        # Calculate mean time differences as aid in illumination drift corrections
        group_mean_time = []
        first_flag = False
        
        for t0 in exposure_strings_list:
            idx = np.where(exp_time_arr == t0)[0]
            if t0 != repeat_exp:
                del_s = (ctime_datetime[idx] - ctime_datetime[0]).total_seconds()
                group_mean_time.append(np.mean(del_s))
            elif t0 == repeat_exp and not first_flag:
                idx_2 = len(idx) // 2
                del_s = (ctime_datetime[idx[:idx_2]] - ctime_datetime[0]).total_seconds()
                group_mean_time.append(np.mean(del_s))
                first_flag = True
        
        if verbose is not None:
            print(group_mean_time)
        
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
                    full_flst[idx] for idx, exp_time in enumerate(exp_time_arr) if exp_time == current_exposure_time
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
        
                        # Subtract row-wise medians
                        row_meds = np.median(frame_1[:, offset_colroi], axis=1)
                        frame_1 -= row_meds[:, np.newaxis]
        
                        # Subtract background
                        frame_1_back1 = np.mean(frame_1[rowback1[0]:rowback1[-1]+1, 
                                                        colback1[0]:colback1[-1]+1])
                        frame_1_back2 = np.mean(frame_1[rowback2[0]:rowback2[-1]+1, 
                                                        colback2[0]:colback2[-1]+1])
                        frame_back = (frame_1_back1 + frame_1_back2) / 2
        
                        # Calculate counts and mean in the ROI after background subtraction
                        roi_frame = frame_1[rowroi[0]:rowroi[-1]+1, colroi[0]:colroi[-1]+1] - frame_back
                        frame_count0 = np.sum(roi_frame)
                        frame_mean0 = frame_1 - frame_back
        
                        # Apply mask and calculate the positive mean
                        frame_mean0 *= mask
                        positive_means = frame_mean0[frame_mean0 > 0]
                        frame_mean1 = np.mean(positive_means) if positive_means.size > 0 else np.nan
        
                        frame_count.append(frame_count0)
                        frame_mean.append(frame_mean1)
                        
                        mean_frame_index += 1
                    mean_signal.append(np.mean(frame_mean))
                elif repeat_flag:
                    # for repeated exposure frames, split into the first half/set
                    # and the second half/set
                    first_half = len(selected_files) // 2
                    for i in range(first_half):

                        frame_1 = selected_files[i]
                        frame_1 = frame_1.astype(np.float64)
        
                        # Subtract row-wise medians
                        row_meds = np.median(frame_1[:, offset_colroi], axis=1)
                        frame_1 -= row_meds[:, np.newaxis]

                        # Subtract background
                        frame_1_back1 = np.mean(frame_1[rowback1[0]:rowback1[-1]+1, 
                                                        colback1[0]:colback1[-1]+1])
                        frame_1_back2 = np.mean(frame_1[rowback2[0]:rowback2[-1]+1, 
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
                        frame_mean1 = np.mean(positive_means) if positive_means.size > 0 else np.nan
                        
                        frame_count.append(frame_count0)
                        frame_mean.append(frame_mean1)
                        
                        mean_frame_index += 1
                    mean_signal.append(np.nanmean(frame_mean))
                    repeat1_mean_signal = np.nanmean(frame_mean)
                    
                    second_half = len(selected_files)
                    for i in range(first_half + 1, second_half):
                       
                        frame_1 = selected_files[i]
                        frame_1 = frame_1.astype(np.float64)
        
                        # Subtract row-wise medians
                        row_meds = np.median(frame_1[:, offset_colroi], axis=1)
                        frame_1 -= row_meds[:, np.newaxis]
        
                        # Subtract background
                        frame_1_back1 = np.mean(frame_1[rowback1[0]:rowback1[-1]+1, colback1[0]:colback1[-1]+1])
                        frame_1_back2 = np.mean(frame_1[rowback2[0]:rowback2[-1]+1, colback2[0]:colback2[-1]+1])
                        frame_back = (frame_1_back1 + frame_1_back2) / 2
        
                        # Calculate counts and mean
                        roi_frame = frame_1[rowroi[0]:rowroi[-1]+1, colroi[0]:colroi[-1]+1] - frame_back
                        frame_count0 = np.sum(roi_frame)
                        frame_mean0 = frame_1 - frame_back
                        frame_mean0 *= mask
                        positive_means = frame_mean0[frame_mean0 > 0]
                        frame_mean1 = np.mean(positive_means) if positive_means.size > 0 else np.nan
        
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
        #exp_time_arr = np.array(all_exposure_strings)  # Convert to numpy array if needed
        
        # Find indices of the frames where the exposure time matches repeat_exp
        repeat_times_idx = np.where(exp_time_arr == repeat_exp)[0]  # np.where returns a tuple, extract first element
        
        # Calculate the mean times for the first and second halves of these indices
        first_half = len(repeat_times_idx) // 2
        first_half_mean_time = delta_ctimes_s.iloc[repeat_times_idx[:first_half]].mean()
        
        second_half = len(repeat_times_idx)
        second_half_mean_time = delta_ctimes_s.iloc[repeat_times_idx[first_half:second_half]].mean()
        
        if verbose is not None:
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
        
        if mkplot is not None:
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
        if rms_y_rel_err < rms_low_limit:
            p1 = np.polyfit(filt_exp_times_sorted, corr_mean_signal_sorted, 1)
        elif (rms_y_rel_err >= rms_low_limit) and (rms_y_rel_err < rms_upp_limit):
            p1 = np.polyfit(filt_exp_times_sorted[fit_low_cutoff1:fit_upp_cutoff1], 
                            corr_mean_signal_sorted[fit_low_cutoff1:fit_upp_cutoff1], 1)
        else:
            p1 = np.polyfit(filt_exp_times_sorted[fit_low_cutoff2:fit_upp_cutoff2], 
                            corr_mean_signal_sorted[fit_low_cutoff2:fit_upp_cutoff2], 1)
        y1 = np.polyval(p1, filt_exp_times_sorted)
        
        if mkplot is not None:
            # Plot the fitted line
            plt.plot(filt_exp_times_sorted, y1, label='Fitted Line')
            
            # Show the plot with legend
            plt.legend()
            plt.show()
        
        # Calculating relative gain
        rel_gain = corr_mean_signal_sorted / y1
        
        # Smoothing the relative gain data; larger 'lowess_frac' gives smoother curve
        rel_gain_smoothed = lowess(rel_gain, 
                            corr_mean_signal_sorted, frac=lowess_frac)[:, 1]
        
        # find the min/max values of corrected measured means and append array
        temp_min = np.min(corr_mean_signal_sorted)
        temp_max = np.max(corr_mean_signal_sorted)
        means_min_max.append([temp_min,temp_max])
        
        if mkplot is not None:
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
            plt.show()
        
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
            warnings.warn('norm_val is not between the minimum and maximum values '
                          'of the means for the current EM gain. Extrapolation '
                          'will be used for norm_val.')
        
        if mkplot is not None:
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
            plt.show()
        
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
    headers_temp = np.array([f"{f:.3e}" for f in actual_gain_arr])
    first_col_head = 'nan'
    # insert new header at start of array
    headers = np.insert(headers_temp, 0, first_col_head)
    
    # Create a string buffer
    output_buffer = io.StringIO()
    # Write the header
    output_buffer.write(','.join(headers) + '\n')
    # Use savetxt to write the data to the buffer
    np.savetxt(output_buffer, nonlin_arr3, delimiter=',', fmt='%s')
    # Get the content of the buffer as a string
    csv_content = output_buffer.getvalue()
    # Convert the string to a list of strings (one per line)
    csv_lines = csv_content.split('\n')
    
    means_min_max = np.array(means_min_max)
    
    return (headers, nonlin_arr3, csv_lines, means_min_max)
