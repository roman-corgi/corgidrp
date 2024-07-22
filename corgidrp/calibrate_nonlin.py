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
from corgidrp.calibrate_kgain import CalKgainException

from corgidrp import check

# Dictionary with constant non-linearity calibration parameters
nonlin_params = {
    # offset column ROI range
    'offset_colroi1': 799,
    'offset_colroi2': 1000,
    
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
    'num_bins': 50,
    'min_bin': 200,
     
    # factor to mutiply bin_edge values when making mask
    'min_mask_factor': 1.1,
    }
 
def check_nonlin_params(
    ):
    """ Checks integrity of kgain parameters in the dictionary nonlin_params. """
    if 'offset_colroi1' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'offset_colroi2' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowroi1' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowroi2' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colroi1' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colroi2' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowback11' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowback12' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowback21' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rowback22' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colback11' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colback12' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colback21' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'colback22' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'min_exp_time' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'num_bins' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'min_bin' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'min_mask_factor' not in nonlin_params:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    
    if not isinstance(nonlin_params['offset_colroi1'], (float, int)):
        raise TypeError('offset_colroi1 is not a number')
    if not isinstance(nonlin_params['offset_colroi2'], (float, int)):
        raise TypeError('offset_colroi2 is not a number')
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

def frameProc(frame, offset_colroi, emgain):
    """
    simple row-bias subtraction using prescan region
    frame is an L1 SCI-size frame, offset_colroi is the
    column range in the prescan region to use to calculate
    the median for each row. EM gain is the actual emgain used
    to collect the frame.

    Args:
      frame (np.array): L1 frame
      offset_colroi (int): column range
      emgain (int): EM gain value

    Returns:
      np.array: bias subtracted frame
    """

    frame = np.float64(frame)
    row_meds = np.median(frame[:,offset_colroi], axis=1)
    row_meds = row_meds[:, np.newaxis]
    frame -= row_meds
    frame = frame/emgain
    return frame

def calibrate_nonlin(dataset_nl, actual_gain_arr, actual_gain_mean_frame,
                     norm_val = 2500, min_write = 800.0, max_write = 10000.0,
                     lowess_frac = 0.1, rms_low_limit = 0.004, rms_upp_limit = 0.2,
                     fit_upp_cutoff1 = -2, fit_upp_cutoff2 = -3,
                     fit_low_cutoff1 = 2, fit_low_cutoff2 = 1,
                     make_plot=True, plot_outdir='figures', show_plot=False,
                     verbose=False):
    """
    Given a large array of stacks with 1 or more EM gains, and sub-stacks of 
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
    pupil region for the longest exposure time at each EM gain setting should 
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
    
    Args:
      dataset_nl (corgidrp.Dataset): dataset, which is implicitly 
        subdivided into smaller ranges of grouped frames. The frames are EXCAM 
        illuminated pupil L1 SCI frames. There must be one or more unique EM 
        gain values and at least 20 unique exposure times for each EM gain. The 
        number of frames for each EM gain can vary. The size of dataset_cal is: 
        Sum(N_t[g]) x 1200 x 2200, where N_t[g] is the number of frames having 
        EM gain value g, and the sum is over g. Each substack of dataset_cal must
        have a group of frames with a repeated exposure time. In addition, there's
        a set of at least 30 frames used to generate a mean frame. These frames
        have the same exp time, such that the mean signal in the pupil regions
        is a few thousand DN, which helps identify the pixels containing the 
        pupil image. They also have unity EM gain. These frames are identified
        with the kewyord 'OBSTYPE'='MNFRAME' (TBD).
      actual_gain_arr (np.array): the array of actual EM gain values (as opposed
        to commanded EM gain) corresponding to the number of EM gain values used
        to construct dataset_nl and in the same order. Note: calibrate_nonlin does
        not calculate actual EM gain values -- they must be provided in this array. 
        The length of actual_gain_arr must equal the length of len_list. Values 
        must be >= 1.0. 
      actual_gain_mean_frame (float):
        The value of the measured/actual EM gain used to collect the frames used
        to build the mean frame in dataset_kgain. note: commanded EM must be unity.
      norm_val (int): (Optional) Value in DN to normalize the nonlinearity values to.
        Must be greater than 0 and must be divisible by 20 without remainder.
        (1500 to 3000 recommended).
      min_write (float): (Optional) Minimum mean value in DN to output in
        nonlin_arr and csv_lines. (800.0 recommended)
      max_write (float): (Optional) Maximum mean value in DN to output in
        nonlin_arr and csv_lines. (10000.0 recommended)
      lowess_frac (float): (Optional) factor to use in lowess smoothing function,
        larger is smoother
      rms_low_limit (float): (Optional) rms relative error selection limits for
        linear fit. Lower limit.
      rms_upp_limit (float): (Optional) rms relative error selection limits for
        linear fit. Upper limit. rms_upp_limit must be greater than rms_low_limit.
      fit_upp_cutoff1 (int): (Optional) polyfit upper cutoff. The following limits were
        determined with simulated frames. If rms_low_limit < rms_y_rel_err < rms_upp_limit,
        this is the upper value applied to select the data to be fitted.
      fit_upp_cutoff2 (int): (Optional) polyfit upper cutoff. The following limits were
        determined with simulated frames. If rms_y_rel_err >= rms_upp_limit,
        this is the upper value applied to select the data to be fitted.
      fit_low_cutoff1 (int): (Optional) polyfit upper cutoff. The following limits were
        determined with simulated frames. If rms_low_limit < rms_y_rel_err < rms_upp_limit,
        this is the lower value applied to select the data to be fitted.
      fit_low_cutoff2 (int): (Optional) polyfit upper cutoff. The following limits were
        determined with simulated frames. If rms_y_rel_err >= rms_upp_limit,
        this is the lower value applied to select the data to be fitted.
      make_plot (bool): (Optional) generate and store plots. Default is True.
      plot_outdir (str): (Optional) Output directory to store figues. Default is
        'figures'. The default directory is not tracked by git.
      show_plot (bool): (Optional) display the plots. Default is False.
      verbose (bool): (Optional) display various diagnostic print messages.
        Default is False.
    
    Returns:
      headers (np.array): 1-D array of headers used to build csv-lines. The length is equal to 
        the number of columns in 'nonlin_arr' and is one greater than the 
        length of 'actual_gain_arr'.
      nonlin_arr (np.array): 2-D array with nonlinearity values for input signal level (DN) in rows 
        and EM gain values in columns. The input signal in DN is the first column. 
        Signal values start with min_write and run through max_write in steps 
        of 20 DN.
      csv_lines (list): List of strings containing the contents of 'headers' and 'nonlin_arr'.
      means_min_max (np.array): minima and maxima of mean values (in DN) used the fit each for EM gain. 
        The size of means_min_max is N x 2, where N is the length of actual_gain_arr.
    """
    # dataset_nl.all_data must be 3-D 
    if np.ndim(dataset_nl.all_data) != 3:
        raise Exception('dataset_nl.all_data must be 3-D')
    # cast dataset objects into np arrays and retrieve aux information
    cal_arr, mean_frame_arr, exp_arr, datetime_arr, len_list, _ = \
        nonlin_dataset_2_stack(dataset_nl)
    # Get relevant constants
    offset_colroi1 = nonlin_params['offset_colroi1']
    offset_colroi2 = nonlin_params['offset_colroi2']
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
    # mean_frame_arr must have at least 30 frames
    if len(mean_frame_arr) < 30:
        raise Exception('mean_frame_arr must have at least 30 frames')
    if np.sum(len_list) != len(cal_arr):
        raise CalNonlinException('Number of sub-stacks in cal_arr must '
                'equal the sum of the elements in len_list')
    if len(len_list) < 1:
        raise CalNonlinException('Number of elements in len_list must '
                'be greater than or equal to 1.')
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
    if len(mean_frame_arr) < 30:
        raise CalNonlinException('Number of frames in mean_frame_arr must '
                'be at least 30.')
    
    check.real_array(exp_arr, 'exp_arr', TypeError)
    check.oneD_array(exp_arr, 'exp_arr', TypeError)
    if (exp_arr <= min_exp_time).any():
        raise CalNonlinException('Each element of exp_arr must be '
            ' greater than min_exp_time.')
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
    if not r_flag:
        raise CalNonlinException('each substack of cal_arr must have a '
            'group of frames with a repeated exposure time.')   
    if len(len_list) != len(actual_gain_arr):
        raise CalNonlinException('Length of actual_gain_arr be the same as the '
                                 'length of len_list.')
    if sum(1 for number in actual_gain_arr if number < 1) != 0:
        raise CalNonlinException('Each element of actual_gain_arr must be greater '
            'than or equal to 1.')
    check.real_array(actual_gain_arr, 'actual_gain_arr', TypeError)
    check.oneD_array(actual_gain_arr, 'actual_gain_arr', TypeError)
    check.real_positive_scalar(actual_gain_mean_frame, 'actual_gain_mean_frame', TypeError)
    if actual_gain_mean_frame < 1:
        raise CalKgainException('Actual gain must be >= 1.')
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
    if not isinstance(fit_upp_cutoff1, (float, int)):
        raise TypeError('fit_upp_cutoff1 is not a number')
    if not isinstance(fit_upp_cutoff2, (float, int)):
        raise TypeError('fit_upp_cutoff2 is not a number')
    if not isinstance(fit_low_cutoff1, (float, int)):
        raise TypeError('fit_low_cutoff1 is not a number')
    if not isinstance(fit_low_cutoff2, (float, int)):
        raise TypeError('fit_low_cutoff2 is not a number')

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
    
    nrow = len(mean_frame_arr[0])
    ncol = len(mean_frame_arr[0][0])
    
    good_mean_frame = np.zeros((nrow, ncol))
    nFrames = len(mean_frame_arr)

    good_mean_frame = good_mean_frame / nFrames
    
    mean_frame_index = 0
    # Loop over the mean_frame_arr frames
    for i in range(nFrames):
        frame = mean_frame_arr[i]
        frame = frameProc(frame.astype(np.float64), offset_colroi,
            actual_gain_mean_frame)
    
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
    means_min_max = []
    
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
                idx_2 = len(idx) // 2
                del_s = (ctime_datetime[idx[:idx_2]] - ctime_datetime[0]).total_seconds()
                group_mean_time.append(np.mean(del_s))
                first_flag = True
        
        if verbose is True:
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
        
        # Find indices of the frames where the exposure time matches repeat_exp
        repeat_times_idx = np.where(exp_em == repeat_exp)[0]  # np.where returns a tuple, extract first element
        
        # Calculate the mean times for the first and second halves of these indices
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
        if rms_y_rel_err < rms_low_limit:
            p1 = np.polyfit(filt_exp_times_sorted, corr_mean_signal_sorted, 1)
        elif (rms_y_rel_err >= rms_low_limit) and (rms_y_rel_err < rms_upp_limit):
            p1 = np.polyfit(filt_exp_times_sorted[fit_low_cutoff1:fit_upp_cutoff1], 
                            corr_mean_signal_sorted[fit_low_cutoff1:fit_upp_cutoff1], 1)
        else:
            p1 = np.polyfit(filt_exp_times_sorted[fit_low_cutoff2:fit_upp_cutoff2], 
                            corr_mean_signal_sorted[fit_low_cutoff2:fit_upp_cutoff2], 1)
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
        temp_min = np.min(corr_mean_signal_sorted)
        temp_max = np.max(corr_mean_signal_sorted)
        means_min_max.append([temp_min,temp_max])
        
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
            warnings.warn('norm_val is not between the minimum and maximum values '
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
    
    breakpoint()
    return (headers, nonlin_arr3, csv_lines, means_min_max)

def nonlin_dataset_2_stack(dataset):
    """
    Casts the CORGIDRP Dataset object for non-linearity calibration into a stack
    of numpy arrays sharing the same commanded gain value. It also returns the list of
    unique EM values and set of exposure times used with each EM. Note: it also
    performs a set of tests about the integrity of the data type and values in
    the dataset.

    Args:
        dataset (corgidrp.Dataset): Dataset with a set of of EXCAM illuminated
        pupil L1 SCI frames (counts in DN)

    Returns:
        numpy array with stack of stacks of data array associated with each frame
        array of exposure times associated with each frame
        array of datetimes associated with each frame
        list with the number of frames with same EM gain
        List of (commanded) EM gains

    """
    # Split Dataset
    dataset_cp = dataset.copy()
    split = dataset_cp.split_dataset(exthdr_keywords=['CMDGAIN'])
    
    # Calibration data
    stack = []
    # Mean frame data
    mean_frame_stack = []
    record_exp_time = True
# Same exposure time, same EM gain (unity??) Docstrings
    # Exposure times
    exp_times = []
    # Datetimes
    datetimes = []
    # Size of each sub stack
    len_sstack = []
    for idx_set, data_set in enumerate(split[0]):
        # Second layer for calibration data (array of different exposure times)
        sub_stack = []
        len_cal_frames = 0
        for frame in data_set.frames:
            if frame.ext_hdr['OBSTYPE'] == 'MNFRAME':
                if record_exp_time:
                    exp_time_mean_frame = frame.ext_hdr['EXPTIME'] 
                    record_exp_time = False
                if frame.ext_hdr['EXPTIME'] != exp_time_mean_frame:
                    raise Exception('Frames used to build the mean frame must have the same exposure time')
                if frame.ext_hdr['CMDGAIN'] != 1:
                    raise Exception('The commanded gain used to build the mean frame must be unity')
                mean_frame_stack.append(frame.data)
            else:
                len_cal_frames += 1
                sub_stack.append(frame.data)
                exp_time = frame.ext_hdr['EXPTIME']
                if isinstance(exp_time, float) is False:
                    raise Exception('Exposure times must be float')
                if exp_time <=0:
                    raise Exception('Exposure times must be positive')
                exp_times.append(exp_time)
                datetime = frame.ext_hdr['DATETIME']
                if isinstance(datetime, str) is False:
                    raise Exception('DATETIME must be a string')
                datetimes.append(datetime)
        # First layer (array of unique EM values)
        stack.append(np.stack(sub_stack))
        len_sstack.append(len_cal_frames)

    # All elements of datetimes must be unique
    if len(datetimes) != len(set(datetimes)):
        raise Exception('DATETIMEs cannot be duplicated')
    # Length of substack must be at least 1
    if len(len_sstack) == 0:
        raise Exception('Substacks must have at least one element')
    # Every EM gain must be greater than or equal to 1
    if np.any(np.array(split[1]) < 1):
        raise Exception('EM gains must be greater than or equal to 1') 

    return (np.vstack(stack), np.stack(mean_frame_stack), np.array(exp_times),
        np.array(datetimes), len_sstack, np.array(split[1]))
