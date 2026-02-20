import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

from corgidrp import check
import corgidrp.data as data
from corgidrp.data import Image
from corgidrp.detector import slice_section, detector_areas
from corgidrp.sorting import extract_datetime
from corgidrp.calibrate_nonlin import nonlin_kgain_dataset_2_stack

# Dictionary with constant kgain calibration parameters
kgain_params_default= {
# ROI constants
'rowroi1': 9,
'rowroi2': 1000,
'colroi1': 1199,
'colroi2': 2000,

# read noise bins range limits
'rn_bins1': -200,
'rn_bins2': 201,

# maximum DN value to be included in PTC
'max_DN_val': 13000,

# number of bins in the signal variables
'signal_bins_N': 400,
}

def check_kgain_params(kgain_params):
    """ Checks integrity of kgain parameters in the dictionary kgain_params. 
    
    Args:
        kgain_params (dict):  Dictionary of parameters used for calibrating the k gain.
    """
    
    if 'rowroi1' not in kgain_params:
        raise ValueError('Missing parameter:  rowroi1.')
    if 'rowroi2' not in kgain_params:
        raise ValueError('Missing parameter:  rowroi2.')
    if 'colroi1' not in kgain_params:
        raise ValueError('Missing parameter:  colroi1.')
    if 'colroi2' not in kgain_params:
        raise ValueError('Missing parameter:  colroi2.')
    if 'rn_bins1' not in kgain_params:
        raise ValueError('Missing parameter:  rn_bins1.')
    if 'rn_bins2' not in kgain_params:
        raise ValueError('Missing parameter:  rn_bins2.')
    if 'max_DN_val' not in kgain_params:
        raise ValueError('Missing parameter:  max_DN_val.')
    if 'signal_bins_N' not in kgain_params:
        raise ValueError('Missing parameter:  signal_bins_N.')

    if not isinstance(kgain_params['rowroi1'], (float, int)):
        raise TypeError('rowroi1 is not a number')
    if not isinstance(kgain_params['rowroi2'], (float, int)):
        raise TypeError('rowroi2 is not a number')
    if not isinstance(kgain_params['colroi1'], (float, int)):
        raise TypeError('colroi1 is not a number')
    if not isinstance(kgain_params['colroi2'], (float, int)):
        raise TypeError('colroi2 is not a number')
    if not isinstance(kgain_params['rn_bins1'], (float, int)):
        raise TypeError('rn_bins1 is not a number')
    if not isinstance(kgain_params['rn_bins2'], (float, int)):
        raise TypeError('rn_bins2 is not a number')
    if not isinstance(kgain_params['max_DN_val'], (float, int)):
        raise TypeError('max_DN_val is not a number')
    if not isinstance(kgain_params['signal_bins_N'], (float, int)):
        raise TypeError('signal_bins_N is not a number')

class CalKgainException(Exception):
    """Exception class for calibrate_kgain."""

################### function defs #####################
    
def ptc_bin2(frame_in, mean_frame, binwidth, max_DN):
    """ 
    frame_in is a bias-corrected frame trimmed to the ROI. mean_frame is 
    the scaled high SNR mean frame made from the >=30 frames of uniform 
    exposure time, binwidth is an integer equal to the width of each bin, 
    max_DN is an integer equal to the maximum DN value to be included in 
    the bins.
       
    Args:
      frame_in (np.array): bias corrected frame
      mean_frame (np.array): mean frame of uniform exposure time
      binwidth (int): width of each bin
      max_DN (int): maximum DN value
      
    Returns:
      np.array: mean array
      np.array: mean array
    """
    # calculate the size of output arrays
    rows, cols = frame_in.shape
    out_rows, out_cols = rows // binwidth, cols // binwidth
    
    local_mean_array = np.zeros((out_rows, out_cols))
    local_noise_array = np.zeros((out_rows, out_cols))
        
    # Define the bin edges
    row_bins = np.arange(1, rows + 1, binwidth)
    col_bins = np.arange(1, cols + 1, binwidth)
        
    tot_bins = len(row_bins) * len(col_bins)
    DN_bin = max_DN / tot_bins
        
    # Flatten the arrays for easier indexing
    mean_flat = mean_frame.flatten()
    frame_in_flat = frame_in.flatten()
        
    DN_val = 0
    for m in range(len(row_bins) - 1):
        for n in range(len(col_bins) - 1):
            DN_val += DN_bin
                
            # Create masks based on DN values
            mean_idx = (mean_flat >= DN_val) & (mean_flat < (DN_val + DN_bin))
                
            # Ensure that there is at least one element to calculate mean and std
            if np.any(mean_idx):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    local_mean_array[m, n] = np.nanmean(frame_in_flat[mean_idx])
                    local_noise_array[m, n] = np.nanstd(frame_in_flat[mean_idx])
            else:
                local_mean_array[m, n] = 0
                local_noise_array[m, n] = 0
        
    return local_mean_array, local_noise_array

def diff2std(diff_frame, detector_regions=None):
    """
    calculate the standard deviation of the frame difference, 
    diff_frame within the ROI row and column boundaries.
    
    Args:
      diff_frame (np.array): frame diffference
      detector_regions (dict): a dictionary of detector geometry properties.
        Keys should be as found in detector_areas in detector.py.  Defaults to
        that dictionary. 
      
    Returns:
      np.array: standard deviation of frame difference
    """
        
    selected_area = slice_section(diff_frame, 'SCI', 'prescan_reliable',
        detector_regions)
    std_value = np.nanstd(selected_area.reshape(-1), ddof=1)
    # dividing by sqrt(2) since we want std of one frame
    return std_value / np.sqrt(2)
    
def gauss(x, A, mean, sigma):
    """
    Gauss function. Called by sgaussfit.
    
    Args:
      x (np.array): input array
      A (float): amplitude
      mean (float): mean value
      sigma (float): sigma
    
    Returns:
      np.array: Gauss function
    """
    return A * np.exp(-0.5 * ((x - mean) ** 2) / (sigma ** 2))
    
def sgaussfit(xdata, ydata, gaussinp):
    """
    Find fitting parameters to Gauss function. Called by Single_peakfit. 
    gaussinp is an array containing initial values of A, mean, sigma.
    
    Args:
      xdata (np.array): input x array
      ydata (np.array): input y array
      gaussinp (np.array): initial guess array
    
    Returns:
      np.array: fit parameters
      np.array: fit parameters
    """
    popt, pcov = curve_fit(gauss, xdata, ydata, p0=gaussinp)
    sse = np.sum((ydata - gauss(xdata, *popt)) ** 2)
    return popt, lambda params: (sse, gauss(xdata, *params))
    
def Single_peakfit(xdata, ydata):
    """
    Fit Gauss function to x, y data. Returns mean and sigma. Only sigma 
    is used in main code call.
    
    Args:
      xdata (np.array): x data
      ydata (np.array): y data
    
    Returns:
      float: sigma
    """
    astart = np.nanmax(ydata)
    mustart = xdata[np.argmax(ydata)]
    sigmastart = 10
    sgaussinp = [astart, mustart, sigmastart]
    
    estimates, model = sgaussfit(xdata, ydata, sgaussinp)
    a1, mu1, sigma = estimates

    return sigma
    
def histc_roi(frame, rn_bins, detector_regions=None):
    """
    Histogram of pixel values of frame within the ROI and bins defined in 
    rn_bins. Returns the counts in each bin.
    
    Args:
      frame (np.array): frame
      rn_bins (int): histogram bins
      detector_regions (dict): a dictionary of detector geometry properties.
        Keys should be as found in detector_areas in detector.py.  Defaults to
        that dictionary.      
    
    Returns:
      np.array: counts in each bin
    """
    selected_area = slice_section(frame, 'SCI', 'prescan_reliable',
        detector_regions)
    data_reshaped = selected_area.ravel()
    counts, _ = np.histogram(data_reshaped, bins=rn_bins)
      
    return counts
    
def calculate_mode(arr):
    """
    calculates histogram of an array
    
    Args:
      arr (np.array): input array
    
    Returns:
      np.array: bin center values
      np.array: bin center counts
    """
    counts, bin_edges = np.histogram(arr)
    # Calculate bin centers (values)
    values = (bin_edges[:-1] + bin_edges[1:]) / 2
    max_count_index = np.argmax(counts)
    return values[max_count_index], counts[max_count_index]

def sigma_clip(data, sigma=2.5, max_iters=6):
    """
    Perform sigma-clipping on the data.
      
    Args:
       data (np.array): The input data to be sigma-clipped.
       sigma (float): The number of standard deviations to use for clipping.
       max_iters (int): The maximum number of iterations to perform.
    
    Returns:
      np.ndarray: The sigma-clipped data.
      np.ndarray: A boolean mask where True indicates a clipped value.
    """
    data = np.asarray(data)
    clipped_data = data.copy()
      
    for i in range(max_iters):
        mean = np.nanmean(clipped_data)
        std = np.nanstd(clipped_data)
        mask = np.abs(clipped_data - mean) > sigma * std
        if not np.any(mask):
            break
        clipped_data = clipped_data[~mask]
        
    return clipped_data, mask
    
######################### start of main code #############################

def calibrate_kgain(dataset_kgain, 
                    n_cal=10, n_mean=30, min_val=800, max_val=3000, binwidth=68,
                    make_plot=False,plot_outdir='figures', show_plot=False,
                    logspace_start=-1, logspace_stop=4, logspace_num=200,
                    verbose=False, detector_regions=None, kgain_params=None, apply_dq = True):
    """
    kgain (e-/DN) is calculated from the means and variances
    within the defined minimum and maximum mean values. A photon transfer curve
    is plotted from the std dev and mean values from the bins. 

    Args:
      dataset_kgain (corgidrp.Dataset): The frames in the dataset are
        bias-subtracted. The dataset contains frames belonging to two different
        sets -- Mean frame and a large array of unity gain frames.
        Mean frame: Unity gain frames with constant exposure time. These frames
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
        at the end of the data collection (see TVAC example below). Two subsets with 
        the same exposure time are not strictly needed for k gain calibration, but they are needed 
        for nonlinearity calibration, and the same visit will be used to cover both 
        k gain and nonlinearity calibrations, so there should be two unity gain frame subsets
        with the same exposure time. The mean
        signal of these two subsets is used to correct for illumination
        brightness/sensor sensitivity drifts in nonlinearity calibration 
        for all the frames in the whole set,
        depending on when the frames were taken. There should be no other repeated
        exposure time among the subsets. In TVAC, a total of 110 frames were taken
        within this category. The 110 frames consisted of 22 subsets, each with
        5 frames. All 5 frames had the same exposure time. The exposure times in
        TVAC in seconds were, each repeated 5 times to collect 5 frames in each
        subset -- 0.077, 0.770, 1.538, 2.308, 3.077, 3.846, 4.615, 5.385, 6.154,
        6.923, 7.692, 8.462, 9.231, 10.000, 11.538, 10.769, 12.308, 13.077,
        13.846, 14.615, 15.385, and 1.538 (again).
      n_cal (int):
        Minimum number of sub-stacks used to calibrate K-Gain. The default value
        is 10.
      n_mean (int):
        Minimum number of frames used to generate the mean frame. The default value
        is 30.
      min_val (int): 
        Minimum value (in DN) of mean values from sub-stacks to use in calculating 
        kgain. (> 400 recommended)  
      max_val (int):
        Maximum value (in DN) of mean values from sub-stacks to use in calculating 
        kgain. Choose value that avoids large deviations from linearity at high 
        counts. (< 6,000 recommended)
      binwidth (int):
        (Optional) Width of each bin for calculating std devs and means from each 
        sub-stack in dataset_kgain. Maximum value of binwidth is 800. Notice that small 
        values increase computation time.
        (minimum 10; binwidth between 65 and 75 recommended)
      make_plot (bool): (Optional) generate and store plots. Default is True.
      plot_outdir (str): (Optional) Output directory to store figues. Default is
        'figures'. The default directory is not tracked by git.
      show_plot (bool): (Optional) display the plots. Default is False.
      logspace_start (int): log plot min value in np.logspace.
      logspace_stop (int): log plot max value in np.logspace.
      logspace_num (int): Number of elements in np.logspace.
      verbose (bool): (Optional) display various diagnostic print messages.
        Default is False. 
      detector_regions (dict): a dictionary of detector geometry properties.
        Keys should be as found in detector_areas in detector.py.  Defaults to
        that dictionary.
      kgain_params (dict): (Optional) Dictionary containing row and col specifications
        for the region of interest (indicated by 'rowroi1','rowroi2','colroi1',and 'colroi2').
        The 'roi' needs one square region specified, and 'back' needs two square regions, 
        where a '1' ending indicates the smaller of two values, and a '2' ending indicates the larger 
        of two values.  The coordinates of the square region are specified by matching 
        up as follows: (rowroi1, colroi1), (rowroi2, colroi1), etc. 
        Also must contain:
        'rn_bins1': lower bound of counts histogram for fitting or read noise
        'rn_bins2': upper bound of counts histogram for fitting or read noise 
        'max_DN_val': maximum DN value to be included in photon transfer curve (PTC)
        'signal_bins_N': number of bins in the signal variables of PTC curve
        Defaults to kgain_params_default included in this file.
      apply_dq (bool): consider the dq mask (from cosmic ray detection) or not
    
    Returns:
      corgidrp.data.KGain: kgain estimate from the least-squares fit to the photon
        transfer curve (in e-/DN). The expected value of kgain for EXCAM with
        flight readout sequence should be between 8 and 9 e-/DN
    """
    if kgain_params is None:
        kgain_params = kgain_params_default
        
    check_kgain_params(kgain_params)

    if detector_regions is None:
        detector_regions = detector_areas

    # cast dataset objects into np arrays for convenience
    #cal_list, mean_frame_list, actual_gain = kgain_dataset_2_list(dataset_kgain, apply_dq = apply_dq)
    cal_list, mean_frame_list, _, _, _, actual_gains, _, truncated_set_len = nonlin_kgain_dataset_2_stack(dataset_kgain, apply_dq = apply_dq, cal_type='kgain')
    split_arr = np.arange(0,len(cal_list[0]), truncated_set_len)[1:]
    cal_list = np.split(cal_list[0], split_arr)
    actual_gain = np.nanmean(actual_gains)
    # check number of frames, unique EM value, exposure times and datetimes
    tmp = cal_list[0]
    for idx in range(4):
        try:
            tmp = tmp[idx]
        except:
            pass
    if idx != 3:
        raise CalKgainException('cal_list must be 4-D (i.e., a stack of '
                '3-D sub-stacks)')
    if len(cal_list) < n_cal:
        raise CalKgainException(f'Number of sub-stacks in cal_list must '
                'be more than {n_cal}.')
    if len(cal_list) < 20 :
        print('Number of sub-stacks in cal_list is less than 20, '
        'which is the recommended minimum number for a good fit ')
    for i in range(len(cal_list)):
        check.threeD_array(cal_list[i], 'cal_list['+str(i)+']', TypeError)
        if len(cal_list[i]) < 5:
            raise CalKgainException(f'A sub-stack in cal_list was found with less than 5 '
            'frames, which is the required minimum number per sub-stack')
        if i > 0:
            if len(cal_list[i-1]) != len(cal_list[i]):
                raise CalKgainException('All sub-stacks must have the '
                            'same number of frames and frame shape.')
    tmp = mean_frame_list[0]
    for idx in range(3):
        try:
            tmp = tmp[idx]
        except:
            pass
    if idx != 2:    
        raise CalKgainException('mean_frame_list must be 3-D (i.e., a stack of '
                '2-D sub-stacks')
    if len(mean_frame_list) < n_mean:
        raise CalKgainException(f'Number of sub-stacks in mean_frame_list must '
                'be equal to or greater than {n_mean}.')

    check.real_positive_scalar(actual_gain, 'actual_gain', TypeError)
    if actual_gain != 1:
        raise CalKgainException('Actual gain must equal 1.')
    check.positive_scalar_integer(min_val, 'min_val', TypeError)
    check.positive_scalar_integer(max_val, 'max_val', TypeError)
    if min_val >= max_val:
        raise CalKgainException('max_val must be greater than min_val')
    check.positive_scalar_integer(binwidth, 'binwidth', TypeError)
    if binwidth < 10:
        raise CalKgainException('binwidth must be >= 10.')
    if binwidth > 800:
        raise CalKgainException('binwidth must be < 800.')
    if not isinstance(logspace_start, (float, int)):
        raise TypeError('logplot1 is not a number')
    if not isinstance(logspace_stop, (float, int)):
        raise TypeError('logplot2 is not a number')
    if not isinstance(logspace_num, (float, int)):
        raise TypeError('logplot3 is not a number')
    
    # get relevant constants
    rowroi1 = kgain_params['rowroi1']
    rowroi2 = kgain_params['rowroi2']
    colroi1 = kgain_params['colroi1']
    colroi2 = kgain_params['colroi2']
    rn_bins1 = kgain_params['rn_bins1']
    rn_bins2 = kgain_params['rn_bins2']
    max_DN_val = kgain_params['max_DN_val']
    signal_bins_N = kgain_params['signal_bins_N']

    if make_plot is True:
        # Avoid issues with importing matplotlib on headless servers without GUI
        # support without proper configuration
        import matplotlib.pyplot as plt
        # Output directory
        if os.path.exists(plot_outdir) is False:
            os.mkdir(plot_outdir)
            if verbose:
                print('Output directory for figures created in ', os.getcwd())
    
    # NOTE: binwidth must be <= rowroi and colroi
    rowroi = slice(rowroi1,rowroi2)
    colroi = slice(colroi1,colroi2)
    
    averages = []
    deviations = []
    read_noise = []
    deviations_shot = []
    #mean_signals = []
    
    rn_bins = np.array(range(rn_bins1, rn_bins2))
    bin_locs = rn_bins[0:-1]

    max_DN = max_DN_val; # maximum DN value to be included in PTC
    nrow = len(cal_list[0][0])
    ncol = len(cal_list[0][0][0])
    
    # prepare "good mean frame"
    good_mean_frame = np.zeros((nrow, ncol))
    nFrames2 = len(mean_frame_list)
    for mean_frame_count in range(nFrames2):
        good_mean_frame += mean_frame_list[mean_frame_count]
    
    good_mean_frame = good_mean_frame / nFrames2
    
    # Slice the good_mean_frame array
    frame_slice = good_mean_frame[rowroi, colroi]
    
    # If requested, create a figure and plot the sliced frame
    if make_plot:
        fname = 'kgain_mean_frame'
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
    
    nFrames = len(cal_list[0]) # number of frames in an exposure set
    nSets = len(cal_list) # number of exposure sets
    
    # Start with specific offset indices for list comprehension in jj loop
    index_offsets = [2, 3, 1, 4, 5]
    
    # Start with specific index pairs for list comprehension in jj loop
    index_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    # if nFrames>5, then append additional terms to index_pairs
    if nFrames > 5:
        # append offsets
        index_offsets += [i for i in range(4, nFrames - 1)]
        
        # append index pairs (i, i + 1) for i from 4 to len(frames) - 2
        index_pairs += [(i, i + 1) for i in range(4, nFrames - 1)]
    
    for jj in range(nSets):
        if verbose:
            print(jj)
        
        # multi-frame analysis method
        frames = [cal_list[jj][nFrames - offset] for offset in index_offsets]
        # Calculate frame differences
        frames_diff = [frames[j] - frames[k] for j, k in index_pairs]
        # calculate read noise with std from prescan
        rn_std = [diff2std(frames_diff[x], detector_regions) for x 
            in range(len(frames_diff))]
        
        counts_diff = [histc_roi(frames_diff[x], rn_bins, detector_regions) for x 
            in range(len(frames_diff))]
        
        # calculate read noise from prescan with Gaussian fit
        rn_gauss = [Single_peakfit(bin_locs,counts_diff[x])/np.sqrt(2) for x 
            in range(len(frames_diff))]
        
        # split each frame up into bins, take std and mean of each region
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            mean_frames_mean_curr0 = np.nanmean(frames, axis=0)
        mean_frames_mean_curr = np.nanmean(mean_frames_mean_curr0[rowroi,colroi])
        
        # Calculate the means
        mean_good_mean_frame_roi = np.nanmean(good_mean_frame[rowroi,colroi])
    
        # Compute the scaling factor
        scaling_factor = mean_frames_mean_curr / mean_good_mean_frame_roi
    
        # Apply the scaling to the selected ROI in good_mean_frame
        good_mean_frame_scaled = scaling_factor * good_mean_frame[rowroi,colroi]
        
        # calculate means and stdevs using good_mean_frame_scaled logical indexing
        local_mean_arrays = [ptc_bin2(frame[rowroi, colroi], 
                            good_mean_frame_scaled, 
                            binwidth, max_DN)[0] for frame in frames]
        
        local_noise_arrays = [ptc_bin2(frame[rowroi, colroi], 
                            good_mean_frame_scaled, 
                            binwidth, max_DN)[1] for frame in frames_diff]
        std_diffs = local_noise_arrays / np.sqrt(2)
        
        averages0 = [array.reshape(-1, 1) for array in local_mean_arrays]
        averages.extend(averages0)
        deviations0 = [array.reshape(-1, 1) for array in std_diffs]
        deviations.extend(deviations0)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            added_deviations_shot_arr = [
                    np.sqrt(np.square(np.reshape(std_diffs[x], 
                    shape=(-1, 1))) - complex(rn_std[x])**2)
                    for x in range(len(rn_std))
                    ]
        
        deviations_shot.extend(added_deviations_shot_arr)

        read_noise.extend(rn_std)
        
    averages = np.concatenate([arr.flatten() for arr in averages])
    deviations = np.concatenate([arr.flatten() for arr in deviations])
    deviations_shot = np.concatenate([arr.flatten() for arr in deviations_shot])
    averages_deviations_vector = np.column_stack((averages, np.abs(deviations_shot)))
    
    # Generate linearly spaced bins
    signal_bins = np.linspace(np.nanmin(averages), np.nanmax(averages), signal_bins_N)
    signal_bins = np.insert(signal_bins, 0, 0)  # Insert 0 at the beginning
    
    # Initialize containers for the results
    binned_averages_compiled = []
    binned_averages_error = []
    binned_shot_deviations_compiled = []
    binned_deviations_error = []
    binned_total_deviations = []

    # Loop through each bin, excluding the first which is just 0
    for b in range(1, len(signal_bins)):
        average_bool = (averages_deviations_vector[:, 0] > signal_bins[b-1]) & (averages_deviations_vector[:, 0] < signal_bins[b])
    
        # Filter data within the current bin
        current_binned_averages = averages_deviations_vector[:, 0][average_bool]
        current_binned_deviations = averages_deviations_vector[:, 1][average_bool]
        current_binned_total_deviations = deviations[average_bool]
    
        # Compute statistics if there are data points within the bin
        if current_binned_averages.size > 0:
            binned_averages_compiled.append(np.mean(current_binned_averages))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                binned_averages_error.append(np.std(current_binned_averages, ddof=1) / np.sqrt(current_binned_averages.size))
            binned_shot_deviations_compiled.append(np.mean(current_binned_deviations))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                binned_deviations_error.append(np.std(current_binned_deviations, ddof=1) / np.sqrt(current_binned_averages.size))
            binned_total_deviations.append(np.mean(current_binned_total_deviations))
        else:
            # Append NaN or some other placeholder if no data points in the bin
            binned_averages_compiled.append(np.nan)
            binned_averages_error.append(np.nan)
            binned_shot_deviations_compiled.append(np.nan)
            binned_deviations_error.append(np.nan)
            binned_total_deviations.append(np.nan)

    # Convert lists to numpy arrays
    compiled_binned_averages = np.array(binned_averages_compiled)
    compiled_binned_shot_deviations = np.array(binned_shot_deviations_compiled)
    compiled_binned_total_deviations = np.array(binned_total_deviations)
    
    compiled_binned_deviations_error = np.array(binned_deviations_error)
    
    # define bounds for linearity fit
    # lower bound indices should include as many data points as possible
    # upper bound indices should avoid nonlin region
    lower_bound = np.nanmin(np.where(compiled_binned_averages > min_val)[0])
    upper_bound = np.nanmax(np.where(compiled_binned_averages < max_val)[0])
    
    # Logarithmic transformation of specific array segments
    logged_averages = np.log10(compiled_binned_averages[lower_bound:upper_bound])
    logged_deviations = np.log10(compiled_binned_shot_deviations[lower_bound:upper_bound])
    
    # Find indices of NaN's
    nan_dev = np.isnan(logged_deviations)
    nan_avg = np.isnan(logged_averages)
    
    # Filter out NaN's from the data
    logged_deviations = logged_deviations[~nan_dev & ~nan_avg]
    logged_averages = logged_averages[~nan_dev & ~nan_avg]
    
    # Calculate the error factor
    logged_deviations_error_factor = compiled_binned_deviations_error / compiled_binned_shot_deviations
    logged_deviations_error_factor = logged_deviations_error_factor[lower_bound:upper_bound]
    logged_deviations_error_factor = logged_deviations_error_factor[~nan_dev & ~nan_avg]
    
    # Calculate the logged deviations error
    logged_deviations_error = logged_deviations_error_factor * logged_deviations
    
    # Create a mask for non-NaN values
    non_nan_mask = ~np.isnan(logged_deviations_error)

    # Use the mask to filter out NaN values
    y_err = logged_deviations_error[non_nan_mask]
    x_vals = logged_averages[non_nan_mask]
    y_vals = logged_deviations[non_nan_mask]
    
    # make array of kgain values from variance and mean values
    # equation: k_gain = mean / signal variance
    kgain_arr = 10**(x_vals)/(10**y_vals)**2
    
    if make_plot:
        fname = 'kgain_histogram'
        plt.figure()
        # 'auto' lets matplotlib decide the number of bins
        plt.hist(kgain_arr, bins='auto', log=True)
        plt.title('Histogram of kgain values')
        plt.savefig(f'{plot_outdir}/{fname}')
        if verbose:
            print(f'Figure {fname} stored in {plot_outdir}')
        if show_plot:
            plt.show()
        plt.close()
    
    if make_plot:
        fname = 'kgain_vs_mean'
        plt.figure()
        plt.plot(10**x_vals, kgain_arr, marker='o', linestyle='-', color='b', label='kgain')
        # Set y-axis range
        plt.ylim(6, 11)
        # Add a horizontal line at y = 10
        plt.axhline(y=8.7, color='r', linestyle='--', label='y = 8.7')
        # Add a grid
        plt.grid(True)
        plt.title('kgain versus mean')
        plt.savefig(f'{plot_outdir}/{fname}')
        if verbose:
            print(f'Figure {fname} stored in {plot_outdir}')
        if show_plot:
            plt.show()
        plt.close()
    
    kgain_clipped, _ = sigma_clip(kgain_arr)
    mode_kgain,_ = calculate_mode(kgain_clipped)

    # adopt 'mode_kgain' as the final value to return
    kgain = mode_kgain

    if make_plot:
        fname = 'kgain_clipped_histogram'
        plt.figure()
        # 'auto' lets matplotlib decide the number of bins
        plt.hist(kgain_clipped, bins='auto', log=True)
        plt.title('Histogram of clipped kgain values')
        plt.savefig(f'{plot_outdir}/{fname}')
        if verbose:
            print(f'Figure {fname} stored in {plot_outdir}')
        if show_plot:
            plt.show()
        plt.close()
    
    # parameter for plot
    parm1 = -0.5*np.log10(kgain)
    
    # Gaussian read noise value in DN
    mean_rn_gauss_DN = np.nanmean(rn_gauss)
    mean_rn_gauss_e = mean_rn_gauss_DN * kgain
    
    mean_rn_std_DN = np.nanmean(read_noise)
    mean_rn_std_e = mean_rn_std_DN * kgain
    
    # If requested, plotting
    if make_plot:
        fname = 'kgain_ptc'
        # Create log-spaced averages for plotting
        full_range_averages = np.logspace(logspace_start, logspace_stop, logspace_num)
        
        plt.figure(num=3, figsize=(20, 10))
        
        # Main data plots
        plt.plot(compiled_binned_averages, compiled_binned_total_deviations, 
                 label='Shot Noise + Read Noise')
        # Assuming `lower_bound` is 6
        plt.plot(compiled_binned_averages, 
                 compiled_binned_shot_deviations, label='Shot Noise')
        plt.errorbar(10**logged_averages, 10**logged_deviations, 
                     yerr=10**(logged_deviations_error), color='gray', 
                     label='Fitted Points')
        
        # Extra lines
        plt.plot(10**full_range_averages, 
                 10**(full_range_averages * 0.50 + parm1), 
                 'k-', label='Forced 0.50 Gradient')
        
        # Reference lines and text annotations
        plt.axhline(y=mean_rn_std_DN, color='k', linestyle='--')
        plt.axvline(x=95000, color='k', linestyle='--')
        plt.text(2, 1.5, f'Fitted k gain = {np.round(kgain,1)} e/DN', fontsize=15)
        plt.text(2, 10, 
                 f'Read noise (Gaussian fit) = {np.round(mean_rn_gauss_DN,1)} DN = {mean_rn_gauss_e} e', 
                 fontsize=15)
        plt.text(2, 8, f'Read noise (Std) = {np.round(mean_rn_std_DN,1)} DN = {np.round(mean_rn_std_e,1)} e', 
                 fontsize=15)
        
        # Scaling and labeling
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Signal (DN)')
        plt.ylabel('Noise (DN)')
        plt.title('PTC curve fit')
        plt.grid(True)
        
        # Setting axis limits
        plt.xlim([1, 20000])
        plt.ylim([1, 1000])
        
        # Legend and aesthetics
        plt.legend(loc='upper left')
        
        plt.gca().tick_params(axis='both', which='major', labelsize=18)
        
        # Adjust figure position
        plt.gcf().set_dpi(100)
        plt.gcf().set_size_inches(20, 10)
        
        plt.savefig(f'{plot_outdir}/{fname}')
        if verbose:
            print(f'Figure {fname} stored in {plot_outdir}')
        if show_plot:
            plt.show()
        plt.close()
    
    # prepare PTC array
    ptc_list = [compiled_binned_averages, 
             compiled_binned_shot_deviations]
    ptc = np.column_stack(ptc_list)

    invalid_krn_keywords = data.typical_cal_invalid_keywords + ['EXPTIME', 'EMGAIN_C', 'EMGAIN_A', 'HVCBIAS']
    # Remove specific keywords
    for key in ['PROGNUM', 'EXECNUM', 'CAMPAIGN', 'SEGMENT', 'OBSNUM']:
        if key in invalid_krn_keywords:
            invalid_krn_keywords.remove(key)
    prihdr, exthdr, errhdr, dqhdr = check.merge_headers(dataset_kgain, invalid_keywords=invalid_krn_keywords)
    # Read noise and error
    exthdr['RN'] = mean_rn_gauss_e
    
    # rn err depends on spread of data that determines rn and the error in kgain,
    # so use error propagation to find error in (rn in DN)*kgain
    kgain_err = np.nanstd(kgain_clipped)
    rn_err_DN = np.nanstd(rn_gauss)
    rn_err_e = np.sqrt((kgain*rn_err_DN)**2 + (mean_rn_gauss_DN*kgain_err)**2)
    exthdr['RN_ERR'] = rn_err_e
    exthdr['RN_UNIT'] = 'detected electron'
    
    # Update history
    exthdr['HISTORY'] = f"Kgain and read noise derived from a set of frames on {exthdr['DATETIME']}"

    k_gain = data.KGain(kgain, err = kgain_err, ptc = ptc, pri_hdr = prihdr, ext_hdr = exthdr, input_dataset=dataset_kgain)
    
    return k_gain

def kgain_dataset_2_list(dataset, apply_dq = True):
    """
    Casts the CORGIDRP Dataset object for K-gain calibration into a list of
    numpy arrays sharing the same exposure time. It also returns the list of
    unique EM values and set of exposure times used with each EM. Note: EM gain
    is the commanded values: EMGAIN_C.

    This function also performs a set of tests about the data type and values in
    dataset.

    Args:
        dataset (corgidrp.Dataset): Dataset with a set of of EXCAM illuminated
        pupil L1 SCI frames (counts in DN)
        apply_dq (bool): consider the dq mask (from cosmic ray detection) or not

    Returns:
        list with stack of stacks of data array associated with each frame
        array of exposure times associated with each frame
        array of datetimes associated with each frame

    """
    # Split Dataset
    dataset_cp = dataset.copy()
    split = dataset_cp.split_dataset(exthdr_keywords=['EXPTIME'])

    # Data
    stack = []
    # Mean frame data
    mean_frame_stack = []
    # Exposure times
    exp_times = []
    # Datetimes
    datetimes = []
    # EM gains (There can only be an EM gain in the data used to calibrate K-gain)
    em_gains = []
    # Size of each sub stack
    len_sstack = []
    # Record measured gain of each substack of calibration frames
    gains = []    
    latest_datetime = 0 #initialize
    for idx_set, data_set in enumerate(split[0]):
        # Second layer (array of different exposure times)
        sub_stack = []
        record_exp_time = True
        record_len = True
        record_gain = True
        for frame in data_set.frames:
            if apply_dq:
                bad = np.where(frame.dq > 0)
                frame.data[bad] = np.nan
            if record_exp_time:
                exp_time_mean_frame = frame.ext_hdr['EXPTIME']
                record_exp_time = False

            if frame.ext_hdr['EXPTIME'] != exp_time_mean_frame:
                raise Exception('Frames in the same data set must have the same exposure time')

            if frame.pri_hdr['OBSNAME'] == 'MNFRAME':
                if frame.ext_hdr['EMGAIN_C'] != 1:
                    raise Exception('The commanded gain used to build the mean frame must be unity')
                mean_frame_stack.append(frame.data)
            else:
                if record_len:
                    len_sstack.append(len(data_set.frames))
                    record_len = False
                sub_stack.append(frame.data)
                exp_time = frame.ext_hdr['EXPTIME']
                if isinstance(exp_time, float) is False:
                    raise Exception('Exposure times must be float')
                if exp_time <=0:
                    raise Exception('Exposure times must be positive')
                exp_times.append(exp_time)
                datetime = frame.ext_hdr['DATETIME']
                date_number = extract_datetime(datetime)
                if date_number > latest_datetime:
                    latest_datetime = date_number
                    repeat_ind = len(sub_stack) - 1
                if isinstance(datetime, str) is False:
                    raise Exception('DATETIME must be a string')
                datetimes.append(datetime)
                em_gain = frame.ext_hdr['EMGAIN_C']
                if em_gain < 1:
                    raise Exception('Commanded EM gain must be >= 1')
                em_gains.append(em_gain)
                if record_gain:
                    try: # if EM gain measured directly from frame 
                        gains.append(frame.ext_hdr['EMGAIN_M'])
                    except:
                        if frame.ext_hdr['EMGAIN_A'] > 0: # use applied EM gain if available
                            gains.append(frame.ext_hdr['EMGAIN_A'])
                        else: # use commanded gain otherwise
                            gains.append(frame.ext_hdr['EMGAIN_C'])
                        record_gain = False
                
        # Calibration data may have different subsets
        if len(sub_stack) != 0:
            stack.append(np.stack(sub_stack))

    # Need to split substacks with the same exposure times
    stack_cp = []
    # Get expected size of substacks
    len_sub = min(len_sstack)
    rep_len_sub = 2*len_sub
    # Length of substack must be at least 1
    if len(len_sstack) == 0:
        raise Exception('Substacks must have at least one element')
    for sub in stack:
        if stack.index(sub) == repeat_ind:
            continue
        if len(sub) == len_sub:
            stack_cp.append(sub)
        else:
            # Add extra care confirming all collected substacks have same # frames
            if len(sub)/len_sub != len(sub)//len_sub:
                # truncate substack to the nearest multiple of len_sub
                sub = sub[:len(sub)//len_sub*len_sub]
            idx_0 = 0
            for rep in range(len(sub)//len_sub):
                stack_cp.append(sub[idx_0:idx_0+len_sub])
                idx_0 += len_sub
    if len(stack[repeat_ind]) == rep_len_sub:
        stack_cp.append(sub)
    else:
        rep_sub = stack[repeat_ind]
        # Add extra care confirming all collected substacks have same # frames
        if len(rep_sub)/rep_len_sub != len(rep_sub)//rep_len_sub:
            # truncate substack to the nearest multiple of len_sub
            rep_sub = rep_sub[:len(rep_sub)//rep_len_sub*rep_len_sub]
        idx_0 = 0
        for rep in range(len(rep_sub)//rep_len_sub):
            stack_cp.append(rep_sub[idx_0:idx_0+rep_len_sub])
            idx_0 += rep_len_sub
    stack = stack_cp        
    # All elements of datetimes must be unique
    if len(datetimes) != len(set(datetimes)):
        raise Exception('DATETIMEs cannot be duplicated')
    # There can only be an EM gain in the data used to calibrate K-gain
    if len(set(em_gains)) != 1:
        raise Exception('There can only be one commanded EM gain when calibrating K-Gain')
    if np.any(np.array(gains) < 1):
        raise Exception('Actual EM gains must be greater than or equal to 1')
    # When measuring k_gain, there can only be one gain for all exposure times
    actual_gain = np.nanmean(gains) # not actually used in k gain calibration since frames already gain-divided
    
    return stack, mean_frame_stack, actual_gain
