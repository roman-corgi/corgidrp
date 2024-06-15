import os
from pathlib import Path
import numpy as np
import warnings
from scipy.optimize import curve_fit

class CalKgainException(Exception):
    """Exception class for calibrate_kgain."""

def calibrate_kgain(stack_arr, stack_arr2, emgain, min_val, max_val, 
                    binwidth=68, mkplot=None, verbose=None):
    """Given an array of frame stacks for various exposure times, each sub-stack
    having at least 5 illuminated pupil L1 SCI-size frames having the same 
    exposure time, this function subtracts the prescan bias from each frame. It 
    also creates a mean pupil array from a separate stack of frames of uniform 
    exposure time. The mean pupil array is scaled to the mean of each stack and 
    statistics (mean and std dev) are calculated for bins from the frames in it. 
    kgain (e-/DN) is calculated from the means and variances within the 
    defined minimum and maximum mean values. A photon transfer curve is plotted 
    from the std dev and mean values from the bins. 
    
    Parameters
    ----------
    stack_arr : array-like
        The stack of stacks of EXCAM illuminated pupil L1 SCI frames (counts 
        in DN) having a range of exp times. stack_arr contains a stack of 
        stacks, and all sub-stacks must have the same number of frames, which 
        is a minimum of 5. The frames in a sub-stack must all have the same 
        exposure time time. Must have at least 10 sub-stacks. (More than 20 
        sub-stacks recommended. The mean signal in the pupil region should 
        span from about 100 to about 10000 DN. note: unity em gain is 
        recommended when k-gain is the primary desired product, since it is 
        known more accurately than non-unity values.)
    
    stack_arr2 : array-like
        The stack of EXCAM illuminated pupil L1 SCI frames (counts in DN). 
        stack_arr2 contains a stack of at least 30 frames of uniform exposure 
        time, such that the net mean counts in the pupil region is a few thousand 
        DN (2000 to 4000 DN recommended; note: unity em gain is recommended 
        when k-gain is the primary desired product, since it is known more 
        accurately than non-unity values. stack_arr and stack_arr2 must be 
        obtained under the same positioning of the pupil relative to the 
        detector.
    
    emgain : float
        The value of the measured/actual EM gain used to collect the frames used 
        to build the stack_arr and stack_arr2 arrays. Must be >= 1.0. (note: 
        unity em gain is recommended when k-gain is the primary desired product, 
        since it is known more accurately than non-unity values.)
    
    min_val : int
        Minimum value (in DN) of mean values from sub-stacks to use in calculating 
        kgain. (> 400 recommended)
        
    max_val : int
        Maximum value (in DN) of mean values from sub-stacks to use in calculating 
        kgain. Choose value that avoids large deviations from linearity at high 
        counts. (< 6,000 recommended)
    
    binwidth : int
        Width of each bin for calculating std devs and means from each 
        sub-stack in stack_arr. Maximum value of binwidth is 800. NOTE: small 
        values increase computation time.
        (minimum 10; binwidth between 65 and 75 recommended)
    
    mkplot : boolean
        Option to display plots. Default is None. If mkplot is anything other 
        than None, then this option is chosen.
        
    verbose : boolean
        Option to display various diagnostic print messages. Default is None. 
        If mkplot is anything other than None, then this option is chosen.
    
    Returns
    -------
    kgain: float
        kgain estimate from the least-squares fit to the photon transfer curve 
        (in e-/DN). The expected value of kgain for EXCAM with flight readout 
        sequence should be between 8 and 9 e-/DN.
    
    read_noise_gauss : float
        Read noise estimate from the prescan regions (in e-), calculated from 
        the Gaussian fit std devs (in DN) multiplied by kgain. This value 
        should be considered the true read noise, not affected by the fixed 
        pattern noise. 
    
    read_noise_stdev : float
        Read noise estimate from the prescan regions (in e-), calculated from 
        simple std devs (in DN) multiplied by kgain. This value should be 
        larger than read_noise_gauss and is affected by the fixed pattern noise.
    
    ptc : array-like
        array of size N x 2, where N is the number of bins set by the 'signal_bins_N' 
        parameter in detector.kgain_params. The first column is the mean (DN) and 
        the second column is standard deviation (DN) corrected for read noise.
    
    """

    # SRH: copy stack_arr and stack_arr2 
    
    # input checks
    from corgidrp.detector import kgain_params as master_files
    # check pointer yaml file
    if 'offset_rowroi1' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'offset_rowroi2' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
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
    if 'rn_bins1' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'rn_bins2' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'max_DN_val' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'signal_bins_N' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'logplot1' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'logplot2' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'logplot3' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')

    if not isinstance(master_files['offset_rowroi1'], (float, int)):
        raise TypeError('offset_rowroi1 is not a number')
    if not isinstance(master_files['offset_rowroi2'], (float, int)):
        raise TypeError('offset_rowroi2 is not a number')
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
    if not isinstance(master_files['rn_bins1'], (float, int)):
        raise TypeError('rn_bins1 is not a number')
    if not isinstance(master_files['rn_bins2'], (float, int)):
        raise TypeError('rn_bins2 is not a number')
    if not isinstance(master_files['max_DN_val'], (float, int)):
        raise TypeError('max_DN_val is not a number')
    if not isinstance(master_files['signal_bins_N'], (float, int)):
        raise TypeError('signal_bins_N is not a number')
    if not isinstance(master_files['logplot1'], (float, int)):
        raise TypeError('logplot1 is not a number')
    if not isinstance(master_files['logplot2'], (float, int)):
        raise TypeError('logplot2 is not a number')
    if not isinstance(master_files['logplot3'], (float, int)):
        raise TypeError('logplot3 is not a number')
    
    # check parameters
    if type(stack_arr) != np.ndarray:
        raise TypeError('stack_arr must be an ndarray.')
    if np.ndim(stack_arr) != 4:
        raise CalKgainException('stack_arr must be 4-D (i.e., a stack of '
                '3-D sub-stacks)')
    if len(stack_arr) <= 10:
        raise CalKgainException('Number of sub-stacks in stack_arr must '
                'be more than 10.')
    if len(stack_arr) < 20 :
        warnings.warn('Number of sub-stacks is less than 20, '
        'which is the recommended minimum number for a good fit ')
    for i in range(len(stack_arr)):
        check.threeD_array(stack_arr[i], 'stack_arr['+str(i)+']', TypeError)
        if len(stack_arr[i]) < 5:
            raise CalKgainException('A sub-stack was found with less than 5 '
            'frames, which is the required minimum number per sub-stack')
        if i > 0:
            if np.shape(stack_arr[i-1]) != np.shape(stack_arr[i]):
                raise CalKgainException('All sub-stacks must have the '
                            'same number of frames and frame shape.')
    if type(stack_arr2) != np.ndarray:
        raise TypeError('stack_arr2 must be an ndarray.')
    if np.ndim(stack_arr2) != 3:
        raise CalKgainException('stack_arr must be 3-D (i.e., a stack of '
                '2-D sub-stacks')
    if len(stack_arr2) < 30:
        raise CalKgainException('Number of sub-stacks in stack_arr2 must '
                'be equal to or greater than 30.')
    check.real_positive_scalar(emgain, 'emgain', TypeError)
    if emgain < 1:
        raise CalKgainException('emgain must be >= 1.')
    check.positive_scalar_integer(min_val, 'min_val', TypeError)
    check.positive_scalar_integer(max_val, 'max_val', TypeError)
    if min_val >= max_val:
        raise CalKgainException('max_val must be greater than min_val')
    check.positive_scalar_integer(binwidth, 'binwidth', TypeError)
    if binwidth < 10:
        raise CalKgainException('binwidth must be >= 10.')
    if binwidth > 800:
        raise CalKgainException('binwidth must be < 800.')
    
    ################### function defs #####################
    
    def frameProc(frame, offset_colroi, emgain):
        """ simple row-bias subtraction using prescan region 
        frame is an L1 SCI-size frame, offset_colroi is the 
        column range in the prescan region to use to calculate 
        the median for each row. em gain is the actual emgain used 
        to collect the frame.
        """
        
        frame = np.float64(frame)

        row_meds = np.median(frame[:,offset_colroi], axis=1)
        row_meds = row_meds[:, np.newaxis]
        frame -= row_meds
        frame = frame/emgain
        return frame
    
    def ptc_bin2(frame_in, mean_frame, binwidth, max_DN):
        """ frame_in is a bias-corrected frame trimmed to the ROI. mean_frame is 
        the scaled high SNR mean frame made from the >=30 frames of uniform 
        exposure time, binwidth is an integer equal to the width of each bin, 
        max_DN is an integer equal to the maximum DN value to be included in 
        the bins.
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
                    local_mean_array[m, n] = np.nanmean(frame_in_flat[mean_idx])
                    local_noise_array[m, n] = np.nanstd(frame_in_flat[mean_idx])
                else:
                    local_mean_array[m, n] = 0
                    local_noise_array[m, n] = 0
        
        return local_mean_array, local_noise_array

    def diff2std(diff_frame, offset_rowroi, offset_colroi):
        """calculate the standard deviation of the frame difference, 
        diff_frame within the ROI row and column boundaries.
        """
        
        selected_area = diff_frame[offset_rowroi, offset_colroi]
        std_value = np.std(selected_area.reshape(-1), ddof=1)
        # dividing by sqrt(2) since we want std of one frame
        return std_value / np.sqrt(2)
    
    def gauss(x, A, mean, sigma):
        """Gauss function. Called by sgaussfit."""
        return A * np.exp(-0.5 * ((x - mean) ** 2) / (sigma ** 2))
    
    def sgaussfit(xdata, ydata, gaussinp):
        """Find fitting parameters to Gauss function. Called by Single_peakfit. 
        gaussinp is an array containing initial values of A, mean, sigma."""
        popt, pcov = curve_fit(gauss, xdata, ydata, p0=gaussinp)
        sse = np.sum((ydata - gauss(xdata, *popt)) ** 2)
        return popt, lambda params: (sse, gauss(xdata, *params))
    
    def Single_peakfit(xdata, ydata):
        """Fit Gauss funxtion to x, y data. Returns mean and sigma. Only sigma 
        is used in main code call."""
        astart = np.max(ydata)
        mustart = xdata[np.argmax(ydata)]
        sigmastart = 10
        sgaussinp = [astart, mustart, sigmastart]
    
        estimates, model = sgaussfit(xdata, ydata, sgaussinp)
        a1, mu1, sigma = estimates

        return sigma
    
    def histc_roi(frame,offset_rowroi,offset_colroi,rn_bins):
        """Histogram of pixel values of frame within the ROI and bins defined in 
        rn_bins. Returns the counts in each bin."""
        selected_area = frame[offset_rowroi,offset_colroi]
        data_reshaped = selected_area.ravel()
        counts, _ = np.histogram(data_reshaped, bins=rn_bins)
        
        return counts
    
    def calculate_mode(arr):
        counts, bin_edges = np.histogram(arr)
        # Calculate bin centers (values)
        values = (bin_edges[:-1] + bin_edges[1:]) / 2
        max_count_index = np.argmax(counts)
        return values[max_count_index], counts[max_count_index]

    def sigma_clip(data, sigma=2.5, max_iters=6):
        """
        Perform sigma-clipping on the data.
        
        Parameters:
        data (array-like): The input data to be sigma-clipped.
        sigma (float): The number of standard deviations to use for clipping.
        max_iters (int): The maximum number of iterations to perform.
    
        Returns:
        clipped_data (np.ndarray): The sigma-clipped data.
        mask (np.ndarray): A boolean mask where True indicates a clipped value.
        """
        data = np.asarray(data)
        clipped_data = data.copy()
        
        for i in range(max_iters):
            mean = np.mean(clipped_data)
            std = np.std(clipped_data)
            mask = np.abs(clipped_data - mean) > sigma * std
            if not np.any(mask):
                break
            clipped_data = clipped_data[~mask]
        
        return clipped_data, mask
    
    ######################### start of main code #############################
    
    # get relevant constants
    from corgidrp.detector import kgain_params as constants_config
    offset_rowroi1 = constants_config['offset_rowroi1']
    offset_rowroi2 = constants_config['offset_rowroi2']
    offset_colroi1 = constants_config['offset_colroi1']
    offset_colroi2 = constants_config['offset_colroi2']
    rowroi1 = constants_config['rowroi1']
    rowroi2 = constants_config['rowroi2']
    colroi1 = constants_config['colroi1']
    colroi2 = constants_config['colroi2']
    rn_bins1 = constants_config['rn_bins1']
    rn_bins2 = constants_config['rn_bins2']
    max_DN_val = constants_config['max_DN_val']
    signal_bins_N = constants_config['signal_bins_N']
    log_plot1 = constants_config['logplot1']
    log_plot2 = constants_config['logplot2']
    log_plot3 = constants_config['logplot3']

    if mkplot is not None:
        # Avoid issues with importing matplotlib on headless servers without GUI
        # support without proper configuration
        import matplotlib.pyplot as plt
    
    # Prescan region
    offset_rowroi = slice(offset_rowroi1,offset_rowroi2)
    offset_colroi = slice(offset_colroi1,offset_colroi2)
    
    # NOTE: binwidth must be <= rowroi and colroi
    rowroi = slice(rowroi1,rowroi2)
    colroi = slice(colroi1,colroi2)
    
    # ROI for frameProc
    colroi_fp = slice(offset_colroi1,offset_colroi2)
    
    averages = []
    deviations = []
    read_noise = []
    deviations_shot = []
    #mean_signals = []
    
    rn_bins = np.array(range(rn_bins1, rn_bins2))
    bin_locs = rn_bins[0:-1]

    max_DN = max_DN_val; # maximum DN value to be included in PTC
    nrow = len(stack_arr[0][0])
    ncol = len(stack_arr[0][0][0])
    
    # prepare "good mean frame"
    good_mean_frame = np.zeros((nrow, ncol))
    nFrames2 = len(stack_arr2)
    for mean_frame_count in range(nFrames2):
        frame = stack_arr2[mean_frame_count]
        frame = frameProc(frame,colroi_fp,emgain)
        
        good_mean_frame += frame  # Accumulate into good_mean_frame
    
    good_mean_frame = good_mean_frame / nFrames2
    
    # Slice the good_mean_frame array
    frame_slice = good_mean_frame[rowroi, colroi]
    
    # If requested, create a figure and plot the sliced frame
    if mkplot is not None:
        plt.figure()
        # 'viridis' is a common colormap
        plt.imshow(frame_slice, aspect='equal', cmap='viridis')
        plt.colorbar()
        plt.title('Good quality mean frame')
        plt.show()
    
    nFrames = len(stack_arr[0]) # number of frames in an exposure set
    nSets = len(stack_arr) # number of exposure sets
    
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
        if verbose is not None:
            print(jj)
        
        # multi-frame analysis method
        frames = [stack_arr[jj][nFrames - offset] for offset in index_offsets]
        # subtract prescan row medians
        frames = [frameProc(frames[x], colroi_fp, emgain) for x in range(len(frames))]
        # Calculate frame differences
        frames_diff = [frames[j] - frames[k] for j, k in index_pairs]
        # calculate read noise with std from prescan
        rn_std = [diff2std(frames_diff[x], offset_rowroi, offset_colroi) for x 
            in range(len(frames_diff))]
        
        # rn_std = [16.0 for x 
        #     in range(len(frames_diff))]
        
        counts_diff = [histc_roi(frames_diff[x], offset_rowroi, offset_colroi, rn_bins) for x 
            in range(len(frames_diff))]
        
        # calculate read noise from prescan with Gaussian fit
        rn_gauss = [Single_peakfit(bin_locs,counts_diff[x])/np.sqrt(2) for x 
            in range(len(frames_diff))]
        
        # split each frame up into bins, take std and mean of each region
        mean_frames_mean_curr0 = np.mean(frames, axis=0)
        mean_frames_mean_curr = np.mean(mean_frames_mean_curr0[rowroi,colroi])
        
        # Calculate the means
        mean_good_mean_frame_roi = np.mean(good_mean_frame[rowroi,colroi])
    
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
        
        added_deviations_shot_arr = [
                np.sqrt(np.square(np.reshape(std_diffs[x], 
                newshape=(-1, 1))) - complex(rn_std[x])**2)
                for x in range(len(rn_std))
                ]
        
        deviations_shot.extend(added_deviations_shot_arr)

        read_noise.extend(rn_std)
        
    averages = np.concatenate([arr.flatten() for arr in averages])
    deviations = np.concatenate([arr.flatten() for arr in deviations])
    deviations_shot = np.concatenate([arr.flatten() for arr in deviations_shot])
    averages_deviations_vector = np.column_stack((averages, np.abs(deviations_shot)))
    
    # Generate linearly spaced bins
    signal_bins = np.linspace(np.min(averages), np.max(averages), signal_bins_N)
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
            binned_averages_error.append(np.std(current_binned_averages, ddof=1) / np.sqrt(current_binned_averages.size))
            binned_shot_deviations_compiled.append(np.mean(current_binned_deviations))
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
    lower_bound = np.min(np.where(compiled_binned_averages > min_val)[0])
    upper_bound = np.max(np.where(compiled_binned_averages < max_val)[0])
    
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
    
    if mkplot is not None:
        plt.figure()
        # 'auto' lets matplotlib decide the number of bins
        plt.hist(kgain_arr, bins='auto', log=True)
        plt.title('Histogram of kgain values')
        plt.show()
    
    if mkplot is not None:
        plt.figure()
        plt.plot(10**x_vals, kgain_arr, marker='o', linestyle='-', color='b', label='kgain')
        # Set y-axis range
        plt.ylim(6, 11)
        # Add a horizontal line at y = 10
        plt.axhline(y=8.7, color='r', linestyle='--', label='y = 8.7')
        # Add a grid
        plt.grid(True)
        plt.title('kgain versus mean')
        plt.show()
    
    kgain_clipped, _ = sigma_clip(kgain_arr)
    mode_kgain,_ = calculate_mode(kgain_clipped)

    # adopt 'mode_kgain' as the final value to return
    kgain = mode_kgain

    if mkplot is not None:
       plt.figure()
       # 'auto' lets matplotlib decide the number of bins
       plt.hist(kgain_clipped, bins='auto', log=True)
       plt.title('Histogram of clipped kgain values')
       plt.show() 
    
    # parameter for plot
    parm1 = -0.5*np.log10(kgain)
    
    # Gaussian read noise value in DN
    mean_rn_gauss_DN = np.mean(rn_gauss)
    mean_rn_gauss_e = mean_rn_gauss_DN * kgain
    
    mean_rn_std_DN = np.mean(read_noise)
    mean_rn_std_e = mean_rn_std_DN * kgain
    
    # If requested, plotting
    if mkplot is not None:
        # Create log-spaced averages for plotting
        full_range_averages = np.logspace(log_plot1, log_plot2, log_plot3)
        
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
        
        # Display the plot
        plt.show()
    
    # prepare PTC array
    ptc_list = [compiled_binned_averages, 
             compiled_binned_shot_deviations]
    ptc = np.column_stack(ptc_list)
    
    return (kgain, mean_rn_gauss_e, mean_rn_std_e, ptc)
    


"""
Class to hold input-checking functions to minimize repetition
Note: This module, used by calibrate_kgain.py and calibrate_nonlin.py is included
here because at this moment it is not clear if the functions in the module are
of general utility for corgidrp
"""
import numbers
# String check support
string_types = (str, bytes)
# Int check support
int_types = (int, np.integer)
class check:
    
    class CheckException(Exception):
        pass
    
    def _checkname(vname):
        """
        Internal check that we can use vname as a string for printing
        """
        if not isinstance(vname, string_types):
            raise CheckException('vname must be a string when fed to check ' + \
                                 'functions')
        pass
    
    
    def _checkexc(vexc):
        """
        Internal check that we can raise from the vexc object
        """
        if not isinstance(vexc, type): # pre-check it is class-like
            raise CheckException('vexc must be a Exception, or an object ' + \
                                 'descended from one when fed to check functions')
        if not issubclass(vexc, Exception):
            raise CheckException('vexc must be a Exception, or an object ' + \
                                 'descended from one when fed to check functions')
        pass
    
    
    def real_positive_scalar(var, vname, vexc):
        """
        Checks whether an object is a real positive scalar.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        if not isinstance(var, numbers.Number):
            raise vexc(vname + ' must be scalar')
        if not np.isrealobj(var):
            raise vexc(vname + ' must be real')
        if var <= 0:
            raise vexc(vname + ' must be positive')
        return var
    
    
    def real_array(var, vname, vexc):
        """
        Checks whether an object is a real numpy array, or castable to one.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        var = np.array(var, copy=False)  # cast to array
        if len(var.shape) == 0:
            raise vexc(vname + ' must have length > 0')
        if not np.isrealobj(var):
            raise vexc(vname + ' must be a real array')
        # skip 'c' as we don't want complex; rest are non-numeric
        if not var.dtype.kind in ['b', 'i', 'u', 'f']:
            raise vexc(vname + ' must be a real numeric type to be real')
        return var
    
    def oneD_array(var, vname, vexc):
        """
        Checks whether an object is a 1D numpy array, or castable to one.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        var = np.array(var, copy=False) # cast to array
        if len(var.shape) != 1:
            raise vexc(vname + ' must be a 1D array')
        if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
            raise vexc(vname + ' must be a real or complex 1D array')
        return var
    
    
    def twoD_array(var, vname, vexc):
        """
        Checks whether an object is a 2D numpy array, or castable to one.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        var = np.array(var, copy=False) # cast to array
        if len(var.shape) != 2:
            raise vexc(vname + ' must be a 2D array')
        if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
            raise vexc(vname + ' must be a real or complex 2D array')
        return var
    
    
    def twoD_square_array(var, vname, vexc):
        """
        Checks whether an object is a 2D square array_like.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        var = np.array(var, copy=False) # cast to array
        if len(var.shape) != 2:
            raise vexc(vname + ' must be a 2D array')
        else: # is 2-D
            if not var.shape[0] == var.shape[1]:
                raise vexc(vname + ' must be a square 2D array')
        if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
            raise vexc(vname + ' must be a real or complex square 2D array')
        return var
    
    def threeD_array(var, vname, vexc):
        """
        Checks whether an object is a 3D numpy array, or castable to one.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        var = np.array(var, copy=False) # cast to array
        if len(var.shape) != 3:
            raise vexc(vname + ' must be a 3D array')
        if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
            raise vexc(vname + ' must be a real or complex 3D array')
        return var
    
    
    
    def real_scalar(var, vname, vexc):
        """
        Checks whether an object is a real scalar.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        if not isinstance(var, numbers.Number):
            raise vexc(vname + ' must be scalar')
        if not np.isrealobj(var):
            raise vexc(vname + ' must be real')
        return var
    
    
    def real_nonnegative_scalar(var, vname, vexc):
        """
        Checks whether an object is a real nonnegative scalar.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        if not isinstance(var, numbers.Number):
            raise vexc(vname + ' must be scalar')
        if not np.isrealobj(var):
            raise vexc(vname + ' must be real')
        if var < 0:
            raise vexc(vname + ' must be nonnegative')
        return var
    
    def positive_scalar_integer(var, vname, vexc):
        """
        Checks whether an object is a positive scalar integer.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        if not isinstance(var, numbers.Number):
            raise vexc(vname + ' must be scalar')
        if not isinstance(var, int_types):
            raise vexc(vname + ' must be integer')
        if var <= 0:
            raise vexc(vname + ' must be positive')
        return var
    
    
    def nonnegative_scalar_integer(var, vname, vexc):
        """
        Checks whether an object is a nonnegative scalar integer.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        if not isinstance(var, numbers.Number):
            raise vexc(vname + ' must be scalar')
        if not isinstance(var, int_types):
            raise vexc(vname + ' must be integer')
        if var < 0:
            raise vexc(vname + ' must be nonnegative')
        return var
    
    
    def scalar_integer(var, vname, vexc):
        """
        Checks whether an object is a scalar integer (no sign dependence).
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        if not isinstance(var, numbers.Number):
            raise vexc(vname + ' must be scalar')
        if not isinstance(var, int_types):
            raise vexc(vname + ' must be integer')
        return var
    
    
    def string(var, vname, vexc):
        """
        Checks whether an object is a string.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        if not isinstance(var, string_types):
            raise vexc(vname + ' must be a string')
        return var
    
    def boolean(var, vname, vexc):
        """
        Checks whether an object is a bool.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        if not isinstance(var, bool):
            raise vexc(vname + ' must be a bool')
        return var
    
    
    def dictionary(var, vname, vexc):
        """
        Checks whether an object is a dictionary.
    
        Arguments:
         var: variable to check
         vname: string to output in case of error for debugging
         vexc: Exception to raise in case of error for debugging
    
        Returns:
         returns var
    
        """
        check._checkname(vname)
        check._checkexc(vexc)
    
        if not isinstance(var, dict):
            raise vexc(vname + ' must be a dict')
        return var
    
