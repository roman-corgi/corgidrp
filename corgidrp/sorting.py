import copy
import numpy as np

import corgidrp.data as data

def extract_frame_id(filename):
    """
    Extract frame ID from an L1 filename. Structure is assumed to be ending
    like '..._frame_id.fits' where frame_id is a series of digits
    OR from timestamp in format 'cgi_visitid_timestamp_l1_.fits'

    Args:
      filename: L1 filename

    Returns:
      Frame id as an integer
    """
    # Updating to account for the new filename format with timestamp
    # Check if filename ends with data level suffixes (new format with timestamp)
    data_level_suffixes = ['_l1_.fits', '_l2a.fits', '_l2b.fits', '_l3_.fits', '_l4_.fits', '_cal.fits']
    
    for suffix in data_level_suffixes:
        if filename.endswith(suffix):
            # Extract timestamp from filename: cgi_visitid_timestamp_l1_.fits
            # Find the timestamp part (between second and third underscore from end)
            parts = filename.split('_')
            if len(parts) >= 4:
                # Extract timestamp (format: YYYYMMDDtHHMMSSS)
                timestamp = parts[-3]  # Third from end
                # Convert timestamp to integer (use last 3 digits as frame ID)
                try:
                    # Use the last 3 digits as frame ID (should be consecutive integers)
                    return int(timestamp[-3:])
                except (ValueError, IndexError):
                    # Fallback: use the entire timestamp as integer
                    return int(timestamp.replace('t', '').replace('T', ''))
    
    # Original logic for old format with frame_id at the end
    idx_0 = len(filename) - filename[::-1].find('_')
    idx_1 = len(filename) - filename[::-1].find('.') - 1

    return int(filename[idx_0:idx_1])

def sort_pupilimg_frames(
    dataset_in,
    cal_type=''):
    """ 
    Sorting algorithm that given a dataset will output a dataset with the
    frames used to generate a mean frame and the frames used to calibrate
    the calibration type: k-gain. non-linearity.

    The output dataset has an added keyword value in its extended header:
    OBSNAME with values 'MNFRAME' (for mean frame), 'KGAIN' (for K-gain),
    and 'NONLIN' (for non-linearity).

    Args:
      dataset_in (corgidrp.Dataset): dataset with all the frames to be sorted.
        By default, it is expected to contain all the frames from the PUPILIMG
        visit files associated with a calibration campaign
      cal_type (string): the calibration type. Case insensitive.
        Accepted values are:
        'k-gain' or 'kgain' for K-gain calibration
        'non-lin(earity)', 'nonlin(earity)' for non-linearity calibration, where
        the letters within parenthesis are optional.

    Returns:

        Dataset with the frames used to generate a mean frame and the frames
        used to calibrate the calibration type: k-gain or non-linearity.
    """
    # Copy dataset
    dataset_cp = dataset_in.copy()
    # Split by CMDGAIN
    split_cmdgain = dataset_cp.split_dataset(exthdr_keywords=['EMGAIN_C'])
    # Mean frame: split by EXPTIME
    idx_unity = np.where(np.array(split_cmdgain[1])==1)[0][0]
    split_exptime = split_cmdgain[0][idx_unity].split_dataset(exthdr_keywords=['EXPTIME'])
    # Get the set with the most frames
    n_frames_list = np.zeros(len(split_exptime[0]))
    for i_sub, subset in enumerate(split_exptime[0]):
        n_frames_list[i_sub] = len(split_exptime[0][i_sub])
    # Choice: choose the subset with the maximum number of frames
    idx_mean_frame = np.argmax(n_frames_list)
    # Use all frames from the exposure time group with the most frames (simplified approach)
    mean_frame_list = []
    n_mean_frame = 0
    for frame in split_exptime[0][idx_mean_frame]:
        exptime_mean_frame = frame.ext_hdr['EXPTIME']
        # Update keyword OBSNAME
        frame.pri_hdr['OBSNAME'] = 'MNFRAME'
        mean_frame_list += [frame]
        n_mean_frame += 1
            
    sorting_summary = (f'Mean frame has {n_mean_frame} unity frames with' + 
        f' exposure time {exptime_mean_frame} seconds. ')

    # K-gain and non-linearity
    cal_frame_list = []
    if cal_type.lower() == 'k-gain' or cal_type.lower() == 'kgain':
        print('Considering K-gain:')
    elif cal_type.lower()[0:7] == 'non-lin' or cal_type.lower()[0:6] == 'nonlin':
        print('Considering Non-linearity:')
    else:
        raise Exception('Unrecognized calibration type (expected k-gain, non-lin)')

    # Remove main frame frames from unity gain frames
    split_exptime[0].remove(split_exptime[0][idx_mean_frame])
    split_exptime[1].remove(split_exptime[1][idx_mean_frame])
    # Simplified approach: Group frames by exposure time and use all frames with sufficient count
    frame_id_list = []
    exptime_list = []
    unity_gain_filepath_list = []
    for subset in split_exptime[0]:
        for frame in subset:
            frame_id_list += [extract_frame_id(frame.filename)]
            exptime_list += [frame.ext_hdr['EXPTIME']]
            unity_gain_filepath_list += [frame.filepath]
    
    # Group frames by exposure time
    exptime_groups = {}
    for i, exptime in enumerate(exptime_list):
        if exptime not in exptime_groups:
            exptime_groups[exptime] = []
        exptime_groups[exptime].append(i)
    
    # Filter groups that have at least 5 frames (minimum required for K-gain)
    valid_groups = {}
    for exptime, indices in exptime_groups.items():
        if len(indices) >= 5:
            valid_groups[exptime] = indices
    
    # Store for summary
    using_valid_groups = len(valid_groups) > 0
    
    if len(valid_groups) == 0:
        print("Warning: No exposure time groups with >= 5 frames found.")
        print("Using all available frames for K-gain calibration.")
        cal_list = unity_gain_filepath_list
        exptime_cons = list(exptime_groups.keys())
        count_cons = [len(exptime_groups[exp]) for exp in exptime_cons]
    else:
        # Use all frames from valid groups
        cal_indices = []
        exptime_cons = []
        count_cons = []
        for exptime, indices in valid_groups.items():
            cal_indices.extend(indices)
            exptime_cons.append(exptime)
            count_cons.append(len(indices))
        
        cal_list = [unity_gain_filepath_list[i] for i in cal_indices]
    # Update OBSNAME and take profit to check files are in the list
    n_kgain = 0
    cal_frame_list = []
    for frame in dataset_cp:
        if frame.filepath in cal_list:
            vistype = frame.pri_hdr['VISTYPE']
            frame.pri_hdr['OBSNAME'] = 'KGAIN'
            cal_frame_list += [frame]
            n_kgain += 1

    # Calculate summary statistics
    if using_valid_groups:
        min_frames = min(count_cons)
        sorting_summary += (f'K-gain has {n_kgain} unity frames with exposure ' +
            f'times {exptime_cons} seconds with ' +
            f'{min_frames} frames each. ')
    else:
        sorting_summary += (f'K-gain has {n_kgain} unity frames with mixed exposure times and frame counts. ')

    # Non-unity gain frames for Non-linearity
    if cal_type.lower()[0:7] == 'non-lin' or cal_type.lower()[0:6] == 'nonlin':
        # Non-unity gain frames
        split_cmdgain[0].remove(split_cmdgain[0][idx_unity])
        split_cmdgain[1].remove(split_cmdgain[1][idx_unity])
        n_nonlin = 0
        nonlin_emgain = []
        for idx_gain_set, gain_set in enumerate(split_cmdgain[0]):
            # Frames must be taken consecutively
            frame_id_list = []
            exptime_list = []
            gain_filepath_list = []
            for frame in gain_set:
                frame_id_list += [extract_frame_id(frame.filename)]
                exptime_list += [frame.ext_hdr['EXPTIME']]
                gain_filepath_list += [frame.filepath]
            # One can set a stronger condition, though in the end the max set
            if len(frame_id_list) < 3:
                continue
            idx_id_sort = np.argsort(frame_id_list)
            exptime_arr = np.array(exptime_list)[idx_id_sort]
            # We need an increasing series of exposure times with the last one
            # the only repeated value in the series
            idx_subsets = np.where(np.diff(exptime_arr) < 0)[0]
            if len(idx_subsets) == 0:
                continue
            # length of candidate subsets
            nonlin_len = []
            for idx, idx_subset in enumerate(idx_subsets):
                # Add 0th element plus the one lost in diff
                if idx == 0:
                    exptime_tmp = exptime_arr[0:idx_subset+2]
                else:
                    exptime_tmp = exptime_arr[idx_subsets[idx-1]+1:idx_subset+2]
                # Check conditions
                if (exptime_tmp[-1] in exptime_tmp[:-1] and
                    len(exptime_tmp) - 1 == len(set(exptime_tmp))):
                    nonlin_len += [len(exptime_tmp)]
                else:
                    nonlin_len += [-1]
            # COntinue of there are no good candidates
            if np.max(nonlin_len) <= 0:
                continue
            # Find maximum set among good candidates
            if np.argmax(nonlin_len) == 0:
                idx_nonlin_first = 0
                idx_nonlin_last = idx_subsets[0] + 1
            else:
                idx_nonlin_first = idx_subsets[np.argmax(nonlin_len)-1] + 1
                idx_nonlin_last = idx_subsets[np.argmax(nonlin_len)] + 1
            # Sort unity gain filenames
            gain_filepath_arr = np.array(gain_filepath_list)[idx_id_sort]
            cal_list = gain_filepath_arr[idx_nonlin_first:idx_nonlin_last+1]
            # Update OBSNAME and take profit to check files are in the list
            for frame in dataset_cp:
                if frame.filepath in cal_list:
                    vistype = frame.pri_hdr['VISTYPE']
                    frame.pri_hdr['OBSNAME'] = 'NONLIN'
                    cal_frame_list += [frame]
                    n_nonlin += 1
            nonlin_emgain += [split_cmdgain[1][idx_gain_set]]

        sorting_summary += (f'Non-linearity has {n_nonlin} frames with gains ' +
            f'{nonlin_emgain}')
        
    # TODO: Add a HISTORY entry
    history = (f'Dataset to calibrate {cal_type.upper()}. A sorting algorithm ' +
        'based on the constraints that NFRAMES, EXPTIME and CMDGAIN have when collecting ' +
        'calibration data for K-gain, Non-linearity and EM-gain vs DAC '
        f"was applied to an input dataset from {vistype} visit files." +
        f'The result is that {sorting_summary}.')
    print(history)

    dataset_sorted = data.Dataset(mean_frame_list + cal_frame_list)
    dataset_sorted.update_after_processing_step(history)
    # Return Datafrane with mean frame and cal type
    return dataset_sorted
