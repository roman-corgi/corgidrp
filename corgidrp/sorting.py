import copy
import numpy as np
import time
from datetime import datetime

import corgidrp.data as data

def extract_datetime(datetime_str):
    """
    Convert the value for ext_hdr's 'DATETIME' to a numerical time stamp.

    Args:
      datetime_str: time string in the format 'YYYY-MM-DDTHH:MM:SS.ssssss'

    Returns:
      numerical time stamp in seconds since the epoch (1970-01-01T00:00:00Z)
    """
    dt_obj = datetime.strptime(datetime_str[:26], "%Y-%m-%dT%H:%M:%S.%f")
    output = time.mktime(dt_obj.timetuple()) + dt_obj.microsecond / 1e6
    return output

def sort_remove_frames(dataset_in):
    '''For a given EM gain, this function groups frames consecutive in time stamp into 
    groups with the same exposure time.  Then it finds the two sets with the same 
    exposure time that are the most separated in time.  It then examines any other sets 
    with repeated exposure times, keeps the set of those with the most frames, and 
    discards the rest.
     
    Args:
      dataset_in (list of corgidrp.Dataset): list of datasets with all the frames to be sorted.
        
    Returns:
      inds_leave_out (list): list of indices of frames to be left out of the final
        dataset.
      del_rep_inds_tot (list): list of indices of repeated exposure time sets to be
        left out of the final dataset.
      exptime_cons (list): list of exposure times for the sets, including the ones to be excluded.
      count_cons (list): list of number of frames for each exposure time set, including the ones to be excluded.
      filepath_list (list): list of file paths for the frames, including the ones to be excluded.    
    '''
    
    
    # Frames must be taken consecutively
    cal_frame_time_list = []
    exptime_list = []
    filepath_list = []
    for subset in dataset_in:
        for frame in subset:
            cal_frame_time_list += [extract_datetime(frame.ext_hdr['DATETIME'])]
            exptime_list += [frame.ext_hdr['EXPTIME']]
            filepath_list += [frame.filepath]
    idx_id_sort = np.argsort(cal_frame_time_list)
    cal_frame_time_list_sorted = np.array(cal_frame_time_list)[idx_id_sort]
    exptime_arr = np.array(exptime_list)[idx_id_sort]
    # Count repeated, consecutive elements
    count_cons = [1]
    exptime_cons = [exptime_arr[0]]
    idx_cons = 0
    for exptime in exptime_arr:
        if exptime == exptime_cons[-1]:
            count_cons[idx_cons] += 1
        else:
            idx_cons += 1
            count_cons += [1]
            exptime_cons += [exptime]
    # First index always has a repetition in the previous loop (id=id)
    count_cons[0] -= 1

    count_cons = np.array(count_cons)
    exptime_cons = np.array(exptime_cons)
    # find exptime_cons entries that are the same, and then pick the pair that has widest time separation
    widest_time_sep = 0 #initialize
    for i in range(len(exptime_cons)):
        arr = np.where(exptime_cons==exptime_cons[i])[0]
        if len(arr) > 1:
            time_ind1 = np.sum(count_cons[0:arr[0]+1]).astype(int)-1 # last of first exposure time set
            time_ind2 = np.sum(count_cons[0:arr[-1]]).astype(int) # first of last exposure time set (exptime_cons sets already time-ordered, so pick the first and last for widest time check)
            delta_time = cal_frame_time_list_sorted[time_ind2] - (cal_frame_time_list_sorted[time_ind1]+exptime_cons[arr[0]])
            if delta_time > widest_time_sep:
                widest_time_sep = delta_time
                set1 = arr[0]
                set2 = arr[-1]
                time_ind1_f = time_ind1
                time_ind2_f = time_ind2
    # other repeat sets: keep the part of each set that has the most frames (making these repeated sets not repeated)
    inds_leave_out = []
    del_rep_inds_tot = []
    for i in range(len(exptime_cons)):
        arr = np.where(exptime_cons==exptime_cons[i])[0]
        if len(arr) > 1:
            # in case the repeated set containing the widest time has more than 2 members:
            if set1 in arr and set2 in arr:
                del_rep_inds = np.delete(arr, np.where(arr==set1))
                del_rep_inds = np.delete(del_rep_inds, np.where(del_rep_inds==set2))
            else:
                del_rep_inds = np.where(count_cons[arr] != np.max(count_cons[arr]))[0]
                if len(del_rep_inds) == 0: # then all within arr have the same number of frames
                    # then just pick one 
                    del_rep_inds = np.array([arr[0]]) 
            del_rep_inds_tot = np.append(del_rep_inds_tot, del_rep_inds)
            for i in del_rep_inds:
                time_ind1 = np.sum(count_cons[0:i]).astype(int) # first of exposure time set
                time_ind2 = time_ind1 + count_cons[i] -1 # last of exposure time set
                for k in range(time_ind1, time_ind2+1):
                    ind_remove = np.where(cal_frame_time_list == cal_frame_time_list_sorted[k])[0][0] #[0][0] b/c should only be unique time stamps for all frames
                    inds_leave_out.append(ind_remove)
    # now leave out sets with just 1 frame 
    for j in range(len(exptime_cons)):
        if count_cons[j] <= 1:
            time_ind1 = np.sum(count_cons[0:j]).astype(int) # first of exposure time set
            time_ind2 = time_ind1 + count_cons[j] -1 # last of exposure time set
            for k in range(time_ind1, time_ind2+1):
                    ind_remove = np.where(cal_frame_time_list == cal_frame_time_list_sorted[k])[0][0]
                    inds_leave_out.append(ind_remove)

    return inds_leave_out, del_rep_inds_tot, exptime_cons, count_cons, filepath_list

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
    split_datetime_ds, _ = dataset_cp.split_dataset(exthdr_keywords=['DATETIME'])
    for ds in split_datetime_ds:
        if len(ds) > 1:
            raise Exception('Dataset contains more than one frame with the same DATETIME value.')
    # Split by EMGAIN_C
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
    frame_time_list = []
    for dataset in split_exptime[0]:
        for frame in dataset:
            frame_time_list += [extract_datetime(frame.ext_hdr['DATETIME'])]
    frame_time_sort = sorted(frame_time_list)
    mean_frame_time_list = []
    for frame in split_exptime[0][idx_mean_frame]:
        mean_frame_time_list += [extract_datetime(frame.ext_hdr['DATETIME'])]
    # Choose the frames with consecutive time stamp values 
    mean_frame_time_sort = np.array(mean_frame_time_list)
    mean_frame_time_sort.sort()
    count_cons = [1]
    idx_cons = 0
    for idx in range(len(mean_frame_time_sort)-1):
        overall_idx = frame_time_sort.index(mean_frame_time_sort[idx])
        if mean_frame_time_sort[idx+1] == frame_time_sort[overall_idx+1]:
            count_cons[idx_cons] += 1
        else:
            idx_cons += 1
            count_cons += [1]
    # Choose the largest subset
    idx_mean_frame_cons = np.argmax(count_cons)
    idx_mean_frame_last = np.sum(count_cons[0:idx_mean_frame_cons+1]).astype(int)
    idx_mean_frame_first = idx_mean_frame_last - count_cons[idx_mean_frame_cons]
    frame_id_mean_frame = mean_frame_time_sort[idx_mean_frame_first:idx_mean_frame_last]
    mean_frame_list = []

    n_mean_frame = 0
    for frame in split_exptime[0][idx_mean_frame]:
        if extract_datetime(frame.ext_hdr['DATETIME']) in frame_id_mean_frame:
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

    # Remove MNFRAME frames from unity gain frames
    split_exptime[0].remove(split_exptime[0][idx_mean_frame])
    split_exptime[1].remove(split_exptime[1][idx_mean_frame])
    inds_leave_out, del_rep_inds_tot, exptime_cons, count_cons, unity_gain_filepath_list = sort_remove_frames(split_exptime[0])
    
    # Sort unity gain filenames
    # Update OBSNAME and take profit to check files are in the list
    n_kgain = 0
    cal_frame_list = []
    index = -1 #initiate
    for subset in split_exptime[0]:
        for i in range(len(subset)):
            index += 1
            frame = subset.frames[i]
            if frame.filepath in unity_gain_filepath_list:
                if index in inds_leave_out:
                    continue
                vistype = frame.pri_hdr['VISTYPE']
                frame.pri_hdr['OBSNAME'] = 'KGAIN'
                cal_frame_list += [frame]
                n_kgain += 1

    sorting_summary += (f'K-gain has {n_kgain} unity frames with exposure ' +
        f'times {list(np.delete(exptime_cons, del_rep_inds_tot.astype(int)))} seconds with ' +
        f'{list(np.delete(count_cons, del_rep_inds_tot.astype(int)))} frames each. ')

    # Non-unity gain frames for Non-linearity
    if cal_type.lower()[0:7] == 'non-lin' or cal_type.lower()[0:6] == 'nonlin':
        # Non-unity gain frames
        split_cmdgain[0].remove(split_cmdgain[0][idx_unity])
        split_cmdgain[1].remove(split_cmdgain[1][idx_unity])
        
        n_nonlin = 0
        nonlin_emgain = []
        for idx_gain_set, gain_set in enumerate(split_cmdgain[0]):
            nonunity_split_exptime = gain_set.split_dataset(exthdr_keywords=['EXPTIME'])
            inds_leave_out, del_rep_inds_tot, exptime_cons, count_cons, nonunity_gain_filepath_list = sort_remove_frames(nonunity_split_exptime[0])
            # Update OBSNAME and take profit to check files are in the list
            index = -1 #initiate
            for subset in nonunity_split_exptime[0]:
                for i in range(len(subset)):
                    index += 1
                    frame = subset.frames[i]
                    if frame.filepath in nonunity_gain_filepath_list:
                        if index in inds_leave_out:
                            continue
                        vistype = frame.pri_hdr['VISTYPE']
                        frame.pri_hdr['OBSNAME'] = 'NONLIN'
                        cal_frame_list += [frame]
                        n_nonlin += 1
            nonlin_emgain += [split_cmdgain[1][idx_gain_set]]

            sorting_summary += (f'Non-linearity has {n_nonlin} frames with gains ' +
                f'{nonlin_emgain}')
        
    history = (f'Dataset to calibrate {cal_type.upper()}. A sorting algorithm ' +
        'based on the constraints that NFRAMES, EXPTIME and CMDGAIN have when collecting ' +
        'calibration data for K-gain, Non-linearity and EM-gain vs DAC '
        f"was applied to an input dataset from {vistype} visit files." +
        f'The result: {sorting_summary}')
    print(history)

    dataset_sorted = data.Dataset(mean_frame_list + cal_frame_list)
    dataset_sorted.update_after_processing_step(history)
    # Return Datafrane with mean frame and cal type
    return dataset_sorted
