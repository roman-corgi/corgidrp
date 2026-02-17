import copy
import numpy as np
import time
from datetime import datetime
from astropy.io import fits

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

def sort_remove_frames(dataset_in_list, cal_type, actual_visit=True):
    '''For a given EM gain, this function groups frames consecutive in time stamp into 
    groups with the same exposure time.  Then it finds the two sets with the same 
    exposure time that are the most separated in time.  It then examines any other sets 
    with repeated exposure times, keeps the set of those with the most frames, and 
    discards the rest.
     
    Args:
      dataset_in_list (list of corgidrp.Dataset): list of datasets with all the frames to be sorted.
      cal_type (string): the calibration type. Case insensitive.
        Accepted values are:
        'k-gain' or 'kgain' for K-gain calibration
        'non-lin(earity)', 'nonlin(earity)' for non-linearity calibration, where
        the letters within parenthesis are optional.
      actual_visit (bool): if True, the dataset is expected to be a dataset from an actual visit (frames taken in the specific order for the visit), 
        as opposed to a generic collection of files from which we might extract files that satisfy the requirements of the calibration script. Default is True.
        
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
    for subset in dataset_in_list:
        for frame in subset:
            cal_frame_time_list += [extract_datetime(frame.ext_hdr['DATETIME'])]
            exptime_list += [frame.ext_hdr['EXPTIME']]
            filepath_list += [frame.filepath]
    idx_id_sort = np.argsort(cal_frame_time_list)
    cal_frame_time_list_sorted = np.array(cal_frame_time_list)[idx_id_sort]
    exptime_arr = np.array(exptime_list)[idx_id_sort]
    # Count repeated, consecutive elements
    count_cons = [0]
    exptime_cons = [exptime_arr[0]]
    idx_cons = 0
    for exptime in exptime_arr:
        if exptime == exptime_cons[-1]:
            count_cons[idx_cons] += 1
        else:
            idx_cons += 1
            count_cons += [1]
            exptime_cons += [exptime]

    count_cons = np.array(count_cons)
    exptime_cons = np.array(exptime_cons)
    if len(exptime_cons) < 3: # must have a repeated set, which must have at least one set between
        return None, None, exptime_cons, count_cons, filepath_list
    # find exptime_cons entries that are the same, and then pick the pair that has its repeated set first (since the frames after that would not be in order of increasing exposure time)
    widest_time_sep = -np.inf #initialize
    widest_time_seps = []
    sets1 = []
    sets2 = []
    arrs = np.array([])
    for i in range(len(exptime_cons)):
        arr = np.where(exptime_cons==exptime_cons[i])[0]
        if len(arr) > 1:
            arrs = np.append(arrs, arr[1:])
            time_ind1 = np.sum(count_cons[0:arr[0]+1]).astype(int)-1 # last of first exposure time set
            time_ind2 = np.sum(count_cons[0:arr[-1]]).astype(int) # first of last exposure time set (exptime_cons sets already time-ordered, so pick the first and last for widest time check)
            delta_time = cal_frame_time_list_sorted[time_ind2] - (cal_frame_time_list_sorted[time_ind1]+exptime_cons[arr[0]])
            if delta_time > widest_time_sep:
                widest_time_sep = delta_time
                widest_time_seps.append(delta_time)
                sets1.append(arr[0])
                sets2.append(arr[-1])
                time_ind1_f = time_ind1
                time_ind2_f = time_ind2
    exptime_cons_wo_repeats = exptime_cons.copy()
    exptime_cons_wo_repeats = np.delete(exptime_cons_wo_repeats, arrs.astype(int))
    if len(exptime_cons_wo_repeats) == len(exptime_cons): # no repeatitions
        return None, None, exptime_cons, count_cons, filepath_list
    # find index of first instance of exptime_cons (ignoring repeats) where exposure times are no longer increasing
    # Beyond this boundary would mean going into the territory of EM gain calibration frames
    ########
    # Find the longest set of consecutive positive values in boundary_wo_indices
    boundary_wo_indices = np.where(np.diff(exptime_cons_wo_repeats) < 0)[0] 
    if len(boundary_wo_indices) > 0:  
        if len(boundary_wo_indices) == 1: 
            # if the boundary is nearer to the beginning 
            if len(exptime_cons_wo_repeats) - 1 - boundary_wo_indices[0] > boundary_wo_indices[0] - 1:
                boundary_index = None
                lower_boundary_wo_index = boundary_wo_indices[0]
                lower_boundary_exptime = exptime_cons_wo_repeats[lower_boundary_wo_index]
                lower_boundary_index = np.where(exptime_cons == lower_boundary_exptime)[0][0]
            else: # if nearer to the end; for actual visit:  should be at most one turnaround in non-repeated sets, nearer to the end in all cases for a non-zero length of boundary_wo_indices
                boundary_wo_index = boundary_wo_indices[0]
                boundary_exptime = exptime_cons_wo_repeats[boundary_wo_index]
                boundary_index = np.where(exptime_cons == boundary_exptime)[0][0] # only one occurence since not a repeated set
                lower_boundary_index = 0
        else: 
            # Find largest run of consecutive indices
            boundary_diffwo_index = np.argmax(np.diff(boundary_wo_indices))+1
            lower_boundary_diffwo_index = boundary_diffwo_index - 1
            boundary_wo_index = boundary_wo_indices[boundary_diffwo_index]
            lower_boundary_wo_index = boundary_wo_indices[lower_boundary_diffwo_index]
            boundary_exptime = exptime_cons_wo_repeats[boundary_wo_index]
            lower_boundary_exptime = exptime_cons_wo_repeats[lower_boundary_wo_index]
            boundary_index = np.where(exptime_cons == boundary_exptime)[0][0] # only one occurence since not a repeated set
            lower_boundary_index = np.where(exptime_cons == lower_boundary_exptime)[0][0]
        
    else:
        boundary_index = None 
        lower_boundary_index = 0

    set1 = None
    set2 = None
    if boundary_index is None:
        # then only one run of increasing exposure time sets present (ignoring repeats), so choose set with repeated index closest to last non-repeated index
        last_nonrepeat = np.where(exptime_cons == exptime_cons_wo_repeats[-1])[0][0] # last non-repeated set
        set_ind = np.argmin(np.abs(sets2 - last_nonrepeat))
        set1 = sets1[set_ind] # for an actual visit, this will be within lower boundary index, which would be 0 index; and it doesn't matter if it isn't within increasing exposure time sets if not an actual visit
        set2 = sets2[set_ind]
    else: # for actual visit, if this happens:  sets1 and sets2 would only have one entry anyways
        if actual_visit:
            for i in range(1, len(sets2)+1):
                if sets2[-i] > boundary_index + 1:
                    continue
                else:
                    # then this is the repeated set with the widest time separation that is within the boundary
                    set1 = sets1[-i]
                    set2 = sets2[-i]
                    break
        else:
            # then just pick the last set (the widest time separation)
            set1 = sets1[-1]
            set2 = sets2[-1]

    if set1 is None or set2 is None:
        raise Exception('No repeated exposure time sets found within the frames for the calibration.')
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
                arr_within_boundary = np.delete(arr, np.where(arr>set2)) # only consider repeated sets within the boundary
                if arr_within_boundary.size > 1: # otherwise, not a repeated set effectively
                    del_rep_inds = np.where(count_cons[arr_within_boundary] != np.max(count_cons[arr_within_boundary]))[0]
                    if len(del_rep_inds) == 0: # then all within arr have the same number of frames
                        # then just pick one 
                        del_rep_inds = np.array([arr_within_boundary[0]]) 
                else:
                    del_rep_inds = np.array([]) # append nothing
            del_rep_inds_tot = np.append(del_rep_inds_tot, del_rep_inds)
            for i in del_rep_inds:
                time_ind1 = np.sum(count_cons[0:i]).astype(int) # first of exposure time set
                time_ind2 = time_ind1 + count_cons[i] -1 # last of exposure time set
                for k in range(time_ind1, time_ind2+1):
                    ind_remove = np.where(cal_frame_time_list == cal_frame_time_list_sorted[k])[0][0] #[0][0] b/c should only be unique time stamps for all frames
                    inds_leave_out.append(ind_remove)
    # now leave out any non-repeated sets after boundary
    del_rep_inds_tot = np.append(del_rep_inds_tot, np.arange(set2+1, len(exptime_cons)))
    for i in range(set2+1, len(exptime_cons)):
                time_ind1 = np.sum(count_cons[0:i]).astype(int) # first of exposure time set
                time_ind2 = time_ind1 + count_cons[i] -1 # last of exposure time set
                for k in range(time_ind1, time_ind2+1):
                    ind_remove = np.where(cal_frame_time_list == cal_frame_time_list_sorted[k])[0][0] #[0][0] b/c should only be unique time stamps for all frames
                    inds_leave_out.append(ind_remove)

    if cal_type.lower() == 'k-gain' or cal_type.lower() == 'kgain':
        # now leave out sets with just 1 frame if k gain
        for j in range(len(exptime_cons)):
            if count_cons[j] <= 1:
                time_ind1 = np.sum(count_cons[0:j]).astype(int) # first of exposure time set
                time_ind2 = time_ind1 + count_cons[j] -1 # last of exposure time set
                for k in range(time_ind1, time_ind2+1):
                        ind_remove = np.where(cal_frame_time_list == cal_frame_time_list_sorted[k])[0][0]
                        inds_leave_out.append(ind_remove)
        # also remove repeated sets if k gain since no drift correction employed for k gain calibration 
        # (and so that no exposure time is more heavily weighted than another)
        del_rep_inds_tot = np.append(del_rep_inds_tot, set2)
        time_ind1 = np.sum(count_cons[0:set2]).astype(int) # first of exposure time set
        time_ind2 = time_ind1 + count_cons[set2] -1 # last of exposure time set
        for k in range(time_ind1, time_ind2+1):
            ind_remove = np.where(cal_frame_time_list == cal_frame_time_list_sorted[k])[0][0] #[0][0] b/c should only be unique time stamps for all frames
            inds_leave_out.append(ind_remove)

    return inds_leave_out, del_rep_inds_tot, exptime_cons, count_cons, filepath_list

def sort_pupilimg_frames(
    dataset_in,
    cal_type='', 
    actual_visit=True):
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
        visit files associated with a calibration campaign.  
        Function works if data in Datset is None (useful for large RAM-heavy Dataset), 
        and data is loaded in at the end.  
      cal_type (string): the calibration type. Case insensitive.
        Accepted values are:
        'k-gain' or 'kgain' for K-gain calibration
        'non-lin(earity)', 'nonlin(earity)' for non-linearity calibration, where
        the letters within parenthesis are optional.
      actual_visit (bool): if True, the dataset is expected to be a dataset from an actual visit (frames taken in the specific order for the visit), 
        as opposed to a generic collection of files from which we might extract files that satisfy the requirements of the calibration script. Default is True.

    Returns:

        Dataset with the frames used to generate a mean frame and the frames
        used to calibrate the calibration type: k-gain or non-linearity.
    """
    if actual_visit: # should have appropriate values populated in exthdr
        split_dpam, vals = dataset_in.split_dataset(exthdr_keywords=['DPAMNAME'])
        for val in vals:
            if 'PUPIL' not in val.upper():
                raise Exception(f'Expected DPAMNAME to contain PUPIL, but found {val}.')
        del split_dpam
        split_cfam, vals = dataset_in.split_dataset(exthdr_keywords=['CFAMNAME'])
        for val in vals:
            if 'DARK' in val.upper():
                raise Exception(f'CFAMNAME should not be DARK.')
        del split_cfam
    split_datetime_ds, _ = dataset_in.split_dataset(exthdr_keywords=['DATETIME'])
    for ds in split_datetime_ds:
        if len(ds) > 1:
            raise Exception('Dataset contains more than one frame with the same DATETIME value.')
    del split_datetime_ds
    # Split by EMGAIN_C
    split_cmdgain = dataset_in.split_dataset(exthdr_keywords=['EMGAIN_C'])
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
    # in the datasets, use the AUXFILE with the most frames
    unity_gain_exptimes_auxs = []
    for dataset in split_exptime[0]:
        subs, vals = dataset.split_dataset(prihdr_keywords=['AUXFILE'])
        subs_lengths = [len(sub) for sub in subs]
        if len(subs_lengths) == 0:
            unity_gain_exptimes_auxs.append(dataset)  # no AUXFILE value, so use the dataset as is
        else:
            dominant_ind = np.argmax(subs_lengths)
            unity_gain_exptimes_auxs.append(subs[dominant_ind])
    inds_leave_out, del_rep_inds_tot, exptime_cons, count_cons, unity_gain_filepath_list = sort_remove_frames(unity_gain_exptimes_auxs, cal_type=cal_type, actual_visit=actual_visit)
    if actual_visit and (inds_leave_out is None or del_rep_inds_tot is None):
        raise Exception('No repeated exposure time sets found within the frames for k gain calibration.')
    elif not actual_visit and (cal_type.lower() == 'k-gain' or cal_type.lower() == 'kgain'):
        # then keep them all
        inds_leave_out = []
        del_rep_inds_tot = []
    
    # Sort unity gain filenames
    # Update OBSNAME and take profit to check files are in the list
    n_kgain = 0
    cal_frame_list = []
    index = -1 #initiate
    for subset in unity_gain_exptimes_auxs:
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
        f'times {list(np.delete(exptime_cons, np.array(del_rep_inds_tot).astype(int)))} seconds with ' +
        f'{list(np.delete(count_cons, np.array(del_rep_inds_tot).astype(int)))} frames each. ')

    # Non-unity gain frames for Non-linearity
    if cal_type.lower()[0:7] == 'non-lin' or cal_type.lower()[0:6] == 'nonlin':
        # Non-unity gain frames
        split_cmdgain[0].remove(split_cmdgain[0][idx_unity])
        split_cmdgain[1].remove(split_cmdgain[1][idx_unity])
        
        n_nonlin = 0
        nonlin_emgain = []
        for idx_gain_set, gain_set in enumerate(split_cmdgain[0]):
            nonunity_split_exptime = gain_set.split_dataset(exthdr_keywords=['EXPTIME'])
            # in the datasets, use the AUXFILE with the most frames
            exptimes_auxs = []
            for dataset in nonunity_split_exptime[0]:
                subs, vals = dataset.split_dataset(prihdr_keywords=['AUXFILE'])
                subs_lengths = [len(sub) for sub in subs]
                if len(subs_lengths) == 0:
                    exptimes_auxs.append(dataset)  # no AUXFILE value, so use the dataset as is
                else:
                    dominant_ind = np.argmax(subs_lengths)
                    exptimes_auxs.append(subs[dominant_ind])
            inds_leave_out, del_rep_inds_tot, exptime_cons, count_cons, nonunity_gain_filepath_list = sort_remove_frames(exptimes_auxs, cal_type=cal_type, actual_visit=actual_visit)
            if inds_leave_out is None or del_rep_inds_tot is None:
                continue # no repeated exposure time sets found within the frames for non-linearity calibration, so ignore that EM gain value (could be for EM gain calibration instead)
            # Update OBSNAME and take profit to check files are in the list
            index = -1 #initiate
            for subset in exptimes_auxs:
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
    # free up memory
    del split_exptime, split_cmdgain, dataset_in
    dataset_sorted = data.Dataset(mean_frame_list + cal_frame_list)
    # now load into memory the actual data

    # if dataset_sorted[0].data is None:  XXX
    #     for frame in dataset_sorted.frames:
    #         temp_frame = data.Image(frame.filepath)
    #         if frame.data is None:
    #             frame.data = temp_frame.data
    #             # no err or dq to worry about here since these frames are L1
    #     # get .all_data and .data restored and linked up
    #     dataset_sorted.all_data = np.array([frame.data for frame in dataset_sorted.frames])
    #     for i, frame in enumerate(dataset_sorted.frames):
    #         frame.data = dataset_sorted.all_data[i]
    # dataset_sorted.data_loaded = True

    dataset_sorted.update_after_processing_step(history)
    # Return Dataset with mean frame and cal type
    return dataset_sorted
