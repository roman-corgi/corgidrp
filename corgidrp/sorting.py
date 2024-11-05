# Mac: ulimit -n 500 to be able to open the 500 files

import copy
import numpy as np

import corgidrp.data as data

def extract_frame_id(filename):
    """ Extract frame ID from an L1 filename. Structure is assumed to be
        ending like '..._frame_id.fits' where frame_id is a series of digits
    """
    idx_0 = len(filename) - filename[::-1].find('_')
    idx_1 = len(filename) - filename[::-1].find('.') - 1

    return filename[idx_0:idx_1]

def sorting(
    dataset_in,
    cal_type=None):
    """ TBW

    Args: TBW

    Returns:
    """
    # Copy dataset
    dataset_cp = dataset_in.copy()

    # Split by CMDGAIN
    split_cmdgain = dataset_cp.split_dataset(exthdr_keywords=['CMDGAIN'])
    # Mean frame: split by EXPTIME
    idx_unity = np.where(np.array(split_cmdgain[1])==1)[0][0]
    split_exptime = split_cmdgain[0][idx_unity].split_dataset(exthdr_keywords=['EXPTIME'])
    # Get the set with the most frames
    n_frames_list = np.zeros(len(split_exptime[0]))
    for i_sub, subset in enumerate(split_exptime[0]):
        n_frames_list[i_sub] = len(split_exptime[0][i_sub])
    # Choice: choose the subset with the maximum number of frames
    idx_mean_frame = np.argmax(n_frames_list)
    frame_id_list = []
    for frame in split_exptime[0][idx_mean_frame]:
        frame_id_list += [extract_frame_id(frame.filename)]
    # Choose the frames with consecutive ID numbers (same row in AUX file)
    frame_id_sort = np.array(frame_id_list).astype(int)
    frame_id_sort.sort()
    frame_id_sort = \
        frame_id_sort[np.where(np.diff(frame_id_sort) == 1)]
    mean_frame_list = []
    n_mean_frame = 0
    for frame in split_exptime[0][idx_mean_frame]:
        if int(extract_frame_id(frame.filename)) in frame_id_sort:
            # Update keyword OBSTYPE
            frame.pri_hdr['OBSTYPE'] = 'MNFRAME'
            mean_frame_list += [frame]
            n_mean_frame += 1
            
    print(f"Mean frame has {n_mean_frame} unity frames with exposure time {frame.ext_hdr['EXPTIME']} seconds")

    # K-gain
    cal_frame_list = []
    if cal_type.lower() == 'k-gain' or cal_type.lower() == 'kgain':
        print('Considering K-gain')
        # Remove main frame frames from unity gain frames
        split_exptime[0].remove(split_exptime[0][idx_mean_frame])
        split_exptime[1].remove(split_exptime[1][idx_mean_frame])
        # Frames must be taken consecutively
        frame_id_list = []
        exptime_list = []
        unity_gain_filepath_list = []
        for subset in split_exptime[0]:
            for frame in subset:
                frame_id_list += [extract_frame_id(frame.filename)]
                exptime_list += [frame.ext_hdr['EXPTIME']]
                unity_gain_filepath_list += [frame.filepath]
        idx_id_sort = np.argsort(frame_id_list)
        exptime_list = np.array(exptime_list)[idx_id_sort]
        # Count repeated, consecutive elements
        count_cons = [1]
        exptime_cons = [exptime_list[0]]
        idx_cons = 0
        for exptime in exptime_list:
            if exptime == exptime_cons[-1]:
                count_cons[idx_cons] += 1
            else:
                idx_cons += 1
                count_cons += [1]
                exptime_cons += [exptime]
        # First index always has a repetition in the previous loop (id=id)
        count_cons[0] -= 1

        idx_cons2 = [0]
        exptime_cons2 = [exptime_list[idx_cons2[0]]]
        kgain_subset = []
        # Iterate over unique counts
        for idx_count in range(len(count_cons) - 1):
            if count_cons[idx_count+1] == count_cons[idx_count]:
                exptime_cons2 += [exptime_cons[idx_count+1]]
                idx_cons2 += [idx_count+1]
            else:
                # Last exposure time must be repeated and only once
                if (exptime_cons2[-1] in exptime_cons2[:-1] and
                    len(exptime_cons2) - 1 == len(set(exptime_cons2))):
                    kgain_subset += [idx_cons2[1] - 1, idx_cons2[-1]]
                    idx_cons2 = [idx_count+1]
                    exptime_cons2 = [exptime_list[idx_cons2[0]]]
                else:
                # It is not a subset for kgain
                    continue
        # Choose the largest subset
        kgain_subset = np.array(kgain_subset)
        idx_kgain = np.argmax(kgain_subset[1::2] - kgain_subset[0::2])
        # Extract first/last index in the subset of consecutive frames
        idx_kgain_0 = kgain_subset[2 * idx_kgain]
        idx_kgain_1 = kgain_subset[2 * idx_kgain + 1]
        # Count frames before and subset length
        idx_kgain_first = np.sum(count_cons[0:idx_kgain_0])
        idx_kgain_last = idx_kgain_first + np.sum(count_cons[idx_kgain_0:idx_kgain_1+1])

        # Sort unity gain filenames
        unity_gain_filepath_list = np.array(unity_gain_filepath_list)[idx_id_sort].tolist()
        cal_list = unity_gain_filepath_list[idx_kgain_first:idx_kgain_last]
        # Update OBSTYPE and take profit to check files are in the list
        n_kgain = 0
        cal_frame_list = []
        for frame in dataset_cp:
            if frame.filepath in cal_list:
                frame.pri_hdr['OBSTYPE'] = 'KGAIN'
                cal_frame_list += [frame]
                n_kgain += 1

        print(f'K-gain has {n_kgain} unity frames with exposure times', exptime_cons[idx_kgain_0:idx_kgain_1+1], f'seconds with each {count_cons[idx_kgain_0]} frames each')
        
    # Non-lin
    elif cal_type.lower() == 'non-lin' or cal_type.lower() == 'nonlin':
        print('Considering Non-linearity')
    # EM-gain: TODO
    elif cal_type.lower() == 'em-gain' or cal_type.lower() == 'emgain':
            print('Considering low EM-gain: TODO')
    else:
        raise Exception('Unrecognized calibration type (expected k-gain, non-lin or em-gain)')

    # Return Datafrane with mean frame and cal type
    return data.Dataset(mean_frame_list + cal_frame_list)
