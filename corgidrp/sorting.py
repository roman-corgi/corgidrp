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
    # Whole list of frames
    filepath_list = []
    for frame in dataset_cp:
        filepath_list += [frame.filepath]

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
        # Update keyword OBSTYPE
        frame.pri_hdr['OBSTYPE'] = 'MNFRAME'
        frame_id_list += [extract_frame_id(frame.filename)]
    # Choose the frames with consecutive ID numbers (frames are taken with the same instruction)
    frame_id_sort = np.array(frame_id_list).astype(int)
    frame_id_sort.sort()
    frame_id_sort = \
        frame_id_sort[np.where(np.diff(frame_id_sort) == 1)]
    mean_frame_list = []
    for frame in split_exptime[0][idx_mean_frame]:
        if int(extract_frame_id(frame.filename)) in frame_id_sort:
            mean_frame_list += [frame.filepath]
    
    # K-gain
    cal_frame_list = []
    if cal_type.lower() == 'k-gain' or cal_type.lower() == 'kgain':
        print('Considering K-gain')
        # Choice: choose those subsets of frames with the mode of n_frames_list
        mode_n_frames = max(set(n_frames_list), key=n_frames_list.tolist().count)
        # Get all sets with at least this number of frames, skipping the one chosen
        # as mean frame
        idx_kgain = np.where(n_frames_list >= mode_n_frames)[0].tolist()
        if len(mean_frame_list) >= mode_n_frames:
            idx_kgain.remove(idx_mean_frame)
        
        breakpoint()
    # Non-lin
    elif cal_type.lower() == 'non-lin' or cal_type.lower() == 'nonlin':
        print('Considering Non-linearity')
    # EM-gain: TODO
    elif cal_type.lower() == 'em-gain' or cal_type.lower() == 'emgain':
            print('Considering low EM-gain')
    else:
        raise Exception('Unrecognized calibration type (expected k-gain, non-lin or em-gain)')

    breakpoint()
    # Return Datafrane with mean frame and cal type
    return data.Dataset(mean_frame_list + cal_frame_list)



