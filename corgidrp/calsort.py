# ulimit -n 500 to be able to open the 500 files

import numpy as np

import corgidrp.data as data

def calsort(
    filename_list,
    cal_type=None):
    """ TBW

    Args: TBW

    Returns:
    """
    # Create Dataset
    dataset_cal = data.Dataset(filename_list)
    # Split by CMDGAIN
    split_cmdgain = dataset_cal.split_dataset(exthdr_keywords=['CMDGAIN'])
    # Mean frame: split by EXPTIME
    idx_unity = np.where(np.array(split_cmdgain[1])==1)[0][0]
    split_exptime = split_cmdgain[0][idx_unity].split_dataset(exthdr_keywords=['EXPTIME'])
    # Get the set with the most frames
    n_frames_list = np.zeros(len(split_exptime[0]))
    for i_sub, subset in enumerate(split_exptime[0]):
        n_frames_list[i_sub] = len(split_exptime[0][i_sub])
    idx_mean_frame = np.argmax(n_frames_list)
    mean_frame_list = []
    for frame in split_exptime[0][idx_mean_frame]:
        mean_frame_list += [frame.filename]
    # Sort by frameID: ascending order
    mean_frame_list.sort()

    # K-gain
    #TODO
    cal_frame_list = []
    if cal_type.lower() == 'k-gain' or cal_type.lower() == 'kgain':
        print('Considering K-gain')
    elif cal_type.lower() == 'non-lin' or cal_type.lower() == 'nonlin':
        print('Considering Non-linearity')
    elif cal_type.lower() == 'em-gain' or cal_type.lower() == 'emgain':
            print('Considering low EM-gain')
    else:
        raise Exception('Unrecognized calibration type (expected k-gain, non-lin or em-gain)')

#    breakpoint()
    # Return list of frames (mean frame and cal type)
    return mean_frame_list, cal_frame_list



