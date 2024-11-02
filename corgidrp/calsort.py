# ulimit -n 500 to be able to open the 500 files

import corgidrp.data as data

def calsort(filename_list):
    """ TBW

    Args: TBW

    Returns:
    """
    # Create Dataset
    dataset_cal = data.Dataset(filename_list)
    for frame in dataset_cal:
        print(frame.filename, frame.ext_hdr['CMDGAIN'], frame.ext_hdr['EXPTIME'])

    # Return list of frames (mean frame and cal type)
    return



