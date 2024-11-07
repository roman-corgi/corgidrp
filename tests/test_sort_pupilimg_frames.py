import os
import random
import numpy as np
from pathlib import Path
import astropy.io.fits as fits

from corgidrp import sorting as calsort
import corgidrp.data as data
from corgidrp.data import Image
from corgidrp.mocks import create_default_headers


# Sub-functions
# NOTE: Most of the values for the different non-unity em gains and the number
# of frames come from either TVAC or some preliminary version of the 
# Commissioning test calculations

def get_cmdgain_exptime_mean_frame(
    exptime_sec=5,
    nframes=30,
    ):
    """
    Create an array of CMDGAIN, EXPTIME for frames that will be used to
    generate a mean frame.

    Args:
      exptime_sec (float): exposure time of the frame in seconds
      nframes (int): (minimum) number of frames to generate a mean frame
    
    Returns:
      cmdgain_list (list): list of commanded gains
      exptime_list (list): list of exposure frames
    """
    cmdgain_list = [1] * nframes
    exptime_list = [exptime_sec] * nframes
    return cmdgain_list, exptime_list

def get_cmdgain_exptime_kgain(
    exptime_sec=[0.077, 0.770, 1.538, 2.308, 3.077, 3.846, 4.615, 5.385, 6.154,
        6.923, 7.692, 8.462, 9.231, 10.000, 11.538, 10.769, 12.308, 13.077,
        13.846, 14.615, 15.385, 1.538],
    nframes=5,
    ):
    """
    Create an array of CMDGAIN, EXPTIME for frames that will be used to
    calibrate k-gain.

    Args:
      exptime_sec (list): set of distinct exposure times n seconds chosen to
      collect frames for K-gain calibration
      nframes (int): number of frames per ditinct exposure time

    Returns:
      cmdgain_list (list): list of commanded gains
      exptime_list (list): list of exposure frames
    """
    cmdgain_list = [1] * (len(exptime_sec) * nframes)
    exptime_list = []
    for exptime in exptime_sec:
        exptime_list += [exptime] * nframes
    return cmdgain_list, exptime_list

def get_cmdgain_exptime_nonlin(
    exptime_sec=[0.076, 0.758, 1.515, 2.273, 3.031, 3.789, 4.546, 5.304, 6.062,
        6.820, 7.577, 8.335, 9.093, 9.851, 10.608, 11.366, 12.124, 12.881,
        13.639, 14.397, 15.155, 1.515],
    nonunity_em=[1.65, 5.24, 8.60, 16.70, 27.50, 45.26, 87.50, 144.10, 237.26,
    458.70, 584.40],
    change_exptime=False,
    ):
    """
    Create an array of CMDGAIN, EXPTIME for frames that will be used to
    calibrate non-linearity.

    Args:
      exptime_sec (list): set of distinct exposure times in seconds chosen to
        collect frames for non-linearity calibration for each EM gain
      nonunity_em (list): set of ditinct (non-unity) EM gains chosen to collect
        data for non-linearity
      change_exptime (bool) (optional): if True, it will change the input exposure
        times by a small amount without changing the ordering of exptime_sec

    Returns:
      cmdgain_list (list): list of commanded gains
      exptime_list (list): list of exposure frames
    """
    cmdgain_list = []
    exptime_list = []
    fac_change = 0
    if change_exptime:
        # Avoid the (unlikely) coincidence of +1/-1 in the uniform distribution
        fac_change = min(np.abs(np.diff(exptime_sec))) / 3
    for emgain in nonunity_em:
        cmdgain_list += [emgain] * len(exptime_sec)
        exptime_sec = (np.array(exptime_sec) *
            (1 + fac_change*random.uniform(-1, 1)))
        exptime_list += exptime_sec.tolist()
    return cmdgain_list, exptime_list

def get_cmdgain_exptime_emgain(
    em_emgain=[1.000, 1.000, 1.007, 1.015, 1.024, 1.035, 1.047, 1.060, 1.076, 1.094,
        1.115, 1.138, 1.165, 1.197, 1.234, 1.276, 1.325, 1.385, 1.453, 1.534, 1.633,
        1.749, 1.890, 2.066, 2.278, 2.541, 2.873, 3.308, 3.858, 4.581, 5.577, 6.189,
        6.906, 7.753, 8.757, 9.955, 11.392, 13.222, 15.351, 17.953, 21.157, 25.128,
        30.082, 36.305, 44.621, 54.768, 67.779, 84.572, 106.378, 134.858, 172.244,
        224.385, 290.538, 378.283, 494.762, 649.232, 853.428],
    exptime_emgain_sec=[5, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    nframes=[3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        ):
    """
    Create an array of CMDGAIN, EXPTIME for frames that will be used to
    calibrate EM-gain vs DAC.

     Args:
      em_gain (list): set of ditinct (non-unity) EM gains chosen to collect
        data for EM gain with pupil images
      exptime_emgain_sec (list): set of distinct exposure times in seconds chosen to
        collect frames for non-linearity calibration for each EM gain
      nframes (int): number of frames per ditinct exposure time

    Returns:
      cmdgain_list (list): list of commanded gains
      exptime_list (list): list of exposure frames
    """
    # Create pairs of frames
    nonunity_gain_list = []
    for idx in range(len(em_emgain)):
        nonunity_gain_list += [em_emgain[idx]] * nframes[idx] 
    cmdgain_list = [1] * np.sum(nframes) + nonunity_gain_list
    exptime_list = []
    for idx in range(len(exptime_emgain_sec)):
        exptime_list += [exptime_emgain_sec[idx]] * nframes[idx]
    # Unity and non-unity gains
    exptime_list += exptime_list
    return cmdgain_list, exptime_list

def make_minimal_image(
    cmdgain=1,
    exptime_sec=0,
    frameid=0,
        ):
    """
    This function makes a mock frame with minimum memory in its data and error
    fields. It is used in this test script only.

    Args:
      cmdgain (float): commanded gain of the frame
      exptime_sec (float): exposure time of the frame
      frameid (int): an integer value used to indentify the frame

    Returns:
      filename (String): filename with path of the generated FITS file
    """
    signal = np.zeros(1)

    prhd, exthd = create_default_headers()
    # Mock error maps
    err = np.ones(1)
    dq = np.zeros(1, dtype = np.uint16)
    # Creating a FITS file to assign it a filename with the frame ID
    prim = fits.PrimaryHDU(header = prhd)
    hdr_img = fits.ImageHDU(signal, header=exthd)
    hdul = fits.HDUList([prim, hdr_img])
    # Record actual commanded EM
    hdul[1].header['CMDGAIN'] = cmdgain
    # Record actual exposure time
    hdul[1].header['EXPTIME'] = exptime_sec
    filename = str(Path('simdata', f'CGI_EXCAM_L1_{frameid:0{10}d}.fits'))
    hdul.writeto(filename, overwrite = True)
    return filename

# Main code

# Mean frame
EXPTIME_MEAN_FRAME = 5
NFRAMES_MEAN_FRAME = 30
# Checks
if NFRAMES_MEAN_FRAME < 30:
    raise Exception(f'Insufficient frames ({NFRAMES_MEAN_FRAME}) for the mean frame')
# Values
cmdgain_mean_frame, exptime_mean_frame = get_cmdgain_exptime_mean_frame(
    exptime_sec=EXPTIME_MEAN_FRAME,
    nframes=NFRAMES_MEAN_FRAME,
    )

if len(cmdgain_mean_frame) != len(exptime_mean_frame):
    raise Exception('Inconsistent lengths in the mean frame')
# Total number of frames
n_mean_frame_total = len(cmdgain_mean_frame)

# K-gain
EXPTIME_KGAIN = [0.077, 0.770, 1.538, 2.308, 3.077, 3.846, 4.615, 5.385, 6.154,
    6.923, 7.692, 8.462, 9.231, 10.000, 11.538, 10.769, 12.308, 13.077, 13.846,
    14.615, 15.385, 1.538]
NFRAMES_KGAIN = 5
# Checks
if NFRAMES_KGAIN < 5:
    raise Exception(f'Insufficient frames ({NFRAMES_KGAIN}) per unique exposure time in k-gain')
if len(EXPTIME_KGAIN) < 22:
    raise Exception(f'Insufficient unique exposure times ({len(EXPTIME_KGAIN)}) in k-gain')
# Values
cmdgain_kgain, exptime_kgain = get_cmdgain_exptime_kgain(
    exptime_sec=EXPTIME_KGAIN,
    nframes=NFRAMES_KGAIN,
    )
if len(cmdgain_kgain) != len(exptime_kgain):
    raise Exception('Inconsistent lengths in k-gain')
# Total number of frames
n_kgain_total = len(cmdgain_kgain)

# Non-linearity
EXPTIME_NONLIN = [0.076, 0.758, 1.515, 2.273, 3.031, 3.789, 4.546, 5.304, 6.062,
    6.820, 7.577, 8.335, 9.093, 9.851, 10.608, 11.366, 12.124, 12.881, 13.639,
    14.397, 15.155, 1.515]
CMDGAIN_NONLIN = [1.65, 5.24, 8.60, 16.70, 27.50, 45.26, 87.50, 144.10, 237.26,
    458.70, 584.40]
# Checks
if len(EXPTIME_NONLIN) < 22:
    raise Exception(f'Insufficient frames ({len(EXPTIME_NONLIN)}) per unique EM value in non-linearity')
if len(CMDGAIN_NONLIN) < 11:
    raise Exception(f'Insufficient values of distinct EM Values ({len(EXPTIME_NONLIN)}) in non-linearity')
if np.sum(np.array(EXPTIME_NONLIN) == EXPTIME_NONLIN[-1]) != 2:
    raise Exception('Last exposure time must be repeated once')
if len(set(EXPTIME_NONLIN)) != len(EXPTIME_NONLIN) - 1:
    raise Exception('Only one exposure time can be repeated in non-linearity')
if EXPTIME_NONLIN[-1] in EXPTIME_NONLIN[0:5] is False:
    raise Exception('The last exposure time must be present at the beginning of the exposure times in non-linearity')

# Values
# w/o changing exposure times among non-unity EM gains
cmdgain_nonlin_wo_change, exptime_nonlin_wo_change = get_cmdgain_exptime_nonlin(
    exptime_sec=EXPTIME_NONLIN,
    nonunity_em=CMDGAIN_NONLIN,
    change_exptime=False,
    )
if len(cmdgain_nonlin_wo_change) != len(exptime_nonlin_wo_change):
    raise Exception('Inconsistent lengths in non-linearity')
# Total number of frames
n_nonlin_wo_changes_total = len(cmdgain_nonlin_wo_change)

# changing exposure times among non-unity EM gains
cmdgain_nonlin_w_change, exptime_nonlin_w_change = get_cmdgain_exptime_nonlin(
    exptime_sec=EXPTIME_NONLIN,
    nonunity_em=CMDGAIN_NONLIN,
    change_exptime=True,
    )
if len(cmdgain_nonlin_w_change) != len(exptime_nonlin_w_change):
    raise Exception('Inconsistent lengths in non-linearity')
# Total number of frames
n_nonlin_w_changes_total = len(cmdgain_nonlin_w_change)

# EM-gain vs DAC (The amount of data is illustrative. It was taken from a draft
# of the Commissioning Activity Report)
# Actual values of the non-unity em gains for low em-gain
EM_EMGAIN = [1.000, 1.007, 1.015, 1.024, 1.035, 1.047, 1.060, 1.076, 1.094,
    1.115, 1.138, 1.165, 1.197, 1.234, 1.276, 1.325, 1.385, 1.453, 1.534, 1.633,
    1.749, 1.890, 2.066, 2.278, 2.541, 2.873, 3.308, 3.858, 4.581, 5.577, 6.189,
    6.906, 7.753, 8.757, 9.955, 11.392, 13.222, 15.351, 17.953, 21.157, 25.128,
    30.082, 36.305, 44.621, 54.768, 67.779, 84.572, 106.378, 134.858, 172.244,
    224.385, 290.538, 378.283, 494.762, 649.232, 853.428]
EXPTIME_EMGAIN_SEC = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11]
NFRAMES_EMGAIN = [3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
# Checks
if len(EM_EMGAIN) < 56:
    raise Exception(f'Insufficient number of EM gain values ({len(EM_EMGAIN)}) in EM-gain vs DAC')
if len(EXPTIME_EMGAIN_SEC) != len(EM_EMGAIN):
    raise Exception(f'Inconsistent number of sets in EM-gain vs DAC')
if len(EXPTIME_EMGAIN_SEC) != len(NFRAMES_EMGAIN):
    raise Exception(f'Inconsistent number of sets in EM-gain vs DAC')
# Values
cmdgain_emgain, exptime_emgain = get_cmdgain_exptime_emgain(
    em_emgain = EM_EMGAIN,
    exptime_emgain_sec = EXPTIME_EMGAIN_SEC,
    nframes = NFRAMES_EMGAIN,
    )
if len(cmdgain_emgain) != len(exptime_emgain):
    raise Exception(f'Inconsistent lengths in em-gain vs dac')
# Total number of frames
n_emgain_total = len(cmdgain_emgain)

# DRP Dataset
# Create directory for temporary data files (not tracked by git)
if not os.path.exists(Path('simdata')):
    os.mkdir(Path('simdata'))

# Loop over the two cases of non-linearity
change_exptime = [False, True]
idx_frame = 0
for change in change_exptime:
    filename_list = []
    if change == False:
        cmdgain_nonlin = cmdgain_nonlin_wo_change
        exptime_nonlin = exptime_nonlin_wo_change
    elif change == True:
        cmdgain_nonlin = cmdgain_nonlin_w_change
        exptime_nonlin = exptime_nonlin_w_change
    else:
        raise Exception('Undefined choice for non-linearity')
    n_nonlin_total = len(exptime_nonlin)
    # Mean frame
    print('Generating frames for mean frame')
    for i_f in range(n_mean_frame_total):
        filename = make_minimal_image(
            cmdgain=cmdgain_mean_frame[i_f],
            exptime_sec=exptime_mean_frame[i_f],
            frameid=idx_frame,
            )
        filename_list += [filename]
        idx_frame += 1
    # K-gain
    print('Generating frames for k-gain')
    for i_f in range(n_kgain_total):
        filename = make_minimal_image(
            cmdgain=cmdgain_kgain[i_f],
            exptime_sec=exptime_kgain[i_f],
            frameid=idx_frame,
            )
        filename_list += [filename]
        idx_frame += 1
    # Non-linearity
    print('Generating frames for non-linearity')
    for i_f in range(n_nonlin_total):
        filename = make_minimal_image(
            cmdgain=cmdgain_nonlin[i_f],
            exptime_sec=exptime_nonlin[i_f],
            frameid=idx_frame,
            )
        filename_list += [filename]
        idx_frame += 1
    # EM-gain
    print('Generating frames for em-gain')
    for i_f in range( n_emgain_total):
        filename = make_minimal_image(
            cmdgain=cmdgain_emgain[i_f],
            exptime_sec=exptime_emgain[i_f],
            frameid=idx_frame,
            )
        filename_list += [filename]
        idx_frame += 1

    # Shuffle file order randomnly
    random.shuffle(filename_list)

    # Create Dataset
    dataset_cal = data.Dataset(filename_list)

    # Apply sorting algorithm and check results (maybe output of sorting is
    # mean frame and the type used as input? Instead of them all. Then, check
    # those properties
    dataset_kgain = calsort.sort_pupilimg_frames(dataset_cal, cal_type='k-gain')

    dataset_nonlin = calsort.sort_pupilimg_frames(dataset_cal, cal_type='non-lin')

    # Erase FITS files
    for file in filename_list:
        os.remove(file)

