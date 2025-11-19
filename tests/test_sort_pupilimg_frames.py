import os
import copy
import pytest
import random
import numpy as np
from pathlib import Path
import astropy.io.fits as fits
import time
from datetime import datetime

from corgidrp import sorting as sorting
import corgidrp.data as data
from corgidrp.data import Image
from corgidrp.mocks import create_default_L1_headers
from datetime import timedelta
from collections import defaultdict

np.random.seed(42)  # For reproducibility
# Functions
def get_cmdgain_exptime_mean_frame(
    exptime_sec=None,
    nframes=None,
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
    if exptime_sec is None:
        raise Exception('Missing input exposure times for mean frame data')
    if nframes is None:
        raise Exception('Missing number of frames for mean frame data')

    cmdgain_list = [1] * nframes
    exptime_list = [exptime_sec] * nframes
    return cmdgain_list, exptime_list

def get_cmdgain_exptime_kgain(
    exptime_sec=None,
    nframes=None,
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
    if exptime_sec is None:
        raise Exception('Missing input exposure times for K=gain calibration data')
    if nframes is None:
        raise Exception('Missing input number of frames for K=gain calibration data')

    cmdgain_list = [1] * (len(exptime_sec) * nframes)
    exptime_list = []
    for exptime in exptime_sec:
        exptime_list += [exptime] * nframes
    return cmdgain_list, exptime_list

def get_cmdgain_exptime_nonlin(
    exptime_sec=None,
    nonunity_em=None,
    change_exptime=False,
    ):
    """
    Create an array of CMDGAIN, EXPTIME for frames that will be used to
    calibrate non-linearity.

    Args:
      exptime_sec (list): set of distinct exposure times in seconds chosen to
        collect frames for non-linearity calibration for each EM gain
      nonunity_em (list): set of distinct (non-unity) EM gains chosen to collect
        data for non-linearity
      change_exptime (bool) (optional): if True, it will change the input exposure
        times by a small amount without changing the ordering of exptime_sec

    Returns:
      cmdgain_list (list): list of commanded gains
      exptime_list (list): list of exposure frames
    """
    if exptime_sec is None:
        raise Exception('Missing input exposure times for non-linearity calibration data')
    if nonunity_em is None:
        raise Exception('Missing input EM gain for non-linearity calibration data')

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
    em_emgain=None,
    exptime_emgain_sec=None,
    nframes=None,
    ):
    """
    Create an array of CMDGAIN, EXPTIME for frames that will be used to
    calibrate EM-gain vs DAC. Notice the pairing between unity and non-unity
    gain frames.

     Args:
      em_emgain (list): set of ditinct (non-unity) EM gains chosen to collect
        data for EM gain with pupil images
      exptime_emgain_sec (list): set of distinct exposure times in seconds chosen to
        collect frames for non-linearity calibration for each EM gain
      nframes (int): number of frames per ditinct exposure time

    Returns:
      cmdgain_list (list): list of commanded gains
      exptime_list (list): list of exposure frames
    """
    if em_emgain is None:
        raise Exception('Missing input EM gain for EM-gain calibration data')
    if exptime_emgain_sec is None:
        raise Exception('Missing input exposure times for EM-gain calibration data')
    if nframes is None:
        raise Exception('Missing input number of frames for EM-gain calibration data')

    # Create pairs of frames
    cmdgain_list = []
    for idx in range(len(em_emgain)):
        cmdgain_list += [em_emgain[idx]] * nframes[idx]
    exptime_list = []
    for idx in range(len(exptime_emgain_sec)):
        exptime_list += [exptime_emgain_sec[idx]] * nframes[idx]
    # Unity and non-unity gains
    return cmdgain_list, exptime_list

def make_minimal_image(
    cmdgain=1,
    exptime_sec=0,
    frameid=0,
    previous_exptime=0,
    previous_timestamp=None,
    auxfile=''
        ):
    """
    This function makes a mock frame with minimum memory in its data and error
    fields. It is used in this test script only.

    Args:
      cmdgain (float): commanded gain of the frame
      exptime_sec (float): exposure time of the frame
      frameid (int): an integer value used to indentify the frame
      previous_exptime (float): exposure time of the previous frame, used to make the time stamp for the frame accurate
      previous_timestamp (str): timestamp of the previous frame, used to make the time stamp for the current frame
      auxfile (str): auxiliary file name to be used in the header of the FITS file.  Defaults to an empty string.
    
    Returns:
      filepath (String): filepath of the generated FITS file
    """
    signal = np.zeros(1)
    prhd, exthd = create_default_L1_headers() # makes timestamp according to current time
    if previous_timestamp is None:
        previous_timestamp = exthd['DATETIME'][:26]
    if len(previous_timestamp) == 19:
        dt_obj = datetime.strptime(previous_timestamp, "%Y-%m-%dT%H:%M:%S")
    else:        
        dt_obj = datetime.strptime(previous_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
    # Add previous_exptime (in seconds) to dt_obj to get updated_time, along with a little offset time (time between frames)
    updated_time = dt_obj + timedelta(seconds=float(previous_exptime)+0.2)
    # Round to nearest tenth of a second
    microsec = int(round(updated_time.microsecond / 1e5) * 1e5)
    if microsec == 1000000:
        updated_time = updated_time.replace(microsecond=0) + timedelta(seconds=1)
    else:
        updated_time = updated_time.replace(microsecond=microsec)
    # Convert output back to ISO format string
    exthd['DATETIME'] = updated_time.isoformat(timespec='microseconds')
    # Creating a FITS file to assign it a filename with the frame ID
    prim = fits.PrimaryHDU(header = prhd)
    hdr_img = fits.ImageHDU(signal, header=exthd)
    hdul = fits.HDUList([prim, hdr_img])
    # Record actual commanded EM
    hdul[1].header['EMGAIN_C'] = cmdgain
    # Record actual exposure time
    hdul[1].header['EXPTIME'] = exptime_sec
    # Add corresponding VISTYPE
    hdul[0].header['VISTYPE'] = 'CGIVST_CAL_PUPIL_IMAGING'
    hdul[0].header['AUXFILE'] = auxfile
    hdul[1].header['DPAMNAME'] = 'PUPIL,PUPIL_FFT' #from latest update of TVAC files from SSC
    hdul[1].header['CFAMNAME'] = 'CLEAR' # would have actual filter names, but for now, just shouldn't be 'DARK'
    year=exthd['DATETIME'][:4]
    month=exthd['DATETIME'][5:7]
    day=exthd['DATETIME'][8:10]
    hour= exthd['DATETIME'][11:13]
    minute= exthd['DATETIME'][14:16]
    seconds= exthd['DATETIME'][17:19]
    tenth = int(updated_time.microsecond/1E5)
    filename = 'cgi_0200001001001001001_{0}{1}{2}t{3}{4}{5}{6}_l1_.fits'.format(year, month, day, hour, minute, seconds, tenth)
    filepath = os.path.join(os.path.dirname(__file__), 'simdata', filename)
    hdul.writeto(filepath, overwrite = True)
    return filepath


def setup_module():
    global EXPTIME_MEAN_FRAME, NFRAMES_MEAN_FRAME
    global EXPTIME_KGAIN,  NFRAMES_KGAIN
    global EXPTIME_NONLIN, CMDGAIN_NONLIN
    global EXPTIME_EMGAIN_SEC, EXPTIME_EMGAIN_SEC
    global n_mean_frame_total, n_kgain_total, n_nonlin_wo_change_total, n_nonlin_w_change_total
    global dataset_w_change, dataset_wo_change 
    # Note: the values for the non-unity em gains and the number
    # of frames used for the mean frame, K-gain, non-linearity and EM-gain vs DAC
    # calibration come from either TVAC or some preliminary version of the
    # Commissioning test calculations
    
    # Global constants
    # Mean frame
    EXPTIME_MEAN_FRAME = 5
    NFRAMES_MEAN_FRAME = 30
    # Checks
    if NFRAMES_MEAN_FRAME < 30:
        raise Exception(f'Insufficient frames ({NFRAMES_MEAN_FRAME}) for the mean frame')
    
    # K-gain
    EXPTIME_KGAIN = [0.077, 0.770, 1.538, 2.308, 3.077, 3.846, 4.615, 5.385, 6.154,
        6.923, 7.692, 8.462, 9.231, 10.000, 10.769, 11.538, 12.308, 13.077, 13.846,
        14.615, 15.385, 1.538]
    NFRAMES_KGAIN = 5
    # Checks
    if NFRAMES_KGAIN < 5:
        raise Exception(f'Insufficient frames ({NFRAMES_KGAIN}) per unique exposure time in k-gain')
    if len(EXPTIME_KGAIN) < 22:
        raise Exception(f'Insufficient unique exposure times ({len(EXPTIME_KGAIN)}) in k-gain')
    if np.all(np.sign(np.diff(EXPTIME_KGAIN[:-1])) == 1) is False:
        raise Exception('Exposure times in K-gain must be monotonic but for the last value')
    
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
    if np.all(np.sign(np.diff(EXPTIME_NONLIN[:-1])) == 1) is False:
        raise Exception('Exposure times in Non-linearity must be monotonic but for the last value')
    
    # Notice the pairing between unity and non-unity gain frames
    EM_EMGAIN=[1.000, 1.000, 1.007, 1.015, 1.024, 1.035, 1.047, 1.060, 1.076, 1.094,
        1.115, 1.138, 1.165, 1.197, 1.234, 1.276, 1.325, 1.385, 1.453, 1.534, 1.633,
        1.749, 1.890, 2.066, 2.278, 2.541, 2.873, 3.308, 3.858, 4.581, 5.577, 6.189,
        6.906, 7.753, 8.757, 9.955, 11.392, 13.222, 15.351, 17.953, 21.157, 25.128,
        30.082, 36.305, 44.621, 54.768, 67.779, 84.572, 106.378, 134.858, 172.244,
        224.385, 290.538, 378.283, 494.762, 649.232, 853.428]
    EXPTIME_EMGAIN_SEC=[5, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    NFRAMES_EMGAIN=[3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    # Checks
    if len(EM_EMGAIN) < 56:
        raise Exception(f'Insufficient number of EM gain values ({len(EM_EMGAIN)}) in EM-gain vs DAC')
    if len(EXPTIME_EMGAIN_SEC) != len(EM_EMGAIN):
        raise Exception(f'Inconsistent number of sets in EM-gain vs DAC')
    if len(EXPTIME_EMGAIN_SEC) != len(NFRAMES_EMGAIN):
        raise Exception(f'Inconsistent number of sets in EM-gain vs DAC')
    
    # Test data: Consider two possible scenarios: identical exposure times among the
    # different subsets of non-unity gain used to calibrate non-linearity or different
    # values of exposure times
    
    # Values for mean frame
    cmdgain_mean_frame, exptime_mean_frame = get_cmdgain_exptime_mean_frame(
        exptime_sec=EXPTIME_MEAN_FRAME,
        nframes=NFRAMES_MEAN_FRAME,
        )
    
    if len(cmdgain_mean_frame) != len(exptime_mean_frame):
        raise Exception('Inconsistent lengths in the mean frame')
    # Total number of frames
    n_mean_frame_total = len(cmdgain_mean_frame)
    
    # Values for K-gain
    cmdgain_kgain, exptime_kgain = get_cmdgain_exptime_kgain(
        exptime_sec=EXPTIME_KGAIN,
        nframes=NFRAMES_KGAIN,
        )
    if len(cmdgain_kgain) != len(exptime_kgain):
        raise Exception('Inconsistent lengths in k-gain')
    # Total number of frames
    n_kgain_total = len(cmdgain_kgain)
    
    # Values for Non-linearity
    cmdgain_nonlin_wo_change, exptime_nonlin_wo_change = get_cmdgain_exptime_nonlin(
        exptime_sec=EXPTIME_NONLIN,
        nonunity_em=CMDGAIN_NONLIN,
        change_exptime=False,
        )
    if len(cmdgain_nonlin_wo_change) != len(exptime_nonlin_wo_change):
        raise Exception('Inconsistent lengths in non-linearity')
    # Total number of frames
    n_nonlin_wo_change_total = len(cmdgain_nonlin_wo_change)
    
    cmdgain_nonlin_w_change, exptime_nonlin_w_change = get_cmdgain_exptime_nonlin(
        exptime_sec=EXPTIME_NONLIN,
        nonunity_em=CMDGAIN_NONLIN,
        change_exptime=True,
        )
    if len(cmdgain_nonlin_w_change) != len(exptime_nonlin_w_change):
        raise Exception('Inconsistent lengths in non-linearity')
    # Total number of frames
    n_nonlin_w_change_total = len(cmdgain_nonlin_w_change)
    
    # Values for EM-gain vs DAC
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
    datadir = os.path.join(os.path.dirname(__file__), 'simdata')
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    # clean out any previous files in there; files are not overwritten since their names depend on the present time
    for filename in os.listdir(datadir):
        if not filename.endswith('.fits'):
            continue
        filepath = os.path.join(datadir, filename)
        os.remove(filepath)
    idx_frame = 0
    filename_list = []
    exptimes = []
    timestamps = []
    # Mean frame
    print('Generating frames for mean frame')
    previous_timestamp = None # initialize
    for i_f in range(n_mean_frame_total):
        filename = make_minimal_image(
            cmdgain=cmdgain_mean_frame[i_f],
            exptime_sec=exptime_mean_frame[i_f],
            frameid=idx_frame,
            previous_exptime=0 if i_f == 0 else exptime_mean_frame[i_f-1],
            previous_timestamp=previous_timestamp
            )
        previous_timestamp = fits.getheader(filename, 1)['DATETIME']
        timestamps += [previous_timestamp]
        exptimes += [fits.getheader(filename, 1)['EXPTIME']]
        filename_list += [filename]
        idx_frame += 1
    # K-gain
    print('Generating frames for k-gain')
    for i_f in range(n_kgain_total):
        filename = make_minimal_image(
            cmdgain=cmdgain_kgain[i_f],
            exptime_sec=exptime_kgain[i_f],
            frameid=idx_frame,
            previous_exptime=exptime_mean_frame[-1] if i_f == 0 else exptime_kgain[i_f-1],
            previous_timestamp=previous_timestamp
            )
        previous_timestamp = fits.getheader(filename, 1)['DATETIME']
        timestamps += [previous_timestamp]
        exptimes += [fits.getheader(filename, 1)['EXPTIME']]
        filename_list += [filename]
        idx_frame += 1
    # EM-gain
    print('Generating frames for em-gain')
    for i_f in range(n_emgain_total):
        filename = make_minimal_image(
            cmdgain=cmdgain_emgain[i_f],
            exptime_sec=exptime_emgain[i_f],
            frameid=idx_frame,
            previous_exptime=exptime_kgain[-1] if i_f == 0 else exptime_emgain[i_f-1],
            previous_timestamp=previous_timestamp
            )
        previous_timestamp = fits.getheader(filename, 1)['DATETIME']
        timestamps += [previous_timestamp]
        exptimes += [fits.getheader(filename, 1)['EXPTIME']]
        filename_list += [filename]
        idx_frame += 1
    # EM-gain (different temperature)
    print('Generating frames for em-gain at different temperature')
    for i_f in range(n_emgain_total):
        # making AUXFILE different; testing AUXFILE filter in sorting.py with this
        filename = make_minimal_image(
            cmdgain=cmdgain_emgain[i_f],
            exptime_sec=exptime_emgain[i_f]+ 11, #makes it big enough to keep an increasing order of exposure times in sorting.py
            frameid=idx_frame,
            previous_exptime=exptime_kgain[-1] if i_f == 0 else exptime_emgain[i_f-1],
            previous_timestamp=previous_timestamp,
            auxfile='TEMPERATURE'
            )
        previous_timestamp = fits.getheader(filename, 1)['DATETIME']
        timestamps += [previous_timestamp]
        exptimes += [fits.getheader(filename, 1)['EXPTIME']]
        filename_list += [filename]
        idx_frame += 1
    # Non-linearity (two cases) 
    print('Generating frames for non-linearity')
    filename_wo_change_list = copy.deepcopy(filename_list)
    for i_f in range(n_nonlin_wo_change_total):
        filename = make_minimal_image(
            cmdgain=cmdgain_nonlin_wo_change[i_f],
            exptime_sec=exptime_nonlin_wo_change[i_f],
            frameid=idx_frame,
            previous_exptime=exptime_emgain[-1] if i_f == 0 else exptime_nonlin_wo_change[i_f-1],
            previous_timestamp=previous_timestamp
            )
        previous_timestamp = fits.getheader(filename, 1)['DATETIME']
        timestamps += [previous_timestamp]
        exptimes += [fits.getheader(filename, 1)['EXPTIME']]
        filename_wo_change_list += [filename]
        idx_frame += 1
    
    filename_w_change_list = copy.deepcopy(filename_list)
    for i_f in range(n_nonlin_w_change_total):
        filename = make_minimal_image(
            cmdgain=cmdgain_nonlin_w_change[i_f],
            exptime_sec=exptime_nonlin_w_change[i_f],
            frameid=idx_frame,
            previous_exptime=exptime_emgain[-1] if i_f == 0 else exptime_nonlin_w_change[i_f-1],
            previous_timestamp=previous_timestamp
            )
        previous_timestamp = fits.getheader(filename, 1)['DATETIME']
        timestamps += [previous_timestamp]
        exptimes += [fits.getheader(filename, 1)['EXPTIME']]
        filename_w_change_list += [filename]
        idx_frame += 1

    # Shuffle file order randomly
    random.shuffle(filename_wo_change_list)
    random.shuffle(filename_w_change_list)
    
    # Create datasets
    dataset_wo_change = data.Dataset(filename_wo_change_list)
    dataset_w_change = data.Dataset(filename_w_change_list)

def test_kgain_sorting():
    """
    Apply the sorting algorithm to a dataset for K-gain and non-linearity
    calibration including EM-gain calibration files in the set to obtain
    the dataset needed for K-gain calibration and check the resulting
    dataset is consistent with the input dataset. K-gain uses unity gain
    frames only. No need to test both non-linearity subsets of data.
    """
    dataset_kgain = sorting.sort_pupilimg_frames(dataset_wo_change, cal_type='k-gain', actual_visit=True)

    # Checks
    n_mean_frame = 0
    n_kgain_test = 0
    filename_kgain_list = []
    exptime_mean_frame_list = []
    exptime_kgain_list = []
    # This way there's no need to perform a sum check and identifies any issue
    for idx_frame, frame in enumerate(dataset_kgain):
        if frame.pri_hdr['OBSNAME'] == 'MNFRAME':
            n_mean_frame += 1
            exptime_mean_frame_list += [frame.ext_hdr['EXPTIME']]
        elif frame.pri_hdr['OBSNAME'] == 'KGAIN':
            n_kgain_test += 1
            filename_kgain_list += [frame.filename]
            exptime_kgain_list += [frame.ext_hdr['EXPTIME']]
        else:
            try:
                raise Exception((f'Frame #{idx_frame}: Misidentified calibration' +
                   f"type in the calibration dataset. OBSNAME={frame.pri_hdr['OBSNAME']}"))
            except:
                raise Exception((f'Frame #{idx_frame}: Unidentified calibration',
                    'type in the Kgain calibration dataset'))
    
    # Same number of files as expected (sort_pupilimg_frames deletes the repeated set since no drift correction is done for k gain cal and so that that exposure time isn't doubly weighted compared to the others)
    assert n_kgain_test == n_kgain_total - NFRAMES_KGAIN
    # Unique exposure time for the mean frame
    assert len(set(exptime_mean_frame_list)) == 1
    # Expected exposure time for the mean frame
    assert exptime_mean_frame_list[0] == EXPTIME_MEAN_FRAME
    # Expected number of frames for the mean frame
    assert n_mean_frame == NFRAMES_MEAN_FRAME
    # Expected identical number of frames per exposure time in K-gain
    kgain_unique, kgain_counts = np.unique(exptime_kgain_list, return_counts=True)
    assert len(set(kgain_counts)) == 1
    assert (kgain_counts == NFRAMES_KGAIN).all()
    # Needs ordering
    idx_kgain_sort = np.argsort(filename_kgain_list)
    # Expected exposure times for K-gain
    exptime_kgain_arr = np.array(exptime_kgain_list)[idx_kgain_sort]
    assert len(set(exptime_kgain_arr[-NFRAMES_KGAIN:])) == 1
    _, vals = dataset_kgain.split_dataset(prihdr_keywords=['AUXFILE'])
    assert len(vals) == 1

def test_nonlin_sorting_wo_change():
    """
    Apply the sorting algorithm to a dataset for K-gain and non-linearity
    calibration including EM-gain calibration files in the set to obtain
    the dataset needed for non-linearity calibration and check the
    resulting dataset is consistent with the input dataset. This test has
    identical exposure times among the different subsets of non-unity gain
    used to calibrate non-linearity
    """        
    dataset_nonlin_wo_change = sorting.sort_pupilimg_frames(dataset_wo_change, cal_type='non-lin',actual_visit=True)

    # Checks
    n_mean_frame = 0
    n_nonlin_test = 0
    filename_nonlin_list = []
    exptime_mean_frame_list = []
    exptime_nonlin_list = []
    cmdgain_nonlin_list = []
    # This way there's no need to perform a sum check and identifies any issue
    for idx_frame, frame in enumerate(dataset_nonlin_wo_change):
        if frame.pri_hdr['OBSNAME'] == 'MNFRAME':
            n_mean_frame += 1
            exptime_mean_frame_list += [frame.ext_hdr['EXPTIME']]
        elif frame.pri_hdr['OBSNAME'] == 'NONLIN':
            n_nonlin_test += 1
            filename_nonlin_list += [frame.filename]
            exptime_nonlin_list += [frame.ext_hdr['EXPTIME']]
            cmdgain_nonlin_list += [frame.ext_hdr['EMGAIN_C']]
        # Testing only non-unity gain frames for Non-linearity
        elif frame.pri_hdr['OBSNAME'] == 'KGAIN':
            pass
        else:
            try:
                raise Exception((f'Frame #{idx_frame}: Misidentified calibration' +
                   f"type in the calibration dataset. OBSNAME={frame.pri_hdr['OBSNAME']}"))
            except:
                raise Exception((f'Frame #{idx_frame}: Unidentified calibration',
                    'type in the Non-linearity calibration dataset'))
    
    # Same number of files as expected
    assert n_nonlin_test == n_nonlin_wo_change_total
    # Unique exposure time for the mean frame
    assert len(set(exptime_mean_frame_list)) == 1
    # Expected exposure time for the mean frame
    assert exptime_mean_frame_list[0] == EXPTIME_MEAN_FRAME
    # Expected number of frames for the mean frame
    assert n_mean_frame == NFRAMES_MEAN_FRAME
    # Needs ordering
    idx_nonlin_sort = np.argsort(filename_nonlin_list)
    # Expected exposure times for Non-linearity
    exptime_nonlin_arr = np.array(exptime_nonlin_list)[idx_nonlin_sort]
    # Subset of unique non-unity EM gains
    nonlin_em_gain_arr = np.unique(cmdgain_nonlin_list)
    nonlin_em_gain_arr.sort()
    assert np.all(nonlin_em_gain_arr == CMDGAIN_NONLIN)
    n_exptime_nonlin = len(EXPTIME_NONLIN)
    # Expected exposure times (this test keeps the same values for all subsets) 
    exptime_nonlin_arr = np.array(exptime_nonlin_list)[idx_nonlin_sort]
    for idx_em, nonlin_em in enumerate(nonlin_em_gain_arr):
        assert np.all(exptime_nonlin_arr[idx_em*n_exptime_nonlin:(idx_em+1)*n_exptime_nonlin] == EXPTIME_NONLIN)
        assert (exptime_nonlin_arr[(idx_em+1)*n_exptime_nonlin-1] in
            exptime_nonlin_arr[idx_em*n_exptime_nonlin:(idx_em+1)*n_exptime_nonlin-1])

def test_nonlin_sorting_w_change():
    """
    Apply the sorting algorithm to a dataset for K-gain and non-linearity
    calibration including EM-gain calibration files in the set to obtain
    the dataset needed for non-linearity calibration and check the
    resulting dataset is consistent with the input dataset. This test has
    different exposure times among the different subsets of non-unity gain
    used to calibrate non-linearity
    """
    dataset_nonlin_w_change = sorting.sort_pupilimg_frames(dataset_w_change, cal_type='non-lin',actual_visit=True)

    # Checks
    n_mean_frame = 0
    n_nonlin_test = 0
    filename_nonlin_list = []
    exptime_mean_frame_list = []
    exptime_nonlin_list = []
    cmdgain_nonlin_list = []
    # This way there's no need to perform a sum check and identifies any issue
    for idx_frame, frame in enumerate(dataset_nonlin_w_change):
        if frame.pri_hdr['OBSNAME'] == 'MNFRAME':
            n_mean_frame += 1
            exptime_mean_frame_list += [frame.ext_hdr['EXPTIME']]
        elif frame.pri_hdr['OBSNAME'] == 'NONLIN':
            n_nonlin_test += 1
            filename_nonlin_list += [frame.filename]
            exptime_nonlin_list += [frame.ext_hdr['EXPTIME']]
            cmdgain_nonlin_list += [frame.ext_hdr['EMGAIN_C']]
        # Testing only non-unity gain frames for Non-linearity
        elif frame.pri_hdr['OBSNAME'] == 'KGAIN':
            pass
        else:
            try:
                raise Exception((f'Frame #{idx_frame}: Misidentified calibration' +
                   f"type in the calibration dataset. OBSNAME={frame.pri_hdr['OBSNAME']}"))
            except:
                raise Exception((f'Frame #{idx_frame}: Unidentified calibration',
                    'type in the Non-linearity calibration dataset'))

    # Same number of files as expected
    assert n_nonlin_test == n_nonlin_wo_change_total
    # Unique exposure time for the mean frame
    assert len(set(exptime_mean_frame_list)) == 1
    # Expected exposure time for the mean frame
    assert exptime_mean_frame_list[0] == EXPTIME_MEAN_FRAME
    # Expected number of frames for the mean frame
    assert n_mean_frame == NFRAMES_MEAN_FRAME
    # Needs ordering
    idx_nonlin_sort = np.argsort(filename_nonlin_list)
    # Expected exposure times for Non-linearity
    exptime_nonlin_arr = np.array(exptime_nonlin_list)[idx_nonlin_sort]
    # Subset of unique non-unity EM gains
    nonlin_em_gain_arr = np.unique(cmdgain_nonlin_list)
    nonlin_em_gain_arr.sort()
    assert np.all(nonlin_em_gain_arr == CMDGAIN_NONLIN)
    n_exptime_nonlin = len(EXPTIME_NONLIN)
    # Expected exposure times (this test changed the values for all subsets)
    exptime_nonlin_arr = np.array(exptime_nonlin_list)[idx_nonlin_sort]
    for idx_em, nonlin_em in enumerate(nonlin_em_gain_arr):
        assert (exptime_nonlin_arr[(idx_em+1)*n_exptime_nonlin-1] in
            exptime_nonlin_arr[idx_em*n_exptime_nonlin:(idx_em+1)*n_exptime_nonlin-1])
    
def test_no_data():
    '''Tests that a Dataset with only metadata gives same results as a normal Dataset.
    '''
    dataset_no_data = dataset_wo_change.copy()
    for frame in dataset_no_data:
        frame.data = None
    dataset_out_no_data = sorting.sort_pupilimg_frames(dataset_no_data, cal_type='k-gain', actual_visit=True)
    dataset_out_wo_change = sorting.sort_pupilimg_frames(dataset_wo_change, cal_type='k-gain', actual_visit=True)
    for i in range(len(dataset_out_wo_change)):
        frame_no_data = dataset_out_no_data[i]
        frame_wo_change = dataset_out_wo_change[i]
        assert np.array_equal(frame_wo_change.data, frame_no_data.data)
        # check that changed pri_hdr values are consistent
        assert frame_wo_change.pri_hdr == frame_no_data.pri_hdr


if __name__ == "__main__":
    setup_module()
    print('Running test_sort_pupilimg_sorting')
    # Testing the sorting algorithm for K-gain calibration
    test_kgain_sorting()
    print('* K-gain tests passed')

    # Testing the sorting algorithm for non-linearity calibration
    test_nonlin_sorting_wo_change()
    print('* Non-linearity tests with identical exposure times among non-unity gains passed')

    test_nonlin_sorting_w_change()
    print('* Non-linearity tests with different exposure times among non-unity gains passed')

    test_no_data()
    print('* Test for reading in Dataset with only metadata works passed')