# Most of the values for the different non-unity em gains and the number of
# frames come from either TVAC or some preliminary version of the Commissioning
# test calculations

import random
import numpy as np

from corgidrp import calsort
import corgidrp.data as data
from corgidrp.data import Image
from corgidrp.mocks import create_default_headers


# Sub-functions
def get_cmdgain_exptime_mean_frame(
    exptime_sec=5,
    nframes=30,
    ):
    """
    Create an array of CMDGAIN, EXPTIME for frames that will be used to
    generate a mean frame.

    Rules for mean frame plus full doc string TBW
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

    Rules for k-gain plus full doc string TBW
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

    Rules for non-linearity plus full doc string TBW
    """
    cmdgain_list = nonunity_em
    exptime_list = []
    fac_change = 0
    if change_exptime:
        fac_change = 1
    for cmdgain in cmdgain_list:
        exptime_sec = np.array(exptime_sec) * (1 + fac_change*random.uniform(-0.1, 0.1))
        exptime_list += exptime_sec.tolist()
    return cmdgain_list, exptime_list

def make_minimal_image(
    cmdgain=1,
    exptime_sec=0,
        ):
    """
    This function makes a mock frame with minimum memory in its data and error
    fields. It is used in this test script only.

    Args:

    Returns:
        corgidrp.data.Image
    """
    signal_arr = np.zeros(1)

    prhd, exthd = create_default_headers()
    # Record actual commanded EM
    exthd['CMDGAIN'] = cmdgain
    # Record actual exposure time
    exthd['EXPTIME'] = exptime_sec
    # Mock error maps
    err = np.ones(1)
    dq = np.zeros(1, dtype = np.uint16)
    image = Image(signal_arr, pri_hdr = prhd, ext_hdr = exthd, err = err,
        dq = dq)
    return image

# Main code

# Mean frame
EXPTIME_MEAN_FRAME = 5
NFRAMES_MEAN_FRAME = 30
cmdgain_mean_frame, exptime_mean_frame = get_cmdgain_exptime_mean_frame(
    exptime_sec=EXPTIME_MEAN_FRAME,
    nframes=NFRAMES_MEAN_FRAME)

if len(cmdgain_mean_frame) != len(exptime_mean_frame):
    raise Exception('Inconsistent lengths in the mean frame')
if len(cmdgain_mean_frame) < 30:
    raise Exception(f'Insufficient frames ({len(cmdgain_mean_frame)}) for the mean frame')

# K-gain
EXPTIME_KGAIN = [0.077, 0.770, 1.538, 2.308, 3.077, 3.846, 4.615, 5.385, 6.154,
    6.923, 7.692, 8.462, 9.231, 10.000, 11.538, 10.769, 12.308, 13.077, 13.846,
    14.615, 15.385, 1.538]
NFRAMES_KGAIN = 5
cmdgain_kgain, exptime_kgain = get_cmdgain_exptime_kgain(
    exptime_sec=EXPTIME_KGAIN,
    nframes=NFRAMES_KGAIN)

if len(cmdgain_kgain) != len(exptime_kgain):
    raise Exception('Inconsistent lengths in K-gain')
frames_per_exptime = len(exptime_kgain) / len(set(exptime_kgain))
if frames_per_exptime < 5:
    raise Exception(f'Insufficient frames ({frames_per_exptime:.1ff}) per unique exposure time in k-gain')
unique_exp_times = len(cmdgain_kgain) / NFRAMES_KGAIN
if unique_exp_times < 22:
    raise Exception(f'Insufficient unique exposure times ({unique_exp_times:.1f}) in k-gain')

# Non-linearity
EXPTIME_NONLIN = [0.076, 0.758, 1.515, 2.273, 3.031, 3.789, 4.546, 5.304, 6.062,
    6.820, 7.577, 8.335, 9.093, 9.851, 10.608, 11.366, 12.124, 12.881,
    13.639, 14.397, 15.155, 1.515]
CMDGAIN_NONLIN = [1.65, 5.24, 8.60, 16.70, 27.50, 45.26, 87.50, 144.10, 237.26,
    458.70, 584.40]
# w/o changing exposure times among non-unity EM gains
cmdgain_nonlin, exptime_nonlin = get_cmdgain_exptime_nonlin(
    exptime_sec=EXPTIME_NONLIN,
    nonunity_em=CMDGAIN_NONLIN,
    change_exptime=False)
# changing exposure times among non-unity EM gains
cmdgain_nonlin_w_change, exptime_nonlin_w_change = get_cmdgain_exptime_nonlin(
    exptime_sec=EXPTIME_NONLIN,
    nonunity_em=CMDGAIN_NONLIN,
    change_exptime=True)

# Actual values of the non-unity em gains for low em-gain
em_emgain = [1.000, 1.007, 1.015, 1.024, 1.035, 1.047, 1.060, 1.076, 1.094,
1.115, 1.138, 1.165, 1.197, 1.234, 1.276, 1.325, 1.385, 1.453, 1.534, 1.633,
1.749, 1.890, 2.066, 2.278, 2.541, 2.873, 3.308, 3.858, 4.581, 5.577, 6.189,
6.906, 7.753, 8.757, 9.955, 11.392, 13.222, 15.351, 17.953, 21.157, 25.128,
30.082, 36.305, 44.621, 54.768, 67.779, 84.572, 106.378, 134.858, 172.244,
224.385, 290.538, 378.283, 494.762, 649.232, 853.428]

# Dataset
# Loop over two cases for non-linearity
change_exptime = [False, True]

Add frameID to the filename

for change in change_exptime:
    # Mean frame
    for 
    # K-gain
    
    # Non-linearity

    # EM-gain


breakpoint()


