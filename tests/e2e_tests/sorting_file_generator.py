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
from corgidrp.mocks import create_default_L1_headers, make_fluxmap_image, nonlin_coefs
from datetime import timedelta
from collections import defaultdict
import pandas as pd

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
    #signal = np.zeros(1)
    #signal = np.random.poisson(lam=1500., size=(1200, 2200)).astype(np.float64)
    # Load the flux map
    here = os.path.abspath(os.path.dirname(__file__))
    fluxmap_init = np.load(Path(os.path.dirname(here), 'test_data', 'FluxMap1024.npy'))
    nonlin_table_path = Path(os.path.dirname(here),'test_data','nonlin_table_TVAC.txt')
    # detector parameters
    kgain = 8.7 # e-/DN
    bias = 2000. # e-
    fluxmap_init[fluxmap_init < 0] = 0 # cleanup flux map a bit
    fluxMap1 = 0.25*fluxmap_init/cmdgain # e/s/px for G=1, then scaled down according to commanded EM gain
    nonlin_flag = True

    # cubic function nonlinearity for emgain of 1
    if nonlin_flag:
        coeffs_1, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)
    else:
        coeffs_1 = [0.0, 0.0, 0.0, 1.0]
        _, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)

    rn = 100.
    image_sim = make_fluxmap_image(fluxMap1,bias,kgain,rn,cmdgain,exptime_sec,coeffs_1,
        nonlin_flag=nonlin_flag)
    # simulating L1 images, which have no err or dq
    del image_sim.err, image_sim.dq
    signal = image_sim.data
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
    datadir = r'E:\E2E_Test_Data3\E2E_Test_Data3\simdata'
    filepath = os.path.join(datadir, filename)
    hdul.writeto(filepath, overwrite = True)
    return filepath


def setup_module():

    #XXX datadir = os.path.join(os.path.dirname(__file__), 'simdata')
    datadir = r'E:\E2E_Test_Data3\E2E_Test_Data3\simdata'
    os.makedirs(datadir, exist_ok=True)
    for f in os.listdir(datadir):
        if f.endswith('.fits'):
            os.remove(os.path.join(datadir, f))

    def read_csv_columns(filepath):
        """
        Reads columns from a CSV file into separate lists.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            list: A list of lists, each containing the values from one column.
        """
        df = pd.read_csv(filepath)
        return [df[col].tolist() for col in df.columns]
    
    pathname = os.path.join('tests','e2e_tests','CAR-121_AUX')
    exptimes = []
    gains = []
    #numframes = []
    tot_numframes = 0
    types = []
    auxfiles = []
    for f in os.listdir(pathname):
        cols = read_csv_columns(os.path.join(pathname, f))
        #auxfiles += len(cols[0]) * [f[:-4]]
        for i in range(len(cols[0])):
            if cols[0][i] == 'MP' or ('NL' in cols[0][i] and cols[1][i] == 1.0):
                numframes = cols[3][i]
                auxfiles += [f[:-4]]*numframes
                types += ['KGain']*numframes
                exptimes += [cols[2][i]]*numframes
                gains += [cols[1][i]]*numframes
            elif 'NL' in cols[0][i] and cols[1][i] != 1.0:
                numframes = cols[3][i]
                auxfiles += [f[:-4]]*numframes
                types += ['NL']*numframes
                exptimes += [cols[2][i]]*numframes
                gains += [cols[1][i]]*numframes
            elif 'EMG' in cols[0][i] and not 'EMGT' in cols[0][i]:
                numframes = cols[3][i]
                auxfiles += [f[:-4]]*numframes
                types += ['EMG']*numframes
                exptimes += [cols[2][i]]*numframes
                gains += [cols[1][i]]*numframes
            elif 'EMGT' in cols[0][i]: # 4 temperatures
                numframes = cols[3][i]*4
                auxfiles += [f[:-4]]*numframes
                types += ['EMGT']*numframes
                exptimes += [cols[2][i]]*numframes
                gains += [cols[1][i]]*numframes
            else:
                raise ValueError('invalid type')
            tot_numframes += numframes
    # now add in 700 DARK frames 
    auxfiles += ['DARK']*700
    types += ['EMG']*700
    exptimes += [10]*700
    gains += [5000]*700 # actually, 50 frames each for 14 different values, but doesn't matter for RAM testing
    tot_numframes += 700

    idx_frame = 0
    filename_list = []
    timestamps = []
    previous_timestamp = None #initialize
    for i_f in range(len(exptimes)):
        filename = make_minimal_image(
        cmdgain=gains[i_f],
        exptime_sec=exptimes[i_f]+ 11, #makes it big enough to keep an increasing order of exposure times in sorting.py
        frameid=idx_frame,
        previous_exptime=0 if i_f == 0 else exptimes[i_f-1],
        previous_timestamp=previous_timestamp,
        auxfile=auxfiles[i_f]
        )   
        idx_frame += 1
        previous_timestamp = fits.getheader(filename, 1)['DATETIME']
        timestamps += [previous_timestamp]
        filename_list += [filename]


if __name__ == "__main__":
    setup_module()
