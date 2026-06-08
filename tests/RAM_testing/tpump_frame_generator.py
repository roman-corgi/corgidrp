import os
import csv
from pathlib import Path
import numpy as np
import warnings
import math
import re
import datetime
import scipy.ndimage
import pandas as pd
import astropy.io.fits as fits
from astropy.io.fits import Header
from astropy.time import Time
from astropy.io.fits import Header
import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord
import astropy.wcs as wcs
from astropy.table import Table
from astropy.convolution import convolve_fft
from astropy.modeling import models
import corgidrp
import astropy.units as u
from astropy.modeling.models import Gaussian2D
import photutils.centroids as centr
import corgidrp.data as data
from corgidrp.data import Image, Dataset, DetectorParams, FpamFsamCal, FluxcalFactor
import corgidrp.detector as detector
import corgidrp.flat as flat
from corgidrp.detector import imaging_area_geom, unpack_geom
from corgidrp.mocks import create_default_L1_TrapPump_headers, rename_files_to_cgi_format
from corgidrp.pump_trap_calibration import (P1, P1_P1, P1_P2, P2, P2_P2, P3, P2_P3, P3_P3, tau_temp)
from pyklip.instruments.utils.wcsgen import generate_wcs
from corgidrp import measure_companions, corethroughput
from corgidrp.astrom import get_polar_dist, seppa2dxdy, seppa2xy
import datetime
import glob
import shutil
from corgidrp import pol

from emccd_detect.emccd_detect import EMCCDDetect
from emccd_detect.util.read_metadata_wrapper import MetadataWrapper
from scipy.interpolate import interp1d

e2edata_dir =  r'E:\E2E_Test_Data3\E2E_Test_Data3'#'/Users/kevinludwick/Documents/DRP_E2E_Test_Files_v2/E2E_Test_Data'
nonlin_path = os.path.join(e2edata_dir, "TV-36_Coronagraphic_Data", "Cals", "nonlin_table_240322.txt")

def generate_car_pump_trap_data(output_dir,meta_path, EMgain=50.,
                                 read_noise = 125, eperdn = None, e2emode=False,
                                 nonlin_path=nonlin_path, arrtype='SCI',
                                 temperatures=[168,178,188,193],cycles_per_injection=1600,
                                 num_cycles={1: 640, 2: 640, 3: 337, 4: 337}, num_frames_per_config=3, num_phase_times=150):
    """
    Generate mock pump trap data, save it to the output_directory.  Using https://collaboration.ipac.caltech.edu/pages/viewpage.action?pageId=136122677&spaceKey=romancoronagraph&title=7%2BCTI
    as a reference.  The default parameters are for the case of EM gain as specified at the link.

    Args:
        output_dir (str): output directory
        meta_path (str): metadata path
        EMgain (float): desired EM gain for frames
        read_noise (float): desired read noise for frames (in e-).
        eperdn (float):  desired k gain (e-/DN conversion factor).  If None, uses value from DetectorParams.
        e2emode (bool):  If True, e2e simulated data made instead of data for the unit test.
            Difference b/w the two:
            This e2emode data differs from the data generated when e2emode is False in the following ways:
            -The bright pixel of each trap is simulated in a more realistic way (i.e., at every phase time frame).
            -Simulated readout is more realistic (read noise, EM gain, k gain, nonlinearity, bias invoked after traps simulated).
            In the other dataset (when e2emode is False), readout was simulated before traps were added, and no nonlinearity was applied.
            Also, the number of electrons in the dark pixels of the dipoles can no longer be negative, and this condition is enforced.
            -The number of pumps and injected charge are much higher in these frames so that traps stand out above the read noise.
            This was not an issue in the other dataset since read noise was added to frames that were EM-gained before charge was injected, which suppressed the effective read noise.
            -The EM gain used is 1.5.  For a large injected charge amount, the EM gain cannot be very high because of the risk of saturation.
            -The number of phase times is 10 per scheme, to reduce the dataset size (compared to 100 when e2emode is False).
        nonlin_path (str): Path of nonlinearity correction file to use.
            The inverse is applied, implementing rather than correcting nonlinearity.
            If None, no nonlinearity is applied.  Defaults to None.
        arrtype (str): array type (for this function, choice of 'SCI' or 'ENG')
        temperatures (list): list of temperatures to simulate (in K)
        cycles_per_injection (int): number of cycles per injection for trap pumping.
        num_cycles (dict): dictionary with keys 1, 2, 3, and 4 for the 4 trap schemes, and values corresponding to the number of cycles to simulate for each scheme.
        num_frames_per_config (int): number of frames to generate per configuration (i.e., per unique combination of trap scheme, temperature, EM gain, and probability function).
        num_phase_times (int): number of phase times to simulate per trap scheme.  The phase times are the same for all trap schemes, and are log spaced between 1 us and 40 ms (or 10 ms when e2emode is True).
    """

    #If output_dir doesn't exist then make it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # here = os.path.abspath(os.path.dirname(__file__))
    # meta_path = Path(here, '..', 'util', 'metadata_test.yaml')
    #meta_path = Path(here, '..', 'util', 'metadata.yaml')
    meta = MetadataWrapper(meta_path)
    #nrows, ncols, _ = meta._imaging_area_geom()
    # the way emccd_detect works is that it takes an input for the selected
    # image area within the viable CCD pixels, so my input here must be that
    # smaller size (below) as opposed to the full useable CCD pixel size
    # (commented out above)
    nrows, ncols, _ = meta._unpack_geom('image')
    #EM gain
    g = EMgain
    cic = 0.02 #200
    rn = read_noise
    dc = {180: 0.163, 190: 0.243, 200: 0.323, 210: 0.403,
          220: 0.483}
    # Interpolate dark current as a function of temperature
    dc_temps = np.array(list(dc.keys())) #in K
    dc_values = np.array(list(dc.values()))
    dc_interp = interp1d(dc_temps, dc_values, kind='linear', fill_value='extrapolate')

    temp_data = np.array(temperatures)
    # Update dc dict with interpolated values for all temperatures
    for temp in temp_data:
        if temp not in dc:
            dc[temp] = float(dc_interp(temp))
    # dc = {180: 0, 190: 0, 195: 0, 200: 0, 210: 0, 220: 0}
    det_params = DetectorParams({})
    if eperdn is None:
        eperdn = det_params.params['KGAINPAR']
    bias = 1000
    #inj_charge = 500 # 0
    inj_charge = 0.13668 * cycles_per_injection - 1.48 #best fit curve from link in doc strings
    # full_well_image=50000.  # e-
    # full_well_serial=50000.
    full_well_image= int(det_params.params['FWC_PP_E'])
    full_well_serial=int(det_params.params['FWC_EM_E'])
    # trap-pumping done when CGI is secondary instrument (i.e., dark):
    fluxmap = np.zeros((nrows, ncols))
    # frametime for pumped frames: 1000ms, or 1 s
    frametime = 1
    # set these to have no effect, then use these with their input values at the end
    later_eperdn = eperdn
    if e2emode:
        eperdn = 1
        cic = 0.02
        num_pumps = 50000 #120000#90000#15000#5000
        inj_charge = 27000 #31000#70000#45000#8000 #num_pumps/2 # more than num_pumps/4, so no mean_field input needed
        num_frames_per_config = 1
        g = 1
        rn = 0
        bias = 0
        full_well_image=105000.  # e-
        full_well_serial=105000.
        phase_times = 10
    bias_dn = bias/eperdn
    nbits = 14 #1

    def _ENF(g, Nem):
        """
        Returns the ENF.

        Args:
            g (float): gain
            Nem (int): Nem

        Returns:
            float: ENF

        """
        return np.sqrt(2*(g-1)*g**(-(Nem+1)/Nem) + 1/g)
    # std dev in e-, before gain division
    std_dev = np.sqrt(100**2 + _ENF(g,604)**2*g**2*(cic+ 1*dc[220]))
    fit_thresh = 3 #standard deviations above mean for trap detection
    #Offset ensures detection.  Physically, shouldn't have to add offset to
    #frame to meet threshold for detection, but in a small-sized frame, the
    # addition of traps increases the std dev a lot
    # (gain divided, w/o offset: from 22 e- before traps to 73 e- after adding)
    # If I run code with lower threshold, though, I can do an offset of 0.
    # For regular full-sized, definitely shouldn't have to add in offset.
    # Also, a trap can't capture more than mean per pixel e-, which is 200e-
    # in this case.  So max amp P1 trap will not be 2500e- but rather the
    #mean e- per pixel!  But this discrepancy doesn't affect validity of tests.

    offset_u = 0
    # offset_u = (bias_dn + ((cic+1*dc[220])*g + fit_thresh*std_dev/g)/eperdn+\
    #    inj_charge/eperdn)
    # #offset_l = bias_dn + ((cic+1*dc[220])*g - fit_thresh*std_dev/g)/eperdn
    # gives these 0 offset in the function (which gives e-), then add it in
    # by hand and convert to DN
    # and I increase dark current with temp linearly (even though it should
    # be exponential, but dc really doesn't affect anything here)
    emccd = {}
    # leaving out 170K
    #170K: gain of 10-20; gives g*CIC ~ 2000 e-
    # emccd[170] = EMCCDDetect(
    #         em_gain=1,#10,
    #         full_well_image=50000.,  # e-
    #         full_well_serial=50000.,  # e-
    #         dark_current=0.083,  # e-/pix/s
    #         cic=200, # e-/pix/frame; lots of CIC from all the prep clocking
    #         read_noise=100.,  # e-/pix/frame
    #         bias=bias,  # e-
    #         qe=0.9,
    #         cr_rate=0.,  # hits/cm^2/s
    #         pixel_pitch=13e-6,  # m
    #         eperdn=7.,
    #         nbits=14,
    #         numel_gain_register=604,
    #         meta_path=meta_path
    #    )
    #180K: gain of 10-20
    for temp in temp_data:
        emccd[temp] = EMCCDDetect(
                em_gain=g,#10,
                full_well_image=full_well_image,  # e-
                full_well_serial=full_well_serial,  # e-
                dark_current=dc[temp], #0.163,  # e-/pix/s
                cic=cic, # e-/pix/frame; lots of CIC from all the prep clocking
                read_noise=rn,  # e-/pix/frame
                bias=bias,  # e-
                qe=0.9,
                cr_rate=0.,  # hits/cm^2/s
                pixel_pitch=13e-6,  # m
                eperdn=eperdn,
                nbits=nbits,
                numel_gain_register=604,
                meta_path=meta_path
            )

    #when tauc is 3e-3, that gives a mean e- field of 2090 e-
    tauc = 1e-8 #3e-3
    tauc2 = 1.2e-8 # 3e-3
    tauc3 = 1e-8 # 3e-3
    # tried for mean field test, but gave low amps that got lost in noise
    tauc4 = 1e-3 #constant Pc over time not a great approximation in theory
    #In order of amplitudes overall (given comparable tau and tau2):
    # P1 biggest, then P3, then P2
    # E,E3 and cs,cs3 params below chosen to ensure a P1 trap found at its
    # peak amp for good eperdn determination
    # E3,cs3: will give tau outside of 1e-6,1e-2
    # for all temps except 220K; we'll just make sure it's present in all
    # scheme 1 stacks for all temps to ensure good eperdn for all temps;
    # E, cs: will give tau outside of 1e-6, 1e-2
    # for just 170K, which I took out of temp_data
    # E2, cs2: fine for all temps
    E = 0.32 #eV
    E2 = 0.28 #0.24 # eV
    E3 = 0.4 #eV
    # tried mean field test (gets tau = 1e-4 for 180K)
    E4 = 0.266 #eV
    cs = 2 #in 1e-19 m^2
    cs2 = 12 #3 #8 # in 1e-19 m^2
    cs3 = 2 # in 1e-19 m^2
    # for mean field test
    cs4 = 4 # in 1e-19 m^2
    #temp_data = np.array([170, 180, 190, 200, 210, 220])
    #temp_data = np.array([180, 190, 195, 200, 210, 220])

    #temp_data = np.array([180])
    taus = {}
    taus2 = {}
    taus3 = {}
    taus4 = {}
    for i in temp_data:
        taus[i] = tau_temp(i, E, cs)
        taus2[i] = tau_temp(i, E2, cs2)
        taus3[i] = tau_temp(i, E3, cs3)
        taus4[i] = tau_temp(i, E4, cs4)
    #tau = 7.5e-3
    #tau2 = 8.8e-3
    if e2emode:
        time_data = (np.logspace(-6, -2, phase_times))*10**6 # in us
    else:
        time_data = (np.logspace(-6, -1.39794, num_phase_times))*10**6 # in us; this is num_phase_times points log-spaced b/w 1 and 40,000 us
    #time_data = (np.linspace(1e-6, 1e-2, 50))*10**6 # in us
    time_data = time_data.astype(float)
    time_data = np.array(time_data.tolist()*num_frames_per_config)
    time_data_s = time_data/10**6 # in s
    # half the # of frames for length limit
    length_limit = 5 #int(np.ceil((len(time_data)/2)))
    # mean of these frames will be a bit more than 2000e-, which is gain*CIC
    # std dev: sqrt(rn^2 + ENF^2 * g^2(e- signal))

    # with offset_u non-zero in below, I expect to get eperdn 4.7 w/ the code
    amps1 = {}; amps2 = {}; amps3 = {}
    amps1_k = {}; amps1_tau2 = {}; amps3_tau2 = {}; amps1_mean_field = {}
    amps2_mean_field = {}
    amps11 = {}; amps12 = {}; amps22 = {}; amps23 = {}; amps33 = {}; amps21 ={}
    for i in temp_data:
        amps1[i] = offset_u + g*P1(time_data_s, 0, tauc, taus[i], num_cycles[1])/eperdn
        amps11[i] = offset_u + g*P1_P1(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_cycles[1])/eperdn
        amps2[i] = offset_u + g*P2(time_data_s, 0, tauc, taus[i], num_cycles[2])/eperdn
        amps12[i] = offset_u + g*P1_P2(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_cycles[1])/eperdn
        amps22[i] = offset_u + g*P2_P2(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_cycles[2])/eperdn
        amps3[i] = offset_u + g*P3(time_data_s, 0, tauc, taus[i], num_cycles[3])/eperdn
        amps33[i] = offset_u + g*P3_P3(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_cycles[3])/eperdn
        amps23[i] = offset_u + g*P2_P3(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_cycles[2])/eperdn
        # just for (98,33)
        amps21[i] =  offset_u + g*P1_P2(time_data_s, 0, tauc2, taus2[i],
            tauc, taus[i], num_cycles[1])/eperdn
        # now a special amps just for ensuring good eperdn determination
        # actually, doesn't usually meet trap_id thresh, but no harm
        # including it
        amps1_k[i] = offset_u + g*P1(time_data_s, 0, tauc3, taus3[i], num_cycles[1])/eperdn
        # for the case of (89,2) with a single trap with tau2
        amps1_tau2[i] = offset_u + g*P1(time_data_s, 0, tauc2, taus2[i], num_cycles[1])/eperdn
        # for the case of (77,90) with a single trap with tau2
        amps3_tau2[i] = offset_u + g*P3(time_data_s, 0, tauc2, taus2[i], num_cycles[3])/eperdn
        #amps1_k[i] = g*2500/eperdn
        # make a trap for the mean_field test (when mean field=400e- < 2500e-)
        #this trap peaks at 250 e-
        amps1_mean_field[i] = offset_u + \
            g*P1(time_data_s,0,tauc4,taus4[i], num_cycles[1])/eperdn
        amps2_mean_field[i] = offset_u + \
            g*P2(time_data_s,0,tauc4,taus4[i], num_cycles[2])/eperdn
    amps_1_trap = {1: amps1, 2: amps2, 3: amps3, 'sp': amps1_k,
            '1b': amps1_tau2, '3b': amps3_tau2, 'mf1': amps1_mean_field,
            'mf2': amps2_mean_field}
    amps_2_trap = {11: amps11, 12: amps12, 21: amps21, 22: amps22, 23: amps23,
        33: amps33}

    #r0c0[0]: starting row for imaging area (physical CCD pixels)
    #r0c0[1]: starting col for imaging area (physical CCD pixels)
    _, _, r0c0 = meta._imaging_area_geom()

    def add_1_dipole(img_stack, row, col, ori, prob, start, end, temp):
        """Adds a dipole to an image stack img_stack at the location of the
        bright pixel given by row and col (relative to image area coordinates)
        that is of orientation 'above' or
        'below' (specified by ori) for a number of unique phase times
        going from start to end (inclusive; don't use -1 for end; 0 for start
        means first frame, length of time array means last frame), and the
        dipole is of the probability function prob (which can be 1, 2, 3,
        'sp', '1b', '3b', 'mf1', or 'mf2').
        The temperature is specified by temp (in K).

        When e2emode is True, the amount subtracted from the dark pixel and added to the bright
        pixel of a given dipole is constrained so that a pixel is not left with a negative number of electrons.
        See doc string of generate_mock_pump_trap_data for full e2emode details.

        Args:
            img_stack (np.array): image stack
            row (int): row
            col (int): col
            ori (str): orientation
            prob (int): probability
            start (int): start
            end (int): end
            temp (int): temperature

        Returns:
            np.array: image stack
        """
        # length limit controlled by how 'long' deficit pixel is since
        #threshold should be met for all frames for bright pixel
        if ori == 'above':
            #img_stack[start:end,r0c0[0]+row+1,r0c0[1]+col] = offset_l
            region = img_stack[start:end,r0c0[0]+row+1,r0c0[1]+col]
            region_c = img_stack[start:end,r0c0[0]+row+1,r0c0[1]+col].copy()
        if ori == 'below':
            #img_stack[start:end,r0c0[0]+row-1,r0c0[1]+col] = offset_l
            region = img_stack[start:end,r0c0[0]+row-1,r0c0[1]+col]
            region_c = img_stack[start:end,r0c0[0]+row-1,r0c0[1]+col].copy()
        region -= amps_1_trap[prob][temp][start:end]
        if e2emode:
            # can't draw more e- than what's there
            neg_inds = np.where(region < 0)
            good_inds = np.where(region >= 0)
            if neg_inds[0].size > 0:
                print(neg_inds[0].size)
                pass
            region[neg_inds[0]] = 0
            img_stack[start:end,r0c0[0]+row,r0c0[1]+col][good_inds[0]] += amps_1_trap[prob][temp][start:end][good_inds[0]]
            img_stack[start:end,r0c0[0]+row,r0c0[1]+col][neg_inds[0]] += region_c[neg_inds[0]]
        else:
            img_stack[: ,r0c0[0]+row,r0c0[1]+col] += amps_1_trap[prob][temp][:]

        return img_stack

    def add_2_dipole(img_stack, row, col, ori1, ori2, prob, start1, end1,
        start2, end2, temp):
        """Adds a 2-dipole to an image stack img_stack at the location of the
        bright pixel given by row and col (relative to image area coordinates)
        that is of orientation 'above' or
        'below' (specified by ori1 and ori2).  The 1st dipole is for a number
        of unique phase times going from start1 to end1, and
        the 2nd dipole starts from start2 and ends at end2 (inclusive; don't
        use -1 for end; 0 for start means first frame, length of time array
        means last frame). The 2-dipole is of probability function
        prob.  Valid values for prob are 11, 12, 22, 23, and 33.
        The temperature is specified by temp (in K).

        When e2emode is True, the amount subtracted from the dark pixel and added to the bright
        pixel of a given dipole is constrained so that a pixel is not left with a negative number of electrons.
        Also, start2:end2 should not overlap with start1:end1, and the ranges should
        cover the whole 0:10 frames.  This condition allows for the simulation of the probability
        distribution across all phase times.
        See doc string of generate_mock_pump_trap_data for full e2emode details.

        Args:
            img_stack (np.array): image stack
            row (int): row
            col (int): col
            ori1 (str): orientation 1
            ori2 (str): orientation 2
            prob (int): probability
            start1 (int): start 1
            end1 (int): end 1
            start2 (int): start 2
            end2 (int): end 2
            temp (int): temperature

        Returns:
            np.array: image stack
        """
        # length limit controlled by how 'long' deficit pixel is since
        #threshold should be met for all frames for bright pixel
        if ori1 == 'above':
            region1 = img_stack[start1:end1,r0c0[0]+row+1,r0c0[1]+col]
            region1_c = img_stack[start1:end1,r0c0[0]+row+1,r0c0[1]+col].copy()
            #img_stack[start1:end1,r0c0[0]+row+1,r0c0[1]+col] = offset_l
        if ori1 == 'below':
            #img_stack[start1:end1,r0c0[0]+row-1,r0c0[1]+col] = offset_l
            region1 = img_stack[start1:end1,r0c0[0]+row-1,r0c0[1]+col]
            region1_c = img_stack[start1:end1,r0c0[0]+row-1,r0c0[1]+col].copy()
        if ori2 == 'above':
            #img_stack[start2:end2,r0c0[0]+row+1,r0c0[1]+col] = offset_l
            region2 = img_stack[start2:end2,r0c0[0]+row+1,r0c0[1]+col]
            region2_c = img_stack[start2:end2,r0c0[0]+row+1,r0c0[1]+col].copy()
        if ori2 == 'below':
            region2 = img_stack[start2:end2,r0c0[0]+row-1,r0c0[1]+col]
            region2_c = img_stack[start2:end2,r0c0[0]+row-1,r0c0[1]+col].copy()
        # technically, should subtract 1 prob distribution at at time (amps_1_trap), but I'm just subtracting
        # a bit more than I'm supposed to, and doesn't matter too much since these
        # are the deficit pixels (or pixel) next to the bright pixel, which is what counts for doing fits
        region1 -= amps_2_trap[prob][temp][start1:end1]
        region2 -= amps_2_trap[prob][temp][start2:end2]
        if e2emode:
            # can't draw more e- than what's there
            neg_inds1 = np.where(region1 < 0)
            if neg_inds1[0].size > 0:
                print(neg_inds1[0].size)
                pass
            good_inds1 = np.where(region1 >= 0)
            region1[neg_inds1] = 0
            img_stack[start1:end1,r0c0[0]+row,r0c0[1]+col][good_inds1[0]] += amps_2_trap[prob][temp][start1:end1][good_inds1[0]]
            img_stack[start1:end1,r0c0[0]+row,r0c0[1]+col][neg_inds1[0]] += region1_c[neg_inds1[0]]

            # can't draw more e- than what's there
            neg_inds2 = np.where(region2 < 0)
            if neg_inds2[0].size > 0:
                print(neg_inds2[0].size)
                pass
            good_inds2 = np.where(region2 >= 0)
            region2[neg_inds2] = 0
            img_stack[start2:end2,r0c0[0]+row,r0c0[1]+col][good_inds2[0]] += amps_2_trap[prob][temp][start2:end2][good_inds2[0]]
            img_stack[start2:end2,r0c0[0]+row,r0c0[1]+col][neg_inds2[0]] += region2_c[neg_inds2[0]]

        else:
            img_stack[:,r0c0[0]+row,r0c0[1]+col] += amps_2_trap[prob][temp][:]
        # technically, if there is overlap b/w start1:end1 and start2:end2,
        # then you are physically causing too big of a deficit since you're
        # saying more emitted than the amount captured in bright pixel, so
        # avoid this
        return img_stack

    def make_scheme_frames(emccd_inst, phase_times = time_data,
        inj_charge = inj_charge ):
        """Makes a series of frames according to the emccd_detect instance
        emccd_inst, one for each element in the array phase_times (assumed to
        be in s).

        Args:
            emccd_inst (EMCCDDetect): emccd instance
            phase_times (np.array): phase times
            inj_charge (int): injection charge

        Returns:
            np.array: full frames
        """
        full_frames = []
        for i in range(len(phase_times)):
            full = (emccd_inst.sim_full_frame(fluxmap,frametime)).astype(float)
            full_frames.append(full)
        # inj charge is before gain, but since it has no variance,
        # g*0 = no noise from this
        full_frames = np.stack(full_frames)
        # lazy and not putting in the last image row and col, but doesn't
        #matter since I only use prescan and image areas
        # add to just image area so that it isn't wiped with bias subtraction
        full_frames[:,r0c0[0]:,r0c0[1]:] += inj_charge
        return full_frames

    def add_defect(sch_imgs, prob, ori, temp):
        """Adds to all frames of an image stack sch_imgs a defect area with
        local mean above image-area mean such that a
        dipole in that area that isn't detectable unless ill_corr is True.
        The dipole is a single trap with orientation
        ori ('above' or 'below') and is of probability function prob
        (can be 1, 2, or 3).  The temperature is specified by temp (in K).

        Note: If a defect region is arbitrarily small (e.g., a 2x2 region of
        very bright pixels hiding a trap dipole), that trap simply will not
        be found since the illumination correction bin size is not allowed to
        be less than 5.  In v2.0, a moving median subtraction can be
        implemented that would be more likely to catch cases similar to that.
        However, physically, a defect region of such a small number of rows is
        improbable; even a cosmic ray hit, which could have this signature for
        perhaps 1 phase time, is very unlikely to hit the same region while
        data for each phase time is being taken.

        When e2emode is True, the amount subtracted from the dark pixel and added to the bright
        pixel of a given dipole is constrained so that a pixel is not left with a negative number of electrons.
        This condition allows for the simulation of the probability
        distribution across all phase times.
        See doc string of generate_mock_pump_trap_data for full e2emode details.

        Args:
            sch_imgs (np.array): scheme images
            prob (int): probability
            ori (str): orientation
            temp (int): temperature


        Returns:
            np.array: scheme images

        """
        # area with defect (high above mean),
        # but no dipole that stands out enough without ill_corr = True
        amount = 9000
        if e2emode:
            amount = inj_charge*2
        sch_imgs[:,r0c0[0]+12:r0c0[0]+22,r0c0[1]+17:r0c0[1]+27]=g*amount/eperdn
        # now a dipole that meets threshold around local mean doesn't meet
        # threshold around frame mean; would be detected only after
        # illumination correction
        if ori == 'above':
            region = sch_imgs[:,r0c0[0]+13+1, r0c0[1]+21]
            region_c = region.copy()
        if ori == 'below':
            region = sch_imgs[:,r0c0[0]+13-1, r0c0[1]+21]
            region_c = region.copy()
                # 2*offset_u - fit_thresh*std_dev/eperdn
        region -= amps_1_trap[prob][temp][:]
        if e2emode: # realistic handling:  can't trap more charge than what's there in a pixel
            neg_inds = np.where(region < 0)
            if neg_inds[0].size > 0:
                print(neg_inds[0].size)
            good_inds = np.where(region >= 0)
            region[neg_inds[0]] = 0
            sch_imgs[good_inds[0],r0c0[0]+13, r0c0[1]+21] += amps_1_trap[prob][temp][good_inds[0]]
            sch_imgs[neg_inds[0],r0c0[0]+13,r0c0[1]+21] += region_c[neg_inds[0]]
        else:
            sch_imgs[:,r0c0[0]+13, r0c0[1]+21] += amps_1_trap[prob][temp][:]

        return sch_imgs

    #initializing
    sch = {1: None, 2: None, 3: None, 4: None}
    #temps = {170: sch, 180: sch, 190: sch, 200: sch, 210: sch, 220: sch}
    # change from last iteration: make copies of sch below b/c make_scheme_frames() below was changing sch present in
    # EVERY temp for every iteration in the temps for loop; however, no actual change in the output since
    # the output .fits files were saved before the next iteration's make_scheme_frames() is called. So, Max's
    # unit test is unchanged.
    temps = {}
    for temp in temp_data:
        temps[temp] = sch.copy()
    #temps = {180: sch, 190: sch.copy(), 200: sch.copy(), 210: sch.copy(), 220: sch.copy()}
    #temps = {180: sch}

    # first, get rid of files already existing in the folders where I'll put
    # the simulated data
    # for temp in temps.keys():
    #     for sch in [1,2,3,4]:
    #         curr_sch_dir = Path(here, 'test_data_sub_frame_noise', str(temp)+'K',
    #             'Scheme_'+str(sch))
    #         for file in os.listdir(curr_sch_dir):
    #             os.remove(Path(curr_sch_dir, file))

    for temp in temps.keys():
        for sc in [1,2,3,4]:
            temps[temp][sc] = make_scheme_frames(emccd[temp])
        # 14 total traps (15 with the (13,19) defect trap); at least 1 in every
        # possible sub-electrode location
        # careful not to add traps in defect region; do that with add_defect()
        # careful not to add, e.g., bright pixel of one trap in the deficit
        # pixel of another trap since that would negate the original trap

        # add in 'LHSel1' trap in midst of defect for all phase times
        # (only detectable with ill_corr)
        add_defect(temps[temp][1], 1, 'below', temp)
        add_defect(temps[temp][3], 3, 'below', temp)
        #this defect was used for k_prob=2 case instead of the 2 lines above
        # 'LHSel2':
    #    add_defect(temps[temp][1], 2, 'above', temp)
    #    add_defect(temps[temp][2], 1, 'below', temp)
    #    add_defect(temps[temp][4], 3, 'above', temp)
        # add in 'special' max amp trap for good eperdn determination
        # has tau value outside of 1e-6 to 1e-2, but provides a peak trap
        # actually, doesn't meet threshold usually to count as trap, but
        #no harm leaving it in
        if not e2emode:
            add_1_dipole(temps[temp][1], 33, 77, 'below', 'sp', 0, 100, temp)
            # add in 'CENel1' trap for all phase times
        #    add_1_dipole(temps[temp][3], 26, 28, 'below', 'mf2', 0, 100, temp)
        #    add_1_dipole(temps[temp][4], 26, 28, 'above', 'mf2', 0, 100, temp)
            add_1_dipole(temps[temp][3], 26, 28, 'below', 2, 0, 100, temp)
            add_1_dipole(temps[temp][4], 26, 28, 'above', 2, 0, 100, temp)
            # add in 'RHSel1' trap for more than length limit (but diff lengths)
            #unused sch2 in this same pixel that is compatible with another trap
            add_1_dipole(temps[temp][1], 50, 50, 'above', 1, 0, 100, temp)
            add_1_dipole(temps[temp][4], 50, 50, 'above', 3, 3, 98, temp)
            add_1_dipole(temps[temp][2], 50, 50, 'below', 1, 2, 99, temp)
            # FALSE TRAPS: 'LHSel2' trap that doesn't meet length limit of unique
            # phase times even though the actual length is met for first 2
            # (and/or doesn't pass trap_id(), but I've already tested this case in
            # its unit test file)
            # (3rd will be 'unused')
            add_1_dipole(temps[temp][1], 71, 84, 'above', 2, 95, 100, temp)
            add_1_dipole(temps[temp][2], 71, 84, 'below', 1, 95, 100, temp)
            add_1_dipole(temps[temp][4], 71, 84, 'above', 3, 9, 20, temp)
            # 'LHSel2' trap
            add_1_dipole(temps[temp][1], 60, 80, 'above', 2, 1, 100, temp)
            add_1_dipole(temps[temp][2], 60, 80, 'below', 1, 1, 100, temp)
            add_1_dipole(temps[temp][4], 60, 80, 'above', 3, 1, 100, temp)
            # 'CENel2' trap
            add_1_dipole(temps[temp][1], 68, 67, 'above', 1, 0, 100, temp)
            add_1_dipole(temps[temp][2], 68, 67, 'below', 1, 0, 100, temp)
        #    add_1_dipole(temps[temp][1], 68, 67, 'above', 'mf1', 0, 100, temp)
        #    add_1_dipole(temps[temp][2], 68, 67, 'below', 'mf1', 0, 100, temp)
            # 'RHSel2' and 'LHSel3' traps in same pixel (could overlap phase time),
            # but good detectability means separation of peaks
            add_1_dipole(temps[temp][1], 98, 33, 'above', 1, 0, 100, temp)
            add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
                60, 100, 0, 40, temp) #80, 100, 0, 20, temp)
            add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
                60, 100, 0, 40, temp)
            # old:
            # add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
            #     50, 100, 0, 50, temp) #80, 100, 0, 20, temp)
            # add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
            #     50, 100, 0, 50, temp)
            # 'CENel3' trap (where sch3 has a 2-trap where one goes unused)
            add_2_dipole(temps[temp][3], 41, 15, 'above', 'above', 23,
            30, 100, 0, 30, temp)
            add_1_dipole(temps[temp][4], 41, 15, 'below', 2, 30, 100, temp)
            # 'RHSel3' and 'LHSel4'
            add_1_dipole(temps[temp][1], 89, 2, 'below', '1b', 0, 100, temp)
            add_2_dipole(temps[temp][2], 89, 2, 'above', 'above', 12,
                60, 100, 0, 30, temp) #30 was 40 in the past
            add_2_dipole(temps[temp][3], 89, 2, 'above', 'above', 33,
                60, 100, 0, 40, temp)
            # 2 'LHSel4' traps; whether the '0' or '1' trap gets assigned tau2 is
            # somewhat random; if one has an earlier starting temp than the other,
            # it would get assigned tau
            add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
                0, 40, 63, 100, temp)
            add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
                0, 40, 63, 100, temp)
            add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
                0, 40, 63, 100, temp) #30, 60, 100
            # old:
            # add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
            #     0, 40, 50, 100, temp)
            # 'CENel4' trap
            add_1_dipole(temps[temp][1], 56, 56, 'below', 1, 1, 100, temp)
            add_1_dipole(temps[temp][2], 56, 56, 'above', 1, 3, 99, temp)
            #'RHSel4' and 'CENel2' trap (tests 'a' and 'b' splitting in trap_fit_*)
            add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
                60, 100, 0, 40, temp)
            add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
                60, 100, 0, 40, temp)
            add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, 40, temp)
            # old:
            # add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
            #     30, 100, 0, 30, temp)
            # add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
            #     53, 100, 0, 53, temp)
            # add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, 30, temp)

        if e2emode: # full range should be covered if trap present
            add_1_dipole(temps[temp][1], 33, 77, 'below', 'sp', 0, phase_times, temp)
            # add in 'CENel1' trap for all phase times
        #    add_1_dipole(temps[temp][3], 26, 28, 'below', 'mf2', 0, 100, temp)
        #    add_1_dipole(temps[temp][4], 26, 28, 'above', 'mf2', 0, 100, temp)
            add_1_dipole(temps[temp][3], 26, 28, 'below', 2, 0, phase_times, temp)
            add_1_dipole(temps[temp][4], 26, 28, 'above', 2, 0, phase_times, temp)
            # add in 'RHSel1' trap for more than length limit (but diff lengths)
            #unused sch2 in this same pixel that is compatible with another trap
            add_1_dipole(temps[temp][1], 50, 50, 'above', 1, 0, phase_times, temp)
            add_1_dipole(temps[temp][4], 50, 50, 'above', 3, 3, phase_times, temp)
            add_1_dipole(temps[temp][2], 50, 50, 'below', 1, 2, phase_times, temp)
            # FALSE TRAPS: 'LHSel2' trap that doesn't meet length limit of unique
            # phase times even though the actual length is met for first 2
            # (and/or doesn't pass trap_id(), but I've already tested this case in
            # its unit test file)
            # (3rd will be 'unused')
            add_1_dipole(temps[temp][1], 71, 84, 'above', 2, 95, phase_times, temp)
            add_1_dipole(temps[temp][2], 71, 84, 'below', 1, 95, phase_times, temp)
            add_1_dipole(temps[temp][4], 71, 84, 'above', 3, 9, phase_times, temp)
            # 'LHSel2' trap
            add_1_dipole(temps[temp][1], 60, 80, 'above', 2, 1, phase_times, temp)
            add_1_dipole(temps[temp][2], 60, 80, 'below', 1, 1, phase_times, temp)
            add_1_dipole(temps[temp][4], 60, 80, 'above', 3, 1, phase_times, temp)
            # 'CENel2' trap
            add_1_dipole(temps[temp][1], 68, 67, 'above', 1, 0, phase_times, temp)
            add_1_dipole(temps[temp][2], 68, 67, 'below', 1, 0, phase_times, temp)
        #    add_1_dipole(temps[temp][1], 68, 67, 'above', 'mf1', 0, 100, temp)
        #    add_1_dipole(temps[temp][2], 68, 67, 'below', 'mf1', 0, 100, temp)
            # 'RHSel2' and 'LHSel3' traps in same pixel (could overlap phase time),
            # but good detectability means separation of peaks
            add_1_dipole(temps[temp][1], 98, 33, 'above', 1, 0, phase_times, temp)
            add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp) #80, 100, 0, 20, temp)
            add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            # old:
            # add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
            #     50, 100, 0, 50, temp) #80, 100, 0, 20, temp)
            # add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
            #     50, 100, 0, 50, temp)
            # 'CENel3' trap (where sch3 has a 2-trap where one goes unused)
            add_2_dipole(temps[temp][3], 41, 15, 'above', 'above', 23,
            int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            add_1_dipole(temps[temp][4], 41, 15, 'below', 2, 0, phase_times, temp)
            # 'RHSel3' and 'LHSel4'
            add_1_dipole(temps[temp][1], 89, 2, 'below', '1b', 0, phase_times, temp)
            add_2_dipole(temps[temp][2], 89, 2, 'above', 'above', 12,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp) #30 was 40 in the past
            add_2_dipole(temps[temp][3], 89, 2, 'above', 'above', 33,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            # 2 'LHSel4' traps; whether the '0' or '1' trap gets assigned tau2 is
            # somewhat random; if one has an earlier starting temp than the other,
            # it would get assigned tau
            add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
                0, int(phase_times/2), int(phase_times/2), phase_times, temp)
            add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
                0, int(phase_times/2), int(phase_times/2), phase_times, temp)
            add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
                0, int(phase_times/2), int(phase_times/2), phase_times, temp) #30, 60, 100
            # old:
            # add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
            #     0, 40, 50, 100, temp)
            # 'CENel4' trap
            add_1_dipole(temps[temp][1], 56, 56, 'below', 1, 1, phase_times, temp)
            add_1_dipole(temps[temp][2], 56, 56, 'above', 1, 3, phase_times, temp)
            #'RHSel4' and 'CENel2' trap (tests 'a' and 'b' splitting in trap_fit_*)
            add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, phase_times, temp)
            # old:
            # add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
            #     30, 100, 0, 30, temp)
            # add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
            #     53, 100, 0, 53, temp)
            # add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, 30, temp)
        pass
        if e2emode:
            readout_emccd = EMCCDDetect(
                em_gain=EMgain, #10,
                full_well_image=full_well_image,  # e-
                full_well_serial=full_well_serial,  # e-
                dark_current=0,  # e-/pix/s
                cic=0, # e-/pix/frame
                read_noise=read_noise,  # e-/pix/frame
                bias=1000,  # e-
                qe=1, # no QE hit here; just simulating readout
                cr_rate=0.,  # hits/cm^2/s
                pixel_pitch=13e-6,  # m
                eperdn=later_eperdn,
                nbits=nbits,
                numel_gain_register=604,
                meta_path=meta_path,
                nonlin_path=nonlin_path
                )
        # save to FITS files
        for sc in [1,2,3,4]:
            for i in range(len(temps[temp][sc])):
                for fr in range(4):
                    if e2emode:
                        if temps[temp][sc][i].any() >= full_well_image:
                            raise Exception('Saturated before EM gain applied.')
                        # Now apply readout things for e2e mode
                        gain_counts = np.reshape(readout_emccd._gain_register_elements(temps[temp][sc][i].ravel()),temps[temp][sc][i].shape)
                        if gain_counts.any() >= full_well_serial:
                            raise Exception('Saturated after EM gain applied.')
                        output_dn = readout_emccd.readout(gain_counts)
                    else:
                        if fr == 0: # save the first one as a bona fide trap-pump; the rest are junk
                            output_dn = temps[temp][sc][i]
                        else:
                            output_dn = np.zeros_like(temps[temp][sc][i])
                            output_dn[1027:1037] += 3
                    prihdr, exthdr = create_default_L1_TrapPump_headers(arrtype)
                    prim = fits.PrimaryHDU(header = prihdr)
                    hdr_img = fits.ImageHDU(output_dn, header=exthdr)
                    hdul = fits.HDUList([prim, hdr_img])
                    ## Fill in the headers that matter to corgidrp
                    hdul[1].header['EXCAMT']  = temp
                    hdul[1].header['EMGAIN_C'] = EMgain
                    hdul[1].header['ARRTYPE'] = arrtype
                    hdul[1].header['OPMODE'] = 'TRAP_PUMPING'
                    hdul[1].header['FRMTYPE'] = 'NUM'
                    hdul[0].header['VISTYPE'] = 'CGIVST_CAL_TPUMP'
                    for j in range(1, 5):
                        if sc == j:
                            hdul[1].header['TPSCHEM' + str(j)] = num_cycles[j]
                        else:
                            hdul[1].header['TPSCHEM' + str(j)] = 0
                    hdul[1].header['TPTAU'] = time_data[i]

                    t = time_data[i]
                    # curr_sch_dir = Path(here, 'test_data_sub_frame_noise', str(temp)+'K',
                    # 'Scheme_'+str(sch))

                    # if os.path.isfile(Path(output_dir,
                    # str(temp)+'K'+'Scheme_'+str(sch)+'TPUMP_Npumps_10000_gain'+str(g)+'_phasetime'+str(t)+'.fits')):
                    #     hdul.writeto(Path(output_dir,
                    #     str(temp)+'K'+'Scheme_'+str(sch)+'TPUMP_Npumps_10000_gain'+str(g)+'_phasetime'+
                    #     str(t)+'_2.fits'), overwrite = True)
                    # else:
                    # Note: have to use old filename format for now and overwrite later because setting
                    # the filename affects data generation
                    mult_counter = 0
                    #print(mult_counter)
                    filename = Path(output_dir,
                        str(temp)+'K'+'Scheme_'+str(sc)+'TPUMP_Npumps_'+str(int(num_cycles[sc]))+'_gain'+str(EMgain)+'_phasetime'+str(t)+'_config'+str(i)+'_fr'+str(fr)+'.fits')
                    if os.path.exists(filename):
                        mult_counter += 1
                        hdul.writeto(str(filename)[:-4]+'_'+str(mult_counter)+'.fits', overwrite = True)
                    else:
                        hdul.writeto(filename, overwrite = True)

    # After all data generation is complete, rename files to CGI format, because changing the filename
    # in the function above somehow affects the content of the file
    rename_files_to_cgi_format(pattern=os.path.join(output_dir, "*K*Scheme_*TPUMP*.fits"), level_suffix="l1")

if __name__ == "__main__":
    output_dir =  r'E:\E2E_Test_Data3\E2E_Test_Data3\TPUMP_RAM_TEST2'
    metadata_path = os.path.join('c:\\Users\\SensorLab\\Documents\\GitHub\\corgidrp\\tests', 'test_data', "metadata.yaml")
    generate_car_pump_trap_data(output_dir=output_dir,meta_path=metadata_path)
    generate_car_pump_trap_data(output_dir=output_dir, meta_path=metadata_path, EMgain=1.,temperatures=[228,233,238],cycles_per_injection=7200,
                                 num_cycles={1: 3200, 2: 3200, 3: 1684, 4: 1684}, num_frames_per_config=2, num_phase_times=100)