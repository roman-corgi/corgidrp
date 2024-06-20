import astropy.io.fits as fits
from astropy.time import Time
import numpy as np

import corgidrp.data as data
import corgidrp.detector as detector
import os
from pathlib import Path

def create_noise_maps(F, Ferr, Fdq, C, Cerr, Cdq, D, Derr, Ddq):
    '''
    Create simulated noise maps for test_masterdark_from_noisemaps.py.

    Arguments:
        F: 2D np.array for fixed-pattern noise (FPN) data array
        Ferr: 2D np.array for FPN err array
        Fdq: 2D np.array for FPN DQ array
        C: 2D np.array for clock-induced charge (CIC) data array
        Cerr: 2D np.array for CIC err array
        Cdq: 2D np.array for CIC DQ array
        D: 2D np.array for dark current data array
        Derr: 2D np.array for dark current err array
        Ddq: 2D np.array for dark current DQ array

    Returns:
        Fnoisemap: corgidrp.data.NoiseMap instance for FPN
        Cnoisemap: corgidrp.data.NoiseMap instance for CIC
        Dnoisemap: corgidrp.data.NoiseMap instance for dark current
    '''

    prihdr, exthdr = create_default_headers()
    # taken from end of calibrate_darks_lsq()
    exthdr['EXPTIME'] = None
    if 'EMGAIN_M' in exthdr.keys():
        exthdr['EMGAIN_M'] = None
    exthdr['CMDGAIN'] = None
    exthdr['KGAIN'] = None
    exthdr['BUNIT'] = 'detected electrons'
    exthdr['HIERARCH DATA_LEVEL'] = None
    # simulate raw data filenames
    exthdr['DRPNFILE'] = 2
    exthdr['FILE0'] = '0.fits'
    exthdr['FILE1'] = '1.fits'

    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electrons'

    exthdr['DATATYPE'] = 'FPN NoiseMap'
    Fnoisemap = data.NoiseMap(F, 'FPN', pri_hdr=prihdr, ext_hdr=exthdr, err=Ferr,
                              dq=Fdq, err_hdr=err_hdr)
    exthdr2 = exthdr.copy()
    exthdr['DATATYPE'] = 'CIC NoiseMap'
    Cnoisemap = data.NoiseMap(C, 'CIC', pri_hdr=prihdr, ext_hdr=exthdr2, err=Cerr,
                              dq=Cdq, err_hdr=err_hdr)
    exthdr3 = exthdr.copy()
    exthdr3['DATATYPE'] = 'dark current NoiseMap'
    Dnoisemap = data.NoiseMap(D, 'DC', pri_hdr=prihdr, ext_hdr=exthdr3, err=Derr,
                              dq=Ddq, err_hdr=err_hdr)

    return Fnoisemap, Cnoisemap, Dnoisemap

def create_synthesized_master_dark_calib():
    '''
    Create simulated data specifically for test_calibrate_darks_lsq.py.

    Returns:
        datasets: List of corgidrp.data.Dataset instances
    The simulated dataset
    '''
    one_up = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    meta_path = Path(one_up, 'corgidrp', 'util', 'metadata_test.yaml')
    meta = detector.Metadata(meta_path)

    dark_current = 8.33e-4 #e-/pix/s
    cic=0.02  # e-/pix/frame
    read_noise=100 # e-/pix/frame
    bias=2000 # e-
    eperdn = 7 # e-/DN conversion; used in this example for all stacks
    g_picks = (np.linspace(2, 5000, 7))
    t_picks = (np.linspace(2, 100, 7))
    grid = np.meshgrid(g_picks, t_picks)
    g_arr = grid[0].ravel()
    t_arr = grid[1].ravel()
    #added in after emccd_detect makes the frames (see below)
    # The mean FPN that will be found is eperdn*(FPN//eperdn)
    # due to how I simulate it and then convert the frame to uint16
    FPN = 21 # e
    # the bigger N is, the better the adjusted R^2 per pixel becomes
    N = 30 #Use N=600 for results with better fits (higher values for adjusted
    # R^2 per pixel)
    # image area, including "shielded" rows and cols:
    imrows, imcols, imr0c0 = meta._imaging_area_geom()
    prerows, precols, prer0c0 = meta._unpack_geom('prescan')

    datasets = []
    for i in range(len(g_arr)):
        frame_list = []
        for l in range(N): #number of frames to produce
            # Simulate full dark frame (image area + the rest)
            frame_dn_dark = np.zeros((meta.frame_rows, meta.frame_cols))
            im = np.random.poisson(cic*g_arr[i]+
                                t_arr[i]*g_arr[i]*dark_current,
                                size=(meta.frame_rows, meta.frame_cols))
            frame_dn_dark = im
            # prescan has no dark current
            pre = np.random.poisson(cic*g_arr[i],
                                    size=(prerows, precols))
            frame_dn_dark[prer0c0[0]:prer0c0[0]+prerows,
                            prer0c0[1]:prer0c0[1]+precols] = pre
            rn = np.random.normal(0, read_noise,
                                    size=(meta.frame_rows, meta.frame_cols))
            with_rn = frame_dn_dark + rn + bias

            frame_dn_dark = with_rn/eperdn
            # simulate a constant FPN in image area (not in prescan
            # so that it isn't removed when bias is removed)
            frame_dn_dark[imr0c0[0]:imr0c0[0]+imrows,imr0c0[1]:
            imr0c0[1]+imcols] += FPN/eperdn # in DN
            # simulate telemetry rows, with the last 5 column entries with high counts
            frame_dn_dark[-1,-5:] = 100000 #DN
            # take raw frames and process them to what is needed for input
            # No simulated pre-processing bad pixels or cosmic rays, so just subtract bias
            # and multiply by k gain
            frame_dn_dark -= bias/eperdn
            frame_dn_dark *= eperdn

            # Now make this into a bunch of corgidrp.Dataset stacks
            prihdr, exthdr = create_default_headers()
            frame = data.Image(frame_dn_dark, pri_hdr=prihdr,
                            ext_hdr=exthdr)
            frame.ext_hdr['CMDGAIN'] = g_arr[i]
            frame.ext_hdr['EXPTIME'] = t_arr[i]
            frame.ext_hdr['KGAIN'] = eperdn
            frame_list.append(frame)
        dataset = data.Dataset(frame_list)
        datasets.append(dataset.copy())

    return datasets

def create_dark_calib_files(filedir=None, numfiles=10):
    """
    Create simulated data to create a master dark.
    Assume these have already undergone L1 processing and are L2a level products

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 10.

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset
    """
    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    filepattern = "simcal_dark_{0:04d}.fits"
    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_headers()
        np.random.seed(456+i); sim_data = np.random.poisson(lam=150., size=(1024, 1024)).astype(np.float64)
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))
        frames.append(frame)
    dataset = data.Dataset(frames)
    return dataset

def create_nonlinear_dataset(filedir=None, numfiles=2,em_gain=2000):
    """
    Create simulated data to non-linear data to test non-linearity correction.

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 2 (not creating the cal here, just testing the function)
        em_gain (int): The EM gain to use for the simulated data.  Defaults to 2000.

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset
    """

    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    filepattern = "simcal_nonlin_{0:04d}.fits"
    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_headers()
        #Add the CMDGAIN to the headers
        exthdr['CMDGAIN'] = em_gain
        # Create a default
        size = 1024
        sim_data = np.zeros([size,size])
        data_range = np.linspace(10,65536,size)
        # Generate data for each row, where the mean increase from 10 to 65536
        for x in range(size):
            np.random.seed(123+x); sim_data[:, x] = np.random.poisson(data_range[x], size).astype(np.float64)

        non_linearity_correction = data.NonLinearityCalibration(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..',"tests","test_data","nonlin_sample.fits"))

        #Apply the non-linearity to the data. When we correct we multiple, here when we simulate we divide
        sim_data /= detector.get_relgains(sim_data,em_gain,non_linearity_correction)

        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))
        frames.append(frame)
    dataset = data.Dataset(frames)
    return dataset

def create_cr_dataset(filedir=None, datetime=None, numfiles=2, em_gain=500, numCRs=5, plateau_length=10):
    """
    Create simulated non-linear data with cosmic rays to test CR detection.

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        datetime (astropy.time.Time): (Optional) Date and time of the observations to simulate.
        numfiles (int): Number of files in dataset.  Defaults to 2 (not creating the cal here, just testing the function)
        em_gain (int): The EM gain to use for the simulated data.  Defaults to 2000.
        numCRs (int): The number of CR hits to inject. Defaults to 5.
        plateau_length (int): The minimum length of a CR plateau that will be flagged by the filter.

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset.
    """

    if datetime is None:
        datetime = Time('2024-01-01T11:00:00.000Z')

    kgain = detector.get_kgain(datetime=datetime)
    fwc_em_dn = detector.get_fwc_em_e(datetime=datetime) / kgain
    fwc_pp_dn = detector.get_fwc_pp_e(datetime=datetime) / kgain
    fwc = np.min([fwc_em_dn,em_gain*fwc_pp_dn])
    dataset = create_nonlinear_dataset(filedir=None, numfiles=numfiles,em_gain=em_gain)

    im_width = dataset.all_data.shape[-1]

    # Overwrite dataset with a poisson distribution
    np.random.seed(123)
    dataset.all_data[:,:,:] = np.random.poisson(lam=150,size=dataset.all_data.shape).astype(np.float64)

    # Loop over images in dataset
    for i in range(len(dataset.all_data)):

        # Save the date
        dataset[i].ext_hdr['DATETIME'] = str(datetime)

        # Pick random locations to add a cosmic ray
        for x in range(numCRs):
            np.random.seed(123+x)
            loc = np.round(np.random.uniform(0,im_width-1, size=2)).astype(int)

            # Add the CR plateau
            tail_start = np.min([loc[1]+plateau_length,im_width])
            dataset.all_data[i,loc[0],loc[1]:tail_start] += fwc

            if tail_start < im_width-1:
                tail_len = im_width-tail_start
                cr_tail = [fwc/(j+1) for j in range(tail_len)]
                dataset.all_data[i,loc[0],tail_start:] += cr_tail

        # Save frame if desired
        if filedir is not None:
            filepattern = "simcal_cosmics_{0:04d}.fits"
            dataset[i].save(filedir=filedir, filename=filepattern.format(i))

    return dataset

def create_prescan_files(filedir=None, numfiles=2, obstype="SCI"):
    """
    Create simulated raw data.

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 2.
        obstype (str): Observation type. Defaults to "SCI".

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset
    """
    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    if obstype == "SCI":
        size = (1200, 2200)
    elif obstype == "ENG":
        size = (2200, 2200)
    elif obstype == "CAL":
        size = (2200,2200)
    else:
        raise ValueError(f'Obstype {obstype} not in ["SCI","ENG","CAL"]')


    filepattern = f"sim_prescan_{obstype}"
    filepattern = filepattern+"{0:04d}.fits"

    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_headers(obstype=obstype)
        sim_data = np.random.poisson(lam=150., size=size).astype(np.float64)
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)

        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))

        frames.append(frame)

    dataset = data.Dataset(frames)

    return dataset

def create_default_headers(obstype="SCI"):
    """
    Creates an empty primary header and an Image extension header with some possible keywords

    Args:
        obstype (str): Observation type. Defaults to "SCI".

    Returns:
        tuple:
            prihdr (fits.Header): Primary FITS Header
            exthdr (fits.Header): Extension FITS Header

    """
    prihdr = fits.Header()
    exthdr = fits.Header()

    if obstype != "SCI":
        NAXIS1 = 2200
        NAXIS2 = 1200
    else:
        NAXIS1 = 2200
        NAXIS2 = 2200

    # fill in prihdr
    prihdr['OBSID'] = 0
    prihdr['BUILD'] = 0
    prihdr['OBSTYPE'] = obstype
    prihdr['MOCK'] = True

    # fill in exthdr
    exthdr['NAXIS'] = 2
    exthdr['NAXIS1'] = NAXIS1
    exthdr['NAXIS2'] = NAXIS2
    exthdr['PCOUNT'] = 0
    exthdr['GCOUNT'] = 1
    exthdr['BSCALE'] = 1
    exthdr['BZERO'] = 32768
    exthdr['ARRTYPE'] = obstype # seems to be the same as OBSTYPE
    exthdr['SCTSRT'] = '2024-01-01T12:00:00.000Z'
    exthdr['SCTEND'] = '2024-01-01T20:00:00.000Z'
    exthdr['STATUS'] = 0
    exthdr['HVCBIAS'] = 1
    exthdr['OPMODE'] = ""
    exthdr['EXPTIME'] = 60.0
    exthdr['CMDGAIN'] = 1.0
    exthdr['CYCLES'] = 100000000000
    exthdr['LASTEXP'] = 1000000
    exthdr['BLNKTIME'] = 10
    exthdr['EXPCYC'] = 100
    exthdr['OVEREXP'] = 0
    exthdr['NOVEREXP'] = 0
    exthdr['EXCAMT'] = 40.0
    exthdr['FCMLOOP'] = ""
    exthdr['FSMINNER'] = ""
    exthdr['FSMLOS'] = ""
    exthdr['FSM_X'] = 50.0
    exthdr['FSM_Y'] = 50.0
    exthdr['DMZLOOP'] = ""
    exthdr['SPAM_H'] = 1.0
    exthdr['SPAM_V'] = 1.0
    exthdr['FPAM_H'] = 1.0
    exthdr['FPAM_V'] = 1.0
    exthdr['LSAM_H'] = 1.0
    exthdr['LSAM_V'] = 1.0
    exthdr['FSAM_H'] = 1.0
    exthdr['FSAM_V'] = 1.0
    exthdr['CFAM_H'] = 1.0
    exthdr['CFAM_V'] = 1.0
    exthdr['DPAM_H'] = 1.0
    exthdr['DPAM_V'] = 1.0
    exthdr['DATETIME'] = '2024-01-01T11:00:00.000Z'
    exthdr['HIERARCH DATA_LEVEL'] = "L1"
    exthdr['MISSING'] = False

    return prihdr, exthdr
