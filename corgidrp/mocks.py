import astropy.io.fits as fits
import numpy as np

import corgidrp.data as data


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
    filepattern = "simcal_dark_{0:04d}.fits"
    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_headers()
        sim_data = np.random.poisson(lam=150, size=(1024, 1024))
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))
        frames.append(frame)
    dataset = data.Dataset(frames)
    return dataset



def create_prescan_files(filedir=None, numfiles=2, obstype="ENG"):
    """
    Create simulated raw data. 

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 2.
        obstype (str): Observation type. Defaults to "ENG".

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset
    """
    
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
        sim_data = np.random.poisson(lam=150, size=size)
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            frame.save(filedir=filedir, filename=filepattern.format(i))
        frames.append(frame)
    dataset = data.Dataset(frames)
    return dataset


def create_default_headers(obstype="ENG"):
    """
    Creates an empty primary header and an Image extension header with some possible keywords

    Args:
        obstype (str): Observation type. Defaults to "ENG".

    Returns:
        tuple:
            prihdr (fits.Header): Primary FITS Header
            exthdr (fits.Header): Extension FITS Header

    """
    prihdr = fits.Header()
    exthdr = fits.Header()

    if obstype != "ENG":
        obstype = "SCI"
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
    exthdr['DATA_LEVEL'] = "L1"
    exthdr['MISSING'] = False

    return prihdr, exthdr
