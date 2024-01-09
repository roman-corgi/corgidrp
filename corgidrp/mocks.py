import numpy as np
import astropy.io.fits as fits
import corgidrp.data as data


def create_dark_calib_files(filedir, numfiles=10):
    """
    Create simulated data to create a master dark. 
    Assume these have already undergone L1 processing
    """
    filepattern = "simcal_dark_{0:04d}.fits"
    for i in range(numfiles):
        prihdr, exthdr = create_default_headers()
        sim_data = np.random.poisson(lam=150, size=(1024, 1024))
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        frame.save(filedir=filedir, filename=filepattern.format(i))


def create_default_headers():
    """
    Creates an empty primary header and an Image extension header with some possible keywords
    """
    prihdr = fits.Header()
    exthdr = fits.Header()

    # fill in prihdr
    prihdr['OBSID'] = 1
    prihdr['BUILD'] = 0
    prihdr['OBSTYPE'] = 'SCI'
    prihdr['MOCK'] = True
    
    # fill in exthdr
    exthdr['PCOUNT'] = 0
    exthdr['GCOUNT'] = 1
    exthdr['BSCALE'] = 1
    exthdr['BZERO'] = 32768
    exthdr['ARRTYPE'] = 'SCI'
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
    exthdr['DATETIME'] = '2024-02-01T10:00:00.000Z'
    exthdr['HIERARCH'] = "DATA_LEVEL= 'L1'"
    exthdr['MISSING'] = False

    return prihdr, exthdr
