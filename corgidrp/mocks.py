import astropy.io.fits as fits
import numpy as np

import corgidrp.data as data
import corgidrp.detector as detector
import os

import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord
import astropy.wcs as wcs
from astropy.table import Table

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
        #Add the EMGAIN to the headers
        exthdr['EMGAIN'] = em_gain
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

def create_astrom_data(field_path, filedir=None):
    """
    Create simulated data for astrometric calibration.

    Args:
        field_path (str): Full path to directory with test field data (ra, dec, vmag, etc.)
        filedir (str): (Optional) Full path to directory to save to.

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset

    """
    if type(field_path) != str:
        raise TypeError('field_path must be a str')

    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    cal_field = ascii.read(field_path)
    cal_SkyCoords = SkyCoord(ra= cal_field['RA'], dec= cal_field['DEC'], 
                             unit='deg', frame='icrs')
    
    # hard coded image properties
    size = (1024, 1024)
    sim_data = np.zeros(size)
    ny, nx = size
    center = [nx //2, ny //2]
    target = (80.553428801, -69.514096821)
    platescale = 21.8   #[mas]
    rotation = 45       #[deg]
    fwhm = 3

    # create the simulated image header
    vert_ang = np.radians(rotation)
    pc = np.array([[-np.cos(vert_ang), np.sin(vert_ang)], [np.sin(vert_ang), np.cos(vert_ang)]])
    cdmatrix = pc * (platescale * 0.001) / 3600.

    new_hdr = {}
    new_hdr['CD1_1'] = cdmatrix[0,0]
    new_hdr['CD1_2'] = cdmatrix[0,1]
    new_hdr['CD2_1'] = cdmatrix[1,0]
    new_hdr['CD2_2'] = cdmatrix[1,1]

    new_hdr['CRPIX1'] = center[0]
    new_hdr['CRPIX2'] = center[1]

    new_hdr['CTYPE1'] = 'RA---TAN'
    new_hdr['CTYPE2'] = 'DEC--TAN'

    new_hdr['CDELT1'] = (platescale * 0.001) / 3600
    new_hdr['CDELT2'] = (platescale * 0.001) / 3600

    new_hdr['CRVAL1'] = target[0]
    new_hdr['CRVAL2'] = target[1]

    w = wcs.WCS(new_hdr)

    # create the image data
    xpix, ypix = wcs.utils.skycoord_to_pixel(cal_SkyCoords, wcs=w)
    xpix_inds = np.where((xpix >= 0) & (xpix <= 1024) & (ypix >= 0) & (ypix <= 1024))[0]
    ypix_inds = np.where((ypix >= 0) & (ypix <= 1024) & (xpix >= 0) & (xpix <= 1024))[0]
    xpix = xpix[xpix_inds]
    ypix = ypix[ypix_inds]

    amplitudes = np.power(10, ((cal_field['VMAG'] - 22.5) / (-2.5))) * 10

    # inject gaussian psf stars
    for xpos, ypos, amplitude in zip(xpix, ypix, amplitudes):
        stampsize = int(np.ceil(3 * fwhm))
        sigma = fwhm/ (2.*np.sqrt(2*np.log(2)))
        
        # coordinate system
        y, x = np.indices([stampsize, stampsize])
        y -= stampsize // 2
        x -= stampsize // 2
        
        # find nearest pixel
        x_int = int(round(xpos))
        y_int = int(round(ypos))
        x += x_int
        y += y_int
        
        xmin = x[0][0]
        xmax = x[-1][-1]
        ymin = y[0][0]
        ymax = y[-1][-1]
        
        psf = amplitude * np.exp(-((x - xpos)**2. + (y - ypos)**2.) / (2. * sigma**2))

        # crop the edge of the injection at the edge of the image
        if xmin <= 0:
            psf = psf[:, -xmin:]
            xmin = 0
        if ymin <= 0:
            psf = psf[-ymin:, :]
            ymin = 0
        if xmax >= nx:
            psf = psf[:, :-(xmax-nx + 1)]
            xmax = nx - 1
        if ymax >= ny:
            psf = psf[:-(ymax-ny + 1), :]
            ymax = ny - 1

        # inject the stars into the image
        sim_data[ymin:ymax + 1, xmin:xmax + 1] += psf

    # add Gaussian random noise
    noise_rng = np.random.default_rng(10)
    gain = 1
    ref_flux = 10
    noise = noise_rng.normal(scale= ref_flux/gain * 0.1, size= size)
    sim_data = sim_data + noise

    # load as an image object
    frames = []
    prihdr, exthdr = create_default_headers()
    newhdr = fits.Header(new_hdr)
    frame = data.Image(sim_data, pri_hdr= prihdr, ext_hdr= newhdr)
    filename = "simcal_astrom.fits"
    if filedir is not None:
        # save source SkyCoord locations and pixel location estimates
        guess = Table()
        guess['x'] = [int(x) for x in xpix]
        guess['y'] = [int(y) for y in ypix]
        guess['RA'] = cal_SkyCoords[xpix_inds].ra
        guess['DEC'] = cal_SkyCoords[ypix_inds].dec
        ascii.write(guess, filedir+'/simcal_guesses.csv', overwrite=True)

        center = Table()
        center['RA'] = [target[0]]
        center['DEC'] = [target[1]]
        ascii.write(center, filedir+'/target_guess.csv', overwrite=True)

        frame.save(filedir=filedir, filename=filename)

    frames.append(frame)
    dataset = data.Dataset(frames)

    return dataset