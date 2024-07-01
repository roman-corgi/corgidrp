import numpy as np

import corgidrp.data

import astropy
import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord

import pyklip.fakes as fakes

import scipy.ndimage as ndi
import scipy.optimize as optimize

def centroid(frame):
    """
    Finds the center coordinates for a given frame.

    Args:
        frame (np.ndarray): 2D array to compute centering

    Returns:
        tuple:
            xcen (float): X centroid coordinate
            ycen (float): Y centroid coordinate

    """
    y, x = np.indices(frame.shape)
    
    ycen = np.sum(y * frame)/np.sum(frame)
    xcen = np.sum(x * frame)/np.sum(frame)
    
    return xcen, ycen

def shift_psf(frame, dx, dy, flux, fitsize=10, stampsize=10):
    """
    Creates a template for matching psfs.

    Args:
        frame (np.ndarray): 2D array 
        dx (float): Value to shift psf in 'x' direction
        dy (float): Value to shift psf in 'y' direction
        flux (float): Peak flux value
        fitsize (float): (Optional) Width of square on frame to fit psf to
        stampsize (float): (Optional) Width of psf stamp to create

    Returns:
        ndi..map_coordinates: New coordinates for a shifted psf

    """
    ystamp, xstamp = np.indices([fitsize, fitsize], dtype=float)
    xstamp -= fitsize//2
    ystamp -= fitsize//2
    
    xstamp += stampsize//2 + dx
    ystamp += stampsize//2 + dy
    
    return ndi.map_coordinates(frame * flux, [ystamp, xstamp], mode='constant', cval=0.0).ravel()
    
def measure_offset(frame, xstar_guess, ystar_guess, xoffset_guess, yoffset_guess, guessflux=1, rad=5, stampsize=10):
    """
    Computes the relative offset between stars.
    
    Args:
        frame (np.ndarray): 2D array of data
        xstar_guess (float): Estimate of first star 'x' coordinate
        ystar_guess (float): Estimate of first star 'y' coordinate
        xoffset_guess (float): Estimate of 'x' direction offset between stars
        yoffset_guess (float): Estimate of 'y' direction offset between stars
        guessflux (float): (Optional) Peak flux of first star
        rad (int): (Optional) Radius around first star to compute centroid on
        stampsize (float): (Optional) Width of square psf stamp

    Returns:
        binary_offset (np.array): List of [x,y] offsets in respective directions
        
    """
    #### Centroid on location of star ###
    yind = ystar_guess
    xind = xstar_guess
        
    ymin = yind - rad
    ymax = yind + rad + 1
    xmin = xind - rad
    xmax = xind + rad + 1
    # check bounds
    if ymin < 0:
        ymin = 0
        ymax = 2*rad + 1
    if xmin < 0:
        xmin = 0
        xmax = 2*rad + 1
    if ymax > frame.shape[0]:  # frame is the entire image here
        ymax = frame.shape[0]
        ymin = frame.shape[0] - 2 * rad - 1
    if xmax > frame.shape[1]:
        xmax = frame.shape[1] 
        xmin = frame.shape[1] - 2 * rad - 1
        
    cutout = frame[ymin:ymax, xmin:xmax]
    
    xstar, ystar = centroid(cutout)
    xstar += xmin
    ystar += ymin
    
    ### Create a PSF stamp ###
    ystamp, xstamp = np.indices([stampsize, stampsize], dtype=float)
    xstamp -= float(stampsize//2)
    ystamp -= float(stampsize//2)
    xstamp += xstar
    ystamp += ystar
    
    stamp = ndi.map_coordinates(frame, [ystamp, xstamp])
    
    ### Create a data stamp ###
    fitsize = stampsize
    ydata,xdata = np.indices([fitsize, fitsize], dtype=float)
    xdata -= fitsize//2
    ydata -= fitsize//2
    xdata += xstar + xoffset_guess
    ydata += ystar + yoffset_guess
    
    data = ndi.map_coordinates(frame, [ydata, xdata])
    
    ### Fit the PSF to the data ###
    popt, pcov = optimize.curve_fit(shift_psf, stamp, data.ravel(), p0=(0,0,guessflux))
    tinyoffsets = popt[0:2]

    binary_offset = [xoffset_guess - tinyoffsets[0], yoffset_guess - tinyoffsets[1]]

    return binary_offset

def compute_combinations(iteration, r=2):
    """ 
    Rough equivalivent to itertools.combinations function to create all r-length combinations from a given array.

    Args:
        iteration (np.array): Array from which to create combinations of elements
        r (int): (Optional) Length of combinations (default: 2)

    Returns:
        combination (generator): Generator object of r-length array element combinations
    
    """
    inds = np.indices(np.shape(iteration))
    pool = tuple(inds[0])
    n = len(pool)
    indices = list(range(2))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(2)):
            if indices[i] != i + n - 2:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, 2):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

def angle_between(pos1, pos2):
    """ 
    Used to find the angle from North counterclockwise between two sources in an image.
    
    Args:
        pos1 (tuple): Position of the first target
        pos2 (tuple): Position of the second target
    
    Returns:
        angle (float): Angle between two sources, from north going counterclockwise
    """
    
    xdif = pos2[0] - pos1[0]
    ydif = pos2[1] - pos1[1]  

    if xdif<0:
        if ydif<0:
            angle = np.pi - np.arctan(xdif/ydif)
        else:
            angle = - np.arctan(xdif/ydif)
    else:
        if ydif<0:
            angle = - np.arctan(xdif/ydif) + np.pi
        else: 
            angle = (2*np.pi) - np.arctan(xdif/ydif)
            
    return angle * 180/np.pi

def compute_platescale_and_northangle(image, source_info, center_coord):
    """
    Used to find the platescale and north angle of the image. Calculates the platescale for each pair of stars in the image
    and returns the averged platescale. Calculates the north angle for pairs of stars with the center target
    and returns the averged north angle.
    
    Args:
        image (numpy.ndarray): 2D array of image data 
        source_info (astropy.table.Table): Estimated pixel positions of sources and true sky positions, must have column names 'x', 'y', 'RA', 'DEC'
        center_coord (tuple):
            (float): RA coordinate of the target source
            (float): Dec coordinate of the target source

    Returns:
        platescale (float): Platescale [mas/pixel]
        north_angle (float): Angle between image north and true north [deg]
        
    """

    # load in the image data and source information
    if type(image) != np.ndarray:
        raise TypeError('Image must be 2D numpy.ndarray')

    if type(source_info) != astropy.table.Table:
        raise TypeError('source_info must be an astropy table with columns \'x\',\'y\',\'RA\',\'DEC\'')
    else:
        guesses = source_info
        skycoords = SkyCoord(ra = guesses['RA'], dec= guesses['DEC'], unit='deg', frame='icrs')

    if type(center_coord) != tuple:
        raise TypeError('center_coord must be a tuple coordinate (RA,DEC)')
    else:
        center_coord = SkyCoord(ra = center_coord[0], dec= center_coord[1], unit='deg', frame='icrs')

    # use only center quadrant
    imageshape = np.shape(image)
    quady, quadx = imageshape[0] // 4, imageshape[1] // 4
    center_source_inds = np.where((guesses['x'] >= quadx) & (guesses['x'] <= imageshape[1] - quadx) & (guesses['y'] >= quady) & (guesses['y'] <= imageshape[0] - quady))
    quad_guesses = guesses[center_source_inds]
    quad_skycoords = skycoords[center_source_inds]

    # Platescale calculation
    # create 1_000 random combinations of stars
    all_combinations = list(compute_combinations(quad_guesses))
    rand_inds = np.random.randint(low=0, high=len(all_combinations), size=1000)
    combo_list = np.array(all_combinations)[rand_inds]

    # gather the skycoord separations for all combinations
    seps = np.empty(len(combo_list))
    for i,c in enumerate(combo_list):
        star1 = quad_skycoords[c[0]]
        star2 = quad_skycoords[c[1]]

        sep = star1.separation(star2).mas
        seps[i] = sep

    # find the separations in pixel space on the image between all combinations
    pixseps = np.empty(len(combo_list))
    for i,c in enumerate(combo_list):
        star1 = quad_guesses[c[0]]
        star2 = quad_guesses[c[1]]

        xguess = star2['x'] - star1['x']
        yguess = star2['y'] - star1['y']
        
        xoff, yoff = measure_offset(image, xstar_guess=star1['x'], ystar_guess=star2['x'], xoffset_guess= xguess, yoffset_guess= yguess)

        pixsep = np.sqrt(np.power(xoff,2) + np.power(yoff,2))
        pixseps[i] = pixsep

    # estimate the platescale from each combination and find the mean
    platescales = seps / pixseps
    platescale = np.mean(platescales)

    # North angle calculation
    # find the true centerings of the sources in the image from the guesses and save into a table
    xs = np.empty(len(quad_guesses))
    ys = np.empty(len(quad_guesses))
    for i, (gx, gy) in enumerate(zip(quad_guesses['x'], quad_guesses['y'])):
        pf, fw, x, y = fakes.gaussfit2d(frame= image, xguess= gx, yguess=gy)
        xs[i] = x
        ys[i] = y

    sources = astropy.table.Table()
    sources['x'] = xs
    sources['y'] = ys

    # find the sky position angles between the center star and all others
    pa_sky = np.empty(len(quad_skycoords))
    for i, star in enumerate(quad_skycoords):
        pa = center_coord.position_angle(star).deg
        pa_sky[i] = pa

    # find the pixel position angles
    pa_pixel = np.empty(len(sources))
    image_center = 512.
    for i, (x, y) in enumerate(zip(sources['x'], sources['y'])):
        pa = angle_between((image_center, image_center), (x,y))
        pa_pixel[i] = pa

    # find the difference between the measured and true positon angles
    offset = np.empty(len(sources))
    same_ind = np.where((quad_skycoords.ra.value == center_coord.ra.value) & (quad_skycoords.dec.value == center_coord.dec.value))[0][0]
    for i, (sky, pix) in enumerate(zip(pa_sky, pa_pixel)):
        if i != same_ind:
            if sky > pix:
                north_offset = sky - pix
            else:
                north_offset = sky - pix + 360 
            offset[i] = north_offset

    # get rid of the comparison with self
    offset = np.delete(offset, same_ind)
    north_angle = np.mean(offset)
    
    return platescale, north_angle

def compute_boresight(image, source_info, target_coordinate, cal_properties):
    """ 
    Used to find the offset between the target and the center of the image.

    Args:
        image (numpy.ndarray): 2D array of image data
        source_info (astropy.table.Table): Estimated pixel positions of sources and true sky positions, must have column names 'x', 'y', 'RA', 'DEC'
        target_coordinate (tuple): 
            (float): RA coordinate of the target source
            (float): DEC coordinate of the target source
        cal_properties (tuple):
            (float): Platescale
            (float): North angle

    Returns:
        image_center_RA (float): RA coordinate of the center pixel
        image_center_DEC (float): Dec coordinate of the center pixel
    
    """
    if type(image) != np.ndarray:
        raise TypeError('Image must be 2D numpy.ndarray')
    
    if type(source_info) != astropy.table.Table:
        raise TypeError('source_info must be an astropy table with columns \'x\',\'y\',\'RA\',\'DEC\'')
    else:
        guesses = source_info
        skycoords = SkyCoord(ra = guesses['RA'], dec= guesses['DEC'], unit='deg', frame='icrs')

    if type(target_coordinate) != tuple:
        raise TypeError('target_coordinate must be tuple (RA,DEC)')

    if type(cal_properties) != tuple:
        raise TypeError('cal_properties must be tuple (platescale, north_angle)')

    # use only center quadrant
    imageshape = np.shape(image)
    quady, quadx = imageshape[0] // 4, imageshape[1] // 4
    center_source_inds = np.where((guesses['x'] >= quadx) & (guesses['x'] <= imageshape[1] - quadx) & (guesses['y'] >= quady) & (guesses['y'] <= imageshape[0] - quady))
    quad_guesses = guesses[center_source_inds]

    # create the predicted image header from found platescale and north angle
    vert_ang = np.radians(cal_properties[1])
    pc = np.array([[-np.cos(vert_ang), np.sin(vert_ang)], [np.sin(vert_ang), np.cos(vert_ang)]])
    cdmatrix = pc * (cal_properties[0] * 0.001) / 3600.

    new_hdr = {}
    new_hdr['CD1_1'] = cdmatrix[0,0]
    new_hdr['CD1_2'] = cdmatrix[0,1]
    new_hdr['CD2_1'] = cdmatrix[1,0]
    new_hdr['CD2_2'] = cdmatrix[1,1]
    new_hdr['CRPIX1'] = np.shape(image)[1] // 2
    new_hdr['CRPIX2'] = np.shape(image)[0] // 2
    new_hdr['CTYPE1'] = 'RA---TAN'
    new_hdr['CTYPE2'] = 'DEC--TAN'
    new_hdr['CDELT1'] = (cal_properties[0] * 0.001) / 3600
    new_hdr['CDELT2'] = (cal_properties[0] * 0.001) / 3600
    new_hdr['CRVAL1'] = target_coordinate[0]
    new_hdr['CRVAL2'] = target_coordinate[1]
    w = astropy.wcs.WCS(new_hdr)

    x_sky_to_pix, y_sky_to_pix = astropy.wcs.utils.skycoord_to_pixel(skycoords, wcs=w)
    x_predict, y_predict = x_sky_to_pix[center_source_inds], y_sky_to_pix[center_source_inds]

    # find offset between measured centers and predicted positions    
    image_centerings = np.zeros((len(quad_guesses), 2))
    boresights = np.zeros((len(quad_guesses), 2))
    searchrad = 5
    for i, (xg, yg) in enumerate(zip(quad_guesses['x'], quad_guesses['y'])):
        p, f, xi_center, yi_center = fakes.gaussfit2d(frame= image, xguess= xg, yguess= yg, searchrad=searchrad)
        x_off = xi_center - x_predict[i]
        y_off = yi_center - y_predict[i]
        boresights[i,:] = [x_off, y_off]
        image_centerings[i,:] = [xi_center, yi_center]

    # average all offsets in x,y directions
    boresight_x, boresight_y = np.mean(boresights[:,0]), np.mean(boresights[:,1])

    # convert back to RA, DEC
    image_center_RA = target_coordinate[0] - ((boresight_x * cal_properties[0]) * astropy.units.mas).to(astropy.units.deg).value
    image_center_DEC = target_coordinate[1] - ((boresight_y * cal_properties[0]) * astropy.units.mas).to(astropy.units.deg).value

    return image_center_RA, image_center_DEC

def astrometric_calibration(input_dataset, guesses, target_coordinate):
    """
    Perform the boresight calibration of a dataset.
    
    Args:
        input_dataset (corgidrp.data.Dataset): Dataset containing a single image for astrometric calibration
        guesses (str): Path to file with x,y [pixel] locations of sources AND RA,DEC [deg] true source positions
        target_coordinate (str): Path to file with RA,DEC coordinate of target source 
        
    Returns:
        corgidrp.data.AstrometricCalibration: Astrometric Calibration data object containing image center coords in (RA,DEC), platescale, and north angle
        
    """
    if type(guesses) is not str:
        raise TypeError('guesses must be a str')
    else:
        guesses = ascii.read(guesses)
        if 'x' and 'y' and 'RA' and 'DEC' not in guesses.colnames:
            raise ValueError('guesses must have column names [\'x\',\'y\',\'RA\',\'DEC\']')

    if type(target_coordinate) is not str:
        raise TypeError('target_coordinate must be a str')
    else:
        target = ascii.read(target_coordinate)
        if 'RA' and 'DEC' not in target.colnames:
            raise ValueError('target_coordinate must have column names [\'RA\',\'DEC\']')
        else:
            target_coordinate = (target['RA'][0], target['DEC'][0])

    # load in the data
    dataset = input_dataset.copy()
    image = dataset[0].data

    # compute the calibration properties
    cal_properties = compute_platescale_and_northangle(image=image, source_info=guesses, center_coord=target_coordinate)
    ra, dec = compute_boresight(image=image, source_info=guesses, target_coordinate=target_coordinate, cal_properties=cal_properties)

    # return a single AstrometricCalibration data file
    astrom_data = np.array([ra, dec, cal_properties[0], cal_properties[1]])
    astrom_cal = corgidrp.data.AstrometricCalibration(astrom_data, pri_hdr=dataset[0].pri_hdr, ext_hdr=dataset[0].ext_hdr, input_dataset=dataset)

    # update the history
    history_msg = "Boresight calibration completed"
    astrom_cal_dataset = corgidrp.data.Dataset([astrom_cal])
    astrom_cal_dataset.update_after_processing_step(history_msg)

    return astrom_cal