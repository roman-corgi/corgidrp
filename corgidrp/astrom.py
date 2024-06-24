import corgidrp.data
import numpy as np
import corgidrp

import astropy
from astropy.coordinates import SkyCoord

import pyklip
import pyklip.fakes as fakes

import measure_offset

def compute_boresight(image, target_coordinate, cal_properties):
    """ 
    Used to find the offset between the target and the center of the image

    Args:
        image (numpy.ndarray): a 2d array of image data
        target_coordinate (tuple): 
            (float): the RA coordinate of the target source
            (float): the DEC coordinate of the target source
        cal_properties (tuple):
            (float): the image platescale
            (float): the image north angle

    Returns:
        tuple:
            image_center_RA (float): the RA coordinate of the center pixel
            image_center_DEC (float): the Dec coordinate of the center pixel
    
    """
    # load in the image data
    if type(image) != np.ndarray:
        print('Wrong image type: must be numpy.ndarray (.fits image data)')
        return
   
    image_shape = np.shape(image)
    image_center_x = (image_shape[0]-1) // 2
    image_center_y = (image_shape[1]-1) // 2    
    
    # estimate the location of the target star with pyklip
    # assuming the target source is meant to fall on the center pixel
    target_guess = (image_center_x, image_center_y)
    peakflx, fwhm, source_center_x, source_center_y = fakes.gaussfit2d(frame= image, xguess= target_guess[0], yguess= target_guess[1])
    
    # find the distance between the center of the image
    offset_x = source_center_x - image_center_x
    offset_y = source_center_y - image_center_y
    pixel_offset = np.sqrt(np.power(offset_x, 2) + np.power(offset_y, 2))

    # convert pixel offset back to SkyCoord separation
    platescale, northangle = cal_properties


    return pixel_offset

def compute_combinations(iteration, r=2):
    """ 
    Rough equivalivent to itertools.combinations function to create all r-length combinations from a given array.

    Args:
        iteration (np.array): array from which to create combinations of elements
        r (int): length of combinations (default: 2)

    Returns:
        combination (generator): generator object of r-length array element combinations
    
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

def compute_platescale(image, source_guesses, skycoords_):
    """
    Used to find the platescale of the image. Calculates the platescale for each pair of stars in the image
    and returns the averged platescale.

    Args:
        image (numpy.ndarray): a 2d array of image data 
        source_guesses (str): path to .csv file holding the x, y [pixel] positions: must have column names 'x', 'y'
        skycoords_ (str): path to .csv holding RA, DEC [deg] locations of sources: must have column names 'RA' and 'DEC'

    Returns: 
        platescale (float): the image platescale [mas/pixel]
    
    """
    # load in the image data and source information
    if type(image) != np.ndarray:
        raise TypeError("Image type must be numpy.ndarray")

    if type(source_guesses) != str:
        raise TypeError("source_guesses must be a filepath (str)")
    else:
        guesses = ascii.read(source_guesses)

    if type(skycoords_) != str:
        raise TypeError("skycoords_ must be a filepath (str)")
    else:
        skycoords_table = ascii.read(skycoords_)
        skycoords = SkyCoord(ra = skycoords_table['RA'], dec= skycoords_table['DEC'], unit=u.deg, frame='icrs')
    
    # create 1_000 random combinations of stars
    all_combinations = list(compute_combinations(guesses['x']))
    rand_inds = np.random.randint(low=0, high=len(all_combinations), size=1000)
    combo_list = np.array(all_combinations)[rand_inds]

    # gather all the skycoord separations for all combinations
    seps = np.empty(len(combo_list))
    for c, i in zip(combo_list, range(len(combo_list))):
        star1 = skycoords[c[0]]
        star2 = skycoords[c[1]]

        sep = star1.separation(star2).mas
        seps[i] = sep

    # find the separations in pixel space on the image between all combinations
    pixseps = np.empty(len(combo_list))
    for c, i in zip(combo_list, range(len(combo_list))):
        star1 = guesses[c[0]]
        star2 = guesses[c[1]]

        xguess = star2['x'] - star1['x']
        yguess = star2['y'] - star1['y']
        
        xoff, yoff = measure_offset.measure_offset(image, xstar_guess=star1['x'], ystar_guess=star2['x'], xoffset_guess= xguess, yoffset_guess= yguess)

        pixsep = np.sqrt(np.power(xoff,2) + np.power(yoff,2))
        pixseps[i] = pixsep

    # estimate the platescale from each combination and find the mean
    platescales = seps / pixseps
    platescale = np.mean(platescales)

    return platescale

def angle_between(pos1, pos2):
    """ 
    Used to find the angle from North counterclockwise between two sources in an image.
    
    Args:
        pos1 (tuple): the (x,y) position of the first target
        pos2 (tuple): the (x,y) position of the second target
    
    Returns:
        angle (float): the angle between image north and true north
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

def compute_northangle(image, source_guesses, center_coord, skycoords_):
    """
    Used to find the north angle of the image. Calculates the north angle for pairs of stars with the center target
    and returns the averged north angle.
    
    Args:
        image (numpy.ndarray): a 2d array of image data 
        source_guesses (str): path to .csv file holding the x, y [pixel] positions: must have column names 'x', 'y'
        center_coords (tuple):
            (float): the RA coordinate of the target source
            (float): the Dec coordinate of the target source
        skycoords_ (str): path to .csv holding RA, DEC [deg] locations of sources: must have column names 'RA' and 'DEC'

    Returns: 
        north_angle (float): the angle between image north and true north [deg]
        
    """

    # load in the image data and source information
    if type(image) != np.ndarray:
        raise TypeError('Image must be 2D numpy.ndarray')

    if type(source_guesses) != str:
        raise TypeError('source_guesses must be a filepath (str)')
    else:
        guesses = ascii.read(source_guesses)

    if type(skycoords_) != str:
        raise TypeError('skycoords_ must be a filepath (str)')
    else:
        skycoords_table = ascii.read(skycoords_)
        skycoords = SkyCoord(ra = skycoords_table['RA'], dec= skycoords_table['DEC'], unit='deg', frame='icrs')

    if type(center_coord) != tuple:
        raise TypeError('center_coord must be a tuple coordinate (RA,DEC)')
    else:
        center_coord = SkyCoord(ra = center_coord[0], dec= center_coord[1], unit='deg', frame='icrs')

    # find the true centerings of the sources in the image from the guesses and save into a table
    xs = np.empty(len(guesses))
    ys = np.empty(len(guesses))
    for gx,gy,i in zip(guesses['x'], guesses['y'], range(len(guesses))):
        pf, fw, x, y = fakes.gaussfit2d(frame= image, xguess= gx, yguess=gy)
        xs[i] = x
        ys[i] = y

    sources = astropy.table.Table()
    sources['x'] = xs
    sources['y'] = ys

    # find the on sky position angles between the center star and all others
    pa_sky = np.empty(len(skycoords))
    for star, i in zip(skycoords, range(len(skycoords))):
        pa = center_coord.position_angle(star).deg
        pa_sky[i] = pa

    # find the pixel space position angles
    pa_pixel = np.empty(len(sources))
    image_center = (np.shape(image)[0] -1) // 2
    for x, y, i in zip(sources['x'], sources['y'], range(len(sources))):
        pa = angle_between((image_center, image_center), (x,y))
        pa_pixel[i] = pa


    # find the difference between the measured and true positon angles
    offset = np.empty(len(sources))
    same_ind = np.where((skycoords.ra.value == center_coord.ra.value) & (skycoords.dec.value == center_coord.dec.value))[0][0]
    for sky, pix, i in zip(pa_sky, pa_pixel, range(len(offset))):
        if i != same_ind:
            if sky > pix:
                north_offset = sky - pix
            else:
                north_offset = sky - pix + 360 
            offset[i] = north_offset

    # get rid of the comparison w self
    offset = np.delete(offset, same_ind)
    north_angle = np.mean(offset)
    
    return north_angle

def astrometric_calibration(input_dataset, guesses):
    """
    Perform the boresight calibration of a dataset.
    
    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images
        guesses (str): path to file with x,y [pixel] locations of sources AND RA,DEC [deg] true source positions
        center_coord (str): path to RA,DEC coordinate of target source 
        
    Returns:
        corgidrp.data.AstrometricCalibration: the astrometric calibration measurements
        
    """
    if columns != guesses.columns:
        raise TypeError('guesses must have column names {\'x\',\'y\',\'RA\',\'DEC\'}')
    else:
        guesses = ascii.read(guesses)

    dataset = input_dataset.copy()
    x_pixels, y_pixels = guesses['x'], guesses['y']
    ra_coords, dec_coords = guesses['RA'], guesses['DEC']

    target_guess = ()
    boresight = compute_boresight(dataset.data, target_guess=target_guess)

    platescale = compute_platescale()

    northangle = compute_northangle()

    astrom_data = np.array([boresight[0], boresight[1], platescale, northangle])
    astrom_cal = corgidrp.data.AstrometricCalibration(astrom_data)


    return astrom_cal