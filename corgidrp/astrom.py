import numpy as np
import os
import warnings

import corgidrp.data
import corgidrp.check as check

import astropy
import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord

import pyklip.fakes as fakes

import scipy.ndimage as ndi
import scipy.optimize as optimize
from scipy.optimize import OptimizeWarning

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


def centroid_with_roi(frame, roi_radius=5, centering_initial_guess=None):
    """
    Finds the centroid in a sub-region around a given initial guess (or the brightest pixel if no guess is provided).

    Args:
        frame (np.ndarray): 2D array to compute centering.
        roi_radius (int or float): Half-size of the box around the initial guess or brightest pixel.
        centering_initial_guess (tuple or None, optional): (x_init, y_init) as initial guess for centroiding.
                                                           If None, defaults to the brightest pixel.

    Returns:
        tuple:
            xcen (float): X centroid coordinate.
            ycen (float): Y centroid coordinate.
    """

    # 1) Unpack initial guess or fall back to brightest pixel
    if centering_initial_guess is not None and None not in centering_initial_guess:
        peak_x, peak_y = int(round(centering_initial_guess[0])), int(round(centering_initial_guess[1]))
    else:
        peak_y, peak_x = np.unravel_index(np.nanargmax(frame), frame.shape)

    # 2) Define the subarray (region of interest) around the peak
    y_min = max(0, peak_y - roi_radius)
    y_max = min(frame.shape[0], peak_y + roi_radius + 1)
    x_min = max(0, peak_x - roi_radius)
    x_max = min(frame.shape[1], peak_x + roi_radius + 1)
    sub_frame = frame[y_min:y_max, x_min:x_max]

    # 3) Create index arrays for sub_frame, offset so they match the full-frame coords
    y_indices, x_indices = np.indices(sub_frame.shape)
    y_indices += y_min
    x_indices += x_min

    # 4) Compute flux-weighted centroid in the subarray
    total_flux = np.sum(sub_frame)
    if total_flux == 0:
        # Edge case: empty or zero frame
        return peak_x, peak_y

    xcen = np.sum(x_indices * sub_frame) / total_flux
    ycen = np.sum(y_indices * sub_frame) / total_flux

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
        fit_errs (np.array): Array of [x,y] fitting errors
    """
    #### Centroid on location of star ###
    yind = int(ystar_guess)
    xind = int(xstar_guess)
        
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning) 
        popt, pcov = optimize.curve_fit(shift_psf, stamp, data.ravel(), p0=(0,0,guessflux), maxfev=2000)
    tinyoffsets = popt[0:2]
    fit_errs = np.sqrt([pcov[0,0], pcov[1,1], pcov[2,2]])

    binary_offset = [xoffset_guess - tinyoffsets[0], yoffset_guess - tinyoffsets[1]]

    return binary_offset, fit_errs

def compute_combinations(iteration, r=2):
    """ 
    Rough equivalivent to itertools.combinations function to create all r-length combinations from a given array.

    Args:
        iteration (np.array): Array from which to create combinations of elements
        r (int): (Optional) Length of combinations (default: 2)

    """
    inds = np.indices(np.shape(iteration))
    pool = tuple(inds[0])
    n = len(pool)
    if r > n:
        return
    
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

def angle_between(pos1, pos2):
    """ 
    Used to find the angle from North counterclockwise between two sources in an image.
    
    Args:
        pos1 (tuple): Position of the first target
        pos2 (tuple): Position of the second target
    
    Returns:
        angle (float): Angle [deg] between two sources, from north going counterclockwise
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


def get_polar_dist(seppa1,seppa2):
    """Computes the linear distance between two points in polar coordinates.

    Args:
        seppa1 (tuple): Separation (in any units) and position angle (in degrees) of the first point.
        seppa2 (tuple): Separation (in same units as above) and position angle (in degrees) of the second point.

    Returns:
        float: Distance between the two points in the input separation units.
    """
    sep1, pa1 = seppa1
    sep2, pa2 = seppa2

    return np.sqrt(sep1**2 + sep2**2 - (2 * sep1 * sep2 * np.cos((pa1-pa2)*np.pi/180.)))


def seppa2dxdy(sep_pix,pa_deg):
    """Converts position in separation (pixels from some reference center) and position angle 
    (counterclockwise from north) to separation in x and y pixels from the center.

    Args:
        sep_pix (float or np.array): Separation in pixels
        pa_deg (float or np.array): Position angle in degrees (counterclockwise from North)

    Returns:
        np.array: array of shape (2,) containing delta x and delta y in pixels from the center 
    """
    dx = -sep_pix * np.sin(pa_deg * np.pi/180.)
    dy = sep_pix * np.cos(pa_deg * np.pi/180.)

    return np.array([dx, dy])


def seppa2xy(sep_pix,pa_deg,cenx,ceny):
    """Converts position in separation (pixels from some reference center) and position angle 
    (counterclockwise from north) to separation in x and y pixels from the center.

    Args:
        sep_pix (float or np.array): Separation in pixels
        pa_deg (float or np.array): Position angle in degrees (counterclockwise from North)
        cenx (float): X location of center reference pixel. (0,0) is center of bottom left pixel
        ceny (float): Y location of center reference pixel. (0,0) is center of bottom left pixel

    Returns:
        np.array: x and y pixel location. (0,0) is center of bottom left pixel.
    """
    dx, dy = seppa2dxdy(sep_pix,pa_deg)

    x = dx + cenx
    y = dy + ceny

    return np.array([x, y])


def find_source_locations(image_data, threshold=10, fwhm=7, mask_rad=1):
    ''' 
    Used to find to [pixel, pixel] locations of the sources in an image
    
    Args:
        image_data (numpy.ndarray): 2D array of image data
        threshold (int): Number of stars to find (default: 100)
        fwhm (float): Full width at half maximum of the stellar psf (default: 7, ~fwhm for a normal distribution with sigma=3)
        mask_rad (int): Radius of mask for stars [in fwhm] (default: 1)
    
    Returns:
        sources (astropy.table.Table): Astropy table with columns 'x', 'y' as pixel locations
    
    '''
    # create a place to store the location arrays and use a copy of the input image
    image = np.copy(image_data)
    image_shape = np.shape(image)
    xs = np.empty(threshold) * np.nan
    ys = np.empty(threshold) * np.nan
    
    fwhm = fwhm * mask_rad
        
    i = 0
    while i < threshold:
        ind = np.where(image == np.nanmax(image))
        if len(ind) > 2:
            ind = ind[0:2]

        # record the location of the star
        image_y, image_x = np.shape(image)
        x = ind[1][0]
        y = ind[0][0]
        
        if i > 2:
            if (ys[i-1] == y and xs[i-1] == x):
                break
            
        ys[i] = y
        xs[i] = x
    
        # mask out the image at this location
        if (x - fwhm) < 0:  # left edge
            startx = 0
            endx = x + int(fwhm) + 1
            
            if (y - fwhm) < 0:  # bottom left corner
                starty = 0
                endy = y + int(fwhm) + 1
                
            elif (y + fwhm) > (image_y - 1):  # upper left corner
                starty = y - int(fwhm)
                endy = image_y

            else:
                starty = y - int(fwhm)
                endy = y + int(fwhm) + 1
        
        elif (x + fwhm) > (image_x - 1):  # right edge
            startx = x - int(fwhm)
            endx = image_x
            
            if (y - fwhm) < 0:  # bottom right corner
                starty = 0
                endy = y + int(fwhm) + 1
                
            elif (y + fwhm) > (image_y - 1):  # upper right corner
                starty = y - int(fwhm)
                endy = image_y

            else:
                starty = y - int(fwhm)
                endy = y + int(fwhm) + 1

        else:
            startx = x - int(fwhm)
            endx = x + int(fwhm) + 1
            
            if (y - fwhm) < 0:  # bottom
                starty = 0
                endy = y + int(fwhm) + 1
                
            elif (y + fwhm) > (image_y - 1):  # upper
                starty = y - int(fwhm)
                endy = image_y

            else:
                starty = y - int(fwhm)
                endy = y + int(fwhm) + 1
        
        stamp =  image[starty: endy, startx: endx]

        image[starty: endy, startx: endx] = np.zeros(np.shape(stamp))
        
        i += 1        
    # record the locations in an astropy.table
    sources = astropy.table.Table()
    sources['x'] = xs[~np.isnan(xs)]
    sources['y'] = ys[~np.isnan(ys)]

    fit_xs = np.empty(len(xs[~np.isnan(xs)]))
    fit_ys = np.empty(len(ys[~np.isnan(ys)]))

    # fit a gaussian to the guess to find true pixel pos
    # pad the image to fit stars along the edge

    pad = 25
    full_frame = np.zeros([image_shape[1]+(pad*2), image_shape[0]+(pad*2)])
    full_frame[pad:-pad, pad:-pad] = image_data
    fit_gauss_image = np.copy(full_frame)

    for i, (gx, gy) in enumerate(zip(xs[~np.isnan(xs)], ys[~np.isnan(ys)])):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pf, fw, x, y = fakes.gaussfit2d(frame= fit_gauss_image, xguess= gx+pad, yguess=gy+pad)
        fit_xs[i] = x - pad
        fit_ys[i] = y - pad

    found_sources = astropy.table.Table()
    found_sources['x'] = fit_xs
    found_sources['y'] = fit_ys
    
    return found_sources

def match_sources(image, sources, field_path, comparison_threshold=50, rad=0.012, platescale_guess=21.8, platescale_tol=0.1):
    ''' 
    Function to find the corresponding RA/Dec positions to image sources, given a particular field.

    Args:
        image (corgidrp.data.Image): Image data as a corgidrp Image object
        sources (astropy.table.Table): Astropy table with columns 'x', 'y' as pixel locations of sources to match
        field_path (str): Full path to directory with search field data (ra, dec, vmag, etc.)
        comparison_threshold (int): How many stars in the field to consider for the initial match
        rad (float): The radius [deg] around the target coordinate for creating a subfield to match image sources to
        platescale_guess (float): An initial guess for the platescale value (default: 21.8 [mas/ pixel])
        platescale_tol (float): A tolerance for finding source matches within a fraction of the initial plate scale guess (default: 0.5)

    Returns:
        matched_sources (astropy.table.Table): Astropy table with columns 'x','y','RA', 'DEC' as pixel locations and corresponding sky positons
        
    '''
    # ensure the search field data has the proper column names
    field = ascii.read(field_path)
    if 'RA' and 'DEC' and 'VMAG' not in field.colnames:
        raise ValueError('field data must have column names [\'RA\',\'DEC\', \'VMAG\']')

    # gather the pixel locations for the (3) brightest sources in the image
    source1, source2, source3 = sources[0], sources[1], sources[2]

    # define the side length to perimeter ratio for the triangle made from these sources [pixels]
    l1, l2, l3 = np.sqrt(np.power(source1['x'] - source2['x'], 2) + np.power(source1['y'] - source2['y'], 2)), np.sqrt(np.power(source2['x'] - source3['x'], 2) + np.power(source2['y'] - source3['y'], 2)), np.sqrt(np.power(source3['x'] - source1['x'], 2) + np.power(source3['y'] - source1['y'], 2))
    perimeter = l1 + l2 + l3

    # the shortest to longest sides get reordered to l1, l2, l3
    l1, l2, l3 = np.sort([l1, l2, l3])

    a, b, c = l1/perimeter, l2/perimeter, l3/perimeter

    # define a search field and load in RA, DEC, Vmag
    field = ascii.read(field_path)
    target = image.pri_hdr['RA'], image.pri_hdr['DEC']
    
    ymid, xmid = image.data.shape   # fit gaussian to find target x,y location (assuming near center)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pf, fw, targetx, targety = fakes.gaussfit2d(frame= image.data, xguess= (xmid-1)//2, yguess= (ymid-1)//2)

    target_skycoord = SkyCoord(ra= target[0], dec= target[1], unit='deg')
    subfield = field[((field['RA'] >= target[0] - rad) & (field['RA'] <= target[0] + rad) & (field['DEC'] >= target[1] - rad) & (field['DEC'] <= target[1] + rad))]

    bright_order_subfield = subfield.copy()
    bright_order_subfield.sort(keys='VMAG')
    brightest_field = bright_order_subfield[:comparison_threshold]

    # create all combinations of triangles with the brightest field sources
    combos = list(compute_combinations(range(len(brightest_field)), r=3))

    #a_prime, b_prime, c_prime = np.empty(len(combos)), np.empty(len(combos)), np.empty(len(combos))
    skycoords = SkyCoord(ra= brightest_field['RA'], dec= brightest_field['DEC'], unit='deg')
    field_side_lengths = np.empty((len(combos), 3))
    best_sky_ind = np.nan
    smallest_lsq = 1e10
    best_ind = np.nan

    for i, ind in enumerate(combos):
        j, k, l = ind
        s1, s2, s3 = skycoords[j], skycoords[k], skycoords[l]

        len1, len2, len3 = np.sort([s1.separation(s2).mas, s2.separation(s3).mas, s3.separation(s1).mas])

        field_side_lengths[i] = np.array([len1, len2, len3])
        perimeter = len1 + len2 + len3
        ap, bp, cp = len1/perimeter, len2/perimeter, len3/perimeter

        # make sure plate scale is within tolerance of the guess or else discard this possibility
        if ((len1 / l1) > platescale_guess* (1 + platescale_tol)) or ((len2 / l2) > platescale_guess* (1 + platescale_tol)) or ((len3 / l3) > platescale_guess* (1 + platescale_tol)):
            ap, bp, cp = 0, 0, 0
        if ((len1 / l1) < platescale_guess* (1 - platescale_tol)) or ((len2 / l2) < platescale_guess* (1 - platescale_tol)) or ((len3 / l3) < platescale_guess* (1 - platescale_tol)):
            ap, bp, cp = 0, 0, 0

        # find the best fit to the brightest image triangle
        lstsq = (a - ap)**2 + (b - bp)**2 + (c - cp)**2
        if lstsq < smallest_lsq:
            smallest_lsq = lstsq
            best_ind = i
            best_sky_ind = ind

    # now use the side length to separations with best fit triangle to define a pseudo plate scale
    best_l1, best_l2, best_l3 = field_side_lengths[best_ind]
    initial_platescale = np.mean(np.array([best_l1 / l1, best_l2 / l2, best_l3 / l3]))  # [deg/mas]

    # find pseudo north angle from difference in triangle rotations from the target value
    # using found target pixel
  
    # rot_image = np.array([angle_between(((xmid-1) //2, (ymid-1) //2), (s['x'], s['y'])) for s in [source1, source2, source3]])
    rot_image = np.array([angle_between((targetx, targety), (s['x'], s['y'])) for s in [source1, source2, source3]])
    rot_field = np.array([target_skycoord.position_angle(t).deg for t in skycoords[[best_sky_ind]]])

    initial_northangle = np.abs(np.mean(rot_field - rot_image))

    # make a new image header with the pseudo platescale and north angle to find matchings
    # allow for some error window and assign the closest star to each source
    vert_ang = np.radians(initial_northangle)
    pc = np.array([[-np.cos(vert_ang), np.sin(vert_ang)], [np.sin(vert_ang), np.cos(vert_ang)]])
    cdmatrix = pc * (initial_platescale * 0.001) / 3600.

    new_hdr = {}
    new_hdr['CD1_1'] = cdmatrix[0,0]
    new_hdr['CD1_2'] = cdmatrix[0,1]
    new_hdr['CD2_1'] = cdmatrix[1,0]
    new_hdr['CD2_2'] = cdmatrix[1,1]
    # new_hdr['CRPIX1'] = (np.shape(image.data)[1]-1) // 2
    # new_hdr['CRPIX2'] = (np.shape(image.data)[0]-1) // 2
    new_hdr['CRPIX1'] = targetx
    new_hdr['CRPIX2'] = targety
    new_hdr['CTYPE1'] = 'RA---TAN'
    new_hdr['CTYPE2'] = 'DEC--TAN'
    new_hdr['CDELT1'] = (initial_platescale * 0.001) / 3600.
    new_hdr['CDELT2'] = (initial_platescale * 0.001) / 3600.
    new_hdr['CRVAL1'] = target[0]
    new_hdr['CRVAL2'] = target[1]
    w = astropy.wcs.WCS(new_hdr)

    # transform the subfield skycoords to pixel locations
    subfield_skycoords = SkyCoord(ra= subfield['RA'], dec= subfield['DEC'], unit='deg')
    x_sky_to_pix, y_sky_to_pix = astropy.wcs.utils.skycoord_to_pixel(subfield_skycoords, wcs=w)
    # restrict to only the sources that fall within 1024 x 1024 pixels
    image_inds = ((x_sky_to_pix >= 0) & (x_sky_to_pix <= 1024) & (y_sky_to_pix >= 0) & (y_sky_to_pix <= 1024))
    x_predict = x_sky_to_pix[image_inds]
    y_predict = y_sky_to_pix[image_inds]
    subfield_skycoords_in_image = subfield_skycoords[image_inds]

    # for each source in the image, find the closest x, y predicted position and record the corresponding RA, DEC in the table
    matched_ra, matched_dec = np.empty(len(sources)), np.empty(len(sources))

    for i, (x, y)in enumerate(zip(sources['x'], sources['y'])):
        lst = 1e6
        i_match = np.nan
        for ii, (xp, yp) in enumerate(zip(x_predict, y_predict)):
            sq = (x - xp)**2 + (y - yp)**2
            if sq < lst:
                lst = sq
                i_match = ii
        matched_ra[i] = subfield_skycoords_in_image[i_match].ra.value
        matched_dec[i] = subfield_skycoords_in_image[i_match].dec.value
    
    matched_image_to_field = astropy.table.Table()
    matched_image_to_field['x'] = sources['x']
    matched_image_to_field['y'] = sources['y']
    matched_image_to_field['RA'] = matched_ra
    matched_image_to_field['DEC'] = matched_dec

    # append each x,y,RA,DEC string to the fits ext_hdr to save the source matches for reference
    for source in np.arange(len(matched_image_to_field)):
        key = 'star' + str(source + 1)

        string = str(matched_image_to_field[source]['x'])
        for col in ['y','RA','DEC']:
            string += ',' + str(matched_image_to_field[source][col])

        if key not in image.ext_hdr:
            image.ext_hdr[key] = string


    return matched_image_to_field

def fit_distortion_solution(params, fitorder, platescale, rotangle, pos1, meas_offset, sky_offset, meas_errs):
    '''
    Cost function used to fit the legendre polynomials for distortion mapping.

    Args:
        params (list): List of the x and y legendre polynomial coefficients
        fitorder (int): The degree of legendre polynomial being used
        platescale (float): The platescale of the image
        rotangle (float): The north angle of the image
        pos1 (np.array): A (2 x N) array of (x, y) pixel positions for the first star in N pairs
        meas_offset (np.array): A (2 x N) array of (x, y) pixel offset from the first star position for N star pairs
        sky_offset (np.array): A (2 x N) array of (sep, pa) true sky offsets in [mas] and [deg] from the first star position for N pairs 
        meas_errs (np.array): A (2 x N) array of (x, y) pixel errors in measured offsets from the first star position for N pairs

    Returns:
        residuals (list): List of residuals between true and measured star positions
    '''
    
    fitparams = (fitorder + 1)**2
    
    leg_params_x = np.array(params[:fitparams])  # the first half of params are for x fitting
    leg_params_x = leg_params_x.reshape(fitorder+1, fitorder+1)

    leg_params_y = np.array(params[fitparams:]) # the last half are for y fitting
    leg_params_y = leg_params_y.reshape(fitorder+1, fitorder+1)

    total_orders = np.arange(fitorder+1)[:,None] + np.arange(fitorder+1)[None,:]  # creating a 4 x 4 matrix of order numbers (?)

    leg_params_x = leg_params_x / 500**(total_orders)  # making the coefficients sufficiently large for fitting (or else ~0)
    leg_params_y = leg_params_y / 500**(total_orders)

    residuals = []

    binary_offsets = np.copy(meas_offset).T
    star1_pos = np.copy(pos1).T
    # center to center of detector
    star1_pos[:,0] -= 511.        # because leg coeffs are defined around (0,0)
    star1_pos[:,1] -= 511.        # make a new param in the function to pass in detector shape?

    # derive star2 position
    star2_pos = star1_pos + binary_offsets

    # undistort x and y for both star positions
    star1_pos_corr = np.copy(star1_pos)
    star1_pos_corr[:,0] = np.polynomial.legendre.legval2d(star1_pos[:,0], star1_pos[:,1], leg_params_x)
    star1_pos_corr[:,1] = np.polynomial.legendre.legval2d(star1_pos[:,0], star1_pos[:,1], leg_params_y)
    star2_pos_corr = np.copy(star2_pos)
    star2_pos_corr[:,0] = np.polynomial.legendre.legval2d(star2_pos[:,0], star2_pos[:,1], leg_params_x)
    star2_pos_corr[:,1] = np.polynomial.legendre.legval2d(star2_pos[:,0], star2_pos[:,1], leg_params_y)

    # translate offsets from [mas] sep, pa to [pixel] sep, pa
    sky_sep = sky_offset[0]
    # this cant be negative by definition, but is defined from north to east while corr_pa starts at (1,0) 
    sky_pa = sky_offset[1]
    
    true_offset_sep = sky_sep / platescale
    true_offset_pa = sky_pa - rotangle  # this is in degrees
    
    # translate star_pos_corr from x, y to sep, pa
    corr_offset_x = star2_pos_corr[:,0] - star1_pos_corr[:,0] 
    corr_offset_y = star2_pos_corr[:,1] - star1_pos_corr[:,1]
    
    corr_offset_sep = np.sqrt(corr_offset_x**2 + corr_offset_y**2)
    corr_offset_pa = np.degrees(np.arctan2(-corr_offset_x, corr_offset_y))
    # corr_offset_pa = np.degrees(np.arctan2(corr_offset_y, corr_offset_x))  # this is in degrees

    res_sep = corr_offset_sep - true_offset_sep  # this is in pixels
    
    res_pa_arctan_num = np.sin(np.radians(corr_offset_pa - true_offset_pa)) 
    res_pa_arctan_denom = np.cos(np.radians(corr_offset_pa - true_offset_pa))
    res_pa = np.arctan(res_pa_arctan_num / res_pa_arctan_denom) # this is in radians

    # translate pixel error in measurement to sep pa errs
    # sep_err = np.sqrt(meas_errs[0]**2 + meas_errs[1]**2) # this is in 
    sep_err = np.mean([meas_errs[0], meas_errs[1]], axis=0)
    # should be the mean of two errors also 

    ## can assume equal errors in cartesian 

    # pa_err = np.arctan2(-meas_errs[0],  meas_errs[1]) # this is in radians
    # pa_err = np.arctan2(meas_errs[1],  meas_errs[0]) # this is in radians
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pa_err = sep_err / true_offset_sep
    # should be avg of errors divided by sep --> pa err is delta arclength ~~ np.avg(m[1], m[0]) /sep
    # or fit x y res instead of sep pa

    res_sep /= sep_err
    res_pa /= pa_err
    ## translate this back to pixels x/y
    res_x = np.cos(res_pa) * res_sep # this is in pixels now
    res_y = np.sin(res_pa) * res_sep # this is in pixels now
    ## have to compute x y first and then subtract for actual res_x, res_y

    residuals = np.append(residuals, np.array([res_sep, res_pa]).ravel())

    return residuals

def compute_platescale_and_northangle(image, source_info, center_coord, center_radius=0.9):
    """
    Used to find the platescale and north angle of the image. Calculates the platescale for each pair of stars in the image
    and returns the averged platescale. Calculates the north angle for pairs of stars with the center target
    and returns the averged north angle.
    
    Args:
        image (numpy.ndarray): 2D array of image data 
        source_info (astropy.table.Table): Estimated pixel positions of sources and true sky positions, must have column names 'x', 'y', 'RA', 'DEC'
        center_coord (tuple):
            (float): RA coordinate of the target pointing
            (float): Dec coordinate of the target pointing
        center_radius (float): Percent of the image radius used to crop the image and compute plate scale and north angle from (default: 1 -- ie: the full image is used)

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

    # translate the center_coord param into a skycoord
    if type(center_coord) != tuple:
        raise TypeError('center_coord must be a tuple coordinate (RA,DEC)')
    else:
        center_coord = SkyCoord(ra = center_coord[0], dec= center_coord[1], unit='deg', frame='icrs')

    # use only center quadrant
    imageshape = np.shape(image)
    cut = 1 - center_radius
    suby, subx = imageshape[0] * cut, imageshape[1] * cut
    center_source_inds = np.where((guesses['x'] >= subx) & (guesses['x'] <= imageshape[1] - subx) & (guesses['y'] >= suby) & (guesses['y'] <= imageshape[0] - suby))
    sub_guesses = guesses[center_source_inds]
    sub_skycoords = skycoords[center_source_inds]

    # Platescale calculation
    # create random combinations of stars
    all_combinations = list(compute_combinations(sub_guesses))
    if len(all_combinations) > 200:
        rand_inds = np.random.randint(low=0, high=len(all_combinations), size=200)
        combo_list = np.array(all_combinations)[rand_inds]
    else:
        combo_list = np.array(all_combinations)

    # gather the skycoord separations for all combinations
    seps = np.empty(len(combo_list))
    for i,c in enumerate(combo_list):
        star1 = sub_skycoords[c[0]]
        star2 = sub_skycoords[c[1]]

        sep = star1.separation(star2).mas
        seps[i] = sep

    # find the separations in pixel space on the image between all combinations
    pixseps = np.empty(len(combo_list))
    for i,c in enumerate(combo_list):
        star1 = sub_guesses[c[0]]
        star2 = sub_guesses[c[1]]

        xguess = star2['x'] - star1['x']
        yguess = star2['y'] - star1['y']
        
        (xoff, yoff), _ = measure_offset(image, xstar_guess=star1['x'], ystar_guess=star1['y'], xoffset_guess= xguess, yoffset_guess= yguess)

        pixsep = np.sqrt(np.power(xoff,2) + np.power(yoff,2))
        pixseps[i] = pixsep

    # estimate the platescale from each combination and find the mean
    platescales = seps / pixseps
    platescale = np.mean(platescales)

    # North angle calculation
    # find the true centerings of the sources in the image from the guesses and save into a table
    xs = np.empty(len(sub_guesses))
    ys = np.empty(len(sub_guesses))
    for i, (gx, gy) in enumerate(zip(sub_guesses['x'], sub_guesses['y'])):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pf, fw, x, y = fakes.gaussfit2d(frame= image, xguess= gx, yguess=gy)
        xs[i] = x
        ys[i] = y

    sources = astropy.table.Table()
    sources['x'] = xs
    sources['y'] = ys

    # find the sky position angles between the center star and all others
    pa_sky = np.empty(len(sub_skycoords))
    for i, star in enumerate(sub_skycoords):
        pa = center_coord.position_angle(star).deg
        pa_sky[i] = pa

    # find the pixel position angles
    pa_pixel = np.empty(len(sub_guesses))
    for i, (x, y) in enumerate(zip(xs, ys)):
        pa = angle_between(((np.shape(image)[0]-1)//2, (np.shape(image)[1]-1)//2), (x,y))
        pa_pixel[i] = pa

    # find the difference between the measured and true positon angles
    offset = np.empty(len(sub_guesses))
    # locate a potential comparison with self
    if len(np.where((sub_skycoords.ra.value == center_coord.ra.value) & (sub_skycoords.dec.value == center_coord.dec.value))[0]) > 0:
        same_ind = np.where((sub_skycoords.ra.value == center_coord.ra.value) & (sub_skycoords.dec.value == center_coord.dec.value))[0][0]
    else:
        same_ind = None

    for i, (sky, pix) in enumerate(zip(pa_sky, pa_pixel)):
        if i != same_ind:
            numerator = np.sin(np.radians(sky - pix))
            denominator = np.cos(np.radians(sky - pix))
            north_offset = np.degrees(np.arctan(numerator / denominator))
            # if sky > pix:
            #     north_offset = sky - pix
            # else:
            #     north_offset = sky - pix + 360 
            offset[i] = north_offset

    # get rid of the comparison with self if it exists
    if same_ind != None:
        offset = np.delete(offset, same_ind)

    # use the median to avoid bias
    north_angle = np.mean(offset)
    
    return platescale, north_angle

def compute_boresight(image, source_info, target_coordinate, cal_properties):
    """ 
    Used to find the offset between the target and the center of the image.

    Args:
        image (numpy.ndarray): 2D array of image data
        source_info (astropy.table.Table): Estimated pixel positions of sources and true sky positions, must have column names 'x', 'y', 'RA', 'DEC'
        target_coordinate (tuple): 
            (float): RA coordinate of the target pointing
            (float): DEC coordinate of the target pointing
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            p, f, xi_center, yi_center = fakes.gaussfit2d(frame= image, xguess= xg, yguess= yg, searchrad=searchrad)
        x_off = xi_center - x_predict[i]
        y_off = yi_center - y_predict[i]
        boresights[i,:] = [x_off, y_off]
        image_centerings[i,:] = [xi_center, yi_center]

    # average all offsets in x,y directions [pix]
    boresight_x, boresight_y = np.mean(boresights[:,0]), np.mean(boresights[:,1])

    # convert back to corrected RA, DEC of target
    # image_center_RA = target_coordinate[0] - ((boresight_x * cal_properties[0]) * astropy.units.mas).to(astropy.units.deg).value
    # image_center_DEC = target_coordinate[1] - ((boresight_y * cal_properties[0]) * astropy.units.mas).to(astropy.units.deg).value

    # report the offsets instead of the new RA/DEC
    boresight_ra = ((boresight_x * cal_properties[0]) * astropy.units.mas).to(astropy.units.deg).value
    boresight_dec = ((boresight_y * cal_properties[0]) * astropy.units.mas).to(astropy.units.deg).value

    return boresight_ra, boresight_dec

def format_distortion_inputs(input_dataset, source_matches, ref_star_pos, position_error=None):
    ''' Function that formats the input data for the distortion map computation * must be run before compute_distortion *
    
    Args:
        input_dataset (corgidrp.data.dataset): corgidrp dataset object with images to compute the distortion from
        source_matches (list of astropy.table.Table() objects): List of length N for N frames in the input dataset. Tables must columns 'x','y','RA','DEC' as pixel locations and corresponding sky positons
        ref_star_pos (list of astropy.table.Table() objects): List of length N for N frames. Tables must have column names 'x', 'y', 'RA', 'DEC' for the position of the reference position to compute pairs with
        position_error (NoneType or int): If int, this is the uniform error value assumed for the offset between pairs of stars in both x and y
                        Should be changed later to accept non-uniform errors
        
    Returns:
        first_stars (np.array): 2D array of the (x, y) pixel positions for the first star in every star pair
        offsets (np.array): 2D array of the (delta_x, delta_y) values for each star from the first star position
        true_offsets (np.array): 2D array of the (delta_ra, delta_dec) offsets between the matched stars in the reference field
        errs (np.array): 2D array of the (x_err, y_err) error in the measured pixel positions
    '''
    ## Create arrays to store values in
    dxs, dys = np.array([]), np.array([])
    xerrs, yerrs = np.array([]), np.array([])
    firstxs, firstys = np.array([]), np.array([])
    seps, pas = np.array([]), np.array([])

    ## Loop over every frame in the dataset
    for frame_ind in range(len(input_dataset)):
        input_image = input_dataset[frame_ind].data

        # create all combinations of the target star with all others
        combo_list = range(len(source_matches[frame_ind]))
        skycoords = SkyCoord(ra= source_matches[frame_ind]['RA'], dec= source_matches[frame_ind]['DEC'], unit='deg', frame='icrs')
        target_coord = SkyCoord(ra= ref_star_pos[frame_ind]['RA'], dec= ref_star_pos[frame_ind]['DEC'], unit='deg', frame='icrs')
    
        for pair_ind in combo_list:
            # get the pixel offset
            star1 = ref_star_pos[frame_ind]
            star2 = source_matches[frame_ind][pair_ind]
    
            x_guess = star2['x'] - star1['x']
            y_guess = star2['y'] - star1['y']
        
            (dx, dy), (xfit_err, yfit_err, _) = measure_offset(input_image, star2['x'], star2['y'], x_guess, y_guess, guessflux=10000)
    
            # get the true sky offset [mas]
            true1 = target_coord
            true2 = skycoords[pair_ind]
        
            # get true sky separation and position angle
            true_sep = true1.separation(true2).mas
            true_pa = true1.position_angle(true2).deg
            
            dxs = np.append(dxs, dx)
            dys = np.append(dys, dy)
            firstxs = np.append(firstxs, star1['x'])
            firstys = np.append(firstys, star1['y'])
            seps = np.append(seps, true_sep)
            pas = np.append(pas, true_pa)
    
            if type(position_error) == type(None):
                xerrs = np.append(xerrs, xfit_err)
                yerrs = np.append(yerrs, yfit_err)
            else:
                xerrs = np.append(xerrs, position_error)
                yerrs = np.append(yerrs, position_error)
   
    # join arrays and reshape to (1, 2, N)
    offsets = np.array([dxs, dys])
    first_stars = np.array([firstxs, firstys])
    true_offsets = np.array([seps, pas])
    errs = np.array([xerrs, yerrs])

    return first_stars, offsets, true_offsets, errs

def compute_distortion(input_dataset, pos1, meas_offset, sky_offset, meas_errs, platescale, northangle, fitorder=3, initial_guess=None):
    ''' 
    Function that computes the legendre polynomial coefficients that describe the image distortion map * must run format_disotrtio_inputs() first *

    Args:
        input_dataset (corgidrp.data.Dataset): corgidrp dataset object with images to compute the distortion from
        pos1 (np.array): 2D array of the (x, y) pixel positions for the first star in every star pair
        meas_offset (np.array): 2D array of the (delta_x, delta_y) values for each star from the first star position
        sky_offset (np.array): 2D array of the (delta_ra, delta_dec) offsets between the matched stars in the reference field
        meas_errs (np.array): 2D array of the (x_err, y_err) error in the measured pixel positions
        platescale (float): Platescale value to use in computing distortion
        northangle (float): Northangle value to use in computing distortion 
        fitorder (int): The order of legendre polynomial to fit to the image distortion (default: 3)
        initial_guess (np.array): Initial guess of fitting parameters (legendre coefficients) length based on fitorder (2 * (fitorder+1)**2), (default: None)

    Returns:
        distortion_coeffs (tuple): The legendre coefficients (np.array) and polynomial order used for the fit (int)

    '''

    ## SET FITTING PARAMS
    # assume all images in dataset have the same shape
    input_image = input_dataset[0].data
    x0 = np.shape(input_image)[1] // 2
    y0 = np.shape(input_image)[0] // 2
    
    # define fitting params            
    fitparams = (fitorder + 1)**2
    
    # initial guesses for the legendre coeffs if none are passed
    if initial_guess is None:
        initial_guess = [0 for _ in range(fitorder+1)] + [500,] + [0 for _ in range(fitparams - fitorder - 2)] + [0,500] + [0 for _ in range(fitparams-2)]
    
    ## OPTIMIZE 
    # first_stars_, offsets_, true_offsets_, errs_ = first_stars, offsets, true_offsets, errs
    (distortion_coeffs, _) = optimize.leastsq(fit_distortion_solution, initial_guess, 
                                              args=(fitorder, platescale, 
                                                northangle, pos1, meas_offset, 
                                                sky_offset, meas_errs))

    return (distortion_coeffs, fitorder)
  
  
def boresight_calibration(input_dataset, field_path='JWST_CALFIELD2020.csv', field_matches=None, find_threshold=10, fwhm=7, mask_rad=1, 
                          comparison_threshold=50, search_rad=0.012, platescale_guess=21.8, platescale_tol=0.1, center_radius=0.9, 
                          frames_to_combine=None, find_distortion=False, fitorder=3, position_error=None, initial_dist_guess=None):
    """
    Perform the boresight calibration of a dataset.
    
    Args:
        input_dataset (corgidrp.data.Dataset): Dataset containing a images for astrometric calibration
        field_path (str): Full path to file with search field data (ra, dec, vmag, etc.) (default: 'JWST_CALFIELD2020.csv')
        field_matches (list of str or astropy.table.Table): List of full paths to files or astropy tables with calibration field matches for each image in the dataset (x, y, ra, dec), if single str the same filepath used for all frames,nif None, automated source matching is used (default: None)
        find_threshold (int): Number of stars to find (default 10)
        fwhm (float): Full width at half maximum of the stellar psf (default: 7, ~fwhm for a normal distribution with sigma=3)
        mask_rad (int): Radius of mask for stars [in fwhm] (default: 1)
        comparison_threshold (int): How many stars in the field to consider for the initial match (default: 50)
        search_rad (float): The radius [deg] around the target coordinate for creating a subfield to match image sources to
        platescale_guess (float): An initial guess for the platescale value (default: 21.8 [mas/ pixel])
        platescale_tol (float): A tolerance for finding source matches within a fraction of the initial plate scale guess (default: 0.1)
        center_radius (float): Percent of the image to compute plate scale and north angle from, centered around the image center (default: 0.9 -- ie: 90% of the image is used)
        frames_to_combine (int): The number of frames to combine in a dataset (default: None)
        find_distortion (boolean): Used to determine if distortion map coeffs will be computed (default: False)
        fitorder (int): The order of legendre polynomials used to fit the distortion map (default: 3)
        position_error (NoneType or int): If int, this is the uniform error value assumed for the offset between pairs of stars in both x and y
        initial_dist_guess (np.array): An initial guess of legendre coefficients used for fitting distortion, if None will use coeffs associated with no distortion (default: None)

    Returns:
        corgidrp.data.AstrometricCalibration: Astrometric Calibration data object containing image center coords in (RA,DEC), platescale, and north angle
        
    """
    # load in the data considering multiple frames in the data
    dataset = input_dataset.copy()

    # load in the source matches if automated source finder is not being used
    matched_sources_multiframe = []
    if field_matches is not None:
        if len(field_matches) == 1: # single str or astropy.table case
            for i in range(len(dataset)):
                if type(field_matches[0]) == str:
                    matched_sources = ascii.read(field_matches[0])
                    matched_sources_multiframe.append(matched_sources)
                else:
                    matched_sources_multiframe.append(field_matches[0])
        # elif len(field_matches) == len(dataset): # this needs to be if the len(field_matches >1)
        elif len(field_matches) > 1:  # unique matches for each frame case
            if len(field_matches) != len(dataset):
                raise TypeError('field_matches must be a single str/ astropy.table OR the same length as input_dataset')
            else:
                for i in range(len(field_matches)):
                    if type(field_matches[0]) == str:
                        matched_sources = ascii.read(field_matches[i])
                        matched_sources_multiframe.append(matched_sources)
                    else:
                        matched_sources_multiframe = field_matches
        # else:
        #     raise TypeError('field_matches must be a single str or the same length as input_dataset')

    # load in field data to refer to
    if field_path == 'JWST_CALFIELD2020.csv':
        full_field_path = os.path.join(os.path.dirname(__file__), "data", field_path)
        field_path = full_field_path

    # combine data frames if requested
    if frames_to_combine is not None:
        num_frames = len(input_dataset)
        data_array = []
        for frame in range(num_frames):
            data_array.append(input_dataset[frame].data)

        image_objects = []
        count = 0
        while count < num_frames:
            count += frames_to_combine
            if count >= num_frames:
                sub_array = data_array[count - frames_to_combine:]
                file_name = input_dataset[-1].filename
                subset_indices = range(count - frames_to_combine, num_frames)
            else:
                sub_array = data_array[count - frames_to_combine: count]
                file_name = input_dataset[count].filename
                subset_indices = range(count - frames_to_combine, count)

            comb = np.mean(sub_array, axis=0)
            
            # Get subset dataset for header merging
            subset_dataset = corgidrp.data.Dataset([input_dataset[j] for j in subset_indices])
            
            # Merge headers for combined frame
            pri_hdr, ext_hdr, err_hdr, dq_hdr = check.merge_headers_for_combined_frame(subset_dataset)
            
            # Create combined image with merged headers
            im = corgidrp.data.Image(comb, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                                     err_hdr=err_hdr, dq_hdr=dq_hdr)
            im.filename = file_name
            image_objects.append(im)
        
        dataset = corgidrp.data.Dataset(image_objects)

    # create a place to store all the calibration measurements
    astroms = []
    target_coord_tables = []

    hold_matches = []   # place to hold the auto-found source matches for each frame
    corrected_positions_boresight = []      # place to hold the corrected target position based on boresight offsets for each frame

    for i in range(len(dataset)):
        in_dataset = corgidrp.data.Dataset([dataset[i]])
        image = dataset[i].data

        # call the target coordinates from the image header
        target_coordinate = (dataset[i].pri_hdr['RA'], dataset[i].pri_hdr['DEC'])
        target_coord_tab = astropy.table.Table()
        target_coord_tab['x'] = [(np.shape(image)[1]-1) // 2]    # assume the target is at the center of the image
        target_coord_tab['y'] = [(np.shape(image)[0]-1) // 2]
        target_coord_tab['RA'] = [target_coordinate[0]]
        target_coord_tab['DEC'] = [target_coordinate[1]]
        target_coord_tables.append(target_coord_tab)
   
        # compute the calibration properties
        found_sources = find_source_locations(image, threshold=find_threshold, fwhm=fwhm, mask_rad=mask_rad)
        matched_sources = match_sources(dataset[i], found_sources, field_path, comparison_threshold=comparison_threshold, rad=search_rad, platescale_guess=platescale_guess, platescale_tol=platescale_tol)
        # if len(hold_matches) < 1:
        hold_matches.append(matched_sources)

        cal_properties = compute_platescale_and_northangle(image, source_info=matched_sources, center_coord=target_coordinate, center_radius=center_radius)
        ra, dec = compute_boresight(image, source_info=matched_sources, target_coordinate=target_coordinate, cal_properties=cal_properties)
        # calculate the corrected target position based on ra, dec offsets
        corr_ra, corr_dec = target_coordinate[0] - ra, target_coordinate[1] - dec
        corrected_positions_boresight.append([corr_ra, corr_dec])

        # return a single AstrometricCalibration data file
        astrom_data = np.array([corr_ra, corr_dec, cal_properties[0], cal_properties[1], ra, dec, np.inf, np.inf])
        astrom_cal = corgidrp.data.AstrometricCalibration(astrom_data, pri_hdr=dataset[i].pri_hdr, ext_hdr=dataset[i].ext_hdr, input_dataset=in_dataset)
        # change the filename here since the astrom_cals will be averaged later and arent individually saved ('_ast_cal' will be added to filename twice otherwise)
        astrom_cal.filename = astrom_cal.filename.split("_ast_cal")[0] + '.fits'
        astroms.append(astrom_cal)

    # average the calibration properties over all frames
    avg_ra = np.mean([astro.avg_offset[0] for astro in astroms])  # this is the average ra offset [deg]
    avg_dec = np.mean([astro.avg_offset[1] for astro in astroms])
    avg_platescale = np.mean([astro.platescale for astro in astroms])
    avg_northangle = np.mean([astro.northangle for astro in astroms])

    # compute the distortion map coeffs
    if find_distortion:
        # use the found matches for distortion
        first_stars, offsets, true_offsets, errs = format_distortion_inputs(input_dataset, source_matches=hold_matches, ref_star_pos=target_coord_tables, position_error=position_error)
        distortion_coeffs, order = compute_distortion(input_dataset, first_stars, offsets, true_offsets, errs, platescale=avg_platescale, northangle=avg_northangle, fitorder=fitorder, initial_guess=initial_dist_guess)
    else:
        # set default coeffs to produce zero distortion
        fitparams = (fitorder + 1)**2
        zero_dist = [0 for _ in range(fitorder+1)] + [500,] + [0 for _ in range(fitparams - fitorder - 2)] + [0,500] + [0 for _ in range(fitparams-2)]
        distortion_coeffs = np.array(zero_dist)
        order = fitorder

    # assume that the undithered image with original pointing position is the first frame in dataset
    corr_pos_ra, corr_pos_dec = corrected_positions_boresight[0]
    astromcal_data = np.concatenate((np.array([corr_pos_ra, corr_pos_dec, avg_platescale, avg_northangle, avg_ra, avg_dec]), np.array(distortion_coeffs), np.array([order])), axis=0)

    astroms_dataset = corgidrp.data.Dataset(astroms)
    avg_cal = corgidrp.data.AstrometricCalibration(astromcal_data, pri_hdr=input_dataset[0].pri_hdr, ext_hdr=input_dataset[0].ext_hdr, input_dataset=astroms_dataset)
    # add the corrected RA/DEC for each frame to the ext_hdr
    for i, corr in enumerate(corrected_positions_boresight):
        name = 'F'+str(i)+'POS'
        avg_cal.ext_hdr[name] = tuple(corr)
        
    # update the history
    history_msg = "Boresight calibration completed"
    astrom_cal_dataset = corgidrp.data.Dataset([avg_cal])
    astrom_cal_dataset.update_after_processing_step(history_msg)
    ## history message should be added to the input dataset(?)
    input_dataset.update_after_processing_step(history_msg)

    return avg_cal


def create_circular_mask(shape_yx, center=None, r=None):
    """Creates a circular mask

    Args:
        shape_yx (list-like of int): 
        center (list of float, optional): Center of mask. Defaults to the 
            center of the array.
        r (float, optional): radius of mask. Defaults to the minimum distance 
            from the center to the edge of the array.

    Returns:
        np.array: boolean array with True inside the circle, False outside.
    """
    shape_yx = np.array(shape_yx)
    if center is None: # use the middle of the image
        center = (shape_yx-1) / 2
    if r is None: # use the smallest distance between the center and image walls
        r = min(center[0], center[1], shape_yx[0]-center[0], shape_yx[1]-center[1])

    Y, X = np.ogrid[:shape_yx[0], :shape_yx[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= r
    return mask

def transform_coeff_to_map(distortion_coeffs, fit_order, image_shape):
    """
    Creates two, 2D maps of distortion for the X and Y directions from given legendre polynomial coefficients.
    
    Args:
        distortion_coeffs (array of float): the array of legendre polynomial coefficients that describe the distortion map in x and y directions
        fit_order (int): the order of legendre polynomial used to fit distortion
        image_shape (list-like of int): the xy pixel shape of the image
        
    Returns:
        x_dist_map (np.ndarray): a 2D array of distortion in the x direction across the image
        y_dist_map (np.ndarray): a 2D array of distortion in the y direction across the image
    
    """
        # correct indices to start at image center
    yorig, xorig = np.indices(image_shape)
    y0, x0 = image_shape[0]//2, image_shape[1]//2
    yorig -= y0
    xorig -= x0

        # get the number of fitting params from the polynomial order
    fitparams = (fit_order + 1)**2

        # reshape the coeff arrays for the given params (X)
    params_x = distortion_coeffs[:fitparams]
    params_x = params_x.reshape(fit_order+1, fit_order+1)
    total_orders = np.arange(fit_order+1)[:,None] + np.arange(fit_order+1)[None,:]
    params_x = params_x / 500**(total_orders)

        # evaluate the polynomial at all pixel positions
    x_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), params_x)
    x_corr = x_corr.reshape(xorig.shape)
    x_diff = x_corr - xorig

        # reshape the coeff arrays for the given params (Y)
        # evaluate the polynomial at all pixel positions
    params_y = distortion_coeffs[fitparams:]
    params_y = params_y.reshape(fit_order+1, fit_order+1)
    params_y = params_y / 500**total_orders

    y_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), params_y)
    y_corr = y_corr.reshape(yorig.shape)
    y_diff = y_corr - yorig

    return x_diff, y_diff