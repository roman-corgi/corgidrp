import numpy as np
import os

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
    popt, pcov = optimize.curve_fit(shift_psf, stamp, data.ravel(), p0=(0,0,guessflux), maxfev=2000)
    tinyoffsets = popt[0:2]

    binary_offset = [xoffset_guess - tinyoffsets[0], yoffset_guess - tinyoffsets[1]]

    return binary_offset

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
    
    return sources

def match_sources(image, sources, field_path, comparison_threshold=50, rad=0.0075, platescale_guess=21.8, platescale_tol=0.1):
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
        elif ((len1 / l1) < platescale_guess* (platescale_tol)) or ((len2 / l2) < platescale_guess* (platescale_tol)) or ((len3 / l3) < platescale_guess* (platescale_tol)):
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
    rot_image = np.array([angle_between((image.ext_hdr['CRPIX1'], image.ext_hdr['CRPIX2']), (s['x'], s['y'])) for s in [source1, source2, source3]])
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
    new_hdr['CRPIX1'] = np.shape(image.data)[1] // 2
    new_hdr['CRPIX2'] = np.shape(image.data)[0] // 2
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

def fit_astrom_solution(params):
    '''
    Function used to fit the legendre polynomials for distortion mapping. Cannot be used outside of compute_distortion() function where most
        hard-coded variables are defined.

    Args:
        params (list): List of the x and y legendre polynomial coefficients

    Returns:
        residuals (list): List of residuals between true and measured star positions
    '''

    platescale, rotangle = the_platescale, the_rotangle

    leg_params_x = np.array(params[:fitparams])  
    leg_params_x = leg_params_x.reshape(fitorder+1, fitorder+1)

    leg_params_y = np.array(params[fitparams:]) 
    leg_params_y = leg_params_y.reshape(fitorder+1, fitorder+1)

    total_orders = np.arange(fitorder+1)[:,None] + np.arange(fitorder+1)[None,:] 

    leg_params_x = leg_params_x / 500**(total_orders)  
    leg_params_y = leg_params_y / 500**(total_orders)

    residuals = []

    for i, (pos1, meas_offset, sky_offset, meas_errs) in enumerate(zip(first_stars, offsets, true_offsets, errs)):
 
        binary_offsets = np.array([meas_offset[0], meas_offset[1]]).T
        star1_pos = np.array([pos1[0], pos1[1]]).T
        
        # recenter to detector center
        star1_pos[:,0] -= x0
        star1_pos[:,1] -= y0

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
        sky_pa = sky_offset[1]
        
        true_offset_sep = sky_sep / platescale
        true_offset_pa = sky_pa + rotangle

        # translate star_pos_corr from x, y to sep, pa
        corr_offset_x = star2_pos_corr[:,0] - star1_pos_corr[:,0]
        corr_offset_y = star2_pos_corr[:,1] - star1_pos_corr[:,1]
        
        corr_offset_sep = np.sqrt(corr_offset_x**2 + corr_offset_y**2)
        corr_offset_pa = np.degrees(np.arctan2(-corr_offset_x, corr_offset_y))
        # ensuring the position angle is always positive
        for i, pa in enumerate(corr_offset_pa):
            if pa < 0:
                corr_offset_pa[i] += 360

        res_sep = corr_offset_sep - true_offset_sep
        res_pa = corr_offset_pa - true_offset_pa
        
        res_sep /= meas_errs[0]
        res_pa /= meas_errs[1]
        
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
            (float): RA coordinate of the target source
            (float): Dec coordinate of the target source
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
    if len(all_combinations) > 100:
        rand_inds = np.random.randint(low=0, high=len(all_combinations), size=100)
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
        
        xoff, yoff = measure_offset(image, xstar_guess=star1['x'], ystar_guess=star1['y'], xoffset_guess= xguess, yoffset_guess= yguess)

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
        pa = angle_between((np.shape(image)[0]//2, np.shape(image)[1]//2), (x,y))
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
            if sky > pix:
                north_offset = sky - pix
            else:
                north_offset = sky - pix + 360 
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

def format_distortion_inputs(input_dataset, source_matches, position_error=None):
    ''' Function that formats the input data for the distortion map computation * must be run before compute_distortion *
    
    Args:
        input_dataset (corgidrp.data.dataset): corgidrp dataset object with images to compute the distortion from
        source_matches (list of astropy.table.Table() objects): List of length N for N frames in the input dataset. Tables must columns 'x','y','RA','DEC' as pixel locations and corresponding sky positons
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

        # create all possible combinations of the given stars
        combo_list = np.array(list(compute_combinations(source_matches[frame_ind])))
        skycoords = SkyCoord(ra= source_matches[frame_ind]['RA'], dec= source_matches[frame_ind]['DEC'], unit='deg', frame='icrs')
    
        for pair_ind in combo_list:
            # get the pixel offset
            first = pair_ind[0]
            second = pair_ind[1]
            
            star1 = source_matches[frame_ind][first][['x','y']]
            star2 = source_matches[frame_ind][second][['x','y']]
    
            x_guess = star2[0] - star1[0]
            y_guess = star2[1] - star1[1]
        
            (dx, dy), (data, model, residual), (xfit_err, yfit_err, _), (x1, y1) = measure_offset(input_image, star1[0], star1[1], x_guess, y_guess, guessflux=1)
    
            # get the true sky offset [mas]
            true1 = skycoords[first]
            true2 = skycoords[second]
        
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
    offsets = np.array([[dxs, dys]])
    first_stars = np.array([[firstxs, firstys]])
    true_offsets = np.array([[seps, pas]])
    errs = np.array([[xerrs, yerrs]])

    return first_stars, offsets, true_offsets, errs

def compute_distortion(input_dataset, first_stars, offsets, true_offsets, errs, platescale, northangle, fitorder=3):
    ''' 
    Function that computes the legendre polynomial coefficients that describe the image distortion map * must run format_disotrtio_inputs() first *

    Args:
        input_dataset (corgidrp.data.Dataset): corgidrp dataset object with images to compute the distortion from
        first_stars (np.array): 2D array of the (x, y) pixel positions for the first star in every star pair
        offsets (np.array): 2D array of the (delta_x, delta_y) values for each star from the first star position
        true_offsets (np.array): 2D array of the (delta_ra, delta_dec) offsets between the matched stars in the reference field
        errs (np.array): 2D array of the (x_err, y_err) error in the measured pixel positions
        platescale (float): Platescale value to use in computing distortion
        northangle (float): Northangle value to use in computing distortion 
        fitorder (int): The order of legendre polynomial to fit to the image distortion (default: 3)

    Returns:
        distortion_coeffs (tuple): The legendre coefficients (np.array) and polynomial order used for the fit (int)

    '''

    ## SET FITTING PARAMS
    # center around image center
    # assume all images in dataset have the same shape
    input_image = input_dataset[0].data
    x0 = np.shape(input_image)[1] // 2
    y0 = np.shape(input_image)[0] // 2
    
    # define fitting params            
    fitparams = (fitorder + 1)**2
    the_platescale = platescale
    the_rotangle = northangle
    
    # initial guesses for the legendre coeffs
    guess_leg = [0 for _ in range(fitorder+1)] + [500,] + [0 for _ in range(fitparams - fitorder - 2)] + [0,500] + [0 for _ in range(fitparams-2)]
    
    ## OPTIMIZE 
    (distortion_coeffs, _) = optimize.leastsq(fit_astrom_solution, guess_leg)

    return (distortion_coeffs, fitorder)
  
  
def boresight_calibration(input_dataset, field_path='JWST_CALFIELD2020.csv', field_matches=None, find_threshold=10, fwhm=7, mask_rad=1, 
                          comparison_threshold=50, search_rad=0.0075, platescale_guess=21.8, platescale_tol=0.1, center_radius=0.9, 
                          frames_to_combine=None, find_distortion=True, fitorder=3, position_error=None):
    """
    Perform the boresight calibration of a dataset.
    
    Args:
        input_dataset (corgidrp.data.Dataset): Dataset containing a images for astrometric calibration
        field_path (str): Full path to file with search field data (ra, dec, vmag, etc.) (default: 'JWST_CALFIELD2020.csv')
        field_matches (list of str): List of full paths to files with calibration field matches for each image in the dataset (x, y, ra, dec), if None, automated source matching is used (default: None)
        find_threshold (int): Number of stars to find (default 10)
        fwhm (float): Full width at half maximum of the stellar psf (default: 7, ~fwhm for a normal distribution with sigma=3)
        mask_rad (int): Radius of mask for stars [in fwhm] (default: 1)
        comparison_threshold (int): How many stars in the field to consider for the initial match (default: 50)
        search_rad (float): The radius [deg] around the target coordinate for creating a subfield to match image sources to
        platescale_guess (float): An initial guess for the platescale value (default: 21.8 [mas/ pixel])
        platescale_tol (float): A tolerance for finding source matches within a fraction of the initial plate scale guess (default: 0.1)
        center_radius (float): Percent of the image to compute plate scale and north angle from, centered around the image center (default: 0.9 -- ie: 90% of the image is used)
        frames_to_combine (int): The number of frames to combine in a dataset (default: None)
        find_distortion (boolean): Used to determine if distortion map coeffs will be computed (default: True)
        fitorder (int): The order of legendre polynomials used to fit the distortion map (default: 3)
        position_error (NoneType or int): If int, this is the uniform error value assumed for the offset between pairs of stars in both x and y

    Returns:
        corgidrp.data.AstrometricCalibration: Astrometric Calibration data object containing image center coords in (RA,DEC), platescale, and north angle
        
    """
    # load in the data considering multiple frames in the data
    dataset = input_dataset.copy()

    # load in the source matches if automated source finder is not being used
    matched_sources_multiframe = []
    if field_matches is not None:
        for i in range(len(input_dataset)):
            matched_sources = ascii.read(field_matches[i])
            matched_sources_multiframe.append(matched_sources)
        
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
            else:
                sub_array = data_array[count - frames_to_combine: count]

            comb = np.mean(sub_array, axis=0)
            im = corgidrp.data.Image(comb, pri_hdr=input_dataset[count - frames_to_combine].pri_hdr, ext_hdr=input_dataset[0].ext_hdr)
            image_objects.append(im)
        
        dataset = corgidrp.data.Dataset(image_objects)

    # create a place to store all the calibration measurements
    astroms = []

    for i in range(len(dataset)):
        in_dataset = corgidrp.data.Dataset([dataset[i]])
        image = dataset[i].data

        # call the target coordinates from the image header
        target_coordinate = (dataset[i].pri_hdr['RA'], dataset[i].pri_hdr['DEC'])

        # run automated source finder if no matches are given
        if field_matches is None:
            found_sources = find_source_locations(image, threshold=find_threshold, fwhm=fwhm, mask_rad=mask_rad)
            matched_sources = match_sources(dataset[i], found_sources, field_path, comparison_threshold=comparison_threshold, rad=search_rad, platescale_guess=platescale_guess, platescale_tol=platescale_tol)
            matched_sources_multiframe.append(matched_sources)

        # compute the calibration properties
        cal_properties = compute_platescale_and_northangle(image, source_info=matched_sources, center_coord=target_coordinate, center_radius=center_radius)
        ra, dec = compute_boresight(image, source_info=matched_sources, target_coordinate=target_coordinate, cal_properties=cal_properties)

        # return a single AstrometricCalibration data file
        astrom_data = np.array([ra, dec, cal_properties[0], cal_properties[1]])
        astrom_cal = corgidrp.data.AstrometricCalibration(astrom_data, pri_hdr=dataset[i].pri_hdr, ext_hdr=dataset[i].ext_hdr, input_dataset=in_dataset)
        astroms.append(astrom_cal)

    # average the calibration properties over all frames
    avg_ra = np.mean([astro.boresight[0] for astro in astroms])
    avg_dec = np.mean([astro.boresight[1] for astro in astroms])
    avg_platescale = np.mean([astro.platescale for astro in astroms])
    avg_northangle = np.mean([astro.northangle for astro in astroms])

    # compute the distortion map coeffs
    if find_distortion:
        first_stars, offsets, true_offsets, errs = format_distortion_inputs(input_dataset, matched_sources_multiframe, position_error=position_error)
        distortion_coeffs = compute_distortion(input_dataset, first_stars, offsets, true_offsets, errs, platescale=avg_platescale, northangle=avg_northangle, fitorder=fitorder)
    else:
        distortion_coeffs = np.nan

    astromcal_data = [np.array([avg_ra, avg_dec]), avg_platescale, avg_northangle, distortion_coeffs]
    astroms_dataset = corgidrp.data.Dataset(astroms)
    avg_cal = corgidrp.data.AstrometricCalibration(astromcal_data, pri_hdr=input_dataset[0].pri_hdr, ext_hdr=input_dataset[0].ext_hdr, input_dataset=astroms_dataset)
        
    # update the history
    history_msg = "Boresight calibration completed"
    astrom_cal_dataset = corgidrp.data.Dataset([avg_cal])
    astrom_cal_dataset.update_after_processing_step(history_msg)
    ## history message should be added to the input dataset(?)
    input_dataset.update_after_processing_step(history_msg)

    return avg_cal