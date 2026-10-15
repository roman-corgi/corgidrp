import numpy as np
import os
import warnings

import corgidrp.data

import astropy
import astropy.io.ascii as ascii
import astropy.units as u
from astropy.coordinates import SkyCoord

import pyklip.fakes as fakes

import scipy.ndimage as ndi
import scipy.optimize as optimize
from scipy.optimize import OptimizeWarning
from scipy.ndimage import median_filter
from astropy.convolution import convolve, Gaussian2DKernel

# LMC astrometric reference catalogue shipped with corgidrp (built by build_lmc_reference_catalog)
default_field_path = os.path.join(os.path.dirname(__file__), "data", "lmc_astrometric_reference.ecsv")


def read_field_catalog(field_path):
    """
    Read an astrometric reference catalogue of the calibration field.

    Accepts the packaged ECSV catalogue and older user-supplied tables with columns
    VMAG, RA, DEC. The magnitude used for brightness ranking is returned as MAG.

    Args:
        field_path (str): path to the catalogue file

    Returns:
        astropy.table.Table: the catalogue, with at least the columns RA, DEC [deg] and MAG

    Raises:
        OSError: if the file is a Git LFS pointer that was never downloaded
        ValueError: if RA, DEC or a magnitude column is missing
    """
    with open(field_path, "rb") as catalog_file:
        start = catalog_file.read(64)
    if start.startswith(b"version https://git-lfs"):
        raise OSError("{0} is a Git LFS pointer, not the catalogue. Run `git lfs pull` in the "
                      "corgidrp repository to download it.".format(field_path))

    field = ascii.read(field_path)
    if "MAG" not in field.colnames and "VMAG" in field.colnames:
        field["MAG"] = field["VMAG"]
    missing = [name for name in ("RA", "DEC", "MAG") if name not in field.colnames]
    if missing:
        raise ValueError("field catalogue {0} is missing column(s) {1} (a magnitude column may be "
                         "called MAG or VMAG)".format(field_path, missing))
    return field


def centroid(frame):
    """
    Finds the center coordinates for a given frame.

    Args:
        frame (np.ndarray): 2D array to compute centering

    Returns:
        tuple: (xcen, ycen) X and Y centroid coordinates.

    """
    y, x = np.indices(frame.shape)
    
    ycen = np.sum(y * frame)/np.sum(frame)
    xcen = np.sum(x * frame)/np.sum(frame)
    
    return xcen, ycen


def centroid_with_roi(frame, roi_radius=5, centering_initial_guess=None, gaussian_kernel_size=None):
    """
    Finds the centroid in a sub-region around a given initial guess (or the brightest pixel if no guess is provided).

    Args:
        frame (np.ndarray): 2D array to compute centering.
        roi_radius (int or float): Half-size of the box around the initial guess or brightest pixel.
        centering_initial_guess (tuple or None, optional): (x_init, y_init) as initial guess for centroiding.
                                                           If None, defaults to the brightest pixel.
        gaussian_kernel_size (int or None, optional): Size of the gaussian kernel convolved with the frame through before guessing the center.
                                                    If None, no filtering is done. 

    Returns:
        tuple: (xcen, ycen) X and Y centroid coordinates.
    """

    # 1) Unpack initial guess or fall back to brightest pixel
    if centering_initial_guess is not None and None not in centering_initial_guess:
        peak_x, peak_y = int(round(centering_initial_guess[0])), int(round(centering_initial_guess[1]))
    elif gaussian_kernel_size is not None:
        # smooth out the frame before taking brightest pixel
        kernel = Gaussian2DKernel(gaussian_kernel_size)
        filtered_frame = convolve(frame, kernel, nan_treatment='interpolate', preserve_nan=False)
        peak_y, peak_x = np.unravel_index(np.nanargmax(filtered_frame), frame.shape)
    else:
        # take the brightest pixel directly with no filtering
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
    # Use nansum to handle NaN pixels from flat field corrections
    total_flux = np.nansum(sub_frame)
    if total_flux == 0 or np.isnan(total_flux):
        # probably should raise error if total flux is 0 or nan
        raise ValueError(f"Cannot compute centroid: ROI has zero or NaN total flux at position ({peak_x}, {peak_y})")

    xcen = np.nansum(x_indices * sub_frame) / total_flux
    ycen = np.nansum(y_indices * sub_frame) / total_flux

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
    # Replace NaN/Inf values with zeros before fitting
    # This allows curve_fit to work, treating bad pixels as background
    stamp_clean = np.copy(stamp)
    data_clean = np.copy(data)

    stamp_clean[~np.isfinite(stamp)] = 0.0
    data_clean[~np.isfinite(data)] = 0.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        popt, pcov = optimize.curve_fit(shift_psf, stamp_clean, data_clean.ravel(), p0=(0,0,guessflux), maxfev=2000)
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

    dtheta = np.mod(np.arctan2(ydif, xdif) - np.pi/2, 2*np.pi)  # this is in radians

    return np.degrees(dtheta)


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
        threshold (int): Number of stars to find (default: 10)
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
            pf, fw, x, y = fakes.gaussfit2d(frame=fit_gauss_image, xguess=gx+pad, yguess=gy+pad, guesspeak=fit_gauss_image[int(gy+pad), int(gx+pad)], refinefit=True)
        # make sure the fit location is not outside the image
        if ((x-pad) > 1024) or ((x-pad) < 0) or ((y-pad) > 1024) or ((y-pad) < 0):
            # if x or y is outside image, save the original guess location (still valid, just not at sub-pixel precision)
            fit_xs[i] = gx
            fit_ys[i] = gy
        else:
            # if the fit is *inside the image, record the new fit position
            fit_xs[i] = x-pad
            fit_ys[i] = y-pad

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
        field_path (str): Full path to the reference catalogue (see read_field_catalog)
        comparison_threshold (int): How many stars in the field to consider for the initial match
        rad (float): The radius [deg] around the target coordinate for creating a subfield to match image sources to
        platescale_guess (float): An initial guess for the platescale value (default: 21.8 [mas/ pixel])
        platescale_tol (float): A tolerance for finding source matches within a fraction of the initial plate scale guess (default: 0.5)

    Returns:
        matched_sources (astropy.table.Table): Astropy table with columns 'x','y','RA', 'DEC' as pixel locations and corresponding sky positons
        
    '''
    # load the reference catalogue (checks for RA, DEC and a magnitude column)
    field = read_field_catalog(field_path)

    # gather the pixel locations for the (3) brightest sources in the image
    source1, source2, source3 = sources[0], sources[1], sources[2]

    # define the side length to perimeter ratio for the triangle made from these sources [pixels]
    l12, l23, l31 = np.sqrt(np.power(source1['x'] - source2['x'], 2) + np.power(source1['y'] - source2['y'], 2)), np.sqrt(np.power(source2['x'] - source3['x'], 2) + np.power(source2['y'] - source3['y'], 2)), np.sqrt(np.power(source3['x'] - source1['x'], 2) + np.power(source3['y'] - source1['y'], 2))
    perimeter = l12 + l23 + l31

    # keep track of which star is furthest from and closest to the other 2
    dstar1 = l12 + l31
    dstar2 = l23 + l12
    dstar3 = l23 + l31

    tri_dist_star = [(dstar1, source1), (dstar2, source2), (dstar3, source3)]
    tri_dist_star.sort(key=lambda a: a[0])  # sort the stars based on their distances to the other two
    short_star, mid_star, long_star = tri_dist_star[0][1], tri_dist_star[1][1], tri_dist_star[2][1]

    # the shortest to longest sides get reordered to l1, l2, l3
    l1, l2, l3 = np.sort([l12, l23, l31])
    a, b, c = l1/perimeter, l2/perimeter, l3/perimeter

    # define a search field around the commanded pointing
    target = image.pri_hdr['RA'], image.pri_hdr['DEC']
    
    subfield = field[((field['RA'] >= target[0] - rad) & (field['RA'] <= target[0] + rad) & (field['DEC'] >= target[1] - (rad*(np.cos(np.radians(target[1]))))) & (field['DEC'] <= target[1] + (rad*np.cos(np.radians(target[1])))))]

    bright_order_subfield = subfield.copy()
    bright_order_subfield.sort(keys='MAG')
    brightest_field = bright_order_subfield[:comparison_threshold]

    # create all combinations of triangles with the brightest field sources
    combos = list(compute_combinations(range(len(brightest_field)), r=3))

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
            ap, bp, cp = -np.inf, -np.inf, -np.inf
        if ((len1 / l1) < platescale_guess* (1 - platescale_tol)) or ((len2 / l2) < platescale_guess* (1 - platescale_tol)) or ((len3 / l3) < platescale_guess* (1 - platescale_tol)):
            ap, bp, cp = -np.inf, -np.inf, -np.inf

        # find the best fit to the brightest image triangle
        lstsq = (a - ap)**2 + (b - bp)**2 + (c - cp)**2
        if lstsq < smallest_lsq:
            smallest_lsq = lstsq
            best_ind = i
            best_sky_ind = ind

    # keep track of which best fit skycoord is the farthest from the other two
    j, k, l = best_sky_ind
    coord1, coord2, coord3 = skycoords[j], skycoords[k], skycoords[l]

    dcoord1 = coord1.separation(coord2).mas + coord1.separation(coord3).mas
    dcoord2 = coord2.separation(coord3).mas + coord2.separation(coord1).mas
    dcoord3 = coord3.separation(coord1).mas + coord3.separation(coord2).mas

    tri_dist_coords = [(dcoord1, coord1), (dcoord2, coord2), (dcoord3, coord3)]
    tri_dist_coords.sort(key=lambda a: a[0])  # sort the stars based on their distances to the other two
    short_coord, mid_coord, long_coord = tri_dist_coords[0][1], tri_dist_coords[1][1], tri_dist_coords[2][1]
    
    # now use the side length to separations with best fit triangle to define a pseudo plate scale
    best_l1, best_l2, best_l3 = field_side_lengths[best_ind]
    initial_platescale = np.mean(np.array([best_l1 / l1, best_l2 / l2, best_l3 / l3]))  # [deg/mas]

    # find the angle between the `target` star and the other two, in the image and in the field
    rot_image = np.array([angle_between((short_star['x'], short_star['y']), (s['x'], s['y'])) for s in [mid_star, long_star]])
    rot_field = np.array([short_coord.position_angle(t).deg for t in [mid_coord, long_coord]])
    # use arctan2 to handle angle wrapping when calculating the difference between the reference field angle and the image
    dtheta1 = np.arctan2(np.sin(np.radians(rot_field[0] - rot_image[0])), np.cos(np.radians(rot_field[0] - rot_image[0])))
    dtheta2 = np.arctan2(np.sin(np.radians(rot_field[1] - rot_image[1])), np.cos(np.radians(rot_field[1] - rot_image[1])))
    initial_northangle = np.mean([dtheta1, dtheta2])

    # make a new image header with the pseudo platescale and north angle to find matchings
    # allow for some error window and assign the closest star to each source
    vert_ang = initial_northangle   # this is in radians
    pc = np.array([[-np.cos(vert_ang), np.sin(vert_ang)], [np.sin(vert_ang), np.cos(vert_ang)]])
    cdmatrix = pc * (initial_platescale * 0.001) / 3600.

    new_hdr = {}
    new_hdr['CD1_1'] = cdmatrix[0,0]
    new_hdr['CD1_2'] = cdmatrix[0,1]
    new_hdr['CD2_1'] = cdmatrix[1,0]
    new_hdr['CD2_2'] = cdmatrix[1,1]
    new_hdr['CRPIX1'] = short_star['x']
    new_hdr['CRPIX2'] = short_star['y']
    new_hdr['CTYPE1'] = 'RA---TAN'
    new_hdr['CTYPE2'] = 'DEC--TAN'
    new_hdr['CDELT1'] = (initial_platescale * 0.001) / 3600.
    new_hdr['CDELT2'] = (initial_platescale * 0.001) / 3600.
    new_hdr['CRVAL1'] = short_coord.ra.value
    new_hdr['CRVAL2'] = short_coord.dec.value
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

def compute_platescale_and_northangle(image, source_info, center_radius=1):
    """
    Used to find the platescale and north angle of the image. Calculates the platescale for each pair of stars in the image
    and returns the averged platescale. Calculates the north angle for pairs of stars with the center target
    and returns the averged north angle.
    
    Args:
        image (numpy.ndarray): 2D array of image data 
        source_info (astropy.table.Table): Estimated pixel positions of sources and true sky positions, must have column names 'x', 'y', 'RA', 'DEC'
        # center_coord (tuple):
        #     (float): RA coordinate of the target pointing
        #     (float): Dec coordinate of the target pointing
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

    imageshape = np.shape(image)
    cut = 1 - center_radius
    suby, subx = imageshape[0] * cut, imageshape[1] * cut
    center_source_inds = np.where((guesses['x'] >= subx) & (guesses['x'] <= imageshape[1] - subx) & (guesses['y'] >= suby) & (guesses['y'] <= imageshape[0] - suby))
    sub_guesses = guesses[center_source_inds]   # these are already guassian fit to find exact pixel location
    sub_skycoords = skycoords[center_source_inds]

    # PLATESCALE
    # create random combinations of stars
    all_combinations = list(compute_combinations(sub_guesses))
    if len(all_combinations) > 200:
        rand_inds = np.random.randint(low=0, high=len(all_combinations), size=200)
        combo_list = np.array(all_combinations)[rand_inds]
    else:
        combo_list = np.array(all_combinations)


    length_diffs = np.empty(len(combo_list))
    for i,c in enumerate(combo_list):
        # calculate the sky separation btw all combos of stars
        star1 = sub_skycoords[c[0]]
        star2 = sub_skycoords[c[1]]

        sep_sky = star1.separation(star2).mas

        # calculate the image separation between the same combos of stars
        star1 = sub_guesses[c[0]]
        star2 = sub_guesses[c[1]]
        xguess = star2['x'] - star1['x']
        yguess = star2['y'] - star1['y']
        
        (xoff, yoff), _ = measure_offset(image, xstar_guess=star1['x'], ystar_guess=star1['y'], xoffset_guess= xguess, yoffset_guess= yguess)
        sep_pix = np.sqrt(np.power(xoff,2) + np.power(yoff,2))

        # calc the ratio of sky to pixel separation for each star pair
        diff = sep_sky / sep_pix
        length_diffs[i] = diff

    # estimate the platescale from each combination
    platescale = np.median(length_diffs)

    # NORTH ANGLE    
    angle_diffs = np.empty(len(combo_list))
    for i,c in enumerate(combo_list):       # compute pa sky vs image for each star combination
        star1 = sub_guesses[c[0]]
        star2 = sub_guesses[c[1]]

        coord1 = sub_skycoords[c[0]]
        coord2 = sub_skycoords[c[1]]

        pa_image = angle_between((star1['x'], star1['y']), (star2['x'], star2['y']))    # these are both in [deg]
        pa_sky = coord1.position_angle(coord2).deg

        # difference in angle between sky and image
        dtheta = np.arctan2(np.sin(np.radians(pa_sky - pa_image)), np.cos(np.radians(pa_sky - pa_image)))
        angle_diffs[i] = np.degrees(dtheta)

    north_angle = np.median(angle_diffs)
    
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
        ra_offset (float): the offset in RA [deg] *from the target pointing RA *to the true RA of the center coord
        dec_offset (float): the offset in Dec [deg] *from the target pointing Dec *to the true Dec of the center coord
    
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
    new_hdr['CDELT1'] = (cal_properties[0] * 0.001) / 3600.
    new_hdr['CDELT2'] = (cal_properties[0] * 0.001) / 3600.
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
    # boresight_x,y is the average offset from a star's predicted image location to its actual position given platescale and northangle alone
    # such that x_predict [predicted pixel location] + offset [pixel] = x_center [image pixel location]
    boresight_x, boresight_y = np.median(boresights[:,0]), np.median(boresights[:,1]) 
    # we want to know what the ~actual (512, 512) pixels location would be in the 'predicted' frame so we have
    # predicted_x + offset = 512.   OR      predicted_x = 512. - offset
    true_center_x, true_center_y = 512.- boresight_x, 512.- boresight_y

    # convert back to skycoord
    true_center_coord = astropy.wcs.utils.pixel_to_skycoord(true_center_x, true_center_y, wcs=w, origin=1)
    # true_center_ra, true_center_dec = true_center_coord.ra.value, true_center_coord.dec.value
    # boresight_ra, boresight_dec = target_coordinate[0] - true_center_ra, target_coordinate[1] - true_center_dec
    target_skycoord = SkyCoord(target_coordinate[0], target_coordinate[1], unit='deg')
    ra_offset, dec_offset = target_skycoord.spherical_offsets_to(true_center_coord)

    return ra_offset.deg, dec_offset.deg

def format_distortion_inputs(input_dataset, source_matches, position_error=None):
    ''' Function that formats the input data for the distortion map computation * must be run before compute_distortion *
    
    Args:
        input_dataset (corgidrp.data.dataset): corgidrp dataset object with images to compute the distortion from
        source_matches (list of astropy.table.Table() objects): List of length N for N frames in the input dataset. Tables must columns 'x','y','RA','DEC' as pixel locations and corresponding sky positons
        # ref_star_pos (list of astropy.table.Table() objects): List of length N for N frames. Tables must have column names 'x', 'y', 'RA', 'DEC' for the position of the reference position to compute pairs with
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

        skycoords = SkyCoord(ra= source_matches[frame_ind]['RA'], dec= source_matches[frame_ind]['DEC'], unit='deg', frame='icrs')
        # compute combinations of all pairs of stars
        combos = list(compute_combinations(range(len(source_matches[frame_ind])), r=2))
        # pair_ind = 1
        # while pair_ind < len(source_matches[frame_ind]):    # compare the first source with all others
        for pair_ind in combos: # restrict to using only the first ~200 combinations
        # for pair_ind in combo_list:
            # get the pixel offset
            star1 = source_matches[frame_ind][pair_ind[0]]  
            star2 = source_matches[frame_ind][pair_ind[1]]
    
            x_guess = star2['x'] - star1['x']
            y_guess = star2['y'] - star1['y']
        
            (dx, dy), (xfit_err, yfit_err, _) = measure_offset(input_image, star2['x'], star2['y'], x_guess, y_guess, guessflux=10000)
    
            # get the true sky offset [mas]
            true1 = skycoords[pair_ind[0]]  
            true2 = skycoords[pair_ind[1]]
        
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

def compute_distortion(pos1, meas_offset, sky_offset, meas_errs, platescale, northangle, fitorder=3, initial_guess=None):
    ''' 
    Function that computes the legendre polynomial coefficients that describe the image distortion map * must run format_disotrtio_inputs() first *

    Args:
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
  
  
def boresight_calibration(input_dataset, field_path='lmc_astrometric_reference.ecsv', field_matches=None, find_threshold=10, fwhm=7, mask_rad=1, 
                          comparison_threshold=50, search_rad=0.012, platescale_guess=21.8, platescale_tol=0.1, center_radius=0.9, 
                          frames_to_combine=False, find_distortion=False, fitorder=3, position_error=None, initial_dist_guess=None, 
                          pa_tolerance=0.1, keywords_to_split_dataset_by=None):
    """
    Perform the boresight calibration of a dataset.
    
    Args:
        input_dataset (corgidrp.data.Dataset): Dataset containing a images for astrometric calibration
        field_path (str): Full path to the reference catalogue, or the default 'lmc_astrometric_reference.ecsv' for the catalogue shipped with corgidrp (default: 'lmc_astrometric_reference.ecsv')
        field_matches (list of str or astropy.table.Table): List of full paths to files or astropy tables with calibration field matches for each image in the dataset (x, y, ra, dec), if single str the same filepath used for all frames,nif None, automated source matching is used (default: None)
        find_threshold (int): Number of stars to find (default 10)
        fwhm (float): Full width at half maximum of the stellar psf (default: 7, ~fwhm for a normal distribution with sigma=3)
        mask_rad (int): Radius of mask for stars [in fwhm] (default: 1)
        comparison_threshold (int): How many stars in the field to consider for the initial match (default: 50)
        search_rad (float): The radius [deg] around the target coordinate for creating a subfield to match image sources to
        platescale_guess (float): An initial guess for the platescale value (default: 21.8 [mas/ pixel])
        platescale_tol (float): A tolerance for finding source matches within a fraction of the initial plate scale guess (default: 0.1)
        center_radius (float): Percent of the image to compute plate scale and north angle from, centered around the image center (default: 0.9 -- ie: 90% of the image is used)
        frames_to_combine (boolean or int): If True or 1, median combine all frames of the same target and instrument configuration into a singular frame.
            If it is an integer value N > 1, the dataset is split into sequences of N frames to be median combined regardless of header keywords. (default: False)
        find_distortion (boolean): Used to determine if distortion map coeffs will be computed (default: False)
        fitorder (int): The order of legendre polynomials used to fit the distortion map (default: 3)
        position_error (NoneType or int): If int, this is the uniform error value assumed for the offset between pairs of stars in both x and y
        initial_dist_guess (np.array): An initial guess of legendre coefficients used for fitting distortion, if None will use coeffs associated with no distortion (default: None)
        pa_tolerance (float, optional): Maximum allowed difference in PA_APER (deg) to group frames together when frames_to_combine is True (Default: 0.1)
        keywords_to_split_dataset_by (list, optional): List of additional header keywords to split the input dataset by for frame combining in addition default keywords the function uses for target, roll, and dither (default: None)

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

    # load in field data to refer to
    if field_path == 'lmc_astrometric_reference.ecsv':
        field_path = default_field_path

    # combine data frames if requested
    grouped_datasets = []
    if int(frames_to_combine) > 1:
        # group N number of frames together as specified by frames_to_combine
        num_frames = len(input_dataset)
        frame_counter = 0
        group = []
        for i in range(num_frames):
            frame = dataset.frames[i]
            frame_counter += 1
            group.append(frame)
            # construct grouping once enough frames are accumulated or end of dataset is reached
            if frame_counter == frames_to_combine or i == num_frames - 1:
                grouped_datasets.append(corgidrp.data.Dataset(group))
                # reset counter and grouping
                frame_counter = 0
                group = []
    elif int(frames_to_combine) == 1: # note that we treat 1 as true here, to disable frame combining 0/False must be passed in
        # group frames based on header keywords
        # first split by PA since it requires handling of slight numerical mismatches and circular behavior
        clusters = []
        for frame in dataset.frames:
            pa = frame.pri_hdr["PA_APER"] % 360.0 # in case PA_APER can be negative..
            pa_rad = np.deg2rad(pa)
            pa_sin = np.sin(pa_rad)
            pa_cos = np.cos(pa_rad)
            best_idx = None
            best_diff = None
            for idx, cluster in enumerate(clusters):
                pa_diff = abs(((pa - cluster["pa_center"] + 180.0) % 360.0) - 180.0)
                if pa_diff <= pa_tolerance and (best_diff is None or pa_diff < best_diff):
                    best_idx = idx
                    best_diff = pa_diff
            if best_idx is None:
                clusters.append({
                    "pa_center": pa,
                    "sum_sin": pa_sin,
                    "sum_cos": pa_cos,
                    "frames": [frame],
                })
            else:
                cluster = clusters[best_idx]
                cluster["frames"].append(frame)
                cluster["sum_sin"] += pa_sin
                cluster["sum_cos"] += pa_cos
                cluster["pa_center"] = np.degrees(
                    np.arctan2(cluster["sum_sin"], cluster["sum_cos"])
                ) % 360.0

        # create master list of all the primary and external header keywords to split by
        split_keywords = ['TARGET', 'RA_APER', 'DEC_APER', 'FSMX', 'FSMY'] # keywords for target and dither splitting
        if keywords_to_split_dataset_by is not None:
            split_keywords.extend(keywords_to_split_dataset_by)
        # separate out primary and external header keywords, ones not found in either will be skipped
        # this assumes the header formats are consistent across all input frames
        prihdr = dataset.frames[0].pri_hdr
        exthdr = dataset.frames[0].ext_hdr
        prihdr_keyword_list = []
        exthdr_keyword_list = []
        for keyword in split_keywords:
            if keyword in prihdr:
                prihdr_keyword_list.append(keyword)
            elif keyword in exthdr:
                exthdr_keyword_list.append(keyword)

        # split pa_dataset one more time by the constructed keyword list
        for cluster in clusters:
            pa_dataset = corgidrp.data.Dataset(cluster["frames"])
            dither_datasets, _ = pa_dataset.split_dataset(prihdr_keywords=prihdr_keyword_list, exthdr_keywords=exthdr_keyword_list)
            # add to the final grouping
            grouped_datasets.extend(dither_datasets)

        # Header grouping sorts the pointings, but the boresight must refer to
        # the first input pointing. Restore the order in which groups appeared.
        frame_indices = {id(frame): index for index, frame in enumerate(dataset)}
        grouped_datasets.sort(key=lambda group: frame_indices[id(group[0])])

    # median combine each grouped dataset into one frame for calibration processing
    if len(grouped_datasets) > 0:
        image_objects = []
        for grouped_dataset in grouped_datasets:
            # grab all frames in the dataset
            data_array = [frame.data for frame in grouped_dataset]
            # median combine
            comb = np.nanmedian(data_array, axis=0)
            # create image object
            im = corgidrp.data.Image(comb, pri_hdr=grouped_dataset[0].pri_hdr, ext_hdr=grouped_dataset[0].ext_hdr)
            im.filename = grouped_dataset[-1].filename
            image_objects.append(im)
        
        # rebuild the input dataset from the grouped and median combined frames
        dataset = corgidrp.data.Dataset(image_objects)

    # create a place to store all the calibration measurements
    astroms = []
    hold_matches = []   # place to hold the auto-found source matches for each frame
    corrected_positions_boresight = []      # place to hold the corrected target position based on boresight offsets for each frame

    for i in range(len(dataset)):
        if int(frames_to_combine) != 0:
            in_dataset = grouped_datasets[i]
        else:
            in_dataset = corgidrp.data.Dataset([dataset[i]])
        image = dataset[i].data

        # call the target coordinates from the image header
        target_coordinate = (dataset[i].pri_hdr['RA'], dataset[i].pri_hdr['DEC'])
        target_skycoord = SkyCoord(ra=target_coordinate[0], dec=target_coordinate[1], unit='deg')

        # compute the calibration properties
        found_sources = find_source_locations(image, threshold=find_threshold, fwhm=fwhm, mask_rad=mask_rad)
        matched_sources = match_sources(dataset[i], found_sources, field_path, comparison_threshold=comparison_threshold, rad=search_rad, platescale_guess=platescale_guess, platescale_tol=platescale_tol)
        # if len(hold_matches) < 1:
        hold_matches.append(matched_sources)

        cal_properties = compute_platescale_and_northangle(image, source_info=matched_sources, center_radius=center_radius)
        ra_off, dec_off = compute_boresight(image, source_info=matched_sources, target_coordinate=target_coordinate, cal_properties=cal_properties)
        # calculate the corrected target position based on ra, dec offsets using skycoord to account for spherical coord system
        corrected_center_coord = target_skycoord.spherical_offsets_by(ra_off * astropy.units.deg, dec_off * astropy.units.deg)
        corr_ra, corr_dec = corrected_center_coord.ra.value, corrected_center_coord.dec.value
        corrected_positions_boresight.append([corr_ra, corr_dec])

        # return a single AstrometricCalibration data file
        astrom_data = np.array([corr_ra, corr_dec, cal_properties[0], cal_properties[1], ra_off, dec_off, np.inf, np.inf])
        astrom_cal = corgidrp.data.AstrometricCalibration(astrom_data, pri_hdr=dataset[i].pri_hdr, ext_hdr=dataset[i].ext_hdr, input_dataset=in_dataset)
        # change the filename here since the astrom_cals will be averaged later and arent individually saved ('_ast_cal' will be added to filename twice otherwise)
        astrom_cal.filename = astrom_cal.filename.split("_ast_cal")[0] + '.fits'
        astroms.append(astrom_cal)

    # average the calibration properties over all frames
    avg_ra = np.median([astro.avg_offset[0] for astro in astroms])  # this is the average ra offset [deg]
    avg_dec = np.median([astro.avg_offset[1] for astro in astroms])
    avg_platescale = np.median([astro.platescale for astro in astroms])
    avg_northangle = np.median([astro.northangle for astro in astroms])

    # compute the distortion map coeffs
    if find_distortion:
        # use the found matches for distortion
        first_stars, offsets, true_offsets, errs = format_distortion_inputs(input_dataset, source_matches=hold_matches, position_error=position_error)
        distortion_coeffs, order = compute_distortion(first_stars, offsets, true_offsets, errs, platescale=avg_platescale, northangle=avg_northangle, fitorder=fitorder, initial_guess=initial_dist_guess)
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

# ---------------------------------------------------------------------------------------------
# Building the LMC astrometric reference catalogue (default_field_path)
#
# Run once by a developer when the catalogue changes, not by the pipeline:
#     python -c "import corgidrp.astrom as a; a.build_lmc_reference_catalog(gaia_file='gaia.ecsv')"
# Needs network access and astroquery (not a corgidrp dependency) unless both input files are
# given. A Gaia query result is saved to gaia_file on first use and reused afterwards.
# ---------------------------------------------------------------------------------------------

_LMC_ANDERSON_URL = "https://www.stsci.edu/~jayander/JWST_CALFIELD2020/master_output.xymvrdeepmeehuGAIA_DR3"
_LMC_ANDERSON_REFERENCE = ("Anderson, J., Fall, M., and the Astrometry Working Group 2021, JWST-STScI-007716, "
                           "The JWST Calibration Field: Absolute Astrometry and Proper Motions with GAIA "
                           "and a Second HST Epoch")
_LMC_GAIA_REFERENCE = "Gaia Collaboration, Vallenari, A., et al. 2023, A&A, 674, A1 (Gaia DR3)"
_LMC_GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

_LMC_FIELD_CENTER = (80.5, -69.5)   # deg, centre of Anderson's reference frame
_LMC_GAIA_CONE_RADIUS = 0.1         # deg, covers the whole HST footprint
_LMC_CATALOG_EPOCH = 2015.5         # Julian year, TCB
_LMC_GAIA_EPOCH = 2016.0            # Julian year, TCB
_LMC_HST_MAG_LIMIT = 22.5           # F606W VEGAMAG, same cut as the former JWST_CALFIELD2020.csv
_LMC_GAIA_MAG_LIMIT = 19.5          # Gaia G; fainter missing stars are mostly blends or poorly measured
_LMC_DUPLICATE_RADIUS = 0.1         # arcsec
_LMC_WIDE_DUPLICATE_RADIUS = 0.3    # arcsec, for Gaia stars without proper motion (see _lmc_near_hst_star)
_LMC_DUPLICATE_MAG_DIFF = 0.5       # mag, G against F606W
_LMC_FOOTPRINT_RADIUS = 5.0         # arcsec
_LMC_FOOTPRINT_MIN_STARS = 10
_LMC_RUWE_LIMIT = 1.4
_LMC_MAX_FRAME_OFFSET_MAS = 0.2
_LMC_SYNC_ROW_LIMIT = 2000          # Gaia archive cap on synchronous query results

LMC_FLAG_PM_ASSUMED = 1
LMC_FLAG_HIGH_RUWE = 2
LMC_FLAG_GAIA_DUPLICATE = 4

_LMC_GAIA_COLUMNS = ["source_id", "ra", "dec", "ra_error", "dec_error", "pmra", "pmra_error",
                     "pmdec", "pmdec_error", "parallax", "parallax_error", "phot_g_mean_mag",
                     "ruwe", "astrometric_params_solved", "duplicated_source"]

_LMC_COLUMNS = {   # name: (unit, description)
    "GAIA_ID": (None, "Gaia DR3 source_id, 0 if the star has no Gaia counterpart"),
    "RA": ("deg", "Right ascension, ICRS, epoch J2015.5"),
    "DEC": ("deg", "Declination, ICRS, epoch J2015.5"),
    "RA_ERR": ("mas", "Uncertainty of RA on the sky (times cos dec) at J2015.5"),
    "DEC_ERR": ("mas", "Uncertainty of DEC at J2015.5"),
    "MAG": ("mag", "Magnitude used for brightness ranking; band given in MAG_BAND"),
    "MAG_BAND": (None, "F606W (HST ACS/WFC, VEGAMAG) or G (Gaia DR3)"),
    "GMAG": ("mag", "Gaia DR3 G magnitude, NaN without Gaia counterpart"),
    "PMRA": ("mas / yr", "Proper motion in RA, mu_alpha* = mu_alpha cos(dec)"),
    "PMDEC": ("mas / yr", "Proper motion in Dec"),
    "PMRA_ERR": ("mas / yr", "Uncertainty of PMRA"),
    "PMDEC_ERR": ("mas / yr", "Uncertainty of PMDEC"),
    "PARALLAX": ("mas", "Gaia DR3 parallax, NaN if not available"),
    "PARALLAX_ERR": ("mas", "Uncertainty of PARALLAX"),
    "RUWE": (None, "Gaia DR3 renormalised unit weight error, NaN if not available"),
    "POS_SOURCE": (None, "Source of RA/DEC: HST or GAIA"),
    "PM_SOURCE": (None, "Source of the proper motion: HST, GAIA, or BULK (LMC bulk motion assumed)"),
    "FLAG": (None, "Bitmask: 1 proper motion assumed, 2 RUWE > 1.4, 4 Gaia duplicated_source"),
}


def build_lmc_reference_catalog(output_path=default_field_path, anderson_file=None, gaia_file=None):
    """
    Build the LMC astrometric reference catalogue used by boresight_calibration.

    Merges Anderson's HST catalogue of the JWST calibration field in the LMC (JWST-STScI-007716)
    with the Gaia DR3 stars HST did not measure (mostly stars that saturated in the deep HST
    exposures), at the common epoch J2015.5, and records origin and selection in the ECSV
    metadata. The catalogue is checked after writing and only then moved to output_path, so a
    failed build never replaces an existing catalogue.

    Args:
        output_path (str): where to write the ECSV catalogue
        anderson_file (str): local copy of Anderson's master_output.xymvrdeepmeehuGAIA_DR3;
            downloaded if None
        gaia_file (str): ECSV with a saved Gaia query; queried and written if it does not exist,
            queried and not saved if None

    Returns:
        astropy.table.Table: the catalogue as written

    Raises:
        RuntimeError: if a Gaia query fails or the catalogue fails its self-checks
    """
    import datetime
    import hashlib
    import tempfile
    import urllib.request
    from astropy.table import Table

    with tempfile.TemporaryDirectory() as scratch:
        if anderson_file is None:
            anderson_file = os.path.join(scratch, os.path.basename(_LMC_ANDERSON_URL))
            urllib.request.urlretrieve(_LMC_ANDERSON_URL, anderson_file)
        with open(anderson_file, "rb") as handle:
            anderson_sha256 = hashlib.sha256(handle.read()).hexdigest()
        hst_all = _lmc_read_anderson(anderson_file)

    bulk = _lmc_bulk_motion(hst_all[np.asarray(hst_all["MAG"], float) < _LMC_HST_MAG_LIMIT])

    if gaia_file is not None and os.path.exists(gaia_file):
        gaia = Table.read(gaia_file, format="ascii.ecsv")
    else:
        gaia = _lmc_query_gaia()
        if gaia_file is not None:
            gaia.write(gaia_file, format="ascii.ecsv")

    gaia_coords = _lmc_gaia_at_epoch(gaia, bulk)
    catalog = _lmc_assemble(hst_all, gaia, gaia_coords, bulk)
    offset = _lmc_frame_offset(catalog, gaia, gaia_coords)
    catalog.sort("MAG")
    for name, (unit, description) in _LMC_COLUMNS.items():
        catalog[name].unit = unit
        catalog[name].description = description
    catalog.meta = _lmc_metadata(catalog, anderson_sha256, gaia.meta, bulk, offset,
                                 datetime.date.today().isoformat())

    linked_ids = np.asarray(hst_all["GAIA_ID"], np.int64)
    temporary = output_path + ".tmp"
    catalog.write(temporary, format="ascii.ecsv", overwrite=True)
    try:
        _lmc_check(Table.read(temporary, format="ascii.ecsv"), linked_ids[linked_ids != 0],
                   gaia["source_id"], offset)
    except RuntimeError:
        os.remove(temporary)
        raise
    os.replace(temporary, output_path)
    return catalog


def _lmc_read_anderson(path):
    """
    Read Anderson's master_output.xymvrdeepmeehuGAIA_DR3 isolated-star catalogue.

    Columns used (1-based, as in the file header): 4 F606W VEGAMAG, 5/6 RA/Dec at 2015.5 [deg],
    7/8 position errors [mas], 9/10 proper motions [mas/yr], 11/12 their errors,
    34 Gaia DR3 source_id (0 if none).

    Args:
        path (str): path to the master file

    Returns:
        astropy.table.Table: columns MAG, RA, DEC, RA_ERR, DEC_ERR, PMRA, PMDEC, PMRA_ERR,
            PMDEC_ERR, GAIA_ID
    """
    from astropy.table import Table

    values = np.loadtxt(path, comments="#", usecols=(3, 4, 5, 6, 7, 8, 9, 10, 11))
    table = Table(values, names=["MAG", "RA", "DEC", "RA_ERR", "DEC_ERR", "PMRA", "PMDEC",
                                 "PMRA_ERR", "PMDEC_ERR"])
    table["GAIA_ID"] = np.loadtxt(path, comments="#", usecols=(33,), dtype=np.int64)
    return table


def _lmc_bulk_motion(hst):
    """
    LMC bulk proper motion (medians) and internal dispersion (mean robust std of both axes).

    Args:
        hst (astropy.table.Table): HST rows with PMRA, PMDEC [mas/yr]

    Returns:
        dict: pmra, pmdec, dispersion [mas/yr]
    """
    pmra = np.asarray(hst["PMRA"], float)
    pmdec = np.asarray(hst["PMDEC"], float)

    def robust_std(values):
        return 1.4826 * np.median(np.abs(values - np.median(values)))

    return {"pmra": float(np.median(pmra)), "pmdec": float(np.median(pmdec)),
            "dispersion": float(0.5 * (robust_std(pmra) + robust_std(pmdec)))}


def _lmc_cone_tiles(center, radius, n_per_side):
    """
    Half-open RA/Dec boxes [ra_min, ra_max) x [dec_min, dec_max) that cover a cone.

    A star on a shared edge belongs to exactly one box; the outer edges are padded slightly.

    Args:
        center (tuple): (ra, dec) of the cone centre [deg]
        radius (float): cone radius [deg]
        n_per_side (int): number of boxes along RA and along Dec

    Returns:
        list of tuple: (ra_min, ra_max, dec_min, dec_max) per box [deg]
    """
    pad = 1e-6
    dec_half = radius + pad
    # widest RA extent is at the declination closest to the pole
    ra_half = radius / np.cos(np.radians(min(abs(center[1]) + radius, 89.9))) + pad
    ra_edges = np.linspace(center[0] - ra_half, center[0] + ra_half, n_per_side + 1)
    dec_edges = np.linspace(center[1] - dec_half, center[1] + dec_half, n_per_side + 1)
    return [(ra_edges[i], ra_edges[i + 1], dec_edges[j], dec_edges[j + 1])
            for i in range(n_per_side) for j in range(n_per_side)]


def _lmc_query_gaia(attempts=3):
    """
    Query Gaia DR3 for all sources in the cone around the field.

    The archive caps synchronous queries at _LMC_SYNC_ROW_LIMIT rows without warning, and its
    asynchronous queue can stall for a long time. So the cone is split into tiles small enough
    for synchronous queries; a tile at the cap, or a total that differs from COUNT(*) of the
    cone, raises.

    Args:
        attempts (int): number of tries per query (the archive is sometimes unstable)

    Returns:
        astropy.table.Table: _LMC_GAIA_COLUMNS with NaN for missing values; meta holds the ADQL
            text of the whole cone and the query date
    """
    import datetime
    from astropy.table import Table, vstack
    from astroquery.gaia import Gaia   # only needed to rebuild the catalogue

    def run(adql):
        error = None
        for _ in range(attempts):
            try:
                return Gaia.launch_job(adql).get_results()
            except Exception as failure:   # network or archive failure: retry
                error = failure
        raise RuntimeError("Gaia query failed {0} times: {1}\n{2}".format(attempts, error, adql))

    center, radius = _LMC_FIELD_CENTER, _LMC_GAIA_CONE_RADIUS
    cone = "1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {0}, {1}, {2}))".format(center[0], center[1], radius)
    adql = "SELECT " + ", ".join(_LMC_GAIA_COLUMNS) + " FROM gaiadr3.gaia_source WHERE " + cone
    expected = int(run("SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE " + cone)["n"][0])

    # aim for about a third of the row cap per tile (stars are not spread evenly)
    n_per_side = int(np.ceil(np.sqrt(3.0 * expected / _LMC_SYNC_ROW_LIMIT))) + 1
    pieces = []
    for ra_min, ra_max, dec_min, dec_max in _lmc_cone_tiles(center, radius, n_per_side):
        piece = run(adql + " AND ra >= {0!r} AND ra < {1!r} AND dec >= {2!r} AND dec < {3!r}".format(
            float(ra_min), float(ra_max), float(dec_min), float(dec_max)))
        if len(piece) >= _LMC_SYNC_ROW_LIMIT:
            raise RuntimeError("Gaia tile hit the {0}-row synchronous limit".format(_LMC_SYNC_ROW_LIMIT))
        pieces.append(piece)
    result = vstack(pieces)
    if len(result) != expected:
        raise RuntimeError("Gaia query returned {0} rows, COUNT(*) says {1}".format(len(result), expected))

    gaia = Table(meta={"adql": adql, "queried": datetime.date.today().isoformat()})
    for name in _LMC_GAIA_COLUMNS:
        if name in ("source_id", "astrometric_params_solved"):
            gaia[name] = np.asarray(result[name]).astype(np.int64)
        elif name == "duplicated_source":
            gaia[name] = np.ma.filled(np.ma.asarray(result[name]), False).astype(bool)
        else:
            gaia[name] = np.ma.filled(np.ma.asarray(result[name], dtype=float), np.nan)
    return gaia


def _lmc_gaia_at_epoch(gaia, bulk):
    """
    Move Gaia DR3 positions from J2016.0 to the catalogue epoch J2015.5.

    Stars without a Gaia proper motion (two-parameter solutions) are moved with the LMC bulk
    motion, the same motion the catalogue assigns them.

    Args:
        gaia (astropy.table.Table): Gaia table (see _lmc_query_gaia)
        bulk (dict): output of _lmc_bulk_motion

    Returns:
        astropy.coordinates.SkyCoord: ICRS positions at J2015.5, in the order of gaia
    """
    from astropy.time import Time
    from erfa import ErfaWarning

    pmra, pmdec, known = _lmc_gaia_pm(gaia)
    coords = SkyCoord(ra=np.asarray(gaia["ra"], float) * u.deg, dec=np.asarray(gaia["dec"], float) * u.deg,
                      pm_ra_cosdec=np.where(known, pmra, bulk["pmra"]) * u.mas / u.yr,
                      pm_dec=np.where(known, pmdec, bulk["pmdec"]) * u.mas / u.yr,
                      obstime=Time(_LMC_GAIA_EPOCH, format="jyear", scale="tcb"), frame="icrs")
    with warnings.catch_warnings():
        # without distances ERFA assumes very distant stars and says so; harmless here
        warnings.simplefilter("ignore", ErfaWarning)
        moved = coords.apply_space_motion(new_obstime=Time(_LMC_CATALOG_EPOCH, format="jyear", scale="tcb"))
    return SkyCoord(ra=moved.ra, dec=moved.dec, frame="icrs")


def _lmc_gaia_pm(gaia):
    """
    Gaia proper motions and which stars have one.

    Args:
        gaia (astropy.table.Table): Gaia table

    Returns:
        tuple: (pmra, pmdec, known) numpy arrays; known is False for two-parameter solutions
    """
    pmra = np.asarray(gaia["pmra"], float)
    pmdec = np.asarray(gaia["pmdec"], float)
    return pmra, pmdec, np.isfinite(pmra) & np.isfinite(pmdec)


def _lmc_quality_flags(ruwe, duplicated):
    """
    FLAG bits from Gaia quality indicators.

    Args:
        ruwe (array-like): RUWE values, NaN where unknown
        duplicated (array-like): Gaia duplicated_source flags

    Returns:
        numpy.ndarray: int16 bitmask
    """
    with np.errstate(invalid="ignore"):
        high_ruwe = np.asarray(ruwe, float) > _LMC_RUWE_LIMIT
    return (high_ruwe * LMC_FLAG_HIGH_RUWE + np.asarray(duplicated, bool) * LMC_FLAG_GAIA_DUPLICATE).astype(np.int16)


def _lmc_hst_rows(hst, gaia):
    """
    Catalogue rows for HST stars, with Gaia information added where Anderson linked a Gaia ID.

    Args:
        hst (astropy.table.Table): kept HST rows (_lmc_read_anderson format)
        gaia (astropy.table.Table): Gaia table

    Returns:
        astropy.table.Table: rows with the catalogue columns
    """
    from astropy.table import Table

    n = len(hst)
    rows = Table()
    rows["GAIA_ID"] = np.asarray(hst["GAIA_ID"], np.int64)
    for name in ("RA", "DEC", "RA_ERR", "DEC_ERR", "MAG"):
        rows[name] = np.asarray(hst[name], float)
    rows["MAG_BAND"] = np.full(n, "F606W")
    rows["GMAG"] = np.full(n, np.nan)
    for name in ("PMRA", "PMDEC", "PMRA_ERR", "PMDEC_ERR"):
        rows[name] = np.asarray(hst[name], float)
    for name in ("PARALLAX", "PARALLAX_ERR", "RUWE"):
        rows[name] = np.full(n, np.nan)
    rows["POS_SOURCE"] = np.full(n, "HST", dtype="<U4")
    rows["PM_SOURCE"] = np.full(n, "HST", dtype="<U4")
    rows["FLAG"] = np.zeros(n, np.int16)

    gaia_index = {int(source_id): i for i, source_id in enumerate(gaia["source_id"])}
    for i in np.flatnonzero(rows["GAIA_ID"] != 0):
        j = gaia_index.get(int(rows["GAIA_ID"][i]))
        if j is None:
            continue   # reported by _lmc_check as a truncated query
        star = gaia[j]
        rows["GMAG"][i] = star["phot_g_mean_mag"]
        rows["PARALLAX"][i] = star["parallax"]
        rows["PARALLAX_ERR"][i] = star["parallax_error"]
        rows["RUWE"][i] = star["ruwe"]
        rows["FLAG"][i] |= _lmc_quality_flags([star["ruwe"]], [star["duplicated_source"]])[0]
        # bright stars saturate in the deep HST frames; Gaia measures their motion far better
        gaia_pm_error = np.hypot(star["pmra_error"], star["pmdec_error"])
        if np.isfinite(gaia_pm_error) and gaia_pm_error < np.hypot(rows["PMRA_ERR"][i], rows["PMDEC_ERR"][i]):
            for name in ("PMRA", "PMDEC", "PMRA_ERR", "PMDEC_ERR"):
                rows[name][i] = star[name.lower().replace("_err", "_error")]
            rows["PM_SOURCE"][i] = "GAIA"
    return rows


def _lmc_gaia_rows(gaia, coords, bulk):
    """
    Catalogue rows for Gaia-only stars.

    Position errors are moved from J2016.0 to J2015.5 by adding half a year of proper-motion
    error in quadrature.

    Args:
        gaia (astropy.table.Table): the Gaia stars to add
        coords (astropy.coordinates.SkyCoord): their positions at J2015.5
        bulk (dict): output of _lmc_bulk_motion, used where Gaia has no proper motion

    Returns:
        astropy.table.Table: rows with the catalogue columns
    """
    from astropy.table import Table

    n = len(gaia)
    pmra, pmdec, known = _lmc_gaia_pm(gaia)
    pmra_error = np.where(known, np.asarray(gaia["pmra_error"], float), bulk["dispersion"])
    pmdec_error = np.where(known, np.asarray(gaia["pmdec_error"], float), bulk["dispersion"])
    years = _LMC_GAIA_EPOCH - _LMC_CATALOG_EPOCH
    gmag = np.asarray(gaia["phot_g_mean_mag"], float)

    rows = Table()
    rows["GAIA_ID"] = np.asarray(gaia["source_id"], np.int64)
    rows["RA"] = coords.ra.deg
    rows["DEC"] = coords.dec.deg
    rows["RA_ERR"] = np.hypot(np.asarray(gaia["ra_error"], float), years * pmra_error)
    rows["DEC_ERR"] = np.hypot(np.asarray(gaia["dec_error"], float), years * pmdec_error)
    rows["MAG"] = gmag
    rows["MAG_BAND"] = np.full(n, "G", dtype="<U5")
    rows["GMAG"] = gmag
    rows["PMRA"] = np.where(known, pmra, bulk["pmra"])
    rows["PMDEC"] = np.where(known, pmdec, bulk["pmdec"])
    rows["PMRA_ERR"] = pmra_error
    rows["PMDEC_ERR"] = pmdec_error
    rows["PARALLAX"] = np.asarray(gaia["parallax"], float)
    rows["PARALLAX_ERR"] = np.asarray(gaia["parallax_error"], float)
    rows["RUWE"] = np.asarray(gaia["ruwe"], float)
    rows["POS_SOURCE"] = np.full(n, "GAIA", dtype="<U4")
    rows["PM_SOURCE"] = np.where(known, "GAIA", "BULK").astype("<U4")
    rows["FLAG"] = (np.where(known, 0, LMC_FLAG_PM_ASSUMED)
                    | _lmc_quality_flags(gaia["ruwe"], gaia["duplicated_source"])).astype(np.int16)
    return rows


def _lmc_assemble(hst_all, gaia, gaia_coords, bulk):
    """
    Merge HST and Gaia DR3 into the reference catalogue (unsorted, no metadata).

    Args:
        hst_all (astropy.table.Table): all rows of Anderson's master file
        gaia (astropy.table.Table): Gaia table
        gaia_coords (astropy.coordinates.SkyCoord): Gaia positions at J2015.5
        bulk (dict): output of _lmc_bulk_motion

    Returns:
        astropy.table.Table: the catalogue columns, in the order of _LMC_COLUMNS
    """
    from astropy.table import vstack

    hst = hst_all[np.asarray(hst_all["MAG"], float) < _LMC_HST_MAG_LIMIT]
    hst_all_coords = SkyCoord(np.asarray(hst_all["RA"], float), np.asarray(hst_all["DEC"], float), unit="deg")
    hst_coords = SkyCoord(np.asarray(hst["RA"], float), np.asarray(hst["DEC"], float), unit="deg")

    # inside the HST footprint: enough Anderson stars around the Gaia star
    gaia_index, _, _, _ = hst_all_coords.search_around_sky(gaia_coords, _LMC_FOOTPRINT_RADIUS * u.arcsec)
    in_footprint = np.bincount(gaia_index, minlength=len(gaia)) >= _LMC_FOOTPRINT_MIN_STARS

    # already in the catalogue: linked by Anderson's Gaia cross-match, or on top of an HST star
    linked_ids = np.asarray(hst_all["GAIA_ID"], np.int64)
    represented = np.isin(np.asarray(gaia["source_id"], np.int64), linked_ids[linked_ids != 0])
    represented |= _lmc_near_hst_star(gaia, gaia_coords, hst, hst_coords)

    with np.errstate(invalid="ignore"):
        bright_enough = np.asarray(gaia["phot_g_mean_mag"], float) < _LMC_GAIA_MAG_LIMIT
    add = in_footprint & ~represented & bright_enough

    catalog = vstack([_lmc_hst_rows(hst, gaia), _lmc_gaia_rows(gaia[add], gaia_coords[add], bulk)])
    return catalog[list(_LMC_COLUMNS)]


def _lmc_near_hst_star(gaia, gaia_coords, hst, hst_coords):
    """
    Which Gaia stars are the same star as a kept HST star, judged by position and brightness.

    Any Gaia star within _LMC_DUPLICATE_RADIUS of an HST star is a duplicate. A star without a
    Gaia proper motion is also one if an HST star within _LMC_WIDE_DUPLICATE_RADIUS differs from
    it by less than _LMC_DUPLICATE_MAG_DIFF in magnitude (G against F606W). In the LMC field such
    pairs are far more common within 0.3 arcsec than chance alignments, and only for those stars.

    Args:
        gaia (astropy.table.Table): Gaia table
        gaia_coords (astropy.coordinates.SkyCoord): Gaia positions at J2015.5
        hst (astropy.table.Table): kept HST rows
        hst_coords (astropy.coordinates.SkyCoord): their positions

    Returns:
        numpy.ndarray: bool, one entry per Gaia star
    """
    near = np.zeros(len(gaia), bool)
    gaia_index, hst_index, separation, _ = hst_coords.search_around_sky(gaia_coords, _LMC_WIDE_DUPLICATE_RADIUS * u.arcsec)
    near[gaia_index[separation.arcsec < _LMC_DUPLICATE_RADIUS]] = True

    # two-parameter positions can be off by tens of mas: a same-brightness HST star nearby is the same star
    _, _, known = _lmc_gaia_pm(gaia)
    same_brightness = np.abs(np.asarray(gaia["phot_g_mean_mag"], float)[gaia_index]
                             - np.asarray(hst["MAG"], float)[hst_index]) < _LMC_DUPLICATE_MAG_DIFF
    near[gaia_index[same_brightness & ~known[gaia_index]]] = True
    return near


def _lmc_frame_offset(catalog, gaia, gaia_coords):
    """
    Median HST minus Gaia position offset for well-measured stars in both catalogues.

    Args:
        catalog (astropy.table.Table): output of _lmc_assemble
        gaia (astropy.table.Table): Gaia table
        gaia_coords (astropy.coordinates.SkyCoord): Gaia positions at J2015.5

    Returns:
        tuple: (median RA offset on the sky, median Dec offset) [mas]; NaN if no star qualifies
    """
    gaia_index = {int(source_id): i for i, source_id in enumerate(gaia["source_id"])}
    hst_rows, gaia_rows = [], []
    for i in np.flatnonzero((catalog["POS_SOURCE"] == "HST") & (catalog["GAIA_ID"] != 0)):
        j = gaia_index.get(int(catalog["GAIA_ID"][i]))
        if j is not None and 14 < gaia["phot_g_mean_mag"][j] < 19 and gaia["ra_error"][j] < 0.5 \
                and np.isfinite(gaia["pmra"][j]):
            hst_rows.append(i)
            gaia_rows.append(j)
    if not hst_rows:
        return float("nan"), float("nan")
    ra_hst = np.asarray(catalog["RA"][hst_rows], float)
    dec_hst = np.asarray(catalog["DEC"][hst_rows], float)
    dra = (ra_hst - gaia_coords.ra.deg[gaia_rows]) * np.cos(np.radians(dec_hst)) * 3.6e6
    ddec = (dec_hst - gaia_coords.dec.deg[gaia_rows]) * 3.6e6
    return float(np.median(dra)), float(np.median(ddec))


def _lmc_check(catalog, linked_ids, queried_ids, offset_mas):
    """
    Check the catalogue and raise if anything is wrong.

    Args:
        catalog (astropy.table.Table): the catalogue, ideally as read back from disk
        linked_ids (array-like): Gaia IDs Anderson linked to HST stars (non-zero)
        queried_ids (array-like): source_id of every star the Gaia query returned
        offset_mas (tuple): (RA, Dec) median HST minus Gaia offset [mas]

    Raises:
        RuntimeError: listing every failed check
    """
    failures = []

    ids = np.asarray(catalog["GAIA_ID"], np.int64)
    ids = ids[ids != 0]
    if len(ids) != len(np.unique(ids)):
        failures.append("duplicate Gaia IDs")

    coords = SkyCoord(np.asarray(catalog["RA"], float), np.asarray(catalog["DEC"], float), unit="deg")
    first, second, _, _ = coords.search_around_sky(coords, _LMC_DUPLICATE_RADIUS * u.arcsec)
    close_pairs = int(np.sum(first < second))
    if close_pairs:
        failures.append("{0} pairs of rows closer than {1} arcsec".format(close_pairs, _LMC_DUPLICATE_RADIUS))

    not_queried = sorted(set(np.asarray(linked_ids, np.int64).tolist())
                         - set(np.asarray(queried_ids, np.int64).tolist()))
    if not_queried:
        failures.append("Gaia query looks truncated: {0} linked IDs not returned, e.g. {1}".format(
            len(not_queried), not_queried[:3]))

    if not all(np.isfinite(offset_mas)) or max(abs(offset_mas[0]), abs(offset_mas[1])) >= _LMC_MAX_FRAME_OFFSET_MAS:
        failures.append("HST minus Gaia frame offset {0} mas exceeds {1} mas".format(
            offset_mas, _LMC_MAX_FRAME_OFFSET_MAS))

    if failures:
        raise RuntimeError("Catalogue self-checks failed:\n  " + "\n  ".join(failures))


def _lmc_metadata(catalog, anderson_sha256, gaia_meta, bulk, offset_mas, today):
    """
    ECSV metadata describing origin, selection and checks.

    Args:
        catalog (astropy.table.Table): the finished catalogue (used for counts)
        anderson_sha256 (str): checksum of the Anderson file used
        gaia_meta (dict): meta of the Gaia table (adql, queried)
        bulk (dict): output of _lmc_bulk_motion
        offset_mas (tuple): output of _lmc_frame_offset
        today (str): ISO build date

    Returns:
        dict: metadata, plain Python types only
    """
    def count(column, value):
        return int(np.sum(catalog[column] == value))

    return {
        "title": "LMC astrometric reference field for Roman CGI astrometric calibration",
        "description": ("HST ACS/WFC F606W catalogue of the JWST/Roman calibration field in the LMC "
                        "(Anderson et al. 2021), supplemented with Gaia DR3 stars that HST did not "
                        "measure. All positions at one epoch; propagate with PMRA/PMDEC to the "
                        "observation time."),
        "epoch": _LMC_CATALOG_EPOCH, "epoch_format": "jyear", "epoch_scale": "tcb", "frame": "ICRS",
        "pmra_convention": "PMRA is mu_alpha* = mu_alpha cos(dec)",
        "sources": [
            {"name": "HST JWST calibration field catalogue (isolated stars)",
             "reference": _LMC_ANDERSON_REFERENCE, "url": _LMC_ANDERSON_URL, "sha256": anderson_sha256,
             "notes": ("HST epochs 2006.39 and 2017.38 tied to the Gaia DR2 frame; positions "
                       "interpolated to 2015.5; MAG is calibrated F606W VEGAMAG")},
            {"name": "Gaia DR3", "reference": _LMC_GAIA_REFERENCE, "service": _LMC_GAIA_TAP_URL,
             "adql": gaia_meta["adql"], "queried": gaia_meta["queried"],
             "notes": ("run as half-open RA/Dec tiles of synchronous queries restricted to this cone; "
                       "row total checked against COUNT(*) of the cone; positions moved from J2016.0 to "
                       "J2015.5 with their proper motions (LMC bulk motion where Gaia has none)")},
        ],
        "selection": {
            "hst": "F606W < {0} (VEGAMAG), as in the former JWST_CALFIELD2020.csv".format(_LMC_HST_MAG_LIMIT),
            "gaia_only": ("inside the HST footprint (>= {0} HST stars within {1} arcsec), no Anderson "
                          "Gaia ID link, no kept HST star within {2} arcsec (for stars without Gaia proper "
                          "motion also none within {3} arcsec and {4} mag), G < {5}").format(
                              _LMC_FOOTPRINT_MIN_STARS, _LMC_FOOTPRINT_RADIUS, _LMC_DUPLICATE_RADIUS,
                              _LMC_WIDE_DUPLICATE_RADIUS, _LMC_DUPLICATE_MAG_DIFF, _LMC_GAIA_MAG_LIMIT),
        },
        "proper_motions": {
            "hst": "HST two-epoch proper motions on the Gaia frame (Anderson et al. 2021)",
            "gaia": "Gaia DR3, for Gaia-only stars and for HST stars whose Gaia error is smaller",
            "bulk": {"pmra": round(bulk["pmra"], 4), "pmdec": round(bulk["pmdec"], 4),
                     "dispersion": round(bulk["dispersion"], 4), "unit": "mas/yr",
                     "notes": "median and robust spread of the HST proper motions; used where Gaia has none"},
        },
        "flags": {"bit_1": "proper motion assumed (LMC bulk motion)", "bit_2": "Gaia RUWE > 1.4",
                  "bit_4": "Gaia duplicated_source"},
        "checks": {"hst_minus_gaia_median_offset_mas": [round(offset_mas[0], 4), round(offset_mas[1], 4)]},
        "build": {"function": "corgidrp.astrom.build_lmc_reference_catalog",
                  "corgidrp_version": corgidrp.__version__, "date": today},
        "counts": {"total": len(catalog), "pos_hst": count("POS_SOURCE", "HST"),
                   "pos_gaia": count("POS_SOURCE", "GAIA"), "pm_hst": count("PM_SOURCE", "HST"),
                   "pm_gaia": count("PM_SOURCE", "GAIA"), "pm_bulk": count("PM_SOURCE", "BULK")},
    }
