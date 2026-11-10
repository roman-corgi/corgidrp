import numpy as np
import os
import warnings

import corgidrp.data

import astropy
import astropy.io.ascii as ascii
import astropy.wcs
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from erfa import ErfaWarning

import pyklip.fakes as fakes

import scipy.ndimage as ndi
import scipy.optimize as optimize
from scipy.optimize import OptimizeWarning
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from astropy.convolution import convolve, Gaussian2DKernel

# LMC astrometric reference catalogue shipped with corgidrp (built by data/build_lmc_reference_catalog.py)
default_field_path = os.path.join(os.path.dirname(__file__), "data", "lmc_astrometric_reference.ecsv")


def read_field_catalog(field_path):
    """
    Read an astrometric reference catalogue of the calibration field.

    Accepts the packaged ECSV catalogue and user-supplied tables (e.g. CSV) with at least
    the columns RA, DEC and VMAG, the magnitude used for brightness ranking.

    Args:
        field_path (str): path to the catalogue file

    Returns:
        astropy.table.Table: the catalogue, with at least the columns RA, DEC [deg] and VMAG

    Raises:
        OSError: if the file is a Git LFS pointer that was never downloaded
        ValueError: if RA, DEC or VMAG is missing
    """
    with open(field_path, "rb") as catalog_file:
        start = catalog_file.read(64)
    if start.startswith(b"version https://git-lfs"):
        raise OSError("{0} is a Git LFS pointer, not the catalogue. Run `git lfs pull` in the "
                      "corgidrp repository to download it.".format(field_path))

    field = ascii.read(field_path)
    missing = [name for name in ("RA", "DEC", "VMAG") if name not in field.colnames]
    if missing:
        raise ValueError("field catalogue {0} is missing column(s) {1}".format(field_path, missing))
    return field


def catalog_at_epoch(field, mjd):
    """
    Move the catalogue positions from the catalogue epoch to the time of observation.

    Uses the proper motions PMRA (mu_alpha*) and PMDEC, in their column units or mas/yr if the
    columns have none, and the epoch recorded in the catalogue metadata (epoch, epoch_format,
    epoch_scale). Stars with a non-finite proper motion keep their catalogue position, with a
    warning. Parallax is ignored: it is ~0.02 mas for LMC members, and the few foreground stars
    in the shipped catalogue with a significant parallax (up to 12 mas, i.e. 0.5 pixel) move by
    at most that much over the year. Catalogues without proper motions or epoch (e.g.
    user-supplied CSV tables), or a missing time of observation, leave the positions unchanged,
    with a warning.
    The returned copy records the time of observation as its epoch, so propagating it again
    starts from there.

    Args:
        field (astropy.table.Table): catalogue from read_field_catalog
        mjd (float): time of observation (MJD, UTC), e.g. the MJDSRT of the frame, or None

    Returns:
        astropy.table.Table: a copy of the catalogue with RA and DEC at the time of observation and
        that time as its epoch; its meta["propagated_to_mjd"] is mjd, or None if the positions were
        not propagated
    """
    field = field.copy()
    field.meta["propagated_to_mjd"] = None
    if not {"PMRA", "PMDEC"} <= set(field.colnames):
        warnings.warn("reference catalogue has no proper motions (PMRA, PMDEC); using its positions "
                      "without propagating them to the time of observation")
        return field
    missing_epoch = [key for key in ("epoch", "epoch_format", "epoch_scale") if key not in field.meta]
    if missing_epoch:
        warnings.warn("reference catalogue has proper motions but its metadata lack {0}; using its "
                      "positions without propagating them to the time of observation".format(", ".join(missing_epoch)))
        return field
    if mjd is None:
        warnings.warn("no time of observation (MJDSRT); using the reference catalogue positions at "
                      "the catalogue epoch")
        return field

    epoch = Time(field.meta["epoch"], format=field.meta["epoch_format"], scale=field.meta["epoch_scale"])
    pmra, pmdec = (np.asarray(field[name].quantity.to_value(u.mas / u.yr) if field[name].unit is not None
                              else field[name], float) for name in ("PMRA", "PMDEC"))
    no_motion = ~(np.isfinite(pmra) & np.isfinite(pmdec))
    if no_motion.any():
        warnings.warn("{0} reference catalogue star(s) have no finite proper motion; they keep their "
                      "catalogue positions".format(int(no_motion.sum())))
        pmra, pmdec = np.where(no_motion, 0., pmra), np.where(no_motion, 0., pmdec)
    stars = SkyCoord(ra=np.asarray(field["RA"], float) * u.deg, dec=np.asarray(field["DEC"], float) * u.deg,
                     pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr, obstime=epoch, frame="icrs")
    with warnings.catch_warnings():
        # without distances ERFA treats the stars as very distant and says so
        warnings.simplefilter("ignore", category=ErfaWarning)
        moved = stars.apply_space_motion(new_obstime=Time(mjd, format="mjd", scale="utc"))
    field["RA"][:] = moved.ra.deg   # in place, so the units are kept
    field["DEC"][:] = moved.dec.deg
    field.meta.update({"epoch": float(mjd), "epoch_format": "mjd", "epoch_scale": "utc",
                       "propagated_to_mjd": float(mjd)})
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

def project_to_pixels(skycoords, crval, crpix, platescale, northangle):
    '''
    Projects sky positions to pixels with a gnomonic projection of a given plate scale and north angle.

    Args:
        skycoords (astropy.coordinates.SkyCoord): Sky positions to project
        crval (astropy.coordinates.SkyCoord): Sky position at the reference pixel
        crpix (tuple): (x, y) of the reference pixel [0-based]
        platescale (float): Plate scale [mas/ pixel]
        northangle (float): North angle [deg], as measured by compute_platescale_and_northangle

    Returns:
        pixels (np.ndarray): Array of shape (N, 2) with the x, y pixel positions
    '''
    vert_ang = np.radians(northangle)
    pc = np.array([[-np.cos(vert_ang), np.sin(vert_ang)], [np.sin(vert_ang), np.cos(vert_ang)]])
    w = astropy.wcs.WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [crval.ra.deg, crval.dec.deg]
    w.wcs.crpix = [crpix[0] + 1, crpix[1] + 1]
    w.wcs.cd = pc * (platescale * 0.001) / 3600.
    x, y = astropy.wcs.utils.skycoord_to_pixel(skycoords, wcs=w, origin=0)
    return np.array([x, y]).T

def find_common_shift(detections, predicted, max_shift, radius):
    '''
    Finds the shift that brings the most detections within a radius of a predicted position.

    Every offset between a detection and a predicted position shorter than max_shift is a candidate.

    Args:
        detections (np.ndarray): Array of shape (N, 2) with the detected x, y positions [pixels]
        predicted (np.ndarray): Array of shape (M, 2) with the predicted x, y positions [pixels]
        max_shift (float): Largest shift to consider [pixels]
        radius (float): Distance within which a detection counts as matched [pixels]

    Returns:
        shift (np.ndarray): (dx, dy) to add to the predicted positions [pixels], or None if no candidate was found
    '''
    offsets = (detections[:, None, :] - predicted[None, :, :]).reshape(-1, 2)
    offsets = offsets[np.hypot(offsets[:, 0], offsets[:, 1]) <= max_shift]
    if len(offsets) == 0:
        return None
    tree = cKDTree(predicted)
    scores = [np.sum(np.isfinite(tree.query(detections - offset, distance_upper_bound=radius)[0])) for offset in offsets]
    return offsets[np.argmax(scores)]

def refine_matches(detections, predicted, match_radius):
    '''
    Matches detections to predicted positions, refining an affine transformation between them.

    The transformation is fitted to the pairs within 4, 2 and finally 1 times match_radius, so that it absorbs
    errors in the predicted plate scale, rotation and shift. Each predicted position is then assigned at most one
    detection, the closest one within match_radius, so repeated detections of one star are dropped.

    Args:
        detections (np.ndarray): Array of shape (N, 2) with the detected x, y positions [pixels]
        predicted (np.ndarray): Array of shape (M, 2) with the predicted x, y positions [pixels]
        match_radius (float): Largest distance between a detection and its matched position [pixels]

    Returns:
        det_ind (np.ndarray): Indices of the matched detections, in their original order
        pred_ind (np.ndarray): Indices of the matched predicted positions
    '''
    empty = np.array([], dtype=int), np.array([], dtype=int)
    if len(detections) < 3 or len(predicted) < 3:
        return empty
    transformed = predicted
    for radius in (4 * match_radius, 2 * match_radius, match_radius):
        dist, ind = cKDTree(transformed).query(detections, distance_upper_bound=radius)
        matched = np.isfinite(dist)
        if np.sum(matched) < 3:
            return empty
        design = np.column_stack([predicted[ind[matched]], np.ones(np.sum(matched))])
        affine = np.linalg.lstsq(design, detections[matched], rcond=None)[0]
        transformed = np.column_stack([predicted, np.ones(len(predicted))]) @ affine

    dist, ind = cKDTree(transformed).query(detections, distance_upper_bound=match_radius)
    det_ind = []
    for i in np.where(np.isfinite(dist))[0]:
        others = np.where(ind == ind[i])[0]
        if dist[i] == np.min(dist[others]):
            det_ind.append(i)
    det_ind = np.array(det_ind, dtype=int)
    return det_ind, ind[det_ind]

def matched_platescale_and_northangle(positions, skycoords):
    '''
    Median plate scale and north angle over all pairs of matched stars, as a check of a match.

    Args:
        positions (np.ndarray): Array of shape (N, 2) with the x, y positions [pixels]
        skycoords (astropy.coordinates.SkyCoord): The matched sky positions

    Returns:
        platescale (float): Plate scale [mas/ pixel]
        northangle (float): North angle [deg], in the convention of compute_platescale_and_northangle
    '''
    i, j = np.triu_indices(len(positions), k=1)
    sep_pix = np.hypot(*(positions[j] - positions[i]).T)
    platescale = np.median(skycoords[i].separation(skycoords[j]).mas / sep_pix)
    dtheta = skycoords[i].position_angle(skycoords[j]).deg - angle_between(positions[i].T, positions[j].T)
    northangle = np.degrees(np.median(np.arctan2(np.sin(np.radians(dtheta)), np.cos(np.radians(dtheta)))))
    return platescale, northangle

def triangle_solutions(sources, brightest_field, platescale_guess, platescale_tol):
    '''
    Plate scale, north angle and anchor of the catalogue triangles that match the triangle of the three
    brightest sources, best match first.

    Args:
        sources (astropy.table.Table): Astropy table with columns 'x', 'y', brightest source first
        brightest_field (astropy.table.Table): The catalogue stars to form triangles from
        platescale_guess (float): An initial guess for the platescale value [mas/ pixel]
        platescale_tol (float): Fractional tolerance on platescale_guess

    Yields:
        platescale (float): Plate scale implied by the triangle [mas/ pixel]
        northangle (float): North angle implied by the triangle [deg]
        anchor_pix (tuple): (x, y) of the source that anchors the triangle [pixels]
        anchor_coord (astropy.coordinates.SkyCoord): Sky position of the matching catalogue star
    '''
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

    # create all combinations of triangles with the brightest field sources
    combos = list(compute_combinations(range(len(brightest_field)), r=3))

    skycoords = SkyCoord(ra= brightest_field['RA'], dec= brightest_field['DEC'], unit='deg')
    field_side_lengths = np.empty((len(combos), 3))
    lstsqs = np.full(len(combos), np.inf)

    for i, ind in enumerate(combos):
        j, k, l = ind
        s1, s2, s3 = skycoords[j], skycoords[k], skycoords[l]

        len1, len2, len3 = np.sort([s1.separation(s2).mas, s2.separation(s3).mas, s3.separation(s1).mas])
        field_side_lengths[i] = np.array([len1, len2, len3])
        perimeter = len1 + len2 + len3
        ap, bp, cp = len1/perimeter, len2/perimeter, len3/perimeter

        # make sure plate scale is within tolerance of the guess or else discard this possibility
        ratios = np.array([len1 / l1, len2 / l2, len3 / l3])
        if np.any(ratios > platescale_guess * (1 + platescale_tol)) or np.any(ratios < platescale_guess * (1 - platescale_tol)):
            continue

        lstsqs[i] = (a - ap)**2 + (b - bp)**2 + (c - cp)**2

    for best_ind in np.argsort(lstsqs):
        if not np.isfinite(lstsqs[best_ind]):
            return
        # keep track of which best fit skycoord is the farthest from the other two
        j, k, l = combos[best_ind]
        coord1, coord2, coord3 = skycoords[j], skycoords[k], skycoords[l]

        dcoord1 = coord1.separation(coord2).mas + coord1.separation(coord3).mas
        dcoord2 = coord2.separation(coord3).mas + coord2.separation(coord1).mas
        dcoord3 = coord3.separation(coord1).mas + coord3.separation(coord2).mas

        tri_dist_coords = [(dcoord1, coord1), (dcoord2, coord2), (dcoord3, coord3)]
        tri_dist_coords.sort(key=lambda a: a[0])  # sort the stars based on their distances to the other two
        short_coord, mid_coord, long_coord = tri_dist_coords[0][1], tri_dist_coords[1][1], tri_dist_coords[2][1]

        # now use the side length to separations with best fit triangle to define a pseudo plate scale
        best_l1, best_l2, best_l3 = field_side_lengths[best_ind]
        initial_platescale = np.mean(np.array([best_l1 / l1, best_l2 / l2, best_l3 / l3]))

        # find the angle between the `target` star and the other two, in the image and in the field
        rot_image = np.array([angle_between((short_star['x'], short_star['y']), (s['x'], s['y'])) for s in [mid_star, long_star]])
        rot_field = np.array([short_coord.position_angle(t).deg for t in [mid_coord, long_coord]])
        # use arctan2 to handle angle wrapping when calculating the difference between the reference field angle and the image
        dtheta1 = np.arctan2(np.sin(np.radians(rot_field[0] - rot_image[0])), np.cos(np.radians(rot_field[0] - rot_image[0])))
        dtheta2 = np.arctan2(np.sin(np.radians(rot_field[1] - rot_image[1])), np.cos(np.radians(rot_field[1] - rot_image[1])))
        initial_northangle = np.degrees(np.mean([dtheta1, dtheta2]))

        yield initial_platescale, initial_northangle, (short_star['x'], short_star['y']), short_coord

def match_sources(image, sources, field_path, comparison_threshold=50, rad=0.012, platescale_guess=21.8, platescale_tol=0.1,
                  match_radius=2., max_pointing_offset=10., northangle_tol=2., min_matches=4, min_match_fraction=0.25):
    '''
    Function to find the corresponding RA/Dec positions to image sources, given a particular field.

    The catalogue positions are first predicted from the header: the pointing RA/DEC, PA_APER as north angle and
    platescale_guess. The shift between these predictions and the sources is found by counting matches, within
    max_pointing_offset of the header pointing. If this does not give a valid match, or the header has no
    PA_APER, the triangle of the three brightest sources is matched to the catalogue instead, trying candidate
    triangles from the best match on. Either way the match is refined with an affine fit, and every catalogue
    star is matched to at most one source within match_radius. A match is valid if it has at least min_matches
    stars, matches at least the fraction min_match_fraction of the brightest 30 sources (or of all sources, if
    there are fewer), and has a plate scale within platescale_tol of platescale_guess and a north angle within
    northangle_tol of PA_APER. The fraction matters when there are many sources: some shift always lines up a
    few of them with catalogue stars by chance, and the affine refinement pulls these within match_radius, so a
    number of stars alone does not reject such a match. It is taken of the brightest sources because the faintest
    detections of a long source list are often noise.

    Args:
        image (corgidrp.data.Image): Image data as a corgidrp Image object
        sources (astropy.table.Table): Astropy table with columns 'x', 'y' as pixel locations of sources to match, brightest first
        field_path (str): Full path to the reference catalogue (see read_field_catalog)
        comparison_threshold (int): How many stars in the field to consider for the triangle match
        rad (float): The radius [deg] around the target coordinate for creating a subfield to match image sources to
        platescale_guess (float): An initial guess for the platescale value (default: 21.8 [mas/ pixel])
        platescale_tol (float): A tolerance for finding source matches within a fraction of the initial plate scale guess (default: 0.1)
        match_radius (float): Largest distance between a source and the matched catalogue position (default: 2 [pixels])
        max_pointing_offset (float): Largest offset of the stars from their positions predicted by the header (default: 10 [arcsec]);
            rad must cover the illuminated field plus this offset
        northangle_tol (float): Largest difference between the north angle of a match and PA_APER (default: 2 [deg])
        min_matches (int): Smallest number of matched stars for a valid match (default: 4)
        min_match_fraction (float): Smallest fraction of the brightest 30 sources that must be matched for a valid match (default: 0.25)

    Returns:
        matched_sources (astropy.table.Table): Astropy table with columns 'x','y','RA', 'DEC' as pixel locations of the matched sources
            and corresponding sky positons at the time of observation (MJDSRT); meta['propagated_to_mjd'] is that MJD, or None if they
            were not propagated, and meta['match_method'] is 'header' or 'triangle'

    '''
    # load the reference catalogue (checks for the columns RA, DEC and VMAG) and move the whole
    # catalogue to the time of observation, so that stars near the edge of the subfield are cut
    # at their current positions
    field = catalog_at_epoch(read_field_catalog(field_path), image.ext_hdr.get('MJDSRT'))

    # define a search field around the commanded pointing
    target = SkyCoord(ra=image.pri_hdr['RA'], dec=image.pri_hdr['DEC'], unit='deg')
    field_skycoords = SkyCoord(ra=field['RA'], dec=field['DEC'], unit='deg')
    subfield = field[field_skycoords.separation(target).deg <= rad]
    subfield_skycoords = SkyCoord(ra=subfield['RA'], dec=subfield['DEC'], unit='deg')

    detections = np.array([sources['x'], sources['y']], dtype=float).T
    pa_aper = image.pri_hdr.get('PA_APER')

    # sources are sorted brightest first; the fraction is counted among the brightest of them
    n_bright = min(len(sources), 30)
    needed_bright = int(np.ceil(min_match_fraction * n_bright))

    def valid(det_ind, pred_ind):
        if len(det_ind) < min_matches or np.sum(det_ind < n_bright) < needed_bright:
            return False
        platescale, northangle = matched_platescale_and_northangle(detections[det_ind], subfield_skycoords[pred_ind])
        if abs(platescale / platescale_guess - 1) > platescale_tol:
            return False
        if pa_aper is not None:
            dangle = np.degrees(np.arctan2(np.sin(np.radians(northangle - pa_aper)), np.cos(np.radians(northangle - pa_aper))))
            if abs(dangle) > northangle_tol:
                return False
        return True

    match_method = None
    if pa_aper is not None:
        crpix = (np.shape(image.data)[1] // 2, np.shape(image.data)[0] // 2)
        predicted = project_to_pixels(subfield_skycoords, target, crpix, platescale_guess, pa_aper)
        shift = find_common_shift(detections, predicted, max_shift=max_pointing_offset * 1000 / platescale_guess,
                                  radius=2 * match_radius)
        if shift is not None:
            det_ind, pred_ind = refine_matches(detections, predicted + shift, match_radius)
            if valid(det_ind, pred_ind):
                match_method = 'header'

    if match_method is None and len(sources) >= 3:
        bright_order_subfield = subfield.copy()
        bright_order_subfield.sort(keys='VMAG')
        brightest_field = bright_order_subfield[:comparison_threshold]
        for platescale, northangle, anchor_pix, anchor_coord in triangle_solutions(sources, brightest_field, platescale_guess, platescale_tol):
            if pa_aper is not None:
                dangle = np.degrees(np.arctan2(np.sin(np.radians(northangle - pa_aper)), np.cos(np.radians(northangle - pa_aper))))
                if abs(dangle) > northangle_tol:
                    continue
            predicted = project_to_pixels(subfield_skycoords, anchor_coord, anchor_pix, platescale, northangle)
            det_ind, pred_ind = refine_matches(detections, predicted, match_radius)
            if valid(det_ind, pred_ind):
                match_method = 'triangle'
                warnings.warn("match_sources: the stars could not be matched from the header pointing and PA_APER, "
                              "the triangle search was used instead")
                break

    if match_method is None:
        raise ValueError("match_sources found no match between the {0} sources and the catalogue: neither the header pointing "
                         "(RA/DEC, PA_APER {1}) nor the triangle search gave at least {2} stars, {3} of them among the brightest {4} sources, within {5} pixels "
                         "with a plate scale within {6:.0%} of {7} mas/pixel and a north angle within {8} deg of PA_APER"
                         .format(len(sources), pa_aper, min_matches, needed_bright, n_bright, match_radius, platescale_tol,
                                 platescale_guess, northangle_tol))

    matched_image_to_field = astropy.table.Table()
    matched_image_to_field['x'] = np.asarray(sources['x'])[det_ind]
    matched_image_to_field['y'] = np.asarray(sources['y'])[det_ind]
    matched_image_to_field['RA'] = subfield_skycoords[pred_ind].ra.value
    matched_image_to_field['DEC'] = subfield_skycoords[pred_ind].dec.value
    matched_image_to_field.meta['propagated_to_mjd'] = field.meta['propagated_to_mjd']
    matched_image_to_field.meta['match_method'] = match_method

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
    # the legendre polynomials are centred on pixel 512, as in transform_coeff_to_map and l3_to_l4.distortion_correction
    star1_pos[:,0] -= 512.
    star1_pos[:,1] -= 512.

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

def compute_platescale_and_northangle(image, source_info, center_radius=1, max_pairs=500, seed=0):
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
        max_pairs (int): Maximum number of star pairs to use. All pairs are used if there are no more than this, otherwise a random subset of this size, drawn without replacement (default: 500)
        seed (int): Seed for drawing the subset of pairs, so that the same input always gives the same result (default: 0)

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
    # create combinations of stars; a seeded generator keeps the subset, and so the result, reproducible
    all_combinations = np.array(list(compute_combinations(sub_guesses)))
    if len(all_combinations) > max_pairs:
        rand_inds = np.random.default_rng(seed).choice(len(all_combinations), size=max_pairs, replace=False)
        combo_list = all_combinations[rand_inds]
    else:
        combo_list = all_combinations


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
   
    # measure_offset can return infinite (or zero) errors, which would remove those pairs from the fit or let them dominate it
    if position_error is None:
        bad = ~(np.isfinite(xerrs) & np.isfinite(yerrs) & (xerrs > 0) & (yerrs > 0))
        if np.all(bad):
            warnings.warn("measure_offset gave no usable errors for any star pair, so the distortion is fitted with "
                          "uniform weights; set position_error to choose them")
            xerrs, yerrs = np.ones_like(xerrs), np.ones_like(yerrs)
        elif np.any(bad):
            xerrs[bad], yerrs[bad] = np.median(xerrs[~bad]), np.median(yerrs[~bad])

    # join arrays and reshape to (1, 2, N)
    offsets = np.array([dxs, dys])
    first_stars = np.array([firstxs, firstys])
    true_offsets = np.array([seps, pas])
    errs = np.array([xerrs, yerrs])

    return first_stars, offsets, true_offsets, errs

def compute_distortion(pos1, meas_offset, sky_offset, meas_errs, platescale, northangle, fitorder=3, initial_guess=None, clip_sigma=5.):
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
        clip_sigma (float): Star pairs whose residual exceeds clip_sigma times the robust scatter of the residuals are rejected and the fit is
            repeated until no more pairs are rejected; None keeps all pairs (default: 5)

    Returns:
        distortion_coeffs (tuple): The legendre coefficients (np.array) and polynomial order used for the fit (int)

    '''
    # define fitting params            
    fitparams = (fitorder + 1)**2
    
    # initial guesses for the legendre coeffs if none are passed
    if initial_guess is None:
        initial_guess = [0 for _ in range(fitorder+1)] + [500,] + [0 for _ in range(fitparams - fitorder - 2)] + [0,500] + [0 for _ in range(fitparams-2)]
    
    pos1, meas_offset, sky_offset, meas_errs = np.asarray(pos1), np.asarray(meas_offset), np.asarray(sky_offset), np.asarray(meas_errs)
    npairs = pos1.shape[1]
    keep = np.ones(npairs, dtype=bool)
    distortion_coeffs = np.array(initial_guess, dtype=float)
    for _ in range(10):
        (distortion_coeffs, _, _, message, ier) = optimize.leastsq(fit_distortion_solution, distortion_coeffs,
                                                                     args=(fitorder, platescale, northangle, pos1[:, keep],
                                                                           meas_offset[:, keep], sky_offset[:, keep], meas_errs[:, keep]),
                                                                     full_output=True)
        if ier not in (1, 2, 3, 4):
            warnings.warn("The distortion fit did not converge: " + message)
        if clip_sigma is None:
            break
        residuals = fit_distortion_solution(distortion_coeffs, fitorder, platescale, northangle, pos1, meas_offset, sky_offset, meas_errs)
        pair_residuals = np.hypot(residuals[:npairs], residuals[npairs:])
        # for 2D gaussian residuals the median of their length is 1.1774 sigma
        scatter = np.median(pair_residuals[keep]) / 1.1774
        new_keep = pair_residuals <= clip_sigma * scatter
        if np.array_equal(new_keep, keep) or np.sum(new_keep) < len(distortion_coeffs):
            break
        keep = new_keep

    if np.allclose(distortion_coeffs, initial_guess):
        warnings.warn("The distortion fit returned its initial guess; check the star pairs and their errors")

    return (distortion_coeffs, fitorder)
  
  
def boresight_calibration(input_dataset, field_path='lmc_astrometric_reference.ecsv', field_matches=None, find_threshold=30, fwhm=7, mask_rad=1, 
                          comparison_threshold=50, search_rad=0.012, platescale_guess=21.8, platescale_tol=0.1, center_radius=0.9, 
                          frames_to_combine=False, find_distortion=False, fitorder=3, position_error=None, initial_dist_guess=None, 
                          pa_tolerance=0.1, keywords_to_split_dataset_by=None, pointing_tolerance=0.5, fsm_tolerance=5.,
                          match_radius=2., max_pointing_offset=10., northangle_tol=2., min_match_fraction=0.25):
    """
    Perform the boresight calibration of a dataset.
    
    Args:
        input_dataset (corgidrp.data.Dataset): Dataset containing a images for astrometric calibration
        field_path (str): Full path to the reference catalogue, or the default 'lmc_astrometric_reference.ecsv' for the catalogue shipped with corgidrp (default: 'lmc_astrometric_reference.ecsv'). If the catalogue has PMRA/PMDEC and epoch metadata, its positions are propagated to the MJDSRT of each frame (see catalog_at_epoch)
        field_matches (list of str or astropy.table.Table): List of full paths to files or astropy tables with calibration field matches for each image in the dataset (x, y, ra, dec), if single str the same filepath used for all frames,nif None, automated source matching is used (default: None)
        find_threshold (int): Number of sources to find, brightest first; about 30 covers the stars of the LMC field without adding noise detections (default 30)
        fwhm (float): Full width at half maximum of the stellar psf (default: 7, ~fwhm for a normal distribution with sigma=3)
        mask_rad (int): Radius of mask for stars [in fwhm] (default: 1)
        comparison_threshold (int): How many stars in the field to consider for the triangle match, which is only used if the match from the header pointing and PA_APER fails (default: 50)
        search_rad (float): The radius [deg] around the target coordinate for creating a subfield to match image sources to; it only needs to cover the illuminated field plus max_pointing_offset
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
        pointing_tolerance (float, optional): Maximum gap [arcsec on the sky] between RA_APER/DEC_APER values that are still grouped together when frames_to_combine is True; these keywords scatter with telemetry noise (Default: 0.5)
        fsm_tolerance (float, optional): Maximum gap [mas] between FSMX/FSMY values that are still grouped together when frames_to_combine is True (Default: 5)
        match_radius (float, optional): Largest distance between a detected source and the matched catalogue star [pixels] (Default: 2)
        max_pointing_offset (float, optional): Largest offset of the stars from their positions predicted by the header pointing [arcsec]; search_rad must cover the illuminated field plus this offset (Default: 10)
        northangle_tol (float, optional): Largest difference between the north angle of a match and PA_APER [deg] (Default: 2)
        min_match_fraction (float, optional): Smallest fraction of the brightest 30 detected sources that must be matched to catalogue stars for a valid match (Default: 0.25)

    Returns:
        corgidrp.data.AstrometricCalibration: Astrometric Calibration data object containing image center coords in (RA,DEC), platescale, and north angle
        
    """
    # load in the data considering multiple frames in the data
    dataset = input_dataset.copy()

    # load in the source matches if automated source finder is not being used
    matched_sources_multiframe = []
    if isinstance(field_matches, (str, astropy.table.Table)):
        field_matches = [field_matches]
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
            # split by target first, so that the RA tolerance below is computed at the declination of each field
            if 'TARGET' in prihdr_keyword_list:
                target_datasets, _ = pa_dataset.split_dataset(prihdr_keywords=['TARGET'])
            else:
                target_datasets = [pa_dataset]
            for target_dataset in target_datasets:
                # RA_APER/DEC_APER and FSMX/FSMY are telemetry and scatter from frame to frame, so group
                # values within a tolerance rather than exactly. RA_APER differences are coordinate degrees;
                # dividing by cos(DEC_APER) makes the tolerance an on-sky distance
                dec_aper = np.median([frame.pri_hdr.get('DEC_APER', 0.) for frame in target_dataset])
                tolerances = {'RA_APER': pointing_tolerance / 3600 / np.cos(np.radians(dec_aper)),
                              'DEC_APER': pointing_tolerance / 3600, 'FSMX': fsm_tolerance, 'FSMY': fsm_tolerance}
                tolerances = {key: value for key, value in tolerances.items()
                              if key in prihdr_keyword_list + exthdr_keyword_list}
                dither_datasets, _ = target_dataset.split_dataset(prihdr_keywords=prihdr_keyword_list,
                                                                  exthdr_keywords=exthdr_keyword_list, tolerances=tolerances)
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

    if len(matched_sources_multiframe) not in (0, len(dataset)):
        if len(field_matches) > 1:
            raise ValueError('field_matches has one table per input frame, but the frames were combined into {0} frames; '
                             'give a single table or set frames_to_combine=False'.format(len(dataset)))
        matched_sources_multiframe = matched_sources_multiframe[:1] * len(dataset)

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
        if len(matched_sources_multiframe) > 0:
            matched_sources = matched_sources_multiframe[i]
        else:
            found_sources = find_source_locations(image, threshold=find_threshold, fwhm=fwhm, mask_rad=mask_rad)
            matched_sources = match_sources(dataset[i], found_sources, field_path, comparison_threshold=comparison_threshold, rad=search_rad, platescale_guess=platescale_guess, platescale_tol=platescale_tol,
                                            match_radius=match_radius, max_pointing_offset=max_pointing_offset, northangle_tol=northangle_tol,
                                            min_match_fraction=min_match_fraction)
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
        # the matches belong to the (combined) frames in dataset
        first_stars, offsets, true_offsets, errs = format_distortion_inputs(dataset, source_matches=hold_matches, position_error=position_error)
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
    if int(frames_to_combine) == 1:
        grouping = "grouped by header keywords"
    elif int(frames_to_combine) > 1:
        grouping = "grouped in sets of up to {0} consecutive frames".format(int(frames_to_combine))
    else:
        grouping = "not combined"
    group_sizes = [len(group) for group in grouped_datasets] if len(grouped_datasets) > 0 else [1] * len(dataset)
    history_msg = ("Boresight calibration completed from {0} frames ({1}) combined into {2} groups of {3} frames"
                   .format(len(input_dataset), grouping, len(group_sizes), ", ".join(str(size) for size in group_sizes)))
    catalog_mjds = [matches.meta.get('propagated_to_mjd') for matches in hold_matches]
    propagated_mjds = [mjd for mjd in catalog_mjds if mjd is not None]
    if len(propagated_mjds) > 0:
        mjd_range = "{0:.4f}".format(min(propagated_mjds))
        if max(propagated_mjds) > min(propagated_mjds):
            mjd_range += " to {0:.4f}".format(max(propagated_mjds))
        measured = ("each measured frame" if len(propagated_mjds) == len(catalog_mjds)
                    else "{0} of the {1} measured frames".format(len(propagated_mjds), len(catalog_mjds)))
        history_msg += ("; reference catalogue positions propagated with proper motions to the MJDSRT "
                        "of {0} (MJD {1})".format(measured, mjd_range))
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
