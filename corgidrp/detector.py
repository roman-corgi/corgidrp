# Place to put detector-related utility functions

import numpy as np
import corgidrp.data as data
from scipy import interpolate
from astropy.time import Time
from scipy.ndimage import median_filter

def create_dark_calib(dark_dataset):
    """
    Turn this dataset of image frames that were taken to measure
    the dark current into a dark calibration frame and determines the corresponding error

    Args:
        dark_dataset (corgidrp.data.Dataset): a dataset of Image frames (L2a-level)

    Returns:
        data.Dark: a dark calibration frame
    """
    combined_frame = np.nanmean(dark_dataset.all_data, axis=0)

    new_dark = data.Dark(combined_frame, pri_hdr=dark_dataset[0].pri_hdr.copy(),
                         ext_hdr=dark_dataset[0].ext_hdr.copy(), input_dataset=dark_dataset)
    
    # determine the standard error of the mean: stddev/sqrt(n_frames)
    new_dark.err = np.nanstd(dark_dataset.all_data, axis=0)/np.sqrt(len(dark_dataset))
    new_dark.err = new_dark.err.reshape((1,)+new_dark.err.shape) #Get it into the right dimensions

    return new_dark

def create_flatfield(flat_dataset):
    
    """
    Turn this dataset of image frames that were taken for performing the flat calibration and
    to make one master flat image
    this is currently a placeholder, until the final flat fielding calibration package is completed. 

    Args:
       flat_dataset (corgidrp.data.Dataset): a dataset of Image frames (L2a-level)

    Returns:
        data.masterflat: a master flat for flat calibration
    """  
    
    
    combined_frame = np.nanmean(flat_dataset.all_data, axis=0)
 
    flat_field = data.FlatField(combined_frame, pri_hdr=flat_dataset[0].pri_hdr.copy(),
                         ext_hdr=flat_dataset[0].ext_hdr.copy(), input_dataset=flat_dataset)
    
    #determine the standard error of the mean: stddev/sqrt(n_frames)
    flat_field.err = np.nanstd(flat_dataset.all_data, axis=0)/np.sqrt(len(flat_dataset))
    flat_field.err=flat_field.err.reshape((1,)+flat_field.err.shape) # Get it into the right dimension

    
    return flat_field

def get_relgains(frame, em_gain, non_lin_correction):
    """
    For a given bias subtracted frame of dn counts, return a same sized
    array of relative gain values.

    This algorithm contains two interpolations:

    - A 2d interpolation to find the relative gain curve for a given EM gain
    - A 1d interpolation to find a relative gain value for each given dn
      count value.

    Both of these interpolations are linear, and both use their edge values as
    constant extrapolations for out of bounds values.

    Parameters:
        frame (array_like): Array of dn count values.
        em_gain (float): Detector EM gain.
        non_lin_correction (corgi.drp.NonLinearityCorrection): A NonLinearityCorrection calibration file.

    Returns:
        array_like: Array of relative gain values.
    """

    # Column headers are gains, row headers are dn counts
    gain_ax = non_lin_correction.data[0, 1:]
    count_ax = non_lin_correction.data[1:, 0]
    # Array is relative gain values at a given dn count and gain
    relgains = non_lin_correction.data[1:, 1:]

    #MMB Note: This check is maybe better placed in the code that is saving the non-linearity correction file?
    # Check for increasing axes
    if np.any(np.diff(gain_ax) <= 0):
        raise ValueError('Gain axis (column headers) must be increasing')
    if np.any(np.diff(count_ax) <= 0):
        raise ValueError('Counts axis (row headers) must be increasing')
    # Check that curves (data in columns) contain or straddle 1.0
    if (np.min(relgains, axis=0) > 1).any() or \
       (np.max(relgains, axis=0) < 1).any():
        raise ValueError('Gain curves (array columns) must contain or '
                              'straddle a relative gain of 1.0')

    # Create interpolation for em gain (x), counts (y), and relative gain (z).
    # Note that this defaults to using the edge values as fill_value for
    # out of bounds values (same as specified below in interp1d)
    f = interpolate.RectBivariateSpline(gain_ax,
                                    count_ax,
                                    relgains.T,
                                    kx=1,
                                    ky=1,
    )
    # Get the relative gain curve for the given gain value
    relgain_curve = f(em_gain, count_ax)[0]

    # Create interpolation for dn counts (x) and relative gains (y). For
    # out of bounds values use edge values
    ff = interpolate.interp1d(count_ax, relgain_curve, kind='linear',
                              bounds_error=False,
                              fill_value=(relgain_curve[0], relgain_curve[-1]))
    # For each dn count, find the relative gain
    counts_flat = ff(frame.ravel())

    return counts_flat.reshape(frame.shape)

detector_areas= {
    'SCI' : {
        'frame_rows' : 1200,
        'frame_cols' : 2200,
        'image' : {
            'rows': 1024,
            'cols': 1024,
            'r0c0': [13, 1088]
            },
        'prescan' : {
            'rows': 1200,
            'cols': 1088,
            'r0c0': [0, 0]
            },
        'prescan_reliable' : {
            'rows': 1200,
            'cols': 200,
            'r0c0': [0, 800]
            },
        'parallel_overscan' : {
            'rows': 163,
            'cols': 1056,
            'r0c0': [1037, 1088]
            },
        'serial_overscan' : {
            'rows': 1200,
            'cols': 56,
            'r0c0': [0, 2144]
            },
        },
    'ENG' :{
        'frame_rows' : 2200,
        'frame_cols' : 2200,
        'image' : {
            'rows': 1024,
            'cols': 1024,
            'r0c0': [13, 1088]
            },
        'prescan' : {
            'rows': 2200,
            'cols': 1088,
            'r0c0': [0, 0]
            },
        'prescan_reliable' : {
            'rows': 2200,
            'cols': 200,
            'r0c0': [0, 800]
            },
        'parallel_overscan' : {
            'rows': 1163,
            'cols': 1056,
            'r0c0': [1037, 1088]
            },
        'serial_overscan' : {
            'rows': 2200,
            'cols': 56,
            'r0c0': [0, 2144]
            },
        },
    }

def slice_section(frame, obstype, key):
    """
    Slice 2d section out of frame

    Args:
        frame (np.ndarray): Full frame consistent with size given in frame_rows, frame_cols
        obstype (str): Keyword referencing the observation type (e.g. 'ENG' or 'SCI')
        key (str): Keyword referencing section to be sliced; must exist in detector_areas

    Returns:
        np.ndarray: a 2D array of the specified detector area
    """
    rows = detector_areas[obstype][key]['rows']
    cols = detector_areas[obstype][key]['cols']
    r0c0 = detector_areas[obstype][key]['r0c0']

    section = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
    if section.size == 0:
        raise Exception('Corners invalid. Tried to slice shape of {0} from {1} to {2} rows and {3} columns'.format(frame.shape, r0c0, rows, cols))
    return section

def plot_detector_areas(detector_areas, areas=('image', 'prescan',
        'prescan_reliable', 'parallel_overscan', 'serial_overscan')):
    """
    Create an image of the detector areas for visualization and debugging

    Args:
        detector_areas (dict): a dictionary of image constants
        areas (tuple): a tuple of areas to create masks for

    Returns:
        np.ndarray: an image of the detector areas
    """
    detector_areas = make_detector_areas(detector_areas, areas=areas)
    detector_area_image = np.zeros(
        (detector_areas['frame_rows'], detector_areas['frame_cols']), dtype=int)
    for i, area in enumerate(areas):
        detector_area_image[detector_areas[area]] = i + 1
    return detector_area_image

def detector_area_mask(detector_areas, area='image'):
    """
    Create a mask for the detector area

    Args:
        detector_areas (dict): a dictionary of image constants
        area (str): the area of the detector to create a mask for

    Returns:
        np.ndarray: a mask for the detector area
    """
    mask = np.zeros((detector_areas['frame_rows'], detector_areas['frame_cols']), dtype=bool)
    mask[detector_areas[area]['r0c0'][0]:detector_areas[area]['r0c0'][0] + detector_areas[area]['rows'],
            detector_areas[area]['r0c0'][1]:detector_areas[area]['r0c0'][1] + detector_areas[area]['cols']] = True
    return mask

def make_detector_areas(detector_areas, areas=('image', 'prescan', 'prescan_reliable',
        'parallel_overscan', 'serial_overscan')):
    """
    Create a dictionary of masks for the different detector areas

    Args:
        detector_areas (dict): a dictionary of image constants
        areas (tuple): a tuple of areas to create masks for

    Returns:
        dict: a dictionary of masks for the different detector areas
    """
    detector_areas = {}
    for area in areas:
        detector_areas[area] = detector_area_mask(detector_areas, area=area)
    return detector_areas


def flag_cosmics(cube, fwc, sat_thresh, plat_thresh, cosm_filter):
    """Identify and remove saturated cosmic ray hits and tails.

    Use sat_thresh (interval 0 to 1) to set the threshold above which cosmics
    will be detected. For example, sat_thresh=0.99 will detect cosmics above
    0.99*fwc.

    Use plat_thresh (interval 0 to 1) to set the threshold under which cosmic
    plateaus will end. For example, if plat_thresh=0.85, once a cosmic is
    detected the beginning and end of its plateau will be determined where the
    pixel values drop below 0.85*fwc.

    Use cosm_filter to determine the smallest plateaus (in pixels) that will
    be identified. A reasonable value is 2.

    Args:
        cube (array_like, float): 
            3D cube of image data (bias of zero).
        fwc (float): 
            Full well capacity of detector *in DNs*.  Note that this may require a
            conversion as FWCs are usually specified in electrons, but the image
            is in DNs at this point.
        sat_thresh (float): 
            Multiplication factor for fwc that determines saturated cosmic pixels.
        plat_thresh (float): 
            Multiplication factor for fwc that determines edges of cosmic plateu.
        cosm_filter (int): 
            Minimum length in pixels of cosmic plateus to be identified.

    Returns:
        array_like, int: 
            Mask for pixels that have been set to zero.

    Notes
    -----
    This algorithm uses a row by row method for cosmic removal. It first finds
    streak rows, which are rows that potentially contain cosmics. It then
    filters each of these rows in order to differentiate cosmic hits (plateaus)
    from any outlier saturated pixels. For each cosmic hit it finds the leading
    ledge of the plateau and kills the plateau and the rest of the row to take
    out the tail.

    |<-------- streak row is the whole row ----------------------->|
     ......|<-plateau->|<------------------tail------------------->|

    B Nemati and S Miller - UAH - 02-Oct-2018

    """
    mask = np.zeros(cube.shape, dtype=int)

    # Do a cheap prefilter for rows that don't have anything bright
    max_rows = np.max(cube, axis=-1,keepdims=True)
    ji_streak_rows = np.transpose(np.array((max_rows >= sat_thresh*fwc).nonzero()[:-1]))
    
    for j,i in ji_streak_rows:
        row = cube[j,i]

        # Find if and where saturated plateaus start in streak row
        i_beg = find_plateaus(row, fwc, sat_thresh, plat_thresh, cosm_filter)

        # If plateaus exist, kill the hit and the rest of the row
        if i_beg is not None:
            mask[j,i, i_beg:] = 1
            pass

    return mask

def find_plateaus(streak_row, fwc, sat_thresh, plat_thresh, cosm_filter):
    """Find the beginning index of each cosmic plateau in a row.

    Note that i_beg is set at one pixel before first plateau pixel, as these
    pixels immediately neighboring the cosmic plateau are very often affected
    by the cosmic hit as well.

    Args: 
        streak_row (array_like, float): 
            Row with possible cosmics.
        fwc (float): 
            Full well capacity of detector *in DNs*.  Note that this may require a
            conversion as FWCs are usually specified in electrons, but the image
            is in DNs at this point.
        sat_thresh (float): 
            Multiplication factor for fwc that determines saturated cosmic pixels.
        plat_thresh (float): 
            Multiplication factor for fwc that determines edges of cosmic plateu.
        cosm_filter (float): 
            Minimum length in pixels of cosmic plateus to be identified.

    Returns:
        array_like, int: 
            Index of plateau beginnings, or None if there is no plateau.
    """
    # Lowpass filter row to differentiate plateaus from standalone pixels
    # The way median_filter works, it will find cosmics that are cosm_filter-1
    # wide. Add 1 to cosm_filter to correct for this
    filtered = median_filter(streak_row, cosm_filter+1, mode='nearest')
    saturated = (filtered >= sat_thresh*fwc).nonzero()[0]

    if len(saturated) > 0:
        i_beg = saturated[0]
        while i_beg > 0 and streak_row[i_beg] >= plat_thresh*fwc:
            i_beg -= 1

        return i_beg
    else:
        return None
    
def calc_sat_fwc(emgain_arr,fwcpp_arr,fwcem_arr,sat_thresh):
    """Calculates the full well capacity saturation threshold for each frame.

    Args:
        emgain_arr (np.array): 1D array of the EM gain value for each frame.
        fwcpp_arr (np.array): 1D array of the full-well capacity in the image 
            frame (before em gain readout) value for each frame.
        fwcem_arr (np.array): 1D array of the full-well capacity in the EM gain 
            register for each frame.
        sat_thresh (float): Multiplier for the full-well capacity to determine 
            what qualifies as saturation. A reasonable value is 0.99

    Returns:
        np.array: _description_
    """
    possible_sat_fwcs_arr = np.append((emgain_arr * fwcpp_arr)[:,np.newaxis], fwcem_arr[:,np.newaxis],axis=1)
    sat_fwcs = sat_thresh * np.min(possible_sat_fwcs_arr,axis=1)
    
    return sat_fwcs