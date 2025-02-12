# Place to put detector-related utility functions

import numpy as np
from scipy import interpolate
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter as gauss
from scipy import ndimage
from scipy.signal import convolve2d
import astropy.io.fits as fits
from astropy.convolution import convolve_fft
import photutils.centroids as centr
from photutils.aperture import CircularAperture

import corgidrp.data as data

def create_flatfield(flat_dataset):

    """
    Turn this dataset of image frames that were taken for performing the flat calibration and
    to make one master flat image
    this is currently a placeholder, until the final flat fielding calibration package is completed.

    Args:
       flat_dataset (corgidrp.data.Dataset): a dataset of Image frames (L2a-level)

    Returns:
        flat_field (corgidrp.data.FlatField): a master flat for flat calibration
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
    if non_lin_correction is None: # then no correction
        return np.ones_like(frame) 
    
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
            'r0c0': [0, 0],
            'col_start': 800,
            'col_end': 1000,
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
            'r0c0': [0, 0],
            'col_start': 800,
            'col_end': 1000,
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
    'ENG_EM' :{
        'frame_rows' : 2200,
        'frame_cols' : 2200,
        'image' : { # combined lower and upper
            'rows': 2048,
            'cols': 1024,
            'r0c0': [13, 1088]
            },
        'lower_image' : {
            'rows': 1024,
            'cols': 1024,
            'r0c0': [13, 1088]
            },
        'upper_image' : {
            'rows': 1024,
            'cols': 1024,
            'r0c0': [1037, 1088]
            },
        'prescan' : {
            'rows': 2200,
            'cols': 1088,
            'r0c0': [0, 0],
            'col_start': 800,
            'col_end': 1000
            },
        'prescan_reliable' : {
            'rows': 2200,
            'cols': 200,
            'r0c0': [0, 800]
            },
        'parallel_overscan' : {
            'rows': 130,
            'cols': 1056,
            'r0c0': [2070, 1088]
            },
        'serial_overscan' : {
            'rows': 2200,
            'cols': 56,
            'r0c0': [0, 2144]
            },
        },
    'ENG_CONV' :{
        'frame_rows' : 2200,
        'frame_cols' : 2200,
        'image' : { # combined lower and upper
            'rows': 2048,
            'cols': 1024,
            'r0c0': [13, 48]
            },
        'lower_image' : {
            'rows': 1024,
            'cols': 1024,
            'r0c0': [13, 48]
            },
        'upper_image' : {
            'rows': 1024,
            'cols': 1024,
            'r0c0': [1037, 48]
            },
        # 'prescan' is actually the serial_overscan region, but the code needs to take
        # the bias from the largest serial non-image region, and the code identifies
        # this region as the "prescan", so we have the prescan and serial_overscan
        # names flipped for this reason.
        'prescan' : {
            'rows': 2200,
            'cols': 1128,
            'r0c0': [0, 1072],
            'col_start': 1200,
            'col_end': 1400
            },
        'prescan_reliable' : {
            # not sure if these are good in the eng_conv case where the geometry is
            # flipped relative to the other cases, but these cols would where the
            # good, reliable cols used for getting row-by-row bias
            # would be
            'rows': 2200,
            'cols': 200,
            'r0c0': [0, 1200]
            },
        'parallel_overscan' : {
            'rows': 130,
            'cols': 1056,
            'r0c0': [2070, 16]
            },
        'serial_overscan' : {
            'rows': 2200,
            'cols': 16,
            'r0c0': [0, 0]
            },
        }
    }


def slice_section(frame, arrtype, key, detector_regions=None):

    """
    Slice 2d section out of frame

    Ported from II&T read_metadata.py

    Args:
        frame (np.ndarray): Full frame consistent with size given in frame_rows, frame_cols
        arrtype (str): Keyword referencing the observation type (e.g. 'ENG' or 'SCI')
        key (str): Keyword referencing section to be sliced; must exist in detector_areas
        detector_regions (dict): a dictionary of detector geometry properties.  Keys should be as found in detector_areas in detector.py.  Defaults to that dictionary.

    Returns:
        np.ndarray: a 2D array of the specified detector area
    """
    if detector_regions is None:
            detector_regions = detector_areas
    rows = detector_regions[arrtype][key]['rows']
    cols = detector_regions[arrtype][key]['cols']
    r0c0 = detector_regions[arrtype][key]['r0c0']

    section = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
    if section.size == 0:
        raise Exception('Corners invalid. Tried to slice shape of {0} from {1} to {2} rows and {3} columns'.format(frame.shape, r0c0, rows, cols))
    return section

def embed(data, arrtype, key, pad_val=0, detector_regions=None):
        '''
        Embed subframe data into a full frame with a specified value per pixel everywhere else. 
        
        Args:
            data (np.ndarray):  Subframe data to embed
            arrtype (str): Keyword referencing the observation type (e.g. 'ENG' or 'SCI')
            key (str): Keyword referencing section to be sliced; must exist in detector_areas
            pad_val (float): Value to fill in each pixel outside the subframe region
            detector_regions (dict): a dictionary of detector geometry properties.  Keys should be as found in detector_areas in detector.py.  Defaults to that dictionary.

        Returns:
            np.ndarray: a 2D array of the full detector area with embedded subframe
        '''
        if detector_regions is None:
            detector_regions = detector_areas
        ff_rows = detector_regions[arrtype]['frame_rows']
        ff_cols = detector_regions[arrtype]['frame_cols']
        full_frame = pad_val*np.ones((ff_rows, ff_cols))
        rows, cols, r0c0 = unpack_geom(arrtype, key, detector_regions)
        try:
            full_frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = data
        except Exception:
            raise Exception('Data does not fit in selected section')
        return full_frame

def unpack_geom(arrtype, key, detector_regions=None):
    """Safely check format of geom sub-dictionary and return values.

    Args:
        arrtype: str
        Keyword referencing the observation type (e.g. 'ENG' or 'SCI')
        key: str
        Desired section
        detector_regions: dict
        a dictionary of detector geometry properties.  Keys should be as found in detector_areas in detector.py.  Defaults to that dictionary.

    Returns:
        rows: int
        Number of rows of frame
        cols : int
        Number of columns of frame
        r0c0: tuple
        Tuple of (row position, column position) of corner closest to (0,0)
    """
    if detector_regions is None:
        detector_regions = detector_areas
    coords = detector_regions[arrtype][key]
    rows = coords['rows']
    cols = coords['cols']
    r0c0 = coords['r0c0']

    return rows, cols, r0c0

def imaging_area_geom(arrtype, detector_regions=None):
    """Return geometry of imaging area (including shielded pixels)
    in reference to full frame.  Different from normal image area.

    Args:
        arrtype: str
        Keyword referencing the observation type (e.g. 'ENG' or 'SCI')
        detector_regions: dict
        a dictionary of detector geometry properties.  Keys should be as found in detector_areas in detector.py.  Defaults to that dictionary.


    Returns:
        rows: int
        Number of rows of imaging area
        cols : int
        Number of columns of imaging area
        r0c0: tuple
        Tuple of (row position, column position) of corner closest to (0,0)
    """
    if detector_regions is None:
        detector_regions = detector_areas
    _, cols_pre, _ = unpack_geom(arrtype, 'prescan', detector_regions)
    _, cols_serial_ovr, _ = unpack_geom(arrtype, 'serial_overscan', detector_regions)
    rows_parallel_ovr, _, _ = unpack_geom(arrtype, 'parallel_overscan', detector_regions)
    #_, _, r0c0_image = self._unpack_geom('image')
    fluxmap_rows, _, r0c0_image = unpack_geom(arrtype, 'image', detector_regions)

    rows_im = detector_regions[arrtype]['frame_rows'] - rows_parallel_ovr
    cols_im = detector_regions[arrtype]['frame_cols'] - cols_pre - cols_serial_ovr
    r0c0_im = r0c0_image.copy()
    r0c0_im[0] = r0c0_im[0] - (rows_im - fluxmap_rows)

    return rows_im, cols_im, r0c0_im

def imaging_slice(arrtype, frame, detector_regions=None):
    """Select only the real counts from full frame and exclude virtual.
    Includes shielded pixels.

    Use this to transform mask and embed from acting on the full frame to
    acting on only the image frame.

    Args:
        arrtype: str
        Keyword referencing the observation type (e.g. 'ENG' or 'SCI')
        frame: array_like
        Input frame
        detector_regions: dict
        a dictionary of detector geometry properties.  Keys should be as found in detector_areas in detector.py.  Defaults to that dictionary.

    Returns:
        sl: array_like
        Imaging slice

    """
    rows, cols, r0c0 = imaging_area_geom(arrtype, detector_regions)
    sl = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
    return sl

def flag_cosmics(cube, fwc, sat_thresh, plat_thresh, cosm_filter, cosm_box,
                   cosm_tail, mode='image', detector_regions=None, arrtype='SCI'):
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
        cosm_box (int):
            Number of pixels out from an identified cosmic head (i.e., beginning of
            the plateau) to mask out.
            For example, if cosm_box is 3, a 7x7 box is masked,
            with the cosmic head as the center pixel of the box.
        cosm_tail (int):
            Number of pixels in the row downstream of the end of a cosmic plateau
            to mask.  If cosm_tail is greater than the number of
            columns left to the end of the row from the cosmic
            plateau, the cosmic masking ends at the end of the row.
        mode (string):
            If 'image', an image-area input is assumed, and if the input
            tail length is longer than the length to the end of the image-area row,
            the mask is truncated at the end of the row.
            If 'full', a full-frame input is assumed, and if the input tail length
            is longer than the length to the end of the full-frame row, the masking
            continues onto the next row.  Defaults to 'image'.
        detector_regions: (dict):  
            A dictionary of detector geometry properties.  Keys should be as 
            found in detector_areas in detector.py. Defaults to detector_areas in detector.py.
        arrtype (string):
            'ARRTYPE' from the header associated with the input data cube.  Defaults to 'SCI'.

    Returns:
        array_like, int:
            Mask for pixels that have been set to zero.

    Notes
    -----
    This algorithm uses a row by row method for cosmic removal. It first finds
    streak rows, which are rows that potentially contain cosmics. It then
    filters each of these rows in order to differentiate cosmic hits (plateaus)
    from any outlier saturated pixels. For each cosmic hit it finds the leading
    ledge of the plateau and kills the plateau (specified by cosm_filter) and
    the tail (specified by cosm_tail).

    |<-------- streak row is the whole row ----------------------->|
     ......|<-plateau->|<------------------tail---------->|.........

    B Nemati and S Miller - UAH - 02-Oct-2018
    Kevin Ludwick - UAH - 2024

    """
    mask = np.zeros(cube.shape, dtype=int)

    if detector_regions is None:
        detector_regions = detector_areas

    if mode=='full':
        im_num_rows = detector_regions[arrtype]['image']['rows']
        im_num_cols = detector_regions[arrtype]['image']['cols']
        im_starting_row = detector_regions[arrtype]['image']['r0c0'][0]
        im_ending_row = im_starting_row + im_num_rows
        im_starting_col = detector_regions[arrtype]['image']['r0c0'][1]
        im_ending_col = im_starting_col + im_num_cols
    else:
        im_starting_row = 0
        im_ending_row = mask.shape[1] - 1 # - 1 to get the index, not size
        im_starting_col = 0
        im_ending_col = mask.shape[2] - 1 # - 1 to get the index, not size

    # Do a cheap prefilter for rows that don't have anything bright
    max_rows = np.max(cube, axis=-1,keepdims=True)
    ji_streak_rows = np.transpose(np.array((max_rows >= sat_thresh*fwc).nonzero()[:-1]))

    for j,i in ji_streak_rows:
        row = cube[j,i]
        if i < im_starting_row or i > im_ending_row:
            continue
        # Find if and where saturated plateaus start in streak row
        i_begs = find_plateaus(row, fwc, sat_thresh, plat_thresh, cosm_filter)

        # If plateaus exist, kill the hit and the tail
        cutoffs = np.array([])
        ex_l = np.array([])
        if i_begs is not None:
            for i_beg in i_begs:
                if i_beg < im_starting_col or i_beg > im_ending_col:
                    continue
                # implement cosm_tail
                if i_beg+cosm_filter+cosm_tail+1 > mask.shape[2]:
                    ex_l = np.append(ex_l,
                            i_beg+cosm_filter+cosm_tail+1-mask.shape[2])
                    cutoffs = np.append(cutoffs, i+1)
                streak_end = int(min(i_beg+cosm_filter+cosm_tail+1,
                                mask.shape[2]))
                mask[j, i, i_beg:streak_end] = 1
                # implement cosm_box
                # can't have cosm_box appear in non-image pixels
                st_row = max(i-cosm_box, im_starting_row)
                end_row = min(i+cosm_box+1, im_ending_row+1)
                st_col = max(i_beg-cosm_box, im_starting_col)
                end_col = min(i_beg+cosm_box+1, im_ending_col+1)
                mask[j, st_row:end_row, st_col:end_col] = 1
                pass

        if mode == 'full' and len(ex_l) > 0:
            mask_rav = mask[j].ravel()
            for k in range(len(ex_l)):
                row = cutoffs[k]
                rav_ind = int(row * mask.shape[2] - 1)
                mask_rav[rav_ind:rav_ind + int(ex_l[k])] = 1


    return mask

def find_plateaus(streak_row, fwc, sat_thresh, plat_thresh, cosm_filter):
    """Find the beginning index of each cosmic plateau in a row.

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
            Multiplication factor for fwc that determines edges of cosmic plateau.
        cosm_filter (float):
            Minimum length in pixels of cosmic plateaus to be identified.

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
        i_begs = np.array([])
        for i in range(len(saturated)):
            i_beg = saturated[i]
            while i_beg > 0 and streak_row[i_beg] >= plat_thresh*fwc:
                i_beg -= 1
            # unless saturated at col 0, shifts forward 1 to plateau start
            if streak_row[i_beg] < plat_thresh*fwc:
                i_beg += 1
            i_begs = np.append(i_begs, i_beg)

        return np.unique(i_begs).astype(int)
    else:
        return None
    
def calc_sat_fwc(emgain_arr,fwcpp_arr,fwcem_arr,sat_thresh):
	"""Calculates the lowest full well capacity saturation threshold for each frame.

	Args:
    	emgain_arr (np.array): 1D array of the EM gain value for each frame.
        fwcpp_arr (np.array): 1D array of the full-well capacity in the image
            frame (before em gain readout) value for each frame.
        fwcem_arr (np.array): 1D array of the full-well capacity in the EM gain
            register for each frame.
        sat_thresh (float): Multiplier for the full-well capacity to determine
            what qualifies as saturation. A reasonable value is 0.99

    Returns:
        np.array: lowest full well capacity saturation threshold for frames
    """
	possible_sat_fwcs_arr = np.append((emgain_arr * fwcpp_arr)[:,np.newaxis], fwcem_arr[:,np.newaxis],axis=1)
	sat_fwcs = sat_thresh * np.min(possible_sat_fwcs_arr,axis=1)

	return sat_fwcs

def raster_kernel(width, image, hard=True):
    """
    Convolution kernel to create flat field circular raster pattern

    Args:
        width (float): radius of circular raster in pixels
        image (np.array): 2-D image to specify the full size of the kernel needed
        hard (bool): if true, use hard edge kernel, otherwise, use Gaussian tapered kernel
    
    Returns:
        np.array: smoothing kernel value at each pixel
    """

    kernel_width = width
    im = image

    # define coordinate grid
    x = np.arange(0,im.shape[1] + 1) - im.shape[1]/2 - 1
    y = np.arange(0,im.shape[0] + 1) - im.shape[0]/2 - 1
    xx,yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    rr_hard = np.sqrt(xx**2 + yy**2)

    # Define the convolution kernel
    if hard == True:
        rr_hard[rr_hard<=kernel_width/2] = 1
        rr_hard[rr_hard>kernel_width/2] = 0
        kernel = rr_hard # option 1: hard-edge kernel
    else:
        kernel = np.exp(-rr**2/(2*kernel_width**2)) # option 2: Gaussian kernel

    return kernel

def flatfield_residuals(images, N=None):
    """Turn this dataset of image frames of neptune or uranus and create matched filters and estimate residuals after 
     dividing from matched filters

     Todo: Propoagate the errors: incorporate the errors in the raster frames of neptune and jupiter and the errors in the matched filter

     Args:
    	images (np.array): 2D array of cropped neptune or uranus image frames
        N (int): number of images to be grouped. defaults to 3 for both Neptune and uranus. (If we use different number of dithers for neptune and uranus, option is provided to change N)

     Returns:
    	matched_residuals (np.array): residual image frames of neptune or uranus divided by matched filter
	"""
    raster_images = np.array(images)
    images_split = np.array(np.split(np.array(raster_images),N))
    matched_filters = np.array([np.nanmedian(np.stack(images_split[i],2),2) for i in np.arange(0,len(images_split))])
    matched_filters_smooth = [gauss(matched_filters[i],3) for i in range(len(matched_filters))] 
    
    matched_residuals=[];
    matched_residuals_err=[]
    for j in range(len(raster_images)):
        n_images=int(np.floor(j//(len(raster_images)//len(matched_filters_smooth))))
        matched_residuals.append(raster_images[j]/matched_filters_smooth[n_images])
    return(matched_residuals)
	    
def combine_flatfield_rasters(residual_images,cent=None,planet=None,band=None,im_size=420,rad_mask=None,  planet_rad=None, n_pix=165, n_pad=302):
    """combine the dataset of residual image frames of neptune or uranus and create flat field 
    	and associated error

    	Args:
        	residual_images (np.array): Residual images frames divided by the mnatched filter of neptune and uranus
            cent (np.array): centroid of the image frames
        	planet (str):   name of the planet neptune or uranus
        	band (str):  band of the observation band1 or band4
            im_size (int): x-dimension of the planet image (in pixels= 420 for the HST images)
            rad_mask (float): radius in pixels used for creating a mask for band (band 1 =1.25, band 4=1.75)
            planet_rad (int): radius of the planet in pixels (planet_rad=54 for neptune, planet_rad=65)
            n_pix (int): Number of pixels in radius covering the Roman CGI imaging FOV defaults to 165 pixels
            n_pad (int): Number of pixels padded with '1s'  to generate the image size 1024X1024 pixels around imaging FOV (defaults to 302 pixels)

        	
    	Returns:
        	full_residuals (np.array): flat field created from uranus or neptune images
        	err_residuals (np.array):  Error in the flatfield estimated using the ideal flat field
    """
    n = im_size
    
    full_residuals = np.zeros((n,n))
    err_residuals= np.zeros((n,n))
    if planet_rad is None:
        if planet.lower() == 'neptune':
             planet_rad = 50
        elif planet.lower() == 'uranus':
             planet_rad = 65
    
    if rad_mask is None:
         if band == 1:
            rad_mask = 1.25
         elif band == 4:
            rad_mask = 1.75
    
    aperture = CircularAperture((np.ceil(rad_mask), np.ceil(rad_mask)), r=rad_mask)
    mask= aperture.to_mask().data
    rad = planet_rad
    for i in np.arange(len(residual_images)):
        nx = np.arange(0,residual_images[i].shape[1])
        ny = np.arange(0,residual_images[i].shape[0])
        nxx,nyy = np.meshgrid(nx,ny)
        nrr = np.sqrt((nxx-rad-5)**2 + (nyy-rad-5)**2)
        nrr_copy = nrr.copy();  nrr_err_copy=nrr.copy()
        nrr_copy[nrr<rad] = residual_images[i][nrr<rad]
        nrr_copy[nrr>=rad] = None
        ymin = int(cent[i][0])
        ymax = int(cent[i][1])
        xmin = int(cent[i][2])
        xmax = int(cent[i][3])
        
        bool_innotzero = np.logical_and(nrr<rad,full_residuals[ymin:ymax,xmin:xmax]!=0)
        bool_iniszero = np.logical_and(nrr<rad,full_residuals[ymin:ymax,xmin:xmax]==0)
        bool_outisnotzero = np.logical_and(nrr>=rad,full_residuals[ymin:ymax,xmin:xmax]!=0)
        
        full_residuals[ymin:ymax,xmin:xmax][bool_innotzero] = (nrr_copy[bool_innotzero] + full_residuals[ymin:ymax,xmin:xmax][bool_innotzero])/2
        full_residuals[ymin:ymax,xmin:xmax][bool_iniszero] = nrr_copy[bool_iniszero]
        full_residuals[ymin:ymax,xmin:xmax][bool_outisnotzero] = full_residuals[ymin:ymax,xmin:xmax][bool_outisnotzero]

    
        full_residuals_resel = ndimage.convolve(full_residuals,mask)
        
        
    
    full_residuals[full_residuals==0] = None
    nx = np.arange(0,full_residuals_resel.shape[1])
    ny = np.arange(0,full_residuals_resel.shape[0])
    nxx,nyy = np.meshgrid(ny,nx)
    nrr = np.sqrt((nxx-n/2)**2 + (nyy-n/2)**2)
    full_residuals[nrr>n_pix]= 1
    
    
    full_residuals=np.pad(full_residuals, ((n_pad,n_pad),(n_pad,n_pad)), mode='constant',constant_values=(1))
    err_residuals=np.pad(err_residuals, ((n_pad,n_pad),(n_pad,n_pad)), mode='constant',constant_values=(0))
    
    return (full_residuals,err_residuals)
    
    
def create_onsky_flatfield(dataset, planet=None,band=None,up_radius=55,im_size=None,N=1,rad_mask=None, planet_rad=None, n_pix=44, n_pad=None, sky_annulus_rin=2, sky_annulus_rout=4):
    """Turn this dataset of image frames of uranus or neptune raster scannned that were taken for performing the flat calibration and create one master flat image. 
    The input image frames are L2b image frames that have been dark subtracted, divided by k-gain, divided by EM gain, desmeared. 

    
        Args:
            dataset (corgidrp.data.Dataset): a dataset of image frames that are raster scanned (L2a-level)
            planet (str): neptune or uranus
            band (str): 1 or 4
            up_radius (int): Number of pixels on either side of centroided planet images (=55 pixels for Neptune and uranus)
            im_size (int): x-dimension of the input image (in pixels; default is size of input dataset; = 420 for the HST images)
            N (int): Number of images to be combined for creating a matched filter (defaults to 1, may not work for N>1 right now)
            rad_mask (float): radius in pixels used for creating a mask for band (band1=1.25, band4=1.75)
            planet_rad (int): radius of the planet in pixels (planet_rad=50 for neptune, planet_rad=65)
            n_pix (int): Number of pixels in radius covering the Roman CGI imaging FOV (defaults to 44 pix for Band1 HLC; 165 pixels for full shaped pupil FOV).
            n_pad (int): Number of pixels padded with '1s'  to generate the image size 1024X1024 pixels around imaging FOV (defaults to None; rest of the FOV to reach 1024)
            sky_annulus_rin (float): Inner radius of annulus to use for sky subtraction. In units of planet_rad. 
                                     If both sky_annulus_rin and sky_annulus_rout = None, skips sky subtraciton.
            sky_annulus_rout (float): Outer radius of annulus to use for sky subtraction. In units of planet_rad. 
            
    	Returns:
    		data.FlatField (corgidrp.data.FlatField): a master flat for flat calibration using on sky images of planet in band specified
    		
	"""
    if im_size is None:
        # assume square images
        im_size = dataset[0].data.shape[0]

    if n_pad is None:
        n_pad = 1024 - im_size
        if n_pad < 0:
            n_pad = 0 

    if planet is None:
         planet=dataset[0].pri_hdr['TARGET']
    if band is None:
         band=dataset[0].pri_hdr['FILTER']
    
    if planet_rad is None:
        if planet.lower() =='neptune':
             planet_rad = 50
        elif planet.lower() == 'uranus':
             planet_rad = 65
    
    if rad_mask is None:
         if band == 1:
            rad_mask = 1.25
         elif band == 4:
            rad_mask = 1.75

    smooth_images=[]; raster_images_cent=[]; cent=[]; act_cents=[]; frames=[];
    for j in range(len(dataset)):
        planet_image=dataset[j].data
        prihdr=dataset[j].pri_hdr
        exthdr=dataset[j].ext_hdr
        image_size=np.shape(planet_image)
        centroid = centr.centroid_com(planet_image)
        centroid[np.isnan(centroid)]=0
        act_cents.append((centroid[1],centroid[0]))
        xc =int( centroid[0])
        yc = int(centroid[1])

        # sky subtraction if needed
        if sky_annulus_rin is not None and sky_annulus_rout is not None:
            ycoords, xcoords = np.indices(planet_image.shape)
            dist_from_planet = np.sqrt((xcoords - xc)**2 + (ycoords - yc)**2)
            sky_annulus = np.where((dist_from_planet >= sky_annulus_rin*planet_rad) & (dist_from_planet < sky_annulus_rout*planet_rad))
            planet_image -= np.nanmedian(planet_image[sky_annulus])

        smooth_images.append(planet_image)
        # cropping the raster scanned images
        raster_images_cent.append(smooth_images[j][yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius])
        #centroid of the cropped images
        cent.append((yc-up_radius,yc+up_radius,xc-up_radius,xc+up_radius))

    resi_images=flatfield_residuals(raster_images_cent,N=N)
    raster_com=combine_flatfield_rasters(resi_images,planet=planet,band=band,cent=cent, im_size=im_size, rad_mask=rad_mask,planet_rad=planet_rad,n_pix=n_pix, n_pad=n_pad)
    onskyflat=raster_com[0]
    onsky_flatfield = data.FlatField(onskyflat, pri_hdr=prihdr, ext_hdr=exthdr, input_dataset=dataset)
    onsky_flatfield.err=raster_com[1]
    
    return(onsky_flatfield)
    
def nan_flags(dataset,threshold=1):
    """Replaces each DQ-flagged pixel (>= the given threshold) in the dataset with np.nan.

    Args:
        dataset (corgidrp.data.Dataset): input dataset.
        threshold (int, optional): DQ threshold to replace with nans. Defaults to 1.

    Returns:
        corgidrp.data.Dataset: dataset with flagged pixels replaced.
    """

    dataset_out = dataset.copy()

    # mask bad pixels
    bad = np.where(dataset_out.all_dq >= threshold)
    dataset_out.all_data[bad] = np.nan
    dataset_out.all_err[bad[0],:,bad[1],bad[2]] = np.nan

    return dataset_out

def flag_nans(dataset,flag_val=1):
    """Assigns a DQ flag to each nan pixel in the dataset.

    Args:
        dataset (corgidrp.data.Dataset): input dataset.
        flag_val (int, optional): DQ value to assign. Defaults to 1.

    Returns:
        corgidrp.data.Dataset: dataset with nan values flagged.
    """

    dataset_out = dataset.copy()

    # mask bad pixels
    bad = np.isnan(dataset_out.all_data)
    dataset_out.all_dq[bad] = flag_val

    return dataset_out
