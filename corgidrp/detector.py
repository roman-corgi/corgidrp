# Place to put detector-related utility functions

import numpy as np
import corgidrp.data as data
from scipy import interpolate
from scipy.ndimage import median_filter
import glob
import os
import astropy.io.fits as fits
import photutils.centroids as centr
from pathlib import Path
from scipy.ndimage import gaussian_filter as gauss
from photutils.aperture import CircularAperture
from scipy import ndimage

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
            'r0c0': [0, 0]
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
            'r0c0': [0, 1072]
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


def slice_section(frame, obstype, key, detector_regions=None):

    """
    Slice 2d section out of frame

    Args:
        frame (np.ndarray): Full frame consistent with size given in frame_rows, frame_cols
        obstype (str): Keyword referencing the observation type (e.g. 'ENG' or 'SCI')
        key (str): Keyword referencing section to be sliced; must exist in detector_areas
        detector_regions (dict): a dictionary of detector geometry properties.  Keys should be as found in detector_areas in detector.py.  Defaults to that dictionary.

    Returns:
        np.ndarray: a 2D array of the specified detector area
    """
    if detector_regions is None:
            detector_regions = detector_areas
    rows = detector_regions[obstype][key]['rows']
    cols = detector_regions[obstype][key]['cols']
    r0c0 = detector_regions[obstype][key]['r0c0']

    section = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
    if section.size == 0:
        raise Exception('Corners invalid. Tried to slice shape of {0} from {1} to {2} rows and {3} columns'.format(frame.shape, r0c0, rows, cols))
    return section

def unpack_geom(obstype, key, detector_regions=None):
        """Safely check format of geom sub-dictionary and return values.

        Args:
            obstype: str
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
        coords = detector_regions[obstype][key]
        rows = coords['rows']
        cols = coords['cols']
        r0c0 = coords['r0c0']

        return rows, cols, r0c0

def imaging_area_geom(obstype, detector_regions=None):
        """Return geometry of imaging area (including shielded pixels)
        in reference to full frame.  Different from normal image area.

        Args:
            obstype: str
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
        _, cols_pre, _ = unpack_geom(obstype, 'prescan', detector_regions)
        _, cols_serial_ovr, _ = unpack_geom(obstype, 'serial_overscan', detector_regions)
        rows_parallel_ovr, _, _ = unpack_geom(obstype, 'parallel_overscan', detector_regions)
        #_, _, r0c0_image = self._unpack_geom('image')
        fluxmap_rows, _, r0c0_image = unpack_geom(obstype, 'image', detector_regions)

        rows_im = detector_regions[obstype]['frame_rows'] - rows_parallel_ovr
        cols_im = detector_regions[obstype]['frame_cols'] - cols_pre - cols_serial_ovr
        r0c0_im = r0c0_image.copy()
        r0c0_im[0] = r0c0_im[0] - (rows_im - fluxmap_rows)

        return rows_im, cols_im, r0c0_im

def imaging_slice(obstype, frame, detector_regions=None):
        """Select only the real counts from full frame and exclude virtual.
        Includes shielded pixels.

        Use this to transform mask and embed from acting on the full frame to
        acting on only the image frame.

        Args:
            obstype: str
            Keyword referencing the observation type (e.g. 'ENG' or 'SCI')
            frame: array_like
            Input frame
            detector_regions: dict
            a dictionary of detector geometry properties.  Keys should be as found in detector_areas in detector.py.  Defaults to that dictionary.

        Returns:
            sl: array_like
            Imaging slice

        """
        rows, cols, r0c0 = imaging_area_geom(obstype, detector_regions)
        sl = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
        return sl

def flag_cosmics(cube, fwc, sat_thresh, plat_thresh, cosm_filter, cosm_box,
                   cosm_tail, mode='image'):
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

    """
    mask = np.zeros(cube.shape, dtype=int)

    # Do a cheap prefilter for rows that don't have anything bright
    max_rows = np.max(cube, axis=-1,keepdims=True)
    ji_streak_rows = np.transpose(np.array((max_rows >= sat_thresh*fwc).nonzero()[:-1]))

    for j,i in ji_streak_rows:
        row = cube[j,i]

        # Find if and where saturated plateaus start in streak row
        i_begs = find_plateaus(row, fwc, sat_thresh, plat_thresh, cosm_filter)

        # If plateaus exist, kill the hit and the tail
        cutoffs = np.array([])
        ex_l = np.array([])
        if i_begs is not None:
            for i_beg in i_begs:
                # implement cosm_tail
                if i_beg+cosm_filter+cosm_tail+1 > mask.shape[2]:
                    ex_l = np.append(ex_l,
                            i_beg+cosm_filter+cosm_tail+1-mask.shape[2])
                    cutoffs = np.append(cutoffs, i+1)
                streak_end = int(min(i_beg+cosm_filter+cosm_tail+1,
                                mask.shape[2]))
                mask[j, i, i_beg:streak_end] = 1
                # implement cosm_box
                st_row = max(i-cosm_box, 0)
                end_row = min(i+cosm_box+1, mask.shape[1])
                st_col = max(i_beg-cosm_box, 0)
                end_col = min(i_beg+cosm_box+1, mask.shape[2])
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

def residuals(images,planims=None):
	"""Turn this dataset of image frames of neptune or uranus and create matched filters and estimate residuals after 
     dividing from matched filters

     Args:
    	images (np.array): 2D array of cropped neptune or uranus image frames
    	planims (str): name of the planet uranus or neptune

     Returns:
    	matched_residuals (np.array): residual image frames of neptune or uranus divided by matched filter
	"""
    
	if planims=='neptune' or planims=='uranus':
		raster_images = np.array([image[2] for image in images])
		masks = np.array([image[3] for image in images]) 
		images_split = np.array(np.split(np.array(raster_images),3))
		matched_filters = np.array([np.nanmedian(np.stack(images_split[i],2),2) for i in np.arange(0,len(images_split))])
		matched_filters_smooth = [gauss(matched_filters[i],3) for i in range(len(matched_filters))] 
		matched_residuals=[];
		for j in range(len(raster_images)):
			matched_residuals.append(raster_images[j]/matched_filters_smooth[int(np.floor(j//(len(raster_images)//len(matched_filters_smooth))))])
		
	return matched_residuals
    
    
def combine_rasters(residual_images,cent=None,planims=None,band=None):
    """combine the dataset of residual image frames of neptune or uranus and create flat field 
    	and associated error

    	Args:
        	residual_images (np.array): residual image frames of neptune and uranus 
        	cent (np.array): centroid of the image frames
        	planims (str):   name of the planet neptune or uranus
        	band (str):  band of the observation band1 or band4
        	
    	Returns:
        	full_qe (np.array): ideal flat field
        	full_residuals (np.array): flat field from uranus or neptune images
        	percens_resel (np.array): percentage error of the flat field
    """
    
    n = 420
    full_qe=np.random.normal(1,.03,(n,n))
    full_residuals = np.zeros((n,n))
    rad1 = 1.26; rad4 = 1.75
    aperture1 = CircularAperture((np.ceil(rad1), np.ceil(rad1)), r=rad1)
    aperture4 = CircularAperture((np.ceil(rad4), np.ceil(rad4)), r=rad4)
    mask1= aperture1.to_mask().data
    mask4= aperture4.to_mask().data
    rad = 50
    for i in np.arange(len(residual_images)):
        nx = np.arange(0,residual_images[i].shape[1])
        ny = np.arange(0,residual_images[i].shape[0])
        nxx,nyy = np.meshgrid(nx,ny)
        nrr = np.sqrt((nxx-rad-5)**2 + (nyy-rad-5)**2)
        nrr_copy = nrr.copy()
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

    if band == 'band4':
        full_qe_resel = ndimage.convolve(full_qe, mask4)
        full_residuals_resel = ndimage.convolve(full_residuals,mask4)
        percens_resel = (((full_qe_resel - full_residuals_resel)/full_qe_resel)*100)
    
    elif band== 'band1':
        full_qe_resel = ndimage.convolve(full_qe, mask1)
        full_residuals_resel = ndimage.convolve(full_residuals,mask1)
        percens_resel = (((full_qe_resel - full_residuals_resel)/full_qe_resel)*100)
    
    full_residuals[full_residuals==0] = None
    nx = np.arange(0,percens_resel.shape[1])
    ny = np.arange(0,percens_resel.shape[0])
    nxx,nyy = np.meshgrid(ny,nx)
    nrr = np.sqrt((nxx-n/2)**2 + (nyy-n/2)**2)
    full_residuals[nrr>165]= None
    percens_resel[nrr>165] = None
    full_qe[nrr>165]= None
    full_qe=np.pad(full_qe, ((302,302),(302,302)), mode='constant',constant_values=(np.nan,))
    full_residuals=np.pad(full_residuals, ((302,302),(302,302)), mode='constant',constant_values=(np.nan,))
    percens_resel=np.pad(percens_resel, ((302,302),(302,302)), mode='constant',constant_values=(np.nan,))
    
    return (full_qe,full_residuals,percens_resel)
    
    
def create_onsky_flatfield(filedir):
	"""Turn this dataset of raster scanned image frames of neptune or uranus into on-sky calibrated flat field 

    	Args:
        	filedir (str):  Full path to directory of images that are raster scanned

    	Returns:
    		nept_band1_flatfield (corgidrp.data.FlatField): a master flat for flat calibration using on sky images of neptune in band 1
    		nept_band4_flatfield (corgidrp.data.FlatField): a master flat for flat calibration using on sky images of neptune in band 4
    		ura_band1_flatfield (corgidrp.data.FlatField): a master flat for flat calibration using on sky images of neptune in band 1
    		ura_band4_flatfield (corgidrp.data.FlatField): a master flat for flat calibration using on sky images of neptune in band 4
	"""
	
	smooth_images=[];nept_band1_raster_images_cent=[];nept_band4_raster_images_cent=[];ura_band1_raster_images_cent=[];
	nept_band1_cent=[];nept_band4_cent=[];ura_band1_cent=[];ura_band4_cent=[];act_cents = [];
	ura_band4_raster_images_cent=[]; planet_names=[]; nept_band1_dataset=[]; frames=[]; nept_band4_dataset=[]; ura_band1_dataset=[]; ura_band4_dataset=[];
	
	image_files=glob.glob(os.path.join(filedir, "*.fits"))
	raster_scanned_images =np.sort(image_files)
	
	for i in range(len(image_files)):
		planet_im=fits.open(raster_scanned_images[i],ignore_missing_simple=True)
		planet_image=planet_im[1].data
		planet=os.path.basename(raster_scanned_images[i]).split('_')[0]
		planet_names.append(planet)
		band='band'+os.path.basename(raster_scanned_images[i]).split('_')[2]
		image_size=np.shape(planet_image)
		nx = np.arange(0,image_size[0])
		ny = np.arange(0,image_size[1])
		qe_prnu_fsm_raster = np.random.normal(1,.03,(image_size[0],image_size[0]))
		nxx,nyy = np.meshgrid(nx,ny)
		nrr = np.sqrt((nxx-nx/2)**2 + (nyy-ny/2)**2)
		centroid = centr.centroid_com(planet_image)
		centroid[np.isnan(centroid)]=0
		act_cents.append((centroid[1],centroid[0]))
		xc =int( centroid[0])
		yc = int(centroid[1])
		up_radius=55
		smooth_images.append(planet_image)
		
		if planet =='neptune' and band == 'band1':
			nept_band1_raster_images_cent.append((planet,band,smooth_images[i][yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius],qe_prnu_fsm_raster[yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius]))
			nept_band1_cent.append((yc-up_radius,yc+up_radius,xc-up_radius,xc+up_radius))
			prihdr=planet_im[0].header
			exthdr=planet_im[1].header
			frame=data.Image(smooth_images[i][yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius], pri_hdr=prihdr, ext_hdr=exthdr)
			frames.append(frame)
			nept_band1_dataset=data.Dataset(frames)
		elif planet =='neptune' and band == 'band4':
			nept_band4_raster_images_cent.append((planet,band,smooth_images[i][yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius],qe_prnu_fsm_raster[yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius]))
			nept_band4_cent.append((yc-up_radius,yc+up_radius,xc-up_radius,xc+up_radius))
			prihdr=planet_im[0].header
			exthdr=planet_im[1].header
			frame=data.Image(smooth_images[i][yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius], pri_hdr=prihdr, ext_hdr=exthdr)
			frames.append(frame)
			nept_band4_dataset=data.Dataset(frames)
		elif planet =='uranus' and band == 'band1':
			ura_band1_raster_images_cent.append((planet,band,smooth_images[i][yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius],qe_prnu_fsm_raster[yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius]))
			ura_band1_cent.append((yc-up_radius,yc+up_radius,xc-up_radius,xc+up_radius))
			prihdr=planet_im[0].header
			exthdr=planet_im[1].header
			frame=data.Image(smooth_images[i][yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius], pri_hdr=prihdr, ext_hdr=exthdr)
			frames.append(frame)
			ura_band1_dataset=data.Dataset(frames)
		elif planet =='uranus' and band == 'band4':
			ura_band4_raster_images_cent.append((planet,band,smooth_images[i][yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius],qe_prnu_fsm_raster[yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius]))
			ura_band4_cent.append((yc-up_radius,yc+up_radius,xc-up_radius,xc+up_radius))
			prihdr=planet_im[0].header
			exthdr=planet_im[1].header
			frame=data.Image(smooth_images[i][yc-up_radius:yc+up_radius,xc-up_radius:xc+up_radius], pri_hdr=prihdr, ext_hdr=exthdr)
			frames.append(frame)
			ura_band4_dataset=data.Dataset(frames)
	
	
	resi_nept_band1=residuals(nept_band1_raster_images_cent,planims='neptune')
	nept_band1_flat=combine_rasters(resi_nept_band1,planims='neptune',band='band1',cent=nept_band1_cent)[1]
	nept_band1_flatfield = data.FlatField(nept_band1_flat, pri_hdr=prihdr,ext_hdr=exthdr,input_dataset=nept_band1_dataset)
	nept_band1_flatfield.err=combine_rasters(resi_nept_band1,planims='neptune',band='band1',cent=nept_band1_cent)[2]
	nept_band1_flatfield.err=nept_band1_flatfield.err.reshape((1,)+nept_band1_flatfield.err.shape)
	
	resi_nept_band4=residuals(nept_band4_raster_images_cent,planims='neptune')
	nept_band4_flat=combine_rasters(resi_nept_band4,planims='neptune',band='band4',cent=nept_band4_cent)[1]
	nept_band4_flatfield = data.FlatField(nept_band4_flat, pri_hdr=prihdr,ext_hdr=exthdr,input_dataset=nept_band4_dataset)
	nept_band4_flatfield.err=combine_rasters(resi_nept_band4,planims='neptune',band='band4',cent=nept_band4_cent)[2]
	nept_band4_flatfield.err=nept_band4_flatfield.err.reshape((1,)+nept_band4_flatfield.err.shape)
	
	resi_ura_band1=residuals(ura_band1_raster_images_cent,planims='uranus')
	ura_band1_flat=combine_rasters(resi_ura_band1,planims='uranus',band='band1',cent=ura_band1_cent)[1]
	ura_band1_flatfield = data.FlatField(ura_band1_flat, pri_hdr=prihdr,ext_hdr=exthdr,input_dataset=ura_band1_dataset)
	ura_band1_flatfield.err=combine_rasters(resi_ura_band1,planims='uranus',band='band1',cent=ura_band1_cent)[2]
	ura_band1_flatfield.err=ura_band1_flatfield.err.reshape((1,)+ura_band1_flatfield.err.shape)
	
	resi_ura_band4=residuals(ura_band4_raster_images_cent,planims='uranus')
	ura_band4_flat=combine_rasters(resi_ura_band4,planims='uranus',band='band4',cent=ura_band4_cent)[1]
	ura_band4_flatfield = data.FlatField(ura_band4_flat, pri_hdr=prihdr,ext_hdr=exthdr,input_dataset=ura_band4_dataset)
	ura_band4_flatfield.err=combine_rasters(resi_ura_band4,planims='uranus',band='band1',cent=ura_band4_cent)[2]
	ura_band4_flatfield.err=ura_band4_flatfield.err.reshape((1,)+ura_band4_flatfield.err.shape)
	
	
	return(nept_band1_flatfield,nept_band4_flatfield,ura_band1_flatfield,ura_band4_flatfield)
