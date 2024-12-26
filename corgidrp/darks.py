import numpy as np
import warnings
from astropy.io import fits
import os

from corgidrp.detector import slice_section, imaging_slice, imaging_area_geom, detector_areas
import corgidrp.check as check
from corgidrp.data import DetectorNoiseMaps, Dark

def mean_combine(image_list, bpmap_list, err=False):
    """
    Get mean frame and corresponding bad-pixel map from L2b data frames.  The
    input image_list should consist of frames with no bad pixels marked or
    removed.  This function takes the bad-pixels maps into account when taking
    the mean.

    The two lists must be the same length, and each 2D array in each list must
    be the same size, both within a list and across lists.

    If the inputs are instead np.ndarray (a single frame or a stack),
    the function will accommodate and convert them to lists of arrays.

    Also Includes outputs for processing darks used for calibrating the
    master dark.

    Args:
        image_list (list or array_like): List (or stack) of L2b data frames
    (with no bad pixels applied to them).
        bpmap_list (list or array_like): List (or stack) of bad-pixel maps
        associated with L2b data frames. Each must be 0 (good) or 1 (bad)
        at every pixel.
        err (bool):  If True, calculates the standard error over all
        the frames.  Intended for the corgidrp.Data.Dataset.all_err
        arrays. Defaults to False.

    Returns:
        comb_image (array_like): Mean-combined frame from input list data.

        comb_bpmap (array_like): Mean-combined bad-pixel map.

        map_im (array-like): Array showing how many frames per pixel were
        unmasked. Used for getting read noise in the calibration of the
        master dark.

        enough_for_rn (bool): Useful only for the calibration of the master dark.
        False:  Fewer than half the frames available for at least one pixel in
        the averaging due to masking, so noise maps cannot be effectively
        determined for all pixels.
        True:  Half or more of the frames available for all pixels, so noise
        mpas can be effectively determined for all pixels.

    """
    # if input is an np array or stack, try to accommodate
    if type(image_list) == np.ndarray:
        if image_list.ndim == 1: # pathological case of empty array
            image_list = list(image_list)
        elif image_list.ndim == 2: #covers case of single 2D frame
            image_list = [image_list]
        elif image_list.ndim == 3: #covers case of stack of 2D frames
            image_list = list(image_list)
    if type(bpmap_list) == np.ndarray:
        if bpmap_list.ndim == 1: # pathological case of empty array
            bpmap_list = list(bpmap_list)
        elif bpmap_list.ndim == 2: #covers case of single 2D frame
            bpmap_list = [bpmap_list]
        elif bpmap_list.ndim == 3: #covers case of stack of 2D frames
            bpmap_list = list(bpmap_list)

    # Check inputs
    if not isinstance(image_list, list):
        raise TypeError('image_list must be a list')
    if not isinstance(bpmap_list, list):
        raise TypeError('bpmap_list must be a list')
    if len(image_list) != len(bpmap_list):
        raise TypeError('image_list and bpmap_list must be the same length')
    if len(image_list) == 0:
        raise TypeError('input lists cannot be empty')
    s0 = image_list[0].shape
    for index, im in enumerate(image_list):
        check.twoD_array(im, 'image_list[' + str(index) + ']', TypeError)
        if im.shape != s0:
            raise TypeError('all input list elements must be the same shape')
        pass
    for index, bp in enumerate(bpmap_list):
        check.twoD_array(bp, 'bpmap_list[' + str(index) + ']', TypeError)
        if np.logical_and((bp != 0), (bp != 1)).any():
            raise TypeError('bpmap_list elements must be 0- or 1-valued')
        if bp.dtype != int:
            raise TypeError('bpmap_list must be made up of int arrays')
        if bp.shape != s0:
            raise TypeError('all input list elements must be the same shape')
        pass


    # Get masked arrays
    ims_m = np.ma.masked_array(image_list, bpmap_list)

    # Add non masked elements
    sum_im = np.zeros_like(image_list[0])
    map_im = np.zeros_like(image_list[0], dtype=int)
    for im_m in ims_m:
        masked = im_m.filled(0)
        if err:
            sum_im += masked**2
        else:
            sum_im += masked
        map_im += (im_m.mask == False).astype(int)

    # Divide sum_im by map_im only where map_im is not equal to 0 (i.e.,
    # not masked).
    # Where map_im is equal to 0, set combined_im to zero
    comb_image = np.divide(sum_im, map_im, out=np.zeros_like(sum_im),
                            where=map_im != 0)
   
    if err: # (sqrt of sum of sigma**2 terms)/sqrt(n)
        comb_image = np.sqrt(comb_image)

    # Mask any value that was never mapped (aka masked in every frame)
    comb_bpmap = (map_im == 0).astype(int)

    enough_for_rn = True
    if map_im.min() < len(image_list)/2:
        enough_for_rn = False

    return comb_image, comb_bpmap, map_im, enough_for_rn

def build_trad_dark(dataset, detector_params, detector_regions=None, full_frame=False):
    """This function produces a traditional master dark from a stack of darks
    taken at a specific EM gain and exposure time to match a corresponding
    observation.  The input dataset represents a stack of dark frames (in e-).
    The frames should be SCI full frames that:

    - have had their bias subtracted (assuming full frame)
    - have had masks made for cosmic rays
    - have been corrected for nonlinearity
    - have been converted from DN to e-
    - have been desmeared if desmearing is appropriate.  Under normal
    circumstances, darks should not be desmeared.  The only time desmearing
    would be useful is in the unexpected case that, for example,
    dark current is so high that it stands far above other noise that is
    not smeared upon readout, such as clock-induced charge, 
    fixed-pattern noise, and read noise.

    Also, add_photon_noise() should NOT have been applied to the frames in
    dataset.  And note that creation of the
    fixed bad pixel mask containing warm/hot pixels and pixels with sub-optimal
    functionality requires a master dark, which requires this function first.

    The steps shown above are a subset of the total number of steps
    involved in going from L1 to L2b.  This function averages
    each stack (which minimizes read noise since it has a mean of 0) while
    accounting for masks.

    There are rows of each frame that are used for telemetry and are irrelevant
    for making a master dark.
    So this function disregards telemetry rows and does not do any fitting for
    master dark for those rows.  They are set to NaN.

    Args:
    dataset (corgidrp.data.Dataset):
        This is an instance of corgidrp.data.Dataset.
        Each frame should accord with the SCI full frame geometry.
    detector_params (corgidrp.data.DetectorParams):
        a calibration file storing detector calibration values
    detector_regions (dict):
        a dictionary of detector geometry properties.  Keys should be as found
        in detector_areas in detector.py.
        Defaults to None, in which case detector_areas from detector.py is used.
    full_frame (bool):
        If True, a full-frame master dark is generated (which
        may be useful for the module that statistically fits a frame to find
        the empirically applied EM gain, for example). If False, an image-area
        master dark is generated.  Defaults to False.

    Returns:
    master_dark : corgidrp.data.DetectorNoiseMaps instance
        The mean-combined master dark, in detected electrons.
        master_dark.err includes the statistical error across all the frames as
        well as any individual err from each frame (and accounts for masked
        pixels in the calculations).
        master_dark.dq: pixels that are masked for all frames are assigned a
        flag value of 1 ("Bad pixel - unspecified reason"),
        and a flag value of 256 is assigned to pixels that are
        masked for half or more of the frames, making them unreliable, with
        large err values, but possibly still usable.
    """
    if detector_regions is None:
            detector_regions = detector_areas

    _, unique_vals = dataset.split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN', 'KGAIN'])
    if len(unique_vals) > 1:
        raise Exception('Input dataset should contain frames of the same exposure time, commanded EM gain, and k gain.')
    # getting telemetry rows to ignore in fit
    telem_rows_start = detector_params.params['telem_rows_start']
    telem_rows_end = detector_params.params['telem_rows_end']
    telem_rows = slice(telem_rows_start, telem_rows_end)

    frames = []
    bpmaps = []
    errs = []
    for fr in dataset.frames:
        # ensure frame is in float so nan can be assigned, though it should
        # already be float
        frame = fr.data.astype(float)
        # For the fit, all types of bad pixels should be masked:
        b1 = fr.dq.astype(bool).astype(int)
        err = fr.err[0]
        frame[telem_rows] = np.nan
        i0 = slice_section(frame, 'SCI', 'image', detector_regions)
        if np.isnan(i0).any():
            raise ValueError('telem_rows cannot be in image area.')
        frame[telem_rows] = 0
        frames.append(frame)
        bpmaps.append(b1)
        errs.append(err)
    mean_frame, combined_bpmap, unmasked_num, _ = mean_combine(frames, bpmaps)
    mean_err, _, _, _ = mean_combine(errs, bpmaps, err=True)
    # combine the error from individual frames to the standard deviation across
    # the frames due to statistical variance
    masked_frames = np.ma.masked_array(frames, bpmaps)
    stat_std = np.ma.std(masked_frames, axis=0)/np.sqrt(unmasked_num)
    total_err = np.sqrt(mean_err**2 + stat_std**2)
    # There are no masked pixels in total_err, and FITS can't save masked arrays,
    # so turn it into a regular array
    total_err = np.ma.getdata(total_err)
    # flag value of 256; unreliable pixels, large err
    output_dq = (unmasked_num < len(dataset)/2).astype(int)*256
    if (output_dq == 256).any():
        warnings.warn('At least one pixel was masked for half or more of the '
                      'frames.')
    # flag value of 1 for those that are masked all the way through for all
    # frames; this overwrites the flag value of 256 that was assigned to
    # these pixels in previous line
    unfittable_ind = np.where(combined_bpmap == 1)
    output_dq[unfittable_ind] = 1
    if not full_frame:
        dq = slice_section(output_dq, 'SCI', 'image', detector_regions)
        err = slice_section(total_err, 'SCI', 'image', detector_regions)
        data = slice_section(mean_frame, 'SCI', 'image', detector_regions)
    else:
        dq = output_dq
        err = total_err
        data = mean_frame

    # get from one of the noise maps and modify as needed
    prihdr = dataset.frames[0].pri_hdr
    exthdr = dataset.frames[0].ext_hdr
    errhdr = dataset.frames[0].err_hdr
    exthdr['NAXIS1'] = data.shape[0]
    exthdr['NAXIS2'] = data.shape[1]
    exthdr['DATATYPE'] = 'Dark'

    master_dark = Dark(data, prihdr, exthdr, dataset, err, dq, errhdr)

    return master_dark



class CalDarksLSQException(Exception):
    """Exception class for calibrate_darks_lsq."""

def calibrate_darks_lsq(dataset, detector_params, detector_regions=None):
    """The input dataset represents a collection of frame stacks of the
    same number of dark frames (in e- units), where the stacks are for various
    EM gain values and exposure times.  The frames in each stack should be
    SCI full frames that:

    - have had their bias subtracted (assuming full frame)
    - have had masks made for cosmic rays
    - have been corrected for nonlinearity
    - have been converted from DN to e-
    - have been desmeared if desmearing is appropriate.  Under normal
    circumstances, darks should not be desmeared.  The only time desmearing
    would be useful is in the unexpected case that, for example,
    dark current is so high that it stands far above other noise that is
    not smeared upon readout, such as clock-induced charge, 
    fixed-pattern noise, and read noise.

    Also, add_photon_noise() should NOT have been applied to the frames in
    dataset.  And note that creation of the
    fixed bad pixel mask containing warm/hot pixels and pixels with sub-optimal
    functionality requires a master dark, which requires this function first.

    The steps shown above are a subset of the total number of steps
    involved in going from L1 to L2b.  This function averages
    each stack (which minimizes read noise since it has a mean of 0) while
    accounting for masks.  It then computes a per-pixel map of fixed-pattern
    noise (due to electromagnetic pick-up before going through the amplifier),
    dark current, and the clock-induced charge (CIC), and it also returns the
    bias offset value.  The function assumes the stacks have the same
    noise profile (at least the same CIC, fixed-pattern noise, and dark
    current).

    There are rows of each frame that are used for telemetry and are irrelevant
    for making a master dark.
    So this function disregards telemetry rows and does not do any fitting for
    master dark for those rows.  They are set to NaN.

    Args:
    dataset (corgidrp.data.Dataset):
        This is an instance of corgidrp.data.Dataset.  The function sorts it into
        stacks where each stack is a stack of dark frames,
        and each stack is for a unique EM gain and frame time combination.
        Each stack should have the same number of frames.
        Each frame should accord with the SCI frame geometry.
        We recommend  >= 1300 frames for each stack if calibrating
        darks for analog frames,
        thousands for photon counting depending on the maximum number of
        frames that will be used for photon counting.
    detector_params (corgidrp.data.DetectorParams):
        a calibration file storing detector calibration values
    detector_regions (dict):
        a dictionary of detector geometry properties.  Keys should be as found
        in detector_areas in detector.py.
        Defaults to None, in which case detector_areas from detector.py is used.

    Returns:
    noise_maps : corgidrp.data.DetectorNoiseMaps instance
        Includes a 3-D stack of frames for the data, err, and the dq.
        input data: np.stack([FPN_map, CIC_map, DC_map])
        input err:  np.stack([FPN_std_map, C_std_map_combo, DC_std_map])
        FPN_std_map, C_std_map_combo, and DC_std_map contain the fitting error, but C_std_map_combo includes
        also the statistical error across the frames and accounts for any err
        content of the individual input frames.  We include it in the CIC part
        since it will not be scaled by EM gain or exposure time when the master
        dark is created.  In all the err, masked pixels are accounted for in
        the calculations.
        input dq:   np.stack([output_dq, output_dq, output_dq])
        unreliable_pix_map is used for the
        output Dark's dq after assigning these pixels a flag value of 256.
        They should have large err values.
        The pixels that are masked for EVERY frame in all sub-stacks
        but 3 (or less) are assigned a flag value of
        1, which falls under the category of "Bad pixel - unspecified reason".
        These pixels would have no reliability for dark subtraction.

        The header info is taken from that of
        one of the frames from the input datasets and can be changed via a call
        to the DetectorNoiseMaps class if necessary.  The bias offset info is
        found in the exthdr under these keys:
        'B_O': bias offset
        'B_O_ERR': bias offset error
        'B_O_UNIT': DN


    Info on intermediate products in this function:
    FPN_map : array-like (full frame)
        A per-pixel map of fixed-pattern noise (in deteceted electrons).  Any negative values
        from the fit are made positive in the end.
    CIC_map : array-like (full frame)
        A per-pixel map of EXCAM clock-induced charge (in deteceted electrons). Any negative
        values from the fit are made positive in the end.
    DC_map : array-like (full frame)
        A per-pixel map of dark current (in deteceted electrons/s). Any negative values
        from the fit are made positive in the end.
    bias_offset : float
        The median for the residual FPN+CIC in the region where bias was
        calculated (i.e., prescan). In DN.
    bias_offset_up : float
        The upper bound of bias offset, accounting for error in input datasets
        and the fit.
    bias_offset_low : float
        The lower bound of bias offset, accounting for error in input datasets
        and the fit.
    FPN_image_map : array-like (image area)
        A per-pixel map of fixed-pattern noise in the image area (in deteceted electrons).
        Any negative values from the fit are made positive in the end.
    CIC_image_map : array-like (image area)
        A per-pixel map of EXCAM clock-induced charge in the image area
        (in deteceted electrons). Any negative values from the fit are made positive in the end.
    DC_image_map : array-like (image area)
        A per-pixel map of dark current in the image area (in deteceted electrons/s).
        Any negative values from the fit are made positive in the end.
    FPNvar : float
        Variance of fixed-pattern noise map (in deteceted electrons).
    CICvar : float
        Variance of clock-induced charge map (in deteceted electrons).
    DCvar : float
        Variance of dark current map (in deteceted electrons).
    read_noise : float
        Read noise estimate from the noise profile of a mean frame (in deteceted electrons).
        It's read off from the sub-stack with the lowest product of EM gain and
        frame time so that the gained variance of C and D is comparable to or
        lower than read noise variance, thus making reading it off doable.
        If read_noise is returned as NaN, the read noise estimate is not
        trustworthy, possibly because not enough frames were used per substack
        for that or because the next lowest gain setting is much larger than
        the gain used in the sub-stack.  The official calibrated read noise
        comes from the k gain calibration, and this is just a rough estimate
        that can be used as a sanity check, for checking agreement with the
        official calibrated value.
    R_map : array-like
        A per-pixel map of the adjusted coefficient of determination
        (adjusted R^2) value for the fit.
    FPN_image_mean : float
        F averaged over all pixels,
        before any negative ones are made positive.  Should be roughly the same
        as taking the mean of F_image_map.  This is just for comparison.
    CIC_image_mean : float
        C averaged over all pixels,
        before any negative ones are made positive.  Should be roughly the same
        as taking the mean of C_image_map.  This is just for comparison.
    DC_image_mean : float
        D averaged over all pixels,
        before any negative ones are made positive.  Should be roughly the same
        as taking the mean of D_image_map.  This is just for comparison.
    unreliable_pix_map : array-like (full frame)
        A pixel value in this array indicates how many sub-stacks are usable
        for a fit for that pixel.  For each sub-stack for which
        a pixel is masked for more than half of
        the frames in the sub-stack, 1 is added to that pixel's value
        in unreliable_pix_map.  Since the least-squares fit function has 3
        parameters, at least 4 sub-stacks are needed for a given pixel in order
        to perform a fit for that pixel.  The pixels in unreliable_pix_map that
        are >= len(stack_arr)-3 cannot be fit.  NOTE:  This is used for the
        output Dark's dq after assigning these pixels a flag value of 256.
        They should have large err values.
        The pixels that are masked for EVERY frame in all sub-stacks
        but 3 (or less) are assigned a flag value of
        1, which falls under the category of "Bad pixel - unspecified reason".
        These pixels would have no reliability for dark subtraction.
    FPN_std_map : array-like (full frame)
        The standard deviation per pixel for the calibrated FPN.
    CIC_std_map : array-like (full frame)
        The standard deviation per pixel for the calibrated CIC.
    DC_std_map : array-like (full frame)
        The standard deviation per pixel for the calibrated dark current.
    stacks_err : array-like (full frame)
        Standard error per pixel coming from the frames in datasets used to
        calibrate the noise maps.
    """
    if detector_regions is None:
            detector_regions = detector_areas

    datasets, _ = dataset.copy().split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN', 'KGAIN'])
    if len(datasets) <= 3:
        raise CalDarksLSQException('Number of sub-stacks in datasets must '
                'be more than 3 for proper curve fit.')
    # getting telemetry rows to ignore in fit
    telem_rows_start = detector_params.params['telem_rows_start']
    telem_rows_end = detector_params.params['telem_rows_end']
    telem_rows = slice(telem_rows_start, telem_rows_end)

    EMgain_arr = np.array([])
    exptime_arr = np.array([])
    kgain_arr = np.array([])
    mean_frames = []
    total_errs = []
    mean_num_good_fr = []
    unreliable_pix_map = np.zeros((detector_regions['SCI']['frame_rows'],
                                   detector_regions['SCI']['frame_cols'])).astype(int)
    unfittable_pix_map = unreliable_pix_map.copy()
    for i in range(len(datasets)):
        frames = []
        bpmaps = []
        errs = []
        check.threeD_array(datasets[i].all_data,
                           'datasets['+str(i)+'].all_data', TypeError)
        if len(datasets[i].all_data) < 1300:
            warnings.warn('A sub-stack was found with less than 1300 frames, '
            'which is the recommended number per sub-stack for an analog '
            'master dark')
        if i > 0:
            if np.shape(datasets[i-1].all_data) != np.shape(datasets[i].all_data):
                raise CalDarksLSQException('All sub-stacks must have the '
                            'same number of frames and frame shape.')
        try: # if EM gain measured directly from frame TODO change hdr name if necessary
            EMgain_arr = np.append(EMgain_arr, datasets[i].frames[0].ext_hdr['EMGAIN_M'])
        except:
            try: # use applied EM gain if available
                EMgain_arr = np.append(EMgain_arr, datasets[i].frames[0].ext_hdr['EMGAIN_A'])
            except: # use commanded gain otherwise
                EMgain_arr = np.append(EMgain_arr, datasets[i].frames[0].ext_hdr['CMDGAIN'])
        exptime = datasets[i].frames[0].ext_hdr['EXPTIME']
        cmdgain = datasets[i].frames[0].ext_hdr['CMDGAIN']
        kgain = datasets[i].frames[0].ext_hdr['KGAIN']
        exptime_arr = np.append(exptime_arr, exptime)
        kgain_arr = np.append(kgain_arr, kgain)

        for fr in datasets[i].frames:
            # ensure frame is in float so nan can be assigned, though it should
            # already be float
            frame = fr.data.astype(float)
            # For the fit, all types of bad pixels should be masked:
            b1 = fr.dq.astype(bool).astype(int)
            err = fr.err[0]
            frame[telem_rows] = np.nan
            i0 = slice_section(frame, 'SCI', 'image', detector_regions)
            if np.isnan(i0).any():
                raise ValueError('telem_rows cannot be in image area.')
            frame[telem_rows] = 0
            frames.append(frame)
            bpmaps.append(b1)
            errs.append(err)
        mean_frame, combined_bpmap, unmasked_num, _ = mean_combine(frames, bpmaps)
        mean_err, _, _, _ = mean_combine(errs, bpmaps, err=True)
        # combine the error from individual frames to the standard deviation across
        # the frames due to statistical variance
        masked_frames = np.ma.masked_array(frames, bpmaps)
        stat_std = np.ma.std(masked_frames, axis=0)/np.sqrt(unmasked_num)
        total_err = np.sqrt(mean_err**2 + stat_std**2)
        # There are no masked pixels in total_err, and FITS can't save masked arrays,
        # so turn it into a regular array
        total_err = np.ma.getdata(total_err)
        pixel_mask = (unmasked_num < len(datasets[i].frames)/2).astype(int)
        mean_num = np.mean(unmasked_num)
        mean_frame[telem_rows] = np.nan
        mean_frames.append(mean_frame)
        total_errs.append(total_err)
        mean_num_good_fr.append(mean_num)
        unreliable_pix_map += pixel_mask
        unfittable_pix_map += combined_bpmap
    unreliable_pix_map = unreliable_pix_map.astype(int)
    mean_stack = np.stack(mean_frames)
    mean_err_stack = np.stack(total_errs)
    if (unreliable_pix_map >= len(datasets)-3).any(): # this condition catches the "unfittable" pixels too
        warnings.warn('At least one pixel was masked for more than half of '
                      'the frames in some sub-stacks, leaving 3 or fewer '
                      'sub-stacks that did not suffer this masking for these '
                      'pixels, which means the fit was unreliable for '
                      'these pixels.  These are the pixels in the output '
                      'unreliable_pixel_map that are >= len(datasets)-3.')
    # flag value of 256; unreliable pixels, large err
    output_dq = (unreliable_pix_map >= len(datasets)-3).astype(int)*256
    # flag value of 1 for those that are masked all the way through for all
    # but 3 (or less) stacks; this overwrites the flag value of 256 that was assigned to
    # these pixels in previous line
    unfittable_ind = np.where(unfittable_pix_map >= len(datasets)-3)
    output_dq[unfittable_ind] = 1

    if len(np.unique(EMgain_arr)) < 2:
        raise CalDarksLSQException("Must have at least 2 unique EM gains "
                                   'represented by the sub-stacks in '
                                   'datasets.')
    if len(EMgain_arr[EMgain_arr<1]) != 0:
        raise CalDarksLSQException('Each EM gain must be 1 or greater.')
    if len(np.unique(exptime_arr)) < 2:
        raise CalDarksLSQException("Must have at 2 unique exposure times.")
    if len(exptime_arr[exptime_arr<0]) != 0:
        raise CalDarksLSQException('Each exposure time cannot be negative.')
    if len(kgain_arr[kgain_arr<=0]) != 0:
        raise CalDarksLSQException('Each element of k_arr must be greater '
            'than 0.')
    unique_sub_stacks = list(zip(EMgain_arr, exptime_arr))
    for el in unique_sub_stacks:
        if unique_sub_stacks.count(el) > 1:
            raise CalDarksLSQException('The EM gain and frame time '
            'combinations for the sub-stacks must be unique.')

    # need the correlation coefficient for FPN for read noise estimate later;
    # other noise sources aren't correlated frame to frame
    # Use correlation b/w mean stacks since read noise is negligible (along
    # with dark current and CIC); correlation b/w mean stacks then
    # approximately equal to the correlation b/w FPN from stack to stack
    # this is the stack that will be used later for estimating read noise:
    min1 = np.argmin(EMgain_arr*exptime_arr)
    # getting next "closest" stack for finding correlation b/w FPN maps:
    # same time, next gain up from least gain (close in time*gain but different
    # gain so that effective read noise values are more uncorrelated)
    tinds = np.where(exptime_arr == exptime_arr[min1])
    nextg = EMgain_arr[EMgain_arr > EMgain_arr[min1]].min()
    ginds = np.where(EMgain_arr == nextg)
    intersect = np.intersect1d(tinds, ginds)
    if intersect.size > 0:
        min2 = intersect[0]
    else: # just get next smallest g_arr*t_arr
        min2 = np.where(np.argsort(EMgain_arr*exptime_arr) == 1)[0][0]
    msi = imaging_slice('SCI', mean_stack[min1], detector_regions)
    msi2 = imaging_slice('SCI', mean_stack[min2], detector_regions)
    avg_corr = np.corrcoef(msi.ravel(), msi2.ravel())[0, 1]

    # number of observations (i.e., # of averaged stacks provided for fit)
    M = len(EMgain_arr)
    FPN_map = np.zeros_like(mean_stack[0])
    CIC_map = np.zeros_like(mean_stack[0])
    DC_map = np.zeros_like(mean_stack[0])

    # input data error comes from .err arrays; could use this for error bars
    # in input data for weighted least squares, but we'll just simply get the
    # std error and add it in quadrature to least squares fit standard dev
    stacks_err = np.sqrt(np.sum(mean_err_stack**2, axis=0)/np.sqrt(len(mean_err_stack)))

    # matrix to be used for least squares and covariance matrix
    X = np.array([np.ones([len(EMgain_arr)]), EMgain_arr, EMgain_arr*exptime_arr]).T
    mean_stack_Y = np.transpose(mean_stack, (2,0,1))
    params_Y = np.linalg.pinv(X)@mean_stack_Y
    #next line: checked with KKT method for including bounds
    #actually, do this after determining everything else so that
    # bias_offset, etc is accurate
    #params_Y[params_Y < 0] = 0
    params = np.transpose(params_Y, (1,2,0))
    FPN_map = params[0]
    CIC_map = params[1]
    DC_map = params[2]
    # using chi squared for ordinary least squares (OLS) variance estiamate
    # This is OLS since the parameters to fit are linear in fit function
    # 3: number of fitted params
    residual_stack = mean_stack - np.transpose(X@params_Y, (1,2,0))
    sigma2_frame = np.sum(residual_stack**2, axis=0)/(M - 3)
    # average sigma2 for image area and use that for all three vars since
    # that's only place where all 3 (F, C, and D) should be present
    sigma2_image = imaging_slice('SCI', sigma2_frame, detector_regions)
    sigma2 = np.mean(sigma2_image)
    cov_matrix = np.linalg.inv(X.T@X)

    # For full frame map of standard deviation:
    FPN_std_map = np.sqrt(sigma2_frame*cov_matrix[0,0])
    CIC_std_map = np.sqrt(sigma2_frame*cov_matrix[1,1])
    # D_std_map here used only for bias_offset error estimate
    DC_std_map = np.sqrt(sigma2_frame*cov_matrix[2,2])
    # Dark current should only be in image area, so error only for that area:
    D_std_map_im = np.sqrt(sigma2_image*cov_matrix[2,2])

    var_matrix = sigma2*cov_matrix
    # variances here would naturally account for masked pixels due to cosmics
    # since mean_combine does this
    FPNvar = var_matrix[0,0]
    CICvar = var_matrix[1,1]
    DCvar = var_matrix[2,2]

    ss_r = np.sum((np.mean(mean_stack, axis=0) -
        np.transpose(X@params_Y, (1,2,0)))**2, axis=0)
    ss_e = np.sum(residual_stack**2, axis=0)
    Rsq = ss_r/(ss_e+ss_r)
    # adjusted coefficient of determination, adjusted R^2:
    # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
    # The closer to 1, the better the fit.
    # Can have negative values. Can never be above 1.
    # If R_map has nan or inf values, then something is probably wrong;
    # this is good feedback to the user. However, nans in the telemetry rows
    # are expected.
    R_map = 1 - (1 - Rsq)*(M - 1)/(M - 3)

    # doesn't matter which k used for just slicing
    DC_image_map = imaging_slice('SCI', DC_map, detector_regions)
    CIC_image_map = imaging_slice('SCI', CIC_map, detector_regions)
    FPN_image_map = imaging_slice('SCI', FPN_map, detector_regions)

    # res: should subtract D_map, too.  D should be zero there (in prescan),
    # but fitting errors may lead to non-zero values there.
    bias_offset = np.zeros([len(mean_stack)])
    bias_offset_up = np.zeros([len(mean_stack)])
    bias_offset_low = np.zeros([len(mean_stack)])
    for i in range(len(mean_stack)):
        # NOTE assume no error in g_arr values (or t_arr or k_arr)
        res = mean_stack[i] - EMgain_arr[i]*CIC_map - FPN_map - EMgain_arr[i]*exptime_arr[i]*DC_map
        # upper and lower bounds
        res_up = ((mean_stack[i]+np.abs(stacks_err)) -
                  EMgain_arr[i]*(CIC_map-CIC_std_map) - (FPN_map-FPN_std_map)
                  - EMgain_arr[i]*exptime_arr[i]*(DC_map-DC_std_map))
        res_low = ((mean_stack[i]-np.abs(stacks_err)) -
                   EMgain_arr[i]*(CIC_map+CIC_std_map) - (FPN_map+FPN_std_map)
                  - EMgain_arr[i]*exptime_arr[i]*(DC_map+DC_std_map))
        res_prescan = slice_section(res, 'SCI', 'prescan', detector_regions)
        res_up_prescan = slice_section(res_up, 'SCI', 'prescan', detector_regions)
        res_low_prescan = slice_section(res_low, 'SCI', 'prescan', detector_regions)
        # prescan may contain NaN'ed telemetry rows, so use nanmedian
        bias_offset[i] = np.nanmedian(res_prescan)/kgain_arr[i] # in DN
        bias_offset_up[i] = np.nanmedian(res_up_prescan)/kgain_arr[i] # in DN
        bias_offset_low[i] = np.nanmedian(res_low_prescan)/kgain_arr[i] # in DN
    bias_offset = np.mean(bias_offset)
    bias_offset_up = np.mean(bias_offset_up)
    bias_offset_low = np.mean(bias_offset_low)

    # don't average read noise for all frames since reading off read noise
    # from a frame with gained variance of C and D much higher than read
    # noise variance is not doable
    # read noise must be comparable to gained variance of D and C
    # so we read it off for lowest-gain-time frame
    l = np.argmin((EMgain_arr*exptime_arr))
    # Num below accounts for pixels lost to cosmic rays
    Num = mean_num_good_fr[l]
    # take std of just image area; more variance if image and different regions
    # included

    mean_stack_image = imaging_slice('SCI', mean_stack[l], detector_regions)
    read_noise2 = (np.var(mean_stack_image)*Num -
        EMgain_arr[l]**2*
        np.var(DC_image_map*exptime_arr[l]+CIC_image_map) -
        ((Num-1)*avg_corr+1)*np.var(FPN_image_map))
    if read_noise2 >= 0:
        read_noise = np.sqrt(read_noise2)
    else:
        read_noise = np.nan
        warnings.warn('read_noise is NaN.  The number of frames per substack '
                     'should be higher in order for this read noise estimate '
                     'to be reliable. However, if the lowest gain setting '
                      'is much larger than the gain used in the substack, '
                      'the best estimate for read noise may not be good.')

    # actual dark current should only be present in CCD pixels (image area),
    # even if we get erroneous non-zero D values in non-CCD pixels.  Let D_map
    # be zeros everywhere except for the image area
    DC_map = np.zeros_like(unreliable_pix_map).astype(float)
    DC_std_map = np.zeros_like(unreliable_pix_map).astype(float)
    # and reset the telemetry rows to NaN
    DC_map[telem_rows] = np.nan

    im_rows, im_cols, r0c0 = imaging_area_geom('SCI', detector_regions)
    DC_map[r0c0[0]:r0c0[0]+im_rows,
                    r0c0[1]:r0c0[1]+im_cols] = DC_image_map
    DC_std_map[r0c0[0]:r0c0[0]+im_rows,
                    r0c0[1]:r0c0[1]+im_cols] = D_std_map_im

    # now catch any elements that were negative for C and D:
    DC_map[DC_map < 0] = 0
    CIC_map[CIC_map < 0] = 0
    #mean taken before zeroing out the negatives for C and D
    FPN_image_mean = np.mean(FPN_image_map)
    CIC_image_mean = np.mean(CIC_image_map)
    DC_image_mean = np.mean(DC_image_map)
    CIC_image_map[CIC_image_map < 0] = 0
    DC_image_map[DC_image_map < 0] = 0

    # Since CIC will not be scaled by gain or exptime when master dark created
    # using these noise maps, just bundle stacks_err in with C_std_map
    C_std_map_combo = np.sqrt(CIC_std_map**2 + stacks_err**2)
    # assume headers from a dataset frame for headers of calibrated noise map
    prihdr = datasets[0].frames[0].pri_hdr
    exthdr = datasets[0].frames[0].ext_hdr
    exthdr['EXPTIME'] = None
    if 'EMGAIN_M' in exthdr.keys():
        exthdr['EMGAIN_M'] = None
    exthdr['CMDGAIN'] = None
    exthdr['KGAIN'] = None
    exthdr['BUNIT'] = 'detected electrons'
    exthdr['HIERARCH DATA_LEVEL'] = None

    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electrons'

    exthdr['DATATYPE'] = 'DetectorNoiseMaps'

    # bias offset
    exthdr['B_O'] = bias_offset
    bo_err_bar = max(bias_offset_up - bias_offset,
                     bias_offset - bias_offset_low)
    exthdr['B_O_ERR'] = bo_err_bar
    exthdr['B_O_UNIT'] = 'DN'

    input_stack = np.stack([FPN_map, CIC_map, DC_map])
    input_err = np.stack([[FPN_std_map, C_std_map_combo, DC_std_map]])
    input_dq = np.stack([output_dq, output_dq, output_dq])

    noise_maps = DetectorNoiseMaps(input_stack, prihdr.copy(), exthdr.copy(), dataset,
                           input_err, input_dq, err_hdr=err_hdr)
    
    l2a_data_filename = dataset.copy()[0].filename
    noise_maps.filename =  l2a_data_filename[:-5] + '_DetectorNoiseMaps.fits'

    return noise_maps


def build_synthesized_dark(dataset, noisemaps, detector_regions=None, full_frame=False):
        """
        Assemble a master dark SCI frame (full or image)
        from individual noise components.

        This is done this way because the actual dark frame varies with gain and
        exposure time, and both of these values may vary over orders of magnitude
        in the course of acquisition, alignment, and HOWFSC.  Better to take data
        sets that don't vary.

        Output is a bias-subtracted, gain-divided master dark in detected electrons.
        (Bias is inherently subtracted as we don't use it as
        one of the building blocks to assemble the dark frame.)

        Master dark = (F + g*t*D + g*C)/g = F/g + t*D + C where
        F = FPN (fixed-pattern noise) map
        g = EM gain
        t = exposure time
        D = dark current map
        C = CIC (clock-induced charge) map

        Arguments:
        dataset: corgidrp.data.Dataset instance.  The dataset should consist of
            frames all with the same EM gain and exposure time, which are read
            off from the dataset headers.
        noisemaps: corgidrp.data.DetectorNoiseMaps instance.  The noise maps used
            to build the master dark.
        detector_regions: dict.  A dictionary of detector geometry properties.
            Keys should be as found in detector_areas in detector.py. Defaults to
            detector_areas in detector.py.
        full_frame: bool.  If True, a full-frame master dark is generated (which
            may be useful for the module that statistically fits a frame to find
            the empirically applied EM gain, for example). If False, an image-area
            master dark is generated.  Defaults to False.

        Returns:
        master_dark:  corgidrp.data.Dark instance.
            This contains the master dark in detected electrons.

        """
        if detector_regions is None:
            detector_regions = detector_areas

        noise_maps = noisemaps.copy()
        Fd = noise_maps.FPN_map
        Dd = noise_maps.DC_map
        Cd = noise_maps.CIC_map
        _, unique_vals = dataset.split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN', 'KGAIN'])
        if len(unique_vals) > 1:
            raise Exception('Input dataset should contain frames of the same exposure time, commanded EM gain, and k gain.')
        try: # use measured EM gain if available TODO change hdr name if necessary
            g = dataset.frames[0].ext_hdr['EMGAIN_M']
        except:
            try: # use applied EM gain if available
                g = dataset.frames[0].ext_hdr['EMGAIN_A']
            except: # otherwise, use commanded EM gain
                g = dataset.frames[0].ext_hdr['CMDGAIN']
        t = dataset.frames[0].ext_hdr['EXPTIME']

        rows = detector_regions['SCI']['frame_rows']
        cols = detector_regions['SCI']['frame_cols']
        if Fd.shape != (rows, cols):
            raise TypeError('F must be ' + str(rows) + 'x' + str(cols))
        if Dd.shape != (rows, cols):
            raise TypeError('D must be ' + str(rows) + 'x' + str(cols))
        if Cd.shape != (rows, cols):
            raise TypeError('C must be ' + str(rows) + 'x' + str(cols))
        if (Dd < 0).any():
            raise TypeError('All elements of D must be >= 0')
        if (Cd < 0).any():
            raise TypeError('All elements of C must be >= 0')
        if g < 1:
            raise TypeError('Gain must be a value >= 1.')

        if not full_frame:
            Fd = slice_section(Fd, 'SCI', 'image', detector_regions)
            Dd = slice_section(Dd, 'SCI','image', detector_regions)
            Cd = slice_section(Cd, 'SCI', 'image', detector_regions)
            Ferr = slice_section(noise_maps.FPN_err, 'SCI', 'image', detector_regions)
            Derr = slice_section(noise_maps.DC_err, 'SCI', 'image', detector_regions)
            Cerr = slice_section(noise_maps.CIC_err, 'SCI', 'image', detector_regions)
            Fdq = slice_section(noise_maps.dq[0], 'SCI', 'image', detector_regions)
            Ddq = slice_section(noise_maps.dq[2], 'SCI', 'image', detector_regions)
            Cdq = slice_section(noise_maps.dq[1], 'SCI', 'image', detector_regions)
        else:
            Ferr = noise_maps.FPN_err
            Cerr = noise_maps.CIC_err
            Derr = noise_maps.DC_err
            Fdq = noise_maps.dq[0]
            Ddq = noise_maps.dq[2]
            Cdq = noise_maps.dq[1]

        # get from one of the noise maps and modify as needed
        prihdr = noise_maps.pri_hdr
        exthdr = noise_maps.ext_hdr
        errhdr = noise_maps.err_hdr
        exthdr['NAXIS1'] = Fd.shape[0]
        exthdr['NAXIS2'] = Fd.shape[1]
        exthdr['DATATYPE'] = 'Dark'
        exthdr['CMDGAIN'] = g # reconciling measured vs applied vs commanded not important for synthesized product; this is simply the user-specified gain
        exthdr['EXPTIME'] = t
        # wipe clean so that the proper documenting occurs for dark
        exthdr.pop('DRPNFILE')
        exthdr.pop('HISTORY')
        # this makes the filename of the dark have "_DetectorNoiseMaps_Dark" in
        # the name so that it is known that this Dark came from noise maps
        input_data = [noise_maps]
        md_data = Fd/g + t*Dd + Cd
        md_noise = np.sqrt(Ferr**2/g**2 + t**2*Derr**2 + Cerr**2)
        # DQ values are 0 or 1 for F, D, and C
        FDdq = np.bitwise_or(Fdq, Ddq)
        FDCdq = np.bitwise_or(FDdq, Cdq)

        master_dark = Dark(md_data, prihdr, exthdr, input_data, md_noise, FDCdq,
                        errhdr)
        
        return master_dark