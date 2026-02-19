import numpy as np
import re
import warnings
from astropy.io import fits

from corgidrp.detector import slice_section, imaging_slice, imaging_area_geom, unpack_geom, detector_areas
import corgidrp.check as check
from corgidrp.data import DetectorNoiseMaps, Dark, Image, Dataset, typical_cal_invalid_keywords, typical_bool_keywords

def mean_combine(dataset_or_image_list, bpmap_list, err=False):
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
        dataset_or_image_list (data.Dataset, list, or array_like): Dataset or list (or stack) of L2b data frames
    (with no bad pixels applied to them).
        bpmap_list (list or array_like): List (or stack) of bad-pixel maps
        associated with L2b data frames. Each must be 0 (good) or 1 (bad)
        at every pixel. If first input is a Dataset, this input is ignored.
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
    # uncomment for RAM check
    # import psutil
    # process = psutil.Process()

    if not isinstance(dataset_or_image_list, Dataset):
        # if input is an np array or stack, try to accommodate
        if type(dataset_or_image_list) == np.ndarray:
            if dataset_or_image_list.ndim == 1: # pathological case of empty array
                dataset_or_image_list = list(dataset_or_image_list)
            elif dataset_or_image_list.ndim == 2: #covers case of single 2D frame
                dataset_or_image_list = [dataset_or_image_list]
            elif dataset_or_image_list.ndim == 3: #covers case of stack of 2D frames
                dataset_or_image_list = list(dataset_or_image_list)
        if type(bpmap_list) == np.ndarray:
            if bpmap_list.ndim == 1: # pathological case of empty array
                bpmap_list = list(bpmap_list)
            elif bpmap_list.ndim == 2: #covers case of single 2D frame
                bpmap_list = [bpmap_list]
            elif bpmap_list.ndim == 3: #covers case of stack of 2D frames
                bpmap_list = list(bpmap_list)

        # Check inputs
        if not isinstance(bpmap_list, list):
            raise TypeError('bpmap_list must be a list')
        if len(dataset_or_image_list) != len(bpmap_list):
            raise TypeError('image_list and bpmap_list must be the same length')
        if len(dataset_or_image_list) == 0:
            raise TypeError('input lists cannot be empty')
        s0 = dataset_or_image_list[0].shape
        for index, im in enumerate(dataset_or_image_list):
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

    # Add non masked elements
    if isinstance(dataset_or_image_list, Dataset):
        temp_fits = Image(dataset_or_image_list[0].filepath)
        if err:
            shape = temp_fits.err.shape[1:] 
        else:
            shape = temp_fits.data.shape 
        sum_im = np.zeros(shape).astype(float)
        map_im = np.zeros(shape, dtype=int)
    elif isinstance(dataset_or_image_list, list) or isinstance(dataset_or_image_list, np.array):  
        sum_im = np.zeros_like(dataset_or_image_list[0]).astype(float)
        map_im = np.zeros_like(dataset_or_image_list[0], dtype=int)
    else:
        raise TypeError('image_list must be a list, array-like, or a Dataset')

    for i in range(len(dataset_or_image_list)):
        if isinstance(dataset_or_image_list, Dataset):
            if dataset_or_image_list[0].data is None:
                temp_fits = Image(dataset_or_image_list[i].filepath)
            else:
                temp_fits = dataset_or_image_list[i]
            if err:
                frame_data = temp_fits.err[0] 
            else:
                frame_data = temp_fits.data.astype(float)
            im_m = np.ma.masked_array(frame_data, temp_fits.dq.astype(bool).astype(int))
        else: #list
            im_m = np.ma.masked_array(dataset_or_image_list[i], bpmap_list[i])
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
    if map_im.min() < len(dataset_or_image_list)/2:
        enough_for_rn = False

    # uncomment for RAM check
    # mem = process.memory_info()
    # # peak_wset is only available on Windows; fall back to rss on other platforms
    # if hasattr(mem, 'peak_wset') and getattr(mem, 'peak_wset') is not None:
    #     peak_memory = mem.peak_wset / (1024 ** 2)  # convert to MB
    # else:
    #     peak_memory = mem.rss / (1024 ** 2)  # convert to MB
    # print(f"mean_combine peak memory usage:  {peak_memory:.2f} MB")

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
    - have been divided by EM gain
    - have NOT been desmeared. Darks should not be desmeared.  The only component 
    of dark frames that would be subject to a smearing effect is dark current 
    since it linearly increases with time, so the extra row read time affects 
    the dark current per pixel.  However, illuminated images
    would also contain this smeared dark current, so dark subtraction should 
    remove this smeared dark current (and then desmearing may be applied to the 
    processed image if appropriate).  

    Also, add_shot_noise_to_err() should NOT have been applied to the frames in
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
        If Dataset has metadata only (as in RAM-heavy case), 
        each frame is read in from its filepath one at a time.  If Dataset has 
        its data, then all the frames are processed at once. 
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
        master_dark.dq: pixels that are masked for all frames have non-zero 
        values.
    """
    if detector_regions is None:
            detector_regions = detector_areas

    _, unique_vals = dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
    if len(unique_vals) > 1:
        raise Exception('Input dataset should contain frames of the same exposure time, commanded EM gain, and k gain.')
    # getting telemetry rows to ignore in fit
    telem_rows_start = detector_params.params['TELRSTRT']
    telem_rows_end = detector_params.params['TELREND']
    telem_rows = slice(telem_rows_start, telem_rows_end)

    frames = []
    bpmaps = []
    errs = []
    if dataset[0].data is not None:
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
    else:
        frames = dataset
        bpmaps = None #not used in this case
        errs = frames
        test_image = Image(dataset.frames[0].filepath)
        test_frame = test_image.data.astype(float)
        test_frame[telem_rows] = np.nan
        i0 = slice_section(test_frame, 'SCI', 'image', detector_regions)
        if np.isnan(i0).any():
            raise ValueError('telem_rows cannot be in image area.')
        test_frame[telem_rows] = 0
    mean_frame, combined_bpmap, unmasked_num, _ = mean_combine(frames, bpmaps)
    mean_err, _, _, _ = mean_combine(errs, bpmaps, err=True)
    if dataset[0].data is None:
        # equivalent to what is done in if statement above for datasets with data
        mean_frame[telem_rows] = 0 
        mean_err[telem_rows] = 0 
    # combine the error from individual frames to the standard deviation across
    # the frames due to statistical variance
    zero_inds = np.where(unmasked_num==0)
    nonzero_inds = np.where(unmasked_num!=0)
    if dataset[0].data is None:
        sum_squares = np.zeros_like(mean_frame).astype(float)
        for f in dataset.frames:
            temp_frame = Image(f.filepath)
            frame_data = temp_frame.data.astype(float)
            # equivalent to what is done in if statement above for datasets with data
            frame_data[telem_rows] = 0
            mask = temp_frame.dq.astype(bool).astype(int)
            masked_frame = np.ma.masked_array(frame_data, mask)
            masked_mean = np.ma.masked_array(mean_frame, combined_bpmap)
            sum_squares += (masked_frame - masked_mean)**2
        stat_std = np.zeros_like(sum_squares).astype(float)
        stat_std[nonzero_inds] = np.ma.sqrt(sum_squares[nonzero_inds]/unmasked_num[nonzero_inds])/np.sqrt(unmasked_num[nonzero_inds]) #standard error=std/sqrt(N)
        stat_std[zero_inds] = 0
        stat_std = np.ma.getdata(stat_std)
    else:
        masked_frames = np.ma.masked_array(frames, bpmaps)
        stat_std = np.zeros_like(frames[0]).astype(float)
        stat_std[nonzero_inds] = np.ma.std(masked_frames[:, nonzero_inds[0], nonzero_inds[1]], axis=0)/np.sqrt(unmasked_num[nonzero_inds])
        stat_std[zero_inds] = 0
        stat_std = np.ma.getdata(stat_std)
    rows_one, cols_one = np.where(unmasked_num==1)
    rows_normal, cols_normal = np.where(unmasked_num == unmasked_num.max())
    if unmasked_num.max() <= 1: # this would virtually never happen
        raise Exception('No pixels found to have more than 1 acceptable frame in a sub-stack.')
    # now pick a pixel from rows_normal and cols_normal to use as a reference for the approximated error for the pixels that have 1 unmasked frame, undo the division by sqrt(unmasked_num), and divide by 1
    stat_std[rows_one, cols_one] = stat_std[rows_normal[0], cols_normal[0]] * np.sqrt(unmasked_num.max())/1
    total_err = np.sqrt(mean_err**2 + stat_std**2)
    # bitwise_or flag value for those that are masked all the way through for all
    # frames
    fittable_inds = np.where(combined_bpmap ==0) 
    if dataset[0].data is None:
        dq_sum = np.zeros_like(mean_frame).astype(float)
        for j in range(len(dataset)):
            dq_temp = Image(dataset[j].filepath).dq
            dq_sum += dq_temp.astype(float)
        dq_sum = np.ma.masked_array(dq_sum, dq_sum == 0)
        output_dq = 2**((np.ma.log(dq_sum)/np.log(2)).astype(int)) - 1
        output_dq = output_dq.filled(0).astype(int)
    else:
        output_dq = np.bitwise_or.reduce(dataset.all_dq, axis=0)
    output_dq[fittable_inds] = 0 
    if not full_frame:
        dq = slice_section(output_dq, 'SCI', 'image', detector_regions)
        err = slice_section(total_err, 'SCI', 'image', detector_regions)
        data = slice_section(mean_frame, 'SCI', 'image', detector_regions)
    else:
        dq = output_dq
        err = total_err
        data = mean_frame

    invalid_trad_drk_keywords = typical_cal_invalid_keywords 
    # Remove specific keywords
    for key in ['PROGNUM', 'EXECNUM', 'CAMPAIGN', 'SEGMENT', 'VISNUM', 'OBSNUM', 'CPGSFILE', 'EXPTIME', 'EMGAIN_C', 'KGAINPAR', 'RN', 'RN_ERR', 'KGAIN_ER', 'HVCBIAS']:
        if key in invalid_trad_drk_keywords:
            invalid_trad_drk_keywords.remove(key)
    prihdr, exthdr, errhdr, dqhdr = check.merge_headers(dataset, any_true_keywords=typical_bool_keywords, invalid_keywords=invalid_trad_drk_keywords)
    
    exthdr['NAXIS1'] = data.shape[1]
    exthdr['NAXIS2'] = data.shape[0]
    exthdr['DATATYPE'] = 'Dark'

    master_dark = Dark(data, prihdr, exthdr, dataset, err, dq, errhdr, dqhdr)
    master_dark.ext_hdr['DRPNFILE'] = int(np.round(np.nanmean(unmasked_num)))
    master_dark.ext_hdr['BUNIT'] = 'detected electron'
    master_dark.err_hdr['BUNIT'] = 'detected electron'

    msg = 'traditional master analog dark (not synthesized from detector noise maps)'
    master_dark.ext_hdr.add_history(msg)

    return master_dark



class CalDarksLSQException(Exception):
    """Exception class for calibrate_darks_lsq."""

def calibrate_darks_lsq(dataset, detector_params, weighting=True, detector_regions=None):
    """The input dataset represents a collection of frame stacks of the
    (in e- units), where the stacks are for various
    EM gain values and exposure times.  Stacks with fewer frames than other 
    stacks are accordingly weighed less in the fit.  
    The frames in each stack should be SCI full frames that:

    - have had their bias subtracted (assuming full frame)
    - have had masks made for cosmic rays
    - have been corrected for nonlinearity
    - have been converted from DN to e-
    - have NOT been desmeared. Darks should not be desmeared.  The only component 
    of dark frames that would be subject to a smearing effect is dark current 
    since it linearly increases with time, so the extra row read time affects 
    the dark current per pixel.  However, illuminated images
    would also contain this smeared dark current, so dark subtraction should 
    remove this smeared dark current (and then desmearing may be applied to the 
    processed image if appropriate).  

    Also, add_shot_noise_to_err() should NOT have been applied to the frames in
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
        We recommend  >= 1176 frames for each stack if calibrating
        darks for analog frames,
        thousands for photon counting depending on the maximum number of
        frames that will be used for photon counting.
        If Dataset has metadata only (as in RAM-heavy case), 
        each frame is read in from its filepath one at a time.  If Dataset has 
        its data, then all the frames are processed at once.
    detector_params (corgidrp.data.DetectorParams):
        a calibration file storing detector calibration values
    weighting (bool):
        If True, weighting is used for the least squares fit, and the weighting
        takes into account the err coming from the input frames, the statistical
        variation among the supposedly identical frames in each sub-stack, and 
        the effect of any DQ masking.  If False, all data is evenly weighted in 
        the least squares fit.  Defaults to True.
    detector_regions (dict):
        a dictionary of detector geometry properties.  Keys should be as found
        in detector_areas in detector.py.
        Defaults to None, in which case detector_areas from detector.py is used.

    Returns:
    noise_maps : corgidrp.data.DetectorNoiseMaps instance
        Includes a 3-D stack of frames for the data, err, and the dq.
        input data: np.stack([FPN_map, CIC_map, DC_map])
        input err:  np.stack([FPN_std_map, C_std_map, DC_std_map])
        FPN_std_map, C_std_map, and DC_std_map contain the fitting error.
        In all the err, masked pixels are accounted for in
        the calculations, and the err from the input frames, along with statistical 
        error due to having fewer frames available per sub-stack due to any masking,
        is used for weighting the data in the least squares fit.
        input dq:   np.stack([output_dq, output_dq, output_dq])
        The pixels that are masked for EVERY frame in all sub-stacks
        but 3 (or less) are assigned a flag value from the combination of the frames.
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
        A per-pixel map of fixed-pattern noise (in detected electrons).  Any negative values
        from the fit are made positive in the end.
    CIC_map : array-like (full frame)
        A per-pixel map of EXCAM clock-induced charge (in detected electrons). Any negative
        values from the fit are made positive in the end.
    DC_map : array-like (full frame)
        A per-pixel map of dark current (in detected electrons/s). Any negative values
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
        A per-pixel map of fixed-pattern noise in the image area (in detected electrons).
        Any negative values from the fit are made positive in the end.
    CIC_image_map : array-like (image area)
        A per-pixel map of EXCAM clock-induced charge in the image area
        (in deteceted electrons). Any negative values from the fit are made positive in the end.
    DC_image_map : array-like (image area)
        A per-pixel map of dark current in the image area (in detected electrons/s).
        Any negative values from the fit are made positive in the end.
    FPNvar : float
        Variance of fixed-pattern noise map (in detected electrons).
    CICvar : float
        Variance of clock-induced charge map (in detected electrons).
    DCvar : float
        Variance of dark current map (in detected electrons).
    read_noise : float
        Read noise estimate from the noise profile of a mean frame (in detected electrons).
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
        are >= len(stack_arr)-3 cannot be fit.  
        The pixels that are masked for EVERY frame in all sub-stacks
        but 3 (or less) are assigned a flag value from the combination of the frames.
        These pixels would have no reliability for dark subtraction.
    FPN_std_map : array-like (full frame)
        The standard deviation per pixel for the calibrated FPN.
    CIC_std_map : array-like (full frame)
        The standard deviation per pixel for the calibrated CIC.
    DC_std_map : array-like (full frame)
        The standard deviation per pixel for the calibrated dark current.
    """
    if type(weighting) != bool:
        raise ValueError('The input weighting should be either True or False.')
    if detector_regions is None:
            detector_regions = detector_areas
    if dataset[0].data is None:
        datasets, _ = dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
    else:
        datasets, _ = dataset.copy().split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
    if len(datasets) <= 3:
        raise CalDarksLSQException('Number of sub-stacks in datasets must '
                'be more than 3 for proper curve fit.')
    # getting telemetry rows to ignore in fit
    telem_rows_start = detector_params.params['TELRSTRT']
    telem_rows_end = detector_params.params['TELREND']
    telem_rows = slice(telem_rows_start, telem_rows_end)

    EMgain_arr = np.array([])
    exptime_arr = np.array([])
    kgain_arr = np.array([])
    mean_frames = []
    total_errs = []
    mean_num_good_fr = []
    output_dqs = []
    unreliable_pix_map = np.zeros((detector_regions['SCI']['frame_rows'],
                                   detector_regions['SCI']['frame_cols'])).astype(int)
    unfittable_pix_map = unreliable_pix_map.copy()
    if len(dataset) < 1176:
            print('The number of frames in dataset is less than 1176 frames, '
            'which is the minimum number for the analog synthesized '
            'master dark')
    for i in range(len(datasets)):
        frames = []
        bpmaps = []
        errs = []
        
        if i > 0:
            if np.shape(datasets[i-1].all_data)[1:] != np.shape(datasets[i].all_data)[1:]:
                raise CalDarksLSQException('All sub-stacks must have the same frame shape.')
        try: # if EM gain measured directly from frame
            EMgain_arr = np.append(EMgain_arr, datasets[i].frames[0].ext_hdr['EMGAIN_M'])
        except:
            if datasets[i].frames[0].ext_hdr['EMGAIN_A'] > 0: # use applied EM gain if available
                EMgain_arr = np.append(EMgain_arr, datasets[i].frames[0].ext_hdr['EMGAIN_A'])
            else: # use commanded gain otherwise
                EMgain_arr = np.append(EMgain_arr, datasets[i].frames[0].ext_hdr['EMGAIN_C'])
        exptime = datasets[i].frames[0].ext_hdr['EXPTIME']
        cmdgain = datasets[i].frames[0].ext_hdr['EMGAIN_C']
        kgain = datasets[i].frames[0].ext_hdr['KGAINPAR']
        exptime_arr = np.append(exptime_arr, exptime)
        kgain_arr = np.append(kgain_arr, kgain)
        
        if not datasets[i][0].data is None:
            check.threeD_array(datasets[i].all_data,
                'datasets['+str(i)+'].all_data', TypeError)
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
        else:
            frames = datasets[i]
            bpmaps = None #not used in this case
            errs = frames
            test_frame = Image(datasets[i].frames[0].filepath).data.astype(float)
            shape = test_frame.shape # for later, after mean_combine()
            test_frame[telem_rows] = np.nan
            i0 = slice_section(test_frame, 'SCI', 'image', detector_regions)
            if np.isnan(i0).any():
                raise ValueError('telem_rows cannot be in image area.')
            test_frame[telem_rows] = 0
        mean_frame, combined_bpmap, unmasked_num, _ = mean_combine(frames, bpmaps)
        mean_err, _, _, _ = mean_combine(errs, bpmaps, err=True)
        if dataset[0].data is None:
            # equivalent to what is done in if statement above for datasets with data
            mean_frame[telem_rows] = 0 
            mean_err[telem_rows] = 0 
        # combine the error from individual frames to the standard deviation across
        # the frames due to statistical variance
        zero_inds = np.where(unmasked_num==0)
        nonzero_inds = np.where(unmasked_num!=0)
        if datasets[i][0].data is None:
            sum_squares = np.zeros(shape).astype(float)
            for f in datasets[i].frames:
                temp_image = Image(f.filepath)
                test_data = temp_image.data.astype(float)
                # equivalent to what is done in if statement above for datasets with data
                test_data[telem_rows] = 0
                mask = temp_image.dq.astype(bool).astype(int) 
                masked_frame = np.ma.masked_array(test_data, mask)
                masked_mean = np.ma.masked_array(mean_frame, combined_bpmap)
                sum_squares += (masked_frame - masked_mean)**2
            stat_std = np.zeros_like(sum_squares).astype(float)
            stat_std[nonzero_inds] = np.ma.sqrt(sum_squares[nonzero_inds]/unmasked_num[nonzero_inds])/np.sqrt(unmasked_num[nonzero_inds]) #standard error=std/sqrt(N)
            stat_std[zero_inds] = 0
            stat_std = np.ma.getdata(stat_std)
        else:
            masked_frames = np.ma.masked_array(frames, bpmaps)
            stat_std = np.zeros_like(frames[0]).astype(float)
            stat_std[nonzero_inds] = np.ma.std(masked_frames[:, nonzero_inds[0], nonzero_inds[1]], axis=0)/np.sqrt(unmasked_num[nonzero_inds])
            stat_std[zero_inds] = 0
            stat_std = np.ma.getdata(stat_std)
        stat_std[telem_rows] = 1 # something non-zero; masked in the DQ anyways, and this assignment here prevents np.inf issues/warnings later
        # where the number of unmasked frames is 1, the std is 0, but we want error to increase as the number of usuable frames decreases, so fudge it a little:
        rows_one, cols_one = np.where(unmasked_num==1)
        if zero_inds[0].size != 0:
            stat_std[zero_inds] = 1 # just assign as something non-zero; doesn't really matter b/c such pixels will be masked in the DQ; this will prevent warning outputs
        rows_normal, cols_normal = np.where(unmasked_num == unmasked_num.max())
        if unmasked_num.max() <= 1: # this would virtually never happen
            raise Exception('No pixels found to have more than 1 acceptable frame in a sub-stack.')
        # now pick a pixel from rows_normal and cols_normal to use as a reference for the approximated error for the pixels that have 1 unmasked frame, undo the division by sqrt(unmasked_num), and divide by 1
        stat_std[rows_one, cols_one] = stat_std[rows_normal[0], cols_normal[0]] * np.sqrt(unmasked_num.max())/1
        total_err = np.sqrt(mean_err**2 + stat_std**2)
        pixel_mask = (unmasked_num < len(datasets[i].frames)/2).astype(int)
        mean_num = np.mean(unmasked_num)
        mean_frame[telem_rows] = np.nan
        mean_frames.append(mean_frame)
        total_errs.append(total_err)
        mean_num_good_fr.append(mean_num)
        unreliable_pix_map += pixel_mask
        unfittable_pix_map += combined_bpmap
        # bitwise_or flag value for those that are masked all the way through for all
        # frames
        fittable_inds = np.where(combined_bpmap != 1)
        if datasets[i][0].data is None:
            dq_sum = np.zeros_like(mean_frame).astype(float)
            for j in range(len(datasets[i])):
                dq_temp = Image(datasets[i][j].filepath).dq
                dq_sum += dq_temp.astype(float)
            dq_sum = np.ma.masked_array(dq_sum, dq_sum == 0)
            output_dq = 2**((np.ma.log(dq_sum)/np.log(2)).astype(int)) - 1
            output_dq = output_dq.filled(0).astype(int)
        else:
            output_dq = np.bitwise_or.reduce(datasets[i].all_dq, axis=0)
        output_dq[fittable_inds] = 0 
        output_dqs.append(output_dq)
    output_dqs = np.stack(output_dqs)
    unreliable_pix_map = unreliable_pix_map.astype(int)
    mean_stack = np.stack(mean_frames)
    mean_err_stack = np.stack(total_errs)

    # uncomment for RAM check 
    # import psutil
    # process = psutil.Process()

    # flag value for those that are masked all the way through for all
    # but 3 (or fewer) stacks
    output_dq = np.bitwise_or.reduce(output_dqs, axis=0)
    fittable_ind = np.where(unfittable_pix_map < len(datasets)-3)
    output_dq[fittable_ind] = 0

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

    # matrix to be used for least squares and covariance matrix
    # Create Xx with shape (M, 3, rows, cols), where M = len(EMgain_arr)
    rows, cols = mean_stack.shape[1], mean_stack.shape[2]
    X = np.array([np.ones([len(EMgain_arr)]).astype(float), EMgain_arr, EMgain_arr*exptime_arr]).T  # (M,3)
    Xx = np.broadcast_to(X[:, :, None, None], (len(EMgain_arr), 3, rows, cols))
    # weighting matrix; sub-stacks with few usable frames get a low weight
    mean_err_stack[telem_rows] = 1 # instead of 0 to avoid inf weighting
    if weighting:
        W = 1/mean_err_stack
    else:
        W = np.ones_like(mean_err_stack) # all weighted the same
    wY = W*mean_stack
    wX = np.transpose(W*np.transpose(Xx, (1,0,2,3)), (1,0,2,3))
    wXTwX = np.einsum('ji...,ik...',np.transpose(wX,(1,0,2,3)), wX)
    wXTwXinv = np.linalg.inv(wXTwX)
    pinv_wX = np.einsum('...ij,jk...', wXTwXinv, np.transpose(wX,(1,0,2,3)))
    params_t = np.einsum('...ij,j...', pinv_wX, wY)
    params = np.transpose(params_t,(2,0,1))
    
    #next line: checked with KKT method for including bounds
    #actually, do this after determining everything else so that
    # bias_offset, etc is accurate
    #params_Y[params_Y < 0]= 0
    FPN_map = params[0]
    CIC_map = params[1]
    DC_map = params[2]
    # using chi squared for ordinary least squares (OLS) variance estiamate
    # This is OLS since the parameters to fit are linear in fit function
    # 3: number of fitted params
    params_Y = np.transpose(params_t, (1,2,0))
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
        res_up = ((mean_stack[i]) -
                  EMgain_arr[i]*(CIC_map-CIC_std_map) - (FPN_map-FPN_std_map)
                  - EMgain_arr[i]*exptime_arr[i]*(DC_map-DC_std_map))
        res_low = ((mean_stack[i]) -
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
    #mean taken before zeroing out the negatives for C and D for better statistical representation of mean value)
    FPN_image_mean = np.mean(FPN_image_map)
    FPN_image_median = np.median(FPN_image_map)
    CIC_image_mean = np.mean(CIC_image_map)
    DC_image_mean = np.mean(DC_image_map)
    CIC_image_map[CIC_image_map < 0] = 0
    DC_image_map[DC_image_map < 0] = 0

    invalid_dnm_keywords = typical_cal_invalid_keywords + ['EXPTIME', 'EMGAIN_C', 'EMGAIN_A', 'KGAINPAR', 'KGAIN_ER', 'HVCBIAS']
    # Remove specific keywords
    for key in ['PROGNUM', 'EXECNUM', 'CAMPAIGN', 'SEGMENT', 'VISNUM', 'OBSNUM', 'CPGSFILE']:
        if key in invalid_dnm_keywords:
            invalid_dnm_keywords.remove(key)
    prihdr, exthdr, err_hdr, dq_hdr = check.merge_headers(dataset, invalid_keywords=invalid_dnm_keywords)
    if 'EMGAIN_M' in exthdr.keys():
        exthdr['EMGAIN_M'] = -999.
    exthdr['BUNIT'] = 'detected electron'

    err_hdr['BUNIT'] = 'detected electron'

    exthdr['DATATYPE'] = 'DetectorNoiseMaps'

    # bias offset
    exthdr['B_O'] = bias_offset
    bo_err_bar = max(bias_offset_up - bias_offset,
                     bias_offset - bias_offset_low)
    exthdr['B_O_ERR'] = bo_err_bar
    exthdr['B_O_UNIT'] = 'DN'

    input_stack = np.stack([FPN_map, CIC_map, DC_map])
    input_err = np.stack([[FPN_std_map, CIC_std_map, DC_std_map]])
    input_dq = np.stack([output_dq, output_dq, output_dq])

    noise_maps = DetectorNoiseMaps(input_stack, prihdr, exthdr, dataset,
                           input_err, input_dq, err_hdr=err_hdr, dq_hdr=dq_hdr)
    
    noise_maps.ext_hdr['DRPNFILE'] = int(np.round(np.sum(mean_num_good_fr)))
    l2a_data_filename = dataset[-1].filename.split('.fits')[0]
    noise_maps.filename =  l2a_data_filename + '_dnm_cal.fits'
    noise_maps.filename = re.sub('_l[0-9].', '', noise_maps.filename)
    noise_maps.ext_hdr.set('FPN_IMM', FPN_image_mean, 'mean of the image-area fixed-pattern noise (e-). -999. if no value supplied.')
    noise_maps.ext_hdr.set('CIC_IMM', CIC_image_mean, 'mean of the image-area clock-induced charge (e-). -999. if no value supplied.')
    noise_maps.ext_hdr.set('DC_IMM', DC_image_mean, 'mean of the image-area dark current (e-/s). -999. if no value supplied.')
    noise_maps.ext_hdr.set('FPN_IMME', FPN_image_median, 'median of the image-area fixed-pattern noise (e-). -999. if no value supplied.')
    vals_list=[]
    for w1,w2,w3 in zip(exptime_arr, EMgain_arr, mean_num_good_fr):
        vals_list.append([float(w1),float(w2),float(w3)])
    noise_maps.ext_hdr['HISTORY'] = 'Detector noise maps created with the following sets of (exposure time (in s), EM gain, and number of frames):  {0}'.format(vals_list)

    # uncomment for RAM check
    # mem = process.memory_info()
    # # peak_wset is only available on Windows; fall back to rss on other platforms
    # if hasattr(mem, 'peak_wset') and getattr(mem, 'peak_wset') is not None:
    #     peak_memory = mem.peak_wset / (1024 ** 2)  # convert to MB
    # else:
    #     peak_memory = mem.rss / (1024 ** 2)  # convert to MB
    # print(f"calibrate_darks_lsq peak memory usage:  {peak_memory:.2f} MB")

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
        _, unique_vals = dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
        if len(unique_vals) > 1:
            raise Exception('Input dataset should contain frames of the same exposure time, commanded EM gain, and k gain.')
        try: # use measured EM gain if available
            g = dataset.frames[0].ext_hdr['EMGAIN_M']
        except:
            g = dataset.frames[0].ext_hdr['EMGAIN_A']
            if g > 0: # use applied EM gain if available
                g = dataset.frames[0].ext_hdr['EMGAIN_A']
            else: # otherwise, use commanded EM gain
                g = dataset.frames[0].ext_hdr['EMGAIN_C']
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
        # remove keywords that would not appear in Dark not made from noise maps
        for key in ['B_O', 'B_O_ERR', 'B_O_UNIT', 'FPN_IMM', 'CIC_IMM', 'DC_IMM',
                    'FPN_IMME']:
            if key in exthdr.keys():
                del exthdr[key]
        exthdr['NAXIS'] = 2
        exthdr['NAXIS1'] = Fd.shape[1]
        exthdr['NAXIS2'] = Fd.shape[0]
        if 'NAXIS3' in exthdr:
            del exthdr['NAXIS3']
        exthdr['DATATYPE'] = 'Dark'
        exthdr['EMGAIN_C'] = g # reconciling measured vs applied vs commanded not important for synthesized product; this is simply the user-specified gain
        exthdr['EXPTIME'] = t
        # one can check HISTORY to see that this Dark was synthesized from noise maps
        input_data = [noise_maps]
        md_data = Fd/g + t*Dd + Cd
        md_noise = np.sqrt(Ferr**2/g**2 + t**2*Derr**2 + Cerr**2)
        # DQ values are 0 or 1 for F, D, and C
        FDdq = np.bitwise_or(Fdq, Ddq)
        FDCdq = np.bitwise_or(FDdq, Cdq)

        master_dark = Dark(md_data, prihdr, exthdr, input_data, md_noise, FDCdq,
                        errhdr)
        master_dark.ext_hdr['DRPNFILE'] = noise_maps.ext_hdr['DRPNFILE']
        
        return master_dark