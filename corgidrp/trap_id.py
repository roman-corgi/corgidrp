import numpy as np

import check as check

def illumination_correction(img, binsize, ill_corr):
    """Performs non-uniform illumination correction by taking sections of the
    image and performing local illumination subtraction.
    Parameters
    ----------
    img : 2-D array
        Image to be corrected.
    binsize : int > 0 or None
        Number of pixels over which to average for subtraction.  If None,
        acts as if ill_corr is False (see below).
    ill_corr : bool
        If True, subtracts the local median of the square region of side length
        equal to binsize from each pixel.  If False, simply subtracts from
        each pixel the median of the whole image region.
    Returns
    -------
    corrected_img : 2-D array
        Corrected image.
    local_ill : 2-D array
        Frame with pixel values equal to the amount that was subtracted from
        each.
    """
    # TODO v2:  use a sliding bin to correct (takes care of linear differences
    # b/w bins)

    # check inputs
    try:
        img = np.array(img).astype(float)
    except:
        raise TypeError("img elements should be real numbers")
    check.twoD_array(img, 'img', TypeError)
    if binsize is not None:
        check.positive_scalar_integer(binsize, 'binsize', TypeError)
    if type(ill_corr) != bool:
        raise TypeError('ill_corr must be True or False')

    if (not ill_corr) or (binsize is None):
        loc_ill = np.median(img)*np.ones_like(img).astype(float)
        corrected_img = img - loc_ill

    if ill_corr and (binsize is not None):
        # ensures there is a bin that runs all the way to the end
        if np.mod(len(img), binsize) == 0:
            row_bins = np.arange(0, len(img)+1, binsize)
        else:
            row_bins = np.arange(0, len(img), binsize)
        # If len(img) not divisible by binsize, this makes last bin of size
        # binsize + the remainder
        row_bins[-1] = len(img)

        # same thing for columns now
        if np.mod(len(img[0]), binsize) == 0:
            col_bins = np.arange(0, len(img[0])+1, binsize)
        else:
            col_bins = np.arange(0, len(img[0]), binsize)
        col_bins[-1] = len(img[0])

        # initializing
        loc_ill = (np.zeros([len(row_bins)-1, len(col_bins)-1])).astype(float)

        corrected_img = (np.zeros([len(img), len(img[0])])).astype(float)

        for i in range(len(row_bins)-1):
            for j in range(len(col_bins)-1):
                loc_ill[i,j]=np.median(img[int(row_bins[i]):int(row_bins[i+1]),
                                    int(col_bins[j]):int(col_bins[j+1])])
                corrected_img[int(row_bins[i]):int(row_bins[i+1]),
                    int(col_bins[j]):int(col_bins[j+1])] = \
                    img[int(row_bins[i]):int(row_bins[i+1]),
                    int(col_bins[j]):int(col_bins[j+1])] - loc_ill[i,j]

    #getting per pixel local_ill:
    local_ill = img - corrected_img

    return corrected_img, local_ill

# length_limit makes sure there are enough points near the peak of the prob
# func for fitting it, essentially.  If the dipole only shows up for 5 frames,
# we still use the amp values from the other frames for that pixel b/c the
# dipole would still obey that prob func even when it's amp doesn't meet
# threshold
def trap_id(cor_img_stack, ill_corr_min, ill_corr_max, timings, thresh_factor,
    length_limit):
    """For a given temperature and scheme, this function finds dipoles by
    identifying adjacent pixels that meet a threshold
    amplitude above and below the mean, and the threshold must be met at a
    sufficient number of phase times.  It then identifies the bright pixel of
    every dipole and categorizes them ('above': bright pixel of dipole toward
    readout direction; 'below': bright pixel of dipole away from readout
    direction; 'both': one 'above' and one 'below' present).
    Parameters
    ----------
    cor_img_stack : 3-D array
        The stack is formed by trap-pumped images taken at different phase
        times.  Each frame should have the same diemsions.
        Units can be in e- or DN, but they are input as e- when this
        function is called in tpump_analysis().
    ill_corr_min : 2-D array
        For a given scheme stack over phase times, a frame with pixel values
        equal to the minimum median taken over phase times
        that was subtracted from each during
        illumination_correction().  If ill_corr was False, the median was
        global over the whole image region.  If ill_corr was True, the median
        was over a local square of side length binsize pixels in
        illumination_correction().
    ill_corr_max : 2-D array
        For a given scheme stack over phase times, a frame with pixel values
        equal to the maximum median taken over phase times
        that was subtracted from each during
        illumination_correction().  If ill_corr was False, the median was
        global over the whole image region.  If ill_corr was True, the median
        was over a local square of side length binsize pixels in
        illumination_correction().
    timings : array or castable to array
        An array of the phase times corresponding to the ordering of the frames
        in cor_img_stack.  Units of seconds.
    thresh_factor : float, > 0
        Number of standard deviations from the mean a dipole should stand out
        in order to be considered for a trap. If this is too high, you get
        dipoles with amplitude that continually increases with phase time (not
        characteristic of an actual trap).  If thresh_factor is too low,
        you get a bad fit because the resulting dipoles have amplitudes that
        are too noisy and low.
    length_limit : int, > 0
        Minimum number of frames for which a dipole needs to meet threshold so
        that it goes forward for consideration as a true trap.
    Returns
    -------
    rc_above : dict
        A dictionay with a key for every bright pixel of an 'above' dipole over
        all phase times for a given scheme at a given temperature, in the
        following format:
        rc_above = { (row,col): {'amps_above': array([amp1, amp2, ...]),
        'loc_med_min': loc_med_min, 'loc_med_max': loc_med_max}, ...}
        'amps_above' is an array in the same order as the phase time order in
        timings (units of cor_image_stack).
        'loc_med_min' and 'loc_med_max' are the minimum and maximum bias values
        over all the phase times, respectively.  If bias subtraction was ideal,
        these values should be 0.  If
        illumination_correction() is also applied, the bias on the frame
        changes, and illumination_correction() will change these values
        accordingly.
    rc_below : dict
        A dictionay with a key for every bright pixel of an 'above' dipole over
        all phase times for a given scheme at a given temperature, in the
        following format:
        rc_below = { (row,col): {'amps_below': array([amp1, amp2, ...]),
        'loc_med_min': loc_med_min, 'loc_med_max': loc_med_max}, ...}
        'amps_below' is an array in the same order as the phase time order in
        timings (units of cor_image_stack).
        'loc_med_min' and 'loc_med_max' are the minimum and maximum bias values
        over all the phase times, respectively.  If bias subtraction was ideal,
        these values should be 0.  If
        illumination_correction() is also applied, the bias on the frame
        changes, and illumination_correction() will change these values
        accordingly.
    rc_both : dict
        A dictionary with a key for every bright pixel of an 'above' that is
        also the bright pixel of a 'below' dipole over
        all phase times for a given scheme at a given temperature, in the
        following format:
        rc_both = { (row,col): {'amps_both': array([amp1, amp2, ...]),
        'loc_med_min': loc_med_min, 'loc_med_max': loc_med_max,
        'above': {'amp': array([amp1a, amp2a, ...]),
                    't': array([t1a, t2a, ...]) },
        'below': {'amp': array([amp1b, amp2b, ...]),
                    't': array([t1b, t2b, ...]) }}, ... }
        'amps' (units of cor_image_stack) and 't' (units of seconds) under the
        'above' key are arrays with the same ordering that are identified
        specifically with the 'above' dipole, and similarly for the 'below'
        key.  The 'amps_both' value is an array of all the amplitudes for that
        pixel in the same order as timings.
    """
    # Check inputs
    try:
        #checks whether elements are good and whether each frame has same dims
        cor_img_stack = np.stack(cor_img_stack).astype(float)
    except:
        raise TypeError("cor_img_stack elements should be real numbers, and "
        "each frame should have the same dimensions.")
        # and complex numbers can't be cast as float
    check.threeD_array(cor_img_stack, 'cor_img_stack', TypeError)
    try:
        ill_corr_min = np.array(ill_corr_min).astype(float)
    except:
        raise TypeError("ill_corr_min elements should be real numbers")
        # and complex numbers can't be cast as float
    check.twoD_array(ill_corr_min, 'ill_corr_min', TypeError)
    if np.shape(ill_corr_min) != np.shape(cor_img_stack[0]):
        raise ValueError('The dimensions of ill_corr_min should match '
        'those of each frame in cor_img_stack.')
    try:
        ill_corr_max = np.array(ill_corr_max).astype(float)
    except:
        raise TypeError("ill_corr_max elements should be real numbers")
        # and complex numbers can't be cast as float
    check.twoD_array(ill_corr_max, 'ill_corr_max', TypeError)
    if np.shape(ill_corr_max) != np.shape(cor_img_stack[0]):
        raise ValueError('The dimensions of ill_corr_max should match '
        'those of each frame in cor_img_stack.')
    try:
        timings = np.array(timings).astype(float)
    except:
        raise TypeError("timings elements should be real numbers")
    check.oneD_array(timings, 'timings', TypeError)
    if len(timings) != len(cor_img_stack):
        raise ValueError('timings must have length equal to number of frames '
        'in cor_img_stack')
    check.real_positive_scalar(thresh_factor, 'thresh_factor', TypeError)
    check.positive_scalar_integer(length_limit, 'length_limit', TypeError)
    if length_limit > len(cor_img_stack):
        raise ValueError('length_limit cannot be longer than the number of '
        'frames in cor_img_stack')
    # This also ensures that cor_img_stack is not 0 frames

    IS_DIPOLE_UPPER = []
    IS_DIPOLE_LOWER = []
    for frame in cor_img_stack:
        IS_DIPOLE_UPPER.append((frame > (np.median(frame) +
                np.std(frame)*thresh_factor)).astype(int))
        IS_DIPOLE_LOWER.append((frame < (np.median(frame) -
                np.std(frame)*thresh_factor)).astype(int))
    IS_DIPOLE_UPPER = np.stack(IS_DIPOLE_UPPER)
    IS_DIPOLE_LOWER = np.stack(IS_DIPOLE_LOWER)

    # Dipole_above means the mean bright pixel of dipole is above
    Dipole_above = IS_DIPOLE_UPPER + np.roll(IS_DIPOLE_LOWER, -1, axis=1)
    Dipole_below = IS_DIPOLE_UPPER + np.roll(IS_DIPOLE_LOWER, 1, axis=1)
    # assign edge rows as 0 to eliminate false positive from "rolling"
    Dipole_above[:,-1,:] = 0
    Dipole_below[:,0,:] = 0

    # will have a 2 if both criteria met.  Turn that into a 1 for every dipole.
    # must be float to allow for fractional values in next step
    dipoles_above_count = (Dipole_above > 1).astype(float)
    dipoles_below_count = (Dipole_below > 1).astype(float)

    for t in range(len(timings)):
        #count how many frames there are for a given phase time
        # timings and dipoles_*_count have same length
        # divide by number of counts for each non-unique phase time
        pt_count = list(timings).count(list(timings)[t])
        dipoles_above_count[t] = dipoles_above_count[t]/pt_count
        dipoles_below_count[t] = dipoles_below_count[t]/pt_count

    # if at least one of the phase times with multiple frames has a dipole
    # in a given pixel, then ceil will count that phase time as a '1' in sum
    ALL_DIPOLES_sum_above = np.ceil(np.sum(dipoles_above_count, axis = 0))
    ALL_DIPOLES_sum_below = np.ceil(np.sum(dipoles_below_count, axis = 0))

    # there should be dipoles present in the locations for enough phase times
    dipoles_above_thresh = (ALL_DIPOLES_sum_above >= length_limit).astype(int)
    dipoles_below_thresh = (ALL_DIPOLES_sum_below >= length_limit).astype(int)
    [row_coords_above, col_coords_above] = np.where(dipoles_above_thresh)
    [row_coords_below, col_coords_below] = np.where(dipoles_below_thresh)
    above_rc = list(zip(row_coords_above, col_coords_above))
    below_rc = list(zip(row_coords_below, col_coords_below))
    # now remove the coordinates shared between both, and collect common
    # coordinates into both_rc
    set_above_rc = set(above_rc) - set(below_rc)
    set_below_rc = set(below_rc) - set(above_rc)
    both_rc = list(set(above_rc) - set_above_rc)
    above_rc = list(set_above_rc)
    below_rc = list(set_below_rc)
    rc_both = {}
    amps_both = np.zeros([len(both_rc), len(cor_img_stack)])

    for i in range(len(both_rc)):
        amps_both[i] = cor_img_stack[:, both_rc[i][0], both_rc[i][1]].copy()
        rc_both[both_rc[i]] = {'amps_both': amps_both[i],
            'loc_med_min': ill_corr_min[both_rc[i][0], both_rc[i][1]].copy(),
            'loc_med_max': ill_corr_max[both_rc[i][0], both_rc[i][1]].copy()}
        # getting times that corresponded to above and below dipoles
        rc_both[both_rc[i]]['above'] = {'amp': [], 't': []}
        rc_both[both_rc[i]]['below'] = {'amp': [], 't': []}
        for z in range(len(dipoles_above_count)):
            if dipoles_above_count[z][both_rc[i]] > 0:
                rc_both[both_rc[i]]['above']['amp'].append(
                            cor_img_stack[z][both_rc[i]])
                rc_both[both_rc[i]]['above']['t'].append(timings[z])
        for z in range(len(dipoles_below_count)):
            if dipoles_below_count[z][both_rc[i]] > 0:
                rc_both[both_rc[i]]['below']['amp'].append(
                        cor_img_stack[z][both_rc[i]])
                rc_both[both_rc[i]]['below']['t'].append(timings[z])
        rc_both[both_rc[i]]['above']['amp'] = np.array(
                        rc_both[both_rc[i]]['above']['amp'])
        rc_both[both_rc[i]]['above']['t'] = np.array(
                        rc_both[both_rc[i]]['above']['t'])
        rc_both[both_rc[i]]['below']['amp'] = np.array(
                        rc_both[both_rc[i]]['below']['amp'])
        rc_both[both_rc[i]]['below']['t'] = np.array(
                        rc_both[both_rc[i]]['below']['t'])

    rc_above = {}
    amps_above = np.zeros([len(above_rc),len(cor_img_stack)])
    for i in range(len(above_rc)):
        amps_above[i] = cor_img_stack[:, above_rc[i][0], above_rc[i][1]].copy()
        rc_above[above_rc[i]] = {'amps_above': amps_above[i],
            'loc_med_min': ill_corr_min[above_rc[i][0], above_rc[i][1]].copy(),
            'loc_med_max': ill_corr_max[above_rc[i][0], above_rc[i][1]].copy()}

    # same thing as above, but now for the "below" dipoles
    rc_below = {}
    amps_below = np.zeros([len(below_rc),len(cor_img_stack)])
    for i in range(len(below_rc)):
        amps_below[i] = cor_img_stack[:, below_rc[i][0], below_rc[i][1]].copy()
        rc_below[below_rc[i]] = {'amps_below': amps_below[i],
            'loc_med_min': ill_corr_min[below_rc[i][0], below_rc[i][1]].copy(),
            'loc_med_max': ill_corr_max[below_rc[i][0], below_rc[i][1]].copy()}

    return rc_above, rc_below, rc_both