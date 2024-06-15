import os
from pathlib import Path
import numpy as np
import warnings

from corgidrp.detector import Metadata
import corgidrp.util.check as check
from corgidrp.util.mean_combine import mean_combine


here = os.path.abspath(os.path.dirname(__file__))
meta_path_default = Path(here, 'util', 'metadata_test.yaml')

class CalDarksLSQException(Exception):
    """Exception class for calibrate_darks_lsq."""

def calibrate_darks_lsq(datasets, meta_path=None):
    """The input datasets represents a collection of frame stacks of the
    same number of dark frames (in e- units), where the stacks are for various
    EM gain values and exposure times.  The frames in each stack should be
    SCI full frames that:

    - have had their bias subtracted (assuming 0 bias offset and full frame;
    this function calibrates bias offset)
    - have had masks made for cosmic rays
    - have been corrected for nonlinearity
    - have been converted from DN to e-
    - have had the cosmic ray masks combined with any bad pixel masks which may
    have come from pre-processing if there are any (because creation of the
    fixed bad pixel mask containing warm/hot pixels and pixels with sub-optimal
    functionality requires a master dark, which requires this function first)
    - have been desmeared if desmearing is appropriate.  Under normal
    circumstances, darks should not be desmeared.  The only time desmearing
    would be useful is in the unexpected case that, for example,
    dark current is so high that it stands far above other noise that is
    not smeared upon readout, such as clock-induced charge
    and fixed-pattern noise.

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
    for making a master dark.  The count values on those rows are such that
    this function may process the values as saturated, and the function would
    then fail if enough frames in a stack suffer from this apparent saturation.
    So this function disregards telemetry rows and does not do any fitting for
    master dark for those rows.

    Args:
    datasets : list, corgidrp.data.Dataset
        This is a list of instances of corgidrp.data.Dataset.  Each instance
        should be for a stack of dark frames (counts in DN), and each stack is
        for a unique EM gain and frame time combination.
        Each sub-stack should have the same number of frames.
        Each frame should accord with the EDU
        science frame (specified by corgidrp.util.metadata.yaml).
        Recommended:  >= 1300 frames for each sub-stack if calibrating
        darks for analog frames,
        thousands for photon counting depending on the maximum number of
        frames that will be used for photon counting.

    meta_path : str
        Full path of .yaml file from which to draw detector parameters.
        For format and names of keys, see corgidrp.util.metadata.yaml.
        If None, uses that file.

    Returns:
    F_map : array-like (full frame)
        A per-pixel map of fixed-pattern noise (in e-).  Any negative values
        from the fit are made positive in the end.

    C_map : array-like (full frame)
        A per-pixel map of EXCAM clock-induced charge (in e-). Any negative
        values from the fit are made positive in the end.

    D_map : array-like (full frame)
        A per-pixel map of dark current (in e-/s). Any negative values
        from the fit are made positive in the end.

    bias_offset : float
        The median for the residual FPN+CIC in the region where bias was
        calculated (i.e., prescan). In DN.

    F_image_map : array-like (image area)
        A per-pixel map of fixed-pattern noise in the image area (in e-).
        Any negative values from the fit are made positive in the end.

    C_image_map : array-like (image area)
        A per-pixel map of EXCAM clock-induced charge in the image area
        (in e-). Any negative values from the fit are made positive in the end.

    D_image_map : array-like (image area)
        A per-pixel map of dark current in the image area (in e-/s).
        Any negative values from the fit are made positive in the end.

    Fvar : float
        Variance of fixed-pattern noise map (in e-).

    Cvar : float
        Variance of clock-induced charge map (in e-).

    Dvar : float
        Variance of dark current map (in e-).

    read_noise : float
        Read noise estimate from the noise profile of a mean frame (in e-).
        It's read off from the sub-stack with the lowest product of EM gain and
        frame time so that the gained variance of C and D is comparable to or
        lower than read noise variance, thus making reading it off doable.
        If read_noise is returned as NaN, the read noise estimate is not
        trustworthy, possibly because not enough frames were used per substack
        for that or because the next lowest gain setting is much larger than
        the gain used in the sub-stack.

    R_map : array-like
        A per-pixel map of the adjusted coefficient of determination
        (adjusted R^2) value for the fit.

    F_image_mean : float
        F averaged over all pixels,
        before any negative ones are made positive.  Should be roughly the same
        as taking the mean of F_image_map.  This is just for comparison.

    C_image_mean : float
        C averaged over all pixels,
        before any negative ones are made positive.  Should be roughly the same
        as taking the mean of C_image_map.  This is just for comparison.

    D_image_mean : float
        D averaged over all pixels,
        before any negative ones are made positive.  Should be roughly the same
        as taking the mean of D_image_map.  This is just for comparison.
    """

    if len(datasets) <= 3:
        raise CalDarksLSQException('Number of sub-stacks in datasets must '
                'be more than 3 for proper curve fit.')
    # getting telemetry rows to ignore in fit
    if meta_path is None:
        meta_path = meta_path_default
    meta = Metadata(meta_path)
    metadata = meta.get_data()
    telem_rows_start = metadata['telem_rows_start']
    telem_rows_end = metadata['telem_rows_end']
    telem_rows = slice(telem_rows_start, telem_rows_end)

    g_arr = np.array([])
    t_arr = np.array([])
    k_arr = np.array([])
    mean_frames = []
    mean_num_good_fr = []
    unreliable_pix_map = np.zeros((meta.frame_rows,
                                   meta.frame_cols)).astype(int)
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
            g_arr = np.append(g_arr, datasets[i].frames[0].ext_hdr['EMGAIN_M'])
        except: # use commanded gain otherwise TODO change hdr name if necessary
            g_arr = np.append(g_arr, datasets[i].frames[0].ext_hdr['CMDGAIN'])
        exptime = datasets[i].frames[0].ext_hdr['EXPTIME']
        cmdgain = datasets[i].frames[0].ext_hdr['CMDGAIN']
        kgain = datasets[i].frames[0].ext_hdr['KGAIN']
        t_arr = np.append(t_arr, exptime)
        k_arr = np.append(k_arr, kgain)
        # check that all frames in sub-stack have same exposure time
        for fr in datasets[i].frames:
            if fr.ext_hdr['EXPTIME'] != exptime:
                raise CalDarksLSQException('The exposure time must be the '
                                           'same for all frames per '
                                           'sub-stack.')
            if fr.ext_hdr['CMDGAIN'] != cmdgain:
                raise CalDarksLSQException('The commanded gain must be the '
                                           'same for all frames per '
                                           'sub-stack.')
            if fr.ext_hdr['KGAIN'] != kgain:
                raise CalDarksLSQException('The k gain must be the '
                                           'same for all frames per '
                                           'sub-stack.')
            # ensure frame is in float so nan can be assigned, though it should
            # already be float
            frame = fr.data.astype(float)
            # For the fit, all types of bad pixels should be masked:
            b1 = fr.dq.astype(bool).astype(int)
            err = fr.err[0]
            frame[telem_rows] = np.nan
            i0 = meta.slice_section(frame, 'image')
            if np.isnan(i0).any():
                raise ValueError('telem_rows cannot be in image area.')
            # setting to 0 prevents failure of mean_combine
            # b0: didn't mask telem_rows b/c they weren't saturated but nan'ed
            frame[telem_rows] = 0
            frames.append(frame)
            bpmaps.append(b1)
            errs.append(err)
        mean_frame, _, map_im, _ = mean_combine(frames, bpmaps)
        mean_err, _, _, _ = mean_combine(errs, bpmaps, err=True)
        pixel_mask = (map_im < len(datasets[i].frames)/2).astype(int)
        mean_num = np.mean(map_im)
        mean_frame[telem_rows] = np.nan
        mean_frames.append(mean_frame)
        mean_num_good_fr.append(mean_num)
        unreliable_pix_map += pixel_mask
    mean_stack = np.stack(mean_frames)
    mean_err_stack = np.stack(mean_err)
    if (unreliable_pix_map >= len(datasets)-3).any():
        warnings.warn('At least one pixel was masked for more than half of '
                      'the frames in some sub-stacks, leaving 3 or fewer '
                      'sub-stacks that did not suffer this masking for these '
                      'pixels, which means the fit was unreliable for '
                      'these pixels.  These are the pixels in the output '
                      'unreliable_pixel_map that are >= len(datasets)-3.')

    if len(np.unique(g_arr)) < 2:
        raise CalDarksLSQException("Must have at least 2 unique EM gains "
                                   'represented by the sub-stacks in '
                                   'datasets.')
    if len(g_arr[g_arr<=1]) != 0:
        raise CalDarksLSQException('Each EM gain must be greater '
            'than 1.')
    if len(np.unique(t_arr)) < 2:
        raise CalDarksLSQException("Must have at 2 unique exposure times.")
    if len(t_arr[t_arr<=0]) != 0:
        raise CalDarksLSQException('Each exposure time must be greater '
            'than 0.')
    if len(k_arr[k_arr<=0]) != 0:
        raise CalDarksLSQException('Each element of k_arr must be greater '
            'than 0.')
    unique_sub_stacks = list(zip(g_arr, t_arr))
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
    min1 = np.argmin(g_arr*t_arr)
    # getting next "closest" stack for finding correlation b/w FPN maps:
    # same time, next gain up from least gain (close in time*gain but different
    # gain so that effective read noise values are more uncorrelated)
    tinds = np.where(t_arr == t_arr[min1])
    nextg = g_arr[g_arr > g_arr[min1]].min()
    ginds = np.where(g_arr == nextg)
    intersect = np.intersect1d(tinds, ginds)
    if intersect.size > 0:
        min2 = intersect[0]
    else: # just get next smallest g_arr*t_arr
        min2 = np.where(np.argsort(g_arr*t_arr) == 1)[0][0]
    msi = meta.imaging_slice(mean_stack[min1])
    msi2 = meta.imaging_slice(mean_stack[min2])
    avg_corr = np.corrcoef(msi.ravel(), msi2.ravel())[0, 1]

    # number of observations (i.e., # of averaged stacks provided for fit)
    M = len(g_arr)
    F_map = np.zeros_like(mean_stack[0])
    C_map = np.zeros_like(mean_stack[0])
    D_map = np.zeros_like(mean_stack[0])

    # input data error comes from .err arrays; could use this for error bars
    # in input data for weighted least squares, but we'll just simply get the
    # std error and add it in quadrature to least squares fit standard dev
    stacks_err = np.sqrt(np.sum(mean_err_stack**2, axis=0))/len(mean_err_stack)

    # matrix to be used for least squares and covariance matrix
    X = np.array([np.ones([len(g_arr)]), g_arr, g_arr*t_arr]).T
    mean_stack_Y = np.transpose(mean_stack, (2,0,1))
    params_Y = np.linalg.pinv(X)@mean_stack_Y
    #next line: checked with KKT method for including bounds
    #actually, do this after determining everything else so that
    # bias_offset, etc is accurate
    #params_Y[params_Y < 0] = 0
    params = np.transpose(params_Y, (1,2,0))
    F_map = params[0]
    C_map = params[1]
    D_map = params[2]
    # using chi squared for ordinary least squares (OLS) variance estiamate
    # This is OLS since the parameters to fit are linear in fit function
    # 3: number of fitted params
    residual_stack = mean_stack - np.transpose(X@params_Y, (1,2,0))
    sigma2_frame = np.sum(residual_stack**2, axis=0)/(M - 3)
    # average sigma2 for image area and use that for all three vars since
    # that's only place where all 3 (F, C, and D) should be present
    sigma2_image = meta.imaging_slice(sigma2_frame)
    sigma2 = np.mean(sigma2_image)
    cov_matrix = np.linalg.inv(X.T@X)

    # For full frame map of standard deviation:
    F_std_map = np.sqrt(sigma2_frame*cov_matrix[0,0])
    C_std_map = np.sqrt(sigma2_frame*cov_matrix[1,1])
    # Dark current should only be in image area, so error only for that area:
    D_std_map_im = np.sqrt(sigma2_image*cov_matrix[2,2])

    var_matrix = sigma2*cov_matrix
    # variances here would naturally account for masked pixels due to cosmics
    # since mean_combine does this
    Fvar = var_matrix[0,0]
    Cvar = var_matrix[1,1]
    Dvar = var_matrix[2,2]

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
    D_image_map = meta.imaging_slice(D_map)
    C_image_map = meta.imaging_slice(C_map)
    F_image_map = meta.imaging_slice(F_map)

    # res: should subtract D_map, too.  D should be zero there (in prescan),
    # but fitting errors may lead to non-zero values there.
    bias_offset = np.zeros([len(mean_stack)])
    for i in range(len(mean_stack)):
        res = mean_stack[i] - g_arr[i]*C_map - F_map - g_arr[i]*t_arr[i]*D_map
        res_prescan = meta.slice_section(res, 'prescan')
        # prescan may contain NaN'ed telemetry rows, so use nanmedian
        bias_offset[i] = np.nanmedian(res_prescan)/k_arr[i] # in DN
    bias_offset = np.mean(bias_offset)

    # don't average read noise for all frames since reading off read noise
    # from a frame with gained variance of C and D much higher than read
    # noise variance is not doable
    # read noise must be comparable to gained variance of D and C
    # so we read it off for lowest-gain-time frame
    l = np.argmin((g_arr*t_arr))
    # Num below accounts for pixels lost to cosmic rays
    Num = mean_num_good_fr[l]
    # take std of just image area; more variance if image and different regions
    # included; below assumes no variance inherent in FPN

    mean_stack_image = meta.imaging_slice(mean_stack[l])
    read_noise2 = (np.var(mean_stack_image)*Num -
        g_arr[l]**2*
        np.var(D_image_map*t_arr[l]+C_image_map) -
        ((Num-1)*avg_corr+1)*np.var(F_image_map))
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
    D_map = np.zeros((meta.frame_rows, meta.frame_cols))
    D_std_map = np.zeros((meta.frame_rows, meta.frame_cols))
    # and reset the telemetry rows to NaN
    D_map[telem_rows] = np.nan

    im_rows, im_cols, r0c0 = meta._imaging_area_geom()
    D_map[r0c0[0]:r0c0[0]+im_rows,
                    r0c0[1]:r0c0[1]+im_cols] = D_image_map
    D_std_map[r0c0[0]:r0c0[0]+im_rows,
                    r0c0[1]:r0c0[1]+im_cols] = D_std_map_im

    # now catch any elements that were negative for C and D:
    D_map[D_map < 0] = 0
    C_map[C_map < 0] = 0
    #mean taken before zeroing out the negatives for C and D
    F_image_mean = np.mean(F_image_map)
    C_image_mean = np.mean(C_image_map)
    D_image_mean = np.mean(D_image_map)
    C_image_map[C_image_map < 0] = 0
    D_image_map[D_image_map < 0] = 0

    return (F_map, C_map, D_map, bias_offset, F_image_map, C_image_map,
            D_image_map, Fvar, Cvar, Dvar, read_noise, R_map, F_image_mean,
            C_image_mean, D_image_mean, unreliable_pix_map, F_std_map,
            C_std_map, D_std_map, stacks_err)


if __name__ == '__main__':

    here = os.path.abspath(os.path.dirname(__file__))
    meta_path = Path(here,'..', 'util', 'metadata_test.yaml')
    meta = Metadata(meta_path)

    dark_current = 8.33e-4 #e-/pix/s
    cic=0.02  # e-/pix/frame
    read_noise=100 # e-/pix/frame
    bias=2000 # e-
    eperdn = 7 # e-/DN conversion; used in this example for all stacks
    g_picks = (np.linspace(2, 5000, 7))
    t_picks = (np.linspace(2, 100, 7))
    grid = np.meshgrid(g_picks, t_picks)
    g_arr = grid[0].ravel()
    t_arr = grid[1].ravel()
    #added in after emccd_detect makes the frames (see below)
    # The mean FPN that will be found is eperdn*(FPN//eperdn)
    # due to how I simulate it and then convert the frame to uint16
    FPN = 21 # e
    # the bigger N is, the better the adjusted R^2 per pixel becomes
    N = 30 #Use N=600 for results with better fits (higher values for adjusted
    # R^2 per pixel)
    # image area, including "shielded" rows and cols:
    imrows, imcols, imr0c0 = meta._imaging_area_geom()
    prerows, precols, prer0c0 = meta._unpack_geom('prescan')

    stack_list = []
    for i in range(len(g_arr)):
        frame_list = []
        for l in range(N): #number of frames to produce
            # Simulate full dark frame (image area + the rest)
            frame_dn_dark = np.zeros((meta.frame_rows, meta.frame_cols))
            im = np.random.poisson(cic*g_arr[i]+
                                t_arr[i]*g_arr[i]*dark_current,
                                size=(meta.frame_rows, meta.frame_cols))
            frame_dn_dark = im
            # prescan has no dark current
            pre = np.random.poisson(cic*g_arr[i],
                                    size=(prerows, precols))
            frame_dn_dark[prer0c0[0]:prer0c0[0]+prerows,
                            prer0c0[1]:prer0c0[1]+precols] = pre
            rn = np.random.normal(0, read_noise,
                                    size=(meta.frame_rows, meta.frame_cols))
            frame_dn_dark += rn + bias
            frame_dn_dark /= eperdn
            # simulate a constant FPN in image area (not in prescan
            # so that it isn't removed when bias is removed)
            frame_dn_dark[imr0c0[0]:imr0c0[0]+imrows,imr0c0[1]:
            imr0c0[1]+imcols] += FPN/eperdn # in DN
            frame_list.append(frame_dn_dark)
            # simulate telemetry rows, with the last 5 column entries with high counts
            frame_dn_dark[-1,-5:] = 100000 #DN
            # to make more like actual frames from detector
        frame_stack = np.stack(frame_list)
        test_data_path = Path(here, 'tests', 'simdata',
                                'calibrate_darks_lsq')
        save = os.path.join(test_data_path, 'g_'+str(int(g_arr[i]))+'_t_'+
            str(int(t_arr[i]))+'_N_'+str(N)+'stack.npy')
        np.save(str(save), frame_stack)
        stack_list.append(frame_stack)