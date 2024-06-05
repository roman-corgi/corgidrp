#least squares method
import os
from pathlib import Path
import numpy as np
import warnings

from corgidrp.detector import Metadata as MetadataWrapper
from cal.util.gsw_process import Process, mean_combine
import corgidrp.util.check as check

TELEM_ROWS = 4 #last 4 rows of frame

class CalDarksLSQException(Exception):
    """Exception class for calibrate_darks_lsq."""

def calibrate_darks_lsq(stack_arr, g_arr, t_arr, k_arr, fwc_em_e, fwc_pp_e,
            meta_path, nonlin_path, Nem = 604, telem_rows=None):
    """Given an array of frame stacks of the same number of dark frames (in DN
    units), where the stacks are for various EM gain values and exposure times,
    this function subtracts the bias from each frame in each stack, masks for
    cosmic rays, corrects for nonlinearity, converts DN to e-, and averages
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

    Parameters
    ----------
    stack_arr : array-like
        The input stack of stacks of dark frames (counts in DN).  stack_arr
        contains a stack of sub-stacks, and in each sub-stack is some number of
        frames for a unique EM gain and frame time combination.
        Each sub-stack should have the same number of frames.
        Each frame should accord with the EDU
        science frame (specified by metadata.yaml).
        Recommended:  >= 1300 frames for each sub-stack if calibrating
        darks for analog frames,
        thousands for photon counting depending on the maximum number of
        frames that will be used for photon counting.

    g_arr : array-like
        The input array of EM gain values corresponding to the sub-stacks in
        stack_arr in the order found there.  > 1 (since extra noise is
        present when EM gain > 1).  Need to have at least 2 unique EM gain
        values.  (More than 2 recommended.)

    t_arr : array-like
        The input array of exposure times (in s) corresponding to the
        sub-stacks in stack_arr in the order found there.  > 0.  Need to have
        at least 2 unique frame time values.  (More than 2 recommended.)

    k_arr : array-like
        The input array of conversion factors in e-/DN corresponding to the
        sub-stacks in stack_arr in the order found there. > 0.

    fwc_em_e : int
        Full well capacity of detector EM gain register (electrons).

    fwc_pp_e : int
        Full well capacity per pixel of detector image area pixels (electrons).

    meta_path : str
        Full path of metadata.yaml.

    nonlin_path : str
        Path to residual nonlinearity relative gain file.

    Nem : int
        Number of gain register cells.  Defaults to in-flight value, 604.

    telem_rows : slice
        The specified slice of rows in all frames where the telemetry rows lie.
        If None, the last 4 rows are used.  Defaults to None.

    Returns
    -------
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
    #input checks
    if type(stack_arr) != np.ndarray:
        raise TypeError('stack_arr must be '
        'castable to a stack.')
    if np.ndim(stack_arr) != 4:
        raise CalDarksLSQException('stack_arr must be 4-D (i.e., a stack of '
                '3-D sub-stacks')
    if len(stack_arr) <= 3:
        raise CalDarksLSQException('Number of sub-stacks in stack_arr must '
                'be more than 3 for proper curve fit.')
    for i in range(len(stack_arr)):
        check.threeD_array(stack_arr[i], 'stack_arr['+str(i)+']', TypeError)
        if len(stack_arr[i]) < 1300:
            warnings.warn('A sub-stack was found with less than 1300 frames, '
            'which is the recommended number per sub-stack for an analog '
            'master dark')
        if i > 0:
            if np.shape(stack_arr[i-1]) != np.shape(stack_arr[i]):
                raise CalDarksLSQException('All sub-stacks must have the '
                            'same number of frames and frame shape.')
    check.real_array(g_arr, 'g_arr', TypeError)
    check.oneD_array(g_arr, 'g_arr', TypeError)
    if len(np.unique(g_arr)) < 2:
        raise CalDarksLSQException("Must have at least 2 unique elements in "
        "g_arr.")
    if len(g_arr) != len(stack_arr):
        raise CalDarksLSQException('The length of g_arr must be the same as '
            'the number of sub-stacks in stack_arr')
    if len(g_arr[g_arr<=1]) != 0:
        raise CalDarksLSQException('Each element of g_arr must be greater '
            'than 1.')
    check.real_array(t_arr, 't_arr', TypeError)
    check.oneD_array(t_arr, 't_arr', TypeError)
    if len(np.unique(t_arr)) < 2:
        raise CalDarksLSQException("Must have at least 2 unique elements in "
        "t_arr.")
    if len(t_arr) != len(stack_arr):
        raise CalDarksLSQException('The length of t_arr must be the same as '
            'the number of sub-stacks in stack_arr')
    if len(t_arr[t_arr<=0]) != 0:
        raise CalDarksLSQException('Each element of t_arr must be greater '
            'than 0.')
    check.real_array(k_arr, 'k_arr', TypeError)
    check.oneD_array(k_arr, 'k_arr', TypeError)
    if len(k_arr) != len(stack_arr):
        raise CalDarksLSQException('The length of k_arr must be the same as '
            'the number of sub-stacks in stack_arr')
    if len(k_arr[k_arr<=0]) != 0:
        raise CalDarksLSQException('Each element of k_arr must be greater '
            'than 0.')
    unique_sub_stacks = list(zip(g_arr, t_arr))
    for el in unique_sub_stacks:
        if unique_sub_stacks.count(el) > 1:
            raise CalDarksLSQException('The EM gain and frame time '
            'combinations for the sub-stacks must be unique.')
    check.positive_scalar_integer(fwc_em_e, 'fwc_em_e', TypeError)
    check.positive_scalar_integer(fwc_pp_e, 'fwc_pp_e', TypeError)
    check.positive_scalar_integer(Nem, 'Nem', TypeError)
    # no checks on meta_path and nonlin_path since they will fail soon into
    # the code if they aren't right
    if telem_rows is not None:
        if not isinstance(telem_rows, slice):
            raise TypeError('telem_rows must be a slice or None.')
    else:
        end = stack_arr.shape[2]
        telem_rows = slice(-TELEM_ROWS, end)

    # no bad pixel maps at this point in the pipeline
    bad_pix = np.zeros_like(stack_arr[0][0])
    # bias offset set to 0 so that it has no effect; bias offset is determined
    # by the calibration of darks (this module)
    bias_offset = 0
    proc_dark = {}
    mean_frames = []
    mean_num_good_fr = []
    for i in range(len(stack_arr)):
        proc_dark[i] = Process(bad_pix=bad_pix, eperdn=k_arr[i],
                               fwc_em_e=fwc_em_e, fwc_pp_e=fwc_pp_e,
                               bias_offset=0, em_gain=g_arr[i],
                               exptime=t_arr[i], nonlin_path=nonlin_path,
                               meta_path=meta_path)
        frames = []
        bpmaps = []
        for fr in stack_arr[i]:
            # ensure frame is in float so nan can be assigned; output of
            # L1_to_L2a converts to float anyways
            fr = fr.astype(float)
            fr[telem_rows] = np.nan
            i0, _, _, _, f0, b0, _ = proc_dark[i].L1_to_L2a(fr)
            if np.isnan(i0).any():
                raise ValueError('telem_rows cannot be in image area.')
            # could just skip L2a_to_L2b(), but I like having a hook for
            # combining b0 with a proc_dark.bad_pix that isn't all zeros
            f1, b1, _ = proc_dark[i].L2a_to_L2b(f0, b0)
            # to undo the division by gain in L2a_to_L2b()
            f1 *= g_arr[i]
            # setting to 0 prevents failure of mean_combine
            # b0: didn't mask telem_rows b/c they weren't saturated but nan'ed
            f1[telem_rows] = 0
            frames.append(f1)
            bpmaps.append(b1)
        mean_frame, _, mean_num, rn_bool = mean_combine(frames, bpmaps)
        if not rn_bool: # if False, due to cosmics
            raise CalDarksLSQException('fewer than half the frames '
            'available for at least one pixel in the averaging due to masking'
            ', so cannot effectively determine noise maps for all pixels')
        # now safe to mark telemetry rows as NaN again
        mean_frame[telem_rows] = np.nan
        mean_frames.append(mean_frame)
        mean_num_good_fr.append(mean_num)
    mean_stack = np.stack(mean_frames)

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
    min2 = np.intersect1d(tinds, ginds)[0]
    msi = proc_dark[i].meta.imaging_slice(mean_stack[min1])
    msi2 = proc_dark[i].meta.imaging_slice(mean_stack[min2])
    avg_corr = np.corrcoef(msi.ravel(), msi2.ravel())[0, 1]

    # number of observations (i.e., # of averaged stacks provided for fit)
    M = len(g_arr)
    F_map = np.zeros_like(mean_stack[0])
    C_map = np.zeros_like(mean_stack[0])
    D_map = np.zeros_like(mean_stack[0])

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
    # doesn't matter which k used for just slicing
    sigma2_image = proc_dark[0].meta.imaging_slice(sigma2_frame)
    sigma2 = np.mean(sigma2_image)

    cov_matrix = np.linalg.inv(X.T@X)
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
    # is expected.
    R_map = 1 - (1 - Rsq)*(M - 1)/(M - 3)

    # doesn't matter which k used for just slicing
    D_image_map = proc_dark[0].meta.imaging_slice(D_map)
    C_image_map = proc_dark[0].meta.imaging_slice(C_map)
    F_image_map = proc_dark[0].meta.imaging_slice(F_map)

    # res: should subtract D_map, too.  D should be zero there (in prescan),
    # but fitting errors may lead to non-zero values there.
    bias_offset = np.zeros([len(mean_stack)])
    for i in range(len(mean_stack)):
        res = mean_stack[i] - g_arr[i]*C_map - F_map - g_arr[i]*t_arr[i]*D_map
        res_prescan = proc_dark[i].meta.slice_section(res, 'prescan')
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

    mean_stack_image = proc_dark[l].meta.imaging_slice(mean_stack[l])
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
    D_map = np.zeros((proc_dark[0].meta.frame_rows,
                                proc_dark[0].meta.frame_cols))
    # and reset the telemetry rows to NaN
    D_map[telem_rows] = np.nan

    im_rows, im_cols, r0c0 = proc_dark[0].meta._imaging_area_geom()
    D_map[r0c0[0]:r0c0[0]+im_rows,
                    r0c0[1]:r0c0[1]+im_cols] = D_image_map
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
            C_image_mean, D_image_mean)

def ENF(g, Nem):
    """Returns the extra-noise function (ENF).
    Parameters
    ----------
    g : float
        EM gain.  >= 1.
    Nem : int
        Number of gain register cells.
    Returns
    -------
    ENF : float
        extra-noise function
    """
    return np.sqrt(2*(g-1)*g**(-(Nem+1)/Nem) + 1/g)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from emccd_detect.emccd_detect import EMCCDDetect

    def imagesc(data, title=None, vmin=None, vmax=None, cmap='viridis',
            aspect='equal', colorbar=True):
        """Plot a scaled colormap."""
        fig, ax = plt.subplots()
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)

        if title:
            ax.set_title(title)
        if colorbar:
            fig.colorbar(im, ax=ax)

        return fig, ax

    here = os.path.abspath(os.path.dirname(__file__))
    #meta_path = Path(here,'..', 'util', 'metadata.yaml')
    meta_path = Path(here,'..', 'util', 'metadata_test.yaml')
    meta = MetadataWrapper(meta_path)
    image_rows, image_cols, r0c0 = meta._unpack_geom('image')

    use_proc_cgi_frame_nonlin = False
    if use_proc_cgi_frame_nonlin:
        nonlin_path = Path(here, '..', 'util', 'nonlin_sample.csv')
    else:
        nonlin_path = Path(here, '..', 'util', 'testdata',
                'ut_nonlin_array_ones.txt') # does no corrections

    fluxmap = np.zeros((image_rows,image_cols)) #for raw dark, photons/s

    fwc_em_e = 90000 #e-
    fwc_pp_e = 50000 #e-
    dark_current = 8.33e-4 #e-/pix/s
    cic=0.02  # e-/pix/frame
    read_noise=100 # e-/pix/frame
    bias=2000 # e-
    qe=0.9  # quantum efficiency, e-/photon
    cr_rate=0.  # hits/cm^2/s
    pixel_pitch=13e-6  # m
    eperdn = 7 # e-/DN conversion; used in this example for all stacks
    nbits=64 # number of ADU bits
    numel_gain_register=604 #number of gain register elements

    #g_picks = np.linspace(2, 300, 7)# np.linspace(300, 5000, 5)
    #g_picks = (np.linspace(2, 5000, 10))
    #g_picks = np.linspace(2, 5000, 7)
    g_picks = (np.linspace(2, 5000, 7))
    #g_picks = np.array([2, 300.0, 1475.0, 2650.0, 3825.0, 5000.0])
    #t_picks = (np.linspace(2, 100, 10))
    #t_picks = np.linspace(2, 100, 7)
    t_picks = (np.linspace(2, 100, 7))
    grid = np.meshgrid(g_picks, t_picks)
    g_arr = grid[0].ravel()
    t_arr = grid[1].ravel()
    k_arr = eperdn*np.ones_like(g_arr) # all the same
    #added in after emccd_detect makes the frames (see below)
    # The mean FPN that will be found is eperdn*(FPN//eperdn)
    # due to how I simulate it and then convert the frame to uint16
    FPN = 21 # e
    # the bigger N is, the better the adjusted R^2 per pixel becomes
    N = 30 #Use N=600 for results with better fits (higher values for adjusted
    # R^2 per pixel)
    # image area, including "shielded" rows and cols:
    imrows, imcols, imr0c0 = meta._imaging_area_geom()

    making_data = False
    if making_data:
        emccd = []
        for i in range(len(g_arr)):
            emccd.append(EMCCDDetect(
            em_gain=g_arr[i],
            full_well_image=fwc_pp_e,
            full_well_serial=fwc_em_e,
            dark_current=dark_current,
            cic=cic,
            read_noise=read_noise,
            bias=bias,
            qe=qe,
            cr_rate=cr_rate,
            pixel_pitch=pixel_pitch,
            eperdn=k_arr[i],
            nbits=nbits,
            numel_gain_register=numel_gain_register,
            meta_path=meta_path)
            )

        stack_list = []
        for i in range(len(g_arr)):
            frame_list = []
            for l in range(N): #number of frames to produce
                # Simulate full dark frame (image area + the rest)
                frame_dn_dark = emccd[i].sim_full_frame(fluxmap, t_arr[i])
                frame_list.append(frame_dn_dark)
            frame_stack = np.stack(frame_list)
            #np.save('C:\\Users\Kevin\\Desktop\\testdata'+
            test_data_path = Path(here, 'testdata_small')
            save = os.path.join(test_data_path, 'g_'+str(int(g_arr[i]))+'_t_'+
                str(int(t_arr[i]))+'_N_'+str(N)+'stack.npy')
            np.save(str(save), frame_stack)
            stack_list.append(frame_stack)

            # simulate a FPN in image area (not in prescan
            # so that it isn't removed when bias is removed)
            stack_list[i] = stack_list[i].astype('float64')
            im_area = stack_list[i][:,imr0c0[0]:imr0c0[0]+imrows,imr0c0[1]:
               imr0c0[1]+imcols]

            # For FPN, could add in a pattern taken from actual data...
            # fpn = fits.getdata(r'/Users/kevinludwick/Documents/Guillermo_TVAC_all_darks/FPN_image.fits')
            # im_area[:] += fpn[300:300+im_area.shape[1],300:300+im_area.shape[2]]/k_arr[i]

            # ...or simulate a cross-hatch-like pattern...
            # groups = (np.round(np.linspace(0, im_area.shape[1],
            #                                40))).astype(int)
            # for j in groups:
            #     expo = np.where(groups==j)[0][0]
            #     im_area[:,:,j:j+1] += FPN/k_arr[i] + \
            #         (-1)**expo*FPN/(k_arr[i]) # in DN
            #     im_area[:,j:j+1,:] += FPN/k_arr[i] + \
            #         (-1)**expo*FPN/(k_arr[i]) # in DN

            # ...or add in a constant offset.
            stack_list[i][:,imr0c0[0]:imr0c0[0]+imrows,imr0c0[1]:
               imr0c0[1]+imcols] += FPN/k_arr[i] # in DN

        stack_arr = np.stack(stack_list)

    if not making_data:
        stack_list = []
        #test_data_path = Path(here, 'testdata')
        test_data_path = Path(here, 'testdata_small')
        for i in range(len(g_arr)):
            load = os.path.join(test_data_path, 'g_'+str(int(g_arr[i]))+'_t_'+
                str(int(t_arr[i]))+'_N_'+str(N)+'stack.npy')
            stack_list.append(np.load(load))

            # simulate a constant FPN in image area (not in prescan
            # so that it isn't removed when bias is removed)
            stack_list[i] = stack_list[i].astype('float64')
            stack_list[i][:,imr0c0[0]:imr0c0[0]+imrows,imr0c0[1]:
               imr0c0[1]+imcols] += FPN/k_arr[i] # in DN

        stack_arr = np.stack(stack_list)

    # simulate telemetry rows, with the last 5 column entries with high counts
    stack_arr[:,:,-4:,-5] = 100000 #DN
    # to make more like actual frames from detector
    stack_arr = stack_arr.astype('uint16')

    (F_map, C_map, D_map, bias_offset, F_image_map, C_image_map, D_image_map,
        Fvar, Cvar, Dvar, read_noise, R_map, F_image_mean, C_image_mean,
        D_image_mean) = \
        calibrate_darks_lsq(stack_arr, g_arr, t_arr, k_arr, fwc_em_e, fwc_pp_e,
            meta_path, nonlin_path, Nem = 604)

    print('read noise (in e-): ', read_noise)
    print('mean image dark current (in e-): ', np.mean(D_image_map))
    print('mean image CIC (in e-): ', np.mean(C_image_map))
    print('mean image FPN (in e-): ', np.mean(F_image_map))
    print('bias offset (in DN): ', bias_offset)
    #print("Fvar: ", Fvar)
    print("average adjusted R^2 value: ", np.nanmean(R_map))

    # master dark noise
    def MDnoise(g, t):
        return np.sqrt(Fvar/g**2 + t**2*Dvar + Cvar)

    g_vals = np.linspace(2, 5000, 100)
    t_vals = np.linspace(2, 100, 100)

    imagesc(D_map, 'Dark Current')
    imagesc(C_map, 'CIC')
    imagesc(F_map, "FPN")
    plt.figure()
    plt.plot(g_vals, MDnoise(g_vals, 10))
    plt.show()
