"""Photon count a stack of analog images and return a mean expected electron 
count array with photometric corrections.

"""
import warnings
import numpy as np
import corgidrp.data as data

def varL23(g, L, T, N):
    '''Expected variance after photometric correction.  
    See https://doi.org/10.1117/1.JATIS.9.4.048006 for details.
    
    Parameters
    ----------
    g (scalar): EM gain
    L (2-D array): mean expected number of electrons
    T (scalar): threshold 
    N (2-D array): number of frames

    Returns 
    -------
    variance from photon counting and the photometric correction

    '''
    Const = 6/(6 + L*(6 + L*(3 + L)))
    eThresh = (Const*(np.e**(-T/g)*L*(2*g**2*(6 + L*(3 + L)) +
            2*g*L*(3 + L)*T + L**2*T**2))/(12*g**2))
    std_dev = np.sqrt(N * eThresh * (1-eThresh))

    return (std_dev)**2*(((np.e**((T/g)))/N) +
    2*((np.e**((2*T)/g)*(g - T))/(2*g*N**2))*(N*eThresh) +
    3*(((np.e**(((3*T)/g)))*(4*g**2 - 8*g*T + 5*T**2))/(
    12*g**2*N**3))*(N*eThresh)**2)**2

class PhotonCountException(Exception):
    """Exception class for photon_count module."""


def photon_count(e_image, thresh):
    """Convert an analog image into a photon-counted image.

    Parameters
    ----------
    e_image : array_like, float
        Analog image (e-).
    thresh : float
        Photon counting threshold (e-). Values > thresh will be assigned a 1,
        values <= thresh will be assigned a 0.

    Returns
    -------
    pc_image : array_like, float
        Output digital image in units of photons.

    B Nemati and S Miller - UAH - 06-Aug-2018

    """
    # Check if input is an array/castable to one
    e_image = np.array(e_image).astype(float)
    if len(e_image.shape) == 0:
        raise PhotonCountException('e_image must have length > 0')

    pc_image = np.zeros(e_image.shape, dtype=int)
    pc_image[e_image > thresh] = 1

    return pc_image


def get_pc_mean(input_dataset, detector_params, niter=2):
    """Take a stack of images, illuminated and dark frames of the same exposure 
    time, k gain, read noise, and EM gain, and return the mean expected value per 
    pixel. The frames are taken in photon-counting mode, which means high EM 
    gain and very low exposure time.  The number of darks can be different from 
    the number of illuminated frames.

    The frames in each stack should be SCI image (1024x1024) frames that have 
    had some of the L2b steps applied:

    - have had their bias subtracted
    - have had masks made for cosmic rays
    - have been corrected for nonlinearity (These first 3 steps make the frames L2a.)
    - have been frame-selected (to weed out bad frames)
    - have been converted from DN to e-
    - have been desmeared if desmearing is appropriate.  Under normal
    circumstances, these frames should not be desmeared.  The only time desmearing
    would be useful is in the unexpected case that, for example,
    dark current is so high that it stands far above other noise that is
    not smeared upon readout, such as clock-induced charge, 
    fixed-pattern noise, and read noise.  For such low-exposure frames, though,
    this really should not be a concern.

    This algorithm will photon count each frame in each of the 2 sets (illuminated and dark) individually,
    then co-add the photon counted frames. The co-added frame is then averaged
    and corrected for thresholding and coincidence loss, returning the mean
    expected array in units of photoelectrons.  The threshold is 
    determined by "T_factor" stored in DetectorParams. The dark mean expected array
    is then subtracted from the illuminated mean expected array.

    Parameters
    ----------
    input_dataset (corgidrp.data.Dataset):
        This is an instance of corgidrp.data.Dataset containing the  
        photon-counted illuminated frames as well as the 
        photon-counted dark frames, and all the frames must have the same 
        exposure time, EM gain, k gain, and read noise.
    detector_params: (corgidrp.data.DetectorParams)
        A calibration file storing detector calibration values.
    niter : int, optional
        Number of Newton's method iterations (used for photometric 
        correction). Defaults to 2.

    Returns
    -------
    mean_expected : array_like
        Mean expected photoelectron array (dark-subtracted)

    References
    ----------
    [1] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11443/114435F/Photon-counting-and-precision-photometry-for-the-Roman-Space-Telescope/10.1117/12.2575983.full
    [2] https://doi.org/10.1117/1.JATIS.9.4.048006

    B Nemati, S Miller - UAH - 13-Dec-2020
    Kevin Ludwick - UAH - 2023

    """
    test_dataset, _ = input_dataset.copy().split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN', 'KGAIN', 'RN'])
    if len(test_dataset) > 1:
        raise PhotonCountException('All frames must have the same exposure time, '
                                   'commanded EM gain, and k gain.')
    datasets, vals = test_dataset[0].split_dataset(prihdr_keywords=['VISTYPE'])
    if len(vals) != 2:
        raise PhotonCountException('There should only be 2 datasets, and one should have \'VISTYPE\' = \'DARK\'.')
    dark_dataset = datasets[vals.index('DARK')]
    ill_dataset = datasets[1-vals.index('DARK')]
    
    pc_means = []
    errs = []
    dqs = []
    for dataset in [ill_dataset, dark_dataset]:
        # getting number of read noise standard deviations at which to set the
        # photon-counting threshold
        T_factor = detector_params.params['T_factor']
        # now get threshold to use for photon-counting
        thresh = T_factor*dataset.frames[0].ext_hdr['RN']
        if thresh < 0:
            raise PhotonCountException('thresh must be nonnegative')
        if not isinstance(niter, (int, np.integer)) or niter < 1:
            raise PhotonCountException('niter must be an integer greater than '
                                        '0')
        try: # if EM gain measured directly from frame TODO change hdr name if necessary
            em_gain = dataset.frames[0].ext_hdr['EMGAIN_M']
        except:
            try: # use applied EM gain if available
                em_gain = dataset.frames[0].ext_hdr['EMGAIN_A']
            except: # use commanded gain otherwise
                em_gain = dataset.frames[0].ext_hdr['CMDGAIN']
        if thresh >= em_gain:
            warnings.warn('thresh should be less than em_gain for effective '
            'photon counting')

        # Photon count stack of frames
        frames_pc = photon_count(dataset.all_data, thresh)
        # frames_pc = np.array([photon_count(frame, thresh) for frame in dataset.all_data]) XXX
        bool_map = dataset.all_dq.astype(bool).astype(float)
        bool_map[bool_map > 0] = np.nan
        bool_map[bool_map == 0] = 1
        nframes = np.nansum(bool_map, axis=0)
        frames_pc_masked = frames_pc * bool_map
        # Co-add frames
        frame_pc_coadded = np.nansum(frames_pc_masked, axis=0)
        
        # Correct for thresholding and coincidence loss; any pixel masked all the 
        # way through the stack will give NaN
        mean_expected = corr_photon_count(frame_pc_coadded, nframes, thresh,
                                            em_gain, niter)
      
        ##### error calculation
        # combine all err frames to get standard error, which is due to inherent error in
        # in the data before any photon counting is done
        err_sum = np.sum(dataset.all_err**2, axis=0)
        bool_map_sum = np.nansum(bool_map, axis=0)
        comb_err = np.divide(np.sqrt(err_sum), np.sqrt(bool_map_sum), out=np.zeros_like(err_sum), 
                            where=bool_map_sum != 0)
        # expected error from photon counting
        pc_variance = varL23(em_gain,mean_expected,thresh,nframes)
        # now combine this error with the statistical error coming from the process of photon counting 
        total_err = np.sqrt(comb_err**2 + pc_variance)
        total_dq = (bool_map_sum == 0).astype(int) # 1 where flagged, 0 otherwise
        pc_means.append(mean_expected)
        errs.append(total_err)
        dqs.append(total_dq)
    
    # now subtract the dark PC mean
    combined_pc_mean = pc_means[0] - pc_means[1]
    combined_err = np.sqrt(errs[0]**2 + errs[1]**2)
    combined_dq = np.bitwise_or(dqs[0], dqs[1])
    pri_hdr = ill_dataset[0].pri_hdr.copy()
    ext_hdr = ill_dataset[0].ext_hdr.copy()
    err_hdr = ill_dataset[0].err_hdr.copy()
    dq_hdr = ill_dataset[0].dq_hdr.copy()
    hdulist = ill_dataset[0].hdu_list.copy()
    new_image = data.Image(combined_pc_mean, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err=combined_err, dq=combined_dq, err_hdr=err_hdr, 
                            dq_hdr=dq_hdr, input_hdulist=hdulist) 

    new_image.filename = ill_dataset[0].filename.split('.')[0]+'_pc.fits'
    new_image._record_parent_filenames(input_dataset)   
    pc_dataset = data.Dataset([new_image])
    pc_dataset.update_after_processing_step("Photon-counted {0} frames using T_factor = {1} and niter={2}".format(len(input_dataset), T_factor, niter))

    return pc_dataset

def corr_photon_count(nobs, nfr, t, g, niter=2):
    """Correct photon counted images.

    Parameters
    ----------
    nobs : array_like
        Number of observations (Co-added photon counted frame).
    nfr : int
        Number of coadded frames.
    t : float
        Photon counting threshold.
    g : float
        EM gain.
    niter : int, optional
        Number of Newton's method iterations. Defaults to 2.

    Returns
    -------
    lam : array_like
        Mean expeted detected electrons per pixel (lambda).

    """
    # Get an approximate value of lambda for the first guess of the Newton fit
    lam0 = calc_lam_approx(nobs, nfr, t, g)

    # Use Newton's method to converge at a value for lambda
    lam = lam_newton_fit(nobs, nfr, t, g, lam0, niter)

    return lam


def calc_lam_approx(nobs, nfr, t, g):
    """Approximate lambda calculation.

    This will calculate the first order approximation of lambda, and for values
    that are out of bounds (e.g. from statistical fluctuations) it will revert
    to the zeroth order.

    Parameters
    ----------
    nobs : array_like
        Number of observations (Co-added photon counted frame).
    nfr : int
        Number of coadded frames.
    t : float
        Photon counting threshold.
    g : float
        EM gain used when taking images.

    Returns
    -------
    array_like
        Mean expeted (lambda).

    """
    # First step of equation (before taking log)
    init = 1 - (nobs/nfr) * np.exp(t/g)
    # Mask out all values less than or equal to 0
    lam_m = np.zeros_like(init).astype(bool)
    lam_m[init > 0] = True

    # Use the first order approximation on all values greater than zero
    lam1 = np.zeros_like(init)
    lam1[lam_m] = -np.log(init[lam_m])

    # For any values less than zero, revert to the zeroth order approximation
    lam0 = nobs / nfr
    lam1[~lam_m] = lam0[~lam_m]

    return lam1


def lam_newton_fit(nobs, nfr, t, g, lam0, niter):
    """Newton fit for finding lambda.

    Parameters
    ----------
    nobs : array_like
        Number of observations (Co-added photon counted frame).
    nfr : int
        Number of coadded frames.
    t : float
        Photon counting threshold.
    g : float
        EM gain used when taking images.
    lam0 : array_like
        Initial guess for lambda.
    niter : int
        Number of Newton's fit iterations to take.

    Returns
    -------
    lam : array_like
        Mean expeted (lambda).

    """
    # Mask out zero values to avoid divide by zero
    lam_est_m = np.ma.masked_where(lam0 == 0, lam0)
    nobs_m = np.ma.masked_where(nobs == 0, nobs)

    # Iterate Newton's method
    for i in range(niter):
        func = _calc_func(nobs_m, nfr, t, g, lam_est_m)
        dfunc = _calc_dfunc(nfr, t, g, lam_est_m)
        lam_est_m -= func / dfunc

    if lam_est_m.min() < 0:
        raise PhotonCountException('negative number of photon counts; '
        'try decreasing the frametime')

    # Fill zero values back in
    lam = lam_est_m.filled(0)

    return lam


def _calc_func(nobs, nfr, t, g, lam):
    """Objective function for lambda."""
    e_thresh = (
        np.exp(-t/g)
        * (
            t**2 * lam**2
            + 2*g * t * lam * (3 + lam)
            + 2*g**2 * (6 + 3*lam + lam**2)
        )
        / (2*g**2 * (6 + 3*lam + lam**2))
    )

    e_coinloss = (1 - np.exp(-lam)) / lam

    #if (lam * nfr * e_thresh * e_coinloss).any() > nobs.any():
    #    warnings.warn('Input photon flux is too high; decrease frametime')
    # This warning isn't necessary; could have a negative func but still
    # close enough to 0 for Newton's method
    func = lam * nfr * e_thresh * e_coinloss - nobs

    return func


def _calc_dfunc(nfr, t, g, lam):
    """Derivative wrt lambda of objective function."""
    dfunc = (
        (np.exp(-t/g - lam) * nfr)
        / (2 * g**2 * (6 + 3*lam + lam**2)**2)
        * (
            2*g**2 * (6 + 3*lam + lam**2)**2
            + t**2 * lam * (
                -12 + 3*lam + 3*lam**2 + lam**3 + 3*np.exp(lam)*(4 + lam)
                )
            + 2*g*t * (
                -18 + 6*lam + 15*lam**2 + 6*lam**3 + lam**4
                + 6*np.exp(lam)*(3 + 2*lam)
                )
        )
    )

    return dfunc