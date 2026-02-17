"""Photon count a stack of analog images and return a mean expected electron 
count array with photometric corrections.

"""
import warnings
import numpy as np
from astropy.io import fits
import corgidrp.data as data
from corgidrp import check

def varL23(g, L, T, N):
    '''Expected variance after photometric correction.  
    See https://doi.org/10.1117/1.JATIS.9.4.048006 for details.
    
    Args:
        g (scalar): EM gain
        L (2-D array): mean expected number of electrons
        T (scalar): threshold 
        N (2-D array): number of frames

    Returns:
        (float): variance from photon counting and the photometric correction

    '''
    Const = 6/(6 + L*(6 + L*(3 + L)))
    eThresh = (Const*(np.e**(-T/g)*L*(2*g**2*(6 + L*(3 + L)) +
            2*g*L*(3 + L)*T + L**2*T**2))/(12*g**2))
    std_dev = np.sqrt(N * eThresh * (1-eThresh))

    with warnings.catch_warnings():
        # prevent RuntimeWarning: divide by zero and invalid value
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        var_after_corr = (std_dev)**2*(((np.e**((T/g)))/N) +
        2*((np.e**((2*T)/g)*(g - T))/(2*g*N**2))*(N*eThresh) +
        3*(((np.e**(((3*T)/g)))*(4*g**2 - 8*g*T + 5*T**2))/(
        12*g**2*N**3))*(N*eThresh)**2)**2

    return var_after_corr

class PhotonCountException(Exception):
    """Exception class for photon_count module."""


def photon_count(e_image, thresh):
    """Convert an analog image into a photon-counted image.

    Args: 
        e_image (array_like, float): Analog image (e-).
        thresh (float): Photon counting threshold (e-). Values > thresh will be assigned a 1,
            values <= thresh will be assigned a 0.

    Returns:
        pc_image (array_like, float): Output digital image in units of photons.

    B Nemati and S Miller - UAH - 06-Aug-2018

    """
    # Check if input is an array/castable to one
    e_image = np.array(e_image).astype(float)
    if len(e_image.shape) == 0:
        raise PhotonCountException('e_image must have length > 0')

    pc_image = np.zeros(e_image.shape, dtype=int)
    pc_image[e_image > thresh] = 1

    return pc_image


def get_pc_mean(input_dataset, pc_master_dark=None, T_factor=None, pc_ecount_max=None, 
                niter=2, mask_filepath=None, safemode=True, inputmode='illuminated', bin_size=None, dataset_copy=True):
    """Take a stack of images, frames of the same exposure 
    time, k gain, read noise, and EM gain, and return the mean expected value per 
    pixel. The frames are taken in photon-counting mode, which means high EM 
    gain and very low exposure time.  

    The frames in each stack should be SCI image (1024x1024) frames that have 
    had some of the L2b steps applied:

    - have had their bias subtracted
    - have had masks made for cosmic rays
    - have been corrected for nonlinearity (These first 3 steps make the frames L2a.)
    - have been frame-selected (to weed out bad frames)
    - have been converted from DN to e-

    This algorithm will photon count each frame individually,
    then co-add the photon-counted frames. The co-added frame is then averaged
    and corrected for thresholding and coincidence loss, returning the mean
    expected array in units of photoelectrons if dark-subtracted (detected electrons 
    if not dark-subtracted).  The threshold is determined by the input "T_factor", 
    and the value stored in DetectorParams is used if this input is None. 
    
    This function can be used for photon-counting illuminated and dark datasets 
    (see Args below). 

    Args:
        input_dataset (corgidrp.data.Dataset): This is an instance of corgidrp.data.Dataset containing the 
            frames to photon-count. All the frames must have the same 
            exposure time, EM gain, k gain, and read noise.  If the input dataset's header key 'VISTYPE' is equal to 'DARK', a 
            photon-counted master dark calibration product will be produced.  Otherwise, the input dataset is assumed to consist of illuminated 
            frames intended for photon-counting.
            If Dataset has metadata only (as in RAM-heavy case), 
            each frame is read in from its filepath one at a time.  If Dataset has 
            its data, then all the frames are processed at once.
        pc_master_dark (corgidrp.data.Dark): Dark containing photon-counted master dark(s) to be used for dark subtraction.  There is a 3-D cube 
            of master darks, 1 2-D slice per subset of frames specified by the input bin_size (see below).
            If None, no dark subtraction is done.
        T_factor (float): The number of read noise standard deviations at which to set the threshold for photon counting.  If None, the value is drawn from corgidrp.data.DetectorParams. 
            Defaults to None.
        pc_ecount_max (float): Maximum allowed electrons/pixel/frame for photon counting.  If None, the value is drawn from corgidrp.data.DetectorParams. 
            Defaults to None.
        niter (int, optional): Number of Newton's method iterations (used for photometric 
            correction). Defaults to 2.
        mask_filepath (str): Filepath to a .fits file with a pixel mask in the default HDU.  The mask should be an array of the same shape as that found in each frame of the input_dataset, and the mask 
            should be 0 where the region of interest is and 1 for pixels to be masked (not considered).
        safemode (bool): If False, the function does not halt due to an exception (useful for the iterative process of digging the dark hole).  
            If True, the function halts with an exception if the mean intensity of the unmasked pixels (or all pixels if no mask provided) is greater than pc_ecount_max or if the minimum photon-counted pixel value is negative.
            If False, the function gives a warning for these instead.
            Defaults to True.
        inputmode (str):  If 'illuminated', the frames are assumed to be illuminated frames.  If 'darks', frames are assumed to be dark frames input for creation of a photon-counted master dark. 
            This flag shows the user's intention with the input, and this input is checked against the file type of the dataset for compatibility (e.g., if this input is 'darks' while 'VISTYPE' is not equal 
            to 'DARK', an exception is raised).
        bin_size (int):  If one wishes to break up the input dataset into subsets of frames to photon-count separately (e.g., for testing the balancing act between 
            good SNR with many frames vs countering speckle time variability with fewer frames), one specifies this number for the size of each subset. If the number does not evenly divide the 
            number of frames in input_dataset, the remainder frames are ignored.  The output is the a dataset containing the more than 1 photon-counted mean-combined frame. Defaults to None, in which case 
            the entire input dataset is used, and the output dataset consists of one frame.
        dataset_copy (bool): flag indicating whether the input dataset will be preserved after this function is executed or not.  If False, the output dataset will be the input dataset modified, and 
            the input and output datasets will be identical.  This is useful when handling a large dataset and when the input dataset is not needed afterwards. Defaults to True.

    Returns:
        corgidrp.data.Dataset or corgidrp.data.Dark: If If the input dataset's header key 'VISTYPE' is not equal to 'CGIVST_CAL_DRK', 
            corgidrp.data.Dataset is the output type, and the output is the processed illuminated set, whether 
            dark subtraction happened or not.  Contains mean expected array (detected electrons if not dark-subtracted, 
            photoelectrons if dark-subtracted). 
            If the input dataset's header key 'VISTYPE' is equal to 'CGIVST_CAL_DRK', corgidrp.data.Dark is the output type, and the output
            is the processed dark set.  Contains mean expected array (detected electrons).

    References
    ----------
    [1] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11443/114435F/Photon-counting-and-precision-photometry-for-the-Roman-Space-Telescope/10.1117/12.2575983.full
    [2] https://doi.org/10.1117/1.JATIS.9.4.048006

    B Nemati, S Miller - UAH - 13-Dec-2020
    Kevin Ludwick - UAH - 2023

    """
    # uncomment for RAM check
    # import psutil 
    # process = psutil.Process()

    if not isinstance(niter, (int, np.integer)) or niter < 1:
            raise PhotonCountException('niter must be an integer greater than '
                                        '0')
    if bin_size is None:
        bin_size = len(input_dataset)
    check.positive_scalar_integer(bin_size, 'bin_size', TypeError)
    if bin_size > len(input_dataset):
        raise ValueError('bin_size must be less than the number of frames in input_dataset.')
        
    num_bins = len(input_dataset)//bin_size 

    lines = []
    for line in input_dataset[0].ext_hdr['HISTORY']:
        lines += [line]
    msg = 'Dark subtracted using dark'
    if msg in lines:
        pc_master_dark = None # dark subtraction was already done, so override any input pc_master_dark
        print("Dark subtraction already done in the dark_subtraction step, so no subtraction done in get_pc_mean.")

    list_new_image = []
    list_err = [] # only used for dark processing case
    list_dq = [] # only used for dark processing case
    index_of_last_frame_used = num_bins*(len(input_dataset)//num_bins)
    # this for loop ignores the remainder from the division above
    for i in range(num_bins):
        subset_frames = input_dataset.frames[bin_size*i:bin_size*(i+1)]
        sub_dataset = data.Dataset(subset_frames)
        if dataset_copy:
            test_dataset, unique_vals = sub_dataset.copy().split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR', 'RN'])
        else:
            test_dataset, unique_vals = sub_dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR', 'RN'])
        if len(test_dataset) > 1:
            raise PhotonCountException('All frames must have the same exposure time, '
                                    'commanded EM gain, and k gain.')
        datasets, val = test_dataset[0].split_dataset(prihdr_keywords=['VISTYPE'])
        if len(val) != 1:
            raise PhotonCountException('There should only be 1 \'VISTYPE\' value for the dataset.')
        if val[0] == 'CGIVST_CAL_DRK':
            if inputmode != 'darks':
                raise PhotonCountException('Inputmode is not \'darks\', but the input dataset has \'VISTYPE\' = \'CGIVST_CAL_DRK\'.')
            if pc_master_dark is not None:
                raise PhotonCountException('The input frames are \'VISTYPE\'=\'CGIVST_CAL_DRK\' frames, so no pc_master_dark should be provided.')
        if val[0] != 'CGIVST_CAL_DRK':
            if inputmode != 'illuminated':
                raise PhotonCountException('Inputmode is not \'illuminated\', but the input dataset has \'VISTYPE\' not equal to \'CGIVST_CAL_DRK\'.')
        if 'ISPC' in datasets[0].frames[0].ext_hdr:
            if datasets[0].frames[0].ext_hdr['ISPC'] != 1:
                raise PhotonCountException('\'ISPC\' header value must be 1 if these frames are to be processed as photon-counted.')

        dataset = datasets[0]
        
        pc_means = []
        errs = []
        dqs = []
        # getting number of read noise standard deviations at which to set the
        # photon-counting threshold
        if T_factor is None:
            detector_params = data.DetectorParams({})
            T_factor = detector_params.params['TFACTOR']
        # getting maximum allowed electrons/pixel/frame for photon counting
        if pc_ecount_max is None:
            detector_params = data.DetectorParams({})
            pc_ecount_max = detector_params.params['PCECNTMX']
        if mask_filepath is None:
            mask = np.zeros_like(dataset.frames[0].data)
        else:
            mask = fits.getdata(mask_filepath)
        
        considered_indices = (mask==0)

        # now get threshold to use for photon-counting
        read_noise = test_dataset[0].frames[0].ext_hdr['RN']
        # Ensure RN is numeric (FITS headers can sometimes preserve string values)
        # if isinstance(read_noise, str): NOTE shouldn't need this when default RN is float, like -999.0
        #     try:
        #         read_noise = float(read_noise.strip()) if read_noise.strip() else 100.0
        #     except (ValueError, TypeError, AttributeError):
        #         read_noise = 100.0
        # else:
        #     read_noise = float(read_noise)
        thresh = T_factor*read_noise
        if thresh < 0:
            raise PhotonCountException('thresh must be nonnegative')
        
        if 'EMGAIN_M' in dataset.frames[0].ext_hdr: # if EM gain measured directly from frame
            em_gain = dataset.frames[0].ext_hdr['EMGAIN_M']
        else:
            em_gain = dataset.frames[0].ext_hdr['EMGAIN_A']
            if em_gain > 0: # use applied EM gain if available
                em_gain = dataset.frames[0].ext_hdr['EMGAIN_A']
            else: # use commanded gain otherwise
                em_gain = dataset.frames[0].ext_hdr['EMGAIN_C']

        if thresh >= em_gain:
            if safemode:
                raise Exception('thresh should be less than em_gain for effective '
                'photon counting')
            else:
                warnings.warn('thresh should be less than em_gain for effective '
                'photon counting')
        if read_noise <=0:
            raise Exception('read noise should be greater than 0 for effective '
            'photon counting')
        if thresh < 4*read_noise: # leave as just a warning
            warnings.warn('thresh should be at least 4 or 5 times read_noise for '
            'accurate photon counting')
        
        if dataset[0].data is None:
            for j, frame in enumerate(dataset.frames): 
                temp_frame = data.Image(frame.filepath)
                frame_data = temp_frame.data
                frame_err = temp_frame.err[0]
                bool_map = temp_frame.dq.astype(bool).astype(float)
                if np.nanmean(frame_data[considered_indices])/em_gain > pc_ecount_max:
                    if safemode:
                        raise Exception('average # of electrons/pixel is > pc_ecount_max, which means '
                        'the average # of photons/pixel may be > pc_ecount_max (depending on the QE).  Can decrease frame '
                        'time to get lower average # of photons/pixel.')
                    else:
                        warnings.warn('average # of electrons/pixel is > pc_ecount_max, which means '
                        'the average # of photons/pixel may be > pc_ecount_max (depending on the QE).  Can decrease frame '
                        'time to get lower average # of photons/pixel.')
                # Photon count stack of frames
                frames_pc = photon_count(frame_data, thresh)
                bool_map[bool_map > 0] = np.nan
                bool_map[bool_map == 0] = 1
                if j == 0:
                    nframes = bool_map
                else:
                    nframes = np.nansum([nframes, bool_map], axis=0)
                # upper and lower bounds for PC (for getting accurate err)
                frames_pc_up = photon_count(frame_data + frame_err, thresh)
                frames_pc_low = photon_count(frame_data - frame_err, thresh)
                frames_pc_masked = frames_pc * bool_map
                frames_pc_masked_up = frames_pc_up * bool_map
                frames_pc_masked_low = frames_pc_low * bool_map
                # Co-add frames
                if j == 0:
                    frame_pc_coadded = frames_pc_masked
                    frame_pc_coadded_up = frames_pc_masked_up
                    frame_pc_coadded_low = frames_pc_masked_low
                else:
                    frame_pc_coadded = np.nansum([frame_pc_coadded, frames_pc_masked], axis=0)
                    frame_pc_coadded_up = np.nansum([frame_pc_coadded_up, frames_pc_masked_up], axis=0)
                    frame_pc_coadded_low = np.nansum([frame_pc_coadded_low, frames_pc_masked_low], axis=0)
        else:    
            if np.nanmean(dataset.all_data[:, considered_indices])/em_gain > pc_ecount_max:
                if safemode:
                    raise Exception('average # of electrons/pixel is > pc_ecount_max, which means '
                    'the average # of photons/pixel may be > pc_ecount_max (depending on the QE).  Can decrease frame '
                    'time to get lower average # of photons/pixel.')
                else:
                    warnings.warn('average # of electrons/pixel is > pc_ecount_max, which means '
                    'the average # of photons/pixel may be > pc_ecount_max (depending on the QE).  Can decrease frame '
                    'time to get lower average # of photons/pixel.')
            # Photon count stack of frames
            frames_pc = photon_count(dataset.all_data, thresh)
            bool_map = dataset.all_dq.astype(bool).astype(float)
            bool_map[bool_map > 0] = np.nan
            bool_map[bool_map == 0] = 1
            nframes = np.nansum(bool_map, axis=0)
            # upper and lower bounds for PC (for getting accurate err)
            frames_pc_up = photon_count(dataset.all_data+dataset.all_err[:,0], thresh)
            frames_pc_low = photon_count(dataset.all_data-dataset.all_err[:,0], thresh)
            frames_pc_masked = frames_pc * bool_map
            frames_pc_masked_up = frames_pc_up * bool_map
            frames_pc_masked_low = frames_pc_low * bool_map
            # Co-add frames
            frame_pc_coadded = np.nansum(frames_pc_masked, axis=0)
            frame_pc_coadded_up = np.nansum(frames_pc_masked_up, axis=0)
            frame_pc_coadded_low = np.nansum(frames_pc_masked_low, axis=0)
        
        # Correct for thresholding and coincidence loss; any pixel masked all the 
        # way through the stack may give NaN, but it should be masked in lam_newton_fit(); 
        # and it doesn't matter anyways since its DQ value will be 1 (it will be masked when the 
        # bad pixel correction is run, which comes after this photon-counting step)
        mean_expected = corr_photon_count(frame_pc_coadded, nframes, thresh,
                                            em_gain, considered_indices, niter)
        mean_expected_up = corr_photon_count(frame_pc_coadded_up, nframes, thresh,
                                            em_gain, considered_indices, niter)
        mean_expected_low = corr_photon_count(frame_pc_coadded_low, nframes, thresh,
                                            em_gain, considered_indices, niter)
        ##### error calculation: accounts for err coming from input dataset and 
        # statistical error from the photon-counting and photometric correction process. 
        # expected error from photon counting (biggest source from the actual values, not 
        # mean_expected_up or mean_expected_low):
        pc_variance = varL23(em_gain,mean_expected,thresh,nframes)
        up = mean_expected_up +  pc_variance
        low = mean_expected_low -  pc_variance
        errs.append(np.max([up - mean_expected, mean_expected - low], axis=0))
        good_inds = np.where(nframes != 0)
        if dataset[0].data is None:
            dq_sum = np.zeros_like(mean_expected).astype(float)
            for j in range(len(dataset)):
                dq_temp = data.Image(dataset[j].filepath).dq 
                dq_sum += dq_temp.astype(float)
            dq_sum = np.ma.masked_array(dq_sum, dq_sum == 0)
            dq = 2**((np.ma.log(dq_sum)/np.log(2)).astype(int)) - 1
            dq = dq.filled(0).astype(int)
        else:
            dq = np.bitwise_or.reduce(dataset.all_dq, axis=0)
        dq[good_inds] = 0 
        pc_means.append(mean_expected)
        dqs.append(dq)
        
        if pc_master_dark is not None:
            if type(pc_master_dark) is not data.Dark:
                raise Exception('Input type for pc_master_dark must be a Dataset of corgidrp.data.Dark instances.')
            if (pc_master_dark.ext_hdr['EXPTIME'], pc_master_dark.ext_hdr['EMGAIN_C']) != (float(unique_vals[0][0]), float(unique_vals[0][1])):
                raise PhotonCountException('Dark should have the same EXPTIME and EMGAIN_C as input_dataset, which are {0} and {1} respectively.'.format((unique_vals[0][0]), unique_vals[0][1]))
            if 'PC_STAT' not in pc_master_dark.ext_hdr:
                raise PhotonCountException('\'PC_STAT\' must be a key in the extension header of each frame of pc_master_dark.')
            if pc_master_dark.ext_hdr['PC_STAT'] == 'photon-counted master dark':
                if 'PCTHRESH' not in pc_master_dark.ext_hdr:
                    raise PhotonCountException('Threshold should be stored under the header \'PCTHRESH\'.')
                if pc_master_dark.ext_hdr['PCTHRESH'] != thresh:
                    raise PhotonCountException('Threshold used for photon-counted master dark should match the threshold to be used for the illuminated frames.')
                if pc_master_dark.ext_hdr['NUM_FR'] < len(sub_dataset):
                    print('Number of frames that created the photon-counted master dark should be greater than or equal to the number of illuminated frames in order for the result to be reliable.')
    
            # in case the number of subsets of darks < number of subsets of brights, which can happen since the number of darks within a subset can be bigger than the number in a bright subset
            j = np.mod(i, pc_master_dark.data.shape[0])
            pc_means.append(pc_master_dark.data[j])
            dqs.append(pc_master_dark.dq[j])
            errs.append(pc_master_dark.err[0][j])
            dark_sub = "yes"
        else:
            pc_means.append(np.zeros_like(pc_means[0]))
            dqs.append(np.zeros_like(pc_means[0]).astype(int))
            errs.append(np.zeros_like(pc_means[0]))
            dark_sub = "no"

        # now subtract the dark PC mean
        combined_dq = np.bitwise_or(dqs[0], dqs[1])
        combined_pc_mean = pc_means[0] - pc_means[1]
        combined_pc_mean[combined_pc_mean<0] = 0
        combined_err = np.sqrt(errs[0]**2 + errs[1]**2)
        combined_dq = np.bitwise_or(dqs[0], dqs[1])
        hdulist = dataset[-1].hdu_list.copy()

        if val[0] != "CGIVST_CAL_DRK":  
            invalid_pc_keywords = data.typical_cal_invalid_keywords
            # keep these since the frames should be from a single visit
            for key in ['PROGNUM', 'EXECNUM', 'CAMPAIGN', 'SEGMENT', 'VISNUM', 'OBSNUM', 'CPGSFILE']:
                if key in invalid_pc_keywords:
                    invalid_pc_keywords.remove(key)
            pri_hdr, ext_hdr, err_hdr, dq_hdr = check.merge_headers(sub_dataset, any_true_keywords=data.typical_bool_keywords, invalid_keywords=invalid_pc_keywords)
            new_image = data.Image(combined_pc_mean, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err=combined_err, dq=combined_dq, err_hdr=err_hdr, 
                                dq_hdr=dq_hdr, input_hdulist=hdulist) 
            new_image.filename = dataset[-1].filename.replace("L2a", "L2b")
            new_image.ext_hdr['PCTHRESH'] = thresh
            new_image.ext_hdr['NUM_FR'] = len(sub_dataset) 
            # Set BUNIT to photoelectron after dark subtraction (same as dark_subtraction function for analog data)
            if dark_sub == "yes":
                new_image.ext_hdr['BUNIT'] = 'photoelectron'
            new_image._record_parent_filenames(sub_dataset) 
            list_new_image.append(new_image)
        else:
            list_new_image.append(combined_pc_mean)
            list_err.append(combined_err)
            list_dq.append(combined_dq)

        # uncomment for RAM check
        # mem = process.memory_info()
        # # peak_wset is only available on Windows; fall back to rss on other platforms
        # if hasattr(mem, 'peak_wset') and getattr(mem, 'peak_wset') is not None:
        #     peak_memory = mem.peak_wset / (1024 ** 2)  # convert to MB
        # else:
        #     peak_memory = mem.rss / (1024 ** 2)  # convert to MB
        # print(f"get_pc_mean peak memory usage:  {peak_memory:.2f} MB")

    if val[0] != "CGIVST_CAL_DRK":
        pc_ill_dataset = data.Dataset(list_new_image)
        pc_ill_dataset.update_after_processing_step("Photon-counted {0} illuminated frames for each PC frame of the output dataset.  Number of subsets: {1}.  Total number of frames in input dataset: {2}. Using T_factor={3} and niter={4}. Dark-subtracted with PC dark: {5}.".format(len(sub_dataset), num_bins, len(input_dataset), T_factor, niter, dark_sub))
        
        return pc_ill_dataset
    else:
        # Dark here may be comprised of a set of master darks, one for each bin, but the frames should be identical, so 
        # use headers from the merging of one of those binned sets to apply for all frames in the output master Dark
        invalid_pc_drk_keywords = data.typical_cal_invalid_keywords 
        # Remove specific keywords
        for key in ['PROGNUM', 'EXECNUM', 'CAMPAIGN', 'SEGMENT', 'VISNUM', 'OBSNUM', 'CPGSFILE']:
            if key in invalid_pc_drk_keywords:
                invalid_pc_drk_keywords.remove(key)
        pri_hdr, ext_hdr, err_hdr, dq_hdr = check.merge_headers(sub_dataset, any_true_keywords=data.typical_bool_keywords, invalid_keywords=invalid_pc_drk_keywords)
        ext_hdr['PC_STAT'] = 'photon-counted master dark'
        ext_hdr['NAXIS1'] = combined_pc_mean.shape[0]
        ext_hdr['NAXIS2'] = combined_pc_mean.shape[1]
        ext_hdr['PCTHRESH'] = thresh
        ext_hdr['NUM_FR'] = len(sub_dataset) 
        ext_hdr['HISTORY'] = "Photon-counted {0} dark frames for each master dark of the output dataset.  Number of subsets: {1}.  Total number of master darks in input dataset: {2}. Using T_factor={3} and niter={4}.".format(len(sub_dataset), num_bins, len(input_dataset), T_factor, niter)
        pc_dark = data.Dark(np.stack(list_new_image), pri_hdr=pri_hdr, ext_hdr=ext_hdr, err=np.stack([list_err]), dq=np.stack(list_dq), err_hdr=err_hdr, dq_hdr=dq_hdr, input_dataset=input_dataset[:index_of_last_frame_used])
        return pc_dark

def corr_photon_count(nobs, nfr, t, g, mask_indices, niter=2):
    """Correct photon counted images.

    Args:
        nobs (array_like): Number of observations (Co-added photon-counted frame).
        nfr (int): Number of coadded frames, accounting for masked pixels in the frames.
        t (float): Photon-counting threshold.
        g (float): EM gain.
        mask_indices (array-like): indices of pixel positions to use.
        niter (int, optional): Number of Newton's method iterations. Defaults to 2.

    Returns:
        lam (array_like): Mean expeted electrons per pixel (lambda).

    """
    # Get an approximate value of lambda for the first guess of the Newton fit
    lam0 = calc_lam_approx(nobs, nfr, t, g)

    # Use Newton's method to converge at a value for lambda
    lam = lam_newton_fit(nobs, nfr, t, g, lam0, niter, mask_indices)

    return lam


def calc_lam_approx(nobs, nfr, t, g):
    """Approximate lambda calculation.

    This will calculate the first order approximation of lambda, and for values
    that are out of bounds (e.g. from statistical fluctuations) it will revert
    to the zeroth order.

    Args:
        nobs (array_like): Number of observations (Co-added photon counted frame).
        nfr (int): Number of coadded frames.
        t (float): Photon counting threshold.
        g (float): EM gain used when taking images.

    Returns:
        lam1 (array_like): Mean expected (lambda).

    """
    # First step of equation (before taking log)
    with warnings.catch_warnings():
        # prevent RuntimeWarning: invalid value in both div statements
        warnings.filterwarnings('ignore', category=RuntimeWarning)
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


def lam_newton_fit(nobs, nfr, t, g, lam0, niter, mask_indices):
    """Newton fit for finding lambda.

    Args:
        nobs (array_like): Number of observations (Co-added photon counted frame).
        nfr (int): Number of coadded frames.
        t (float): Photon counting threshold.
        g (float): EM gain used when taking images.
        lam0 (array_like): Initial guess for lambda.
        niter (int): Number of Newton's fit iterations to take.
        mask_indices (array-like): indices of pixel positions to use.

    Returns:
        lam (array_like): Mean expected (lambda).

    """
    # Mask out zero values to avoid divide by zero
    lam_est_m = np.ma.masked_array(lam0, mask=(lam0==0))
    nobs_m = np.ma.masked_array(nobs, mask=(nobs==0))

    # Iterate Newton's method
    for i in range(niter):
        func = _calc_func(nobs_m, nfr, t, g, lam_est_m)
        dfunc = _calc_dfunc(nfr, t, g, lam_est_m)
        lam_est_m -= func / dfunc

    if np.nanmin(lam_est_m.data[mask_indices]) < 0:
        raise PhotonCountException('negative number of photon counts; '
        'try decreasing the frametime')

    # Fill zero values back in
    lam = lam_est_m.filled(0)

    return lam

def _calc_func(nobs, nfr, t, g, lam):
    """Objective function for lambda for Newton's method for all applying photometric correction.
    
    Args:
        nobs (array-like): number of frames per pixel that passed the threshold
        nfr (array-like): number of unmasked frames per pixel total
        t (float): threshold for photon counting
        g (float): EM gain
        lam (array-like): estimated mean expected electron count

    Returns:
        func (array-like): objective function

    """
    epsilon_prime = (lam*(2*g**2*(6 + lam*(3 + lam)) + 2*g*lam*(3 + lam)*t + 
            lam**2*t**2))/(2.*np.e**(t/g)*g**2*(6 + lam*(6 + lam*(3 + lam))))

    #if (nfr * epsilon_prime).any() > nobs.any():
    #    warnings.warn('Input photon flux is too high; decrease frametime')
    # This warning isn't necessary; could have a negative func but still
    # close enough to 0 for Newton's method
    func = nfr * epsilon_prime - nobs

    return func


def _calc_dfunc(nfr, t, g, lam):
    """Derivative with respect to lambda of objective function.
    
    Args:
        nfr (array-like): number of unmasked frames per pixel total
        t (float): threshold for photon counting
        g (float): EM gain
        lam (array-like): estimated mean expected electron count
    
    Returns:
        dfunc (array-like): derivative with respect to lambda of objective function
    """
    dfunc = (lam*nfr*(2*g**2*(3 + 2*lam) + 2*g*lam*t + 2*g*(3 + lam)*t + 
            2*lam*t**2))/(2.*np.e**(t/g)*g**2*(6 + lam*(6 + lam*(3 + lam)))) - (lam*(6 + 
            lam*(3 + lam) + lam*(3 + 2*lam))*nfr*
         (2*g**2*(6 + lam*(3 + lam)) + 2*g*lam*(3 + lam)*t + lam**2*t**2))/(2.*np.e**(t/g)*g**2*(6 + 
        lam*(6 + lam*(3 + lam)))**2) + (nfr*(2*g**2*(6 + lam*(3 + lam)) + 2*g*lam*(3 + lam)*t + 
        lam**2*t**2))/(2.*np.e**(t/g)*g**2*(6 + lam*(6 + lam*(3 + lam))))

    return dfunc