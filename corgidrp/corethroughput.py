import numpy as np
import matplotlib.pyplot as plt

from corgidrp.data import Dataset


# CTC requirements
"""
1090881 - Given a core throughput dataset consisting of M clean frames 
(nominally 1024x1024) taken at different FSM positions, the CTC GSW shall
estimate the pixel location and core throughput of each PSF.

NOTE: the list of M clean frames may be a subset of the frames collected during
core throughput data collection, to allow for the removal of outliers.

1090882 - Given 1) the location of the center of the FPM coronagraphic mask in
EXCAM pixels during the coronagraphic observing sequence and 2) the FPAM and
FSAM encoder positions during both the coronagraphic and core throughput observing
sequences, the CTC GSW shall compute the center of the FPM coronagraphic mask
during the core throughput observing sequence. 

1090883 - Given 1) an array of PSF pixel locations and 2) the location of the
center of the FPAM coronagraphic mask in EXCAM pixels during core throughput
calibrations, and 3) corresponding core throughputs for each PSF, the CTC GSW
shall compute a 2D floating-point interpolated core throughput map.

1090884 - Given 1) a core throughput dataset consisting of a set of clean frames
(nominally 1024x1024) taken at different FSM positions, and 2) a list of N (x, y)
coordinates, in units of EXCAM pixels, which fall within the area covered by the
core throughput dataset, the CTC GSW shall produce a 1024x1024xN cube of PSF
images best centered at each set of coordinates.
"""

def get_psf_pix(
    dataset,
    method='max',
    ):
    """ Estimate the PSF positions of a set of PSF images. 
 
    Args:
      dataset (Dataset): a collection of off-axis PSFs.
      
      method (string): the method used to estimate the PSF positions. Default:
        'max'.

    Returns:
      Array of pair of values with PSFs position in (fractional) EXCAM pixels
      with respect to the pixel (0,0) in the PSF images
    """ 
    if method.lower() == 'max':
        psf_pix = []
        for psf in dataset:
            psf_pix += [np.unravel_index(psf.data.argmax(), psf.data.shape)]
        psf_pix = np.array(psf_pix)
    else:
        raise Exception('Method to estimate PSF pixels unrecognized')

    return psf_pix

def get_psf_ct(
    dataset,
    unocc_psf_norm=1,
    method='max',
    ):
    """ Estimate the core throughput of a set of PSF images.

    Definition of core throughput: divide the summed intensity counts of the
      region with intensity >= 50% of the peak by the summed intensity counts
      w/o any masks.

    Args:
      dataset (Dataset): a collection of off-axis PSFs.

      unocc_psf_norm (float): sum of the 2-d array corresponding to the
        unocculted psf. Default: off-axis PSF are normalized to the unocculted
        PSF already. That is, unocc_psf_norm equals 1.

      method (string): the method used to estimate the PSF core throughput.
        Default: 'direct'. This method finds the set of EXCAM pixels that
        satisfy the condition to derive the core throughput with no approximations.

    Returns:
      Array of core throughput values between 0 and 1.
    """
    if method.lower() == 'direct':
        psf_ct = []
        for psf in dataset:
            psf_ct += [psf.data[psf.data >= psf.data.max()/2].sum()/unocc_psf_norm]
        psf_ct = np.array(psf_ct)
    else:
        raise Exception('Method to estimate core throughput unrecognized')

    return psf_ct

def estimate_psf_pix_and_ct(
    dataset_in,
    pix_method=None,
    ct_method=None,
    ):
    """
    1090881 - Given a core throughput dataset consisting of M clean frames
    (nominally 1024x1024) taken at different FSM positions, the CTC GSW shall
    estimate the pixel location and core throughput of each PSF.

    NOTE: the list of M clean frames may be a subset of the frames collected during
    core throughput data collection, to allow for the removal of outliers.

    Args:
      dataset_in (corgidrp.Dataset): A core throughput dataset consisting of
        M clean frames (nominally 1024x1024) taken at different FSM positions.
        Units: photoelectrons / second / pixel.

        NOTE: the dataset contains the pupil image(s) of the unocculted source.

      pix_method (string): the method used to estimate the PSF positions.
        Default: 'max'.

      ct_method (string): the method used to estimate the PSF core throughput.
        Default: 'direct'.        

    Returns:
      psf_pix (array): Array with PSF's pixel positions. Units: EXCAM pixels
        referred to the (0,0) pixel.

      psf_ct (array): Array with PSF's core throughput values. Units:
        dimensionless (Values must be within 0 and 1).
    """
    dataset = dataset_in.copy()

    # default methods
    if pix_method is None:
        pix_method = 'max'
    if ct_method is None:
        ct_method = 'direct'

    # identify the pupil images in the dataset (pupil images are extended)
    n_pix_up = [np.sum(np.where(frame.data > 3*frame.data.std())) for frame in dataset]
    # frames are mostly off-axis PSFs
    pupil_img_idx = np.where( n_pix_up > 10 * np.median(n_pix_up))[0]
    print(f'Found {len(pupil_img_idx)} pupil images for the core throughput estimation') 
    # mean combine the total values (photo-electrons/sec)
    unocc_psf_norm = 0
    for frame in dataset[pupil_img_idx]:
        unocc_psf_norm += frame.data.sum()
    unocc_psf_norm /= len(pupil_img_idx)
    # Remove pupil frames
    offaxis_frames = []
    for i_f, frame in enumerate(dataset):
        if i_f not in pupil_img_idx:
            offaxis_frames += [frame]
    dataset_offaxis = Dataset(offaxis_frames)

    # find the PSF positions of the off-axis PSFs
    psf_pix = get_psf_pix(
        dataset_offaxis,
        method=pix_method)

    # find the PSF corethroughput of the off-axis PSFs
    psf_ct = get_psf_ct(
        dataset_offaxis,
        unocc_psf_norm = unocc_psf_norm,
        method=ct_method)

    # same number of estimates. One per PSF 
    if len(psf_pix) != len(psf_ct) or len(psf_pix) != len(dataset_offaxis):
        raise Exception('PSF positions and CT values are inconsistent')

    return psf_pix, psf_ct

def get_ct_fpm_center(
    fpm_center_cor,
    fpam_pos_cor=None,
    fpam_pos_ct=None,
    fsam_pos_cor=None,
    fsam_pos_ct=None,
    ):
    """
    1090882 - Given 1) the location of the center of the FPM coronagraphic mask
    in EXCAM pixels during the coronagraphic observing sequence and 2) the FPAM
    and FSAM encoder positions during both the coronagraphic and core throughput
    observing sequences, the CTC GSW shall compute the center of the FPM
    coronagraphic mask during the core throughput observing sequence.

    Args:
      fpm_center_cor (array): 2-dimensional array with the center of the focal
        plane mask during coronagraphic observations. Units: EXAM pixels.
      fpam_pos_cor (array): 2-dimensional array with the H/V values of the FPAM
        positions during coronagraphic observations. Units: micrometers.
      fpam_pos_ct (array): 2-dimensional array with the H/V values of the FPAM
        positions during core throughput observations. Units: micrometers.
      fsam_pos_cor (array): 2-dimensional array with the H/V values of the FSAM
        positions during coronagraphic observations. Units: micrometers.
      fsam_pos_ct (array): 2-dimensional array with the H/V values of the FSAM
        positions during core throughput observations. Units: micrometers.

    Returns:
      New center of the focal plane mask during core throughput observations in
      units of EXCAM pixels.
    """
    # Checks
    try:
        if (type(fpm_center_cor) != np.ndarray or len(fpm_center_cor) !=2 or
            type(fpam_pos_cor) != np.ndarray or len(fpam_pos_cor) !=2 or 
            type(fpam_pos_ct) != np.ndarray or len(fpam_pos_ct) !=2 or
            type(fsam_pos_cor) != np.ndarray or len(fsam_pos_cor) !=2 or
            type(fsam_pos_ct) != np.ndarray or len(fsam_pos_ct) !=2):
            raise IOError('Input values are not 2-dimensional arrays')
    except:
        raise IOError('Input values are not 2-dimensional arrays')



