import numpy as np
from scipy.interpolate import griddata
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

def fpam_mum2pix(
    fpam_pos_um,
    excam_pix_mas=None,
    ):
    """ Translate FPAM positions in micrometers to EXCAM pixels.
    Args:
      fpam_pos_um (array): Value of the FPAM position in units of micrometers.
      excam_pix_mas (float): Value of EXCAM's pixel pitch. Best value from
      TVAC is 21.8 mas, same as as-designed.
    Returns:
      Value of the FPAM position in units of EXCAM pixels
    """
    if excam_pix_mas == None:
        # Best known value (TVAC, same as design)
        excam_pix_mas = 21.8
    # Theoretical value. Replace by measured value during TVAC
    fpam_mum2mas = 2.67
    return fpam_pos_um * fpam_mum2mas / excam_pix_mas

def fsam_mum2pix(
    fsam_pos_um,
    excam_pix_mas=None,
    ):
    """ Translate FSAM positions in micrometers to EXCAM pixels.
    Args:
      fsam_pos_um (array): Value of the FSAM position in units of micrometers.
      excam_pix_mas (float): Value of EXCAM's pixel pitch. Best value from
      TVAC is 21.8 mas, same as as-designed.
    Returns:
      Value of the FSAM position in units of EXCAM pixels
    """
    if excam_pix_mas == None:
        # Best known value (TVAC, same as design)
        excam_pix_mas = 21.8
    # Theoretical value. Replace by measured value during TVAC
    fsam_mum2mas = 2.10
    return fsam_pos_um * fsam_mum2mas / excam_pix_mas

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
    # FPM center must be within EXCAM boundaries and with enough space to
    # accommodate the HLC mask area (OWA radius <=9.7 l/D ~ 487 mas ~ 22.34
    # EXCAM pixels)
    if (np.any(fpm_center_cor <= 23) or np.any(fpm_center_cor >= 1000)):
      raise ValueError("Inout focal plane mask's center is too close to the edges")

    # Translate FPAM positions into EXCAM pixels
    fpam_pos_cor_px = fpam_mum2pix(fpam_pos_cor)
    fpam_pos_ct_px = fpam_mum2pix(fpam_pos_ct)
    # FPAM center must be within EXCAM boundaries and with enough space to
    # accommodate the HLC mask area (OWA radius <=9.7 l/D ~ 487 mas ~ 22.34
    # EXCAM pixels)
    if (np.any(fpam_pos_cor_px <= 23) or np.any(fpam_pos_cor_px >= 1000)):
      raise ValueError("Input FPAM's center is too close to the edges")
    # Translate FSAM positions into EXCAM pixels
    fsam_pos_cor_px = fsam_mum2pix(fsam_pos_cor)
    fsam_pos_ct_px = fsam_mum2pix(fsam_pos_ct)
    print(fpam_pos_cor_px, fpam_pos_ct_px)
    print(fsam_pos_cor_px, fsam_pos_ct_px)    
    # FSAM center must be within EXCAM boundaries and with enough space to
    # accommodate the HLC mask area (OWA radius <=9.7 l/D ~ 487 mas ~ 22.34
    # EXCAM pixels)
    if (np.any(fsam_pos_cor_px <= 23) or np.any(fsam_pos_cor_px >= 1000)):
      raise ValueError("Input FSAM's center is too close to the edges")
    # FPAM center must be within EXCAM boundaries and with enough space to
    # accommodate the HLC mask area (OWA radius <=9.7 l/D ~ 487 mas ~ 22.34
    # EXCAM pixels)
    if (np.any(fpam_pos_ct_px <= 23) or np.any(fpam_pos_ct_px >= 1000)):
      raise ValueError("New FPAM's center is too close to the edges")
    # FSAM center must be within EXCAM boundaries and with enough space to
    # accommodate the HLC mask area (OWA radius <=9.7 l/D ~ 487 mas ~ 22.34
    # EXCAM pixels)
    if (np.any(fsam_pos_ct_px <= 23) or np.any(fsam_pos_ct_px >= 1000)):
      raise ValueError("New FSAM's center is too close to the edges")

    # New FPM center in units of EXCAM pixels
    delta_fpm_ct_cor_px = 0.5*((fpam_pos_ct_px - fpam_pos_cor_px) +
        (fsam_pos_ct_px - fsam_pos_cor_px))
    fpm_center_ct_px = fpm_center_cor + delta_fpm_ct_cor_px
    # FPM center must be within EXCAM boundaries and with enough space to
    # accommodate the HLC mask area (OWA radius <=9.7 l/D ~ 487 mas ~ 22.34
    # EXCAM pixels
    if (np.any(fpm_center_ct_px <= 23) or np.any(fpm_center_ct_px >= 1000)):
      raise ValueError("New focal plane mask's center is too close to the edges")

    return fpm_center_ct_px

def ct_map(
    psf_pix,
    fpam_pix,
    ct,
    target_pix,
    ):
    """
    Function satisfying CTC requirement 1090883.

    Args:
      psf_pix (array): Nx2 array containing the pixel positions for N PSFs in
        EXCAM pixels with respect to (0,0).

      fpam_pix (array): 2-dimensional array with the pixel location of the
        center of the focal plane mask in EXCAM pixels with respect to (0,0).

      ct (array): 1-dimensional array of core throughput values (0,1] associated
        with each PSF.

      target_pix (array): Mx2 array containing the pixel positions for M target
        pixels where the core throughput will be derived by interpolation. the
        target pixels are measured with respect the center of the focal plane
        mask in (fractional) EXCAM pixels.

    Returns:
      Core throughput map: 3-dimensional array (x,y,ct_target) where (x,y) is
      the position of each target pixel location and ct_target is the
      interpolated core throughput value corresponding to each target pixel
      location. 

    """
    # Checks
    # FPAM
    if fpam_pix.shape != (2,):
        raise TypeError('FPAM input must be a two-dimensional array')
    # FPAM center cannnot be closer than 150 pixels to the 1024x1024 boundaries
    fpam_rad_pix = 150
    if ((fpam_pix[0] < fpam_rad_pix) or (1023-fpam_pix[0] < fpam_rad_pix)
      or (fpam_pix[1] < fpam_rad_pix) or (1023-fpam_pix[1] < fpam_rad_pix)):
        raise ValueError(f'FPAM position cannot be closer than {fpam_rad_pix}' + 
            'pixels from the edges of the image')
    # PSF positions must be a 2-dimensional array
    if psf_pix.shape[0] != 2:
      raise TypeError('PSF positions must be a 2-dimensional array')
    # There must be more than one PSF to be able to interpolate
    if psf_pix.shape[1] < 2:
      raise IndexError('There must be at least two PSF positions to ' +
          'construct a ct map')
    # Same number of PSF positions as ct values
    if psf_pix.shape[1] != len(ct):
      raise ValueError('The number of PSF positions and core throughput ' +
          'values must be equal')     
    # ct b/w (0,1]
    if np.any(np.array(ct) <= 0):
      raise ValueError('Core throughput must be positive')
    if np.any(np.array(ct) > 1):
      raise ValueError('Core throughput cannot be greater than 1')
    
    # Use linear interpolation
    ct_interp = griddata((psf_pix[0], psf_pix[1]), ct, (target_pix[0],
        target_pix[1]), method='linear')
    # Check if any PSF position was out of the range for interpolation
    if np.any(np.isnan(ct_interp)):
        # Find where the issue is
        isnan = np.where(np.isnan(ct_interp))[0]
        # Optional diagnosis plot
        if False:
            plt.plot(psf_pix[0], psf_pix[1], 'k+', label='PSF locations')
            plt.plot(target_pix[0], target_pix[1], 'g+', label='Target locations')
            for bad in isnan:
                plt.plot(target_pix[0, bad], target_pix[1, bad], 'rx')
            plt.title('Red crosses indicate target locations that failed')
            plt.legend()
            plt.grid()
            plt.show()
        raise ValueError(f'Target positions at {isnan} gave NaN. Are ' + 
            'this/these position/s within the range of input PSF locations?') 

    return np.array([target_pix[0], target_pix[1], ct_interp])

