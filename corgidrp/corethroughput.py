import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from scipy.interpolate import griddata

from corgidrp.data import Dataset

here = os.path.abspath(os.path.dirname(__file__))

def di_over_pil_transmission(
    filter='1F',
    version=0,
    ):
    """ Derives the relative transmission between the pupil lens and the imaging
      lens: trans_imaging/trans_pupil.
 
      Multiplying the counts of the pupil image by this factor translates them
      into equivalent counts of the direct imaging lens.
 
    Args:
      filter (string): Filter in CFAM. For instance, '1F', '4A', '3B' or '2C'.
        Default: '1F'.

      version (int): version number of the filters (CFAM, pupil, imaging
        lens). Default is 0.

    Returns:
      Ratio trans_imaging/trans_pupil.

    """
    # Read pupil and direct imaging lenses
    try:
        lambda_pupil_A, trans_pupil = np.loadtxt(os.path.join(here, 'data',
            'filter_curves', f'pupil_lens_v{version}.txt'),
            delimiter=',', unpack=True)
        lambda_pupil_nm = lambda_pupil_A / 10
    except:
        raise Exception('* File with the transmission of the pupil lens not found')
 
    try:
        lambda_imaging_A, trans_imaging = np.loadtxt(os.path.join(here, 'data',
            'filter_curves', f'imaging_lens_v{version}.txt'),
            delimiter=',', unpack=True)
        lambda_imaging_nm = lambda_imaging_A / 10
    except:
        raise Exception('* File with the transmission of the imaging lens not found')
 
    # Read filter (CFAM)
    datadir = os.path.join(here, 'data', 'filter_curves')
    filter_names = os.listdir(datadir)
    filter_name = [name for name in filter_names if filter in name]
    if filter_name == []:
        raise ValueError('there is no filter available with name {filter}')
    tab = ascii.read(os.path.join(datadir,filter_name[0]), format='csv',
        header_start = 3, data_start = 4)
    lambda_nm_filter = tab['lambda_nm'].data

    # Linear interpolation
    trans_lambda_pupil_band = np.interp(
        lambda_nm_filter,
        lambda_pupil_nm,
        trans_pupil)
 
    trans_lambda_imaging_band = np.interp(
        lambda_nm_filter,
        lambda_imaging_nm,
        trans_imaging)
    # Ratio of both transmissions:
    ratio_imaging_pupil_trans = \
        np.sum(trans_lambda_imaging_band)/np.sum(trans_lambda_pupil_band)
 
    return ratio_imaging_pupil_trans

def get_psf_pix(
    dataset,
    method='max',
    ):
    """ Estimate the PSF positions of a set of PSF images. 
 
    Args:
      dataset (corgidrp.data.Dataset): a collection of off-axis PSFs.
      
      method (string): the method used to estimate the PSF positions. Default:
        'max'.

    Returns:
      Array of pair of values with PSFs position in (fractional) EXCAM pixels
      with respect to the pixel (0,0) in the PSF images
    """ 
    psf_pix = []
    if method.lower() == 'max':
        for psf in dataset:
            psf_pix += [np.unravel_index(psf.data.argmax(), psf.data.shape)]
    else:
        raise Exception('Method to estimate PSF pixels unrecognized')

    return np.array(psf_pix)

def get_psf_ct(
    dataset,
    unocc_psf_norm=1,
    method='max',
    filter='1F',
    ):
    """ Estimate the core throughput of a set of PSF images.

    Definition of core throughput: The numerator in CT (counts above 50% peak)
    is measured with pupil masks (Lyot stop, SPC pupil mask) in place, DMs at
    dark hole solution, but no FPM.  The denominator (total stellar flux) is
    measured without any masks in place and an infinite aperture.

    NOTE: The FPM are kept in place while measuring the CT because near the
    region of 6 lam/D, the FPM effects are negligible and the CT data set allows
    one to quantify the effect of the FPM in other areas, near the IWA and OWA,
    respectively.

    See  Journal of Astronomical Telescopes, Instruments, and Systems, Vol. 9,
    Issue 4, 045002 (October 2023). https://doi.org/10.1117/1.JATIS.9.4.045002
    and figures 9-13 for details.

    Args:
      dataset (corgidrp.data.Dataset): a collection of off-axis PSFs.

      unocc_psf_norm (float): sum of the 2-d array corresponding to the
        unocculted psf. Default: off-axis PSF are normalized to the unocculted
        PSF already. That is, unocc_psf_norm equals 1.

      method (string): the method used to estimate the PSF core throughput.
        Default: 'direct'. This method finds the set of EXCAM pixels that
        satisfy the condition to derive the core throughput with no approximations.

      filter (string): Filter in CFAM. For instance, '1F', '4A', '3B' or '2C'.
        Default: '1F'.      

      version (int): version number of the filters (CFAM, pupil, imaging
        lens). Default is 0.

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
    version=None,
    ):
    """
    1090881 - Given a core throughput dataset consisting of M clean frames
    (nominally 1024x1024) taken at different FSM positions, the CTC GSW shall
    estimate the pixel location and core throughput of each PSF.

    NOTE: the list of M clean frames may be a subset of the frames collected during
    core throughput data collection, to allow for the removal of outliers.

    Some of the images are pupil images of the unocculted source.

    Args:
      dataset_in (corgidrp.data.Dataset): A core throughput dataset consisting of
        M clean frames (nominally 1024x1024) taken at different FSM positions.
        It includes some pupil images of the unocculted source.
        Units: photoelectrons / second / pixel.

      pix_method (string): The method used to estimate the PSF positions.
        Default: 'max'.

      ct_method (string): The method used to estimate the PSF core throughput.
        Default: 'direct'.        

      version (int): version number of the filters (CFAM, pupil, imaging
        lens). Default is 0.

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
    if version is None:
        version = 0

    # All frames must have the same CFAM setup
    cfam_list = []
    for frame in dataset:
        try:
            cfam_list += [frame.ext_hdr['CFAMNAME']]
        except:
            raise Exception('Frame w/o CFAM specification. All frames must have CFAM specified')
    if len(set(cfam_list)) != 1:
        raise Exception('All frames must have the same CFAM filter')

    # identify the pupil images in the dataset
    pupil_img_frames = []
    for frame in dataset:
        try:
        # Pupil images of the unocculted source satisfy:
        # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
            exthd = frame.ext_hdr
            if (exthd['DPAMNAME']=='PUPIL' and exthd['LSAMNAME']=='OPEN' and
                exthd['FSAMNAME']=='OPEN' and exthd['FPAMNAME']=='OPEN_12'):
                pupil_img_frames += [frame]
        except:
            pass
  
    
    if len(pupil_img_frames):
        print(f'Found {len(pupil_img_frames)} pupil images for the core throughput estimation.')
    else:
        raise Exception('No pupil image found. At least there must be one pupil image.')
    # mean combine the total values (photo-electrons/sec) of the pupil images
    unocc_psf_norm = 0
    for frame in pupil_img_frames:
        unocc_psf_norm += frame.data.sum()
    unocc_psf_norm /= len(pupil_img_frames)
    # Transform pupil counts into direct imaging counts. Recall all frames have
    # the same cfam filter or an Exception is raised
    unocc_psf_norm *= di_over_pil_transmission(filter=cfam_list[0], version=version)
    
    # Remove pupil frames
    offaxis_frames = []
    for frame in dataset:
        if frame not in pupil_img_frames:
            offaxis_frames += [frame]
    dataset_offaxis = Dataset(offaxis_frames)
    if len(dataset_offaxis):
        print(f'Found {len(dataset_offaxis)} off-axis PSFs for the core throughput estimation.')
    else:
        raise Exception('No off-axis PSF found. At least there must be one off-axis PSF.')
    
    # find the PSF positions of the off-axis PSFs
    psf_pix = get_psf_pix(
        dataset_offaxis,
        method=pix_method)

    # find the PSF corethroughput of the off-axis PSFs
    psf_ct = get_psf_ct(
        dataset_offaxis,
        unocc_psf_norm = unocc_psf_norm,
        method=ct_method,
        filter=filter)

    # same number of estimates. One per PSF 
    if len(psf_pix) != len(psf_ct) or len(psf_pix) != len(dataset_offaxis):
        raise Exception('PSF positions and CT values are inconsistent')

    return psf_pix, psf_ct

def fpam_mum2pix(
    delta_fpam_pos_um,
    fpam2excam_matrix=None,
    ):
    """ Translate FPAM delta positions in micrometers to EXCAM pixels.
    Args:
      delta_fpam_pos_um (array): Value of the FPAM delta positions in units of
        micrometers.
      fpam2excam_matrix (string): FITS file with full path included that contains
        the rotation matrix from delta FPAM positions in mum to EXCAM pixels.
    Returns:
      Value of the FPAM delta position in units of EXCAM pixels
    """
    if fpam2excam_matrix == None:
         fpam2excam_matrix = os.path.join(here, 'data', 'fpm_matrices',
           'fpam_to_excam_modelbased.fits')
    try:
        rot_matrix = fits.getdata(fpam2excam_matrix)
    except:
        raise OSError('The rotation matrix for FPAM could not be loaded.')

    return (rot_matrix @ delta_fpam_pos_um)

def fsam_mum2pix(
    delta_fsam_pos_um,
    fsam2excam_matrix=None,
    ):
    """ Translate FSAM delta positions in micrometers to EXCAM pixels.
    Args:
      delta_fsam_pos_um (array): Value of the FSAM delta positions in units of
        micrometers.
      fsam2excam_matrix (string): FITS file with full path included that contains
        the rotation matrix from delta FSAM positions in mum to EXCAM pixels.
    Returns:
      Value of the FSAM delta position in units of EXCAM pixels
    """
    if fsam2excam_matrix == None:
         fsam2excam_matrix = os.path.join(here, 'data', 'fpm_matrices',
           'fsam_to_excam_modelbased.fits')
    try:
        rot_matrix = fits.getdata(fsam2excam_matrix)
    except:
        raise OSError('The rotation matrix for FSAM could not be loaded.')

    return (rot_matrix @ delta_fsam_pos_um)

def get_ct_fpm_center(
    fpm_center_cor,
    fpam_pos_cor=None,
    fpam_pos_ct=None,
    fsam_pos_cor=None,
    fsam_pos_ct=None,
    fpam2excam_matrix=None,
    fsam2excam_matrix=None,
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
      fpam_pos_cor (array): 2-dimensional array with the [H,V] values of the FPAM
        positions during coronagraphic observations. Units: micrometers.
      fpam_pos_ct (array): 2-dimensional array with the [H,V] values of the FPAM
        positions during core throughput observations. Units: micrometers.
      fsam_pos_cor (array): 2-dimensional array with the [H,V] values of the FSAM
        positions during coronagraphic observations. Units: micrometers.
      fsam_pos_ct (array): 2-dimensional array with the [H,V] values of the FSAM
        positions during core throughput observations. Units: micrometers.
      fpam2excam_matrix (string): FITS file with full path included that contains
        the rotation matrix from delta FPAM positions in mum to EXCAM pixels.
      fsam2excam_matrix (string): FITS file with full path included that contains
        the rotation matrix from delta FSAM positions in mum to EXCAM pixels.

        Note: the use of the delta FPAM/FSAM positions and the rotation matrices
        is based on the prescription provided on 1/14/25:
        "H/V values to EXCAM row/column pixels"

          delta_pam = np.array([[dh], [dv]]) # fill these in
          M = np.array([[ 0.        ,  0.12285012],
              [-0.12285012, -0.        ]], dtype=float32)
          delta_pix = M @ delta_pam

    Returns:
      New center of the FPAM and FSAM mask during core throughput observations
      in units of EXCAM pixels.
    """
    # Checks
    try:
        if (type(fpm_center_cor) != np.ndarray or len(fpm_center_cor) !=2 or
            type(fpam_pos_cor) != np.ndarray or len(fpam_pos_cor) !=2 or 
            type(fpam_pos_ct) != np.ndarray or len(fpam_pos_ct) !=2 or
            type(fsam_pos_cor) != np.ndarray or len(fsam_pos_cor) !=2 or
            type(fsam_pos_ct) != np.ndarray or len(fsam_pos_ct) !=2):
            raise OSError('Input values are not 2-dimensional arrays')
    except:
        raise OSError('Input values are not 2-dimensional arrays')
    # FPM center must be within EXCAM boundaries and with enough space to
    # accommodate the HLC mask area (OWA radius <=9.7 l/D ~ 487 mas ~ 22.34
    # EXCAM pixels)
    if (np.any(fpm_center_cor <= 23) or np.any(fpm_center_cor >= 1000)):
      raise ValueError("Input focal plane mask's center is too close to the edges")

    # Translate FPAM delta positions into EXCAM delta pixels
    delta_fpam_pos_um = np.array([[fpam_pos_ct[0]-fpam_pos_cor[0]],
         [fpam_pos_ct[1]-fpam_pos_cor[1]]])
    delta_fpam_pos_px = fpam_mum2pix(delta_fpam_pos_um,
        fpam2excam_matrix=fpam2excam_matrix).transpose()[0]

    # Translate FSAM positions into EXCAM pixels
    delta_fsam_pos_um = np.array([[fsam_pos_ct[0]-fsam_pos_cor[0]],
         [fsam_pos_ct[1]-fsam_pos_cor[1]]])
    delta_fsam_pos_px = fsam_mum2pix(delta_fsam_pos_um,
        fsam2excam_matrix=fsam2excam_matrix).transpose()[0]

    # New FPM center in units of EXCAM pixels
    fpam_center_ct = fpm_center_cor + delta_fpam_pos_px
    fsam_center_ct = fpm_center_cor + delta_fsam_pos_px
    # FPM center must be within EXCAM boundaries and with enough space to
    # accommodate the HLC mask area (OWA radius <=9.7 l/D ~ 487 mas ~ 22.34
    # EXCAM pixels
    if (np.any(fpam_center_ct <= 23) or np.any(fpam_center_ct >= 1000)):
      raise ValueError("New FPAM mask's center is too close to the edges")
    if (np.any(fsam_center_ct <= 23) or np.any(fsam_center_ct >= 1000)):
      raise ValueError("New FSAM mask's center is too close to the edges")

    return fpam_center_ct, fsam_center_ct

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
    # Check if all PSF positions fall out of the range of the input PSFs
    if np.all(np.isnan(ct_interp)):
        raise ValueError('There are no valid target positions within the ' + 
            'range of input PSF locations') 

    isvalid = np.where(np.isnan(ct_interp) == False)[0]
    return np.array([target_pix[0][isvalid], target_pix[1][isvalid], ct_interp[isvalid]])

