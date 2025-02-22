import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import fits, ascii
from scipy.interpolate import griddata

import corgidrp
import corgidrp.data as data
from corgidrp.data import Dataset
from corgidrp.astrom import centroid_with_roi
from corgidrp import corethroughput

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
    roi_radius=3,
    ):
    """ Estimate the PSF positions of a set of PSF images. 
 
    Args:
      dataset (corgidrp.data.Dataset): a collection of off-axis PSFs.
      
      roi_radius (int or float): Half-size of the box around the peak,
        in pixels. Adjust based on desired λ/D.

    Returns:
      Array of pair of values with PSFs position in (fractional) EXCAM pixels
      with respect to the pixel (0,0) in the PSF images
    """ 
    psf_pix = []
    for psf in dataset:
        psf_pix += [centroid_with_roi(psf.data,roi_radius=roi_radius)]

    return np.array(psf_pix)

def get_psf_ct(
    dataset,
    unocc_psf_norm=1,
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

      filter (string): Filter in CFAM. For instance, '1F', '4A', '3B' or '2C'.
        Default: '1F'.      

      version (int): version number of the filters (CFAM, pupil, imaging
        lens). Default is 0.

    Returns:
      Array of core throughput values between 0 and 1.
    """
    psf_ct = []
    for psf in dataset:
        psf_ct += [psf.data[psf.data >= psf.data.max()/2].sum()/unocc_psf_norm]
    psf_ct = np.array(psf_ct)

    return psf_ct

def estimate_psf_pix_and_ct(
    dataset_in,
    roi_radius=None,
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

      roi_radius (int or float): Half-size of the box around the peak,
        in pixels. Adjust based on desired λ/D.

      version (int): version number of the filters (CFAM, pupil, imaging
        lens). Default is 0.

    Returns:
      psf_pix (array): Array with PSF's pixel positions. Units: EXCAM pixels
        referred to the (0,0) pixel.

      psf_ct (array): Array with PSF's core throughput values. Units:
        dimensionless (Values must be within 0 and 1).
    """
    dataset = dataset_in.copy()

    if roi_radius is None:
        roi_radius = 3

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
        roi_radius=roi_radius)

    # find the PSF corethroughput of the off-axis PSFs
    psf_ct = get_psf_ct(
        dataset_offaxis,
        unocc_psf_norm = unocc_psf_norm,
        filter=filter)

    # same number of estimates. One per PSF 
    if len(psf_pix) != len(psf_ct) or len(psf_pix) != len(dataset_offaxis):
        raise Exception('PSF positions and CT values are inconsistent')

    return psf_pix, psf_ct

def read_rot_matrix():
    """ Read latest calibration file with the FPAM and FSAM rotation matrices."""

    # Check for latest time with FPAM/FSAM rotation matrices
    try:
        idx1 = len('FpamFsamRotMat')
        for _, _, files in os.walk(corgidrp.default_cal_dir):
            calfile_list = [file for file in files if 'FpamFsamRotMat' in file]
            calfile_date = [Time(file[idx1+1:-5]) for file in calfile_list]
        calfile_latest = calfile_list[np.array(calfile_date).argmax()]
        rot_matrix = fits.getdata(os.path.join(corgidrp.default_cal_dir, calfile_latest))
        try:
            fpam2excam_matrix = rot_matrix[0]
            fsam2excam_matrix = rot_matrix[1]
        except:
            raise ValueError(f'The data in {calfile_latest} does not have two (2x2) rotation matrices.')
    except:
        raise OSError('The rotation matrix for FPAM and FSAM could not be loaded.')

    return fpam2excam_matrix, fsam2excam_matrix

def pam_mum2pix(
    pam2excam_matrix,
    delta_pam_pos_um,
    ):
    """ Translate PAM delta positions in micrometers to EXCAM pixels.
    Args:
      pam2excam_matrix (array): Rotation matrix to translate delta PAM positions
        in micrometer to EXCAM (direct imaging) pixels
      delta_pam_pos_um (array): Value of the PAM delta positions in units of
        micrometer.
    Returns:
      Value of the PAM delta position in units of EXCAM (direct imaging) pixels
    """
    # Enforce vertical array. Transpose if it is a horizontal array
    try:
        if delta_pam_pos_um.shape != (2,1):
            delta_pam_pos_um = np.array([delta_pam_pos_um]).transpose()
            if delta_pam_pos_um.shape != (2,1):
                raise ValueError('Input delta PAM must be a 2-1 array')
    except:
        raise ValueError('Input delta PAM must be a 2-1 array')

    return (pam2excam_matrix @ delta_pam_pos_um).transpose()[0]

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
      fpam_pos_cor (array): 2-dimensional array with the [H,V] values of the FPAM
        positions during coronagraphic observations. Units: micrometers.
      fpam_pos_ct (array): 2-dimensional array with the [H,V] values of the FPAM
        positions during core throughput observations. Units: micrometers.
      fsam_pos_cor (array): 2-dimensional array with the [H,V] values of the FSAM
        positions during coronagraphic observations. Units: micrometers.
      fsam_pos_ct (array): 2-dimensional array with the [H,V] values of the FSAM
        positions during core throughput observations. Units: micrometers.

      FPAM and FSAM rotation matrices are read from a FpamFsamRotMat calibration
      file with the date closest to the time of running this script.

        Note: the use of the delta FPAM/FSAM positions and the rotation matrices
        is based on the prescription provided on 1/14/25:
        "H/V values to EXCAM row/column pixels"

          delta_pam = np.array([[dh], [dv]]) # fill these in
          M = np.array([[ M00, M01], [M10, M11]], dtype=float32)
          delta_pix = M @ delta_pam

    Returns:
      New center of the FPAM and FSAM mask during core throughput observations
      in units of EXCAM (direct imaging) pixels.
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

    # Read FPAM and FSAM rotation matrices from their calibration file
    fpam2excam_matrix, fsam2excam_matrix = read_rot_matrix()
    # Translate FPAM delta positions into EXCAM delta pixels
    delta_fpam_pos_um = np.array([[fpam_pos_ct[0]-fpam_pos_cor[0]],
         [fpam_pos_ct[1]-fpam_pos_cor[1]]])
    delta_fpam_pos_px = pam_mum2pix(fpam2excam_matrix, delta_fpam_pos_um)

    # Translate FSAM positions into EXCAM pixels
    delta_fsam_pos_um = np.array([[fsam_pos_ct[0]-fsam_pos_cor[0]],
         [fsam_pos_ct[1]-fsam_pos_cor[1]]])
    delta_fsam_pos_px = pam_mum2pix(fsam2excam_matrix, delta_fsam_pos_um)

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

def write_ct_calfile(
    dataset_in,
    fpm_center_cor,
    fpam_pos_cor,
    fpam_pos_ct,
    fsam_pos_cor,
    fsam_pos_ct,
    roi_radius=None,
    version=None,
    n_pix_psf=None,
    ):
    """
    1090884 - Given 1) a core throughput dataset consisting of a set of clean
    frames (nominally 1024x1024) taken at different FSM positions, and 2) a list
    of N (x, y) coordinates, in units of EXCAM pixels, which fall within the area
    covered by the core throughput dataset, the CTC GSW shall produce a
    1024x1024xN cube of PSF images best centered at each set of coordinates

    A CoreThroughput calibration file has two main data arrays:
    
      3-d cube of PSF images, i.e, a N1xN1xN array where N1<=1024 is set by a
      keyword argument, with default value of 1024. The N PSF images are the ones
      in the CT dataset (obtained in 1090881 and 1090884)
      
      N sets of (x,y, CT measurements). The (x,y) are pixel coordinates of the
      PSF images wrt the FPAM's center (obtained in 1090881 and 1090882)

      The CoreThroughput calibration file will also include the FPAM, FSAM
      position during coronagraphic and core throughput observing sequences in
      units of EXCAM pixels (Obtained in 1090882)

    Args:
      dataset_in (corgidrp.data.Dataset): A core throughput dataset consisting of
        M clean frames (nominally 1024x1024) taken at different FSM positions.
        It includes some pupil images of the unocculted source.
        Units: photoelectrons / second / pixel.
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
      roi_radius (int or float): Half-size of the box around the peak,
        in pixels. Adjust based on desired λ/D.
      version (int): version number of the filters (CFAM, pupil, imaging
        lens). Default is 0.
      n_pix_psf (int): The number of pixels of each PSF array dimension. The
        PSF array is centered at the EXCAM pixel closest to the PSF's location.
        Default: 15 EXCAM pixels (corresponding to radius from PSF's centroid
        of 3 l/D, where the PSF intensity is ~1e-10 its peak). 

    Returns:
      CoreThroughputCalibration file.
    """
    dataset = dataset_in.copy()

    # default methods
    if roi_radius is None:
        roi_radius = 3
    if version is None:
        version = 0
    if n_pix_psf is None:
        n_pix_psf = 15

    # Get PSF centers and CT
    psf_loc_ct, ct = \
        corethroughput.estimate_psf_pix_and_ct(dataset,
            roi_radius=roi_radius,
            version=version)
    # Get FPAM and FSAM centers during CT in EXCAM pixels
    fpam_center_ct_pix, fsam_center_ct_pix = \
            corethroughput.get_ct_fpm_center(fpm_center_cor,
            fpam_pos_cor=fpam_pos_cor,
            fpam_pos_ct=fpam_pos_ct,
            fsam_pos_cor=fsam_pos_cor,
            fsam_pos_ct=fsam_pos_ct)
    # Collect data
    # First extension: 3-d cube of PSF images cut around the PSF's location
    psf_cube = []
    n_pix_psf_1 = n_pix_psf // 2
    n_pix_psf_2 = n_pix_psf - n_pix_psf_1
    i_psf = 0
    for frame in dataset:
        # Skip pupil images of the unocculted source, which satisfy:
        # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
        try:
        # Pupil images of the unocculted source satisfy:
        # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
            exthd = frame.ext_hdr
            if (exthd['DPAMNAME']=='PUPIL' and exthd['LSAMNAME']=='OPEN' and
                exthd['FSAMNAME']=='OPEN' and exthd['FPAMNAME']=='OPEN_12'):
                continue
        except:
           pass 
        idx_0_0 = max(int(np.round(psf_loc_ct[i_psf][1])) - n_pix_psf_1,0)
        idx_0_1 = min(frame.data.shape[0],
            int(np.round(psf_loc_ct[i_psf][1])) + n_pix_psf_2)
        idx_1_0 = max(int(np.round(psf_loc_ct[i_psf][0])) - n_pix_psf_1,0)
        idx_1_1 = min(frame.data.shape[1],
            int(np.round(psf_loc_ct[i_psf][0])) + n_pix_psf_2)
        psf_cube += [frame.data[idx_0_0:idx_0_1, idx_1_0:idx_1_1]]
        i_psf += 1 
        # Get headers from an off-axis PSF
        prhd_offaxis = frame.pri_hdr
        exthd_offaxis = frame.ext_hdr

    psf_cube = np.array(psf_cube)
    # Check
    if len(psf_cube) != len(psf_loc_ct) or len(psf_cube) != len(ct):
        raise Exception(('The number of PSFs does not match the number of PSF '+
            ' locations and/or core throughput values'))
   
    # Add history
    exthd_offaxis['HISTORY'] = ('Core Throughput calibration derived from a '
        f'set of frames on {exthd_offaxis["DATETIME"]}')
    # Add specific information
    exthd_offaxis['UNITS'] = 'photoelectron/pix/s'
    exthd_offaxis['COMMENT'] = ('Set of PSFs derived from a core throughput '
        'observing sequence. PSFs are not normalized. They are the L2b images '
        'of the off-axis source. The data cube is centered around each PSF location')

    # N sets of (x,y, CT measurements)
    # x, y: PSF centers wrt FPAM's center
    psf_loc = psf_loc_ct - fpam_center_ct_pix
    ct_map = np.array([psf_loc[:,0], psf_loc[:,1], ct])
    ct_hdr = fits.Header()
    ct_hdr['COMMENT'] = ('PSF location with respect to FPAM center. '
        'Core throughput value for each PSF. (x,y,ct)=(data[0], data[1], data[2])')
    ct_hdr['UNITS'] = 'PSF locsation: EXCAM pixels. Core throughput: values between 0 and 1.'

    # FPM information:
    fpm_info = np.array([fpam_center_ct_pix, fsam_center_ct_pix,fpm_center_cor,
        fpam_pos_cor,fpam_pos_ct,fsam_pos_cor,fsam_pos_ct])
    fpm_hdr = fits.Header()
    fpm_hdr['COMMENT'] = ('fpm_info[0]=FPAM center during core throughput '
        'observing sequences in units of EXCAM pixels.'
        'fpm_info[1]=FSAM center during core throughput observing sequences in '
        'units of EXCAM pixels. '
        'fpm_info[2]=FPM center during coronagraphic observing sequences in '
        'units of EXCAM pixels.'
        'fpm_info[3]=FPAM H/V values during coronagraphic observing sequences '
        'in units of micron. '
        'fpm_info[4]=FPAM H/V values during core throughput observing sequences '
        'in units of micron.'
        'fpm_info[5]=FSAM H/V values during coronagraphic observing sequences '
        'in units of micron. '
        'fpm_info[6]=FSAM H/V values during core throughput observing sequences '
        'in units of micron.')

    # Create an instance of the CoreThroughputCalibration class and save it
    ct_cal_file = data.CoreThroughputCalibration(psf_cube, 
        pri_hdr = prhd_offaxis, ext_hdr = exthd_offaxis,
        ct_map=ct_map, ct_hdr=ct_hdr,
        fpm_info=fpm_info, fpm_hdr=fpm_hdr,
        input_dataset=dataset)
    ct_cal_file.save(filedir=corgidrp.default_cal_dir)

def read_ct_cal_file():
    """ Read latest core throughput calibration file."""

    # Check for latest time with FPAM/FSAM rotation matrices
    try:
        idx1 = len('CoreThroughputCalibration')
        for _, _, files in os.walk(corgidrp.default_cal_dir):
            calfile_list = [file for file in files if 'CoreThroughputCalibration' in file]
            calfile_date = [Time(file[idx1+1:-5]) for file in calfile_list]
        calfile_latest = calfile_list[np.array(calfile_date).argmax()]
        
        with fits.open(os.path.join(corgidrp.default_cal_dir, calfile_latest)) as hdul:
            pri_hdr = hdul[0].header
            psf_cube = hdul[1].data
            psf_hdr = hdul[1].header
            err_cube = hdul[2].data
            err_hdr = hdul[2].header
            dq_cube = hdul[3].data
            dq_hdr = hdul[3].header
            ct_map = hdul[4].data
            ct_hdr = hdul[4].header
            fpm_info = hdul[5].data
            fpm_hdr = hdul[5].header
    except:
        raise OSError('The core throughput calibration file could not be loaded.')

    # Print a reminder of the content of the CT calibration file
    print('Core throughput calibration file:')
    print('Primary header: [0]')
    print('PSF cube: [1]')
    print('PSF cube header: [2]')
    print('PSF cube err: [3]')
    print('PSF cube err header: [4]')
    print('PSF cube dq: [5]')
    print('PSF cube dq header: [6]')
    print('CT map: [7]. PSF location: [7][0], [7][1]. CT values: [7][2]')
    print('CT map header: [8]')
    print('FPM info: [9]. FPAM during CT observations: [9][0], FSAM during CT observations: [9][1].')
    print('FPM during coronagraphic observations: [9][2].')
    print('FPAM H/V values during coronagraphic observations: [9][3], FPAM H/V values during corethroughput observations: [9][4]')
    print('FSAM H/V values during coronagraphic observations: [9][5], FSAM H/V values during corethroughput observations: [9][6]')
    print('FPM info header: [10]')

    return [pri_hdr, psf_cube, psf_hdr, err_cube, err_hdr, dq_cube, dq_hdr,
        ct_map, ct_hdr, fpm_info, fpm_hdr]
