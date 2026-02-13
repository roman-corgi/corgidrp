import os
import numpy as np
from astropy.time import Time
from astropy.io import fits, ascii

import corgidrp
from corgidrp import astrom, data

here = os.path.abspath(os.path.dirname(__file__))

def get_cfam(
    cfam_name='1F',
    cfam_version=0,
    ):
    """ Read CFAM filter wavelength in nm and transmission.

    Args:
      cfam_name (string): Filter in CFAM. For instance, '1F', '4A', '3B' or '2C'.
      cfam_version (int): version number of the filters (CFAM, pupil, imaging
        lens).

    Returns:
      CFAM filter wavelength in nm and transmission.
    """
    datadir = os.path.join(here, 'data', 'filter_curves')
    filter_names = os.listdir(datadir)
    filter_name = [name for name in filter_names if name.find(cfam_name) >= 0]
    if filter_name == []:
        raise ValueError(f'there is no filter available with name {cfam_name}')
    filter_name = [name for name in filter_name if f'v{cfam_version}' in name]
    if filter_name == []:
        raise ValueError(f'there is no filter {cfam_name} available with version {cfam_version}')
    tab = ascii.read(os.path.join(datadir,filter_name[0]), format='csv',
        header_start = 3, data_start = 4)
    lambda_nm_filter = tab['lambda_nm'].data
    trans_filter = tab['%T'].data / tab['%T'].data.max()
    return lambda_nm_filter, trans_filter

def di_over_pil_transmission(
    cfam_name='1F',
    cfam_version=0,
    ):
    """ Derives the relative transmission between the pupil lens and the imaging
      lens: trans_imaging/trans_pupil.
 
      Multiplying the counts of the pupil image by this factor translates them
      into equivalent counts of the direct imaging lens.
 
    Args:
      cfam_name (string): Filter in CFAM. For instance, '1F', '4A', '3B' or '2C'.
      cfam_version (int): version number of the filters (CFAM, pupil, imaging
        lens).

    Returns:
      Ratio trans_imaging/trans_pupil.
    """
    # Read pupil and direct imaging lenses
    try:
        lambda_pupil_A, trans_pupil = np.loadtxt(os.path.join(here, 'data',
            'filter_curves', f'pupil_lens_v{cfam_version}.txt'),
            delimiter=',', unpack=True)
        lambda_pupil_nm = lambda_pupil_A / 10
    except:
        raise Exception('* File with the transmission of the pupil lens not found')

    try:
        lambda_imaging_A, trans_imaging = np.loadtxt(os.path.join(here, 'data',
            'filter_curves', f'imaging_lens_v{cfam_version}.txt'),
            delimiter=',', unpack=True)
        lambda_imaging_nm = lambda_imaging_A / 10
    except:
        raise Exception('* File with the transmission of the imaging lens not found')

    # Get CFAM filter wavelength and transmission
    lambda_nm_filter, trans_lambda_filter = get_cfam(cfam_name=cfam_name,
        cfam_version=cfam_version)

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
    ratio_imaging_pupil_trans = (np.sum(trans_lambda_imaging_band*trans_lambda_filter)/
        np.sum(trans_lambda_pupil_band*trans_lambda_filter))
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
        psf_pix += [astrom.centroid_with_roi(psf.data,roi_radius=roi_radius)]
    return np.array(psf_pix)

def get_psf_ct(
    dataset,
    unocc_psf_norm=1,
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

    Returns:
      Array of core throughput values between 0 and 1.
    """
    psf_ct = []
    for psf in dataset:
        psf_ct += [psf.data[psf.data >= psf.data.max()/2].sum()/unocc_psf_norm]
    psf_ct = np.array(psf_ct, dtype=float)

    return psf_ct

def estimate_psf_pix_and_ct(
    dataset_in,
    roi_radius=3,
    cfam_version=0,
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
        It includes some pupil images of the unocculted source.  photoelectrons / second / pixel.
      roi_radius (int or float): Half-size of the box around the peak,
        in pixels. Adjust based on desired λ/D.
      cfam_version (int): version number of the filters (CFAM, pupil, imaging
        lens).

    Returns:
      psf_pix (array): Array with PSF's pixel positions. Units: EXCAM pixels
        referred to the (0,0) pixel.
      psf_ct (array): Array with PSF's core throughput values. Units:
        dimensionless (Values must be within 0 and 1).
    """
    dataset = dataset_in.copy()

    # All frames must have the same CFAM filter
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
    if pupil_img_frames:
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
    unocc_psf_norm *= di_over_pil_transmission(cfam_name=cfam_list[0],
        cfam_version=cfam_version)
    # Remove pupil frames
    offaxis_frames = []
    for frame in dataset:
        if frame not in pupil_img_frames:
            offaxis_frames += [frame]
    dataset_offaxis = corgidrp.data.Dataset(offaxis_frames)
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
        unocc_psf_norm = unocc_psf_norm)
    # same number of estimates. One per PSF
    if len(psf_pix) != len(psf_ct) or len(psf_pix) != len(dataset_offaxis):
        raise Exception('PSF positions and CT values are inconsistent')
    return psf_pix, psf_ct

def generate_psf_cube(
    dataset_in,
    psf_loc,
    cfam_name='1F',
    cfam_version=0,
    ):
    """
    Function that derives a 3-d cube of PSF images from a core throughput dataset.

    # TODO: error data cubes will be added in a release after R3.0.2

    Args:
      dataset_in (corgidrp.data.Dataset): A core throughput dataset consisting of
        M clean frames (nominally 1024x1024) taken at different FSM positions.
        It includes some pupil images of the unocculted source.
      psf_loc (array): Array of pair of values with PSFs position in (fractional)
        EXCAM pixels with respect to the pixel (0,0) in the PSF images.
      cfam_name (string): Filter in CFAM. For instance, '1F', '4A', '3B' or '2C'.
      cfam_version (int): version number of the filters (CFAM, pupil, imaging
        lens).

    Returns:
      3-d PSF cube of PSF images from a core throughput dataset, including their
      data quality, and corresponding headers as HDU units. 
    """
    dataset = dataset_in.copy()

    # 3-d cube of PSF images cut around the PSF's location
    psf_cube = []
    dq_cube = []
    # Pixels arounf PSF's location +/- n_pix_psf in both dimensions that
    # correspond to 3 lambda/D in units of EXCAM pixels:
    # 3 * lambda_mean_nm * 1e-9 / D * rad_to_mas / EXCAM_pixel_pitch in mas
    n_pix_psf = int(np.ceil(3*get_cfam(cfam_name=cfam_name,
        cfam_version=cfam_version)[0].mean()*1e-9/2.36*180/np.pi*3600e3/21.8))
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
        idx_0_0 = max(int(np.round(psf_loc[i_psf][1])) - n_pix_psf,0)
        idx_0_1 = min(frame.data.shape[0],
            int(np.round(psf_loc[i_psf][1])) + n_pix_psf + 1)
        idx_1_0 = max(int(np.round(psf_loc[i_psf][0])) - n_pix_psf,0)
        idx_1_1 = min(frame.data.shape[1],
            int(np.round(psf_loc[i_psf][0])) + n_pix_psf + 1)
        psf_cube += [frame.data[idx_0_0:idx_0_1, idx_1_0:idx_1_1]]
        dq_cube += [frame.dq[idx_0_0:idx_0_1, idx_1_0:idx_1_1]]
        i_psf += 1

    psf_cube = np.array(psf_cube)
    dq_cube = np.array(dq_cube)
    # Check
    if len(psf_cube) != len(psf_loc):
        raise Exception(('The number of PSFs does not match the number of PSF '+
            ' locations.'))

    # PSF cube header
    ext_hdr = dataset[0].ext_hdr
    # Add history
    ext_hdr['HISTORY'] = ('Core Throughput calibration derived from a '
        f'set of frames from {dataset[0].ext_hdr["DATETIME"]} to '
        f'{dataset[-1].ext_hdr["DATETIME"]}')
    # Add specific information
    ext_hdr['BUNIT'] = 'photoelectron/pix/s'
    ext_hdr['COMMENT'] = ('Set of PSFs derived from a core throughput '
        'observing sequence. PSFs are not normalized. They are the images of the '
        'off-axis source. The data cube is centered around each PSF location')
    # Add EXTNAME
    psf_hdu = fits.ImageHDU(data=psf_cube, header=ext_hdr, name='PSFCUBE')

    # Data quality cube
    dq_hdr = dataset[0].dq_hdr
    # Add specific information
    dq_hdr['COMMENT'] = 'Data quality for each image' 
    # Add EXTNAME
    dq_hdu = fits.ImageHDU(data=dq_cube, header=dq_hdr, name='DQCUBE')

    return psf_hdu, dq_hdu

def generate_ct_cal(
    dataset_in,
    roi_radius=3,
    cfam_version=0,
    ):
    """
    Generate the elements needed to create a core throughput calibration file.

    A CoreThroughput calibration file has two main data arrays:

    3-d cube of PSF images, e.g, a N1xN1xN array where N1= +/- 3l/D about 
      PSF's centroid in EXCAM pixels. The N PSF images are the ones in the CT
      dataset.

    N sets of (x, y, CT measurements). The (x, y) are pixel coordinates of the
      PSF images in the CT dataset wrt EXCAM (0,0) pixel during core throughput
      observation.

    Args:
      dataset_in (corgidrp.data.Dataset): A core throughput dataset consisting of
        M clean frames (nominally 1024x1024) taken at different FSM positions.
        It includes some pupil images of the unocculted source.
      roi_radius (int or float): Half-size of the box around the peak,
        in pixels. Adjust based on desired λ/D.
      cfam_version (int): version number of the filters (CFAM, pupil, imaging
        lens).

    Returns:
      PSF cube, data quality cube, HDU list with the CT array measurements,
      including the PSF locations, FPAM/FSAM positions, and corrresponding
      headers. 
    """
    dataset = dataset_in.copy()

    # All frames must have the same CFAM filter
    cfam_list = []
    for frame in dataset:
        try:
            cfam_list += [frame.ext_hdr['CFAMNAME']]
        except:
            raise Exception('Frame w/o CFAM specification. All frames must have CFAM specified')
    if len(set(cfam_list)) != 1:
        raise Exception('All frames must have the same CFAM filter')

    # Get estimated PSF centers and CT
    psf_loc_est, ct_est = \
        corgidrp.corethroughput.estimate_psf_pix_and_ct(dataset,
            roi_radius=roi_radius,
            cfam_version=cfam_version)

    psf_hdu, dq_hdu = generate_psf_cube(dataset, psf_loc_est,
        cfam_name=cfam_list[0], cfam_version=cfam_version)
    # N sets of (x,y, CT measurements)
    # x, y: PSF centers wrt EXCAM's (0,0) pixel
    ct_excam = np.array([psf_loc_est[:,0], psf_loc_est[:,1], ct_est])
    ct_hdr = fits.Header()
    # Core throughput values on EXCAM wrt pixel (0,0) (not a "CT map", which is
    # wrt FPM's center 
    ct_hdr['COMMENT'] = ('PSF location with respect to EXCAM (0,0) pixel. '
        'Core throughput value for each PSF. (x,y,ct)=(data[0], data[1], data[2])')
    ct_hdr['UNITS'] = 'PSF location: EXCAM pixels. Core throughput: values between 0 and 1.'
    ct_hdu_list = [fits.ImageHDU(data=ct_excam, header=ct_hdr, name='CTEXCAM')]
    # Values of FPAM during CT observations (needed to derive the FPM's center
    # during CT observations given a coronagraphic dataset). The values do not
    # change during CT observations
    fpam_hv = [dataset_in[0].ext_hdr['FPAM_H'], dataset_in[0].ext_hdr['FPAM_V']]
    fpam_hdr = fits.Header()
    fpam_hdr['COMMENT'] = 'FPAM H and V values during the core throughput observations'
    fpam_hdr['UNITS'] = 'micrometer'
    ct_hdu_list += [fits.ImageHDU(data=fpam_hv, header=fpam_hdr, name='CTFPAM')]
    # Values of FSAM during CT observations (needed to derive the FPM's center
    # during CT observations given a coronagraphic dataset). The values do not
    # change during CT observations
    fsam_hv = [dataset_in[0].ext_hdr['FSAM_H'], dataset_in[0].ext_hdr['FSAM_V']]
    fsam_hdr = fits.Header()
    fsam_hdr['COMMENT'] = 'FSAM H and V values during the core throughput observations'
    fsam_hdr['UNITS'] = 'micrometer'
    ct_hdu_list += [fits.ImageHDU(data=fsam_hv, header=fsam_hdr, name='CTFSAM')]

    # Generate core throughput calibration file
    ct_cal = data.CoreThroughputCalibration(psf_hdu.data,
        pri_hdr=dataset[0].pri_hdr,
        ext_hdr=psf_hdu.header,
        input_hdulist=ct_hdu_list,
        dq=dq_hdu.data,
        dq_hdr=dq_hdu.header,
        input_dataset=dataset)

    return ct_cal

def get_1d_ct(ct_cal,frame,seps,
              method='nearest'):
    """Fetches core throughput values at specific separations from the mask center.
    Currently only the 'nearest' method is configured. 

    Args:
        ct_cal (corgidrp.data.CoreThroughputCalibration): the core throughput calibration 
            object.
        frame (corgidrp.data.Image): data frame containing mask location and detector 0,0 coordinate 
            in the header
        seps (np.array of float): separations (pixels from the mask center) at which to sample 
            the CT curve.
        method (str, optional): Method of calculating CT at a given separation. Defaults to 'nearest'.
            'nearest': grabs the core throughput measured at a location nearest to the desired 
            separation and assumes CT is radially symmetric.

    Returns:
        np.array: Array of shape (2,len(seps)), where the first row is the list of separations 
            sampled, and the second row is the ct value for each separation.
    """
    x, y, ct = ct_cal.ct_excam

    # Get location of mask center in CT coordinates
    xcen = frame.ext_hdr['STARLOCX'] + frame.ext_hdr.get("DETPIX0X",0.) + 0.5
    ycen = frame.ext_hdr['STARLOCY'] + frame.ext_hdr.get("DETPIX0Y",0.) + 0.5

    ct_seps = np.sqrt((x-xcen)**2 + (y-ycen)**2)

    if method == 'nearest':
        cts_out = []
        for sep in seps:
            argmin = np.argmin(np.abs(sep-ct_seps))
            ct_out = ct[argmin]
            cts_out.append(ct_out)
        
        ct_arr_out = np.array([seps,cts_out])
        return ct_arr_out
    else:
        raise NotImplementedError

def create_ct_map(
    corDataset,
    fpamfsamcal,
    ct_cal,
    x_range=[-23,23],
    y_range=[-23,23],
    n_gridx=47,
    n_gridy=47,
    target_pix=None,
    logr=False,
    filepath=None,
    save=False):
    """
      Create a core throughput map: Given a core throughput calibration file and
      a coronagraphic dataset, derive 3-D list (x,y,ct) where (x,y) are some
      target locations on EXCAM relative to the FPM's center and with valid
      values of the throughput.

        The core throughmap may be saved, optionally, as a CSV file.

        The creation of the core throughput map relies on InterpolateCT(), a 
      method of the CoreThroughputCalibration class in data.py. Valid core
      throughput values are within the minimum and maxium radial distance from
      the FPM's center in the core throughput dataset used to generate the
      core throughput calibration file. Its options are inluded in the call of
      this method too.

      If an external list of locations is not provided, a default grid of points
      is condidered.

    Args:
      corDataset (corgidrp.data.Dataset): a dataset containing some
        coronagraphic observations.
      fpamfsamcal (corgidrp.data.FpamFsamCal): an instance of the
        FpamFsamCal class. That is, a FpamFsamCal calibration.
      ct_cal (corgidrp.data.CoreThroughputCalibration): an instance of the
        CoreThroughputCalibration class. That is, a core throughput calibration
        file.
      x_range (array): Two values [xmin, xmax] specifying the range of pixels to
        be considered. Units are EXCAM pixels measured with respect the center
        of the FPM. Notice that [-23,23] is approx. +/-10 l/D in band 1.
      y_range (array): Two values [ymin, ymax] specifying the range of pixels to
        be considered. Units are EXCAM pixels measured with respect the center
        of the FPM. Notice that [-23,23] is approx. +/-10 l/D in band 1.
      n_gridx (int) (optional): Number of x gridpoints.
      n_gridy (int) (optional): Number of y gridpoints.
      target_pix (array) (optional): a user-defined Mx2 array containing the pixel
        positions for M target pixels where the core throughput will be derived
        by interpolation. The target pixels are measured with respect the center
        of the focal plane mask in (fractional) EXCAM pixels. Default is None.
        In this case, a rectangular grid of pixel positions is used. Using
        matplotlib.pyplot, target_pix[0] is the horizontal axis (x), and
        target_pix[1] is the vertical axis (y).
      logr (bool) (optional): If True, radii are mapped into their logarithmic
        values before constructing the interpolant.
      filepath (string) (optional): String with the path and filename of the 
        file that will store the core throughput map as a CSV file.
      save (bool) (optionla): Whether the core throughput map will be stored or not.

    Returns:
        A core throughput map with (x,y,ct) where x and y are locations
        on EXCAM relative to the FPM's center with valid interpolated values of
        the core throughput.
    """
    # If no target pixels are provided, create a grid:
    if target_pix is None:
        x_tmp = np.linspace(x_range[0], x_range[1], n_gridx)
        y_tmp = np.linspace(y_range[0], y_range[1], n_gridy)
        target_pix = np.array(np.meshgrid(x_tmp, y_tmp)).reshape(2, n_gridx*n_gridy)
    # Get interpolated CT values at valid positions
    ct_interp = ct_cal.InterpolateCT(
            target_pix[0], target_pix[1], corDataset, fpamfsamcal, logr=logr)

    # Generate the core throughput map object
    # Re-order output to match the required order: (x,y,ct)
    ct_map = data.CoreThroughputMap(ct_interp[[1,2,0]],
        pri_hdr=corDataset[0].pri_hdr,
        ext_hdr=corDataset[0].ext_hdr,
        input_dataset=corDataset)

    return ct_map
