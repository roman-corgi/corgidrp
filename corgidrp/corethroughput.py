import os
import numpy as np
from scipy.ndimage import shift as ndi_shift
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
                exthd['FSAMNAME']=='OPEN' and exthd['FPAMNAME'] in ['OPEN_12','OPEN_34']):
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
        # prevent NaNs from corrupting the entire pupil image value
        unocc_psf_norm += np.nansum(frame.data)
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

def subpixel_center_stamp(
    cutout_data,
    cutout_dq,
    dx,
    dy,
    spline_order=3,
    ):
    """ Apply a sub-pixel shift to a PSF stamp and its DQ array so that the
    PSF centroid lands centered on the stamp's central pixel.

    The data array is shifted with a cubic spline. NaNs in the input would 
    propagate through the spline coefficients and be an issue in the stamp, 
    so they are temporarily replaced with zeros. The NaN mask is shifted
    separately and reapplied to the output so that originally NaN regions 
    remain NaN.

    The DQ array is shifted with nearest-neighbour interpolation since its 
    values are integers. Pixels shifted in from outside the stamp are flagged 
    as bad.

    Args:
        cutout_data (np.ndarray): 2-D PSF stamp (could contain NaN)
        cutout_dq (np.ndarray): 2-D DQ array of the same shape
        dx (float): Sub-pixel shift along the x axis
        dy (float): Sub-pixel shift along the y axis
        spline_order (int): Spline order for the data shift. Default 3.

    Returns:
        shifted_data (np.ndarray): Data stamp shifted by (dy, dx) 
        shifted_dq (np.ndarray): DQ stamp shifted by (dy, dx) 
    """
    # Track NaN regions separately so the spline does not smear them into PSF data
    nan_mask = np.isnan(cutout_data)
    data_filled = np.where(nan_mask, 0.0, cutout_data) if nan_mask.any() else cutout_data

    # Shift the data with a spline. Pixels that get shifted in are marked with 
    # NaN
    shifted_data = ndi_shift(
        data_filled,
        shift=(dy, dx),
        order=spline_order,
        mode='constant',
        cval=0.0,
    )
    # Shift the NaN mask as well (skip when no NaN)
    if nan_mask.any():
        shifted_nan_mask = ndi_shift(
            nan_mask.astype(np.float32),
            shift=(dy, dx),
            order=spline_order,
            mode='constant',
            cval=0.0,
        ) > 0.5
        shifted_data = np.where(shifted_nan_mask, np.nan, shifted_data)

    # Shift the DQ array with nearest-neighbour interpolation
    shifted_dq = ndi_shift(
        cutout_dq,
        shift=(dy, dx),
        order=0,
        mode='constant',
        cval=1,
    ).astype(cutout_dq.dtype)

    return shifted_data, shifted_dq

def generate_psf_cube(
    dataset_in,
    psf_loc,
    cfam_name='1F',
    cfam_version=0,
    spline_order=3,
    ):
    """
    Function that derives a 3-d cube of PSF images from a core throughput dataset.

    Each PSF stamp is centered on the central pixel.

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
      spline_order (int): Spline order used to shift and center each stamp.
        Default 3 (cubic). Use 1 for linear interpolation.

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
    expected_size = 2*n_pix_psf + 1

    i_psf = 0
    for frame in dataset:
        # Skip pupil images of the unocculted source, which satisfy:
        # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
        try:
            exthd = frame.ext_hdr
            if (exthd['DPAMNAME']=='PUPIL' and exthd['LSAMNAME']=='OPEN' and
                exthd['FSAMNAME']=='OPEN' and exthd['FPAMNAME']=='OPEN_12'):
                continue
        except:
            pass
        
        # Sub-pixel centroid of this PSF on EXCAM, and the integer pixel it falls on
        cx = float(psf_loc[i_psf][0])
        cy = float(psf_loc[i_psf][1])
        cx_round = int(np.round(cx))
        cy_round = int(np.round(cy))

        # Bounding box of size (2*n_pix_psf + 1)^2 around the rounded centroid
        # pixel. Axis 0 = y (rows), axis 1 = x (cols). max/min clip to the frame
        # edges so PSFs near the boundary produce smaller cutouts; these are
        # padded back to uniform size below.
        idx_0_0 = max(cy_round - n_pix_psf, 0)
        idx_0_1 = min(frame.data.shape[0], cy_round + n_pix_psf + 1)
        idx_1_0 = max(cx_round - n_pix_psf, 0)
        idx_1_1 = min(frame.data.shape[1], cx_round + n_pix_psf + 1)
        cutout_data = frame.data[idx_0_0:idx_0_1, idx_1_0:idx_1_1]
        cutout_dq = frame.dq[idx_0_0:idx_0_1, idx_1_0:idx_1_1]

        # PSFs near field stop boundary produce clipped cutouts with varying shapes
        # (eg not all 15x15). Pad them to uniform size so np.array() can create 
        # the PSF cube. Padded regions are filled with NaN (data) and DQ flag 1 
        # (bad pixel).
        if cutout_data.shape[0] != expected_size or cutout_data.shape[1] != expected_size:
            padded_data = np.full((expected_size, expected_size), np.nan, dtype=cutout_data.dtype)
            padded_dq = np.full((expected_size, expected_size), 1, dtype=cutout_dq.dtype)
            # Place the clipped cutout so the rounded centroid pixel lands at
            # (n_pix_psf, n_pix_psf) in the padded array.
            y_offset = n_pix_psf - (cy_round - idx_0_0)
            x_offset = n_pix_psf - (cx_round - idx_1_0)
            padded_data[y_offset:y_offset+cutout_data.shape[0],
                       x_offset:x_offset+cutout_data.shape[1]] = cutout_data
            padded_dq[y_offset:y_offset+cutout_dq.shape[0],
                     x_offset:x_offset+cutout_dq.shape[1]] = cutout_dq
            
            cutout_data = padded_data
            cutout_dq = padded_dq

        # Sub-pixel centering:
        # The cutout is currently centered on the rounded centroid pixel.
        # Shift it by (round(c) - c) so the centroid lands on the central 
        # pixel of the stamp.
        dx = cx_round - cx
        dy = cy_round - cy
        # Skip the shift for negligible offsets to avoid introducing
        # interpolation noise when the PSF is already pixel-centered.
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            cutout_data, cutout_dq = subpixel_center_stamp(
                cutout_data, cutout_dq, dx, dy, spline_order=spline_order)

        psf_cube += [cutout_data]
        dq_cube += [cutout_dq]
        i_psf += 1

    psf_cube = np.array(psf_cube)
    dq_cube = np.array(dq_cube)
    # Check
    if len(psf_cube) != len(psf_loc):
        raise Exception(('The number of PSFs does not match the number of PSF '+
            ' locations.'))

    # PSF cube header: use first off-axis frame so pupil headers don't propagate
    first_offaxis_frame = None
    for frame in dataset:
        exthd = frame.ext_hdr
        if not (exthd['DPAMNAME'] == 'PUPIL' and exthd['LSAMNAME'] == 'OPEN' and
                exthd['FSAMNAME'] == 'OPEN' and exthd['FPAMNAME'] == 'OPEN_12'):
            first_offaxis_frame = frame
            break
    if first_offaxis_frame is None:
        raise Exception('No off-axis PSF frame found in dataset')
    ext_hdr = first_offaxis_frame.ext_hdr
    # Add EXTNAME
    psf_hdu = fits.ImageHDU(data=psf_cube, header=ext_hdr, name='PSFCUBE')

    # Data quality cube
    dq_hdr = first_offaxis_frame.dq_hdr
    # Add specific information
    dq_hdr['COMMENT'] = 'Data quality for each image' 
    # Add EXTNAME
    dq_hdu = fits.ImageHDU(data=dq_cube, header=dq_hdr, name='DQCUBE')

    return psf_hdu, dq_hdu

def generate_ct_cal(
    dataset_in,
    roi_radius=3,
    cfam_version=0,
    spline_order=3,
    combine_frames=False
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
      spline_order (int): Spline order for sub-pixel centering of the PSF
        stamps. Default 3. Use 1 for linear interpolation.
      combine_frames (boolean): Combine frames of the same FSMX/Y position, 
        pupil images are unaffected.

    Returns:
      PSF cube, data quality cube, HDU list with the CT array measurements,
      including the PSF locations, FPAM/FSAM positions, and corrresponding
      headers. 
    """
    dataset = dataset_in.copy()

    # split dataset by dither if combine_frames is true
    if combine_frames:
        # first take out pupil frames from the dataset
        pupil_frames = []
        img_frames = []
        for frame in dataset:
            if frame.ext_hdr["DPAMNAME"] == "PUPIL":
                pupil_frames.append(frame)
            else:
                img_frames.append(frame)
        # split the non-pupil frames by FSM dither position
        dither_datasets, _ = corgidrp.data.Dataset(img_frames).split_dataset(exthdr_keywords=["FSMX", "FSMY"])
        # median combine the frames in each dither group
        combined_img_frames = []
        # add the pupil dataset back into the datasets split by dither
        dither_datasets.append(corgidrp.data.Dataset(pupil_frames))
        # only combine frames if each dataset actually contains multiple frames, if each individual frame is already
        # unique in dither position then the original input dataset works fine as is
        if len(dither_datasets) < len(dataset):
            for dither_dataset in dither_datasets:
                # grab all frames in the dataset
                data_array = [frame.data for frame in dither_dataset]
                # median combine
                comb = np.nanmedian(data_array, axis=0)
                # create image object
                im = corgidrp.data.Image(comb, pri_hdr=dither_dataset[0].pri_hdr, ext_hdr=dither_dataset[0].ext_hdr)
                im.filename = dither_dataset[-1].filename
                combined_img_frames.append(im)
            dataset = corgidrp.data.Dataset(combined_img_frames)

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

    # Build the PSF cube. Each stamp is centered so the PSF lands on
    # the stamp's central pixel.
    psf_hdu, dq_hdu = generate_psf_cube(dataset, psf_loc_est,
        cfam_name=cfam_list[0], cfam_version=cfam_version,
        spline_order=spline_order)

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
    offaxis = [f for f in dataset_in if f.ext_hdr.get('DPAMNAME') != 'PUPIL'] # ensure FPAM/FSAM values are not from pupil images in the dataset
    fpam_hv = [offaxis[0].ext_hdr['FPAM_H'], offaxis[0].ext_hdr['FPAM_V']]
    fpam_hdr = fits.Header()
    fpam_hdr['COMMENT'] = 'FPAM H and V values during the core throughput observations'
    fpam_hdr['UNITS'] = 'micrometer'
    ct_hdu_list += [fits.ImageHDU(data=fpam_hv, header=fpam_hdr, name='CTFPAM')]
    # Values of FSAM during CT observations (needed to derive the FPM's center
    # during CT observations given a coronagraphic dataset). The values do not
    # change during CT observations
    fsam_hv = [offaxis[0].ext_hdr['FSAM_H'], offaxis[0].ext_hdr['FSAM_V']]
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