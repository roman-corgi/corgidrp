# A file that holds the functions that transmogrify l3 data to l4 data 
import warnings
import numpy as np
import os
import pyklip.rdi
from pyklip.klip import rotate
import scipy.ndimage
from astropy.wcs import WCS

from corgidrp import data
from corgidrp.detector import flag_nans,nan_flags
from corgidrp import star_center
import corgidrp
from corgidrp.klip_fm import meas_klip_thrupt
from corgidrp.corethroughput import get_1d_ct
from scipy.ndimage import rotate as rotate_scipy # to avoid duplicated name
from scipy.ndimage import shift
from astropy.io import fits
from scipy.ndimage import generic_filter
from corgidrp.spec import compute_psf_centroid, create_wave_cal, read_cent_wave
from corgidrp import pol
from corgidrp import fluxcal
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning


def replace_bad_pixels(input_dataset,kernelsize=3,dq_thresh=1):
    """Interpolate over bad pixels in image and error arrays using a median filter.
    TODO: Add additional options for bad pixel replacement (e.g. 2d interpolation that
    can handle nans, constant value, etc.)

    Args:
        input_dataset (corgidrp.data.Dataset): input L3 dataset with bad pixels
        kernelsize (int, optional): Size of median filter window in pixels. Defaults to 3.
        dq_thresh (int, optional): Minimum DQ value for a pixel to be replaced. Defaults to 1.

    Returns:
        corgidrp.data.Dataset: A copy of the dataset with the bad pixels and error values interpolated over. 
        The bad pixel map is unchanged.
    """

    # Copy input dataset
    dataset = input_dataset.copy()
    im_data = dataset.all_data
    im_err = dataset.all_err

    # Get the pixels where the dq array is above the threshold
    im_dq_bool = dataset.all_dq >= dq_thresh
    
    # Set the bad pixels to np.nan
    im_data[im_dq_bool] = np.nan
    
    # Apply the correct dq frame to each error frame
    for f,frame in enumerate(im_err):
        for e,_ in enumerate(frame):
            im_err[f][e] = np.where(im_dq_bool[f], np.nan,im_err[f][e])

    # Interpolate over the bad pixels using nanmedian
    im_filtered = generic_filter(im_data,np.nanmedian,size=kernelsize,axes=[-1,-2])
    err_filtered = generic_filter(im_err,np.nanmedian,size=kernelsize,axes=[-1,-2])
    
    # Replace the bad pixels with the interpolated pixels
    im_replaced = np.where(np.isnan(im_data),im_filtered,im_data)
    err_replaced = np.where(np.isnan(im_err),err_filtered,im_err)
    
    # Update dataset
    bp_count = np.sum(np.isnan(im_data))
    history_msg = f"Interpolated over {bp_count} bad pixels with median filter size {kernelsize}."
    dataset.update_after_processing_step(history_msg,
                                         new_all_data=im_replaced, 
                                         new_all_err=err_replaced)

    return dataset


def distortion_correction(input_dataset, astrom_calibration):
    """ 
    Applies the distortion correction to the dataset. The function assumes bad 
    pixels have been corrected beforehand. Furthermore it also applies the distortion 
    correction to the error maps.
    

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)
        astrom_calibration (corgidrp.data.AstrometricCalibration): an AstrometricCalibration calibration file to model the distortion

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the distortion correction applied
    """
    undistorted_dataset = input_dataset.copy()
    distortion_coeffs = astrom_calibration.distortion_coeffs[:-1]
    distortion_order = int(astrom_calibration.distortion_coeffs[-1])

    undistorted_ims = []
    undistorted_errs = []

    # apply the distortion correction to each image in the dataset
    for undistorted_data in undistorted_dataset:

        im_data = undistorted_data.data
        im_err = undistorted_data.err
        imgsizeX, imgsizeY = im_data.shape

        # set image size to the largest axis if not square imagea
        imgsize = np.max([imgsizeX,imgsizeY])

        yorig, xorig = np.indices(im_data.shape)
        y0, x0 = imgsize//2, imgsize//2
        yorig -= y0
        xorig -= x0

        ### compute the distortion map based on the calibration file passed in
        fitparams = (distortion_order + 1)**2

            # reshape the coefficient arrays
        x_params = distortion_coeffs[:fitparams]
        x_params = x_params.reshape(distortion_order+1, distortion_order+1)

        total_orders = np.arange(distortion_order+1)[:,None] + np.arange(distortion_order+1)[None,:]
        x_params = x_params / 500**(total_orders)

            # evaluate the legendre polynomial at all pixel positions
        x_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), x_params)
        x_corr = x_corr.reshape(xorig.shape)

        distmapX = x_corr - xorig

            # reshape and evaluate the same way for the y coordinates
        y_params = distortion_coeffs[fitparams:]
        y_params = y_params.reshape(distortion_order+1, distortion_order+1)
        y_params = y_params /500**(total_orders)

        y_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), y_params)
        y_corr = y_corr.reshape(yorig.shape)
        distmapY = y_corr - yorig

        # apply the distortion grid to the image indeces and map the image
        gridx, gridy = np.meshgrid(np.arange(imgsize), np.arange(imgsize))
        gridx = gridx - distmapX
        gridy = gridy - distmapY
        
        undistorted_image = scipy.ndimage.map_coordinates(im_data, [gridy, gridx])
        
        # undistort the errors
        if len(im_err.shape) == 2:
            undistorted_errors = scipy.ndimage.map_coordinates(im_err, [gridy, gridx])
        else:
            undistorted_errors = []
            for err in im_err:
                und_err = scipy.ndimage.map_coordinates(err, [gridy, gridx])
                
                undistorted_errors.append(und_err)
        
        undistorted_ims.append(undistorted_image)
        undistorted_errs.append(undistorted_errors)

    history_msg = 'Distortion correction completed'

    undistorted_dataset.update_after_processing_step(history_msg, new_all_data=np.array(undistorted_ims), new_all_err=np.array(undistorted_errs))

    return undistorted_dataset


def find_star(input_dataset,
              star_coordinate_guess=None,
              thetaOffsetGuess=0,
              satellite_spot_parameters=None,
              drop_satspots_frames=True):
    """
    Determines the star position within a coronagraphic dataset by analyzing frames that 
    contain satellite spots (indicated by ``SATSPOTS=1`` in the primary header). The 
    function computes the median of all science frames (``SATSPOTS=0``) and the median 
    of all satellite spot frames (``SATSPOTS=1``), then estimates the star location 
    based on these median images and the initial guess provided.

    The star's (x, y) location is stored in each frame's extension header under 
    ``STARLOCX`` and ``STARLOCY``.

    You can replace many of the default settings for by adjusting the satellite_spot_parameters 
    dictionary. You only need to replace the parameters of interest and the rest will stay as defaults. 

    satellite_spot_parameters of the form: 
         offset : dict
                Parameters for estimating the offset of the star center:

                spotSepPix : float
                    Expected (model-based) separation of the satellite spots from the star.
                    Units: pixels.
                roiRadiusPix : float
                    Radius of the region of interest around each satellite spot.
                    Units: pixels.
                probeRotVecDeg : array_like
                    Angles (degrees CCW from x-axis) specifying the position of satellite spot pairs.
                nSubpixels : int
                    Number of subpixels across for calculating region-of-interest mask edges.
                nSteps : int
                    Number of points in grid search along each direction.
                stepSize : float
                    Step size for the grid search.
                    Units: pixels.
                nIter : int
                    Number of iterations refining the radial separation.

            separation : dict
                Parameters for estimating the separation of satellite spots from the star:

                spotSepPix : float
                    Expected separation between star and satellite spots.
                    Units: pixels.
                roiRadiusPix : float
                    Radius of the region of interest around each satellite spot.
                    Units: pixels.
                probeRotVecDeg : array_like
                    Angles (degrees CCW from x-axis) specifying the position of satellite spot pairs.
                nSubpixels : int
                    Number of subpixels across for calculating region-of-interest mask edges.
                nSteps : int
                    Number of points in grid search along each direction.
                stepSize : float
                    Step size for the grid search.
                    Units: pixels.
                nIter : int
                    Number of iterations refining the radial separation.

    

    Args:
        input_dataset (corgidrp.data.Dataset):
            A dataset of L3-level frames. Frames should be labeled in their primary 
            headers with ``SATSPOTS=0`` (science frames) or ``SATSPOTS=1`` 
            (satellite spot frames).
        star_coordinate_guess (tuple of float or None, optional):
            Initial guess for the star's (x, y) location as absolute coordinates.
            If ``None``, defaults to the center of the median satellite spot image.
            Defaults to None.
        thetaOffsetGuess (float, optional):
            Initial guess for any angular rotation of the star center 
            (in degrees, for example). Defaults to 0.
        satellite_spot_parameters (dict, optional):
            Dictionary containing tuning parameters for spot separation and offset estimation. The dictionary
            can contain the following keys and structure. Only provided parameters will be changed,
            otherwise defaults for the mode will be used:
            If None, default parameters corresponding to the specified observing_mode will be used.     
        drop_satspots_frames (bool, optional):
            If True, frames with satellite spots (``SATSPOTS=1``) will be removed from 
            the returned dataset. Defaults to False.

    Returns:
        corgidrp.data.Dataset:
            The original dataset, augmented with the star's (x, y) location stored in 
            the extension header (``ext_hdr``) of each frame under the keys 
            ``STARLOCX`` and ``STARLOCY``.

    Raises:
        AssertionError:
            If any frames have an invalid ``SATSPOTS`` keyword (not 0 or 1), or if 
            the frames do not all share the same observing mode (as determined by 
            the ``FSMPRFL`` keyword).

    Notes:
        • This function merges the science frames (for reference) and the satellite 
          spot frames (for analysis) by taking a median image of each set.
        • The star center is computed using the median images and the 
          ``star_center.star_center_from_satellite_spots`` routine.
        • Future enhancements may include separate handling of positive vs. negative 
          satellite spot frames once the relevant metadata keywords are defined.
        • This routine can fail, if the guess position is off by more than a few pixel.
          A significantly wrong guess of the angle offset can also lead to failure.
    """

    # Copy input dataset
    dataset = input_dataset.copy()

    satellite_spot_parameters_defaults = star_center.satellite_spot_parameters_defaults

    # Separate the dataset into frames with and without satellite spots
    sci_frames = []
    sat_spot_frames = []

    observing_mode = []

    for frame in dataset.frames:
        if frame.pri_hdr["SATSPOTS"] == 0:
            sci_frames.append(frame)
            observing_mode.append(frame.ext_hdr['FSMPRFL'])
        elif frame.pri_hdr["SATSPOTS"] == 1:
            sat_spot_frames.append(frame)
            observing_mode.append(frame.ext_hdr['FSMPRFL'])
        else:
            raise AssertionError("Input frames do not have a valid SATSPOTS keyword.")

    assert all(mode == observing_mode[0] for mode in observing_mode), \
        "All frames should have the same observing mode."

    observing_mode = observing_mode[0]

    sci_dataset = data.Dataset(sci_frames)
    sat_spot_dataset = data.Dataset(sat_spot_frames)

    # Compute median images
    img_ref = np.median(sci_dataset.all_data, axis=0)
    img_sat_spot = np.median(sat_spot_dataset.all_data, axis=0)

    # Default star_coordinate_guess to center of img_sat_spot if None
    if star_coordinate_guess is None:
        star_coordinate_guess = (img_sat_spot.shape[1] // 2, img_sat_spot.shape[0] // 2)

    tuningParamDict = satellite_spot_parameters_defaults[observing_mode]
    # See if the satellite spot parameters are provided, if not used defaults
    if satellite_spot_parameters is not None:
        tuningParamDict = star_center.update_parameters(tuningParamDict, satellite_spot_parameters)

    # Find star center
    star_xy, list_spots_xy = star_center.star_center_from_satellite_spots(
        img_ref=img_ref,
        img_sat_spot=img_sat_spot,
        star_coordinate_guess=star_coordinate_guess,
        thetaOffsetGuess=thetaOffsetGuess,
        satellite_spot_parameters=tuningParamDict,
    )

    # Add star location to frame headers
    header_entries = {'STARLOCX': star_xy[0], 'STARLOCY': star_xy[1]}

    if drop_satspots_frames:
        dataset = sci_dataset

    history_msg = (
        f"Satellite spots analyzed. Star location at x={star_xy[0]} "
        f"and y={star_xy[1]}."
    )

    dataset.update_after_processing_step(
        history_msg,
        header_entries=header_entries)

    return dataset


def do_psf_subtraction(input_dataset, 
                       ct_calibration=None,
                       reference_star_dataset=None,
                       outdir=None,fileprefix="",
                       measure_klip_thrupt=True,
                       measure_1d_core_thrupt=True,
                       cand_locs=None,
                       kt_seps=None,
                       kt_pas=None,
                       kt_snr=20.,
                       num_processes=None,
                       **klip_kwargs
                       ):
    
    """
    Perform PSF subtraction on the dataset. Optionally using a reference star dataset.
    TODO: 
        Handle/propagate DQ array
        Propagate error correctly
        What info is missing from output dataset headers?
        Add comments to new ext header cards
        Require pyklip output to be centered on 1 pixel. can use the aligned_center kw to do this.
        Make sure psfsub test output data gets saved in a reasonable place
        Update output filename to: CGI_<Last science target VisitID>_<Last science target TimeUTC>_L<>.fits
        
    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)
        ct_calibration (corgidrp.data.CoreThroughputCalibration, optional): core throughput calibration object. Required 
            if measuring KLIP throughput or 1D core throughput. Defaults to None.
        reference_star_dataset (corgidrp.data.Dataset, optional): a dataset of Images of the reference 
            star. If not provided, references will be searched for in the input dataset.
        outdir (str or path, optional): path to output directory. Defaults to "KLIP_SUB".
        fileprefix (str, optional): prefix of saved output files. Defaults to "".
        measure_klip_thrupt (bool, optional): Whether to measure KLIP throughput via injection-recovery. Separations 
            and throughput levels for each separation and KL mode are saved in Dataset[0].hdu_list['KL_THRU'].
            Defaults to True.
        measure_1d_core_thrupt (bool, optional): Whether to measure the core throughput as a function of separation. 
            Separations and throughput levels for each separation are saved in Dataset[0].hdu_list['CT_THRU'].
            Defaults to True.
        cand_locs (list of tuples, optional): Locations of known off-axis sources, so we don't inject a fake 
            PSF too close to them. This is a list of tuples (sep_pix,pa_degrees) for each source. Defaults to [].
        kt_seps (np.array, optional): Separations (in pixels from the star center) at which to inject fake 
            PSFs for KLIP throughput calibration. If not provided, a linear spacing of separations between the IWA & OWA 
            will be chosen.
        kt_pas (np.array, optional): Position angles (in degrees counterclockwise from north/up) at which to inject fake 
            PSFs at each separation for KLIP throughput calibration. Defaults to [0.,90.,180.,270.].
        kt_snr (float, optional): SNR of fake signals to inject during KLIP throughput calibration. Defaults to 20.
        num_processes (int): number of processes for parallelizing the PSF subtraction
        klip_kwargs: Additional keyword arguments to be passed to pyKLIP fm.klip_dataset, as defined `here <https://pyklip.readthedocs.io/en/latest/pyklip.html#pyklip.fm.klip_dataset>`. 
            'mode', e.g. ADI/RDI/ADI+RDI, is chosen autonomously if not specified. 'annuli' defaults to 1. 'annuli_spacing' 
            defaults to 'constant'. 'subsections' defaults to 1. 'movement' defaults to 1. 'numbasis' defaults to [1,4,8,16].

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the PSF subtraction applied (L4-level)

    """

    sci_dataset = input_dataset.copy()
    
    # Need CT calibration object to measure KLIP throughput and 1D core throughput
    if measure_klip_thrupt or measure_1d_core_thrupt:
        assert ct_calibration != None

    # Use input reference dataset if provided
    if not reference_star_dataset is None:
        ref_dataset = reference_star_dataset.copy()

    # Try getting PSF references via the "PSFREF" header kw
    else:
        split_datasets, unique_vals = sci_dataset.split_dataset(prihdr_keywords=["PSFREF"])
        unique_vals = np.array(unique_vals)

        if 0. in unique_vals:
            sci_dataset = split_datasets[int(np.nonzero(np.array(unique_vals) == 0)[0].item())]
        else:
            raise UserWarning('No science files found in input dataset.')

        if 1. in unique_vals:
            ref_dataset = split_datasets[int(np.nonzero(np.array(unique_vals) == 1)[0].item())]
        else:
            ref_dataset = None

    assert len(sci_dataset) > 0, "Science dataset has no data."

    if 'mode' not in klip_kwargs.keys():
        # Choose PSF subtraction mode if unspecified
        if not ref_dataset is None and len(sci_dataset)==1:
            klip_kwargs['mode'] = 'RDI' 
        elif not ref_dataset is None:
            klip_kwargs['mode'] = 'ADI+RDI'
        else:
            klip_kwargs['mode'] = 'ADI' 
    else: assert klip_kwargs['mode'] in ['RDI','ADI+RDI','ADI'], f"Mode {klip_kwargs['mode']} is not configured."

    if 'numbasis' not in klip_kwargs.keys():
        klip_kwargs['numbasis'] = [1,4,8,16]
    elif isinstance(klip_kwargs['numbasis'],int):
        klip_kwargs['numbasis'] = [klip_kwargs['numbasis']]

    if 'annuli' not in klip_kwargs.keys():
        klip_kwargs['annuli'] = 1

    if 'annuli_spacing' not in klip_kwargs.keys():
        klip_kwargs['annuli_spacing'] = 'constant'

    if 'subsections' not in klip_kwargs.keys():
        klip_kwargs['subsections'] = 1

    if 'movement' not in klip_kwargs.keys():
        klip_kwargs['movement'] = 1
    
    # Set up outdir
    if outdir is None: 
        outdir = os.path.join(corgidrp.config_folder, 'KLIP_SUB')
    
    outdir = os.path.join(outdir,klip_kwargs['mode'])
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Mask data where DQ > 0, let pyklip deal with the nans
    sci_dataset_masked = nan_flags(sci_dataset)
    ref_dataset_masked = None if ref_dataset is None else nan_flags(ref_dataset)

    # Initialize pyklip dataset class
    pyklip_dataset = data.PyKLIPDataset(sci_dataset_masked,psflib_dataset=ref_dataset_masked)
    
    # Run pyklip
    pyklip.parallelized.klip_dataset(pyklip_dataset, outputdir=outdir,
                            **klip_kwargs,
                            calibrate_flux=False,psf_library=pyklip_dataset._psflib,
                            fileprefix=fileprefix, numthreads=num_processes)
    
    # Construct corgiDRP dataset from pyKLIP result
    result_fpath = os.path.join(outdir,f'{fileprefix}-KLmodes-all.fits')   
    pyklip_data = fits.getdata(result_fpath)
    pyklip_hdr = fits.getheader(result_fpath)

    err = np.zeros([1,*pyklip_data.shape])
    dq = np.zeros_like(pyklip_data) # This will get filled out later

    # Collapse sci_dataset headers
    pri_hdr = sci_dataset[0].pri_hdr.copy()
    ext_hdr = sci_dataset[0].ext_hdr.copy()    
    
    # Add relevant info from the pyklip headers:
    skip_kws = ['PSFCENTX','PSFCENTY','CREATOR','CTYPE3']
    for kw, val, comment in pyklip_hdr._cards:
        if not kw in skip_kws:
            ext_hdr.set(kw,val,comment)

    # Record KLIP algorithm explicitly
    pri_hdr.set('KLIP_ALG',klip_kwargs['mode'])
    
    # Add info from pyklip to ext_hdr
    ext_hdr['STARLOCX'] = pyklip_hdr['PSFCENTX']
    ext_hdr['STARLOCY'] = pyklip_hdr['PSFCENTY']

    if "HISTORY" in sci_dataset[0].ext_hdr.keys():
        history_str = str(sci_dataset[0].ext_hdr['HISTORY'])
        ext_hdr['HISTORY'] = ''.join(history_str.split('\n'))
    
    # Construct Image and Dataset object
    frame = data.Image(pyklip_data,
                        pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                        err=err, dq=dq)
    # NOTE: product of psfsubtraction should take: CGI_<Last science target VisitID>_<Last science target TimeUTC>_L<>.fits
    # upgrade to L4 should be done by a serpate receipe
    frame.filename = sci_dataset.frames[-1].filename
    
    dataset_out = data.Dataset([frame])

    # Flag nans in the dq array and then add nans to the error array
    dataset_out = flag_nans(dataset_out,flag_val=1)
    dataset_out = nan_flags(dataset_out,threshold=1)
    
    history_msg = f'PSF subtracted via pyKLIP {klip_kwargs["mode"]}.'
    dataset_out.update_after_processing_step(history_msg)
    
    if measure_klip_thrupt:
        
        # Determine flux of objects to inject (units?)

        # Use same KLIP parameters
        klip_params = klip_kwargs.copy()
        klip_params['outdir'] = outdir
        klip_params['fileprefix'] = fileprefix,
        
        if cand_locs is None:
            cand_locs = []

        klip_thpt = meas_klip_thrupt(sci_dataset_masked,ref_dataset_masked, # pre-psf-subtracted dataset
                            dataset_out,
                            ct_calibration,
                            klip_params,
                            kt_snr,
                            cand_locs = cand_locs, # list of (sep_pix,pa_deg) of known off axis source locations
                            seps=kt_seps,
                            pas=kt_pas,
                            num_processes=num_processes
                            )
        thrupt_hdr = fits.Header()
        # Core throughput values on EXCAM wrt pixel (0,0) (not a "CT map", which is
        # wrt FPM's center 
        # thrupt_hdr['COMMENT'] = ('KLIP Throughput and retrieved FWHM as a function of separation for each KLMode '
        #                         '(r, KL1, KL2, ...) = (data[0], data[1], data[2]). The last axis contains the'
        #                         'KL throughput in the 0th index and the FWHM in the 1st index')
        thrupt_hdr['COMMENT'] = ('array of shape (N,n_seps,2), where N is 1 + the number of KL mode truncation choices and n_seps '
            'is the number of separations (in pixels from the star center) sampled. Index 0 contains the separations sampled (twice, to fill up the last axis of dimension 2), and each following index '
            'contains the dimensionless KLIP throughput and FWHM in pixels measured at each separation for each KL mode '
            'truncation choice. An example for 4 KL mode truncation choices, using r1 and r2 for separations and n_seps=2: '
            '[ [[r1,r1],[r2,r2]], '
            '[[KL_thpt_r1_KL1, FWHM_r1_KL1],[KL_thpt_r2_KL1, FWHM_r2_KL1]], '
            '[[KL_thpt_r1_KL2, FWHM_r1_KL2],[KL_thpt_r2_KL2, FWHM_r2_KL2]], '
            '[[KL_thpt_r1_KL3, FWHM_r1_KL3],[KL_thpt_r2_KL3, FWHM_r2_KL3]], '
            '[[KL_thpt_r1_KL4, FWHM_r1_KL4],[KL_thpt_r2_KL4, FWHM_r2_KL4]] ]')
        thrupt_hdr['UNITS'] = 'Separation: EXCAM pixels. KLIP throughput: values between 0 and 1. FWHM: EXCAM pixels'
        thrupt_hdu_list = [fits.ImageHDU(data=klip_thpt, header=thrupt_hdr, name='KL_THRU')]
        
        dataset_out[0].hdu_list.extend(thrupt_hdu_list)
    
        # Save throughput as an extension on the psf-subtracted Image

        # Add history msg
        history_msg = f'KLIP throughput measured and saved to Image class HDU List extension "KL_THRU".'
        dataset_out.update_after_processing_step(history_msg)

    if measure_1d_core_thrupt:
        
        # Use the same separations as for KLIP throughput
        if measure_klip_thrupt:
            seps = dataset_out[0].hdu_list['KL_THRU'].data[0,:,0]
        else:
            seps = np.array([5.,10.,15.,20.,25.,30.,35.])

        ct_1d = get_1d_ct(ct_calibration,dataset_out[0],seps)

        ct_hdr = fits.Header()
        # Core throughput values on EXCAM wrt pixel (0,0) (not a "CT map", which is
        # wrt FPM's center 
        ct_hdr['COMMENT'] = ('KLIP Throughput as a function of separation for each KLMode '
                                '(r, KL1, KL2, ...) = (data[0], data[1], data[2])')
        ct_hdr['UNITS'] = 'Separation: EXCAM pixels. CT throughput: values between 0 and 1.'
        ct_hdu_list = [fits.ImageHDU(data=ct_1d, header=ct_hdr, name='CT_THRU')]
        
        dataset_out[0].hdu_list.extend(ct_hdu_list)
        # Save throughput as an extension on the psf-subtracted Image

        # Add history msg
        history_msg = f'1D CT throughput measured and saved to Image class HDU List extension "CT_THRU".'
        dataset_out.update_after_processing_step(history_msg)
            
    return dataset_out


def northup(input_dataset,use_wcs=True,rot_center='im_center'):
    """
    Derotate the Image, ERR, and DQ data by the angle offset to make the FoV up to North. 
    The northup function looks for 'STARLOCX' and 'STARLOCY' for the star location. If not, it uses the center of the FoV as the star location.
    With use_wcs=True it uses WCS infomation to calculate the north position angle, or use just 'ROLL' header keyword if use_wcs is False (not recommended).
  
    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level) - now handles pol datasets shapes
        use_wcs: if you want to use WCS to correct the north position angle, set True (default). 
	    rot_center: 'im_center', 'starloc', or manual coordinate (x,y). 'im_center' uses the center of the image. 'starloc' refers to 'STARLOCX' and 'STARLOCY' in the header. 

    Returns:
        corgidrp.data.Dataset: North is up, East is left
    
    """
    # make a copy 
    processed_dataset = input_dataset.copy()
    new_all_data = []; new_all_err = []; new_all_dq = []
    for processed_data in processed_dataset:
        ## image extension ##
        sci_hd = processed_data.ext_hdr
        sci_data = processed_data.data
        # See if it's pol data (each array is 3D since has two pol modes)
        is_pol = sci_data.ndim == 3
        if is_pol:
            num_pols, ylen, xlen = sci_data.shape # set number of pol modes (nominally 2)
        else:
            ylen, xlen = sci_data.shape # for regular data
        # define the center for rotation
        if rot_center == 'im_center':
            xcen, ycen = [(xlen-1) // 2, (ylen-1) // 2]
        elif rot_center == 'starloc':
            try:
                xcen, ycen = sci_hd['STARLOCX'], sci_hd['STARLOCY'] 
            except KeyError:
                warnings.warn('"STARLOCX/Y" missing from ext_hdr. Rotating about center of array.')
                xcen, ycen = [(xlen-1) // 2, (ylen-1) // 2]
        else:
            xcen = rot_center[0]
            ycen = rot_center[1]

            # look for WCS solutions
        if use_wcs is True:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)
                astr_hdr = WCS(sci_hd)
            CD1_2 = sci_hd['CD1_2']
            CD2_2 = sci_hd['CD2_2']
            roll_angle = -np.rad2deg(np.arctan2(-CD1_2, CD2_2))  # Compute North Position Angle from the WCS solutions
        else:
            warnings.warn('use "ROLL" instead of WCS to estimate the north position angle')
            astr_hdr = None
            # read the roll angle parameter, assuming this info is recorded in the primary header as requested
            roll_angle = processed_data.pri_hdr['ROLL']

        # derotate - handle 2D or 3D
        if is_pol:
            # one header applies to all pols. Have to rotate both pol images, but header only gets updated once in loop
            sci_derot = np.stack([rotate(sci_data[i], roll_angle, (xcen, ycen), astr_hdr=astr_hdr if i == 0 else None)
                                  for i in range(num_pols)])
        else:
            sci_derot = rotate(sci_data, roll_angle, (xcen, ycen), astr_hdr=astr_hdr)  # astr_hdr is corrected at above lines
        new_all_data.append(sci_derot)
        log = f'FoV rotated by {roll_angle}deg counterclockwise at a roll center {xcen, ycen}'
        sci_hd['HISTORY'] = log

        # update WCS solutions
        if use_wcs:
            sci_hd['CD1_1'] = astr_hdr.wcs.cd[0, 0]
            sci_hd['CD1_2'] = astr_hdr.wcs.cd[0, 1]
            sci_hd['CD2_1'] = astr_hdr.wcs.cd[1, 0]
            sci_hd['CD2_2'] = astr_hdr.wcs.cd[1, 1]
        #############
        ## HDU ERR ##
        err_data = processed_data.err
        if is_pol:
            # err has shape (num_err_layers, num_pols, ylen, xlen)
            err_derot = np.stack([rotate(err_data[0, i], roll_angle, (xcen, ycen))
                                  for i in range(num_pols)])
            err_derot = np.expand_dims(err_derot, axis=0)  # Add back error layer dimension
        else:
            err_derot = np.expand_dims(rotate(err_data[0], roll_angle, (xcen, ycen)), axis=0)
        new_all_err.append(err_derot)

        #############
        ## HDU DQ ##
        dq_data = processed_data.dq

        # pol version has to loop through both images
        if is_pol:

            dq_derot_planes = []
            for plane_idx in range(num_pols):
                dq_pol = dq_data[plane_idx]
                if xcen != xlen / 2 or ycen != ylen / 2:
                    # padding, shifting (rot center to image center), rotating, re-shift (image center to rot center), and cropping
                    xshift = xcen - xlen / 2;
                    yshift = ycen - ylen / 2
                    pad_x = int(np.ceil(abs(xshift)));
                    pad_y = int(np.ceil(abs(yshift)))
                    dq_pol_padded = np.pad(dq_pol, pad_width=((pad_y, pad_y), (pad_x, pad_x)), mode='constant',
                                             constant_values=0)
                    dq_pol_padded_shifted = shift(dq_pol_padded, (-yshift, -xshift), order=0, mode='constant',
                                                    cval=0)
                    crop_x = slice(pad_x, pad_x + xlen);
                    crop_y = slice(pad_y, pad_y + ylen)
                    dq_pol_derot = shift(
                        rotate_scipy(dq_pol_padded_shifted, -roll_angle, order=0, mode='constant', reshape=False,
                                     cval=0), (yshift, xshift), order=0, mode='constant', cval=0)[crop_y, crop_x]
                else:
                    dq_pol_derot = rotate_scipy(dq_pol, -roll_angle, order=0, mode='constant', reshape=False,
                                                  cval=0)
                dq_derot_planes.append(dq_pol_derot)

            dq_derot = np.stack(dq_derot_planes)

        else: #non pol version
            if xcen != xlen / 2 or ycen != ylen / 2:
                # padding, shifting (rot center to image center), rotating, re-shift (image center to rot center), and cropping
                # calculate shift values
                xshift = xcen - xlen / 2;
                yshift = ycen - ylen / 2

                # pad and shift
                pad_x = int(np.ceil(abs(xshift)));
                pad_y = int(np.ceil(abs(yshift)))
                dq_data_padded = np.pad(dq_data, pad_width=((pad_y, pad_y), (pad_x, pad_x)), mode='constant',
                                        constant_values=0)
                dq_data_padded_shifted = shift(dq_data_padded, (-yshift, -xshift), order=0, mode='constant', cval=0)
                # define slices for cropping
                crop_x = slice(pad_x,pad_x+xlen); crop_y = slice(pad_y,pad_y+ylen)

                # rotate (invserse direction to pyklip.rotate), re-shift, and crop
                dq_derot = shift(
                    rotate_scipy(dq_data_padded_shifted, -roll_angle, order=0, mode='constant', reshape=False,
                                 cval=0), \
                    (yshift, xshift), order=0, mode='constant', cval=0)[crop_y, crop_x]
            else:
                # simply rotate
                dq_derot = rotate_scipy(dq_data, -roll_angle, order=0, mode='constant', reshape=False, cval=0)

        new_all_dq.append(dq_derot)
        ############
    history_msg = 'North is Up and East is Left'
    processed_dataset.update_after_processing_step(history_msg, new_all_data=np.array(new_all_data),
                                                   new_all_err=np.array(new_all_err), new_all_dq=np.array(new_all_dq))

    return processed_dataset


def determine_wave_zeropoint(input_dataset, template_dataset = None, xcent_guess = None, ycent_guess = None, bb_nb_dx = None, bb_nb_dy = None, return_all = False):
    """ 
    A procedure for estimating the centroid of the zero-point image
    (satellite spot or PSF) taken through the narrowband filter (2C or 3D) and slit.

    Args:
        input_dataset (corgidrp.data.Dataset): Dataset containing 2D PSF or satellite spot images taken through the narrowband filter and slit.
        template_dataset (corgidrp.data.Dataset): dataset of the template PSF, if None, a simulated PSF from the data/spectroscopy/template 
                                                  path is taken
        xcent_guess (float): initial x guess for the centroid fit for all frames
        ycent_guess (float): initial y guess for the centroid fit for all frames
        bb_nb_dx (float): horizontal image offset between the narrowband and broadband filters, in EXCAM pixels. 
                          This will override the offset in the existing lookup table. 
        bb_nb_dy (float): vertical image offset between the narrowband and broadband filters, in EXCAM pixels. 
                          This will override the offset in the existing lookup table. 
        return_all (boolean): if false (default) returns only the broad band science frames, if true it returns all (including narrow band) frames
    
    Returns:
        corgidrp.data.Dataset: the returned science dataset without the satellite spots images and the wavelength zeropoint 
                               information as header keywords, which is WAVLEN0, WV0_X, WV0_XERR, WV0_Y, WV0_YERR, WV0_DIMX, WV0_DIMY
    """
    dataset = input_dataset.copy()
    dpamname = dataset.frames[0].ext_hdr["DPAMNAME"]
    if not dpamname.startswith("PRISM"):
        raise AttributeError("This is not a spectroscopic observation. but {0}").format(dpamname)
    slit = dataset.frames[0].ext_hdr['FSAMNAME']
    if not slit.startswith("R"):
        raise AttributeError("not a slit observation")
    # Assumed that only narrowband filter (includes sat spots) frames are taken to fit the zeropoint
    narrow_dataset, band = dataset.split_dataset(exthdr_keywords=["CFAMNAME"])
    band = np.array([s.upper() for s in band])
    if len(band) < 2:
        raise AttributeError("there needs to be at least 1 narrowband and 1 science band prism frame in the dataset\
                             to determine the wavelength zero point")
        
    if "3D" in band:
        sat_dataset = narrow_dataset[int(np.nonzero(band == "3D")[0].item())]
        sci_dataset = narrow_dataset[int(np.nonzero(band != "3D")[0].item())]
    elif "2C" in band:
        sat_dataset = narrow_dataset[int(np.nonzero(band == "2C")[0].item())]
        sci_dataset = narrow_dataset[int(np.nonzero(band != "2C")[0].item())]
    else:
        raise AttributeError("No narrowband frames found in input dataset")
    
    if len(sci_dataset) == 0:
        raise AttributeError("No science frames found in input dataset")
    
    if xcent_guess is not None and ycent_guess is not None:
        n = len(sat_dataset)
        initial_cent = {"xcent": np.repeat(xcent_guess, n),
                        "ycent": np.repeat(ycent_guess, n)}
    else:
        initial_cent = None
    spot_centroids = compute_psf_centroid(dataset = sat_dataset, template_dataset = template_dataset, initial_cent = initial_cent)
    
    nb_filter = sat_dataset[0].ext_hdr["CFAMNAME"]
    bb_filter = nb_filter[0]
    cen_wave, _, xoff_nb, yoff_nb = read_cent_wave(nb_filter)
    _, _, xoff_bb, yoff_bb = read_cent_wave(bb_filter)
    # Correct the centroid for the filter-to-filter image offset, so that
    # the coordinates (x0,y0) correspond to the wavelength location in the broadband filter. 
    if bb_nb_dx is not None and bb_nb_dy is not None:
        x0 = np.mean(spot_centroids.xfit) + bb_nb_dx
        y0 = np.mean(spot_centroids.yfit) + bb_nb_dy
    else:
        x0 = np.mean(spot_centroids.xfit) + (xoff_bb - xoff_nb)
        y0 = np.mean(spot_centroids.yfit) + (yoff_bb - yoff_nb)
    x0err = np.sqrt(np.sum(spot_centroids.xfit_err**2)/len(spot_centroids.xfit_err))
    y0err = np.sqrt(np.sum(spot_centroids.yfit_err**2)/len(spot_centroids.yfit_err))
    if return_all:
        sci_dataset = dataset

    for frame in sci_dataset:
        frame.ext_hdr["WAVLEN0"] = cen_wave
        frame.ext_hdr["WV0_X"] = x0
        frame.ext_hdr["WV0_XERR"] = x0err
        frame.ext_hdr["WV0_Y"] = y0
        frame.ext_hdr["WV0_YERR"] = y0err
        frame.ext_hdr["WV0_DIMX"] = sat_dataset[0].ext_hdr['NAXIS1']
        frame.ext_hdr["WV0_DIMY"] = sat_dataset[0].ext_hdr['NAXIS2']
                              
    history_msg = "wavelength zeropoint values added to header"
    sci_dataset.update_after_processing_step(history_msg)
    return sci_dataset


def add_wavelength_map(input_dataset, disp_model, pixel_pitch_um = 13.0, ntrials = 1000):
    """
    add_wavelength_map adds the wavelength map + error and the position lookup table as extensions to the frames
    
    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of spectroscopy Images (L3-level)
        disp_model (corgidrp.data.DispersionModel): dispersion model of the corresponding band
        pixel_pitch_um (float): EXCAM pixel pitch in microns, default: 13.0
        ntrials (int): number of trials when applying a Monte Carlo error propagation to estimate the uncertainties of the
                       values in the wavelength calibration map

    Returns:
        corgidrp.data.Dataset: dataset with appended wavelength map and error
    """
    dataset = input_dataset.copy()
    
    for frames in dataset:
        #get the corgidrp.data.Dataset:wavelength zeropoint information from the input science frames header
        head = frames.ext_hdr
        wave_zero = {
        'wavlen': head['WAVLEN0'],
        'x' : head['WV0_X'],
        'xerr': head['WV0_XERR'],
        'y': head['WV0_Y'],
        'yerr': head['WV0_YERR'],
        'shapex': head['WV0_DIMX'],
        'shapey': head['WV0_DIMY']
        }
    
        wave_map, wave_err, pos_lookup, x_refwav, y_refwav = create_wave_cal(disp_model, wave_zero, pixel_pitch_um = pixel_pitch_um, ntrials = ntrials)
        wave_hdr = fits.Header()
        wave_hdr["BUNIT"] = "nm"
        wave_hdr["REFWAVE"] = disp_model.ext_hdr["REFWAVE"]
        wave_hdr["XREFWAV"] = x_refwav
        wave_hdr["YREFWAV"] = y_refwav
        wave_err_hdr = fits.Header()
        wave_err_hdr["BUNIT"] = "nm"
        frames.add_extension_hdu("WAVE" ,data = wave_map, header = wave_hdr)
        frames.add_extension_hdu("WAVE_ERR", data = wave_err, header = wave_err_hdr)
        pos_hdu = fits.BinTableHDU(data = pos_lookup, header = fits.Header(), name = "POSLOOKUP")
        frames.hdu_list.append(pos_hdu.copy())
        frames.hdu_names.append("POSLOOKUP")
    
    history_msg = "wavelength map and position lookup table extension added"
    dataset.update_after_processing_step(history_msg)
    return dataset

def extract_spec(input_dataset, halfwidth = 2, halfheight = 9, apply_weights = False):
    """
    extract an optionally error weighted 1D - spectrum and wavelength information of a point source from a box around 
    the wavelength zero point with units photoelectron/s/bin.
    
    Args:
        input_dataset (corgidrp.data.Dataset): 
        halfwidth (int): The width of the fitting region is 2 * halfwidth + 1 pixels across dispersion
        halfheight (int): The height of the fitting region is 2 * halfheight + 1 pixels along dispersion.
        apply_weights (boolean): if true a weighted sum is calculated using 1/error^2 as weights.
        
    Returns:
        corgidrp.data.Dataset: dataset containing the spectral 1D data, error and corresponding wavelengths
    """
    dataset = input_dataset.copy()
    
    for image in dataset:
        xcent_round, ycent_round = (int(np.rint(image.ext_hdr["WV0_X"])), int(np.rint(image.ext_hdr["WV0_Y"])))
        image_cutout = image.data[ycent_round - halfheight:ycent_round + halfheight + 1,
                                  xcent_round - halfwidth:xcent_round + halfwidth + 1]
        dq_cutout = image.dq[ycent_round - halfheight:ycent_round + halfheight + 1,
                                  xcent_round - halfwidth:xcent_round + halfwidth + 1]
        wave_cal_map_cutout = image.hdu_list["WAVE"].data[ycent_round - halfheight:ycent_round + halfheight + 1,
                                                          xcent_round - halfwidth:xcent_round + halfwidth + 1]
        wave_err_cutout = image.hdu_list["WAVE_ERR"].data[ycent_round - halfheight:ycent_round + halfheight + 1,
                                                          xcent_round - halfwidth:xcent_round + halfwidth + 1]
        err_cutout = image.err[:,ycent_round - halfheight:ycent_round + halfheight + 1,
                                  xcent_round - halfwidth:xcent_round + halfwidth + 1]
        bad_ind = np.where(dq_cutout > 0)
        image_cutout[bad_ind] = np.nan
        err_cutout[bad_ind] = np.nan
        wave = np.mean(wave_cal_map_cutout, axis=1)
        wave_err = np.mean(wave_err_cutout, axis=1)
        err = np.sqrt(np.nansum(np.square(err_cutout), axis=2))
        # dq collpase: keep all flags on
        dq_collapse = np.bitwise_or.reduce(dq_cutout, axis=1)
 
        if apply_weights:
            err_cutout[0][err_cutout[0] == 0] = np.nan
            whts = 1./np.square(err_cutout[0])
            spec = np.nansum(image_cutout * whts, axis = 1) / np.nansum (whts, axis = 1) * (2 * halfwidth + 1)
            err[0] = 1./np.sqrt(np.nansum(whts, axis = 1))
            weight_str = "weights applied"
        else:
            spec = np.nansum(image_cutout, axis=1)
            weight_str = "no weights applied"
        image.data = spec
        image.err = err
        image.dq = dq_collapse
        image.hdu_list["WAVE"].data = wave
        image.hdu_list["WAVE_ERR"].data = wave_err
        del(image.hdu_list["POSLOOKUP"])
    history_msg = "spectral extraction within a box of half width of {0}, half height of {1} and with ".format(halfwidth, halfheight) + weight_str
    dataset.update_after_processing_step(history_msg, header_entries={'BUNIT': "photoelectron/s/bin"})
    return dataset

def subtract_stellar_polarization(input_dataset, system_mueller_matrix_cal, nd_mueller_matrix_cal):
    """
    Takes in polarimetric L3 images and their unocculted polarimetric observations,
    computes and subtracts off the stellar polarization component from each image
    TODO: make issue about error propagation, need to check that it is done correctly
          and make changes if necessary to ensure the errors are accurate

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of L3 images, must include unocculted observations
                                               taken with both wollastons at the same roll angle. All frames for the
                                               same target star must have the same x and y dimensions
        system_mueller_matrix_cal (corgidrp.data.MuellerMatrix): mueller matrix calibration of the system without a ND filter
        nd_mueller_matrix_cal (corgidrp.data.MuellerMatrix): mueller matrix calibration of the system with the ND filter used for unocculted observations

    Returns:
        corgidrp.data.Dataset: The input data with stellar polarization removed, excluding the unocculted observations
    """
    
    # check that the data is at the L3 level, and only polarimetric observations are inputted
    dataset = input_dataset.copy()
    for frame in dataset:
        if frame.ext_hdr['DATALVL'] != "L3":
            err_msg = "{0} needs to be L3 data, but it is {1} data instead".format(frame.filename, frame.ext_hdr['DATALVL'])
            raise ValueError(err_msg)
        if frame.ext_hdr['DPAMNAME'] not in ['POL0', 'POL45']:
            raise ValueError("{0} must be a polarimetric observation".format(frame.filename))
        
    # split the dataset by the target star
    split_datasets, unique_vals = dataset.split_dataset(prihdr_keywords=['TARGET'])

    # process each target star
    updated_frames = []
    for target_dataset in split_datasets:
        # split further based on if the observation is unocculted or not, and the wollaston used
        coron_frames = []
        unocculted_pol0_frames = []
        unocculted_pol45_frames = []
        target_name = target_dataset.frames[0].pri_hdr['TARGET']
        for frame in target_dataset:
            if frame.ext_hdr['FPAMNAME'] == 'ND225':
                # unocculted observations, separate by wollaston
                if frame.ext_hdr['DPAMNAME'] == 'POL0':
                    unocculted_pol0_frames.append(frame)
                else:
                    unocculted_pol45_frames.append(frame)
            else:
                # coronagraphic observation
                coron_frames.append(frame)
        
        # make sure input dataset contains unocculted frames taken with both wollastons
        if len(unocculted_pol0_frames) == 0:
            raise ValueError(f"Input dataset must contain unocculted POL0 frame(s) for target {target_name}")
        if len(unocculted_pol45_frames) == 0:
            raise ValueError(f"Input dataset must contain unocculted POL45 frame(s) for target {target_name}")
        
        unocculted_pol0_img = unocculted_pol0_frames[0]
        unocculted_pol45_img = unocculted_pol45_frames[0]

        # construct image for each polarization to pass into aper_phot function in order to obtain flux
        I_0_img = data.Image(unocculted_pol0_img.data[0], 
                             err=unocculted_pol0_img.err[:,0,:,:], 
                             pri_hdr=unocculted_pol0_img.pri_hdr.copy(),
                             ext_hdr=unocculted_pol0_img.ext_hdr.copy())
        I_90_img = data.Image(unocculted_pol0_img.data[1], 
                             err=unocculted_pol0_img.err[:,1,:,:], 
                             pri_hdr=unocculted_pol0_img.pri_hdr.copy(),
                             ext_hdr=unocculted_pol0_img.ext_hdr.copy())
        I_45_img = data.Image(unocculted_pol45_img.data[0], 
                             err=unocculted_pol45_img.err[:,0,:,:], 
                             pri_hdr=unocculted_pol45_img.pri_hdr.copy(),
                             ext_hdr=unocculted_pol45_img.ext_hdr.copy())
        I_135_img = data.Image(unocculted_pol45_img.data[1], 
                             err=unocculted_pol45_img.err[:,1,:,:], 
                             pri_hdr=unocculted_pol45_img.pri_hdr.copy(),
                             ext_hdr=unocculted_pol45_img.ext_hdr.copy())
        # calculate flux
        I_0_flux, I_0_flux_err = fluxcal.aper_phot(I_0_img, encircled_radius=5)
        I_90_flux, I_90_flux_err = fluxcal.aper_phot(I_90_img, encircled_radius=5)
        I_45_flux, I_45_flux_err = fluxcal.aper_phot(I_45_img, encircled_radius=5)
        I_135_flux, I_135_flux_err = fluxcal.aper_phot(I_135_img, encircled_radius=5)
        
        ## construct I, Q, U components after instrument with ND filter
        # I = I_0 +I_90
        I_nd = I_0_flux + I_90_flux
        # Q = I_0 - I_90
        Q_nd = I_0_flux - I_90_flux
        # U = I_45 - I_135
        U_nd = I_45_flux - I_135_flux
        # assume V is basically 0
        V_nd = 0
        # construct stokes vector after instrument with ND filter
        S_nd = [I_nd, Q_nd, U_nd, V_nd]

        # S_nd = M_nd * R(roll_angle) * S_in
        # invert M_nd * R(roll_angle) to recover S_in
        roll_angle = unocculted_pol0_img.pri_hdr['ROLL']
        total_system_mm_nd = nd_mueller_matrix_cal.data @ pol.rotation_mueller_matrix(roll_angle)
        system_nd_inv = np.linalg.pinv(total_system_mm_nd)
        S_in = system_nd_inv @ S_nd

        # propagate errors to find uncertainty of S_in
        I_nd_var = I_0_flux_err**2 + I_90_flux_err**2
        Q_nd_var = I_nd_var
        U_nd_var = I_45_flux_err**2 + I_135_flux_err**2
        v_nd_var = 0
        # construct covariance matrix for S_nd
        C_nd = np.array([[I_nd_var, 0, 0, 0],
                         [0, Q_nd_var, 0, 0],
                         [0, 0, U_nd_var, 0],
                         [0, 0, 0, v_nd_var]])
        # solve for covariance matrix of input stokes vector
        # C_in = pinv(M) * C_nd * pinv(M)^T
        #TODO: incoporate the error terms of the nd mueller matrix into this calculation if necessary 
        C_in = system_nd_inv @ C_nd @ system_nd_inv.T
        # contract back to just the variance
        S_in_var = np.array([
            C_in[0,0],
            C_in[1,1],
            C_in[2,2],
            C_in[3,3]
        ])
        S_in_err = np.sqrt(S_in_var)

        # subtract stellar polarization from the rest of the frames
        for frame in coron_frames:
            # propagate S_in back through the non-ND system mueller matrix to calculate star polarization as observed with coronagraph mask
            frame_roll_angle = frame.pri_hdr['ROLL']
            total_system_mm = system_mueller_matrix_cal.data @ pol.rotation_mueller_matrix(frame_roll_angle)
            S_out = total_system_mm @ S_in
            # construct I0, I45, I90, and I135 back from stokes vector
            I_0_star = (S_out[0] + S_out[1]) / 2
            I_90_star = (S_out[0] - S_out[1]) / 2
            I_45_star = (S_out[0] + S_out[2]) / 2
            I_135_star = (S_out[0] - S_out[2]) / 2

            # propagate errors back to the new intensity terms for the unocculted star, assuming independence
            # σS_out^2 = (σM^2)(I_in^2) + (M^2)(σI_in^2)
            #TODO: double check if this is valid/invalid, change if necessary
            system_mm_var = (system_mueller_matrix_cal.err[0])**2
            system_mm_sq = (system_mueller_matrix_cal.data)**2
            S_in_sq = S_in**2
            S_out_var = (system_mm_var @ S_in_sq) + (system_mm_sq @ S_in_var)
            I_0_star_err = np.sqrt(S_out_var[0] + S_out_var[1]) / 2
            I_90_star_err = I_0_star_err
            I_45_star_err = np.sqrt(S_out_var[0] + S_out_var[2]) / 2
            I_135_star_err = I_45_star_err

            with warnings.catch_warnings():
                # catch divide by zero warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                # calculate normalized difference for the specific wollaston
                if frame.ext_hdr['DPAMNAME'] == 'POL0':
                    normalized_diff = (I_0_star - I_90_star) / (I_0_star + I_90_star)
                    # error
                    normalized_diff_err = normalized_diff * np.sqrt(
                        (np.sqrt(I_0_star_err**2 + I_90_star_err**2) / (I_0_star - I_90_star))**2 +
                        (np.sqrt(I_0_star_err**2 + I_90_star_err**2) / (I_0_star + I_90_star))**2
                    )
                else:
                    normalized_diff = (I_45_star - I_135_star) / (I_45_star + I_135_star)
                    # error
                    normalized_diff_err = normalized_diff * np.sqrt(
                        (np.sqrt(I_45_star_err**2 + I_135_star_err**2) / (I_45_star - I_135_star))**2 +
                        (np.sqrt(I_45_star_err**2 + I_135_star_err**2) / (I_45_star + I_135_star))**2
                    )
            # subtract
            sum = frame.data[0] + frame.data[1]
            diff = frame.data[0] - frame.data[1]
            diff -= sum * normalized_diff
            frame.data[0] = (sum + diff) / 2
            frame.data[1] = (sum - diff) / 2

            # propagate errors for the subtraction
            sum_err = np.sqrt(frame.err[0,0,:,:]**2 + frame.err[0,1,:,:]**2)
            diff_err = sum_err
            diff_err = np.sqrt(diff_err**2 + 
                               (sum * normalized_diff * np.sqrt((sum_err/sum)**2 + (normalized_diff_err/normalized_diff)**2))**2
                        )
            frame.err[0,0,:,:] = np.sqrt(sum_err**2 + diff_err**2) / 2
            frame.err[0,1,:,:] = frame.err[0,0,:,:]
            
            updated_frames.append(frame)

    updated_dataset = data.Dataset(updated_frames)
    history_msg = f"Subtracted Apparent Stellar Polarization, stellar Q value: {S_in[1]}, stellar Q err: {S_in_err[1]}, \
    stellar U value: {S_in[2]}, stellar U err: {S_in_err[2]}."
    updated_dataset.update_after_processing_step(history_msg)
    return updated_dataset

def combine_polarization_states(input_dataset,
                                system_mueller_matrix_cal,
                                ct_calibration=None,
                                reference_star_dataset=None,
                                measure_klip_thrupt=True,
                                measure_1d_core_thrupt=True,
                                cand_locs=None,
                                kt_seps=None,
                                kt_pas=None,
                                kt_snr=20.,
                                num_processes=None,
                                **klip_kwargs):
    """
    Takes in a L3 polarimetric dataset, performs PSF subtraction on total intensity, calculates
    and returns the on-sky Stokes datacube

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of polarimetric Images (L3-level), should be of the same size and same target
         system_mueller_matrix_cal (corgidrp.data.MuellerMatrix): mueller matrix calibration of the instrument
        ct_calibration (corgidrp.data.CoreThroughputCalibration, optional): For PSF Subtraction. core throughput calibration object. Required 
            if measuring KLIP throughput or 1D core throughput. Defaults to None.
        reference_star_dataset (corgidrp.data.Dataset, optional): For PSF Subtraction. a dataset of Images of the reference 
            star. If not provided, references will be searched for in the input dataset.
        measure_klip_thrupt (bool, optional): For PSF Subtraction. Whether to measure KLIP throughput via injection-recovery. Separations 
            and throughput levels for each separation and KL mode are saved in Dataset[0].hdu_list['KL_THRU']. 
            Defaults to True.
        measure_1d_core_thrupt (bool, optional): For PSF Subtraction. Whether to measure the core throughput as a function of separation. 
            Separations and throughput levels for each separation are saved in Dataset[0].hdu_list['CT_THRU'].
            Defaults to True.
        cand_locs (list of tuples, optional): For PSF Subtraction. Locations of known off-axis sources, so we don't inject a fake 
            PSF too close to them. This is a list of tuples (sep_pix,pa_degrees) for each source. Defaults to [].
        kt_seps (np.array, optional): For PSF Subtraction. Separations (in pixels from the star center) at which to inject fake 
            PSFs for KLIP throughput calibration. If not provided, a linear spacing of separations between the IWA & OWA 
            will be chosen.
        kt_pas (np.array, optional): For PSF Subtraction. Position angles (in degrees counterclockwise from north/up) at which to inject fake 
            PSFs at each separation for KLIP throughput calibration. Defaults to [0.,90.,180.,270.].
        kt_snr (float, optional): For PSF Subtraction. SNR of fake signals to inject during KLIP throughput calibration. Defaults to 20.
        num_processes (int): For PSF Subtraction. number of processes for parallelizing the PSF subtraction
        klip_kwargs: For PSF Subtraction. Additional keyword arguments to be passed to pyKLIP fm.klip_dataset, as defined `here <https://pyklip.readthedocs.io/en/latest/pyklip.html#pyklip.fm.klip_dataset>`. 
            'mode', e.g. ADI/RDI/ADI+RDI, is chosen autonomously if not specified. 'annuli' defaults to 1. 'annuli_spacing' 
            defaults to 'constant'. 'subsections' defaults to 1. 'movement' defaults to 1. 'numbasis' defaults to [1,4,8,16].

    Returns:
        corgidrp.data.Dataset: Dataset consisting of a 4xnxm on-sky Stokes datacube, where the first dimension corresponds to IQUV
            with I being the PSF-subtracted total intensity. 
    """

    dataset = input_dataset.copy()
    # construct total intensity for PSF subtraction
    total_intensity_frames = []
    for frame in dataset:
        # check that the data is at the L3 level, and only polarimetric observations are inputted
        if frame.ext_hdr['DATALVL'] != "L3":
            err_msg = "{0} needs to be L3 data, but it is {1} data instead".format(frame.filename, frame.ext_hdr['DATALVL'])
            raise ValueError(err_msg)
        if frame.ext_hdr['DPAMNAME'] not in ['POL0', 'POL45']:
            raise ValueError("{0} must be a polarimetric observation".format(frame.filename))
        
        # sum orthogonal polarization axis to obtain total intensity
        total_intensity_data = frame.data[0] + frame.data[1]
        # error propagation
        total_intensity_err = np.sqrt(frame.err[:,0,:,:]**2 + frame.err[:,1,:,:]**2)
        # construct image
        total_intensity_img = data.Image(total_intensity_data,
                                         pri_hdr=frame.pri_hdr.copy(),
                                         ext_hdr=frame.ext_hdr.copy(),
                                         err=total_intensity_err)
        total_intensity_frames.append(total_intensity_img)
    total_intensity_dataset = data.Dataset(total_intensity_frames)

    # ensure only one KL basis is used for PSF subtraction, so that only one image is returned
    if 'numbasis' in klip_kwargs:
        if isinstance(klip_kwargs['numbasis'], list) and len(klip_kwargs['numbasis']) > 1:
            # raise error if multiple KL basis is passed in
            raise ValueError('Only one KL basis should be used for PSF subtraction')
    else:
        # set default values is numbasis is not passed into klip_kwargs
        if reference_star_dataset is None:
            # default to one if no reference star dataset is provided
            klip_kwargs['numbasis'] = 1
        else:
            # set to the length of the reference star dataset if one is provided
            klip_kwargs['numbasis'] = len(reference_star_dataset)
    
    # perform PSF subtraction on total intensity
    with warnings.catch_warnings():
        # suppress astropy warnings
        warnings.filterwarnings('ignore', category=VerifyWarning)
        warnings.filterwarnings('ignore', category=FITSFixedWarning)
        psf_subtracted_dataset = do_psf_subtraction(total_intensity_dataset,
                                                    ct_calibration=ct_calibration,
                                                    reference_star_dataset=reference_star_dataset,
                                                    measure_klip_thrupt=measure_klip_thrupt,
                                                    measure_1d_core_thrupt=measure_1d_core_thrupt,
                                                    cand_locs=cand_locs,
                                                    kt_seps=kt_seps,
                                                    kt_pas=kt_pas,
                                                    kt_snr=kt_snr,
                                                    num_processes=num_processes,
                                                    **klip_kwargs)
    psf_subtracted_intensity = psf_subtracted_dataset.frames[0]
    # derotate input polarimetric data to North-up East-left
    with warnings.catch_warnings():
        # suppress astropy warnings
        warnings.filterwarnings('ignore', category=VerifyWarning)
        warnings.filterwarnings('ignore', category=FITSFixedWarning)
        derotated_dataset = northup(dataset)

    # construct polarimetric measurement matrix and output intensity vector
    dataset_size = len(derotated_dataset)
    image_size_y = psf_subtracted_intensity.data.shape[1]
    image_size_x = psf_subtracted_intensity.data.shape[2]
    measurement_matrix = np.zeros(shape=(2 * dataset_size, 4))
    output_intensities = np.zeros(shape = (2 * dataset_size, image_size_y, image_size_x))
    # err propagation
    output_intensities_cov = np.zeros(shape = (2 * dataset_size, 2 * dataset_size, image_size_y, image_size_x))
    # system mueller matrix calibration
    system_mm = system_mueller_matrix_cal.data
    for i in range(dataset_size):
        # fill in output intensity vector
        output_intensities[2*i] = derotated_dataset.frames[i].data[0]
        output_intensities[(2*i)+1] = derotated_dataset.frames[i].data[1]
        # fill in diagonal terms of the cov matrix
        output_intensities_cov[2*i,2*i,:,:] = derotated_dataset.frames[i].err[0,0,:,:]**2
        output_intensities_cov[(2*i)+1,(2*i)+1,:,:] = derotated_dataset.frames[i].err[0,1,:,:]**2
        ## fill in measurement matrix
        # roll angle rotation matrix
        roll = derotated_dataset.frames[i].pri_hdr['ROLL']
        rotation_mm = pol.rotation_mueller_matrix(roll)
        if derotated_dataset.frames[i].ext_hdr['DPAMNAME'] == 'POL0':
            # use correct polarizer mueller matrix depending on wollaston used
            # o and e denotes ordinary and extraordinary axis of the wollaston
            polarizer_mm_o = pol.lin_polarizer_mueller_matrix(0)
            polarizer_mm_e = pol.lin_polarizer_mueller_matrix(90)
        else:
            polarizer_mm_o = pol.lin_polarizer_mueller_matrix(45)
            polarizer_mm_e = pol.lin_polarizer_mueller_matrix(135)
        # construct full mueller matrix with roll angle and wollaston
        total_mm_o = polarizer_mm_o @ system_mm @ rotation_mm
        total_mm_e = polarizer_mm_e @ system_mm @ rotation_mm
        # row at current index of the measurement matrix corresponds to the first row of the full system mueller matrix
        measurement_matrix[2*i] = total_mm_o[0]
        measurement_matrix[(2*i)+1] = total_mm_e[0]
    
    # invert measurement matrix to obtain on-sky Stokes datacube
    # if S is the on-sky Stokes vector, M is the measurement matrix, and I is the output intensity vector
    # compute the measurement matrix pseudoinverse using SVD
    u,s,v=np.linalg.svd(measurement_matrix, full_matrices=False)
    # limit the singular values to improve the conditioning of the inversion
    s[s < 1e-5] = 1e-5
    measurement_matrix_inv = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
    # measurement matrix inverse is of size (i, j) where i is 4 and j is the number of output intensities gathered from the input dataset
    # output intensity vector is of size (j, y, x), where y and x are spatial coordinates
    # calling einsum with the input 'ij,jyx->iyx' computes the matrix multiplication of the measurement matrix invese and
    # the output intensity vector at each point (y,x) for all points in space in order to recover the Stokes datacube of size (4, y, x)
    stokes_datacube = np.einsum('ij,jyx->iyx', measurement_matrix_inv, output_intensities)

    # replace I component of the Stokes datacube with the PSF subtracted intensity
    stokes_datacube[0] = psf_subtracted_intensity.data[0]

    # construct final error terms for output Stokes datacube
    stokes_cov = np.einsum('ij,jkyx,kl->ilyx', measurement_matrix_inv, output_intensities_cov, measurement_matrix_inv.T)
    output_err = np.zeros(shape=(1,4,image_size_y,image_size_x))
    output_err[0,0] = psf_subtracted_intensity.err[0]
    output_err[0,1] = np.sqrt(stokes_cov[1, 1])
    output_err[0,2] = np.sqrt(stokes_cov[2, 2])
    output_err[0,3] = np.sqrt(stokes_cov[3, 3])

    # construct output
    output_frame = data.Image(stokes_datacube,
                              pri_hdr=psf_subtracted_intensity.pri_hdr.copy(),
                              ext_hdr=psf_subtracted_intensity.ext_hdr.copy(),
                              err=output_err)
    updated_dataset = data.Dataset([output_frame])
    history_msg = "Combined Polarization States"
    updated_dataset.update_after_processing_step(history_msg)
    return updated_dataset


        
def update_to_l4(input_dataset, corethroughput_cal, flux_cal):
    """
    Updates the data level to L4. Only works on L3 data.

    Currently only checks that data is at the L3 level

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)
        corethroughput_cal (corgidrp.data.CoreThroughputCalibration): a CoreThroughputCalibration calibration file. Can be None
        flux_cal (corgidrp.data.FluxCalibration): a FluxCalibration calibration file. Cannot be None

    Returns:
        corgidrp.data.Dataset: same dataset now at L4-level
    """
    # check that we are running this on L1 data
    for orig_frame in input_dataset:
        if orig_frame.ext_hdr['DATALVL'] != "L3":
            err_msg = "{0} needs to be L3 data, but it is {1} data instead".format(orig_frame.filename, orig_frame.ext_hdr['DATALVL'])
            raise ValueError(err_msg)

    # we aren't altering the data
    updated_dataset = input_dataset.copy(copy_data=False)

    for frame in updated_dataset:
        # update header
        frame.ext_hdr['DATALVL'] = "L4"
        if corethroughput_cal is not None:
            frame.ext_hdr['CTCALFN'] = corethroughput_cal.filename.split("/")[-1] #Associate the ct calibration file
        else:
            frame.ext_hdr['CTCALFN'] = ''
        frame.ext_hdr['FLXCALFN'] = flux_cal.filename.split("/")[-1] #Associate the flux calibration file
        # update filename convention. The file convention should be
        # "CGI_[dataleel_*]" so we should be same just replacing the just instance of L1
        frame.filename = frame.filename.replace("_l3_", "_l4_", 1)

    history_msg = "Updated Data Level to L4"
    updated_dataset.update_after_processing_step(history_msg)

    return updated_dataset
