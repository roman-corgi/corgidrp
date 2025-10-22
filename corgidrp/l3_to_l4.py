# A file that holds the functions that transmogrify l3 data to l4 data 

from pyklip.klip import rotate
import scipy.ndimage
from astropy.wcs import WCS

from corgidrp import data
from corgidrp.detector import flag_nans,nan_flags
from corgidrp import star_center
import corgidrp
from corgidrp.klip_fm import meas_klip_thrupt
from corgidrp.corethroughput import get_1d_ct
from corgidrp.spec import create_wave_cal
from scipy.ndimage import rotate as rotate_scipy # to avoid duplicated name
from scipy.ndimage import shift
import warnings
import numpy as np
import pyklip.rdi
import os
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

def distortion_correction(input_dataset, astrom_calibration):
    """
    
    Applies the distortion correction to the dataset. The function interpolates the bad pixels 
    before applying the distortion correction to avoid creating more bad pixels. It then adds 
    the bad pixels back in after the correction is applied, keeping the bad pixel maps the same. 
    Furthermore it also applies the distortion correction to the error maps.
    

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
        im_dq = undistorted_data.dq
        imgsizeX, imgsizeY = im_data.shape

        # set image size to the largest axis if not square imagea
        if (imgsizeX >= imgsizeY): imgsize = imgsizeX
        else: imgsize = imgsizeY

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

        # interpolating bad pixels to not spillover during the interpolation while saving bpix
        im_bpixs = np.zeros_like(im_data)
        im_bpixs[im_dq.astype(bool)] = im_data[im_dq.astype(bool)]
        
        im_data[im_dq.astype(bool)] = np.nan
        
        kernel = Gaussian2DKernel(3)
        im_data = interpolate_replace_nans(im_data, kernel)
    
        undistorted_image = scipy.ndimage.map_coordinates(im_data, [gridy, gridx])
        
        # interpolate the errors
        if len(im_err.shape) == 2:
            
            err_bpixs = np.zeros_like(im_err)
            err_bpixs[im_dq.astype(bool)] = im_err[im_dq.astype(bool)]
            
            im_err[im_dq.astype(bool)] = np.nan
            im_err = interpolate_replace_nans(im_err, kernel)
            
            undistorted_errors = scipy.ndimage.map_coordinates(im_err, [gridy, gridx])
            undistorted_errors[im_dq.astype(bool)] = err_bpixs[im_dq.astype(bool)]
        else:
            undistorted_errors = []
            for err in im_err:
                err_bpixs = np.zeros_like(err)
                err_bpixs[im_dq.astype(bool)] = err[im_dq.astype(bool)]
            
                err[im_dq.astype(bool)] = np.nan
                err = interpolate_replace_nans(err, kernel)
                
                und_err = scipy.ndimage.map_coordinates(err, [gridy, gridx])
                und_err[im_dq.astype(bool)] = err_bpixs[im_dq.astype(bool)]
                
                undistorted_errors.append(und_err)
        
        # put the bad pixels back in
        
        undistorted_image[im_dq.astype(bool)] = im_bpixs[im_dq.astype(bool)]

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


def crop(input_dataset, sizexy=None, centerxy=None):
    """
    
    Crop the Images in a Dataset to a desired field of view. Default behavior is to 
    crop the image to the dark hole region, centered on the pixel intersection nearest 
    to the star location. Assumes 3D Image data is a stack of 2D data arrays, so only 
    crops the last two indices. Currently only configured for HLC mode.

    TODO: 
        - Pad with nans if you try to crop outside the array (handle err & DQ too)
        - Option to crop to an odd data array and center on a pixel?

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (any level)
        sizexy (int or array of int): desired frame size, if only one number is provided the 
            desired shape is assumed to be square, otherwise xy order. If not provided, 
            defaults to 60 for NFOV (narrow field-of-view) observations. Defaults to None.
        centerxy (float or array of float): desired center (xy order), should be a pixel intersection (a.k.a 
            half-integer) otherwise the function rounds to the nearest intersection. Defaults to the 
            "STARLOCX/Y" header values.

    Returns:
        corgidrp.data.Dataset: a version of the input dataset cropped to the desired FOV.
    """

    # Copy input dataset
    dataset = input_dataset.copy()

    # Require even data shape
    if not sizexy is None and not np.all(np.array(sizexy)%2==0):
        raise UserWarning('Even sizexy is required.')
       
    # Need to loop over frames and reinit dataset because array sizes change
    frames_out = []

    for frame in dataset:
        prihdr = frame.pri_hdr
        exthdr = frame.ext_hdr
        dqhdr = frame.dq_hdr
        errhdr = frame.err_hdr

        # Pick default crop size based on the size of the effective field of view
        if sizexy is None:
            if exthdr['LSAMNAME'] == 'NFOV':
                sizexy = 60
            else:
                raise UserWarning('Crop function is currently only configured for NFOV (narrow field-of-view) observations if sizexy is not provided.')

        # Assign new array sizes and center location
        frame_shape = frame.data.shape
        if isinstance(sizexy,int):
            sizexy = [sizexy]*2
        if isinstance(centerxy,float):
            centerxy = [centerxy] * 2
        elif centerxy is None:
            if ("STARLOCX" in exthdr.keys()) and ("STARLOCY" in exthdr.keys()):
                centerxy = np.array([exthdr["STARLOCX"],exthdr["STARLOCY"]])
            else: raise ValueError('centerxy not provided but STARLOCX/Y are missing from image extension header.')
        
        # Round to centerxy to nearest half-pixel
        centerxy = np.array(centerxy)
        if not np.all((centerxy-0.5)%1 == 0):
            old_centerxy = centerxy.copy()
            centerxy = np.round(old_centerxy-0.5)+0.5
            print(f'Desired center {old_centerxy} is not at the intersection of 4 pixels. Centering on the nearest intersection {centerxy}')
            
        # Crop the data
        start_ind = (centerxy + 0.5 - np.array(sizexy)/2).astype(int)
        end_ind = (centerxy + 0.5 + np.array(sizexy)/2).astype(int)
        x1,y1 = start_ind
        x2,y2 = end_ind

        # Check if cropping outside the FOV
        xleft_pad = -x1 if (x1<0) else 0
        xrright_pad = x2-frame_shape[-1]+1 if (x2 > frame_shape[-1]) else 0
        ybelow_pad = -y1 if (y1<0) else 0
        yabove_pad = y2-frame_shape[-2]+1 if (y2 > frame_shape[-2]) else 0
        
        if np.any(np.array([xleft_pad,xrright_pad,ybelow_pad,yabove_pad])> 0) :
            raise ValueError("Trying to crop to a region outside the input data array. Not yet configured.")

        if frame.data.ndim == 2:
            cropped_frame_data = frame.data[y1:y2,x1:x2]
            cropped_frame_err = frame.err[:,y1:y2,x1:x2]
            cropped_frame_dq = frame.dq[y1:y2,x1:x2]
        elif frame.data.ndim == 3:
            cropped_frame_data = frame.data[:,y1:y2,x1:x2]
            cropped_frame_err = frame.err[:,:,y1:y2,x1:x2]
            cropped_frame_dq = frame.dq[:,y1:y2,x1:x2]
        else:
            raise ValueError('Crop function only supports 2D or 3D frame data.')

        # Update headers
        exthdr["NAXIS1"] = sizexy[0]
        exthdr["NAXIS2"] = sizexy[1]
        dqhdr["NAXIS1"] = sizexy[0]
        dqhdr["NAXIS2"] = sizexy[1]
        errhdr["NAXIS1"] = sizexy[0]
        errhdr["NAXIS2"] = sizexy[1]
        errhdr["NAXIS3"] = cropped_frame_err.shape[-3]
        if frame.data.ndim == 3:
            exthdr["NAXIS3"] = frame.data.shape[0]
            dqhdr["NAXIS3"] = frame.dq.shape[0]
            errhdr["NAXIS4"] = frame.err.shape[0]
        
        updated_hdrs = []
        if ("STARLOCX" in exthdr.keys()):
            exthdr["STARLOCX"] -= x1
            exthdr["STARLOCY"] -= y1
            updated_hdrs.append('STARLOCX/Y')
        if ("CRPIX1" in prihdr.keys()):
            prihdr["CRPIX1"] -= x1
            prihdr["CRPIX2"] -= y1
            updated_hdrs.append('CRPIX1/2')
        if not ("DETPIX0X" in exthdr.keys()):
            exthdr.set('DETPIX0X',0)
            exthdr.set('DETPIX0Y',0)
        exthdr.set('DETPIX0X',exthdr["DETPIX0X"]+x1)
        exthdr.set('DETPIX0Y',exthdr["DETPIX0Y"]+y1)

        new_frame = data.Image(cropped_frame_data,prihdr,exthdr,cropped_frame_err,cropped_frame_dq,frame.err_hdr,frame.dq_hdr)
        new_frame.filename = frame.filename
        frames_out.append(new_frame)

    output_dataset = data.Dataset(frames_out)

    history_msg1 = f"""Frames cropped to new shape {list(output_dataset[0].data.shape)} on center {list(centerxy)}. Updated header kws: {", ".join(updated_hdrs)}."""
    output_dataset.update_after_processing_step(history_msg1)
    
    return output_dataset


def do_psf_subtraction(input_dataset, 
                       ct_calibration=None,
                       reference_star_dataset=None,
                       outdir=None,fileprefix="",
                       do_crop=True,
                       crop_sizexy=None,
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
        Handle propagate DQ array
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
        do_crop (bool, optional): whether to crop data before PSF subtraction. Defaults to True.
        crop_sizexy (list of int, optional): Desired size to crop the images to before PSF subtraction. Defaults to 
            None, which results in the step choosing a crop size based on the imaging mode. 
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

    # Crop data
    if do_crop:
        sci_dataset = crop(sci_dataset,sizexy=crop_sizexy)
        ref_dataset = None if ref_dataset is None else crop(ref_dataset,sizexy=crop_sizexy) 

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
        thrupt_hdr['COMMENT'] = ('KLIP Throughput and retrieved FWHM as a function of separation for each KLMode '
                                '(r, KL1, KL2, ...) = (data[0], data[1], data[2]). The last axis contains the'
                                'KL throughput in the 0th index and the FWHM in the 1st index')
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
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L3-level)
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
        ylen, xlen = sci_data.shape

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
            astr_hdr = WCS(sci_hd)
            CD1_2 = sci_hd['CD1_2']
            CD2_2 = sci_hd['CD2_2']
            roll_angle = -np.rad2deg(np.arctan2(-CD1_2, CD2_2)) # Compute North Position Angle from the WCS solutions

        else:
            warnings.warn('use "ROLL" instead of WCS to estimate the north position angle')
            astr_hdr = None
            # read the roll angle parameter, assuming this info is recorded in the primary header as requested
            roll_angle = processed_data.pri_hdr['ROLL']

        # derotate
        sci_derot = rotate(sci_data,roll_angle,(xcen,ycen),astr_hdr=astr_hdr) # astr_hdr is corrected at above lines
        new_all_data.append(sci_derot)

        log = f'FoV rotated by {roll_angle}deg counterclockwise at a roll center {xcen, ycen}'
        sci_hd['HISTORY'] = log 

        # update WCS solutions
        if use_wcs:
            sci_hd['CD1_1'] = astr_hdr.wcs.cd[0,0]
            sci_hd['CD1_2'] = astr_hdr.wcs.cd[0,1]
            sci_hd['CD2_1'] = astr_hdr.wcs.cd[1,0]
            sci_hd['CD2_2'] = astr_hdr.wcs.cd[1,1]
        #############

        ## HDU ERR ##
        err_data = processed_data.err
        err_derot = np.expand_dims(rotate(err_data[0],roll_angle,(xcen,ycen)), axis=0) # err data shape is 1x1024x1024
        new_all_err.append(err_derot)
        #############

        ## HDU DQ ##
        # all DQ pixels must have integers, use scipy.ndimage.rotate with order=0 instead of klip.rotate (rotating the other way)
        dq_data = processed_data.dq
        if xcen != xlen/2 or ycen != ylen/2:
                # padding, shifting (rot center to image center), rotating, re-shift (image center to rot center), and cropping
                # calculate shift values
                xshift = xcen-xlen/2; yshift = ycen-ylen/2

                # pad and shift
                pad_x = int(np.ceil(abs(xshift))); pad_y = int(np.ceil(abs(yshift)))
                dq_data_padded = np.pad(dq_data,pad_width=((pad_y, pad_y), (pad_x, pad_x)),mode='constant',constant_values=0)
                dq_data_padded_shifted = shift(dq_data_padded,(-yshift,-xshift),order=0,mode='constant',cval=0)

                # define slices for cropping
                crop_x = slice(pad_x,pad_x+xlen); crop_y = slice(pad_y,pad_y+ylen)

                # rotate (invserse direction to pyklip.rotate), re-shift, and crop
                dq_derot = shift(rotate_scipy(dq_data_padded_shifted, -roll_angle, order=0, mode='constant', reshape=False, cval=0),\
                 (yshift,xshift),order=0,mode='constant',cval=0)[crop_y,crop_x]
        else:
                # simply rotate 
                dq_derot = rotate_scipy(dq_data, -roll_angle, order=0, mode='constant', reshape=False, cval=0)

        new_all_dq.append(dq_derot)
        ############

    history_msg = 'North is Up and East is Left'
    processed_dataset.update_after_processing_step(history_msg, new_all_data=np.array(new_all_data), new_all_err=np.array(new_all_err),\
                                                   new_all_dq=np.array(new_all_dq))

    return processed_dataset 


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
        'wavlen': head['wavlen0'],
        'x' : head['x0'],
        'xerr': head['x0err'],
        'y': head['y0'],
        'yerr': head['y0err'],
        'shapex': head['shapex0'],
        'shapey': head['shapey0']
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
