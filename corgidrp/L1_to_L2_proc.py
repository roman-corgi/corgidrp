# -*- coding: utf-8 -*-
"""Process EMCCD images from L1 to L2."""

import numpy as np
import os
from pathlib import Path

from . import check
from .read_metadata import Metadata
from .gsw_nonlin import get_relgains
from .gsw_emccd_frame import EMCCDFrame

here = Path(os.path.dirname(os.path.abspath(__file__)))
meta_path_default = Path(here, 'metadata.yaml')

class L1_to_L2_proc(object):
    """Process EMCCD frames from L1 to L2.

    L1 to L2a
    cleaned, single frame, with applicable calibrations applied
    
    --
    Use cases:
        - TTR5 Analog Non-PSF-subtracted on-sky observation data 
        (These are non-calibration data on astrophysical objects 
         (eg: unocculted star)
        - TTR5 Analog PSF-subtracted on-sky data
        - TTR5 Photon-Counted Non-PSF-subtracted On-Sky Observation Data
        - TTR5 Photon-counted PSF-subtracted on-sky data
             
    cleaned single image.

    input units: initial DN / pixel / frame

    output units: initial DN / pixel / frame

    CTC WFSC-delivered GSW algorithm

    Separate image from prescan and overscan
    Compute bias, including offset subtraction, and subtract from image
    Build per-frame cosmic ray masks
    Correct for nonlinearity
    provide:
        frame
        per-frame cosmic ray masks
        bias estimate         
    --

    L2a to L2b
    cleaned, calibrated image
    
    --
    Use cases:
        - TTR5 Analog Non-PSF-subtracted on-sky observation data 
        (These are non-calibration data on astrophysical objects 
         (eg: unocculted star)
        - TTR5 Photon-Counted Non-PSF-subtracted On-Sky Observation Data

    
    input units: initial DN / pixel / frame

    output units: photoelectrons / pixel / frame

    CTC WFSC-delivered GSW algorithm

    Convert from DN to electrons
    Divide by EM gain
    Subtract bias-subtracted, gain-divided master dark in electrons
    Divide by corresponding flat field
    Compute per-frame bad-pixel map from fixed bad pixel map (master bad-pixel map constructed from calibration data such as dark and flat as well as other sources of knowledge where present) and additional flagged pixels at the frame level if any.
    provide:
        frame
        per-frame bad-pixel maps
    --
    
    L2a to L2b
    cleaned, calibrated image
    
    --
    Use cases:
        - TTR5 Analog PSF-subtracted on-sky data
        
    Analog observation data should not be coadded before archiving; this ensures users have freedom to do their own frame selection.

    input units: initial DN / pixel / frame

    output units: photoelectrons / pixel / frame

    PS+CPP-delivered GSW algorithm

    Apply frame rejection (eg: high tip/tilt residuals)
    Convert from DN to electrons
    Divide by EM gain
    Apply CTI correction
    Subtract bias-subtracted, gain-divided master dark in electrons
    Divide by corresponding flat field
    Run additional recheck post-correction to see if any of the bad pixels from cosmic-ray flagging are no longer bad–e.g. warm pixels that were well-subtracted.
    Compute per-frame bad-pixel map from fixed bad pixel map (master bad-pixel map constructed from calibration data such as dark and flat as well as other sources of knowledge where present) and additional flagged pixels at the frame level if any.
    Correct for (either mask out or replace) bad pixels
    report/provide:
        EM gain measured from frames
        CTI correction
        frames rejected & criteria
        frame
        per-frame bad-pixel maps

    --
    
    L2a to L2b
    cleaned, calibrated image
    
    --
    Use cases:
        - TTR5 Photon-counted PSF-subtracted on-sky data
        
    input units: initial DN / pixel / frame

    output units: photoelectrons / pixel / combined frame

    PS+CPP-delivered GSW algorithm

    Read in N frames.  N must be ≥ 1 and in general will be >> 1.  N will be less than or equal to the number of frames in that pointing (eg: single roll in a roll/chop sequence).
    For each of N frames:
        Apply frame rejection (eg: high tip/tilt residuals)
        Convert from DN to electrons
        Apply photon-counting threshold
    Mean-combine all N thresholded frames together.
    Mean-combine all N bad pixel maps together.
    Apply photon-counting photometric corrections: thresholding loss, coincidence loss
    Do NOT divide by EM gain (this was covered in the thresholding)
    Apply CTI correction
    Subtract bias-subtracted, photon-counted master dark in electrons 
    Divide by corresponding flat field
    Do NOT run additional recheck post-correction to see if any of the bad pixels from cosmic-ray flagging are no longer bad–this step is not meaningful for photon-counted frames.
    Compute per-frame bad-pixel map from fixed bad pixel map (master bad-pixel map constructed from calibration data such as dark and flat as well as other sources of knowledge where present) and additional flagged pixels at the frame level if any. In this case, the second element will be the combined bad pixel map from N frames.
    Correct for (either mask out or replace) bad pixels
    Report/provide:
        EM gain measured from frames
        CTI correction
        frames rejected & criteria
        photon-counting threshold
        combined frame
        combined bad-pixel map     
   -- 
    
    Parameters
    ----------
    bad_pix : array_like
        Bad pixel mask. Bad pixels are True.
    eperdn : float
        Electrons per dn conversion factor (detector k gain).
    fwc_em_e : int
        Full well capacity of detector EM gain register (electrons).
    fwc_pp_e : int
        Full well capacity of detector image area pixels (electrons).
    bias_offset : float
        Median number of counts in the bias region due to fixed non-bias noise
        not in common with the image region.  Basically we compute the bias
        for the image region based on the prescan from each frame, and the
        bias_offset is how many additional counts the prescan had from e.g.
        fixed-pattern noise.  This value is subtracted from each measured bias.
        Units of DNs.
    em_gain : float
        Electron multiplying gain that was used for the
        master dark, flat, and the observation. >= 1.
    exptime : float
        Exposure time that was used for the master dark, flat,
        and the observation, in seconds. >=0.
    nonlin_path : str
        Path to nonlinearity relative gain file.
    meta_path : str
        Full path of desired metadata YAML file.  Defaults to full path of
        metadata.yaml included in proc_cgi_frame.
    dark : array_like
        Dark frame used for dark subtraction. This frame is expected to have
        been pre-processed (bias subtracted, flat field corrected,
        nonlinearity corrected, gain divided). If None, dark will be all zeros
        and have no effect on the image. Defaults to None.
    flat : array_like
        Flat field array for pixel scaling of image section. If None, flat will
        be all ones and have no effect on the image. Defaults to None.
    sat_thresh : float
        Multiplication factor for fwc that determines saturated cosmic pixels.
        Defaults to 0.99.
    plat_thresh : float
        Multiplication factor for fwc that determines edges of cosmic plateau.
        Defaults to 0.85.
    cosm_filter : int
        Minimum length in pixels of cosmic plateus to be identified.  Defaults
        to 2.

    Attributes
    ----------
    meta : instance
        Instance of Metadata class containing detector metadata.

    """

    def __init__(self, bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                 bias_offset, em_gain, exptime,
                 nonlin_path, meta_path=meta_path_default,
                 dark=None, flat=None, sat_thresh=0.99, plat_thresh=0.85,
                 cosm_filter=2):
        check.twoD_array(bad_pix, 'bad_pix', TypeError)
        check.real_positive_scalar(eperdn, 'eperdn', TypeError)
        check.positive_scalar_integer(fwc_em_e, 'fwc_em_e', TypeError)
        check.positive_scalar_integer(fwc_pp_e, 'fwc_pp_e', TypeError)
        #meta_path checked further below, and nonlin_path is
        # checked by _parse_file
        check.real_nonnegative_scalar(bias_offset, 'bias_offset', TypeError)
        check.real_positive_scalar(em_gain, 'em_gain', TypeError)
        if em_gain < 1:
            raise ValueError('em_gain must be >= 1.')
        check.real_positive_scalar(exptime, 'exptime', TypeError)
        if dark is None:
            dark = np.zeros_like(bad_pix, dtype=float)
        if flat is None:
            flat = np.ones_like(bad_pix, dtype=float)
        check.twoD_array(dark, 'dark', TypeError)
        check.twoD_array(flat, 'flat', TypeError)
        if np.shape(dark) != np.shape(bad_pix):
            raise ValueError('The dimensions of dark and bad_pix '
                'should agree.')
        if np.shape(flat) != np.shape(bad_pix):
            raise ValueError('The dimensions of flat and bad_pix '
                'should agree.')
        check.real_positive_scalar(sat_thresh, 'sat_thresh', TypeError)
        check.real_positive_scalar(plat_thresh, 'plat_thresh', TypeError)
        check.positive_scalar_integer(cosm_filter, 'cosm_filter', TypeError)

        self.bad_pix = bad_pix
        self.eperdn = eperdn
        # Need DNs for cosmics, but FWC is normally in electrons
        self.fwc_em_dn = fwc_em_e/eperdn
        self.fwc_pp_dn = fwc_pp_e/eperdn
        self.bias_offset = bias_offset
        self.em_gain = em_gain
        self.exptime = exptime
        self.meta_path = meta_path
        self.dark = dark
        self.flat = flat
        self.sat_thresh = sat_thresh
        self.plat_thresh = plat_thresh
        self.cosm_filter = cosm_filter

        # Read metadata file
        try:
            self.meta = Metadata(meta_path)
        except:
            raise FileNotFoundError('metadata file not found')

        self.nonlin_path = nonlin_path


    def L1_to_L2a(self, frame_dn):
        """
        Take a level 1 CGI data product (raw EXCAM frame) and return a Level 2a
        CGI data product.

        Input units are initial DN/pixel/frame.
        Output units on frame are initial DN/pixel/frame.

        This will implement the following steps:
         - Separate image from prescan
         - Compute bias from prescan, including offset subtraction, and
           subtract from image
         - Build per-frame cosmic ray masks and remove cosmic rays
         - Correct for nonlinearity

        Despite taking EM gain as an input, this function does not divide by
        gain.  Gain is used as part of cosmic-ray processing to understand the
        expected saturation limit and as part of the nonlinearity correction to
        determine the appropriate correction level.

        Parameters
        ----------
        frame_dn : array_like
            Raw EMCCD frame (dn).

        Returns
        -------
        image : array_like
            Processed image area of frame with bias subtracted and
            nonlinearity corrected.
        bpmap : array_like
            Bad-pixel map flagging all bad pixels in that image.
            All unmasked pixels have the value 0, and all
            masked pixels have the value 1.
        image_r : array_like
            Same as "image" but with the bad pixels removed (i.e., designated
            as zeros).
        bias : array_like
            Row-by-row bias estimate for the image area.
        frame : array_like
            Processed full frame with bias subtracted and
            nonlinearity corrected.
        bpmap_frame : array_like
            Bad-pixel map flagging all bad pixels in that full frame.
            All unmasked pixels have the value 0, and all
            masked pixels have the value 1.
        bias_frame : array_like
            Row-by-row bias estimate for the full prescan.

        """

        check.twoD_array(frame_dn, 'frame_dn', TypeError)

        frameobj = EMCCDFrame(frame_dn,
                              self.meta,
                              self.fwc_em_dn,
                              self.fwc_pp_dn,
                              self.em_gain,
                              self.bias_offset,
        )

        #for full and image-area outputs:
        # Subtract bias and bias offset and get cosmic mask
        image = frameobj.image_bias0
        frame = frameobj.frame_bias0
        bpmap, bpmap_frame = frameobj.remove_cosmics(
                sat_thresh=self.sat_thresh, plat_thresh=self.plat_thresh,
                cosm_filter=self.cosm_filter)
        # Correct for nonlinearity
        image *= get_relgains(image, self.em_gain, self.nonlin_path)
        frame *= get_relgains(frame, self.em_gain, self.nonlin_path)

        # remove bad pixels for image_r:
        image_r = np.ma.masked_array(image, bpmap)
        image_r = image_r.filled(0)


        return (image, bpmap, image_r, frameobj.bias,
                frame, bpmap_frame, frameobj.frame_bias)


    def L2a_to_L2b(self, image, bpmap):
        """
        Take a level 2a CGI data product (partially-processed in DNs) and
        return a Level 2b CGI data product (fully processed, in photo-e)

        Input units are initial DN/pixel/frame.
        Output units on frame are photoelectrons/pixel/frame.

        This will implement the following steps:
         - Convert from DN to electrons
         - Divide by EM gain
         - Subtract bias-subtracted, gain-divided master dark in electrons
         - Divide by corresponding flat field
         - Compute per-frame bad-pixel map from fixed bad pixel map (master bad
           pixel map constructed from calibration data such as dark and flat as
           well as other sources of knowledge where present) and additional
           flagged pixels at the frame level if any.

        If dark and flat have been left at their default values, the third and
        fourth steps will be identity operations (useful for master dark
        calibration).

        Parameters
        ----------
        image : array_like
            Processed image area of frame with bias subtracted and nonlinearity
            corrected.
        bpmap : array_like
            Bad-pixel map flagging all bad pixels in that frame.  Must be 0
            (good) or 1 (bad) at every pixel.

        Returns
        -------
        L2b_image : array_like
            Processed image area of frame with bias subtracted, nonlinearity
            corrected, and dark frame, flat-field, and gain compensated for.
            Converted from DNs to photoelectrons.
        L2b_bpmap : array_like
            Bad-pixel map flagging all bad pixels in the frame, including
            known fixed bad pixels.
        L2b_image_r : array_like
            Same as "image" but with the bad pixels removed (i.e., designated
            as zeros).

        """

        check.twoD_array(image, 'image', TypeError)
        check.twoD_array(bpmap, 'bpmap', TypeError)
        if np.shape(image) != np.shape(bpmap):
            raise ValueError('image and bpmap must have the same dimensions.')
        if np.logical_and((bpmap != 0), (bpmap != 1)).any():
            raise TypeError('bpmap must be 0- or 1-valued')

        L2b_image = image.copy()

        # Convert from DN to e- (k gain conversion)
        L2b_image *= self.eperdn

        # Correct for gain
        L2b_image /= self.em_gain

        # Combine masks
        L2b_bpmap = np.logical_or(self.bad_pix, bpmap).astype(int)

        L2b_image -= self.dark

        # Divide by flat
        # Divide image by flat only where flat is not equal to 0.
        # Where flat is equal to 0, set image to zero
        L2b_image = np.divide(L2b_image,
                              self.flat,
                              out=np.zeros_like(L2b_image),
                              where=self.flat != 0)

        # remove bad pixels for L2b_image_r:
        L2b_image_r = np.ma.masked_array(L2b_image, L2b_bpmap)
        L2b_image_r = L2b_image_r.filled(0)

        return L2b_image, L2b_bpmap, L2b_image_r

#---------------------------------------
# Median and mean combines for L2b data
#---------------------------------------

def median_combine(image_list, bpmap_list):
    """
    Get median frame and corresponding bad-pixel map from L2b data frames.  The
    input "image_list" should consist of frames with no bad pixels marked or
    removed.  This function takes the bad-pixels maps into account when taking
    the median.

    The two lists must be the same length, and each 2D array in each list must
    be the same size, both within a list and across lists.

    If the inputs are instead np.ndarray (a single frame or a stack),
    the function will accommodate and convert them to lists of arrays.

    Parameters
    ----------
    image_list : list or array-like
        List (or stack) of L2b data frames
        (with no bad pixels applied to them).
    bpmap_list : list or array-like
        List (or stack) of bad-pixel maps associated with L2b data frames.
        Each must be 0 (good) or 1 (bad) at every pixel.

    Returns
    -------
    comb_image : array_like
        Median-combined frame from input list data.

    comb_bpmap : array_like
        Median-combined bad-pixel map.

    """
    # if input is an np array or stack, try to accommodate
    if type(image_list) == np.ndarray:
        if image_list.ndim == 1: # pathological case of empty array
            image_list = list(image_list)
        elif image_list.ndim == 2: #covers case of single 2D frame
            image_list = [image_list]
        elif image_list.ndim == 3: #covers case of stack of 2D frames
            image_list = list(image_list)
    if type(bpmap_list) == np.ndarray:
        if bpmap_list.ndim == 1: # pathological case of empty array
            bpmap_list = list(bpmap_list)
        elif bpmap_list.ndim == 2: #covers case of single 2D frame
            bpmap_list = [bpmap_list]
        elif bpmap_list.ndim == 3: #covers case of stack of 2D frames
            bpmap_list = list(bpmap_list)
    # Check inputs
    if not isinstance(image_list, list):
        raise TypeError('image_list must be a list')
    if not isinstance(bpmap_list, list):
        raise TypeError('bpmap_list must be a list')
    if len(image_list) != len(bpmap_list):
        raise TypeError('image_list and bpmap_list must be the same length')
    if len(image_list) == 0:
        raise TypeError('input lists cannot be empty')
    s0 = image_list[0].shape
    for index, im in enumerate(image_list):
        check.twoD_array(im, 'image_list[' + str(index) + ']', TypeError)
        if im.shape != s0:
            raise TypeError('all input list elements must be the same shape')
        pass
    for index, bp in enumerate(bpmap_list):
        check.twoD_array(bp, 'bpmap_list[' + str(index) + ']', TypeError)
        if np.logical_and((bp != 0), (bp != 1)).any():
            raise TypeError('bpmap_list elements must be 0- or 1-valued')
        if bp.dtype != int:
            raise TypeError('bpmap_list must be made up of int arrays')
        if bp.shape != s0:
            raise TypeError('all input list elements must be the same shape')
        pass

    # Get masked arrays
    ims_m = np.ma.masked_array(image_list, bpmap_list)
    # take median, ignoring masked pixels
    med = np.ma.median(ims_m, axis=0)
    # combined mask:
    comb_bpmap = med.mask
    # combined image, setting any pixels that were masked all the way through
    # to 0:
    comb_image = med.filled(0)

    return comb_image, comb_bpmap


def mean_combine(image_list, bpmap_list):
    """
    Get mean frame and corresponding bad-pixel map from L2b data frames.  The
    input "image_list" should consist of frames with no bad pixels marked or
    removed.  This function takes the bad-pixels maps into account when taking
    the mean.

    The two lists must be the same length, and each 2D array in each list must
    be the same size, both within a list and across lists.

    If the inputs are instead np.ndarray (a single frame or a stack),
    the function will accommodate and convert them to lists of arrays.

    Also Includes outputs for processing darks used for calibrating the
    master dark.

    Parameters
    ----------
    image_list : list or array_like
        List (or stack) of L2b data frames
        (with no bad pixels applied to them).
    bpmap_list : list or array_like
        List (or stack) of bad-pixel maps associated with L2b data frames.
        Each must be 0 (good) or 1 (bad) at every pixel.

    Returns
    -------
    comb_image : array_like
        Mean-combined frame from input list data.

    comb_bpmap : array_like
        Mean-combined bad-pixel map.

    mean_num_good_fr : int
        Mean number of good, unmasked pixels.  Used for getting read
        noise in the calibration of the master dark.

    enough_for_rn : bool
        Useful only for the calibration of the master dark.
        False:  Fewer than half the frames available for at least one pixel in
        the averaging due to masking, so noise maps cannot be effectively
        determined for all pixels.
        True:  Half or more of the frames available for all pixels, so noise
        mpas can be effectively determined for all pixels.

    """
    # if input is an np array or stack, try to accommodate
    if type(image_list) == np.ndarray:
        if image_list.ndim == 1: # pathological case of empty array
            image_list = list(image_list)
        elif image_list.ndim == 2: #covers case of single 2D frame
            image_list = [image_list]
        elif image_list.ndim == 3: #covers case of stack of 2D frames
            image_list = list(image_list)
    if type(bpmap_list) == np.ndarray:
        if bpmap_list.ndim == 1: # pathological case of empty array
            bpmap_list = list(bpmap_list)
        elif bpmap_list.ndim == 2: #covers case of single 2D frame
            bpmap_list = [bpmap_list]
        elif bpmap_list.ndim == 3: #covers case of stack of 2D frames
            bpmap_list = list(bpmap_list)

    # Check inputs
    if not isinstance(image_list, list):
        raise TypeError('image_list must be a list')
    if not isinstance(bpmap_list, list):
        raise TypeError('bpmap_list must be a list')
    if len(image_list) != len(bpmap_list):
        raise TypeError('image_list and bpmap_list must be the same length')
    if len(image_list) == 0:
        raise TypeError('input lists cannot be empty')
    s0 = image_list[0].shape
    for index, im in enumerate(image_list):
        check.twoD_array(im, 'image_list[' + str(index) + ']', TypeError)
        if im.shape != s0:
            raise TypeError('all input list elements must be the same shape')
        pass
    for index, bp in enumerate(bpmap_list):
        check.twoD_array(bp, 'bpmap_list[' + str(index) + ']', TypeError)
        if np.logical_and((bp != 0), (bp != 1)).any():
            raise TypeError('bpmap_list elements must be 0- or 1-valued')
        if bp.dtype != int:
            raise TypeError('bpmap_list must be made up of int arrays')
        if bp.shape != s0:
            raise TypeError('all input list elements must be the same shape')
        pass


    # Get masked arrays
    ims_m = np.ma.masked_array(image_list, bpmap_list)

    # Add non masked elements
    sum_im = np.zeros_like(image_list[0])
    map_im = np.zeros_like(image_list[0], dtype=int)
    for im_m in ims_m:
        masked = im_m.filled(0)
        sum_im += masked
        map_im += (im_m.mask == False).astype(int)

    # Divide sum_im by map_im only where map_im is not equal to 0 (i.e.,
    # not masked).
    # Where map_im is equal to 0, set combined_im to zero
    comb_image = np.divide(sum_im, map_im, out=np.zeros_like(sum_im),
                            where=map_im != 0)

    # Mask any value that was never mapped (aka masked in every frame)
    comb_bpmap = (map_im == 0).astype(int)

    enough_for_rn = True
    if map_im.min() < len(image_list)/2:
        enough_for_rn = False

    mean_num_good_fr = np.mean(map_im)

    return comb_image, comb_bpmap, mean_num_good_fr, enough_for_rn