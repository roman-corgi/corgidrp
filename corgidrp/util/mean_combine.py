import numpy as np
import corgidrp.util.check as check

def mean_combine(image_list, bpmap_list, err=False):
    """
    Get mean frame and corresponding bad-pixel map from L2b data frames.  The
    input image_list should consist of frames with no bad pixels marked or
    removed.  This function takes the bad-pixels maps into account when taking
    the mean.

    The two lists must be the same length, and each 2D array in each list must
    be the same size, both within a list and across lists.

    If the inputs are instead np.ndarray (a single frame or a stack),
    the function will accommodate and convert them to lists of arrays.

    Also Includes outputs for processing darks used for calibrating the
    master dark.

    Args:
        image_list : list or array_like
    List (or stack) of L2b data frames
    (with no bad pixels applied to them).
        bpmap_list : list or array_like
    List (or stack) of bad-pixel maps associated with L2b data frames.
    Each must be 0 (good) or 1 (bad) at every pixel.
        err : bool
    If True, calculates the standard error over all the frames.  Intended
    for the corgidrp.Data.Dataset.all_err arrays. Defaults to False.

    Returns:
        comb_image : array_like
    Mean-combined frame from input list data.

        comb_bpmap : array_like
    Mean-combined bad-pixel map.

        map_im : array-like
    Array showing how many frames per pixel were unmasked.
    Used for getting read
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
        if err:
            sum_im += masked**2
        else:
            sum_im += masked
        map_im += (im_m.mask == False).astype(int)

    if err: # sqrt of sum of sigma**2 terms
        sum_im = np.sqrt(sum_im)

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

    return comb_image, comb_bpmap, map_im, enough_for_rn
