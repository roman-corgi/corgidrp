import numpy as np
import corgidrp.data as data
from scipy.ndimage import median_filter

def create_dark_calib(dark_dataset):
    """
    Turn this dataset of image frames that were taken to measure 
    the dark current into a dark calibration frame

    Args:
        dark_dataset (corgidrp.data.Dataset): a dataset of Image frames (L2a-level)
    Returns:
        data.Dark: a dark calibration frame
    """
    combined_frame = np.nanmean(dark_dataset.all_data, axis=0)

    new_dark = data.Dark(combined_frame, pri_hdr=dark_dataset[0].pri_hdr.copy(), 
                         ext_hdr=dark_dataset[0].ext_hdr.copy(), input_dataset=dark_dataset)
    
    return new_dark


def dark_subtraction(input_dataset, dark_frame):
    """
    Perform dark current subtraction of a dataset using the corresponding dark frame

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need dark subtraction (L2a-level)
        dark_frame (corgidrp.data.Dark): a Dark frame to model the dark current
    Returns:
        corgidrp.data.Dataset: a dark subtracted version of the input dataset
    """
    # you should make a copy the dataset to start
    darksub_dataset = input_dataset.copy()

    darksub_cube = darksub_dataset.all_data - dark_frame.data

    history_msg = "Dark current subtracted using dark {0}".format(dark_frame.filename)

    # update the output dataset with this new dark subtracted data and update the history
    darksub_dataset.update_after_processing_step(history_msg, new_all_data=darksub_cube)

    return darksub_dataset

def reject_crs(input_dataset, sat_thresh, plat_thresh, cosm_filter):
    """
    Perform cosmic ray flagging and masking on a dataset

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need cosmic ray rejection (L1-level)
        fsat_thresh (float): 
            Multiplication factor for fwc that determines saturated cosmic
            pixels.
        plat_thresh (float): 
            Multiplication factor for fwc that determines edges of cosmic
            plateau.
        cosm_filter (int): 
            Minimum length in pixels of cosmic plateus to be identified.
    Returns:
        corgidrp.data.Dataset: a version of the input dataset with cosmic rays masked
    """
    # you should make a copy the dataset to start
    crmasked_dataset = input_dataset.copy()

    crmasked_cube = crmasked_dataset.all_data

    # pick the FWC that will get saturated first, depending on gain
    sat_fwc = sat_thresh*min(self.fwc_em, self.fwc_pp*self.em_gain)

    # threshold the frame to catch any values above sat_fwc --> this is
    # mask 1
    m1 = (self.image_bias0 >= sat_fwc)
    # run remove_cosmics() with fwc=fwc_em since tails only come from
    # saturation in the gain register --> this is mask 2
    m2 = flag_crs(image=self.image_bias0,
                    fwc=self.fwc_em,
                    sat_thresh=sat_thresh,
                    plat_thresh=plat_thresh,
                    cosm_filter=cosm_filter,
                    )
    # same thing, but now making masks for full frame (for calibrate_darks)

    # threshold the frame to catch any values above sat_fwc --> this is
    # mask 1
    m1_full = (self.frame_bias0 >= sat_fwc)
    # run remove_cosmics() with fwc=fwc_em since tails only come from
    # saturation in the gain register --> this is mask 2
    m2_full = remove_cosmics(image=self.frame_bias0,
                        fwc=self.fwc_em,
                        sat_thresh=sat_thresh,
                        plat_thresh=plat_thresh,
                        cosm_filter=cosm_filter,
                        )

    # OR the two masks together and return
    return np.logical_or(m1, m2), np.logical_or(m1_full, m2_full)

    #history_msg = "Dark current subtracted using dark {0}".format(dark_frame.filename)

    # update the output dataset with this new dark subtracted data and update the history
    crmasked_dataset.update_after_processing_step(history_msg, new_all_data=crmasked_cube)

    return crmasked_dataset

def flag_crs(image, fwc, sat_thresh, plat_thresh, cosm_filter):
    """Identify and remove saturated cosmic ray hits and tails.

    Use sat_thresh (interval 0 to 1) to set the threshold above which cosmics
    will be detected. For example, sat_thresh=0.99 will detect cosmics above
    0.99*fwc.

    Use plat_thresh (interval 0 to 1) to set the threshold under which cosmic
    plateaus will end. For example, if plat_thresh=0.85, once a cosmic is
    detected the beginning and end of its plateau will be determined where the
    pixel values drop below 0.85*fwc.

    Use cosm_filter to determine the smallest plateaus (in pixels) that will
    be identified. A reasonable value is 2.

    Parameters
    ----------
    image : array_like, float
        Image area of frame (bias of zero).
    fwc : float
        Full well capacity of detector *in DNs*.  Note that this may require a
        conversion as FWCs are usually specified in electrons, but the image
        is in DNs at this point.
    sat_thresh : float
        Multiplication factor for fwc that determines saturated cosmic pixels.
    plat_thresh : float
        Multiplication factor for fwc that determines edges of cosmic plateu.
    cosm_filter : int
        Minimum length in pixels of cosmic plateus to be identified.

    Returns
    -------
    mask : array_like, int
        Mask for pixels that have been set to zero.

    Notes
    -----
    This algorithm uses a row by row method for cosmic removal. It first finds
    streak rows, which are rows that potentially contain cosmics. It then
    filters each of these rows in order to differentiate cosmic hits (plateaus)
    from any outlier saturated pixels. For each cosmic hit it finds the leading
    ledge of the plateau and kills the plateau and the rest of the row to take
    out the tail.

    |<-------- streak row is the whole row ----------------------->|
     ......|<-plateau->|<------------------tail------------------->|

    B Nemati and S Miller - UAH - 02-Oct-2018

    """
    mask = np.zeros(image.shape, dtype=int)

    # Do a cheap prefilter for rows that don't have anything bright
    max_rows = np.max(image, axis=1)
    i_streak_rows = (max_rows >= sat_thresh*fwc).nonzero()[0]

    for i in i_streak_rows:
        row = image[i]

        # Find if and where saturated plateaus start in streak row
        i_beg = find_plateaus(row, fwc, sat_thresh, plat_thresh, cosm_filter)

        # If plateaus exist, kill the hit and the rest of the row
        if i_beg is not None:
            mask[i, i_beg:] = 1
            pass

    return mask

def find_plateaus(streak_row, fwc, sat_thresh, plat_thresh, cosm_filter):
    """Find the beginning index of each cosmic plateau in a row.

    Note that i_beg is set at one pixel before first plateau pixel, as these
    pixels immediately neighboring the cosmic plateau are very often affected
    by the cosmic hit as well.

    Parameters
    ----------
    streak_row : array_like, float
        Row with possible cosmics.
    fwc : float
        Full well capacity of detector *in DNs*.  Note that this may require a
        conversion as FWCs are usually specified in electrons, but the image
        is in DNs at this point.
    sat_thresh : float
        Multiplication factor for fwc that determines saturated cosmic pixels.
    plat_thresh : float
        Multiplication factor for fwc that determines edges of cosmic plateu.
    cosm_filter : int
        Minimum length in pixels of cosmic plateus to be identified.

    Returns
    -------
    i_beg : array_like, int
        Index of plateau beginnings, or None if there is no plateau.

    """
    # Lowpass filter row to differentiate plateaus from standalone pixels
    # The way median_filter works, it will find cosmics that are cosm_filter-1
    # wide. Add 1 to cosm_filter to correct for this
    filtered = median_filter(streak_row, cosm_filter+1, mode='nearest')
    saturated = (filtered >= sat_thresh*fwc).nonzero()[0]

    if len(saturated) > 0:
        i_beg = saturated[0]
        while i_beg > 0 and streak_row[i_beg] >= plat_thresh*fwc:
            i_beg -= 1

        return i_beg
    else:
        return None