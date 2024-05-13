import glob
import os

import corgidrp.data as data
import corgidrp.mocks as mocks
from corgidrp.l1_to_l2a import detect_cosmic_rays

import numpy as np
from astropy.io import fits
from pathlib import Path
from scipy.ndimage import median_filter
from pytest import approx

## Copy-pasted II&T code from https://github.com/roman-corgi/cgi_iit_drp/blob/main/proc_cgi_frame_NTR/proc_cgi_frame/gsw_remove_cosmics.py ##

def remove_cosmics(image, fwc, sat_thresh, plat_thresh, cosm_filter):
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

## Run tests ##

def test_iit_vs_corgidrp():
    """
    Generate mock raw data ('SCI' & 'ENG') and pass into prescan processing function. 
    Check output dataset shapes, maintain pointers in the Dataset and Image class,
    and check that output is consistent with results II&T code.
    """

    tol = 0.01

    # TODO: Make sure these values make sense
    fwc_em = 100
    fwc_pp = 100
    em_gain = 2000

    fwc = np.min(fwc_em, fwc_pp*em_gain)
    sat_thresh = 0.99
    plat_thresh = 0.85
    cosm_filter = 2

    ###### create simulated data
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    
    # create simulated data
    dataset = mocks.create_cr_dataset(filedir=datadir, numfiles=2,em_gain=em_gain)

    iit_masks = []


    # II&T version
    for frame in dataset:
        
        cr_mask = remove_cosmics(frame, fwc=fwc, 
                                 sat_thresh=sat_thresh, 
                                 plat_thresh=plat_thresh, 
                                 cosm_filter=cosm_filter)
        
        iit_masks.append(cr_mask)
        

def test_crs_zeros_frame():
    """Verify detect_cosmics does not break for a frame of all zeros 
    (should return all zeros)."""
    
    tol = 1e-13

    ###### create simulated data
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    
    for obstype in ['SCI', 'ENG']:
        # create simulated data
        dataset = mocks.create_prescan_files(filedir=datadir, obstype=obstype,numfiles=1)

        # Overwrite data with zeros
        dataset.all_data[:,:,:] = 0.

        output_dataset = detect_cosmic_rays(dataset)

        if output_dataset.all_dq != approx(0,abs=tol):
            raise Exception(f'Operating on all zero frames did not return all zero dq mask.')

# TODO: test tophat above cr threshold
# TODO: test tophat above plateau threshold but below cr threshold
# TODO: test tophat below plateau threshold
# TODO: test double tophat (one > CR thresh, one b/n CR and plateau thresh)
# TODO: test that fwc_em and fwc_pp are saved to ext headers
# TODO: test that sat_fwc is calculated correctly
# TODO: test that consistent em_gain assertion error works correctly


if __name__ == "__main__":
    test_prescan_sub()
    test_bias_zeros_frame()
    test_bias_hvoff()
    test_bias_hvon()
    test_bias_uniform_value()
    test_bias_offset()
