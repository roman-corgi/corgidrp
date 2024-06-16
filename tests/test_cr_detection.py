import glob
import os

import corgidrp.data as data
import corgidrp.mocks as mocks
from corgidrp.l1_to_l2a import detect_cosmic_rays
from corgidrp.detector import find_plateaus, calc_sat_fwc

import numpy as np
from astropy.time import Time
from scipy.ndimage import median_filter
from pytest import approx

## Copy-pasted II&T code from https://github.com/roman-corgi/cgi_iit_drp/blob/main/proc_cgi_frame_NTR/proc_cgi_frame/gsw_remove_cosmics.py ##

def find_plateaus_iit(streak_row, fwc, sat_thresh, plat_thresh, cosm_filter):
    """Find the beginning index of each cosmic plateau in a row.

    Note that i_beg is set at one pixel before first plateau pixel, as these
    pixels immediately neighboring the cosmic plateau are very often affected
    by the cosmic hit as well.

    Args:
        streak_row (float):
            Row with possible cosmics.
        fwc (float): 
            Full well capacity of detector *in DNs*.  Note that this may require a
            conversion as FWCs are usually specified in electrons, but the image
            is in DNs at this point.
        sat_thresh (float): 
            Multiplication factor for fwc that determines saturated cosmic pixels.
        plat_thresh (float): 
            Multiplication factor for fwc that determines edges of cosmic plateu.
        cosm_filter (int): 
            Minimum length in pixels of cosmic plateus to be identified.

    Returns:
        i_beg (array_like, int): 
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

def remove_cosmics_iit(image, fwc, sat_thresh, plat_thresh, cosm_filter):
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

    Args:
        image (array_like, float): 
            Image area of frame (bias of zero).
        fwc (float): 
            Full well capacity of detector *in DNs*.  Note that this may require a
            conversion as FWCs are usually specified in electrons, but the image
            is in DNs at this point.
        sat_thresh (float): 
            Multiplication factor for fwc that determines saturated cosmic pixels.
        plat_thresh (float): 
            Multiplication factor for fwc that determines edges of cosmic plateu.
        cosm_filter (int): 
            Minimum length in pixels of cosmic plateus to be identified.

    Returns:
        mask (array_like, int): 
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
        i_beg = find_plateaus_iit(row, fwc, sat_thresh, plat_thresh, cosm_filter)

        # If plateaus exist, kill the hit and the rest of the row
        if i_beg is not None:
            mask[i, i_beg:] = 1
            pass

    return mask

## Run tests ##

###### create simulated data
datadir = os.path.join(os.path.dirname(__file__), "simdata")

detector_params = data.DetectorParams({}, date_valid=Time("2023-11-01 00:00:00"))

def test_iit_vs_corgidrp():
    """
    Generate mock raw data ('SCI' & 'ENG') and pass into prescan processing function. 
    Check output dataset shapes, maintain pointers in the Dataset and Image class,
    and check that output is consistent with results II&T code.
    """

    kgain = 8.7
    fwc_em_e = 100000
    fwc_pp_e = 90000
    em_gain = 500

    fwc_em = fwc_em_e / kgain
    fwc_pp = fwc_pp_e / kgain
    fwc = np.min([fwc_em, fwc_pp*em_gain])
    sat_thresh = 0.99
    plat_thresh = 0.85
    cosm_filter = 2
    
    # create simulated data
    dataset = mocks.create_cr_dataset(filedir=datadir, numfiles=2,em_gain=em_gain, numCRs=5, plateau_length=10)

    iit_masks = []


    # II&T version
    for frame in dataset:
        
        cr_mask = remove_cosmics_iit(frame.data, fwc=fwc, 
                                 sat_thresh=sat_thresh, 
                                 plat_thresh=plat_thresh, 
                                 cosm_filter=cosm_filter)
        
        iit_masks.append(cr_mask)
    iit_masks_arr = np.array(iit_masks)

    # corgidrp version
    crmasked_dataset = detect_cosmic_rays(dataset, detector_params)
    corgi_crmask_bool = np.where(crmasked_dataset.all_dq>0,1,0)

    if not corgi_crmask_bool == approx(iit_masks_arr):
        raise Exception(f'Corgidrp and II&T functions do not result in the same CR masks.')

def test_crs_zeros_frame():
    """Verify detect_cosmics does not break for a frame of all zeros 
    (should return all zeros)."""
    
    tol = 1e-13

    # create simulated data
    dataset = mocks.create_cr_dataset(filedir=datadir, numfiles=2,numCRs=5, plateau_length=10)

    # Overwrite data with zeros
    dataset.all_data[:,:,:] = 0.

    output_dataset = detect_cosmic_rays(dataset, detector_params)

    if output_dataset.all_dq != approx(0,abs=tol):
        raise Exception(f'Operating on all zero frames did not return all zero dq mask.')

def test_correct_headers():
    """
    Asserts "FWC_EM_E", "FWC_PP_E", and "SAT_DN" are tracked in the frame headers.
    """
    # create simulated data
    dataset = mocks.create_cr_dataset(filedir=datadir, numfiles=2,numCRs=5, plateau_length=10)
    output_dataset = detect_cosmic_rays(dataset, detector_params)

    for frame in output_dataset:
        if not ("FWC_EM_E" in frame.ext_hdr):
            raise Exception("'FWC_EM_E' missing from frame header.")
        
        if not ("FWC_PP_E" in frame.ext_hdr):
            raise Exception("'FWC_PP_E' missing from frame header.")

        if not ("SAT_DN" in frame.ext_hdr):
            raise Exception("'SAT_DN' missing from frame header.")

def test_saturation_calc():
    """
    Asserts that FWC saturation threshold is calculated correctly.
    """
    sat_thresh = 0.99

    # fwc_em = fwc_pp * em_gain
    fwc_em = np.array([90000,90000])
    fwc_pp = np.array([90,90])
    em_gain = np.array([1000,1000])
    sat_fwcs = calc_sat_fwc(em_gain,fwc_pp,fwc_em,sat_thresh)
    
    expected = fwc_em * sat_thresh
    if not sat_fwcs == approx(expected):
        raise Exception(f"Saturation full-well capacity calculation incorrect when fwc_em = fwc_pp * em_gain. \nReturned {sat_fwcs} when {expected} was expected.")

    # fwc_em < fwc_pp * em_gain
    fwc_em = np.array([50000,50000])
    fwc_pp = np.array([90,90])
    em_gain = np.array([1000,1000])
    sat_fwcs = calc_sat_fwc(em_gain,fwc_pp,fwc_em,sat_thresh)
    
    expected = fwc_em * sat_thresh
    if not sat_fwcs == approx(expected):
        raise Exception(f"Saturation full-well capacity calculation incorrect when fwc_em < fwc_pp * em_gain. \nReturned {sat_fwcs} when {expected} was expected.")

    # fwc_em > fwc_pp * em_gain
    fwc_em = np.array([90000,90000])
    fwc_pp = np.array([50,50])
    em_gain = np.array([1000,1000])
    sat_fwcs = calc_sat_fwc(em_gain,fwc_pp,fwc_em,sat_thresh)
    
    expected = fwc_pp * em_gain * sat_thresh
    if not sat_fwcs == approx(expected):
        raise Exception(f"Saturation full-well capacity calculation incorrect when fwc_em > fwc_pp * em_gain. \nReturned {sat_fwcs} when {expected} was expected.")
    
    # fwc_em > fwc_pp * em_gain for first frame, fwc_em < fwc_pp * em_gain for second frame
    fwc_em = np.array([90000,50000])
    fwc_pp = np.array([500,90])
    em_gain = np.array([100,1000])
    sat_fwcs = calc_sat_fwc(em_gain,fwc_pp,fwc_em,sat_thresh)
    
    expected = np.array([49500.,49500.])
    if not sat_fwcs == approx(expected):
        raise Exception(f"Saturation full-well capacity calculation incorrect when frames have different fwc_em, fwc_pp, em_gain. \nReturned {sat_fwcs} when {expected} was expected.")


## Useful constructs from JPL II&T unit tests:

fwc = 90000
cosm_filter = 2
plat_thresh = 0.85
sat_thresh = 0.99

# Create a bias subtracted image with cosmics that cover all corner cases
# Make a variety of plateaus
p_basic = np.array([fwc]*cosm_filter)  # Smallest allowed through filter
p_small = np.array([fwc]*(cosm_filter-1))  # Smaller than filter
p_large = np.array([fwc]*cosm_filter*10)  # Larger than filter
p_dip = np.append(p_basic, [plat_thresh*fwc, fwc])  # Shallow dip mid cosmic
p_dip_deep = np.hstack((p_basic, [0.], p_basic))  # Deep dip mid cosmic
p_uneven = np.array([fwc*sat_thresh, fwc, fwc*plat_thresh, fwc,
                     fwc*plat_thresh])  # Uneven cosmic
p_below_min = np.array([fwc*sat_thresh - 1]*cosm_filter)  # Below min value

# Create tail
# An exponential tail with no noise should be able to be perfectly removed
tail = np.exp(np.linspace(0, -10, 50)) * 0.1*fwc

# Create bias subtracted image

bs_dataset = mocks.create_cr_dataset(datadir,numfiles=1,numCRs=0)
bs_dataset.all_data[:,:,:] = 1.
im_width = bs_dataset.all_data.shape[-1]
i_streak_rows_t = np.array([0, im_width//2-1, im_width//2, im_width-1])
cosm_bs = np.append(p_basic, tail)
not_cosm_bs = np.append(p_below_min, tail)

streak_row = np.ones(bs_dataset.all_data.shape[-1])

bs_dataset_below_thresh = bs_dataset.copy()
bs_dataset_single_pix = bs_dataset.copy()
bs_dataset_two_cosm = bs_dataset.copy()

bs_dataset.all_data[0,i_streak_rows_t[0], 0:len(cosm_bs)] = cosm_bs
bs_dataset.all_data[0,i_streak_rows_t[1], 50:50+len(cosm_bs)] = cosm_bs
bs_dataset.all_data[0,i_streak_rows_t[1], 50+len(cosm_bs):50+len(cosm_bs)*2] = cosm_bs
bs_dataset.all_data[0,i_streak_rows_t[2], 51:51+len(cosm_bs)] = cosm_bs
bs_dataset.all_data[0,i_streak_rows_t[3], im_width-len(p_basic):] = p_basic
bs_dataset_below_thresh.all_data[0,im_width//2, 50:50+len(not_cosm_bs)] = not_cosm_bs
bs_dataset_single_pix.all_data[0,im_width//2, im_width//2] = fwc

    
def test_mask():
    """Assert correct elements are masked."""
    dataset = mocks.create_cr_dataset(filedir=datadir, numfiles=1,numCRs=0, plateau_length=10)
    dataset.all_data[:,:,:] = 1.
    dataset.all_data[0,1, 2:2+len(cosm_bs)] = cosm_bs
    check_mask = np.zeros_like(dataset.all_dq, dtype=int)
    check_mask[0,1, 1:] = 1  # Mask starts 1 before cosmic
    dataset_masked = detect_cosmic_rays(dataset, detector_params)
    if not np.where(dataset_masked.all_dq>0,1,0) == approx(check_mask):
        raise Exception("Incorrect pixels were masked.")
    
def test_i_begs():
    """Verify that function returns correct i_begs result."""
    beg = 50
    streak_row_copy = streak_row.copy()
    streak_row_copy[beg:beg+len(cosm_bs)] = cosm_bs
    i_beg = find_plateaus(streak_row_copy, fwc, sat_thresh,
                            plat_thresh, cosm_filter)
    if not i_beg == beg-1:
        raise Exception("find_plateaus returned incorrect streak beginning.")

def test_left_edge_i_begs():
    """Verify that function returns correct i_begs result at left edge."""
    beg = 0
    streak_row_copy = streak_row.copy()
    streak_row_copy[beg:len(cosm_bs)] = cosm_bs
    i_beg = find_plateaus(streak_row_copy, fwc, sat_thresh,
                            plat_thresh, cosm_filter)
    if not i_beg == beg:
        raise Exception("find_plateaus returned incorrect streak beginning at left edge.")

def test_right_edge_i_begs():
    """Verify that function returns correct i_begs result at right edge."""
    cosm = p_basic
    beg = len(streak_row)-len(cosm)
    streak_row_copy = streak_row.copy()
    streak_row_copy[beg:] = cosm
    i_beg = find_plateaus(streak_row_copy, fwc, sat_thresh,
                            plat_thresh, cosm_filter)
    if not i_beg == beg-1:
        # print(streak_row)
        # print(i_beg,beg)
        raise Exception("find_plateaus returned incorrect streak beginning at right edge.")

def test_two_cosm_i_begs():
    """Verify that function returns correct i_begs result for two cosm."""
    beg1 = len(streak_row) - len(cosm_bs)*2
    beg2 = beg1 + len(cosm_bs)
    streak_row_copy = streak_row.copy()
    streak_row_copy[beg1:beg1 + len(cosm_bs)] = cosm_bs
    streak_row_copy[beg2:beg2 + len(cosm_bs)] = cosm_bs

    i_beg = find_plateaus(streak_row_copy, fwc, sat_thresh,
                            plat_thresh, cosm_filter)
    if not i_beg == beg1-1:
        raise Exception("find_plateaus returned incorrect streak beginning for two CRs in one row.")

def test_p_small():
    """Verify that function ignores plateaus smaller than filter size."""
    cosm = np.append(p_small, tail)
    beg = 50
    streak_row_copy = streak_row.copy()
    streak_row_copy[beg:beg+len(cosm)] = cosm
    i_beg = find_plateaus(streak_row_copy, fwc,
                            sat_thresh, plat_thresh,
                            cosm_filter)
    if not i_beg is None:
        raise Exception("find_plateaus did not ignore plateau smaller than filter size.")

def test_p_large():
    """Verify that function returns correct results for large plateaus."""
    cosm = np.append(p_large, tail)
    beg = 50
    streak_row_copy = streak_row.copy()
    streak_row_copy[beg:beg+len(cosm)] = cosm
    i_beg = find_plateaus(streak_row_copy, fwc,
                            sat_thresh, plat_thresh,
                            cosm_filter)
    if not i_beg == beg-1:
        raise Exception("find_plateaus returned incorrect streak beginning for large plateau.")

def test_p_dip():
    """Verify that function still recognizes a plateau with a dip."""
    cosm = np.append(p_dip, tail)
    beg = 50
    streak_row_copy = streak_row.copy()
    streak_row_copy[beg:beg+len(cosm)] = cosm
    i_beg = find_plateaus(streak_row_copy, fwc,
                            sat_thresh, plat_thresh,
                            cosm_filter)
    if not i_beg == beg-1:
        raise Exception("find_plateaus did not recognize plateau with a dip.")

def test_p_dip_deep():
    """Verify that the function recognizes plateau with a single pixel dip
    below plat_thresh and does not set the end at the dip."""
    cosm = np.append(p_dip_deep, tail)
    beg = 50
    streak_row_copy = streak_row.copy()
    streak_row_copy[beg:beg+len(cosm)] = cosm
    i_beg = find_plateaus(streak_row_copy, fwc,
                            sat_thresh, plat_thresh,
                            cosm_filter)
    if not i_beg == beg-1:
        raise Exception("find_plateaus returned incorrect result for plateau with a deep dip.")

def test_p_uneven():
    """Verify that function still recognizes an uneven plateau."""
    cosm = np.append(p_uneven, tail)
    beg = 50
    streak_row_copy = streak_row.copy()
    streak_row_copy[beg:beg+len(cosm)] = cosm
    i_beg = find_plateaus(streak_row_copy, fwc,
                            sat_thresh, plat_thresh,
                            cosm_filter)
    if not i_beg == beg-1:
        raise Exception("find_plateaus returned incorrect streak beginning for uneven plateau.")

def test_p_below_min():
    """Verify that function ignores plateaus below saturation thresh."""
    cosm = np.append(p_below_min, tail)
    beg = 50
    streak_row_copy = streak_row.copy()
    streak_row_copy[beg:beg+len(cosm)] = cosm
    i_beg = find_plateaus(streak_row_copy, fwc,
                            sat_thresh, plat_thresh,
                            cosm_filter)
    if not i_beg is None:
        raise Exception("find_plateaus did not ignore plateau below sat threshold.")


if __name__ == "__main__":
    test_iit_vs_corgidrp()
    test_crs_zeros_frame()
    test_correct_headers()
    test_saturation_calc()
    test_mask()
    test_i_begs()
    test_left_edge_i_begs()
    test_right_edge_i_begs()
    test_two_cosm_i_begs()
    test_p_small()
    test_p_large()
    test_p_dip()
    test_p_dip_deep()
    test_p_uneven()
    test_p_below_min()