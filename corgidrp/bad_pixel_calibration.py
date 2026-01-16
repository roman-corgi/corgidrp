import numpy as np
from corgidrp.data import BadPixelMap, Dataset, DetectorNoiseMaps
from corgidrp.darks import build_synthesized_dark
import corgidrp.check as check

def create_bad_pixel_map(dataset, master_dark, master_flat, dthresh = 5., ffrac = 0.8, fwidth = 32, dark_outputdir=None):
    """
    Compute a fixed bad pixel map for EXCAM from a master dark and flat.


    Args:
        dataset (corgidrp.data.Dataset): A dataset that is ignored by this class. 
        master_dark (corgidrp.data.Dark or corgidrp.data.DetectorNoiseMaps): 
            a dark frame, or, if it is a Noise Map,  a synthesized master is created using 
            calibrated noise maps for the EM gain and exposure time used in master_flat        
        master_flat (corgidrp.data.FlatField): A master flat field object.
        dthresh (float): Number of standard deviations above the mean to threshold for hot pixels. Must be >= 0.
        ffrac (float): Fraction of local mean value below which poorly-functioning pixels are flagged. Must be >=0 and < 1.
        fwidth (int): Number of pixels to include in local mean check with ffrac. Must be >0.
        dark_outputdir (str): if not None, a file directory to save any synthetic darks created from noise maps

    Returns:
        corgi.data.BadPixelMap: A 2-D boolean array the same size as dark and flat, with bad pixels marked as True.
    """
    if isinstance(master_dark, DetectorNoiseMaps):
        noise_map = master_dark
        master_dark = build_synthesized_dark(Dataset([master_flat]), noise_map)
        if dark_outputdir is not None:
            master_dark.save(filedir=dark_outputdir)

    #Extract data 
    dark_data = master_dark.data
    flat_data = master_flat.data

    #Calculate the hot and warm bad pixel maps
    hot_warm_pixels_bool = detect_hot_warm_pixels_from_dark(dark_data, dthresh)
    hot_warm_pixels = np.zeros_like(dark_data,dtype=np.uint8)
    hot_warm_pixels[hot_warm_pixels_bool] = 8 

    #Calculate the cold and dead pixels
    dead_pixels_bool = detect_dead_pixels_from_flat(flat_data, ffrac, fwidth)
    dead_pixels = np.zeros_like(dark_data,dtype=np.uint8) 
    dead_pixels[dead_pixels_bool] = 4

    #Combined the two maps 
    combined_badpixels = np.bitwise_or(hot_warm_pixels,dead_pixels)

    # Merge headers from dark and flat (may differ in BUNIT, EXPTIME, EMGAIN_C, KGAINPAR)
    input_dataset = Dataset([master_dark, master_flat])
    pri_hdr, ext_hdr, err_hdr, dq_hdr = check.merge_headers_for_combined_frame(input_dataset, 
                                                                                 allow_differing_keywords={'BUNIT', 'EXPTIME', 'EMGAIN_C', 'KGAINPAR'})

    ext_hdr['DATALVL']  = 'CAL'
    ext_hdr.add_history("Bad Pixel Map created using {} and {}".format(master_dark.filename, master_flat.filename))

    #Make the BadPixelMap object
    badpixelmap = BadPixelMap(combined_badpixels, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=input_dataset)

    return badpixelmap
    
def detect_hot_warm_pixels_from_dark(dark, dthresh):
    """
    
    Detects warm/hot pixels from the dark frame, flagging anything above
    dthresh standard deviations above the mean dark level in the frame. For
    example, dthresh = 5 will flag any pixel > 5 standard deviations above the
    mean of dark.

    Ported from II&T pipeline

    Args:
        dark (array_like): 2-D array containing the master dark frame.
        dthresh (float): Threshold for flagging hot pixels, specified as the number of standard deviations above the mean dark level. Must be >= 0.

    Returns:
        fixedbp_dark (array_like): A 2-D boolean array the same size as the input dark frame, with hot pixels marked as True.
    """

    # Process dark frame
    fixedbp_dark = np.zeros(dark.shape).astype('bool')
    fixedbp_dark[dark > np.mean(dark) + dthresh*np.std(dark)] = True

    return fixedbp_dark

def detect_dead_pixels_from_flat(flat, ffrac, fwidth):
    """
    Compute a fixed bad pixel map from a flat field.

    Detects low- or non-functional pixels from the flat frame, flagging any
    pixels less than ffrac times the local mean flat level.  Flat uses
    local mean as flats may have low-spatial-frequency variations due to e.g.
    fringing or vignetting.  For example, ffrac = 0.8 and fwidth = 32 will
    flag any pixel which is < 80% of the mean value in a 32-pixel box
    centered on the pixel.  (Centration will use FFT rules, where odd-sized
    widths center on the pixel, and even-sized place the pixel to the right of
    center, e.g.:
     odd: [. x .]
     even: [. . x .]
    For boxes near the edge, only the subset of pixels within the box will be
    used for the calculation.

    Ported from II&T pipeline
    
    Args:
        flat (array_like): 2-D array containing the master flat field.
        ffrac (float): Fraction of the local mean value below which pixels are considered poorly-functioning.
        fwidth (int): Width of the box used to calculate the local mean, in pixels.

    Returns:
        fixedbp_flat (array_like): A 2-D boolean array the same size as the input flat frame, with dead pixels marked as True.

    """
    # Check inputs
    # check.twoD_array(flat, 'flat', TypeError)
    # check.real_nonnegative_scalar(ffrac, 'ffrac', TypeError)
    # check.positive_scalar_integer(fwidth, 'fwidth', TypeError)

    # Process flat frame
    fixedbp_flat = np.zeros(flat.shape).astype('bool')
    nrow, ncol = flat.shape
    for r in range(nrow):
        tmpr = np.arange(fwidth) - fwidth//2 + r
        rind = np.logical_and(tmpr >= 0, tmpr < nrow)
        for c in range(ncol):
            tmpc = np.arange(fwidth) - fwidth//2 + c
            cind = np.logical_and(tmpc >= 0, tmpc < ncol)

            # rind/cind removes indices that fall out of flat
            subflat = flat[tmpr[rind], :][:, tmpc[cind]]

            localm = np.mean(subflat)
            if flat[r, c] < ffrac*localm:
                fixedbp_flat[r, c] = True
                pass

            pass
        pass

    return fixedbp_flat
