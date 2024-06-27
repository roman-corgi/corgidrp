"""
Function to assemble a master dark frame from calibrated subcomponents
"""
import numpy as np

import corgidrp.check as check
from corgidrp.detector import slice_section, detector_areas
from corgidrp.data import Dark

def build_dark(F, D, C, g, t, d_areas=detector_areas, full_frame=False):
    """
    Assemble a master dark SCI frame (full or image)
    from individual noise components.

    This is done this way because the actual dark frame varies with gain and
    exposure time, and both of these values may vary over orders of magnitude
    in the course of acquisition, alignment, and HOWFSC.  Better to take data
    sets that don't vary.

    Output is a bias-subtracted, gain-divided master dark in electrons.
    (Bias is inherently subtracted as we don't use it as
    one of the building blocks to assemble the dark frame.)

    Master dark = (F + g*t*D + g*C)/g = F/g + t*D + C

    Arguments:
     F: corgidrp.data.NoiseMap instance.  This contains a per-pixel map of
        fixed-pattern noise in electrons.  The values may be positive or
        negative.
     D: corgidrp.data.NoiseMap instance.  This contains a per-pixel map
        of dark current noise in electrons per second.
        Each array element should be >= 0.
     C: corgidrp.data.NoiseMap instance.  This contains a per-pixel map of
        clock-induced charge in electrons.
        Each array element should be >= 0.
     g: Desired EM gain, >= 1.  Unitless.
     t: Desired exposure time in seconds.  >= 0.
     d_areas: dict.  A dictionary of detector geometry properties.
        Keys should be as found in detector_areas in detector.py. Defaults to
        detector_areas in detector.py.
     full_frame: bool.  If True, a full-frame master dark is generated (which
        may be useful for the module that statistically fits a frame to find
        the empirically applied EM gain, for example). If False, an image-area
        master dark is generated.  Defaults to False.

    Returns:
     master_dark:  corgidrp.data.Dark instance.
        This contains the master dark in electrons.

    """
    F = F.copy()
    D = D.copy()
    C = C.copy()
    # Check inputs
    Fd = F.data
    Dd = D.data
    Cd = C.data
    check.real_scalar(g, 'g', TypeError)
    check.real_nonnegative_scalar(t, 't', TypeError)

    rows = d_areas['SCI']['frame_rows']
    cols = d_areas['SCI']['frame_cols']
    if Fd.shape != (rows, cols):
        raise TypeError('F must be ' + str(rows) + 'x' + str(cols))
    if Dd.shape != (rows, cols):
        raise TypeError('D must be ' + str(rows) + 'x' + str(cols))
    if Cd.shape != (rows, cols):
        raise TypeError('C must be ' + str(rows) + 'x' + str(cols))
    if (Dd < 0).any():
        raise TypeError('All elements of D must be >= 0')
    if (Cd < 0).any():
        raise TypeError('All elements of C must be >= 0')
    if g < 1:
        raise TypeError('Gain must be a value >= 1.')

    if not full_frame:
        Fd = slice_section(Fd, d_areas, 'SCI', 'image')
        Dd = slice_section(Dd, d_areas, 'SCI','image')
        Cd = slice_section(Cd, d_areas, 'SCI', 'image')
        Ferr = slice_section(F.err[0], d_areas, 'SCI', 'image')
        Derr = slice_section(D.err[0], d_areas, 'SCI', 'image')
        Cerr = slice_section(C.err[0], d_areas, 'SCI', 'image')
        Fdq = slice_section(F.dq, d_areas, 'SCI', 'image')
        Ddq = slice_section(D.dq, d_areas, 'SCI', 'image')
        Cdq = slice_section(C.dq, d_areas, 'SCI', 'image')
    else:
        Ferr = F.err[0]
        Cerr = C.err[0]
        Derr = D.err[0]
        Fdq = F.dq
        Ddq = D.dq
        Cdq = C.dq

    # get from one of the noise maps and modify as needed
    prihdr = F.pri_hdr
    exthdr = F.ext_hdr
    errhdr = F.err_hdr
    exthdr['NAXIS1'] = Fd.shape[0]
    exthdr['NAXIS2'] = Fd.shape[1]
    exthdr['DATATYPE'] = 'Dark'
    exthdr['CMDGAIN'] = g
    exthdr['EXPTIME'] = t
    # wipe clean so that the proper documenting occurs for dark
    exthdr.pop('DRPNFILE')
    exthdr.pop('HISTORY')
    input_data = [F, C, D]
    md_data = Fd/g + t*Dd + Cd
    md_noise = np.sqrt(Ferr**2/g**2 + t**2*Derr**2 + Cerr**2)
    # DQ values are 0 or 1 for F, D, and C
    FDdq = np.logical_or(Fdq, Ddq)
    FDCdq = np.logical_or(FDdq, Cdq)

    master_dark = Dark(md_data, prihdr, exthdr, input_data, md_noise, FDCdq,
                       errhdr)

    return master_dark