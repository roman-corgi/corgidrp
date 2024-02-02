#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process EMCCD images from L2 to L3."""

import numpy as np

class L2_to_L3_proc(object):
    """Process EMCCD frames from L2 to L3.

    cleaned, calibrated image, converted to photoelectrons per second
    provide astrometric solution for imaging data
    
    --
    Use cases:
        - TTR5 Analog Non-PSF-subtracted on-sky observation data 
        (These are non-calibration data on astrophysical objects 
         (eg: unocculted star)
        - TTR5 Analog PSF-subtracted on-sky data
        - TTR5 Photon-Counted Non-PSF-subtracted On-Sky Observation Data
        - TTR5 Photon-counted PSF-subtracted on-sky data
             
    single cleaned image, normalized by exposure time, with WCS provided 

    input units: photoelectrons / pixel / frame

    output units: photoelectrons / pixel / second

    PS+CPP-delivered GSW algorithm

    construct WCS: distortion correction, plate scale, North and East angles
    normalize by exposure time
    provide:
        frame
        distortion correction
        plate scale
        North and East angles
         
    --
    
    Parameters
    ----------

    Attributes
    ----------
    meta : instance
        Instance of Metadata class containing detector metadata.

    """

    def __init__():
        
        pass
    
    def func1():
        
        pass
    
    def func2():
        
        pass
    
    
    