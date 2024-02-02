#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process EMCCD images from L4 to o TDA (Tech Demo Analysis) products."""

import numpy as np

class L4_to_TDA_proc(object):
    """Process EMCCD frames from L4 to TDA (Tech Demo Analysis) products.

    Final analysis results in calibrated physical units

    --
    Use cases:
        - TTR5 Analog Non-PSF-subtracted on-sky observation data 
        (These are non-calibration data on astrophysical objects 
         (eg: unocculted star)
        - TTR5 Photon-Counted Non-PSF-subtracted On-Sky Observation Data

    Analysis result used to verify TTR5 and write the final tech demo phase report.

    input units: photoelectrons / pixel / second

    PS+CPP-delivered GSW algorithm

    Inputs:
        PSF-subtracted star L4 data product
        unocculted star apparent magnitude (TDA Product)
        absolute flux calibration
        core throughput calibration

    calculate:
        target star apparent magnitude
        Flux Ratio Noise vs separation
    provide:
        assumed spectral type or spectrum of target star

    If a companion is detected, also:

        calculate SNR of the companion
        extract companion apparent magnitude and flux ratio
        report/provide: 
            companion SNR, flux ratio, and apparent magnitude
            assumed spectral type or spectrum of companion

    --
    
    --
    Use cases:
        - TTR5 Analog PSF-subtracted on-sky data
        - TTR5 Photon-counted PSF-subtracted on-sky data

    Analysis result used to verify TTR5 and write the final tech demo phase report.

    input units: photoelectrons / pixel / second

    PS+CPP-delivered GSW algorithm

    Inputs:
        PSF-subtracted star L4 data product
        unocculted star apparent magnitude (TDA Product)
        absolute flux calibration
        core throughput calibration

    calculate:
        target star apparent magnitude
        Flux Ratio Noise vs separation
    provide
        assumed spectral type or spectrum of target star

    If a companion is detected, also:

        calculate SNR of the companion
        companion apparent magnitude and flux ratio
        report/provide: 
            companion SNR, flux ratio, and apparent magnitude
            assumed spectral type or spectrum of companion

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
    
    
    