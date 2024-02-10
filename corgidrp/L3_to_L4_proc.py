#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process EMCCD images from L3 to L4."""

import numpy as np

class L3_to_L4_proc(object):
    """Process EMCCD frames from L3 to L4.

    distortion-corrected final image in photoelectrons per second with adequate 
    information to do the final conversion to photons
    
    --
    Use cases:
        - TTR5 Analog Non-PSF-subtracted on-sky observation data 
        (These are non-calibration data on astrophysical objects 
         (eg: unocculted star)
        - TTR5 Photon-Counted Non-PSF-subtracted On-Sky Observation Data
    
    final combined image for entire observing sequence in a given configuration.

    input units: photoelectrons / second / pixel

    output units: photoelectrons / second / pixel

    PS+CPP-delivered GSW algorithm

    apply distortion correction to frames and bad pixel maps, if requested by user
    align images and per-frame bad pixel maps to North Up / East Left
    Combine frames (mean or median as per user preference) to produce single final image
    Mean-combine per-frame bad pixel maps to produce single final bad pixel map
    Report/provide:
        combined frame
        combined bad pixel map
        frame selection criteria and frames included in analysis
    associate applicable data sets:
        absolute flux calibration
    --
    
    --
    Use cases:
        - TTR5 Analog PSF-subtracted on-sky data
        - TTR5 Photon-counted PSF-subtracted on-sky data
    
    final combined image for entire observing sequence. PSF-subtracted, if applicable.

    input units: photoelectrons / second / pixel

    output units: photoelectrons / second / pixel

    PS+CPP-delivered GSW algorithm

    For each N ≥ 2 L3 data sets, taken from target and/or reference at Roll A and/or B:

        apply distortion correction to frames and bad pixel maps, if requested by user
        estimate location of star under coronagraph from satellite spot data and/or other CGI telemetry

    For all the N ≥ 2 data sets together:

        perform PSF subtraction (eg: RDI); this step may involve frame combination (mean or median as per user preference) if the data is low-SNR
        align PSF-subtracted images and corresponding per-frame bad pixel maps to North Up / East Left
        Combine all PSF-subtracted target frames (mean or median as per user preference) to produce single final image
        Mean-combine all PSF-subtracted target per-frame bad pixel maps to produce single final bad pixel map
        report/provide:
            combined frame
            combined bad pixel map
            PSF subtraction algorithm throughput for point sources as a function of separation
            frame selection criteria and frames included in analysis
            star location estimate 
            associate applicable data sets:
                associated core throughput correction map
                associated absolute flux calibration
    
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
    
    
    