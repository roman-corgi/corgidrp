# Spectroscopy functions

#can import astropy, but don't import photutils

#want to load in 2 files (template and data) 
import numpy as np
import os

import corgidrp.data

import astropy
import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord

#import pyklip.fakes as fakes

import scipy.ndimage as ndi
import scipy.optimize as optimize


def centroid(frame):
    """
    Finds the center coordinates for a given frame.

    Args:
        frame (np.ndarray): 2D array to compute centering

    Returns:
        tuple:
            xcen (float): X centroid coordinate
            ycen (float): Y centroid coordinate

    """
    y, x = np.indices(frame.shape)
    
    ycen = np.sum(y * frame)/np.sum(frame)
    xcen = np.sum(x * frame)/np.sum(frame)
    
    return xcen, ycen

def shift_psf(frame, dx, dy, flux, fitsize=10, stampsize=10):
    """
    Creates a template for matching psfs.

    Args:
        frame (np.ndarray): 2D array 
        dx (float): Value to shift psf in 'x' direction
        dy (float): Value to shift psf in 'y' direction
        flux (float): Peak flux value
        fitsize (float): (Optional) Width of square on frame to fit psf to
        stampsize (float): (Optional) Width of psf stamp to create

    Returns:
        ndi..map_coordinates: New coordinates for a shifted psf

    """
    ystamp, xstamp = np.indices([fitsize, fitsize], dtype=float)
    xstamp -= fitsize//2
    ystamp -= fitsize//2
    
    xstamp += stampsize//2 + dx
    ystamp += stampsize//2 + dy
    
    return ndi.map_coordinates(frame * flux, [ystamp, xstamp], mode='constant', cval=0.0).ravel()

def fit_centroid(psf_template, psf_data, guessflux=1, rad=5, stampsize=30):
#def measure_offset(frame, xstar_guess, ystar_guess, xoffset_guess, yoffset_guess, guessflux=1, rad=5, stampsize=10):
    """
    Use this code to calculate the psf centroid for the template and psf data.
    
    Args:
        frame (np.ndarray): 2D array of data
        xstar_guess (float): Estimate of first star 'x' coordinate
        ystar_guess (float): Estimate of first star 'y' coordinate
        xoffset_guess (float): Estimate of 'x' direction offset between stars
        yoffset_guess (float): Estimate of 'y' direction offset between stars
        guessflux (float): (Optional) Peak flux of first star
        rad (int): (Optional) Radius around first star to compute centroid on
        stampsize (float): (Optional) Width of square psf stamp

    Returns:
        binary_offset (np.array): List of [x,y] offsets in respective directions
        
    """
    #### Centroid on location of star ###

    # Use a guess first?  - take tools from photutils
    # Make a guess of the centroid of the psf template, then follow this script



    yind = int(ystar_guess)
    xind = int(xstar_guess)
        
    ymin = yind - rad
    ymax = yind + rad + 1
    xmin = xind - rad
    xmax = xind + rad + 1
    # check bounds
    if ymin < 0:
        ymin = 0
        ymax = 2*rad + 1
    if xmin < 0:
        xmin = 0
        xmax = 2*rad + 1
    if ymax > frame.shape[0]:  # frame is the entire image here
        ymax = frame.shape[0]
        ymin = frame.shape[0] - 2 * rad - 1
    if xmax > frame.shape[1]:
        xmax = frame.shape[1] 
        xmin = frame.shape[1] - 2 * rad - 1
        
    cutout = frame[ymin:ymax, xmin:xmax]
    
    xstar, ystar = centroid(cutout)
    xstar += xmin
    ystar += ymin
    
    ### Create a PSF stamp ###
    ystamp, xstamp = np.indices([stampsize, stampsize], dtype=float)
    xstamp -= float(stampsize//2)
    ystamp -= float(stampsize//2)
    xstamp += xstar
    ystamp += ystar
    
    stamp = ndi.map_coordinates(frame, [ystamp, xstamp])
    
    ### Create a data stamp ###
    fitsize = stampsize
    ydata,xdata = np.indices([fitsize, fitsize], dtype=float)
    xdata -= fitsize//2
    ydata -= fitsize//2
    xdata += xstar + xoffset_guess
    ydata += ystar + yoffset_guess
    
    data = ndi.map_coordinates(frame, [ydata, xdata])
    
    ### Fit the PSF to the data ###
    popt, pcov = optimize.curve_fit(shift_psf, stamp, data.ravel(), p0=(0,0,guessflux))
    tinyoffsets = popt[0:2]

    binary_offset = [xoffset_guess - tinyoffsets[0], yoffset_guess - tinyoffsets[1]]

    return binary_offset
