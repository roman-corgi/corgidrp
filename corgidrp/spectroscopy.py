# Spectroscopy functions

#can import astropy, but don't import photutils

#want to load in 2 files (template and data) 
import numpy as np
import os

import corgidrp.data

import astropy
import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord
from astropy.modeling.models import Const2D, Gaussian2D

import pyklip.fakes as fakes

import scipy.ndimage as ndi
import scipy.optimize as optimize
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt


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
    fitsize = frame.shape[0]
    ystamp, xstamp = np.indices([fitsize, fitsize], dtype=float)
    xstamp -= fitsize//2
    ystamp -= fitsize//2
    
    stampsize=frame.shape[0]

    xstamp += stampsize//2 + dx
    ystamp += stampsize//2 + dy
    
    #print(f"shift_psf result shape: {ndi.map_coordinates(frame * flux, [ystamp, xstamp], mode='constant', cval=0.0).ravel().shape}")
    return ndi.map_coordinates(frame * flux, [ystamp, xstamp], mode='constant', cval=0.0).ravel()

def fit_centroid(psf_template, psf_data, guessflux=1, rad=20, stampsize=30):
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

    # Subtract the minimum of the data as a rough background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties. Moments from negative data
    # values can yield undefined Gaussian parameters, e.g., x/y_stddev.

    # Get a rough estimate of the psf_template center of mass (x and y coord)
    y_com, x_com = center_of_mass(psf_template)
    print(y_com,x_com)  
    #pf, fw, x_centroid, y_centroid = fakes.gaussfit2d(frame=psf_template, xguess=x_com, yguess=y_com)

    x_centroid = x_com
    y_centroid = y_com

    # replace frame with psf_template
    frame = psf_template
    plt.imshow(psf_template)
    plt.title('psf_template')

    yind = int(y_centroid)
    xind = int(x_centroid)
    print(xind, yind, 'xind and yind')
        
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
    plt.figure()
    plt.imshow(cutout)
    plt.title('cutout')
    
    xstar, ystar = centroid(cutout)  # 
    xstar += xmin
    ystar += ymin
    
    ### Create a PSF stamp ###
    ystamp, xstamp = np.indices([stampsize, stampsize], dtype=float)
    xstamp -= float(stampsize//2)
    ystamp -= float(stampsize//2)
    xstamp += xstar
    ystamp += ystar
    
    stamp = ndi.map_coordinates(frame, [ystamp, xstamp])  # this image doesn't have psf
    
    ### Create a data stamp ###  - this is from psf_data (in a file test_spectroscopy.py we will open this fits file)
    # I am not sure how else to estimate the offset? Unless I assume it is 0 (which is true in our example case)
    y_com_data, x_com_data = center_of_mass(psf_template)
    #pf, fw, x_centroid_data, y_centroid_data = fakes.gaussfit2d(frame=psf_data, xguess=x_com_data, yguess=y_com_data)
    x_centroid_data = x_com_data
    y_centroid_data = y_com_data

    print(x_centroid_data, y_centroid_data, 'x and y centroid')
    xoffset_guess = x_centroid - x_centroid_data
    yoffset_guess = y_centroid - y_centroid_data

    fitsize = stampsize
    ydata,xdata = np.indices([fitsize, fitsize], dtype=float)
    xdata -= fitsize//2
    ydata -= fitsize//2
    xdata += xstar + xoffset_guess  # not sure what to do for guesses? Or if I should get coords from this image as well?
    ydata += ystar + yoffset_guess

    #print(xdata, ydata)

    #Pull data from psf_data
    data = ndi.map_coordinates(psf_data, [ydata, xdata])
    plt.figure()
    plt.imshow(data)
    plt.title('data')

    plt.figure()
    plt.imshow(stamp)
    plt.title('stamp')

    # print(f"stamp shape: {stamp.shape[0]}")
    # print(f"data shape: {data.shape}")
    # print(data.ravel().shape, 'data ravel shape')

    
    ### Fit the PSF to the data ###
    popt, pcov = optimize.curve_fit(shift_psf, stamp, data.ravel(), p0=(0,0,guessflux))
    tinyoffsets = popt[0:2]

    binary_offset = [xoffset_guess - tinyoffsets[0], yoffset_guess - tinyoffsets[1]]

    return binary_offset
