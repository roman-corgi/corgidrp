from corgidrp.mocks import create_psfsub_dataset,gaussian_array
from corgidrp.l3_to_l4 import do_psf_subtraction
from corgidrp.data import PyKLIPDataset, Image, Dataset
from corgidrp.detector import nan_flags, flag_nans
from corgidrp.klip_fm import meas_klip_thrupt
from scipy.ndimage import shift, rotate
import pytest
import numpy as np

## Helper functions/quantities

def create_circular_mask(h, w, center=None, r=None):
    """Creates a circular mask

    Args:
        h (int): array height
        w (int): array width
        center (list of float, optional): Center of mask. Defaults to the 
            center of the array.
        r (float, optional): radius of mask. Defaults to the minimum distance 
            from the center to the edge of the array.

    Returns:
        np.array: boolean array with True inside the circle, False outside.
    """

    if center is None: # use the middle of the image
        center = (w/2, h/2)
    if r is None: # use the smallest distance between the center and image walls
        r = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= r
    return mask

iwa_lod = 3.
owa_lod = 9.7
d = 2.36 #m
lam = 573.8e-9 #m
pixscale_arcsec = 0.0218

iwa_pix = iwa_lod * lam / d * 206265 / pixscale_arcsec
owa_pix = owa_lod * lam / d * 206265 / pixscale_arcsec

st_amp = 100.
noise_amp=1e-11
pl_contrast=1e-4

stamp = gaussian_array(array_shape=[20,20],sigma=1,amp=100.,xoffset=0.,yoffset=0.)

ct_calibration = { # Assume these are in 
    'dx' : [10,20,30,10,20,30,10,20,30],
    'dy' : [10,10,10,20,20,20,30,30,30],
    'psfs' : [
        stamp,
        stamp,
        stamp,
        stamp,
        stamp,
        stamp,
        stamp,
        stamp,
        stamp
    ],
    'input_flux' : 1.
}

## pyKLIP data class tests

def test_meas_klip_ADI():
    """Tests that psf subtraction step can correctly split an input dataset into
    science and reference dataset, if they are not passed in separately.
    """

    numbasis = [1,4,8]
    inject_contrast = 1e-7
    rolls = [60,70]
    mock_sci,_ = create_psfsub_dataset(2,0,rolls,
                                              st_amp=st_amp,
                                              noise_amp=noise_amp,
                                              pl_contrast=pl_contrast)
    


    meas_klip_thrupt(mock_sci, # pre-psf-subtracted dataset
                     ct_calibration,
                     inject_contrast,
                     numbasis,
                     seps=[20.], # in pixels from mask center
                     pas=[30.], # Degrees
                     )

    # See if it runs
    pass

if __name__ == '__main__':  
    test_meas_klip_ADI()
