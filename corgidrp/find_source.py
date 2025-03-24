import numpy as np
from pyklip.kpp.metrics.crossCorr import calculate_cc
from pyklip.kpp.stat.statPerPix_utils import get_image_stat_map_perPixMasking

def make_snmap(image, psf_binarymask, image_without_planet=None, coronagraph=True):
    """
    Generates a signal-to-noise (S/N) map by convolving the image with the PSF.
    
    Args:
        image (ndarray): The input image.
        psf_binarymask (ndarray): A binary mask for PSF convolution.
        image_without_planet (ndarray, optional): An image without any sources (~noise map) to make snmap more accurate.
        coronagraph (bool, optional): If True, an IWA is applied to derive the snmap. Defaults to True.
    
    Returns:
        ndarray: The computed S/N map.
    """
    
    image_convo = calculate_cc(image, psf_binarymask, nans2zero=True)

    if image_without_planet is not None:
        image_wop_convo = calculate_cc(image_without_planet, psf_binarymask, nans2zero=True)
    else: image_wop_convo = None
    
    if coronagraph:
        # Compute distance map from PSF center and mask radius from the image.
        x = np.arange(image.shape[1]) ; y = np.arange(image.shape[0])
        xx, yy = np.meshgrid(x, y)
        center = np.array(image.shape) // 2
        distance_map = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
        idx = np.where( (np.isnan(image) == False) )
        mask_radius = np.ceil( np.min(distance_map[idx]) )
    else: mask_radius = 0.

    image_snmap = get_image_stat_map_perPixMasking(image_convo, image_without_planet=image_wop_convo,
                                                   mask_radius=mask_radius, Dr=3.)

    return image_snmap


def psf_scalesub(image, xy, psf, fwhm):
    """
    Scales and subtracts the PSF at a given location in the image.
    
    Args:
        image (ndarray): The input image.
        xy (tuple): The (y, x) coordinates where the PSF should be subtracted.
        psf (ndarray): The point spread function.
        fwhm (float): Full width at half maximum of the PSF.
    
    Returns:
        ndarray: The image after PSF subtraction.
    """
    # Compute distance map from PSF center
    x = np.arange(psf.shape[1]) ; y = np.arange(psf.shape[0])
    xx, yy = np.meshgrid(x, y)
    center = np.array(psf.shape) // 2
    distance_map = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
    
    # Mask central region of the PSF
    idx_masked = np.where( (distance_map < fwhm*1.0) )

    # Scale PSF to match local flux and subtract it
    psf_window = psf.shape[0] // 2
    psf_scaled = psf * np.nanmedian(image[
        xy[0]-psf_window:xy[0]+psf_window+1,
        xy[1]-psf_window:xy[1]+psf_window+1] / psf)
    psf_scaled[idx_masked] = np.nan # Apply masking
    
    image[xy[0]-psf_window:xy[0]+psf_window+1,
          xy[1]-psf_window:xy[1]+psf_window+1] -= psf_scaled
    
    return image