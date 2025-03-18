import os, glob, copy
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import pyklip.klip as klip
import pyklip.instruments.GPI as GPI
from pyklip.kpp.metrics.crossCorr import calculate_cc

from corgidrp import data
from corgidrp.l4_to_tda import find_source, make_snmap

def simulate_image(psf, image_size=256, fwhm=3.5):
    """
    Simulates an image with random Gaussian noise and generates an S/N map.
    
    Args:
        psf (ndarray): Point Spread Function.
        image_size (int): Size of the image (square).
        fwhm (float): Full Width at Half Maximum of the PSF.

    Returns:
        image_snmap (ndarray): Signal-to-Noise (S/N) map of the image.
    """
    image = np.random.normal(loc=0.0, scale=1.0, size=[image_size, image_size])
    image_snmap = make_snmap(image, psf)

    return image_snmap


def test_find_source(fwhm=3.5, nsigma_threshold=5.0):
    """
    Tests the source detection algorithm by injecting artificial planets into an image,
    applying detection, and evaluating the results.

    Args:
        fwhm (float): Full Width at Half Maximum of the PSF.
        nsigma_threshold (float): Detection threshold in terms of sigma.
    """
    # Define paths for test data
    mockfilepath = os.path.join(os.path.dirname(__file__), 'test_data/')
    input_dataset = data.Dataset(glob.glob(mockfilepath+'example_L1_input.fits'))
    filepath_out = mockfilepath
    os.makedirs(filepath_out, exist_ok=True)

    # Construct the PSF kernel for convolution
    boxsize = int(fwhm * 3)
    boxsize += 1 if boxsize % 2 == 0 else 0 # Ensure an odd box size
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    y, x = np.indices((boxsize, boxsize))
    y -= boxsize // 2 ; x -= boxsize // 2
    psf = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

    # Create a binary mask for convolution with a circular aperture
    distance_map = np.sqrt(x**2 + y**2)  
    idx_psf = np.where( (distance_map <= fwhm*0.5) )   
    psf_forconv = np.zeros_like(psf) ; psf_forconv[idx_psf] = 1

    # Simulate an image with noise and generate an S/N map
    image = simulate_image(psf, fwhm=fwhm)

    # Generate a distance map from the image center
    xx, yy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    center = np.floor( image.shape ) // 2 
    distance_map = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)

    # Define inner and outer working angles (IWA & OWA) based on GPI scale
    dataset_owa = 0.45 / GPI.GPIData.lenslet_scale
    dataset_iwa = 0.14 / GPI.GPIData.lenslet_scale
    image[np.where(distance_map > dataset_owa)] = np.nan
    image[np.where(distance_map < dataset_iwa)] = np.nan
    
    # Center of the dataset
    dataset_center = np.array([image.shape[0] // 2, image.shape[1] // 2])

    
    # Generate contrast profile
    pyklip_contrast = True ; pyklip_contrast = False
    if pyklip_contrast:
        contrast_seps, contrast = klip.meas_contrast(
            image, dataset_iwa, dataset_owa, fwhm, center=dataset_center, low_pass_filter=0.
        )
    else:
        
        image_convo = calculate_cc(image, psf_forconv, nans2zero=True)
        contrast_seps = np.arange(dataset_iwa, dataset_owa, fwhm)
        contrast = np.zeros_like(contrast_seps)
        for i in range( len(contrast_seps)-1 ):
            idx = np.where( (distance_map >= contrast_seps[i]) & (distance_map < contrast_seps[i+1]) )
            contrast[i] = np.nanstd(image_convo[idx])
        contrast_seps = contrast_seps[:-1] ; contrast = contrast[:-1]

    f = interp1d(contrast_seps, contrast)

    
    # Initialize arrays to store detection results
    detection = np.empty((0, 6))
    nondetection, misdetection = np.empty((0, 3)), np.empty((0, 3))
    n_threshold, n_source = 20, 3
    n_total = n_threshold // n_source

    # Inject artificial planets and run the detection process
    for _ in range(n_threshold // n_source):
        iwa = dataset_iwa
        if contrast_seps[0] > iwa:
            iwa = contrast_seps[0]
        owa = dataset_owa
        if contrast_seps[-1] < owa:
            owa = contrast_seps[-1]
        radius_rand = np.random.uniform(iwa, owa, n_source)

        pa_rand = np.random.uniform(0, 360, n_source)
        sn_rand = np.random.uniform(5, 10, n_source)
        inputflux_rand = f(radius_rand) * sn_rand / np.nansum(psf[idx_psf])
        x_rand = radius_rand * np.cos(np.radians(pa_rand)) + dataset_center[1] ; x_rand = np.array(x_rand, dtype=int)
        y_rand = radius_rand * np.sin(np.radians(pa_rand)) + dataset_center[0] ; y_rand = np.array(y_rand, dtype=int)

        image_copy = copy.deepcopy(image)
        for i in range(n_source):
            psf_window = psf.shape[0] // 2
            image_copy[y_rand[i]-psf_window:y_rand[i]+psf_window+1,
                       x_rand[i]-psf_window:x_rand[i]+psf_window+1] += psf * inputflux_rand[i]
        
        ##### ##### #####
        # Run the source detection algorithm
        sn_source, xy_source = find_source(image_copy, psf=psf, fwhm=fwhm, nsigma_threshold=4.)#nsigma_threshold)

        # Store detected sources in FITS header
        for i in range(len(sn_source)):
            input_dataset[0].ext_hdr[f'snyx{i:03d}'] = f'{sn_source[i]:5.1f},{xy_source[i][0]:4d},{xy_source[i][1]:4d}'
        # names of the header keywords are tentative
        ##### ##### #####

        # Extract detected sources from FITS header
        header = input_dataset[0].ext_hdr
        snyx = np.array([list(map(float, header[key].split(','))) for key in header if key.startswith("SNYX")])

        # Compute distances between injected and detected sources
        distance_matrix = cdist(np.column_stack((y_rand, x_rand)), snyx[:,1:])
        idx1, idx2 = np.where( (distance_matrix <= 5) )

        # Categorize results into detections, non-detections, and misdetections
        detection = np.vstack((detection, np.hstack((np.column_stack((sn_rand, y_rand, x_rand))[idx1], snyx[idx2]))))
        nondetection = np.vstack((nondetection, np.delete(np.column_stack((sn_rand, y_rand, x_rand)), idx1, axis=0)))
        misdetection = np.vstack((misdetection, np.delete(snyx, idx2, axis=0)))

        
    dx = detection[:,4] - detection[:,1]
    assert np.nanmean(dx) < 1., 'x-coordinates of simulated sources could not be recovered'
    
    dy = detection[:,5] - detection[:,2]
    assert np.nanmean(dy) < 1., 'y-coordinates of simulated sources could not be recovered'

    assert len(detection) > n_total*0.5, 'Many sources were missed' 

    return detection


if __name__ == '__main__':
    test_find_source()
