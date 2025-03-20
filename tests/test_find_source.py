import os, glob, copy
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import pyklip.klip as klip
from pyklip.kpp.metrics.crossCorr import calculate_cc

from corgidrp import data
from corgidrp.l4_to_tda import find_source
from corgidrp.find_source import make_snmap

def simulate_image(image_size=64):
    """
    Simulates an image with random Gaussian noise
    
    Args:
        image_size (int): Size of the image (square).

    Returns:
        image (ndarray): an image with random Gaussian noise
    """
    image = np.random.normal(loc=0.0, scale=1.0, size=[image_size, image_size])

    return image

def generate_angles(n_random, lower=0, upper=360, min_diff=30):
    
    angles = []
    while len(angles) < n_random:
        candidate = np.random.uniform(lower, upper)
        if all(min(abs(candidate - a), 360 - abs(candidate - a)) >= min_diff for a in angles):
            angles.append(candidate)
    
    return np.array(angles)

def test_find_source(fwhm=2.8, nsigma_threshold=5.0):
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
    psf_binarymask = np.zeros_like(psf) ; psf_binarymask[idx_psf] = 1

    # Simulate an image with noise and generate an S/N map
    image = simulate_image()

    # Generate a distance map from the image center
    xx, yy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    center = np.floor( image.shape ) // 2 
    distance_map = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)

    # Define inner and outer working angles (IWA & OWA) based on Roman/CGI scale
    pixel_scale = 21.8 # mas
    dataset_owa = 0.45 / (pixel_scale / 1e3)
    dataset_iwa = 0.14 / (pixel_scale / 1e3)
    #fwhm = np.degrees(1.2 * 0.6*1e-6 / 2.4) * 60.**2. * 1e3 / pixel_scale ; print(fwhm)
    image[np.where(distance_map > dataset_owa)] = np.nan
    image[np.where(distance_map < dataset_iwa)] = np.nan
    
    # Center of the dataset
    dataset_center = np.array([image.shape[0] // 2, image.shape[1] // 2])

    
    # Generate contrast curve
    pyklip_contrast = True ; pyklip_contrast = False
    if pyklip_contrast:
        contrast_seps, contrast = klip.meas_contrast(
            image, dataset_iwa, dataset_owa, fwhm, center=dataset_center, low_pass_filter=0.
        )
    else:
        image_convo = calculate_cc(image, psf_binarymask, nans2zero=True)
        contrast_seps = np.arange(dataset_iwa, dataset_owa, fwhm)
        contrast = np.zeros_like(contrast_seps)
        for i in range( len(contrast_seps)-1 ):
            idx = np.where( (distance_map >= contrast_seps[i]) & (distance_map < contrast_seps[i+1]) )
            contrast[i] = np.nanstd(image_convo[idx])
        contrast_seps = contrast_seps[:-1] ; contrast = contrast[:-1]

    f = interp1d(contrast_seps, contrast)

    
    # Create input data with a simulated image and a sample header
    Image = data.Image
    Image.data = image ; Image.ext_hdr = input_dataset[0].ext_hdr

    # Initialize arrays to store detection results
    detection, detection_lowsn = np.empty((0, 6)), np.empty((0, 6))
    nondetection, nondetection_lowsn, misdetection = np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
    n_loop, n_source = 50, 3
    n_total = n_loop * n_source

    # Inject artificial planets and run the detection process
    for _ in range(n_loop):
        iwa = dataset_iwa
        if contrast_seps[0] > iwa:
            iwa = contrast_seps[0]
        owa = dataset_owa
        if contrast_seps[-1] < owa:
            owa = contrast_seps[-1]
        radius_rand = np.random.uniform(iwa, owa, n_source)

        #pa_rand = np.random.uniform(0, 360, n_source)
        pa_rand = generate_angles(n_source)
        sn_rand = np.random.uniform(0, 10, n_source)
        #while np.max(sn_rand) < 5.: sn_rand = np.random.uniform(0, 10, n_source)
        inputflux_rand = f(radius_rand) * sn_rand / np.nansum(psf[idx_psf])
        x_rand = radius_rand * np.cos(np.radians(pa_rand)) + dataset_center[1] ; x_rand = np.array(x_rand, dtype=int)
        y_rand = radius_rand * np.sin(np.radians(pa_rand)) + dataset_center[0] ; y_rand = np.array(y_rand, dtype=int)

        image_copy = copy.deepcopy(image)
        for i in range(n_source):
            psf_window = psf.shape[0] // 2
            image_copy[y_rand[i]-psf_window:y_rand[i]+psf_window+1,
                       x_rand[i]-psf_window:x_rand[i]+psf_window+1] += psf * inputflux_rand[i]
        Image.data = image_copy

        
        ##### ##### #####
        # Run the source detection algorithm
        nsigma_threshold = 5.
        find_source(Image, psf=psf, fwhm=fwhm, nsigma_threshold=nsigma_threshold)
        ##### ##### #####

        
        # Extract detected sources from FITS header
        header = Image.ext_hdr
        snyx = np.array([list(map(float, header[key].split(','))) for key in header if key.startswith("SNYX")])

        if len(snyx) > 0:
        
            # Compute distances between injected and detected sources
            distance_matrix = cdist(np.column_stack((y_rand, x_rand)), snyx[:,1:])
            #idx1, idx2 = np.where( (distance_matrix <= fwhm) )
            idx1 = [] ; idx2 = []
            yx_input = np.column_stack((y_rand, x_rand))
            for i in range(yx_input.shape[0]):
                idx = np.argmin(distance_matrix[i])
                if distance_matrix[i,idx] < fwhm:
                    idx1.append(i) ; idx2.append(idx)

            # Categorize results into detections, detections (low input-SNR), non-detections, non-detections (low input-SNR), and misdetections
            snyx_det = np.hstack((np.column_stack((sn_rand, y_rand, x_rand))[idx1], snyx[idx2]))
            snyx_det_lowsn = snyx_det[ np.where( snyx_det[:,0] < nsigma_threshold ) ]
            detection_lowsn = np.vstack((detection_lowsn, snyx_det_lowsn))
            snyx_det = snyx_det[ np.where( snyx_det[:,0] >= nsigma_threshold ) ]
            detection = np.vstack((detection, snyx_det))

            snyx_nondet = np.delete(np.column_stack((sn_rand, y_rand, x_rand)), idx1, axis=0)
            snyx_nondet_lowsn = snyx_nondet[ np.where( snyx_nondet[:,0] < nsigma_threshold ) ]
            nondetection_lowsn = np.vstack((nondetection_lowsn, snyx_nondet_lowsn))
            snyx_nondet = snyx_nondet[ np.where( snyx_nondet[:,0] >= nsigma_threshold ) ]
            nondetection = np.vstack((nondetection, snyx_nondet))
            
            snyx_misdet = np.delete(snyx, idx2, axis=0)
            misdetection = np.vstack((misdetection, snyx_misdet))

    # Remove duplicate false positives (misdetections) associated with a particular pattern of the simulated image
    idx = []
    seen = []
    for i, row in enumerate(misdetection):
        if not any(np.linalg.norm(row[1:3] - s) <= fwhm for s in seen):
            seen.append(row[1:3])
            idx.append(i)
    misdetection = misdetection[idx]

    
    dx = np.nanmean(detection[:,4] - detection[:,1])
    dy = np.nanmean(detection[:,5] - detection[:,2])
    dsn = np.nanmean(abs(detection[:,3] - detection[:,0]))
    detection_rate = float(len(detection)) / (len(detection)+len(nondetection))

    assert dx < 1., 'x-coordinates of simulated sources could not be recovered'
    assert dy < 1., 'y-coordinates of simulated sources could not be recovered'
    assert dsn < 1.5, 'SNRs of simulated sources could not be recovered'
    assert detection_rate > 0.80, 'Many sources were missed' 

    return


if __name__ == '__main__':
    test_find_source()
