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
from corgidrp.mocks import create_default_L3_headers

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

def plot_results(image, nsigma_threshold,
                 detection, detection_lowsn,
                 nondetection, nondetection_lowsn,
                 misdetection):
    
    fig = plt.figure(figsize=(8,8)) ; cmap = "bwr"

    # plot 1
    plt.subplot(2, 2, 1)
    clim = [np.nanmedian(image)-np.nanstd(image)*nsigma_threshold,
            np.nanmedian(image)+np.nanstd(image)*nsigma_threshold]
    plt.imshow(np.flip(image, 0), clim=(clim[0],clim[1]), cmap=cmap)
    plt.title('An example with artificial sources')

    # plot 2
    plt.subplot(2, 2, 2)
    for i in range( len(detection) ):
        plt.plot([detection[i,1], detection[i,4]], [detection[i,2], detection[i,5]])
        
    for i in range( len(nondetection) ):
        if i == 0: plt.plot(nondetection[i,1], nondetection[i,2], marker='d', markersize=12, label='Non-detect')
        else: plt.plot(nondetection[i,1], nondetection[i,2], marker='d', markersize=12)
    
    for i in range( len(misdetection) ):
        if i == 0: plt.plot(misdetection[i,1], misdetection[i,2], marker='x', markersize=12, label='Mis-detect')
        else: plt.plot(misdetection[i,1], misdetection[i,2], marker='x', markersize=12)

    x = np.arange(image.shape[1]) ; y = np.arange(image.shape[0])
    xx, yy = np.meshgrid(x, y)
    center = np.array(image.shape) // 2
    distance_map = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
    idx = np.where( (np.isnan(image) == False) )
    iwa = np.nanmin(distance_map[idx]) ; owa = np.nanmax(distance_map[idx])

    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta) ; y = np.sin(theta)
    plt.plot(iwa*x+center[1], iwa*y+center[0], label='IWA')
    plt.plot(owa*x+center[1], owa*y+center[0], label='OWA')
    
    n_total = len(detection) + len(detection_lowsn) + len(nondetection) + len(nondetection_lowsn)
    txt = 'Detection (low input-SNR): '+str(len(detection)).zfill(2)+'('+str(len(detection_lowsn)).zfill(2)+')/'+str(n_total).zfill(2)+'\n' 
    txt = txt+'Non-detection (low input-SNR): '+str(len(nondetection)).zfill(2)+'('+str(len(nondetection_lowsn)).zfill(2)+')/'+str(n_total).zfill(2)+'\n'
    txt = txt+'Mis-detection: '+str(len(misdetection)).zfill(2)
    plt.title(txt)
    plt.legend()
            
    # plot 3
    dx = detection[:,4] - detection[:,1]
    dy = detection[:,5] - detection[:,2]
    h_x, hlocs = np.histogram(dx, bins=20, range=[-3, 3])
    h_y, hlocs = np.histogram(dy, bins=20, range=[-3, 3])
    plt.subplot(2, 2, 3)
    plt.step(hlocs[1:], h_x, label='delta_x')
    plt.step(hlocs[1:], h_y, label='delta_y')
    plt.xlabel('Separation [pix]')
    plt.legend()

    # plot 4
    plt.subplot(2, 2, 4)
    plt.scatter(detection[:,0], detection[:,3], label='Detection')
    plt.scatter(detection_lowsn[:,0], detection_lowsn[:,3], label='Detection (low input-SNR)')        
    plt.plot(np.arange(0,100), np.arange(0,100))
    plt.hlines(nsigma_threshold, 0., 10., linestyle='dashed')
    plt.vlines(nsigma_threshold, 0., np.nanmax(detection[:,3])+1, linestyle='dashed')
    plt.xlim(0, 10)
    plt.ylim(0., np.nanmax(detection[:,3])+1)
    plt.xlabel('SN_injected') ; plt.ylabel('SN_detected')
    
    plt.show()
    plt.cla() ; plt.clf() ; plt.close()

    return

    
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
    image_convo = calculate_cc(image, psf_binarymask, nans2zero=True)
     
    pyklip_contrast = True #; pyklip_contrast = False
    if pyklip_contrast:
        contrast_seps, contrast = klip.meas_contrast(
            image_convo, dataset_iwa, dataset_owa, 1., center=dataset_center, low_pass_filter=False)
        contrast /= 5.
    else:
        image_convo = calculate_cc(image, psf_binarymask, nans2zero=True)
        contrast_seps = np.arange(dataset_iwa, dataset_owa, fwhm)
        contrast = np.zeros_like(contrast_seps)
        for i in range( len(contrast_seps)-1 ):
            idx = np.where( (distance_map >= contrast_seps[i]) & (distance_map < contrast_seps[i+1]) )
            contrast[i] = np.nanstd(image_convo[idx])
        contrast_seps = contrast_seps[:-1] ; contrast = contrast[:-1]

    f = interp1d(contrast_seps, contrast)


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
        sn_rand = np.random.uniform(3, 10, n_source)
        inputflux_rand = f(radius_rand) * sn_rand / np.nansum(psf[idx_psf])
        x_rand = radius_rand * np.cos(np.radians(pa_rand)) + dataset_center[1] ; x_rand = np.array(x_rand, dtype=int)
        y_rand = radius_rand * np.sin(np.radians(pa_rand)) + dataset_center[0] ; y_rand = np.array(y_rand, dtype=int)

        image_copy = image.copy()
        for i in range(n_source):
            psf_window = psf.shape[0] // 2
            image_copy[y_rand[i]-psf_window:y_rand[i]+psf_window+1,
                       x_rand[i]-psf_window:x_rand[i]+psf_window+1] += psf * inputflux_rand[i]
        
        # Create input data with a simulated image and a sample header
        pri_hdr, _, errhdr, dqhdr = create_default_L3_headers()
        input_dataset[0].ext_hdr['CRPIX1'] = image_copy.shape[1]/2
        input_dataset[0].ext_hdr['CRPIX2'] = image_copy.shape[0]/2
        image_with_point_source = data.Image(image_copy,pri_hdr=pri_hdr,ext_hdr=input_dataset[0].ext_hdr)
        image_with_point_source.data = image_copy

        ##### ##### #####
        # Run the source detection algorithm
        nsigma_threshold = 5.
        image_with_point_source = find_source(image_with_point_source, psf=psf, fwhm=fwhm, nsigma_threshold=nsigma_threshold, image_without_planet=image)
        ##### ##### #####

        
        # Extract detected sources from FITS header
        header = image_with_point_source.ext_hdr
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

    # Plot results
    #plot_results(image, nsigma_threshold, detection, detection_lowsn, nondetection, nondetection_lowsn, misdetection)
    dx = np.nanmedian(detection[:,4] - detection[:,1])
    dy = np.nanmedian(detection[:,5] - detection[:,2])
    dsn = np.nanmedian(abs(detection[:,3] - detection[:,0]))
    detection_rate = float(len(detection)) / (len(detection)+len(nondetection))
    
    assert dx < 1., 'dx_mean = '+str(dx)+', x-coordinates of simulated sources could not be recovered'
    assert dy < 1., 'dy_mean = '+str(dy)+', y-coordinates of simulated sources could not be recovered'
    assert dsn < 1.5, 'dSNR_mean = '+str(dsn)+', SNRs of simulated sources could not be recovered'
    assert detection_rate > 0.85, 'detection_rate = '+str(detection_rate)+', Many sources were missed'
    print(dx, dy, dsn, detection_rate)

    return


if __name__ == '__main__':
    test_find_source()
