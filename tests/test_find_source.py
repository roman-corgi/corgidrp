import os, glob, copy
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from astropy.io import fits
import matplotlib.pyplot as plt
import pyklip.klip as klip
import pyklip.instruments.GPI as GPI
import pyklip.fakes as fakes

from corgidrp import data
from find_source import find_source

if __name__ == "__main__":
    # Define file paths for test data
    filepath_test = './tests/test_data/'
    input_dataset = data.Dataset(glob.glob(filepath_test + 'mock_northup.fits'))

    filepath_out = filepath_test
    os.makedirs(filepath_out, exist_ok=True)

    # Set parameters for the source detection process
    psf = None
    fwhm = 3.5
    nsigma_threshold = 5.0

    # Load the dataset from GPI data file
    file_gpi = filepath_test+'NoCompNoDisk-GPIcube.fits'
    dataset = GPI.GPIData(file_gpi)
    
    x = np.arange(dataset.input.shape[1]) ; y = np.arange(dataset.input.shape[2])
    xx, yy = np.meshgrid(x, y)
    center = np.floor( np.array(dataset.input.shape)[1:] // 2. )
    distance_map = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
    idx_owa = np.where( (distance_map > 0.7/GPI.GPIData.lenslet_scale) )
    dataset.input[dataset.input.shape[0]//2][idx_owa] = np.nan
    idx_iwa = np.where( (np.isnan(dataset.input[dataset.input.shape[0]//2]) == False) )
    dataset_iwa = np.min(distance_map[idx_iwa])
    dataset_owa = 0.7 / GPI.GPIData.lenslet_scale
    
    # Compute the dataset center and create an array of repeated centers
    dataset_center = np.array([dataset.input.shape[1] // 2, dataset.input.shape[2] // 2])
    dataset_centers = np.tile(dataset_center, [dataset.input.shape[0], 1])

    # Generate contrast profile
    contrast_seps, contrast = klip.meas_contrast(
        dataset.input[dataset.input.shape[0] // 2], dataset_iwa, dataset_owa, fwhm, center=dataset_center, low_pass_filter=1.0
    )
    f = interp1d(contrast_seps, contrast)

    # Initialize arrays to store detection results
    detection = np.empty((0, 6))
    nondetection, misdetection = np.empty((0, 3)), np.empty((0, 3))
    n_threshold, n_source = 50, 3

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
        inputflux_rand = f(radius_rand) * (sn_rand/5.)
        ##### ##### #####
        inputflux_rand*= 1.5 # somehow needed ?
        ##### ##### #####
        x_rand = radius_rand * np.cos(np.radians(pa_rand)) + dataset_center[1]
        y_rand = radius_rand * np.sin(np.radians(pa_rand)) + dataset_center[0]

        # Create a copy of the dataset and inject fake sources
        dataset_copy = copy.deepcopy(dataset)
        for i in range(n_source):
            fakes.inject_planet(
                dataset_copy.input, dataset_centers, inputflux_rand[i], dataset.wcs, radius_rand[i], pa_rand[i],
                fwhm=fwhm, thetas=np.zeros(dataset.input.shape[0]) + pa_rand[i]
            )
        input_dataset.data = dataset_copy.input[dataset.input.shape[0] // 2]
        
        ##### ##### #####
        # Run the source detection algorithm
        input_dataset = find_source(input_dataset, psf=psf, fwhm=fwhm, nsigma_threshold=nsigma_threshold)
        ##### ##### #####

        # Extract detected sources from FITS header
        header = input_dataset[0].pri_hdr
        snyx = np.array([list(map(float, header[key].split(','))) for key in header if key.startswith("SNYX")])

        # Compute distances between injected and detected sources
        distance_matrix = cdist(np.column_stack((y_rand, x_rand)), snyx[:,1:])
        idx1, idx2 = np.where( (distance_matrix <= 5) )

        # Categorize results into detections, non-detections, and misdetections
        detection = np.vstack((detection, np.hstack((np.column_stack((sn_rand, y_rand, x_rand))[idx1], snyx[idx2]))))
        nondetection = np.vstack((nondetection, np.delete(np.column_stack((sn_rand, y_rand, x_rand)), idx1, axis=0)))
        misdetection = np.vstack((misdetection, np.delete(snyx, idx2, axis=0)))

    # Print summary of detection results
    print('Detection, Non-detection, Mis-detection:', len(detection), len(nondetection), len(misdetection))

    show = True #; show = False
    if show:
        
        fig = plt.figure(figsize=(10,10)) ; cmap = "bwr"

        
        plt.subplot(2, 2, 1)
        image = dataset_copy.input[dataset.input.shape[0]//2]
        clim = [np.nanmedian(image)-np.nanstd(image)*nsigma_threshold,
                np.nanmedian(image)+np.nanstd(image)*nsigma_threshold]
        plt.imshow(np.flip(image[85:-85,85:-85], 0), clim=(clim[0],clim[1]), cmap=cmap)
        plt.title('An example with 3 artificial point sources')

        
        plt.subplot(2, 2, 2)
        for i in range( len(detection) ):
            plt.plot([detection[i,1], detection[i,4]], [detection[i,2], detection[i,5]])

        for i in range( len(nondetection) ):
            if i == 0: label = 'Non-detect'
            else: label = ''
            plt.plot(nondetection[i,1], nondetection[i,2], marker='d', markersize=12, label=label)
        
        for i in range( len(misdetection) ):
            if i == 0: label = 'Mis-detect'
            else: label = ''
            plt.plot(misdetection[i,1], misdetection[i,2], marker='x', markersize=12, label=label)

        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(theta) ; y = np.sin(theta)
        plt.plot(iwa*x+dataset_center[1], iwa*y+dataset_center[0], label='IWA')
        plt.plot(owa*x+dataset_center[1], owa*y+dataset_center[0], label='OWA')
        
        plt.plot(dataset_center[1], dataset_center[0], marker='*', markersize=12)
        
        plt.ylim(dataset_center[0]-dataset_owa*1.1,dataset_center[0]+dataset_owa*1.1)
        plt.xlim(dataset_center[1]-dataset_owa*1.1,dataset_center[1]+dataset_owa*1.1)

        n_total = len(detection) + len(nondetection)
        txt = 'Good:'+str(len(detection)).zfill(2)+'/'+str(n_total).zfill(2)
        txt = txt+', Non:'+str(len(nondetection)).zfill(2)+'/'+str(n_total).zfill(2)
        txt = txt+', Mis:'+str(len(misdetection)).zfill(2)
        plt.title(txt)
        plt.legend()


        dx = detection[:,4] - detection[:,1]
        dy = detection[:,5] - detection[:,2]
        h_x, hlocs = np.histogram(dx, bins=20, range=[-fwhm, fwhm])
        h_y, hlocs = np.histogram(dy, bins=20, range=[-fwhm, fwhm])
        plt.subplot(2, 2, 3)
        plt.step(hlocs[1:], h_x, label='delta_x')
        plt.step(hlocs[1:], h_y, label='delta_y')
        plt.xlabel('Separation [pix]')
        plt.legend()

    
        plt.subplot(2, 2, 4)
        plt.scatter(detection[:,0], detection[:,3])
        plt.plot(np.arange(1,100), np.arange(1,100))
        plt.xlim(np.nanmin(detection[:,0])-2, np.nanmax(detection[:,0])+2)
        plt.ylim(np.nanmin(detection[:,3])-2, np.nanmax(detection[:,3])+2)
        plt.xlabel('SN_injected') ; plt.ylabel('SN_detected')

        plt.show()
        outfile = filepath_out+'test_find_source.png'
        plt.savefig(outfile)
        plt.cla() ; plt.clf() ; plt.close()
