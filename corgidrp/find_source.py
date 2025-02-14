import numpy as np
from pyklip.kpp.metrics.crossCorr import calculate_cc
from pyklip.kpp.stat.statPerPix_utils import get_image_stat_map_perPixMasking
from corgidrp.data import Dataset

def find_source(input_dataset: Dataset, psf=None, fwhm=3.5, nsigma_threshold=5.0):
    """
    Detect sources in a given coronagraphic or non-coronagraphic image above a specified SNR threshold.
    
    Args:
        input_dataset (Dataset): Input dataset containing images.
        psf (np.ndarray, optional): 2D PSF array. If None, a Gaussian PSF is generated.
        fwhm (float, optional): Full-width at half-maximum of the PSF.
        nsigma_threshold (float, optional): SNR threshold for detection.
    
    Returns:
        Dataset: The input dataset updated with detected sources.
    """
    
    # Generate a Gaussian PSF if none is provided
    if psf is None:
        boxsize = int(fwhm * 3)
        boxsize += 1 if boxsize % 2 == 0 else 0 # Ensure an odd box size
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
        y, x = np.indices((boxsize, boxsize))
        y -= boxsize // 2 ; x -= boxsize // 2
        psf = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
           
    # Compute the distance map from the PSF center 
    x = np.arange(psf.shape[1]) ; y = np.arange(psf.shape[0])
    xx, yy = np.meshgrid(x, y)
    center = np.array(psf.shape) // 2
    distance_map = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
    idx_masked = np.where( (distance_map < fwhm*.5) ) # Mask central region
    
    # Compute the cross-correlation between the image and the PSF
    image_residual = np.copy(input_dataset.data)
    image_convo = calculate_cc(image_residual, psf, nans2zero=True)
    image_snmap = get_image_stat_map_perPixMasking(image_convo)
    
    sn_source, xy_source = [], []
       
    # Iteratively find sources above the SNR threshold
    while np.nanmax(image_snmap) >= nsigma_threshold:
        sn = np.nanmax(image_snmap)
        xy = np.unravel_index(np.nanargmax(image_snmap), image_snmap.shape)
        
        if sn > nsigma_threshold:
            sn_source.append(sn)
            xy_source.append(xy)

            # Scale and subtract the detected PSF from the image
            psf_window = psf.shape[0] // 2
            psf_scaled = psf * np.nanmedian(image_residual[
                xy[0]-psf_window:xy[0]+psf_window+1,
                xy[1]-psf_window:xy[1]+psf_window+1] / psf)
            psf_scaled[idx_masked] = np.nan # Apply masking
            image_residual[
                xy[0]-psf_window:xy[0]+psf_window+1,
                xy[1]-psf_window:xy[1]+psf_window+1] -= psf_scaled

            show = True ; show = False # will be removed
            if show:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(10,10)) ; cmap = "bwr"
                plt.subplot(2, 2, 3)
                plt.imshow(np.flip(image_snmap[80:-80,80:-80], 0), clim=(-5,5), cmap=cmap)
                
            # Update the SNR map after source removal
            image_convo = calculate_cc(image_residual, psf, nans2zero=True)
            image_snmap = get_image_stat_map_perPixMasking(image_convo)

            if show: # will be removed
                print(sn, xy)
                clim = [np.nanmedian(image_residual)-np.nanstd(image_residual)*nsigma_threshold,
                        np.nanmedian(image_residual)+np.nanstd(image_residual)*nsigma_threshold]
                plt.subplot(2, 2, 1)
                plt.imshow(np.flip(input_dataset.data[80:-80,80:-80], 0), clim=(clim[0],clim[1]), cmap=cmap)
                plt.subplot(2, 2, 2)
                plt.imshow(np.flip(image_residual[80:-80,80:-80], 0), clim=(clim[0],clim[1]), cmap=cmap)
                plt.subplot(2, 2, 4)
                plt.imshow(np.flip(image_snmap[80:-80,80:-80], 0), clim=(-5,5), cmap=cmap)
                plt.show()
                
    # Store detected sources in FITS header
    for i in range(len(sn_source)):
        input_dataset[0].pri_hdr[f'snyx{i:03d}'] = f'{sn_source[i]:5.1f},{xy_source[i][0]:4d},{xy_source[i][1]:4d}'
    
    return input_dataset
