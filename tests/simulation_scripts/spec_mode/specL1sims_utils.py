"""
Utility functions for generating end-to-end (e2e) test data files.

This module contains common utility functions needed by various e2e test
data generation scripts.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def write_png_from_sceneobj(scene, outdir, loc_x, loc_y, output_dim):
    """
    Write a PNG image for a scene's detector image.

    Parameters
    ----------
    scene : corgisim.Scene
        The scene object containing the image_on_detector attribute
    outdir : str
        Output directory path where files will be saved
    loc_x : int
        X-coordinate location on the detector array
    loc_y : int
        Y-coordinate location on the detector array
    output_dim : int
        Dimension (width/height) of the output image in pixels

    Returns
    -------
    L1_png_fname : str
        Path to the saved PNG file
    """
    L1_fitsname = os.path.join(outdir, scene.image_on_detector[0].header['FILENAME'])
    L1_png_fname = str.replace(L1_fitsname, '.fits', '.png')
    plt.figure(figsize=(8,6))
    plt.imshow(scene.image_on_detector[1].data[loc_y + 13 - output_dim//2:loc_y + 13 + output_dim//2,
                                               loc_x + 1088 - output_dim//2:loc_x + 1088 + output_dim//2], origin='lower')
    plt.colorbar()
    plt.savefig(L1_png_fname, dpi=300)
    plt.close()

    return L1_png_fname

def write_png_from_fits(fits_file, loc_x, loc_y, output_dim, ext=1):
    """
    Write a PNG image from a FITS file's detector image.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file containing the detector image
    loc_x : int
        X-coordinate location on the detector array
    loc_y : int
        Y-coordinate location on the detector array
    output_dim : int
        Dimension (width/height) of the output image in pixels
    ext : int, optional
        FITS extension index containing the image array (default=1)

    Returns
    -------
    L1_png_fname : str
        Path to the saved PNG file
    """
    L1_png_fname = str.replace(fits_file, '.fits', '.png')
    
    with fits.open(fits_file, mode='readonly') as hdul:
        image_data = hdul[ext].data
        
    plt.figure(figsize=(8,6))
    plt.imshow(image_data[loc_y + 13 - output_dim//2:loc_y + 13 + output_dim//2,
                          loc_x + 1088 - output_dim//2:loc_x + 1088 + output_dim//2], origin='lower')
    plt.colorbar()
    plt.savefig(L1_png_fname, dpi=300)
    plt.close()

    return L1_png_fname

def write_headers_to_text(fits_files):
    """
    Write header text files for a list of FITS images, storing them with the same name and location,
    replacing the .fits extenstion with .txt. 

    Parameters
    ----------
    fits_files : list of str
        List of FITS files to extract headers from

    Returns
    -------
    header_txt_fname_list : list of str
        List of paths to the saved header text files
    """
    header_txt_fname_list = []
    for fits_file in fits_files:
        header_txt_fname = str.replace(fits_file, '.fits', '_header.txt')
        header_txt_fname_list.append(header_txt_fname)
        with fits.open(fits_file, mode='readonly') as hdul:
            with open(header_txt_fname, 'w') as f:
                f.write(repr(hdul[0].header))
                f.write('\n')
                f.write(repr(hdul[1].header))

    return header_txt_fname_list

def update_fits_headers(fits_files, header_updates):
    """
    Update FITS header values for a list of FITS files.

    Parameters
    ----------
    fits_files : list of str
        List of FITS file paths to update
    header_updates : list of tuple
        Nested list where each item is a (key, value) tuple.
        key is the header keyword string, value is the new value to set.

    Returns
    -------
    None
        Files are modified in place

    Examples
    --------
    >>> update_fits_headers(['file1.fits', 'file2.fits'], 
    ...                     [('VISTYPE', 'CGIVST_CAL_SPEC_TGTREF'), ('EXPTIME', 10.0)])
    """
    for fits_file in fits_files:
        with fits.open(fits_file, mode='update') as hdul:
            for key, value in header_updates:
                hdul[0].header[key] = value
            hdul.flush()

def average_L1_images(fits_files, outdir, ext=1, median=False):
    """
    Compute the mean or median of an image stack from a list of FITS files.

    Parameters
    ----------
    fits_files : list of str
        List of FITS file paths to average
    outdir : str
        Output directory to store the result in
    ext : int, optional
        FITS extension index containing the image array (default=1)
    median : bool, optional
        If True, compute median instead of mean (default=False)

    Returns
    -------
    output_fname : str
        Path to the saved averaged FITS file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Load all images into a stack
    image_stack = []
    for fits_file in fits_files:
        with fits.open(fits_file, mode='readonly') as hdul:
            image_stack.append(hdul[ext].data)
    
    # Convert to numpy array for efficient computation
    image_stack = np.array(image_stack)
    
    # Compute mean or median
    if median:
        averaged_image = np.median(image_stack, axis=0)
        suffix = '_med.fits'
    else:
        averaged_image = np.mean(image_stack, axis=0)
        suffix = '_mean.fits'
    
    # Extract subdirectory name from first file
    first_file_dir = os.path.dirname(fits_files[0])
    subdir_name = os.path.basename(first_file_dir)
    
    # Construct output filename
    output_fname = os.path.join(outdir, subdir_name + suffix)
    
    # Create FITS file with the averaged image
    # Copy header from first file and update
    with fits.open(fits_files[0], mode='readonly') as hdul:
        primary_hdu = fits.PrimaryHDU(header=hdul[0].header)
        image_hdu = fits.ImageHDU(data=averaged_image, header=hdul[ext].header)
        hdu_list = fits.HDUList([primary_hdu, image_hdu])
        hdu_list.writeto(output_fname, overwrite=True)
    
    return output_fname

def threshold_sum_L1_images(fits_files, outdir, ext=1, thresh=None, nsigma=5.0,
                            noise_box_x_start=100, noise_box_y_start=100, noise_box_width=800):
    """
    Apply a counting threshold to images in a stack and compute the sum of thresholded images.

    Parameters
    ----------
    fits_files : list of str
        List of FITS file paths to process
    outdir : str
        Output directory to store the result in
    ext : int, optional
        FITS extension index containing the image array (default=1)
    thresh : float, optional
        Counting threshold in DN units. If None, threshold is computed from
        noise statistics (default=None)
    nsigma : float, optional
        Number of standard deviations above the mean to set the counting
        threshold when thresh=None (default=5.0)
    noise_box_x_start : int, optional
        Array column index of left edge of noise measurement box (default=100)
    noise_box_y_start : int, optional
        Array row index of bottom edge of noise measurement box (default=100)
    noise_box_width : int, optional
        Width of noise measurement box (default=800)

    Returns
    -------
    output_fname : str
        Path to the saved sum of thresholded count image FITS file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # If threshold not provided, compute from noise statistics
    if thresh is None:
        # Load first image to compute noise statistics
        with fits.open(fits_files[0], mode='readonly') as hdul:
            first_image = hdul[ext].data
        
        # Extract noise measurement box
        noise_box = first_image[noise_box_y_start:noise_box_y_start + noise_box_width,
                                noise_box_x_start:noise_box_x_start + noise_box_width]
        
        # Compute statistics
        mean_noise = np.mean(noise_box)
        stddev_noise = np.std(noise_box)
        
        # Set threshold
        thresh = mean_noise + nsigma * stddev_noise
        print(f"Computed threshold: {thresh:.2f} DN (mean={mean_noise:.2f}, stddev={stddev_noise:.2f}, nsigma={nsigma})")
    else:
        print(f"Using provided threshold: {thresh:.2f} DN")
    
    # Initialize accumulator for thresholded counts
    thresholded_sum = None
    
    # Process each FITS file
    for fits_file in fits_files:
        with fits.open(fits_file, mode='readonly') as hdul:
            image_data = hdul[ext].data
        
        # Apply binary threshold filter (use integer type)
        binary_count = (image_data >= thresh).astype(np.int32)
        
        # Add to rolling sum
        if thresholded_sum is None:
            thresholded_sum = binary_count.astype(np.int32)
        else:
            thresholded_sum += binary_count
    
    # Keep the sum as integer array (no division by number of images)
    sum_thresholded_count = thresholded_sum
    
    # Extract subdirectory name from first file
    first_file_dir = os.path.dirname(fits_files[0])
    subdir_name = os.path.basename(first_file_dir)
    
    # Construct output filename
    suffix = '_threshsum.fits'
    output_fname = os.path.join(outdir, subdir_name + suffix)
    
    # Create FITS file with the sum of thresholded count image
    # Copy header from first file and update
    with fits.open(fits_files[0], mode='readonly') as hdul:
        primary_hdu = fits.PrimaryHDU(header=hdul[0].header)
        image_hdu = fits.ImageHDU(data=sum_thresholded_count, header=hdul[ext].header)
        
        # Add threshold information to header
        image_hdu.header['THRESH'] = (thresh, 'Counting threshold in DN units')
        image_hdu.header['NSIGMA'] = (nsigma, 'Number of sigma for threshold computation')
        image_hdu.header['NIMAGES'] = (len(fits_files), 'Number of images in stack')
        
        hdu_list = fits.HDUList([primary_hdu, image_hdu])
        hdu_list.writeto(output_fname, overwrite=True)
    
    return output_fname

def get_L1_config_dict(fits_file, file_trunc=None, descrip=None, calib_product=None, header_keys=None):
    """
    Extract simulation configuration information from an L1 FITS file into a dictionary.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file to extract configuration from
    file_trunc : str, optional
        Truncated file path to store in the dictionary. If None, uses the
        basename of fits_file (default=None)
    descrip : str, optional
        Description string to store in the dictionary. If None, uses an
        empty string (default=None)
    calib_product : str, optional
        Data calibration product type
    header_keys : list of tuple, optional
        List of (ext, key) tuples defining which header values to extract.
        ext is the FITS extension index, key is the header keyword string.
        If None, only File and Description are included (default=None)

    Returns
    -------
    config_dict : dict
        Dictionary containing the configuration information with at minimum
        'File' and 'Description' keys, plus any additional header values
        specified in header_keys

    Examples
    --------
    >>> config = get_L1_config_dict('path/to/image.fits', 
    ...                             descrip='Test observation',
    ...                             header_keys=[(0, 'VISTYPE'), (1, 'EXPTIME')])
    >>> print(config)
    {'File': 'image.fits', 'Description': 'Test observation', 
     'VISTYPE': 'CGIVST_CAL_SPEC_TGTREF', 'EXPTIME': 10.0}
    """
    # Initialize the configuration dictionary with minimum required fields
    config_dict = {}
    
    # Set file_trunc to basename if not provided
    if file_trunc is None:
        file_trunc = os.path.basename(fits_file)
    config_dict['File'] = file_trunc
    
    # Set description to empty string if not provided
    if descrip is None:
        descrip = ''
    config_dict['Description'] = descrip

    # Set description to empty string if not provided
    if calib_product is None:
        calib_product = ''
    config_dict['Calibration Product'] = calib_product
    
    # Extract additional header values if specified
    if header_keys is not None:
        with fits.open(fits_file, mode='readonly') as hdul:
            for ext, key in header_keys:
                try:
                    config_dict[key] = hdul[ext].header[key]
                except (KeyError, IndexError) as e:
                    # If header key or extension doesn't exist, store None
                    config_dict[key] = None
                    print(f"Warning: Could not extract {key} from extension {ext}: {e}")
    
    return config_dict
