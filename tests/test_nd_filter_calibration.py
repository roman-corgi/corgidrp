import os
from pathlib import Path
import pytest
import math
import numpy as np
from astropy.io import fits
from corgidrp.nd_filter_calibration import main
from corgidrp.data import Dataset, Image
from corgidrp.mocks import create_default_headers
import astropy.units as u

# From fluxcal.py bright standards
bright_stars = ['109 Vir', 'Vega', 'Eta Uma', 'Lam Lep', 'KSI2 CETI']

# From fluxcal.py dim standards
dim_stars = [
    'TYC 4433-1800-1',
    'TYC 4205-1677-1',
    'TYC 4212-455-1',
    'TYC 4209-1396-1',
    'TYC 4413-304-1',
    'UCAC3 313-62260',
    'BPS BS 17447-0067',
    'TYC 4424-1286-1',
    'GSC 02581-02323',
    'TYC 4207-219-1'
]

def create_flux_image(flux, fwhm, background, nx=1024, ny=1024):
    """
    Create a mock image with a Gaussian source:
    - flux: total flux (e.g. erg/s/cm^2/Å) or arbitrary units
    - fwhm: full width at half maximum of the Gaussian source in pixels
    - background: background level in the same units as flux
    - nx, ny: image size in pixels

    Returns:
        Image: a corgidrp.data.Image object
    """

    # Create an empty image with background
    data = np.full((ny, nx), background, dtype=float)

    # Compute Gaussian parameters
    x0 = nx/2
    y0 = ny/2
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    # Normalize Gaussian so that the integral over all pixels equals the desired total flux
    # The integral of a 2D Gaussian = 2 * pi * sigma_x * sigma_y * peak
    # Here sigma_x = sigma_y = sigma
    # flux = peak * 2 * pi * sigma^2
    # peak = flux / (2 * pi * sigma^2)
    peak = flux / (2 * math.pi * sigma**2)

    y_indices, x_indices = np.indices((ny, nx))
    r2 = (x_indices - x0)**2 + (y_indices - y0)**2
    gaussian = peak * np.exp(-r2/(2*sigma**2))
    data += gaussian

    # Create error and DQ arrays (if needed)
    # For now, set a uniform error and no data quality flags:
    err = np.sqrt(np.abs(data))  # Poisson-like error
    dq = np.zeros((ny, nx), dtype=int)

    pri_hdr, ext_hdr = create_default_headers()
    image = Image(data, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err=err, dq=dq)

    return image

def save_image_to_fits(image, filename):
    """
    Save an Image object to a FITS file following the given recipe:
    - Primary HDU with pri_hdr (no data)
    - Second HDU as an ImageHDU with data and ext_hdr
    """
    primary_hdu = fits.PrimaryHDU(header=image.pri_hdr)
    image_hdu = fits.ImageHDU(data=image.data, header=image.ext_hdr)

    hdul = fits.HDUList([primary_hdu, image_hdu])
    hdul.writeto(filename, overwrite=True)


def mock_dim_dataset_files(output_path):
    """
    Create 10 dim star images using the names from dim_stars in fluxcal.py
    These are without ND filter and serve as calibration references.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Simulate dim stars, pick a faint flux:
    star_flux = (30.0 * u.STmag).to(u.erg/u.s/u.cm**2/u.AA)
    fwhm = 3
    background = star_flux.value / 20

    file_paths = []
    for star_name in dim_stars:
        flux_image = create_flux_image(star_flux.value, fwhm, background)
        # Update headers
        flux_image.ext_hdr['TARGET'] = star_name
        flux_image.ext_hdr['CFAMNAME'] = '3C'  # filter name must match a known filter 
        flux_image.ext_hdr['FPAM_H'] = 3.0
        flux_image.ext_hdr['FPAM_V'] = 2.5
        flux_image.ext_hdr['EXPTIME'] = 10.0  # Example exposure time

        filename = os.path.join(output_path, f"mock_dim_dataset_{star_name.replace(' ', '_')}.fits")
        save_image_to_fits(flux_image, filename)
        file_paths.append(str(filename))

    return file_paths

def mock_bright_dataset_files(output_path):
    """
    Create 4 sets of 9 bright star images using the names from bright_stars.
    Choose the first 4 from the bright_stars list provided in fluxcal.py.
    Each star gets a 3x3 raster of images.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Simulate bright stars. Pick a random bright flux for now:
    bright_star_flux = (20.0 * u.STmag).to(u.erg/u.s/u.cm**2/u.AA)
    fwhm = 3
    background = bright_star_flux.value / 20

    # Pick the first 4 bright standards
    selected_bright_stars = bright_stars[:4]
    x_offsets = [-10, 0, 10]
    y_offsets = [-10, 0, 10]

    file_paths = []
    for star_name in selected_bright_stars:
        index = 1
        for dy in y_offsets:
            for dx in x_offsets:
                flux_image = create_flux_image(bright_star_flux.value * 10, fwhm, background)
                flux_image.ext_hdr['TARGET'] = star_name
                flux_image.ext_hdr['CFAMNAME'] = '3C'  # same filter as dim stars for consistency
                flux_image.ext_hdr['FPAM_H'] = 3.0
                flux_image.ext_hdr['FPAM_V'] = 2.5
                flux_image.ext_hdr['FSM_X'] = dx
                flux_image.ext_hdr['FSM_Y'] = dy
                flux_image.ext_hdr['EXPTIME'] = 5.0  # shorter exposure for bright sources

                filename = os.path.join(output_path, f"mock_bright_dataset_{star_name.replace(' ', '_')}_{index}.fits")
                save_image_to_fits(flux_image, filename) 
                file_paths.append(str(filename))
                index += 1

    return file_paths


if __name__ == "__main__":
    print('Running test_nd_filter_calibration')
    
    # User paths
    dim_data_path = '/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks_dim'
    bright_data_path = '/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks_bright'
    output_path = '/Users/jmilton/Github/corgidrp/tests/e2e_tests/nd_filter_output'

    dim_files = mock_dim_dataset_files(dim_data_path)
    bright_files = mock_bright_dataset_files(bright_data_path)

    input_dim_dataset = Dataset(dim_files)
    input_bright_dataset = Dataset(bright_files)

    # Run the main ND filter calibration routine
    main(input_dim_dataset, input_bright_dataset, output_path)