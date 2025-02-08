import os
import math
import numpy as np
from astropy.io import fits
from corgidrp.nd_filter_calibration import compute_expected_band_irradiance
from corgidrp.nd_filter_calibration import main
from corgidrp.data import Dataset, Image
from corgidrp.mocks import create_default_headers, create_flux_image
import astropy.units as u
import matplotlib.pyplot as plt

# From fluxcal.py bright standards
#bright_stars = ['109 Vir', 'Vega', 'Eta Uma', 'Lam Lep', 'KSI2 CETI']
print("for debugging")
bright_stars = ['109 Vir']

# From fluxcal.py dim standards
dim_stars = ['TYC 4433-1800-1']
'''
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
'''

def mock_dim_dataset_files(output_path, dim_exptime, filter, cal_factor):
    """
    Create FITS files for dim star images to serve as calibration references.

    For each dim star name in the global `dim_stars` list, a mock image is
    created, its headers are updated with calibration metadata, and the image
    is saved to a FITS file in the specified output directory.

    Args:
        output_path (str): The directory where the mock dim star FITS files 
            will be saved.
        dim_exptime (float): Exposure time the dim star dataset files
            would be taken at.
        filter (str): The CFAM filter used.

    Returns:
        file_paths (str): A list of file paths to the generated FITS files.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    dim_star_images = []
    for star_name in dim_stars:
        # Simulate dim stars using a faint flux.
        star_flux = compute_expected_band_irradiance(star_name, filter)
        total_dim_flux = star_flux * dim_exptime
        fwhm = 3

        flux_image = create_flux_image(total_dim_flux, fwhm, cal_factor, filter, star_name,
                                       fsm_x = 0, fsm_y = 0, exptime = dim_exptime,
                                       filedir=output_path, color_cor = 1., platescale=21.8, 
                                       add_gauss_noise=True, noise_scale=1., file_save=False)

        dim_star_images.append(flux_image)

    return dim_star_images


def mock_bright_dataset_files(output_path, bright_exptime, filter, OD, cal_factor):
    """
    Create FITS files for bright star images arranged in a raster pattern.

    For the first four bright stars in the global `bright_stars` list, a
    3x3 raster (total of 9 images per star) is created with slight positional
    offsets. Each image is saved as a FITS file with updated calibration
    headers.

    Args:
        output_path (str): The directory where the mock bright star FITS files
            will be saved.
        bright_exptime (float): Exposure time the bright star dataset files
            would be taken at.
        filter (str): The CFAM filter used.
        OD: The optical density of the ND filter.

    Returns:
        file_paths (str): A list of file paths to the generated FITS files.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    ND_transmission = 10**(-OD)
    
    # Pick the first 4 bright standards.
    selected_bright_stars = bright_stars[:4]
    x_offsets = [-10, 0, 10]
    y_offsets = [-10, 0, 10]

    bright_star_images = []
    for star_name in selected_bright_stars:
        for dy in y_offsets:
            for dx in x_offsets:
                bright_star_flux = compute_expected_band_irradiance(star_name, filter)
                total_bright_flux = bright_star_flux * bright_exptime
                attenuated_flux = total_bright_flux * ND_transmission
                fwhm = 3

                flux_image = create_flux_image(attenuated_flux, fwhm, cal_factor, filter, star_name,
                                               fsm_x = dx, fsm_y = dy, exptime = dim_exptime,
                                               filedir=output_path, color_cor = 1., platescale=21.8, 
                                               add_gauss_noise=True, noise_scale=1., file_save=False)

                bright_star_images.append(flux_image)

    return bright_star_images


if __name__ == "__main__":
    print('Running test_nd_filter_calibration')

    # User paths for dim and bright mock datasets and output.
    dim_data_path = (
        '/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks_dim'
    )
    bright_data_path = (
        '/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks_bright'
    )
    output_path = (
        '/Users/jmilton/Github/corgidrp/tests/e2e_tests/nd_filter_output'
    )

    # Observation settings
    dim_exptime = 10.0
    bright_exptime = 5.0
    filter = '3C'
    OD = 2.75
    cal_factor = 0.2

    input_dim_dataset = Dataset(mock_dim_dataset_files(dim_data_path, dim_exptime, filter, 
                                       cal_factor))
    input_bright_dataset = Dataset(mock_bright_dataset_files(bright_data_path, bright_exptime,
                                             filter, OD, cal_factor))
    

    # Run the main ND filter calibration routine.
    main(input_dim_dataset, input_bright_dataset, output_path)

   