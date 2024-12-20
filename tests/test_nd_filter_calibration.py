import os
from pathlib import Path
import pytest
import numpy as np
from astropy.io import fits
from corgidrp.nd_filter_calibration import main
from corgidrp.data import Dataset
from corgidrp.mocks import create_default_headers 

# @pytest.fixture
def mock_bright_dataset_files(output_path):
    """
      - 4 sets of 9 files for bright stars (ND filter in)
    """
    width = 2200
    height = 1200

    # Simple 3x3 Gaussian spot
    spot = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 5.0, 2.0],
        [1.0, 2.0, 1.0]
    ])

    base_x = 1100
    base_y = 600
    x_offsets = [-10, 0, 10]
    y_offsets = [-10, 0, 10]

    file_paths = []
    star_names = ["Star1", "Star2", "Star3", "Star4"]

    for star_name in star_names:
        index = 1
        for dy in y_offsets:
            for dx in x_offsets:
                image_data = np.zeros((height, width), dtype=float)

                x_center = base_x + dx
                y_center = base_y + dy

                y_min = y_center - 1
                y_max = y_center + 2
                x_min = x_center - 1
                x_max = x_center + 2
                image_data[y_min:y_max, x_min:x_max] = spot

                pri_hdr, ext_hdr = create_default_headers()
                # For demonstration, let's say these files are bright star files with ND filter:
                ext_hdr['FPAM_H'] = 3.0
                ext_hdr['FPAM_V'] = 2.5
                ext_hdr['FSM_X'] = x_center
                ext_hdr['FSM_Y'] = y_center
                ext_hdr['TARGET'] = star_name

                primary_hdu = fits.PrimaryHDU(header=pri_hdr)
                image_hdu = fits.ImageHDU(data=image_data, header=ext_hdr)
                hdul = fits.HDUList([primary_hdu, image_hdu])
                filename = os.path.join(output_path, f"mock_bright_dataset_{star_name}_{index}.fits")
                hdul.writeto(filename, overwrite=True)

                file_paths.append(str(filename))
                index += 1

    return file_paths

def mock_dim_dataset_files(output_path):
    width = 2200
    height = 1200

    # Simple 3x3 Gaussian spot
    spot = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 5.0, 2.0],
        [1.0, 2.0, 1.0]
    ])

    x_center = 1100
    y_center = 600

    y_min = y_center - 1
    y_max = y_center + 2
    x_min = x_center - 1
    x_max = x_center + 2

    file_paths = []
    for index in range(1,11):
        image_data = np.zeros((height, width), dtype=float)
        image_data[y_min:y_max, x_min:x_max] = spot

        pri_hdr, ext_hdr = create_default_headers()

        star_name = f"Star{index}"
        ext_hdr['FPAM_H'] = 3.0
        ext_hdr['FPAM_V'] = 2.5
        ext_hdr['FSM_X'] = 0
        ext_hdr['FSM_Y'] = 0
        ext_hdr['TARGET'] = star_name

        primary_hdu = fits.PrimaryHDU(header=pri_hdr)
        image_hdu = fits.ImageHDU(data=image_data, header=ext_hdr)
        hdul = fits.HDUList([primary_hdu, image_hdu])
        filename = os.path.join(output_path, f"mock_dim_dataset_{star_name}.fits")
        hdul.writeto(filename, overwrite=True)

        file_paths.append(str(filename))
    
    return file_paths


if __name__ == "__main__":
    print('Running test_nd_filter_calibration')
    input_dim_dataset = Dataset(mock_dim_dataset_files('/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks'))
    input_bright_dataset = Dataset(mock_bright_dataset_files('/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks'))
    main(input_dim_dataset, input_bright_dataset,'/Users/jmilton/Github/corgidrp/tests/e2e_tests/nd_filter_output')