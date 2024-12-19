import os
from pathlib import Path
import pytest
import re
import numpy as np
from astropy.io import fits
from corgidrp.nd_filter_calibration import main
from corgidrp.data import Dataset
from corgidrp.mocks import create_default_headers 

#@pytest.fixture
def mock_dataset_files(tmp_path):
    """
    Create 9 mock FITS files with both primary and extension headers,
    each 2200x1200 pixels, containing a small 3x3 "Gaussian-like" bright spot.
    """
    # Dimensions of the image
    width = 2200
    height = 1200

    # Define a simple 3x3 Gaussian-like spot pattern
    spot = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 5.0, 2.0],
        [1.0, 2.0, 1.0]
    ])

    # Choose a base center position (somewhere near the middle of the image)
    base_x = 1100
    base_y = 600

    # Offsets for the 3x3 grid of positions
    x_offsets = [-10, 0, 10]
    y_offsets = [-10, 0, 10]

    # List to hold the file paths
    file_paths = []

    # Create 9 files corresponding to the 3x3 grid
    index = 0
    for dy in y_offsets:
        for dx in x_offsets:
            # Create an empty image
            image_data = np.zeros((height, width), dtype=float)

            # Compute the spot center for this file
            x_center = base_x + dx
            y_center = base_y + dy

            # Place the 3x3 spot pattern in the image
            y_min = y_center - 1
            y_max = y_center + 2
            x_min = x_center - 1
            x_max = x_center + 2
            image_data[y_min:y_max, x_min:x_max] = spot

            # Create default headers
            pri_hdr, ext_hdr = create_default_headers()  # Get primary and extension headers
            ext_hdr['FPAM_H'] = 3.0
            ext_hdr['FPAM_V'] = 2.5

            # Create PrimaryHDU with primary header
            primary_hdu = fits.PrimaryHDU(header=pri_hdr)

            # Create ImageHDU with image data and extension header
            image_hdu = fits.ImageHDU(data=image_data, header=ext_hdr)

            # Combine into an HDUList and write to file
            hdul = fits.HDUList([primary_hdu, image_hdu])
            filename = os.path.join(tmp_path, f"test_dataset_{index}.fits")
            hdul.writeto(filename, overwrite=True)

            file_paths.append(str(filename))  # Store the filename
            index += 1

    return file_paths

def test_sweet_spot_dataset_run(input_dataset, tmp_path):
    """
    Test running the sweet_spot_dataset main function on a mock dataset.
    """
    flux_calibration = 1.0
    visit_id = 'PPPPPCCAAASSSOOOVVV'

    # Pattern to match the existing files in the directory
    pattern = re.compile(rf"CGI_{visit_id}_(\d+)_NDF_CAL\.fits")

    max_serial = 0
    for filename in os.listdir(tmp_path):
        match = pattern.match(filename)
        if match:
            current_serial = int(match.group(1))
            if current_serial > max_serial:
                max_serial = current_serial

    # Increment the serial number by 1, format to three digits if needed
    serial_number = f"{max_serial + 1:03d}"

    # Construct output filename with the new serial number
    output_filename = f"CGI_{visit_id}_{serial_number}_NDF_CAL.fits"
    output_file = os.path.join(tmp_path, output_filename)

    # Run the main function
    main(input_dataset, flux_calibration, str(output_file))

    # Optionally, open the output and check its contents
    with fits.open(output_file) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
        # Check shape of data (M x 3)
        # Since we only made one test file, M=1
        assert data.shape[1] == 3, "Data should have 3 columns (OD, x_center, y_center)."
        # Check that FPAM_POS keyword is present


if __name__ == "__main__":
    print('Running test_nd_filter_calibration')
    dataset_filepaths = Dataset(mock_dataset_files('/Users/jmilton/Github/corgidrp/corgidrp/data/nd_filter_mocks'))
    test_sweet_spot_dataset_run(dataset_filepaths, '/Users/jmilton/Github/corgidrp/tests/e2e_tests/nd_filter_output')
    
