"""
Simple pipeline runner that creates mock L1 input files.

This script demonstrates the basic workflow for:
1. Creating mock L1 input frames
2. (Future) Running them through pipeline stages

Usage:
    python run_pipeline.py
"""

import os
import numpy as np
import astropy.io.fits as fits
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.detector as detector

def create_mock_l1_files(output_dir, num_files=3):
    """
    Create mock L1 input files for pipeline testing.

    Args:
        output_dir (str): Directory where L1 files will be saved
        num_files (int): Number of mock L1 files to create

    Returns:
        list: List of file paths for created L1 files
    """
    print(f"Creating {num_files} mock L1 files in {output_dir}...")

    # Create output directory if it doesn't exist
    l1_dir = os.path.join(output_dir, 'input_l1')
    os.makedirs(l1_dir, exist_ok=True)

    l1_files = []

    for i in range(num_files):
        # Create default L1 headers
        pri_hdr, ext_hdr = mocks.create_default_L1_headers(arrtype="SCI")

        # Create simple mock image data (using test detector geometry for speed)
        # Using the smaller test geometry: 120x220 instead of full 1200x2200
        frame_rows = mocks.detector_areas_test['SCI']['frame_rows']
        frame_cols = mocks.detector_areas_test['SCI']['frame_cols']

        # Create mock data with some gaussian noise and a constant background
        image_data = np.random.normal(loc=1000, scale=50, size=(frame_rows, frame_cols))

        # Add some "stars" (bright spots) to make it more realistic
        if i == 0:
            # Add a bright source in the first frame
            image_data[60:70, 150:160] += 5000

        # Update headers with correct dimensions
        ext_hdr['NAXIS1'] = frame_cols
        ext_hdr['NAXIS2'] = frame_rows

        # Create HDU list and write to file
        pri_hdu = fits.PrimaryHDU(header=pri_hdr)
        img_hdu = fits.ImageHDU(data=image_data.astype(np.float32), header=ext_hdr)
        hdul = fits.HDUList([pri_hdu, img_hdu])

        # Create filename and save
        filename = os.path.join(l1_dir, f"mock_l1_frame_{i:03d}.fits")
        hdul.writeto(filename, overwrite=True)
        l1_files.append(filename)

        print(f"  Created: {filename}")

    return l1_files


def main():
    """
    Main pipeline execution function.
    """
    # Set up output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'pipeline_output')

    print("="*60)
    print("CORGIDRP Pipeline Runner")
    print("="*60)

    # Step 1: Create mock L1 input files
    l1_files = create_mock_l1_files(output_dir, num_files=3)

    print(f"\nSuccessfully created {len(l1_files)} L1 files:")
    for f in l1_files:
        print(f"  - {f}")

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)

    # Future steps will go here:
    # Step 2: Run L1 -> L2a processing
    # Step 3: Run L2a -> L2b processing
    # etc.


if __name__ == "__main__":
    main()
