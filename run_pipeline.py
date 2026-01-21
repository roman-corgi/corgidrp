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
import astropy.time as time
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector

def save_headers_to_file(fits_file, output_dir, stage_name=""):
    """
    Save all header information from a FITS file to a text file for tracking.

    Args:
        fits_file (str): Path to the FITS file
        output_dir (str): Directory where header info will be saved
        stage_name (str): Optional stage name for organization (e.g., "L1", "L2a")
    """
    headers_dir = os.path.join(output_dir, 'header_tracking')
    if stage_name:
        headers_dir = os.path.join(headers_dir, stage_name)
    os.makedirs(headers_dir, exist_ok=True)

    base_name = os.path.basename(fits_file).replace('.fits', '')
    header_file = os.path.join(headers_dir, f"{base_name}_headers.txt")

    with fits.open(fits_file) as hdul:
        with open(header_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"HEADER INFORMATION FOR: {os.path.basename(fits_file)}\n")
            f.write(f"Total HDUs: {len(hdul)}\n")
            f.write("="*80 + "\n\n")

            # Loop through all HDUs
            for i, hdu in enumerate(hdul):
                # Determine HDU type/name
                if i == 0:
                    hdu_name = "PRIMARY HEADER (HDU 0)"
                else:
                    extname = hdu.header.get('EXTNAME', f'Extension {i}')
                    hdu_name = f"{extname.upper()} HEADER (HDU {i})"

                # Write HDU header
                f.write(f"{hdu_name}\n")
                f.write("-"*80 + "\n")

                # Add data shape information if available
                if hdu.data is not None:
                    f.write(f"Data shape: {hdu.data.shape}\n")
                    f.write(f"Data type: {hdu.data.dtype}\n")
                    f.write("-"*80 + "\n")

                # Write all header keywords with dtype
                for key, value in hdu.header.items():
                    comment = hdu.header.comments[key]
                    dtype = type(value).__name__
                    f.write(f"{key:8s} = {value!s:20s} [{dtype:10s}] / {comment}\n")

                # Separator between HDUs
                if i < len(hdul) - 1:
                    f.write("\n" + "="*80 + "\n\n")

    return header_file


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

        # Create simple mock image data
        # Use full detector frame dimensions (1200x2200 for SCI)
        frame_rows = detector.detector_areas['SCI']['frame_rows']  # 1200
        frame_cols = detector.detector_areas['SCI']['frame_cols']  # 2200

        # Create mock data with some gaussian noise and a constant background
        image_data = np.random.normal(loc=1000, scale=50, size=(frame_rows, frame_cols))

        # Add some "stars" (bright spots) to make it more realistic
        if i == 0:
            # Add a bright source in the first frame
            image_data[500:510, 1100:1110] += 5000

        # Update headers to match actual data dimensions
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


def setup_calibrations():
    """
    Set up calibration database and load calibration files for L1->L2a processing.

    Returns:
        tuple: (caldb instance, calibrations directory path)
    """
    print("\n" + "="*60)
    print("Setting up calibration database...")
    print("="*60)

    # Path to existing calibration data
    calib_data_dir = os.path.join(os.path.dirname(__file__), 'calibration_data')

    # Initialize calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_pipeline_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()

    # Load calibration files from calibration_data directory
    print(f"  Loading calibrations from: {calib_data_dir}")

    # Load NonLinearity calibration
    nonlin_path = os.path.join(calib_data_dir, 'nonlin_table_TVAC.fits')
    if os.path.exists(nonlin_path):
        print(f"    Loading NonLinearity: {os.path.basename(nonlin_path)}")
        nonlinear_cal = data.NonLinearityCalibration(nonlin_path)
        this_caldb.create_entry(nonlinear_cal)

    # Load KGain calibration
    # Note: Skipping KGain as it's OPTIONAL in l1_to_l2a_basic recipe
    # kgain_path = os.path.join(calib_data_dir, 'mock_kgain.fits')
    # if os.path.exists(kgain_path):
    #     print(f"    Loading KGain: {os.path.basename(kgain_path)}")
    #     kgain = data.KGain(kgain_path)
    #     this_caldb.create_entry(kgain)

    # Load DetectorNoiseMaps calibration
    noisemaps_path = os.path.join(calib_data_dir, 'mock_detnoisemaps.fits')
    if os.path.exists(noisemaps_path):
        print(f"    Loading DetectorNoiseMaps: {os.path.basename(noisemaps_path)}")
        noise_map = data.DetectorNoiseMaps(noisemaps_path)
        this_caldb.create_entry(noise_map)

    # Scan default calibration directory for any additional calibrations
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    print("  Calibration setup complete!")

    return this_caldb, calib_data_dir


def process_l1_to_l2a(l1_files, output_dir):
    """
    Process L1 files to L2a using the walker framework.

    Args:
        l1_files (list): List of L1 FITS files
        output_dir (str): Output directory for L2a files

    Returns:
        list: List of L2a output file paths
    """
    print("\n" + "="*60)
    print("Processing L1 -> L2a...")
    print("="*60)

    # Create L2a output directory
    l2a_outputdir = os.path.join(output_dir, 'l1_to_l2a')
    os.makedirs(l2a_outputdir, exist_ok=True)

    # Run the walker to process L1 -> L2a
    print("  Running walker.walk_corgidrp()...")
    walker.walk_corgidrp(l1_files, "", l2a_outputdir, template="l1_to_l2a_basic.json")

    # Find the L2a output files
    l2a_files = [
        os.path.join(l2a_outputdir, f)
        for f in os.listdir(l2a_outputdir)
        if f.endswith('.fits') and '_cal' not in f and 'recipe' not in f
    ]

    print(f"\n  Successfully created {len(l2a_files)} L2a files:")
    for f in l2a_files:
        print(f"    - {os.path.basename(f)}")

    return sorted(l2a_files)


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

    # Step 2: Track L1 headers
    print("\n" + "="*60)
    print("Tracking L1 header information...")
    print("="*60)

    for l1_file in l1_files:
        header_file = save_headers_to_file(l1_file, output_dir, stage_name="L1")
        print(f"  Saved: {os.path.basename(header_file)}")

    # Step 3: Set up calibrations
    _, calibrations_dir = setup_calibrations()

    # Step 4: Process L1 -> L2a
    l2a_files = process_l1_to_l2a(l1_files, output_dir)

    # Step 5: Track L2a headers
    print("\n" + "="*60)
    print("Tracking L2a header information...")
    print("="*60)

    for l2a_file in l2a_files:
        header_file = save_headers_to_file(l2a_file, output_dir, stage_name="L2a")
        print(f"  Saved: {os.path.basename(header_file)}")

    # Clean up temporary caldb file
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(corgidrp.caldb_filepath)

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)
    print("\nOutput structure:")
    print(f"  L1 files: {os.path.join(output_dir, 'input_l1')}")
    print(f"  L2a files: {os.path.join(output_dir, 'l1_to_l2a')}")
    print(f"  Calibrations: {calibrations_dir}")
    print(f"  L1 headers: {os.path.join(output_dir, 'header_tracking', 'L1')}")
    print(f"  L2a headers: {os.path.join(output_dir, 'header_tracking', 'L2a')}")

    # Future steps will go here:
    # Step 6: Run L2a -> L2b processing
    # Step 7: Run L2b -> L3 processing
    # etc.


if __name__ == "__main__":
    main()
