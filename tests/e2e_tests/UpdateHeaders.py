#!/usr/bin/env python3

"""
update_headers_all.py

Reads *all* .fits files from an input directory, assigns new "mock" calibration 
headers (from corgidrp.mocks.create_default_calibration_product_headers), 
and writes each updated file to the specified output directory. The new 
filename is (original name)_updated_headers.fits.

Designed to be run directly in VS Code (no command-line arguments needed).
"""

import os
import astropy.io.fits as fits
import corgidrp.mocks as mocks

def apply_mock_headers_and_save(input_fits, output_fits):
    """
    Reads a FITS file at 'input_fits', overwrites the primary and extension 
    headers with 'mock' calibration headers, then writes the result to 
    'output_fits'.

    :param input_fits:  str. Path to an existing FITS file.
    :param output_fits: str. Path for the updated FITS file to be written.
    """
    # 1) Load the original FITS file in memory
    with fits.open(input_fits, mode='readonly') as hdulist:
        # Extract data from each HDU so we can rewrite it later
        hdus_data = [hdu.data for hdu in hdulist]

    # 2) Generate mock calibration product headers
    pri_hdr, ext_hdr = mocks.create_default_calibration_product_headers()

    # 3) Build new HDUList with the mock headers
    new_hdulist = []
    # Overwrite the primary HDU
    primary_hdu = fits.PrimaryHDU(data=hdus_data[0], header=pri_hdr.copy())
    new_hdulist.append(primary_hdu)

    # If there's at least one extension, overwrite its header as well
    if len(hdus_data) > 1:
        extension_hdu = fits.ImageHDU(data=hdus_data[1], header=ext_hdr.copy())
        new_hdulist.append(extension_hdu)

    # 4) Save to output path (overwrite if exists)
    hdul_out = fits.HDUList(new_hdulist)
    hdul_out.writeto(output_fits, overwrite=True)

    print(f"Updated headers for: {input_fits}\n  --> {output_fits}")

def update_headers_in_directory(input_dir, output_dir):
    """
    Loop over every .fits file in 'input_dir', apply mock headers, and
    save to 'output_dir' as (basename)_updated_headers.fits.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all files in the input directory
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".fits"):
            continue  # skip non-FITS files

        # Full path to the input file
        full_input_path = os.path.join(input_dir, filename)

        # Construct the new output filename
        root, ext = os.path.splitext(filename)  # e.g. ("dark_current_20240322", ".fits")
        new_filename = f"{root}_updated_headers{ext}"  # e.g. "dark_current_20240322_updated_headers.fits"
        full_output_path = os.path.join(output_dir, new_filename)

        # Apply mock headers
        apply_mock_headers_and_save(full_input_path, full_output_path)

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Change these to the folder(s) you want to process, then run in VS Code
    # ---------------------------------------------------------------------
    input_dir = "/Users/jmilton/Library/CloudStorage/Box-Box/CGI_TVAC_Data/Working_Folder/TV-20_EXCAM_noise_characterization/darkmap"
    output_dir = "/Users/jmilton/Documents/CGI/CGI_TVAC_Data/Updated_Header_Files/TV-20_EXCAM_noise_characterization/darkmap"

    # Validate that input_dir exists
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    # Process each .fits in input_dir
    update_headers_in_directory(input_dir, output_dir)
    print("\nDone")

