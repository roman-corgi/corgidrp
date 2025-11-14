#!/usr/bin/env python3
"""
Convert analog spectroscopy images to photon-counting images.

This script modifies FITS headers and optionally reduces flux to convert
analog images into photon-counting format. The modifications include:
- Setting ISPC to 1 in the extension header
- Setting PHTCNT to 1 in the primary header  
- Setting EMGAIN to 5000
- Setting EXPTIME to 0.05 seconds
- Optionally reducing flux in the image data

Usage:
    python convert_analog_to_photon_counting.py <input_directory> [--reduce-flux FACTOR] [--output-dir DIR]
"""

import argparse
import os
import glob
from astropy.io import fits
import numpy as np


def convert_to_photon_counting(filepath, reduce_flux_factor=None, output_dir=None):
    """
    Convert a single FITS file from analog to photon-counting format.
    
    Args:
        filepath (str): Path to input FITS file
        reduce_flux_factor (float, optional): Factor to multiply image data by to reduce flux.
                                             If None, flux is not modified.
        output_dir (str, optional): Output directory. If None, overwrites input file.
    
    Returns:
        str: Path to output file
    """
    with fits.open(filepath, mode='update' if output_dir is None else 'readonly') as hdul:
        # Get primary and extension headers
        pri_hdr = hdul[0].header
        ext_hdr = hdul[1].header if len(hdul) > 1 else None
        
        # Update primary header
        pri_hdr['PHTCNT'] = 1
        
        # Update extension header if it exists
        if ext_hdr is not None:
            ext_hdr['ISPC'] = 1
            ext_hdr['EMGAIN'] = 5000.0
            ext_hdr['EMGAIN_C'] = 5000.0
            ext_hdr['EMGAIN_A'] = 5000.0
            if 'EXPTIME' in ext_hdr:
                ext_hdr['EXPTIME'] = 0.05
        
        # Optionally reduce flux in image data
        if reduce_flux_factor is not None and len(hdul) > 1:
            image_hdu = hdul[1]
            if image_hdu.data is not None:
                image_hdu.data = image_hdu.data * reduce_flux_factor
        
        # Determine output path
        if output_dir is None:
            output_path = filepath
            # Write changes back to file
            hdul.flush()
        else:
            basename = os.path.basename(filepath)
            output_path = os.path.join(output_dir, basename)
            os.makedirs(output_dir, exist_ok=True)
            # Write to new file
            hdul.writeto(output_path, overwrite=True)
        
        print(f"Converted: {os.path.basename(filepath)} -> {os.path.basename(output_path)}")
        if reduce_flux_factor is not None:
            print(f"  Flux reduced by factor {reduce_flux_factor}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert analog spectroscopy images to photon-counting format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all FITS files in a directory in place (overwrites originals):
  python convert_analog_to_photon_counting.py /path/to/directory
  
  # Convert with flux reduction:
  python convert_analog_to_photon_counting.py /path/to/directory --reduce-flux 0.1
  
  # Convert to new directory:
  python convert_analog_to_photon_counting.py /path/to/directory --output-dir pc_output/
        """
    )
    parser.add_argument(
        'input_directory',
        help='Input directory containing FITS files to convert'
    )
    parser.add_argument(
        '--reduce-flux',
        type=float,
        default=None,
        metavar='FACTOR',
        help='Factor to multiply image data by to reduce flux (e.g., 0.1 for 10%% of original). '
             'If not specified, flux is not modified.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='Output directory. If not specified, files are modified in place.'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_directory):
        print(f"Error: {args.input_directory} is not a directory")
        return 1
    
    # Find all FITS files in the directory
    fits_files = sorted(glob.glob(os.path.join(args.input_directory, '*.fits')))
    
    if not fits_files:
        print(f"No FITS files found in {args.input_directory}")
        return 1
    
    print(f"Found {len(fits_files)} FITS file(s) in {args.input_directory}")
    if args.reduce_flux is not None:
        print(f"Flux will be reduced by factor {args.reduce_flux}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    print()
    
    # Process each file
    converted_count = 0
    for filepath in fits_files:
        try:
            convert_to_photon_counting(
                filepath,
                reduce_flux_factor=args.reduce_flux,
                output_dir=args.output_dir
            )
            converted_count += 1
        except Exception as e:
            print(f"Error processing {os.path.basename(filepath)}: {e}")
            continue
    
    print(f"\nSuccessfully converted {converted_count} file(s)")
    return 0


if __name__ == "__main__":
    exit(main())

