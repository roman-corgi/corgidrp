#!/usr/bin/env python3
"""
Script to set ISPC (extension header) and PHTCNT (primary header) to 0 for analog data.

Usage:
    python fix_ispc_headers.py <input_directory> [--output-directory <output_dir>]

If output directory is not specified, files will be modified in place.
"""

import argparse
import os
import astropy.io.fits as fits
import glob


def fix_ispc_headers(input_dir, output_dir=None):
    """
    Fix ISPC and PHTCNT headers in FITS files.
    
    Args:
        input_dir (str): Directory containing FITS files to fix
        output_dir (str, optional): Output directory. If None, modifies files in place.
    """
    # Find all FITS files
    fits_files = glob.glob(os.path.join(input_dir, '*.fits'))
    
    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        return
    
    print(f"Found {len(fits_files)} FITS files")
    
    for fits_file in fits_files:
        filename = os.path.basename(fits_file)
        
        with fits.open(fits_file, mode='readonly') as hdul:
            # Check if already fixed
            pri_hdr = hdul[0].header
            if len(hdul) > 1:
                ext_hdr = hdul[1].header
            else:
                ext_hdr = None
            
            # Determine output filepath
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, filename)
            else:
                output_file = fits_file
            
            # Update headers if needed
            updated = False
            
            # Fix PHTCNT in primary header
            if 'PHTCNT' in pri_hdr:
                if pri_hdr['PHTCNT'] != 0:
                    print(f"  {filename}: Setting PHTCNT from {pri_hdr['PHTCNT']} to 0")
                    pri_hdr['PHTCNT'] = 0
                    updated = True
            else:
                print(f"  {filename}: Adding PHTCNT=0 to primary header")
                pri_hdr['PHTCNT'] = 0
                updated = True
            
            # Fix ISPC in extension header
            if ext_hdr is not None:
                if 'ISPC' in ext_hdr:
                    if ext_hdr['ISPC'] not in (False, 0):
                        print(f"  {filename}: Setting ISPC from {ext_hdr['ISPC']} to 0")
                        ext_hdr['ISPC'] = 0
                        updated = True
                else:
                    print(f"  {filename}: Adding ISPC=0 to extension header")
                    ext_hdr['ISPC'] = 0
                    updated = True
            
            # Write updated file
            if updated or output_dir:
                hdul.writeto(output_file, overwrite=True)
                if output_dir:
                    print(f"  {filename}: Wrote to {output_file}")
            else:
                print(f"  {filename}: Already correct, skipping")
    
    print(f"\nFixed {len(fits_files)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix ISPC and PHTCNT headers in FITS files for analog data"
    )
    parser.add_argument(
        "input_directory",
        help="Directory containing FITS files to fix"
    )
    parser.add_argument(
        "-o", "--output-directory",
        help="Output directory. If not specified, files are modified in place.",
        default=None
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_directory):
        print(f"Error: {args.input_directory} is not a directory")
        exit(1)
    
    fix_ispc_headers(args.input_directory, args.output_directory)
    print("Done!")

