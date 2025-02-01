import os
import glob
from astropy.io import fits
import corgidrp

print(corgidrp.default_cal_dir)

def check_obsid_in_directory(directory):
    print(f"Checking for missing OBSID in {directory}...")
    missing = []
    for fitsfile in glob.glob(os.path.join(directory, "**", "*.fits"), recursive=True):
        print(fitsfile)
        with fits.open(fitsfile) as hdul:
            if "OBSID" not in hdul[0].header:
                missing.append(fitsfile)
    return missing

if __name__ == "__main__":
    dir_to_check = corgidrp.default_cal_dir  # Typically corgidrp.default_cal_dir
    bad_files = check_obsid_in_directory(dir_to_check)
    if bad_files:
        print("Files missing OBSID keyword:")
        for bf in bad_files:
            print("  ", bf)
    else:
        print("All files in directory have OBSID.")
