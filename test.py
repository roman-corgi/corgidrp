from astropy.io import fits

file_path = "tests/test_data/spectroscopy/g0v_vmag6_spc-spec_band3_unocc_CFAM3d_NOSLIT_PRISM3_offset_array.fits"

try:
    with fits.open(file_path) as hdul:
        print(f"✅ Successfully opened FITS file: {file_path}")
        hdul.info()  # Print FITS file structure
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
except OSError:
    print(f"❌ FITS file is corrupted or unreadable: {file_path}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
