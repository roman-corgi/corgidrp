import os
import sys
import glob
import numpy as np
import astropy.io.fits as fits
from datetime import datetime, timedelta

from corgidrp.data import Dataset
from corgidrp.spec import compute_psf_centroid, calibrate_dispersion_model

# --- CONFIGURATION ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'spec_prism_disp_output')
INPUT_DATA_DIR = os.path.join(OUTPUT_DIR, 'input_data')
os.makedirs(INPUT_DATA_DIR, exist_ok=True)

# --- USE THE SAME INPUT DATA AS test_spec.py ---
print('Using the same input data as test_spec.py...')
from astropy.io import fits
from astropy.table import Table
from corgidrp.data import Image
from corgidrp.mocks import create_default_L2b_headers

# Use the exact same data source as test_spec.py
datadir = os.path.join(os.path.dirname(__file__), '../test_data/spectroscopy')
file_path = os.path.join(datadir, "g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep_withoffsets.fits")

if not os.path.exists(file_path):
    raise RuntimeError(f'Test file not found: {file_path}')

# Load the same data that test_spec.py uses
psf_array = fits.getdata(file_path, ext=0)
psf_table = Table(fits.getdata(file_path, ext=1))

print(f"Loaded PSF array with shape: {psf_array.shape}")
print(f"PSF table has {len(psf_table)} rows")

# Create the same dataset structure as test_spec.py
pri_hdr, ext_hdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
ext_hdr["DPAMNAME"] = 'PRISM3'
ext_hdr["FSAMNAME"] = 'OPEN'

# Add random noise to the filter sweep template images to serve as fake data (same as test_spec.py)
np.random.seed(5)
read_noise = 200
noisy_data_array = (np.random.poisson(np.abs(psf_array) / 2) + 
                    np.random.normal(loc=0, scale=read_noise, size=psf_array.shape))

psf_images = []
for i in range(noisy_data_array.shape[0]):
    data_2d = np.copy(noisy_data_array[i])
    err = np.zeros_like(data_2d)
    dq = np.zeros_like(data_2d, dtype=int)
    image = Image(
        data_or_filepath=data_2d,
        pri_hdr=pri_hdr.copy(),
        ext_hdr=ext_hdr.copy(),
        err=err,
        dq=dq
    )
    image.ext_hdr['CFAMNAME'] = psf_table['CFAM'][i]
    psf_images.append(image)

l2b_dataset = Dataset(psf_images)

# --- SAVE INPUT DATA TO input_data SUBFOLDER ---
print('Saving input data to input_data subfolder...')
os.makedirs(INPUT_DATA_DIR, exist_ok=True)

def get_formatted_filename(pri_hdr, dt, suffix="l2b"):
    visitid = pri_hdr.get('VISITID', '0000000000000000000')
    now = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
    return f"cgi_{visitid}_{now}_{suffix}.fits"

basetime = datetime.now()
for i, img in enumerate(psf_images):
    cfam = img.ext_hdr.get('CFAMNAME', 'unknown')
    fname = get_formatted_filename(img.pri_hdr, basetime + timedelta(seconds=i), suffix="l2b")
    fpath = os.path.join(INPUT_DATA_DIR, fname)
    
    # Save with both primary and extension headers
    primary_hdu = fits.PrimaryHDU(header=img.pri_hdr)
    image_hdu = fits.ImageHDU(data=img.data, header=img.ext_hdr)
    hdul = fits.HDUList([primary_hdu, image_hdu])
    hdul.writeto(fpath, overwrite=True)
    print(f"Saved {cfam} to {fname}")

# --- CHECK INPUT DATASET ---
print('Checking input dataset headers and dimensions...')
for i, frame in enumerate(l2b_dataset):
    ext_hdr = frame.ext_hdr
    assert 'CFAMNAME' in ext_hdr, f"Frame {i} missing CFAMNAME!"
    print(f"Frame {i}: CFAMNAME={ext_hdr['CFAMNAME']}, shape={frame.data.shape}")

# --- CENTROIDING ---
print('Computing PSF centroids...')

# Load the saved files back into a dataset so they have proper filenames
# This is needed for the SpectroscopyCentroidPSF constructor to auto-generate filenames
saved_files = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, 'cgi_*_l2b.fits')))
if not saved_files:
    raise RuntimeError(f'No saved L2b files found in {INPUT_DATA_DIR}!')

print(f"Loading {len(saved_files)} saved files for centroid computation...")
l2b_dataset_with_filenames = Dataset(saved_files)

# Use the same approach as test_spec.py - let it use the default template dataset
centroid_cal = compute_psf_centroid(l2b_dataset_with_filenames)

# The SpectroscopyCentroidPSF class should auto-generate a filename
# Just set the output directory
centroid_cal.filedir = OUTPUT_DIR
centroid_cal.save()

# --- PRINT CENTROID CALIBRATION CONTENTS ---
print('\nCentroid calibration results:')
print(f'Number of centroids: {len(centroid_cal.xfit)}')
print('Sub-bands (FILTERS header):', centroid_cal.ext_hdr.get('FILTERS', 'N/A'))
print('xfit:', centroid_cal.xfit)
print('yfit:', centroid_cal.yfit)
print('xfit_err:', centroid_cal.xfit_err)
print('yfit_err:', centroid_cal.yfit_err)

# --- NOTE: Do NOT remove any centroids - calibrate_dispersion_model handles reference bands internally ---

# --- DISPERSION CALIBRATION ---
print('Calibrating dispersion model...')
disp_model = calibrate_dispersion_model(centroid_cal)

# --- SAVE OUTPUT ---
print('Saving output calibration product to current directory...')
disp_model.save(filedir=OUTPUT_DIR)

# --- VERIFY OUTPUT ---
# Find the most recent file ending with _dpm_cal.fits in OUTPUT_DIR
cal_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*_dpm_cal.fits')), key=os.path.getmtime, reverse=True)
if not cal_files:
    raise RuntimeError(f'No _dpm_cal.fits file found in {OUTPUT_DIR}!')
cal_file = cal_files[0]
print(f'Verifying output calibration product: {cal_file}')
hdul = fits.open(cal_file)



# Print polynomial coefficients and orientation
coeffs = disp_model.pos_vs_wavlen_polycoeff
angle = disp_model.clocking_angle
print(f"Polynomial coefficients (dispersion): {coeffs}")
print(f"Dispersion axis orientation angle: {angle} deg")

print('End-to-end test complete.')
