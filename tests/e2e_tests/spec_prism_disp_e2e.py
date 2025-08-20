import os
import sys
import glob
import numpy as np
import astropy.io.fits as fits
from datetime import datetime, timedelta

# Ensure test_spec.py is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tests.test_spec import convert_tvac_to_dataset, template_dir
from corgidrp.data import Dataset
from corgidrp.spec import compute_psf_centroid, calibrate_dispersion_model

# --- CONFIGURATION ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'spec_prism_disp_output')
INPUT_DATA_DIR = os.path.join(OUTPUT_DIR, 'input_data')
os.makedirs(INPUT_DATA_DIR, exist_ok=True)

# --- GENERATE MOCK/TVAC L3 DATASET AND TEMPLATES ---
print('Generating mock/TVAC L3 images and templates using convert_tvac_to_dataset()...')
# Point to input data directory for this run
import tests.test_spec as test_spec_module
test_spec_module.template_dir = INPUT_DATA_DIR
convert_tvac_to_dataset()
# --- GATHER INPUT FILES ---
print('Gathering generated FITS files for L2b dataset and templates...')
# Match the new filename format: cgi_*_l2b*.fits (with or without trailing underscore)
l2b_files = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, 'cgi_*_l2b*.fits')))

# Deduplicate by CFAMNAME in a deterministic way (always select the first file for each sub-band)
from collections import OrderedDict
import astropy.io.fits as afits
cfam_to_file = OrderedDict()
for f in l2b_files:
    with afits.open(f) as hdul:
        cfam = hdul[1].header.get('CFAMNAME')
        if cfam and cfam not in cfam_to_file:
            cfam_to_file[cfam] = f
l2b_files_unique = [cfam_to_file[cfam] for cfam in sorted(cfam_to_file.keys())]

# --- LOAD INPUT DATA ---
print('Loading L2b dataset...')
l2b_dataset = Dataset(l2b_files_unique)

# --- GENERATE TEMPLATE DATASET FROM FILTER SWEEP FILE ---
print('Generating template dataset from filter sweep template file...')
from astropy.io import fits
from astropy.table import Table
from corgidrp.data import Image

# Relative path to the filter sweep template file
template_fits_path = os.path.join(os.path.dirname(__file__), '../test_data/spectroscopy/g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep_withoffsets.fits')

# Load the template PSF array and sub-band info
psf_array = fits.getdata(template_fits_path, ext=0)
psf_table = Table(fits.getdata(template_fits_path, ext=1))

# Use headers from the first science frame as a base
if len(l2b_dataset) > 0:
    base_pri_hdr = l2b_dataset[0].pri_hdr.copy()
    base_ext_hdr = l2b_dataset[0].ext_hdr.copy()
else:
    base_pri_hdr = fits.Header()
    base_ext_hdr = fits.Header()

# Seed and add noise to template images (mirroring test_spec.py)
np.random.seed(5)
read_noise = 200
noisy_data_array = (np.random.poisson(np.abs(psf_array) / 2) +
                    np.random.normal(loc=0, scale=read_noise, size=psf_array.shape))

template_images = []
for i in range(noisy_data_array.shape[0]):
    data_2d = np.copy(noisy_data_array[i])
    err = np.zeros_like(data_2d)
    dq = np.zeros_like(data_2d, dtype=int)
    image = Image(
        data_or_filepath=data_2d,
        pri_hdr=base_pri_hdr.copy(),
        ext_hdr=base_ext_hdr.copy(),
        err=err,
        dq=dq
    )
    image.ext_hdr['CFAMNAME'] = psf_table['CFAM'][i]
    template_images.append(image)
template_dataset = type(l2b_dataset)(template_images)

# --- SAVE TEMPLATE IMAGES TO template_images SUBFOLDER ---
TEMPLATE_IMAGE_DIR = os.path.join(OUTPUT_DIR, 'template_images')
os.makedirs(TEMPLATE_IMAGE_DIR, exist_ok=True)
print('Saving template images to template_images folder...')
from astropy.io import fits as pyfits

def get_formatted_filename(pri_hdr, dt, suffix="l2b"):
    visitid = pri_hdr.get('VISITID', '0000000000000000000')
    now = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
    return f"cgi_{visitid}_{now}_{suffix}.fits"

basetime = datetime.now()
i = 0
for img in template_images:
    cfam = img.ext_hdr.get('CFAMNAME', 'unknown')
    fname = get_formatted_filename(img.pri_hdr, basetime + timedelta(seconds=i), suffix="l2b")
    fpath = os.path.join(TEMPLATE_IMAGE_DIR, fname)
    # Save with both primary and extension headers
    primary_hdu = pyfits.PrimaryHDU(header=img.pri_hdr)
    image_hdu = pyfits.ImageHDU(data=img.data, header=img.ext_hdr)
    hdul = pyfits.HDUList([primary_hdu, image_hdu])
    hdul.writeto(fpath, overwrite=True)
    i += 1

# --- CHECK INPUT DATASET ---
print('Checking input dataset headers and dimensions...')
for i, frame in enumerate(l2b_dataset):
    ext_hdr = frame.ext_hdr
    assert 'CFAMNAME' in ext_hdr, f"Frame {i} missing CFAMNAME!"
    print(f"Frame {i}: CFAMNAME={ext_hdr['CFAMNAME']}, shape={frame.data.shape}")

# --- CHECK TEMPLATE DATASET ---
print('Checking template dataset headers and dimensions...')
for i, frame in enumerate(template_dataset):
    ext_hdr = frame.ext_hdr
    assert 'CFAMNAME' in ext_hdr, f"Template {i} missing CFAMNAME!"
    print(f"Template {i}: CFAMNAME={ext_hdr['CFAMNAME']}, shape={frame.data.shape}")

# --- CENTROIDING ---
print('Computing PSF centroids...')
centroid_cal = compute_psf_centroid(l2b_dataset, template_dataset=template_dataset, filtersweep=True)
centroid_cal.save(filedir=OUTPUT_DIR)

# --- PRINT CENTROID CALIBRATION CONTENTS ---
print('\nCentroid calibration results:')
print(f'Number of centroids: {len(centroid_cal.xfit)}')
print('Sub-bands (FILTERS header):', centroid_cal.ext_hdr.get('FILTERS', 'N/A'))
print('xfit:', centroid_cal.xfit)
print('yfit:', centroid_cal.yfit)
print('xfit_err:', centroid_cal.xfit_err)
print('yfit_err:', centroid_cal.yfit_err)

# --- MIRROR test_spec.py: Remove broadband filter (last entry) before dispersion calibration ---
centroid_cal.xfit = centroid_cal.xfit[:-1]
centroid_cal.yfit = centroid_cal.yfit[:-1]
centroid_cal.xfit_err = centroid_cal.xfit_err[:-1]
centroid_cal.yfit_err = centroid_cal.yfit_err[:-1]

# --- DISPERSION CALIBRATION ---
print('Calibrating dispersion model...')
disp_model = calibrate_dispersion_model(centroid_cal, prism='PRISM3')

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
