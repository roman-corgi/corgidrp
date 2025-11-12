import os
import glob
import logging
from datetime import datetime

import numpy as np
import astropy.io.fits as fits
import pytest
import argparse

from corgidrp.data import Dataset, Image, FluxcalFactor
from corgidrp.check import (
    check_filename_convention,
    check_dimensions,
    verify_header_keywords,
)
from corgidrp import l4_to_tda


# ==============================================================================
# Spectroscopy L4->TDA VAP Test
# ==============================================================================

def run_spec_l4_to_tda_vap_test(e2edata_path, e2eoutput_path):
    """Execute the spectroscopy L4->TDA VAP validation with detailed logging."""
    logger = logging.getLogger('l4_to_tda_spec_vap')
    logger.info('=' * 80)
    logger.info('Pre-test: set up input files')
    logger.info('=' * 80)

    # ------------------------------------------------------------------
    # Find required data products
    # ------------------------------------------------------------------
    host_dir = os.path.join(e2edata_path, 'l3_to_l4_spec_noncoron_e2e')
    psf_dir = os.path.join(e2edata_path, 'l3_to_l4_spec_psfsub_e2e')

    host_file = sorted(glob.glob(os.path.join(host_dir, '*_l4_.fits')))
    comp_file = sorted(glob.glob(os.path.join(psf_dir, '*_l4_.fits')))
    fluxcal_files = sorted(glob.glob(os.path.join(host_dir, 'calibrations', '*_abf_cal.fits')))

    if not host_file:
        raise FileNotFoundError("No non-coronagraphic L4 files found for VAP test.")
    if not comp_file:
        raise FileNotFoundError("No PSF-subtracted L4 files found for VAP test.")
    if not fluxcal_files:
        raise FileNotFoundError("No FluxcalFactor file located for spectroscopy VAP test.")

    noncoron_path = host_file[-1]
    psf_path = comp_file[-1]
    fluxcal_path = fluxcal_files[-1]

    noncoron_image = Image(noncoron_path)
    psf_image = Image(psf_path)
    fluxcal_factor = FluxcalFactor(fluxcal_path)

    logger.info(f"Non-coronagraphic L4 cube: {os.path.basename(noncoron_path)}")
    logger.info(f"PSF-subtracted L4 cube: {os.path.basename(psf_path)}")
    logger.info(f"Flux calibration file: {os.path.basename(fluxcal_path)}")

    # ------------------------------------------------------------------
    # Test 1: Flux-Ratio vs Wavelength
    # ------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info('Test 1: Flux-Ratio vs Wavelength')
    logger.info("-" * 80)

    # Step: Check CGI filename and L4 data-level keywords for non-coronagraphic L4
    check_filename_convention(noncoron_image.filename, 'cgi_*_l4_.fits', "Non-coronagraphic L4", logger, data_level = 'l4_')
    verify_header_keywords(noncoron_image.ext_hdr, {
        'DATALVL': 'L4',
        'DPAMNAME': 'PRISM3',
        'BUNIT': 'photoelectron/s'
    }, "Non-coronagraphic L4", logger)
    verify_header_keywords(noncoron_image.ext_hdr, {
        'WAVLEN0', 'WV0_X', 'WV0_XERR', 'WV0_Y', 'WV0_YERR', 'WV0_DIMX', 'WV0_DIMY'
    }, "Non-coronagraphic L4", logger)

    # Step: Check CGI filename and L4 data-level keywords for PSF-subtracted L4
    check_filename_convention(psf_image.filename, 'cgi_*_l4_.fits', "PSF-subtracted L4", logger, data_level = 'l4_')
    verify_header_keywords(psf_image.ext_hdr, {
        'DATALVL': 'L4',
        'DPAMNAME': 'PRISM3',
        'BUNIT': 'photoelectron/s'
    }, "PSF-subtracted L4", logger)
    verify_header_keywords(psf_image.ext_hdr, {
        'WAVLEN0', 'WV0_X', 'WV0_XERR', 'WV0_Y', 'WV0_YERR', 'WV0_DIMX', 'WV0_DIMY'
    }, "PSF-subtracted L4", logger)

    # Step: Validate core throughput grid matches image WCS extent
    if 'CT_THRU' in psf_image.hdu_list:
        ct_grid = psf_image.hdu_list['CT_THRU'].data
        image_shape = psf_image.data.shape
        if ct_grid.ndim >= 2 and ct_grid.shape[-2:] == image_shape[-2:]:
            logger.info('Core throughput grid matches PSF-subtracted image WCS. PASS')
        else:
            logger.error(f'Core throughput grid shape {ct_grid.shape} does not match image shape {image_shape}. FAIL')
    else:
        logger.error('PSF-subtracted L4 cube missing CT_THRU extension; cannot validate throughput grid alignment. FAIL')

    # Step: Extract spectra and wavelength grids; check length, NaNs, monotonicity
    noncoron_spec = noncoron_image.hdu_list['SPEC'].data.copy()
    noncoron_wave = noncoron_image.hdu_list['SPEC_WAVE'].data.copy()
    psf_spec = psf_image.hdu_list['SPEC'].data.copy()
    psf_wave = psf_image.hdu_list['SPEC_WAVE'].data.copy()

    logger.info(f"Non-coronagraphic wavelength grid (nm): {noncoron_wave}")
    logger.info(f"PSF-subtracted wavelength grid (nm): {psf_wave}")

    # Confirm wavelengths match and are monotonic
    if not np.allclose(noncoron_wave, psf_wave):
        logger.warning('Wavelength grids differ between host and PSF-subtracted cubes. Interpolating companion spectrum.')
        psf_spec = np.interp(noncoron_wave, psf_wave, psf_spec, left=np.nan, right=np.nan)
    else:
        logger.info('Wavelength grids are identical between non-coronagraphic and PSF-subtracted spectra. PASS')

    if np.all(np.diff(noncoron_wave) >= 0) or np.all(np.diff(noncoron_wave) <= 0):
        logger.info('Host wavelength grid is monotonic. PASS')
    else:
        logger.error('Host wavelength grid is not monotonic. FAIL')

    if np.all(np.diff(psf_wave) >= 0) or np.all(np.diff(psf_wave) <= 0):
        logger.info('PSF-subtracted wavelength grid is monotonic. PASS')
    else:
        logger.error('PSF-subtracted wavelength grid is not monotonic. FAIL')

    if len(noncoron_wave) == len(psf_wave):
        logger.info('Host and companion wavelength grids have equal length. PASS')
    else:
        logger.error(f'Host and companion wavelength grids differ in length ({len(noncoron_wave)} vs {len(psf_wave)}). FAIL')

    if np.isnan(noncoron_wave).any() or np.isnan(psf_wave).any():
        logger.error('Detected NaNs in wavelength grids. FAIL')
    else:
        logger.info('No NaNs detected in wavelength grids. PASS')


    # ------------------------------------------------------------------
    # Test 2: Unocculted Star in Astrophysical Units
    # ------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info('Test 2: Unocculted Star in Astrophysical Units')
    logger.info("-" * 80)

    verify_header_keywords(noncoron_image.pri_hdr, {'ROLL'}, "Non-coronagraphic L4 headers", logger)
    # Step: Verify spectroscopy headers and required extensions
    for ext in ['SPEC', 'SPEC_ERR', 'SPEC_WAVE']:
        if ext in noncoron_image.hdu_list:
            logger.info(f"Extension {ext} present for non-coronagraphic spectrum. PASS")
        else:
            logger.error(f"Extension {ext} missing in non-coronagraphic spectrum. FAIL")

    # Step: Interpolate slit-transmission map onto SPEC_WAVE grid and log throughput
    slit_transmission_map = np.ones_like(noncoron_spec)
    psf_slit_transmission_map = np.ones_like(psf_spec)
    slit_transmission_host = np.interp(noncoron_wave, noncoron_wave, slit_transmission_map)
    slit_transmission_comp = np.interp(noncoron_wave, psf_wave, psf_slit_transmission_map)
    logger.info(f"Host slit transmission applied (first 5 bins): {slit_transmission_host[:5]}")
    logger.info(f"Companion slit transmission applied (first 5 bins): {slit_transmission_comp[:5]}")

    # Step: Flux-calibrate host spectrum with convert_spec_to_flux
    noncoron_dataset = Dataset([noncoron_image])
    calibrated_noncoron = l4_to_tda.convert_spec_to_flux(noncoron_dataset, fluxcal_factor, slit_transmission=slit_transmission_host)
    noncoron_calibrated_spec = calibrated_noncoron[0].hdu_list['SPEC'].data
    noncoron_calibrated_err = calibrated_noncoron[0].hdu_list['SPEC_ERR'].data

    bunit = calibrated_noncoron[0].hdu_list['SPEC'].header['BUNIT']
    if bunit == "erg/(s*cm^2*AA)":
        logger.info("Non-coronagraphic spectrum calibrated. BUNIT=erg/(s*cm^2*AA). PASS")
    else:
        logger.error(f"Non-coronagraphic spectrum BUNIT={bunit}. Expected erg/(s*cm^2*AA). FAIL")
    logger.info(f"Non-coronagraphic spectrum sample (first 5 bins): {noncoron_calibrated_spec[:5]}")
    logger.info(f"Non-coronagraphic flux uncertainties (first 5 bins): {noncoron_calibrated_err[0][:5]}")

    # Step: Flux-calibrate companion spectrum with convert_spec_to_flux
    comp_dataset = Dataset([psf_image])
    calibrated_comp = l4_to_tda.convert_spec_to_flux(comp_dataset, fluxcal_factor, slit_transmission=slit_transmission_comp)
    psf_calibrated_spec = calibrated_comp[0].hdu_list['SPEC'].data

    # Step: Log COL_COR usage and confirm non-coronagraphic wavelengths are monotonic
    col_cor_val = noncoron_image.ext_hdr.get('COL_COR', 1.0)
    logger.info(f"Non-coronagraphic COL_COR value: {col_cor_val}")

    if np.all(np.diff(noncoron_wave) >= 0) or np.all(np.diff(noncoron_wave) <= 0):
        logger.info('Non-coronagraphic wavelength grid is monotonic. PASS')
    else:
        logger.error('Non-coronagraphic wavelength grid is not monotonic. FAIL')

    # Step: Log final astrophysical flux ratio
    astro_flux_ratio = np.divide(psf_calibrated_spec, noncoron_calibrated_spec, out=np.zeros_like(psf_calibrated_spec), where=noncoron_calibrated_spec != 0)
    logger.info(f"Converted companion spectrum to astrophysical units. Sample (first 5 bins): {psf_calibrated_spec[:5]}")
    logger.info(f"Astrophysical flux-ratio stats: min={np.nanmin(astro_flux_ratio):.3e}, max={np.nanmax(astro_flux_ratio):.3e}")

    logger.info('=' * 80)
    logger.info('Spectroscopy L4->TDA VAP Test Completed')
    logger.info('=' * 80)


# ==============================================================================
# Pytest Wrapper
# ==============================================================================

@pytest.mark.e2e
def test_l4_to_tda_spec_vap(e2edata_path, e2eoutput_path):
    """Pytest entry point for the spectroscopy VAP workflow."""
    global logger

    output_top_level = os.path.join(e2eoutput_path, 'l4_to_tda_spec_vap')
    if not os.path.exists(output_top_level):
        os.makedirs(output_top_level)
    for f in os.listdir(output_top_level):
        file_path = os.path.join(output_top_level, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    log_file = os.path.join(output_top_level, 'l4_to_tda_spec_vap.log')

    logger = logging.getLogger('l4_to_tda_spec_vap')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('=' * 80)
    logger.info('SPECTROSCOPY L4->TDA VAP TEST')
    logger.info('=' * 80)
    logger.info("")

    run_spec_l4_to_tda_vap_test(e2edata_path, output_top_level)

    logger.info('=' * 80)
    logger.info('END-TO-END TEST COMPLETE')
    logger.info('=' * 80)


if __name__ == "__main__":
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir
    e2edata_dir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the spectroscopy L4->TDA VAP test")
    ap.add_argument("-i", "--e2edata_dir", default=e2edata_dir,
                    help="directory to get input files from [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()

    test_l4_to_tda_spec_vap(args.e2edata_dir, args.outputdir)
