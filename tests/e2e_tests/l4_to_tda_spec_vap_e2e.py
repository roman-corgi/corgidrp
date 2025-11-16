import os
import glob
import logging

import numpy as np
import pytest
import argparse

from corgidrp.data import Dataset, Image
from corgidrp.check import check_filename_convention, verify_header_keywords
from corgidrp import l4_to_tda, spec, mocks


# ==============================================================================
# Spectroscopy L4->TDA VAP Test
# ==============================================================================

def run_spec_l4_to_tda_vap_test(e2edata_path, e2eoutput_path):
    """Execute the spectroscopy L4->TDA VAP validation with logging.

    Args:
        e2edata_path (str): Directory containing the pre-generated E2E input data.
        e2eoutput_path (str): Directory where test artifacts and logs should be written.
    """
    logger = logging.getLogger('l4_to_tda_spec_vap')
    logger.info('=' * 80)
    logger.info('Pre-test: set up input files')
    logger.info('=' * 80)

    # ------------------------------------------------------------------
    # Create required calibration products on the fly
    # ------------------------------------------------------------------
    # Build a lightweight mock FluxcalFactor instead of reading an abf_cal file.
    fluxcal_factor = mocks.make_mock_fluxcal_factor(
        value=1.0,
        err=0.0,
        cfam_name='3D',
        dpam_name='PRISM3',
        fsam_name='R1C2',
    )
    logger.info("Mock flux calibration factor created")

    host_dir = os.path.join(e2edata_path, 'non-coron')
    psf_dir = os.path.join(e2edata_path, 'coron')
    host_file = sorted(glob.glob(os.path.join(host_dir, '*_l4_.fits')))
    comp_file = sorted(glob.glob(os.path.join(psf_dir, '*_l4_.fits')))

    if not host_file:
        raise FileNotFoundError("No non-coronagraphic L4 files found for VAP test.")
    if not comp_file:
        raise FileNotFoundError("No PSF-subtracted L4 files found for VAP test.")

    noncoron_path = host_file[-1]
    psf_path = comp_file[-1]

    noncoron_image = Image(noncoron_path)
    psf_image = Image(psf_path)

    logger.info(f"Non-coronagraphic L4 cube: {os.path.basename(noncoron_path)}")
    logger.info(f"PSF-subtracted L4 cube: {os.path.basename(psf_path)}")

    # Build mock core-throughput and FPAM/FSAM calibrations
    ct_cal = mocks.create_ct_cal(fwhm_mas=50.0, cfam_name='3D', cenx=0.0, ceny=0.0, nx=11, ny=11)
    fpamfsam_cal = mocks.create_mock_fpamfsam_cal()

    # ------------------------------------------------------------------
    # Test 1: L4 Spectroscopy Input Data Validation
    # ------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info('Test 1: L4 Spectroscopy Input Data Validation')
    logger.info("-" * 80)

    # Step: Check CGI filename and L4 data-level keywords for non-coronagraphic L4
    check_filename_convention(noncoron_image.filename, 'cgi_*_l4_.fits', "Non-coronagraphic L4", logger, data_level = 'l4_')
    verify_header_keywords(noncoron_image.ext_hdr, {
        'DATALVL': 'L4',
        'DPAMNAME': 'PRISM3',
        'BUNIT': 'photoelectron/s'
    }, "Non-coronagraphic L4", logger)
    verify_header_keywords(noncoron_image.pri_hdr, {'ROLL'}, "Non-coronagraphic L4", logger)
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
    verify_header_keywords(psf_image.pri_hdr, {'ROLL'}, "PSF-subtracted L4", logger)
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

    # Step: Verify spectroscopy headers and required extensions
    for ext in ['SPEC', 'SPEC_ERR', 'SPEC_WAVE']:
        if ext in noncoron_image.hdu_list:
            logger.info(f"Extension {ext} present for non-coronagraphic spectrum. PASS")
        else:
            logger.error(f"Extension {ext} missing in non-coronagraphic spectrum. FAIL")

    # Step: Check if COL_COR exists; if not, compute it using determine_color_cor
    col_cor_val = noncoron_image.ext_hdr.get('COL_COR', None)
    if col_cor_val is None:
        # TODO: This requires reference and source star info
        logger.warning('COL_COR not found in header. Using default value of 1.0.')
        col_cor_val = 1.0
    else:
        logger.info(f"COL_COR found in header: {col_cor_val}")

    # Step: Apply core-throughput correction to PSF-subtracted L4 cube
    try:
        ct_factor = l4_to_tda.apply_core_throughput_correction(
            psf_image, ct_cal, fpamfsam_cal, logr=False
        )
        spec_hdr = psf_image.hdu_list['SPEC'].header
        ctcor_flag = spec_hdr.get('CTCOR', False)
        ok = ctcor_flag and np.isfinite(ct_factor) and (ct_factor > 0)
        message = (
            f"Core throughput correction applied to PSF-subtracted L4 cube. "
            f"CTCOR={ctcor_flag}, CTFAC={ct_factor}"
        )
        logger.info(f"{message}. {'PASS' if ok else 'FAIL'}")
    except Exception as exc:
        logger.error(
            f"Core throughput correction failed for PSF-subtracted L4 cube: {exc}. FAIL"
        )

    # Step: Build a slit-transmission map (map, x, y)
    slit_map = np.ones((1, noncoron_wave.size), dtype=float)
    slit_x = np.array([noncoron_image.ext_hdr.get('WV0_X', 0.0)])
    slit_y = np.array([noncoron_image.ext_hdr.get('WV0_Y', 0.0)])
    slit_transmission = (slit_map, slit_x, slit_y)
    logger.info(f"Slit transmission map sample (first 5 bins): {slit_map[0][:5]}")

    # Step: Check SPEC BUNIT and flux-calibrate host and companion spectra
    spec_bunit_input = noncoron_image.hdu_list['SPEC'].header.get('BUNIT')

    noncoron_calibrated_spec = None
    noncoron_calibrated_err = None
    psf_calibrated_spec = None

    if spec_bunit_input != "photoelectron/s/bin":
        logger.error(
            f"Non-coronagraphic SPEC BUNIT before flux calibration: {spec_bunit_input}. FAIL."
        )
    else:
        # Flux-calibrate host spectrum with convert_spec_to_flux
        logger.info(
            f"Non-coronagraphic SPEC BUNIT before flux calibration: {spec_bunit_input}. PASS."
        )
        noncoron_dataset = Dataset([noncoron_image])
        calibrated_noncoron = l4_to_tda.convert_spec_to_flux(
            noncoron_dataset, fluxcal_factor, slit_transmission=slit_transmission
        )
        noncoron_calibrated_spec = calibrated_noncoron[0].hdu_list['SPEC'].data
        noncoron_calibrated_err = calibrated_noncoron[0].hdu_list['SPEC_ERR'].data

        bunit = calibrated_noncoron[0].hdu_list['SPEC'].header.get('BUNIT')
        if bunit == "erg/(s*cm^2*AA)":
            logger.info("Non-coronagraphic spectrum calibrated. BUNIT=erg/(s*cm^2*AA). PASS")
        else:
            logger.error(
                f"Non-coronagraphic spectrum BUNIT={bunit}. "
                "Expected erg/(s*cm^2*AA). FAIL"
            )
        logger.info(
            f"Non-coronagraphic spectrum sample (first 5 bins): "
            f"{noncoron_calibrated_spec[:5]}"
        )
        logger.info(
            f"Non-coronagraphic flux uncertainties (first 5 bins): "
            f"{noncoron_calibrated_err[0][:5]}"
        )

        # Flux-calibrate companion spectrum with convert_spec_to_flux
        comp_dataset = Dataset([psf_image])
        comp_ctcor = psf_image.hdu_list['SPEC'].header.get('CTCOR', False)

        if not comp_ctcor:
            logger.error(
                "PSF-subtracted SPEC header missing CTCOR=True after attempted "
                "core throughput correction; skipping convert_spec_to_flux for "
                "companion spectrum. FAIL"
            )
            calibrated_comp = None
            psf_calibrated_spec = None
        else:
            logger.info(
                "PSF-subtracted SPEC header has CTCOR=True; proceeding with "
                "convert_spec_to_flux for companion spectrum. PASS"
            )
            calibrated_comp = l4_to_tda.convert_spec_to_flux(
                comp_dataset, fluxcal_factor, slit_transmission=slit_transmission
            )
            psf_calibrated_spec = calibrated_comp[0].hdu_list['SPEC'].data

    # Step: Log COL_COR usage and confirm non-coronagraphic wavelengths are monotonic
    # Check the actual COL_COR value used (from header or default).
    if 'calibrated_noncoron' in locals():
        final_col_cor = calibrated_noncoron[0].ext_hdr.get('COL_COR', 1.0)
        source_hdr = "flux-calibrated host header"
    else:
        final_col_cor = noncoron_image.ext_hdr.get('COL_COR', 1.0)
        source_hdr = "input host header (no flux calibration applied)"

    if final_col_cor != 1.0:
        logger.info(
            f"Color correction applied: COL_COR={final_col_cor} "
            f"from {source_hdr}"
        )
    else:
        logger.info(
            f"No color correction applied: COL_COR={final_col_cor} "
            f"from {source_hdr}"
        )

    if np.all(np.diff(noncoron_wave) >= 0) or np.all(np.diff(noncoron_wave) <= 0):
        logger.info('Non-coronagraphic wavelength grid is monotonic. PASS')
    else:
        logger.error('Non-coronagraphic wavelength grid is not monotonic. FAIL')

    # Step: Log final astrophysical flux ratio (only if flux-calibration succeeded)
    if (noncoron_calibrated_spec is not None) and (psf_calibrated_spec is not None):
        astro_flux_ratio = np.divide(
            psf_calibrated_spec,
            noncoron_calibrated_spec,
            out=np.zeros_like(psf_calibrated_spec),
            where=noncoron_calibrated_spec != 0,
        )
        logger.info(
            "Converted companion spectrum to astrophysical units. "
            f"Sample (first 5 bins): {psf_calibrated_spec[:5]}"
        )
        logger.info(
            "Astrophysical flux-ratio stats: "
            f"min={np.nanmin(astro_flux_ratio):.3e}, "
            f"max={np.nanmax(astro_flux_ratio):.3e}"
        )

    # ------------------------------------------------------------------
    # Test 3: Companion-to-Host Flux-Ratio (PSF-subtracted)
    # ------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info('Test 3: Companion-to-Host Flux-Ratio (PSF-subtracted)')
    logger.info("-" * 80)

    # The following steps were already completed in previous tests:
    # - Extract 1-D spectra from PSF-subtracted and non-PSF-subtracted cubes (Test 1)
    # - Convert host and companion spectra to physical units via convert_spec_to_flux() (Test 2)
    # - Apply slit-transmission loss to both host and companion fluxes (Test 2)
    # - Apply color-correction via convert_spec_to_flux() (Test 2)

    # Step: Compute flux ratio using all available host and companion spectra
    all_host_files = sorted(glob.glob(os.path.join(host_dir, '*_l4_.fits')))
    all_comp_files = sorted(glob.glob(os.path.join(psf_dir, '*_l4_.fits')))

    if not all_host_files or not all_comp_files:
        logger.error('No host and/or companion L4 files found. Cannot compute flux ratio. FAIL')
        flux_ratio = np.full_like(noncoron_wave, np.nan)
        ratio_err = np.full_like(noncoron_wave, np.nan)
        wavelength = noncoron_wave
    else:
        host_images = [Image(f) for f in all_host_files]
        comp_images = [Image(f) for f in all_comp_files]

        logger.info(f'Number of host L4 spectra used: {len(host_images)}')
        logger.info(f'Number of companion L4 spectra used: {len(comp_images)}')
        logger.info(f'Host rolls: {[img.pri_hdr.get("ROLL") for img in host_images]}')
        logger.info(f'Companion rolls: {[img.pri_hdr.get("ROLL") for img in comp_images]}')

        # Check that all host and companion cubes contain a SPEC_WAVE (wavelength) extension.
        host_wave_ext = all('SPEC_WAVE' in img.hdu_list for img in host_images)
        comp_wave_ext = all('SPEC_WAVE' in img.hdu_list for img in comp_images)
        logger.info(
            f"All host cubes contain SPEC_WAVE extension: {'PASS' if host_wave_ext else 'FAIL'}"
        )
        logger.info(
            f"All companion cubes contain SPEC_WAVE extension: {'PASS' if comp_wave_ext else 'FAIL'}"
        )

        # Combine all host and all companion spectra (raw units) using inverse-variance weighting.
        # This produces a single 1-D host spectrum and a single 1-D companion spectrum
        # that can be passed into compute_spec_flux_ratio for flux calibration and ratio.
        host_ds = Dataset(host_images)
        comp_ds = Dataset(comp_images)

        host_spec, host_wave, host_err, host_rolls = l4_to_tda.combine_spectra(host_ds)
        comp_spec, comp_wave, comp_err, comp_rolls = l4_to_tda.combine_spectra(comp_ds)

        logger.info(f'Host rolls contributing to combined spectrum: {host_rolls}')
        logger.info(f'Companion rolls contributing to combined spectrum: {comp_rolls}')

        # Check monotonicity and length of the combined wavelength grids.
        host_mono = np.all(np.diff(host_wave) >= 0) or np.all(np.diff(host_wave) <= 0)
        comp_mono = np.all(np.diff(comp_wave) >= 0) or np.all(np.diff(comp_wave) <= 0)
        logger.info(
            f"Combined host wavelength grid monotonic: {'PASS' if host_mono else 'FAIL'}"
        )
        logger.info(
            f"Combined companion wavelength grid monotonic: {'PASS' if comp_mono else 'FAIL'}"
        )
        same_len = host_wave.size == comp_wave.size
        logger.info(
            f"Combined host/companion wavelength grids equal length "
            f"({host_wave.size} vs {comp_wave.size}): {'PASS' if same_len else 'FAIL'}"
        )

        # Build 1-D spectroscopy Images from the combined spectra
        host_template = host_images[0]
        comp_template = comp_images[0]

        host_comb = Image(
            host_template.data.copy(),
            pri_hdr=host_template.pri_hdr.copy(),
            ext_hdr=host_template.ext_hdr.copy(),
            err=host_template.err.copy(),
            dq=host_template.dq.copy(),
            err_hdr=host_template.err_hdr.copy(),
            dq_hdr=host_template.dq_hdr.copy(),
            input_hdulist=host_template.hdu_list,
        )
        comp_comb = Image(
            comp_template.data.copy(),
            pri_hdr=comp_template.pri_hdr.copy(),
            ext_hdr=comp_template.ext_hdr.copy(),
            err=comp_template.err.copy(),
            dq=comp_template.dq.copy(),
            err_hdr=comp_template.err_hdr.copy(),
            dq_hdr=comp_template.dq_hdr.copy(),
            input_hdulist=comp_template.hdu_list,
        )

        host_comb.hdu_list['SPEC'].data = host_spec
        host_comb.hdu_list['SPEC_ERR'].data = host_err
        host_comb.hdu_list['SPEC_WAVE'].data = host_wave

        comp_comb.hdu_list['SPEC'].data = comp_spec
        comp_comb.hdu_list['SPEC_ERR'].data = comp_err
        comp_comb.hdu_list['SPEC_WAVE'].data = comp_wave

        # Before computing the flux ratio, confirm that the combined host spectrum
        # is in correct units 
        host_spec_bunit = host_comb.hdu_list['SPEC'].header.get('BUNIT')
        logger.info(
            f"Combined host SPEC BUNIT before flux-ratio computation: {host_spec_bunit}"
        )

        if host_spec_bunit != "photoelectron/s":
            logger.error(
                "Combined host SPEC BUNIT is not 'photoelectron/s'; "
                "skipping compute_spec_flux_ratio in Test 3. FAIL"
            )
            flux_ratio = np.full_like(host_wave, np.nan, dtype=float)
            wavelength = host_wave
            ratio_err = np.full_like(host_wave, np.nan, dtype=float)
        else:
            # Compute flux ratio
            flux_ratio, wavelength, metadata = l4_to_tda.compute_spec_flux_ratio(
                host_comb,
                comp_comb,
                fluxcal_factor,
                slit_transmission=slit_transmission,
            )
            ratio_err = metadata.get('ratio_err')

        # Final flux-ratio diagnostics
        logger.info(f"Final combined flux ratio sample (first 5 bins): {flux_ratio[:5]}")
        if ratio_err is not None:
            logger.info(f"Final combined flux-ratio uncertainty (first 5 bins): {ratio_err[:5]}")

        # Logical checks on the final ratio arrays
        shapes_match = flux_ratio.shape == wavelength.shape
        logger.info(
            f"Flux ratio and wavelength arrays have identical shape "
            f"{flux_ratio.shape}: {'PASS' if shapes_match else 'FAIL'}"
        )

        if ratio_err is not None:
            err_shape_ok = ratio_err.shape == flux_ratio.shape
            logger.info(
                f"Flux-ratio uncertainty shape matches flux ratio shape "
                f"{ratio_err.shape}: {'PASS' if err_shape_ok else 'FAIL'}"
            )

        has_nan_ratio = np.isnan(flux_ratio).any()
        logger.info(
            f"Flux ratio contains no NaNs: {'PASS' if not has_nan_ratio else 'FAIL'}"
        )

        if ratio_err is not None:
            has_nan_err = np.isnan(ratio_err).any()
            logger.info(
                f"Flux-ratio uncertainty contains no NaNs: {'PASS' if not has_nan_err else 'FAIL'}"
            )
    

    logger.info('=' * 80)
    logger.info('Spectroscopy L4->TDA VAP Test Completed')
    logger.info('=' * 80)


# ==============================================================================
# Pytest Wrapper
# ==============================================================================

@pytest.mark.e2e
def test_l4_to_tda_spec_vap(e2edata_path, e2eoutput_path):
    """Pytest entry point for the spectroscopy VAP workflow.

    Args:
        e2edata_path (str): Fixture-provided path to E2E input data products.
        e2eoutput_path (str): Fixture-provided path where test outputs should be stored.
    """
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
    e2edata_dir = "/Users/jmilton/Documents/CGI/E2E_Test_Data2/SPEC_sims"

    ap = argparse.ArgumentParser(description="run the spectroscopy L4->TDA VAP test")
    ap.add_argument("-i", "--e2edata_dir", default=e2edata_dir,
                    help="directory to get input files from [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()

    test_l4_to_tda_spec_vap(args.e2edata_dir, args.outputdir)
