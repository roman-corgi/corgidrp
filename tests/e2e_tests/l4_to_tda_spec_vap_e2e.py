import os
import glob
import logging

import numpy as np
import pytest
import argparse

from corgidrp.data import Dataset, Image, SlitTransmission
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
    # Find/ create required data products
    # ------------------------------------------------------------------
    # Build mock core-throughput, FPAM/FSAM, and flux calibrations
    ct_cal = mocks.create_ct_cal(fwhm_mas=50.0, cfam_name='3D', cenx=0.0, ceny=0.0, nx=11, ny=11)
    fpamfsam_cal = mocks.create_mock_fpamfsam_cal()
    fluxcal_factor = mocks.make_mock_fluxcal_factor(
        value=1.0,
        err=0.0,
        cfam_name='3D',
        dpam_name='PRISM3',
        fsam_name='R1C2',
    )
    logger.info("Mock core-throughput, FPAM/FSAM, and flux calibration factors created")

    host_dir = os.path.join(e2edata_path, 'SPEC_sims/non-coron')
    psf_dir = os.path.join(e2edata_path, 'SPEC_sims/coron')
    host_files = sorted(glob.glob(os.path.join(host_dir, '*_l4_.fits')))
    comp_files = sorted(glob.glob(os.path.join(psf_dir, '*_l4_.fits')))

    if not host_files:
        raise FileNotFoundError("No non-coronagraphic L4 files found for VAP test.")
    if not comp_files:
        raise FileNotFoundError("No PSF-subtracted L4 files found for VAP test.")
    logger.info(
        f"Non-coronagraphic L4s: {[os.path.basename(f) for f in host_files]}"
    )
    logger.info(
        f"PSF-subtracted L4s: {[os.path.basename(f) for f in comp_files]}"
    )

    noncoron_path = [Image(f) for f in host_files]
    psf_path = [Image(f) for f in comp_files]

    noncoron_images = Dataset(noncoron_path)
    psf_images = Dataset(psf_path)


    # ------------------------------------------------------------------
    # Test 1: L4 Spectroscopy Input Data Validation
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info('Test 1: L4 Spectroscopy Input Data Validation')
    logger.info("=" * 80)

    # Step: Check CGI filename, headers, and wavelength grids for all L4 images
    image_groups = [
        ("Non-coronagraphic L4 image", noncoron_images),
        ("PSF-subtracted L4 image", psf_images),
    ]

    for group_label, images in image_groups:
        for idx, img in enumerate(images, start=1):
            base_label = f"{group_label} {idx}"

            # Header for this image
            logger.info("-" * 80)
            logger.info(base_label)
            logger.info("-" * 80)

            # Filename + basic header checks
            check_filename_convention(img.filename, 'cgi_*_l4_.fits', f"    ", logger, data_level='l4_')
            verify_header_keywords(img.ext_hdr, {'DATALVL': 'L4', 'DPAMNAME': 'PRISM3', 'BUNIT': 'photoelectron/s'}, f"    ", logger)
            verify_header_keywords(img.pri_hdr, {'PA_APER'}, f"    ", logger)
            verify_header_keywords(img.ext_hdr, {'WAVLEN0', 'WV0_X', 'WV0_XERR', 'WV0_Y', 'WV0_YERR', 'WV0_DIMX', 'WV0_DIMY'}, f"    ", logger)

            # Wavelength grid checks (monotonicity + NaNs)
            wave = img.hdu_list['SPEC_WAVE'].data.copy()
            logger.info(f"    wavelength grid (nm): {wave}")

            if np.all(np.diff(wave) >= 0) or np.all(np.diff(wave) <= 0):
                logger.info("    wavelength grid is monotonic: PASS")
            else:
                logger.error("    wavelength grid is monotonic: FAIL")

            if np.isnan(wave).any():
                logger.error("    No NaNs in wavelength grid: FAIL")
            else:
                logger.info("    No NaNs in wavelength grid: PASS")


    # ------------------------------------------------------------------
    # Test 2: Unocculted Star in Astrophysical Units
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info('Test 2: Unocculted Star in Astrophysical Units')
    logger.info("=" * 80)

    # Step: Build a slit-transmission calibration product (map, x, y) using the first non-coronagraphic wavelength grid
    reference_wave = noncoron_images[0].hdu_list['SPEC_WAVE'].data
    slit_map = np.ones((1, reference_wave.size), dtype=float)
    slit_x = np.array([noncoron_images[0].ext_hdr.get('WV0_X', 0.0)])
    slit_y = np.array([noncoron_images[0].ext_hdr.get('WV0_Y', 0.0)])
    slit_transmission = SlitTransmission(slit_map, pri_hdr = noncoron_images[0].pri_hdr, ext_hdr = noncoron_images[0].ext_hdr, x_offset = slit_x, y_offset = slit_y, input_dataset = Dataset([noncoron_images[0]]))
    logger.info(f"Slit transmission map sample (first 5 bins): {slit_map[0][:5]}")

    # Step: Verify spectroscopy headers, extensions, COL_COR, and core throughput
    required_exts = ['SPEC', 'SPEC_ERR', 'SPEC_WAVE']
    image_groups = [
        ("Non-coronagraphic L4 image", noncoron_images, False),
        ("PSF-subtracted L4 image", psf_images, True),
    ]

    for group_label, images, is_coron in image_groups:
        for idx, img in enumerate(images, start=1):
            base_label = f"{group_label} {idx}"

            logger.info("-" * 80)
            logger.info(base_label)
            logger.info("-" * 80)

            # Extensions
            present = [ext for ext in required_exts if ext in img.hdu_list]
            missing = [ext for ext in required_exts if ext not in img.hdu_list]

            if missing:
                logger.error(
                    f"    Extensions present {present}, missing {missing}. FAIL"
                )
            else:
                logger.info(
                    f"    Extensions {present} present. PASS"
                )

            # COL_COR
            col_cor_val = img.ext_hdr.get('COL_COR', None)
            if col_cor_val is None:
                logger.warning(
                    "    COL_COR not found in header. Using default value of 1.0."
                )
                col_cor_val = 1.0
            else:
                logger.info(f"    COL_COR found in header: {col_cor_val}")

            # SPEC BUNIT precondition
            spec_bunit_input = img.hdu_list['SPEC'].header.get('BUNIT')
            if spec_bunit_input != "photoelectron/s/bin":
                logger.error(
                    f"    SPEC BUNIT before flux calibration: {spec_bunit_input}. FAIL."
                )
                # Skip further flux calibration for this image
                continue
            else:
                logger.info(
                    f"    SPEC BUNIT before flux calibration: {spec_bunit_input}. PASS."
                )

            # PSF-subtracted images: core throughput + CT_THRU grid + CTCOR 
            if is_coron:
                try:
                    ct_factor, _ = l4_to_tda.apply_core_throughput_correction(
                        img, ct_cal, fpamfsam_cal, logr=False
                    )
                    spec_hdr = img.hdu_list['SPEC'].header
                    ctcor_flag = spec_hdr.get('CTCOR', False)
                    ok = ctcor_flag and np.isfinite(ct_factor) and (ct_factor > 0)
                    logger.info(
                        f"    Core throughput correction applied. "
                        f"CTCOR={ctcor_flag}, CTFAC={ct_factor:.4f}. "
                        f"{'PASS' if ok else 'FAIL'}"
                    )
                except Exception as exc:
                    logger.error(
                        f"    Core throughput correction failed: {exc}. FAIL"
                    )

                # ALGO_THRU extension existence check
                if 'ALGO_THRU' in img.hdu_list:
                    logger.info("    ALGO_THRU extension present. PASS")
                else:
                    logger.error("    ALGO_THRU extension not present in PSF-subtracted L4 image. FAIL")

                # ALGO_THRU shape check (should be 1-D matching SPEC shape)
                if 'ALGO_THRU' in img.hdu_list:
                    algo_thru = img.hdu_list['ALGO_THRU'].data
                    spec_shape = img.hdu_list['SPEC'].data.shape
                    if algo_thru.ndim == 1 and algo_thru.shape == spec_shape:
                        logger.info(
                            f"    ALGO_THRU extension is 1-D with shape {algo_thru.shape} matching SPEC. PASS"
                        )
                    else:
                        logger.error(
                            f"    ALGO_THRU extension has shape {algo_thru.shape}, expected 1-D shape {spec_shape} matching SPEC. FAIL"
                        )

                # Check CTCOR for convert_spec_to_flux
                comp_ctcor = img.hdu_list['SPEC'].header.get('CTCOR', False)
                if not comp_ctcor:
                    logger.error(
                        "    SPEC header missing CTCOR=True after attempted core "
                        "throughput correction; skipping convert_spec_to_flux for "
                        "companion spectrum. FAIL"
                    )
                    continue
                else:
                    logger.info(
                        "    SPEC header has CTCOR=True; proceeding with "
                        "convert_spec_to_flux. PASS"
                    )
            # TODO: why not outside the image loop?
            # correct for slit transmission and algo thru
            corrected_ds = l4_to_tda.apply_slit_transmission(Dataset([img]), slit_transmission)
            # Flux calibration 
            calibrated_img = l4_to_tda.convert_spec_to_flux(
                corrected_ds, fluxcal_factor
            )
            calibrated_spec = calibrated_img[0].hdu_list['SPEC'].data
            calibrated_err = calibrated_img[0].hdu_list['SPEC_ERR'].data

            bunit = calibrated_img[0].hdu_list['SPEC'].header.get('BUNIT')
            if bunit == "erg/(s*cm^2*AA)":
                logger.info(
                    "    Spectrum calibrated. BUNIT=erg/(s*cm^2*AA). PASS"
                )
            else:
                logger.error(
                    f"    Spectrum BUNIT={bunit}. Expected erg/(s*cm^2*AA). FAIL"
                )

            logger.info(
                f"    Spectrum sample (first 5 bins): {calibrated_spec[:5]}"
            )
            logger.info(
                f"    Flux uncertainties (first 5 bins): {calibrated_err[0][:5]}"
            )




    # ------------------------------------------------------------------
    # Test 3: Companion-to-Host Flux-Ratio (PSF-subtracted)
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info('Test 3: Companion-to-Host Flux-Ratio (PSF-subtracted)')
    logger.info("=" * 80)

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
        flux_ratio = np.array([])
        ratio_err = np.array([])
        wavelength = np.array([])
    else:
        host_images = [Image(f) for f in all_host_files]
        comp_images = [Image(f) for f in all_comp_files]

        logger.info(f'Number of host L4 spectra used: {len(host_images)}')
        logger.info(f'Number of companion L4 spectra used: {len(comp_images)}')
        logger.info(f'Host rotation angles: {[img.pri_hdr.get("PA_APER") for img in host_images]}')
        logger.info(f'Companion rotation angles: {[img.pri_hdr.get("PA_APER") for img in comp_images]}')

        # Check that all host and companion cubes contain a SPEC_WAVE (wavelength) extension.
        host_wave_ext = all('SPEC_WAVE' in img.hdu_list for img in host_images)
        comp_wave_ext = all('SPEC_WAVE' in img.hdu_list for img in comp_images)
        logger.info(
            f"All host cubes contain SPEC_WAVE extension: {'PASS' if host_wave_ext else 'FAIL'}"
        )
        logger.info(
            f"All companion cubes contain SPEC_WAVE extension: {'PASS' if comp_wave_ext else 'FAIL'}"
        )

        # Combine all host and all companion images using inverse-variance weighting.
        # This produces a single 1-D host image and a single 1-D companion image
        # that can be passed into compute_spec_flux_ratio.
        host_ds = Dataset(host_images)
        comp_ds = Dataset(comp_images)

        host_spec, host_wave, host_err, host_rotation_angles = l4_to_tda.combine_spectra(host_ds)
        comp_spec, comp_wave, comp_err, comp_rotation_angles = l4_to_tda.combine_spectra(comp_ds)

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

        # Apply core throughput correction to the combined companion spectrum if coronagraphic
        is_coron_comb = comp_comb.ext_hdr.get('FSMLOS', 0) == 1
        if is_coron_comb:
            try:
                ct_factor_comb, _ = l4_to_tda.apply_core_throughput_correction(
                    comp_comb, ct_cal, fpamfsam_cal, logr=False
                )
                spec_hdr_comb = comp_comb.hdu_list['SPEC'].header
                ctcor_flag_comb = spec_hdr_comb.get('CTCOR', False)
                ok_comb = ctcor_flag_comb and np.isfinite(ct_factor_comb) and (ct_factor_comb > 0)
                logger.info(
                    "Core throughput correction applied to combined companion spectrum. "
                    f"CTCOR={ctcor_flag_comb}, CTFAC={ct_factor_comb:.4f}. "
                    f"{'PASS' if ok_comb else 'FAIL'}"
                )
                if not ok_comb:
                    # Do not attempt flux ratio if CT correction looks invalid
                    flux_ratio = np.array([])
                    ratio_err = np.array([])
                    wavelength = np.array([])
                    logger.info('=' * 80)
                    logger.info('Spectroscopy L4->TDA VAP Test Completed')
                    logger.info('=' * 80)
                    return
            except Exception as exc:
                logger.error(
                    f"Core throughput correction failed for combined companion spectrum: {exc}. FAIL"
                )
                flux_ratio = np.array([])
                ratio_err = np.array([])
                wavelength = np.array([])
                logger.info('=' * 80)
                logger.info('Spectroscopy L4->TDA VAP Test Completed')
                logger.info('=' * 80)
                return

        # Before computing the flux ratio, confirm that the combined host spectrum
        # is in correct units 
        host_spec_bunit = host_comb.hdu_list['SPEC'].header.get('BUNIT')
        logger.info(
            f"Combined host SPEC BUNIT before flux-ratio computation: {host_spec_bunit}"
        )

        if host_spec_bunit != "photoelectron/s/bin":
            logger.error(
                "Combined host SPEC BUNIT is not 'photoelectron/s/bin'; "
                "skipping compute_spec_flux_ratio in Test 3. FAIL"
            )
            flux_ratio = np.full_like(host_wave, np.nan, dtype=float)
            wavelength = host_wave
            ratio_err = np.full_like(host_wave, np.nan, dtype=float)
        else:
            # Compute flux ratio)
            flux_ratio, wavelength, metadata = l4_to_tda.compute_spec_flux_ratio(
                host_comb,
                comp_comb
            )
            ratio_err = metadata.get('ratio_err')

        # correct for slit transmission and algo thru
        comp_corrected_ds = l4_to_tda.apply_slit_transmission(Dataset([comp_comb]), slit_transmission)
        # Check algorithm throughput correction is applied
        comp_cal_check = l4_to_tda.convert_spec_to_flux(
            comp_corrected_ds, fluxcal_factor
        )
        comp_spec_hdr_cal = comp_cal_check[0].hdu_list['SPEC'].header
        if 'ALGOCOR' in comp_spec_hdr_cal:
            algocor_value = comp_spec_hdr_cal['ALGOCOR']
            logger.info(
                f"Algorithm throughput correction applied to combined companion spectrum. "
                f"ALGOCOR={algocor_value}. PASS"
            )
        else:
            logger.error(
                "ALGOCOR flag not found in header after flux calibration. "
                "Algorithm throughput correction may not have been applied. FAIL"
            )

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
    e2edata_dir = "/Users/jmilton/Documents/CGI/E2E_Test_Data2"

    ap = argparse.ArgumentParser(description="run the spectroscopy L4->TDA VAP test")
    ap.add_argument("-i", "--e2edata_dir", default=e2edata_dir,
                    help="directory to get input files from [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()

    test_l4_to_tda_spec_vap(args.e2edata_dir, args.outputdir)
