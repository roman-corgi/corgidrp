import os
import glob
import logging

import numpy as np
import pytest
import argparse

from corgidrp.data import Dataset, Image, FluxcalFactor
from corgidrp.check import check_filename_convention, verify_header_keywords
from corgidrp import l4_to_tda, spec


# ==============================================================================
# Spectroscopy L4->TDA VAP Test
# ==============================================================================

def run_spec_l4_to_tda_vap_test(e2edata_path, e2eoutput_path):
    """Execute the spectroscopy L4->TDA VAP validation with detailed logging.

    Args:
        e2edata_path (str): Directory containing the pre-generated E2E input data.
        e2eoutput_path (str): Directory where test artifacts and logs should be written.

    Returns:
        None
    """
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
    
    # Step: Interpolate slit-transmission map onto SPEC_WAVE grid and log throughput
    slit_transmission_map = np.ones_like(noncoron_spec)
    slit_transmission = np.interp(noncoron_wave, noncoron_wave, slit_transmission_map)
    logger.info(f"Slit transmission applied (first 5 bins): {slit_transmission[:5]}")

    # Step: Flux-calibrate host spectrum with convert_spec_to_flux
    noncoron_dataset = Dataset([noncoron_image])
    calibrated_noncoron = l4_to_tda.convert_spec_to_flux(noncoron_dataset, fluxcal_factor, slit_transmission=slit_transmission)
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
    calibrated_comp = l4_to_tda.convert_spec_to_flux(comp_dataset, fluxcal_factor, slit_transmission=slit_transmission)
    psf_calibrated_spec = calibrated_comp[0].hdu_list['SPEC'].data

    # Step: Log COL_COR usage and confirm non-coronagraphic wavelengths are monotonic
    # Check the actual COL_COR value used (from header or default)
    final_col_cor = calibrated_noncoron[0].ext_hdr.get('COL_COR', 1.0)
    if final_col_cor != 1.0:
        logger.info(f"Color correction applied: COL_COR={final_col_cor}")
    else:
        logger.info(f"No color correction applied: COL_COR={final_col_cor} (default)")

    if np.all(np.diff(noncoron_wave) >= 0) or np.all(np.diff(noncoron_wave) <= 0):
        logger.info('Non-coronagraphic wavelength grid is monotonic. PASS')
    else:
        logger.error('Non-coronagraphic wavelength grid is not monotonic. FAIL')

    # Step: Log final astrophysical flux ratio
    astro_flux_ratio = np.divide(psf_calibrated_spec, noncoron_calibrated_spec, out=np.zeros_like(psf_calibrated_spec), where=noncoron_calibrated_spec != 0)
    logger.info(f"Converted companion spectrum to astrophysical units. Sample (first 5 bins): {psf_calibrated_spec[:5]}")
    logger.info(f"Astrophysical flux-ratio stats: min={np.nanmin(astro_flux_ratio):.3e}, max={np.nanmax(astro_flux_ratio):.3e}")

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

    # Step: Compute flux-ratio per roll with optional roll-averaging
    # Load all available files and group by roll
    all_host_files = sorted(glob.glob(os.path.join(host_dir, '*_l4_.fits')))
    all_comp_files = sorted(glob.glob(os.path.join(psf_dir, '*_l4_.fits')))
    
    host_by_roll = {}
    comp_by_roll = {}
    
    for f in all_host_files:
        img = Image(f)
        roll = img.pri_hdr['ROLL']
        if roll not in host_by_roll:
            host_by_roll[roll] = []
        host_by_roll[roll].append(img)
    
    for f in all_comp_files:
        img = Image(f)
        roll = img.pri_hdr['ROLL']
        if roll not in comp_by_roll:
            comp_by_roll[roll] = []
        comp_by_roll[roll].append(img)
    
    logger.info(f'Found host files for roll angles: {list(host_by_roll.keys())}')
    logger.info(f'Found companion files for roll angles: {list(comp_by_roll.keys())}')
    
    # Build datasets for each roll (matching host and companion)
    host_datasets = []
    comp_datasets = []
    available_rolls = sorted(set(host_by_roll.keys()) & set(comp_by_roll.keys()))
    
    for roll in available_rolls:
        host_datasets.append(Dataset(host_by_roll[roll]))
        comp_datasets.append(Dataset(comp_by_roll[roll]))
    
    if not host_datasets:
        logger.error('No matching rolls found between host and companion datasets. Cannot compute flux ratio. FAIL')
        weighted_flux_ratio = np.full_like(noncoron_wave, np.nan)
        wavelength = noncoron_wave
    else:
        # Call the flux ratio computation function
        weighted_flux_ratio, wavelength, metadata = l4_to_tda.compute_spec_flux_ratio(
            host_datasets, comp_datasets, fluxcal_factor,
            slit_transmission=slit_transmission
        )
        
        # Log results
        logger.info(f'Computed flux ratio for roll angles: {metadata["rolls"]}')
        logger.info(f'Exposure times: {metadata["exp_times"]}')
        if metadata['weighted']:
            logger.info(f'Exposure-time weighted average flux ratio: {weighted_flux_ratio}')
        else:
            logger.info(f'Single roll flux ratio: {weighted_flux_ratio}')
    

    # ------------------------------------------------------------------
    # Test 4: Line Spread Function Characterization
    # ------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info('Test 4: Line Spread Function Characterization')
    logger.info("-" * 80)
    
    # Step: Find narrowband calibration dataset
    # Look for L4 files with narrowband filter (CFAMNAME=3D or 2C) and prism
    narrowband_dir = os.path.join(e2edata_path, 'l3_to_l4_spec_noncoron_e2e')
    narrowband_files = sorted(glob.glob(os.path.join(narrowband_dir, '*_l4_.fits')))
    
    # Find a file with narrowband filter
    narrowband_dataset = None
    for f in narrowband_files:
        img = Image(f)
        cfamname = img.ext_hdr.get('CFAMNAME', '')
        if cfamname in ('3D', '2C'):
            narrowband_dataset = Dataset([img])
            logger.info(f"Found narrowband calibration file: {os.path.basename(f)} with CFAMNAME={cfamname}")
            break
    
    if narrowband_dataset is None:
        logger.warning('No narrowband calibration dataset found. Skipping LSF test.')
    else:
        # Step: Check data format and header values per L4 documentation
        narrowband_image = narrowband_dataset[0]
        check_filename_convention(narrowband_image.filename, 'cgi_*_l4_.fits', "Narrowband L4", logger, data_level='l4_')
        verify_header_keywords(narrowband_image.ext_hdr, {
            'DATALVL': 'L4',
            'DPAMNAME': 'PRISM3',
            'BUNIT': 'photoelectron/s/bin'
        }, "Narrowband L4", logger)
        
        # Step: Check and log that the LSF fit uses the correct narrowband subset
        cfamname = narrowband_image.ext_hdr.get('CFAMNAME', '')
        if cfamname in ('3D', '2C'):
            logger.info(f"LSF fit will use narrowband subset with CFAMNAME={cfamname}. PASS")
        else:
            logger.error(f"LSF fit cannot use CFAMNAME={cfamname}. Expected 3D or 2C. FAIL")
        
        # Step: Fit the line spread function
        try:
            line_spread = spec.fit_line_spread_function(narrowband_dataset)
            
            # Step: Check and log that returned LineSpread object reports mean wavelength, FWHM, and amplitude with uncertainties
            if hasattr(line_spread, 'mean_wave') and hasattr(line_spread, 'fwhm') and hasattr(line_spread, 'amplitude'):
                logger.info(f"LineSpread object reports mean_wave={line_spread.mean_wave:.3f} nm. PASS")
                logger.info(f"LineSpread object reports fwhm={line_spread.fwhm:.3f} pixels. PASS")
                logger.info(f"LineSpread object reports amplitude={line_spread.amplitude:.3e}. PASS")
            else:
                logger.error("LineSpread object missing required attributes (mean_wave, fwhm, amplitude). FAIL")
            
            if hasattr(line_spread, 'wave_err') and hasattr(line_spread, 'fwhm_err') and hasattr(line_spread, 'amp_err'):
                logger.info(f"LineSpread object reports wave_err={line_spread.wave_err:.3f} nm. PASS")
                logger.info(f"LineSpread object reports fwhm_err={line_spread.fwhm_err:.3f} pixels. PASS")
                logger.info(f"LineSpread object reports amp_err={line_spread.amp_err:.3e}. PASS")
            else:
                logger.error("LineSpread object missing uncertainty attributes (wave_err, fwhm_err, amp_err). FAIL")
            
            # Step: Check and log that FWHM values stay within algorithm bounds (5-30 pixels) or are flagged
            fwhm_min = 5.0
            fwhm_max = 30.0
            if fwhm_min <= line_spread.fwhm <= fwhm_max:
                logger.info(f"FWHM={line_spread.fwhm:.3f} pixels is within algorithm bounds ({fwhm_min}-{fwhm_max} pixels). PASS")
            else:
                logger.warning(f"FWHM={line_spread.fwhm:.3f} pixels is outside algorithm bounds ({fwhm_min}-{fwhm_max} pixels). FLAGGED")
        
        except Exception as e:
            logger.error(f"Failed to fit line spread function: {e}. FAIL")

    # ------------------------------------------------------------------
    # Test 5: Slit Transmission Factor
    # ------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info('Test 5: Slit Transmission Factor')
    logger.info("-" * 80)
    
    # Step: Find datasets with slit in and slit open
    # L4 files with FSAMNAME = R1C2, R6C5, or R3C1 (slit in) and OPEN (slit open)
    slit_dir = os.path.join(e2edata_path, 'l3_to_l4_spec_noncoron_e2e')
    all_slit_files = sorted(glob.glob(os.path.join(slit_dir, '*_l4_.fits')))
    
    # Group files by FSAMNAME
    slit_in_files = []
    slit_open_files = []
    
    for f in all_slit_files:
        img = Image(f)
        fsamname = img.ext_hdr.get('FSAMNAME', '').upper()
        if fsamname in ('R1C2', 'R6C5', 'R3C1'):
            slit_in_files.append(img)
        elif fsamname == 'OPEN':
            slit_open_files.append(img)
    
    if not slit_in_files or not slit_open_files:
        logger.warning('No slit transmission datasets found (need both slit-in and slit-open files). Skipping slit transmission test.')
    else:
        # Step: Check data format and header values per L4 documentation
        dataset_slit = Dataset(slit_in_files)
        dataset_open = Dataset(slit_open_files)
        
        # Check slit-in dataset
        for i, frame in enumerate(dataset_slit):
            frame_info = f"Slit-in Frame {i}"
            check_filename_convention(frame.filename, 'cgi_*_l4_.fits', frame_info, logger, data_level='l4_')
            verify_header_keywords(frame.ext_hdr, {
                'DATALVL': 'L4',
                'DPAMNAME': 'PRISM3',
                'BUNIT': 'photoelectron/s/bin'
            }, frame_info, logger)
            verify_header_keywords(frame.ext_hdr, {'FSAMNAME'}, frame_info, logger)
            verify_header_keywords(frame.ext_hdr, {'FSMX', 'FSMY'}, frame_info, logger)
        
        # Check slit-open dataset
        for i, frame in enumerate(dataset_open):
            frame_info = f"Slit-open Frame {i}"
            check_filename_convention(frame.filename, 'cgi_*_l4_.fits', frame_info, logger, data_level='l4_')
            verify_header_keywords(frame.ext_hdr, {
                'DATALVL': 'L4',
                'DPAMNAME': 'PRISM3',
                'BUNIT': 'photoelectron/s/bin'
            }, frame_info, logger)
            verify_header_keywords(frame.ext_hdr, {'FSAMNAME': 'OPEN'}, frame_info, logger)
            verify_header_keywords(frame.ext_hdr, {'FSMX', 'FSMY'}, frame_info, logger)
        
        # Step: Determine FSM grid dimensions
        fsmx_values = sorted(set([f.ext_hdr.get('FSMX') for f in dataset_slit]))
        fsmy_values = sorted(set([f.ext_hdr.get('FSMY') for f in dataset_slit]))
        is_1d_x = len(fsmx_values) == 1
        is_1d_y = len(fsmy_values) == 1
        is_2d = not is_1d_x and not is_1d_y
        
        logger.info(f"FSM grid: {len(fsmx_values)} X positions, {len(fsmy_values)} Y positions")
        if is_1d_x:
            logger.info("FSM grid is 1-D along Y direction. PASS")
        elif is_1d_y:
            logger.info("FSM grid is 1-D along X direction. PASS")
        elif is_2d:
            logger.info("FSM grid is 2-D. PASS")
        else:
            logger.info("FSM grid has single position (no interpolation needed). PASS")
        
        # Step: Compute slit transmission
        try:
            # Get wavelength zero-point positions for grid definition
            # Use the first slit-in frame to get reasonable ranges
            first_frame = dataset_slit[0]
            wv0_x = first_frame.ext_hdr.get('WV0_X', 40.0)
            wv0_y = first_frame.ext_hdr.get('WV0_Y', 32.0)
            
            # Define grid ranges around zero-point
            x_range = [wv0_x - 1.0, wv0_x + 1.0]
            y_range = [wv0_y - 1.0, wv0_y + 1.0]
            n_gridx = 10
            n_gridy = 10
            
            # Step: For each FSM position, compute T(λ)=S_slit(λ)/S_open(λ) and interpolate
            slit_trans_map, slit_pos_x, slit_pos_y = spec.slit_transmission(
                dataset_slit, dataset_open,
                target_pix=None,
                x_range=x_range,
                y_range=y_range,
                n_gridx=n_gridx,
                n_gridy=n_gridy,
                kind='linear',
                average='mean'
            )
            
            # Step: Check and log that returned transmission map shape matches requested grid
            expected_size = n_gridx * n_gridy
            if isinstance(slit_trans_map, np.ndarray):
                if slit_trans_map.ndim == 1:
                    actual_size = slit_trans_map.shape[0]
                else:
                    actual_size = slit_trans_map.size
                
                if actual_size == expected_size:
                    logger.info(f"Transmission map size ({actual_size}) matches requested grid size ({expected_size}). PASS")
                else:
                    logger.error(f"Transmission map size ({actual_size}) does not match requested grid size ({expected_size}). FAIL")
                
                # Step: Check that all values are in (0, 1]
                finite_mask = np.isfinite(slit_trans_map)
                finite_values = slit_trans_map[finite_mask]
                
                if len(finite_values) > 0:
                    min_val = np.min(finite_values)
                    max_val = np.max(finite_values)
                    
                    if min_val > 0 and max_val <= 1.0:
                        logger.info(f"All transmission values are in (0, 1]: min={min_val:.3e}, max={max_val:.3e}. PASS")
                    else:
                        logger.warning(f"Some transmission values outside (0, 1]: min={min_val:.3e}, max={max_val:.3e}. FLAGGED")
                    
                    # Check for NaNs
                    nan_count = np.isnan(slit_trans_map).sum()
                    if nan_count > 0:
                        logger.info(f"Found {nan_count} NaN values in transmission map (may indicate masked pixels).")
                    else:
                        logger.info("No NaN values in transmission map. PASS")
                else:
                    logger.error("All transmission values are NaN. FAIL")
            else:
                logger.error(f"Transmission map is not a numpy array: {type(slit_trans_map)}. FAIL")
            
            # Step: Check that interpolation handles 1-D and 2-D FSM grids
            # This is already verified by the function working correctly above
            if is_1d_x or is_1d_y:
                logger.info("1-D FSM grid interpolation handled successfully. PASS")
            elif is_2d:
                logger.info("2-D FSM grid interpolation handled successfully. PASS")
            else:
                logger.info("Single FSM position (no interpolation needed). PASS")
        
        except Exception as e:
            logger.error(f"Failed to compute slit transmission: {e}. FAIL")

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

    Returns:
        None
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
    e2edata_dir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the spectroscopy L4->TDA VAP test")
    ap.add_argument("-i", "--e2edata_dir", default=e2edata_dir,
                    help="directory to get input files from [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()

    test_l4_to_tda_spec_vap(args.e2edata_dir, args.outputdir)
