"""
End-to-end tests for L4 to TDA polarimetry processing

These tests verify the complete workflow of L4→TDA analysis functions
for polarimetric data, including:
- Flux ratio noise curve calculation
- Companion detection and characterization (polarization, photometry)
- Extended source (disk) analysis (Qφ, Uφ, P, p, EVPA)
- HISTORY logging and metadata validation

Author: corgidrp team
"""

import os
import numpy as np
import pytest
from astropy.io import fits

import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.l4_to_tda as l4_to_tda


# ============================================================================
# Flux Ratio Noise Tests
# ============================================================================

def test_compute_flux_ratio_noise_with_stokes_cube():
    """
    Test FRN curve computation using PSF-subtracted Stokes I from L4 data
    
    Requirements:
    - Input: L4 Stokes cube with PSF-subtracted I
    - Output: FRN_CRV extension with shape [2+M, N]
    - HISTORY entry added
    """
    # Create mock L4 Stokes cube (I, Q, U, V)
    stokes_image = mocks.create_mock_stokes_image_l4(
        badpixel_fraction=0.0,
        fwhm=3.0,
        I0=1e8,
        p=0.2,
        theta_deg=30.0
    )
    stokes_image.ext_hdr['DATALVL'] = 'L4'
    stokes_image.ext_hdr['BUNIT'] = 'photoelectron/s'
    
    # Create mock unocculted star dataset
    unocculted_dataset = mocks.create_mock_stokes_image_l4(
        badpixel_fraction=0.0,
        fwhm=3.0,
        I0=1e9,  # Brighter for unocculted
        p=0.01,
        theta_deg=0.0
    )
    unocculted_dataset = data.Dataset([unocculted_dataset])
    
    # Create mock ND calibration
    nd_cal = mocks.create_nd_cal()
    
    # Create input dataset
    input_dataset = data.Dataset([stokes_image])
    
    # TODO: Implement compute_flux_ratio_noise for polarimetric data
    # This test is a placeholder for when the function is implemented
    pytest.skip("compute_flux_ratio_noise for polarimetry not yet implemented")
    
    # output_dataset = l4_to_tda.compute_flux_ratio_noise(
    #     input_dataset,
    #     NDcalibration=nd_cal,
    #     unocculted_star_dataset=unocculted_dataset
    # )
    
    # # Verify FRN_CRV extension exists
    # assert 'FRN_CRV' in output_dataset[0].hdu_list
    # frn_data = output_dataset[0].hdu_list['FRN_CRV'].data
    
    # # Check shape: [2+M, N] where row 0=pixels, row 1=mas, rows 2+=FRN values
    # assert frn_data.shape[0] >= 3
    # assert frn_data.shape[1] > 10  # At least 10 separation points
    
    # # Check separations are positive and increasing
    # sep_pixels = frn_data[0]
    # assert np.all(sep_pixels > 0)
    # assert np.all(np.diff(sep_pixels) > 0)
    
    # # Check FRN values are positive
    # frn_values = frn_data[2:]
    # assert np.all(frn_values > 0)
    
    # # Verify HISTORY entry
    # history_entries = output_dataset[0].ext_hdr['HISTORY']
    # assert any('flux ratio noise' in str(h).lower() for h in history_entries)


# ============================================================================
# Companion Detection Tests
# ============================================================================

def test_find_source_single_companion():
    """
    Test detection of a single companion in Stokes I
    
    Requirements:
    - Detect companion above SNR threshold
    - Report SNR and pixel location in header keywords
    - Add HISTORY entry
    """
    # Create mock L4 Stokes cube with companion
    # Base image with low background
    base_image = mocks.create_mock_stokes_image_l4(
        badpixel_fraction=0.0,
        fwhm=100,  # Large fwhm = diffuse background
        I0=1e6,
        p=0.0,
        theta_deg=0.0
    )
    
    # Add companion as point source in Stokes I
    companion_x, companion_y = 150, 120
    companion_flux = 5e7  # Strong signal
    y, x = np.indices(base_image.data[0].shape)
    companion_psf = companion_flux * np.exp(-((x - companion_x)**2 + (y - companion_y)**2) / (2 * 3**2))
    base_image.data[0] += companion_psf  # Add to Stokes I only
    
    base_image.ext_hdr['DATALVL'] = 'L4'
    base_image.ext_hdr['BUNIT'] = 'photoelectron/s'
    
    # Extract just Stokes I for source detection
    stokes_i_image = data.Image(base_image.data[0], 
                                 pri_hdr=base_image.pri_hdr, 
                                 ext_hdr=base_image.ext_hdr)
    
    # Detect sources
    output_image = l4_to_tda.find_source(stokes_i_image, fwhm=3.0, nsigma_threshold=5.0)
    
    # Verify source was detected
    assert 'snyx000' in output_image.ext_hdr
    
    # Parse detection: format is 'SNR,Y,X'
    detection = output_image.ext_hdr['snyx000'].split(',')
    detected_snr = float(detection[0])
    detected_y = int(detection[1])
    detected_x = int(detection[2])
    
    # Check SNR is above threshold
    assert detected_snr >= 5.0
    
    # Check location is close to injected (within 2 pixels)
    assert abs(detected_x - companion_x) <= 2
    assert abs(detected_y - companion_y) <= 2
    
    print(f"Detected companion: SNR={detected_snr:.1f}, location=({detected_x},{detected_y})")
    print(f"Expected location: ({companion_x},{companion_y})")


def test_find_source_multiple_companions():
    """Test detection of multiple companions"""
    # Create mock L4 Stokes cube with two companions
    base_image = mocks.create_mock_stokes_image_l4(
        badpixel_fraction=0.0,
        fwhm=100,
        I0=1e6,
        p=0.0,
        theta_deg=0.0
    )
    
    # Add two companions
    companions = [(150, 120, 5e7), (180, 90, 4e7)]
    y, x = np.indices(base_image.data[0].shape)
    
    for comp_x, comp_y, comp_flux in companions:
        companion_psf = comp_flux * np.exp(-((x - comp_x)**2 + (y - comp_y)**2) / (2 * 3**2))
        base_image.data[0] += companion_psf
    
    base_image.ext_hdr['DATALVL'] = 'L4'
    base_image.ext_hdr['BUNIT'] = 'photoelectron/s'
    
    # Extract just Stokes I for source detection
    stokes_i_image = data.Image(base_image.data[0], 
                                 pri_hdr=base_image.pri_hdr, 
                                 ext_hdr=base_image.ext_hdr)
    
    # Detect sources
    output_image = l4_to_tda.find_source(stokes_i_image, fwhm=3.0, nsigma_threshold=5.0)
    
    # Verify both sources detected
    assert 'snyx000' in output_image.ext_hdr
    assert 'snyx001' in output_image.ext_hdr
    
    print(f"Detected {2} companions as expected")


def test_find_source_no_detection_below_threshold():
    """Test that weak sources below threshold are not detected"""
    # Create mock with only faint background
    base_image = mocks.create_mock_stokes_image_l4(
        badpixel_fraction=0.0,
        fwhm=100,
        I0=1e6,  # Low background only
        p=0.0,
        theta_deg=0.0
    )
    
    base_image.ext_hdr['DATALVL'] = 'L4'
    base_image.ext_hdr['BUNIT'] = 'photoelectron/s'
    
    # Extract just Stokes I for source detection
    stokes_i_image = data.Image(base_image.data[0], 
                                 pri_hdr=base_image.pri_hdr, 
                                 ext_hdr=base_image.ext_hdr)
    
    # Detect sources with high threshold
    output_image = l4_to_tda.find_source(stokes_i_image, fwhm=3.0, nsigma_threshold=10.0)
    
    # Verify no detections
    assert 'snyx000' not in output_image.ext_hdr
    
    print("No false detections - test passed")


# ============================================================================
# Companion Characterization Tests
# ============================================================================

def test_companion_polarization_properties():
    """
    Test measurement of companion polarization (P, p, EVPA, pSNR)
    
    Requirements:
    - Extract P, p, EVPA at companion location
    - Compute pSNR = p / σ_p
    - Verify values match input within uncertainties
    """
    # Create mock Stokes cube with polarized companion
    p_input = 0.25
    theta_input = 45.0  # degrees
    
    stokes_image = mocks.create_mock_stokes_image_l4(
        badpixel_fraction=0.0,
        fwhm=3.0,
        I0=1e8,
        p=p_input,
        theta_deg=theta_input
    )
    
    # Calculate polarization products
    pol_image = l4_to_tda.calc_pol_p_and_pa_image(stokes_image)
    
    # Extract from center (where mock places the source)
    center = (pol_image.data.shape[1] // 2, pol_image.data.shape[2] // 2)
    
    P_measured = pol_image.data[0, center[0], center[1]]
    p_measured = pol_image.data[1, center[0], center[1]]
    evpa_measured = pol_image.data[2, center[0], center[1]]
    
    p_err = pol_image.err[0, 1, center[0], center[1]]
    evpa_err = pol_image.err[0, 2, center[0], center[1]]
    
    # Calculate pSNR
    psnr = p_measured / p_err
    
    # Verify values match input
    assert p_measured == pytest.approx(p_input, abs=3*p_err)
    assert evpa_measured == pytest.approx(theta_input, abs=3*evpa_err)
    assert psnr > 3.0  # Should have significant detection
    
    # Verify HISTORY entry
    assert 'HISTORY' in pol_image.ext_hdr
    history_entries = pol_image.ext_hdr['HISTORY']
    assert any('polarization products' in str(h).lower() for h in history_entries)
    
    print(f"Companion polarization: p={p_measured:.3f}±{p_err:.3f} (pSNR={psnr:.1f})")
    print(f"                       EVPA={evpa_measured:.1f}±{evpa_err:.1f} deg")


def test_companion_flux_ratio():
    """
    Test measurement of companion flux ratio relative to star
    
    Requirements:
    - Measure companion flux in Stokes I
    - Measure unocculted star flux
    - Calculate flux ratio (companion/star)
    """
    # This test is a placeholder for photometry functions
    # TODO: Implement when photometry functions are available for L4 data
    pytest.skip("Companion photometry functions not yet implemented for L4")


# ============================================================================
# Extended Source (Disk) Analysis Tests
# ============================================================================

def test_compute_qphi_uphi_for_disk():
    """
    Test azimuthal Stokes decomposition (Qφ, Uφ) for disk
    
    Requirements:
    - Compute Qφ, Uφ from Q, U using stellar center
    - Output shape: [I, Q, U, V, Qφ, Uφ]
    - For tangentially-polarized disk: Qφ >> Uφ
    - HISTORY includes center coordinates
    """
    # Create mock disk with tangential polarization
    disk_image = mocks.create_mock_IQUV_image()
    
    # Compute Qphi, Uphi
    result = l4_to_tda.compute_QphiUphi(disk_image)
    
    # Verify output shape
    assert result.data.shape[0] == 6  # [I, Q, U, V, Qφ, Uφ]
    
    # For a perfectly tangentially-polarized disk, Uphi should be ~0
    Uphi = result.data[5]
    assert np.allclose(Uphi, 0.0, atol=1e-6)
    
    # Verify error propagation
    assert result.err.shape == result.data.shape
    assert result.dq.shape == result.data.shape
    
    # Verify HISTORY entry includes center
    history_entries = result.ext_hdr['HISTORY']
    assert any('Q_phi/U_phi' in str(h) for h in history_entries)
    assert any('center=' in str(h) for h in history_entries)
    
    print("Qφ/Uφ computation successful for disk")


def test_disk_polarized_intensity_map():
    """
    Test computation of P, p, EVPA maps for extended disk
    
    Requirements:
    - Compute P = sqrt(Q² + U²) for entire field
    - Compute p = P / I
    - Compute EVPA = 0.5 * arctan2(U, Q)
    - Output shape: [P, p, EVPA]
    """
    # Create mock disk
    disk_image = mocks.create_mock_stokes_image_l4(
        badpixel_fraction=0.0,
        fwhm=20,  # Extended source
        I0=1e8,
        p=0.3,
        theta_deg=60.0
    )
    
    # Calculate polarization products
    pol_image = l4_to_tda.calc_pol_p_and_pa_image(disk_image)
    
    # Verify output shape
    assert pol_image.data.shape[0] == 3  # [P, p, EVPA]
    
    # Verify P, p are positive
    P = pol_image.data[0]
    p = pol_image.data[1]
    assert np.all(P >= 0)
    assert np.all((p >= 0) & (p <= 1))
    
    # Calculate median polarization fraction
    p_median = np.nanmedian(p)
    print(f"Disk median fractional polarization: p={p_median:.3f}")
    
    # Verify it's close to input (within 20% for extended source)
    assert p_median == pytest.approx(0.3, rel=0.2)
    
    # Verify HISTORY entry
    history_entries = pol_image.ext_hdr['HISTORY']
    assert any('polarization products' in str(h).lower() for h in history_entries)


def test_disk_with_wrong_center():
    """
    Test that incorrect stellar center produces nonzero Uφ
    
    This validates that center coordinate matters for azimuthal decomposition
    """
    # Create mock disk
    disk_image = mocks.create_mock_IQUV_image()
    
    # Use incorrect center (offset by 5 pixels)
    disk_image.ext_hdr['STARLOCX'] += 5.0
    disk_image.ext_hdr['STARLOCY'] += 5.0
    
    # Compute Qphi, Uphi with wrong center
    result = l4_to_tda.compute_QphiUphi(disk_image)
    
    # Uphi should NOT be zero with wrong center
    Uphi = result.data[5]
    assert not np.allclose(Uphi, 0.0, atol=1e-6)
    
    print("Confirmed: wrong center produces nonzero Uφ as expected")


# ============================================================================
# TDA Level Update Tests
# ============================================================================

def test_update_to_tda_level():
    """
    Test updating L4 data to TDA level
    
    Requirements:
    - DATALVL updated from 'L4' to 'TDA'
    - Filename updated from '_l4_' to '_tda_'
    - HISTORY entry added
    """
    # Create mock L4 dataset
    l4_image = mocks.create_mock_stokes_image_l4(
        badpixel_fraction=0.0,
        fwhm=3.0,
        I0=1e8,
        p=0.2,
        theta_deg=30.0
    )
    l4_image.ext_hdr['DATALVL'] = 'L4'
    l4_image.filename = 'cgi_0089001001001001027_20251106t1234567_l4_stokes.fits'
    
    l4_dataset = data.Dataset([l4_image])
    
    # Update to TDA
    tda_dataset = l4_to_tda.update_to_tda(l4_dataset)
    
    # Verify data level updated
    assert tda_dataset[0].ext_hdr['DATALVL'] == 'TDA'
    
    # Verify filename updated
    assert '_tda_' in tda_dataset[0].filename
    assert '_l4_' not in tda_dataset[0].filename
    
    # Verify HISTORY entry
    history_entries = tda_dataset[0].ext_hdr['HISTORY']
    assert any('TDA' in str(h) for h in history_entries)
    
    print(f"Updated: {l4_dataset[0].filename} -> {tda_dataset[0].filename}")


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_companion_workflow():
    """
    Test complete workflow: detect companion, characterize polarization
    
    Workflow:
    1. Create L4 Stokes cube with companion
    2. Detect companion in Stokes I
    3. Calculate P, p, EVPA maps
    4. Extract values at companion location
    5. Update to TDA level
    """
    # Step 1: Create mock data with companion
    p_input = 0.3
    theta_input = 45.0
    
    stokes_image = mocks.create_mock_stokes_image_l4(
        badpixel_fraction=0.0,
        fwhm=3.0,
        I0=1e8,
        p=p_input,
        theta_deg=theta_input
    )
    stokes_image.ext_hdr['DATALVL'] = 'L4'
    stokes_image.ext_hdr['BUNIT'] = 'photoelectron/s'
    stokes_image.filename = 'test_l4_stokes.fits'
    
    # Step 2: Detect companion (extract Stokes I)
    stokes_i_image = data.Image(stokes_image.data[0], 
                                 pri_hdr=stokes_image.pri_hdr, 
                                 ext_hdr=stokes_image.ext_hdr)
    detected_image = l4_to_tda.find_source(stokes_i_image, fwhm=3.0, nsigma_threshold=5.0)
    assert 'snyx000' in detected_image.ext_hdr
    
    # Parse detection
    detection = detected_image.ext_hdr['snyx000'].split(',')
    comp_snr = float(detection[0])
    comp_y = int(detection[1])
    comp_x = int(detection[2])
    
    print(f"Step 2: Detected companion at ({comp_x},{comp_y}) with SNR={comp_snr:.1f}")
    
    # Step 3: Calculate polarization products
    pol_image = l4_to_tda.calc_pol_p_and_pa_image(detected_image)
    
    # Step 4: Extract at companion location
    p_measured = pol_image.data[1, comp_y, comp_x]
    evpa_measured = pol_image.data[2, comp_y, comp_x]
    p_err = pol_image.err[0, 1, comp_y, comp_x]
    psnr = p_measured / p_err
    
    print(f"Step 4: Companion p={p_measured:.3f}±{p_err:.3f} (pSNR={psnr:.1f}), EVPA={evpa_measured:.1f}°")
    
    # Verify measurements
    assert p_measured == pytest.approx(p_input, abs=3*p_err)
    assert psnr > 3.0
    
    # Step 5: Update to TDA
    tda_dataset = l4_to_tda.update_to_tda(data.Dataset([pol_image]))
    assert tda_dataset[0].ext_hdr['DATALVL'] == 'TDA'
    assert '_tda_' in tda_dataset[0].filename
    
    print("Step 5: Updated to TDA level")
    print("Full companion workflow completed successfully!")


def test_full_disk_workflow():
    """
    Test complete workflow: analyze extended disk structure
    
    Workflow:
    1. Create L4 Stokes cube with disk
    2. Compute Qφ, Uφ (azimuthal decomposition)
    3. Compute P, p, EVPA maps
    4. Update to TDA level
    """
    # Step 1: Create mock disk
    disk_image = mocks.create_mock_stokes_image_l4(
        badpixel_fraction=0.0,
        fwhm=20,  # Extended
        I0=1e8,
        p=0.25,
        theta_deg=30.0
    )
    disk_image.ext_hdr['DATALVL'] = 'L4'
    disk_image.filename = 'test_l4_disk.fits'
    
    print("Step 1: Created mock disk")
    
    # Step 2: Compute azimuthal Stokes
    qphi_image = l4_to_tda.compute_QphiUphi(disk_image)
    assert qphi_image.data.shape[0] == 6
    
    Qphi = qphi_image.data[4]
    Uphi = qphi_image.data[5]
    print(f"Step 2: Computed Qφ (median={np.nanmedian(Qphi):.2e}) and Uφ (median={np.nanmedian(Uphi):.2e})")
    
    # Step 3: Compute polarization maps
    pol_image = l4_to_tda.calc_pol_p_and_pa_image(disk_image)
    assert pol_image.data.shape[0] == 3
    
    p_map = pol_image.data[1]
    p_median = np.nanmedian(p_map)
    print(f"Step 3: Computed polarization map (median p={p_median:.3f})")
    
    # Step 4: Update to TDA
    tda_dataset = l4_to_tda.update_to_tda(data.Dataset([pol_image]))
    assert tda_dataset[0].ext_hdr['DATALVL'] == 'TDA'
    
    print("Step 4: Updated to TDA level")
    print("Full disk workflow completed successfully!")


# ============================================================================
# Main execution for standalone testing
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("L4→TDA Polarimetry End-to-End Tests")
    print("="*60)
    
    # Test companion detection
    print("\n--- Companion Detection Tests ---")
    test_find_source_single_companion()
    test_find_source_multiple_companions()
    test_find_source_no_detection_below_threshold()
    
    # Test companion characterization
    print("\n--- Companion Characterization Tests ---")
    test_companion_polarization_properties()
    
    # Test extended source analysis
    print("\n--- Extended Source (Disk) Tests ---")
    test_compute_qphi_uphi_for_disk()
    test_disk_polarized_intensity_map()
    test_disk_with_wrong_center()
    
    # Test TDA level update
    print("\n--- TDA Level Update Tests ---")
    test_update_to_tda_level()
    
    # Test integration workflows
    print("\n--- Integration Tests ---")
    test_full_companion_workflow()
    test_full_disk_workflow()
    
    print("\n" + "="*60)
    print("All L4→TDA tests completed!")
    print("="*60)
