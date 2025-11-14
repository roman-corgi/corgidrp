Implement a VAP test with logging according to the following plan/ requirements for L4->TDA processing, based on the Polarimetry FDD:

-----

### **Test 1: Flux Ratio Noise vs Separation**
**Use/adapt**: `test_flux_ratio_noise_curve.py::test_expected_flux_ratio_noise()`  
**Function**: `l4_to_tda.compute_flux_ratio_noise()`  
_**Input data**:_ 
- L4 PSF-subtracted Stokes I
- Unocculted star dataset
- ND filter calibration
- Core throughput calibration

**1.1 Data format:**

- [ ] Check/log that L4 data input complies with cgi format
- [ ] Check/log that DATALVL = L4
- [ ] Check/log that BUNIT = photoelectron/s
- [ ] Check/log that output is fits file with extension header 'FRN_CRV'

**1.2 Flux ratio noise vs separation:**

- [ ] Check/log that FRN_CRV has correct shape [2+M, N] where:
        - Row 0: separations in pixels
        - Row 1: separations in mas
        - Rows 2+: FRN values for M KL mode truncations
- [ ] Check/log that separations are within IWA-OWA range
- [ ] Check/log that FRN values are positive
- [ ] Check/log that interpolation works for custom separation grid


### **Test 2: Companion Polarization Properties**
**Use/adapt**: `test_polarimetry.py::test_calc_pol_p_and_pa_image()`  
**Function**: `l4_to_tda.calc_pol_p_and_pa_image()`  
**Input data**: L4 Stokes cube [I, Q, U, ...] with known (p, theta) polarization

**2.1 Data format:**
- [ ] Check/log that L4 data input complies with cgi format
- [ ] Check/log that DATALVL = L4
- [ ] Check/log that BUNIT = photoelectron/s

**2.2 Calculate and log polarization fraction, pSNR, and polarization position angle of the companion:**
- [ ] Check/log that polarized intensity P is computed correctly
- [ ] Check/log that fractional polarization p matches input (within error)
- [ ] Check/log that polarization position angle matches input angle (within error)
- [ ] Check/log that pSNR computed correctly as p / σ_p
- [ ] Check/log that error propagation from Q, U is correct
- [ ] Check/log that flags propagate from I, Q, U to P, p, polarization angle


### **Test 3: Companion Photometry Test**
**Create (new test needed)**: `test_l4_companion_photometry()`  
**Functions**: `l4_to_tda.determine_flux()`, `l4_to_tda.determine_app_mag()`  
**Input data**: 
- L4 Stokes cube with known companion flux
- Unocculted star dataset with known flux
- Flux calibration factor (mock)
- Optional: COL_COR in header for color correction test

**3.1 Data format:**
- [ ] Check/log that L4 data input complies with cgi format
- [ ] Check/log that DATALVL = L4
- [ ] Check/log that BUNIT = photoelectron/s

**3.2 Calculate the apparent magnitude and integrated flux ratio of the companion:**
- [ ] Check/log that companion flux is computed and reported
- [ ] Check/log that flux ratio matches input ratio (within error)
- [ ] Check/log that apparent magnitude is computed using Vega zeropoint
- [ ] Check/log that color correction is applied if COL_COR in header
- [ ] Check/log that uncertainty propagation is done through aperture sum


### **Test 4: Extended Source (Disk) Azimuthal Stokes Test**
**Use/adapt**: `test_compute_QphiUphi.py` (4 existing tests)  
**Function**: `l4_to_tda.compute_QphiUphi()`  
**Input data**: L4 Stokes cube [I, Q, U, V] with tangentially-polarized disk, stellar center in header

**4.1 Data format:**
- [ ] Check/log that L4 data input complies with cgi format
- [ ] Check/log that DATALVL = L4
- [ ] Check/log that BUNIT = photoelectron/s

**4.2 Azimuthal components test:**
Calculate and provide azimuthal components Qφ and Uφ:
- [ ] Check/log that Qφ matches expected tangential polarization pattern
- [ ] Check/log that Uφ ≈ 0 for perfect tangential polarization
- [ ] Check/log that incorrect center produces nonzero Uφ
- [ ] Check/log that output shape is [6, n, m]
- [ ] Check/log error propagation σ_Qφ, σ_Uφ from σ_Q, σ_U
- [ ] Check/log that DQ flags combine Q and U masks (bitwise OR)
- [ ] Check/log using header STARLOCX/Y vs manual center


### **Test 5: Extended Source (Disk) Polarized and Total Intensity Test**
**Use/adapt**: `test_polarimetry.py::test_calc_pol_p_and_pa_image()`  
**Function**: `l4_to_tda.calc_pol_p_and_pa_image()`  
**Input data**: L4 Stokes cube [I, Q, U] with known (p, theta) polarization

**5.1 Data format:**
- [ ] Check/log that L4 data input complies with cgi format
- [ ] Check/log that DATALVL = L4
- [ ] Check/log that BUNIT = photoelectron/s

**5.2 Polarized and total intensity:**
Calculate and provide polarized and total intensity:
- [ ] Check/log that P computed correctly from Q, U
- [ ] Check/log that p = P/I matches input polarization fraction
- [ ] Check/log that EVPA matches input polarization angle
- [ ] Check/log that Output shape is [3, H, W]
- [ ] Check/log error propagation through sqrt and division
- [ ] Check/log handling I ≈ 0 (avoid divide by zero)
- [ ] Check/log that DQ flags propagate correctly

