Implement a VAP test with logging according to the following plan/requirements for L4→TDA spectroscopy processing, aligned with the Spectroscopy FDD and tech demo phase analysis deliverables.  
Run all tests twice per detector mode—photon-counted (PC) and analog—and for each mode execute the full suite on:
- PSF-subtracted spectrograms or extracted spectra
- Non-PSF-subtracted (direct-extracted) spectrograms or spectra  
Single-roll data execute the same checks, skipping roll-averaging steps when the second roll is absent.

-----


### **Test 1: Flux-Ratio vs Wavelength**
**New test needed**
- L4 spectroscopy FITS images of the source (if analog PSF-subtracted, taken at different roll angles A and B)
    - Extract the 1-D spectrum and wavelength array from the image data and the `WAVE` extension  
- If PSF-subtracted test, L4 spectroscopy FITS image cube of the target star taken at different roll angles A and B
    - Extract the 1-D spectrum and wavelength array from the image data and the `WAVE` extension
- Spectroscopy core throughput calibration map (`spec.SpectroscopyThroughput`). 

**Test:**  
- [ ] Check and log that L4 data input complies with cgi format
- [ ] Check and log that `DATALVL=L4`, `DPAMNAME = PRISM*`, `BUNIT=photoelectron/s` for source and target images.  
- [ ] For each roll, compute the companion-to-host flux ratio `R(λ) = S_science(λ) / S_target(λ)` and log the spectrum.  
- [ ] If two roll angles are present, combine the two `R(λ)` values using exposure time weighting. Log value.


### **Test 2: Unocculted Star in Astrophysical Units**
**Use/adapt**: `tests/test_fluxcal.py::test_convert_to_flux()` (adapted for 1-D spectra)  
**Functions**: `l4_to_tda.convert_spec_to_flux()`  
**Inputs:**  
- L4 spectroscopy FITS cube of the unocculted host star (per roll used in Test 1)  
- Absolute flux calibration factor (`FluxcalFactor`)  
- Slit-transmission map from `spec.slit_transmission()`  

**Test:**  
- [ ] Verify spectroscopy headers (`DATALVL=L4`, `DPAMNAME=PRISM*`, `BUNIT=photoelectron/s`, `ROLL`) and confirm `SPEC`, `SPEC_ERR`, `SPEC_WAVE` extensions exist.  
- [ ] Interpolate the slit-transmission map onto the `SPEC_WAVE` grid if necessary and log the throughput applied per roll.  
- [ ] Run `convert_spec_to_flux()` with the fluxcal factor and slit-transmission vector. Ensure `SPEC`/`SPEC_ERR` units update to `erg/(s*cm^2*Å)` and uncertainties propagate.  
- [ ] Log whether `COL_COR` was applied and confirm the wavelength array remains strictly increasing.  
- [ ] Log the value 


### **Test 3: Companion‑to‑Host Flux‑Ratio ( PSF‑subtracted)**
**Use/adapt**: `tests/test_fluxcal.py::test_convert_to_flux()`  
**Functions**: `l4_to_tda.convert_to_flux()`, `l4_to_tda.determine_flux()`  
**Inputs:**:
- L4 PSF‑subtracted cube that contains the companion data
- L4 non‑PSF‑subtracted (un‑occulted) cube of the host star (the reference star).
- Absolute flux‑calibration factor (FluxcalFactor).
- Slit‑transmission map (wavelength-dependent transmission for the FSAM slit)

**Test:**  
- [ ] Check data format and header values per L4 documentation (DATALVL=L4, DPAMNAME = PRISM*, BUNIT=photoelectron/s, ROLL)
- [ ] Confirm both L4 cubes contain a WAVE extension (? confirm how this is stored) with a monotonic wavelength array and that the arrays are identical in length.
- [ ] Extract 1‑D spectra from the PSF‑subtracted cube (companion(λ)) and from the non‑PSF‑subtracted cube (host(λ)).
- [ ] Convert host and companion spectra to physical units - call convert_to_flux() on host(λ) and companion(λ).
- [ ] Apply slit‑transmission loss to host and companion fluxes.
- [ ] Apply color‑correction if present to host and companion fluxes.
- [ ] Compute flux‑ratio per roll. For each roll A and B): R(λ)= companion_corrected(λ)/host_corrected(λ)
- [ ] Roll‑averaging. If both rolls are present, weight the flux ratio by exposure time.


### **Test 4: Line Spread Function Characterization**
**Use/adapt**: `tests/test_spec.py::test_fit_line_spread_function()`  
**Function**: `spec.fit_line_spread_function()`  
**Input data**: Narrowband + prism calibration dataset with `WAVE` map and slit metadata

**Test:**
- [ ] Check and log that L4 data input complies with cgi format
- [ ] Check and log that `DATALVL=L4`, `DPAMNAME = PRISM*`, `BUNIT=photoelectron/s` for source and target images.  
- [ ] Check and log that the LSF fit uses the correct narrowband subset (`CFAMNAME=` 3D or 2C) 
- [ ] Check and log that returned `LineSpread` object reports mean wavelength, FWHM, and amplitude with uncertainties
- [ ] Check and log that FWHM values stay within algorithm bounds (5–30 pixels?) or are flagged


### **Test 5: Slit Transmission Factor**
**Use/adapt**: `tests/test_spec.py::test_slit_transmission_map()` *(add if missing)*  
**Function**: `spec.slit_transmission()`  
**Input data**:
- Averaged spectra with the slit in (FSAMNAME = Slit) for each FSM position.
- Averaged spectra with the slit open (FSAMNAME = Open) for the same FSM grid.
- Target-pixel grid for interpolation

**Test:**
- [ ] Check and log that L4 data input complies with cgi format
- [ ] Check and log that `DATALVL=L4`, `DPAMNAME = PRISM*`, `BUNIT=photoelectron/s` for source and target images.  
- [ ] For each FSM position, compute the transmission ratio T(λ)=S_slit(λ)/S_open(λ).
- [ ] Interpolate the ratios onto the grid
- [ ] Check and log that the returned transmission map shape matches the requested grid and that all values are (0, 1]. If NaNs exist, they should match locations of masked pixels (if any).
- [ ] Check and log that interpolation handles 1-D and 2-D FSM grids.


