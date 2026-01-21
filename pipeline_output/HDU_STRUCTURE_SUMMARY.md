# HDU Structure Summary

This document describes the Header Data Unit (HDU) structure for each pipeline stage.

---

## L1 (Raw Data) - 2 HDUs

### HDU 0: Primary Header
- **Data**: None (header only)
- **Purpose**: Top-level observation metadata
- **Key Headers**: 41 total
  - Observation info (VISITID, VISTYPE, TARGET, coordinates)
  - System info (TELESCOP, INSTRUME, DETECTOR)
  - Program details (PROGNUM, CAMPAIGN, etc.)
  - Operational settings (OPGAIN, PHTCNT, FRAMET)

### HDU 1: Science Data (Extension 1)
- **Data Shape**: (1200, 2200) for SCI array
- **Data Type**: float32
- **Purpose**: Raw detector data in DN (Data Numbers)
- **Key Headers**: 113 total
  - Array configuration (NAXIS1, NAXIS2, ARRTYPE, BUNIT)
  - Detector settings (HVCBIAS, OPMODE, EXPTIME, EMGAIN_C)
  - Timing (SCTSRT, SCTEND, DATETIME)
  - Mechanisms (SPAM, FPAM, LSAM, FSAM, CFAM, DPAM positions)
  - Deformable mirror (Zernike coefficients Z2-Z14)
  - Fast steering mirror (FSMLOOP, FSMX, FSMY)
  - Data level (DATALVL = L1)

**Total L1 Headers**: 154 (41 + 113)

---

## L2a (Basic Corrections) - 5 HDUs

### HDU 0: Primary Header
- **Data**: None (header only)
- **Purpose**: Top-level observation metadata (inherited from L1 + updates)
- **Changes from L1**:
  - `ORIGIN` = DRP (was SSC)
  - All L1 observation metadata preserved

### HDU 1: Science Data (Extension 1)
- **Data Shape**: (1024, 1024) - cropped to science region
- **Data Type**: float32
- **Purpose**: Processed science data after L1→L2a corrections
- **Key Headers**: ~125 total (L1 headers + new L2a headers)
- **New L2a Headers**:
  - `DATALVL` = L2a (updated from L1)
  - `DESMEAR` = False (desmear applied?)
  - `CTI_CORR` = False (CTI correction applied?)
  - `IS_BAD` = False (frame deemed bad?)
  - `DRPVERSN` = 3.1 (DRP version)
  - `DRPCTIME` = timestamp (processing time)
  - `FWC_PP_E` = 90000.0 (pre-amp full well capacity)
  - `FWC_EM_E` = 100000.0 (EM full well capacity)
  - `SAT_DN` = 7241.38 (saturation level in DN)
  - `RECIPE` = Full JSON of processing recipe
  - `HISTORY` entries (6 total):
    1. "Frames cropped and bias subtracted"
    2. "Removed 0 frames as bad"
    3. "Cosmic ray mask created..."
    4. "...with hash ..."
    5. "Data corrected for non-linearity..."
    6. "Updated Data Level to L2a"

### HDU 2: ERR (Error Extension)
- **Data Shape**: (1, 1024, 1024)
- **Data Type**: float64 (>f8)
- **Purpose**: Error propagation through pipeline
- **Key Headers**:
  - `EXTNAME` = ERR
  - `TRK_ERRS` = False (tracking individual error terms?)
  - `LAYER_1` = combined_error (error layer description)
  - `HISTORY` = "Added error term: prescan_bias_sub"

### HDU 3: DQ (Data Quality Extension)
- **Data Shape**: (1024, 1024)
- **Data Type**: uint16
- **Purpose**: Data quality flags for each pixel
- **Key Headers**:
  - `EXTNAME` = DQ
  - `BSCALE` = 1
  - `BZERO` = 32768 (offset for unsigned integers)
- **Data Quality Flags** (bit meanings):
  - Bit 0: Bad pixel
  - Bit 1: Cosmic ray
  - Bit 2: Saturated
  - Bit 3: Non-linear
  - (etc. - full flag definitions in DQ extension)

### HDU 4: BIAS (Bias Extension)
- **Data Shape**: (1024,)
- **Data Type**: float32 (>f4)
- **Purpose**: Bias values used in prescan bias subtraction (1D array per column)
- **Key Headers**:
  - `EXTNAME` = BIAS

**Total L2a Headers**: ~237 lines in header file
- HDU 0: Primary (41 headers)
- HDU 1: Science (125 headers including new L2a headers)
- HDU 2: ERR (13 headers)
- HDU 3: DQ (10 headers)
- HDU 4: BIAS (7 headers)

---

## L2b (Calibrated) - Expected HDUs

*To be populated after L2a→L2b processing*

Expected HDUs:
- HDU 0: Primary
- HDU 1: Science Data (flat-fielded, dark-subtracted)
- HDU 2: ERR (propagated errors)
- HDU 3: DQ (updated quality flags)
- HDU 4: Additional calibration info

---

## L3 (Science-Ready) - Expected HDUs

*To be populated after L2b→L3 processing*

Expected HDUs:
- HDU 0: Primary
- HDU 1: Science Data (distortion-corrected, wavelength-calibrated)
- HDU 2: ERR (propagated errors)
- HDU 3: DQ (updated quality flags)
- HDU 4+: WCS, wavelength maps, etc.

---

## L4 (Analyzed) - Expected HDUs

*To be populated after L3→L4 processing*

Expected HDUs:
- HDU 0: Primary
- HDU 1: Processed science data (PSF-subtracted, extracted)
- HDU 2: ERR (propagated errors)
- HDU 3: DQ (updated quality flags)
- HDU 4+: Companion detection maps, spectral cubes, etc.

---

## Key Insights

### Error Propagation (ERR HDU)
The ERR extension tracks uncertainties through the pipeline:
- L2a: Adds prescan_bias_sub error term
- L2b: Will add flat field, dark subtraction errors
- L3: Will add distortion, wavelength calibration errors
- L4: Will add PSF subtraction, extraction errors

### Data Quality Tracking (DQ HDU)
The DQ extension is a bitmask tracking pixel-level issues:
- Bad pixels (detector defects)
- Cosmic ray hits
- Saturation
- Non-linearity corrections
- Each stage can add new flags

### Bias Tracking (BIAS HDU)
The BIAS extension preserves the bias correction:
- 1D array (one value per column)
- Allows verification of prescan bias subtraction
- Can be used for bias residual analysis

### Data Size Changes
- L1: (1200, 2200) full frame
- L2a: (1024, 1024) cropped to science region
- Prescan, overscan regions removed after bias subtraction

---

## Header Tracking Files

All header tracking files are in `pipeline_output/header_tracking/`:
- `L1/` - 2 HDUs per file
- `L2a/` - 5 HDUs per file
- *(Future: L2b/, L3/, L4/)*

Each HDU header includes:
- HDU name and number
- Data shape and type (if data present)
- All keyword-value-comment triplets
