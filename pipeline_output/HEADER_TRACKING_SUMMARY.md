# Header Tracking Summary

This document tracks header information as data progresses through the pipeline stages.

## Pipeline Overview

The CORGIDRP pipeline processes data through these levels:
- **L1** (Raw Data) → **L2a** (Basic Corrections) → **L2b** (Calibrated) → **L3** (Science-Ready) → **L4** (Analyzed)

---

## L1 (Raw Data) Headers

**Total Headers:** 154 (41 primary + 113 image)

### Primary HDU (41 keys)

**Observation Metadata:**
- `VISITID`, `VISTYPE`, `OBSNAME`, `TARGET`
- `RA`, `DEC`, `EQUINOX` (celestial coordinates)
- `ROLL`, `PITCH`, `YAW` (spacecraft attitude)

**System Information:**
- `TELESCOP` = ROMAN, `INSTRUME` = CGI, `DETECTOR` = EXCAM
- `CDMSVERS`, `FSWDVERS` (software versions)

**Program Details:**
- `PROGNUM`, `EXECNUM`, `CAMPAIGN`, `SEGMENT`, `OBSNUM`, `VISNUM`

**Operational Settings:**
- `OPGAIN`, `PHTCNT`, `FRAMET`, `SATSPOTS`, `PSFREF`

### Image HDU (113 keys)

**Array Configuration:**
- `NAXIS1`, `NAXIS2` (2200 x 1200 for SCI array)
- `ARRTYPE` = SCI, `BUNIT` = DN

**Detector Settings:**
- `HVCBIAS`, `OPMODE`, `EXPTIME`
- `EMGAIN_C` (commanded), `EMGAIN_A` (actual)
- `KGAINPAR`

**Timing:**
- `SCTSRT`, `SCTEND`, `DATETIME`, `FTIMEUTC`

**Mechanisms:**
All mask positions (SPAM, FPAM, LSAM, FSAM, CFAM, DPAM)

**Deformable Mirror:**
Zernike coefficients Z2-Z14

**Fast Steering Mirror:**
`FSMLOOP`, `FSMX`, `FSMY`, `FSMSG1-3`

**Data Level:**
- `DATALVL` = L1

---

## L2a (Basic Corrections) Headers

**Total HDUs:** 5 (Primary + SCI + ERR + DQ + BIAS)
**Total Headers:** ~196 across all HDUs

### HDU Structure:

**HDU 0: Primary Header** (inherited from L1)
- Observation metadata unchanged
- `ORIGIN` changed from SSC to DRP

**HDU 1: Science Data (Extension 1)**
- Data shape: (1024, 1024) - cropped from (1200, 2200)
- Contains corrected science data + new L2a headers

**HDU 2: ERR (Error Extension)** - NEW
- Data shape: (1, 1024, 1024)
- Data type: float64
- Tracks error propagation through pipeline
- Headers:
  - `EXTNAME` = ERR
  - `TRK_ERRS` = False
  - `LAYER_1` = combined_error
  - `HISTORY` = "Added error term: prescan_bias_sub"

**HDU 3: DQ (Data Quality Extension)** - NEW
- Data shape: (1024, 1024)
- Data type: uint16 (bitmask)
- Flags bad pixels, cosmic rays, saturation, etc.
- Headers:
  - `EXTNAME` = DQ
  - `BSCALE` = 1
  - `BZERO` = 32768

**HDU 4: BIAS (Bias Extension)** - NEW
- Data shape: (1024,) - 1D array
- Data type: float32
- Stores bias values used in prescan subtraction
- Headers:
  - `EXTNAME` = BIAS

### New Headers Added in HDU 1 (Science Data):

**Data Level:**
- `DATALVL` = L2a (updated from L1)

**Processing Flags:**
- `DESMEAR` = False (Was desmear applied?)
- `CTI_CORR` = False (Was CTI correction applied?)
- `IS_BAD` = False (Was this frame deemed bad?)

**DRP Information:**
- `DRPVERSN` = 3.1 (DRP version)
- `DRPCTIME` = 2026-01-21T11:27:41.380 (processing time)

**Detector Saturation Parameters:**
- `FWC_PP_E` = 90000.0 (Pre-amp full well capacity in e-)
- `FWC_EM_E` = 100000.0 (EM full well capacity in e-)
- `SAT_DN` = 7241.38 (Saturation level in DN)

**Recipe Information:**
- `RECIPE` = Full JSON describing the processing recipe:
  - Recipe name: "l1_to_l2a_basic"
  - Input files list
  - Output directory
  - All processing steps with calibration files used:
    - DetectorNoiseMaps (mock_detnoisemaps.fits)
    - DetectorParams (DetectorParams_2023-11-01T00.00.00.000.fits)
    - NonLinearityCalibration (nonlin_table_TVAC.fits)
  - Processing parameters

**Processing History (6 HISTORY entries):**
1. "Frames cropped and bias subtracted"
2. "Removed 0 frames as bad"
3. "Cosmic ray mask created. Used detector parameters from DetectorParams_2023-11-01T00.00.00.000.fits"
4. "with hash 8102519725012926413"
5. "Data corrected for non-linearity with nonlin_table_TVAC.fits"
6. "Updated Data Level to L2a"

### L1→L2a Processing Summary:

The L1→L2a stage performs basic detector corrections using the `l1_to_l2a_basic` recipe:

1. **Prescan Bias Subtraction** (`prescan_biassub`)
   - Removes detector bias using prescan regions
   - Crops frame to science region
   - Uses DetectorNoiseMaps calibration

2. **Cosmic Ray Detection** (`detect_cosmic_rays`)
   - Identifies cosmic ray hits
   - Creates cosmic ray mask
   - Uses DetectorParams calibration
   - Optional: KGain calibration (not used in this run)

3. **Nonlinearity Correction** (`correct_nonlinearity`)
   - Corrects for detector response nonlinearity
   - Uses NonLinearityCalibration table (TVAC-derived)

4. **Update to L2a** (`update_to_l2a`)
   - Updates DATALVL header to L2a
   - Adds processing metadata

5. **Save** (`save`)
   - Writes L2a FITS files to disk

**Calibrations Used:**
- `nonlin_table_TVAC.fits` - TVAC nonlinearity correction table
- `mock_detnoisemaps.fits` - Detector noise maps (FPN, CIC, dark current)
- `DetectorParams_2023-11-01T00.00.00.000.fits` - Detector parameters (from default calibs)

---

## L2b (Calibrated) Headers
*To be populated after L2a→L2b processing*

Expected additions:
- Dark subtraction details
- Flat field correction
- Bad pixel interpolation flags
- CTI/desmear corrections

---

## L3 (Science-Ready) Headers
*To be populated after L2b→L3 processing*

Expected additions:
- Distortion correction
- Wavelength calibration (spectroscopy)
- Astrometric solution
- WCS information

---

## L4 (Analyzed) Headers
*To be populated after L3→L4 processing*

Expected additions:
- PSF subtraction method
- Reference star information
- KLIP parameters
- Spectral extraction details
- Companion detection results

---

## Header Tracking Files

All header files are saved in `pipeline_output/header_tracking/`:
- `L1/` - L1 raw data headers (2 HDUs per file)
- `L2a/` - L2a processed headers (5 HDUs per file)
- *(Future: L2b/, L3/, L4/, etc.)*

Each FITS file gets a corresponding `*_headers.txt` file with:
- Total HDU count
- Complete header dumps from **ALL HDUs**
- Data shape and type for each HDU
- All keys, values, and comments
- Clear separation between HDUs

### Example Header File Format:

```
================================================================================
HEADER INFORMATION FOR: mock_l2aframe_000.fits
Total HDUs: 5
================================================================================

PRIMARY HEADER (HDU 0)
--------------------------------------------------------------------------------
[All primary header keywords...]

================================================================================

EXTENSION 1 HEADER (HDU 1)
--------------------------------------------------------------------------------
Data shape: (1024, 1024)
Data type: float32
--------------------------------------------------------------------------------
[All science data keywords...]

================================================================================

ERR HEADER (HDU 2)
--------------------------------------------------------------------------------
Data shape: (1, 1024, 1024)
Data type: >f8
--------------------------------------------------------------------------------
[All error extension keywords...]

[...and so on for all HDUs]
```

### Additional Documentation:

See `HDU_STRUCTURE_SUMMARY.md` for detailed information about:
- Complete HDU structure for each pipeline stage
- Data shapes, types, and purposes
- Error propagation tracking
- Data quality flag meanings
- Processing history
