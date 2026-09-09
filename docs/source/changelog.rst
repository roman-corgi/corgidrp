Change Log
==========

v5.1
----

New features:

* Chained recipes recorded in output FITS headers, with the full input file list written once and referenced by other products to reduce header bloat (@semaphoreP, @kjl0025)
* Cosmic ray median filter mode with prescan cosmic ray streak handling (@kjl0025)
* Coronagraph inner working angle excluded from cosmic ray flagging (@everlastingEric)
* ``inject_psf`` supports a relative scaling factor in addition to peak flux (@cychung0312)
* Wavelength zeropoint determination from stellar templates for broadband-only spectroscopy data (@neilzim)
* PRISM2 PSF templates and dispersion model for L2b to L3 spectroscopy processing (@neilzim)
* DRP updates for SPECROT images (@aneeshb97)
* ND filter calibration also returns the absolute flux calibration; walker distinguishes the two via bright/dim datasets (@everlastingEric)
* Boresight calibration frame splitting and combining by fixed frame count, RA/Dec dither, and user-specified header keywords (@everlastingEric)
* ``compute_flux_ratio_noise`` accepts a single Image in addition to a Dataset (@WhiteDogos)
* ``generate_mueller_matrix_cal`` looks in ``~/.corgidrp`` for a polarization reference file (@m-samland)
* Polarization standard star database expanded with additional Band 1 standards (@toshiyukimizuki)
* Polarization calibration inputs optionally filtered by dither position matching, with tolerancing in ``split_datasets`` (@maxwellmb)
* ``create_onsky_flatfield`` works with point-source star raster flats, including polarimetry mode (@juliamilton)
* Dataset constructor option to skip building combined arrays for frames with mismatched dimensions (@everlastingEric)
* HD37962 added as a faint CALSPEC standard (@JuergenSchreiber)
* Propagate ND and system Mueller matrix errors through ``subtract_stellar_polarization`` (@john-livingston)
* Filename suffixes added to the auto-generated data format documentation (@rt538-del)

Bug fixes:

* Negative photon-counted pixels masked with DQ 256 instead of crashing; ``EMGAIN_M`` cosmic ray crash fixed (@kjl0025)
* Astrometry source finding falls back to the guess location when the fit lands outside the image bounds (@manduhmia)
* Check for multiple targets before combining datasets in flux calibration (@JuergenSchreiber)
* Core throughput correctly identifies PUPIL images with ``FPAMNAME`` ``OPEN_34`` (@aneeshb97, @alexisyslau)
* ``Image.save`` casts only ndarray extension HDUs and handles ``SPEC_DQ`` (@JuergenSchreiber)
* Fix incorrect input filepath at the L2a level in walker.py (@adrienmaillard)
* Chained recipes removed for two non-calibration science vistypes (@adrienmaillard)
* Crop spec frames before wavelength zeropoint fitting in recipe templates (@JuergenSchreiber)
* ``determine_wave_zeropoint`` robust to filter-sweep sub-bands (@neilzim)
* Skip redundant ``subtract_no_offset_frames`` warning when already disabled (@SAY-5)
* Support for the new ``emccd_detect`` release (@kjl0025)
* L1 headers add ``RA_BORE``, ``DEC_BORE``, ``RA_APER``, and ``DEC_APER``; the ``SATSPOTS`` extension keyword is now boolean (@everlastingEric)
* Standardized ``unpack_geom()`` Returns docstring to type-first format (@rt538-del)
* Polarization setup observations with no dither processed to L3 are no longer identified as Mueller matrix calibraiton data (@shalverson)

Testing:

* New end-to-end tests: L1 to polcal, L1 to polarized flux calibration, L1 to NDFilterSweetSpot, L1 to astrometry, L1 to core throughput from simulated data, and L1 to core throughput map (@everlastingEric)
* Regular Mueller matrix calibration exercised in the L1 to polcal end-to-end test (@everlastingEric)
* Dither data added to mock polarization calibration data and tests (@everlastingEric)
* Data format end-to-end test updates (@everlastingEric)
* Fix ``flux_source`` calls in ``test_fluxcal.py`` (@semaphoreP)
* End-to-end test fixes for ND filter dithers and the pol flat ``NUM_FR`` keyword (@semaphoreP)

Other:

* README ported to ReadTheDocs with walker and recipe documentation updates (@semaphoreP)
* GitHub Actions workflows updated from v3 to v6 (@semaphoreP)


v5.0
----

New features:

* Use latest input UTC timestamp for calibration filenames
* Print recipes after finding them
* Add 2 recipes with polarization
* Spectral wave zeropoint now accounts for satspot background
* Center PSFs on the central pixel in the core throughput calibration cube
* CR frames retained through L2a processing and marked as IS_BAD
* User recipes can override default recipes
* Intermediate products in chained recipes now have updated filenames and data levels
* L3 to L4 step includes combine functionality
* Median full cosmic ray handling

Bug fixes:

* Updated walker.py due to satspot convention changes
* Enhanced cosmic ray handling in k gain/nonlin with corrected effective FWC
* Fix ND filter OD interpolation on cropped image
* Satspot processing corrections
* Polarization flatfield warning resolution
* NaN handling in astrometry centroiding; standardized PSF cut-out sizes
* Recipe generation consistency for analog satspots and science PC
* CFAMNAME 3F alignment with broadband name 3
* Fix northangle convention in l2b_to_l3
* CRPIX updates when shifting images
* NaN pixel fixes for boresight calibration
* Vignetting corrections

Testing:

* Analog HLC band1 end-to-end test
* L1 to spec dispersion testing with autodetection
* Cosmic ray parameter adjustments for test stability
* L1 to ND spec calibration using sweet spot dataset

v4.1
----

Functional improvements:

* L3 to L4 script expects a slit mask keyword in spectroscopy flux calibration data
* Legacy frame filtering improvements
* Propagate reference star polarization errors into Mueller matrix calibration
* Calibration database enhanced to handle missing files robustly
* Spectroscopic PSF subtraction routine optimization
* Dark subtraction now supports datasets with multiple detector settings
* ND transmission factor corrections
* CT calibration FPAMNAME corruption prevention when pupil image appears first
* Use SCTSRT for frame ordering and fix MJD in mocks
* Default calibrations added
* POL0 and POL45 flux calibrations both recorded in L4 pol headers
* Synthetic dark generation recipe added

Testing:

* FluxcalFactor implementation testing with input_dataset requirement
* L1 to linespread end-to-end test
* New L1 to L4 non-coron spec end-to-end test

v4.0
----

Functional improvements:

* Add nsigma scaling and small sample statistics correction to contrast curves
* ``find_star`` function now splits on VISITID by default
* Spectroscopic band 1 handling added
* Non-science frames filtering via ``discard_setup_frames`` step
* Polarization flatfield selection logic
* Noise column in ``add_shot_noise_to_err()`` now includes read noise
* Thermal pump updates
* K-gain nonlinearity RAM optimization
* Calibration database automatic product selection criteria refinement

Data format changes:

* Default image dtype changed to 32-bit
* NORTHANG replaces PA_APER; added to default L3 headers with WCS keyword validation
* Coron modes use VISTYPE
* Single frame L2a and L2b image keywords not invalidated by default
* ND filter spectroscopy calibration product
* Spec flux calibration data format standardized

Testing:

* L1 to Cal end-to-end test for FluxcalFactor
* L4 to TDA spec VAP test updates
* Pol L1 to L4 end-to-end test
* L1 to SpecFluxCal end-to-end testing
* Nonlinearity testing of default behavior

Other:

* Migrated setup.py to pyproject.toml

v3.2
----

New features:

* Automated data format array size checking
* Introduced SpecFilterOffset calibration file with default application
* Huge dataset accommodation in walker.py
* Spectral flux calibration implementation
* Combined frame header logic
* Convert slit transmission to a calibration product and add corresponding pipeline step
* IS_SYNTH header addition

Data format changes:

* Header keyword update to accommodate SSC 4.1.1 L1 keywords
* Updated documentation for -999 values where appropriate
* Comprehensive data product page updates

Bug fixes:

* Cropping bugfix
* Pandas 3.0 deprecation handling
* WFOV Band 4 support

v3.1
----

* Synchronized main and develop branches
* Spectroscopy: generate a wavelength dependent absolute flux calibration file
* Fixed EACQ_ROW assignment to use psfcenty
* WCS fixes implementation
* Data format and general documentation updates

v3.0-alpha
----------

* Alpha-version pre-release for new VAP testing x e2e testing setup
* E2E test in new VAP format for spectroscopy dispersion calibration
* New spectroscopy module with calibration functions
* Begin changing of filenames and headers to match new file standards
* End-to-end tests for some band 3 calibrations
* Suppression of some warnings

v2.2
----

* Specific print statements in unit tests for AAC testing
* E2E test for ND filter calibration
* Revert to using only 1 platescale, instead of separate x and y axis platescales

v2.1
----

* E2E test bug fixes:

  * Fix bug that traditional dark requires a clean output directory to pass
  * Fix bug where photon counting depends on a missing calibration file sometimes

* Added E2E tests for core throughput and core throughput map

v2.0
----

* Major release with L3, L4, and TDA processing
* Distortion, absolute flux, core throughput, and ND filter calibrations
* Photon counting L2 processing
* New file naming and header keywords
* Bug fixes

v1.1.2
------

* Flat field correction marks pixels divided by 0 as bad

v1.1.1
------

* Fix unit test that wasn't cleaning up environment properly

v1.1
----

* Bug fix so that ``corgidrp`` classes can be pickled
* New ``corgidrp.ops`` interface
* Improved agreement with II&T pipeline in updated e2e tests
* Ability to embed just the illuminated part of the detector back into a full engineering frame
* Updated DRP throughout to handle recently updated data header specification

v1.0
----

* First official pipeline release
* Step functions to produce the necessary calibration files for analog L1 to L2b implemented and tested
* Step function to produce boresight calibration implemented and tested
* Automated data processing handling for analog L1/L2 calibration files and for boresight calibration
* End-to-end testing demonstrating analog L1/L2 calibration files and boresight calibration file can be produced from realistic/real L1 data files

v0.2.1
------

* Update end-to-end tests to handle updated test data filenames

v0.2
----

* All L1 to L2b step functions implemented and tested
* Automated data processing for analog L1 to L2b
* End-to-end testing for analog L1 to L2b processing
* Minor bug fixes throughout

v0.1.2
------

* Added ability to change paths for analog L1 to L2a end-to-end test from command line
* Note: v0.1.1 was a partial release of this version and should not be used

v0.1
----

* First preliminary release of pipeline including step functions, calibration database, walker (pipeline automation framework)
* All L1 to L2a step functions implemented and tested
* Automated data processing for analog L1 to L2a
* End-to-end test demonstrating analog L1 to L2a processing
