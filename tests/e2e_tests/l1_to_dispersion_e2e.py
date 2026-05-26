# Flux calibration E2E Test Code

import argparse
import os, shutil
import warnings
import glob
import pytest
import numpy as np
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.detector as detector
from corgidrp import caldb
from corgidrp import check
import astropy.time as time
import astropy.io.fits as fits
from corgidrp.darks import build_synthesized_dark


def setup_caldb(l1_datadir, processed_cal_path, calibrations_dir):
    """
    Make a temporary CalDB populated with detector calibrations and the
    default spectroscopy calibrations.

    Args:
        l1_datadir (str): Directory containing the L1 input files. 
        processed_cal_path (str):  Directory containing flat detector calibration files.  
            The following filenames are expected:
                nonlin_table.txt    — nonlinearity correction table (CSV)
                dark_current.fits   — per-pixel dark current image
                flat.fits           — flat field image
                fpn.fits            — fixed-pattern noise image
                cic.fits            — clock-induced charge image
                bad_pix.fits        — bad pixel map image
        calibrations_dir (str): Output directory where the built calibration FITS files 
            are saved.

    Returns:
        this_caldb (corgidrp.caldb.CalDB): Populated temporary calibration database 
            containing:
                NonLinearityCalibration, KGain, DetectorNoiseMaps,
                SynthesizedDark (analog mode only), FlatField, BadPixelMap,
                DispersionModel, SpecFilterOffset  (from corgidrp.default_cal_dir)
        is_pc_data (bool): True if the L1 data are photon-counted (ISPC=1).
            If True, a PC dark is built later from L2a frames rather than here.
    """
    # Use a temporary CSV to avoid issues with real CalDB
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_linespread_e2e_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()

    # Get default spectroscopy calibrations 
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    print(f"Loaded default calibrations from {corgidrp.default_cal_dir}")

    # Paths to detector calibration files
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table.txt")
    dark_path   = os.path.join(processed_cal_path, "dark_current.fits")
    flat_path   = os.path.join(processed_cal_path, "flat.fits")
    fpn_path    = os.path.join(processed_cal_path, "fpn.fits")
    cic_path    = os.path.join(processed_cal_path, "cic.fits")
    bp_path     = os.path.join(processed_cal_path, "bad_pix.fits")

    # Build a mock input_dataset from a couple of L1 files so that
    # calibration products can record them
    all_l1 = sorted(f for f in os.listdir(l1_datadir)
                    if f.endswith('l1.fits') or f.endswith('l1_.fits'))
    mock_cal_files = [os.path.join(l1_datadir, f) for f in all_l1[-2:]]
    mock_cal_dir   = os.path.join(os.path.dirname(calibrations_dir), 'mock_cal_input')
    os.makedirs(mock_cal_dir, exist_ok=True)
    mock_cal_files = [
        shutil.copy2(f, os.path.join(mock_cal_dir, os.path.basename(f)))
        for f in mock_cal_files
    ]
    mock_input_dataset = data.Dataset(mock_cal_files)

    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr['DRPCTIME'] = time.Time.now().isot
    ext_hdr['DRPVERSN'] = corgidrp.__version__

    # Determine whether the dataset contains any photon-counted frames
    sample_hdr = fits.getheader(mock_cal_files[0], ext=1)
    is_pc_data = any(
        bool(int(fits.getheader(os.path.join(l1_datadir, f), ext=1).get('ISPC', 0)))
        for f in all_l1
    )

    # Nonlinearity
    nonlin_dat  = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(
        nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format([nonlinear_cal], calibrations_dir, "nln_cal")
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    signal_array = np.linspace(0, 50)
    noise_array  = np.sqrt(signal_array)
    ext_hdr['RN']    = 100.0
    ext_hdr['RN_ERR'] = 0.0
    kgain = data.KGain(
        8.7, ptc=np.column_stack([signal_array, noise_array]),
        pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format([kgain], calibrations_dir, "krn_cal")
    this_caldb.create_entry(kgain)

    # Noise map
    fpn_dat  = fits.getdata(fpn_path)
    cic_dat  = fits.getdata(cic_path)
    dark_dat = fits.getdata(dark_path)
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    nm_dat = np.zeros((3,
                       detector.detector_areas['SCI']['frame_rows'],
                       detector.detector_areas['SCI']['frame_cols']))
    nm_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = np.array([fpn_dat, cic_dat, dark_dat])
    err_hdr_nm = fits.Header()
    err_hdr_nm['BUNIT'] = 'detected electron'
    ext_hdr['B_O']     = 0.
    ext_hdr['B_O_ERR'] = 0.
    noise_map = data.DetectorNoiseMaps(
        nm_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=mock_input_dataset,
        err=np.zeros([1] + list(nm_dat.shape)),
        dq=np.zeros(nm_dat.shape, dtype=int),
        err_hdr=err_hdr_nm)
    mocks.rename_files_to_cgi_format([noise_map], calibrations_dir, "dnm_cal")
    this_caldb.create_entry(noise_map)

    # Synthesized dark: caldb drk entry only for analog. PC pipeline dark
    # comes from L2a, but we still need a drk FITS for BadPixelMap input_dataset.
    fname0 = all_l1[0]
    h0 = fits.getheader(os.path.join(l1_datadir, fname0), ext=1)
    exptime = float(h0['EXPTIME'])
    emgain_c = float(h0['EMGAIN_C'])
    src = os.path.join(l1_datadir, fname0)
    tmp_f = shutil.copy2(
        src, os.path.join(mock_cal_dir, os.path.basename(src)))
    tmp_ds = data.Dataset([tmp_f])
    tmp_ds.frames[0].ext_hdr['EXPTIME'] = exptime
    tmp_ds.frames[0].ext_hdr['EMGAIN_C'] = emgain_c
    dark_cal = build_synthesized_dark(tmp_ds, noise_map)
    mocks.rename_files_to_cgi_format([dark_cal], calibrations_dir, "drk_cal")
    if not is_pc_data:
        this_caldb.create_entry(dark_cal)
    bpm_drk_input_path = dark_cal.filepath

    # Flat field
    flat = data.FlatField(
        fits.getdata(flat_path),
        pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format([flat], calibrations_dir, "flt_cal")
    this_caldb.create_entry(flat)

    # Bad pixel map
    bp_map = data.BadPixelMap(
        fits.getdata(bp_path),
        pri_hdr=pri_hdr, ext_hdr=ext_hdr,
        input_dataset=data.Dataset([bpm_drk_input_path]))
    mocks.rename_files_to_cgi_format([bp_map], calibrations_dir, "bpm_cal")
    this_caldb.create_entry(bp_map)

    print("Calibration database populated.")
    return this_caldb, is_pc_data



@pytest.mark.e2e
def test_l1_to_dispersion(e2edata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(e2edata_path, "spec_dispersion_sims")
    processed_cal_path = os.path.join(e2edata_path, "ND_SPEC", "Cals")

    # make output directory if needed
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_dispersion_e2e")
    if os.path.exists(test_outputdir):
        shutil.rmtree(test_outputdir)
    os.makedirs(test_outputdir)

    calibrations_dir = os.path.join(test_outputdir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)

    l2b_outputdir = os.path.join(test_outputdir, "l2b_results")
    if not os.path.exists(l2b_outputdir):
        os.mkdir(l2b_outputdir)

    # clean up by removing old files
    for file in os.listdir(l2b_outputdir):
        os.remove(os.path.join(l2b_outputdir, file))
    
    this_caldb, _ = setup_caldb(
        l1_datadir, processed_cal_path, calibrations_dir)    
    
    l1_data_filelist=[os.path.join(l1_datadir, os.listdir(l1_datadir)[i]) for i in range(len(os.listdir(l1_datadir))) if os.listdir(l1_datadir)[i].endswith("l1_.fits")]
    
    ####### Run the walker on some test_data
    # ------------------------------------------------------------------ 
    # L1 -> L2a                                                        
    # ------------------------------------------------------------------ 
    print("Running L1 -> L2a …")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l1_data_filelist, "", l2b_outputdir,
                             template="l1_to_l2a_basic.json")

    l2a_filelist = sorted(
        os.path.join(l2b_outputdir, f)
        for f in os.listdir(l2b_outputdir) if f.endswith('_l2a.fits')
    )
    print(f"L1 -> L2a complete: {len(l2a_filelist)} L2a files produced.")
    
    # ------------------------------------------------------------------ 
    # L2a -> L2b                                                        
    # ------------------------------------------------------------------ 
    print("Running L2a -> L2b …")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l2a_filelist, "", l2b_outputdir,
                             template="l2a_to_l2b_spec.json")

    l2b_filelist = sorted(
        os.path.join(l2b_outputdir, f)
        for f in os.listdir(l2b_outputdir) if f.endswith('_l2b.fits')
    )
    print(f"L2a -> L2b complete: {len(l2b_filelist)} L2b files produced.")
    
        # ------------------------------------------------------------------ 
    # L2b -> spec dispersion                                                        
    # ------------------------------------------------------------------ 
    print("Running L2b -> DispersionModel …")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l2b_filelist, "", l2b_outputdir,
                             template="l2b_to_spec_prism_disp.json")

    print(f"L2b -> DispersionModel complete.")
    
    ####### Load in the output data. It should be the latest line spread calibration file produced.
    dispersion_cal_file = glob.glob(os.path.join(l2b_outputdir, '*dpm_cal*.fits'))[0]
    dispersion = data.DispersionModel(dispersion_cal_file)
    
    ### validate LineSpread product 
    check.compare_to_mocks_hdrs(dispersion_cal_file)

    assert dispersion.ext_hdr["DATATYPE"] == "DispersionModel"
    assert dispersion.ext_hdr["DATALVL"] == "CAL"
    assert dispersion.ext_hdr['BAND'] == '3'
    assert dispersion.ext_hdr['REFWAVE'] == 730
    
    assert np.abs(dispersion.clocking_angle) == pytest.approx(90, abs = dispersion.clocking_angle_uncertainty) 
    assert len(dispersion.pos_vs_wavlen_polycoeff) == len(dispersion.wavlen_vs_pos_polycoeff) == 4
    print("clocking angle: ", dispersion.clocking_angle)
    print("clocking angle uncertainty: ", dispersion.clocking_angle_uncertainty)
    print("pos_vs_wavlen_polycoeff: ", dispersion.pos_vs_wavlen_polycoeff)
    print("pos_vs_wavlen_cov: ", dispersion.pos_vs_wavlen_cov)
    print("wavlen_vs_pos_polycoeff: ", dispersion.wavlen_vs_pos_polycoeff)
    print("wavlen_vs_pos_cov: ", dispersion.wavlen_vs_pos_cov)
    
    # Remove temporary CalDB
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_dispersion_e2e_caldb.csv')
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    # Print success message
    print('e2e test for dispersion calibration passed')
    
if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    e2edata_dir = '/home/schreiber/DataCopy/E2E_Test_Data'
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1-> Dispersion end-to-end test")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    test_l1_to_dispersion(e2edata_dir, outputdir)
