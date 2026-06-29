# E2E Test Code for l1 to CoreThroughput Calibration

import argparse
import os, shutil
import glob
import pytest
import warnings
import numpy as np
import astropy.time as time
from astropy.io import fits

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.detector as detector
import corgidrp.corethroughput as corethroughput
from corgidrp import caldb
from corgidrp import check
from corgidrp.check import fix_hdrs_for_tvac, compare_to_mocks_hdrs

# this file's folder
thisfile_dir = os.path.dirname(__file__)

@pytest.mark.e2e
def test_expected_results_band1_nfov_e2e(e2edata_path, e2eoutput_path):
    """Test corethroughput calibration with mock data

    Args:
        e2edata_path (str): Path to the test data
        e2eoutput_path (str): Path to the output directory

    """
    
    # create output directory if none exists
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_corethrpughput_e2e")
    if os.path.exists(test_outputdir):
        shutil.rmtree(test_outputdir)
    os.makedirs(test_outputdir)
    
    # grab required sims and mock calibration data
    sci_datadir = os.path.join(e2edata_path, "corethroughput_sims_HLC_B1")
    l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")

    calibrations_dir = os.path.join(test_outputdir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)

    l2b_outputdir = os.path.join(test_outputdir, "l2b_results")
    if not os.path.exists(l2b_outputdir):
        os.mkdir(l2b_outputdir)
    # clean up by removing old files
    for file in os.listdir(l2b_outputdir):
        os.remove(os.path.join(l2b_outputdir, file))
    # assume all cals are in the same directory
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")
    mock_cal_filelist = [os.path.join(l1_datadir, os.listdir(l1_datadir)[i]) for i in range(5)]
    # Copy and fix mock cal headers 
    mock_cal_dir = os.path.join(test_outputdir, 'mock_cal_input')
    os.makedirs(mock_cal_dir, exist_ok=True)
    mock_cal_filelist = [
        shutil.copy2(f, os.path.join(mock_cal_dir, os.path.basename(f)))
        for f in mock_cal_filelist
    ]
    mock_cal_filelist = check.fix_hdrs_for_tvac(mock_cal_filelist, mock_cal_dir)

    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make calibration files using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    kgain_val = 8.7 # 8.7 is what is in the TVAC headers
    signal_array = np.linspace(0, 50)
    noise_array = np.sqrt(signal_array)
    ptc = np.column_stack([signal_array, noise_array])
    kgain = data.KGain(kgain_val, ptc=ptc, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    this_caldb.create_entry(kgain)

    # NoiseMap
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_path) as hdulist:
        dark_dat = hdulist[0].data
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    ext_hdr['B_O'] = 0.
    ext_hdr['B_O_ERR'] = 0.
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    mocks.rename_files_to_cgi_format(list_of_fits=[noise_map], output_dir=calibrations_dir, level_suffix="dnm_cal")
    this_caldb.create_entry(noise_map)

    ## Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    flat.ext_hdr['FPAMNAME'] = 'OPEN_12'
    flat.ext_hdr['CFAMNAME'] = '1F'
    flat.ext_hdr['DPAMNAME'] = 'IMAGING'
    mocks.rename_files_to_cgi_format(list_of_fits=[flat], output_dir=calibrations_dir, level_suffix="flt_cal")
    this_caldb.create_entry(flat)

    # bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    # Make sure BPM includes a dark(-like) frame
    bp_dark_pri, bp_dark_ext, _, _ = mocks.create_default_calibration_product_headers()
    bp_dark_ext['EXPTIME'] = float(mock_input_dataset.frames[0].ext_hdr.get('EXPTIME', 0.0))
    bp_dark_ext['EMGAIN_C'] = float(mock_input_dataset.frames[0].ext_hdr.get('EMGAIN_C', 1.0))
    bp_dark_ext['DRPNFILE'] = 1
    bp_dark = data.Dark(np.zeros_like(bp_dat, dtype=float), pri_hdr=bp_dark_pri, ext_hdr=bp_dark_ext,
                        input_dataset=mock_input_dataset,
                        err=np.zeros((1,) + bp_dat.shape, dtype=float),
                        dq=np.zeros(bp_dat.shape, dtype='uint16'),
                        err_hdr=fits.Header())
    bp_map_inputs = data.Dataset([bp_dark, flat])
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=bp_map_inputs)
    mocks.rename_files_to_cgi_format(list_of_fits=[bp_map], output_dir=calibrations_dir, level_suffix="bpm_cal")
    this_caldb.create_entry(bp_map)
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    # define the raw science data to process
    l1_data_filelist = [os.path.join(sci_datadir, os.listdir(sci_datadir)[i]) for i in range(len(os.listdir(sci_datadir))) if os.listdir(sci_datadir)[i].endswith("l1_.fits")] 

    ## delete some headers introduced in corgisim that aren't part of real data
    for filename in l1_data_filelist:
        with fits.open(filename) as hdulist:
            prihdr = hdulist[0].header
            if 'SATSPOTS' in prihdr:
                del prihdr['SATSPOTS']
            if 'COMMENT' in prihdr:
                del prihdr['COMMENT']
            hdulist.writeto(filename, overwrite=True)
            

    # run the walker
    with warnings.catch_warnings():
        # suppress warning about number of detectornoisemap frames
        warnings.simplefilter("ignore", category=UserWarning)
        walker.walk_corgidrp(l1_data_filelist, "", l2b_outputdir)

    # Load in the output data. It should be the latest ctp_cal file produced.
    corethroughput_drp_file = glob.glob(os.path.join(l2b_outputdir,
        '*ctp_cal.fits'))[0]
    ct_cal_drp = data.CoreThroughputCalibration(corethroughput_drp_file)

    # check headers
    check.compare_to_mocks_hdrs(corethroughput_drp_file, header_template=mocks.create_default_calibration_product_headers)
    assert ct_cal_drp.ext_hdr["DATATYPE"] == "CoreThroughputCalibration"
    assert ct_cal_drp.ext_hdr["DATALVL"] == "CAL"

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

    # Print success message
    print('e2e test for corethroughput calibration with mock data passed')
    

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    outputdir = thisfile_dir
    e2edata_path = '/home/eshen12345/dev/E2E_Test_Data'

    ap = argparse.ArgumentParser(description='run the l21 to CoreThroughput end-to-end test')
    ap.add_argument('-e2e', '--e2edata_dir', default=e2edata_path,
                    help='Path to CGI_TVAC_Data Folder [%(default)s]')
    ap.add_argument('-o', '--outputdir', default=outputdir,
                    help='directory to write results to [%(default)s]')
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_band1_nfov_e2e(args.e2edata_dir, args.outputdir)
