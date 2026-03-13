# Flux calibration E2E Test Code

import argparse
import os, shutil
import glob
import pytest
import numpy as np
from datetime import datetime
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.detector as detector
import corgidrp.fluxcal as fluxcal
from corgidrp import caldb
from corgidrp import check
import astropy.time as time
import astropy.io.fits as fits

@pytest.mark.e2e
def test_l1_to_fluxcal(e2edata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    sci_datadir = os.path.join(e2edata_path, "Fluxcal_sims")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")

    # make output directory if needed
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_fluxcal_e2e")
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
    # assume all cals are in the same directory
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")
    mock_cal_filelist = [os.path.join(l1_datadir, os.listdir(l1_datadir)[i]) for i in [-2,-1]]
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
    mocks.rename_files_to_cgi_format(list_of_fits=[flat], output_dir=calibrations_dir, level_suffix="flt_cal")
    this_caldb.create_entry(flat)

    # bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[bp_map], output_dir=calibrations_dir, level_suffix="bpm_cal")
    this_caldb.create_entry(bp_map)
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    # define the raw science data to process
    l1_data_filelist = [os.path.join(sci_datadir, os.listdir(sci_datadir)[i]) for i in range(len(os.listdir(sci_datadir))) if os.listdir(sci_datadir)[i].endswith("l1_.fits")] 

    ####### Run the walker on some test_data
    walker.walk_corgidrp(l1_data_filelist, "", l2b_outputdir)

    ####### Load in the output data. It should be the latest absflux file produced.
    fluxcal_file = glob.glob(os.path.join(l2b_outputdir, '*abf_cal*.fits'))[0]
    flux_fac = data.FluxcalFactor(fluxcal_file)
    print("used color filter", flux_fac.filter)
    print("used ND filter", flux_fac.nd_filter)
    print("fluxcal factor", flux_fac.fluxcal_fac)
    print("fluxcal factor error", flux_fac.fluxcal_err)
    
    #sanity check using the real flux of the target calstar.
    new_l2a_filenames = [os.path.join(l2b_outputdir, f) for f in os.listdir(l2b_outputdir) if f.endswith('l2b.fits')] 
    dataset=data.Dataset(new_l2a_filenames)
    image = dataset[0]
    image_data = np.nanmedian(dataset.all_data, 0)
    #estimate expected flux of calspec standard
    filter_file = fluxcal.get_filter_name(image)
    wave, filter_trans = fluxcal.read_filter_curve(filter_file)
    
    star_name = image.pri_hdr["TARGET"]
    calspec_filepath, calspec_filename = fluxcal.get_calspec_file(star_name)
    flux_ref = fluxcal.read_cal_spec(calspec_filepath, wave)
    flux = fluxcal.calculate_band_flux(filter_trans, flux_ref, wave)
    
    #calculate the photoelectron counts around the center image and divide by exposure time
    center = 512
    counts = np.nansum(image_data[center-5:center+5, center-5:center+5])/image.ext_hdr["EXPTIME"]
    flux_count = flux_fac.fluxcal_fac * counts
    assert flux == pytest.approx(flux_count, abs = 1.5 * flux_fac.fluxcal_err * counts)
    
    check.compare_to_mocks_hdrs(fluxcal_file)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

   # Print success message
    print('e2e test for flux calibration factor passed')
    
if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    e2edata_dir = '/home/schreiber/DataCopy/E2E_Test_Data'
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1-> FluxcalFactor end-to-end test")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    test_l1_to_fluxcal(e2edata_dir, outputdir)
