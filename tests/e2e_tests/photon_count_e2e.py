# Photon Counting E2E Test Code

import argparse
import os
import pytest
import numpy as np
import astropy.time as time
from astropy.io import fits

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector

@pytest.mark.e2e
def test_expected_results_e2e(e2edata_path, e2eoutput_path):
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")

    np.random.seed(1234)
    ill_dataset, dark_dataset, ill_mean, dark_mean = mocks.create_photon_countable_frames(Nbrights=160, Ndarks=161, cosmic_rate=1, flux=0.5)
    output_dir = os.path.join(e2eoutput_path, 'pc_sim_test_data')
    output_ill_dir = os.path.join(output_dir, 'ill_frames')
    output_dark_dir = os.path.join(output_dir, 'dark_frames')
    output_l2a_dir = os.path.join(output_dir, 'l2a')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # empty out directory of any previous files
    for f in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir,f)):
            continue
        os.remove(os.path.join(output_dir,f))
    if not os.path.exists(output_ill_dir):
        os.mkdir(output_ill_dir)
    if not os.path.exists(output_dark_dir):
        os.mkdir(output_dark_dir)
    if not os.path.exists(output_l2a_dir):
        os.mkdir(output_l2a_dir)
    # empty out directory of any previous files
    for f in os.listdir(output_ill_dir):
        os.remove(os.path.join(output_ill_dir,f))
    for f in os.listdir(output_dark_dir):
        os.remove(os.path.join(output_dark_dir,f))
    ill_dataset.save(output_ill_dir, ['pc_frame_ill_{0}.fits'.format(i) for i in range(len(ill_dataset))])
    dark_dataset.save(output_dark_dir, ['pc_frame_dark_{0}.fits'.format(i) for i in range(len(dark_dataset))])
    l1_data_ill_filelist = []
    l1_data_dark_filelist = []
    for f in os.listdir(output_ill_dir):
        l1_data_ill_filelist.append(os.path.join(output_ill_dir, f))
    for f in os.listdir(output_dark_dir):
        l1_data_dark_filelist.append(os.path.join(output_dark_dir, f))

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB

    # KGain
    kgain_val = 7. # default value used in mocks.create_photon_countable_frames()
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(l1_data_ill_filelist)
    kgain = data.KGain(kgain_val, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    # add in keywords that didn't make it into mock_kgain.fits, using values used in mocks.create_photon_countable_frames()
    kgain.ext_hdr['RN'] = 100
    kgain.ext_hdr['RN_ERR'] = 0
    kgain.save(filedir=output_dir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)

    # NoiseMap
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=output_dir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    here = os.path.abspath(os.path.dirname(__file__))
    tests_dir = os.path.join(here, os.pardir)
    # Nonlinearity calibration
    ### Create a dummy non-linearity file ####
    #Create a mock dataset because it is a required input when creating a NonLinearityCalibration
    dummy_dataset = mocks.create_prescan_files()
    # Make a non-linearity correction calibration file
    input_non_linearity_filename = "nonlin_table_TVAC.txt"
    input_non_linearity_path = os.path.join(tests_dir, "test_data", input_non_linearity_filename)
    test_non_linearity_filename = input_non_linearity_filename.split(".")[0] + ".fits"
    nonlin_fits_filepath = os.path.join(tests_dir, "test_data", test_non_linearity_filename)
    tvac_nonlin_data = np.genfromtxt(input_non_linearity_path, delimiter=",")

    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    new_nonlinearity = data.NonLinearityCalibration(tvac_nonlin_data,pri_hdr=pri_hdr,ext_hdr=ext_hdr,input_dataset = dummy_dataset)
    new_nonlinearity.filename = nonlin_fits_filepath
    new_nonlinearity.pri_hdr = pri_hdr
    new_nonlinearity.ext_hdr = ext_hdr
    this_caldb.create_entry(new_nonlinearity)

    ## Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    flat.save(filedir=output_dir, filename="mock_flat.fits")
    this_caldb.create_entry(flat)

    # bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    bp_map.save(filedir=output_dir, filename="mock_bpmap.fits")
    this_caldb.create_entry(bp_map)

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    
    # make PC dark
    # below I leave out the template specification to check that the walker recipe guesser works as expected
    walker.walk_corgidrp(l1_data_dark_filelist, '', output_dir)#, template="l1_to_l2b_pc_dark.json")
    # calDB was just updated with the PC Dark that was created with the walker above
    master_dark_filename_list = []
    master_dark_filepath_list = []
    for f in os.listdir(output_dir):
        if not f.endswith('.fits'):
            continue
        if f.endswith('_drk_cal.fits'):
            master_dark_filename_list.append(f)
            master_dark_filepath_list.append(os.path.join(output_dir, f))
    

    # make PC illuminated, subtracting the PC dark
    # below I leave out the template specification to check that the walker recipe guesser works as expected
    # L1 to L2a
    walker.walk_corgidrp(l1_data_ill_filelist, '', output_l2a_dir)#, template="l1_to_l2b_pc.json")
    
    # grab L2a files to go to L2b
    l2a_files = []
    for filepath in l1_data_ill_filelist:
        # emulate naming change behaviors
        new_filename = filepath.split(os.path.sep)[-1].replace("_l1_", "_l2a") 
        # loook in new dir
        new_filepath = os.path.join(output_l2a_dir, new_filename)
        l2a_files.append(new_filepath)
    walker.walk_corgidrp(l2a_files, '', output_dir)

    # get photon-counted frame
    master_ill_filename_list = []
    master_ill_filepath_list = []
    # helper function that doesn't rely on naming scheme
    def get_last_modified_file(directory):
        files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        if not files:
            return None
        last_modified_file = max(files, key=os.path.getmtime)
        return last_modified_file
    pc_processed_filepath = get_last_modified_file(output_dir)
    pc_processed_filename = os.path.split(pc_processed_filepath)[-1]
    for f in os.listdir(output_dir):
        if f == pc_processed_filename: 
            master_ill_filename_list.append(f)
            master_ill_filepath_list.append(os.path.join(output_dir, f))
    for i in range(len(master_ill_filepath_list)):
        pc_frame = fits.getdata(master_ill_filepath_list[i])
        pc_frame_err = fits.getdata(master_ill_filepath_list[i], 'ERR')
        pc_dark_frame = fits.getdata(master_dark_filepath_list[i])
        pc_dark_frame_err = fits.getdata(master_dark_filepath_list[i], 'ERR')

        # more frames gets a better agreement; agreement to 1% for ~160 darks and illuminated
        assert np.isclose(np.nanmean(pc_frame), ill_mean - dark_mean, rtol=0.02) 
        assert np.isclose(np.nanmean(pc_dark_frame), dark_mean, rtol=0.01) 
        assert pc_frame_err.min() >= 0
        assert pc_dark_frame_err.min() >= 0

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)



if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir
    e2edata_dir =  '/Users/kevinludwick/Documents/ssc_tvac_test/E2E_test_data2/'#'/home/jwang/Desktop/CGI_TVAC_Data/'

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    e2edata_dir = args.e2edata_dir
    test_expected_results_e2e(e2edata_dir, outputdir)
