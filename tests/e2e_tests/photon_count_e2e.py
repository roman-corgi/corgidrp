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
from corgidrp.darks import build_trad_dark
import shutil
import glob

@pytest.mark.e2e
def test_expected_results_e2e(e2edata_path, e2eoutput_path):
    #Checks that a photon-counted master dark works fine in the pipeline, for both cases of master dark (PC master dark or synthesized master dark)
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")
    # for noisemaps later in the script
    dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")

    np.random.seed(1234)
    # using CIC and dark current average values which come from the corresponding values from cic_path and dark_path above; FPN mean is already 0 in fpn_path and simulated set below
    ill_dataset, dark_dataset, ill_mean, dark_mean = mocks.create_photon_countable_frames(Nbrights=2, Ndarks=3, cosmic_rate=1, flux=0.5, cic=0.0035075, dark_current=0.00086158)
    output_dir = os.path.join(e2eoutput_path, 'photon_count_e2e')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Create input_data subfolder
    input_data_dir = os.path.join(output_dir, 'input_l1')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    # Create calibrations subfolder
    calibrations_dir = os.path.join(output_dir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)

    # Create l2a_to_l2b_output subfolder
    l2a_to_l2b_output_dir = os.path.join(output_dir, 'l2a_to_l2b')
    if not os.path.exists(l2a_to_l2b_output_dir):
        os.makedirs(l2a_to_l2b_output_dir)

    output_ill_dir = os.path.join(input_data_dir, 'ill_l1_frames')
    output_dark_dir = os.path.join(input_data_dir, 'dark_l1_frames')
    output_l2a_dir = os.path.join(output_dir, 'l1_to_l2a')
    if not os.path.exists(output_ill_dir):
        os.makedirs(output_ill_dir)
    if not os.path.exists(output_dark_dir):
        os.makedirs(output_dark_dir)
    if not os.path.exists(output_l2a_dir):
        os.makedirs(output_l2a_dir)

    ill_dataset.save(output_ill_dir)
    dark_dataset.save(output_dark_dir)
    del ill_dataset
    del dark_dataset
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
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    this_caldb.create_entry(kgain)

    # NoiseMap (meaningless data; won't be used in dark subtraction for this first test which instead uses PC master dark)
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
    mocks.rename_files_to_cgi_format(list_of_fits=[noise_map], output_dir=calibrations_dir, level_suffix="dnm_cal")
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
    mocks.rename_files_to_cgi_format(list_of_fits=[flat], output_dir=calibrations_dir, level_suffix="flt_cal")
    this_caldb.create_entry(flat)

    # bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[bp_map], output_dir=calibrations_dir, level_suffix="bpm_cal")
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
    walker.walk_corgidrp(l1_data_ill_filelist, '', output_l2a_dir)

    # grab L2a files to go to L2b
    l2a_files = []
    for filepath in l1_data_ill_filelist:
        # emulate naming change behaviors
        new_filename = filepath.split(os.path.sep)[-1].replace("_l1_", "_l2a")
        # loook in new dir
        new_filepath = os.path.join(output_l2a_dir, new_filename)
        l2a_files.append(new_filepath)

    recipe = walker.autogen_recipe(l2a_files, l2a_to_l2b_output_dir)
    ### Modify they keywords of some of the steps
    for step in recipe[0]['steps']:
        if step['name'] == "dark_subtraction":
            step['calibs']['Dark'] = master_dark_filepath_list[0] # to find PC dark
    walker.run_recipe(recipe[0], save_recipe_file=True)
    recipe[1]['inputs'] = glob.glob(os.path.join(recipe[0]['outputdir'], '*_l2a.fits'))
    walker.run_recipe(recipe[1], save_recipe_file=True)
    # files are overwritten with same filenames, so glob up the same filepaths
    recipe[2]['inputs'] = glob.glob(os.path.join(recipe[1]['outputdir'], '*_l2a.fits'))
    walker.run_recipe(recipe[2], save_recipe_file=True)
    
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
    pc_processed_filepath = get_last_modified_file(l2a_to_l2b_output_dir)
    pc_processed_filename = os.path.split(pc_processed_filepath)[-1]
    for f in os.listdir(l2a_to_l2b_output_dir):
        if f == pc_processed_filename:
            master_ill_filename_list.append(f)
            master_ill_filepath_list.append(os.path.join(l2a_to_l2b_output_dir, f))
    for i in range(len(master_ill_filepath_list)):
        pc_frame = fits.getdata(master_ill_filepath_list[i])
        pc_frame_err = fits.getdata(master_ill_filepath_list[i], 'ERR')
        pc_dark_frame = fits.getdata(master_dark_filepath_list[i])
        pc_dark_frame_err = fits.getdata(master_dark_filepath_list[i], 'ERR')

        # more frames gets a better agreement; agreement to 2% for ~160 darks and illuminated
        assert np.isclose(np.nanmean(pc_frame), ill_mean - dark_mean, rtol=0.02)
        assert np.isclose(np.nanmean(pc_dark_frame), dark_mean, rtol=0.01)
        assert pc_frame_err.min() >= 0
        assert pc_dark_frame_err.min() >= 0

    # remove PC master dark
    for f in os.listdir(output_dir):
        if f.endswith('_drk_cal.fits'):
            os.remove(os.path.join(output_dir, f))

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

    print('First part of e2e test for photon counting with PC master dark passed')

    #__________________________________________________
    # now test that the pipeline works fine if we start with a synthesized master dark instead
    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB
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
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=output_dir, filename="mock_detnoisemap_dnm_cal.fits")
    this_caldb.create_entry(noise_map)

    this_caldb.create_entry(flat)
    this_caldb.create_entry(bp_map)
    this_caldb.create_entry(kgain)
    this_caldb.create_entry(new_nonlinearity)
    # now get any default cal files that might be needed; if any reside in the folder that are not
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    # go from L2a, where now dark subtraction should be performed using the noise_map made above, 
    # and get_pc_mean() should ignore the Dark created from noise_map when it 
    # detects that dark-subtraction has already occurred (during the dark_subtraction() step)
    walker.walk_corgidrp(l2a_files, '', output_dir)

    # get photon-counted frame
    master_ill_filename_list = []
    master_ill_filepath_list = []
    pc_processed_filepath = get_last_modified_file(output_dir)
    pc_processed_filename = os.path.split(pc_processed_filepath)[-1]
    for f in os.listdir(output_dir):
        if f == pc_processed_filename:
            master_ill_filename_list.append(f)
            master_ill_filepath_list.append(os.path.join(output_dir, f))
    for i in range(len(master_ill_filepath_list)):
        pc_frame = fits.getdata(master_ill_filepath_list[i])
        pc_frame_err = fits.getdata(master_ill_filepath_list[i], 'ERR')

        # slightly bigger rtol here for synthesized dark vs pc dark above
        assert np.isclose(np.nanmean(pc_frame), ill_mean - dark_mean, rtol=0.05)
        assert pc_frame_err.min() >= 0
    
    # remove synthesized master dark
    for f in os.listdir(output_dir):
        if f.endswith('_drk_cal.fits'):
            os.remove(os.path.join(output_dir, f))

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

    print('Second part of e2e test for photon counting with synthesized master dark passed')
    
    #___________________________________________________
    # now test that the pipeline works if we use an analog traditional master dark (most likely will not be used in practice)
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB
    this_caldb.create_entry(noise_map)
    this_caldb.create_entry(flat)
    this_caldb.create_entry(bp_map)
    this_caldb.create_entry(kgain)
    this_caldb.create_entry(new_nonlinearity)
    # now get any default cal files that might be needed; if any reside in the folder that are not
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    
    # this will add the traditional dark to the caldb
    walker.walk_corgidrp(l1_data_dark_filelist, '', output_dir, template="build_trad_dark_image.json")
    for f in os.listdir(output_dir):
        if f.endswith('_drk_cal.fits'):
            trad_dark_filepath = os.path.join(output_dir, f)
            break
    trad_dark_cal = data.Dark(trad_dark_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=np.zeros((1,1024,1024)).astype(float),
                                    dq = np.zeros((1024,1024)).astype('uint16'), err_hdr=err_hdr)
    trad_dark_cal.save(filedir=output_dir)
    this_caldb.create_entry(trad_dark_cal)
    # go from L2a, where now dark subtraction should be performed using the traditional dark made above, 
    # and get_pc_mean() should ignore the Dark created from traditional dark when it 
    # detects that dark-subtraction has already occurred (during the dark_subtraction() step)
    recipe = walker.autogen_recipe(l2a_files, output_dir)
    ### Modify they keywords of some of the steps
    for step in recipe[0]['steps']:
        if step['name'] == "dark_subtraction":
            step['calibs']['Dark'] = trad_dark_cal.filepath # to find traditional dark
    walker.run_recipe(recipe[0], save_recipe_file=True)
    recipe[1]['inputs'] = glob.glob(os.path.join(recipe[0]['outputdir'], '*_l2a.fits'))
    walker.run_recipe(recipe[1], save_recipe_file=True)
    recipe[2]['inputs'] = glob.glob(os.path.join(recipe[1]['outputdir'], '*_l2a.fits'))
    walker.run_recipe(recipe[2], save_recipe_file=True)

    # get photon-counted frame
    master_ill_filename_list = []
    master_ill_filepath_list = []
    pc_processed_filepath = get_last_modified_file(output_dir)
    pc_processed_filename = os.path.split(pc_processed_filepath)[-1]
    for f in os.listdir(output_dir):
        if f == pc_processed_filename:
            master_ill_filename_list.append(f)
            master_ill_filepath_list.append(os.path.join(output_dir, f))
    for i in range(len(master_ill_filepath_list)):
        pc_frame = fits.getdata(master_ill_filepath_list[i])
        pc_frame_err = fits.getdata(master_ill_filepath_list[i], 'ERR')

        # using trad dark 
        assert np.isclose(np.nanmean(pc_frame), ill_mean - dark_mean, rtol=0.02)
        assert pc_frame_err.min() >= 0

    # Print success message
    print('e2e test for photon counting calibration passed')


if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir
    e2edata_dir =  '/Users/kevinludwick/Documents/DRP E2E Test Files v2/E2E_Test_Data'#'/Users/jmilton/Documents/CGI/E2E_Test_Data2'#'/home/jwang/Desktop/CGI_TVAC_Data/'

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    e2edata_dir = args.e2edata_dir
    test_expected_results_e2e(e2edata_dir, outputdir)
