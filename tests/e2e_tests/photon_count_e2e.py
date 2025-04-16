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
    # e2edata_path not used at all for this test
    np.random.seed(1234)
    ill_dataset, dark_dataset, ill_mean, dark_mean = mocks.create_photon_countable_frames(Nbrights=2, Ndarks=2, cosmic_rate=1, flux=0.5, bad_frames=1)#Nbrights=160, Ndarks=161, cosmic_rate=1, flux=0.5, bad_frames=1)
    output_dir = os.path.join(e2eoutput_path, 'pc_sim_test_data')
    output_ill_dir = os.path.join(output_dir, 'ill_frames')
    output_dark_dir = os.path.join(output_dir, 'dark_frames')
    output_l2a_dir = os.path.join(output_dir, 'l2a')
    output_l2a_dark_dir = os.path.join(output_dir, 'l2a_dark')
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
    if not os.path.exists(output_l2a_dark_dir):
        os.mkdir(output_l2a_dark_dir)
    # empty out directory of any previous files
    for f in os.listdir(output_ill_dir):
        os.remove(os.path.join(output_ill_dir,f))
    for f in os.listdir(output_dark_dir):
        os.remove(os.path.join(output_dark_dir,f))
    ill_dataset.save(output_ill_dir)#, ['pc_frame_ill_{0}.fits'.format(i) for i in range(len(ill_dataset))])
    dark_dataset.save(output_dark_dir)#, ['pc_frame_dark_{0}.fits'.format(i) for i in range(len(dark_dataset))])
    l1_data_ill_filelist = []
    l1_data_dark_filelist = []
    for f in os.listdir(output_ill_dir):
        l1_data_ill_filelist.append(os.path.join(output_ill_dir, f))
    for f in os.listdir(output_dark_dir):
        l1_data_dark_filelist.append(os.path.join(output_dark_dir, f))

    this_caldb = caldb.CalDB() # connection to cal DB
    # remove other KGain calibrations that may exist in case they don't have the added header keywords
    for i in range(len(this_caldb._db['Type'])):
        if this_caldb._db['Type'][i] == 'KGain':
            this_caldb._db = this_caldb._db.drop(i)
        elif this_caldb._db['Type'][i] == 'Dark':
            this_caldb._db = this_caldb._db.drop(i)
    this_caldb.save()

    # KGain
    kgain_val = 7 # default value used in mocks.create_photon_countable_frames()
    pri_hdr, ext_hdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(l1_data_ill_filelist)
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
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
    err_hdr['BUNIT'] = 'detected electrons'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=output_dir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    corgidrp_dir = os.path.split(corgidrp.__path__[0])[0]
    tests_dir = os.path.join(corgidrp_dir, 'tests')
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

    pri_hdr, ext_hdr = mocks.create_default_calibration_product_headers()
    new_nonlinearity = data.NonLinearityCalibration(tvac_nonlin_data,pri_hdr=pri_hdr,ext_hdr=ext_hdr,input_dataset = dummy_dataset)
    new_nonlinearity.filename = nonlin_fits_filepath
    new_nonlinearity.pri_hdr = pri_hdr
    new_nonlinearity.ext_hdr = ext_hdr
    this_caldb.create_entry(new_nonlinearity)

    ## Flat field
    flat_dat = np.ones((1024,1024))
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    flat.save(filedir=output_dir, filename="mock_flat.fits")
    this_caldb.create_entry(flat)

    # bad pixel map
    bp_dat = np.zeros((1024,1024))
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    bp_map.save(filedir=output_dir, filename="mock_bpmap.fits")
    this_caldb.create_entry(bp_map)

    # make PC dark
    # you can leave out the template specification to check that the walker recipe guesser works as expected, going from L1 to L2b for PC dark
    #walker.walk_corgidrp(l1_data_dark_filelist, '', output_dir)#, template="l1_to_l2b_pc_dark.json")
    # instead, for VAP testing, we choose to go to L2a first and then to L2b PC dark
    walker.walk_corgidrp(l1_data_dark_filelist, '', output_l2a_dark_dir, template="l1_to_l2a_basic.json")
    # grab L2a dark files to go to L2b dark
    l2a_dark_files = []
    for filepath in l1_data_dark_filelist:
        # emulate naming change behaviors
        new_filename = filepath.split(os.path.sep)[-1].replace("_L1_", "_L2a") 
        # loook in new dir
        new_filepath = os.path.join(output_l2a_dark_dir, new_filename)
        l2a_dark_files.append(new_filepath)
    walker.walk_corgidrp(l2a_dark_files, '', output_dir, template="l2a_to_l2b_pc_dark_VAP.json")
    # calDB was just updated with the PC Dark that was created with the walker above
    master_dark_filename_list = []
    master_dark_filepath_list = []
    for f in os.listdir(output_dir):
        if not f.endswith('.fits'):
            continue
        if f.endswith('_DRK_CAL.fits'):
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
        new_filename = filepath.split(os.path.sep)[-1].replace("_L1_", "_L2a") 
        # loook in new dir
        new_filepath = os.path.join(output_l2a_dir, new_filename)
        l2a_files.append(new_filepath)
    walker.walk_corgidrp(l2a_files, '', output_dir, template="l2a_to_l2b_pc_VAP.json")

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

        ############# extra checks for VAP testing
        # dimensions of output arrays
        assert pc_frame.shape == (1024,1024)
        # check frame rejection was correctly applied (for illuminated and dark)
        pc_ext_hdr = fits.getheader(master_ill_filepath_list[i], 1)
        assert pc_ext_hdr['NUM_FR'] == len(l1_data_ill_filelist) - 1
        pc_dark_ext_hdr = fits.getheader(master_dark_filepath_list[i], 1)
        assert pc_dark_ext_hdr['NUM_FR'] == len(l1_data_dark_filelist) - 1

    # dimensions of input dark arrays
    for f in l2a_dark_files:
        if not f.endswith('.fits'):
            continue        
        input_frame = fits.getdata(f)
        assert input_frame.shape == (1024,1024)
    # dimensions of input illuminated arrays
    for f in l2a_files:
        if not f.endswith('.fits'):
            continue        
        input_frame = fits.getdata(f)
        assert input_frame.shape == (1024,1024)
    # bpmap is an array of 0 and 1 
    inds_0 = np.where(bp_map.data.ravel() == 0)
    inds_1 = np.where(bp_map.data.ravel() == 1)
    assert len(inds_0[0]) + len(inds_1[0]) == bp_map.data.size
#XXX separate out each test into its own test function so pytest will show results of all tests simultaneously (and not stop output after first failure when all checks in one large test function)?
#XXX fix all TPSCHEME* instances to TPSCHEM*, and adjust wiki pages accordingly? (Check with Julia on Wiki pages, though)

# check image conversion from DN to electrons (covered in run_vi_tdd_05_case_02.py)
# check photon-counting threshold was correctly applied (new test to write)
# check that the mean-combine of all N thresholded frames has been done correctly (covered in
# run_vi_tdd_04_case_03.py)

# check that the mean-combine of all N bad pixel maps has been done correctly (covered in run_vi_tdd_04_case_04.py)
# check that the photon-counting photometric corrections have been properly applied NOTE 
# check that master dark is correctly generated NOTE
# check that master dark is divided by gain (covered in run_vi_tdd_05_case_02.py)
# check that master dark has been correctly photon-counted NOTE
# check that the master dark has been subtracted from the image NOTE
# check that desmearing is correctly applied to image if desmear_flag is set to True (currently no desmear_flag, but want to check that it is indeed applied if part of the recipe) (covered in run_vi_tdd_05_case_02.py)
# check that image is divided by flat field (covered in run_vi_tdd_05_case_02.py)
# check that flat zero values are correctly handled (covered in run_vi_tdd_05_case_02.py)
# check that per-frame bad-pixel map are correctly computed from fixed bad pixel map (covered in run_vi_tdd_05_case_02.py)
# check that pixels are correctly flagged at the frame level (covered in run_vi_tdd_05_case_02.py)
# Visually inspect the output images from a subject matter standpoint to make sure they look correct.

    # load in CalDB again to reflect the PC Dark that was implicitly added in (but not found in this_caldb, which was loaded before the Dark was created)
    post_caldb = caldb.CalDB()
    post_caldb.remove_entry(kgain)
    post_caldb.remove_entry(noise_map)
    post_caldb.remove_entry(new_nonlinearity)
    post_caldb.remove_entry(flat)
    post_caldb.remove_entry(bp_map)
    for filepath in master_dark_filepath_list:
        pc_dark = data.Dark(filepath)
        post_caldb.remove_entry(pc_dark)


if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir
    e2edata_dir =  r"/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/Working_Folder/"#'/home/jwang/Desktop/CGI_TVAC_Data/'

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    e2edata_dir = args.e2edata_dir
    test_expected_results_e2e(e2edata_dir, outputdir)
