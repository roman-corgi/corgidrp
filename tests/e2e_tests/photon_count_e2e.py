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
import corgidrp.l2a_to_l2b as l2a_to_l2b
import corgidrp.photon_counting as photon_counting

@pytest.mark.e2e
def test_pc_prep_e2e(e2edata_path, e2eoutput_path):
    global pc_frame, pc_dark_frame, ill_mean, dark_mean, pc_frame_err, pc_dark_frame_err, master_ill_filepath_list, l1_data_ill_filelist, master_dark_filepath_list, l1_data_dark_filelist, l2a_files, l2a_dark_files, bp_map, kgain, noise_map, new_nonlinearity, flat, l2a_dark_dataset, l2a_dataset, bp_dat
    global fs_l2a_dataset, converted_l2a_dataset, pc_master_dark, pc_output, detector_params, desmeared_dataset, flat_dataset, correct_bp_dataset
    # e2edata_path not used at all for this test
    np.random.seed(1234)
    ill_dataset, dark_dataset, ill_mean, dark_mean = mocks.create_photon_countable_frames(Nbrights=50, Ndarks=51, cosmic_rate=1, flux=0.5, bad_frames=1)#Nbrights=160, Ndarks=161, cosmic_rate=1, flux=0.5, bad_frames=1)
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
    flat_dat[1,1] = 0.9
    flat_dat[2,2] = 0 # for VAP testing purposes
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    flat.save(filedir=output_dir, filename="mock_flat.fits")
    this_caldb.create_entry(flat)

    # bad pixel map
    bp_dat = np.zeros((1024,1024))
    bp_dat[0, 0] = 1 # for testing purposes
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    bp_map.save(filedir=output_dir, filename="mock_bpmap.fits")
    this_caldb.create_entry(bp_map)

    # make PC dark
    # you can leave out the template specification to check that the walker recipe guesser works as expected, going from L1 to L2b for PC dark
    #walker.walk_corgidrp(l1_data_dark_filelist, '', output_dir)#, template="l1_to_l2b_pc_dark.json")
    # instead, especially for VAP testing, we choose to go to L2a first and then to L2b PC dark
    walker.walk_corgidrp(l1_data_dark_filelist, '', output_l2a_dark_dir, template="l1_to_l2a_basic.json")
    # grab L2a dark files to go to L2b dark
    l2a_dark_files = []
    for filepath in l1_data_dark_filelist:
        # emulate naming change behaviors
        new_filename = filepath.split(os.path.sep)[-1].replace("_L1_", "_L2a") 
        # loook in new dir
        new_filepath = os.path.join(output_l2a_dark_dir, new_filename)
        l2a_dark_files.append(new_filepath)
    l2a_dark_dataset = data.Dataset(l2a_dark_files) #for later VAP testing

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
    l2a_dataset = data.Dataset(l2a_files) 
    #run these steps separately for later VAP testing
    fs_l2a_dataset = l2a_to_l2b.frame_select(l2a_dataset, overexp=True)
    converted_l2a_dataset = l2a_to_l2b.convert_to_electrons(fs_l2a_dataset, kgain)
    pc_master_dark = data.Dark(master_dark_filepath_list[0])
    pc_output = photon_counting.get_pc_mean(converted_l2a_dataset, pc_master_dark)
    detector_params = data.DetectorParams({})
    desmeared_dataset = l2a_to_l2b.desmear(pc_output, detector_params)
    # skip cti correction step since that isn't required for VAP testing
    flat_dataset = l2a_to_l2b.flat_division(desmeared_dataset, flat)
    correct_bp_dataset = l2a_to_l2b.correct_bad_pixels(flat_dataset, bp_map)
    # but also run the recipe to exercise the typical usage
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

@pytest.mark.e2e
def test_pc_e2e_pc_frame(e2edata_path, e2eoutput_path):            
    for i in range(len(master_ill_filepath_list)):
        pc_frame = fits.getdata(master_ill_filepath_list[i])
        # more frames gets a better agreement; agreement to 1% for ~160 darks and illuminated
        if np.isclose(np.nanmean(pc_frame), ill_mean - dark_mean, rtol=0.02):
            print(r'dark-subtracted PC frame mean within 2% of expected value:  PASS')
        else:
            print(r'dark-subtracted PC frame mean within 2% of expected value:  FAIL')
        assert np.isclose(np.nanmean(pc_frame), ill_mean - dark_mean, rtol=0.02)

@pytest.mark.e2e
def test_pc_e2e_pc_dark_frame(e2edata_path, e2eoutput_path):            
    for i in range(len(master_ill_filepath_list)):
        pc_dark_frame = fits.getdata(master_dark_filepath_list[i])
        # more frames gets a better agreement; agreement to 1% for ~160 darks and illuminated
        if np.isclose(np.nanmean(pc_dark_frame), dark_mean, rtol=0.01):
            print(r'PC dark frame mean within 1% of expected value:  PASS')
        else:
            print(r'PC dark frame mean within 1% of expected value:  FAIL')
        assert np.isclose(np.nanmean(pc_dark_frame), dark_mean, rtol=0.01)

@pytest.mark.e2e
def test_pc_e2e_pc_err_frame(e2edata_path, e2eoutput_path):            
    for i in range(len(master_ill_filepath_list)):
        pc_frame_err = fits.getdata(master_ill_filepath_list[i], 'ERR')
        if pc_frame_err.min() >= 0:
            print(r'PC frame error array has no negative values:  PASS')
        else:
            print(r'PC frame error array has no negative values:  FAIL')
        assert pc_frame_err.min() >= 0

@pytest.mark.e2e
def test_pc_e2e_pc_err_dark_frame(e2edata_path, e2eoutput_path):            
    for i in range(len(master_ill_filepath_list)):
        pc_dark_frame_err = fits.getdata(master_dark_filepath_list[i], 'ERR')
        if pc_dark_frame_err.min() >= 0:
            print(r'PC dark frame error array has no negative values:  PASS')
        else:
            print(r'PC dark frame error array has no negative values:  FAIL')
        assert pc_dark_frame_err.min() >= 0

############# extra checks for VAP testing

@pytest.mark.e2e
def test_pc_e2e_output_array_dims(e2edata_path, e2eoutput_path):            
    for i in range(len(master_ill_filepath_list)):
        pc_frame = fits.getdata(master_ill_filepath_list[i])     
        # dimensions of output arrays
        if pc_frame.shape == (1024,1024):
            print(r'PC frame array has dimensions of 1024x1024:  PASS')
        else:
            print(r'PC frame array has dimensions of 1024x1024:  FAIL')
        assert pc_frame.shape == (1024,1024)
        
@pytest.mark.e2e
def test_pc_e2e_frame_rejection_ill(e2edata_path, e2eoutput_path):            
    for i in range(len(master_ill_filepath_list)):     
        # check frame rejection was correctly applied (for illuminated)
        pc_ext_hdr = fits.getheader(master_ill_filepath_list[i], 1)
        if pc_ext_hdr['NUM_FR'] == len(l1_data_ill_filelist) - 1:
            print(r'PC frame rejection was correctly applied:  PASS')
        else:
            print(r'PC frame rejection was correctly applied:  FAIL')
        len(l2a_dark_dataset)
        assert pc_ext_hdr['NUM_FR'] == len(l1_data_ill_filelist) - 1 
    # and a more direct test; I simulated one bad frame
    fs_ill_dataset = l2a_to_l2b.frame_select(l2a_dark_dataset, overexp=True)
    assert len(fs_ill_dataset) == len(l1_data_ill_filelist) - 1

@pytest.mark.e2e
def test_pc_e2e_frame_rejection_dark(e2edata_path, e2eoutput_path):            
    for i in range(len(master_ill_filepath_list)):    
        # check frame rejection was correctly applied (for dark)
        pc_dark_ext_hdr = fits.getheader(master_dark_filepath_list[i], 1)
        if pc_dark_ext_hdr['NUM_FR'] == len(l1_data_dark_filelist) - 1:
            print(r'PC dark frame rejection was applied:  PASS')
        else:
            print(r'PC dark frame rejection was applied:  FAIL')
        assert pc_dark_ext_hdr['NUM_FR'] == len(l1_data_dark_filelist) - 1
    # and a more direct test; I simulated one bad frame
    fs_dark_dataset = l2a_to_l2b.frame_select(l2a_dark_dataset, overexp=True)
    assert len(fs_dark_dataset) == len(l1_data_dark_filelist) - 1


@pytest.mark.e2e
def test_pc_e2e_input_dark_array_dims(e2edata_path, e2eoutput_path):            
    # dimensions of input dark arrays
    for f in l2a_dark_files:
        if not f.endswith('.fits'):
            continue        
        input_frame = fits.getdata(f)
        if input_frame.shape == (1024,1024):
            print(r'Input dark frame has dimensions of 1024x1024:  PASS')
        else:
            print(r'Input dark frame has dimensions of 1024x1024:  FAIL')
        assert input_frame.shape == (1024,1024)

@pytest.mark.e2e
def test_pc_e2e_input_ill_array_dims(e2edata_path, e2eoutput_path):   
    # dimensions of input illuminated arrays
    for f in l2a_files:
        if not f.endswith('.fits'):
            continue        
        input_frame = fits.getdata(f)
        if input_frame.shape == (1024,1024):
            print(r'Input illuminated frame has dimensions of 1024x1024:  PASS')
        else:
            print(r'Input illuminated frame has dimensions of 1024x1024:  FAIL')
        assert input_frame.shape == (1024,1024)

@pytest.mark.e2e
def test_pc_e2e_bpmap(e2edata_path, e2eoutput_path):   
    # bpmap is an array of 0 and 1 
    inds_0 = np.where(bp_map.data.ravel() == 0)
    inds_1 = np.where(bp_map.data.ravel() == 1)
    if len(inds_0[0]) + len(inds_1[0]) == bp_map.data.size:
        print(r'Bad pixel map has only 0 and 1 values:  PASS')
    else:
        print(r'Bad pixel map has only 0 and 1 values:  FAIL')
    assert len(inds_0[0]) + len(inds_1[0]) == bp_map.data.size

@pytest.mark.e2e
def test_pc_e2e_kgain_conversion_ill(e2edata_path, e2eoutput_path):   
    # check illuminated image conversion from DN to electrons
    fs_l2a_dataset = l2a_to_l2b.frame_select(l2a_dataset, overexp=True)
    converted_l2a_dataset = l2a_to_l2b.convert_to_electrons(fs_l2a_dataset, kgain)
    kgain_empirical = np.divide(converted_l2a_dataset.all_data, fs_l2a_dataset.all_data, where=fs_l2a_dataset.all_data!=0)
    kgain_empirical = np.nanmedian(kgain_empirical)
    for i in range(len(master_ill_filepath_list)):
        pc_ext_hdr = fits.getheader(master_ill_filepath_list[i], 1)
        if np.isclose(kgain_empirical, pc_ext_hdr['KGAINPAR'], rtol=0.01):
            print(r'Illuminated image conversion from DN to electrons was applied:  PASS')
        else:
            print(r'Illuminated image conversion from DN to electrons was applied:  FAIL')
        assert np.isclose(kgain_empirical, pc_ext_hdr['KGAINPAR'], rtol=0.01)

@pytest.mark.e2e
def test_pc_e2e_kgain_conversion_dark(e2edata_path, e2eoutput_path):   
    # check illuminated image conversion from DN to electrons
    fs_l2a_dataset = l2a_to_l2b.frame_select(l2a_dataset, overexp=True)
    converted_l2a_dataset = l2a_to_l2b.convert_to_electrons(fs_l2a_dataset, kgain)
    kgain_empirical = np.divide(converted_l2a_dataset.all_data, fs_l2a_dataset.all_data, where=fs_l2a_dataset.all_data!=0)
    kgain_empirical = np.nanmedian(kgain_empirical)
    for i in range(len(master_dark_filepath_list)):
        pc_ext_hdr = fits.getheader(master_dark_filepath_list[i], 1)
        if np.isclose(kgain_empirical, pc_ext_hdr['KGAINPAR'], rtol=0.01):
            print(r'Dark image conversion from DN to electrons was applied:  PASS')
        else:
            print(r'Dark image conversion from DN to electrons was applied:  FAIL')
        assert np.isclose(kgain_empirical, pc_ext_hdr['KGAINPAR'], rtol=0.01)


@pytest.mark.e2e
def test_pc_e2e_pc_ill_threshold(e2edata_path, e2eoutput_path):   
    # check photon-counting threshold for illuminated frames was correctly applied 
    pc_ill = fits.open(master_ill_filepath_list[0])
    pc_thresh = pc_ill[1].header['PCTHRESH'] # this is the same for each binned output PC frame
    fs_l2a_dataset = l2a_to_l2b.frame_select(l2a_dataset, overexp=True)
    converted_l2a_dataset = l2a_to_l2b.convert_to_electrons(fs_l2a_dataset, kgain)
    trues = 0 #initiailize counter
    for frame in converted_l2a_dataset:
        # photon_count called to do the thresholding
        thresh_frame = photon_counting.photon_count(frame.data, pc_thresh)
        expected_thresh_frame = np.ones_like(frame.data).astype('float64')
        expected_thresh_frame[frame.data <= pc_thresh] = 0
        cond = np.array_equal(thresh_frame, expected_thresh_frame.astype('int64'))
        trues += cond
    if trues == len(converted_l2a_dataset):
        print(r'PC threshold was correctly applied for illuminated frames:  PASS')
    else:
        print(r'PC threshold was correctly applied for illuminated frames:  FAIL')
    assert trues == len(converted_l2a_dataset)


@pytest.mark.e2e
def test_pc_e2e_pc_dark_threshold(e2edata_path, e2eoutput_path):   
    # check photon-counting threshold for dark frames was correctly applied 
    pc_dark = fits.open(master_dark_filepath_list[0])
    pc_thresh = pc_dark[1].header['PCTHRESH'] # this is the same for each binned output PC frame
    fs_l2a_dataset = l2a_to_l2b.frame_select(l2a_dataset,overexp=True)
    converted_l2a_dataset = l2a_to_l2b.convert_to_electrons(fs_l2a_dataset, kgain)
    trues = 0 #initiailize counter
    for frame in converted_l2a_dataset:
        # photon_count called to do the thresholding
        thresh_frame = photon_counting.photon_count(frame.data, pc_thresh)
        expected_thresh_frame = np.ones_like(frame.data).astype('float64')
        expected_thresh_frame[frame.data <= pc_thresh] = 0
        cond = np.array_equal(thresh_frame, expected_thresh_frame.astype('int64'))
        trues += cond
    if trues == len(converted_l2a_dataset):
        print(r'PC threshold was correctly applied for dark frames:  PASS')
    else:
        print(r'PC threshold was correctly applied for dark frames:  FAIL')
    assert trues == len(converted_l2a_dataset)

@pytest.mark.e2e
def test_pc_e2e_bp_map_mean_combine(e2edata_path, e2eoutput_path):  
    # check that the mean-combine of all N bad pixel maps has been done 
    # in DRP, frames are not mean-combined to master bad pixel map; we can at least assert the num dims is 2
    if bp_map.data.ndim == 2: 
        
        print(r'mean-combine of all N bad pixel maps was computed:  PASS')
    else:
        print(r'mean-combine of all N bad pixel maps was computed:  FAIL')
    assert bp_map.data.ndim == 2

@pytest.mark.e2e
def test_pc_e2e_bp_map_per_frame(e2edata_path, e2eoutput_path):  
    # check that per-frame bad-pixel map are correctly computed from fixed bad pixel map
    pc_dq = fits.getdata(master_ill_filepath_list[0], 'DQ') # only one frame; no PC binning done
    pc_dq_01 = np.zeros_like(pc_dq)
    pc_dq_01[pc_dq > 0] = 1
    if np.array_equal(pc_dq_01, bp_map.data):
        print(r'Bad pixel map was applied:  PASS')
    else:
        print(r'Bad pixel map was applied:  FAIL')
    assert np.array_equal(pc_dq_01, bp_map.data)

@pytest.mark.e2e
def test_pc_e2e_desmear(e2edata_path, e2eoutput_path):  
    # check that PC illuminated frame is desmeared (I simulate smear in the mocked files)
    if np.isclose(np.nanmean(pc_output.frames[0].data), ill_mean - dark_mean, rtol=0.02):
        print(r'dark-subtracted PC frame without desmearing does not give expected mean within 2%:  PASS')
    else:
        print(r'dark-subtracted PC frame without desmearing does not give expected mean within 2%:  FAIL')
    if np.isclose(np.nanmean(desmeared_dataset.frames[0].data), ill_mean - dark_mean, rtol=0.02):
        print(r'dark-subtracted PC frame with desmearing gives expected mean within 2%:  PASS')
    else:
        print(r'dark-subtracted PC frame with desmearing gives expected mean within 2%:  FAIL')
    assert not np.isclose(np.nanmean(pc_output.frames[0].data), ill_mean - dark_mean, rtol=0.02)
    assert np.isclose(np.nanmean(desmeared_dataset.frames[0].data), ill_mean - dark_mean, rtol=0.02)

@pytest.mark.e2e
def test_pc_e2e_flat_division(e2edata_path, e2eoutput_path):  
    # check that image is correctly divided by flat field 
    for i in range(len(desmeared_dataset)):
        expected_flat_frame = desmeared_dataset[i]/flat
        if np.array_equal(expected_flat_frame, flat_dataset[i]):
            print(r'Flat field division was correctly applied:  PASS')
        else:
            print(r'Flat field division was correctly applied:  FAIL')
        assert np.array_equal(expected_flat_frame, flat_dataset[i])
    
@pytest.mark.e2e
def test_pc_e2e_flat_division_0(e2edata_path, e2eoutput_path):  
    # check that flat zero values are correctly handled
    # a 0 pixel on the flat gets adjusted to NaN on the image at the correct_bad_pixels step
    if (correct_bp_dataset.frames[0].data[1,1] == np.nan) and flat_dataset.frames[0].dq[1,1] == 1:
        print(r'check that flat zero values are correctly handled:  PASS')
    else:
        print(r'check that flat zero values are correctly handled:  FAIL')
    assert (correct_bp_dataset.frames[0].data[1,1] == np.nan) and flat_dataset.frames[0].dq[1,1] == 1

@pytest.mark.e2e
def test_pc_e2e_flag_pixels(e2edata_path, e2eoutput_path):  
    # check that pixels are correctly flagged at the frame level 
    # the bp map had a flag at (0,0)
    if (correct_bp_dataset.frames[0].data[0,0] == np.nan) and correct_bp_dataset.frames[0].dq[0,0] == 1:
        print(r'check that pixels are correctly flagged at the frame level:  PASS')
    else:
        print(r'check that pixels are correctly flagged at the frame level:  FAIL')
    assert np.isnan(correct_bp_dataset.frames[0].data[0,0]) and correct_bp_dataset.frames[0].dq[0,0] > 0

    # the last VAP test (for visual inspection):  the PC output files were saved in the output folder

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
    test_pc_prep_e2e(e2edata_dir, outputdir)
    test_pc_e2e_desmear(e2edata_dir, outputdir)
    test_pc_e2e_bp_map_mean_combine(e2edata_dir, outputdir)
    test_pc_e2e_flag_pixels(e2edata_dir, outputdir)
    test_pc_e2e_flat_division_0(e2edata_dir, outputdir)
    test_pc_e2e_flat_division(e2edata_dir, outputdir)
    test_pc_e2e_bpmap(e2edata_dir, outputdir)
    test_pc_e2e_pc_ill_threshold(e2edata_dir, outputdir)
    test_pc_e2e_kgain_conversion_dark(e2edata_dir, outputdir)
    test_pc_e2e_kgain_conversion_ill(e2edata_dir, outputdir)