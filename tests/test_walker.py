"""
Test the walker infrastructure to read and execute recipes
"""
import os
import glob
import json
import warnings
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.caldb as caldb
import corgidrp.walker as walker
import corgidrp.detector as detector

np.random.seed(456)

def test_autoreducing():
    """
    Tests both generating and processing a basic L1->L2a SCI recipe
    """
    # create dirs
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    outputdir = os.path.join(os.path.dirname(__file__), "walker_output")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)


    # create simulated data
    l1_dataset = mocks.create_prescan_files(filedir=datadir, arrtype="SCI", numfiles=2)
    # simulate the expected CGI naming convention
    fname_template = "cgi_0200001999001000{:03d}_20250415t0305102_l1_.fits"
    for i, image in enumerate(l1_dataset):
        image.filename = fname_template.format(i)
    l1_dataset.save(filedir=datadir)
    filelist = [frame.filepath for frame in l1_dataset]


    ###########################################
    ### Create a dummy non-linearity file ####
    #Create a mock dataset because it is a required input when creating a NonLinearityCalibration
    dummy_dataset = mocks.create_prescan_files()

    # Make a non-linearity correction calibration file
    input_non_linearity_filename = "nonlin_table_TVAC.txt"
    input_non_linearity_path = os.path.join(os.path.dirname(__file__), "test_data", input_non_linearity_filename)
    test_non_linearity_filename = input_non_linearity_filename.split(".")[0] + ".fits"
    nonlin_fits_filepath = os.path.join(os.path.dirname(__file__), "test_data", test_non_linearity_filename)
    tvac_nonlin_data = np.genfromtxt(input_non_linearity_path, delimiter=",")

    pri_hdr, ext_hdr = mocks.create_default_L1_headers()
    new_nonlinearity = data.NonLinearityCalibration(tvac_nonlin_data,pri_hdr=pri_hdr,ext_hdr=ext_hdr,input_dataset = dummy_dataset)
    new_nonlinearity.filename = nonlin_fits_filepath
    new_nonlinearity.save()
    # index the sample nonlinearity correction that we need for processing
    # fake the headers because this frame doesn't have the proper headers
    prihdr, exthdr = mocks.create_default_L1_headers("SCI")
    new_nonlinearity.pri_hdr = prihdr
    new_nonlinearity.ext_hdr = exthdr
    new_nonlinearity.ext_hdr.set('DRPCTIME', time.Time.now().isot, "When this file was saved")
    new_nonlinearity.ext_hdr.set('DRPVERSN', corgidrp.__version__, "corgidrp version that produced this file")
    mycaldb = caldb.CalDB()
    mycaldb.create_entry(new_nonlinearity)
    
    #Make a KGain calibration file
    kgain = 8.8
    new_kgain = data.KGain(kgain,pri_hdr=prihdr,ext_hdr=exthdr,input_dataset = dummy_dataset)
    new_kgain.ext_hdr.set('DRPCTIME', time.Time.now().isot, "When this file was saved")
    new_kgain.ext_hdr.set('DRPVERSN', corgidrp.__version__, "corgidrp version that produced this file")
    new_kgain.save(filedir = os.path.join(os.path.dirname(__file__), "test_data"), filename = "kgain.fits")
    
    mycaldb.create_entry(new_kgain)
    

    CPGS_XML_filepath = "" # not yet implemented

    # generate recipe and run it
    # basic l2a recipe to keep things simple
    recipe = walker.walk_corgidrp(filelist, CPGS_XML_filepath, outputdir, template="l1_to_l2a_basic.json")

    # check that the output dataset is saved to the output dir
    # filenames have been updated to L2a. 
    output_files = [os.path.join(outputdir, frame.filename.replace("_l1_", "_l2a")) for frame in l1_dataset]
    output_dataset = data.Dataset(output_files)
    assert len(output_dataset) == len(l1_dataset) # check the same number of files
    # check that the recipe is saved into the header.

    for frame in output_dataset:
        assert "RECIPE" in frame.ext_hdr
        # test recipe was correctly written into the header
        # do a string comparison, easiest way to check
        hdr_recipe = json.loads(frame.ext_hdr["RECIPE"])
        assert json.dumps(hdr_recipe) == json.dumps(recipe)

    # clean up
    mycaldb.remove_entry(new_nonlinearity)
    mycaldb.remove_entry(new_kgain)

def test_auto_template_identification():
    """
    Tests the process of coming up with the right template and filling in calibration files
    """
    # create dirs
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    outputdir = os.path.join(os.path.dirname(__file__), "walker_output")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # create simulated data
    l1_dataset = mocks.create_prescan_files(filedir=datadir, arrtype="SCI", numfiles=2)
    # simulate the expected CGI naming convention
    fname_template = "cgi_0200001999001000{:03d}_20250415t0305102_l1_.fits"
    for i, image in enumerate(l1_dataset):
        image.filename = fname_template.format(i)
    l1_dataset.save(filedir=datadir)
    filelist = [frame.filepath for frame in l1_dataset]


    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make calibration files using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(filelist)

    this_caldb = caldb.CalDB() # connection to cal DB

    # Nonlinearity calibration
    ### Create a dummy non-linearity file ####
    #Create a mock dataset because it is a required input when creating a NonLinearityCalibration
    dummy_dataset = mocks.create_prescan_files()

    # Make a non-linearity correction calibration file
    input_non_linearity_filename = "nonlin_table_TVAC.txt"
    input_non_linearity_path = os.path.join(os.path.dirname(__file__), "test_data", input_non_linearity_filename)
    test_non_linearity_filename = input_non_linearity_filename.split(".")[0] + ".fits"
    nonlin_fits_filepath = os.path.join(os.path.dirname(__file__), "test_data", test_non_linearity_filename)
    tvac_nonlin_data = np.genfromtxt(input_non_linearity_path, delimiter=",")

    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    new_nonlinearity = data.NonLinearityCalibration(tvac_nonlin_data,pri_hdr=pri_hdr,ext_hdr=ext_hdr,input_dataset = dummy_dataset)
    new_nonlinearity.filename = nonlin_fits_filepath
    # fake the headers because this frame doesn't have the proper headers
    new_nonlinearity.pri_hdr = pri_hdr
    new_nonlinearity.ext_hdr = ext_hdr
    this_caldb.create_entry(new_nonlinearity)

    # KGain
    kgain_val = 8.7
    kgain = data.KGain(kgain_val, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)

    # NoiseMap
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=outputdir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    ## Flat field
    flat_dat = np.ones((1024, 1024))
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    flat.save(filedir=outputdir, filename="mock_flat.fits")
    this_caldb.create_entry(flat)

    # bad pixel map
    bp_dat = np.zeros((1024, 1024), dtype=int)
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    bp_map.save(filedir=outputdir, filename="mock_bpmap.fits")
    this_caldb.create_entry(bp_map)


    ### Finally, test the recipe identification
    recipe = walker.autogen_recipe(filelist, outputdir)

    assert recipe['name'] == 'l1_to_l2a_basic'
    assert recipe['template'] == False

    # now cleanup
    this_caldb.remove_entry(new_nonlinearity)
    this_caldb.remove_entry(kgain)
    this_caldb.remove_entry(noise_map)
    this_caldb.remove_entry(flat)
    this_caldb.remove_entry(bp_map)


def test_saving():
    """
    Tests the special save function including suffix. Tries both calibration image and non-calibration image
    """
    ### create dirs
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    outputdir = os.path.join(os.path.dirname(__file__), "walker_output")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    ### load test recipe
    testdatadir = os.path.join(os.path.dirname(__file__), "test_data")
    save_recipe_path = os.path.join(testdatadir, "saving_only.json")
    save_recipe = json.load(open(save_recipe_path, 'r'))
    
    ######################
    ## Test regular images
    ######################

    ### Create mock Image data
    l1_dataset = mocks.create_dark_calib_files(filedir=datadir, numfiles=2)
    # simulate the expected CGI naming convention
    fname_template = "cgi_0200001999001000{:03d}_20250415t0305102_l1_.fits"
    for i, image in enumerate(l1_dataset):
        image.filename = fname_template.format(i)
    l1_dataset.save(filedir=datadir)
    filelist = [frame.filepath for frame in l1_dataset]

    this_recipe = walker.autogen_recipe(filelist, outputdir, template=save_recipe)
    walker.run_recipe(this_recipe)

    # check that the output dataset is saved to the output dir
    # filenames have been appended with a suffix
    output_files = [os.path.join(outputdir, "cgi_0200001999001000{:03d}_20250415t0305102_l1__test.fits".format(i)) for i in range(len(l1_dataset))]
    output_dataset = data.Dataset(output_files)
    assert len(output_dataset) == len(l1_dataset) # check the same number of files
    
    ##########################
    ## Test calibration image 
    ##########################
    # Fake a nonlinearity dataset
    ### Create a dummy non-linearity file ####
    #Create a mock dataset because it is a required input when creating a NonLinearityCalibration
    dummy_dataset = mocks.create_prescan_files()

    # Make a non-linearity correction calibration file
    input_non_linearity_filename = "nonlin_table_TVAC.txt"
    input_non_linearity_path = os.path.join(os.path.dirname(__file__), "test_data", input_non_linearity_filename)
    tvac_nonlin_data = np.genfromtxt(input_non_linearity_path, delimiter=",")
    test_non_linearity_filename = input_non_linearity_filename.split(".")[0] + ".fits"
    nonlin_fits_filepath = os.path.join(os.path.dirname(__file__), "test_data", test_non_linearity_filename)
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    fake_nonlinearity = data.NonLinearityCalibration(tvac_nonlin_data,pri_hdr=pri_hdr,ext_hdr=ext_hdr,input_dataset = dummy_dataset)
    fake_nonlinearity.filename = nonlin_fits_filepath
    # fake the headers because this frame doesn't have the proper headers
    prihdr, exthdr = mocks.create_default_L1_headers("SCI")
    fake_nonlinearity.pri_hdr = prihdr
    fake_nonlinearity.ext_hdr = exthdr
    fake_nonlinearity.ext_hdr['DATATYPE'] = 'NonLinearityCalibration'
    fake_nonlinearity.ext_hdr.set('DRPCTIME', time.Time.now().isot, "When this file was saved")
    fake_nonlinearity.ext_hdr.set('DRPVERSN', corgidrp.__version__, "corgidrp version that produced this file")
    fake_nonlinearity.filename = "CGI_test.fits"

    # tested the run_recipe portion of the code already (nothing different)
    # test save_data when we only pass in a single calibration file and not a dataset to see how it goes
    walker.save_data(fake_nonlinearity, outputdir, "nonlin")
    output_filepath = os.path.join(outputdir, "CGI_test_nonlin.fits")
    new_nonlinearity = data.NonLinearityCalibration(output_filepath)

    # remove this entry. should only work if it's already in the database
    # also used for cleanup
    this_caldb = caldb.CalDB()
    this_caldb.remove_entry(new_nonlinearity)



def test_skip_missing_calib():
    """
    Tests the option of skipping steps with missing calibrations
    """
    # turn on skipping
    old_setting = corgidrp.skip_missing_cal_steps
    corgidrp.skip_missing_cal_steps = True

    # use an empty test caldb
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    testcaldb_filepath = os.path.join(calibdir, "empty_caldb.csv")
    old_caldb_filepath = corgidrp.caldb_filepath
    corgidrp.caldb_filepath = testcaldb_filepath

    # create dirs
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    outputdir = os.path.join(os.path.dirname(__file__), "walker_output")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # create simulated data
    l1_dataset = mocks.create_prescan_files(filedir=datadir, arrtype="SCI", numfiles=2)
    # simulate the expected CGI naming convention
    fname_template = "cgi_0200001999001000{:03d}_20250415t0305102_l1_.fits"
    for i, image in enumerate(l1_dataset):
        image.filename = fname_template.format(i)
    l1_dataset.save(filedir=datadir)
    filelist = [frame.filepath for frame in l1_dataset]


    ### Test that we are skipping the steps without calibrations
    # use l1 to l2b recipe
    template_filepath = os.path.join(os.path.dirname(walker.__file__), "recipe_templates", "l1_to_l2b.json")
    template_recipe = json.load(open(template_filepath, "r"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # because walker throws a UserWarning about skipping missing calibs, catch here in tests
        recipe = walker.autogen_recipe(filelist, outputdir, template=template_recipe)

    assert recipe['name'] == 'l1_to_l2b'
    assert recipe['template'] == False

    assert recipe['steps'][0]['skip'] # prescan bias sub    
    assert recipe['steps'][2]['skip'] # nonlinearity
    assert recipe['steps'][6]['skip'] # kgain
    
    # cut down to recipe to just the first 5 steps (to the first save)
    recipe['steps'] = recipe['steps'][:5]

    # run with all the ksips
    walker.run_recipe(recipe, save_recipe_file=False)

    # check that the output dataset is saved to the output dir
    # filenames have been appended with a suffix
    output_files = [os.path.join(outputdir, "cgi_0200001999001000{:03d}_20250415t0305102_l2a.fits".format(i)) for i in range(len(l1_dataset))]
    output_dataset = data.Dataset(output_files)
    assert len(output_dataset) == len(l1_dataset) # check the same number of files

    for hist_entry in output_dataset[0].ext_hdr['HISTORY']:
        assert 'non-linearity' not in hist_entry.lower()

    corgidrp.skip_missing_cal_steps = old_setting
    corgidrp.caldb_filepath = old_caldb_filepath



def test_skip_missing_optional_calib():
    """
    Tests optional calibrtion behavior when skpip_missing_calibs is True
    The behavior is that the step should not be skipped, given the calibration is optional
    """
    # turn on skipping
    old_setting = corgidrp.skip_missing_cal_steps
    corgidrp.skip_missing_cal_steps = True

    # use an empty test caldb
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    testcaldb_filepath = os.path.join(calibdir, "empty_caldb.csv")
    old_caldb_filepath = corgidrp.caldb_filepath
    corgidrp.caldb_filepath = testcaldb_filepath

    # create dirs
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    outputdir = os.path.join(os.path.dirname(__file__), "walker_output")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # create simulated data
    l1_dataset = mocks.create_prescan_files(filedir=datadir, arrtype="SCI", numfiles=2)
    # simulate the expected CGI naming convention
    fname_template = "cgi_0200001999001000{:03d}_20250415t0305102_l1_.fits"
    for i, image in enumerate(l1_dataset):
        image.filename = fname_template.format(i)
    l1_dataset.save(filedir=datadir)
    filelist = [frame.filepath for frame in l1_dataset]


    # use l1 to l2a recipe since it as optional calibration
    template_filepath = os.path.join(os.path.dirname(walker.__file__), "recipe_templates", "l1_to_l2a_basic.json")
    template_recipe = json.load(open(template_filepath, "r"))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # because walker throws a UserWarning about skipping missing calibs, catch here in tests
        recipe = walker.autogen_recipe(filelist, outputdir, template=template_recipe)

    # check prescan bias sub is not skipped and Detector Noise Maps is None
    assert 'skip' not in recipe['steps'][0] # prescan biassub
    assert recipe['steps'][0]['calibs']['DetectorNoiseMaps'] is None

    # assert nonlinearity is indeed skipped
    assert recipe['steps'][2]['skip'] # nonlinearity

    corgidrp.skip_missing_cal_steps = old_setting
    corgidrp.caldb_filepath = old_caldb_filepath

def test_jit_calibs():
    """
    Tests defining calibrations just in time
    """
    old_setting = corgidrp.jit_calib_id
    corgidrp.jit_calib_id = True

    # create dirs
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    outputdir = os.path.join(os.path.dirname(__file__), "walker_output")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)


    # create simulated data
    l1_dataset = mocks.create_prescan_files(filedir=datadir, arrtype="SCI", numfiles=2)
    # simulate the expected CGI naming convention
    fname_template = "cgi_0200001999001000{:03d}_20250415t0305102_l1_.fits"
    for i, image in enumerate(l1_dataset):
        image.filename = fname_template.format(i)
    l1_dataset.save(filedir=datadir)
    filelist = [frame.filepath for frame in l1_dataset]


    # index the sample nonlinearity correction that we need for processing
    # Fake a nonlinearity dataset
    ### Create a dummy non-linearity file ####
    #Create a mock dataset because it is a required input when creating a NonLinearityCalibration
    dummy_dataset = mocks.create_prescan_files()

    # Make a non-linearity correction calibration file
    input_non_linearity_filename = "nonlin_table_TVAC.txt"
    input_non_linearity_path = os.path.join(os.path.dirname(__file__), "test_data", input_non_linearity_filename)
    tvac_nonlin_data = np.genfromtxt(input_non_linearity_path, delimiter=",")
    test_non_linearity_filename = input_non_linearity_filename.split(".")[0] + ".fits"
    nonlin_fits_filepath = os.path.join(os.path.dirname(__file__), "test_data", test_non_linearity_filename)
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    new_nonlinearity = data.NonLinearityCalibration(tvac_nonlin_data,pri_hdr=pri_hdr,ext_hdr=ext_hdr,input_dataset = dummy_dataset)
    new_nonlinearity.filename = nonlin_fits_filepath
    # fake the headers because this frame doesn't have the proper headers
    prihdr, exthdr = mocks.create_default_L1_headers("SCI")
    new_nonlinearity.pri_hdr = prihdr
    new_nonlinearity.ext_hdr = exthdr
    new_nonlinearity.ext_hdr.set('DRPCTIME', time.Time.now().isot, "When this file was saved")
    new_nonlinearity.ext_hdr.set('DRPVERSN', corgidrp.__version__, "corgidrp version that produced this file")
    mycaldb = caldb.CalDB()
    mycaldb.create_entry(new_nonlinearity)

    #Make a KGain calibration file
    kgain_val = 8.8
    new_kgain = data.KGain(kgain_val,pri_hdr=prihdr,ext_hdr=exthdr,input_dataset = dummy_dataset)
    new_kgain.ext_hdr.set('DRPCTIME', time.Time.now().isot, "When this file was saved")
    new_kgain.ext_hdr.set('DRPVERSN', corgidrp.__version__, "corgidrp version that produced this file")
    new_kgain.save(filedir = os.path.join(os.path.dirname(__file__), "test_data"), filename = "kgain.fits")
    
    mycaldb.create_entry(new_kgain)

    CPGS_XML_filepath = "" # not yet implemented

    # generate recipe and check we haven't defined anyhting yet
    template_filepath = os.path.join(os.path.dirname(walker.__file__), "recipe_templates", "l1_to_l2a_basic.json")
    template_recipe = json.load(open(template_filepath, "r"))
    recipe = walker.autogen_recipe(filelist, outputdir, template=template_recipe)

    assert recipe['steps'][2]['calibs']['NonLinearityCalibration'] == 'AUTOMATIC' 

    walker.run_recipe(recipe)

    # check that the output dataset is saved to the output dir
    # filenames have been updated to L2a. 
    output_files = [os.path.join(outputdir, frame.filename.replace("_l1_", "_l2a")) for frame in l1_dataset]
    output_dataset = data.Dataset(output_files)
    assert len(output_dataset) == len(l1_dataset) # check the same number of files
    
    # check that the recipe is saved into the header with specified calibrations
    new_recipe = json.loads(output_dataset[0].ext_hdr['RECIPE'])
    assert recipe['steps'][2]['calibs']['NonLinearityCalibration'] != 'AUTOMATIC' 


    #### Test cases where JIT should be enabled or not
    # already tested pipeline setting True, nothing set in recipe. Resulted in keeping automatic keyword

    # pipeline setting false, and nothing set in recipe. Should define calibrations
    corgidrp.jit_calib_id = False
    template_recipe = json.load(open(template_filepath, "r"))
    recipe = walker.autogen_recipe(filelist, outputdir, template=template_recipe)
    assert recipe['steps'][2]['calibs']['NonLinearityCalibration'] != 'AUTOMATIC' 

    # pipeline setting false, but recipe says JIT. Should keep automatic calibration
    corgidrp.jit_calib_id = False
    template_recipe = json.load(open(template_filepath, "r"))
    template_recipe['drpconfig']['jit_calib_id'] = True
    recipe = walker.autogen_recipe(filelist, outputdir, template=template_recipe)
    assert recipe['steps'][2]['calibs']['NonLinearityCalibration'] == 'AUTOMATIC' 

    # pipeline setting false, and recipe says no JIT. Should define calibrations
    corgidrp.jit_calib_id = False
    template_recipe = json.load(open(template_filepath, "r"))
    template_recipe['drpconfig']['jit_calib_id'] = False
    recipe = walker.autogen_recipe(filelist, outputdir, template=template_recipe)
    assert recipe['steps'][2]['calibs']['NonLinearityCalibration'] != 'AUTOMATIC' 

    # pipeline setting True, and recipe says no JIT. Should define calibrations
    corgidrp.jit_calib_id = True
    template_recipe = json.load(open(template_filepath, "r"))
    template_recipe['drpconfig']['jit_calib_id'] = False
    recipe = walker.autogen_recipe(filelist, outputdir, template=template_recipe)
    assert recipe['steps'][2]['calibs']['NonLinearityCalibration'] != 'AUTOMATIC' 

    # pipeline setting True, and recipe says JIT. Should keep automatic calibration
    corgidrp.jit_calib_id = True
    template_recipe = json.load(open(template_filepath, "r"))
    template_recipe['drpconfig']['jit_calib_id'] = True
    recipe = walker.autogen_recipe(filelist, outputdir, template=template_recipe)
    assert recipe['steps'][2]['calibs']['NonLinearityCalibration'] == 'AUTOMATIC' 


    # clean up
    mycaldb.remove_entry(new_nonlinearity)
    mycaldb.remove_entry(new_kgain)

    corgidrp.jit_calib_id = old_setting



def test_generate_multiple_recipes():
    """
    Tests that we can generate multiple recipes when passing in a dataset
    """
    # create dirs
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    outputdir = os.path.join(os.path.dirname(__file__), "walker_output")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    # Make a non-linearity correction calibration file
    input_non_linearity_filename = "nonlin_table_TVAC.txt"
    test_non_linearity_filename = input_non_linearity_filename.split(".")[0] + ".fits"
    test_non_linearity_path = os.path.join(os.path.dirname(__file__), "test_data", test_non_linearity_filename)

    dataset = mocks.create_nonlinear_dataset(test_non_linearity_path, filedir=datadir)
    # add vistype
    for frame in dataset:
        frame.pri_hdr['VISTYPE'] = "PUPILIMG"
    dataset.save()
    filelist = [frame.filepath for frame in dataset]

    recipes = walker.autogen_recipe(filelist, outputdir)

    assert len(recipes) == 2




if __name__ == "__main__":#
    test_autoreducing()
    test_auto_template_identification()
    test_saving()
    test_skip_missing_calib()
    test_skip_missing_optional_calib()
    test_jit_calibs()
    test_generate_multiple_recipes()



