"""
Test the walker infrastructure to read and execute recipes
"""
import os
import glob
import json
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.caldb as caldb
import corgidrp.walker as walker
import corgidrp.detector as detector


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
    l1_dataset = mocks.create_prescan_files(filedir=datadir, obstype="SCI", numfiles=2)
    # fake the emgain
    for image in l1_dataset:
        image.ext_hdr['EMGAIN'] = 1
    # simulate the expected CGI naming convention
    fname_template = "CGI_L1_100_0200001001001100001_20270101T120000_{0:03d}.fits"
    for i, image in enumerate(l1_dataset):
        image.filename = fname_template.format(i)
    l1_dataset.save(datadir)
    filelist = [frame.filepath for frame in l1_dataset]


    # index the sample nonlinearity correction that we need for processing
    new_nonlinearity = data.NonLinearityCalibration(os.path.join(os.path.dirname(__file__),"test_data",'nonlin_sample.fits'))
    # fake the headers because this frame doesn't have the proper headers
    prihdr, exthdr = mocks.create_default_headers("SCI")
    new_nonlinearity.pri_hdr = prihdr
    new_nonlinearity.ext_hdr = exthdr
    new_nonlinearity.ext_hdr.set('DRPCTIME', time.Time.now().isot, "When this file was saved")
    new_nonlinearity.ext_hdr.set('DRPVERSN', corgidrp.__version__, "corgidrp version that produced this file")
    mycaldb = caldb.CalDB()
    mycaldb.create_entry(new_nonlinearity)

    CPGS_XML_filepath = "" # not yet implemented

    # generate recipe and run it
    # basic l2a recipe to keep things simple
    recipe = walker.walk_corgidrp(filelist, CPGS_XML_filepath, outputdir, template="l1_to_l2a_basic.json")

    # check that the output dataset is saved to the output dir
    # filenames have been updated to L2a. 
    output_files = [os.path.join(outputdir, frame.filename.replace("_L1_", "_L2a_")) for frame in l1_dataset]
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
    l1_dataset = mocks.create_prescan_files(filedir=datadir, obstype="SCI", numfiles=2)
    # fake the emgain
    for image in l1_dataset:
        image.ext_hdr['EMGAIN'] = 1
    # simulate the expected CGI naming convention
    fname_template = "CGI_L1_100_0200001001001100001_20270101T120000_{0:03d}.fits"
    for i, image in enumerate(l1_dataset):
        image.filename = fname_template.format(i)
    l1_dataset.save(datadir)
    filelist = [frame.filepath for frame in l1_dataset]


    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make calibration files using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(filelist)

    this_caldb = caldb.CalDB() # connection to cal DB

    # Nonlinearity calibration
    new_nonlinearity = data.NonLinearityCalibration(os.path.join(os.path.dirname(__file__),"test_data",'nonlin_sample.fits'))
    # fake the headers because this frame doesn't have the proper headers
    new_nonlinearity.pri_hdr = pri_hdr
    new_nonlinearity.ext_hdr = ext_hdr
    this_caldb.create_entry(new_nonlinearity)

    # KGain
    kgain_val = 8.7
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)

    # NoiseMap
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected EM electrons'
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

    assert recipe['name'] == 'l1_to_l2b'
    assert recipe['template'] == False

    # now cleanup
    this_caldb.remove_entry(new_nonlinearity)
    this_caldb.remove_entry(kgain)
    this_caldb.remove_entry(noise_map)
    this_caldb.remove_entry(flat)
    this_caldb.remove_entry(bp_map)

if __name__ == "__main__":
    test_autoreducing()



