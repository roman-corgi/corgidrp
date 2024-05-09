"""
Test the walker infrastructure to read and execute recipes
"""
import os
import glob
import json
import astropy.time as time
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.caldb as caldb
import corgidrp.walker as walker


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
    l1_dataset.save(datadir)
    filelist = [frame.filepath for frame in l1_dataset]


    # index the sample nonlinearity correction that we need for processing
    new_nonlinearity = data.NonLinearityCalibration(os.path.join(os.path.dirname(__file__),"test_data",'nonlin_sample.fits'))
    # fake the headers because this frame doesn't have the proper headers
    prihdr, exthdr = mocks.create_default_headers("SCI")
    new_nonlinearity.pri_hdr = prihdr
    new_nonlinearity.ext_hdr = exthdr
    new_nonlinearity.ext_hdr.set('DRPCTIME', time.Time.now().isot, "When this file was saved")
    new_nonlinearity.ext_hdr.set('DRPVERSN', corgidrp.version, "corgidrp version that produced this file")
    mycaldb = caldb.CalDB()
    mycaldb.create_entry(new_nonlinearity)

    # generate recipe
    recipe = walker.autogen_recipe(filelist, outputdir)

    # process_recipe
    walker.run_recipe(recipe)

    # check that the output dataset is saved to the output dir with the same filename as the input filenames
    output_files = [os.path.join(outputdir, frame.filename) for frame in l1_dataset]
    output_dataset = data.Dataset(output_files)
    assert len(output_dataset) == len(l1_dataset) # check the same number of files
    # check that the recipe is saved into the header.
    for frame in output_dataset:
        assert "RECIPE" in frame.ext_hdr
        # test recipe was correctly written into the header
        # do a string comparison, easiest way to check
        hdr_recipe = json.loads(frame.ext_hdr["RECIPE"])
        assert json.dumps(hdr_recipe) == json.dumps(recipe)
if __name__ == "__main__":
    test_autoreducing()



