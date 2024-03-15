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
    recipe = walker.autogen_reicpe(filelist, outputdir)
    # save recipe
    recipe_path = os.path.join(outputdir, "my_l1_to_l2b.json")
    with open(recipe_path, "w") as json_file:
        json.dump(recipe, json_file, indent=4)

    # process_recipe
    walker.run_recipe(recipe)


if __name__ == "__main__":
    test_autoreducing()



