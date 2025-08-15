import os
import numpy as np
import corgidrp
import astropy.time as time
import corgidrp.data as data
import corgidrp.ops as ops
import corgidrp.caldb as caldb
import corgidrp.mocks as mocks



def test_ops_produces_expected_file(): 
    """
    Tests that the ops module produces the expected files. Based on test_autoreducingin test_walker.py
    """

    # create dirs
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    outputdir = os.path.join(os.path.dirname(__file__), "ops_output")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    main_cal_dir = os.path.join(os.path.dirname(__file__), "ops_cal_dir")
    if not os.path.exists(main_cal_dir):
        os.mkdir(main_cal_dir)

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
    ###########################################
    #Create a mock dataset because it is a required input when creating a NonLinearityCalibration
    dummy_dataset = mocks.create_prescan_files()

    # Make a non-linearity correction calibration file
    input_non_linearity_filename = "nonlin_table_TVAC.txt"
    input_non_linearity_path = os.path.join(os.path.dirname(__file__), "test_data", input_non_linearity_filename)
    test_non_linearity_filename = input_non_linearity_filename.split(".")[0] + ".fits"
    nonlin_fits_filepath = os.path.join(os.path.dirname(__file__), main_cal_dir, test_non_linearity_filename)
    tvac_nonlin_data = np.genfromtxt(input_non_linearity_path, delimiter=",")

    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
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

    CPGS_XML_filepath = "" # not yet implemented

    #######################
    ## Test the Ops code ##
    #######################

    #Initialize the caldb and rescan the main_cal_directory
    this_caldb = ops.step_1_initialize()
    ops.step_2_load_cal(this_caldb, main_cal_dir)

    #Process the data. Ops generally won't have a template, but a template-less 
    # test would require generating more calibrations than are necessary for just testing this functionality.
    ops.step_3_process_data(filelist, CPGS_XML_filepath, outputdir,template="l1_to_l2a_basic.json")

    #Check that the output files are as expected. 
    output_filelist = [os.path.join(outputdir,os.path.basename(filename).replace("_l1_", "_l2a")) for filename in filelist]
    for output_file in output_filelist:
        assert os.path.exists(output_file), f"Expected output file {output_file} does not exist."

    ### Clean up
    mycaldb.remove_entry(new_nonlinearity)

if __name__ == "__main__":#
    test_ops_produces_expected_file()