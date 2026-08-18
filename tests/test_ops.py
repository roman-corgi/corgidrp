import os
import json
import tempfile
import numpy as np
import corgidrp
import astropy.time as time
import corgidrp.data as data
import corgidrp.ops as ops
import corgidrp.caldb as caldb
import corgidrp.mocks as mocks


def _setup_ops_test_data(tmp_path):
    """
    Build the L1 dataset + nonlinearity calibration used by the ops tests.

    Returns:
        tuple: (filelist, main_cal_dir, outputdir, new_nonlinearity)
    """

    # create dirs
    datadir = tmp_path / "simdata"
    datadir.mkdir(parents=True, exist_ok=True)
    outputdir = tmp_path / "ops_output"
    outputdir.mkdir(parents=True, exist_ok=True)
    main_cal_dir = tmp_path / "ops_cal_dir"
    main_cal_dir.mkdir(parents=True, exist_ok=True)

    # create simulated data
    l1_dataset = mocks.create_prescan_files(filedir=str(datadir), arrtype="SCI", numfiles=2)
    # simulate the expected CGI naming convention
    fname_template = "cgi_0200001999001000{:03d}_20250415t0305102_l1_.fits"
    for i, image in enumerate(l1_dataset):
        image.filename = fname_template.format(i)
    l1_dataset.save(filedir=str(datadir))
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
    nonlin_fits_filepath = str(main_cal_dir / test_non_linearity_filename)
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

    return filelist, main_cal_dir, outputdir, new_nonlinearity


def test_ops_produces_expected_file(tmp_path):
    """
    Tests that the ops module produces the expected files. Based on test_autoreducing in test_walker.py
    """
    filelist, main_cal_dir, outputdir, new_nonlinearity = _setup_ops_test_data(tmp_path)

    CPGS_XML_filepath = "" # not yet implemented

    #######################
    ## Test the Ops code ##
    #######################

    #Initialize the caldb and rescan the main_cal_directory
    this_caldb = ops.step_1_initialize()
    ops.step_2_load_cal(this_caldb, main_cal_dir)

    #Process the data. Ops generally won't have a template, but a template-less 
    # test would require generating more calibrations than are necessary for just testing this functionality.
    ops.step_3_process_data(filelist, CPGS_XML_filepath, str(outputdir),template="l1_to_l2a_basic.json")

    #Check that the output files are as expected. 
    output_filelist = [os.path.join(outputdir,os.path.basename(filename).replace("_l1_", "_l2a")) for filename in filelist]
    for output_file in output_filelist:
        assert os.path.exists(output_file), f"Expected output file {output_file} does not exist."

    ### Clean up
    mycaldb = caldb.CalDB()
    mycaldb.remove_entry(new_nonlinearity)


def test_ops_with_user_templates_dir(tmp_path):
    """
    Tests that the ops module works with a custom user_templates_dir argument.
    Same as test_ops_produces_expected_file but with user template override.
    """
    import copy
    import corgidrp.walker as walker

    filelist, main_cal_dir, outputdir, new_nonlinearity = _setup_ops_test_data(tmp_path)

    CPGS_XML_filepath = ""

    with tempfile.TemporaryDirectory() as user_templates_dir:
        # Load default template and modify one keyword
        default_template_path = os.path.join(
            os.path.dirname(walker.__file__), "recipe_templates", "l1_to_l2a_basic.json"
        )
        with open(default_template_path, 'r') as f:
            user_template = json.load(f)

        # Override sat_thresh in detect_cosmic_rays step
        detect_cr_step = next(s for s in user_template['steps'] if s['name'] == 'detect_cosmic_rays')
        detect_cr_step['keywords']['sat_thresh'] = 0.7

        # Write modified template to user templates dir
        template_path = os.path.join(user_templates_dir, "l1_to_l2a_basic.json")
        with open(template_path, 'w') as f:
            json.dump(user_template, f)

        # Run the full ops chain with custom user_templates_dir
        this_caldb = ops.step_1_initialize(user_templates_dir=user_templates_dir)
        ops.step_2_load_cal(this_caldb, main_cal_dir)
        ops.step_3_process_data(filelist, CPGS_XML_filepath, str(outputdir), template="l1_to_l2a_basic.json")

        # Check output files exist
        output_filelist = [os.path.join(outputdir, os.path.basename(filename).replace("_l1_", "_l2a")) for filename in filelist]
        for output_file in output_filelist:
            assert os.path.exists(output_file), f"Expected output file {output_file} does not exist."

        # Verify RECIPE_SRC traces to user template directory
        output_dataset = data.Dataset(output_filelist)
        for frame in output_dataset:
            recipe = json.loads(frame.ext_hdr["RECIPE"])
            assert "RECIPE_SRC" in recipe
            assert user_templates_dir in recipe["RECIPE_SRC"]
            # Verify user-modified keyword propagated
            recipe_cr_step = next(s for s in recipe['steps'] if s['name'] == 'detect_cosmic_rays')
            assert recipe_cr_step['keywords']['sat_thresh'] == 0.7

    ### Clean up
    mycaldb = caldb.CalDB()
    mycaldb.remove_entry(new_nonlinearity)


if __name__ == "__main__":#
    test_ops_produces_expected_file()
    test_ops_with_user_templates_dir()