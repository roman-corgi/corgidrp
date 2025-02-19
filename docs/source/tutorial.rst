Tests 
======

This section demonstrates how to run an end-to-end test for processing Level 1 (L1) data into Level 2b (L2b) data using the `corgidrp` package.

The test involves processing raw L1 data through calibration procedures and comparing the results against known TVAC L2b data. It also ensures calibration file creation and removal from the calibration database.
Before running the test, ensure you have set up the necessary paths to your data and calibration files. The following example demonstrates the steps involved in processing the data and running the test.

.. contents:: Table of Contents
   :depth: 2
   :local:
   
Test Setup
----------

The script imports necessary libraries and determines the directory containing the script.

.. code-block:: python

    import argparse
    import os
    import pytest
    import numpy as np
    import astropy.time as time
    import astropy.io.fits as fits
    import corgidrp
    import corgidrp.data as data
    import corgidrp.mocks as mocks
    import corgidrp.walker as walker
    import corgidrp.caldb as caldb
    import corgidrp.detector as detector

    thisfile_dir = os.path.dirname(__file__)  # this file's folder

The `test_l1_to_l2b` function initializes paths for input data, calibration files, and output directories.

.. code-block:: python

    @pytest.mark.e2e
    def test_l1_to_l2b(tvacdata_path, e2eoutput_path):
        # Define input data paths
        l1_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L1")
        l2b_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L2b")
        processed_cal_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals")

        # Create output directory if it does not exist
        l2b_outputdir = os.path.join(e2eoutput_path, "l1_to_l2b_output")
        if not os.path.exists(l2b_outputdir):
            os.mkdir(l2b_outputdir)

Calibration files such as non-linearity tables, dark current, flat fields, and bad pixel maps are loaded.

.. code-block:: python

        nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
        dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
        flat_path = os.path.join(processed_cal_path, "flat.fits")
        fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
        cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
        bp_path = os.path.join(processed_cal_path, "bad_pix.fits")

The raw science data files and mock calibration files are defined.

.. code-block:: python

        l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]]
        mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]]
        tvac_l2b_filelist = [os.path.join(l2b_datadir, "{0}.fits".format(i)) for i in [90529, 90531]]

The calibration database (`caldb`) is initialized, and calibration entries are created.

.. code-block:: python

        pri_hdr, ext_hdr = mocks.create_default_headers()
        ext_hdr["DRPCTIME"] = time.Time.now().isot
        ext_hdr['DRPVERSN'] = corgidrp.__version__
        mock_input_dataset = data.Dataset(mock_cal_filelist)

        this_caldb = caldb.CalDB()  # Connection to the calibration database

Non-linearity calibration, KGain, noise maps, flat field, and bad pixel maps are generated and stored in the calibration database.

.. code-block:: python

        # Nonlinearity calibration
        nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
        nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                     input_dataset=mock_input_dataset)
        nonlinear_cal.save(filedir=l2b_outputdir, filename="mock_nonlinearcal.fits")
        this_caldb.create_entry(nonlinear_cal)

        # KGain
        kgain_val = 8.7
        kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                           input_dataset=mock_input_dataset)
        kgain.save(filedir=l2b_outputdir, filename="mock_kgain.fits")
        this_caldb.create_entry(kgain)

The `walker.walk_corgidrp` function processes the L1 data.

.. code-block:: python

        walker.walk_corgidrp(l1_data_filelist, "", l2b_outputdir)

Calibration entries are removed from the database.

.. code-block:: python

        this_caldb.remove_entry(nonlinear_cal)
        this_caldb.remove_entry(kgain)

The processed L2b data is compared against TVAC data to verify correctness.

.. code-block:: python

        new_l2b_filenames = [os.path.join(l2b_outputdir, "{0}.fits".format(i)) for i in [90499, 90500]]

        for new_filename, tvac_filename in zip(new_l2b_filenames, tvac_l2b_filelist):
            img = data.Image(new_filename)

            with fits.open(tvac_filename) as hdulist:
                tvac_dat = hdulist[1].data

            diff = img.data - tvac_dat

            assert np.all(np.abs(diff) < 1e-5)

The test can be run using command-line arguments.

.. code-block:: python

    if __name__ == "__main__":
        tvacdata_dir = "/path/to/CGI_TVAC_Data/"
        outputdir = thisfile_dir

        ap = argparse.ArgumentParser(description="Run the L1->L2b end-to-end test")
        ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                        help="Path to CGI_TVAC_Data Folder [%(default)s]")
        ap.add_argument("-o", "--outputdir", default=outputdir,
                        help="Directory to write results to [%(default)s]")
        args = ap.parse_args()
        tvacdata_dir = args.tvacdata_dir
        outputdir = args.outputdir
        test_l1_to_l2b(tvacdata_dir, outputdir)

This script ensures that the `corgidrp` pipeline correctly processes L1 data into L2b. The validation step confirms that the output matches expected results, ensuring data integrity.
