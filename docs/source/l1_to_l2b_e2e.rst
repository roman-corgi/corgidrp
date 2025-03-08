Process L1 to L2b data
-----------------------

This section demonstrates how process Level 1 (L1) data into Level 2b (L2b) data using the ``corgidrp`` package.

The focus is on explaining the key steps involved in processing data, including setting up calibration files, running the data processing pipeline, and understanding the core steps for transforming raw data into processed output. The test can be run by processing raw L1 data into L2b data using the ``walker.walk_corgidrp`` function. This function is the core of the pipeline and is responsible for the actual data processing. Here is how you can set up and run the test:

Calibration Files
~~~~~~~~~~~~~~~~~

To process the L1 data, you will need several calibration files. These files are used in the calibration process to adjust the raw data accordingly. The following calibration files are required:

    - Noise Maps: Used to account for noise in the sensor data.
    - Non-Linearity Calibration: Corrects for any non-linearities in the sensor's response.
    - Dark Current Calibration: Accounts for any dark current noise in the sensor.
    - Flat Field Calibration: Used to correct for uneven sensitivity across the detector.
    - Bad Pixel Maps: Identifies pixels that are defective or not functioning properly.


Test Setup
~~~~~~~~~~

Before running the test, ensure you have the necessary datasets and calibration files. You will need:

1. **L1 Data** - Raw Level 1 data files.
2. **Cals** - Calibration files required for processing as outlined above. 

The test script is written to process the L1 data, apply the calibration files, and produce the output in the L2b format.

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

The calibration database (`caldb`) is initialized, and calibration entries are created. The following is mocking code for squeezing the test data in the official format. 

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

The ``walker.walk_corgidrp`` function is the main part of the pipeline responsible for transforming the raw L1 data into L2b data. This function applies all necessary calibration steps and generates the output files.

The processed L2b data is compared against TVAC data to verify correctness.

.. code-block:: python

        new_l2b_filenames = [os.path.join(l2b_outputdir, "{0}.fits".format(i)) for i in [90499, 90500]]

        for new_filename, tvac_filename in zip(new_l2b_filenames, tvac_l2b_filelist):
            img = data.Image(new_filename)

            with fits.open(tvac_filename) as hdulist:
                tvac_dat = hdulist[1].data

            diff = img.data - tvac_dat

            assert np.all(np.abs(diff) < 1e-5)

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

There are several ways to run test:

1. Using pytest

.. code-block:: python
        
        # From the root directory of corgidrp
        pytest tests/test_l1_to_l2b.py -v

2. Direct execution

.. code-block:: python

         # Run the script directly with default paths
        python tests/test_l1_to_l2b.py
        # Or specify custom paths
        python tests/test_l1_to_l2b.py --tvacdata_dir /path/to/CGI_TVAC_Data --outputdir /path/to/output


Output
~~~~~~

Once the test has been successfully run, the results will be stored in the output directory you specified. To view and analyze the output data, you will need to use a suitable image viewer, such as **SAOImageDS9**.

To analyze the output FITS files:

1. Load your processed L2b files in DS9:
   ``saoimageds9 90500.fits``

2. Quick analysis steps:

   - Press 's' for scale menu (zscale recommended)
   - Press 'c' for colormap options (heat shows features well)
   - Use Analysis -> Statistics to verify calibration values

For more information on using DS9, including detailed tutorials on viewing and manipulating FITS images, check the `official DS9 documentation <https://sites.google.com/cfa.harvard.edu/saoimageds9/documentation>`_

Here is an example of the output:


.. figure:: /_static/Output.png
   :width: 600px
   :align: center
   
   Sample L2b processed image "90500.fits"