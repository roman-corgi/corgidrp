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

nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
flat_path = os.path.join(processed_cal_path, "flat.fits")
fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
bp_path = os.path.join(processed_cal_path, "bad_pix.fits")

l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]]
mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]]
tvac_l2b_filelist = [os.path.join(l2b_datadir, "{0}.fits".format(i)) for i in [90529, 90531]]

pri_hdr, ext_hdr = mocks.create_default_headers()
ext_hdr["DRPCTIME"] = time.Time.now().isot
ext_hdr['DRPVERSN'] = corgidrp.__version__
mock_input_dataset = data.Dataset(mock_cal_filelist)

this_caldb = caldb.CalDB()  # Connection to the calibration database