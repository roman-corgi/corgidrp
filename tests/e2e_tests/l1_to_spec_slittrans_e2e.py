# Flux calibration E2E Test Code

import argparse
import os, shutil
import warnings
import glob
import pytest
import numpy as np
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.detector as detector
from corgidrp import caldb
from corgidrp import check
import astropy.time as time
import astropy.io.fits as fits
from corgidrp.darks import build_synthesized_dark


@pytest.mark.e2e
def test_l1_to_slittrans(e2edata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(e2edata_path, "slit_trans_simdata")

    # make output directory if needed
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_spec_slittrans_e2e")
    if os.path.exists(test_outputdir):
        shutil.rmtree(test_outputdir)
    os.makedirs(test_outputdir)

    l2b_outputdir = os.path.join(test_outputdir, "l2b_results")
    if not os.path.exists(l2b_outputdir):
        os.mkdir(l2b_outputdir)

    # clean up by removing old files
    for file in os.listdir(l2b_outputdir):
        os.remove(os.path.join(l2b_outputdir, file))
    
    # Use a temporary CSV to avoid issues with real CalDB
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_slittrans_e2e_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()

    # Get default spectroscopy calibrations 
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    print(f"Loaded default calibrations from {corgidrp.default_cal_dir}")
    
    l1_data_filelist=[os.path.join(l1_datadir, os.listdir(l1_datadir)[i]) for i in range(len(os.listdir(l1_datadir))) if os.listdir(l1_datadir)[i].endswith("l1_.fits")]
    
    ####### Run the walker on some test_data
    # ------------------------------------------------------------------ 
    # L1 -> L2a                                                        
    # ------------------------------------------------------------------ 
    print("Running L1 -> L2a …")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l1_data_filelist, "", l2b_outputdir,
                             template="l1_to_l2a_basic.json")

    l2a_filelist = sorted(
        os.path.join(l2b_outputdir, f)
        for f in os.listdir(l2b_outputdir) if f.endswith('_l2a.fits')
    )
    print(f"L1 -> L2a complete: {len(l2a_filelist)} L2a files produced.")
    
    # ------------------------------------------------------------------ 
    # L2a -> L2b                                                        
    # ------------------------------------------------------------------ 
    print("Running L2a -> L2b …")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l2a_filelist, "", l2b_outputdir,
                             template="l2a_to_l2b_spec.json")

    l2b_filelist = sorted(
        os.path.join(l2b_outputdir, f)
        for f in os.listdir(l2b_outputdir) if f.endswith('_l2b.fits')
    )
    print(f"L2a -> L2b complete: {len(l2b_filelist)} L2b files produced.")
    
        # ------------------------------------------------------------------ 
    # L2b -> spec dispersion                                                        
    # ------------------------------------------------------------------ 
    print("Running L2b -> SlitTransmission …")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l2b_filelist, "", l2b_outputdir,
                             template="l2b_to_spec_slittrans.json")

    print(f"L2b -> SlitTransmission complete.")
    
    ####### Load in the output data. It should be the latest slit transmission calibration file produced.
    slittrans_cal_file = glob.glob(os.path.join(l2b_outputdir, '*slt_cal*.fits'))[0]
    slittrans = data.SlitTransmission(slittrans_cal_file)
    
    ### validate SlitTransmission product 
    check.compare_to_mocks_hdrs(slittrans_cal_file)

    assert slittrans.ext_hdr["DATATYPE"] == "SlitTransmission"
    assert slittrans.ext_hdr["DATALVL"] == "CAL"
    assert slittrans.ext_hdr['BAND'] == '3'
    assert slittrans.ext_hdr['REFWAVE'] == 730
    
    #check the values
    
    
    
    
    # Remove temporary CalDB
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    # Print success message
    print('e2e test for slit transmission calibration passed')
    
if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    e2edata_dir = '/home/schreiber/DataCopy/E2E_Test_Data'
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1-> Slit Transmission end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    e2edata_dir = args.e2edata_dir
    test_l1_to_slittrans(e2edata_dir, outputdir)
