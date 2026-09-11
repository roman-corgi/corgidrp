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
from corgidrp import spec
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
    
def l2b_to_slittrans(e2eoutput_path):    
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_spec_slittrans_e2e")
    l2b_outputdir = os.path.join(test_outputdir, "l2b_results")
    l2b_filelist = sorted(
        os.path.join(l2b_outputdir, f)
        for f in os.listdir(l2b_outputdir) if f.endswith('_l2b.fits')
    )
    
    # ------------------------------------------------------------------ 
    # L2b -> slit transmission                                                        
    # ------------------------------------------------------------------ 
    # first we have to split the dataset to slit and open
    l2b_open_dir = os.path.join(test_outputdir, "l2b_open")
    if not os.path.exists(l2b_open_dir):
        os.mkdir(l2b_open_dir)
    # clean up by removing old files
    for file in os.listdir(l2b_open_dir):
        os.remove(os.path.join(l2b_open_dir, file))
    l2b_slit_dir = os.path.join(test_outputdir, "l2b_slit")
    if not os.path.exists(l2b_slit_dir):
        os.mkdir(l2b_slit_dir)
    # clean up by removing old files
    for file in os.listdir(l2b_slit_dir):
        os.remove(os.path.join(l2b_slit_dir, file))
    
    l3_open_dir = os.path.join(test_outputdir, "l3_open")
    if not os.path.exists(l3_open_dir):
        os.mkdir(l3_open_dir)
    # clean up by removing old files
    for file in os.listdir(l3_open_dir):
        os.remove(os.path.join(l3_open_dir, file))
    l3_slit_dir = os.path.join(test_outputdir, "l3_slit")
    if not os.path.exists(l3_slit_dir):
        os.mkdir(l3_slit_dir)
    # clean up by removing old files
    for file in os.listdir(l3_slit_dir):
        os.remove(os.path.join(l3_slit_dir, file))   
        
    l2b_dataset = data.Dataset(l2b_filelist)
    fsam_dataset, fsam = l2b_dataset.split_dataset(exthdr_keywords=["FSAMNAME"])
    for i in range(len(fsam)):
        if fsam[i] == "OPEN":
            open_dataset = fsam_dataset[i]
            open_dataset.save(filedir = l2b_open_dir)
        else:
            slit_dataset = fsam_dataset[i]
            slit_dataset.save(filedir = l2b_slit_dir) 
    
    l2b_open_filelist = sorted(
        os.path.join(l2b_open_dir, f)
        for f in os.listdir(l2b_open_dir) if f.endswith('_l2b.fits')
    )
    l2b_slit_filelist = sorted(
        os.path.join(l2b_slit_dir, f)
        for f in os.listdir(l2b_slit_dir) if f.endswith('_l2b.fits')
    )
    
    print("Running L2b -> l3 spec open")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l2b_open_filelist, "", l3_open_dir,
                             template="l2b_to_spec_slittrans.json")
    print("Running L2b -> l3 spec slit")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l2b_slit_filelist, "", l3_slit_dir,
                             template="l2b_to_spec_slittrans.json")
    
    l3_slit_list = sorted(
        os.path.join(l3_slit_dir, f)
        for f in os.listdir(l3_slit_dir) if f.endswith('.fits')
    )
    
    l3_open_list = sorted(
        os.path.join(l3_open_dir, f)
        for f in os.listdir(l3_open_dir) if f.endswith('.fits')
    )
    slit_data = data.Dataset(l3_slit_list)
    open_data = data.Dataset(l3_open_list)
    slittrans = spec.slit_transmission(slit_data, open_data, x_range=[39,89], y_range =[65,71])
    print(f"L2b -> SlitTransmission complete.")
    
    slittrans.save(filedir = l3_slit_dir)
    
    ### validate SlitTransmission product 
    #check.compare_to_mocks_hdrs(slittrans_cal_file)

    assert slittrans.ext_hdr["DATATYPE"] == "SlitTransmission"
    assert slittrans.ext_hdr["DATALVL"] == "CAL"
    assert slittrans.ext_hdr['CFAMNAME'] == '3F'
    
    #check the values
    print(slittrans.x_offset)
    print(slittrans.y_offset)
    
    
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
    #test_l1_to_slittrans(e2edata_dir, outputdir)
    l2b_to_slittrans(outputdir)