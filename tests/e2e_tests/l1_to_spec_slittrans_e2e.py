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
    
    # ------------------------------------------------------------------ 
    # L2b -> slit transmission                                                        
    # ------------------------------------------------------------------ 
    
    l3_out_dir = os.path.join(test_outputdir, "l3_out")
    if not os.path.exists(l3_out_dir):
        os.mkdir(l3_out_dir)
    # clean up by removing old files
    for file in os.listdir(l3_out_dir):
        os.remove(os.path.join(l3_out_dir, file))
        
    l2b_dataset = data.Dataset(l2b_filelist)
    
    print("Running L2b -> slit transmission")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        walker.walk_corgidrp(l2b_filelist, "", l3_out_dir,
                             template="l2b_to_spec_slittrans.json")
    
    ####### Load in the output data. It should be the latest slit transmission calibration file produced.
    slittrans_file = glob.glob(os.path.join(l3_out_dir, '*slt_cal*.fits'))[0]
    slittrans = data.SlitTransmission(slittrans_file)
    # Remove temporary CalDB
    if os.path.exists(tmp_caldb_csv):
        os.remove(tmp_caldb_csv)
    
    ### validate SlitTransmission product 
    check.compare_to_mocks_hdrs(slittrans_file)

    assert slittrans.ext_hdr["DATATYPE"] == "SlitTransmission"
    assert slittrans.ext_hdr["DATALVL"] == "CAL"
    assert slittrans.ext_hdr['CFAMNAME'] == '3F'
    assert slittrans.slitname == "R1C2"
    
    #check the values
    assert len(slittrans.x_offset) == 100
    assert len(slittrans.y_offset) == 100
    assert 39 <= np.min(slittrans.x_offset) 
    assert 89 >= np.max(slittrans.x_offset)
    assert 65 <= np.min(slittrans.y_offset) 
    assert 71 >= np.max(slittrans.y_offset)
    assert np.shape(slittrans.data) == (100, 51)
    print("mean value of the slit transmission:",np.mean(slittrans.data))
    
    #the values at the edge of the slit should be smaller than around the center
    assert np.mean(slittrans.data[0:10,25]) < np.mean(slittrans.data[40:50,25])
    assert np.mean(slittrans.data[90:100,25]) < np.mean(slittrans.data[40:50,25])
    l3_list = sorted(
        os.path.join(l3_out_dir, f)
        for f in os.listdir(l3_out_dir) if f.endswith('.fits')
    )
    
    im_slit = data.Image(l3_list[0]).data
    im_open = data.Image(l3_list[-1]).data
    
    x_max_slit = int(np.median(np.argmax(im_slit[60:80,:], axis = 1)))
    x_max_open = int(np.median(np.argmax(im_open[60:80,:], axis = 1)))
    #estimate the throughput of the slit due to the slit width of about 6 pixels in band 3 using a slitless measurement
    est_trans_open = np.sum(im_open[60:80,x_max_open -3:x_max_open+3])/np.sum(im_open[60:80,x_max_open -30:x_max_open+30])
    assert np.mean(slittrans.data[40:50, 25]) == pytest.approx(est_trans_open, abs = 0.08)
    #estimate the ratio of a corresponding slit and slitless measurement at the same position
    est_trans_slit = np.sum(im_slit[60:80,x_max_slit -15:x_max_slit+15])/np.sum(im_open[60:80,x_max_open -15:x_max_open+15])
    assert np.mean(slittrans.data[40:50, 25]) == pytest.approx(est_trans_slit, abs = 0.05)
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