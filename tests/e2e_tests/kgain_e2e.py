import argparse
import os, shutil
import glob
import pytest
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.walker as walker

thisfile_dir = os.path.dirname(__file__) # this file's folder

tvac_kgain = 8.8145 #e/DN, result from new iit drp code
tvac_readnoise = 130.12 #e, result from new iit drp code

@pytest.mark.e2e
def test_l1_to_kgain(tvacdata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(tvacdata_path, "TV-20_EXCAM_noise_characterization", "kgain")

    # make output directory if needed
    kgain_outputdir = os.path.join(e2eoutput_path, "l1_to_kgain_output")
    if os.path.exists(kgain_outputdir):
        shutil.rmtree(kgain_outputdir)
    os.mkdir(kgain_outputdir)

    # define the raw science data to process

    l1_data_filelist_same_exp = [os.path.join(l1_datadir, "CGI_EXCAM_L1_00000{0}.fits".format(i)) for i in np.arange(51841,51871)]#[51841, 51851]] # just grab some files of it
    l1_data_filelist_range_exp = [os.path.join(l1_datadir, "CGI_EXCAM_L1_00000{0}.fits".format(i)) for i in np.arange(51731, 51841)]#[51731, 51761]]
    
    ####### Run the walker on some test_data

    for file in l1_data_filelist_same_exp:
        image = data.Image(file)
        # This should not be necessary anymore after the updates of the OBSTYPE keyword, up to now it is only "SCI"
        if image.pri_hdr['OBSTYPE'] != 'MNFRAME':
            image.pri_hdr['OBSTYPE'] = 'MNFRAME'
            image.save(filename = file)
        l1_data_filelist_range_exp.append(file)

    walker.walk_corgidrp(l1_data_filelist_range_exp, "", kgain_outputdir, template="l1_to_kgain.json")
    kgain_file = os.path.join(kgain_outputdir, "CGI_EXCAM_L1_0000051731_kgain.fits")
    kgain = data.KGain(kgain_file)
    
    ##### Check against TVAC kgain, readnoise
    new_kgain = kgain.value
    new_readnoise = kgain.ext_hdr["RN"]
    print("determined kgain:", new_kgain)
    print("determined read noise", new_readnoise)    
    
    diff_kgain = new_kgain - tvac_kgain
    diff_readnoise = new_readnoise - tvac_readnoise
    print ("difference to TVAC kgain:", diff_kgain)
    print ("difference to TVAC read noise:", diff_readnoise)
    print ("error of kgain:", kgain.error)
    print ("error of readnoise:", kgain.ext_hdr["RN_ERR"])

    assert np.abs(diff_kgain) < 0.01
    assert np.abs(diff_readnoise) < 3

    
if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    tvacdata_dir = "/home/schreiber/DataCopy/corgi/CGI_TVAC_Data/"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->kgain end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    test_l1_to_kgain(tvacdata_dir, outputdir)
