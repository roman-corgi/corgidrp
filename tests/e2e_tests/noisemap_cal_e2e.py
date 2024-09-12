import argparse
import os
import pytest
import numpy as np
import astropy.time as time
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector
from glob import glob

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_noisemap_calibration_from_l1(tvacdata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location as in the TVAC Box drive
    l1_datadir = os.path.join(tvacdata_path, "TV-20_EXCAM_noise_characterization", "noisemap_test_data", "test_l1_data")
    iit_noisemap_datadir = os.path.join(tvacdata_path, "TV-20_EXCAM_noise_characterization", "noisemap_test_data")
    processed_cal_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals")

    # make output directory if needed
    noisemap_outputdir = os.path.join(e2eoutput_path, "noisemap_output")
    if not os.path.exists(noisemap_outputdir):
        os.mkdir(noisemap_outputdir)

    # assume all cals are in the same directory
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    
    # define the raw science data to process
    l1_data_filelist = sorted(glob(os.path.join(l1_datadir,"*.fits")))
    mock_cal_filelist = l1_data_filelist [-2:] # grab the last two input data to mock the calibration 
    
    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make calibration files using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    this_caldb = caldb.CalDB() # connection to cal DB

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=noisemap_outputdir, filename="mock_nonlinearcal.fits" )
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    kgain_val = 8.7 
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=noisemap_outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)


    ####### Run the walker on some test_data

    template = "l1_to_l2a_noisemap.json"
    walker.walk_corgidrp(l1_data_filelist, "", noisemap_outputdir,template=template)


    # clean up by removing entry
    this_caldb.remove_entry(nonlinear_cal)
    this_caldb.remove_entry(kgain)

    ##### Check against TVAC data
    corgidrp_noisemap_fname = os.path.join(noisemap_outputdir,"CGI_EXCAM_L2a_0000050775_DetectorNoiseMaps.fits")
    iit_noisemap_fname = os.path.join(iit_noisemap_datadir,"iit_test_noisemaps.fits")

    corgidrp_noisemap = data.autoload(corgidrp_noisemap_fname)
    iit_noisemap = data.autoload(iit_noisemap_fname)
    
    for noise_ext in ["FPN_map","CIC_map","DC_map"]:

        corgi_dat = detector.imaging_slice('SCI', corgidrp_noisemap.__dict__[noise_ext])
        iit_dat = detector.imaging_slice('SCI', iit_noisemap.__dict__[noise_ext])

        diff = corgi_dat - iit_dat

        # # Plot for debugging:
        # import matplotlib.pyplot as plt
        # vmaxes = {
        #     "FPN_map" : 20,
        #     "CIC_map" : 0.5,
        #     "DC_map"  : 0.5
        # }

        # _,axes = plt.subplots(1,3,figsize=(16,4))

    
        # dataset1_comparison_value = np.nanmean(corgi_dat)
        # dataset2_comparison_value = np.nanmean(iit_dat)
        # diff_comparison_value = np.nanmean(diff)
        
        # im = axes[0].imshow(corgi_dat,origin='lower',vmax=vmaxes[noise_ext],vmin=0)
        # plt.colorbar(im, ax=axes[0])
        # axes[0].set_title("CorgiDRP(mean={:.2E})".format(dataset1_comparison_value))

        # im = axes[1].imshow(iit_dat,origin='lower',vmax=vmaxes[noise_ext],vmin=0)
        # plt.colorbar(im, ax=axes[1])
        # axes[1].set_title("II&T(mean={:.2E})".format(dataset2_comparison_value))

        # im = axes[2].imshow(diff,origin='lower',vmax=1e-5,vmin=-1e-5,cmap='seismic')
        # plt.colorbar(im, ax=axes[2])
        # axes[2].set_title("Difference(mean={:.2E})".format(diff_comparison_value))

        # plt.suptitle(noise_ext[:-4])
        # plt.tight_layout()
        # plt.savefig(os.path.join(noisemap_outputdir,f"CorgiDRP_TVAC_Comparison_l1_to_l2a_{noise_ext}.pdf"))
        # plt.close()

        assert np.all(np.abs(diff) < 1e-5)

@pytest.mark.e2e
def test_noisemap_calibration_from_l2a(tvacdata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location as in the TVAC Box drive
    l1_datadir = os.path.join(tvacdata_path, "TV-20_EXCAM_noise_characterization", "noisemap_test_data", "test_l1_data")
    l2a_datadir = os.path.join(tvacdata_path, "TV-20_EXCAM_noise_characterization", "noisemap_test_data", "test_l2a_data")
    iit_noisemap_datadir = os.path.join(tvacdata_path, "TV-20_EXCAM_noise_characterization", "noisemap_test_data")
    
    # make output directory if needed
    noisemap_outputdir = os.path.join(e2eoutput_path, "noisemap_output")
    if not os.path.exists(noisemap_outputdir):
        os.mkdir(noisemap_outputdir)

    # define the raw science data to process
    l1_data_filelist = sorted(glob(os.path.join(l1_datadir,"*.fits")))
    l2a_data_filelist = sorted(glob(os.path.join(l2a_datadir,"*.fits")))
    mock_cal_filelist = l1_data_filelist [-2:] # grab the last two input data to mock the calibration 
    
    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make calibration files using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    this_caldb = caldb.CalDB() # connection to cal DB

    # KGain
    kgain_val = 8.7
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=noisemap_outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)


    ####### Run the walker on some test_data

    template = "l2a_to_l2a_noisemap.json"
    walker.walk_corgidrp(l2a_data_filelist, "", noisemap_outputdir,template=template)


    # clean up by removing entry
    this_caldb.remove_entry(kgain)

    ##### Check against TVAC data
    corgidrp_noisemap_fname = os.path.join(noisemap_outputdir,"CGI_EXCAM_L2a_0000050775_DetectorNoiseMaps.fits")
    iit_noisemap_fname = os.path.join(iit_noisemap_datadir,"iit_test_noisemaps.fits")

    corgidrp_noisemap = data.autoload(corgidrp_noisemap_fname)
    iit_noisemap = data.autoload(iit_noisemap_fname)
    
    for noise_ext in ["FPN_map","CIC_map","DC_map"]:

        corgi_dat = detector.imaging_slice('SCI', corgidrp_noisemap.__dict__[noise_ext])
        iit_dat = detector.imaging_slice('SCI', iit_noisemap.__dict__[noise_ext])

        diff = corgi_dat - iit_dat

        # # Plot for debugging:
        # import matplotlib.pyplot as plt
        # vmaxes = {
        #     "FPN_map" : 20,
        #     "CIC_map" : 0.5,
        #     "DC_map"  : 0.5
        # }

        # _,axes = plt.subplots(1,3,figsize=(16,4))

    
        # dataset1_comparison_value = np.nanmean(corgi_dat)
        # dataset2_comparison_value = np.nanmean(iit_dat)
        # diff_comparison_value = np.nanmean(diff)
        
        # im = axes[0].imshow(corgi_dat,origin='lower',vmax=vmaxes[noise_ext],vmin=0)
        # plt.colorbar(im, ax=axes[0])
        # axes[0].set_title("CorgiDRP(mean={:.2E})".format(dataset1_comparison_value))

        # im = axes[1].imshow(iit_dat,origin='lower',vmax=vmaxes[noise_ext],vmin=0)
        # plt.colorbar(im, ax=axes[1])
        # axes[1].set_title("II&T(mean={:.2E})".format(dataset2_comparison_value))

        # im = axes[2].imshow(diff,origin='lower',vmax=1e-5,vmin=-1e-5,cmap='seismic')
        # plt.colorbar(im, ax=axes[2])
        # axes[2].set_title("Difference(mean={:.2E})".format(diff_comparison_value))

        # plt.suptitle(noise_ext[:-4])
        # plt.tight_layout()
        # plt.savefig(os.path.join(noisemap_outputdir,f"CorgiDRP_TVAC_Comparison_l2a_to_l2a_{noise_ext}.pdf"))
        # plt.close()

        assert np.all(np.abs(diff) < 1e-5)

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    tvacdata_dir = "/Users/sbogat/Documents/01_Research/Roman/corgidrp_workspace/CGI_TVAC_Data"
    outputdir = "/Users/sbogat/Documents/01_Research/Roman/corgidrp_workspace/results"

    ap = argparse.ArgumentParser(description="run the l2a->l2a_noisemap end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    test_noisemap_calibration_from_l2a(tvacdata_dir, outputdir)
    test_noisemap_calibration_from_l1(tvacdata_dir, outputdir)
