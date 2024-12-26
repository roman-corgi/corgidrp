# DetectorNoiseMap E2E Test Code

import argparse
import os
import pytest
import numpy as np
import astropy.time as time
from astropy.io import fits

from glob import glob

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector

try:
    from cal.calibrate_darks.calibrate_darks_lsq import calibrate_darks_lsq 
    from proc_cgi_frame.gsw_process import Process
except:
    pass

thisfile_dir = os.path.dirname(__file__) # this file's folder

def set_obstype_for_darks(
    list_of_fits,
    ):
    """ Adds proper values to VISTYPE for the NoiseMap calibration: DARKS
    (data used to calibrate the dark noise sources).

    This function is unnecessary with future data because data will have
    the proper values in VISTYPE. 

    Args:
    list_of_fits (list): list of FITS files that need to be updated.

    """
    # Folder with files
    for file in list_of_fits:
        fits_file = fits.open(file)
        prihdr = fits_file[0].header
        prihdr['VISTYPE'] = 'DARK'
        # Update FITS file
        fits_file.writeto(file, overwrite=True)

@pytest.mark.e2e
def test_noisemap_calibration_from_l1(tvacdata_path, e2eoutput_path):
    """End-to-End test for generating NoiseMap calibration files, starting with L1 data.

    Args:
        tvacdata_path (str or path): Path to the directory holding all TVAC data.
        e2eoutput_path (str or path): Path for test output files.
    """

    # figure out paths for both II&T and DRP runs, assuming everything is located in the same relative location as in the TVAC Box drive
    l1_datadir = os.path.join(tvacdata_path, "TV-20_EXCAM_noise_characterization", "noisemap_test_data", "test_l1_data")

    # define the raw science data to process
    l1_data_filelist = sorted(glob(os.path.join(l1_datadir,"*.fits")))
    #l2a_data_filelist = sorted(glob(os.path.join(l2a_datadir,"*.fits")))
    # l2a_data_filename = corgidrp.data.Dataset(l2a_data_filelist[:1])[0].filename
    # output_filename = l2a_data_filename[:24] + '_DetectorNoiseMaps.fits'
    mock_cal_filelist = l1_data_filelist[-2:] # grab the last two input data to mock the calibration 
    
    ########## prepping inputs for II&T run
    # drawing same parameters and metadata as found in DRP
    corgidrp_folder = os.path.split(corgidrp.__file__)[0]
    corgidrp_f = os.path.split(corgidrp_folder)[0]
    meta_path = os.path.join(corgidrp_f, 'tests', 'test_data', 'metadata.yaml')
    processed_cal_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals")
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    det_params = data.DetectorParams({})
    fwc_pp_e = int(det_params.params['fwc_pp']) # same as what is in DRP's DetectorParams
    fwc_em_e = int(det_params.params['fwc_em']) # same as what is in DRP's DetectorParams
    telem_rows_start = det_params.params['telem_rows_start']
    telem_rows_end = det_params.params['telem_rows_end']
    telem_rows = slice(telem_rows_start, telem_rows_end)
    stack_arr_f_l1 = []
    for f in os.listdir(l1_datadir):
        file = os.path.join(l1_datadir, f)
        if not file.endswith('.fits'):
            continue
        stack_arr_f_l1.append(file)

    stackl1_dat = data.Dataset(stack_arr_f_l1)
    splitl1, splitl1_params = stackl1_dat.split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN'])
    stackl1_arr = []
    exptime_arr = []
    gain_arr = []
    stack_arr_files = [] # in case split_dataset scrambled order some
    for i, dset in enumerate(splitl1):
        stackl1_arr.append(dset.all_data[:10]) #get first 10 frames, to speed up runs
        for j in range(len(dset.all_data[:10])):
            stack_arr_files.append(dset.frames[j].filepath)
        exptime_arr.append(splitl1_params[i][0])
        gain_arr.append(splitl1_params[i][1])
    stackl1_arr = np.stack(stackl1_arr)
    kgain_arr = [8.7]*len(exptime_arr)
    
    exptime_arr = np.array(exptime_arr)
    gain_arr = np.array(gain_arr)
    kgain_arr = np.array(kgain_arr)

    ####### call II&T code
    (F_map, C_map, D_map, bias_offset, F_image_map, C_image_map,
                D_image_map, Fvar, Cvar, Dvar, read_noise, R_map, F_image_mean,
                C_image_mean, D_image_mean, unreliable_pix_map) = \
    calibrate_darks_lsq(stackl1_arr, gain_arr, exptime_arr, kgain_arr, fwc_em_e, fwc_pp_e,
                meta_path, nonlin_path, Nem = 604, telem_rows=telem_rows, 
                sat_thresh=0.7, plat_thresh=0.7, cosm_filter=1, cosm_box=3,
                cosm_tail=10, desmear_flags=None, rowreadtime=223.5e-6)
    ##########

    ####### Now prep and setup necessary calibration files for DRP run

    # make DRP output directory if needed
    if not os.path.exists(e2eoutput_path):
        os.mkdir(e2eoutput_path)
    noisemap_outputdir = os.path.join(e2eoutput_path, "noisemap_output")
    if not os.path.exists(noisemap_outputdir):
        os.mkdir(noisemap_outputdir)

    # remove old DetectorNoiseMaps
    old_DNMs = sorted(glob(os.path.join(noisemap_outputdir,'*_DetectorNoiseMaps.fits')))
    for old_DNM in old_DNMs:
        os.remove(old_DNM)
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    this_caldb = caldb.CalDB() # connection to cal DB

    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=noisemap_outputdir, filename="mock_nonlinearcal.fits" )
    this_caldb.create_entry(nonlinear_cal)

    # KGain calibration 
    kgain_val = 8.7 # From TVAC-20 noise characterization measurements
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=noisemap_outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)

    # getting output filename
    output_filenamel1 = os.path.split(stack_arr_files[0])[1][:-5] + '_DetectorNoiseMaps.fits'
    #Since the walker updates to L2a and the filename accordingly:
    output_filename = output_filenamel1.replace('L1','L2a',1)

    # Update VISTYPE to "DARK" for DRP run
    set_obstype_for_darks(stack_arr_files)

    ####### Run the DRP walker
    template = "l1_to_l2a_noisemap.json"
    walker.walk_corgidrp(stack_arr_files, "", noisemap_outputdir,template=template)

    # clean up by removing entry
    this_caldb.remove_entry(kgain)
    this_caldb.remove_entry(nonlinear_cal)

    ##### Check against II&T ("TVAC") data
    corgidrp_noisemap_fname = os.path.join(noisemap_outputdir,output_filename)
    # iit_noisemap_fname = os.path.join(iit_noisemap_datadir,"iit_test_noisemaps.fits")

    corgidrp_noisemap = data.autoload(corgidrp_noisemap_fname)
    this_caldb.remove_entry(corgidrp_noisemap)

    assert(np.nanmax(np.abs(corgidrp_noisemap.data[0]- F_map)) < 1e-11)
    assert(np.nanmax(np.abs(corgidrp_noisemap.data[1]- C_map)) < 1e-11)
    assert(np.nanmax(np.abs(corgidrp_noisemap.data[2]- D_map)) < 1e-11)
    assert(np.abs(corgidrp_noisemap.ext_hdr['B_O']- bias_offset) < 1e-11)
    pass

    # for noise_ext in ["FPN_map","CIC_map","DC_map"]:
        # corgi_dat = detector.imaging_slice('SCI', corgidrp_noisemap.__dict__[noise_ext])
        # iit_dat = detector.imaging_slice('SCI', iit_noisemap.__dict__[noise_ext])

        # diff = corgi_dat - iit_dat

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

        # assert np.all(np.abs(diff) < 1e-5)

@pytest.mark.e2e
def test_noisemap_calibration_from_l2a(tvacdata_path, e2eoutput_path):
    """End-to-End test for generating NoiseMap calibration files, starting with L2a data.

    Args:
        tvacdata_path (str or path): Path to the directory holding all TVAC data.
        e2eoutput_path (str or path): Path for test output files.
    """

    # figure out paths for both II&T and DRP runs, assuming everything is located in the same relative location as in the TVAC Box drive
    l1_datadir = os.path.join(tvacdata_path, "TV-20_EXCAM_noise_characterization", "noisemap_test_data", "test_l1_data")

    # define the raw science data to process
    l1_data_filelist = sorted(glob(os.path.join(l1_datadir,"*.fits")))
    mock_cal_filelist = l1_data_filelist [-2:] # grab the last two input data to mock the calibration 
    
    ########## prepping inputs for II&T run
    # drawing same parameters and metadata as found in DRP
    corgidrp_folder = os.path.split(corgidrp.__file__)[0]
    corgidrp_f = os.path.split(corgidrp_folder)[0]
    meta_path = os.path.join(corgidrp_f, 'tests', 'test_data', 'metadata.yaml')
    processed_cal_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals")
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    bad_pix = np.zeros((1200,2200)) # what is used in DRP
    eperdn = 8.7 # what is used in DRP
    b_offset = 0 # what is used in DRP
    det_params = data.DetectorParams({})
    fwc_pp_e = int(det_params.params['fwc_pp']) # same as what is in DRP's DetectorParams
    fwc_em_e = int(det_params.params['fwc_em']) # same as what is in DRP's DetectorParams
    telem_rows_start = det_params.params['telem_rows_start']
    telem_rows_end = det_params.params['telem_rows_end']
    telem_rows = slice(telem_rows_start, telem_rows_end)
    stack_arr_f_l1 = []
    for f in os.listdir(l1_datadir):
        file = os.path.join(l1_datadir, f)
        if not file.endswith('.fits'):
            continue
        stack_arr_f_l1.append(file)

    # Need to run II&T code on L1 data b/c that's what II&T code expects as input
    # For DRP in this test, L2a is expected, so for consistency between the II&T and DRP tests, 
    # we process from L1 to L2a before inputting to DRP since that is what II&T code does with L1 input before calibration for noisemaps
    stackl1_dat = data.Dataset(stack_arr_f_l1)
    splitl1, splitl1_params = stackl1_dat.split_dataset(exthdr_keywords=['EXPTIME', 'CMDGAIN'])
    stackl1_arr = []
    # make folder for saving the II&T processed L2a files to be used by DRP code later
    # make output directory if needed
    if not os.path.exists(e2eoutput_path):
        os.mkdir(e2eoutput_path)
    noisemap_outputdir = os.path.join(e2eoutput_path, "noisemap_output")
    if not os.path.exists(noisemap_outputdir):
        os.mkdir(noisemap_outputdir)
    L2a_output_dir = os.path.join(noisemap_outputdir, 'L2a_output')
    # keep track of file order
    l2a_filepaths = []
    if not os.path.exists(L2a_output_dir):
        os.mkdir(L2a_output_dir)
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    exptime_arr = []
    gain_arr = []
    for i, dset in enumerate(splitl1):
        stackl1_arr.append(dset.all_data[:10]) #get first 10 frames, to speed up runs
        exptime_arr.append(splitl1_params[i][0])
        gain_arr.append(splitl1_params[i][1])
        for j, frame_data in enumerate(dset.all_data[:10]):
            exptime = splitl1_params[i][0]
            em_gain = splitl1_params[i][1]
            proc_dark = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                 b_offset, em_gain, exptime,
                 nonlin_path)
            _, _, _, _, d0, bp0, _ = proc_dark.L1_to_L2a(frame_data)
            d1, bp1, _ = proc_dark.L2a_to_L2b(d0, bp0)
            d1 /= eperdn  # undo k gain division that L2a_to_L2b() does
            d1 *= em_gain # undo EM gain division that L2a_to_L2b() does
            ext_hdr["CMDGAIN"] = em_gain
            ext_hdr['EXPTIME'] = exptime
            ext_hdr['KGAIN'] = 8.7
            d1_data = data.Image(d1, pri_hdr=pri_hdr, ext_hdr=ext_hdr, dq=bp1)
            fname = dset.frames[j].filename.replace('L1','L2a',1)
            d1_data.save(L2a_output_dir, fname)
            l2a_filepaths.append(d1_data.filepath)
    stackl1_arr = np.stack(stackl1_arr)
    kgain_arr = [8.7]*len(exptime_arr)
    exptime_arr = np.array(exptime_arr)
    gain_arr = np.array(gain_arr)
    kgain_arr = np.array(kgain_arr)
    
    ####### Run the II&T code
    (F_map, C_map, D_map, bias_offset, F_image_map, C_image_map,
                D_image_map, Fvar, Cvar, Dvar, read_noise, R_map, F_image_mean,
                C_image_mean, D_image_mean, unreliable_pix_map) = \
    calibrate_darks_lsq(stackl1_arr, gain_arr, exptime_arr, kgain_arr, fwc_em_e, fwc_pp_e,
                meta_path, nonlin_path, Nem = 604, telem_rows=telem_rows, 
                sat_thresh=0.7, plat_thresh=0.7, cosm_filter=1, cosm_box=3,
                cosm_tail=10, desmear_flags=None, rowreadtime=223.5e-6)
    ##########

    ####### Now prep and setup necessary calibration files for DRP run

    # remove old DetectorNoiseMaps
    old_DNMs = sorted(glob(os.path.join(noisemap_outputdir,'*_DetectorNoiseMaps.fits')))
    for old_DNM in old_DNMs:
        os.remove(old_DNM)
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    this_caldb = caldb.CalDB() # connection to cal DB

    # KGain calibration
    kgain_val = 8.7 # From TVAC-20 noise characterization measurements
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=noisemap_outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)

    # Update VISTPYE to "DARK" for DRP run
    set_obstype_for_darks(l2a_filepaths)

    ####### Run the DRP walker
    template = "l2a_to_l2a_noisemap.json"
    walker.walk_corgidrp(l2a_filepaths, "", noisemap_outputdir,template=template)

    # getting output filename
    l2a_data_filename = os.path.split(l2a_filepaths[0])[1]
    output_filename = l2a_data_filename[:-5] + '_DetectorNoiseMaps.fits'

    # clean up by removing entry
    this_caldb.remove_entry(kgain)

    ##### Check against II&T ("TVAC") data
    corgidrp_noisemap_fname = os.path.join(noisemap_outputdir,output_filename)

    corgidrp_noisemap = data.autoload(corgidrp_noisemap_fname)
    # iit_noisemap = data.autoload(iit_noisemap_fname)
    

    assert(np.nanmax(np.abs(corgidrp_noisemap.data[0]- F_map)) < 1e-11)
    assert(np.nanmax(np.abs(corgidrp_noisemap.data[1]- C_map)) < 1e-11)
    assert(np.nanmax(np.abs(corgidrp_noisemap.data[2]- D_map)) < 1e-11)
    assert(np.abs(corgidrp_noisemap.ext_hdr['B_O']- bias_offset) < 1e-11)
    pass
    # remove from caldb
    this_caldb.remove_entry(corgidrp_noisemap)
    
    # for noise_ext in ["FPN_map","CIC_map","DC_map"]:
        # corgi_dat = detector.imaging_slice('SCI', corgidrp_noisemap.__dict__[noise_ext])
        # iit_dat = detector.imaging_slice('SCI', iit_noisemap.__dict__[noise_ext])

    


        # diff = corgi_dat - iit_dat

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

        # assert np.all(np.abs(diff) < 1e-5)

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    tvacdata_dir = "/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/Working_Folder/"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l2a->l2a_noisemap end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    args_here = ['--tvacdata_dir', tvacdata_dir, '--outputdir', outputdir]#, '--e2e_flag',False]
    #args = ap.parse_args()
    args = ap.parse_args(args_here)
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    #test_noisemap_calibration_from_l1(tvacdata_dir, outputdir)
    test_noisemap_calibration_from_l2a(tvacdata_dir, outputdir)
