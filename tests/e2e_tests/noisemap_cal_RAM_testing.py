# RAM testing 
'''This script is designed to test ~26000 frames for generating noise maps.  It uses a shortcut 
through all the preliminary step functions before calibrate_darks_lsq(), the main 
RAM-heavy function for generating noise maps.  

For this shortcut to work, in walker.run_recipe(), change 
curr_dataset = data.Dataset(filelist, no_data=True) 
 
 to 

curr_dataset = data.Dataset(filelist, no_data=True, no_err=True, no_dq=True)

and add in 

from memory_profiler import profile

and add @profile decorator above walker.walk_corgidrp() and darks.calibrate_darks_lsq()
'''

import argparse
import os
import shutil
import pytest
import numpy as np
import warnings
import astropy.time as time
from astropy.io import fits

from glob import glob

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
from corgidrp.darks import build_synthesized_dark
import logging
from datetime import date
from memory_profiler import profile

try:
    from cal.calibrate_darks.calibrate_darks_lsq import calibrate_darks_lsq
    from proc_cgi_frame.gsw_process import Process
except:
    pass

thisfile_dir = os.path.dirname(__file__) # this file's folder

def set_obstype_for_darks(
    list_of_fits,
    ):
    """ Adds proper values to VISTYPE for the NoiseMap calibration: CGIVST_CAL_DRK
    (data used to calibrate the dark noise sources).

    This function is unnecessary with future data because data will have
    the proper values in VISTYPE.

    Args:
    list_of_fits (list): list of FITS files that need to be updated.

    """
    # Folder with files
    for file in list_of_fits:
        with fits.open(file, mode='update') as fits_file:
            prihdr = fits_file[0].header
            exthdr = fits_file[1].header
            if float(exthdr['EMGAIN_A']) == 1 and exthdr['HVCBIAS'] <= 0:
                exthdr['EMGAIN_A'] = -1  # for new SSC-updated TVAC files which have EMGAIN_A by default as 1 regardless of the commanded EM gain
            prihdr['VISTYPE'] = 'CGIVST_CAL_DRK'
            prihdr['PHTCNT'] = False
            # Update FITS file in-place
            fits_file.flush()


@profile
def test_noisemap_calibration_from_l1(e2edata_path, e2eoutput_path):
    """End-to-End test for generating NoiseMap calibration files, starting with L1 data.

    Args:
        e2edata_path (str or path): Path to the directory holding all TVAC data.
        e2eoutput_path (str or path): Path for test output files.
    """
    import tracemalloc
    tracemalloc.start()

    import psutil
    pr = psutil.Process()
    import datetime
    # figure out paths for both II&T and DRP runs, assuming everything is located in the same relative location as in the TVAC Box drive
    l1_datadir = os.path.join(e2edata_path, "TV-20_EXCAM_noise_characterization", "noisemap_test_data", "test_l1_data_2")

    while len(os.listdir(l1_datadir)) < 26000:
        for filename in os.listdir(l1_datadir):
            f = os.path.join(l1_datadir, filename)
            if f.endswith('l1_.fits'):
                f_dest = f
                base_time = datetime.datetime.now()
                time_offset = datetime.timedelta(seconds=os.listdir(l1_datadir).index(filename))
                unique_time = base_time + time_offset
                time_str = data.format_ftimeutc(unique_time.isoformat())
                f_dest = f_dest[:len(f_dest)-25] + time_str + f_dest[len(f_dest)-9:]
                shutil.copy(f, f_dest)

    # define the raw science data to process
    l1_data_filelist = glob(os.path.join(l1_datadir,"*.fits"))[:7] #XXX sorted(glob(os.path.join(l1_datadir,"*.fits")))
    #l2a_data_filelist = sorted(glob(os.path.join(l2a_datadir,"*.fits")))
    # l2a_data_filename = corgidrp.data.Dataset(l2a_data_filelist[:1])[0].filename
    # output_filename = l2a_data_filename[:24] + '_DNM_CAL.fits'
    mock_cal_filelist = l1_data_filelist[-2:] # grab the last two input data to mock the calibration

    # Create main noisemap_cal_e2e directory
    main_output_dir = os.path.join(e2eoutput_path, "noisemap_cal_e2e")
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)

    # Create l1_to_dnm subdirectory
    l1_to_dnm_dir = os.path.join(main_output_dir, "l1_to_dnm")
    # if os.path.exists(l1_to_dnm_dir):
    #     shutil.rmtree(l1_to_dnm_dir)
    # os.makedirs(l1_to_dnm_dir)

    # Create subdirectories for l1_to_dnm
    input_l1_dir = os.path.join(l1_to_dnm_dir, 'input_l1')
    processed_l2a_dir = os.path.join(l1_to_dnm_dir, 'l1_to_l2a')
    calibrations_dir = os.path.join(l1_to_dnm_dir, 'calibrations')

    # os.makedirs(input_l1_dir)
    # os.makedirs(processed_l2a_dir)
    # os.makedirs(calibrations_dir)

    noisemap_outputdir = l1_to_dnm_dir
    input_data_dir = input_l1_dir


    # Copy files to input_data directory with proper naming
    # for i, file_path in enumerate(l1_data_filelist):
    #     shutil.copy2(file_path, input_data_dir)

    # Update l1_data_filelist to point to new files
    # l1_data_filelist = []
    # for f in os.listdir(input_data_dir):
    #     if f.endswith('.fits'):
    #         l1_data_filelist.append(os.path.join(input_data_dir, f))

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB

    ########## prepping inputs for II&T run
    # drawing same parameters and metadata as found in DRP
    corgidrp_folder = os.path.split(corgidrp.__file__)[0]
    corgidrp_f = os.path.split(corgidrp_folder)[0]
    meta_path = os.path.join(corgidrp_f, 'tests', 'test_data', 'metadata.yaml')
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    det_params = this_caldb.get_calib(None, data.DetectorParams)
    #mocks.rename_files_to_cgi_format(list_of_fits=[det_params], output_dir=calibrations_dir, level_suffix="dpm_cal")
    fwc_pp_e = int(det_params.params['FWC_PP_E']) # same as what is in DRP's DetectorParams
    fwc_em_e = int(det_params.params['FWC_EM_E']) # same as what is in DRP's DetectorParams
    telem_rows_start = det_params.params['TELRSTRT']
    telem_rows_end = det_params.params['TELREND']
    telem_rows = slice(telem_rows_start, telem_rows_end)
    stack_arr_f_l1 = []
    for f in os.listdir(l1_datadir)[:26000]: #XXX 
        file = os.path.join(l1_datadir, f)
        if not file.endswith('.fits'):
            continue
        stack_arr_f_l1.append(file)

    stack_arr_files = stack_arr_f_l1

    # stackl1_dat = data.Dataset(stack_arr_f_l1)
    # splitl1, splitl1_params = stackl1_dat.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C'])
    # stackl1_arr = []
    # exptime_arr = []
    # gain_arr = []
    # stack_arr_files = [] # in case split_dataset scrambled order some
    # for i, dset in enumerate(splitl1):
    #     stackl1_arr.append(dset.all_data[:10]) #get first 10 frames, to speed up runs
    #     for j in range(len(dset.all_data[:10])):
    #         stack_arr_files.append(dset.frames[j].filepath)
    #     exptime_arr.append(splitl1_params[i][0])
    #     gain_arr.append(splitl1_params[i][1])
    # stackl1_arr = np.stack(stackl1_arr)
    # kgain_arr = [8.7]*len(exptime_arr)

    # exptime_arr = np.array(exptime_arr)
    # gain_arr = np.array(gain_arr)
    # kgain_arr = np.array(kgain_arr)

    # ####### call II&T code
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore', category=UserWarning)
    #     (F_map, C_map, D_map, bias_offset, F_image_map, C_image_map,
    #                 D_image_map, Fvar, Cvar, Dvar, read_noise, R_map, F_image_mean,
    #                 C_image_mean, D_image_mean, unreliable_pix_map) = \
    #     calibrate_darks_lsq(stackl1_arr, gain_arr, exptime_arr, kgain_arr, fwc_em_e, fwc_pp_e,
    #                 meta_path, nonlin_path, Nem = 604, telem_rows=telem_rows,
    #                 sat_thresh=0.7, plat_thresh=0.7, cosm_filter=1, cosm_box=3,
    #                 cosm_tail=10, desmear_flags=None, rowreadtime=223.5e-6)
    # ##########

    ####### Now prep and setup necessary calibration files for DRP run

    # remove old DetectorNoiseMaps
    old_DNMs = sorted(glob(os.path.join(noisemap_outputdir,'*_DNM_CAL.fits')))
    old_DNMs2 = sorted(glob(os.path.join(noisemap_outputdir,'*_dnm_cal.fits')))
    for old_DNM in old_DNMs:
        os.remove(old_DNM)
    # for old_DNM in old_DNMs2:
    #     os.remove(old_DNM)
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    pri_hdr, ext_hdr = mocks.create_default_L1_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")
    this_caldb.create_entry(nonlinear_cal)

    # KGain calibration
    kgain_val = 8.7 # From TVAC-20 noise characterization measurements
    kgain = data.KGain(kgain_val, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                    input_dataset=mock_input_dataset)
    # add in keywords that didn't make it into mock_kgain.fits, using values used in mocks.create_photon_countable_frames()
    kgain.ext_hdr['RN'] = 100
    kgain.ext_hdr['RN_ERR'] = 0
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    this_caldb.create_entry(kgain)

    # getting output filename
    # output_filenamel1 = os.path.split(stack_arr_files[0])[1][:-5] + '_DNM_CAL.fits'
    # #Since the walker updates to L2a and the filename accordingly:
    # output_filename = output_filenamel1.replace('L1','L2a',1)

    # Update VISTYPE to "DARK" for DRP run
    #set_obstype_for_darks(stack_arr_files)
    # update headers
    #fix_headers_for_tvac(stack_arr_files)

    ####### Run the DRP walker
    #template = "l1_to_l2a_noisemap.json"
    #template = "l2a_to_l2a_noisemap.json"
    #guess template should work
    # walker.walk_corgidrp(stack_arr_files, "", noisemap_outputdir,template=None)

    # for no weighting:
    recipe = walker.autogen_recipe(stack_arr_files, noisemap_outputdir)
    ### Modify a keyword
    # for step in recipe[1]['steps']:
    #     if step['name'] == "calibrate_darks":
    #         step['keywords'] = {}
    #         step['keywords']['weighting'] = False # to be comparable to II&T code, which does no weighting
    # output_filepaths = walker.run_recipe(recipe[0], save_recipe_file=True) XXX
    # recipe[1]['inputs'] = output_filepaths XXX shortcut to skip to the RAM-heavy part  
    recipe[1]['inputs'] = recipe[0]['inputs']
    walker.run_recipe(recipe[1], save_recipe_file=True)

    mem = pr.memory_info()
    # peak_wset is only available on Windows; fall back to rss on other platforms
    if hasattr(mem, 'peak_wset') and getattr(mem, 'peak_wset') is not None:
        peak_memory = mem.peak_wset / (1024 ** 2)  # convert to MB
    else:
        peak_memory = mem.rss / (1024 ** 2)  # convert to MB
    print(f"noisemap_cal_e2e peak memory usage:  {peak_memory:.2f} MB")
    logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), "noisemap_cal_e2e_memory_usage.log"), level=logging.INFO)
    todays_date = date.today()
    logging.info(todays_date.strftime("%Y-%m-%d"))
    logging.info(f"psutil noisemap_cal_e2e peak memory usage:  {peak_memory} MB")
    # Get current and peak memory usage
    current, peak = tracemalloc.get_traced_memory()

    # Stop tracing
    tracemalloc.stop()

    # Print the peak memory usage
    print(f"tracemalloc Peak memory usage was {peak / (1024 * 1024):.2f} MB")
    logging.info(f"tracemalloc noisemap_cal_e2e peak memory usage:  {peak/(1024 * 1024)} MB")

    # Move L2a files to processed_l2a directory
    for f in os.listdir(noisemap_outputdir):
        if f.endswith('_l2a.fits'):
            shutil.move(os.path.join(noisemap_outputdir, f), os.path.join(processed_l2a_dir, f))

    ##### Check against II&T ("TVAC") data
    for f in os.listdir(noisemap_outputdir):
        if f.endswith('_dnm_cal.fits'):
            output_filename = f
            break
    corgidrp_noisemap_fname = os.path.join(noisemap_outputdir,output_filename)
    # iit_noisemap_fname = os.path.join(iit_noisemap_datadir,"iit_test_noisemaps.fits")
    corgidrp_noisemap = data.autoload(corgidrp_noisemap_fname)

    # assert(np.nanmax(np.abs(corgidrp_noisemap.data[0]- F_map)) < 1e-9)
    # assert(np.nanmax(np.abs(corgidrp_noisemap.data[1]- C_map)) < 1e-9)
    # assert(np.nanmax(np.abs(corgidrp_noisemap.data[2]- D_map)) < 1e-9)
    # assert(np.abs(corgidrp_noisemap.ext_hdr['B_O']- bias_offset) < 1e-9)
    # pass

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

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    #e2edata_dir = '/home/jwang/Desktop/CGI_TVAC_Data/'
    # e2edata_dir = '/Users/kevinludwick/Documents/DRP E2E Test Files v2/E2E_Test_Data'
    # outputdir = thisfile_dir

    outputdir = r'E:\E2E_tests'#thisfile_dir
    e2edata_dir =  r'E:\E2E_Test_Data3\E2E_Test_Data3'

    ap = argparse.ArgumentParser(description="run the l2a->l2a_noisemap end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()

    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_noisemap_calibration_from_l1(e2edata_dir, outputdir)

