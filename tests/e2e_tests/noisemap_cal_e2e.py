# DetectorNoiseMap E2E Test Code

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
import corgidrp.check as check
from corgidrp.darks import build_synthesized_dark

try:
    from cal.calibrate_darks.calibrate_darks_lsq import calibrate_darks_lsq 
    from proc_cgi_frame.gsw_process import Process
except:
    pass

thisfile_dir = os.path.dirname(__file__) # this file's folder

def fix_str_for_tvac(
    list_of_fits,
    ):
    """ 
    Makes type for each header to what it should be.

    Gets around EMGAIN_A being set to 1 in TVAC data.

    Adds proper values to VISTYPE for the NoiseMap calibration: CGIVST_CAL_DRK
    (data used to calibrate the dark noise sources).

    This function is unnecessary with future data because data will have
    the proper values in VISTYPE. 

    Args:
    list_of_fits (list): list of FITS files that need to be updated.

    """
    for file in list_of_fits:
        with fits.open(file, mode='update') as fits_file:
        #fits_file = fits.open(file)
            exthdr = fits_file[1].header
            prihdr = fits_file[0].header
            errhdr = fits_file[2].header if len(fits_file) > 2 else None
            dqhdr = fits_file[3].header if len(fits_file) > 3 else None
            ref_errhdr = None
            ref_dqhdr = None
            prihdr['VISTYPE'] = 'CGIVST_CAL_DRK'
            if exthdr['DATALVL'].lower() == 'l1':
                ref_prihdr, ref_exthdr = mocks.create_default_L1_headers(exthdr['ARRTYPE'], prihdr['VISTYPE'])
            elif exthdr['DATALVL'].lower() == 'l2a':
                ref_prihdr, ref_exthdr, ref_errhdr, ref_dqhdr, ref_biashdr = mocks.create_default_L2a_headers(exthdr['ARRTYPE'])
            elif exthdr['DATALVL'].lower() == 'l2b':
                ref_prihdr, ref_exthdr, ref_errhdr, ref_dqhdr, ref_biashdr = mocks.create_default_L2b_headers(exthdr['ARRTYPE'])
            elif exthdr['DATALVL'].lower() == 'cal':
                ref_prihdr, ref_exthdr, ref_errhdr, ref_dqhdr = mocks.create_default_calibration_product_headers()
            ##could add in more
            else:
                raise ValueError(f"Unrecognized DATALVL {exthdr['DATALVL']} in file {file}")
            for el in [(ref_prihdr, prihdr), (ref_exthdr, exthdr), (ref_errhdr, errhdr), (ref_dqhdr, dqhdr)]:
                if el[0] is None or el[1] is None:
                    continue
                for key in el[0].keys():
                    if 'NAXIS' in key or 'HISTORY' in key:
                        continue
                    if key not in el[1].keys():
                        el[1][key] = el[0][key]
                    else: 
                        if type(el[1][key]) != type(el[0][key]):
                            type_class = type(el[0][key])
                            if el[1][key] == 'N/A' and type_class != str:
                                el[1][key] = el[0][key]
                            else:
                                try:
                                    el[1][key] = type_class(el[1][key])
                                except: 
                                    if el[1][key] == "OPEN":
                                        el[1][key] = 0
                                    elif el[1][key] == "CLOSED":
                                        el[1][key] = 1
            # don't delete any headers that do not appear in the reference headers, although there shouldn't be any
            if float(exthdr['EMGAIN_A']) == 1. and exthdr['HVCBIAS'] <= 0:
                exthdr['EMGAIN_A'] = -1. #for new SSC-updated TVAC files which have EMGAIN_A by default as 1 regardless of the commanded EM gain
            if type(exthdr['EMGAIN_C']) is str:
                exthdr['EMGAIN_C'] = float(exthdr['EMGAIN_C'])
            
            # Update FITS file
            #fits_file.writeto(file, overwrite=True)
            fits_file.flush()



@pytest.mark.e2e
def test_noisemap_calibration_from_l1(e2edata_path, e2eoutput_path):
    """End-to-End test for generating NoiseMap calibration files, starting with L1 data.

    Args:
        e2edata_path (str or path): Path to the directory holding all TVAC data.
        e2eoutput_path (str or path): Path for test output files.
    """

    # figure out paths for both II&T and DRP runs, assuming everything is located in the same relative location as in the TVAC Box drive
    l1_datadir = os.path.join(e2edata_path, "TV-20_EXCAM_noise_characterization", "noisemap_test_data", "test_l1_data")

    # define the raw science data to process
    l1_data_filelist = sorted(glob(os.path.join(l1_datadir,"*.fits")))
    #l2a_data_filelist = sorted(glob(os.path.join(l2a_datadir,"*.fits")))
    # l2a_data_filename = corgidrp.data.Dataset(l2a_data_filelist[:1])[0].filename
    # output_filename = l2a_data_filename[:24] + '_DNM_CAL.fits'

    # Create main noisemap_cal_e2e directory
    main_output_dir = os.path.join(e2eoutput_path, "noisemap_cal_e2e")
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    
    # Create l1_to_dnm subdirectory
    l1_to_dnm_dir = os.path.join(main_output_dir, "l1_to_dnm")
    if os.path.exists(l1_to_dnm_dir):
        shutil.rmtree(l1_to_dnm_dir)
    os.makedirs(l1_to_dnm_dir)
    
    # Create subdirectories for l1_to_dnm
    input_l1_dir = os.path.join(l1_to_dnm_dir, 'input_l1')
    processed_l2a_dir = os.path.join(l1_to_dnm_dir, 'l1_to_l2a')
    calibrations_dir = os.path.join(l1_to_dnm_dir, 'calibrations')
    
    os.makedirs(input_l1_dir)
    os.makedirs(processed_l2a_dir)
    os.makedirs(calibrations_dir)
    
    noisemap_outputdir = l1_to_dnm_dir
    input_data_dir = input_l1_dir

    #fix_str_for_tvac(l1_data_filelist)
    # Fix L1 headers in the copied inputs
    l1_data_filelist = check.fix_hdrs_for_tvac(
        l1_data_filelist,
        input_data_dir,
        header_template=mocks.create_default_L1_headers,
    )

    # Set VISTYPE/PHTCNT after header fix
    for file in l1_data_filelist:
        with fits.open(file, mode='update') as fits_file:
            prihdr = fits_file[0].header
            prihdr['VISTYPE'] = 'CGIVST_CAL_DRK'
            prihdr['PHTCNT'] = 0
    
    mock_cal_filelist = l1_data_filelist[-2:] # grab the last two input data to mock the calibration

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
    mocks.rename_files_to_cgi_format(list_of_fits=[det_params], output_dir=calibrations_dir, level_suffix="dpm_cal")
    fwc_pp_e = int(det_params.params['FWC_PP_E']) # same as what is in DRP's DetectorParams
    fwc_em_e = int(det_params.params['FWC_EM_E']) # same as what is in DRP's DetectorParams
    telem_rows_start = det_params.params['TELRSTRT']
    telem_rows_end = det_params.params['TELREND']
    telem_rows = slice(telem_rows_start, telem_rows_end)
    #stack_arr_f_l1 = list(l1_data_filelist)

    # stackl1_dat = data.Dataset(stack_arr_f_l1)
    stackl1_dat = data.Dataset(l1_data_filelist)
    splitl1, splitl1_params = stackl1_dat.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C'])
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
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
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
    old_DNMs = sorted(glob(os.path.join(noisemap_outputdir,'*_DNM_CAL.fits')))
    old_DNMs2 = sorted(glob(os.path.join(noisemap_outputdir,'*_dnm_cal.fits')))
    for old_DNM in old_DNMs:
        os.remove(old_DNM)
    for old_DNM in old_DNMs2:
        os.remove(old_DNM)
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    pri_hdr, ext_hdr = mocks.create_default_L1_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")
    #fix_str_for_tvac([nonlinear_cal.filepath])
    this_caldb.create_entry(nonlinear_cal)

    # KGain calibration 
    kgain_val = 8.7 # From TVAC-20 noise characterization measurements
    kgain = data.KGain(kgain_val, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    # add in keywords that didn't make it into mock_kgain.fits, using values used in mocks.create_photon_countable_frames()
    kgain.ext_hdr['RN'] = 100.
    kgain.ext_hdr['RN_ERR'] = 0.
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    #fix_str_for_tvac([kgain.filepath])
    this_caldb.create_entry(kgain)

    # getting output filename
    # output_filenamel1 = os.path.split(stack_arr_files[0])[1][:-5] + '_DNM_CAL.fits'
    # #Since the walker updates to L2a and the filename accordingly:
    # output_filename = output_filenamel1.replace('L1','L2a',1)

    ####### Run the DRP walker
    #template = "l1_to_l2a_noisemap.json"
    #template = "l2a_to_l2a_noisemap.json"
    #guess template should work
    # walker.walk_corgidrp(stack_arr_files, "", noisemap_outputdir,template=None)

    # for no weighting:
    recipe = walker.autogen_recipe(stack_arr_files, noisemap_outputdir)
    ### Modify a keyword
    for step in recipe[1]['steps']:
        if step['name'] == "calibrate_darks":
            step['keywords'] = {}
            step['keywords']['weighting'] = False # to be comparable to II&T code, which does no weighting
    output_filepaths = walker.run_recipe(recipe[0], save_recipe_file=True)
    recipe[1]['inputs'] = output_filepaths
    walker.run_recipe(recipe[1], save_recipe_file=True)
    

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
    
    assert(np.nanmax(np.abs(corgidrp_noisemap.data[0]- F_map)) < 1e-9)
    assert(np.nanmax(np.abs(corgidrp_noisemap.data[1]- C_map)) < 1e-9)
    assert(np.nanmax(np.abs(corgidrp_noisemap.data[2]- D_map)) < 1e-9)
    assert(np.abs(corgidrp_noisemap.ext_hdr['B_O']- bias_offset) < 1e-9)
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
    
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

@pytest.mark.e2e
def test_noisemap_calibration_from_l2a(e2edata_path, e2eoutput_path):
    """End-to-End test for generating NoiseMap calibration files, starting with L2a data.

    Args:
        e2edata_path (str or path): Path to the directory holding all TVAC data.
        e2eoutput_path (str or path): Path for test output files.
    """

    # figure out paths for both II&T and DRP runs, assuming everything is located in the same relative location as in the TVAC Box drive
    l1_datadir = os.path.join(e2edata_path, "TV-20_EXCAM_noise_characterization", "noisemap_test_data", "test_l1_data")

    # define the raw science data to process
    l1_data_filelist = sorted(glob(os.path.join(l1_datadir,"*.fits")))
    
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
    bad_pix = np.zeros((1200,2200)) # what is used in DRP
    eperdn = 8.7 # what is used in DRP
    b_offset = 0. # what is used in DRP
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    det_params = this_caldb.get_calib(None, data.DetectorParams)
    fwc_pp_e = int(det_params.params['FWC_PP_E']) # same as what is in DRP's DetectorParams
    fwc_em_e = int(det_params.params['FWC_EM_E']) # same as what is in DRP's DetectorParams
    telem_rows_start = det_params.params['TELRSTRT']
    telem_rows_end = det_params.params['TELREND']
    telem_rows = slice(telem_rows_start, telem_rows_end)

    # Need to run II&T code on L1 data b/c that's what II&T code expects as input
    # For DRP in this test, L2a is expected, so for consistency between the II&T and DRP tests, 
    # we process from L1 to L2a before inputting to DRP since that is what II&T code does with L1 input before calibration for noisemaps
    stackl1_dat = data.Dataset(l1_data_filelist)
    splitl1, splitl1_params = stackl1_dat.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C'])
    stackl1_arr = []
    # make folder for saving the II&T processed L2a files to be used by DRP code later
    # Create l2a_to_dnm subdirectory under main noisemap_cal_e2e directory
    main_output_dir = os.path.join(e2eoutput_path, "noisemap_cal_e2e")
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    
    l2a_to_dnm_dir = os.path.join(main_output_dir, "l2a_to_dnm")
    if os.path.exists(l2a_to_dnm_dir):
        shutil.rmtree(l2a_to_dnm_dir)
    os.makedirs(l2a_to_dnm_dir)
    
    # Create subdirectories for l2a_to_dnm
    input_l1_dir = os.path.join(l2a_to_dnm_dir, 'input_l1')
    input_l2a_dir = os.path.join(l2a_to_dnm_dir, 'input_l2a')
    calibrations_dir = os.path.join(l2a_to_dnm_dir, 'calibrations')
    
    os.makedirs(input_l1_dir)
    os.makedirs(input_l2a_dir)
    os.makedirs(calibrations_dir)
    
    
    for i, file_path in enumerate(l1_data_filelist):
        shutil.copy2(file_path, input_l1_dir)
    
    # keep track of file order
    l2a_filepaths = []
    pri_hdr, ext_hdr, errhdr, dqhdr, biashdr = mocks.create_default_L2a_headers()
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
            ext_hdr["EMGAIN_C"] = em_gain
            ext_hdr['EXPTIME'] = exptime
            ext_hdr['KGAINPAR'] = eperdn
            d1_data = data.Image(d1, pri_hdr=pri_hdr, ext_hdr=ext_hdr, dq=bp1)
            fname = dset.frames[j].filename.replace('_l1_.fits','_l2a.fits')
            d1_data.save(input_l2a_dir, fname)
            l2a_filepaths.append(d1_data.filepath)
    stackl1_arr = np.stack(stackl1_arr)
    kgain_arr = [8.7]*len(exptime_arr)
    exptime_arr = np.array(exptime_arr)
    gain_arr = np.array(gain_arr)
    kgain_arr = np.array(kgain_arr)
    
    ####### Run the II&T code
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
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
    old_DNMs = sorted(glob(os.path.join(l2a_to_dnm_dir,'*_DNM_CAL.fits')))
    old_DNMs2 = sorted(glob(os.path.join(l2a_to_dnm_dir,'*_dnm_cal.fits')))
    for old_DNM in old_DNMs:
        os.remove(old_DNM)
    for old_DNM in old_DNMs2:
        os.remove(old_DNM)
    
    mock_cal_filelist = l1_data_filelist[-2:] # grab the last two input data to mock the calibration 
    mock_cal_filelist = check.fix_hdrs_for_tvac(
        mock_cal_filelist,
        input_l1_dir,
        header_template=mocks.create_default_L1_headers,
    )
    mock_input_dataset = data.Dataset(mock_cal_filelist)
    
    # KGain calibration
    kgain_val = 8.7 # From TVAC-20 noise characterization measurements
    kgain = data.KGain(kgain_val, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    # add in keywords that didn't make it into mock_kgain.fits, using values used in mocks.create_photon_countable_frames()
    kgain.ext_hdr['RN'] = 100.
    kgain.ext_hdr['RN_ERR'] = 0.
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    #fix_str_for_tvac([kgain.filepath])
    this_caldb.create_entry(kgain)

    # Update VISTPYE to "CGIVST_CAL_DRK" for DRP run
    #fix_str_for_tvac(l2a_filepaths)
    # Fix L2a headers in the copied inputs
    l2a_filepaths = check.fix_hdrs_for_tvac(
        l2a_filepaths,
        input_l2a_dir,
        header_template=mocks.create_default_L2a_headers,
    )
    for file in l2a_filepaths:
        with fits.open(file, mode='update') as fits_file:
            prihdr = fits_file[0].header
            prihdr['VISTYPE'] = 'CGIVST_CAL_DRK'
            prihdr['PHTCNT'] = 0


    ####### Run the DRP walker
    # template = "l2a_to_l2a_noisemap.json"
    # walker.walk_corgidrp(l2a_filepaths, "", l2a_to_dnm_dir,template=template)

    recipe = walker.autogen_recipe(l2a_filepaths, l2a_to_dnm_dir)
    ### Modify a keyword
    for step in recipe[1]['steps']:
        if step['name'] == "calibrate_darks":
            step['keywords'] = {}
            step['keywords']['weighting'] = False # to be comparable to II&T code, which does no weighting
    output_filepaths = walker.run_recipe(recipe[0], save_recipe_file=True)
    recipe[1]['inputs'] = output_filepaths
    walker.run_recipe(recipe[1], save_recipe_file=True) 


    # getting output filename
    for f in os.listdir(l2a_to_dnm_dir):
        if f.endswith('_dnm_cal.fits'):
            output_filename = f
            break

    ##### Check against II&T ("TVAC") data
    corgidrp_noisemap_fname = os.path.join(l2a_to_dnm_dir,output_filename)

    corgidrp_noisemap = data.autoload(corgidrp_noisemap_fname)
    # iit_noisemap = data.autoload(iit_noisemap_fname)
    

    assert(np.nanmax(np.abs(corgidrp_noisemap.data[0]- F_map)) < 1e-9)
    assert(np.nanmax(np.abs(corgidrp_noisemap.data[1]- C_map)) < 1e-9)
    assert(np.nanmax(np.abs(corgidrp_noisemap.data[2]- D_map)) < 1e-9)
    assert(np.abs(corgidrp_noisemap.ext_hdr['B_O']- bias_offset) < 1e-9)
    pass

    # create synthesized master dark in output folder (for inspection and for having a sample synthesized dark with all the right headers)
    mock_dataset = mocks.create_prescan_files() # dummy dataset with an EM gain and exposure time for creating synthesized dark
    master_dark = build_synthesized_dark(mock_dataset, corgidrp_noisemap)
    master_dark.save(filedir=calibrations_dir)
    
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
    e2edata_dir = '/Users/kevinludwick/Documents/DRP_E2E_Test_Files_v2/E2E_Test_Data'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l2a->l2a_noisemap end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_noisemap_calibration_from_l2a(e2edata_dir, outputdir)
    test_noisemap_calibration_from_l1(e2edata_dir, outputdir)
