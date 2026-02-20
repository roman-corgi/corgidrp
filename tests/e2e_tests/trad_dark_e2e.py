import argparse
import os
import pytest
import re
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
from datetime import datetime, timedelta
import corgidrp
import corgidrp.data as data
import corgidrp.detector as detector
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.check as check
import shutil
try:
    from proc_cgi_frame.gsw_process import Process, mean_combine
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
            comparison_list = [(ref_prihdr, prihdr), (ref_exthdr, exthdr), (ref_errhdr, errhdr), (ref_dqhdr, dqhdr)]
            for el in comparison_list:
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
            for key in ['HOWFSLNK', 'ISHOWFSC', 'SATSPOTS']:
                if key in prihdr.keys():
                    prihdr.pop(key)
            # don't delete any headers that do not appear in the reference headers, although there shouldn't be any
            if float(exthdr['EMGAIN_A']) == 1. and exthdr['HVCBIAS'] <= 0:
                exthdr['EMGAIN_A'] = -1. #for new SSC-updated TVAC files which have EMGAIN_A by default as 1 regardless of the commanded EM gain
            if type(exthdr['EMGAIN_C']) is str:
                exthdr['EMGAIN_C'] = float(exthdr['EMGAIN_C'])
            
            # Update FITS file
            #fits_file.writeto(file, overwrite=True)
            fits_file.flush()


@pytest.mark.e2e
def test_trad_dark(e2edata_path, e2eoutput_path):
    '''There is no official II&T code for creating a "traditional" master dark (i.e., a dark made from taking the 
    mean of several darks at the same EM gain and exposure time), but all the parts are there in proc_cgi_frame.  
    So this function compares the DRP's output of build_trad_dark()
    to the output of proc_cgi_frame code to make a traditional master dark. This is for the full-frame case.

    Args:
        e2edata_path (str): path to TVAC data root directory
        e2eoutput_path (str): path to output files made by this test
    '''
    # figure out paths, assuming everything is located in the same relative location    
    trad_dark_raw_datadir = os.path.join(e2edata_path, 'TV-20_EXCAM_noise_characterization', 'darkmap')
    #TVAC_dark_path = os.path.join(e2edata_path, 'TV-20_EXCAM_noise_characterization', "results", "dark_current_20240322.fits")
    #TVAC_dark_path = os.path.join(e2edata_path, 'TV-20_EXCAM_noise_characterization', "results", "proc_cgi_frame_trad_dark.fits")

    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_current_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")

    # Create main output directory and full-frame test subfolder
    main_output_dir = os.path.join(e2eoutput_path, "trad_dark_e2e")
    build_trad_dark_outputdir = os.path.join(main_output_dir, "trad_dark_full_frame")
    if os.path.exists(build_trad_dark_outputdir):
        shutil.rmtree(build_trad_dark_outputdir)
    os.makedirs(build_trad_dark_outputdir)

    # Create input_data and calibrations subfolders
    input_data_dir = os.path.join(build_trad_dark_outputdir, 'input_l1')
    calibrations_dir = os.path.join(build_trad_dark_outputdir, 'calibrations')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB


    # define the raw science data to process
    trad_dark_data_filelist = []
    trad_dark_filename = None
    for root, _, files in os.walk(trad_dark_raw_datadir):
        for name in files:
            if not name.endswith('.fits'):
                continue
            if trad_dark_filename is None:
                trad_dark_filename = name # get first filename fed to walk_corgidrp for finding cal file later
            f = os.path.join(root, name)
            trad_dark_data_filelist.append(f)
    # run in order of files input to II&T code to get exactly the same results
    # trad_dark_data_filelist = np.load(os.path.join(e2edata_path, 'TV-20_EXCAM_noise_characterization', "results",'proc_cgi_frame_trad_dark_filelist_order.npy'), allow_pickle=True)
    # trad_dark_data_filelist = trad_dark_data_filelist.tolist()

    # Update headers
    trad_dark_data_filelist = check.fix_hdrs_for_tvac(
        trad_dark_data_filelist,
        input_data_dir,
        header_template=mocks.create_default_L1_headers,
    )
    #fix_str_for_tvac(trad_dark_data_filelist)
    # Set correct vistype for darks
    for filepath in trad_dark_data_filelist:
        with fits.open(filepath, mode='update') as hdul:
            hdul[0].header['VISTYPE'] = 'CGIVST_CAL_DRK'



    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    # dummy data; basically just need the header info to combine with II&T nonlin calibration
    #l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    #mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]]
    #mock_cal_filelist = [os.path.join(l1_datadir, os.listdir(l1_datadir)[i]) for i in [0,1]] # use first two files in L1 directory
    mock_cal_filelist = trad_dark_data_filelist[:2]
    pri_hdr, ext_hdr, _, _ = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    # Set unique timestamp and use rename_files_to_cgi_format for proper CGI filename
    base_time = datetime.now()
    nonlinear_cal.ext_hdr['FILETIME'] = base_time.isoformat()
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")
    #fix_str_for_tvac([nonlinear_cal.filepath])


    # Load and combine noise maps from various calibration files into a single array
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_current_path) as hdulist:
        dark_current_dat = hdulist[0].data

    # Combine all noise data into one 3D array
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_current_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'],
                              detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img

    # Initialize additional noise map parameters
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    # from CGI_TVAC_Data/TV-20_EXCAM_noise_characterization/tvac_noisemap_original_data/results/bias_offset.txt
    ext_hdr['B_O'] = 0. # bias offset not simulated in the data, so set to 0;  -0.0394 DN from tvac_noisemap_original_data/results
    ext_hdr['B_O_ERR'] = 0. # was not estimated with the II&T code

    # Create a DetectorNoiseMaps object and save it
    noise_maps = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                        input_dataset=mock_input_dataset, err=noise_map_noise,
                                        dq=noise_map_dq, err_hdr=err_hdr)
    # Set unique timestamp and use rename_files_to_cgi_format for proper CGI filename
    noise_maps.ext_hdr['FILETIME'] = (base_time + timedelta(seconds=1)).isoformat()
    mocks.rename_files_to_cgi_format(list_of_fits=[noise_maps], output_dir=calibrations_dir, level_suffix="dnm_cal")
    #fix_str_for_tvac([noise_maps.filepath])

    # create a k gain object and save it
    kgain_val = fits.getheader(os.path.join(trad_dark_raw_datadir, os.listdir(trad_dark_raw_datadir)[0]), 1)['KGAINPAR'] # read off header from TVAC files
    kgain_dat = kgain_val # KGain value from TVAC files
    kgain = data.KGain(kgain_dat,
                                pri_hdr=pri_hdr,
                                ext_hdr=ext_hdr,
                                input_dataset=mock_input_dataset)
    # Set unique timestamp and use rename_files_to_cgi_format for proper CGI filename
    kgain.ext_hdr['FILETIME'] = (base_time + timedelta(seconds=2)).isoformat()
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    #fix_str_for_tvac([kgain.filepath])

    # add calibration files to caldb
    this_caldb.create_entry(nonlinear_cal)
    this_caldb.create_entry(noise_maps)
    this_caldb.create_entry(kgain)

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    ####### Run the walker on some test_data; use template in recipes folder, so we can use walk_corgidrp()
    walker.walk_corgidrp(trad_dark_data_filelist, "", build_trad_dark_outputdir, template="build_trad_dark_full_frame.json")

    # find cal file (naming convention for data.Dark class)
    for f in os.listdir(build_trad_dark_outputdir):
        if f.endswith('_drk_cal.fits'):
            trad_dark_filename = f
            break
    generated_trad_dark_file = os.path.join(build_trad_dark_outputdir, trad_dark_filename) 
    
    ###################### run II&T code on data
    bad_pix = np.zeros((1200,2200)) # what is used in DRP
    eperdn = kgain_val # what is used in DRP above
    bias_offset = 0 # what is used in DRP
    emgain_val = fits.getheader(os.path.join(trad_dark_raw_datadir, os.listdir(trad_dark_raw_datadir)[0]), 1)['EMGAIN_C']
    em_gain = emgain_val # read off header from TVAC files
    exptime_val = fits.getheader(os.path.join(trad_dark_raw_datadir,os.listdir(trad_dark_raw_datadir)[0]), 1)['EXPTIME']
    exptime = exptime_val # read off header from TVAC files
    detector_params = this_caldb.get_calib(None, data.DetectorParams)
    fwc_pp_e = int(detector_params.params['FWC_PP_E']) # same as what is in DRP's DetectorParams
    fwc_em_e = int(detector_params.params['FWC_EM_E']) # same as what is in DRP's DetectorParams
    telem_rows_start = detector_params.params['TELRSTRT']
    telem_rows_end = detector_params.params['TELREND']
    telem_rows = slice(telem_rows_start, telem_rows_end)
    proc_dark = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                 bias_offset, em_gain, exptime,
                 nonlin_path)
    dark_frames = []
    bp_frames = []
    filelist = []
    for f in trad_dark_data_filelist: #os.listdir(trad_dark_raw_datadir):
        # filelist.append(os.path.join(trad_dark_raw_datadir, f))
        # file = os.path.join(trad_dark_raw_datadir, f)
        d = fits.getdata(f).astype(float) #file)
        d[telem_rows] = np.nan
        _, _, _, _, d0, bp0, _ = proc_dark.L1_to_L2a(d)
        d1, bp1, _ = proc_dark.L2a_to_L2b(d0, bp0)
        # d1 *= em_gain # undo gain division
        d1[telem_rows] = 0
        dark_frames.append(d1)
        bp_frames.append(bp1)

    # The last output of mean_combine() are useful for calibrate_darks
    # module in the calibration repository:
    mean_frame, _, mean_num_good_fr, _ = mean_combine(dark_frames, bp_frames)
    mean_frame[telem_rows] = np.nan
    # TVAC_dark_path = os.path.join(e2edata_dir, 'TV-20_EXCAM_noise_characterization', "results", "proc_cgi_frame_trad_dark.fits")
    # fits.writeto(TVAC_dark_path, mean_frame, overwrite=True)
    # np.save(TVAC_dark_path, trad_dark_data_filelist)
    # TVAC_dark_path = os.path.join(e2edata_dir, 'TV-20_EXCAM_noise_characterization', "results", "proc_cgi_frame_trad_dark.fits")
    with fits.open(generated_trad_dark_file) as trad_dark_fits:
        trad_dark_data = trad_dark_fits[1].data
    ###################
    
    ##### Check against TVAC traditional dark result

    TVAC_trad_dark = mean_frame #fits.getdata(TVAC_dark_path) 

    # Set telemetry rows to NaN in DRP data to match reference (which has NaN in telem_rows and 
    # was handled by np.nanmax before)
    trad_dark_data[telem_rows] = np.nan

    # Use allclose for float32 save/load and small numerical differences (rtol/atol=1e-5)
    assert np.allclose(TVAC_trad_dark, trad_dark_data, rtol=1e-5, atol=1e-5, equal_nan=True)
    print('e2e test for trad_dark calibration passed')
    
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)


@pytest.mark.e2e
def test_trad_dark_im(e2edata_path, e2eoutput_path):
    '''There is no official II&T code for creating a "traditional" master dark (i.e., a dark made from taking the 
    mean of several darks at the same EM gain and exposure time), but all the parts are there in proc_cgi_frame.  
    So this function compares the DRP's output of build_trad_dark()
    to the output of proc_cgi_frame code to make a traditional master dark. This is for the image-area case.

    Args:
        e2edata_path (str): path to TVAC data root directory
        e2eoutput_path (str): path to output files made by this test
    '''
    # figure out paths, assuming everything is located in the same relative location    
    trad_dark_raw_datadir = os.path.join(e2edata_path, 'TV-20_EXCAM_noise_characterization', 'darkmap')
    #TVAC_dark_path = os.path.join(e2edata_path, 'TV-20_EXCAM_noise_characterization', "results", "dark_current_20240322.fits")
    #TVAC_dark_path = os.path.join(e2edata_path, 'TV-20_EXCAM_noise_characterization', "results", "proc_cgi_frame_trad_dark.fits")

    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_current_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")

    # Create main output directory and image-area test subfolder
    main_output_dir = os.path.join(e2eoutput_path, "trad_dark_e2e")
    build_trad_dark_outputdir = os.path.join(main_output_dir, "trad_dark_image_area")
    if os.path.exists(build_trad_dark_outputdir):
        shutil.rmtree(build_trad_dark_outputdir)
    os.makedirs(build_trad_dark_outputdir)

    # Create input_data and calibrations subfolders
    input_data_dir = os.path.join(build_trad_dark_outputdir, 'input_l1')
    calibrations_dir = os.path.join(build_trad_dark_outputdir, 'calibrations')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)
    
    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB

    # define the raw science data to process
    trad_dark_data_filelist = []
    trad_dark_filename = None
    for root, _, files in os.walk(trad_dark_raw_datadir):
        for name in files:
            if not name.endswith('.fits'):
                continue
            if trad_dark_filename is None:
                trad_dark_filename = name # get first filename fed to walk_corgidrp for finding cal file later
            f = os.path.join(root, name)
            trad_dark_data_filelist.append(f)
    # run in order of files input to II&T code to get exactly the same results
    # trad_dark_data_filelist = np.load(os.path.join(e2edata_path, 'TV-20_EXCAM_noise_characterization', "results",'proc_cgi_frame_trad_dark_filelist_order.npy'), allow_pickle=True)
    # trad_dark_data_filelist = trad_dark_data_filelist.tolist()

    # Copy files to input_data directory with proper naming
    for i, file_path in enumerate(trad_dark_data_filelist):
        shutil.copy2(file_path, input_data_dir)
    
    # Update trad_dark_data_filelist to point to new files
    trad_dark_data_filelist = []
    for f in os.listdir(input_data_dir):
        if f.endswith('.fits'):
            trad_dark_data_filelist.append(os.path.join(input_data_dir, f))

    # modify headers from TVAC to in-flight headers
    trad_dark_data_filelist = check.fix_hdrs_for_tvac(
        trad_dark_data_filelist,
        input_data_dir,
        header_template=mocks.create_default_L1_headers,
    )
    fix_str_for_tvac(trad_dark_data_filelist)
    # Set correct vistype for darks
    for filepath in trad_dark_data_filelist:
        with fits.open(filepath, mode='update') as hdul:
            hdul[0].header['VISTYPE'] = 'CGIVST_CAL_DRK'


    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    # dummy data; basically just need the header info to combine with II&T nonlin calibration
    l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    #mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]]
    mock_cal_filelist = [os.path.join(l1_datadir, os.listdir(l1_datadir)[i]) for i in [0,1]] # use first two files in L1 directory
    pri_hdr, ext_hdr, _, _ = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    # Set unique timestamp and use rename_files_to_cgi_format for proper CGI filename
    base_time = datetime.now()
    nonlinear_cal.ext_hdr['FILETIME'] = base_time.isoformat()
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")
    fix_str_for_tvac([nonlinear_cal.filepath])


    # Load and combine noise maps from various calibration files into a single array
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_current_path) as hdulist:
        dark_current_dat = hdulist[0].data

    # Combine all noise data into one 3D array
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_current_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'],
                              detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img

    # Initialize additional noise map parameters
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    # from CGI_TVAC_Data/TV-20_EXCAM_noise_characterization/tvac_noisemap_original_data/results/bias_offset.txt
    ext_hdr['B_O'] = 0. # bias offset not simulated in the data, so set to 0;  -0.0394 DN from tvac_noisemap_original_data/results
    ext_hdr['B_O_ERR'] = 0. # was not estimated with the II&T code

    # Create a DetectorNoiseMaps object and save it
    noise_maps = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                        input_dataset=mock_input_dataset, err=noise_map_noise,
                                        dq=noise_map_dq, err_hdr=err_hdr)
    # Set unique timestamp and use rename_files_to_cgi_format for proper CGI filename
    noise_maps.ext_hdr['FILETIME'] = (base_time + timedelta(seconds=1)).isoformat()
    mocks.rename_files_to_cgi_format(list_of_fits=[noise_maps], output_dir=calibrations_dir, level_suffix="dnm_cal")
    fix_str_for_tvac([noise_maps.filepath])

    # create a k gain object and save it
    kgain_val = fits.getheader(os.path.join(trad_dark_raw_datadir, os.listdir(trad_dark_raw_datadir)[0]), 1)['KGAINPAR'] # read off header from TVAC files
    kgain_dat = kgain_val # KGain value from TVAC files
    kgain = data.KGain(kgain_dat,
                                pri_hdr=pri_hdr,
                                ext_hdr=ext_hdr,
                                input_dataset=mock_input_dataset)
    # Set unique timestamp and use rename_files_to_cgi_format for proper CGI filename
    kgain.ext_hdr['FILETIME'] = (base_time + timedelta(seconds=2)).isoformat()
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    fix_str_for_tvac([kgain.filepath])

    # add calibration files to caldb
    this_caldb.create_entry(nonlinear_cal)
    this_caldb.create_entry(noise_maps)
    this_caldb.create_entry(kgain)

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    ####### Run the walker on some test_data; use template in recipes folder, so we can use walk_corgidrp()
    walker.walk_corgidrp(trad_dark_data_filelist, "", build_trad_dark_outputdir) #, template="build_trad_dark_image.json")

    # find cal file (naming convention for data.Dark class)
    for f in os.listdir(build_trad_dark_outputdir):
        if f.endswith('_drk_cal.fits'):
            trad_dark_filename = f
            break
    if not trad_dark_filename or not trad_dark_filename.endswith('_drk_cal.fits'):
        raise FileNotFoundError(
            f"No *_drk_cal.fits found in {build_trad_dark_outputdir}; "
        )
    generated_trad_dark_file = os.path.join(build_trad_dark_outputdir, trad_dark_filename) 
    
    ###################### run II&T code on data
    bad_pix = np.zeros((1200,2200)) # what is used in DRP, full SCI frame
    eperdn = kgain_val # what is used in DRP above
    bias_offset = 0 # what is used in DRP
    emgain_val = fits.getheader(os.path.join(trad_dark_raw_datadir, os.listdir(trad_dark_raw_datadir)[0]), 1)['EMGAIN_C']
    em_gain = emgain_val # read off header from TVAC files
    exptime_val = fits.getheader(os.path.join(trad_dark_raw_datadir,os.listdir(trad_dark_raw_datadir)[0]), 1)['EXPTIME']
    exptime = exptime_val # read off header from TVAC files
    detector_params = this_caldb.get_calib(None, data.DetectorParams)
    fwc_pp_e = int(detector_params.params['FWC_PP_E']) # same as what is in DRP's DetectorParams
    fwc_em_e = int(detector_params.params['FWC_EM_E']) # same as what is in DRP's DetectorParams
    telem_rows_start = detector_params.params['TELRSTRT']
    telem_rows_end = detector_params.params['TELREND']
    telem_rows = slice(telem_rows_start, telem_rows_end)
    proc_dark = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                 bias_offset, em_gain, exptime,
                 nonlin_path)
    dark_frames = []
    bp_frames = []
    filelist = []
    for f in trad_dark_data_filelist: #os.listdir(trad_dark_raw_datadir):
        # filelist.append(os.path.join(trad_dark_raw_datadir, f))
        # file = os.path.join(trad_dark_raw_datadir, f)
        d = fits.getdata(f).astype(float) #file)
        d[telem_rows] = np.nan
        _, _, _, _, d0, bp0, _ = proc_dark.L1_to_L2a(d)
        d1, bp1, _ = proc_dark.L2a_to_L2b(d0, bp0)
        #d1 *= em_gain # undo gain division
        d1[telem_rows] = 0
        dark_frames.append(d1)
        bp_frames.append(bp1)

    # The last output of mean_combine() are useful for calibrate_darks
    # module in the calibration repository:
    mean_frame, _, mean_num_good_fr, _ = mean_combine(dark_frames, bp_frames)
    mean_frame[telem_rows] = np.nan
    # TVAC_dark_path = os.path.join(e2edata_dir, 'TV-20_EXCAM_noise_characterization', "results", "proc_cgi_frame_trad_dark.fits")
    # fits.writeto(TVAC_dark_path, mean_frame, overwrite=True)
    # np.save(TVAC_dark_path, trad_dark_data_filelist)
    # TVAC_dark_path = os.path.join(e2edata_dir, 'TV-20_EXCAM_noise_characterization', "results", "proc_cgi_frame_trad_dark.fits")
    with fits.open(generated_trad_dark_file) as trad_dark_fits:
        trad_dark_data = trad_dark_fits[1].data
    ###################
    
    ##### Check against TVAC traditional dark result

    TVAC_trad_dark = detector.slice_section(mean_frame, 'SCI', 'image')

    # Use allclose for float32 save/load and small numerical differences (rtol/atol=1e-5)
    assert np.allclose(TVAC_trad_dark, trad_dark_data, rtol=1e-5, atol=1e-5, equal_nan=True)
    trad_dark = data.Dark(generated_trad_dark_file)
    assert trad_dark.ext_hdr['BUNIT'] == 'detected electron'
    assert trad_dark.err_hdr['BUNIT'] == 'detected electron'
    test_filepath = trad_dark_data_filelist[-1].split('.fits')[0] + '_drk_cal.fits'
    test_filename = os.path.basename(test_filepath)
    test_filename = re.sub('_l[0-9].', '', test_filename)
    assert(trad_dark.filename == test_filename)
    print('e2e test for trad_dark_im calibration passed')

    check.compare_to_mocks_hdrs(generated_trad_dark_file)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)


if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.

    e2edata_dir =  '/Users/kevinludwick/Documents/DRP_E2E_Test_Files_v2/E2E_Test_Data' #'/home/jwang/Desktop/CGI_TVAC_Data/'

    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the build traditional dark end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    # args = ap.parse_args(args_here)
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_trad_dark(e2edata_dir, outputdir)
    test_trad_dark_im(e2edata_dir, outputdir)
