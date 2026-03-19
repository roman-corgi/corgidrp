import argparse
import os
import glob
import pytest
import os, shutil
import numpy as np
import scipy.ndimage
import datetime
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import re
import logging
import warnings
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector
import warnings
import corgidrp.check as check

thisfile_dir = os.path.dirname(__file__) # this file's folder

#Get path to this file
current_file_path = os.path.dirname(os.path.abspath(__file__))
thisfile_dir = os.path.dirname(__file__)

def fix_str_for_tvac(
    list_of_fits,
    ):
    """ 
    Gets around EMGAIN_A being set to 1 in TVAC data.
    
    Args:
        list_of_fits (list): list of FITS files that need to be updated.
    """
    for file in list_of_fits:
        fits_file = fits.open(file)
        exthdr = fits_file[1].header
        if float(exthdr['EMGAIN_A']) == 1:
            exthdr['EMGAIN_A'] = -1 #for new SSC-updated TVAC files which have EMGAIN_A by default as 1 regardless of the commanded EM gain
        if type(exthdr['EMGAIN_C']) is str:
            exthdr['EMGAIN_C'] = float(exthdr['EMGAIN_C'])
        # Update FITS file
        fits_file.writeto(file, overwrite=True)


@pytest.mark.e2e
def test_flat_creation_neptune(e2edata_path, e2eoutput_path):
    """
    Tests e2e flat field using Neptune in Band 4, full FOV

    Args:
        e2edata_path (str): path to L1 data files
        e2eoutput_path (str): output directory
    """
    # create output dir first (delete existing to start fresh)
    output_dir = os.path.join(e2eoutput_path, "flatfield_cal_e2e")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # set up logging
    global logger
    log_file = os.path.join(output_dir, 'flatfield_cal_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('flatfield_e2e')
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info('='*80)
    logger.info('Flat Calibration END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    logger.info('='*80)
    logger.info('Pre-test: set up input files and save to disk')
    logger.info('='*80)

     # figure out paths, assuming everything is located in the same relative location
    l1_dark_datadir = os.path.join(e2edata_path, "TV-20_EXCAM_noise_characterization", "darkmap")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")

    # clear and recreate output directory
    flat_outputdir = os.path.join(e2eoutput_path, "flatfield_cal_e2e/flat_neptune_output")
    if os.path.exists(flat_outputdir):
        shutil.rmtree(flat_outputdir)
    os.makedirs(flat_outputdir)
    flat_mock_inputdir = os.path.join(flat_outputdir, "input_l1")
    os.makedirs(flat_mock_inputdir, exist_ok=True)  
    
    l2a_mock_outdir = os.path.join(flat_outputdir, "output_l2a")
    os.makedirs(l2a_mock_outdir, exist_ok=True) 
    
    calibrations_dir = os.path.join(flat_outputdir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)

    logger.info('='*80)
    logger.info('Pre-test: set up input calibration files')
    logger.info('='*80)

    # assume all cals are in the same directory
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")

     # mock flat field is all ones
    input_flat = np.ones([1024, 1024], dtype=float)
    # input_flat = np.random.normal(1,.03,(1024, 1024))
    # create mock onsky rasters
    hstdata_filedir = os.path.join(thisfile_dir,"..", "test_data")
    hstdata_filenames = glob.glob(os.path.join(hstdata_filedir, "med*.fits"))
    hstdata_dataset = data.Dataset(hstdata_filenames)

    raster_dataset = mocks.create_onsky_rasterscans(hstdata_dataset, planet='neptune', band='4', im_size=1024, d=50, n_dith=3, 
                                                    radius=54, snr=25000, snr_constant=4.95, flat_map=input_flat, 
                                                    raster_radius=40, raster_subexps=6)
    
    logger.info('='*80)
    logger.info('Test Case 1: Raster scanned image for Neptune')
    logger.info('='*80)
    
    # raw science data to mock from
    l1_dark_filelist = glob.glob(os.path.join(l1_dark_datadir, "cgi_*.fits"))
    fix_str_for_tvac(l1_dark_filelist)
    l1_dark_filelist.sort()
    # l1_dark_dataset = data.Dataset(l1_dark_filelist[:len(raster_dataset)])
    l1_dark_dataset = mocks.create_prescan_files(numfiles=len(raster_dataset))
    # determine average noise
    noise_map = np.std(l1_dark_dataset.all_data, axis=0)
    r0c0 = detector.detector_areas["SCI"]["image"]['r0c0']
    rows = detector.detector_areas["SCI"]["image"]['rows']
    cols = detector.detector_areas["SCI"]["image"]['cols']
    avg_noise = np.mean(noise_map[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols])
    target_snr = 250/np.sqrt(4.95) # per pix

    # Create properly formatted CGI filenames
    base_time = datetime.datetime.now()
    l1_flat_dataset = []

    for i in range(len(raster_dataset)):
        base_image = l1_dark_dataset[i % len(l1_dark_dataset)].copy()
        base_image.pri_hdr['TARGET'] = "Neptune"
        base_image.ext_hdr['CFAMNAME'] = "4F"
        base_image.pri_hdr['VISTYPE'] = "CGIVST_CAL_FLAT"
        base_image.ext_hdr['EXPTIME'] = 60. # needed to mitigate desmear processing effect
        base_image.ext_hdr['DPAMNAME'] = "Imaging"
        base_image.data = base_image.data.astype(float)
       
       
        # Generate CGI filename with correct timestamp format
        visitid = base_image.pri_hdr.get('VISITID', '0089001001001001027')
        unique_time = base_time + datetime.timedelta(seconds=i)
        time_str = data.format_ftimeutc(unique_time.isoformat())
        base_image.filename = f"cgi_{visitid}_{time_str}_l1_.fits"

        # scale the raster image by the noise to reach a desired snr
        raster_frame = raster_dataset[i].data
        scale_factor = target_snr * avg_noise / np.percentile(raster_frame, 99)
        # get the location to inject the raster image into
        x_start = r0c0[1] + cols//2 - raster_frame.shape[1]//2
        y_start = r0c0[0] + rows//2 - raster_frame.shape[0]//2
        x_end = x_start + raster_frame.shape[1]
        y_end = y_start + raster_frame.shape[0] 

        base_image.data[y_start:y_end, x_start:x_end] += raster_frame * scale_factor
        l1_flat_dataset.append(base_image)
    
    l1_flat_dataset = data.Dataset(l1_flat_dataset)
    l1_flat_dataset.save(filedir=flat_mock_inputdir)
    l1_flatfield_filelist = glob.glob(os.path.join(flat_mock_inputdir, "*.fits"))
    l1_flatfield_filelist.sort()

    # define the raw science data to process

    mock_cal_filelist = l1_dark_filelist[-2:]

    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make calibration files using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=calibrations_dir, filename="cgi_0000000000000000000_20251031t1200000_nln_cal.fits" )
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    kgain_val = 8.7
    # add in keywords not provided by create_default_L1_headers() (since L1 headers are simulated from that function)
    ext_hdr['RN'] = 100.
    ext_hdr['RN_ERR'] = 0.
    signal_array = np.linspace(0, 50)
    noise_array = np.sqrt(signal_array)
    ptc = np.column_stack([signal_array, noise_array])
    kgain = data.KGain(kgain_val, ptc=ptc, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=calibrations_dir, filename="cgi_0000000000000000000_20251031t1200000_krn_cal.fits")
    this_caldb.create_entry(kgain, to_disk=False)
    this_caldb.save()

    # NoiseMap
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_path) as hdulist:
        dark_dat = hdulist[0].data
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    ext_hdr['B_O'] = 0.
    ext_hdr['B_O_ERR'] = 0.
    ext_hdr['MJDSRT'] -= 1. # make this earlier in time than the flat field header
    ext_hdr['MJDEND'] -= 1.
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=calibrations_dir, filename="cgi_0000000000000000000_20251031t1200000_dnm_cal.fits")
    this_caldb.create_entry(noise_map)

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)


     ####### Run the DRP walker for flatfield neptune
    logger.info('='*80)
    logger.info('Running processing pipeline')
    logger.info('='*80)
    logger.info('Step 1: Processing L1 -> Flat Field')
    
    # Step 1: Process L1 to L2a
    with warnings.catch_warnings():  
        warnings.filterwarnings('ignore', category=UserWarning)# prevent UserWarning: Number of frames which made the DetectorNoiseMaps product is less than the number of frames in input_dataset
        walker.walk_corgidrp(l1_flatfield_filelist, "", flat_outputdir)

    
    ####### Test the flat field result
    # the requirement: <=0.71% error per resolution element
    flat_filename = l1_flatfield_filelist[-1].split(os.path.sep)[-1].replace("_l1_", "_flt_cal")
    flat = data.FlatField(os.path.join(flat_outputdir, flat_filename))
    good_region = np.where(flat.data != 1)
    diff = flat.data - input_flat # compute residual from true
    smoothed_diff = scipy.ndimage.gaussian_filter(diff, 1.4) # smooth by the size of the resolution element, since we care about that
    print(np.std(smoothed_diff[good_region]))
    assert np.std(smoothed_diff[good_region]) < 0.0071

    logger.info('Processing flat field complete')

    check.compare_to_mocks_hdrs(os.path.join(flat_outputdir, flat_filename))

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

@pytest.mark.e2e
def test_flat_creation_uranus(e2edata_path, e2eoutput_path):
    """
    Tests e2e flat field using Uranus in Band 1, only HLC FOV

    Args:
        e2edata_path (str): path to L1 data files
        e2eoutput_path (str): output directory
    """
    # create output dir first, don't delete since Neptune e2e test should have just run.
    output_dir = os.path.join(e2eoutput_path, "flatfield_cal_e2e")
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set up logging
    global logger
    log_file = os.path.join(output_dir, 'flatfield_cal_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('flatfield_e2e')
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info('='*80)
    logger.info('Flat Calibration END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    logger.info('='*80)
    logger.info('Pre-test: set up input files and save to disk')
    logger.info('='*80)

     # figure out paths, assuming everything is located in the same relative location
    l1_dark_datadir = os.path.join(e2edata_path, "TV-20_EXCAM_noise_characterization", "darkmap")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")


    # clear and recreate output directory
    flat_outputdir = os.path.join(e2eoutput_path, "flatfield_cal_e2e/flat_uranus_output")
    if os.path.exists(flat_outputdir):
        shutil.rmtree(flat_outputdir)
    os.makedirs(flat_outputdir)
    flat_mock_inputdir = os.path.join(flat_outputdir, "input_l1")
    os.makedirs(flat_mock_inputdir, exist_ok=True)  
    
    l2a_mock_outdir = os.path.join(flat_outputdir, "output_l2a")
    os.makedirs(l2a_mock_outdir, exist_ok=True) 
    
    calibrations_dir = os.path.join(flat_outputdir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)

    logger.info('='*80)
    logger.info('Pre-test: set up input calibration files')
    logger.info('='*80)

    # assume all cals are in the same directory
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")

     # mock flat field is all ones
    input_flat = np.ones([1024, 1024], dtype=float)
    # input_flat = np.random.normal(1,.03,(1024, 1024))
    # create mock onsky rasters
    hstdata_filedir = os.path.join(thisfile_dir,"..", "test_data")
    hstdata_filenames = glob.glob(os.path.join(hstdata_filedir, "med*.fits"))
    hstdata_dataset = data.Dataset(hstdata_filenames)
    raster_dataset = mocks.create_onsky_rasterscans(hstdata_dataset, planet='uranus',band='1',im_size=1024, d=65, n_dith=1, 
                                                    radius=90, snr=250000, snr_constant=9.66, flat_map=input_flat, 
                                                    raster_radius=40, raster_subexps=6)
    logger.info('='*80)
    logger.info('Test Case 1: Raster scanned image for Uranus')
    logger.info('='*80)
    
    # raw science data to mock from
    l1_dark_filelist = glob.glob(os.path.join(l1_dark_datadir, "cgi_*.fits"))
    fix_str_for_tvac(l1_dark_filelist)
    l1_dark_filelist.sort()
    # l1_dark_dataset = data.Dataset(l1_dark_filelist[:len(raster_dataset)])
    l1_dark_dataset = mocks.create_prescan_files(numfiles=len(raster_dataset))
    # determine average noise
    noise_map = np.std(l1_dark_dataset.all_data, axis=0)
    r0c0 = detector.detector_areas["SCI"]["image"]['r0c0']
    rows = detector.detector_areas["SCI"]["image"]['rows']
    cols = detector.detector_areas["SCI"]["image"]['cols']
    avg_noise = np.mean(noise_map[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols])
    target_snr = 250/np.sqrt(4.95) # per pix

    # Create properly formatted CGI filenames
    base_time = datetime.datetime.now()
    l1_flat_dataset = []

    for i in range(len(raster_dataset)):
        base_image = l1_dark_dataset[i % len(l1_dark_dataset)].copy()
        base_image.pri_hdr['TARGET'] = "Uranus"
        base_image.ext_hdr['CFAMNAME'] = "1F"
        base_image.pri_hdr['VISTYPE'] = "CGIVST_CAL_FLAT"
        base_image.ext_hdr['EXPTIME'] = 60. # needed to mitigate desmear processing effect
        base_image.ext_hdr['DPAMNAME'] = "Imaging"
        base_image.data = base_image.data.astype(float)
       
        # Generate CGI filename with correct timestamp format
        visitid = base_image.pri_hdr.get('VISITID', '0089001001001001027')
        unique_time = base_time + datetime.timedelta(seconds=i)
        time_str = data.format_ftimeutc(unique_time.isoformat())
        base_image.filename = f"cgi_{visitid}_{time_str}_l1_.fits"

        # scale the raster image by the noise to reach a desired snr
        raster_frame = raster_dataset[i].data
        scale_factor = target_snr * avg_noise / np.percentile(raster_frame, 99)
        # get the location to inject the raster image into
        x_start = r0c0[1] + cols//2 - raster_frame.shape[1]//2
        y_start = r0c0[0] + rows//2 - raster_frame.shape[0]//2
        x_end = x_start + raster_frame.shape[1]
        y_end = y_start + raster_frame.shape[0] 

        base_image.data[y_start:y_end, x_start:x_end] += raster_frame * scale_factor
        l1_flat_dataset.append(base_image)
    
    l1_flat_dataset = data.Dataset(l1_flat_dataset)
    l1_flat_dataset.save(filedir=flat_mock_inputdir)
    l1_flatfield_filelist = glob.glob(os.path.join(flat_mock_inputdir, "*.fits"))
    l1_flatfield_filelist.sort()

    # define the raw science data to process

    mock_cal_filelist = l1_dark_filelist[-2:]

    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make calibration files using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=calibrations_dir, filename="cgi_0000000000000000000_20251031t1200000_nln_cal.fits" )
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    kgain_val = 8.7
    # add in keywords not provided by create_default_L1_headers() (since L1 headers are simulated from that function)
    ext_hdr['RN'] = 100.
    ext_hdr['RN_ERR'] = 0.
    signal_array = np.linspace(0, 50)
    noise_array = np.sqrt(signal_array)
    ptc = np.column_stack([signal_array, noise_array])
    kgain = data.KGain(kgain_val, ptc=ptc, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=calibrations_dir, filename="cgi_0000000000000000000_20251031t1200000_krn_cal.fits")
    this_caldb.create_entry(kgain, to_disk=False)
    this_caldb.save()

    # NoiseMap
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_path) as hdulist:
        dark_dat = hdulist[0].data
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    ext_hdr['B_O'] = 0.
    ext_hdr['B_O_ERR'] = 0.
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=calibrations_dir, filename="cgi_0000000000000000000_20251031t1200000_dnm_cal.fits")
    this_caldb.create_entry(noise_map)

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)


     ####### Run the DRP walker for flatfield
    logger.info('='*80)
    logger.info('Running processing pipeline')
    logger.info('='*80)
    logger.info('Step 1: Processing L1 -> Flat Field')
    
    # Step 1: Process L1 to L2a
    with warnings.catch_warnings():  
        warnings.filterwarnings('ignore', category=UserWarning)# prevent UserWarning: Number of frames which made the DetectorNoiseMaps product is less than the number of frames in input_dataset
        walker.walk_corgidrp(l1_flatfield_filelist, "", flat_outputdir)

    ####### Test the flat field result
    # the requirement: <=0.71% error per resolution element
    flat_filename = l1_flatfield_filelist[-1].split(os.path.sep)[-1].replace("_l1_", "_flt_cal")
    flat = data.FlatField(os.path.join(flat_outputdir, flat_filename))
    good_region = np.where(flat.data != 1)
    diff = flat.data - input_flat # compute residual from true
    smoothed_diff = scipy.ndimage.gaussian_filter(diff, 1.4) # smooth by the size of the resolution element, since we care about that
    print(np.std(smoothed_diff[good_region]))
    assert np.std(smoothed_diff[good_region]) < 0.0071

    logger.info('Processing flat field complete')

if __name__ == "__main__":
    
    outputdir = thisfile_dir
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'

    ap = argparse.ArgumentParser(description="run the l2b-> Flatfield end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    
    test_flat_creation_neptune(e2edata_dir, outputdir)
    test_flat_creation_uranus(e2edata_dir, outputdir)
