import argparse
import os
import glob
import pytest
import numpy as np
import scipy.ndimage
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import re
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector

thisfile_dir = os.path.dirname(__file__) # this file's folder


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
        e2edata_path (str): path to CGI_TVAC_Data dir
        e2eoutput_path (str): output directory
    """
    # figure out paths, assuming everything is located in the same relative location
    l1_dark_datadir = os.path.join(e2edata_path, "TV-20_EXCAM_noise_characterization", "darkmap")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")

    # make output directory if needed
    flat_outputdir = os.path.join(e2eoutput_path, "flat_neptune_output")
    if not os.path.exists(flat_outputdir):
        os.mkdir(flat_outputdir)
    flat_mock_inputdir = os.path.join(flat_outputdir, "mock_input_data")
    if not os.path.exists(flat_mock_inputdir):
        os.mkdir(flat_mock_inputdir)    

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

    # change the UTC time using the UTC time from the first time as the start
    #start_filenum = int(l1_dark_filelist[0][:-5].split("_")[-1])
    #base_filename = l1_dark_filelist[0].split(os.path.sep)[-1][:-15]
    l1_dark_st_filename = l1_dark_filelist[0].split(os.path.sep)[-1]
    match = re.findall(r'\d{2,}', l1_dark_st_filename)
    last_num_str = match[-1] if match else None
    start_utc = int(last_num_str)
    l1_flat_dataset = []
    for i in range(len(raster_dataset)):
        base_image = l1_dark_dataset[i % len(l1_dark_dataset)].copy()
        base_image.pri_hdr['TARGET'] = "Neptune"
        base_image.ext_hdr['CFAMNAME'] = "4F"
        base_image.pri_hdr['VISTYPE'] = "FFIELD"
        base_image.ext_hdr['EXPTIME'] = 60 # needed to mitigate desmear processing effect
        base_image.data = base_image.data.astype(float)
        # add 1 millisecond each time to UTC time
        base_image.filename = l1_dark_st_filename.replace(last_num_str, str(start_utc + i))

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
    nonlinear_cal.save(filedir=flat_outputdir, filename="mock_nonlinearcal.fits" )
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    kgain_val = 8.7
    # add in keywords not provided by create_default_L1_headers() (since L1 headers are simulated from that function)
    ext_hdr['RN'] = 100
    ext_hdr['RN_ERR'] = 0
    signal_array = np.linspace(0, 50)
    noise_array = np.sqrt(signal_array)
    ptc = np.column_stack([signal_array, noise_array])
    kgain = data.KGain(kgain_val, ptc=ptc, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=flat_outputdir, filename="mock_kgain.fits")
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
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=flat_outputdir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    ####### Run the walker on some test_data

    recipe = walker.autogen_recipe(l1_flatfield_filelist, flat_outputdir)
     ### Modify they keywords of some of the steps
    for step in recipe['steps']:
        # if step['name'] in ["desmear", "cti_correction"]:
        #     step['skip'] = True
        if step['name'] == "create_onsky_flatfield":
            step['keywords']['n_pix'] = 165 # full shaped pupil FOV
    walker.run_recipe(recipe, save_recipe_file=True)


    ####### Test the flat field result
    # the requirement: <=0.71% error per resolution element
    flat_filename = l1_flatfield_filelist[-1].split(os.path.sep)[-1].replace("_l1_", "_flt_cal")
    flat = data.FlatField(os.path.join(flat_outputdir, flat_filename))
    good_region = np.where(flat.data != 1)
    diff = flat.data - input_flat # compute residual from true
    smoothed_diff = scipy.ndimage.gaussian_filter(diff, 1.4) # smooth by the size of the resolution element, since we care about that
    print(np.std(smoothed_diff[good_region]))
    assert np.std(smoothed_diff[good_region]) < 0.0071


    ####### Check the bad pixel map result
    bp_map_filename = l1_flatfield_filelist[-1].split(os.path.sep)[-1].replace("_l1_", "_bpm_cal")
    bpmap = data.BadPixelMap(os.path.join(flat_outputdir, bp_map_filename))
    assert np.all(bpmap.data == 0) # this bpmap should have no bad pixels

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)


@pytest.mark.e2e
def test_flat_creation_uranus(e2edata_path, e2eoutput_path):
    """
    Tests e2e flat field using Uranus in Band 1, only HLC FOV

    Args:
        e2edata_path (str): path to CGI_TVAC_Data dir
        e2eoutput_path (str): output directory
    """
    # figure out paths, assuming everything is located in the same relative location
    l1_dark_datadir = os.path.join(e2edata_path, "TV-20_EXCAM_noise_characterization", "darkmap")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")

    # make output directory if needed
    flat_outputdir = os.path.join(e2eoutput_path, "flat_uranus_output")
    if not os.path.exists(flat_outputdir):
        os.mkdir(flat_outputdir)
    flat_mock_inputdir = os.path.join(flat_outputdir, "mock_input_data")
    if not os.path.exists(flat_mock_inputdir):
        os.mkdir(flat_mock_inputdir)    

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
    # raw science data to mock from
    l1_dark_filelist = glob.glob(os.path.join(l1_dark_datadir, "cgi_*.fits"))
    l1_dark_filelist.sort()
    # l1_dark_dataset = data.Dataset(l1_dark_filelist[:len(raster_dataset)])
    l1_dark_dataset = mocks.create_prescan_files(numfiles=len(raster_dataset))
    # determine average noise of the L1 dark images, so we can scale the images appropriately.
    noise_map = np.std(l1_dark_dataset.all_data, axis=0)
    r0c0 = detector.detector_areas["SCI"]["image"]['r0c0']
    rows = detector.detector_areas["SCI"]["image"]['rows']
    cols = detector.detector_areas["SCI"]["image"]['cols']
    avg_noise = np.mean(noise_map[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols])
    target_snr = 250/np.sqrt(4.95) # per pix

    # change the UTC time using the UTC time from the first time as the start
    #start_filenum = int(l1_dark_filelist[0][:-5].split("_")[-1])
    #base_filename = l1_dark_filelist[0].split(os.path.sep)[-1][:-15]
    l1_dark_st_filename = l1_dark_filelist[0].split(os.path.sep)[-1]
    match = re.findall(r'\d{2,}', l1_dark_st_filename)
    last_num_str = match[-1] if match else None
    start_utc = int(last_num_str)
    l1_flat_dataset = []
    for i in range(len(raster_dataset)):
        base_image = l1_dark_dataset[i % len(l1_dark_dataset)].copy()
        base_image.pri_hdr['TARGET'] = "Uranus"
        base_image.ext_hdr['CFAMNAME'] = "1F"
        base_image.pri_hdr['VISTYPE'] = "FFIELD"
        base_image.ext_hdr['EXPTIME'] = 60 # needed to mitigate desmear processing effect
        base_image.data = base_image.data.astype(float)
        # add 1 millisecond each time to UTC time
        base_image.filename = l1_dark_st_filename.replace(last_num_str, str(start_utc + i))

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
    nonlinear_cal.save(filedir=flat_outputdir, filename="mock_nonlinearcal.fits" )
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    kgain_val = 8.7
    # add in keywords not provided by create_default_L1_headers() (since L1 headers are simulated from that function)
    ext_hdr['RN'] = 100
    ext_hdr['RN_ERR'] = 0
    signal_array = np.linspace(0, 50)
    noise_array = np.sqrt(signal_array)
    ptc = np.column_stack([signal_array, noise_array])
    kgain = data.KGain(kgain_val, ptc=ptc, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=flat_outputdir, filename="mock_kgain.fits")
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
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=flat_outputdir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    
    ####### Run the walker on some test_data

    recipe = walker.autogen_recipe(l1_flatfield_filelist, flat_outputdir)
    walker.run_recipe(recipe, save_recipe_file=True)


    ####### Test the result
    # the requirement: <=0.71% error per resolution element
    flat_filename = l1_flatfield_filelist[-1].split(os.path.sep)[-1].replace("_l1_", "_flt_cal")
    flat = data.FlatField(os.path.join(flat_outputdir, flat_filename))
    good_region = np.where(flat.data != 1)
    diff = flat.data - input_flat
    smoothed_diff = scipy.ndimage.gaussian_filter(diff, 1.4) # smooth by the size of the resolution element, since we care about that
    print(np.std(smoothed_diff[good_region]))
    assert np.std(smoothed_diff[good_region]) < 0.0071

    ####### Check the bad pixel map result
    bp_map_filename = l1_flatfield_filelist[-1].split(os.path.sep)[-1].replace("_l1_", "_bpm_cal")
    bpmap = data.BadPixelMap(os.path.join(flat_outputdir, bp_map_filename))
    assert np.all(bpmap.data == 0) # this bpmap should have no bad pixels

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)



if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    # e2edata_dir = '/home/jwang/Desktop/CGI_TVAC_Data/'
    e2edata_dir = '/Users/kevinludwick/Documents/ssc_tvac_test/E2E_Test_Data2'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_flat_creation_neptune(e2edata_dir, outputdir)
    test_flat_creation_uranus(e2edata_dir, outputdir)
    