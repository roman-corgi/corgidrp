import argparse
import os
import glob
import pytest
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_flat_creation(tvacdata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_dark_datadir = os.path.join(tvacdata_path, "TV-20_EXCAM_noise_characterization", "darkmap")
    processed_cal_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals")

    # make output directory if needed
    flat_outputdir = os.path.join(e2eoutput_path, "flat_output")
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
    raster_dataset = mocks.create_onsky_rasterscans(hstdata_dataset, planet='neptune', band='1', im_size=1024, d=50, n_dith=3, numfiles=36, radius=54, snr=25000, snr_constant=4.55, flat_map=input_flat, raster_radius=95)
    # raw science data to mock from
    l1_dark_filelist = glob.glob(os.path.join(l1_dark_datadir, "CGI_*.fits"))
    l1_dark_filelist.sort()
    # l1_dark_dataset = data.Dataset(l1_dark_filelist[:len(raster_dataset)])
    l1_dark_dataset = mocks.create_prescan_files(numfiles=len(raster_dataset))
    # determine average noise
    noise_map = np.std(l1_dark_dataset.all_data, axis=0)
    r0c0 = detector.detector_areas["SCI"]["image"]['r0c0']
    rows = detector.detector_areas["SCI"]["image"]['rows']
    cols = detector.detector_areas["SCI"]["image"]['cols']
    avg_noise = np.mean(noise_map[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols])
    target_snr = 200 # per pix
    for i in range(len(l1_dark_dataset)):
        l1_dark_dataset[i].pri_hdr['TARGET'] = "Neptune"
        l1_dark_dataset[i].pri_hdr['FILTER'] = 1
        l1_dark_dataset[i].pri_hdr['OBSTYPE'] = "FLT"
        l1_dark_dataset[i].data = l1_dark_dataset[i].data.astype(float)
        l1_dark_dataset[i].filename = l1_dark_filelist[i].split(os.path.sep)[-1]

        # scale the raster image by the noise to reach a desired snr
        raster_frame = raster_dataset[i].data
        scale_factor = target_snr * avg_noise / np.percentile(raster_frame, 99)
        # get the location to inject the raster image into
        x_start = r0c0[1] + cols//2 - raster_frame.shape[1]//2
        y_start = r0c0[0] + rows//2 - raster_frame.shape[0]//2
        x_end = x_start + raster_frame.shape[1]
        y_end = y_start + raster_frame.shape[0] 

        l1_dark_dataset[i].data[y_start:y_end, x_start:x_end] += raster_frame * scale_factor

    l1_dark_dataset.save(filedir=flat_mock_inputdir)
    l1_flatfield_filelist = glob.glob(os.path.join(flat_mock_inputdir, "*.fits"))
    l1_flatfield_filelist.sort()
    
    # define the raw science data to process

    mock_cal_filelist = l1_dark_filelist[-2:]

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
    nonlinear_cal.save(filedir=flat_outputdir, filename="mock_nonlinearcal.fits" )
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    kgain_val = 8.7
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=flat_outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)

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
    err_hdr['BUNIT'] = 'detected EM electrons'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=flat_outputdir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    ####### Run the walker on some test_data

    recipe = walker.autogen_recipe(l1_flatfield_filelist, flat_outputdir)
     ### Modify they keywords of some of the steps
    for step in recipe['steps']:
        if step['name'] in ["correct_nonlinearity", "desmear", "cti_correction"]:
            step['skip'] = True
    walker.run_recipe(recipe, save_recipe_file=True)


    ####### Test the result
    flat_filename = l1_flatfield_filelist[0].split(os.path.sep)[-1].replace("_L1_", "_L2a_")[:-5] + "_flatfield.fits"
    flat = data.FlatField(os.path.join(flat_outputdir, flat_filename))
    good_region = np.where(flat.data != 1)
    diff = flat.data - input_flat
    print(np.std(diff[good_region]))
    assert np.std(diff[good_region]) < 0.0071

    # clean up by removing entry
    this_caldb.remove_entry(nonlinear_cal)
    this_caldb.remove_entry(kgain)
    this_caldb.remove_entry(noise_map)



if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    tvacdata_dir = "/home/jwang/Jason/Documents/DataCopy/corgi/CGI_TVAC_Data"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    test_flat_creation(tvacdata_dir, outputdir)