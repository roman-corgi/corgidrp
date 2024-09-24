import argparse
import os
import pytest
import json
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.detector as detector
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_trad_dark(tvacdata_path, e2eoutput_path):
    '''There is no official II&T code for creating a "traditional" master dark (i.e., a dark made from taking the 
    mean of several darks at the same EM gain and exposure time), but all the parts are there in proc_cgi_frame.  
    So this function compares the DRP's output of build_trad_dark()
    to the output of CGI_TVAC_Data/TV-20_EXCAM_noise_characterization/results/run_TVAC_data_ENG_code_trad_dark.py, 
    which uses proc_cgi_frame code to make a traditional master dark. 

    Args:
        tvacdata_path (str): path to TVAC data root directory
        e2eoutput_path (str): path to output files made by this test
    '''
    # figure out paths, assuming everything is located in the same relative location    
    trad_dark_raw_datadir = os.path.join(tvacdata_path, 'TV-20_EXCAM_noise_characterization', 'darkmap')
    #TVAC_dark_path = os.path.join(tvacdata_path, 'TV-20_EXCAM_noise_characterization', "results", "dark_current_20240322.fits")
    TVAC_dark_path = os.path.join(tvacdata_path, 'TV-20_EXCAM_noise_characterization', "results", "proc_cgi_frame_trad_dark.fits")

    processed_cal_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals")
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_current_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")

    # make output directory if needed
    build_trad_dark_outputdir = os.path.join(e2eoutput_path, "build_trad_dark_output")
    if not os.path.exists(build_trad_dark_outputdir):
        os.mkdir(build_trad_dark_outputdir)

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

    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    # dummy data; basically just need the header info to combine with II&T nonlin calibration
    l1_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L1")
    mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]]
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=build_trad_dark_outputdir, filename="mock_nonlinearcal.fits" )


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
    err_hdr['BUNIT'] = 'detected EM electrons'
    # from CGI_TVAC_Data/TV-20_EXCAM_noise_characterization/tvac_noisemap_original_data/results/bias_offset.txt
    ext_hdr['B_O'] = 0 # bias offset not simulated in the data, so set to 0;  -0.0394 DN from tvac_noisemap_original_data/results
    ext_hdr['B_O_ERR'] = 0 # was not estimated with the II&T code

    # Create a DetectorNoiseMaps object and save it
    noise_maps = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                        input_dataset=mock_input_dataset, err=noise_map_noise,
                                        dq=noise_map_dq, err_hdr=err_hdr)
    noise_maps.save(filedir=build_trad_dark_outputdir, filename="mock_detnoisemaps.fits")
    
    # create a DetectorParams object and save it
    detector_params = data.DetectorParams({})
    detector_params.save(filedir=build_trad_dark_outputdir, filename="detector_params.fits")

    # create a k gain object and save it
    kgain_dat = np.array([[8.7]])
    kgain = data.KGain(kgain_dat,
                                pri_hdr=pri_hdr,
                                ext_hdr=ext_hdr,
                                input_dataset=mock_input_dataset)
    kgain.save(filedir=build_trad_dark_outputdir, filename="mock_kgain.fits")

    # add calibration files to caldb
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(nonlinear_cal)
    this_caldb.create_entry(noise_maps)
    this_caldb.create_entry(kgain)
    this_caldb.create_entry(detector_params)

    ####### Run the walker on some test_data; use template in recipes folder, so we can use walk_corgidrp()
    walker.walk_corgidrp(trad_dark_data_filelist, "", build_trad_dark_outputdir, template="build_trad_dark_full_frame.json")

    # clean up by removing entry
    this_caldb.remove_entry(nonlinear_cal)
    this_caldb.remove_entry(noise_maps)
    this_caldb.remove_entry(kgain)
    this_caldb.remove_entry(detector_params)
    # find cal file (naming convention for data.Dark class)
    generated_trad_dark_file = trad_dark_filename[:-5]+'_dark.fits'
    generated_trad_dark_file = os.path.join(build_trad_dark_outputdir, generated_trad_dark_file) 
    # Load
    trad_dark = fits.getdata(generated_trad_dark_file.replace("_L1_", "_L2a_", 1)) 
    
    ##### Check against TVAC traditional dark result
    TVAC_trad_dark = fits.getdata(TVAC_dark_path)
    # clean_trad_dark = (trad_dark.data - fpn_dat/1.340000033378601 - cic_dat)/100
    # master darks are exactly equal, ignoring the telemetry rows (first and last in this case)
    assert(np.allclose(TVAC_trad_dark[1:-1], trad_dark[1:-1], atol=1e-10))
    
    # remove from caldb
    trad_dark = data.Dark(generated_trad_dark_file.replace("_L1_", "_L2a_", 1))
    this_caldb.remove_entry(trad_dark)

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.

    tvacdata_dir = "/home/maxmb/Data/corgidrp/CGI_TVAC_Data/"

    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the build traditional dark end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args_here = ['--tvacdata_dir', tvacdata_dir, '--outputdir', outputdir]
    #args = ap.parse_args()
    args = ap.parse_args(args_here)
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    test_trad_dark(tvacdata_dir, outputdir)
