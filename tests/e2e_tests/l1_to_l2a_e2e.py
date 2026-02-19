import argparse
import os
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
import corgidrp.check as check
import shutil

try:
    from proc_cgi_frame.gsw_process import Process
except:
    pass

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_l1_to_l2a(e2edata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    #l2a_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L2a")
    nonlin_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals", "nonlin_table_240322.txt")
    dark_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals", "dark_current_20240322.fits")
    fpn_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals", "fpn_20240322.fits")
    cic_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals", "cic_20240322.fits")

    # make output directory if needed
    l2a_outputdir = os.path.join(e2eoutput_path, "l1_to_l2a_e2e")
    if os.path.exists(l2a_outputdir):
        shutil.rmtree(l2a_outputdir)
    os.makedirs(l2a_outputdir)

    # Create input_data subfolder
    input_data_dir = os.path.join(l2a_outputdir, 'input_l1')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)
    
    # Create calibrations subfolder
    calibrations_dir = os.path.join(l2a_outputdir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)

    l2a_tvac_outputdir = os.path.join(l2a_outputdir, "tvac_reference_data")
    if not os.path.exists(l2a_tvac_outputdir):
        os.makedirs(l2a_tvac_outputdir)
    # clean up by removing old files
    for file in os.listdir(l2a_tvac_outputdir):
        os.remove(os.path.join(l2a_tvac_outputdir, file))

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB

    # define the raw science data to process

    l1_data_filelist = [os.path.join(l1_datadir, os.listdir(l1_datadir)[i]) for i in [0,1]] #[os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]] # just grab the first two files
    mock_cal_filelist = [os.path.join(l1_datadir, os.listdir(l1_datadir)[i]) for i in [-2,-1]] #[os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]] # grab the last two real data to mock the calibration

    # Update headers for TVAC files
    l1_data_filelist = check.fix_hdrs_for_tvac(l1_data_filelist, input_data_dir)
    #tvac_l2a_filelist = [os.path.join(l2a_datadir, os.listdir(l2a_datadir)[i]) for i in [0,1]] #[os.path.join(l2a_datadir, "{0}.fits".format(i)) for i in [90528, 90530]] # just grab the first two files
    # run the L1 data through the II&T code to process to L2a
    tvac_l2a_filelist = []
    bad_pix = np.zeros((1200,2200)) # what is used in DRP
    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    det_params = this_caldb.get_calib(None, data.DetectorParams)
    fwc_pp_e = int(det_params.params['FWC_PP_E']) # same as what is in DRP's DetectorParams
    fwc_em_e = int(det_params.params['FWC_EM_E']) # same as what is in DRP's DetectorParams
    telem_rows_start = det_params.params['TELRSTRT']
    telem_rows_end = det_params.params['TELREND']
    telem_rows = slice(telem_rows_start, telem_rows_end)
    for j, file in enumerate(l1_data_filelist):
        frame_data = fits.getdata(file)
        ext_hdr = fits.getheader(file, ext=1)
        exptime = ext_hdr['EXPTIME']
        em_gain = float(ext_hdr['EMGAIN_C'])
        eperdn = float(ext_hdr['KGAINPAR'])
        b_offset = 0 # what is used in DRP by default
        proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                b_offset, em_gain, exptime,
                nonlin_path)
        image, _, _, _, _, _, _ = proc.L1_to_L2a(frame_data)
        with fits.open(file) as hdul:
            hdul_copy = fits.HDUList([hdu.copy() for hdu in hdul])
            hdul_copy[1].data = image
            # not important to change headers in the way DRP would; we are just comparing data values
            l2a_tvac_filename = file.split(os.path.sep)[-1].replace('l1','l2a',1) 
            hdul_copy.writeto(os.path.join(l2a_tvac_outputdir, l2a_tvac_filename), overwrite=True)
        tvac_l2a_filelist.append(os.path.join(l2a_tvac_outputdir, l2a_tvac_filename))


    ###### Setup necessary calibration files
    # add calibration file to caldb

    # Create necessary calibration files
    # we are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")
    this_caldb.create_entry(nonlinear_cal)

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
    mocks.rename_files_to_cgi_format(list_of_fits=[noise_map], output_dir=calibrations_dir, level_suffix="dnm_cal")
    this_caldb.create_entry(noise_map)

    # KGain
    kgain_val = eperdn # 8.7 is what is in the TVAC headers
    signal_array = np.linspace(0, 50)
    noise_array = np.sqrt(signal_array)
    ptc = np.column_stack([signal_array, noise_array])
    kgain = data.KGain(kgain_val, ptc=ptc, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    this_caldb.create_entry(kgain)

    ####### Run the walker on some test_data

    walker.walk_corgidrp(l1_data_filelist, "", l2a_outputdir, template="l1_to_l2a_basic.json")
    
    ##### Check against TVAC data
    # Filter out calibration files and only get L2a data files
    all_files = [f for f in os.listdir(l2a_outputdir) if f.endswith('.fits')]
    new_l2a_filenames = [os.path.join(l2a_outputdir, f) for f in all_files if '_l2a' in f and '_cal' not in f]
    #new_l2a_filenames = [os.path.join(l2a_outputdir, "{0}.fits".format(i)) for i in [90499, 90500]]

    for new_filename, tvac_filename in zip(sorted(new_l2a_filenames), sorted(tvac_l2a_filelist)):
        img = data.Image(new_filename)

        with fits.open(tvac_filename) as hdulist:
            tvac_dat = hdulist[1].data
        
        diff = img.data - tvac_dat

        assert np.all(np.abs(diff) < 1e-5)
        
        check.compare_to_mocks_hdrs(new_filename, mocks.create_default_L2a_headers)
        
        # # plotting script for debugging
        # import matplotlib.pylab as plt
        # fig = plt.figure(figsize=(10,3.5))
        # fig.add_subplot(131)
        # plt.imshow(img.data, vmin=-20, vmax=2000, cmap="viridis")
        # plt.title("corgidrp")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # fig.add_subplot(132)
        # plt.imshow(tvac_dat, vmin=-20, vmax=2000, cmap="viridis")
        # plt.title("TVAC")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # fig.add_subplot(133)
        # plt.imshow(diff, vmin=-0.01, vmax=0.01, cmap="inferno")
        # plt.title("difference")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # plt.show()
    
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    #e2edata_dir = '/home/jwang/Desktop/CGI_TVAC_Data/'
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_l1_to_l2a(e2edata_dir, outputdir)