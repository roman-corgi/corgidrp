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

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_l1_to_l2b(tvacdata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L1")
    l2b_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L2b")
    processed_cal_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals")

    # make output directory if needed
    l2b_outputdir = os.path.join(e2eoutput_path, "l1_to_l2b_output")
    if not os.path.exists(l2b_outputdir):
        os.mkdir(l2b_outputdir)

    # assume all cals are in the same directory
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_path = os.path.join(processed_cal_path, "Dark_map_240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    fpn_path = os.path.join(processed_cal_path, "FPN_map_240318.fits")
    cic_path = os.path.join(processed_cal_path, "CIC_map_240322.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")

    # define the raw science data to process

    l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]] # just grab the first two files
    mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]] # grab the last two real data to mock the calibration 
    tvac_l2b_filelist = [os.path.join(l2b_datadir, "{0}.fits".format(i)) for i in [90529, 90531]] # just grab the first two files


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
    nonlinear_cal.save(filedir=l2b_outputdir, filename="mock_nonlinearcal.fits" )
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    kgain_val = 8.7
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=l2b_outputdir, filename="mock_kgain.fits")
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
    noise_map.save(filedir=l2b_outputdir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    ## Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    flat.save(filedir=l2b_outputdir, filename="mock_flat.fits")
    this_caldb.create_entry(flat)

    # bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    bp_map.save(filedir=l2b_outputdir, filename="mock_bpmap.fits")
    this_caldb.create_entry(bp_map)

    ####### Run the walker on some test_data

    walker.walk_corgidrp(l1_data_filelist, "", l2b_outputdir)


    # clean up by removing entry
    this_caldb.remove_entry(nonlinear_cal)
    this_caldb.remove_entry(kgain)
    this_caldb.remove_entry(noise_map)
    this_caldb.remove_entry(flat)
    this_caldb.remove_entry(bp_map)

    ##### Check against TVAC data
    new_l2b_filenames = [os.path.join(l2b_outputdir, "{0}.fits".format(i)) for i in [90499, 90500]]

    for new_filename, tvac_filename in zip(new_l2b_filenames, tvac_l2b_filelist):
        img = data.Image(new_filename)

        with fits.open(tvac_filename) as hdulist:
            tvac_dat = hdulist[1].data
        
        diff = img.data - tvac_dat

        assert np.all(np.abs(diff) < 1e-5)

        # plotting script for debugging
        # import matplotlib.pylab as plt
        # fig = plt.figure(figsize=(10,3.5))
        # fig.add_subplot(131)
        # plt.imshow(img.data, vmin=-0.01, vmax=45, cmap="viridis")
        # plt.title("corgidrp")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # fig.add_subplot(132)
        # plt.imshow(tvac_dat, vmin=-0.01, vmax=45, cmap="viridis")
        # plt.title("TVAC")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # fig.add_subplot(133)
        # plt.imshow(diff, vmin=-0.01, vmax=0.01, cmap="inferno")
        # plt.title("difference")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # plt.show()

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    tvacdata_dir = "/Users/jmilton/Documents/CGI/CGI_TVAC_Data"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    test_l1_to_l2b(tvacdata_dir, outputdir)