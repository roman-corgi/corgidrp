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

thisfile_dir = os.path.dirname(__file__) # this file's folder


@pytest.mark.e2e
def test_l1_to_l2a(tvacdata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L1")
    l2a_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L2a")
    nonlin_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals", "nonlin_table_240322.txt")

    # make output directory if needed
    l2a_outputdir = os.path.join(e2eoutput_path, "l1_to_l2a_output")
    if not os.path.exists(l2a_outputdir):
        os.mkdir(l2a_outputdir)

    # define the raw science data to process

    l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]] # just grab the first two files
    mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]] # grab the last two real data to mock the calibration 
    tvac_l2a_filelist = [os.path.join(l2a_datadir, "{0}.fits".format(i)) for i in [90528, 90530]] # just grab the first two files

    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=l2a_outputdir, filename="mock_nonlinearcal.fits" )

    # add calibration file to caldb
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(nonlinear_cal)

    ####### Run the walker on some test_data

    walker.walk_corgidrp(l1_data_filelist, "", l2a_outputdir, template="l1_to_l2a_basic.json")

    # clean up by removing entry
    this_caldb.remove_entry(nonlinear_cal)

    
    ##### Check against TVAC data
    new_l2a_filenames = [os.path.join(l2a_outputdir, "{0}.fits".format(i)) for i in [90499, 90500]]

    for new_filename, tvac_filename in zip(new_l2a_filenames, tvac_l2a_filelist):
        img = data.Image(new_filename)

        with fits.open(tvac_filename) as hdulist:
            tvac_dat = hdulist[1].data
        
        diff = img.data - tvac_dat

        assert np.all(np.abs(diff) < 1e-5)

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
    test_l1_to_l2a(tvacdata_dir, outputdir)