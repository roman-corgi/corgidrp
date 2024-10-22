import argparse
import glob
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

def process_one_dataset(input_filelist, cpgs_xml_filepath, outputdir, main_cal_dir=None, template=None):
    """
    Processes a filelist of data and saves data to outputdir

    Args:
        input_filelist (list): a list of filepaths for the input data
        cpgs_xml_filepath (str): path to CPGS XML file (currently not implemented)
        outputdir (str): filepath to directory to save data
        main_cal_dir (str): if specified, a directory containing processed and validated calibration files.
                            Any files in this directory will be added to the calibration database before
                            processing, if they aren't already in there
        template (str): if specified, a specific recipe to run. If not specified, the corgidrp.walker will
                        guess the apporpriate template
    """
    if main_cal_dir is not None:
        this_caldb = caldb.CalDB()
        this_caldb.scan_dir_for_new_entries(main_cal_dir)
    walker.walk_corgidrp(input_filelist, cpgs_xml_filepath, outputdir, template=template)

@pytest.mark.e2e
def test_l1_to_l2b_fromscratch(tvacdata_path, e2eoutput_path):

    # make output directory if needed
    outputdir = os.path.join(e2eoutput_path, "l1_to_l2b_fromscratch_output")
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    cal_outputdir = os.path.join(outputdir, "cals")
    if not os.path.exists(cal_outputdir):
        os.mkdir(cal_outputdir)

    #################################################
    ###### Process all necessary calibration files
    #################################################

    ##### Nonlinearity calibration
    # Identify filelist of L1 data for nonlinearity calibration
    nonlin_l1_datadir = os.path.join(tvacdata_path, 'TV-20_EXCAM_noise_characterization', 'nonlin')
    nonlin_l1_list = glob.glob(os.path.join(nonlin_l1_datadir, "*.fits"))
    nonlin_l1_list.sort()
    # Create nonlinearity calibration
    process_one_dataset(nonlin_l1_list, '', cal_outputdir)

    ##### KGain
    kgain_l1_datadir = os.path.join(tvacdata_path, "TV-20_EXCAM_noise_characterization", "kgain")
    kgain_l1_filelist = glob.glob(os.path.join(kgain_l1_datadir, "*.fits"))
    kgain_l1_filelist.sort() 
    # Crate Kgain calibration
    process_one_dataset(kgain_l1_filelist, '', cal_outputdir)


    ##### NoiseMap
    noisemap_l1_datadir = os.path.join(tvacdata_path, "TV-20_EXCAM_noise_characterization", "noisemap_test_data", "test_l1_data")
    noisemap_l1_data_filelist = sorted(glob.glob(os.path.join(noisemap_l1_datadir,"*.fits")))
    noisemap_l1_data_filelist.sort()
    process_one_dataset(noisemap_l1_data_filelist, '', cal_outputdir)

    ##### Flat field and Bad Pixel Map
    flat_mock_inputdir = os.path.join(e2eoutput_path, "flat_uranus_output", "mock_input_data")
    l1_flatfield_filelist = glob.glob(os.path.join(flat_mock_inputdir, "*.fits"))
    l1_flatfield_filelist.sort()
    process_one_dataset(l1_flatfield_filelist, '', cal_outputdir)

    ############################################
    ####### Run the walker on some test_data
    ############################################

    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L1")
    l2b_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L2b")
    # define the raw science data to process
    l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]] # just grab the first two files
    tvac_l2b_filelist = [os.path.join(l2b_datadir, "{0}.fits".format(i)) for i in [90529, 90531]] # TVAC L2b data for comparison
    # FINALLY, run the processing of the L1 data now that all calibrtation products are also created
    process_one_dataset(l1_data_filelist, '', outputdir)

    #####################
    #### Test Clean-up
    #####################
    # # clean up by removing entries form caldb
    this_caldb = caldb.CalDB()
    cal_files = glob.glob(os.path.join(cal_outputdir, "*[a-zA-Z].fits"))
    for filename in cal_files:
        frame = data.autoload(filename)
        try:
            this_caldb.remove_entry(frame)
        except ValueError:
            # skip ones not in caldb
            print("{0} not in caldb, skipping".format(filename))
            pass

    ###########################
    ##### Test Verification
    ###########################
    # check against TVAC data that we produced basically the same result
    new_l2b_filenames = [os.path.join(outputdir, "{0}.fits".format(i)) for i in [90499, 90500]]

    for new_filename, tvac_filename in zip(new_l2b_filenames, tvac_l2b_filelist):
        img = data.Image(new_filename)

        with fits.open(tvac_filename) as hdulist:
            tvac_dat = hdulist[1].data
        
        diff = img.data - tvac_dat
        middle_diff = diff[475:535, 500:560]

        assert np.median(np.abs(middle_diff)) < 1.0 # assert not too different

        # # plotting script for debugging
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
    tvacdata_dir = "/home/jwang/Desktop/CGI_TVAC_Data"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    test_l1_to_l2b_fromscratch(tvacdata_dir, outputdir)