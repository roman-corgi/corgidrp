
import argparse
import glob
import numpy as np
import os
import astropy.time as time
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb

corgidrp_dir = os.path.join(os.path.dirname(corgidrp.__file__), "..") # basedir of entire corgidrp github repo

# paths (CHANGE THESE!!!) - now optional. Can use arguments to override these
# rather than edit them.
nonlin_path = "/home/jwang/Jason/Downloads/20240723_TVAC_data_for_DRP_Testing_LRS/nonlin_table_240322.txt"
l1_datadir = "/home/jwang/Jason/Downloads/20240723_TVAC_data_for_DRP_Testing_LRS/input_data_TV-36/"
outputdir = "./l1_to_l2a_output/"

def main():
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # define the raw science data to process
    all_l1_dat = glob.glob(os.path.join(l1_datadir, "*.fits"))
    all_l1_dat.sort()
    l1_data_filelist = all_l1_dat[:2] # just grab the first two files
    mock_nonlinear_filelist = all_l1_dat[-2:] # grab the last two real data to mock the nonlinearity file

    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    #test_nonlinear_filename = os.path.join(corgidrp_dir, "tests", "test_data", "nonlin_sample.fits")
    #nonlinear_cal = data.NonLinearityCalibration(test_nonlinear_filename) # use the same headers
    #nonlinear_cal.data = nonlin_dat # replace the data with the real data from II&T
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_nonlinear_filelist)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=outputdir, filename="mock_nonlinearcal.fits" )

    # add calibration file to caldb
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(nonlinear_cal)

    ####### Run the walker on some test_data

    walker.walk_corgidrp(l1_data_filelist, "", outputdir)

    # clean up by removing entry
    this_caldb.remove_entry(nonlinear_cal)

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-np", "--nonlin_path", default=nonlin_path,
                    help="text file containing the non-linear table to use [%(default)s]")
    ap.add_argument("-l1", "--l1_datadir", default=l1_datadir,
                    help="directory that contains the L1 data files [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results and it will be created if it does not exist [%(default)s]")
    args = ap.parse_args()
    nonlin_path = args.nonlin_path
    l1_datadir = args.l1_datadir
    outputdir = args.outputdir
    main()
