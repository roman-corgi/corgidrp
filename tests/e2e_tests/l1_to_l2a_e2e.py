import os
import glob
import numpy as np
import astropy.time as time
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb

corgidrp_dir = os.path.join(os.path.dirname(corgidrp.__file__), "..") # basedir of entire corgidrp github repo

# paths (CHANGE THESE!!!)
nonlin_path = "/home/jwang/Jason/Downloads/20240723_TVAC_data_for_DRP_Testing_LRS/nonlin_table_240322.txt"
l1_datadir = "/home/jwang/Jason/Downloads/20240723_TVAC_data_for_DRP_Testing_LRS/input_data_TV-36/"
outputdir = "./l1_to_l2a_output/"

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
nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                             input_dataset=mock_input_dataset)
nonlinear_cal.save(filedir=outputdir, filename="mock_nonlinearcal.fits" )

# add calibration file to caldb
this_caldb = caldb.CalDB()
this_caldb.create_entry(nonlinear_cal)

####### Run the walker on some test_data

walker.walk_corgidrp(l1_data_filelist, "", outputdir)

# clean up by removing entry
this_caldb.remove_entry(nonlinear_cal)