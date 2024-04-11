## This file is meant to provide an example of the different pipeline steps
## that we will need to run for a TTR5-like observation. This is not meant to
## be a working example, but rather a template for the different 
## steps that we will need to run. It's not even really meant to show how 
## the pipeline will actually work. Most of the functions are just placeholders
## and their exact names and home files are only preliminary suggestions. 

from corgidrp.data import Dataset, Dark, Flat, generate_filelist
from corgidrp import l1_to_l2a, l2a_to_l2b, l2b_to_l3, l3_to_l4
from corgidrp.caldb import CalDB

input_filepath = "my/bank/passwords/"
output_filepath = "where/the/planets/will/go/"
caldb_filepath = "where/the/caldb/is/"

#Load up the file list
filelist = generate_filelist(input_filepath) # This function doesn't exist

#Read it into a Dataset object. Here we'll be starting with L1 data and the pipeline will figure that out. 
l1_dataset = Dataset(filelist)

#Read in the caldb
calibration_db = CalDB(caldb_filepath)

#############################
######### L1 -> L2a #########
#############################

#Crop the data to only the main region of interest
l1_dataset_prescan_bias_subtracted = l1_to_l2a.prescan_biassub(l1_dataset,return_full_frame=False) 

#Detect cosmic rays and make a mask
l1_dataset_cosmic_ray_masked = l1_to_l2a.detect_cosmic_rays(l1_dataset_prescan_bias_subtracted) # There is II&T code that we can port over for this

#Correct for non-linearity
#This may need an outside calibration file if nonlinearity is not constant
l1_dataset_nonlinearity_corrected = l1_to_l2a.correct_nonlinearity(l1_dataset_cosmic_ray_masked) # There is II&T code that we can port over for this

#Change the dataset level from L1 to L2a
l2a_dataset = l1_dataset_nonlinearity_corrected.update_to_l2a() # This function doesn't exist yet

l2a_dataset.save(output_filepath)

#############################
######### L2a -> L2b ########
#############################

#Select the frames that we want to use
l2_dataset_frame_selected = l2a_to_l2b.frame_select(l2a_dataset) # This function doesn't exist yet

#Convert to e-
# If we expect the detector gain to vary with time, we may need to pass in the caldb here
l2_dataset_electrons = l2a_to_l2b.convert_to_electrons(l2_dataset_frame_selected) # This function doesn't exist yet

#Divide by the em gain -- if applicable -- not assuming photon counting in this example
#This may need an outside calibration file if em_gain < 1000
# l2_dataset_em_gain_divided = detector.divide_by_em_gain(l2_dataset_electrons,cal_db = calibration_db) # This function doesn't exist yet

#Subtract_master_dark
master_dark = CalDB.get_calib(l2_dataset_electrons.frames[0],Dark)
l2_dataset_dark_subtracted = l2a_to_l2b.dark_subtraction(l2_dataset_electrons, master_dark)

#Apply CTI Correction
l2_dataset_cti_corrected = l2a_to_l2b.cti_correction(l2_dataset_dark_subtracted) # This function doesn't exist yet

#Divide by master flat
master_flat = CalDB.get_calib(l2_dataset_cti_corrected.frames[0],Flat)
l2_dataset_flat_divided = l2a_to_l2b.flat_division(l2_dataset_cti_corrected, master_flat) # This function doesn't exist yet

#Compute bad_pixel map and correct for bad pixels
l2_dataset_bad_pixel_corrected = l2a_to_l2b.correct_bad_pixels(l2_dataset_flat_divided) # This function doesn't exist yet

#Change the dataset level from L2a to L2b
l2b_dataset = l2_dataset_bad_pixel_corrected.update_to_l2b() # This function doesn't exist yet

#Save the dataset
l2b_dataset.save(output_filepath)

#############################
######### L2b -> L3 #########
#############################

#Create the WCS headers
l2b_dataset_wcs = l2b_to_l3.create_wcs(l2b_dataset) # This function doesn't exist yet. currently in image_utils, but doesn't have to be there

#Divide by exposure time
l2b_dataset_exptime_divided = l2b_to_l3.divide_by_exptime(l2b_dataset_wcs)

#Change the dataset level from L2b to L3
l3_dataset = l2b_dataset_exptime_divided.update_to_l3() # This function doesn't exist yet

#Save the dataset
l3_dataset.save(output_filepath)

############################
######### L3 -> L4 #########
############################

#Apply distortion correction
l3_dataset_distortion_corrected = l3_to_l4.distortion_correction(l3_dataset) # This function doesn't exist yet

#Find the location of the star
l3_dataset_star_located = l3_to_l4.find_star(l3_dataset_distortion_corrected) # This function doesn't exist yet

#Do PSF subtraction
l3_psf_subtracted = l3_to_l4.do_psf_subtraction(l3_dataset_star_located) # This function doesn't exist yet

l4_dataset = l3_psf_subtracted.update_to_l4() # This function doesn't exist yet

#Save the dataset
l4_dataset.save(output_filepath)
