## This file is meant to provide an example of the different pipeline steps
## that we will need to run for a TTR5-like observation. This is not meant to
## be a working example, but rather a template for the different 
## steps that we will need to run. It's not even really meant to show how 
## the pipeline will actually work. Most of the functions are just placeholders
## and their exact names and home files are only preliminary suggestions. 

from corgidrp.data import Dataset, Dark, Flat, generate_filelist
# from corgidrp.detector import crop_dataset, subtract_bias, detect_cosmic_rays, correct_nonlinearity, convert_to_electrons, divide_by_em_gain, dark_subtraction, cti_correction
import corgidrp.detector as detector
import corgidrp.image_utils as image_utils
from corgidrp.caldb import CalDB, find_best_dark
import corgidrp.L1_to_L2_proc as L1_to_L2_proc
import corgidrp.L2_to_L3_proc as L2_to_L3_proc
import corgidrp.L3_to_L4_proc as L3_to_L4_proc
import corgidrp.L4_to_TDA_proc as L1_to_TDA_proc

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

# MY: I suggest implementing 2 different options:
# Option 1: Process data step by step for detailed inspection (best for development and sanity checks)
# Option 2: Perform L1 to L2a processing in one single line (best for testing and operations)  

# Option 1: Process data step by step for detailed inspection (best for development and sanity checks)
######################################################################################################

#Crop the data to only the main region of interest
# l1_dataset_cropped = detector.crop_dataset(l1_dataset) 
# This function doesn't exist yet
# MY: Note that this function exists already see: EMCCDFrame in gsw_emccd_frame.py, 
# MY: I suggest to port that function as a sub-function of L1_to_L2a and call something like:
l1_dataset_cropped = L1_to_L2_proc.L1_to_L2a.crop_dataset(l1_dataset)

#Measure and subtract the bias
# l1_dataset_bias_subtracted = detector.subtract_bias(l1_dataset_cropped) 
# There is II&T code that we can port over for this
# MY: That's correct, I suggest porting that code currently in L1_to_L2a to a sub-function 
# of L1_to_L2a and call something like: 
l1_dataset_bias_subtracted = L1_to_L2_proc.L1_to_L2a.subtract_bias(l1_dataset_cropped)

#Detect cosmic rays and make a mask
l1_dataset_cosmic_ray_masked = detector.detect_cosmic_rays(l1_dataset_bias_subtracted) 
# There is II&T code that we can port over for this
# MY: That's correct, I suggest porting that code currently in L1_to_L2a to a sub-function 
# of L1_to_L2a and call something like: 
l1_dataset_cosmic_ray_masked = L1_to_L2_proc.L1_to_L2a.detect_cosmic_rays(l1_dataset_bias_subtracted) 

#Correct for non-linearity
l1_dataset_nonlinearity_corrected = detector.correct_nonlinearity(l1_dataset_cosmic_ray_masked) # There is II&T code that we can port over for this
#This may need an outside calibration file if nonlinearity is not constant
# MY: That's correct, I suggest porting the existing code currently in L1_to_L2a to a sub-function 
# of L1_to_L2a and call something like: 
l1_dataset_nonlinearity_corrected = L1_to_L2_proc.L1_to_L2a.correct_nonlinearity(l1_dataset_cosmic_ray_masked)

#Change the dataset level from L1 to L2a
l2a_dataset = l1_dataset_nonlinearity_corrected.update_to_l2a() # This function doesn't exist yet
# MY: It would be useful to have a description of what this function should be doing, 
# MY: I assume it should handle the headers at the very least. 
# MY: There's no existing function yet but there's is some code that handle the headers 
# and they are up-to-date with what is being done during iit

# Option 2: Perform L1 to L2a processing in one single line (best for testing and operations) 
################################################################################################

# Directly perform L1 to L2a processing (which involves all steps described above)
l2a_dataset = l1_dataset.L1_to_L2_proc.L1_to_L2a() 

# MY: alternatively, one could call l1_dataset.L1_to_L2_proc() to perform all 
# the L1 to L2 processing steps in one line (including L1 to L2a, L2a to L2b, 
# mean combine and median combine) but some work is needed to update L1_to_L2_proc 
# so that calling that function performs all those steps

#Save the dataset
l2a_dataset.save(output_filepath)

#############################
######### L2a -> L2b ########
#############################

# MY: I suggest implementing 2 different options:
# Option 1: Process data step by step for detailed inspection (best for development and sanity checks)
# Option 2: Perform L2a to L2b processing in one single line (best for testing and operations)  

# Option 1: Process data step by step for detailed inspection (best for development and sanity checks)
######################################################################################################

#Select the frames that we want to use
l2_dataset_frame_selected = image_utils.frame_select(l2a_dataset) # This function doesn't exist yet
# MY: Pros and cons of adding that function to image_utils instead of L1_to_L2_proc.L2a_to_L2b??

#Convert to e-
# If we expect the detector gain to vary with time, we may need to pass in the caldb here
#l2_dataset_electrons = detector.convert_to_electrons(l2_dataset_frame_selected) # This function doesn't exist yet
# MY: Note that code to do this already exists in L2a_to_L2b: 
# MY: I suggest porting the existing code currently in L2a_to_L2b to a sub-function 
# of L2a_to_L2b and call something like: 
l2_dataset_electrons = L1_to_L2_proc.L2a_to_L2b.convert_to_electrons(l2_dataset_frame_selected)

#Divide by the em gain -- if applicable -- not assuming photon counting in this example
#This may need an outside calibration file if em_gain < 1000
# l2_dataset_em_gain_divided = detector.divide_by_em_gain(l2_dataset_electrons,cal_db = calibration_db) # This function doesn't exist yet
# MY: Note that code to do this already exists in L2a_to_L2b (no calib file in that version, 
# the commanded em_gain is an input of the L2a_to_L2b function right now): 
# MY: I suggest porting the existing code currently in L2a_to_L2b to a sub-function 
# of L2a_to_L2b and call something like: 
em_gain = 1 # placeholder
l2_dataset_em_gain_divided = L1_to_L2_proc.L2a_to_L2b.divide_by_em_gain(l2_dataset_electrons, em_gain = em_gain)

#Subtract_master_dark
master_dark = CalDB.get_calib(l2_dataset_electrons.frames[0],Dark)
# MY: there's some exiting code to import calibration frames in the iitdata, 
# it would worth discussing whether we should keep that implementation or not
#l2_dataset_dark_subtracted = detector.dark_subtraction(l2_dataset_electrons, master_dark)
# MY: Note that code to do this already exists in L2a_to_L2b: 
# MY: I suggest porting the existing code currently in L2a_to_L2b to a sub-function 
# of L2a_to_L2b and call something like: 
l2_dataset_dark_subtracted = L1_to_L2_proc.L2a_to_L2b.dark_subtraction(l2_dataset_electrons, master_dark = master_dark)

#Apply CTI Correction
# l2_dataset_cti_corrected = detector.cti_correction(l2_dataset_dark_subtracted) # This function doesn't exist yet
# MY: I suggest calling something like: 
l2_dataset_cti_corrected = L1_to_L2_proc.L2a_to_L2b.cti_correction(l2_dataset_dark_subtracted) 
   
#Divide by master flat
master_flat = CalDB.get_calib(l2_dataset_cti_corrected.frames[0],Flat)
# MY: there's some exiting code to import calibration frames in the iitdata, 
# it would worth discussing whether we should keep that implementation or not
#l2_dataset_flat_divided = detector.flat_division(l2_dataset_cti_corrected, master_flat) # This function doesn't exist yet
# MY: Note that code to do this already exists in L2a_to_L2b: 
# MY: I suggest porting the existing code currently in L2a_to_L2b to a sub-function 
# of L2a_to_L2b and call something like: 
l2_dataset_flat_divided = L1_to_L2_proc.L2a_to_L2b.flat_division(l2_dataset_cti_corrected, master_flat = master_flat) # This function doesn't exist yet
  
#Compute bad_pixel map and correct for bad pixels
l2_dataset_bad_pixel_corrected = detector.bad_pixel_correction(l2_dataset_flat_divided) # This function doesn't exist yet
# MY: Note that code to do this already exists in L2a_to_L2b: 
# MY: I suggest porting the existing code currently in L2a_to_L2b to a sub-function 
# of L2a_to_L2b and call something like: 
l2_dataset_bad_pixel_corrected = L1_to_L2_proc.L2a_to_L2b.bad_pixel_correction(l2_dataset_flat_divided) # This function doesn't exist yet
  
#Change the dataset level from L2a to L2b
l2b_dataset = l2_dataset_bad_pixel_corrected.update_to_l2b() # This function doesn't exist yet
# MY: It would be useful to have a description of what this function should be doing, 
# MY: I assume it should handle the headers at the very least. 
# MY: There's no existing function yet but there's is some code that handle the headers 
# and they are up-to-date with what is being done during iit

# Option 2: Perform L2a to L2b processing in one single line (best for testing and operations) 
################################################################################################
l2b_dataset = l2a_dataset.L1_to_L2_proc.L2a_to_L2b()
# MY: some work is needed to update L2_to_L2a_proc so that calling that function performs 
# all the L2a to L2b processing steps above

# MY: alternatively, one could call l1_dataset.L1_to_L2a_proc() to perform all 
# the L1 to L2 processing steps in one line (including mean combine and median combine) 
# but some work is needed to update L1_to_L2_proc so that calling that function performs 
# all those steps

#Save the dataset
l2b_dataset.save(output_filepath)

#############################
## Mean and median combine ##
#############################

l2b_dataset_mean_combine = l2b_dataset.L1_to_L2_proc.mean_combine(l2b_dataset)
l2b_dataset_median_combine = l2b_dataset.L1_to_L2_proc.median_combine(l2b_dataset)

#Save the mean and median combined dataset
l2b_dataset_mean_combine.save(output_filepath)
l2b_dataset_median_combine.save(output_filepath)

#############################
######### L2b -> L3 #########
#############################

# MY: I suggest implementing 2 different options:
# Option 1: Process data step by step for detailed inspection (best for development and sanity checks)
# Option 2: Perform L2 to L3 processing in one single command (best for testing and operations)  

# Option 1: Process data step by step for detailed inspection (best for development and sanity checks)
######################################################################################################

#Create the WCS headers
l2b_dataset_wcs = image_utils.create_wcs(l2b_dataset) # This function doesn't exist yet. currently in image_utils, but doesn't have to be there
    
#Divide by exposure time
l2b_dataset_exptime_divided = image_utils.divide_by_exptime(l2b_dataset_wcs)
# MY: Pros and cons of adding that function to image_utils instead of L2_to_L3_proc.L2_to_L3??

#Change the dataset level from L2b to L3
l3_dataset = l2b_dataset_exptime_divided.update_to_l3() # This function doesn't exist yet
# MY: It would be useful to have a description of what this function should be doing, 
# MY: I assume it should handle the headers at the very least. 
# MY: There's no existing function yet but there's is some code that handle the headers 
# and they are up-to-date with what is being done during iit

# Option 2: Perform L2 to L3 processing in one single line (best for testing and operations) 
################################################################################################

l3_dataset = l2b_dataset.L2_to_L3_proc.L2_to_L3()

#Save the dataset
l3_dataset.save(output_filepath)

############################
######### L3 -> L4 #########
############################

# MY: I suggest implementing 2 different options:
# Option 1: Process data step by step for detailed inspection (best for development and sanity checks)
# Option 2: Perform L3 to L4 processing in one single command (best for testing and operations)  

# Option 1: Process data step by step for detailed inspection (best for development and sanity checks)
######################################################################################################

#Apply distortion correction
l3_dataset_distortion_corrected = image_utils.distortion_correction(l3_dataset) # This function doesn't exist yet
# MY: Pros and cons of adding that function to image_utils instead of L3_to_L4_proc.L3_to_L4??
    
#Find the location of the star
l3_dataset_star_located = image_utils.find_star(l3_dataset_distortion_corrected) # This function doesn't exist yet
# MY: Pros and cons of adding that function to image_utils instead of L3_to_L4_proc.L3_to_L4??

#Do PSF subtraction
l3_psf_subtracted = image_utils.do_psf_subtraction(l3_dataset_star_located) # This function doesn't exist yet
# MY: Pros and cons of adding that function to image_utils instead of L3_to_L4_proc.L3_to_L4??
# MY: Note that there's existing code to do this, I can take an action to port it

l4_dataset = l3_psf_subtracted.update_to_l4() # This function doesn't exist yet
# MY: It would be useful to have a description of what this function should be doing, 
# MY: I assume it should handle the headers at the very least. 
# MY: There's no existing function yet but there's is some code that handle the headers 
# and they are up-to-date with what is being done during iit

# Option 2: Perform L2a to L2b processing in one single line (best for testing and operations) 
################################################################################################

l4_dataset = l3_dataset.L3_to_L4_proc.L3_to_L4()

#Save the dataset
l4_dataset.save(output_filepath)

############################
######### L4 -> TDA ########
############################

# MY: I suggest implementing 2 different options:
# Option 1: Process data step by step for detailed inspection (best for development and sanity checks)
# Option 2: Perform L2a to L2b processing in one single command (best for testing and operations)  

# Option 1: Process data step by step for detailed inspection (best for development and sanity checks)
######################################################################################################

# MY: Individuals steps TBD

# Option 2: Perform L2a to L2b processing in one single line (best for testing and operations) 
################################################################################################

tda_dataset = l4_dataset.L4_to_TDA_proc()

#Save the dataset
tda_dataset.save(output_filepath)

# MY: General note: for all existing functions above, there are some unit tests as well,
# I can take an action to port those


