'''
This script runs an end-to-end test of processing photon-counted SPC WFOV band 1 data.
'''
# Import modules required for data processing and testing
import numpy as np
import shutil
import corgidrp.check as check
import corgidrp.mocks as mocks
import astropy.time as time
import corgidrp
import corgidrp.data as data
import corgidrp.caldb as caldb # This part can cause trouble if imported earlier.
import corgidrp.detector as detector
import astropy.io.fits as fits
import corgidrp.walker as walker
import corgidrp.astrom as astrom
from corgidrp import corethroughput
from astropy.io import fits
import os
import pytest
import argparse
import time
import datetime

try:
    from proc_cgi_frame.gsw_process import Process
except:
    pass

this_file_dir = os.path.dirname(__file__) # this file's folder

def create_and_clean_folder(folder_name):
    # Create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # Clean up the folder by removing any old files
    for file in os.listdir(folder_name):
        os.remove(os.path.join(folder_name,file))
        
def extract_visit_files(filelist):
    visit_file_groups = {}
    
    for file in filelist:
        with fits.open(file) as hdu_temp:
            visitid = hdu_temp[0].header["VISITID"]
            
        if visitid not in visit_file_groups:
            visit_file_groups[visitid] = []
            
        visit_file_groups[visitid].append(file)
        
    return list(visit_file_groups.values())
        
def separate_sci_and_satspots(filelist):
    '''
    This script separates the files with satspots from the files without.
    
    Args:
        filelist: A list of files, in no particular order, that may or may not
                  have satspots
                  
    Returns:
        sci_images: A list of images without satspots
        satspots_images: A list of images with satspots
    '''
    sci_images=[]
    satspots_images=[]
    for file in filelist:
        hdu_temp = fits.open(file)
        if hdu_temp[1].header['SATSPOTS']==1:
            satspots_images.append(file)
        elif hdu_temp[1].header['SATSPOTS']==0:
            sci_images.append(file)
    return sci_images, satspots_images  

def process_l1_to_l2a(filelist,l2a_outputdir):
    '''
    This script separates the files in the provided list into those with satspots
    and those without. It then processess the two subsets of files from L1 to L2a.
    
    Args:
        filelist: a list of the data files to process
        l2a_outputdir: the directory to store the output l2a files
    '''
    images_sci, images_spots = separate_sci_and_satspots(filelist)
    walker.walk_corgidrp(images_spots, '', l2a_outputdir)
    walker.walk_corgidrp(images_sci, '', l2a_outputdir)
    
def process_l2a_to_l2b(filelist,l2b_outputdir):
    '''
    This script separates the files in the provided list into those with satspots
    and those without. It then processess the two subsets of files from L2a to L2b.
    
    Args:
        filelist: a list of data files to process
        l2b_outputdir: the directory to store the output l2b files
    '''
    images_sci, images_spots = separate_sci_and_satspots(filelist)
    walker.walk_corgidrp(images_spots, '', l2b_outputdir)
    walker.walk_corgidrp(images_sci, '', l2b_outputdir)
    
def create_mock_calibrations(calibrations_dir):
    # Setup
    
    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder,'tmp_pc_wfov_band1_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    
    # Remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()   
    #--------------------------------------------------------------------------
    # Astrometric Calibration 
    
    # Simulated image
    pkg_dir = Path(corgidrp.__file__).resolve().parent
    field_path = os.path.join(pkg_dir,'data/JWST_CALFIELD2020.csv')
    
    # Astrometric calibration input
    astrom_input_dir = os.path.join(calibrations_dir,'astrom_cal_input')
    create_and_clean_folder(astrom_input_dir)
    
    mock_dataset = mocks.create_astrom_data(field_path=field_path,filedir=None,rotation=0)
    mock_dataset.save(filedir=astrom_input_dir)
    
    # Add calibration file to caldb
    astrom_cal = astrom.boresight_calibration(input_dataset=mock_dataset, field_path=field_path, find_threshold=5)
    astrom_cal.save(filedir=calibrations_dir)

    this_caldb.create_entry(astrom_cal)
    #--------------------------------------------------------------------------
    # Core Throughput Calibration
    # Dataset with some CT profile defined in create_ct_interp
    # The DRP will return an error if it does not find at least one pupil image
    # Pupil image
    pupil_image = np.zeros([1024, 1024])
    # Set it to some known value for a selected range of pixels
    pupil_image[510:530, 510:530]=1
    prhd, exthd_pupil, errhdr, dqhdr = mocks.create_default_L3_headers()
    # DRP
    # cfam filter
    exthd_pupil['CFAMNAME'] = '1F'
    # Add specific values for pupil images:
        # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'
    
    data_psf, psf_loc_in, half_psf = mocks.create_ct_psfs(50, cfam_name='1F',
                                                          n_psfs=100)
    
    err = np.ones([1024,1024]) 
    data_ct_interp = [data.Image(pupil_image,pri_hdr = prhd,
                                 ext_hdr = exthd_pupil, err = err)]
    # Set of off-axis PSFs with a CT profile defined in create_ct_interp
    # First, we need the CT FPM center to create the CT radial profile
    # We can use a miminal dataset to get to know it
    data_ct_interp += [data_psf[0]]
    ct_cal_tmp = corethroughput.generate_ct_cal(corgidrp.data.Dataset(data_ct_interp))
    # Change the FPAMNAME to match the sim data
    ct_cal_tmp.ext_hdr['FPAMNAME']='SPC12_R2C1'
    mocks.rename_files_to_cgi_format(list_of_fits=[ct_cal_tmp], output_dir=calibrations_dir, level_suffix="ctm_cal")
    this_caldb.create_entry(ct_cal_tmp)
    #--------------------------------------------------------------------------
    # Flux Calibration
    #Create a mock flux calibration file
    fluxcal_factor = 2e-12
    fluxcal_factor_error = 1e-14
    prhd, exthd, errhd, dqhd = mocks.create_default_L3_headers()
    # Set consistent header values for flux calibration factor
    exthd['CFAMNAME'] = '1F'
    exthd['DPAMNAME'] = 'PUPIL'
    exthd['LSAMNAME'] = 'OPEN'
    exthd['FSAMNAME'] = 'OPEN'
    exthd['FPAMNAME'] = 'OPEN_12'
    fluxcal_fac = corgidrp.data.FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, err_hdr = errhd, input_dataset = mock_dataset)

    mocks.rename_files_to_cgi_format(list_of_fits=[fluxcal_fac], output_dir=calibrations_dir, level_suffix="abf_cal")
    this_caldb.create_entry(fluxcal_fac)
    
    

@pytest.mark.e2e    
def test_l1_to_l4_pc_SPCWFOV_band1_e2e(e2edata_path,outputdir):
    '''
    This test function loads simulated SPC WFOV band 1 photon-counted data
    and processes it from L1 to L4.
    
    Args:
        e2edata_path: location of the TVAC test data folder containing the data
                      used for mock calibrations
        outputdir:    directory of the e2e output data
    '''
    # Initial Setup: specifying file paths and creating output directories

    # Define the file paths for the TVAC test data (used for example calibrations)
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    TVAC_datadir = os.path.join(e2edata_path,"TV-36_Coronagraphic_Data","L1")
    
    # Locate the L1 data relative to the e2edata_path
    l1_datadir = os.path.join(e2edata_path,"Photon_Counting_SPC_WFOV_Band1")
    
    # Create output directories for the test

    # Top-level output folder
    test_outputdir = os.path.join(outputdir,"l1_to_l4_e2e")
    for d in [test_outputdir]:
        os.makedirs(d, exist_ok=True)

    # Create input_data subfolder
    input_data_dir = os.path.join(test_outputdir,'input_l1')
    create_and_clean_folder(input_data_dir)
     
    # Create calibrations subfolder
    calibrations_dir = os.path.join(test_outputdir,'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)
        
    # Create a folder for the input L1 data, and copy the L1 files into it
    input_data_dir = os.path.join(test_outputdir,'input_l1')
    create_and_clean_folder(input_data_dir)
    l1_data_filelist = [os.path.join(l1_datadir,f) for f in os.listdir(l1_datadir) if f.endswith('.fits')]
    for file in l1_data_filelist:
        shutil.copy2(file,os.path.join(input_data_dir,os.path.basename(file)))
        
    # Create separate L2a and L2b outputdirs
    l2a_outputdir = os.path.join(test_outputdir,'l1_to_l2a')
    create_and_clean_folder(l2a_outputdir)
    
    l2b_outputdir = os.path.join(test_outputdir,'l1_to_l2b')
    create_and_clean_folder(l2b_outputdir)
   
    # Create L3 outputdir
    l3_outputdir = os.path.join(test_outputdir,'l1_to_l3')
    create_and_clean_folder(l3_outputdir)
    
    # Create L4 outputdir
    l4_outputdir = os.path.join(test_outputdir,'l1_to_l4')
    create_and_clean_folder(l4_outputdir)
    #--------------------------------------------------------------------------
    # Adapting TVAC test data for use as sample calibration data
    # This step should now be automated by the DRP, so it is no longer included
    # here explicitly.
    #--------------------------------------------------------------------------
    # Processing from L1 to L2a
    
    # Run the walker to process the data from L1 to L2a
    # Note that the reference star images must be processed separately from the
    # target star images. We will also separate out the images with satspots.
    
    # Separate the L1 files by visit ID
    L1_visit_file_groups = extract_visit_files(l1_data_filelist)
    # We do not need to explicitly identify which group of files is which.
    # The important part is to process each group separately.
    
    # Process each set of files
    Nvisits = len(L1_visit_file_groups)
    for ivisit in range(Nvisits):
        process_l1_to_l2a(L1_visit_file_groups[ivisit], l2a_outputdir)
    print('Completed processing L1 to L2a')
    #--------------------------------------------------------------------------
    # Processing from L2a to L2b
    
    # Again, the reference and target star images must be processed separately.
    # We will also separately process images with and without spots.
    
    l2a_filelist = [os.path.join(l2a_outputdir,f) for f in os.listdir(l2a_outputdir) if f.endswith('l2a.fits')]
    
    # Separate the L2a files by visit ID
    L2a_visit_file_groups = extract_visit_files(l2a_filelist)
    
    # Process each set of files
    for ivisit in range(Nvisits):
        process_l2a_to_l2b(L2a_visit_file_groups[ivisit], l2b_outputdir)
    print('Completed processing from L2a to L2b')
    #--------------------------------------------------------------------------
    # Processing from L2b to L3
    
    # Processing from L2b to L3 requires three calibrations:
    #   astrometric calibration
    #   core throughput calibration
    #   flux calibration
    #
    # The DRP does not automatically mock these calibrations, so we will create
    # the appropriate files.
    create_mock_calibrations(calibrations_dir)
    
    # Run the walker to process from L2b to L3
    # Now, all of the images could be processed together. However, for a
    # large dataset, it may be significantly faster to continue to separate
    # the files by visit ID.
    
    l2b_filelist = [os.path.join(l2b_outputdir,f) for f in os.listdir(l2b_outputdir) if f.endswith('l2b.fits')]
    
    L2b_visit_file_groups = extract_visit_files(l2b_filelist)
    for ivisit in range(Nvisits):
        walker.walk_corgidrp(L2b_visit_file_groups[ivisit],'',l3_outputdir)
    print('Completed processing from L2b to L3')
    #--------------------------------------------------------------------------
    # Processing from L3 to L4
    
    # Now all of the files must be processed together.
    l3_filelist = [os.path.join(l3_outputdir,f) for f in os.listdir(l3_outputdir) if f.endswith('l3.fits')]
    walker.walk_corgidrp(l3_filelist,'',l4_outputdir)
    print('completed processing from L3 to L4')
    
    
if __name__=='__main__':
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    
    e2edata_dir = os.path.join(this_file_dir,'../../../TVAC_Test_Data/E2E_Test_Data/')
    outputdir = 'l1_to_l4_PC_WFOV_band1_e2e'
     
    ap = argparse.ArgumentParser(description='run the l1->l4 end-to-end test')
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    
    # Run the test
    test_l1_to_l4_pc_SPCWFOV_band1_e2e(e2edata_dir,outputdir)