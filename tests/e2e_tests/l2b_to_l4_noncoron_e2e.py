import argparse
import os
import numpy as np
from astropy.io import fits
import corgidrp
from corgidrp.data import Image
from corgidrp.mocks import (create_default_L2b_headers, create_default_L3_headers)
import corgidrp.mocks as mocks
import corgidrp.caldb as caldb
import corgidrp.astrom as astrom
import corgidrp.data as corgidata
import corgidrp.check as check
import corgidrp.walker as walker
import pyklip.fakes
import pytest
import glob
import shutil
from datetime import datetime, timedelta

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_l2b_to_l3(e2edata_path, e2eoutput_path):
    '''

    An end-to-end test that takes the noncoronagraphic OS11 data and runs it through the L2b to L4 pipeline.

        It checks that: 
            - The two OS11 planets are detected within 1 pixel of their expected separations
            - The calibration files are correctly associated with the output file

        Data needed: 
            - Noncoronagraphic dataset - taken from OS11 data
            - Reference star dataset - taken from OS11 data
            - Satellite spot dataset - created in the test
        
        Calibrations needed: 
            - AstrometricCalibration
            - CoreThroughputCalibration
            - FluxCalibration
    
    Args:
        e2edata_path (str): Path to the test data
        e2eoutput_path (str): Path to the output directory


    '''

    main_output_dir = os.path.join(e2eoutput_path, "l2b_to_l4_noncoron_e2e")
    if os.path.exists(main_output_dir):
        shutil.rmtree(main_output_dir)
    os.makedirs(main_output_dir)

    # Create input_data subfolder
    input_data_dir = os.path.join(main_output_dir, 'input_l2b')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    calibrations_dir = os.path.join(main_output_dir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)

    l2b_to_l3_dir = os.path.join(main_output_dir, "l2b_to_l3")
    if not os.path.exists(l2b_to_l3_dir):
        os.makedirs(l2b_to_l3_dir)

    ##################################################
    #### Generate an astrometric calibration file ####
    ##################################################

    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not

    field_path = os.path.join(os.path.dirname(__file__),"..","test_data", "JWST_CALFIELD2020.csv")

    #Create the mock dataset (without saving to disk to avoid simcal_astrom.fits)
    mock_dataset = mocks.create_astrom_data(field_path=field_path, filedir=None)

    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=mock_dataset, field_path=field_path, find_threshold=5)
    astrom_cal.save(filedir=calibrations_dir)

    # add calibration file to caldb
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(astrom_cal)

    ############################################
    #### Put the OS 11 data into L2B format ####
    ############################################

    #Read in the images
    input_file = 'hlc_os11_no_fpm.fits'
    input_hdul = fits.open(os.path.join(e2edata_path, "hlc_os11_v3", input_file))
    input_data = input_hdul[0].data
    header = input_hdul[0].header
    
    # number of frames we are making
    num_frames = 5

    big_array_size = [1024,1024]

    image_list = []
    for i in range(num_frames): 

        big_array = np.zeros(big_array_size)
        big_rows, big_cols = big_array_size
        small_rows, small_cols = input_data.shape

        # Find the middle indices for the big array
        row_start = (big_rows - small_rows) // 2
        col_start = (big_cols - small_cols) // 2

        # Insert the small array into the middle of the big array
        input_img = input_data / np.nanmax(input_data) * 1000 # peak counts of 1000
        big_array[row_start:row_start + small_rows, col_start:col_start + small_cols] += input_img

        # add some gaussian noise so each frame is not the same. 1 count of noise
        big_array += np.random.normal(loc=0, scale=1, size=big_array.shape)
        big_err = np.ones(big_array.shape) * 1.0

        #Make a BIAS HDU
        bias_hdu = fits.ImageHDU(data=np.zeros_like(big_array[0]))
        bias_hdu.nane = 'BIAS'
        bias_hdu.header['PCOUNT'] = 0
        bias_hdu.header['GCOUNT'] = 1
        bias_hdu.header['EXTNAME'] = 'BIAS'

        #Create the new Image object passing in the error header
        mock_pri_header,mock_ext_header,mock_err_header,mock_dq_header,_=create_default_L2b_headers()
        new_image=Image(big_array,mock_pri_header,mock_ext_header,
                        err=big_err,err_hdr=mock_err_header,
                        dq_hdr=mock_dq_header,
                        input_hdulist=[bias_hdu])
        # Check if LAYER_1 present in error header
        #print(f"Image err_hdr has LAYER_1: {new_image.err_hdr.get('LAYER_1','NOT FOUND')}")
        #print("="*60+"\n")

        new_image.pri_hdr.set('FRAMET', 1)
        new_image.ext_hdr.set('EXPTIME', 1)
        new_image.pri_hdr.set('PA_APER', 0)
        new_image.ext_hdr.set('CFAMNAME','1F')
        new_image.ext_hdr.set('FSMLOS', 0) # non-coron
        new_image.ext_hdr.set('LSAMNAME', 'OPEN') # non-coron

        # Generate proper filename with visitid and current time
        visitid = "0200001999001000001"  # Use consistent visitid
        unique_time = (datetime.now() + timedelta(milliseconds=i*100)).strftime('%Y%m%dt%H%M%S%f')[:-5]
        new_image.filename = f"cgi_{visitid}_{unique_time}_l2b.fits"

        image_list.append(new_image)


    #########################################
    #### Save the dataset to a directory ####
    #########################################

    mock_dataset = corgidata.Dataset(image_list)
    mock_dataset.save(filedir=input_data_dir)

    ## Next step run things through the walker. 

    #####################################
    #### Pass the data to the walker ####
    #####################################

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    l2b_data_filelist = sorted(glob.glob(os.path.join(input_data_dir, "*.fits")))
    walker.walk_corgidrp(l2b_data_filelist, "", l2b_to_l3_dir)

    #Read in an L3 file
    l3_filename = glob.glob(os.path.join(l2b_to_l3_dir, "*l3_.fits"))[0]
    l3_image = Image(l3_filename)

    #Check if there's a WCS header
    assert l3_image.ext_hdr['CTYPE1'] == 'RA---TAN'
    assert l3_image.ext_hdr['CTYPE2'] == 'DEC--TAN'

    #Check if the Bunit is correct
    assert l3_image.ext_hdr['BUNIT'] == 'photoelectron/s'
    
    check.compare_to_mocks_hdrs(l3_filename, mocks.create_default_L2b_headers) #L2b leaves out CDELT1 and CDELT2

    #Clean up
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)
    

@pytest.mark.e2e
def test_l3_to_l4(e2eoutput_path):
    '''
    An end-to-end test that takes the noncoronagraphic L3 data and runs it through the L3 to L4 pipeline.

        It checks that: 
            - The two OS11 planets are detected within 1 pixel of their expected separations
            - The calibration files are correctly associated with the output file

        Data needed: 
            - L3 data - taken from the L3 data
            - Satellite spot dataset - created in the test
        
        Calibrations needed: 
            - AstrometricCalibration
            - CoreThroughputCalibration
            - FluxCalibration
    
    Args:
        e2eoutput_path (str): Path to the output directory
    '''

    main_output_dir = os.path.join(e2eoutput_path, "l2b_to_l4_noncoron_e2e")
    e2eintput_path = os.path.join(main_output_dir, "l2b_to_l3")
    calibrations_dir = os.path.join(main_output_dir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)


    ##################################################
    #### Generate an astrometric calibration file ####
    ##################################################

    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not

    field_path = os.path.join(os.path.dirname(__file__),"..","test_data", "JWST_CALFIELD2020.csv")

    #Create the mock dataset (without saving to disk to avoid simcal_astrom.fits)
    mock_dataset = mocks.create_astrom_data(field_path=field_path, filedir=None)


    astrom_cal = astrom.boresight_calibration(input_dataset=mock_dataset, field_path=field_path, find_threshold=5)
    astrom_cal.save(filedir=calibrations_dir)

    # add calibration file to caldb
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(astrom_cal)

    ##########################################
    #### Generate a flux calibration file ####
    ##########################################

    #Create a mock flux calibration file
    fluxcal_factor = 2e-12
    fluxcal_factor_error = 1e-14
    prhd, exthd, errhdr, _ = create_default_L3_headers()
    fluxcal_fac = corgidata.FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, err_hdr = errhdr, input_dataset = mock_dataset)


    fluxcal_fac.save(filedir=calibrations_dir)
    this_caldb.create_entry(fluxcal_fac)

    #####################################
    #### Read in the L3 data and run ####
    #####################################

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    
    l3_data_filelist = sorted(glob.glob(os.path.join(e2eintput_path, "*l3_.fits")))

    walker.walk_corgidrp(l3_data_filelist, "", main_output_dir)

    ########################################################################
    #### Read in the images and test for source detection ###
    ########################################################################

    l4_filename = glob.glob(os.path.join(main_output_dir, "*l4_.fits"))[0]
    combined_image = Image(l4_filename)
    assert combined_image.filename == l3_data_filelist[-1].split(os.path.sep)[-1].replace("_l3_", "_l4_")
    
    #Find the sources and get their (x,y) coordinate
    y_source, x_source = np.unravel_index(np.nanargmax(combined_image.data), combined_image.data.shape)
    assert combined_image.dq[y_source, x_source] == 0 # check DQ
    peakflux, _, x_source, y_source = pyklip.fakes.gaussfit2d(combined_image.data, x_source, y_source, guesspeak=np.nanmax(combined_image.data))
    xcen = combined_image.ext_hdr['CRPIX1']
    ycen = combined_image.ext_hdr['CRPIX2']
    assert np.isclose(x_source, xcen, atol=1)
    assert np.isclose(y_source, ycen, atol=1)
    assert peakflux == pytest.approx(1000, rel=1e-2)


    #Filename format will be checked in data format test

    input_filenames = [filepath.split(os.path.sep)[-1] for filepath in l3_data_filelist]

    assert combined_image.ext_hdr['FILE0'] in input_filenames
    assert combined_image.ext_hdr['FILE1'] in input_filenames
    assert combined_image.ext_hdr['FILE2'] in input_filenames
    assert combined_image.ext_hdr['FILE3'] in input_filenames
    assert combined_image.ext_hdr['FILE4'] in input_filenames

    check.compare_to_mocks_hdrs(l4_filename, mocks.create_default_L2b_headers)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)



if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.


    outputdir = thisfile_dir
    #This folder should contain an OS11 folder: ""hlc_os11_v3" with the OS11 data in it.
    e2edata_dir = '/Users/kevinludwick/Documents/DRP_E2E_Test_Files_v2/E2E_Test_Data'# '/Users/jmilton/Documents/CGI/E2E_Test_Data2''/Users/clarissardoo/Projects/E2E_Test_Data'
    #Not actually TVAC Data, but we can put it in the TVAC data folder. 
    ap = argparse.ArgumentParser(description="run the l2b->l4 end-to-end test")

    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir

    test_l2b_to_l3(e2edata_dir, outputdir)
    test_l3_to_l4(outputdir)
