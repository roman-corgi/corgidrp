import argparse
import os
import numpy as np
from astropy.io import fits
from corgidrp.data import Image
from corgidrp.mocks import (create_default_L2b_headers, create_default_L3_headers, 
                            create_synthetic_satellite_spot_image, create_ct_psfs)
from corgidrp.l4_to_tda import find_source
import corgidrp.mocks as mocks
import corgidrp.caldb as caldb
import corgidrp.astrom as astrom
import corgidrp.data as corgidata
import corgidrp.walker as walker
import pyklip.fakes
import pytest
import glob
import shutil
import pathlib

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_l2b_to_l3(e2edata_path, e2eoutput_path):
    '''

    An end-to-end test that takes the OS11 data and runs it through the L2b to L4 pipeline.

        It checks that: 
            - The two OS11 planets are detected within 1 pixel of their expected separations
            - The calibration files are correctly associated with the output file

        Data needed: 
            - Coronagraphic dataset - taken from OS11 data
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

    e2e_data_path = os.path.join(e2eoutput_path, "l2_files_noncoron")
    if not os.path.exists(e2e_data_path):
        os.mkdir(e2e_data_path)

    e2eoutput_path = os.path.join(e2eoutput_path, "l2b_to_l3_noncoron_output")
    if not os.path.exists(e2eoutput_path):
        os.mkdir(e2eoutput_path)

    ##################################################
    #### Generate an astrometric calibration file ####
    ##################################################

    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not

    field_path = os.path.join(os.path.dirname(__file__),"..","test_data", "JWST_CALFIELD2020.csv")

    #Create the mock dataset
    mocks.create_astrom_data(field_path=field_path, filedir=e2eoutput_path)
    image_path = os.path.join(e2eoutput_path, 'simcal_astrom.fits')

    # open the image
    dataset = corgidata.Dataset([image_path])
    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, find_threshold=5)

    astrom_cal.save(filedir=e2eoutput_path, filename="mock_astro.fits" )

    # add calibration file to caldb
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(astrom_cal)

    ############################################
    #### Put the OS 11 data into L2B format ####
    ############################################

    #Read in the PSFs
    input_file = 'hlc_os11_no_fpm.fits'
    input_hdul = fits.open(os.path.join(e2edata_path, "hlc_os11_v3", input_file))
    input_image = input_hdul[0].data
    header = input_hdul[0].header
    # I think we work with (0,0) at the center of the pixel
    # if we work with (0, 0) at the bottom left corner of the pixel, then remove the -0.5
    psf_center_x = header['XCENTER'] - 0.5
    psf_center_y = header['YCENTER'] - 0.5
    
    # number of frames we are making
    num_frames = 5

    big_array_size = [1024,1024]

    image_list = []
    for i in range(num_frames): 

        big_array = np.zeros(big_array_size)
        big_rows, big_cols = big_array_size
        small_rows, small_cols = input_image.shape

        # Find the middle indices for the big array
        row_start = (big_rows - small_rows) // 2
        col_start = (big_cols - small_cols) // 2

        # Insert the small array into the middle of the big array
        input_psf = input_image / np.nanmax(input_image) * 1000 # peak counts of 1000
        big_array[row_start:row_start + small_rows, col_start:col_start + small_cols] += input_psf

        # add some gaussian noise so each frame is not the same. 1 count of noise
        big_array += np.random.normal(loc=0, scale=1, size=big_array.shape)
        big_err = np.ones(big_array.shape) * 1.0

        #Update the PSF Center. Might have columns and row mixed up; doesn't matter for now since psf_center_x and psf_center_y are the same. 
        new_psf_center_x = psf_center_x + col_start
        new_psf_center_y = psf_center_y + row_start

        #Make a BIAS HDU
        bias_hdu = fits.ImageHDU(data=np.zeros_like(big_array[0]))
        bias_hdu.nane = 'BIAS'
        bias_hdu.header['PCOUNT'] = 0
        bias_hdu.header['GCOUNT'] = 1
        bias_hdu.header['EXTNAME'] = 'BIAS'

        #Create the new Image object
        mock_pri_header, mock_ext_header, errhdr, dqhdr, biashdr = create_default_L2b_headers()
        new_image = Image(big_array, mock_pri_header, mock_ext_header, err=big_err, input_hdulist=[bias_hdu])
        # new_image.ext_hdr.set('PSF_CEN_X', new_psf_center_x)
        # new_image.ext_hdr.set('PSF_CEN_Y', new_psf_center_y)
        new_image.pri_hdr.set('FRAMET', 1)
        new_image.ext_hdr.set('EXPTIME', 1)
        new_image.pri_hdr.set('ROLL', 0)
        new_image.ext_hdr.set('CFAMNAME','1F')
        new_image.ext_hdr.set('FSMLOS', 0) # non-coron

        # new_image.filename ="CGI_020000199900100{}00{}_20250415T0305102_L2b.fits".format(ibatch,i)
        new_image.filename = "CGI_0200001999001000001_20250415T0305{0:02d}_L2b.fits".format(i)


        image_list.append(new_image)


    #########################################
    #### Save the dataset to a directory ####
    #########################################

    mock_dataset = corgidata.Dataset(image_list)
    mock_dataset.save(filedir=e2e_data_path)

    ## Next step run things through the walker. 

    #####################################
    #### Pass the data to the walker ####
    #####################################

    l2b_data_filelist = sorted(glob.glob(os.path.join(e2e_data_path, "*.fits")))
    walker.walk_corgidrp(l2b_data_filelist, "", e2eoutput_path)

    #Read in an L3 file
    l3_filename = glob.glob(os.path.join(e2eoutput_path, "*L3_.fits"))[0]
    l3_image = Image(l3_filename)

    #Check if there's a WCS header
    assert l3_image.ext_hdr['CTYPE1'] == 'RA---TAN'
    assert l3_image.ext_hdr['CTYPE2'] == 'DEC--TAN'

    #Check if the Bunit is correct
    assert l3_image.ext_hdr['BUNIT'] == 'photoelectron/s'
    
    #Clean up
    this_caldb.remove_entry(astrom_cal)
    shutil.rmtree(e2e_data_path)
    # shutil.rmtree(e2eoutput_path)
    

@pytest.mark.e2e
def test_l3_to_l4(e2eoutput_path):
    '''
    An end-to-end test that takes the L3 data and runs it through the L3 to L4 pipeline.

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

    e2eintput_path = os.path.join(e2eoutput_path, "l2b_to_l3_noncoron_output")

    e2eoutput_path_l4 = os.path.join(e2eoutput_path, "l3_to_l4_noncoron_output")
    if not os.path.exists(e2eoutput_path_l4):
        os.mkdir(e2eoutput_path_l4)

    ##################################################
    #### Generate an astrometric calibration file ####
    ##################################################

    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not

    field_path = os.path.join(os.path.dirname(__file__),"..","test_data", "JWST_CALFIELD2020.csv")

    #Create the mock dataset
    mocks.create_astrom_data(field_path=field_path, filedir=e2eoutput_path_l4)
    image_path = os.path.join(e2eoutput_path_l4, 'simcal_astrom.fits')

    # open the image
    dataset = corgidata.Dataset([image_path])
    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, find_threshold=5)

    astrom_cal.save(filedir=e2eoutput_path_l4, filename="mock_astro.fits" )

    # add calibration file to caldb
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(astrom_cal)

    ##########################################
    #### Generate a flux calibration file ####
    ##########################################

    #Create a mock flux calibration file
    fluxcal_factor = 2e-12
    fluxcal_factor_error = 1e-14
    prhd, exthd, errhdr, dqhdr = create_default_L3_headers()
    fluxcal_fac = corgidata.FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, err_hdr = errhdr, input_dataset = dataset)

    fluxcal_fac.save(filedir=e2eoutput_path_l4, filename="mock_fluxcal.fits")
    this_caldb.create_entry(fluxcal_fac)

    #####################################
    #### Read in the L3 data and run ####
    #####################################

    l3_data_filelist = sorted(glob.glob(os.path.join(e2eintput_path, "*L3_.fits")))

    walker.walk_corgidrp(l3_data_filelist, "", e2eoutput_path_l4)

    ########################################################################
    #### Read in the psf_subtracted images and test for source detection ###
    ########################################################################

    l4_filename = glob.glob(os.path.join(e2eoutput_path_l4, "*L4_.fits"))[0]
    combined_image = Image(l4_filename)
    assert combined_image.filename == l3_data_filelist[-1].split(os.path.sep)[-1].replace("_L3_", "_L4_")
    
    #Find the sources and get their (x,y) coordinate
    y_source, x_source = np.unravel_index(np.nanargmax(combined_image.data), combined_image.data.shape)
    assert combined_image.dq[y_source, x_source] == 0 # check DQ
    peakflux, fwhm, x_source, y_source = pyklip.fakes.gaussfit2d(combined_image.data, x_source, y_source, guesspeak=np.nanmax(combined_image.data))
    xcen = combined_image.ext_hdr['CRPIX1']
    ycen = combined_image.ext_hdr['CRPIX2']
    assert np.isclose(x_source, xcen, atol=1)
    assert np.isclose(y_source, ycen, atol=1)
    assert peakflux == pytest.approx(1000, rel=1e-2)


    #Check that the calibration filenames are appropriately associated
    assert combined_image.ext_hdr['CTCALFN'] == '' # no calibration frame associated for L4 non-coron
    assert combined_image.ext_hdr['FLXCALFN'] == "mock_fluxcal.fits"

    input_filenames = [filepath.split(os.path.sep)[-1] for filepath in l3_data_filelist]

    assert combined_image.ext_hdr['FILE0'] in input_filenames
    assert combined_image.ext_hdr['FILE1'] in input_filenames
    assert combined_image.ext_hdr['FILE2'] in input_filenames
    assert combined_image.ext_hdr['FILE3'] in input_filenames
    assert combined_image.ext_hdr['FILE4'] in input_filenames


    #Clean up
    this_caldb.remove_entry(astrom_cal)
    this_caldb.remove_entry(fluxcal_fac)
    # shutil.rmtree(e2eoutput_path_l4)
    # shutil.rmtree(e2eintput_path)



if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.


    outputdir = thisfile_dir
    #This folder should contain an OS11 folder: ""hlc_os11_v3" with the OS11 data in it.
    e2edata_dir = "/home/jwang/Desktop/CGI_TVAC_Data/" 
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
