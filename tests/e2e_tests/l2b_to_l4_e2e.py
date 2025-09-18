import argparse
import os
import shutil
import numpy as np
from astropy.io import fits
import corgidrp
from corgidrp.data import Image
from corgidrp.mocks import (create_default_L2b_headers, create_default_L3_headers, 
                            create_synthetic_satellite_spot_image, create_ct_psfs)
from corgidrp.l4_to_tda import find_source
import corgidrp.mocks as mocks
import corgidrp.caldb as caldb
import corgidrp.astrom as astrom
import corgidrp.data as corgidata
import corgidrp.walker as walker
from corgidrp import corethroughput
from corgidrp.check import generate_fits_excel_documentation
import pytest
import glob
from datetime import datetime, timedelta

thisfile_dir = os.path.dirname(__file__) # this file's folder


@pytest.mark.e2e
def test_l2b_to_l3(e2edata_path, e2eoutput_path):
    '''

    An end-to-end test that takes the OS11 data and runs it through the L2B to L4 pipeline.

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

    main_output_dir = os.path.join(e2eoutput_path, "l2b_to_l4_e2e")
    if os.path.exists(main_output_dir):

        shutil.rmtree(main_output_dir)
    os.makedirs(main_output_dir)
    calibrations_dir = os.path.join(main_output_dir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)
    
    # Create input_data subfolder
    input_data_dir = os.path.join(main_output_dir, 'input_l2b')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    ##################################################
    #### Generate an astrometric calibration file ####
    ##################################################

    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not

    field_path = os.path.join(os.path.dirname(__file__),"..","test_data", "JWST_CALFIELD2020.csv")

    #Create the mock dataset (without saving to disk to avoid simcal_astrom.fits)
    mock_dataset = mocks.create_astrom_data(field_path=field_path, filedir=None)
    
    # Create separate directory for astrometric calibration input (L1 data)
    astrom_input_dir = os.path.join(main_output_dir, 'astrom_cal_input')
    if not os.path.exists(astrom_input_dir):
        os.makedirs(astrom_input_dir)
    
    # Generate proper filename and save with standard naming convention
    current_time = datetime.now().strftime('%Y%m%dt%H%M%S%f')[:-5]
    visitid = "0200001999001000001"  # Use consistent visitid
    astrom_filename = f'cgi_{visitid}_{current_time}_l1_.fits'  # This is L1 data, not L2b
    mock_dataset[0].filename = astrom_filename
    mock_dataset[0].save(filedir=astrom_input_dir)
    
    image_path = os.path.join(astrom_input_dir, astrom_filename)

    # open the image
    dataset = corgidata.Dataset([image_path])
    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, find_threshold=5)

    # Generate timestamp for AstrometricCalibration
    base_time = datetime.now()
    ast_time_str = corgidata.format_ftimeutc((base_time.replace(second=(base_time.second + 1) % 60)).isoformat())
    ast_filename = f"cgi_0000000000000000000_{ast_time_str}_ast_cal.fits"
    astrom_cal.save(filedir=calibrations_dir, filename=ast_filename)

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

    #Read in the PSFs
    input_file = 'hlc_os11_frames_with_planets.fits'
    input_hdul = fits.open(os.path.join(e2edata_path,"hlc_os11_v3",input_file))
    input_images = input_hdul[0].data
    header = input_hdul[0].header
    psf_center_x = header['XCENTER']
    psf_center_y = header['YCENTER']
    
    #Get the auxilliary data
    data = np.loadtxt(os.path.join(e2edata_path,"hlc_os11_v3",'hlc_os11_batch_info.txt'), skiprows=2)
    batch = data[:,0].astype(int)
    star = data[:,2].astype(int)
    roll = data[:,3]
    frame_exptime_sec = data[:,4]
    nframes = data[:,6].astype(int)

    nbatch = 22
    istart = 0

    big_array_size = [1024,1024]

    image_list = []
    for ibatch in range(nbatch): 
        # iend = istart + nframes_per_batch
        iend = istart + nframes[ibatch] 

        batch_images = input_images[istart:iend,:,:]

        #collapse these images and stick them into a big array.
        psf_image = np.nanmedian(batch_images, axis=0)

        big_array = np.zeros(big_array_size)
        big_rows, big_cols = big_array_size
        small_rows, small_cols = psf_image.shape

        # Find the middle indices for the big array
        row_start = (big_rows - small_rows) // 2
        col_start = (big_cols - small_cols) // 2

        # Insert the small array into the middle of the big array
        big_array[row_start:row_start + small_rows, col_start:col_start + small_cols] = psf_image

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
        new_image = Image(big_array, mock_pri_header, mock_ext_header, input_hdulist=[bias_hdu])
        # new_image.ext_hdr.set('PSF_CEN_X', new_psf_center_x)
        # new_image.ext_hdr.set('PSF_CEN_Y', new_psf_center_y)
        new_image.pri_hdr.set('FRAMET', frame_exptime_sec[ibatch])
        new_image.ext_hdr.set('EXPTIME', frame_exptime_sec[ibatch])
        new_image.pri_hdr.set('ROLL',-roll[ibatch])
        new_image.ext_hdr.set('FSMPRFL','NFOV')
        new_image.ext_hdr.set('LSAMNAME','NFOV')
        new_image.ext_hdr.set('CFAMNAME','1F')
        new_image.ext_hdr.set('FSMLOS', 1) # tip/tilt enabled only in coronagraphic images
        new_image.ext_hdr.set('FPAMNAME', 'HLC12_C2R1')
        new_image.ext_hdr.set('MASKLOCX', big_cols//2)
        new_image.ext_hdr.set('MASKLOCY', big_cols//2)

        #If Reference star then flag it. 
        if star[ibatch] == 2:
            new_image.pri_hdr.set('PSFREF', 1)

        # Generate proper filename with visitid and current time
        current_time = datetime.now().strftime('%Y%m%dt%H%M%S%f')[:-5]
        visitid = "0200001999001000001"  # Use consistent visitid
        unique_time = (datetime.now() + timedelta(milliseconds=ibatch*100)).strftime('%Y%m%dt%H%M%S%f')[:-5]
        new_image.filename = f"cgi_{visitid}_{unique_time}_l2b.fits"
        #Save the last science filename for later. 
        if star[ibatch] == 1:
            last_sci_filename = new_image.filename


        image_list.append(new_image)

        istart = iend

    #############################################
    #### Add a sat spot image to the dataset ####
    #############################################

    #For now assuming there's just one since the step function in progress doesn't break things up. 
    #We'll want it do later. 

    #Create a mock satellite spot image with the same center as the PSF images. 
    satellite_spot_image = create_synthetic_satellite_spot_image([55,55],1e-4,0,2,14.79,amplitude_multiplier=1000)
    big_array = np.zeros(big_array_size)
    big_rows, big_cols = big_array_size
    small_rows, small_cols = satellite_spot_image.shape

    # Find the middle indices for the big array
    row_start = (big_rows - small_rows) // 2
    col_start = (big_cols - small_cols) // 2

    # Insert the small array into the middle of the big array
    big_array[row_start:row_start + small_rows, col_start:col_start + small_cols] = satellite_spot_image

    mock_satspot_pri_header, mock_satspot_ext_header, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    mock_satspot_pri_header['SATSPOTS'] = 1
    mock_satspot_ext_header['FSMPRFL']='NFOV'

    sat_spot_image = Image(big_array, mock_satspot_pri_header, mock_satspot_ext_header)
    # Generate proper filename for satellite spot
    unique_time = (datetime.now() + timedelta(milliseconds=(ibatch+1)*100)).strftime('%Y%m%dt%H%M%S%f')[:-5]
    sat_spot_image.filename = f"cgi_{visitid}_{unique_time}_l2b.fits"

    image_list.append(sat_spot_image)

    #########################################
    #### Save the dataset to a directory ####
    #########################################

    mock_dataset = corgidata.Dataset(image_list)
    mock_dataset.save(filedir=input_data_dir)

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    ## Next step run things through the walker. 
    
    #####################################
    #### Pass the data to the walker ####
    #####################################

    l2b_data_filelist = sorted(glob.glob(os.path.join(input_data_dir, "*.fits")))
    
    walker.walk_corgidrp(l2b_data_filelist, "", main_output_dir)

    # Organize L3 files into l2b_to_l3 subfolder
    l2b_to_l3_dir = os.path.join(main_output_dir, "l2b_to_l3")
    if not os.path.exists(l2b_to_l3_dir):
        os.mkdir(l2b_to_l3_dir)
    
    # Move L3 files and related recipes to l2b_to_l3 subfolder
    for filename in os.listdir(main_output_dir):
        filepath = os.path.join(main_output_dir, filename)
        if os.path.isfile(filepath):
            if '_l3_' in filename and filename.endswith('.fits'):
                shutil.move(filepath, os.path.join(l2b_to_l3_dir, filename))
            elif 'l2b_to_l3' in filename and filename.endswith('_recipe.json'):
                shutil.move(filepath, os.path.join(l2b_to_l3_dir, filename))

    #Read in an L3 file (now from l2b_to_l3 subfolder)
    l3_filename = glob.glob(os.path.join(l2b_to_l3_dir, "*l3_.fits"))[0]
    l3_image = Image(l3_filename)

    #Check if there's a WCS header
    assert l3_image.ext_hdr['CTYPE1'] == 'RA---TAN'
    assert l3_image.ext_hdr['CTYPE2'] == 'DEC--TAN'

    #Check if the Bunit is correct
    assert l3_image.ext_hdr['BUNIT'] == 'photoelectron/s'
    
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)
    # shutil.rmtree(e2e_data_path)
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

    main_output_dir = os.path.join(e2eoutput_path, "l2b_to_l4_e2e")
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
    
    # Create separate directory for astrometric calibration input (L1 data)
    astrom_input_dir = os.path.join(main_output_dir, 'astrom_cal_input')
    if not os.path.exists(astrom_input_dir):
        os.makedirs(astrom_input_dir)
    
    # Generate proper filename and save with standard naming convention
    current_time = datetime.now().strftime('%Y%m%dt%H%M%S%f')[:-5]
    visitid = "0200001999001000001"  # Use consistent visitid
    astrom_filename = f'cgi_{visitid}_{current_time}_l1_.fits'  # This is L1 data, not L2b
    mock_dataset[0].filename = astrom_filename
    mock_dataset[0].save(filedir=astrom_input_dir)
    
    image_path = os.path.join(astrom_input_dir, astrom_filename)

    # open the image
    dataset = corgidata.Dataset([image_path])
    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=dataset, field_path=field_path, find_threshold=5)

    # Generate timestamp for AstrometricCalibration
    base_time = datetime.now()
    ast_time_str = corgidata.format_ftimeutc((base_time.replace(second=(base_time.second + 1) % 60)).isoformat())
    ast_filename = f"cgi_0000000000000000000_{ast_time_str}_ast_cal.fits"
    astrom_cal.save(filedir=calibrations_dir, filename=ast_filename)

    # add calibration file to caldb
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(astrom_cal)

    ###########################
    #### Make dummy CT cal ####
    ###########################

        # Dataset with some CT profile defined in create_ct_interp
    # Pupil image
    pupil_image = np.zeros([1024, 1024])
    # Set it to some known value for a selected range of pixels
    pupil_image[510:530, 510:530]=1
    prhd, exthd_pupil, errhdr, dqhdr = create_default_L3_headers()
    # DRP
    # cfam filter
    exthd_pupil['CFAMNAME'] = '1F'
    # Add specific values for pupil images:
    # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'

    data_psf, psf_loc_in, half_psf = create_ct_psfs(50, cfam_name='1F',
    n_psfs=100)
    
    err = np.ones([1024,1024]) 
    data_ct_interp = [Image(pupil_image,pri_hdr = prhd,
        ext_hdr = exthd_pupil, err = err)]
    # Set of off-axis PSFs with a CT profile defined in create_ct_interp
    # First, we need the CT FPM center to create the CT radial profile
    # We can use a miminal dataset to get to know it
    data_ct_interp += [data_psf[0]]
    ct_cal_tmp = corethroughput.generate_ct_cal(corgidata.Dataset(data_ct_interp))
    # Generate timestamp for CoreThroughputMap calibration
    ctm_time_str = corgidata.format_ftimeutc((base_time.replace(second=(base_time.second + 2) % 60)).isoformat())
    ctm_filename = f"cgi_0000000000000000000_{ctm_time_str}_ctm_cal.fits"
    ct_cal_tmp.save(filedir=calibrations_dir, filename=ctm_filename)
    this_caldb.create_entry(ct_cal_tmp)

    ##########################################
    #### Generate a flux calibration file ####
    ##########################################

    #Create a mock flux calibration file
    fluxcal_factor = 2e-12
    fluxcal_factor_error = 1e-14
    prhd, exthd, errhd, dqhd = create_default_L3_headers()
    # Set consistent header values for flux calibration factor
    exthd['CFAMNAME'] = '1F'
    exthd['DPAMNAME'] = 'PUPIL'
    exthd['LSAMNAME'] = 'OPEN'
    exthd['FSAMNAME'] = 'OPEN'
    exthd['FPAMNAME'] = 'OPEN_12'
    fluxcal_fac = corgidata.FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, err_hdr = errhd, input_dataset = dataset)

    # Generate timestamp for FluxcalFactor calibration
    abf_time_str = corgidata.format_ftimeutc((base_time.replace(second=(base_time.second + 3) % 60)).isoformat())
    abf_filename = f"cgi_0000000000000000000_{abf_time_str}_abf_cal.fits"
    fluxcal_fac.save(filedir=calibrations_dir, filename=abf_filename)
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
    #### Read in the psf_subtracted images and test for source detection ###
    ########################################################################

    l4_filename = glob.glob(os.path.join(main_output_dir, "*l4_.fits"))[0]
    psf_subtracted_image = Image(l4_filename)
    psf_subtracted_image.data = psf_subtracted_image.data[-1,:,:] #Just pick one of the KL modes for now
    
    #Find the sources and get their distances from the center
    psf_subtracted_image_with_source = find_source(psf_subtracted_image)
    source_header = psf_subtracted_image_with_source.ext_hdr

    snyx = np.array([list(map(float, source_header[key].split(','))) for key in source_header if key.startswith("SNYX")])
    xcen = psf_subtracted_image_with_source.ext_hdr['CRPIX1']
    ycen = psf_subtracted_image_with_source.ext_hdr['CRPIX2']
    source_distances =np.sort(np.sqrt((snyx[:,1] - xcen)**2 + (snyx[:,2] - ycen)**2))

    ### Get the expected distances
    pixel_scale = 21 #mas/pixel
    lambda_c = 575e-9 #center wavelength of band 1 in m
    roman_D = 2.37 #roman primary diameter
    expected_separations_lambda_over_D = np.array([3.5, 4.5]) #OS11 injected planets
    expected_separations_arcsec = expected_separations_lambda_over_D *  lambda_c / roman_D * 206265
    expected_separations_pixels = expected_separations_arcsec / pixel_scale * 1000 

    #Check that the detected sources are within 1 pixel of the expected separations
    #Assumes that only the correct sources were detected. 
    for i,source_distance in enumerate(expected_separations_pixels):
        assert np.isclose(source_distance, source_distances[i], atol=1)
    print("Found all the sources!")

    #Check that the calibration filenames are appropriately associated
    assert psf_subtracted_image.ext_hdr['CTCALFN'] == ctm_filename
    assert psf_subtracted_image.ext_hdr['FLXCALFN'] == abf_filename
    print("Filenames associated correctly!")
    
    print('e2e test for l3_to_l4 calibration passed')

    # Generate Excel documentation for L4 science data files
    l4_files = glob.glob(os.path.join(main_output_dir, "*_l4_.fits"))
    if l4_files:
        # Document the first L4 file as a representative example
        l4_file = l4_files[0]
        excel_output_path = os.path.join(main_output_dir, "l4_data_documentation.xlsx")
        generate_fits_excel_documentation(l4_file, excel_output_path)
        print(f"Excel documentation generated for L4 data: {excel_output_path}")

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)
    # shutil.rmtree(e2eoutput_path_l4)
    # shutil.rmtree(e2eintput_path)
    # shutil.rmtree(os.path.join(pathlib.Path.home(), ".corgidrp",'KLIP_SUB'))



if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.


    outputdir = thisfile_dir
    #This folder should contain an OS11 folder: ""hlc_os11_v3" with the OS11 data in it.
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
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
