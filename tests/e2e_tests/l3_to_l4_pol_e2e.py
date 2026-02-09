import os, shutil
import pytest
import logging
import numpy as np
import argparse
import glob

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.caldb as caldb
import corgidrp.pol as pol
from corgidrp import l2b_to_l3
from corgidrp.walker import walk_corgidrp
from corgidrp import corethroughput
from astropy.io import fits

import json

from corgidrp.check import (check_filename_convention, check_dimensions, 
                           verify_hdu_count, verify_header_keywords, 
                           validate_binary_table_fields, get_latest_cal_file)

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_l3_to_l4_pol_e2e(e2edata_path, e2eoutput_path):
    # create output dir first
    output_dir = os.path.join(e2eoutput_path, 'l3_to_l4_pol_e2e')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    calibrations_dir = os.path.join(output_dir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)

    # set up logging
    global logger
    log_file = os.path.join(output_dir, 'l3_to_l4_pol_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('l3_to_l4_pol_e2e')
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    ## Create the astrometric calibration

    # Create calibrations subfolder for mock calibration products
    calibrations_dir = os.path.join(output_dir, "calibrations")
    if not os.path.exists(calibrations_dir):
        os.mkdir(calibrations_dir)

    ###### Setup necessary calibration files
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()  # connection to cal DB
    
    # ================================================================================
    # (1) Setup Input Files
    # ================================================================================
    logger.info('='*80)
    logger.info('Pre-test: set up input files and save to disk')
    logger.info('='*80)
    ########################################
    ########### Create astrom cal ##########
    ########################################

    # CREATE DISTORTED POL DATASET
    # Create a bad pixel map for testing
    datashape = (2, 1024, 1024)  # pol data shape
    bpixmap = np.zeros(datashape)

    field_path = os.path.join(os.path.dirname(__file__), "..","test_data", "JWST_CALFIELD2020.csv")

    distortion_coeffs_path = os.path.join(os.path.dirname(__file__), "..","test_data", "distortion_expected_coeffs.csv")
    distortion_coeffs = np.genfromtxt(distortion_coeffs_path)
    distortion_dataset_base = mocks.create_astrom_data(field_path, distortion_coeffs_path=distortion_coeffs_path,
                                                       bpix_map=bpixmap[0], sim_err_map=True)


    astromcal_data = np.concatenate((np.array([80.553428801, -69.514096821, 21.8, 45, 0, 0]), distortion_coeffs),
                                    axis=0)
    astrom_cal = data.AstrometricCalibration(astromcal_data,
                                             pri_hdr=distortion_dataset_base[0].pri_hdr,
                                             ext_hdr=distortion_dataset_base[0].ext_hdr,
                                             input_dataset=distortion_dataset_base)
    mocks.rename_files_to_cgi_format(list_of_fits=[astrom_cal], output_dir=calibrations_dir, level_suffix="ast_cal")
    
    
    this_caldb.create_entry(astrom_cal)

    ###########################
    #### Make dummy CT cal ####
    ###########################

    # Dataset with some CT profile defined in create_ct_interp
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
    ct_dataset0 = data_psf[0]
    ct_dataset0.ext_hdr['FPAMNAME'] = 'HLC12_C2R1'  # set FPM to a coronagraphic one
    
    err = np.ones([1024,1024])
    data_ct_interp = [ct_dataset0, data.Image(pupil_image,pri_hdr = prhd,
        ext_hdr = exthd_pupil, err = err)]
    # Set of off-axis PSFs with a CT profile defined in create_ct_interp
    # First, we need the CT FPM center to create the CT radial profile
    # We can use a miminal dataset to get to know it
    ct_cal_tmp = corethroughput.generate_ct_cal(data.Dataset(data_ct_interp))
    mocks.rename_files_to_cgi_format(list_of_fits=[ct_cal_tmp], output_dir=calibrations_dir, level_suffix="ctm_cal")
    this_caldb.create_entry(ct_cal_tmp)

    ##########################################
    #### Generate a flux calibration file ####
    ##########################################

    #Create a mock flux calibration file
    fluxcal_factor = 2e-12
    fluxcal_factor_error = 1e-14
    prhd, exthd, errhd, dqhd = mocks.create_default_L3_headers()
    # Set consistent header values for flux calibration factor
    exthd['CFAMNAME'] = '1F'
    exthd['DPAMNAME'] = 'POL0'
    exthd['FPAMNAME'] = 'HLC12_C2R1'
    fluxcal_fac = data.FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, err_hdr = errhd, input_dataset = distortion_dataset_base)

    mocks.rename_files_to_cgi_format(list_of_fits=[fluxcal_fac], output_dir=calibrations_dir, level_suffix="abf_cal")
    this_caldb.create_entry(fluxcal_fac)

    #################################################
    ########### Create Mueller Matrix cals ##########
    #################################################

        # define mueller matrices and stokes vectors
    nd_mueller_matrix = np.array([
        [0.8, 0.1, 0, 0],
        [0.05, -0.75, 0, 0],
        [0.05, 0.05, 0.75, 0],
        [0, 0, 0, 0.95]
    ])
    system_mueller_matrix = np.array([
        [0.9, -0.02, 0, 0],
        [0.01, -0.8, 0, 0],
        [0, 0, 0.8, 0.005],
        [0, 0, -0.01, 0.9]
    ])

    #Create dataset because we need to input something: 
    prihdr, exthdr = mocks.create_default_L1_headers()
    dummy1 = data.Image(np.zeros((10,10)), pri_hdr=prihdr, ext_hdr=exthdr)
    dummy2 = data.Image(np.zeros((10,10)), pri_hdr=prihdr, ext_hdr=exthdr)
    input_dataset = data.Dataset([dummy1, dummy2])

    mm_prihdr, mm_exthdr, _, _ = mocks.create_default_calibration_product_headers()
    system_mm_cal = data.MuellerMatrix(system_mueller_matrix, pri_hdr=mm_prihdr.copy(), ext_hdr=mm_exthdr.copy(), input_dataset=input_dataset)
    nd_exthdr = mm_exthdr.copy()
    nd_exthdr['FPAMNAME'] = 'ND225'
    nd_mm_cal = data.NDMuellerMatrix(nd_mueller_matrix, pri_hdr=mm_prihdr.copy(), ext_hdr=nd_exthdr, input_dataset=input_dataset)

    mocks.rename_files_to_cgi_format(list_of_fits=[system_mm_cal], output_dir=calibrations_dir, level_suffix="mmx_cal")
    mocks.rename_files_to_cgi_format(list_of_fits=[nd_mm_cal], output_dir=calibrations_dir, level_suffix="ndm_cal")

    this_caldb.create_entry(system_mm_cal)
    this_caldb.create_entry(nd_mm_cal)

    ####################################################
    ########### Generate Main science Dataset ##########
    ####################################################

    #The dataset needs to include:
    # - L3 data with unocculted observations of science and reference stars
    # - L3 data with sat spots for science and reference stars
    # - L3 science and reference data. 

    stellar_stokes_vectors = {"ScienceStar": {"stokes_vector": [1,0,0,0],
                                              "nd_rotation_angle":45,
                                              "nd_amplitude": 100,
                                              "fpm_rotation_angles":[32, 62],
                                              "amplitude": 50000},
                              "RefStar": {"stokes_vector": [1,-0.01,-0.02,0],
                                          "nd_rotation_angle":45,
                                          "nd_amplitude": 150,
                                          "fpm_rotation_angles":[32, 62],
                                          "amplitude": 100000}}

    # create mock images and dataset
    # use gaussians as mock star, scale by polarized intensity to construct polarimetric data
    # add background noise to star to prevent divide by zero

    # Create the mock stellar polarization measurements (from the ND data)
    # Get the size we need (see the defaults in split_image_by_polarization_states
    diam = 2.363114 #telescope diameter in meters
    passband_center = 573.8  # nm
    radius_arcsec = 9.7 * ((passband_center * 1e-9) / diam) * 206265 #9.7 lambda/D in arcseconds
    radius_pix = int(round(radius_arcsec / 0.0218)) # pixel scale is 0.0218 arcsec/pix
    padding = 5
    image_size = 2 * (radius_pix + padding)
    
    # polarizer_angles = [[0,45],[90,135]]  # in degrees
    polarizers = {"POL0":[0,90], "POL45":[45,135]}

    number_nd_images = 3
    number_of_science_images = 4
    number_of_sat_spot_images = 2

    wide_psf_sigma = 10

    #Set up numpy random seed for reproducibility
    # np.random.seed(42)

    input_image_list = []
    # Build the FPM Datasets
    for targetname, stokes_info in stellar_stokes_vectors.items():
        for rotation_angle in stokes_info["fpm_rotation_angles"]:
            stokes_vector = stokes_info["stokes_vector"]

            stellar_sys_stokes = system_mueller_matrix @ pol.rotation_mueller_matrix(rotation_angle) @ stokes_vector
            
            ## Make the normal science data
            #TODO: Add offsets between the two Wollaston beams, and offsets for different rotation angles. 
            for i in range(number_of_science_images):
                for wollaston in polarizers.keys():
                    pol_angles = polarizers[wollaston]
                    # find intensities at each polarizer angle for star 1
                    stellar_sys_o_beam = (pol.lin_polarizer_mueller_matrix(pol_angles[0]) @ stellar_sys_stokes)[0]
                    stellar_sys_e_beam = (pol.lin_polarizer_mueller_matrix(pol_angles[1]) @ stellar_sys_stokes)[0]

                    
                    #First a central PSF - we'll make a wider Gaussian
                    stellar_image_1 = mocks.gaussian_array(amp=stokes_info["amplitude"],array_shape=[image_size,image_size],sigma=wide_psf_sigma) + np.random.normal(loc=0.0, scale=5, size=(image_size,image_size))
                    stellar_image_2 = mocks.gaussian_array(amp=stokes_info["amplitude"],array_shape=[image_size,image_size],sigma=wide_psf_sigma) + np.random.normal(loc=0.0, scale=5, size=(image_size,image_size))
                    #TODO: Add a polarized companion here later.

                    stellar_sys_wp_data = np.array([stellar_sys_o_beam * stellar_image_1, stellar_sys_e_beam * stellar_image_2])

                    # Create the new Image object passing in the error header
                    prihdr,exthdr,errhdr,dqhdr=mocks.create_default_L3_headers()
                    stellar_sys_wp_img=data.Image(stellar_sys_wp_data,
                                                  pri_hdr=prihdr.copy(),
                                                  ext_hdr=exthdr.copy(),
                                                  err_hdr=errhdr.copy(),
                                                  dq_hdr=dqhdr.copy())
                    #Check if error header has LAYER_1
                    #print(f"Image err_hdr has LAYER_1: {stellar_sys_wp_img.err_hdr.get('LAYER_1','NOT FOUND')}")
                    #print("="*60+"\n")

                    #Update Headers
                    stellar_sys_wp_img.pri_hdr['TARGET'] = targetname
                    stellar_sys_wp_img.ext_hdr['DPAMNAME'] = wollaston
                    stellar_sys_wp_img.pri_hdr['PA_APER'] = rotation_angle
                    stellar_sys_wp_img.ext_hdr['FSMPRFL'] = 'NFOV'

                    # wcs_header = generate_wcs(rotation_angles[i], 
                    #                   [psfcentx,psfcenty],
                    #                   platescale=0.0218).to_header()
            
                    # # wcs_header._cards = wcs_header._cards[-1]
                    # exthdr.extend(wcs_header)

                    input_image_list.append(stellar_sys_wp_img)

            ## Make the normal science data with sat spots
            for i in range(number_of_sat_spot_images):
                for wollaston in polarizers.keys():
                    pol_angles = polarizers[wollaston]
                    # find intensities at each polarizer angle for star 1
                    stellar_sys_o_beam = (pol.lin_polarizer_mueller_matrix(pol_angles[0]) @ stellar_sys_stokes)[0]
                    stellar_sys_e_beam = (pol.lin_polarizer_mueller_matrix(pol_angles[1]) @ stellar_sys_stokes)[0]

                    
                    #Here we'll make the simmed images the same as in the unit test test_align_frames() in test_polarimetry.py
                    image_WP_nfov_sp = mocks.create_mock_l2b_polarimetric_image_with_satellite_spots(dpamname=wollaston,
                    observing_mode='NFOV',
                    left_image_value=1,
                    right_image_value=1,
                    bg_sigma=1,
                    amplitude_multiplier=stokes_info["amplitude"]*1000)

                    #Split the images
                    temp_dataset = data.Dataset([image_WP_nfov_sp])
                    temp_dataset = l2b_to_l3.divide_by_exptime(temp_dataset)
                    split_dataset = l2b_to_l3.split_image_by_polarization_state(temp_dataset)
                    split_dataset = l2b_to_l3.update_to_l3(split_dataset)

                    split_frame = split_dataset.frames[0]

                    #Add in the central star
                    split_frame.data[0] += mocks.gaussian_array(amp=stokes_info["amplitude"],array_shape=[image_size,image_size],sigma=wide_psf_sigma) + np.random.normal(loc=0.0, scale=1, size=(image_size,image_size))
                    split_frame.data[1] += mocks.gaussian_array(amp=stokes_info["amplitude"],array_shape=[image_size,image_size],sigma=wide_psf_sigma) + np.random.normal(loc=0.0, scale=1, size=(image_size,image_size))

                    #Update Headers
                    split_frame.pri_hdr['TARGET'] = targetname
                    split_frame.ext_hdr['DPAMNAME'] = wollaston
                    split_frame.pri_hdr['PA_APER'] = rotation_angle
                    split_frame.ext_hdr['SATSPOTS'] = 1

                    input_image_list.append(split_frame)


    #Build the ND Datasets - For each target cycle over rotation angles and Wollastons
    for targetname, stokes_info in stellar_stokes_vectors.items():
        rotation_angle = stokes_info["nd_rotation_angle"]
        stokes_vector = stokes_info["stokes_vector"]

        stellar_nd_stokes = nd_mueller_matrix @ pol.rotation_mueller_matrix(rotation_angle) @ stokes_vector    

        for i in range(number_nd_images):   
            for wollaston in polarizers.keys():
                pol_angles = polarizers[wollaston]
                # find intensities at each polarizer angle for star 1
                stellar_nd_o_beam = (pol.lin_polarizer_mueller_matrix(pol_angles[0]) @ stellar_nd_stokes)[0]
                stellar_nd_e_beam = (pol.lin_polarizer_mueller_matrix(pol_angles[1]) @ stellar_nd_stokes)[0]

                # create stellar images
                stellar_image_1 = mocks.gaussian_array(amp=stokes_info["nd_amplitude"],array_shape=[image_size,image_size]) + np.random.normal(loc=0.0, scale=1, size=(image_size,image_size))
                stellar_image_2 = mocks.gaussian_array(amp=stokes_info["nd_amplitude"],array_shape=[image_size,image_size]) + np.random.normal(loc=0.0, scale=1, size=(image_size,image_size))

                stellar_nd_wp_data = np.array([stellar_nd_o_beam * stellar_image_1, stellar_nd_e_beam * stellar_image_2])

                # create default header including error header this time
                prihdr,exthdr,errhdr,dqhdr=mocks.create_default_L3_headers()
                stellar_nd_wp_img=data.Image(stellar_nd_wp_data,
                                             pri_hdr=prihdr.copy(),
                                             ext_hdr=exthdr.copy(),
                                             err_hdr=errhdr.copy(),
                                             dq_hdr=dqhdr.copy())
                #Update Headers
                stellar_nd_wp_img.pri_hdr['TARGET'] = targetname
                stellar_nd_wp_img.ext_hdr['DPAMNAME'] = wollaston
                stellar_nd_wp_img.ext_hdr['FPAMNAME'] = "ND225"
                stellar_nd_wp_img.pri_hdr['PA_APER'] = rotation_angle
                stellar_nd_wp_img.ext_hdr['FSMPRFL'] = 'NFOV'

                input_image_list.append(stellar_nd_wp_img)

    # input_dataset = data.Dataset(input_image_list)
    saved_files = mocks.rename_files_to_cgi_format(input_image_list, level_suffix="l3", output_dir=output_dir)

    # ================================================================================
    # (2) Validate Input Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 1: Input L3 Image Data Format and Content')
    logger.info('='*80)

    # Validate all input images
    for i, frame in enumerate(input_image_list):
        frame_info = f"Frame {i}"
        
        check_filename_convention(getattr(frame, 'filename', None), 'cgi_*_l3_.fits', frame_info, logger,data_level='l3_')
        check_dimensions(frame.data, (2,54,54), frame_info, logger)
        verify_header_keywords(frame.ext_hdr, ['CFAMNAME'], frame_info, logger)
        verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L3'}, frame_info, logger)
        logger.info("")

    logger.info(f"Total input images validated: {len(input_image_list)}")
    logger.info("")

    # ================================================================================
    # (3) Run Processing Pipeline
    # ================================================================================
    logger.info('='*80)
    logger.info('Running processing pipeline')
    logger.info('='*80)

    # import IPython; IPython.embed()
    logger.info('Running e2e recipe...')
    filelist = sorted(glob.glob(os.path.join(output_dir, '*_l3_.fits')))
    recipe = walk_corgidrp(
        filelist=filelist, 
        CPGS_XML_filepath="",
        outputdir=output_dir
    )
    logger.info("")

    # ================================================================================
    # (4) Validate Output L4 Image
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Output L4 Image Data Format and Content')
    logger.info('='*80)

    l4_filelist = sorted(glob.glob(os.path.join(output_dir, '*_l4_.fits')))
    
    n_l4_files = len(l4_filelist)
    if n_l4_files == 1:
        logger.info(f"L4 Output File Count: {n_l4_files}. PASS")
    else:
        logger.info(f"L4 Output File Count: {n_l4_files}. Expected 1. FAIL")


    #There should only be one. 

    frame_info = f"L4 Output Frame"

    img = data.Image(l4_filelist[0])
    # Verify HDU count
    with fits.open(l4_filelist[0]) as hdul:
        verify_hdu_count(hdul, 6, frame_info, logger)  # L4 should have 5 HDUs

    # Verify data level
    verify_header_keywords(img.ext_hdr, {'DATALVL': 'L4'}, frame_info, logger)
    
    # Check this is polarimetry data
    dpam = img.ext_hdr.get('DPAMNAME', '')
    if dpam in ('POL0', 'POL45'):
        logger.info(f"{frame_info}: DPAMNAME = '{dpam}' (polarimetry). PASS")
    else:
        logger.info(f"{frame_info}: DPAMNAME = '{dpam}'. Expected POL0 or POL45. FAIL")
    

    # Check data dimensions - should always be polarimetry datacube (4, N, N)
    if len(img.data.shape) == 3 and img.data.shape[0] == 4:
        logger.info(f"{frame_info}: Stokes datacube shape {img.data.shape}. PASS")
    else:
        logger.info(f"{frame_info}: Expected Stokes datacube (4, N, N), got {img.data.shape}. FAIL")

    #Check that core throughput and flux calibration products have been linked. 
    verify_header_keywords(img.ext_hdr, ['CTCALFN', 'FLXCALFN', 'STARLOCX','STARLOCY'], frame_info, logger)

    #Check that the stellar polarization is report: 
    if "stellar Q value: " in str(img.ext_hdr['HISTORY']) and "stellar U value: " in str(img.ext_hdr['HISTORY']):
        logger.info(f"{frame_info}: Stellar polarization Q reported in HISTORY. PASS")
    else:
        logger.info(f"{frame_info}: Stellar polarization Q not found in HISTORY. FAIL")

    #Check that the mueller matrix calibrations are reported: 
    recipe = json.loads(img.ext_hdr['RECIPE'])
    step_names = [step['name'] for step in recipe['steps']]
    if "combine_polarization_states" in step_names:
        if "MuellerMatrix" in recipe['steps'][step_names.index("combine_polarization_states")]['calibs']:
            logger.info(f"{frame_info}: Mueller Matrix calibration used reported in RECIPE. PASS")
        else:
            logger.info(f"{frame_info}: Mueller Matrix calibration used not found in RECIPE. FAIL")


if __name__ == "__main__":
    #e2edata_dir = "/Users/macuser/Roman/corgidrp_develop/calibration_notebooks/TVAC"
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2b->boresight end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_l3_to_l4_pol_e2e(e2edata_dir, outputdir)