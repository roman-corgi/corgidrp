import pytest
import os, shutil
import logging
import argparse
import glob

import corgidrp.mocks as mocks
import corgidrp.check as checks
from corgidrp.data import Dataset, MuellerMatrix, NDMuellerMatrix
from corgidrp.walker import walk_corgidrp

from astropy.io import fits

def run_polcal_test(output_dir, do_ND=False,
                    q_instrumental_polarization = 0.03,  # in percent
                    u_instrumental_polarization = -0.04,  # in percent
                    q_efficiency = 0.9,
                    u_efficiency = 0.85,
                    uq_cross_talk = 0.05,
                    qu_cross_talk = 0.03):
    
    
    logger.info('='*80)
    if do_ND:
        logger.info('ND Polarization Mueller Matrix Calibration END-TO-END TEST')
    else:
        logger.info('Polarization Mueller Matrix Calibration END-TO-END TEST')
    logger.info('='*80)
    logger.info("")
    
    logger.info('='*80)
    logger.info('Pre-test: set up input files and save to disk')
    logger.info('='*80)

    logger.info("Generating new input files...")

    #Get path to this file
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    # path_to_pol_ref_file = os.path.join(current_file_path, "../test_data/stellar_polarization_database.csv")
        # if path_to_pol_ref_file is None:
    # path_to_pol_ref_file = os.path.join(os.path.dirname(__file__), "..", "..","corgidrp","data", "stellar_polarization_database.csv")
#To test this we need this catalog with the same values as the test data also in corgidrp/data/stellar_polarization_database.csv
    path_to_pol_ref_file = os.path.join(current_file_path, "..","test_data","stellar_polarization_database.csv")

    
    mock_dataset = mocks.generate_mock_polcal_dataset(path_to_pol_ref_file,
                                           q_inst=q_instrumental_polarization,
                                           u_inst=u_instrumental_polarization,
                                           q_eff=q_efficiency,
                                           u_eff=u_efficiency,
                                           uq_ct=uq_cross_talk,
                                           qu_ct=qu_cross_talk)
    
    frames = [frame for frame in mock_dataset]
    if do_ND: 
        visit_id = 1234567891234567891
        for frame in frames:
            frame.pri_hdr["VISITID"] = visit_id  #19 digit visit ID for ND
    else:
        visit_id = 2345678912345678912
        for frame in frames:
            frame.pri_hdr["VISITID"] = visit_id  #19 digit visit ID for non-ND
    new_filenames = mocks.rename_files_to_cgi_format(frames, level_suffix="l2b", output_dir=output_dir)
    for i,frame in enumerate(mock_dataset):
        frame.pri_hdr["FILENAME"] = new_filenames[i]
    if do_ND:
        for frame in mock_dataset:
            frame.ext_hdr["FPAMNAME"] = "ND225"


    mock_dataset.save(filedir=output_dir, filenames=new_filenames)

    saved_files = sorted(glob.glob(os.path.join(output_dir, f'cgi_*{visit_id}*_l2b.fits')))
    assert len(saved_files) > 0, f'No saved L2b files found in {output_dir}!'
    l2b_dataset_with_filenames = Dataset(saved_files)
    logger.info(f"Generated and saved {len(saved_files)} new input files")

    logger.info('')
    
    # ================================================================================
    # (2) Validate Input Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 1: Input Image Data Format and Content')
    logger.info('='*80)

    # Validate all input images
    for i, frame in enumerate(l2b_dataset_with_filenames):
        frame_info = f"Frame {i}"
        
        checks.check_filename_convention(getattr(frame, 'filename', None), 'cgi_*_l2b.fits', frame_info, logger)
        checks.check_dimensions(frame.data, (1024, 1024), frame_info, logger)
        checks.verify_header_keywords(frame.ext_hdr, ['CFAMNAME'], frame_info, logger)
        checks.verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L2b'}, frame_info, logger)
        logger.info("")

    logger.info(f"Total input images validated: {len(l2b_dataset_with_filenames)}")
    logger.info("")



    # ================================================================================
    # (3) Run Processing Pipeline
    # ================================================================================
    logger.info('='*80)
    logger.info('Running processing pipeline')
    logger.info('='*80)

    logger.info('Running e2e recipe...')
    recipe = walk_corgidrp(
        filelist=saved_files, 
        CPGS_XML_filepath="",
        outputdir=output_dir
    )
    logger.info("")

    # ================================================================================
    # (4) Validate Output Calibration Product
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Output Calibration Product Data Format and Content')
    logger.info('='*80)

    # Validate output calibration product
    if do_ND:
        suffix = '_ndm_cal.fits'
        product_type = "NDMuellerMatrix calibration product"
        datatype = "NDMuellerMatrix"
    else:
        suffix = '_mmx_cal.fits'
        product_type = "MuellerMatrix calibration product"
        datatype = "MuellerMatrix"
    cal_file = checks.get_latest_cal_file(output_dir, f'*{suffix}', logger)
    checks.check_filename_convention(os.path.basename(cal_file), f'cgi_*{suffix}', product_type, logger)
    
    with fits.open(cal_file) as hdul:
        checks.verify_hdu_count(hdul, 3, product_type, logger)
        
        # Verify HDU0 (header only)
        hdu0 = hdul[0]
        if hdu0.data is None:
            logger.info("HDU0: Header only. Expected: header only. PASS.")
        else:
            logger.info(f"HDU0: Contains data with shape {hdu0.data.shape}. Expected: header only. FAIL.")
        
        checks.check_dimensions(hdul[1].data, (4, 4), cal_file, logger)
        
        # Verify header keywords
        if len(hdul) > 1:
            checks.verify_header_keywords(hdul[1].header, {'DATALVL': 'CAL', 'DATATYPE': datatype}, product_type, logger)

    # ================================================================================
    # (5) Baseline Performance Checks
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 3: Baseline Performance Checks')
    logger.info('='*80)

    # Load and display dispersion model results
    cal_file = checks.get_latest_cal_file(output_dir, f'*{suffix}', logger)
    if do_ND:
        mueller_matrix = NDMuellerMatrix(cal_file)
    else:   
        mueller_matrix = MuellerMatrix(cal_file)

    tolerance = 1e-3
    #Check that the measured mueller matrix is close to the input values
    assert mueller_matrix.data[1,0] == pytest.approx(q_instrumental_polarization/100.0, abs=tolerance)
    assert mueller_matrix.data[2,0] == pytest.approx(u_instrumental_polarization/100.0, abs=tolerance)
    assert mueller_matrix.data[1,1] == pytest.approx(q_efficiency, abs=tolerance)
    assert mueller_matrix.data[2,2] == pytest.approx(u_efficiency, abs=tolerance)
    assert mueller_matrix.data[1,2] == pytest.approx(uq_cross_talk, abs=tolerance)
    assert mueller_matrix.data[2,1] == pytest.approx(qu_cross_talk, abs=tolerance)

    logger.info(f"Recovered Mueller Matrix parameters match input values within {tolerance} tolerance. PASS.")
    logger.info("")

    logger.info("")
    

@pytest.mark.e2e
def test_polcal_e2e(e2edata_path, e2eoutput_path, 
                    q_instrumental_polarization = 0.03,  # in percent
                    u_instrumental_polarization = -0.04,  # in percent
                    q_efficiency = 0.9,
                    u_efficiency = 0.85,
                    uq_cross_talk = 0.05,
                    qu_cross_talk = 0.03):

    #################################################################
    ### First do a bunch of setup, then run the test w and w/o ND ###
    #################################################################

    # create output dir first
    output_dir = os.path.join(e2eoutput_path, 'polcal_e2e')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # set up logging
    global logger
    log_file = os.path.join(output_dir, 'polcal_e2e.log')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('polcal_e2e')
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
    
    #first without ND:
    run_polcal_test(output_dir, do_ND=False,
                    q_instrumental_polarization=q_instrumental_polarization,
                    u_instrumental_polarization=u_instrumental_polarization,
                    q_efficiency=q_efficiency,
                    u_efficiency=u_efficiency,
                    uq_cross_talk=uq_cross_talk,
                    qu_cross_talk=qu_cross_talk)
    #then with ND:
    run_polcal_test(output_dir, do_ND=True,
                    q_instrumental_polarization=q_instrumental_polarization,
                    u_instrumental_polarization=u_instrumental_polarization,
                    q_efficiency=q_efficiency,
                    u_efficiency=u_efficiency,
                    uq_cross_talk=uq_cross_talk,
                    qu_cross_talk=qu_cross_talk)
    
    
@pytest.mark.e2e
def test_polcal_stokes_vap(e2edata_path, e2eoutput_path, 
                    q_instrumental_polarization = 0.03,  # in percent
                    u_instrumental_polarization = -0.04,  # in percent
                    q_efficiency = 0.9,
                    u_efficiency = 0.85,
                    uq_cross_talk = 0.05,
                    qu_cross_talk = 0.03):
    
    #Assume the logger is already set up from the previous test. 
    output_dir = os.path.join(e2eoutput_path, 'polcal_stokes_e2e')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    logger.info('Polarization Calibration Stokes Output TEST')
    logger.info('='*80)
    logger.info("")
    
    logger.info('='*80)
    logger.info('Pre-test: set up input files and save to disk')
    logger.info('='*80)

    logger.info("Generating new input files...")

    #Get path to this file
    current_file_path = os.path.dirname(os.path.abspath(__file__))

    # Get path to polarization reference file
    path_to_pol_ref_file = os.path.join(current_file_path, "..","test_data","stellar_polarization_database.csv")

    mock_dataset = mocks.generate_mock_polcal_dataset(path_to_pol_ref_file,
                                           q_inst=q_instrumental_polarization,
                                           u_inst=u_instrumental_polarization,
                                           q_eff=q_efficiency,
                                           u_eff=u_efficiency,
                                           uq_ct=uq_cross_talk,
                                           qu_ct=qu_cross_talk)
    
    frames = [frame for frame in mock_dataset]
    visit_id = 2345678912345678912
    for frame in frames:
        frame.pri_hdr["VISITID"] = visit_id  #19 digit visit ID for non-ND
    new_filenames = mocks.rename_files_to_cgi_format(frames, level_suffix="l2b", output_dir=output_dir)
    for i,frame in enumerate(mock_dataset):
        frame.pri_hdr["FILENAME"] = new_filenames[i]
    
    mock_dataset.save(filedir=output_dir, filenames=new_filenames)

    saved_files = sorted(glob.glob(os.path.join(output_dir, f'cgi_*{visit_id}*_l2b.fits')))
    assert len(saved_files) > 0, f'No saved L2b files found in {output_dir}!'
    l2b_dataset_with_filenames = Dataset(saved_files)
    logger.info(f"Generated and saved {len(saved_files)} new input files")

    logger.info('')
    
    # ================================================================================
    # (2) Validate Input Images
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 1: Input Image Data Format and Content')
    logger.info('='*80)

    # Validate all input images
    for i, frame in enumerate(l2b_dataset_with_filenames):
        frame_info = f"Frame {i}"
        
        checks.check_filename_convention(getattr(frame, 'filename', None), 'cgi_*_l2b.fits', frame_info, logger)
        checks.check_dimensions(frame.data, (1024, 1024), frame_info, logger)
        checks.verify_header_keywords(frame.ext_hdr, ['CFAMNAME'], frame_info, logger)
        checks.verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L2b'}, frame_info, logger)
        logger.info("")

    logger.info(f"Total input images validated: {len(l2b_dataset_with_filenames)}")
    logger.info("")

    # ================================================================================
    # (3) Run Processing Pipeline
    # ================================================================================
    logger.info('='*80)
    logger.info('Running processing pipeline')
    logger.info('='*80)

    recipe_filename= os.path.join(current_file_path, "l2b_to_polcal_stokes.json")

    logger.info('Running e2e recipe...')
    recipe = walk_corgidrp(
        filelist=saved_files, 
        CPGS_XML_filepath="",
        outputdir=output_dir,
        template=recipe_filename
    )
    logger.info("")

    # ================================================================================
    # (4) Validate Output Calibration Products
    # ================================================================================
    logger.info('='*80)
    logger.info('Test Case 2: Output Calibration Product Data Format and Content')
    logger.info('='*80)

    # Validate output calibration product
    suffix = '_stokes.fits'
    product_type = "Stokes Vector" #No official product type
    datatype = "L3" #No official datatype. 

    l3_files = sorted(glob.glob(os.path.join(output_dir, f'cgi_*{suffix}')))
    assert len(l3_files) > 0, f'No L3 files found in {output_dir}!'


    #No formal filename convention for stokes files. 
    # for i, frame in enumerate(l3_files):
    #     checks.check_filename_convention(os.path.basename(frame), f'cgi_*{suffix}', product_type, logger)

    for i, frame in enumerate(l3_files):
        with fits.open(frame) as hdul:
            checks.verify_hdu_count(hdul, 3, product_type, logger)
            
            # Verify HDU0 (header only)
            hdu0 = hdul[0]
            if hdu0.data is None:
                logger.info("HDU0: Header only. Expected: header only. PASS.")
            else:
                logger.info(f"HDU0: Contains data with shape {hdu0.data.shape}. Expected: header only. FAIL.")
            
            checks.check_dimensions(hdul[1].data, (4,), product_type, logger)
            
            #No formal data level or datatype for stokes files.
            # Verify header keywords
            # if len(hdul) > 1:
                # checks.verify_header_keywords(hdul[1].header, {'DATALVL': 'CAL', 'DATATYPE': datatype}, product_type, logger)

    




if __name__ == "__main__":
    
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir
    e2edata_dir =  "/Users/maxmb/Data/corgi/E2E_Test_Data/"

    ap = argparse.ArgumentParser(description="run the l2b-> Polcal end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to corgidrp e2e Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    test_polcal_e2e(e2edata_dir, outputdir)
    test_polcal_stokes_vap(e2edata_dir, outputdir)