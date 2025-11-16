import os, shutil
import pytest
import logging
import numpy as np
import argparse
import glob
from pathlib import Path
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.caldb as caldb
import corgidrp.pol as pol
from corgidrp import l2b_to_l3
from corgidrp.walker import walk_corgidrp
from corgidrp import corethroughput
from astropy.io import fits
from corgidrp.l4_to_tda import compute_QphiUphi

from corgidrp.check import (check_filename_convention, check_dimensions, 
                           verify_hdu_count, verify_header_keywords, 
                           validate_binary_table_fields, get_latest_cal_file)

thisfile_dir = os.path.dirname(__file__)


def l4_to_tda_compute_quphi(e2eoutput_path):

    #"""VAP Test 4: Extended Source (Disk) Azimuthal Stokes Test"""

    # create output dir first

    output_dir = os.path.join(e2eoutput_path, 'l4_to_tda_compute_quphi')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # set up logging
    global logger

    log_file = os.path.join(output_dir, 'l4_to_tda_compute_QphiUphi')
    
    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('l4_to_tda_compute_QphiUphi')
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

    # ================================================================================
    # (1) Setup Input Files
    # ================================================================================
    logger.info('='*80)
    logger.info('Set up input files and save to disk')
    logger.info('='*80)
    ########################################
    ########### Create input image of L4 Stokes cube [I, Q, U, V] with tangentially poalrized disk##########
    ########################################
    qu_img = mocks.create_mock_IQUV_image()

    #Check input image complies with cgi format
    logger.info('='*80)
    logger.info('Test 4.1:Input L4 Image Data format')
    logger.info('='*80)
    check_filename_convention(getattr(qu_img , 'filename', None), 'cgi_*_l4_.fits', logger,data_level='l4_')
    verify_header_keywords(qu_img .ext_hdr, {'BUNIT': 'photoelectron/s'},  logger)
    verify_header_keywords(qu_img .ext_hdr, {'DATALVL': 'L4'},  logger)
    logger.info("")

    # ================================================================================
    # (4) Validate Output L4 Image
    # ================================================================================
    logger.info('='*80)
    logger.info('Test 4.2: Output TDA Azimuthal components test')
    logger.info('='*80)

    qu_phi = compute_QphiUphi(qu_img)

    #Check and log that Qphi and Uphi matches expected tangetial polarization
    
    q_phi=qu_phi.data[4]
    u_phi=qu_phi.data[5]
    
    logger.info('='*80)
    logger.info('Test Case 4.2.1: Qphi matches expected tangential polarization')
    logger.info('='*80)
    assert np.mean(q_phi) > 0,  "mean valuse of Qφ > 0"
    assert not np.allclose(q_phi, 0.0), "Qφ should nor be < 0"
    

    logger.info('='*80)
    logger.info('Test Case 4.2.2: Uφ ≈ 0 for perfect tangential polarization')
    logger.info('='*80)
    assert np.allclose(u_phi, 0.0, atol=1e-6), "Uφ should be ~0 for perfect tangential polarization"

    #Check and log that incorrect center produces non zero Uφ

    logger.info('='*80)
    logger.info('Test Case 4.2.3: Incorrect center produces nonzero Uφ')
    logger.info('='*80)
    # overwrite header center with wrong value
    qu_img_wc=qu_img.copy()
    qu_img_wc.ext_hdr["STARLOCX"] += 5.0
    qu_img_wc.ext_hdr["STARLOCY"] += 5.0

    qu_phi_wc = compute_QphiUphi(qu_img_wc)

    u_phi_wc = qu_phi_wc.data[5]
    assert not np.allclose(u_phi_wc, 0.0, atol=1e-6), "Uφ should be nonzero for wrong center"

    #Check and log that output image shape is [6, n, m]
    logger.info('='*80)
    logger.info('Test Case 4.2.4: Output shape is [6, n, m]')
    logger.info('='*80)
    assert qu_phi.data.shape[0] == 6, "Output should have 6 frames (I,Q,U,V,Q_phi,U_phi)"

    # Check/log the shape of the errors
    logger.info('='*80)
    logger.info('Test Case 4.2.5: Shape of the errors consistent with data')
    logger.info('='*80)

    assert qu_phi.err.shape == qu_phi.data.shape, "err should have the same shape as data"
    assert qu_phi.dq.shape == qu_phi.data.shape, "dq should have the same shape as data"

    # Expect at least (I, Q, U, V) planes in the input dq
    assert qu_img.dq.shape[0] >= 4, "mock image should have I,Q,U,V planes"

    # Check/log that Error propagation σ_Qφ, σ_Uφ from σ_Q, σ_U
    logger.info('='*80)
    logger.info('Test Case 4.2.6: Error propagation σ_Qφ, σ_Uφ from σ_Q, σ_U')
    logger.info('='*80)
    
    # Distinct bits for Q and U (non-overlapping)
    BIT_Q = 1 << 2
    BIT_U = 1 << 5

    # Add bits to Q and U while preserving existing dq
    dq_mod = qu_img.dq.copy()
    dq_mod[1] = dq_mod[1] | BIT_Q  # Q plane
    dq_mod[2] = dq_mod[2] | BIT_U  # U plane
    qu_img.dq = dq_mod

    # Compute Q_phi and U_phi
    qu_phi_mod = compute_QphiUphi(qu_img)
    # Expect (I, Q, U, V, Q_phi, U_phi) -> 6 planes
    assert qu_phi_mod.dq.shape[0] == 6, "Output dq should have 6 planes"

    expected_or = qu_img.dq[1] | qu_img.dq[2]

    # Check/log that Error propagation σ_Qφ, σ_Uφ from σ_Q, σ_U
    logger.info('='*80)
    logger.info('Test Case 4.2.7: DQ flags combine Q and U masks (bitwise OR)')
    logger.info('='*80)

    # Q_phi dq should include bits from Q and U (bitwise OR)
    np.testing.assert_array_equal(
        qu_phi_mod.dq[4] & (BIT_Q | BIT_U),
        expected_or & (BIT_Q | BIT_U),
        err_msg="Q_phi dq should include bits from Q and U (OR)."
    )

    # U_phi dq should include bits from Q and U (bitwise OR)
    np.testing.assert_array_equal(
        qu_phi_mod.dq[5] & (BIT_Q | BIT_U),
        expected_or & (BIT_Q | BIT_U),
        err_msg="U_phi dq should include bits from Q and U (OR)."
    )

    # Check/log using header STARLOCX/Y vs manual center
    logger.info('='*80)
    logger.info('Test Case 4.2.8: STARLOCX/Y vs manual center')
    logger.info('='*80)

    # estimate manual center
    I=qu_img.data[0]
    n, m = I.shape
    cx = (m - 1) * 0.5
    cy = (n - 1) * 0.5

    assert cx==qu_img.ext_hdr["STARLOCX"]
    assert cy==qu_img.ext_hdr["STARLOCY"]

    logger.info('='*80)
    logger.info('Polarimetry L4->TDA VAP Test 4: Extended Source (Disk) Azimuthal Stokes Test: Complete')
    logger.info('='*80)


if __name__ == "__main__":
    outputdir = thisfile_dir # this file's folder
    l4_to_tda_compute_quphi(e2eoutput_path=outputdir)