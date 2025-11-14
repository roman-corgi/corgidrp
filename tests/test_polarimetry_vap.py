import os, glob
import pandas as pd
import numpy as np
import logging
import pytest

from corgidrp.data import Dataset, Image
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.l4_to_tda as l4_to_tda

# ================================================================================
# Logger Setup
# ================================================================================
def setup_logger(output_dir, name='my_e2e_logger'):
    log_file = os.path.join(output_dir, f'{name}.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # formatter
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    # add
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ================================================================================
# Polarimetry Test Function
# ================================================================================
def test_calc_pol_p_and_pa_image(nsigma_tol=3.,
                                 logger=None, log_head=""):
    """
    End-to-end test of `calc_pol_p_and_pa_image` using mock L4 Stokes cubes.

    This function performs the following checks:

    1. Validates that the input Image is in the expected CGI format
       and that its headers (DATALVL, BUNIT) are correct.
    2. Computes the polarization products: 
       - Polarized intensity (P)
       - Fractional polarization (p)
       - Electric-vector position angle (EVPA)
    3. Validates the output images:
       - Shape is [3, H, W]
       - Data Quality (DQ) flags propagate correctly from input to output
    4. Performs statistical consistency checks using chi:
       - (measured - true) / sigma for P, p, and EVPA
       - Checks that median ~ 0 and standard deviation ~ 1 within tolerance

    Args:
        nsigma_tol (float): Tolerance in units of standard errors for chi statistics.
        logger (logging.Logger, optional): Logger instance for output messages.
        log_head (str, optional): Prefix for log messages.

    """
    if logger is None:
        logger = setup_logger('./', name='polVAP')
        
    # ================================================================================
    # (0) Setup Input L4 Image
    # ================================================================================
    seed = 0
    rng = np.random.default_rng(seed)

    # --- Simulation parameters ---
    p_input = 0.1
    theta_input = 10.0
    
    common_kwargs = dict(
        fwhm=1e2,
        I0=1e10,
        p=p_input,
        theta_deg=theta_input,
        rng=rng,
    )
    # Generate mock Stokes cube
    Image_input = mocks.create_mock_stokes_image_l4(badpixel_fraction=1e-3, **common_kwargs)
    
    Image_input_noerr = mocks.create_mock_stokes_image_l4(badpixel_fraction=0.0, add_noise=False, **common_kwargs)
    P_input = np.sqrt(Image_input_noerr.data[1]**2. + Image_input_noerr.data[2]**2.)
    idx = np.where( Image_input.dq[0] != 0 )
    P_input[idx] = np.nan

    # ================================================================================
    # (1) Validate Input Image and Header
    # ================================================================================
    # Check/log that L4 data input complies with cgi format
    if isinstance(Image_input, Image): logger.info(log_head + "Input check passed: CGI format")
    else: logger.info(log_head + "Input check FAILED: non-CGI format")
    
    ext_hdr = Image_input.ext_hdr

    # Check/log that DATALVL = L4
    if ext_hdr['DATALVL'] == "L4": logger.info(log_head + "Header check passed: DATALVL=L4")
    else: logger.info(log_head + "Header check FAILED, Expected:DATALVL=4, but "+ext_hdr['DATALVL'])
    
    # Check/log that BUNIT = photoelectron/s
    if ext_hdr['BUNIT'] == "photoelectron/s": logger.info(log_head + "Header check passed: BUNIT=photoelectron/s")
    else: logger.info(log_head + "Header check FAILED: Expected:BUNIT=photoelectron/s, but "+ext_hdr['BUNIT'])
    
    # ================================================================================
    # (2) Compute polarization products
    # ================================================================================
    Image_pol = l4_to_tda.calc_pol_p_and_pa_image(Image_input)

    P_map = Image_pol.data[0]       # Polarized intensity
    p_map = Image_pol.data[1]       # fractional polarization
    evpa_map = Image_pol.data[2]    # EVPA
    P_map_err = Image_pol.err[0][0]
    p_map_err = Image_pol.err[0][1]
    evpa_map_err = Image_pol.err[0][2]
    P_dq = Image_pol.dq[0]
    p_dq = Image_pol.dq[1]
    evpa_dq = Image_pol.dq[2]

    # ================================================================================
    # (3) Validate Output Images
    # ================================================================================
    # Check/log that Output shape is [3, H, W]
    shape_expected = (3, Image_input.data.shape[1], Image_input.data.shape[2])
    if Image_pol.data.shape == shape_expected:
        logger.info(log_head + f"Output check passed: Image size {shape_expected}")
    else:
        logger.info(log_head + f"Output check FAILED: Image size, expected:{shape_expected}, but {Image_pol.data.shape}")

    # Check/log that DQ flags propagate correctly
    dq_input = np.bitwise_or(np.bitwise_or(Image_input.dq[0], Image_input.dq[1]), Image_input.dq[2])
    dq_sets = [
        (P_dq, "Polarized intensity"),
        (p_dq, "Polarized fraction"),
        (evpa_dq, "Polarization angle"),
    ]
    for dq, label in dq_sets:
        idx = np.where( dq_input != dq )[0]
        if idx.size == 0:
            logger.info(log_head + f"Output check passed: DQ for {label} was propagated correctly")
        else:
            logger.info(log_head + f"Output check FAILED: DQ for {label} was not propagated correctly")

    # Compute chi statistics
    P_chi = (P_map - P_input) / P_map_err
    p_chi = (p_map - p_input) / p_map_err
    evpa_chi = (evpa_map - theta_input) / evpa_map_err
    
    # --- Statistical consistency check ---
    # Each pixel in the mock images is independent, so total samples = n_pixel
    n_pixel = P_map.size
    tol_mean = 1. / np.sqrt(n_pixel) * nsigma_tol
    tol_std = 1. / np.sqrt(2.*(n_pixel-1.)) * nsigma_tol

    idx_P = np.where( P_dq == 0 )
    idx_p = np.where( p_dq == 0 )
    idx_evpa = np.where( evpa_dq == 0 )
    chi_sets = [
        (P_chi[idx_P], "Polarized intensity"),
        (p_chi[idx_p], "Polarized fraction"),
        (evpa_chi[idx_evpa], "Polarization angle"),
    ]
    #print(np.median(P_chi[idx_P]), np.std(P_chi[idx_P]),
    #      np.median(p_chi[idx_p]), np.std(p_chi[idx_p]),
    #      np.median(evpa_chi[idx_evpa]), np.std(evpa_chi[idx_evpa]))

    # Check/log that P computed correctly from Q, U
    # Check/log that p = P/I matches input polarization fraction
    # Check/log that EVPA matches input polarization angle
    # Check/log error propagation through sqrt and division
    for chi, label in chi_sets:
        mean_ok = np.median(chi) == pytest.approx(0.0, abs=tol_mean)
        std_ok =  np.std(chi)    == pytest.approx(1.0, abs=tol_std)
        if mean_ok and std_ok:
            logger.info(log_head + f"Output check passed: {label} matches input within errors")
        else:
            logger.info(log_head + f"Output check FAILED: {label} outside propagated errors")

    return

if __name__ == "__main__":

    logger = setup_logger('./', name='polVAPtest')
    log_head = ""#"l4_to_tda:test_calc_pol_p_and_pa_image, "
    
    test_calc_pol_p_and_pa_image(logger=logger, log_head=log_head)
