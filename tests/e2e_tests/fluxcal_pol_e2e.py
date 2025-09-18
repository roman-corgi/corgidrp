import argparse
import os, shutil
import glob
import pytest
import numpy as np
from datetime import datetime, timedelta

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.fluxcal as fluxcal
from corgidrp import caldb
from corgidrp.check import generate_fits_excel_documentation

@pytest.mark.e2e
def test_expected_results_e2e(e2eoutput_path):
    #mock a point source image
    fwhm = 3
    star_flux = 1.5e-09 #erg/(s*cm^2*AA)
    cal_factor = star_flux/200
    # split the flux unevenly by polarization
    star_flux_left = 0.6 * star_flux
    star_flux_right = 0.4 * star_flux
    flux_image_WP1 = mocks.create_pol_flux_image(star_flux_left, star_flux_right, fwhm, cal_factor, dpamname='POL0')
    flux_image_WP1.ext_hdr['BUNIT'] = 'photoelectron'
    flux_dataset_WP1 = data.Dataset([flux_image_WP1])
    flux_image_WP2 = mocks.create_pol_flux_image(star_flux_left, star_flux_right, fwhm, cal_factor, dpamname='POL45')
    flux_image_WP2.ext_hdr['BUNIT'] = 'photoelectron'
    flux_dataset_WP2 = data.Dataset([flux_image_WP2])

    output_dir = os.path.join(e2eoutput_path, 'flux_cal_pol_e2e')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Create input_data subfolders
    wp1_dir = os.path.join(output_dir, 'WP1')
    wp2_dir = os.path.join(output_dir, 'WP2')
    if not os.path.exists(wp1_dir):
        os.makedirs(wp1_dir)
    if not os.path.exists(wp2_dir):
        os.makedirs(wp2_dir)
    input_data_dir_WP1 = os.path.join(wp1_dir, 'input_l2b')
    input_data_dir_WP2 = os.path.join(wp2_dir, 'input_l2b')
    if not os.path.exists(input_data_dir_WP1):
        os.makedirs(input_data_dir_WP1)
    if not os.path.exists(input_data_dir_WP2):
        os.makedirs(input_data_dir_WP2)

    # Generate proper filenames with visitid and current time
    base_time = datetime.now()
    current_time = base_time.strftime('%Y%m%dt%H%M%S%f')[:-5]
    # Create second timestamp with 0.1 seconds added
    current_time_wp2 = (base_time + timedelta(milliseconds=100)).strftime('%Y%m%dt%H%M%S%f')[:-5]
    
    # Extract visit ID from primary header VISITID keyword
    visitid = flux_image_WP1.pri_hdr.get('VISITID', None)
    if visitid is not None:
        # Convert to string and pad to 19 digits
        visitid = str(visitid).zfill(19)
    else:
        # Fallback: use default visitid
        visitid = "0000000000000000000"

    # Create proper L2b filenames with unique timestamps
    flux_dataset_WP1.save(input_data_dir_WP1, [f'cgi_{visitid}_{current_time}_l2b.fits'])
    flux_dataset_WP2.save(input_data_dir_WP2, [f'cgi_{visitid}_{current_time_wp2}_l2b.fits'])

    data_filelist_WP1 = []
    data_filelist_WP2 = []

    for f in os.listdir(input_data_dir_WP1):
        data_filelist_WP1.append(os.path.join(input_data_dir_WP1, f))
    
    for f in os.listdir(input_data_dir_WP2):
        data_filelist_WP2.append(os.path.join(input_data_dir_WP2, f))

    ####### Run the DRP walker for WP1
    print('Running walker for WP1')
    walker.walk_corgidrp(data_filelist_WP1, '', wp1_dir)
    fluxcal_file_WP1 = glob.glob(os.path.join(wp1_dir, '*abf_cal*.fits'))[0]
    fluxcal_image_WP1 = data.Image(fluxcal_file_WP1)

    #check that the calibration file is configured correctly
    assert fluxcal_image_WP1.pri_hdr['NAXIS'] == 0
    assert fluxcal_image_WP1.ext_hdr['DPAMNAME'] == 'POL0'
    assert fluxcal_image_WP1.data.shape == (1,)
    assert fluxcal_image_WP1.err.shape == (1,)
    assert fluxcal_image_WP1.dq.shape == (1,)

    #output values
    flux_fac_WP1 = data.FluxcalFactor(fluxcal_file_WP1)
    print("used color filter", flux_fac_WP1.filter)
    print("used ND filter", flux_fac_WP1.nd_filter)
    print("fluxcal factor", flux_fac_WP1.fluxcal_fac)
    print("fluxcal factor error", flux_fac_WP1.fluxcal_err)
    assert flux_fac_WP1.fluxcal_fac == pytest.approx(cal_factor, abs = 1.5 * flux_fac_WP1.fluxcal_err)


    ####### Run the DRP walker for WP2
    print('Running walker for WP2')
    walker.walk_corgidrp(data_filelist_WP2, '', wp2_dir)
    fluxcal_file_WP2 = glob.glob(os.path.join(wp2_dir, '*abf_cal*.fits'))[0]
    fluxcal_image_WP2 = data.Image(fluxcal_file_WP2)

    #check that the calibration file is configured correctly
    assert fluxcal_image_WP2.pri_hdr['NAXIS'] == 0
    assert fluxcal_image_WP2.ext_hdr['DPAMNAME'] == 'POL45'
    assert fluxcal_image_WP2.data.shape == (1,)
    assert fluxcal_image_WP2.err.shape == (1,)
    assert fluxcal_image_WP2.dq.shape == (1,)

    #output values
    flux_fac_WP2 = data.FluxcalFactor(fluxcal_file_WP2)
    print("used color filter", flux_fac_WP2.filter)
    print("used ND filter", flux_fac_WP2.nd_filter)
    print("fluxcal factor", flux_fac_WP2.fluxcal_fac)
    print("fluxcal factor error", flux_fac_WP2.fluxcal_err)
    assert flux_fac_WP2.fluxcal_fac == pytest.approx(cal_factor, abs = 1.5 * flux_fac_WP2.fluxcal_err)

    #check the flux values are similar regardless of the wollaston used
    assert flux_fac_WP1.fluxcal_fac == pytest.approx(flux_fac_WP2.fluxcal_fac, rel=0.05)

    # Generate Excel documentation for the polarimetric flux calibration factor products
    excel_output_path_wp1 = os.path.join(wp1_dir, "abf_cal_documentation.xlsx")
    generate_fits_excel_documentation(fluxcal_file_WP1, excel_output_path_wp1)
    print(f"Excel documentation generated for WP1: {excel_output_path_wp1}")
    
    excel_output_path_wp2 = os.path.join(wp2_dir, "abf_cal_documentation.xlsx")
    generate_fits_excel_documentation(fluxcal_file_WP2, excel_output_path_wp2)
    print(f"Excel documentation generated for WP2: {excel_output_path_wp2}")

    # Print success message
    print('e2e test for polarimetric flux calibration factor passed')



if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l2b-> PolFluxcalFactor end-to-end test")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_e2e(outputdir)