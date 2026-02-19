# Flux calibration E2E Test Code

import argparse
import os, shutil
import glob
import pytest
import numpy as np
from datetime import datetime

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.detector as detector
import corgidrp.fluxcal as fluxcal
from corgidrp import caldb
from corgidrp import check

@pytest.mark.e2e
def test_expected_results_e2e(e2eoutput_path):
    # Test Band 3
    cfam_name = '3F'
    #mock a point source image
    fwhm = 3
    star_flux = 1.5e-09 #erg/(s*cm^2*AA)
    cal_factor = star_flux/200
    flux_image = mocks.create_flux_image(star_flux, fwhm, cal_factor)
    flux_image.ext_hdr['BUNIT'] = 'photoelectron'
    # Changes to test Band 3F
    # There's nothing specific to band 3 in the primary header
    # In the extension header, we can set
    flux_image.ext_hdr['CFAMNAME']= cfam_name
    # These settings are related to imaging, which is what is tested now. They
    # do not need to be changed. No other settings in the extension header are
    # specific to band 3
    #FSAMNAME= 'R1C1'
    #LSAMNAME= 'NFOV'
    #FPAMNAME= 'HOLE'
    #SPAMNAME= 'OPEN
    output_dir = os.path.join(e2eoutput_path, 'flux_cal_band3_e2e')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Create input_data subfolder
    input_data_dir = os.path.join(output_dir, 'input_l2b')
    os.makedirs(input_data_dir, exist_ok=True)
    
    flux_image.save(input_data_dir)
    flux_data_filelist = [flux_image.filepath]

    # One last check
    if flux_image.ext_hdr['CFAMNAME'] != cfam_name:
        raise ValueError(f'CFAMNAME should be {cfam_name:s}')

    ####### Run the DRP walker
    print('Running walker')
    walker.walk_corgidrp(flux_data_filelist, '', output_dir)
    
    ####### Load in the output data. It should be the latest kgain file produced.
    fluxcal_file = glob.glob(os.path.join(output_dir, '*abf_cal*.fits'))[0]
    flux_fac = data.FluxcalFactor(fluxcal_file)
    print("used color filter", flux_fac.filter)
    print("used ND filter", flux_fac.nd_filter)
    print("fluxcal factor", flux_fac.fluxcal_fac)
    print("fluxcal factor error", flux_fac.fluxcal_err)
    assert flux_fac.fluxcal_fac == pytest.approx(cal_factor, abs = 1.5 * flux_fac.fluxcal_err)

    check.compare_to_mocks_hdrs(fluxcal_file, mocks.create_default_L2b_headers)

   # Print success message
    print('e2e test for flux calibration factor passed')
    
if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l2b-> FluxcalFactor end-to-end test")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_e2e(outputdir)
