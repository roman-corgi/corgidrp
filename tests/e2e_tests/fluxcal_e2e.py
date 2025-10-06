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

@pytest.mark.e2e
def test_expected_results_e2e(e2eoutput_path):
    #mock a point source image
    fwhm = 3
    star_flux = 1.5e-09 #erg/(s*cm^2*AA)
    cal_factor = star_flux/200
    flux_image = mocks.create_flux_image(star_flux, fwhm, cal_factor)
    flux_dataset = data.Dataset([flux_image])
    output_dir = os.path.join(e2eoutput_path, 'flux_cal_e2e')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
 
    # Create input_data subfolder
    input_data_dir = os.path.join(output_dir, 'input_l2b')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    # Generate proper filename with visitid and current time
    current_time = datetime.now().strftime('%Y%m%dt%H%M%S%f')[:-5]
    # Extract visit ID from primary header VISITID keyword
    visitid = flux_image.pri_hdr.get('VISITID', None)
    if visitid is not None:
        # Convert to string and pad to 19 digits
        visitid = str(visitid).zfill(19)
    else:
        # Fallback: use default visitid
        visitid = "0000000000000000000"

    # Create proper L2b filename: cgi_{visitid}_{current_time}_l2b.fits
    flux_dataset.save(input_data_dir, [f'cgi_{visitid}_{current_time}_l2b.fits'])
    flux_data_filelist = []
    for f in os.listdir(input_data_dir):
        flux_data_filelist.append(os.path.join(input_data_dir, f))
    #print(flux_data_filelist)


    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)

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
    
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

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
