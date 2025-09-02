# Flux calibration E2E Test Code

import argparse
import os, shutil
import glob
import pytest
import numpy as np

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.detector as detector
import corgidrp.fluxcal as fluxcal
from corgidrp import caldb

@pytest.mark.e2e
def test_expected_results_e2e(e2edata_path, e2eoutput_path):
    #mock a point source image
    fwhm = 3
    star_flux = 1.5e-09 #erg/(s*cm^2*AA)
    cal_factor = star_flux/200
    flux_image = mocks.create_flux_image(star_flux, fwhm, cal_factor)
    flux_image.ext_hdr['BUNIT'] = 'photoelectron'
    flux_dataset = data.Dataset([flux_image])
    output_dir = os.path.join(e2eoutput_path, 'flux_sim_test_data')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
 
    flux_dataset.save(output_dir, ['flux_e2e_{0}.fits'.format(i) for i in range(len(flux_dataset))])
    flux_data_filelist = []
    for f in os.listdir(output_dir):
        flux_data_filelist.append(os.path.join(output_dir, f))
    print(flux_data_filelist)

    # make DRP output directory if needed
    fluxcal_outputdir = os.path.join(e2eoutput_path, "l2b_to_fluxcal_factor_output")
    if os.path.exists(fluxcal_outputdir):
        shutil.rmtree(fluxcal_outputdir)
    os.mkdir(fluxcal_outputdir)

    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)

    ####### Run the DRP walker
    print('Running walker')
    walker.walk_corgidrp(flux_data_filelist, '', fluxcal_outputdir)
    
        ####### Load in the output data. It should be the latest kgain file produced.
    fluxcal_file = glob.glob(os.path.join(fluxcal_outputdir, '*abf_cal*.fits'))[0]
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
    e2edata_dir =  "/home/schreiber/DataCopy/fluxcal_mock_data/"

    ap = argparse.ArgumentParser(description="run the l2b-> FluxcalFactor end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_e2e(e2edata_dir, outputdir)
