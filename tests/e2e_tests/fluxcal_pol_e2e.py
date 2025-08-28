import argparse
import os, shutil
import glob
import pytest
import numpy as np

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.fluxcal as fluxcal

@pytest.mark.e2e
def test_expected_results_e2e(e2edata_path, e2eoutput_path):
    # Load in L1 polarimetric data
    WP1_input_path = os.path.join('..', 'test_data', 'polarimetry', 'simulated_polarimetric_L1_HLC_WP1.fits')
    WP2_input_path = os.path.join('..', 'test_data', 'polarimetry', 'simulated_polarimetric_L1_HLC_WP2.fits')
    L1_image_WP1 = data.Image(WP1_input_path)
    L1_image_WP2 = data.Image(WP2_input_path)

    # update image size to be 1024 x 1024
    data_WP1 = np.zeros(shape=(1024,1024))
    data_WP2 = np.zeros(shape=(1024,1024))
    data_WP1 += L1_image_WP1.data[13:1037, 1088:2112]
    data_WP2 += L1_image_WP2.data[13:1037, 1088:2112]

    #create L2b headers
    prihdr, exthdr, errhdr, dqhdr, biashdr = mocks.create_default_L2b_headers()
    prihdr['VISTYPE'] = 'ABSFLXBT'
    prihdr['TARGET'] = 'Vega'
    exthdr['EXPTIME'] = 1000
    exthdr['CFAMNAME'] = '1F'
    exthdr['DPAMNAME'] = 'POL0'
    exthdr['FPAMNAME'] = 'ND475'
    exthdr['BUNIT'] = 'photoelectron'

    #create L2b images using L1 data
    L2b_image_WP1 = data.Image(data_WP1, pri_hdr=prihdr, ext_hdr=exthdr)
    L2b_image_WP2 = data.Image(data_WP2, pri_hdr=prihdr, ext_hdr=exthdr)
    flux_dataset_WP1 = data.Dataset([L2b_image_WP1])
    flux_dataset_WP2 = data.Dataset([L2b_image_WP2])

    output_dir = os.path.join(e2eoutput_path, 'pol_flux_sim_test_data')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    output_dir_WP1 = os.path.join(output_dir, 'WP1')
    output_dir_WP2 = os.path.join(output_dir, 'WP2')
    os.mkdir(output_dir_WP1)
    os.mkdir(output_dir_WP2)
    flux_dataset_WP1.save(output_dir_WP1, ['flux_e2e_WP1_{0}.fits'.format(i) for i in range(len(flux_dataset_WP1))])
    flux_dataset_WP2.save(output_dir_WP2, ['flux_e2e_WP2_{0}.fits'.format(i) for i in range(len(flux_dataset_WP2))])

    data_filelist_WP1 = []
    data_filelist_WP2 = []

    for f in os.listdir(output_dir_WP1):
        data_filelist_WP1.append(os.path.join(output_dir_WP1, f))
    
    for f in os.listdir(output_dir_WP2):
        data_filelist_WP2.append(os.path.join(output_dir_WP2, f))
    
    # make DRP output directory if needed
    fluxcal_outputdir = os.path.join(e2eoutput_path, "l2b_to_pol_fluxcal_factor_output")
    if os.path.exists(fluxcal_outputdir):
        shutil.rmtree(fluxcal_outputdir)
    os.mkdir(fluxcal_outputdir)

    fluxcal_outputdir_WP1 = os.path.join(fluxcal_outputdir, 'WP1')
    fluxcal_outputdir_WP2 = os.path.join(fluxcal_outputdir, 'WP2')
    os.mkdir(fluxcal_outputdir_WP1)
    os.mkdir(fluxcal_outputdir_WP2)

    ####### Run the DRP walker for WP1
    print('Running walker for WP1')
    walker.walk_corgidrp(data_filelist_WP1, '', fluxcal_outputdir_WP1)
    fluxcal_file_WP1 = glob.glob(os.path.join(fluxcal_outputdir_WP1, '*abf_cal*.fits'))[0]
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


     ####### Run the DRP walker for WP2
    print('Running walker for WP2')
    walker.walk_corgidrp(data_filelist_WP2, '', fluxcal_outputdir_WP2)
    fluxcal_file_WP2 = glob.glob(os.path.join(fluxcal_outputdir_WP2, '*abf_cal*.fits'))[0]
    fluxcal_image_WP2 = data.Image(fluxcal_file_WP2)

    #check that the calibration file is configured correctly
    assert fluxcal_image_WP2.pri_hdr['NAXIS'] == 0
    assert fluxcal_image_WP2.ext_hdr['DPAMNAME'] == 'POL0'
    assert fluxcal_image_WP2.data.shape == (1,)
    assert fluxcal_image_WP2.err.shape == (1,)
    assert fluxcal_image_WP2.dq.shape == (1,)

    #output values
    flux_fac_WP2 = data.FluxcalFactor(fluxcal_file_WP2)
    print("used color filter", flux_fac_WP2.filter)
    print("used ND filter", flux_fac_WP2.nd_filter)
    print("fluxcal factor", flux_fac_WP2.fluxcal_fac)
    print("fluxcal factor error", flux_fac_WP2.fluxcal_err)




if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir
    e2edata_dir =  "/home/ericshen/corgi/TVAC_data/"

    ap = argparse.ArgumentParser(description="run the l2b-> FluxcalFactor end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_e2e(e2edata_dir, outputdir)