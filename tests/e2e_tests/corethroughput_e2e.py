# E2E Test Code for CoreThroughput Calibration

import argparse
import os, shutil
import glob
import pytest
import numpy as np
import astropy.time as time

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.detector as detector
import corgidrp.corethroughput as corethroughput
from corgidrp import caldb

# this file's folder
thisfile_dir = os.path.dirname(__file__)

@pytest.mark.e2e
def test_expected_results_e2e(e2edata_path, e2eoutput_path):
    # Mock a CT dataset (CT PAM, pupil image and off-axis PSFs)
    # Some arbitrary positive value
    exp_time_s = np.pi
    # Choose some H/V values for FPAM/FSAM  during corethroughput observations
    FPAM_H_CT, FPAM_V_CT, FSAM_H_CT, FSAM_V_CT = 6854, 22524, 29471, 12120
    # Choose a band
    cfam_name = '1F'
    prhd, exthd, errhdr, dqhdr, biashdr = mocks.create_default_L2b_headers()
    # Mock error
    err = np.ones([1024,1024])
    # Add pupil image(s) of the unocculted source's observation to test that
    # the corethroughput calibration function can handle more than one pupil image
    pupil_image = np.zeros([1024, 1024])
    # Set it to some known value for some known pixels
    pupil_image[510:530, 510:530]=1
    # Add specific values for pupil images:
    # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
    exthd_pupil = exthd.copy()
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'

    # Add pupil images
    corethroughput_image_list = [data.Image(pupil_image,pri_hdr=prhd,
        ext_hdr=exthd_pupil, err=err)]
    corethroughput_image_list += mocks.create_ct_psfs(50, e2e=True)[0]
    # Make sure all dataframes share the same common header values
    for image in corethroughput_image_list:
        image.pri_hdr['VISTYPE'] = 'CORETPUT'
        image.ext_hdr['EXPTIME'] = exp_time_s
        # DRP
        image.ext_hdr['DRPCTIME'] = time.Time.now().isot
        image.ext_hdr['DRPVERSN'] = corgidrp.__version__
        # cfam filter
        image.ext_hdr['CFAMNAME'] = cfam_name
        # FPAM/FSAM during CT observing sequence
        image.ext_hdr['FPAM_H'] = FPAM_H_CT
        image.ext_hdr['FPAM_V'] = FPAM_V_CT
        image.ext_hdr['FSAM_H'] = FSAM_H_CT
        image.ext_hdr['FSAM_V'] = FSAM_V_CT
    corethroughput_dataset = data.Dataset(corethroughput_image_list)

    output_dir = os.path.join(e2eoutput_path, 'corethroughput_test_data')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
    # List of filenames
    corethroughput_data_filelist = ['corethroughput_e2e_{0}_L2b.fits'.format(i) for i in range(len(corethroughput_dataset))]
    corethroughput_dataset.save(output_dir, corethroughput_data_filelist)

    # make DRP output directory if needed
    corethroughput_outputdir = os.path.join(e2eoutput_path, 'l2b_to_corethroughput_output')
    if os.path.exists(corethroughput_outputdir):
        shutil.rmtree(corethroughput_outputdir)
    os.mkdir(corethroughput_outputdir)

    # Run the DRP walker
    print('Running walker')
    # Add path to files
    corethroughput_data_filepath = [os.path.join(output_dir, f) for f in corethroughput_data_filelist]
    walker.walk_corgidrp(corethroughput_data_filepath, '', corethroughput_outputdir)
    
    # Load in the output data. It should be the latest CTP_CAL file produced.
    corethroughput_drp_file = glob.glob(os.path.join(corethroughput_outputdir,
        '*CTP_CAL*.fits'))[0]
    ct_cal_drp = data.CoreThroughputCalibration(corethroughput_drp_file)

    # CT cal file from mock data directly
    # Divide by exposure time
    ct_cal_mock = corethroughput.generate_ct_cal(corethroughput_dataset)

    # Check DRP CT cal file and mock one coincide
    # Remember that DRP divides by exposure time to go from L2b to L3 and
    # generate_ct_cal() does not, so we need to divide by EXPTIME the off-axis PSFs
    assert ct_cal_drp.data == pytest.approx(ct_cal_mock.data/exp_time_s, abs=1e-12)
    assert ct_cal_drp.ct_excam == pytest.approx(ct_cal_mock.ct_excam, abs=1e-12)
    assert np.all(ct_cal_drp.err == ct_cal_mock.err)
    assert np.all(ct_cal_drp.dq == ct_cal_mock.dq)
    assert np.all(ct_cal_drp.ct_fpam == ct_cal_mock.ct_fpam)
    assert np.all(ct_cal_drp.ct_fsam == ct_cal_mock.ct_fsam)

    # Remove entry from caldb
    this_caldb = caldb.CalDB()
    this_caldb.remove_entry(ct_cal_drp)

    # Print success message
    print('e2e test for corethroughput calibration passed')
    
if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    outputdir = thisfile_dir
    e2edata_path =  '.'

    ap = argparse.ArgumentParser(description='run the l2b-> CoreThroughput end-to-end test')
    ap.add_argument('-e2e', '--e2edata_dir', default=e2edata_path,
                    help='Path to CGI_TVAC_Data Folder [%(default)s]')
    ap.add_argument('-o', '--outputdir', default=outputdir,
                    help='directory to write results to [%(default)s]')
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_e2e(args.e2edata_dir, args.outputdir)
