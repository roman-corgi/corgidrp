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
# If a run has crashed before being able to remove the test CT cal file,
# remove it before importing caldb, which scans for new entries in the cal folder
ct_cal_files = os.path.join(corgidrp.default_cal_dir, '*_CTP_CAL.fits')
for ct_filepath_object in glob.glob(ct_cal_files):
    os.remove(ct_filepath_object)
# Updates in main that change this file, do not update it, and may crash the run
drp_cfg_file = os.path.join(corgidrp.default_cal_dir, '..', 'corgidrp_caldb.csv')
if os.path.exists(drp_cfg_file):
    os.remove(drp_cfg_file)
from corgidrp import caldb

@pytest.mark.e2e
def test_expected_results_e2e(tvacdata_path, e2eoutput_path):
    # Mock a CT dataset (CT PAM, pupil image and off-axis PSFs)
    # Choose some H/V values for FPAM/FSAM  during corethroughput observations
    FPAM_H_CT, FPAM_V_CT, FSAM_H_CT, FSAM_V_CT = 6854, 22524, 29471, 12120
    # Choose a band
    cfam_name = '1F'
    prhd, exthd = mocks.create_default_L2b_headers()
    prhd['VISTYPE'] = 'CORETPUT'
    # Some arbitrary positive value
    exthd['EXPTIME'] = np.pi
    # DRP
    exthd['DRPCTIME'] = time.Time.now().isot
    exthd['DRPVERSN'] = corgidrp.__version__
    # cfam filter
    exthd['CFAMNAME'] = cfam_name
    # FPAM/FSAM during CT observing sequence
    exthd['FPAM_H'] = FPAM_H_CT
    exthd['FPAM_V'] = FPAM_V_CT
    exthd['FSAM_H'] = FSAM_H_CT
    exthd['FSAM_V'] = FSAM_V_CT
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
    corethroughput_dataset = data.Dataset(corethroughput_image_list)

    output_dir = os.path.join(e2eoutput_path, 'corethroughput_test_data')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
    corethroughput_dataset.save(output_dir, ['corethroughput_e2e_{0}_L2b.fits'.format(i) for i in range(len(corethroughput_dataset))])
    corethroughput_data_filelist = []
    for f in os.listdir(output_dir):
        corethroughput_data_filelist.append(os.path.join(output_dir, f))
    print(corethroughput_data_filelist)

    # make DRP output directory if needed
    corethroughput_outputdir = os.path.join(e2eoutput_path, 'l2b_to_corethroughput_output')
    if os.path.exists(corethroughput_outputdir):
        shutil.rmtree(corethroughput_outputdir)
    os.mkdir(corethroughput_outputdir)

    # Run the DRP walker
    print('Running walker')
    walker.walk_corgidrp(corethroughput_data_filelist, '', corethroughput_outputdir)
    
    # Load in the output data. It should be the latest CTP_CAL file produced.
    corethroughput_drp_file = glob.glob(os.path.join(corethroughput_outputdir,
        '*CTP_CAL*.fits'))[0]
    ct_cal_drp = data.CoreThroughputCalibration(corethroughput_drp_file)

    # CT cal file from mock data directly
    # Divide by exposure time
    for 
    ct_cal_mock = corethroughput.generate_ct_cal(corethroughput_dataset)

    # Check DRP CT cal file and mock one coincide
    assert np.all(ct_cal_drp.data == ct_cal_mock.data)
    assert np.all(ct_cal_drp.err == ct_cal_mock.err)
    assert np.all(ct_cal_drp.dq == ct_cal_mock.dq)
    assert np.all(ct_cal_drp.ct_excam == ct_cal_mock.ct_excam)
    assert np.all(ct_cal_drp.ct_fpam == ct_cal_mock.ct_fpam)
    assert np.all(ct_cal_drp.ct_fsam == ct_cal_mock.ct_fsam)

    # Remove entry from caldb
    this_caldb = caldb.CalDB()
    this_caldb.remove_entry(ct_cal)

    # Print success message
    print('e2e test for corethroughput calibration passed')
    
if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir
    tvacdata_dir =  '.'

    ap = argparse.ArgumentParser(description="run the l2b-> CoreThroughput end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_e2e(tvacdata_dir, outputdir)
