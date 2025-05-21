# e2e test code for corethroughput map

import os, shutil
import argparse
import glob
import pytest
import numpy as np
import astropy.time as time
from astropy.io import fits

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
    # Choose some arbitrary H/V values for FPAM/FSAM  during corethroughput observations
    FPAM_H_CT, FPAM_V_CT, FSAM_H_CT, FSAM_V_CT = 6854, 22524, 29471, 12120
    # Choose a band
    cfam_name = '1F'
    prhd, exthd = mocks.create_default_L2b_headers()
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
    corethroughput_image_list += mocks.create_ct_psfs(50, e2e=True, n_psfs=100)[0]
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
        # Some location for the center of the mask
        image.ext_hdr['STARLOCX'] = 509
        image.ext_hdr['STARLOCY'] = 513

    # Create CT dataset
    corethroughput_dataset = data.Dataset(corethroughput_image_list)

    # Create a mock coronagrahoc dataset with a different FPM's center than the
    # CT dataset
    corDataset_image_list = mocks.create_ct_psfs(50)[0]
    # Make sure all dataframes share the same common header values
    for image in corDataset_image_list:
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
        # Some location for the center of the mask
        image.ext_hdr['STARLOCX'] = 522
        image.ext_hdr['STARLOCY'] = 517
    # Create coronagraphic dataset
    corDataset = data.Dataset(corDataset_image_list)

    # Define temporary directory to store the individual frames
    output_dir = os.path.join(e2eoutput_path, 'ctmap_test_data')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
    # List of filenames
    corDataset_filelist = ['ctmap_e2e_{0}_L2b.fits'.format(i)
        for i in range(len(corDataset))]
    # Save them
    corDataset.save(output_dir, corDataset_filelist)

    # Make directory for the CT cal file
    ctmap_outputdir = os.path.join(e2eoutput_path, 'l2a_to_ct_map')
    if os.path.exists(ctmap_outputdir):
        shutil.rmtree(ctmap_outputdir)
    os.mkdir(ctmap_outputdir)

    # Create CT cal file from the mock data directly
    ct_cal_mock = corethroughput.generate_ct_cal(corethroughput_dataset)
    # Save it
    ct_cal_mock.filedir = ctmap_outputdir
    ct_cal_mock.save()
    # Add it to caldb
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(ct_cal_mock)

    # Create the CT map. Do not save it. We will compare it with the map from
    # the walker
    # FPAM/FSAM
    fpam_fsam_cal = data.FpamFsamCal(os.path.join(corgidrp.default_cal_dir,
        'FpamFsamCal_2024-02-10T00:00:00.000.fits'))
    # The first entry (dataset) is only used to get the FPM's center
    ct_map_mock = corethroughput.create_ct_map(corDataset, fpam_fsam_cal,
        ct_cal_mock)

    # Run the DRP walker
    print('Running walker')
    # Add path to files
    corDataset_filepath = [os.path.join(output_dir, f) for f in corDataset_filelist]
    walker.walk_corgidrp(corDataset_filepath, '', ctmap_outputdir,
        template='l2a_to_corethroughput_map.json')
    
    # Read CT map produced by the walker
    ct_map_filepath = os.path.join(ctmap_outputdir, ct_map_mock.filename)
    ct_map_walker = fits.open(ct_map_filepath)

    # Check whether direct ct map and the one from the walker are the same
    # CT map values: (x, y, CT) for each location
    assert np.all(ct_map_walker[1].data == ct_map_mock.data)
    # ERR
    assert np.all(ct_map_walker[2].data == ct_map_mock.err)
    # DQ
    assert np.all(ct_map_walker[3].data == ct_map_mock.dq)

    # Clean test data
    # Remove entry from caldb
    corethroughput_drp_file = glob.glob(os.path.join(ctmap_outputdir,
        '*CTP_CAL*.fits'))[0]
    ct_cal_drp = data.CoreThroughputCalibration(corethroughput_drp_file)
    this_caldb = caldb.CalDB()
    this_caldb.remove_entry(ct_cal_drp)

    # Delete mock data files
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # Delete e2e CT cal file
    if os.path.exists(ctmap_outputdir):
        shutil.rmtree(ctmap_outputdir)

    # Delete CT map file
    if os.path.exists(ct_map_filepath):
        os.remove(ct_map_filepath)

    # Print success message
    print('e2e test for corethroughput map passed')
    
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
