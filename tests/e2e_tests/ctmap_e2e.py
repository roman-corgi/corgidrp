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
import corgidrp.caldb as caldb
import corgidrp.check as check

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

    # Create a mock coronagrahic dataset with a different FPM's center than the
    # CT dataset
    corDataset_image_list = mocks.create_ct_psfs(50, e2e=True)[0]
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

    # Make directory for the CT cal file
    ctmap_outputdir = os.path.join(e2eoutput_path, 'ctmap_cal_e2e')
    if os.path.exists(ctmap_outputdir):
        shutil.rmtree(ctmap_outputdir)
    os.mkdir(ctmap_outputdir)
    
    # Define directory to store the individual frames under the output directory
    output_dir = os.path.join(ctmap_outputdir, 'input_l2b')
    os.mkdir(output_dir)

    calibrations_dir = os.path.join(ctmap_outputdir, 'calibrations')
    os.mkdir(calibrations_dir)
    
    renamed_files = mocks.rename_files_to_cgi_format(list_of_fits=list(corDataset), output_dir=output_dir, level_suffix="l2b")
    
    # Update the dataset with the new filenames
    corDataset_filelist = [os.path.basename(f) for f in renamed_files]
    
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB

    # Create CT cal file from the mock data directly
    ct_cal_mock = corethroughput.generate_ct_cal(corethroughput_dataset)
    # Save it
    ct_cal_mock.filedir = calibrations_dir
    ct_cal_mock.save()
    # Add it to caldb
    this_caldb.create_entry(ct_cal_mock)

    # Create the CT map. Do not save it. We will compare it with the map from
    # the walker
    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    fpam_fsam_cal = this_caldb.get_calib(None, data.FpamFsamCal)
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
    # Use allclose for floating point comparison to account for bit depth differences
    assert np.allclose(ct_map_walker[1].data, ct_map_mock.data, rtol=1e-5, atol=1e-5, equal_nan=True)
    # ERR
    assert np.allclose(ct_map_walker[2].data, ct_map_mock.err, rtol=1e-5, atol=1e-5, equal_nan=True)
    # DQ (integer comparison, but cast to same dtype for consistency)
    assert np.all(ct_map_walker[3].data.astype(ct_map_mock.dq.dtype) == ct_map_mock.dq)

    check.compare_to_mocks_hdrs(ct_map_filepath)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

    # Print success message
    print('e2e test for corethroughput map passed')
    
if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    outputdir = thisfile_dir
    e2edata_path = '/Users/kevinludwick/Documents/DRP_E2E_Test_Files_v2/E2E_Test_Data'# '/Users/jmilton/Documents/CGI/E2E_Test_Data2'

    ap = argparse.ArgumentParser(description='run the l2b-> CoreThroughput end-to-end test')
    ap.add_argument('-e2e', '--e2edata_dir', default=e2edata_path,
                    help='Path to CGI_TVAC_Data Folder [%(default)s]')
    ap.add_argument('-o', '--outputdir', default=outputdir,
                    help='directory to write results to [%(default)s]')
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_e2e(args.e2edata_dir, args.outputdir)
