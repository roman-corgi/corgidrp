# E2E Test Code for CoreThroughput Calibration

import argparse
import os, shutil
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
import corgidrp.l2b_to_l3 as l2b_to_l3
from corgidrp import caldb
from corgidrp.check import fix_hdrs_for_tvac

# this file's folder
thisfile_dir = os.path.dirname(__file__)

@pytest.mark.e2e
def test_expected_results_e2e(e2edata_path, e2eoutput_path):
    """Test corethroughput calibration with mock data

    Args:
        e2edata_path (str): Path to the test data
        e2eoutput_path (str): Path to the output directory

    """
    # make output directory if needed
    corethroughput_outputdir = os.path.join(e2eoutput_path, "corethroughput_cal_e2e/mock_data")
    if os.path.exists(corethroughput_outputdir):
        shutil.rmtree(corethroughput_outputdir)
    os.makedirs(corethroughput_outputdir)
    
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
    for i, image in enumerate(corethroughput_image_list):
        image.pri_hdr['VISTYPE'] = 'CGIVST_CAL_CORETHRPT'
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

    # Create input data directory
    input_l2b_dir = os.path.join(corethroughput_outputdir, 'input_l2b')
    os.makedirs(input_l2b_dir)
    
    # Save with proper CGI naming convention
    corethroughput_data_filepath = mocks.rename_files_to_cgi_format(
        list_of_fits=corethroughput_image_list, 
        output_dir=input_l2b_dir, 
        level_suffix="l2b"
    )
    corethroughput_data_filepath = fix_hdrs_for_tvac(
        corethroughput_data_filepath,
        input_l2b_dir,
        header_template=mocks.create_default_L2b_headers,
    )
    
    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)

    # Run the DRP walker
    print('Running walker')
    walker.walk_corgidrp(corethroughput_data_filepath, '', corethroughput_outputdir)
    
    # Load in the output data. It should be the latest ctp_cal file produced.
    corethroughput_drp_file = glob.glob(os.path.join(corethroughput_outputdir,
        '*ctp_cal.fits'))[0]
    ct_cal_drp = data.CoreThroughputCalibration(corethroughput_drp_file)

    # CT cal file from mock data directly
    # Divide by exposure time
    ct_cal_mock = corethroughput.generate_ct_cal(corethroughput_dataset)

    # Check DRP CT cal file and mock one coincide
    # Remember that DRP divides by exposure time to go from L2b to L3 and
    # generate_ct_cal() does not, so we need to divide by EXPTIME the off-axis PSFs
    # Use allclose for floating point comparison to account for bit depth differences
    assert np.allclose(ct_cal_drp.data, ct_cal_mock.data/exp_time_s, rtol=1e-6, atol=1e-8)
    assert np.allclose(ct_cal_drp.ct_excam, ct_cal_mock.ct_excam, rtol=1e-6, atol=1e-8)
    assert np.allclose(ct_cal_drp.err, ct_cal_mock.err, rtol=1e-6, atol=1e-8)
    assert np.all(ct_cal_drp.dq == ct_cal_mock.dq)
    assert np.all(ct_cal_drp.ct_fpam == ct_cal_mock.ct_fpam)
    assert np.all(ct_cal_drp.ct_fsam == ct_cal_mock.ct_fsam)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

    # Print success message
    print('e2e test for corethroughput calibration with mock data passed')
    
@pytest.mark.e2e
def test_expected_results_spc_band3_simdata_e2e(e2edata_path, e2eoutput_path):
    """Test corethroughput calibration with simulated band 3 shaped pupil data

    Args:
        e2edata_path (str): Path to the test data
        e2eoutput_path (str): Path to the output directory

    """
    
    # Create the output directory
    corethroughput_outputdir = os.path.join(e2eoutput_path, 'corethroughput_cal_e2e/band3_spc_data')
    if os.path.exists(corethroughput_outputdir):
        shutil.rmtree(corethroughput_outputdir)
    os.makedirs(corethroughput_outputdir)
    
    # Read the files in the directory
    input_dir = os.path.join(e2edata_path, "ct_band3_shapedpupil")
    files = os.listdir(input_dir)
    files.sort()
    datafiles = glob.glob(os.path.join(input_dir, "*.fits"))
    # datafiles = datafiles[::5] # only use every 5 files to make it smaller

    # Create the input data in the right format
    images = []
    for file in datafiles:
        image = fits.open(file)
        new_image = data.Image(image[0].data, pri_hdr=image[0].header, ext_hdr=image[1].header)
        new_image.pri_hdr['VISTYPE'] = 'CGIVST_CAL_CORETHRPT'
        new_image.ext_hdr['DATALVL'] = "L2b"
        new_image.ext_hdr['BUNIT'] = "photoelectron"
        new_image.ext_hdr['EXPTIME'] = 1.0
        ftimeutc = data.format_ftimeutc(new_image.ext_hdr['FTIMEUTC'])
        new_image.filename = f'cgi_{new_image.pri_hdr["VISITID"]}_{ftimeutc}_l2b.fits'
        images.append(new_image)
    filenames = [img.filename for img in images]
    def get_filename(img):
        return img.filename
    images.sort(key=get_filename)

    dataset = data.Dataset(images)

    # save the input data
    l2b_data_dir = os.path.join(corethroughput_outputdir, 'input_l2b')
    os.makedirs(l2b_data_dir)
    dataset.save(filedir=l2b_data_dir)
    l2b_filenames = glob.glob(os.path.join(l2b_data_dir, '*.fits'))
    l2b_filenames.sort()
    l2b_filenames = fix_hdrs_for_tvac(
        l2b_filenames,
        l2b_data_dir,
        header_template=mocks.create_default_L2b_headers,
    )

    walker.walk_corgidrp(l2b_filenames, '', corethroughput_outputdir)
    
    # Load in the output data. It should be the latest CTP_CAL file produced.
    corethroughput_drp_file = glob.glob(os.path.join(corethroughput_outputdir,
        '*ctp_cal.fits'))[0]
    ct_cal_drp = data.CoreThroughputCalibration(corethroughput_drp_file)
    
    # run the recipe directly to check out it comes
    dataset_l2b = data.Dataset(l2b_filenames)
    dataset_normed = l2b_to_l3.divide_by_exptime(dataset_l2b)
    ct_cal_sim = corethroughput.generate_ct_cal(dataset_normed)

    # Asserts
    # Use allclose for floating point comparison to account for bit depth differences
    assert np.allclose(ct_cal_drp.data, ct_cal_sim.data, rtol=1e-6, atol=1e-8)
    assert np.allclose(ct_cal_drp.ct_excam, ct_cal_sim.ct_excam, rtol=1e-6, atol=1e-8)
    assert np.allclose(ct_cal_drp.err, ct_cal_sim.err, rtol=1e-6, atol=1e-8)
    assert np.all(ct_cal_drp.dq == ct_cal_sim.dq)
    assert np.all(ct_cal_drp.ct_fpam == ct_cal_sim.ct_fpam)
    assert np.all(ct_cal_drp.ct_fsam == ct_cal_sim.ct_fsam)

    assert ct_cal_drp.data is not None, "CoreThroughput calibration data is None"
    assert len(ct_cal_drp.data) == len(dataset) - 1, "CoreThroughput calibration data length does not match dataset length"
    assert ct_cal_drp.data.shape[0] == ct_cal_drp.ct_excam.shape[1], "CoreThroughput calibration excam length does not match data length"
    assert ct_cal_drp.err is not None, "CoreThroughput error data is None"
    assert ct_cal_drp.dq is not None, "CoreThroughput dq data is None"
    
    assert np.min(ct_cal_drp.ct_excam[2]) > 0, "CoreThroughput measurements have non-positive values"
    assert np.max(ct_cal_drp.ct_excam[2]) <= 1, "CoreThroughput measurements exceed 1"

    # Print success message
    print('e2e test for corethroughput calibration with simulated band 3 shaped pupil data passed')
    
if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    outputdir = thisfile_dir
    e2edata_path = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'

    ap = argparse.ArgumentParser(description='run the l2b-> CoreThroughput end-to-end test')
    ap.add_argument('-e2e', '--e2edata_dir', default=e2edata_path,
                    help='Path to CGI_TVAC_Data Folder [%(default)s]')
    ap.add_argument('-o', '--outputdir', default=outputdir,
                    help='directory to write results to [%(default)s]')
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_e2e(args.e2edata_dir, args.outputdir)
    test_expected_results_spc_band3_simdata_e2e(args.e2edata_dir, args.outputdir)
