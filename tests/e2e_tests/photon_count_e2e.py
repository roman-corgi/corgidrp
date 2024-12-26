# Photon Counting E2E Test Code

import argparse
import os
import pytest
import numpy as np
import astropy.time as time
from astropy.io import fits

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector

@pytest.mark.e2e
def test_expected_results_e2e(file_dir):
    np.random.seed(1234)
    ill_dataset, dark_dataset, ill_mean, dark_mean = mocks.create_photon_countable_frames(Nbrights=160, Ndarks=161, cosmic_rate=1)
    output_dir = os.path.join(file_dir, 'pc_sim_test_data')
    output_ill_dir = os.path.join(output_dir, 'ill_frames')
    output_dark_dir = os.path.join(output_dir, 'dark_frames')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # empty out directory of any previous files
    for f in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir,f)):
            continue
        os.remove(os.path.join(output_dir,f))
    if not os.path.exists(output_ill_dir):
        os.mkdir(output_ill_dir)
    if not os.path.exists(output_dark_dir):
        os.mkdir(output_dark_dir)
    # empty out directory of any previous files
    for f in os.listdir(output_ill_dir):
        os.remove(os.path.join(output_ill_dir,f))
    for f in os.listdir(output_dark_dir):
        os.remove(os.path.join(output_dark_dir,f))
    ill_dataset.save(output_ill_dir, ['pc_frame_ill_{0}.fits'.format(i) for i in range(len(ill_dataset))])
    dark_dataset.save(output_dark_dir, ['pc_frame_dark_{0}.fits'.format(i) for i in range(len(dark_dataset))])
    l1_data_ill_filelist = []
    l1_data_dark_filelist = []
    for f in os.listdir(output_ill_dir):
        l1_data_ill_filelist.append(os.path.join(output_ill_dir, f))
    for f in os.listdir(output_dark_dir):
        l1_data_dark_filelist.append(os.path.join(output_dark_dir, f))

    this_caldb = caldb.CalDB() # connection to cal DB
    # remove other KGain calibrations that may exist in case they don't have the added header keywords
    for i in range(len(this_caldb._db['Type'])):
        if this_caldb._db['Type'][i] == 'KGain':
            this_caldb._db = this_caldb._db.drop(i)
        elif this_caldb._db['Type'][i] == 'Dark':
            this_caldb._db = this_caldb._db.drop(i)
    this_caldb.save()

    # KGain
    kgain_val = 7 # default value used in mocks.create_photon_countable_frames()
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(l1_data_ill_filelist)
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    # add in keywords that didn't make it into mock_kgain.fits, using values used in mocks.create_photon_countable_frames()
    kgain.ext_hdr['RN'] = 100
    kgain.ext_hdr['RN_ERR'] = 0
    kgain.save(filedir=output_dir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)

    # NoiseMap
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electrons'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=output_dir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    corgidrp_dir = os.path.split(corgidrp.__path__[0])[0]
    tests_dir = os.path.join(corgidrp_dir, 'tests')
    # Nonlinearity calibration
    ### Create a dummy non-linearity file ####
    #Create a mock dataset because it is a required input when creating a NonLinearityCalibration
    dummy_dataset = mocks.create_prescan_files()
    # Make a non-linearity correction calibration file
    input_non_linearity_filename = "nonlin_table_TVAC.txt"
    input_non_linearity_path = os.path.join(tests_dir, "test_data", input_non_linearity_filename)
    test_non_linearity_filename = input_non_linearity_filename.split(".")[0] + ".fits"
    nonlin_fits_filepath = os.path.join(tests_dir, "test_data", test_non_linearity_filename)
    tvac_nonlin_data = np.genfromtxt(input_non_linearity_path, delimiter=",")

    pri_hdr, ext_hdr = mocks.create_default_headers()
    new_nonlinearity = data.NonLinearityCalibration(tvac_nonlin_data,pri_hdr=pri_hdr,ext_hdr=ext_hdr,input_dataset = dummy_dataset)
    new_nonlinearity.filename = nonlin_fits_filepath
    new_nonlinearity.pri_hdr = pri_hdr
    new_nonlinearity.ext_hdr = ext_hdr
    this_caldb.create_entry(new_nonlinearity)
    # make PC dark
    walker.walk_corgidrp(l1_data_dark_filelist, '', output_dir, template="l1_to_l2b_pc_dark.json")
    for f in os.listdir(output_dir):
        if f.endswith('_pc_dark.fits'):
            pc_dark_filename = f
    # add the Dark to the Caldb for processing the illuminated
    pc_dark_file = os.path.join(output_dir, pc_dark_filename)
    dark_entry = data.Dark(pc_dark_file)
    this_caldb.create_entry(dark_entry)
    # make PC illuminated, subtracting the PC dark
    walker.walk_corgidrp(l1_data_ill_filelist, '', output_dir, template="l1_to_l2b_pc.json")
    # get photon-counted frame
    for f in os.listdir(output_dir):
        if f.endswith('_pc.fits'):
            pc_filename = f
    pc_file = os.path.join(output_dir, pc_filename)
    pc_frame = fits.getdata(pc_file)
    pc_frame_err = fits.getdata(pc_file, 'ERR')
    pc_dark_frame = fits.getdata(pc_dark_file)
    pc_dark_frame_err = fits.getdata(pc_dark_file, 'ERR')

    # more frames gets a better agreement; agreement to 1% for ~160 darks and illuminated
    assert np.isclose(np.nanmean(pc_frame), ill_mean - dark_mean, rtol=0.01) 
    assert np.isclose(np.nanmean(pc_dark_frame), dark_mean, rtol=0.01) 
    assert pc_frame_err.min() >= 0
    assert pc_dark_frame_err.min() >= 0

    this_caldb.remove_entry(kgain)
    this_caldb.remove_entry(noise_map)
    this_caldb.remove_entry(new_nonlinearity)


if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2b PC end-to-end test")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_e2e(outputdir)
