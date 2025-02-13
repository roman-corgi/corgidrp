"""
    End-to-end test for processing L1 data to L2b data.

    This function processes raw L1 data through a series of calibration and noise correction steps,
    then compares the output against the expected TVAC dataset. If the difference is below a threshold,
    the test is considered successful.

    Parameters:
    
    tvacdata_path : str
        Path to the directory containing the TVAC data.
    e2eoutput_path : str
        Path to the directory where processed output will be saved.

    Steps:
    
    1. Define paths for L1, L2b, and calibration data.
    2. Generate necessary calibration files and add them to the calibration database.
    3. Run the `walker` function to process L1 data and create L2b output.
    4. Compare the processed L2b output against the TVAC data to ensure accuracy.

    Asserts:
    
    Compares the processed L2b files against the expected TVAC files and asserts that the difference
    is less than 1e-5.
    
"""

import argparse
import os
import pytest
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector

thisfile_dir = os.path.dirname(__file__)  # This file's folder

@pytest.mark.e2e
def test_l1_to_l2b(tvacdata_path, e2eoutput_path):
    """
    Test the conversion of L1 data to L2b data by processing the raw science data, 
    generating necessary calibration files, and comparing the output with the TVAC data.

    Args:
        tvacdata_path (str): Path to the directory containing the TVAC data.
        e2eoutput_path (str): Path to the directory to store output files.
    """
    # Figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L1")
    l2b_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L2b")
    processed_cal_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals")

    # Make output directory if needed
    l2b_outputdir = os.path.join(e2eoutput_path, "l1_to_l2b_output")
    if not os.path.exists(l2b_outputdir):
        os.mkdir(l2b_outputdir)

    # Assume all calibration files are in the same directory
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")

    # Define the raw science data to process
    l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]]  # Just grab the first two files
    mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]]  # Grab the last two real data to mock the calibration
    tvac_l2b_filelist = [os.path.join(l2b_datadir, "{0}.fits".format(i)) for i in [90529, 90531]]  # Just grab the first two files

    ###### Setup necessary calibration files
    # Create necessary calibration files
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] = corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    this_caldb = caldb.CalDB()  # Connection to calibration DB

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=l2b_outputdir, filename="mock_nonlinearcal.fits")
    this_caldb.create_entry(nonlinear_cal)

    # KGain calibration
    kgain_val = 8.7
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                       input_dataset=mock_input_dataset)
    kgain.save(filedir=l2b_outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)

    # NoiseMap calibration
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_path) as hdulist:
        dark_dat = hdulist[0].data
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electrons'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                       input_dataset=mock_input_dataset, err=noise_map_noise,
                                       dq=noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=l2b_outputdir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    # Flat field calibration
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    flat.save(filedir=l2b_outputdir, filename="mock_flat.fits")
    this_caldb.create_entry(flat)

    # Bad pixel map calibration
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    bp_map.save(filedir=l2b_outputdir, filename="mock_bpmap.fits")
    this_caldb.create_entry(bp_map)

    ####### Run the walker on some test data
    walker.walk_corgidrp(l1_data_filelist, "", l2b_outputdir)

    # Clean up by removing entries
    this_caldb.remove_entry(nonlinear_cal)
    this_caldb.remove_entry(kgain)
    this_caldb.remove_entry(noise_map)
    this_caldb.remove_entry(flat)
    this_caldb.remove_entry(bp_map)

    ##### Check against TVAC data
    new_l2b_filenames = [os.path.join(l2b_outputdir, "{0}.fits".format(i)) for i in [90499, 90500]]

    for new_filename, tvac_filename in zip(new_l2b_filenames, tvac_l2b_filelist):
        img = data.Image(new_filename)

        with fits.open(tvac_filename) as hdulist:
            tvac_dat = hdulist[1].data
        
        diff = img.data - tvac_dat

        assert np.all(np.abs(diff) < 1e-5)

        # Debugging plot code (if needed)
        # import matplotlib.pylab as plt
        # fig = plt.figure(figsize=(10,3.5))
        # fig.add_subplot(131)
        # plt.imshow(img.data, vmin=-0.01, vmax=45, cmap="viridis")
        # plt.title("corgidrp")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # fig.add_subplot(132)
        # plt.imshow(tvac_dat, vmin=-0.01, vmax=45, cmap="viridis")
        # plt.title("TVAC")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # fig.add_subplot(133)
        # plt.imshow(diff, vmin=-0.01, vmax=0.01, cmap="inferno")
        # plt.title("difference")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # plt.show()

if __name__ == "__main__":
    """
    Main entry point for running the test. Accepts command-line arguments for the 
    paths to the TVAC data and output directory.
    """
    # Default paths
    tvacdata_dir = "/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/Working_Folder/"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="Run the L1 -> L2b end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to the CGI_TVAC_Data folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="Directory to write results to [%(default)s]")
    args = ap.parse_args()
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    test_l1_to_l2b(tvacdata_dir, outputdir)
