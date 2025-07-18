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

thisfile_dir = os.path.dirname(__file__) # this file's folder


def fix_headers_for_tvac(
    list_of_fits,
    ):
    """ 
    Fixes TVAC headers to be consistent with flight headers. 
    Writes headers back to disk

    Args:
        list_of_fits (list): list of FITS files that need to be updated.
    """
    print("Fixing TVAC headers")
    for file in list_of_fits:
        fits_file = fits.open(file)
        prihdr = fits_file[0].header
        exthdr = fits_file[1].header
        # Adjust VISTYPE
        prihdr['OBSNUM'] = prihdr['OBSID']
        exthdr['EMGAIN_C'] = exthdr['CMDGAIN']
        exthdr['EMGAIN_A'] = -1
        exthdr['DATALVL'] = exthdr['DATA_LEVEL']
        prihdr["OBSNAME"] = prihdr['OBSTYPE']
        prihdr['PHTCNT'] = False
        exthdr['ISPC'] = False
        # Update FITS file
        fits_file.writeto(file, overwrite=True)

@pytest.mark.e2e
def test_l1_to_l2a(e2edata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    l2a_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L2a")
    nonlin_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals", "nonlin_table_240322.txt")
    dark_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals", "dark_current_20240322.fits")
    fpn_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals", "fpn_20240322.fits")
    cic_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals", "cic_20240322.fits")

    # make output directory if needed
    l2a_outputdir = os.path.join(e2eoutput_path, "l1_to_l2a_output")
    if not os.path.exists(l2a_outputdir):
        os.mkdir(l2a_outputdir)

    # define the raw science data to process

    l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]] # just grab the first two files
    mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]] # grab the last two real data to mock the calibration 
    tvac_l2a_filelist = [os.path.join(l2a_datadir, "{0}.fits".format(i)) for i in [90528, 90530]] # just grab the first two files

    # fix headers for TVAC
    fix_headers_for_tvac(l1_data_filelist)

    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=l2a_outputdir, filename="mock_nonlinearcal.fits" )

    # add calibration file to caldb
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(nonlinear_cal)


    # NoiseMap
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
    err_hdr['BUNIT'] = 'detected electron'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=l2a_outputdir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    # KGain
    kgain_val = 8.7
    kgain = data.KGain(kgain_val, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=l2a_outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)

    ####### Run the walker on some test_data

    walker.walk_corgidrp(l1_data_filelist, "", l2a_outputdir, template="l1_to_l2a_basic.json")

    # clean up by removing entry
    this_caldb.remove_entry(nonlinear_cal)
    this_caldb.remove_entry(noise_map)
    this_caldb.remove_entry(kgain)
    
    ##### Check against TVAC data
    new_l2a_filenames = [os.path.join(l2a_outputdir, "{0}.fits".format(i)) for i in [90499, 90500]]

    for new_filename, tvac_filename in zip(new_l2a_filenames, tvac_l2a_filelist):
        img = data.Image(new_filename)

        with fits.open(tvac_filename) as hdulist:
            tvac_dat = hdulist[1].data
        
        diff = img.data - tvac_dat

        assert np.all(np.abs(diff) < 1e-5)

        # # plotting script for debugging
        # import matplotlib.pylab as plt
        # fig = plt.figure(figsize=(10,3.5))
        # fig.add_subplot(131)
        # plt.imshow(img.data, vmin=-20, vmax=2000, cmap="viridis")
        # plt.title("corgidrp")
        # plt.xlim([500, 560])
        # plt.ylim([475, 535])

        # fig.add_subplot(132)
        # plt.imshow(tvac_dat, vmin=-20, vmax=2000, cmap="viridis")
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
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    e2edata_dir = '/home/jwang/Desktop/CGI_TVAC_Data/'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_l1_to_l2a(e2edata_dir, outputdir)