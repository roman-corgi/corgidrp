import argparse
import os
import glob
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
        exthdr['ISPC'] = False
    # exthdr['KGAINPAR'] = exthdr['KGAIN']
        prihdr["OBSNAME"] = prihdr['OBSTYPE']
        prihdr['PHTCNT'] = False
        # Update FITS file
        fits_file.writeto(file, overwrite=True)


@pytest.mark.e2e
def test_astrom_e2e(e2edata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    noise_characterization_path = os.path.join(e2edata_path, "TV-20_EXCAM_noise_characterization", "darkmap")

    # make output directory if needed
    astrom_cal_outputdir = os.path.join(e2eoutput_path, "astrom_cal_output")
    if not os.path.exists(astrom_cal_outputdir):
        os.mkdir(astrom_cal_outputdir)

    # assume all cals are in the same directory
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")  
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")

    # create raw data that includes injected stars with gaussian psfs
    jwst_calfield_path = os.path.join(os.path.dirname(thisfile_dir), "test_data", "JWST_CALFIELD2020.csv")
    image_sources = mocks.create_astrom_data(jwst_calfield_path, add_gauss_noise=False)
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    # create a directory in the output dir to hold the simulated data files
    rawdata_dir = os.path.join(astrom_cal_outputdir, 'data')
    if not os.path.exists(rawdata_dir):
        os.mkdir(rawdata_dir)

    # get an idea of the noise rms
    noise_datacube = []
    for dark in os.listdir(noise_characterization_path):
        with fits.open(os.path.join(noise_characterization_path, dark)) as hdulist:
            dark_dat = hdulist[1].data
            noise_datacube.append(dark_dat)
    noise_std = np.std(noise_datacube, axis=0)
    noise_rms = np.mean(noise_std)

    for dark in os.listdir(noise_characterization_path):
        with fits.open(os.path.join(noise_characterization_path, dark)) as hdulist:
            dark_dat = hdulist[1].data
            hdulist[0].header['VISTYPE'] = "BORESITE"
            # setting SNR to ~250 (arbitrary SNR)
            scaled_image = ((250 * noise_rms) / np.max(image_sources[0].data)) * image_sources[0].data
            scaled_image = scaled_image.astype(type(dark_dat[0][0]))
            hdulist[1].data[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] += scaled_image
            # update headers
            for key in image_sources[0].pri_hdr:
                if key in ["RA", "DEC"]:
                    # must overwrite!
                    hdulist[0].header[key] = image_sources[0].pri_hdr[key]
                elif key not in hdulist[0].header:
                    hdulist[0].header[key] = image_sources[0].pri_hdr[key]

            for ext_key in image_sources[0].ext_hdr:
                if ext_key == "HISTORY":
                    for item in image_sources[0].ext_hdr[ext_key]:
                        hdulist[1].header.add_history(item)
                elif ext_key not in hdulist[1].header:
                    hdulist[1].header[ext_key] = image_sources[0].ext_hdr[ext_key]

            # save to the data dir in the output directory
            hdulist.writeto(os.path.join(rawdata_dir, dark[:-5]+'.fits'), overwrite=True)

    # define the raw science data to process
    ## replace w my raw data sets
    sim_data_filelist = [os.path.join(rawdata_dir, f) for f in os.listdir(rawdata_dir)] # full paths to simulated data
    mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]] # grab the last two real data to mock the calibration 

    # update headers of TVAC data
    fix_headers_for_tvac(sim_data_filelist)

    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make calibration files using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    this_caldb = caldb.CalDB() # connection to cal DB

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=astrom_cal_outputdir, filename="mock_nonlinearcal.fits" )
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    kgain_val = 8.7
    kgain = data.KGain(kgain_val, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    kgain.save(filedir=astrom_cal_outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)

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
    noise_map.save(filedir=astrom_cal_outputdir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    ## Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    flat.save(filedir=astrom_cal_outputdir, filename="mock_flat.fits")
    this_caldb.create_entry(flat)

    # bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    bp_map.save(filedir=astrom_cal_outputdir, filename="mock_bpmap.fits")
    this_caldb.create_entry(bp_map)


    ####### Run the walker on some test_data

    walker.walk_corgidrp(sim_data_filelist, "", astrom_cal_outputdir)


    # clean up by removing entry
    this_caldb.remove_entry(nonlinear_cal)
    this_caldb.remove_entry(kgain)
    this_caldb.remove_entry(noise_map)
    this_caldb.remove_entry(flat)
    this_caldb.remove_entry(bp_map)

    ## Check against astrom ground truth -- target= [80.553428801, -69.514096821],
    ## plate scale = 21.8[mas/pixel], north angle = 45 [deg]
    output_files = []
    for file in os.scandir(astrom_cal_outputdir): # sort between added files and subdirectories
        if file.is_file():
            output_files.append(file)

    expected_platescale = 21.8
    expected_northangle = 45
    target = (80.553428801, -69.514096821)

    astrom_cal = data.AstrometricCalibration(glob.glob(os.path.join(astrom_cal_outputdir, '*_AST_CAL.fits'))[0])

    # check that the astrometric calibration filename is based on the last file in the input file list
    expected_last_filename = sim_data_filelist[-1].split('L1_')[-1].split('.fits')[0]
    assert astrom_cal.filename.split('L2b')[-1] == expected_last_filename + '_AST_CAL.fits'

    # check orientation is correct within 0.05 [deg]
    # and plate scale is correct within 0.5 [mas] (arbitrary)
    assert astrom_cal.platescale == pytest.approx(expected_platescale, abs=0.5)

    assert astrom_cal.northangle == pytest.approx(expected_northangle, abs=0.05)

    # check that the center is correct within 3 [mas]
    # the simulated image should have no shift from the target
    ra, dec = astrom_cal.boresight[0], astrom_cal.boresight[1]
    assert ra == pytest.approx(target[0], abs=8.333e-7)
    assert dec == pytest.approx(target[1], abs=8.333e-7)

    this_caldb.remove_entry(astrom_cal)

if __name__ == "__main__":
    e2edata_dir = "/Users/macuser/Roman/corgidrp_develop/calibration_notebooks/TVAC"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2b->boresight end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_astrom_e2e(e2edata_dir, outputdir)