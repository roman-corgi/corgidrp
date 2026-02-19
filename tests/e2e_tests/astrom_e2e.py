import argparse
import os
import shutil
import glob
import pytest
import warnings
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.check as check
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector

thisfile_dir = os.path.dirname(__file__) # this file's folder


def fix_str_for_tvac(
    list_of_fits,
    ):
    """ 
    Gets around EMGAIN_A being set to 1 in TVAC data.
    
    Args:
        list_of_fits (list): list of FITS files that need to be updated.
    """
    for file in list_of_fits:
        fits_file = fits.open(file)
        prihdr = fits_file[0].header
        exthdr = fits_file[1].header
        if float(exthdr['EMGAIN_A']) == 1 and exthdr['HVCBIAS'] <= 0:
            exthdr['EMGAIN_A'] = -1 #for new SSC-updated TVAC files which have EMGAIN_A by default as 1 regardless of the commanded EM gain
        if type(exthdr['EMGAIN_C']) is str:
            exthdr['EMGAIN_C'] = float(exthdr['EMGAIN_C'])
        
        if 'OBSTYPE' in prihdr:
            prihdr["OBSNAME"] = prihdr['OBSTYPE']
        else:
            prihdr["OBSNAME"] = "BORESITE"  # Default value
        prihdr["PHTCNT"] = False
        
        exthdr["ISPC"] = False
        
        # Update FITS file
        fits_file.writeto(file, overwrite=True)


@pytest.mark.e2e
def test_astrom_e2e(e2edata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    noise_characterization_path = os.path.join(e2edata_path, "TV-20_EXCAM_noise_characterization", "darkmap")

    # make output directory if needed
    astrom_cal_outputdir = os.path.join(e2eoutput_path, "astrom_cal_e2e")
    if not os.path.exists(astrom_cal_outputdir):
        os.makedirs(astrom_cal_outputdir)
    # clean out any files from a previous run
    for f in os.listdir(astrom_cal_outputdir):
        file_path = os.path.join(astrom_cal_outputdir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
    # Create calibrations subfolder for mock calibration products
    calibrations_dir = os.path.join(astrom_cal_outputdir, "calibrations")
    if not os.path.exists(calibrations_dir):
        os.mkdir(calibrations_dir)

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
    input_data_dir = os.path.join(astrom_cal_outputdir, 'input_l1')
    if not os.path.exists(input_data_dir):
        os.mkdir(input_data_dir)
    # clean out any files from a previous run
    for f in os.listdir(input_data_dir):
        file_path = os.path.join(input_data_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # get an idea of the noise rms
    noise_datacube = []
    for dark in os.listdir(noise_characterization_path):
        if not dark.lower().endswith('.fits'):
            continue
        with fits.open(os.path.join(noise_characterization_path, dark)) as hdulist:
            dark_dat = hdulist[1].data
            noise_datacube.append(dark_dat)
    noise_std = np.std(noise_datacube, axis=0)
    noise_rms = np.mean(noise_std)

    for dark in os.listdir(noise_characterization_path):
        if not dark.lower().endswith('.fits'):
                continue
        with fits.open(os.path.join(noise_characterization_path, dark)) as hdulist:
            dark_dat = hdulist[1].data
            hdulist[0].header['VISTYPE'] = "CGIVST_CAL_BORESIGHT"
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
            
            hdulist[0].header['RA'] = image_sources[0].pri_hdr['RA'] 
            hdulist[0].header['DEC'] = image_sources[0].pri_hdr['DEC']
            # save to the data dir in the output directory
            hdulist.writeto(os.path.join(input_data_dir, dark[:-5]+'.fits'), overwrite=True)

    # define the raw science data to process
    ## replace w my raw data sets
    sim_data_filelist = [os.path.join(input_data_dir, f) for f in os.listdir(input_data_dir)] # full paths to simulated data
    mock_cal_filelist = []
    # grab 2 files of real data to mock the calibration
    for filename in os.listdir(l1_datadir):
        if filename.lower().endswith('.fits'):
            mock_cal_filelist.append(os.path.join(l1_datadir, filename))
        if len(mock_cal_filelist) == 2:
            break

    # Copy and fix mock cal headers
    mock_cal_dir = os.path.join(astrom_cal_outputdir, 'mock_cal_input')
    os.makedirs(mock_cal_dir, exist_ok=True)
    mock_cal_filelist = [
        shutil.copy2(f, os.path.join(mock_cal_dir, os.path.basename(f)))
        for f in mock_cal_filelist
    ]
    mock_cal_filelist = check.fix_hdrs_for_tvac(mock_cal_filelist, mock_cal_dir)

    # Fix string values
    fix_str_for_tvac(sim_data_filelist)

    ###### Setup necessary calibration files
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()  # connection to cal DB

    # Create necessary calibration files
    # we are going to make calibration files using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    kgain_val = 8.7
    signal_array = np.linspace(0, 50)
    noise_array = np.sqrt(signal_array)
    ptc = np.column_stack([signal_array, noise_array])
    kgain = data.KGain(kgain_val, ptc=ptc, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
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
    ext_hdr['B_O'] = 0.
    ext_hdr['B_O_ERR'] = 0.
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    mocks.rename_files_to_cgi_format(list_of_fits=[noise_map], output_dir=calibrations_dir, level_suffix="dnm_cal")
    this_caldb.create_entry(noise_map)

    ## Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[flat], output_dir=calibrations_dir, level_suffix="flt_cal")
    this_caldb.create_entry(flat)

    # bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[bp_map], output_dir=calibrations_dir, level_suffix="bpm_cal")
    this_caldb.create_entry(bp_map)

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    ####### Run the walker on some test_data

    with warnings.catch_warnings():  
        warnings.filterwarnings('ignore', category=UserWarning)# prevent UserWarning: Number of frames which made the DetectorNoiseMaps product is less than the number of frames in input_dataset
        walker.walk_corgidrp(sim_data_filelist, "", astrom_cal_outputdir)
    
    # Organize output files into subdirectories
    subdirs = {
        'l1_to_l2a': os.path.join(astrom_cal_outputdir, "l1_to_l2a"),
        'l2a_to_l2b': os.path.join(astrom_cal_outputdir, "l2a_to_l2b")
    }
    
    # Create subdirectories
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    # Move files to appropriate subdirectories
    for filename in os.listdir(astrom_cal_outputdir):
        filepath = os.path.join(astrom_cal_outputdir, filename)
        if not os.path.isfile(filepath):
            continue
            
        if '_l2a' in filename and filename.endswith('.fits'):
            shutil.move(filepath, os.path.join(subdirs['l1_to_l2a'], filename))
        elif '_l2b' in filename and filename.endswith('.fits'):
            shutil.move(filepath, os.path.join(subdirs['l2a_to_l2b'], filename))
        elif filename.endswith('_recipe.json'):
            if 'l1_to_l2a' in filename:
                shutil.move(filepath, os.path.join(subdirs['l1_to_l2a'], filename))
            elif 'l2a_to_l2b' in filename:
                shutil.move(filepath, os.path.join(subdirs['l2a_to_l2b'], filename))
        elif '_cal.fits' in filename and not filename.endswith('_ast_cal.fits'):
            shutil.move(filepath, os.path.join(calibrations_dir, filename))

    ## Check against astrom ground truth -- target= [80.553428801, -69.514096821],
    ## plate scale = 21.8[mas/pixel], north angle = 45 [deg]
    output_files = []
    for file in os.scandir(astrom_cal_outputdir): # sort between added files and subdirectories
        if file.is_file():
            output_files.append(file)

    expected_platescale = 21.8
    expected_northangle = 45.
    target = (80.553428801, -69.514096821)

    # Look for astrometric calibration file in the main directory (it's not L2a or L2b data)
    astrom_cal_files = glob.glob(os.path.join(astrom_cal_outputdir, '*_ast_cal.fits'))
    if not astrom_cal_files:
        # If not in main directory, check subdirectories
        astrom_cal_files = glob.glob(os.path.join(astrom_cal_outputdir, '**', '*_ast_cal.fits'), recursive=True)
    astrom_cal = data.AstrometricCalibration(astrom_cal_files[0])

    # check that the astrometric calibration filename is based on the last file in the input file list
    expected_last_filename = sim_data_filelist[-1].split('_l1_')[0].split(os.path.sep)[-1]
    assert astrom_cal.filename.split('l2b')[-1] == expected_last_filename + '_ast_cal.fits'

    # check orientation is correct within 0.05 [deg]
    # and plate scale is correct within 0.5 [mas] (arbitrary)
    assert astrom_cal.platescale == pytest.approx(expected_platescale, abs=0.5)

    assert astrom_cal.northangle == pytest.approx(expected_northangle, abs=0.05)

    # check that the center is correct within 3 [mas]
    # the simulated image should have no shift from the target
    ra, dec = astrom_cal.boresight[0], astrom_cal.boresight[1]
    assert ra == pytest.approx(target[0], abs=8.333e-7)
    assert dec == pytest.approx(target[1], abs=8.333e-7)

    check.compare_to_mocks_hdrs(astrom_cal_files[0], mocks.create_default_L2b_headers)
    
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

if __name__ == "__main__":
    #e2edata_dir = "/Users/macuser/Roman/corgidrp_develop/calibration_notebooks/TVAC"
    e2edata_dir = '/Users/kevinludwick/Documents/DRP_E2E_Test_Files_v2/E2E_Test_Data'#
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