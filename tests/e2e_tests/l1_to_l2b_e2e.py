import argparse
import os
import glob
import pytest
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
from datetime import datetime
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
        prihdr['VISTYPE'] = "TDEMO"
        prihdr['OBSNUM'] = prihdr['OBSID']
        exthdr['EMGAIN_C'] = exthdr['CMDGAIN']
        exthdr['EMGAIN_A'] = -1
        exthdr['DATALVL'] = exthdr['DATA_LEVEL']
        if 'KGAIN' in exthdr:
            exthdr['KGAINPAR'] = exthdr['KGAIN']
        else:
            exthdr['KGAINPAR'] = 8.7
        prihdr["OBSNAME"] = prihdr['OBSTYPE']
        prihdr['PHTCNT'] = False
        exthdr['ISPC'] = False
        exthdr['BUNIT'] = 'DN'
        # Update FITS file
        fits_file.writeto(file, overwrite=True)

@pytest.mark.e2e
def test_l1_to_l2b(e2edata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    l2a_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L2a")
    l2b_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L2b")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")

    # make output directory if needed
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_l2b_output")
    if not os.path.exists(test_outputdir):
        os.mkdir(test_outputdir)
    # separate L2a and L2b outputdirs
    l2a_outputdir = os.path.join(test_outputdir, "l2a")
    if not os.path.exists(l2a_outputdir):
        os.mkdir(l2a_outputdir)
    l2b_outputdir = os.path.join(test_outputdir, "l2b")
    if not os.path.exists(l2b_outputdir):
        os.mkdir(l2b_outputdir)

    # assume all cals are in the same directory
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")

    # define the raw science data to process

    l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]] # just grab the first two files
    mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]] # grab the last two real data to mock the calibration 
    tvac_l2a_filelist = [os.path.join(l2a_datadir, "{0}.fits".format(i)) for i in [90528, 90530]] # just grab the first two files
    tvac_l2b_filelist = [os.path.join(l2b_datadir, "{0}.fits".format(i)) for i in [90529, 90531]] # just grab the first two files

    # modify TVAC headers for produciton
    fix_headers_for_tvac(l1_data_filelist)

    ########### Prepare input files with proper L1 filenames

    # Create input_data subfolder
    input_data_dir = os.path.join(test_outputdir, 'input_data')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)
    
    # Save input files with proper L1 filenames
    # Get current time once outside the loop
    base_time = datetime.now()
    
    input_filelist = []
    for i, filepath in enumerate(l1_data_filelist):
        # Extract frame number from original filename
        original_filename = os.path.basename(filepath)
        
        # Extract frame number from filename (e.g., "90499" from "90499.fits")
        frame_number = original_filename.split('.')[0]
        
        if frame_number.isdigit():
            visitid = frame_number.zfill(19)  # Pad with zeros to make 19 digits
        else:
            visitid = f"{i:019d}"  # Fallback- use file index padded to 19 digits
        
        # Create unique timestamp by incrementing seconds for each file
        # Handle second rollover properly
        new_second = (base_time.second + i) % 60
        new_minute = base_time.minute + ((base_time.second + i) // 60)
        file_time = base_time.replace(minute=new_minute, second=new_second)
        # Use the format_ftimeutc function from data.py to get consistent 3-digit seconds format
        time_str = data.format_ftimeutc(file_time.isoformat())
        
        # Load the file
        with fits.open(filepath) as hdulist:
            # Create new filename: cgi_{visitid}_{time_str}_l1_.fits
            new_filename = f"cgi_{visitid}_{time_str}_l1_.fits"
            new_filepath = os.path.join(input_data_dir, new_filename)
            
            # Save with new filename
            hdulist.writeto(new_filepath, overwrite=True)
            input_filelist.append(new_filepath)
    
    # Use the renamed input files for the DRP walker
    l1_data_filelist = input_filelist

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

    # create a DetectorParams object and save it
    detector_params = data.DetectorParams({})
    # Generate unique timestamp for detector params
    dp_time = base_time.replace(second=(base_time.second + 103) % 60, minute=(base_time.minute + (base_time.second + 103) // 60))
    dp_time_str = data.format_ftimeutc(dp_time.isoformat())
    dp_filename = f"cgi_0000000000000090526_{dp_time_str}_dpr_cal.fits"
    detector_params.save(filedir=test_outputdir, filename=dp_filename)
    this_caldb.create_entry(detector_params)

    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                input_dataset=mock_input_dataset)
    # Generate unique timestamp for nonlinear calibration
    nln_time = base_time.replace(second=(base_time.second + 104) % 60, minute=(base_time.minute + (base_time.second + 104) // 60))
    nln_time_str = data.format_ftimeutc(nln_time.isoformat())
    nln_filename = f"cgi_0000000000000090526_{nln_time_str}_nln_cal.fits"
    nonlinear_cal.save(filedir=test_outputdir, filename=nln_filename)
    this_caldb.create_entry(nonlinear_cal)

    # KGain
    kgain_val = 8.7
    kgain = data.KGain(kgain_val, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    # Generate unique timestamp for KGain calibration
    kgain_time = base_time.replace(second=(base_time.second + 105) % 60, minute=(base_time.minute + (base_time.second + 105) // 60))
    kgain_time_str = data.format_ftimeutc(kgain_time.isoformat())
    kgain_filename = f"cgi_0000000000000090526_{kgain_time_str}_krn_cal.fits"
    kgain.save(filedir=test_outputdir, filename=kgain_filename)
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
    # Generate unique timestamp for noise map calibration
    dnm_time = base_time.replace(second=(base_time.second + 106) % 60, minute=(base_time.minute + (base_time.second + 106) // 60))
    dnm_time_str = data.format_ftimeutc(dnm_time.isoformat())
    dnm_filename = f"cgi_0000000000000090527_{dnm_time_str}_dnm_cal.fits"
    noise_map.save(filedir=test_outputdir, filename=dnm_filename)
    this_caldb.create_entry(noise_map)

    ## Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    # Generate unique timestamp for flat field calibration
    flat_time = base_time.replace(second=(base_time.second + 107) % 60, minute=(base_time.minute + (base_time.second + 107) // 60))
    flat_time_str = data.format_ftimeutc(flat_time.isoformat())
    flat_filename = f"cgi_0000000000000090526_{flat_time_str}_flt_cal.fits"
    flat.save(filedir=test_outputdir, filename=flat_filename)
    this_caldb.create_entry(flat)

    # bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    # Generate unique timestamp for bad pixel map calibration
    bpm_time = base_time.replace(second=(base_time.second + 108) % 60, minute=(base_time.minute + (base_time.second + 108) // 60))
    bpm_time_str = data.format_ftimeutc(bpm_time.isoformat())
    bpm_filename = f"cgi_0000000000000090527_{bpm_time_str}_bpm_cal.fits"
    bp_map.save(filedir=test_outputdir, filename=bpm_filename)
    this_caldb.create_entry(bp_map)

    ####### Run the walker on some test_data

    # l1 -> l2a processing
    walker.walk_corgidrp(l1_data_filelist, "", l2a_outputdir)

    # l2a -> l2b processing
    # Find the actual L2a output files generated by the DRP walker
    new_l2a_filenames = glob.glob(os.path.join(l2a_outputdir, "*_l2a.fits"))
    
    # Sort by modification time to get the most recent files
    new_l2a_filenames.sort(key=os.path.getmtime)
    
    # Take the first two files (should correspond to our input files)
    new_l2a_filenames = new_l2a_filenames[:2]
    
    walker.walk_corgidrp(new_l2a_filenames, "", l2b_outputdir)


    # clean up by removing entry
    this_caldb.remove_entry(nonlinear_cal)
    this_caldb.remove_entry(kgain)
    this_caldb.remove_entry(noise_map)
    this_caldb.remove_entry(flat)
    this_caldb.remove_entry(bp_map)
    this_caldb.remove_entry(detector_params)

    ##### Check against TVAC data
    # l2a data
    for new_filename, tvac_filename in zip(new_l2a_filenames, tvac_l2a_filelist):
        img = data.Image(new_filename)

        with fits.open(tvac_filename) as hdulist:
            tvac_dat = hdulist[1].data
        
        diff = img.data - tvac_dat

        assert np.all(np.abs(diff) < 1e-5)

    # l2b data
    # Find the actual L2b output files generated by the DRP walker
    new_l2b_filenames = glob.glob(os.path.join(l2b_outputdir, "*_l2b.fits"))
    
    # Sort by modification time to get the most recent files
    new_l2b_filenames.sort(key=os.path.getmtime)
    
    # Take the first two files (should correspond to our input files)
    new_l2b_filenames = new_l2b_filenames[:2]

    for new_filename, tvac_filename in zip(new_l2b_filenames, tvac_l2b_filelist):
        img = data.Image(new_filename)

        with fits.open(tvac_filename) as hdulist:
            tvac_dat = hdulist[1].data
        
        diff = img.data - tvac_dat

        assert np.all(np.abs(diff) < 1e-5)

        # plotting script for debugging
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
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    e2edata_dir =  r'/Users/jmilton/Documents/CGI/CGI_TVAC_Data/'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_l1_to_l2b(e2edata_dir, outputdir)