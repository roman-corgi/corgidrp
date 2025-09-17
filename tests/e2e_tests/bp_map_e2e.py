import argparse
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.io import fits
import corgidrp
import datetime
from corgidrp import data
from corgidrp import caldb
from corgidrp import detector
from corgidrp import darks
from corgidrp import walker
from corgidrp import mocks

# Get the directory of the current script file
thisfile_dir = os.path.dirname(__file__)

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
        exthdr = fits_file[1].header
        if float(exthdr['EMGAIN_A']) == 1:
            exthdr['EMGAIN_A'] = -1 #for new SSC-updated TVAC files which have EMGAIN_A by default as 1 regardless of the commanded EM gain
        if type(exthdr['EMGAIN_C']) is str:
            exthdr['EMGAIN_C'] = float(exthdr['EMGAIN_C'])
        # Update FITS file
        fits_file.writeto(file, overwrite=True)

def fix_headers_for_tvac(
    list_of_fits,
    output_dir,
    ):
    """ 
    Fixes TVAC headers to be consistent with flight headers and updates filenames.
    Writes headers back to disk with proper L1 filename convention.

    Args:
        list_of_fits (list): list of FITS files that need to be updated.
        output_dir (str): directory to write results to
    """
    print("Fixing TVAC headers and filenames")
    for i, file in enumerate(list_of_fits):
        fits_file = fits.open(file)
        prihdr = fits_file[0].header
        exthdr = fits_file[1].header
        
        # Extract visit ID from primary header VISITID keyword
        visitid = prihdr.get('VISITID', None)
        if visitid is not None:
            # Convert to string and pad to 19 digits
            visitid = str(visitid).zfill(19)
        else:
            # Fallback: try to extract from filename or use file index
            current_filename = os.path.basename(file)
            if current_filename.replace('.fits', '').isdigit():
                # Handle numbered files like 90500.fits
                frame_number = current_filename.replace('.fits', '')
                visitid = frame_number.zfill(19)  # Pad with zeros to make 19 digits
            else:
                visitid = f"{i:019d}"  # Fallback: use file index padded to 19 digits
        
        filetime = exthdr.get('FILETIME', prihdr.get('FILETIME', None))
        
        # Convert filetime to the format expected in filenames (YYYYMMDDtHHMMSS)
        if filetime and 'T' in filetime:
            try:
                dt = datetime.datetime.fromisoformat(filetime.replace('Z', '+00:00'))
                filetime = dt.strftime('%Y%m%dt%H%M%S')
            except:
                filetime = datetime.datetime.now().strftime('%Y%m%dt%H%M%S')  # fallback to current time
        elif not filetime:
            filetime = datetime.datetime.now().strftime('%Y%m%dt%H%M%S')  # fallback to current time
        
        # Create new filename with proper L1 convention
        input_data_dir = os.path.join(output_dir, 'input_l1')
        if not os.path.exists(input_data_dir):
            os.mkdir(input_data_dir)
        new_filename = os.path.join(input_data_dir, f'cgi_{visitid}_{filetime}_l1_.fits')
        
        # Adjust VISTYPE
        # Set OBSNUM - use OBSID if it exists, otherwise use a default value
        if 'OBSID' in prihdr:
            prihdr['OBSNUM'] = prihdr['OBSID']
        else:
            prihdr['OBSNUM'] = '90500'  # Default OBSNUM value
        
        exthdr['EMGAIN_A'] = -1
        
        # Update FITS file with new filename (create a copy)
        fits_file.writeto(new_filename, overwrite=True)
        fits_file.close()

@pytest.mark.e2e
def test_bp_map_master_dark_e2e(e2edata_path, e2eoutput_path):
    # Define paths for input L1 data and calibration files
    l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")

    # Create main output directory and master dark subfolder
    main_output_dir = os.path.join(e2eoutput_path, "bp_map_cal_e2e")
    bp_map_outputdir = os.path.join(main_output_dir, "bp_map_master_dark")
    if not os.path.exists(bp_map_outputdir):
        os.makedirs(bp_map_outputdir)

    calibrations_dir = os.path.join(bp_map_outputdir, "calibrations")
    if not os.path.exists(calibrations_dir):
        os.mkdir(calibrations_dir)

    # Paths to calibration files
    dark_current_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_ref_path = os.path.join(processed_cal_path, "fixed_bp_zeros.fits")

    # Define the list of raw science data files for input, selecting the first two files as examples
    input_image_filelist = []
    l1_data_filelist = []
    for filename in os.listdir(l1_datadir)[:2]:
        l1_data_filelist.append(os.path.join(l1_datadir, filename))
    #l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]]

    # update TVAC headers
    fix_headers_for_tvac(l1_data_filelist, bp_map_outputdir)
    fix_str_for_tvac(l1_data_filelist)

    # Update file list to reflect the new filenames
    input_data_dir = os.path.join(bp_map_outputdir, 'input_l1')
    l1_data_filelist = [os.path.join(input_data_dir, f) for f in os.listdir(input_data_dir) if f.endswith('.fits')]
    
    # Extract visit ID from the first file's primary header
    def get_visitid_from_header(filepath):
        with fits.open(filepath) as hdulist:
            prihdr = hdulist[0].header
            visitid = prihdr.get('VISITID', None)
            if visitid is not None:
                return str(visitid).zfill(19)
            else:
                return "0000000000000000000"  # fallback
    
    visitid = get_visitid_from_header(l1_data_filelist[0])

    ###### Setup necessary calibration files
    # Modify input files to set KGAIN value in their headers
    for file in l1_data_filelist:
        with fits.open(file, mode='update') as hdulist:
            # Modify the extension header to set KGAIN to 8.7
            pri_hdr = hdulist[0].header
            ext_hdr = hdulist[1].header if len(hdulist) > 1 else None
            ext_hdr["KGAINPAR"] = 8.7
            
            # Ensure OBSNUM is set
            if 'OBSNUM' not in pri_hdr:
                pri_hdr['OBSNUM'] = '90500'  # Default OBSNUM value

    # Create a mock dataset object using the input files
    mock_input_dataset = data.Dataset(l1_data_filelist)

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()

    # Load and combine noise maps from various calibration files into a single array
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_current_path) as hdulist:
        dark_current_dat = hdulist[0].data

    # Combine all noise data into one 3D array
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_current_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'],
                              detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img

    # Initialize additional noise map parameters
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    pri_hdr, ext_hdr, _, _ = mocks.create_default_calibration_product_headers()
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0

    # Create a DetectorNoiseMaps object and save it
    noise_maps = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                        input_dataset=mock_input_dataset, err=noise_map_noise,
                                        dq=noise_map_dq, err_hdr=err_hdr)
    # Generate filename with visitid and current time
    current_time = datetime.datetime.now().strftime('%Y%m%dt%H%M%S')
    noise_maps_filename = f"cgi_{visitid}_{current_time}_dnm_cal.fits"
    noise_maps.save(filedir=calibrations_dir, filename=noise_maps_filename)
    this_caldb.create_entry(noise_maps)
    this_caldb.create_entry(noise_maps)

    ## Load and save flat field calibration data
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                          input_dataset=mock_input_dataset)
    # Generate filename with visitid and current time
    flat_filename = f"cgi_{visitid}_{current_time}_flt_cal.fits"
    flat.save(filedir=calibrations_dir, filename=flat_filename)
    this_caldb.create_entry(flat)

    # Load and save bad pixel map data
    with fits.open(bp_ref_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                              input_dataset=mock_input_dataset)
    # Generate filename with visitid and current time
    bp_map_filename = f"cgi_{visitid}_{current_time}_bpm_cal.fits"
    bp_map.save(filedir=calibrations_dir, filename=bp_map_filename)
    this_caldb.create_entry(bp_map)

    # Build and save a synthesized master dark frame
    master_dark = darks.build_synthesized_dark(mock_input_dataset, noise_maps)
    # Generate filename with visitid and current time
    master_dark_filename = f"cgi_{visitid}_{current_time}_drk_cal.fits"
    master_dark.save(filedir=calibrations_dir, filename=master_dark_filename)
    this_caldb.create_entry(master_dark)
    master_dark_ref = master_dark.filepath

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    ####### Run the CorGI DRP walker script
    walker.walk_corgidrp(input_image_filelist, "", bp_map_outputdir, template="bp_map.json")

    # Clean up the calibration database entries
    this_caldb.remove_entry(noise_maps)
    this_caldb.remove_entry(flat)
    this_caldb.remove_entry(master_dark)
    this_caldb.remove_entry(bp_map)

    generated_bp_map_file = os.path.join(bp_map_outputdir, f"cgi_{visitid}_{current_time}_bpm_cal.fits")

    # Load the generated bad pixel map image and master dark reference data
    generated_bp_map_img = data.BadPixelMap(generated_bp_map_file)

    with fits.open(master_dark_ref) as hdulist:
        dark_ref_dat = hdulist[1].data

    with fits.open(dark_current_path) as hdulist:
        naxis1 = hdulist[0].header['NAXIS1']  # X-axis size
        naxis2 = hdulist[0].header['NAXIS2']  # Y-axis size

    with fits.open(bp_ref_path) as hdulist:
        bp_ref_dat = hdulist[0].data

        diff = generated_bp_map_img.data - bp_ref_dat.data

        # Check for differences between the generated and reference bad pixel maps
        assert np.all(np.abs(diff) < 1e-5)

        # Identify and print any bad pixels found
        bad_pixels = generated_bp_map_img.data[generated_bp_map_img.data > 0]
        if bad_pixels.size > 0:
            print(f"Bad pixel values identified: {bad_pixels}")
        else:
            print("No bad pixels identified")

        # Plot noise maps, master dark, and bad pixel maps
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Plot each subplot with relevant data and titles
        im1 = axes[0].imshow(fpn_dat, vmin=-240, vmax=240, cmap="gray")
        axes[0].set_title("FPN noise map")
        axes[0].set_xlim([0, naxis1])
        axes[0].set_ylim([0, naxis2])
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(cic_dat, vmin=0, vmax=0.02, cmap="gray")
        axes[1].set_title("CIC noise map")
        axes[1].set_xlim([0, naxis1])
        axes[1].set_ylim([0, naxis2])
        fig.colorbar(im2, ax=axes[1])

        im3 = axes[2].imshow(dark_current_dat, vmin=0, vmax=0.003, cmap="gray")
        axes[2].set_title("Dark current noise map")
        axes[2].set_xlim([0, naxis1])
        axes[2].set_ylim([0, naxis2])
        fig.colorbar(im3, ax=axes[2])

        im4 = axes[3].imshow(dark_ref_dat, vmin=-240, vmax=240, cmap="gray")
        axes[3].set_title("DRP-produced master dark")
        axes[3].set_xlim([0, naxis1])
        axes[3].set_ylim([0, naxis2])
        fig.colorbar(im4, ax=axes[3])

        im5 = axes[4].imshow(generated_bp_map_img.data, vmin=0, vmax=8, cmap="gray")
        axes[4].set_title("DRP-produced BP map")
        axes[4].set_xlim([0, naxis1])
        axes[4].set_ylim([0, naxis2])
        fig.colorbar(im5, ax=axes[4])

        im6 = axes[5].imshow(bp_ref_dat, vmin=0, vmax=8, cmap="gray")
        axes[5].set_title("Reference BP map")
        axes[5].set_xlim([0, naxis1])
        axes[5].set_ylim([0, naxis2])
        fig.colorbar(im6, ax=axes[5])

        plt.tight_layout()

        # Save the figure to a file
        output_path = os.path.join(bp_map_outputdir, "bp_map_master_dark_test.png")
        plt.savefig(output_path)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

@pytest.mark.e2e
def test_bp_map_simulated_dark_e2e(e2edata_path, e2eoutput_path):
    # Define paths for input L1 data and calibration files
    l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")

    # Create main output directory and simulated dark subfolder
    main_output_dir = os.path.join(e2eoutput_path, "bp_map_cal_e2e")
    bp_map_outputdir = os.path.join(main_output_dir, "bp_map_simulated_dark")
    if not os.path.exists(bp_map_outputdir):
        os.makedirs(bp_map_outputdir)

    calibrations_dir = os.path.join(bp_map_outputdir, "calibrations")
    if not os.path.exists(calibrations_dir):
        os.mkdir(calibrations_dir)

    # Paths to calibration files
    dark_current_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat.fits")

    # Define the list of raw science data files for input, selecting the first two files as examples
    input_image_filelist = []
    l1_data_filelist = []
    for filename in os.listdir(l1_datadir)[:2]:
        l1_data_filelist.append(os.path.join(l1_datadir, filename))
    # l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]]

    # update TVAC headers
    fix_headers_for_tvac(l1_data_filelist, bp_map_outputdir)
    
    # Update file list to reflect the new filenames
    input_data_dir = os.path.join(bp_map_outputdir, 'input_l1')
    l1_data_filelist = [os.path.join(input_data_dir, f) for f in os.listdir(input_data_dir) if f.endswith('.fits')]
    
    # Extract visit ID from the first file's primary header
    def get_visitid_from_header(filepath):
        with fits.open(filepath) as hdulist:
            prihdr = hdulist[0].header
            visitid = prihdr.get('VISITID', None)
            if visitid is not None:
                return str(visitid).zfill(19)
            else:
                return "0000000000000000000"  # fallback
    
    visitid = get_visitid_from_header(l1_data_filelist[0])

    ###### Setup necessary calibration files
    # Modify input files to set KGAIN value in their headers
    for file in l1_data_filelist:
        with fits.open(file, mode='update') as hdulist:
            # Modify the extension header to set KGAIN to 8.7
            pri_hdr = hdulist[0].header
            ext_hdr = hdulist[1].header if len(hdulist) > 1 else None
            ext_hdr["KGAIN"] = 8.7
            
            # Ensure OBSNUM is set
            if 'OBSNUM' not in pri_hdr:
                pri_hdr['OBSNUM'] = '90500'  # Default OBSNUM value

    # Create a mock dataset object using the input files
    mock_input_dataset = data.Dataset(l1_data_filelist)

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()

    # Generate filename variables for this test
    current_time = datetime.datetime.now().strftime('%Y%m%dt%H%M%S')

    ## Load and save flat field calibration data
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    pri_hdr, ext_hdr, _, _ = mocks.create_default_calibration_product_headers()
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                          input_dataset=mock_input_dataset)
    # Generate filename with visitid and current time
    flat_filename = f"cgi_{visitid}_{current_time}_flt_cal.fits"
    flat.save(filedir=calibrations_dir, filename=flat_filename)
    this_caldb.create_entry(flat)

    # Create a simulated dark frame with random hot pixels for testing
    with fits.open(dark_current_path) as hdulist:
        naxis1 = hdulist[0].header['NAXIS1']  # X-axis size
        naxis2 = hdulist[0].header['NAXIS2']  # Y-axis size

    # Initialize a simple dark data array and simulate hot pixel clusters
    simple_dark_data = np.zeros_like(flat.data)
    cluster_size = 10
    hot_pixel_value = 8

    for _ in range(10):
        cluster_center = (np.random.randint(0, naxis2), np.random.randint(0, naxis1))
        for _ in range(cluster_size):
            offset_x = np.random.randint(-5, 6)
            offset_y = np.random.randint(-5, 6)
            x = cluster_center[0] + offset_x
            y = cluster_center[1] + offset_y

            # Ensure the pixel is within image bounds
            if 0 <= x < naxis2 and 0 <= y < naxis1:
                simple_dark_data[x, y] = hot_pixel_value

    # Create a dark object and save it
    master_dark = data.Dark(simple_dark_data, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                            input_dataset=mock_input_dataset)

    # Generate filename with visitid and current time
    master_dark_filename = f"cgi_{visitid}_{current_time}_drk_cal.fits"
    master_dark.save(filedir=calibrations_dir, filename=master_dark_filename)
    this_caldb.create_entry(master_dark)
    master_dark_ref = master_dark.filepath

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    ####### Run the CorGI DRP walker script
    walker.walk_corgidrp(input_image_filelist, "", bp_map_outputdir, template="bp_map.json")

    # Clean up the calibration database entries
    this_caldb.remove_entry(flat)
    this_caldb.remove_entry(master_dark)

    generated_bp_map_file = os.path.join(bp_map_outputdir, f"cgi_{visitid}_{current_time}_bpm_cal.fits")

    # Load the generated bad pixel map image and reference dark data
    generated_bp_map_img = data.BadPixelMap(generated_bp_map_file)

    with fits.open(master_dark_ref) as hdulist:
        dark_ref_dat = hdulist[1].data
        diff = generated_bp_map_img.data - dark_ref_dat.data

        # Check for differences between the generated bad pixel map and the reference dark data
        assert np.all(np.abs(diff) < 1e-5)

        # Plotting the results: generated bad pixel map, reference dark file, and their difference
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # First subplot: Generated bad pixel map
        im1 = axes[0].imshow(generated_bp_map_img.data, cmap="viridis",
                                norm=colors.PowerNorm(gamma=0.1, vmin=0, vmax=1))
        axes[0].set_title("CorGI DRP generated bad pixel map")
        axes[0].set_xlim([0, naxis1])
        axes[0].set_ylim([0, naxis2])
        fig.colorbar(im1, ax=axes[0])

        # Second subplot: Reference dark file
        im2 = axes[1].imshow(dark_ref_dat, cmap="viridis",
                                norm=colors.PowerNorm(gamma=0.1, vmin=0, vmax=1))
        axes[1].set_title("Referenced dark file")
        axes[1].set_xlim([0, naxis1])
        axes[1].set_ylim([0, naxis2])
        fig.colorbar(im2, ax=axes[1])

        # Third subplot: Difference between generated BP map and reference
        im3 = axes[2].imshow(diff, vmin=-0.01, vmax=0.01, cmap="inferno")
        axes[2].set_title("Difference")
        axes[2].set_xlim([0, naxis1])
        axes[2].set_ylim([0, naxis2])
        fig.colorbar(im3, ax=axes[2])

        plt.tight_layout()

        # Save the figure to a file
        output_path = os.path.join(bp_map_outputdir, "bp_map_simulated_dark_test.png")
        plt.savefig(output_path)
    
    # Skip removal of generated_bp_map_img since it doesn't have a proper filepath
    # this_caldb.remove_entry(generated_bp_map_img)

if __name__ == "__main__":
    # Set default paths and parse command-line arguments
    # e2edata_dir = "/home/jwang/Desktop/CGI_TVAC_Data"
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    outputdir = thisfile_dir

    # Argument parser setup
    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()

    # Assign parsed arguments to variables
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir

    # Run the main functions with parsed arguments
    test_bp_map_master_dark_e2e(e2edata_dir, outputdir)
    test_bp_map_simulated_dark_e2e(e2edata_dir, outputdir)
    