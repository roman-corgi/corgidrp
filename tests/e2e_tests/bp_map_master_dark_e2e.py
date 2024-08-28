import argparse
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.io import fits
from corgidrp import data
from corgidrp import caldb
from corgidrp import walker

# Get the directory of the current script file
thisfile_dir = os.path.dirname(__file__)

@pytest.mark.e2e
def bp_map_simulated_dark_e2e(tvacdata_path, e2eoutput_path):
    """
    Performs an end-to-end test to generate a bad pixel map using simulated dark data.
    
    This function sets up a simulated master dark frame with random hot pixels. 
    It then generates a bad pixel map and compares it against the simulated dark data, 
    checks for discrepancies, and visualizes the results. 
    The generated figures are saved to the specified output directory.
    """

    # Define paths for input L1 data and calibration files
    l1_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L1")
    processed_cal_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals")

    # Create output directory for bad pixel map results if it doesn't exist
    bp_map_outputdir = os.path.join(e2eoutput_path, "bp_map_output")
    if not os.path.exists(bp_map_outputdir):
        os.mkdir(bp_map_outputdir)

    # Paths to calibration files
    dark_current_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat_ones.fits")

    # Define the list of raw science data files for input, selecting the first two files as examples
    input_image_filelist = []
    l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]]

    ###### Setup necessary calibration files
    # Modify input files to set KGAIN value in their headers
    for file in l1_data_filelist:
        with fits.open(file, mode='update') as hdulist:
            # Modify the extension header to set KGAIN to 8.7
            pri_hdr = hdulist[0].header
            ext_hdr = hdulist[1].header if len(hdulist) > 1 else None
            ext_hdr["KGAIN"] = 8.7

    # Create a mock dataset object using the input files
    mock_input_dataset = data.Dataset(l1_data_filelist)

    # Initialize a connection to the calibration database
    this_caldb = caldb.CalDB()

    ## Load and save flat field calibration data
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                          input_dataset=mock_input_dataset)
    flat.save(filedir=bp_map_outputdir, filename="mock_flat.fits")
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
    master_dark = data.Dark(simple_dark_data, pri_hdr=pri_hdr, ext_hdr=ext_hdr)
    master_dark.save(filedir=bp_map_outputdir, filename="dark_mock.fits")
    this_caldb.create_entry(master_dark)
    master_dark_ref = master_dark.filepath

    ####### Run the CorGI DRP walker script
    walker.walk_corgidrp(input_image_filelist, "", bp_map_outputdir, template="bp_map.json")

    # Clean up the calibration database entries
    this_caldb.remove_entry(flat)
    this_caldb.remove_entry(master_dark)

    generated_bp_map_file = os.path.join(bp_map_outputdir, "dark_mock_bad_pixel_map.fits")

    # Load the generated bad pixel map image and reference dark data
    generated_bp_map_img = data.Image(generated_bp_map_file)

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

if __name__ == "__main__":
    # Set default paths and parse command-line arguments
    tvacdata_dir = "/Users/jmilton/Documents/CGI/CGI_TVAC_Data"
    outputdir = thisfile_dir

    # Argument parser setup
    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()

    # Assign parsed arguments to variables
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir

    # Run the main function with parsed arguments
    bp_map_simulated_dark_e2e(tvacdata_dir, outputdir)
