import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.io import fits
from corgidrp import data
from corgidrp import caldb
from corgidrp import detector
from corgidrp import darks
from corgidrp import walker

thisfile_dir = os.path.dirname(__file__)  # this file's folder

def test_bp_map(tvacdata_path, output_path, use_master_dark):
    """
    Generate and test the bad pixel map using TVAC data.

    Args:
        tvacdata_path (str): Path to the TVAC data directory.
        output_path (str): Directory to save the output files.
        use_master_dark (bool): Whether to use a master dark or a simple dark.

    Returns:
        None
    """
    cals_dir = os.path.join(tvacdata_path, "Cals")

    # Make output directory if needed
    bp_map_outputdir = os.path.join(output_path, "bp_map_output")
    if not os.path.exists(bp_map_outputdir):
        os.mkdir(bp_map_outputdir)
    dark_current_filename = "dark_current_20240322.fits"
    fpn_filename = "fpn_20240322.fits"
    cic_filename = "cic_20240322.fits"
    flat_filename = "flat_ones.fits"

    # Construct the full paths to the noise map files
    noise_maps_filelist = [
        os.path.join(cals_dir, dark_current_filename),
        os.path.join(cals_dir, fpn_filename),
        os.path.join(cals_dir, cic_filename)
    ]

    flats_filelist = [os.path.join(cals_dir, flat_filename)]
    input_image_filelist = []

    ## Handle header issues with TVAC files
    # Need to change this to use appropriate values for KGAIN, for now setting them all the same.
    # Master dark calculation checks to make sure KGAIN is the the same for every input noise map.
    for file in noise_maps_filelist:
        with fits.open(file, mode='update') as hdulist:
            # Primary header from HDU[0]
            pri_hdr = hdulist[0].header
            # Extension header from HDU[1] if it exists
            ext_hdr = hdulist[1].header if len(hdulist) > 1 else None
            ext_hdr["KGAIN"] = 0

    noise_maps_dataset = data.Dataset(noise_maps_filelist)
    flats_dataset = data.Dataset(flats_filelist)

    this_caldb = caldb.CalDB()  # Connection to cal DB

    # assume all cals are in the same directory
    dark_current_path = os.path.join(cals_dir, dark_current_filename)
    flat_path = os.path.join(cals_dir, flat_filename)
    fpn_path = os.path.join(cals_dir, fpn_filename)
    cic_path = os.path.join(cals_dir, cic_filename)

    # NoiseMap
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_current_path) as hdulist:
        dark_current_dat = hdulist[0].data
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_current_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'],
                              detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected EM electrons'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=noise_maps_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    # card is too long, comment will be truncated warning
    noise_map.save(filedir=bp_map_outputdir, filename="tvac_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)

    ## Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=flats_dataset)
    flat.save(filedir=bp_map_outputdir, filename="tvac_ones_flat.fits")
    this_caldb.create_entry(flat)
    with fits.open(dark_current_path) as hdulist:
        naxis1 = hdulist[0].header['NAXIS1']  # X-axis size
        naxis2 = hdulist[0].header['NAXIS2']  # Y-axis size

    # Master dark
    if use_master_dark:
        master_dark = darks.build_synthesized_dark(noise_maps_dataset, noise_map)
        master_dark.save(filedir=bp_map_outputdir, filename="dark_mock.fits")
        this_caldb.create_entry(master_dark)
        master_dark_ref = master_dark.filepath
    else:
        # Simple dark for comparison
        simple_dark_data = np.zeros_like(flat.data)
        cluster_center = (naxis2 // 2, naxis1 // 2)
        cluster_size = 10
        hot_pixel_value = 1

        for _ in range(10):
            # Randomly choose a center for each cluster
            cluster_center = (np.random.randint(0, naxis2), np.random.randint(0, naxis1))

            for _ in range(cluster_size):
                # Offset the position from the cluster center
                offset_x = np.random.randint(-5, 6)
                offset_y = np.random.randint(-5, 6)
                x = cluster_center[0] + offset_x
                y = cluster_center[1] + offset_y

                # Ensure the pixel is within image bounds
                if 0 <= x < naxis2 and 0 <= y < naxis1:
                    simple_dark_data[x, y] = hot_pixel_value
        master_dark = data.Dark(simple_dark_data, pri_hdr=pri_hdr, ext_hdr=ext_hdr)
        master_dark.save(filedir=bp_map_outputdir, filename = "dark_mock.fits")
        this_caldb.create_entry(master_dark)
        master_dark_ref = master_dark.filepath

    ####### Run the walker
    walker.walk_corgidrp(input_image_filelist, "", bp_map_outputdir, template="bp_map.json")

    # Clean up by removing entry
    this_caldb.remove_entry(noise_map)
    this_caldb.remove_entry(flat)
    this_caldb.remove_entry(master_dark)

    generated_bp_map_file = os.path.join(bp_map_outputdir, "dark_mock_bad_pixel_map.fits")

    # Test generated BP map against TVAC data or simple dark
    generated_bp_map_img = data.Image(generated_bp_map_file)

    if use_master_dark:
        with fits.open(master_dark_ref) as hdulist:
            dark_ref_dat = hdulist[1].data

            # Check for bad pixels
            bad_pixels = generated_bp_map_img.data[generated_bp_map_img.data > 0]
            if bad_pixels.size > 0:
                print(f"Bad pixel values identified: {bad_pixels}")
            else:
                print("No bad pixels identified")
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            # Flatten the axes array for easier indexing
            axes = axes.flatten()

            # First subplot
            im1 = axes[0].imshow(fpn_dat, vmin=-240, vmax=240, cmap="gray")
            axes[0].set_title("FPN noise map")
            axes[0].set_xlim([0, naxis1])
            axes[0].set_ylim([0, naxis2])
            fig.colorbar(im1, ax=axes[0])

            # Second subplot
            im2 = axes[1].imshow(cic_dat, vmin=0, vmax=0.02, cmap="gray")
            axes[1].set_title("CIC noise map")
            axes[1].set_xlim([0, naxis1])
            axes[1].set_ylim([0, naxis2])
            fig.colorbar(im2, ax=axes[1])

            # Third subplot
            im3 = axes[2].imshow(dark_current_dat, vmin=0, vmax=0.003, cmap="gray")
            axes[2].set_title("Dark current noise map")
            axes[2].set_xlim([0, naxis1])
            axes[2].set_ylim([0, naxis2])
            fig.colorbar(im3, ax=axes[2])

            # Fourth subplot (on the second row, left)
            im4 = axes[3].imshow(dark_ref_dat, vmin=-240, vmax=240, cmap="gray")
            axes[3].set_title("DRP-produced master dark")
            axes[3].set_xlim([0, naxis1])
            axes[3].set_ylim([0, naxis2])
            fig.colorbar(im4, ax=axes[3])

            # Fifth subplot (on the second row, right)
            im5 = axes[4].imshow(generated_bp_map_img.data, vmin=0, vmax=8, cmap="gray")
            axes[4].set_title("DRP-produced BP map")
            axes[4].set_xlim([0, naxis1])
            axes[4].set_ylim([0, naxis2])
            fig.colorbar(im5, ax=axes[4])

            # Hide the unused subplot (bottom-right corner)
            axes[5].axis('off')

            plt.tight_layout()
            plt.show()
    else:
        with fits.open(master_dark_ref) as hdulist:
            dark_ref_dat = hdulist[1].data
            diff = generated_bp_map_img.data - dark_ref_dat.data

            #assert np.all(np.abs(diff) < 1e-5)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a 1x3 grid of subplots

            # First subplot
            im1 = axes[0].imshow(generated_bp_map_img.data, cmap="viridis",
                                 norm=colors.PowerNorm(gamma=0.1, vmin=0, vmax=1))
            axes[0].set_title("CorGI DRP generated bad pixel map")
            axes[0].set_xlim([0, naxis1])
            axes[0].set_ylim([0, naxis2])
            fig.colorbar(im1, ax=axes[0])

            # Second subplot
            im2 = axes[1].imshow(dark_ref_dat, cmap="viridis",
                                 norm=colors.PowerNorm(gamma=0.1, vmin=0, vmax=1))
            axes[1].set_title("Referenced dark file")
            axes[1].set_xlim([0, naxis1])
            axes[1].set_ylim([0, naxis2])
            fig.colorbar(im2, ax=axes[1])

            # Third subplot
            im3 = axes[2].imshow(diff, vmin=-0.01, vmax=0.01, cmap="inferno")
            axes[2].set_title("Difference")
            axes[2].set_xlim([0, naxis1])
            axes[2].set_ylim([0, naxis2])
            fig.colorbar(im3, ax=axes[2])

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Use arguments to run the test.
    TVACDATA_DIR = "/Users/jmilton/Documents/CGI/CGI_Data"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="Run the bad pixel map test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=TVACDATA_DIR,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="Directory to write results to [%(default)s]")
    ap.add_argument("-m", "--use_master_dark", action='store_true',
                    help="Use master dark instead of simple dark")
    args = ap.parse_args()

    test_bp_map(args.TVACDATA_DIR, args.outputdir, args.use_master_dark)
