import argparse
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.io import fits
from corgidrp import data
from corgidrp import caldb
from corgidrp import detector
from corgidrp import darks
from corgidrp import walker

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_header(tvacdata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
        # figure out paths, assuming everything is located in the same relative location
    l1_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L1")
    processed_cal_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals")

    # make output directory if needed
    bp_map_outputdir = os.path.join(e2eoutput_path, "bp_map_output")
    if not os.path.exists(bp_map_outputdir):
        os.mkdir(bp_map_outputdir)

    # assume all cals are in the same directory
    dark_current_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat_ones.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_ref_path = os.path.join(processed_cal_path, "fixed_bp_zeros.fits")
    noise_maps_filelist = [dark_current_path, fpn_path, cic_path]

    # define the raw science data to use as mock frames where necessary
    input_image_filelist = []
    l1_data_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90499, 90500]] # just grab the first two files

    ###### Setup necessary calibration files
    # Need to change this to use appropriate values for KGAIN, for now setting them all the same.
    # Master dark calculation checks to make sure KGAIN is the the same for every input noise map.
    for file in l1_data_filelist:
        with fits.open(file, mode='update') as hdulist:
            # Primary header from HDU[0]
            pri_hdr = hdulist[0].header
            # Extension header from HDU[1] if it exists
            ext_hdr = hdulist[1].header if len(hdulist) > 1 else None
            ext_hdr["KGAIN"] = 8.7
    mock_input_dataset = data.Dataset(l1_data_filelist)

    this_caldb = caldb.CalDB() # connection to cal DB

    # NoiseMap
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_current_path) as hdulist:
        dark_current_dat = hdulist[0].data
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_current_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected EM electrons'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_maps = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=mock_input_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_maps.save(filedir=bp_map_outputdir, filename="mock_detnoisemaps.fits")
    this_caldb.create_entry(noise_maps)

    ## Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    flat.save(filedir=bp_map_outputdir, filename="mock_flat.fits")
    this_caldb.create_entry(flat)

    # bad pixel map
    with fits.open(bp_ref_path) as hdulist:
        bp_dat = hdulist[0].data
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    bp_map.save(filedir=bp_map_outputdir, filename="mock_bpmap.fits")
    this_caldb.create_entry(bp_map)

    # Master dark
    master_dark = darks.build_synthesized_dark(mock_input_dataset, noise_maps)
    master_dark.save(filedir=bp_map_outputdir, filename="dark_mock.fits")
    this_caldb.create_entry(master_dark)
    master_dark_ref = master_dark.filepath

    ####### Run the walker on some test_data
    ####### Run the walker
    walker.walk_corgidrp(input_image_filelist, "", bp_map_outputdir, template="bp_map.json")

    # Clean up by removing entry
    this_caldb.remove_entry(noise_maps)
    this_caldb.remove_entry(flat)
    this_caldb.remove_entry(master_dark)

    generated_bp_map_file = os.path.join(bp_map_outputdir, "dark_mock_bad_pixel_map.fits")

    # Test generated BP map against TVAC data or simple dark
    generated_bp_map_img = data.Image(generated_bp_map_file)

    with fits.open(master_dark_ref) as hdulist:
        dark_ref_dat = hdulist[1].data
    
    with fits.open(dark_current_path) as hdulist:
        naxis1 = hdulist[0].header['NAXIS1']  # X-axis size
        naxis2 = hdulist[0].header['NAXIS2']  # Y-axis size

    with fits.open(bp_ref_path) as hdulist:
        bp_ref_dat = hdulist[0].data

        diff = generated_bp_map_img.data - bp_ref_dat.data

        # Assert for values greater than 1e-5
        assert np.all(np.abs(diff) < 1e-5)

        # Check for bad pixels
        bad_pixels = generated_bp_map_img.data[generated_bp_map_img.data > 0]
        if bad_pixels.size > 0:
            print(f"Bad pixel values identified: {bad_pixels}")
        else:
            print("No bad pixels identified")
        
        # Plotting
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

        # Sixth subplot (bottom-right corner)
        im5 = axes[5].imshow(bp_ref_dat, vmin=0, vmax=8, cmap="gray")
        axes[5].set_title("Reference BP map")
        axes[5].set_xlim([0, naxis1])
        axes[5].set_ylim([0, naxis2])
        fig.colorbar(im5, ax=axes[5])

        plt.tight_layout()

        plt.show()

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    #tvacdata_dir = "/Users/jmilton/Documents/CGI/CGI_Data"
    tvacdata_dir = "/Users/jmilton/Documents/CGI/CGI_TVAC_Data"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    test_header(tvacdata_dir, outputdir)
