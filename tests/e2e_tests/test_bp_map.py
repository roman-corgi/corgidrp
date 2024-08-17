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
import corgidrp.darks as darks
import corgidrp.bad_pixel_calibration

all_steps = {
    "create_bad_pixel_map" : corgidrp.bad_pixel_calibration.create_bad_pixel_map,
    "detect_hot_warm_pixels_from_dark" : corgidrp.bad_pixel_calibration.detect_hot_warm_pixels_from_dark,
    "detect_dead_pixels_from_flat" : corgidrp.bad_pixel_calibration.detect_dead_pixels_from_flat
}

thisfile_dir = os.path.dirname(__file__) # this file's folder

def test_bp_map(tvacdata_path, output_path):
    # figure out paths, assuming everything is located in the same relative location
    inputs_dir = os.path.join(tvacdata_path, "Inputs")
    cals_dir = os.path.join(tvacdata_path, "Cals")

    # make output directory if needed
    bp_map_outputdir = os.path.join(output_path, "bp_map_output")
    if not os.path.exists(bp_map_outputdir):
        os.mkdir(bp_map_outputdir)
    
    dark_filename = "dark_current_20240322.fits"
    fpn_filename = "fpn_20240322.fits"
    cic_filename = "cic_20240322.fits"
    flat_filename = "flat_ones.fits"

    # Construct the full paths to the noise map files
    noise_maps_filelist = [
        os.path.join(cals_dir, dark_filename),
        os.path.join(cals_dir, fpn_filename),
        os.path.join(cals_dir, cic_filename)
    ]

    flats_filelist = [os.path.join(cals_dir, flat_filename)]
    input_image_filelist = []


    # Noise map fits files are missing headers for some reason
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__

    # Print the HDU lists of the FITS files
    for file in noise_maps_filelist:
        print(f"Inspecting file: {file}")
        with fits.open(file, mode='update') as hdulist:
            # Apply the default headers
            hdulist[0].header.update(pri_hdr)
            
            # Add extension header if there's more than one HDU
            if len(hdulist) > 1:
                hdulist[1].header.update(ext_hdr)
            else:
                # If there is no extension, we can add one
                new_hdu = fits.ImageHDU(header=ext_hdr)
                hdulist.append(new_hdu)
            
            # Save changes to the FITS file
            hdulist.flush()
    
    for file in noise_maps_filelist:
        with fits.open(file) as hdulist:
            dataCheck = hdulist[0].data
            if dataCheck is None:
                print(f"Warning: {file} contains no data in the primary HDU.")
            else:
                print(f"{file} data shape: {dataCheck.shape}")
            print("time for all checks. hdulist[0]: ", hdulist[0].header)
            print("time for all checks. hdulist[1]: ", hdulist[1].header)
            #print("time for all checks. hdulist[0]: ", hdulist[2].header)
    
    
    noise_maps_dataset = data.Dataset(noise_maps_filelist)
    flats_dataset = data.Dataset(flats_filelist)

    this_caldb = caldb.CalDB() # connection to cal DB
    

    # assume all cals are in the same directory
    dark_path = os.path.join(cals_dir, dark_filename)
    flat_path = os.path.join(cals_dir, flat_filename)
    fpn_path = os.path.join(cals_dir, fpn_filename)
    cic_path = os.path.join(cals_dir, cic_filename)
    bp_path = os.path.join(cals_dir, "bad_pix.fits")

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
    err_hdr['BUNIT'] = 'detected EM electrons'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                    input_dataset=noise_maps_dataset, err=noise_map_noise,
                                    dq = noise_map_dq, err_hdr=err_hdr)
    noise_map.save(filedir=bp_map_outputdir, filename="tvac_detnoisemaps.fits")
    this_caldb.create_entry(noise_map)
    noise_map_ref = noise_map.filepath

    # Master dark
    master_dark = darks.build_synthesized_dark(noise_maps_dataset, noise_map)
    master_dark.save(filedir=bp_map_outputdir, filename="master_dark.fits")
    this_caldb.create_entry(master_dark)
    master_dark_ref = master_dark.filepath

    ## Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=flats_dataset)
    flat.save(filedir=bp_map_outputdir, filename="ones_flat.fits")
    this_caldb.create_entry(flat)
    flat_ref = flat.filepath

 
    ####### Run the walker on some test_data
    print(type(noise_map), type(flat))
    walker.walk_corgidrp(input_image_filelist, master_dark_ref, flat_ref, "", bp_map_outputdir)


    # clean up by removing entry
    this_caldb.remove_entry(noise_map)
    this_caldb.remove_entry(flat)


if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    tvacdata_dir = "/Users/jmilton/Documents/CGI/CGI_Data"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the bad pixel map test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    test_bp_map(tvacdata_dir, outputdir)
