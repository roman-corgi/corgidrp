import os
import glob
import numpy as np
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.detector as detector
from corgidrp.bad_pixel_calibration import create_bad_pixel_map
from corgidrp.darks import build_trad_dark

np.random.seed(456)
def test_badpixelmap(): 
    '''

    Tests the creation of badpixelmaps.

    Create master darks and master flats, inject some hot and cold pixels
    to create a master bad pixel map. 

    '''

    ###### create simulated dark data
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    
    mocks.create_dark_calib_files(filedir = datadir)

    ####### test data architecture
    dark_filenames = glob.glob(os.path.join(datadir, "simcal_dark*.fits"))

    dark_dataset = data.Dataset(dark_filenames)

    ###### create dark
    # use default parameters
    detector_params = data.DetectorParams({})
    dark_frame = build_trad_dark(dark_dataset,detector_params)
    
    ###### make some hot pixels:
    # Add some hot pixels
    col_hot_pixels_test = [12, 123, 234, 456, 678, 890]
    row_hot_pixels_test = [546, 789, 123, 43, 547, 675]

    for i_col in col_hot_pixels_test:
        for i_row in row_hot_pixels_test:
            dark_frame.data[i_col, i_row] = 300

    ###### create simulated dark data
    mocks.create_simflat_dataset(filedir=datadir)
    
    # simulated images to be checked in flat division
    simdata_filenames=glob.glob(os.path.join(datadir, "sim_flat*.fits"))
    simflat_dataset=data.Dataset(simdata_filenames)
     
    # creat one dummy flat field perform flat division
    mocks.create_flatfield_dummy(filedir=datadir)
	#test data architecture
    flat_filenames = glob.glob(os.path.join(datadir, "flat_field*.fits"))
    flat_dataset = data.Dataset(flat_filenames)
    
    ###### create flatfield
    flat_frame = detector.create_flatfield(flat_dataset)

    ###### make some hot pixels:
    col_dead_pixel_test=[12, 120, 234, 450, 678, 990]
    row_dead_pixel_test=[546, 89, 123, 243, 447, 675]

    for i_col in col_dead_pixel_test:
        for i_row in row_dead_pixel_test:
            flat_frame.data[i_col, i_row] = 0.3

    ###### make the badpixel map (input the flat_dataset just as a dummy):
    badpixelmap = create_bad_pixel_map(flat_dataset, dark_frame,flat_frame, dthresh=6)
    # Use np.unpackbits to unpack the bits - big endien
    badpixelmap_bits = np.unpackbits(badpixelmap.data[:, :, np.newaxis], axis=2)

    # Checking that everywhere there's a badpixel is in one of the two lists
    bp_locations = np.argwhere(badpixelmap.data)
    for ii in bp_locations[:,0]:
        assert ii in col_hot_pixels_test or ii in col_dead_pixel_test
    for jj in bp_locations[:,1]:
        assert jj in row_hot_pixels_test or jj in row_dead_pixel_test

    # Checking that hot pixels are at the expected locations - bit #4
    hot_pixel_locations = np.where(badpixelmap_bits[:,:,-4])
    for ii in hot_pixel_locations[0]:
        assert ii in col_hot_pixels_test
    for jj in hot_pixel_locations[1]:
        assert jj in row_hot_pixels_test

    # Checking that CR are at the expected locations - bit #3
    dead_pixel_locations = np.where(badpixelmap_bits[:,:,-3])
    for ii in dead_pixel_locations[0]:
        assert ii in col_dead_pixel_test
    for jj in dead_pixel_locations[1]:
        assert jj in row_dead_pixel_test 

if __name__ == "__main__":
    test_badpixelmap()