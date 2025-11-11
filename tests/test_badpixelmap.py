import os
import numpy as np
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.detector as detector
import corgidrp.flat as flat
from corgidrp.bad_pixel_calibration import create_bad_pixel_map
from corgidrp.darks import build_trad_dark
import re

np.random.seed(456)

# Get the flag to bit map and flag to value map
FLAG_TO_BIT_MAP = data.get_flag_to_bit_map()
FLAG_TO_VALUE_MAP = data.get_flag_to_value_map()

def generate_badpixel_map(datadir=None, dthresh=6):
    """
    Create simulated dark and flat calibration data, inject hot and dead pixels,
    and then generate a bad pixel map. Returns the bad pixel map,
    the dark frame, and the flat field frame.
    
    Args:
        datadir (str, optional): Directory to store simulated data. If None,
                                 a "simdata" folder in the current directory is used.
        dthresh (int, optional): Threshold parameter to pass into create_bad_pixel_map.
                                 Defaults to 6.
    
    Returns:
        tuple: (badpixelmap, dark_frame, flat_frame)
    """
    # Set the data directory.
    if datadir is None:
        datadir = os.path.join(os.path.dirname(__file__), "simdata")
    os.makedirs(datadir, exist_ok=True)
    
    # --- Create and load dark data ---
    dark_dataset = mocks.create_dark_calib_files(filedir=datadir)
    
    # Create the dark frame using default detector parameters.
    detector_params = data.DetectorParams({})
    dark_frame = build_trad_dark(dark_dataset, detector_params)
    
    # Inject "hot" pixels into the dark frame.
    col_hot_pixels = [12, 123, 234, 456, 678, 890]
    row_hot_pixels = [546, 789, 123, 43, 547, 675]
    for i_col in col_hot_pixels:
        for i_row in row_hot_pixels:
            dark_frame.data[i_col, i_row] = 300

    # --- Create and load flat data ---
    simflat_dataset = mocks.create_simflat_dataset(filedir=datadir)
    
    # Create a dummy flat field (flat division) image.
    flat_dataset = mocks.create_flatfield_dummy(filedir=datadir)
    flat_frame = flat.create_flatfield(flat_dataset)
    
    # Inject "dead" pixels into the flat field image.
    col_dead_pixels = [12, 120, 234, 450, 678, 990]
    row_dead_pixels = [546, 89, 123, 243, 447, 675]
    for i_col in col_dead_pixels:
        for i_row in row_dead_pixels:
            flat_frame.data[i_col, i_row] = 0.3

    # --- Create the bad pixel map ---
    bpm = create_bad_pixel_map(flat_dataset, dark_frame, flat_frame, dthresh=dthresh)
    
    return bpm, dark_frame, flat_frame


def test_badpixelmap(): 
    '''

    Tests the creation of badpixelmaps.

    Create master darks and master flats, inject some hot and cold pixels
    to create a master bad pixel map. 

    '''

    ###### make the badpixel map (input the flat_dataset just as a dummy):
    badpixelmap, dark_frame, flat_frame = generate_badpixel_map()

    # # Use np.unpackbits to unpack the bits - big endien demical to binary
    badpixelmap_bits = data.unpackbits_64uint(badpixelmap.data[:, :, np.newaxis], axis=2)  # unit64 to binary
    badpixelmap_repacked = data.packbits_64uint(badpixelmap_bits, axis=2).reshape(badpixelmap.data.shape)
    assert np.array_equal(badpixelmap_repacked, badpixelmap.data) # check if the repacked data is the same as the original

    # Checking that everywhere there's a badpixel is in one of the two lists
    bp_locations = np.argwhere(badpixelmap.data)
    col_hot = [12, 123, 234, 456, 678, 890]
    col_dead = [12, 120, 234, 450, 678, 990]
    row_hot = [546, 789, 123, 43, 547, 675]
    row_dead = [546, 89, 123, 243, 447, 675]
    
    for (i_col, i_row) in bp_locations:
        assert (i_col in col_hot or i_col in col_dead), f"Column {i_col} not expected."
        assert (i_row in row_hot or i_row in row_dead), f"Row {i_row} not expected."

    # Checking that hot pixels are at the expected locations - bit #3
    hot_pixel_bit_position = 63 - FLAG_TO_BIT_MAP["hot_pixel"]
    hot_pixel_locations = np.where(badpixelmap_bits[:,:,hot_pixel_bit_position])
    for ii in hot_pixel_locations[0]:
        assert ii in col_hot, f"Hot pixel at column {ii} not expected."
    for jj in hot_pixel_locations[1]:
        assert jj in row_hot, f"Hot pixel at row {jj} not expected."

    # Checking that CR are at the expected locations - bit #2
    dead_pixel_bit_position = 63 - FLAG_TO_BIT_MAP["bad_pixel"]
    dead_pixel_locations = np.where(badpixelmap_bits[:,:,dead_pixel_bit_position])
    for ii in dead_pixel_locations[0]:
        assert ii in col_dead, f"Dead pixel at column {ii} not expected." 
    for jj in dead_pixel_locations[1]:
       assert jj in row_dead, f"Dead pixel at row {jj} not expected."


def test_packing_unpacking_uint64():
    '''
    Test the packing and unpacking of uint64 data
    ''' 

    # Checking whether we can assign demical higher than 255 to the badpixelmap to test our function for bit unpacking and packing
    packed = np.zeros((3, 3), dtype='>u8')

    # Get the bit position for 'TBD'
    bit_position = FLAG_TO_BIT_MAP["TBD"]
    flag_value = FLAG_TO_VALUE_MAP["TBD"]

    packed[0, 0] = flag_value

    unpacked_bits = data.unpackbits_64uint(packed[:, :, np.newaxis], axis=2)

    # Compute the expected unpacked index for 'big' endian
    expected_index = 63 - bit_position

    # Check the expected bit index is set
    assert unpacked_bits[0, 0, expected_index] == 1, f"Bit {bit_position} should be at index {expected_index}"

    # Check that only one bit is set
    assert np.sum(unpacked_bits[0, 0]) == 1, "Only one bit should be set"


def test_output_filename_convention():
    print("**Testing output filename naming conventions**")

    bp_fake_data = np.array([
        [0,  1,  0],   
        [1,  0, 0],  
        [0, 0, 0]
    ], dtype=float)

    # Create a fake input dataset to set the filename
    input_prihdr, input_exthdr, errhdr, dqhdr, biashdr = mocks.create_default_L2b_headers()
    fake_input_image = data.Image(bp_fake_data, pri_hdr=input_prihdr, ext_hdr=input_exthdr)
    fake_input_image.filename = f"cgi_{input_prihdr['VISITID']}_{data.format_ftimeutc(input_exthdr['FTIMEUTC'])}_l2b.fits"
    fake_input_dataset = data.Dataset(frames_or_filepaths=[fake_input_image, fake_input_image])

    bpcal_prihdr, bpcal_exthdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    
    # Create test output directory
    test_output_dir = os.path.join(os.path.dirname(__file__), "testcalib")
    os.makedirs(test_output_dir, exist_ok=True)
    
    badpixelmap = data.BadPixelMap(bp_fake_data, pri_hdr=bpcal_prihdr, ext_hdr=bpcal_exthdr, input_dataset=fake_input_dataset)
    badpixelmap.save(filedir=test_output_dir)

     # Construct the expected filename from the last input dataset filename.
    expected_filename = re.sub('_l[0-9].', '_bpm_cal', fake_input_dataset[-1].filename)
    full_expected_path = os.path.join(test_output_dir, expected_filename)
    
    assert os.path.exists(full_expected_path), (
        f"Expected file {expected_filename} not found in {test_output_dir}."
    )
    print("The bad pixel map calibration product file exists and meets the expected naming convention.")


if __name__ == "__main__":
    test_badpixelmap()
    test_packing_unpacking_uint64()
    test_output_filename_convention()