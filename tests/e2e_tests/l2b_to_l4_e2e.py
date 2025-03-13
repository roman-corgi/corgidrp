import argparse
import os
import numpy as np
from astropy.io import fits
from corgidrp.data import Image
from corgidrp.mocks import create_default_L2b_headers, create_synthetic_satellite_spot_image

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_l2b_to_l4(test_data_path, e2eoutput_path):
    '''
    Data needed: 
        - Coronagraphic dataset
        - Reference star dataset
        - Satellite spot dataset
    
    Calibrations needed: 
        - AstrometricCalibration
    '''

    #TODO: Generate mock astrometric calibration data. 

    # Read in the os11 data and format it into a dataset
    # Reading things in based on hlc_os11_example.py

    #Read in the PSFs
    input_file = 'hlc_os11_frames_with_planets.fits'
    input_hdul = fits.open(os.path.join(test_data_path,input_file), header = True, ignore_missing_end = True)
    input_images = input_hdul[0].data
    header = input_hdul[0].header
    psf_center_x = header['PSF_CEN_X']
    psf_center_y = header['PSF_CEN_Y']
    
    #Get the auxilliary data
    data = np.loadtxt(os.path.join(test_data_path,'hlc_os11_batch_muf_info.txt'), skiprows=2)
    batch = data[:,0].astype(int)
    star = data[:,2].astype(int)
    roll = data[:,3]
    frame_exptime_sec = data[:,4]
    nframes = data[:,6].astype(int)

    mock_pri_header, mock_ext_header = create_default_L2b_headers()

    nbatch = 5 #We'll just do one reference and 2x 2 rolls for now. 

    istart = 0
    nframes_per_batch = 2 #Limit the number of frames we're going to generate for now. 

    big_array_size = [1024,1024]

    image_list = []
    for ibatch in range(nbatch): 
        iend = istart + nframes_per_batch

        batch_images = input_images[istart:iend,:,:]

        #Stick each one of these images into an L2b corgidrp Image object. 
        for i in range(nframes_per_batch): 

            big_array = np.zeros(big_array_size)
            psf_image = batch_images[i,:,:]

            big_rows, big_cols = big_array_size
            small_rows, small_cols = psf_image.shape

            # Find the middle indices for the big array
            row_start = (big_rows - small_rows) // 2
            col_start = (big_cols - small_cols) // 2

            # Insert the small array into the middle of the big array
            big_array[row_start:row_start + small_rows, col_start:col_start + small_cols] = psf_image

            #Update the PSF Center. Might have columns and row mixed up; doesn't matter for now since psf_center_x and psf_center_y are the same. 
            new_psf_center_x = psf_center_x + col_start
            new_psf_center_y = psf_center_y + row_start

            #Create the new Image object
            new_image = Image(big_array, mock_pri_header, mock_ext_header)
            new_image.ext_hdr.set('PSF_CEN_X', new_psf_center_x)
            new_image.ext_hdr.set('PSF_CEN_Y', new_psf_center_y)
            new_image.pri_hdr.set('FRAMET', frame_exptime_sec[ibatch])
            new_image.ext_hdr.set('EXPTIME', frame_exptime_sec[ibatch])
            new_image.pri_hdr.set('ROLL',roll[ibatch])

            #TODO add header keywords that WCS wants. 

            #If Reference star then flag it. 
            if star[ibatch] == 2:
                new_image.pri_hdr.set('PSFREF', 1)

            image_list.append(new_image)



        istart = istart + nframes[ibatch] 

    #############################################
    #### Add a sat spot image to the dataset ####
    #############################################
    #For now assuming there's just one since the step function in progress doesn't break things up. 
    #We'll want it do later. 

    #Create a mock satellite spot image with the same center as the PSF images. 
    satellite_spot_image = create_synthetic_satellite_spot_image([55,55],1e-4,0,2,14.79)
    big_array = np.zeros(big_array_size)
    big_rows, big_cols = big_array_size
    small_rows, small_cols = satellite_spot_image.shape

    # Find the middle indices for the big array
    row_start = (big_rows - small_rows) // 2
    col_start = (big_cols - small_cols) // 2

    # Insert the small array into the middle of the big array
    big_array[row_start:row_start + small_rows, col_start:col_start + small_cols] = psf_image

    sat_spot_image = Image(big_array, mock_pri_header, mock_ext_header)
    
    image_list.append(sat_spot_image)

    mock_dataset = data.Dataset(image_list)

    ## Next step run things through the walker. 


    


if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.


    outputdir = thisfile_dir
    ap = argparse.ArgumentParser(description="run the l2b->l4 end-to-end test")


    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    
    args = ap.parse_args()
    outputdir = args.outputdir
