import numpy as np
import astropy.wcs as wcs

# A file that holds the functions that transmogrify l2b data to l3 data 

def create_wcs(input_dataset, astrom_calibration):
    """
    
    Create the WCS headers for the dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)
        astrom_calibration (corgidrp.data.AstrometricCalibration): an astrometric calibration file for the input dataset

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the WCS headers added
    """
    updated_dataset = input_dataset.copy()

    northangle = astrom_calibration.northangle
    platescale = astrom_calibration.platescale
    center_coord = astrom_calibration.boresight

    # create wcs for each image in the dataset
    for image in updated_dataset:

        im_data = image.data
        image_shape = im_data.shape
        center_pixel = [image_shape[1] // 2, image_shape[0] // 2]
        roll_ang = image.pri_hdr['ROLL']

        vert_ang = np.radians(northangle + roll_ang)  ## might be -roll_ang
        pc = np.array([[-np.cos(vert_ang), np.sin(vert_ang)], [np.sin(vert_ang), np.cos(vert_ang)]])
        cdmatrix = pc * (platescale * 0.001) / 3600.

        # create dictionary with wcs information
        wcs_info = {}
        wcs_info['CD1_1'] = cdmatrix[0,0]
        wcs_info['CD1_2'] = cdmatrix[0,1]
        wcs_info['CD2_1'] = cdmatrix[1,0]
        wcs_info['CD2_2'] = cdmatrix[1,1]

        wcs_info['CRPIX1'] = center_pixel[0]
        wcs_info['CRPIX2'] = center_pixel[1]

        wcs_info['CTYPE1'] = 'RA---TAN'
        wcs_info['CTYPE2'] = 'DEC--TAN'

        wcs_info['CDELT1'] = (platescale * 0.001) / 3600  ## converting to degrees
        wcs_info['CDELT2'] = (platescale * 0.001) / 3600

        wcs_info['CRVAL1'] = center_coord[0]
        wcs_info['CRVAL2'] = center_coord[1]

        # update the image header with wcs information
        for key, value in wcs_info.items():
            image.ext_hdr[key] = value

    # update the dataset with new header entries an dhistory message
    history_msg = 'WCS created'

    updated_dataset.update_after_processing_step(history_msg)

    return updated_dataset

def divide_by_exptime(input_dataset):
    """
    
    Divide the data by the exposure time to get the units in electrons/s

    TODO: Make sure to update the headers to reflect the change in units

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in units of electrons/s
    """

    return input_dataset.copy()

def update_to_l3(input_dataset):
    """
    Updates the data level to L3. Only works on L2b data.

    Currently only checks that data is at the L2b level

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)

    Returns:
        corgidrp.data.Dataset: same dataset now at L3-level
    """
    # check that we are running this on L1 data
    for orig_frame in input_dataset:
        if orig_frame.ext_hdr['DATA_LEVEL'] != "L2b":
            err_msg = "{0} needs to be L2b data, but it is {1} data instead".format(orig_frame.filename, orig_frame.ext_hdr['DATA_LEVEL'])
            raise ValueError(err_msg)

    # we aren't altering the data
    updated_dataset = input_dataset.copy(copy_data=False)

    for frame in updated_dataset:
        # update header
        frame.ext_hdr['DATA_LEVEL'] = "L3"
        # update filename convention. The file convention should be
        # "CGI_[dataleel_*]" so we should be same just replacing the just instance of L1
        frame.filename = frame.filename.replace("_L2b_", "_L3_", 1)

    history_msg = "Updated Data Level to L3"
    updated_dataset.update_after_processing_step(history_msg)

    return updated_dataset