import numpy as np
import astropy.wcs as wcs

# A file that holds the functions that transmogrify l2b data to l3 data 
import numpy as np

def create_wcs(input_dataset, astrom_calibration, offset=None):
    """
    
    Create the WCS headers for the dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)
        astrom_calibration (corgidrp.data.AstrometricCalibration): an astrometric calibration file for the input dataset
        offset (optional, tuple(float, float)): x and y offset in units of pixel between the dataset and WCS center (for spectroscopy or other optics offset from imaging mode)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the WCS headers added
    """
    updated_dataset = input_dataset.copy()

    northangle = astrom_calibration.northangle
    platescale = astrom_calibration.platescale
    ra_offset, dec_offset = astrom_calibration.avg_offset

    # create wcs for each image in the dataset
    for image in updated_dataset:

        im_data = image.data
        image_y, image_x = im_data.shape
        center_pixel = [(image_x-1) // 2, (image_y-1) // 2]
        if offset is not None:
            center_pixel[0] += offset[0]
            center_pixel[1] += offset[1]
        target_ra, target_dec = image.pri_hdr['RA'], image.pri_hdr['DEC']
        corrected_ra, corrected_dec = target_ra - ra_offset, target_dec - dec_offset
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

        wcs_info['CRVAL1'] = corrected_ra
        wcs_info['CRVAL2'] = corrected_dec

        wcs_info['PLTSCALE'] = platescale  ## [mas] / pixel

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

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in units of electrons/s
    """
    if input_dataset[0].ext_hdr['BUNIT'] != "photoelectron":
        raise ValueError("input dataset must have unit photoelectron for the conversion, not {0}".format(input_dataset[0].ext_hdr['BUNIT']))
    data = input_dataset.copy()

    all_data_new = np.zeros(data.all_data.shape)
    all_err_new = np.zeros(data.all_err.shape)

    for i in range(len(data.frames)):
        exposure_time = float(data.frames[i].ext_hdr['EXPTIME'])

        data.frames[i].data = data.frames[i].data / exposure_time
        # data.frames[i].err = data.frames[i].err / exposure_time
        scale_factor = (1/exposure_time) * np.ones([data.frames[i].err.shape[1], data.frames[i].err.shape[2]])
        # print(data.frames[i].err.shape)
        # print(data.frames[i].err[0])
        # print(scale_factor.shape)
        #data.frames[i].rescale_error(data.frames[i].err, 'normalized by the exposure time')
        data.frames[i].rescale_error(scale_factor, 'normalized by the exposure time')

        all_data_new[i] = data.frames[i].data
        all_err_new[i] = data.frames[i].err

        data.frames[i].ext_hdr.set('BUNIT', 'photoelectron/s')
    
    history_msg = 'divided by the exposure time'
    data.update_after_processing_step(history_msg, new_all_data = all_data_new, new_all_err = all_err_new)


    return data

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
        if orig_frame.ext_hdr['DATALVL'] != "L2b":
            err_msg = "{0} needs to be L2b data, but it is {1} data instead".format(orig_frame.filename, orig_frame.ext_hdr['DATALVL'])
            raise ValueError(err_msg)

    # we aren't altering the data
    updated_dataset = input_dataset.copy(copy_data=False)

    for frame in updated_dataset:
        # update header
        frame.ext_hdr['DATALVL'] = "L3"
        # update filename convention. The file convention should be
        # "CGI_[dataleel_*]" so we should be same just replacing the just instance of L1
        frame.filename = frame.filename.replace("_l2b", "_l3_", 1)

    history_msg = "Updated Data Level to L3"
    updated_dataset.update_after_processing_step(history_msg)

    return updated_dataset