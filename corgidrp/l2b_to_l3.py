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

def split_image_by_polarization_state(input_dataset, image_center=(512,512), separation_diameter_arcsec=7.5, image_size=None):
    """
    Split each polarimetric input image into two images by its polarization state, 
    recompose the two images into a 2 x image_size x image_size datacube

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)
        image_center (optional, tuple(int, int)): x and y pixel coordinate location of the center location between the two polarized images on the detector,
            default is the detector center at 512, 512
        separation_diameter_arcsec (optional, float): Distance between the centers of the two polarized images on the detector in arcsec, 
            Default for Roman CGI is 7.5"
        image_size (optional, int): length/width of the cropped polarized images, if none is provided, 
            the size is automatically determined based on the coronagraph mask used
    
    Returns:
        corgidrp.data.Dataset: The input dataset with each image now being a 2 x image_size x image_size datacube
    """

    passband = input_dataset[0].ext_hdr['CFAMNAME']
    if passband != '1F' and passband != '4F':
        raise ValueError(f'Polarimetric datasets must be imaged in band 1F or 4F, not {passband}')
    
    updated_dataset = input_dataset.copy()

    # determine coronagraph FOV
    bandpass_center_um = {
        '1F': 0.5738,
        '4F': 0.8255
    }
    diam = 2.363114
    fov = input_dataset[0].ext_hdr['FSMPRFL']
    if fov == 'NFOV':
        # NFOV outer radius is 9.7 λ/D
        # convert to arcsec: λ/D * 206265
        radius_arcsec = 9.7 * ((bandpass_center_um[passband] * 1e-6) / diam) * 206265
    elif fov == 'WFOV':
        # WFOV outer radius is 20.1 λ/D
        # convert to arcsec: λ/D * 206265
        radius_arcsec = 20.1 * ((bandpass_center_um[passband] * 1e-6) / diam) * 206265
    else:
        # default to unvignetted polarimetry FOV diameter of 3.8"
        radius_arcsec = 3.8 / 2
    # convert to pixel: 0.0218" = 1 pixel
    radius_pix = int(round(radius_arcsec / 0.0218))

    # raise error if the polarized images are close enough to overlap
    if separation_diameter_arcsec < 2 * radius_arcsec:
        raise ValueError(f'The inputted separation diameter of {separation_diameter_arcsec}" must be greater than {2 * radius_arcsec}"')
    
    # auto determine image size based on FOV if none is provided
    if image_size == None:
        # number of pixels between the coronagraph focal plane's outer radius and the image edge
        padding = 5
        image_size = 2 * (radius_pix + padding)
    
    for image in updated_dataset:
        im_data = image.data
        image_y, image_x = im_data.shape
        prism = image.ext_hdr['DPAMNAME']
        # make sure input image is polarized
        if prism != 'POL0' and prism != 'POL45':
            raise ValueError('Input image must be a polarimetric observation')
        
        # find polarized image centers
        if prism == 'POL0':
            # polarized images placed horizontally on detector
            displacement = int(round(separation_diameter_arcsec / (2 * 0.0218)))
            center_left = (image_center[0] - displacement, image_center[1])
            center_right = (image_center[0] + displacement, image_center[1])
        else:
            # polarized images placed diagonally on detector
            displacement = int(round(separation_diameter_arcsec / (2 * 0.0218 * np.sqrt(2))))
            center_left = (image_center[0] - displacement, image_center[1] + displacement)
            center_right = (image_center[0] + displacement, image_center[1] - displacement)
        
        # find starting point for cropping
        image_radius = image_size // 2
        start_left = (center_left[0] - image_radius, center_left[1] - image_radius)
        start_right = (center_right[0] - image_radius, center_right[1] - image_radius)

        # check that cropped image doesn't exceed full image bounds
        if start_left[0] < 0 or start_left[1] < 0 or start_right[0] < 0 or start_right[1] < 0\
        or start_left[0] + image_size >= image_x or start_left[1] + image_size >= image_y\
        or start_right[1] + image_size >= image_x or start_right[1] + image_size >= image_y:
            raise ValueError('Image bounds exceed that of the input data, please decrease the image size')
        
        # construct new datacube
        im_data_new = np.zeros(shape=(2, image_size, image_size))

        # fill in the first dimension, corresponding to 0 or 45 degree polarization
        # for each pixel in the cropped area, if it's inside the radius of the other polarized image, replace it with a NaN
        for i in range(image_size):
            for j in range(image_size):
                y = start_left[1] + i
                x = start_left[0] + j
                # mark anything on the other side of the center line dividing the two images as NaN to avoid including the other image
                if (prism == 'POL0' and x >= image_center[0]) or\
                (prism=='POL45' and y + image_center[1] <= x + image_center[0]):
                    im_data_new[0, i, j] = float('nan')
                else:
                    im_data_new[0, i, j] = im_data[y, x]
        # fill in the second dimension, corresponding to the 90 or 135 degree polarization
        for i in range(image_size):
            for j in range(image_size):
                y = start_right[1] + i
                x = start_right[0] + j
                # mark anything on the other side of the center line dividing the two images as NaN to avoid including the other image
                if (prism == 'POL0' and x <= image_center[0]) or\
                (prism=='POL45' and y + image_center[1] >= x + image_center[0]):
                    print('NaN')
                    im_data_new[1, i, j] = float('nan')
                else:
                    im_data_new[1, i, j] = im_data[y, x]

        #update data
        image.data = im_data_new
    
    return updated_dataset



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