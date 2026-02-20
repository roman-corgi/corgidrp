import numpy as np
import astropy.wcs as wcs
from corgidrp.spec import read_cent_wave
from corgidrp import data
from corgidrp import check

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
        pa_aper_deg = image.pri_hdr['PA_APER']

        #TO DO: double check this. northangle may be defined as the full rotation angle, 
        # not north offset, in which case, adding the two below would be adding two absolute rotation angles from north
        vert_ang = np.radians(northangle + pa_aper_deg)  ## might be -pa_aper_deg
        pc = np.array([[-np.cos(vert_ang), np.sin(vert_ang)], [np.sin(vert_ang), np.cos(vert_ang)]])
        cdmatrix = pc * (platescale * 0.001) / 3600.

        # create dictionary with wcs information
        wcs_info = {}
        wcs_info['CD1_1'] = float(cdmatrix[0,0])
        wcs_info['CD1_2'] = float(cdmatrix[0,1])
        wcs_info['CD2_1'] = float(cdmatrix[1,0])
        wcs_info['CD2_2'] = float(cdmatrix[1,1])

        wcs_info['CRPIX1'] = float(center_pixel[0])
        wcs_info['CRPIX2'] = float(center_pixel[1])

        wcs_info['CTYPE1'] = 'RA---TAN'
        wcs_info['CTYPE2'] = 'DEC--TAN'

        #wcs_info['CDELT1'] = (platescale * 0.001) / 3600  ## converting to degrees
        #wcs_info['CDELT2'] = (platescale * 0.001) / 3600

        wcs_info['CRVAL1'] = float(corrected_ra)
        wcs_info['CRVAL2'] = float(corrected_dec)

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


def split_image_by_polarization_state(input_dataset,
                                      image_center_x=512,
                                      image_center_y=512,
                                      separation_diameter_arcsec=7.5, 
                                      alignment_angle_WP1=0.0,
                                      alignment_angle_WP2=45.0,
                                      image_size=None,
                                      padding=5):
    """
    Split each polarimetric input image into two images by its polarization state, 
    recompose the two images into a 2 x image_size x image_size datacube

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level), should all be taken with the same color filter (same CFAMNAME header)
        image_center_x (optional, float): x pixel coordinate location of the center location between the two polarized images on the detector,
            default is the detector center at x=512.0
        image_center_y (optional, float): y pixel coordinate location of the center location between the two polarized images on the detector,
            default is the detector center at y=512.0
        separation_diameter_arcsec (optional, float): Distance between the centers of the two polarized images on the detector in arcsec, 
            default for Roman CGI is 7.5"
        alignment_angle_WP1 (optional, float): the angle in degrees of how the two polarized images are aligned with respect to the horizontal
            for WP1, defaults to 0
        alignment_angle_WP2 (optional, float): the angle in degrees of how the two polarized images are aligned with respect to the horizontal
            for WP1, defaults to 45
        image_size (optional, int): length/width of the cropped polarized images, if none is provided, 
            the size is automatically determined based on the coronagraph mask used
        padding (optional, int): number of pixels to leave as blank space between the outer radius of each PSF and the edge of the image,
            default is 5, is overriden if a custom image size is provided
    
    Returns:
        corgidrp.data.Dataset: The input dataset with each image now being a 2 x image_size x image_size datacube
    """

    passband = input_dataset[0].ext_hdr['CFAMNAME']
    #remove the letter F to make it compatible with table
    if passband[1] == 'F':
        passband = passband[0]
    
    updated_dataset = input_dataset.copy()

    # determine coronagraph FOV
    diam = 2.363114
    fov = input_dataset[0].ext_hdr['LSAMNAME']
    if fov == 'NFOV':
        # NFOV outer radius is 9.7 λ/D
        # convert to arcsec: λ/D * 206265
        radius_arcsec = 9.7 * ((read_cent_wave(passband)[0] * 1e-9) / diam) * 206265
    elif fov == 'WFOV':
        # WFOV outer radius is 20.1 λ/D
        # convert to arcsec: λ/D * 206265
        radius_arcsec = 20.1 * ((read_cent_wave(passband)[0] * 1e-9) / diam) * 206265
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
        image_size = 2 * (radius_pix + padding)
    
    for image in updated_dataset:
        im_data = image.data
        im_err = image.err
        im_dq = image.dq
        image_y, image_x = im_data.shape
        prism = image.ext_hdr['DPAMNAME']
        # make sure input image is polarized
        if prism != 'POL0' and prism != 'POL45':
            raise ValueError('Input image must be a polarimetric observation')
        
        # make sure every image in the input dataset uses the same filter
        if image.ext_hdr['CFAMNAME'] != input_dataset[0].ext_hdr['CFAMNAME']:
            raise ValueError('The color filter must be the same for all images in the dataset')

        # find polarized image centers
        if prism == 'POL0':
            #place image according to specified angle
            angle_rad = (alignment_angle_WP1 * np.pi) / 180
        else:
            angle_rad = (alignment_angle_WP2 * np.pi) / 180
        displacement_x = int(round((separation_diameter_arcsec * np.cos(angle_rad)) / (2 * 0.0218)))
        displacement_y = int(round((separation_diameter_arcsec * np.sin(angle_rad)) / (2 * 0.0218)))
        center_left = (image_center_x - displacement_x, image_center_y + displacement_y)
        center_right = (image_center_x + displacement_x, image_center_y - displacement_y)
        
        # find starting point for cropping
        image_radius = image_size // 2
        start_left = (center_left[0] - image_radius, center_left[1] - image_radius)
        start_right = (center_right[0] - image_radius, center_right[1] - image_radius)

        # check that cropped image doesn't exceed full image bounds
        if start_left[0] < 0 or start_left[1] < 0 or start_right[0] < 0 or start_right[1] < 0\
        or start_left[0] + image_size >= image_x or start_left[1] + image_size >= image_y\
        or start_right[1] + image_size >= image_x or start_right[1] + image_size >= image_y:
            raise ValueError('Image bounds exceed that of the input data, please decrease the image size')
        
        # construct new datacube for data, err, and dq
        im_data_new = np.zeros(shape=(2, image_size, image_size))
        # make sure to keep the additional dimension for err array
        im_err_new = np.zeros(shape=(im_err.shape[0], 2, image_size, image_size))
        im_dq_new = np.zeros(shape=(2, image_size, image_size))

        # define coordinates
        y, x = np.indices([image_size, image_size])
        x_left = x + start_left[0]
        y_left = y + start_left[1]
        x_right = x + start_right[0]
        y_right = y + start_right[1]
        # fill in the first dimension, corresponding to 0 or 45 degree polarization
        im_data_new[0,:,:] = im_data[start_left[1]:start_left[1] + image_size, start_left[0]:start_left[0] + image_size]
        im_err_new[:,0,:,:] = im_err[:, start_left[1]:start_left[1] + image_size, start_left[0]:start_left[0] + image_size]
        im_dq_new[0,:,:] = im_dq[start_left[1]:start_left[1] + image_size, start_left[0]:start_left[0] + image_size]
        # fill in the second dimension, corresponding to the 90 or 135 degree polarization
        im_data_new[1,:,:] = im_data[start_right[1]:start_right[1] + image_size, start_right[0]:start_right[0] + image_size]
        im_err_new[:,1,:,:] = im_err[:, start_right[1]:start_right[1] + image_size, start_right[0]:start_right[0] + image_size]
        im_dq_new[1,:,:] = im_dq[start_right[1]:start_right[1] + image_size, start_right[0]:start_right[0] + image_size]
        # mark anything on the other side of the center line dividing the two images as NaN to avoid including the other image
        if prism == 'POL0':
            im_data_new[0, x_left >= image_center_x] = np.nan
            im_data_new[1, x_right <= image_center_x] = np.nan
            im_err_new[:, 0, x_left >= image_center_x] = np.nan
            im_err_new[:, 1, x_right <= image_center_x] = np.nan
            # update dq with corresponding flag for bad pixel
            im_dq_new[0, x_left >= image_center_x] = 1
            im_dq_new[1, x_right <= image_center_x] = 1
        else:
            im_data_new[0, y_left - image_center_y <= x_left - image_center_x] = np.nan
            im_data_new[1, y_right - image_center_y >= x_right - image_center_x] = np.nan
            im_err_new[:, 0, y_left - image_center_y <= x_left - image_center_x] = np.nan
            im_err_new[:, 1, y_right - image_center_y >= x_right - image_center_x] = np.nan
            im_dq_new[0, y_left - image_center_y <= x_left - image_center_x] = 1
            im_dq_new[1, y_right - image_center_y >= x_right - image_center_x] = 1           

        #update data, err, and dq
        image.data = im_data_new
        image.err = im_err_new
        image.dq = im_dq_new

    history_msg = 'images split by polarization state'
    updated_dataset.update_after_processing_step(history_msg)
    
    return updated_dataset


def crop(input_dataset, sizexy=None, centerxy=None):
    """
    
    Crop the Images in a Dataset to a desired field of view. Default behavior is to 
    crop the image to the dark hole region, centered on the pixel intersection nearest 
    to the star location. Assumes 3D Image data is a stack of 2D data arrays, so only 
    crops the last two indices.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (any level)
        sizexy (int or array of int): desired frame size, if only one number is provided the 
            desired shape is assumed to be square, otherwise xy order. If not provided, 
            defaults to 61 for NFOV (narrow field-of-view) observations. Defaults to None.
        centerxy (float or array of float): desired center (xy order), should be a pixel center 
            (aka integer) or intersection (aka  half-integer) otherwise the function rounds 
            to the nearest pixel or pixel intersection. Defaults to the "EACQ_ROW/COL" header values.

    Returns:
        corgidrp.data.Dataset: a version of the input dataset cropped to the desired FOV.
    """

    # Copy input dataset
    dataset = input_dataset.copy()
       
    # Need to loop over frames and reinit dataset because array sizes change
    frames_out = []

    for frame in dataset:
        prihdr = frame.pri_hdr
        exthdr = frame.ext_hdr
        dqhdr = frame.dq_hdr
        errhdr = frame.err_hdr

        # Pick default crop size based on the size of the effective field of view
        if sizexy is None:

            filter_band = exthdr['CFAMNAME']
            # change filter names ending in F to just the number
            if filter_band[1] == 'F':
                filter_band = filter_band[0]
            prism = exthdr['DPAMNAME']
            slit = exthdr['FSAMNAME']
            cor_mode = exthdr['LSAMNAME']
            spec_slits = ['R1C2', 'R2C3', 'R2C4', 'R2C5', 'R3C1', 'R3C2', 'R4C6', 'R5C1', 'R5C2', 'R6C3', 'R6C4', 'R6C5']
            spec_prisms = ['PRISM2', 'PRISM3']
            color_filters = ['1', '1A', '1B', '1C', '2', '2A', '2B', '2C', '3', '3A', '3B', '3C', '3D', '3E', '3F', '3G', '4', '4A', '4B', '4C']
            # outer working angle in lambda/D
            cor_outer_working_angle = {
                'WFOV': 20.1,
                'SPEC': 9.1
            }
            # Skip cropping by default if observation is non-coronagraphic
            if cor_mode == 'OPEN':
                return dataset
            if cor_mode == 'NFOV':
                # set size to 61 if coronagraph is HLC NFOV
                sizexy = 61
            elif cor_mode not in cor_outer_working_angle or filter_band not in color_filters:
                # raise warning if unable to calculate image size
                raise UserWarning('Unable to determine image size, please change instrument configuration or provide a sizexy value')
            else:
                ## calculate image size using coronagraph information
                padding = 5
                diam = 2.363114
                # convert lambda/D to arcsec
                radius_arcsec = cor_outer_working_angle[cor_mode] * ((read_cent_wave(filter_band)[0] * 1e-9) / diam) * 206265
                # convert arcsec to pix, 1 pix = 0.0218", round to nearest integer
                radius_pix = int(round(radius_arcsec / 0.0218))
                # update sizexy
                sizexy = 2 * (padding + radius_pix) + 1
                # add additional 60 pixels to account for increase in size with spec slit or prism
                if slit in spec_slits or prism in spec_prisms:
                    sizexy += 60
                          

        # Assign new array sizes and center location
        frame_shape = frame.data.shape
        if isinstance(sizexy,int):
            sizexy = [sizexy]*2
        if isinstance(centerxy,float):
            centerxy = [centerxy] * 2
        elif centerxy is None:
            if ("EACQ_COL" in exthdr.keys()) and ("EACQ_ROW" in exthdr.keys()):
                centerxy = np.array([exthdr["EACQ_COL"],exthdr["EACQ_ROW"]])
                if float(exthdr["EACQ_COL"]) == -999.0 or float(exthdr["EACQ_ROW"]) == -999.0:
                    raise ValueError('EACQ_ROW/COL header values are invalid (-999.0)')
            else: raise ValueError('centerxy not provided but EACQ_ROW/COL are missing from image extension header.')
        # Round to center to nearest half-pixel if size is even, nearest pixel if odd
        size_evenness = (np.array(sizexy) % 2) == 0
        centerxy_input = np.array(centerxy)
        centerxy = np.where(size_evenness,np.round(centerxy_input-0.5)+0.5,np.round(centerxy_input))
        if not np.all(centerxy == centerxy_input):
            print(f'Desired center was {centerxy_input}. Centering crop on {centerxy}.')
        # Crop the data
        start_ind = (centerxy + 0.5 - np.array(sizexy)/2).astype(int)
        end_ind = (centerxy + 0.5 + np.array(sizexy)/2).astype(int)
        x1,y1 = start_ind
        x2,y2 = end_ind

        # Check if cropping outside the FOV
        left_pad = -x1 if (x1<0) else 0
        right_pad = x2-frame_shape[-1] if (x2 > frame_shape[-1]) else 0
        below_pad = -y1 if (y1<0) else 0
        above_pad = y2-frame_shape[-2] if (y2 > frame_shape[-2]) else 0


        if frame.data.ndim == 2:

            cropped_frame_data = np.full(sizexy[::-1],np.nan)
            cropped_frame_data[below_pad:sizexy[1]-above_pad,
                               left_pad:sizexy[0]-right_pad] = frame.data[y1+below_pad:y2-above_pad,
                                                                          x1+left_pad:x2-right_pad]
            cropped_frame_err = np.full((frame.err.shape[0],*sizexy[::-1]),np.nan)
            cropped_frame_err[:,below_pad:sizexy[1]-above_pad,
                               left_pad:sizexy[0]-right_pad] = frame.err[:,y1+below_pad:y2-above_pad,
                                                                         x1+left_pad:x2-right_pad]
            cropped_frame_dq = np.full(sizexy[::-1],np.nan)
            cropped_frame_dq[below_pad:sizexy[1]-above_pad,
                               left_pad:sizexy[0]-right_pad] = frame.dq[y1+below_pad:y2-above_pad,
                                                                          x1+left_pad:x2-right_pad]
            
            # cropped_frame_data = frame.data[y1:y2,x1:x2]
            # cropped_frame_err = frame.err[:,y1:y2,x1:x2]
            # cropped_frame_dq = frame.dq[y1:y2,x1:x2]

        elif frame.data.ndim == 3:

            cropped_frame_data = np.full((frame.data.shape[0],*sizexy[::-1]),np.nan)
            cropped_frame_data[:,below_pad:sizexy[1]-above_pad,
                               left_pad:sizexy[0]-right_pad] = frame.data[:,y1+below_pad:y2-above_pad,
                                                                          x1+left_pad:x2-right_pad]
            cropped_frame_err = np.full((*frame.err.shape[:2],*sizexy[::-1]),np.nan)
            cropped_frame_err[:,:,below_pad:sizexy[1]-above_pad,
                               left_pad:sizexy[0]-right_pad] = frame.err[:,:,y1+below_pad:y2-above_pad,
                                                                         x1+left_pad:x2-right_pad]
            cropped_frame_dq = np.full((frame.dq.shape[0],*sizexy[::-1]),0).astype(int)
            cropped_frame_dq[:,below_pad:sizexy[1]-above_pad,
                               left_pad:sizexy[0]-right_pad] = frame.dq[:,y1+below_pad:y2-above_pad,
                                                                          x1+left_pad:x2-right_pad]
            
            
            # cropped_frame_data = frame.data[:,y1:y2,x1:x2]
            # cropped_frame_err = frame.err[:,:,y1:y2,x1:x2]
            # cropped_frame_dq = frame.dq[:,y1:y2,x1:x2]
        else:
            raise ValueError('Crop function only supports 2D or 3D frame data.')

        # Update headers
        exthdr["NAXIS1"] = sizexy[0]
        exthdr["NAXIS2"] = sizexy[1]
        dqhdr["NAXIS1"] = sizexy[0]
        dqhdr["NAXIS2"] = sizexy[1]
        errhdr["NAXIS1"] = sizexy[0]
        errhdr["NAXIS2"] = sizexy[1]
        errhdr["NAXIS3"] = cropped_frame_err.shape[-3]
        if frame.data.ndim == 3:
            exthdr["NAXIS3"] = frame.data.shape[0]
            dqhdr["NAXIS3"] = frame.dq.shape[0]
            errhdr["NAXIS4"] = frame.err.shape[0]
        
        updated_hdrs = []
        if ("EACQ_COL" in exthdr.keys()):
            exthdr["EACQ_COL"] -= x1
            exthdr["EACQ_ROW"] -= y1
            updated_hdrs.append('EACQ_ROW/COL')
        if ("CRPIX1" in exthdr.keys()):
            exthdr["CRPIX1"] -= x1
            exthdr["CRPIX2"] -= y1
            updated_hdrs.append('CRPIX1/2')
        if ("STARLOCX" in exthdr.keys()):
            exthdr["STARLOCX"] -= x1
            exthdr["STARLOCY"] -= y1
            updated_hdrs.append('STARLOCX/Y')
        if not ("DETPIX0X" in exthdr.keys()):
            exthdr.set('DETPIX0X',0)
            exthdr.set('DETPIX0Y',0)
        exthdr.set('DETPIX0X',exthdr["DETPIX0X"]+x1)
        exthdr.set('DETPIX0Y',exthdr["DETPIX0Y"]+y1)

        new_frame = data.Image(cropped_frame_data,prihdr,exthdr,cropped_frame_err,cropped_frame_dq,frame.err_hdr,frame.dq_hdr)
        new_frame.filename = frame.filename
        frames_out.append(new_frame)

    output_dataset = data.Dataset(frames_out)

    history_msg1 = f"""Frames cropped to new shape {list(output_dataset[0].data.shape)} on center {list(centerxy)}. Updated header kws: {", ".join(updated_hdrs)}."""
    output_dataset.update_after_processing_step(history_msg1)
    
    return output_dataset


def update_to_l3(input_dataset):
    """
    Updates the data level to L3. Only works on L2b data.

    Currently only checks that data is at the L2b level

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2b-level)

    Returns:
        corgidrp.data.Dataset: same dataset now at L3-level
    """
    # check that we are running this on L2b data
    for orig_frame in input_dataset:
        if orig_frame.ext_hdr['DATALVL'] != "L2b":
            err_msg = "{0} needs to be L2b data, but it is {1} data instead".format(orig_frame.filename, orig_frame.ext_hdr['DATALVL'])
            raise ValueError(err_msg)

    # we aren't altering the data
    updated_dataset = input_dataset.copy(copy_data=False)

    for frame in updated_dataset:
        # Apply header rules to each frame
        pri_hdr, ext_hdr, err_hdr, dq_hdr = check.merge_headers(data.Dataset([frame]))
        frame.pri_hdr = pri_hdr
        frame.ext_hdr = ext_hdr
        frame.err_hdr = err_hdr
        frame.dq_hdr = dq_hdr
        frame.ext_hdr['DATALVL'] = "L3"
        # update filename convention. The file convention should be
        # "CGI_[dataleel_*]" so we should be same just replacing the just instance of L1
        frame.filename = frame.filename.replace("_l2b", "_l3_", 1)
        #updating filename in the primary header
        frame.pri_hdr['FILENAME'] = frame.filename

    history_msg = "Updated Data Level to L3"
    updated_dataset.update_after_processing_step(history_msg)

    return updated_dataset