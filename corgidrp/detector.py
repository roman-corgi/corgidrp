import numpy as np

import corgidrp.data as data

image_constants= {
    'SCI' : {
        'frame_rows' : 1200,
        'frame_cols' : 2200,
        'image' : {
            'rows': 1024,
            'cols': 1024,
            'r0c0': [13, 1088]
            },
        'prescan' : {
            'rows': 1200,
            'cols': 1088,
            'r0c0': [0, 0]
            },
        'prescan_reliable' : {
            'rows': 1200,
            'cols': 200,
            'col_start': 800,
            'col_end': 1000,
            'r0c0': [0, 800]
            },
        'parallel_overscan' : {
            'rows': 163,
            'cols': 1056,
            'r0c0': [1037, 1088]
            },
        'serial_overscan' : {
            'rows': 1200,
            'cols': 56,
            'r0c0': [0, 2144]
            },
        },
    'ENG' :{
        'frame_rows' : 2200,
        'frame_cols' : 2200,
        'image' : {
            'rows': 1024,
            'cols': 1024,
            'r0c0': [13, 1088]
            },
        'prescan' : {
            'rows': 2200,
            'cols': 1088,
            'r0c0': [0, 0]
            },
        'prescan_reliable' : {
            'rows': 2200,
            'cols': 200,
            'col_start': 800,
            'col_end': 1000,
            'r0c0': [0, 800]
            },
        'parallel_overscan' : {
            'rows': 1163,
            'cols': 1056,
            'r0c0': [1037, 1088]
            },
        'serial_overscan' : {
            'rows': 2200,
            'cols': 56,
            'r0c0': [0, 2144]
            },
        },
    }

def create_dark_calib(dark_dataset):
    """
    Turn this dataset of image frames that were taken to measure 
    the dark current into a dark calibration frame

    Args:
        dark_dataset (corgidrp.data.Dataset): a dataset of Image frames (L2a-level)
    Returns:
        data.Dark: a dark calibration frame
    """
    combined_frame = np.nanmean(dark_dataset.all_data, axis=0)

    new_dark = data.Dark(combined_frame, pri_hdr=dark_dataset[0].pri_hdr.copy(), 
                         ext_hdr=dark_dataset[0].ext_hdr.copy(), input_dataset=dark_dataset)
    
    return new_dark

def slice_section(frame, obstype, key):
    """Slice 2d section out of frame.

    Parameters
    ----------
    frame : array_like
        Full frame consistent with size given in frame_rows, frame_cols.
    key : str
        Keyword referencing section to be sliced; must exist in geom.

    """
    rows = image_constants[obstype][key]['rows']
    cols = image_constants[obstype][key]['cols']
    r0c0 = image_constants[obstype][key]['r0c0']

    section = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
    if section.size == 0:
        raise Exception('Corners invalid')
    return section

def prescan_biassub_v2(input_dataset, bias_offset=0.):
    """
    Perform pre-scan bias subtraction of a dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L1a-level)
        bias_offset (float): an offset value to be subtracted from the bias
    Returns:
        corgidrp.data.Dataset: a pre-scan bias subtracted version of the input dataset
    """



    # Make a copy of the input dataset to operate on
    output_dataset = input_dataset.copy()

    # Iterate over frames
    for i, frame in enumerate(output_dataset.frames):

        frame_data = frame.data

        # Determine what type of file it is (engineering or science), then choose detector area dict
        obstype = frame.pri_hdr['OBSTYPE']
        assert(obstype in ['SCI','ENG'], f"Observation type of frame {i} is not 'SCI' or 'ENG'")

        # Get the reliable prescan area
        prescan = slice_section(frame_data, obstype, 'prescan_reliable')

        # Get the image area
        image = slice_section(frame_data, obstype, 'image')

        # Get the part of the prescan that lines up with the image, and do a
        # row-by-row bias subtraction on it
        i_r0 = image_constants[obstype]['image']['r0c0'][0]
        p_r0 = image_constants[obstype]['prescan']['r0c0'][0]
        i_nrow = image_constants[obstype]['image']['rows']
        # select the good cols for getting row-by-row bias
        st = image_constants[obstype]['prescan_reliable']['col_start']
        end = image_constants[obstype]['prescan_reliable']['col_end']
        
        # prescan aligned with image rows
        al_prescan = prescan[(i_r0-p_r0):(i_r0-p_r0+i_nrow), :]
        medbyrow = np.median(al_prescan[:,st:end], axis=1)[:, np.newaxis]

        # # Get data from prescan (alined with image area)
        bias = medbyrow - bias_offset
        image_bias_corrected = image - bias

        # # over total frame
        # medbyrow_tot = np.median(prescan[:,st:end], axis=1)[:, np.newaxis]
        # frame_bias = medbyrow_tot - bias_offset
        # frame_bias_corrected = frame_data[p_r0:, :] -  frame_bias

        # Update frame data and header in the dataset
        output_dataset.frames[i].data = image_bias_corrected
        output_dataset.frames[i].ext_hdr['NAXIS1'] = image_bias_corrected.shape[1]
        output_dataset.frames[i].ext_hdr['NAXIS2'] = image_bias_corrected.shape[0]
        
    history_msg = "Frames cropped and bias subtracted"

    # update the output dataset with this new dark subtracted data and update the history
    output_dataset.update_after_processing_step(history_msg)

    return output_dataset

def prescan_biassub(input_dataset, bias_offset=0.):
    """
    Perform pre-scan bias subtraction of a dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L1a-level)
        bias_offset (float): an offset value to be subtracted from the bias
    Returns:
        corgidrp.data.Dataset: a pre-scan bias subtracted version of the input dataset
    """


    # Create a dictionary of image constants. Describing the different areas on the detector.
    
    def detector_area_mask(image_constants, area='image'):
        """
        Create a mask for the detector area

        Args:
            area (str): the area of the detector to create a mask for
        Returns:
            np.ndarray: a mask for the detector area
        """
        mask = np.zeros((image_constants['frame_rows'], image_constants['frame_cols']), dtype=bool)
        mask[image_constants[area]['r0c0'][0]:image_constants[area]['r0c0'][0] + image_constants[area]['rows'],
             image_constants[area]['r0c0'][1]:image_constants[area]['r0c0'][1] + image_constants[area]['cols']] = True
        return mask


    def make_detector_areas(image_constants, areas=('image', 'prescan', 'prescan_reliable',
            'parallel_overscan', 'serial_overscan')):
        """
        Create a dictionary of masks for the different detector areas

        Args:
            image_constants (dict): a dictionary of image constants
            areas (tuple): a tuple of areas to create masks for

        Returns:
            dict: a dictionary of masks for the different detector areas
        """
        detector_areas = {}
        for area in areas:
            detector_areas[area] = detector_area_mask(image_constants, area=area)
        return detector_areas

    
    def make_detector_area_image(image_constants, areas=('image', 'prescan',
            'prescan_reliable', 'parallel_overscan', 'serial_overscan')):
        """
        Create an image of the detector areas for visualization and debugging

        Args:
            image_constants (dict): a dictionary of image constants
            areas (tuple): a tuple of areas to create masks for

        Returns:
            np.ndarray: an image of the detector areas
        """
        detector_areas = make_detector_areas(image_constants, areas=areas)
        detector_area_image = np.zeros(
            (image_constants['frame_rows'], image_constants['frame_cols']), dtype=int)
        for i, area in enumerate(areas):
            detector_area_image[detector_areas[area]] = i + 1
        return detector_area_image

    detector_areas = make_detector_areas(image_constants_sci)
    detector_area_image = make_detector_area_image(
        image_constants_sci,
        areas=('image', 'prescan', 'prescan_reliable', 'parallel_overscan', 'serial_overscan'))

    plt.imshow(detector_area_image, origin='lower')
    plt.show()

    

    # output_dataset = input_dataset.copy()

    # Determine what type of file it is (engineering or science), then read Metadata file?

    # cube = output_dataset.all_data
    # frame = cube[0]
    frame = fits.getdata('/home/samland/science/python/projects/corgidrp/example_L1_input.fits')

    # prescan_reliable_mask = detector_area_mask(image_constants, area='prescan_reliable')
    # image = frame[detector_areas['image']]
    

    prescan_reliable = frame[detector_areas['prescan_reliable']]
    median_prescan = np.median(prescan_reliable, axis=1)[:, np.newaxis]

    i_r0 = image_constants['image']['r0c0'][0]
    p_r0 = image_constants['prescan']['r0c0'][0]
    i_nrow = image_constants['image']['rows']

    median_prescan_for_image_region = np.median(
        prescan_reliable[(i_r0-p_r0):(i_r0-p_r0+i_nrow), :], axis=1)[:, np.newaxis]

    image_bias_corr = image - (median_prescan_for_image_region - bias_offset)
    full_image_bias_corr = frame - (median_prescan - bias_offset)

    return image_bias_corr


    # Subtract the pre-scan bias
    for frame in output_dataset:
        frame.data = frame.data - np.nanmean(frame.data, axis=1)[:, np.newaxis]

    # Get the part of the prescan that lines up with the image, and do a
    # row-by-row bias subtraction on it
    # i_r0 = self.meta.geom['image']['r0c0'][0]
    # p_r0 = self.meta.geom['prescan']['r0c0'][0]
    # i_nrow = self.meta.geom['image']['rows']
    # # select the good cols for getting row-by-row bias
    # st = self.meta.geom['prescan']['col_start']
    # end = self.meta.geom['prescan']['col_end']
    # # over all prescan rows
    # medbyrow_tot = np.median(self.prescan[:,st:end], axis=1)[:, np.newaxis]
    # # prescan relative to image rows
    # self.al_prescan = self.prescan[(i_r0-p_r0):(i_r0-p_r0+i_nrow), :]
    # medbyrow = np.median(self.al_prescan[:,st:end], axis=1)[:, np.newaxis]

    # # Get data from prescan (image area)
    # self.bias = medbyrow - self.bias_offset
    # self.image_bias0 = self.image - self.bias

    # # over total frame
    # self.frame_bias = medbyrow_tot - self.bias_offset
    # self.frame_bias0 = self.frame_dn[p_r0:, :] -  self.frame_bias



    # cube = darksub_dataset.all_data - dark_frame.data

    # history_msg = "Dark current subtracted using dark {0}".format(dark_frame.filename)

    # update the output dataset with this new dark subtracted data and update the history
    # darksub_dataset.update_after_processing_step(history_msg, new_all_data=darksub_cube)

    return output_dataset


def dark_subtraction(input_dataset, dark_frame):
    """
    Perform dark current subtraction of a dataset using the corresponding dark frame

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need dark subtraction (L2a-level)
        dark_frame (corgidrp.data.Dark): a Dark frame to model the dark current
    Returns:
        corgidrp.data.Dataset: a dark subtracted version of the input dataset
    """
    # you should make a copy the dataset to start
    darksub_dataset = input_dataset.copy()

    darksub_cube = darksub_dataset.all_data - dark_frame.data

    history_msg = "Dark current subtracted using dark {0}".format(dark_frame.filename)

    # update the output dataset with this new dark subtracted data and update the history
    darksub_dataset.update_after_processing_step(history_msg, new_all_data=darksub_cube)

    return darksub_dataset
