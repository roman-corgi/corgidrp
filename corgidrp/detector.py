# Place to put detector-related utility functions

import numpy as np

import corgidrp.data as data

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



def plot_detector_areas(image_constants, areas=('image', 'prescan',
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
