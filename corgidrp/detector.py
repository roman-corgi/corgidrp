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

detector_areas= {
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

def slice_section(frame, obstype, key):
    """
    Slice 2d section out of frame

    Args:
        frame (np.ndarray): Full frame consistent with size given in frame_rows, frame_cols
        key (str): Keyword referencing section to be sliced; must exist in detector_areas

    Returns: 
        np.ndarray: a 2D array of the specified detector area
    """
    rows = detector_areas[obstype][key]['rows']
    cols = detector_areas[obstype][key]['cols']
    r0c0 = detector_areas[obstype][key]['r0c0']

    section = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
    if section.size == 0:
        raise Exception('Corners invalid')
    return section

def plot_detector_areas(detector_areas, areas=('image', 'prescan',
        'prescan_reliable', 'parallel_overscan', 'serial_overscan')):
    """
    Create an image of the detector areas for visualization and debugging

    Args:
        detector_areas (dict): a dictionary of image constants
        areas (tuple): a tuple of areas to create masks for

    Returns:
        np.ndarray: an image of the detector areas
    """
    detector_areas = make_detector_areas(detector_areas, areas=areas)
    detector_area_image = np.zeros(
        (detector_areas['frame_rows'], detector_areas['frame_cols']), dtype=int)
    for i, area in enumerate(areas):
        detector_area_image[detector_areas[area]] = i + 1
    return detector_area_image

def detector_area_mask(detector_areas, area='image'):
    """
    Create a mask for the detector area

    Args:
        detector_areas (dict): a dictionary of image constants
        area (str): the area of the detector to create a mask for
    Returns:
        np.ndarray: a mask for the detector area
    """
    mask = np.zeros((detector_areas['frame_rows'], detector_areas['frame_cols']), dtype=bool)
    mask[detector_areas[area]['r0c0'][0]:detector_areas[area]['r0c0'][0] + detector_areas[area]['rows'],
            detector_areas[area]['r0c0'][1]:detector_areas[area]['r0c0'][1] + detector_areas[area]['cols']] = True
    return mask

def make_detector_areas(detector_areas, areas=('image', 'prescan', 'prescan_reliable',
        'parallel_overscan', 'serial_overscan')):
    """
    Create a dictionary of masks for the different detector areas

    Args:
        detector_areas (dict): a dictionary of image constants
        areas (tuple): a tuple of areas to create masks for

    Returns:
        dict: a dictionary of masks for the different detector areas
    """
    detector_areas = {}
    for area in areas:
        detector_areas[area] = detector_area_mask(detector_areas, area=area)
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
