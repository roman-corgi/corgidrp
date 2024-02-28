# A file that holds the functions that transmogrify l1 data to l2a data 
from corgidrp.detector import get_relgains


def prescan_biassub(input_dataset, bias_offset=0., return_full_frame=False):
    """
    Perform pre-scan bias subtraction of a dataset.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L1a-level)
        bias_offset (float): an offset value to be subtracted from the bias
        return_full_frame (bool): flag indicating whether to return the full frame or only the bias-subtracted image area
    
    Returns:
        corgidrp.data.Dataset: a pre-scan bias subtracted version of the input dataset
    """
    # Make a copy of the input dataset to operate on
    output_dataset = input_dataset.copy()

    #initialize list of output frames to be concatenated
    out_frames_data = []
    out_frames_err = []
    out_frames_dq = []

    # Iterate over frames
    for i, frame in enumerate(output_dataset):

        frame_data = frame.data
        frame_err = frame.err
        frame_dq = frame.dq

        # Determine what type of file it is (engineering or science), then choose detector area dict
        obstype = frame.pri_hdr['OBSTYPE']
        if not obstype in ['SCI','ENG'] :
                raise Exception(f"Observation type of frame {i} is not 'SCI' or 'ENG'")

        # Get the reliable prescan area
        prescan = slice_section(frame_data, obstype, 'prescan_reliable')

        if not return_full_frame:
            # Get the image area
            image_data = slice_section(frame_data, obstype, 'image')
            image_err = slice_section(frame_err, obstype, 'image')
            image_dq = slice_section(frame_dq, obstype, 'image')
            
            # Get the part of the prescan that lines up with the image, and do a
            # row-by-row bias subtraction on it
            i_r0 = detector_areas[obstype]['image']['r0c0'][0]
            p_r0 = detector_areas[obstype]['prescan']['r0c0'][0]
            i_nrow = detector_areas[obstype]['image']['rows']
     
            # Get data from prescan (alined with image area)
            al_prescan = prescan[(i_r0-p_r0):(i_r0-p_r0+i_nrow), :]    
            medbyrow = np.median(al_prescan, axis=1)[:, np.newaxis]
        else:
            image_data = frame_data
            image_err = frame_err
            image_dq = frame_dq
            
            medbyrow = np.median(prescan, axis=1)[:, np.newaxis]  

        bias = medbyrow - bias_offset
        image_bias_corrected = image_data - bias

        out_frames_data.append(image_bias_corrected)
        out_frames_err.append(image_err)
        out_frames_dq.append(image_dq)

        # Update header in the dataset
        frame.ext_hdr['NAXIS1'] = image_bias_corrected.shape[1]
        frame.ext_hdr['NAXIS2'] = image_bias_corrected.shape[0]

    
    # Update all_data and reassign frame pointers (only necessary because the array size has changed)
    out_frames_data_arr = np.array(out_frames_data)
    out_frames_err_arr = np.array(out_frames_err)
    out_frames_dq_arr = np.array(out_frames_dq)

    output_dataset.all_data = out_frames_data_arr
    output_dataset.all_err = out_frames_err_arr
    output_dataset.all_dq = out_frames_dq_arr

    for i,frame in enumerate(output_dataset):
        frame.data = out_frames_data_arr[i]
        frame.err = out_frames_err_arr[i]
        frame.dq = out_frames_dq_arr[i]

            
    history_msg = "Frames cropped and bias subtracted"

    # update the output dataset with this new dark subtracted data and update the history
    output_dataset.update_after_processing_step(history_msg)

    return output_dataset

def detect_cosmic_rays(input_dataset):
    """
    
    Detects cosmis rays in a given images. Updates the DQ to reflect the pixels that are affected. 
    TODO: Decide if we want this step to optionally compensate for them, or if that's a different step. 

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need cosmic ray identification (L1-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset of the input dataset where the cosmic rays have been identified. 
    """

    return None

def correct_nonlinearity(input_dataset, non_lin_correction):
    """
    Perform non-linearity correction of a dataset using the corresponding non-linearity correction

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need non-linearity correction (L2a-level)
        non_lin_correction (corgidrp.data.NonLinearityCorrection): a NonLinearityCorrection calibration file to model the non-linearity

    Returns:
        corgidrp.data.Dataset: a non-linearity corrected version of the input dataset
    """
    #Copy the dataset to start
    linearized_dataset = input_dataset.copy()

    #Apply the non-linearity correction to the data
    linearized_cube = linearized_dataset.all_data
    #Check to see if EM gain is in the header, if not, raise an error
    if "EMGAIN" not in linearized_dataset[0].ext_hdr.keys():
        raise ValueError("EM gain not found in header of input dataset. Non-linearity correction requires EM gain to be in header.")

    em_gain = linearized_dataset[0].ext_hdr["EMGAIN"] #NOTE THIS REQUIRES THAT THE EM GAIN IS MEASURED ALREADY

    for i in range(linearized_cube.shape[0]):
        linearized_cube[i] *= get_relgains(linearized_cube[i], em_gain, non_lin_correction)

    history_msg = "Data corrected for non-linearity with {0}".format(non_lin_correction.filename)

    linearized_dataset.update_after_processing_step(history_msg, new_all_data=linearized_cube)

    return linearized_dataset
