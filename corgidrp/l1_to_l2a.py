# A file that holds the functions that transmogrify l1 data to l2a data 
from corgidrp.detector import get_relgains, slice_section, detector_areas, flag_cosmics
import numpy as np


def prescan_biassub(input_dataset, bias_offset=0., return_full_frame=False):
    """
    Measure and subtract the median bias in each row of the pre-scan detector region. 
    This step also crops the images to just the science area, or 
    optionally returns the full detector frames.


    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L1a-level)
        bias_offset (float): an offset value to be subtracted from the bias. Defaults to 0.
        return_full_frame (bool): flag indicating whether to return the full frame or 
            only the bias-subtracted image area. Defaults to False.
    
    Returns:
        corgidrp.data.Dataset: a pre-scan bias subtracted version of the input dataset
    """
    # Make a copy of the input dataset to operate on
    output_dataset = input_dataset.copy()

    # Initialize list of output frames to be concatenated
    out_frames_data = []
    out_frames_err = []
    out_frames_dq = []
    out_frames_bias = []

    # Place to save new error estimates to be added later via Image.add_error_term()
    new_err_list = []

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
            image_dq = slice_section(frame_dq, obstype, 'image')
            
            # Special treatment for 3D error array
            image_err = []
            for err_slice in frame_err: 
                image_err.append(slice_section(err_slice, obstype, 'image'))
            image_err = np.array(image_err)

            # Get the part of the prescan that lines up with the image
            i_r0 = detector_areas[obstype]['image']['r0c0'][0]
            p_r0 = detector_areas[obstype]['prescan']['r0c0'][0]
            i_nrow = detector_areas[obstype]['image']['rows']
            al_prescan = prescan[(i_r0-p_r0):(i_r0-p_r0+i_nrow), :]    
            
        else:
            # Use full frame
            image_data = frame_data
            image_dq = frame_dq

            # Special treatment for 3D error array
            image_err = []
            for err_slice in frame_err: 
                image_err.append(err_slice)
            image_err = np.array(image_err)

            al_prescan = prescan

        # Measure bias and error (standard error of the median for each row, add this to 3D image array)
        medbyrow = np.median(al_prescan, axis=1)[:, np.newaxis]
        sterrbyrow = np.std(al_prescan, axis=1)[:, np.newaxis] * np.ones_like(image_data) / np.sqrt(al_prescan.shape[1])
        new_err_list.append(sterrbyrow)   
            

        bias = medbyrow - bias_offset
        image_bias_corrected = image_data - bias

        out_frames_data.append(image_bias_corrected)
        out_frames_err.append(image_err)
        out_frames_dq.append(image_dq)
        out_frames_bias.append(bias[:,0]) # save 1D version of array

        # Update header with new frame dimensions
        frame.ext_hdr['NAXIS1'] = image_bias_corrected.shape[1]
        frame.ext_hdr['NAXIS2'] = image_bias_corrected.shape[0]
    
    # Update all_data and reassign frame pointers (only necessary because the array size has changed)
    out_frames_data_arr = np.array(out_frames_data)
    out_frames_err_arr = np.array(out_frames_err)
    out_frames_dq_arr = np.array(out_frames_dq)
    out_frames_bias_arr = np.array(out_frames_bias, dtype=np.float32)

    output_dataset.all_data = out_frames_data_arr
    output_dataset.all_err = out_frames_err_arr
    output_dataset.all_dq = out_frames_dq_arr

    for i,frame in enumerate(output_dataset):
        frame.data = out_frames_data_arr[i]
        frame.err = out_frames_err_arr[i]
        frame.dq = out_frames_dq_arr[i]
        frame.bias = out_frames_bias_arr[i]
        
    # Add new error component from this step to each frame using the Dataset class method
    output_dataset.add_error_term(np.array(new_err_list),"prescan_bias_sub")

    history_msg = "Frames cropped and bias subtracted" if not return_full_frame else "Bias subtracted"

    # update the output dataset with this new dark subtracted data and update the history
    output_dataset.update_after_processing_step(history_msg)

    return output_dataset

def detect_cosmic_rays(input_dataset, sat_thresh=0.99, plat_thresh=0.85, cosm_filter=2):
    """
    Detects cosmic rays in a given dataset. Updates the DQ to reflect the pixels that are affected. 
    TODO: Decide if we want this step to optionally compensate for them, or if that's a different step. 
    TODO: Decide if we want to invest time in improving CR rejection (modeling and subtracting the hit 
    and tail rather than flagging the whole row.)
    TODO: Decode incoming DQ mask to avoid double counting saturation/CR flags in case a similar custom step has been run beforehand.
    TODO: Enable processing of datasets where each frame has a different saturation threshold (determined by em_gain, fwc_em,fwc_pp)
    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need cosmic ray identification (L1-level)
        fwc_em (float): 
            Detector EM gain register full well capacity (DN).
        fwc_pp (float): 
            Detector image area per-pixel full well capacity (DN).
        fsat_thresh (float): 
            Multiplication factor for fwc that determines saturated cosmic
            pixels. Interval 0 to 1, defaults to 0.99.
        plat_thresh (float): 
            Multiplication factor for fwc that determines edges of cosmic
            plateau. Interval 0 to 1, defaults to 0.85
        cosm_filter (int): 
            Minimum length in pixels of cosmic plateus to be identified. Defaults to 2
    Returns:
        corgidrp.data.Dataset: a version of the input dataset of the input dataset where the cosmic rays have been identified. 
    """
    sat_dqval = 32 # DQ value corresponding to full well saturation
    cr_dqval = 128 # DQ value corresponding to CR hit

    # you should make a copy the dataset to start
    crmasked_dataset = input_dataset.copy()

    crmasked_cube = crmasked_dataset.all_data

    # Assert that full well capacity is the same for every frame in the dataset
    fwc_arr = np.array([(frame.ext_hdr['EMGAIN'], frame.ext_hdr['FWC_PP'], frame.ext_hdr['FWC_EM']) for frame in crmasked_dataset])
    if len(fwc_arr.unique) > 1:
        raise ValueError("Not all Frames in the Dataset have the same FWC_EM, FWC_PP, and EMGAIN).")
    
    # pick the FWC that will get saturated first, depending on gain
    sat_fwc = sat_thresh*min(crmasked_dataset[0].ext_hdr['EMGAIN'] * crmasked_dataset[0].ext_hdr['FWC_PP'], crmasked_dataset[0].ext_hdr['FWC_EM'])

    # threshold the frame to catch any values above sat_fwc --> this is
    # mask 1
    m1 = (crmasked_cube >= sat_fwc) * sat_dqval
    
    # run remove_cosmics() with fwc=fwc_em since tails only come from
    # saturation in the gain register --> this is mask 2
    m2 = flag_cosmics(image=crmasked_cube,
                    fwc=crmasked_dataset[0].ext_hdr['FWC_EM'],
                    sat_thresh=sat_thresh,
                    plat_thresh=plat_thresh,
                    cosm_filter=cosm_filter,
                    ) * cr_dqval

    # add the two masks to the all_dq mask
    new_all_dq = crmasked_dataset.all_dq + m1 + m2

    history_msg = "Cosmic ray mask created."

    # update the output dataset with this new dark subtracted data and update the history
    crmasked_dataset.update_after_processing_step(history_msg, new_all_dq=new_all_dq)

    return crmasked_dataset

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
