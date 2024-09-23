# A file that holds the functions that transmogrify l1 data to l2a data
from corgidrp.detector import get_relgains, slice_section, detector_areas, flag_cosmics, calc_sat_fwc, imaging_slice, imaging_area_geom
import numpy as np
import corgidrp.data as data

def prescan_biassub(input_dataset, noise_maps=None, return_full_frame=False, 
                    detector_regions=None, use_imaging_area = False):
    """
    Measure and subtract the median bias in each row of the pre-scan detector region.
    This step also crops the images to just the science area, or
    optionally returns the full detector frames.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L1a-level)
        noise_maps (corgidrp.data.DetectorNoiseMaps): the bias offset (an offset value to be subtracted from the bias) is extracted from this calibration class instance.
        If None, a default value of 0 is used for the bias offset and its error.
        return_full_frame (bool): flag indicating whether to return the full frame or
            only the bias-subtracted image area. Defaults to False.
        detector_regions: (dict):  A dictionary of detector geometry properties.
            Keys should be as found in detector_areas in detector.py. Defaults to detector_areas in detector.py.
        use_imaging_area (bool): flag indicating whether to use the imaging area (like in the trap pump code) or use the defualt (equivalent to EMCCDFrame)

    Returns:
        corgidrp.data.Dataset: a pre-scan bias subtracted version of the input dataset
    """
    # Make a copy of the input dataset to operate on
    output_dataset = input_dataset.copy()

    if detector_regions is None:
        detector_regions = detector_areas

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
        obstype = frame.ext_hdr['ARRTYPE']
        if not obstype in ['SCI','ENG','ENG_EM','ENG_CONV'] :
                raise Exception(f"Observation type of frame {i} is not 'SCI' or 'ENG' or 'ENG_EM' or 'EMG_CONV'")

        if detector_regions[obstype]['frame_rows'] != frame_data.shape[0] or detector_regions[obstype]['frame_cols'] != frame_data.shape[1]:
            raise Exception('Frame size incompatible with specified detector_regions.')
        # Get the reliable prescan area
        prescan = slice_section(frame_data, obstype, 'prescan', detector_regions=detector_regions)

        if not return_full_frame:
            # Get the image area
            if use_imaging_area: 
                image_data = imaging_slice(obstype, frame_data, detector_regions=detector_regions)
                image_dq = imaging_slice(obstype, frame_dq, detector_regions=detector_regions)

                image_err = []
                for err_slice in frame_err:
                    image_err.append(imaging_slice(obstype, err_slice, detector_regions=detector_regions))
                image_err = np.array(image_err)

                prows, _, r0c0 = imaging_area_geom(obstype,detector_regions=detector_regions)
                i_r0 = r0c0[0]
                p_r0 = detector_regions[obstype]['prescan']['r0c0'][0]
                al_prescan = prescan[(i_r0-p_r0):(i_r0-p_r0+prows), :]

            else: 
                image_data = slice_section(frame_data, obstype, 'image', detector_regions)
                image_dq = slice_section(frame_dq, obstype, 'image', detector_regions)

                # Special treatment for 3D error array
                image_err = []
                for err_slice in frame_err:
                    image_err.append(slice_section(err_slice, obstype, 'image', detector_regions))
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

        st = detector_regions[obstype]['prescan']['col_start']
        end = detector_regions[obstype]['prescan']['col_end']

        # Measure bias and error (standard error of the median for each row, add this to 3D image array)
        medbyrow = np.median(al_prescan[:,st:end], axis=1)[:, np.newaxis]
        sterrbyrow = np.std(al_prescan[:,st:end], axis=1)[:, np.newaxis] * np.ones_like(image_data) / np.sqrt(al_prescan[:,st:end].shape[1])
        if noise_maps is not None:
            bias_offset = noise_maps.bias_offset
            bias_offset_err = noise_maps.bias_offset_err
        else:
            bias_offset = 0
            bias_offset_err = 0
        sterrbyrow = np.sqrt(sterrbyrow**2 + bias_offset_err**2)
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
        # frame.bias = out_frames_bias_arr[i]
        frame.add_extension_hdu("BIAS",data=out_frames_bias_arr[i])

    # Add new error component from this step to each frame using the Dataset class method
    output_dataset.add_error_term(np.array(new_err_list),"prescan_bias_sub")

    history_msg = "Frames cropped and bias subtracted" if not return_full_frame else "Bias subtracted"

    # update the output dataset with this new dark subtracted data and update the history
    output_dataset.update_after_processing_step(history_msg)

    return output_dataset

def detect_cosmic_rays(input_dataset, detector_params, sat_thresh=0.7,
                       plat_thresh=0.7, cosm_filter=1, cosm_box=3, cosm_tail=10,
                       mode='image'):
    """
    Detects cosmic rays in a given dataset. Updates the DQ to reflect the pixels that are affected.
    TODO: (Eventually) Decide if we want to invest time in improving CR rejection (modeling and subtracting the hit
    and tail rather than just flagging the whole row.)
    TODO: Decode incoming DQ mask to avoid double counting saturation/CR flags in case a similar custom step has been run beforehand.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need cosmic ray identification (L1-level)
        detector_params (corgidrp.data.DetectorParams): a calibration file storing detector calibration values
        sat_thresh (float):
            Multiplication factor for the pixel full-well capacity (fwc) that determines saturated cosmic
            pixels. Interval 0 to 1, defaults to 0.7. Lower numbers are more aggressive in flagging saturation.
        plat_thresh (float):
            Multiplication factor for pixel full-well capacity (fwc) that determines edges of cosmic
            plateau. Interval 0 to 1, defaults to 0.7. Lower numbers are more aggressive in flagging cosmic
            ray hits.
        cosm_filter (int):
            Minimum length in pixels of cosmic plateaus to be identified. Defaults to 1.
        cosm_box (int):
            Number of pixels out from an identified cosmic head (i.e., beginning of
            the plateau) to mask out.
            For example, if cosm_box is 3, a 7x7 box is masked,
            with the cosmic head as the center pixel of the box. Defaults to 3.
        cosm_tail (int):
            Number of pixels in the row downstream of the end of a cosmic plateau
            to mask.  If cosm_tail is greater than the number of
            columns left to the end of the row from the cosmic
            plateau, the cosmic masking ends at the end of the row. Defaults to 10.
        mode (string):
            If 'image', an image-area input is assumed, and if the input
            tail length is longer than the length to the end of the image-area row,
            the mask is truncated at the end of the row.
            If 'full', a full-frame input is assumed, and if the input tail length
            is longer than the length to the end of the full-frame row, the masking
            continues onto the next row.  Defaults to 'image'.

    Returns:
        corgidrp.data.Dataset:
            A version of the input dataset of the input dataset where the cosmic rays have been identified.
    """
    sat_dqval = 32 # DQ value corresponding to full well saturation
    cr_dqval = 128 # DQ value corresponding to CR hit

    # you should make a copy the dataset to start
    crmasked_dataset = input_dataset.copy()

    crmasked_cube = crmasked_dataset.all_data


    # Calculate the full well capacity for every frame in the dataset
    kgain = np.array([detector_params.params['kgain'] for frame in crmasked_dataset])
    emgain_list = []
    for frame in crmasked_dataset:
        try: # use measured gain if available TODO change hdr name if necessary
            emgain = frame.ext_hdr['EMGAIN_M']
        except:
            try: # use applied EM gain if available
                emgain = frame.ext_hdr['EMGAIN_A']
            except: # otherwise use commanded EM gain
                emgain = frame.ext_hdr['CMDGAIN']
        emgain_list.append(emgain)
    emgain_arr = np.array(emgain_list)
    fwcpp_e_arr = np.array([detector_params.params['fwc_pp'] for frame in crmasked_dataset])
    fwcem_e_arr = np.array([detector_params.params['fwc_em'] for frame in crmasked_dataset])

    fwcpp_dn_arr = fwcpp_e_arr / kgain
    fwcem_dn_arr = fwcem_e_arr / kgain

    # pick the FWC that will get saturated first, depending on gain
    sat_fwcs = calc_sat_fwc(emgain_arr,fwcpp_dn_arr,fwcem_dn_arr,sat_thresh)

    for i,frame in enumerate(crmasked_dataset):
        frame.ext_hdr['FWC_PP_E'] = fwcpp_e_arr[i]
        frame.ext_hdr['FWC_EM_E'] = fwcem_e_arr[i]
        frame.ext_hdr['SAT_DN'] = sat_fwcs[i]

    sat_fwcs_array = np.array([np.full_like(crmasked_cube[0],sat_fwcs[i]) for i in range(len(sat_fwcs))])

    # threshold the frame to catch any values above sat_fwc --> this is
    # mask 1
    m1 = (crmasked_cube >= sat_fwcs_array) * sat_dqval

    # run remove_cosmics() with fwc=fwc_em since tails only come from
    # saturation in the gain register --> this is mask 2
    # Do a for loop since it's calling a for loop in the sub-routine anyway
    # and can't handle different 'FWC_EM's for different frames.
    m2 = np.zeros_like(crmasked_cube)

    for i in range(len(crmasked_cube)):
        m2[i,:,:] = flag_cosmics(cube=crmasked_cube[i:i+1,:,:],
                        fwc=fwcem_dn_arr[i],
                        sat_thresh=sat_thresh,
                        plat_thresh=plat_thresh,
                        cosm_filter=cosm_filter,
                        cosm_box=cosm_box,
                        cosm_tail=cosm_tail,
                        mode=mode
                        ) * cr_dqval

    # add the two masks to the all_dq mask
    new_all_dq = np.bitwise_or(crmasked_dataset.all_dq, m1)
    new_all_dq =  np.bitwise_or(new_all_dq, m2.astype(int))

    history_msg = ("Cosmic ray mask created. "
                   "Used detector parameters from {0}"
                   "with hash {1}").format(detector_params.filename, detector_params.get_hash())

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
    if "CMDGAIN" not in linearized_dataset[0].ext_hdr.keys():
        raise ValueError("EM gain not found in header of input dataset. Non-linearity correction requires EM gain to be in header.")

    for i in range(linearized_cube.shape[0]):
        try: # use measured gain if available TODO change hdr name if necessary
            em_gain = linearized_dataset[i].ext_hdr["EMGAIN_M"]
        except:
            try: # use applied EM gain if available
                em_gain = linearized_dataset[i].ext_hdr["EMGAIN_A"]
            except: # otherwise use commanded EM gain
                em_gain = linearized_dataset[i].ext_hdr["CMDGAIN"]
        linearized_cube[i] *= get_relgains(linearized_cube[i], em_gain, non_lin_correction)
    
    if non_lin_correction is not None:
        history_msg = "Data corrected for non-linearity with {0}".format(non_lin_correction.filename)

        linearized_dataset.update_after_processing_step(history_msg, new_all_data=linearized_cube)

    return linearized_dataset

def update_to_l2a(input_dataset):
    """
    Updates the data level to L2a. Only works on L1 data.

    Currently only checks that data is at the L1 level

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L1-level)

    Returns:
        corgidrp.data.Dataset: same dataset now at L2-level
    """
    # check that we are running this on L1 data
    for orig_frame in input_dataset:
        if orig_frame.ext_hdr['DATA_LEVEL'] != "L1":
            err_msg = "{0} needs to be L1 data, but it is {1} data instead".format(orig_frame.filename, orig_frame.ext_hdr['DATA_LEVEL'])
            raise ValueError(err_msg)

    # we aren't altering the data
    updated_dataset = input_dataset.copy(copy_data=False)

    for frame in updated_dataset:
        # update header
        frame.ext_hdr['DATA_LEVEL'] = "L2a"
        # update filename convention. The file convention should be
        # "CGI_[dataleel_*]" so we should be same just replacing the just instance of L1
        frame.filename = frame.filename.replace("_L1_", "_L2a_", 1)

    history_msg = "Updated Data Level to L2a"
    updated_dataset.update_after_processing_step(history_msg)

    return updated_dataset
