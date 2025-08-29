# A file that holds the functions that transmogrify l2a data to l2b data
import numpy as np
from scipy.interpolate import interp1d
import copy
import warnings
import corgidrp.data as data
from corgidrp.darks import build_synthesized_dark
from corgidrp.detector import detector_areas, ENF


def add_shot_noise_to_err(input_dataset, kgain, detector_params):
    """
    Propagate the Poisson/shot noise determined from the image signal to the error map. 
    Estimation of photon/poisson/shot noise by interpolation of the photon transfer curve,
    added excess noise in case of em_gain > 1.
    Especially useful when a dataset has very few frames so that the shot noise is not
    accurately obtained by averaging the frames.

    Args:
       input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)
       kgain (corgidrp.data.KGain): kgain calibration object
       detector_params (corgidrp.data.DetectorParams): detector parameters calibration object

    Returns:
        corgidrp.data.Dataset: shot noise propagated to the image error extensions of the input dataset
    """
    # you should make a copy the dataset to start
    phot_noise_dataset = input_dataset.copy()
    
    #get the noise from the ptc curve
    ptc = kgain.ptc
    nem = detector_params.params['NEMGAIN']
    
    for i, frame in enumerate(phot_noise_dataset.frames):
        # use measured gain if available TODO change hdr name if necessary
        try:
            em_gain = frame.ext_hdr["EMGAIN_M"]
        except:
            # use EM applied EM gain if available
            em_gain = frame.ext_hdr.get("EMGAIN_A", 0)
            if em_gain > 0:
                em_gain = em_gain
            else : # otherwise use commanded EM gain
                em_gain = frame.ext_hdr.get("EMGAIN_C", 0)
                
        #estimate of photon/poisson/shot noise by interpolation of the photon transfer curve
        interp_func = interp1d(ptc[:,0], ptc[:,1], kind='linear', fill_value='extrapolate')

        phot_err = interp_func(frame.data)
        #add excess noise in case of em_gain > 1
        if em_gain > 1:
            phot_err *= ENF(em_gain, nem)           
        frame.add_error_term(phot_err, "shotnoise_error")
    
    new_all_err = np.array([frame.err for frame in phot_noise_dataset.frames])        
    
    history_msg = "Poisson/shot noise propagated to error map"
    # update the output dataset
    phot_noise_dataset.update_after_processing_step(history_msg, new_all_err = new_all_err)

    return phot_noise_dataset


def dark_subtraction(input_dataset, dark, detector_regions=None, outputdir=None):
    """

    Perform dark subtraction of a dataset using the corresponding dark frame.  The dark frame can be either a synthesized master dark (made for any given EM gain and exposure time)
    or for a traditional master dark (average of darks taken at the EM gain and exposure time of the corresponding observation).  The master dark is also saved if it is of the synthesized type after it is built.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images that need dark subtraction (L2a-level)
        dark (corgidrp.data.Dark or corgidrp.data.DetectorNoiseMaps): If dark is of the corgidrp.data.Dark type, dark subtraction will be done immediately.
            If dark is of the corgidrp.data.DetectorNoiseMaps type, a synthesized master is created using calibrated noise maps for the EM gain and exposure time used in the frames in input_dataset.
        detector_regions: (dict):  A dictionary of detector geometry properties.  Keys should be as found in detector_areas in detector.py. Defaults to detector_areas in detector.py.
        outputdir (string): Filepath for output directory where to save the master dark if it is a synthesized master dark.  Defaults to current directory.

    Returns:
        corgidrp.data.Dataset: a dark-subtracted version of the input dataset including error propagation
    """
    if input_dataset[0].ext_hdr['BUNIT'] != "detected electron":
        raise ValueError ("input dataset must have unit 'detected electron' for dark subtraction, not {0}".format(input_dataset[0].ext_hdr['BUNIT']))
    _, unique_vals = input_dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
    if len(unique_vals) > 1:
        raise Exception('Input dataset should contain frames of the same exposure time, commanded EM gain, and k gain.')

    if detector_regions is None:
        detector_regions = detector_areas
    # you should make a copy the dataset to start
    darksub_dataset = input_dataset.copy()
    rows = detector_regions['SCI']['frame_rows']
    cols = detector_regions['SCI']['frame_cols']
    im_rows = detector_regions['SCI']['image']['rows']
    im_cols = detector_regions['SCI']['image']['cols']

    if type(dark) is data.DetectorNoiseMaps:
        if input_dataset.frames[0].data.shape == (rows, cols):
            full_frame = True
        elif input_dataset.frames[0].data.shape == (im_rows, im_cols):
            full_frame = False
        else:
            raise Exception('Frames in input_dataset do not have valid SCI full-frame or image dimensions.')
        dark = build_synthesized_dark(input_dataset, dark, detector_regions=detector_regions, full_frame=full_frame)
        if outputdir is None:
            outputdir = '.' #current directory
        dark.save(filedir=outputdir)
    elif type(dark) is data.Dark:
        if 'PC_STAT' in dark.ext_hdr:
            if dark.ext_hdr['PC_STAT'] != 'analog master dark':
                raise Exception('The input \'dark\' is a photon-counted dark and cannot be used in the dark_subtraction step function, which is intended for analog frames.')
        # In this case, the Dark loaded in should already match the arry dimensions
        # of input_dataset, specified by full_frame argument of build_trad_dark
        # when this Dark was built
        if (dark.ext_hdr['EXPTIME'], dark.ext_hdr['EMGAIN_C'], dark.ext_hdr['KGAINPAR']) != unique_vals[0]:
            raise Exception('Dark should have the same EXPTIME, EMGAIN_C, and KGAINPAR as input_dataset.')
    else:
        raise Exception('dark type should be either corgidrp.data.Dark or corgidrp.data.DetectorNoiseMaps.')

    darksub_cube = darksub_dataset.all_data - dark.data

    # propagate the error of the dark frame
    if hasattr(dark, "err"):
        darksub_dataset.add_error_term(dark.err[0], "dark_error")
    else:
        raise Warning("no error attribute in the dark frame")

    if hasattr(dark, "dq"):
        new_all_dq = np.bitwise_or(darksub_dataset.all_dq, dark.dq)
    else:
        new_all_dq = None

    #darksub_dataset.all_err = np.array([frame.err for frame in darksub_dataset.frames])
    history_msg = "Dark subtracted using dark {0}.  Units changed from detected electrons to photoelectrons.".format(dark.filename)

    # update the output dataset with this new dark subtracted data and update the history
    darksub_dataset.update_after_processing_step(history_msg, new_all_data=darksub_cube, new_all_dq = new_all_dq, header_entries = {"BUNIT":"photoelectron"})

    return darksub_dataset

def flat_division(input_dataset, flat_field):
    """

    Divide the dataset by the master flat field.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)
        flat_field (corgidrp.data.FlatField): a master flat field to divide by

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the flat field divided out
    """

     # copy of the dataset
    flatdiv_dataset = input_dataset.copy()

    #Divide by the master flat
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning) # catch divide by zero
        flatdiv_cube = flatdiv_dataset.all_data /  flat_field.data

    #Find where the flat_field is 0 and set a DQ flag: 
    where_zero = np.where(flat_field.data == 0)
    flatdiv_dq = copy.deepcopy(flatdiv_dataset.all_dq)
    for i in range(len(flatdiv_dataset)):
       flatdiv_dq[i][where_zero] = np.bitwise_or(flatdiv_dataset[i].dq[where_zero], 4)

    # propagate the error of the master flat frame
    if hasattr(flat_field, "err"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning) # catch divide by zero
            flatdiv_dataset.rescale_error(1/flat_field.data, "FlatField")
            flatdiv_dataset.add_error_term(flatdiv_dataset.all_data*flat_field.err[0]/(flat_field.data**2), "FlatField_error")
    else:
        raise Warning("no error attribute in the FlatField")

    history_msg = "Flat calibration done using Flat field {0}".format(flat_field.filename)

    # update the output dataset with this new flat calibrated data and update the history
    flatdiv_dataset.update_after_processing_step(history_msg,new_all_data=flatdiv_cube, new_all_dq = flatdiv_dq)

    return flatdiv_dataset

def frame_select(input_dataset, bpix_frac=1., allowed_bpix=0, overexp=False, tt_rms_thres=None, tt_bias_thres=None, discard_bad=True):
    """

    Selects the frames that we want to use for further processing.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)
        bpix_frac (float): greater than fraction of the image needs to be bad to discard. Default: 1.0 (not used)
        allowed_bpix (int): sum of DQ values that are allowed and not counted towards to bpix fraction
                            (e.g., 6 means 2 and 4 are not considered bad).
                            Default is 0 (all nonzero DQ flags are considered bad)
        overexp (bool): if True, removes frames where the OVEREXP keyword is True. Default: False
        tt_rms_thres (float): maximum allowed RMS tip or tilt in image to be considered good. Default: None (not used)
        tt_bias_thres (float): maximum allowed bias in tip/tilt over the course of an image to be consdiered good. Default: None (not used)
        discard_bad (bool): if True, drops the bad frames rather than keeping them through processing
        
    Returns:
        corgidrp.data.Dataset: a version of the input dataset with only the frames we want to use
    """
    pruned_dataset = input_dataset.copy()
    reject_flags = np.zeros(len(input_dataset))
    reject_reasons = {}

    disallowed_bits = np.invert(allowed_bpix) # invert the mask

    for i, frame in enumerate(pruned_dataset.frames):
        reject_reasons[i] = [] # list of rejection reasons
        if bpix_frac < 1:
            masked_dq = np.bitwise_and(frame.dq, disallowed_bits) # handle allowed_bpix values
            numbadpix = np.size(np.where(masked_dq > 0)[0])
            frame_badpix_frac = numbadpix / np.size(masked_dq)
            # if fraction of bad pixel over threshold, mark is as bad
            if frame_badpix_frac > bpix_frac:
                reject_flags[i] += 1
                reject_reasons[i].append("bad pix frac {0:.5f} > {1:.5f}"
                                         .format(frame_badpix_frac, bpix_frac))
        frame.ext_hdr['FRMSEL01'] = (bpix_frac, "Bad Pixel Fraction < This Value. Doesn't include DQflags summed to {0}".format(allowed_bpix)) # record selection criteria

        if overexp:
            if frame.ext_hdr['OVEREXP']:
                reject_flags[i] += 2 # use distinct bits in case it's useful
                reject_reasons[i].append("OVEREXP = T")
        frame.ext_hdr['FRMSEL02'] = (overexp, "Are we selecting on the OVEREXP flag?") # record selection criteria

        if tt_rms_thres is not None:
            if frame.ext_hdr['Z2VAR'] > tt_rms_thres:
                reject_flags[i] += 4 # use distinct bits in case it's useful
                reject_reasons[i].append("tip rms (Z2VAR) {0:.1f} > {1:.1f}"
                                         .format(frame.ext_hdr['Z2VAR'], tt_rms_thres))
            if frame.ext_hdr['Z3VAR'] > tt_rms_thres:
                reject_flags[i] += 8 # use distinct bits in case it's useful
                reject_reasons[i].append("tilt rms (Z3VAR) {0:.1f} > {1:.1f}"
                                         .format(frame.ext_hdr['Z3VAR'], tt_rms_thres))
        frame.ext_hdr['FRMSEL03'] = (tt_rms_thres, "tip rms (Z2VAR) threshold") # record selection criteria
        frame.ext_hdr['FRMSEL04'] = (tt_rms_thres, "tilt rms (Z3VAR) threshold") # record selection criteria
        
        if tt_bias_thres is not None:
            if frame.ext_hdr['Z2RES'] > tt_bias_thres:
                reject_flags[i] += 16 # use distinct bits in case it's useful
                reject_reasons[i].append("tip bias (Z2RES) {0:.1f} > {1:.1f}"
                                         .format(frame.ext_hdr['Z2RES'], tt_bias_thres))
            if frame.ext_hdr['Z3RES'] > tt_bias_thres:
                reject_flags[i] += 32 # use distinct bits in case it's useful
                reject_reasons[i].append("tilt bias (Z3RES) {0:.1f} > {1:.1f}"
                                         .format(frame.ext_hdr['Z3RES'], tt_bias_thres))
        frame.ext_hdr['FRMSEL05'] = (tt_bias_thres, "tip bias (Z2RES) threshold") # record selection criteria
        frame.ext_hdr['FRMSEL06'] = (tt_bias_thres, "tilt bias (Z3RES) threshold") # record selection criteria
                
        # if rejected, mark as bad in the header
        if reject_flags[i] > 0:
            frame.ext_hdr['IS_BAD'] = True
                                
    good_frames = np.where(reject_flags == 0)
    bad_frames = np.where(reject_flags > 0)
    # check that we didn't remove all of the good frames
    if np.size(good_frames) == 0:
        raise ValueError("No good frames were selected. Unable to continue")

    # if we need to discard bad, do that here. 
    if discard_bad:
        pruned_frames = pruned_dataset.frames[good_frames]
        pruned_dataset = data.Dataset(pruned_frames)
        
        # history message of which frames were removed and why
        history_msg = "Removed {0} frames as bad:".format(np.size(bad_frames))
    else:
        # history message of which frames were marked and why
        history_msg = "Marked {0} frames as bad:".format(np.size(bad_frames))

    for bad_index in bad_frames[0]:
        bad_frame = input_dataset.frames[bad_index]
        bad_reasons = "; ".join(reject_reasons[bad_index])
        history_msg += " {0} ({1}),".format(bad_frame.filename, bad_reasons)
    history_msg = history_msg[:-1] # remove last comma or :

    pruned_dataset.update_after_processing_step(history_msg)

    return pruned_dataset

def convert_to_electrons(input_dataset, k_gain):
    """

    Convert the data from ADU to electrons.
    TODO: Establish the interaction with the CalDB for obtaining gain calibration

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)
        k_gain (corgidrp.data.KGain): KGain calibration file

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in electrons
    """
    if input_dataset[0].ext_hdr['BUNIT'] != "DN":
        raise ValueError("input dataset must have unit DN for the conversion, not {0}".format(input_dataset[0].ext_hdr['BUNIT']))
    # you should make a copy the dataset to start
    kgain_dataset = input_dataset.copy()
    kgain_cube = kgain_dataset.all_data

    kgain = k_gain.value #extract from caldb
    error_frame = kgain_cube * k_gain.error
    kgain_cube *= kgain
    
    #scale also the old error with kgain and propagate the error 
    kgain_dataset.rescale_error(kgain, "kgain") 
    kgain_dataset.add_error_term(error_frame, "kgain_error")

    history_msg = "data converted to detected EM electrons by kgain {0}".format(str(kgain))

    # update the output dataset with this converted data and update the history
    kgain_dataset.update_after_processing_step(history_msg, new_all_data=kgain_cube, header_entries = {"BUNIT":"detected EM electron", "KGAINPAR":kgain, 
                                               "KGAIN_ER": k_gain.error, "RN":k_gain.ext_hdr['RN'], "RN_ERR":k_gain.ext_hdr["RN_ERR"]})
    return kgain_dataset

def em_gain_division(input_dataset):
    """

    Convert the data from detected EM electrons to detected electrons by dividing by the EM gain.
    Update the change in units in the header [detected electrons].

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the data in units "detected electrons"
    """
    if input_dataset[0].ext_hdr['BUNIT'] != "detected EM electron":
        raise ValueError("input dataset must have unit 'detected EM electron' for the conversion, not {0}".format(input_dataset[0].ext_hdr['BUNIT']))
    # you should make a copy the dataset to start
    emgain_dataset = input_dataset.copy()
    emgain_cube = emgain_dataset.all_data
    emgain_error = emgain_dataset.all_err

    for i in range(len(emgain_dataset)):
        try: # use measured gain if available
            emgain = emgain_dataset[i].ext_hdr["EMGAIN_M"]
        except:
            emgain = emgain_dataset[i].ext_hdr["EMGAIN_A"]
            if emgain > 0: # use EM applied EM gain if available
                emgain = emgain_dataset[i].ext_hdr["EMGAIN_A"]
            else: # otherwise use commanded EM gain
                emgain = emgain_dataset[i].ext_hdr["EMGAIN_C"]
        emgain_cube[i] /= emgain
        emgain_error[i] /= emgain

    dataset_list, _ = emgain_dataset.split_dataset(exthdr_keywords=['EMGAIN_C'])
    if len(dataset_list) > 1:
        history_msg = "data divided by EM gain for dataset with frames with different commanded EM gains"
    else:
        history_msg = "data divided by EM gain for dataset with frames with the same commanded EM gain"

    # update the output dataset with this EM gain divided data and update the history
    emgain_dataset.update_after_processing_step(history_msg, new_all_data=emgain_cube, new_all_err=emgain_error, header_entries = {"BUNIT":"detected electron"})

    return emgain_dataset

def cti_correction(input_dataset, pump_trap_cal):
    """

    Apply the CTI correction to the dataset.

    Currently a no-op step function. 

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)
        pump_trap_cal (corgidrp.data.TrapCalibration): Pump trap calibration file

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with the CTI correction applied
    """
    # also remember to update CTI_CORR ext header keyword
    return input_dataset.copy()


def correct_bad_pixels(input_dataset, bp_mask):

    """

    Correct for bad pixels: Bad pixels are identified as part of the data
        calibration. This function replaces bad pixels by NaN values. It also
        updates its DQ storing the type of bad pixel at each bad pixel location,
        and it records the fact that the pixel has been replaced by NaN.

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)
        bp_mask (corgidrp.data.BadPixelMap): Bad-pixel mask built from the bad
        pixel calibration file.


    Returns:
        corgidrp.data.Dataset: a version of the input dataset with bad detector
        pixels and cosmic rays replaced by NaNs

    """

    data = input_dataset.copy()
    data_cube = data.all_data
    dq_cube = data.all_dq.astype(np.uint8)

    for i in range(data_cube.shape[0]):
        # combine DQ and BP masks
        bp_dq_mask = np.bitwise_or(dq_cube[i],bp_mask.data.astype(np.uint8))
        # mask affected pixels with NaN
        bp = np.where(bp_dq_mask != 0)
        data_cube[i, bp[0], bp[1]] = np.nan
        # Update DQ to keep track of replaced bad pixel values
        bp_dq_mask[bp[0], bp[1]]=np.bitwise_or(bp_dq_mask[bp[0], bp[1]], 2)
        dq_cube[i] = bp_dq_mask

    history_msg = "removed pixels affected by bad pixels"
    data.update_after_processing_step(history_msg, new_all_data=data_cube,
        new_all_dq=dq_cube)

    return data

def desmear(input_dataset, detector_params):
    """
    EXCAM has no shutter, and so continues to illuminate the detector during
    readout. This creates a "smearing" effect into the resulting images. The
    desmear function corrects for this effect. There are a small number of use
    cases for not desmearing data (e.g. time-varying raster data).

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)
        detector_params (corgidrp.data.DetectorParams): a calibration file storing detector calibration values

    Returns:
        corgidrp.data.Dataset: a version of the input dataset with desmear applied

    """

    data = input_dataset.copy()
    data_cube = data.all_data

    rowreadtime_sec = detector_params.params['ROWREADT']

    for i in range(data_cube.shape[0]):
        exptime_sec = float(data[i].ext_hdr['EXPTIME'])
        smear = np.zeros_like(data_cube[i])
        m = len(smear)
        for r in range(m):
            columnsum = 0
            for s in range(r+1):
                columnsum = columnsum + rowreadtime_sec/exptime_sec*((1
                + rowreadtime_sec/exptime_sec)**((s+1)-(r+1)-1))*data_cube[i,s,:]
            smear[r,:] = columnsum
        data_cube[i] -= smear

    history_msg = "Desmear applied to data"
    header_update = {'DESMEAR' : True}
    data.update_after_processing_step(history_msg, new_all_data=data_cube, header_entries=header_update)

    return data

def update_to_l2b(input_dataset):
    """
    Updates the data level to L2b. Only works on L2a data.

    Currently only checks that data is at the L2a level

    Args:
        input_dataset (corgidrp.data.Dataset): a dataset of Images (L2a-level)

    Returns:
        corgidrp.data.Dataset: same dataset now at L2b-level
    """
    # check that we are running this on L1 data
    for orig_frame in input_dataset:
        if orig_frame.ext_hdr['DATALVL'] != "L2a":
            err_msg = "{0} needs to be L2a data, but it is {1} data instead".format(orig_frame.filename, orig_frame.ext_hdr['DATALVL'])
            raise ValueError(err_msg)

    # we aren't altering the data
    updated_dataset = input_dataset.copy(copy_data=False)

    for frame in updated_dataset:
        # update header
        frame.ext_hdr['DATALVL'] = "L2b"
        # update filename convention. The file convention should be
        # "CGI_[dataleel_*]" so we should be same just replacing the just instance of L1
        frame.filename = frame.filename.replace("_l2a", "_l2b", 1)

    history_msg = "Updated Data Level to L2b"
    updated_dataset.update_after_processing_step(history_msg)

    return updated_dataset
