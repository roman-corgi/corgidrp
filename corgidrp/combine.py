"""
Module to support frame combination
"""
import warnings
import numpy as np
import corgidrp.data as data
from pyklip.klip import rotate
import corgidrp

def combine_images(data_subset, err_subset, dq_subset, collapse, num_frames_scaling, other_hdus=None):
    """
    Combines several images together

    Args:
        data_subset (np.array): 3-D array of N 2-D images
        err_subset (np.array): 4-D array of N 3-D error maps
        dq_subset (np.array): 3-D array of N 2-D DQ maps
        collapse (str): "mean" or "median". 
        num_frames_scaling (bool): Multiply by number of frames in sequence in order to ~conserve photons
        other_hdus (list of HDULists, optional): list of other HDULists to be combined in the same way

    Returns:
        tuple: combined images
            np.array: 2-D array of combined images
            np.array: 3-D array of combined error map
            np.array: 2-D array of combined DQ maps
            list of np.array: list of the combined data of other HUDs
    """
    tot_frames = data_subset.shape[0]
    # mask bad pixels
    bad = np.where(dq_subset > 0)
    data_subset[bad] = np.nan
    err_subset[bad[0],:,bad[1],bad[2]] = np.nan
    # track the number of good values that go into the combination
    n_samples = np.ones(data_subset.shape)
    n_samples[bad] = 0
    n_samples = np.sum(n_samples, axis=0)
    if collapse.lower() == "mean":
        with warnings.catch_warnings():
            # prevent RuntimeWarning: Mean of empty slice
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            data_collapse = np.nanmean(data_subset, axis=0) 
            err_collapse = np.sqrt(np.nanmean(err_subset**2, axis=0)) /np.sqrt(n_samples) # correct assuming standard error propagation
    elif collapse.lower() == "median":
        with warnings.catch_warnings():
            # prevent RuntimeWarning: Mean of empty slice
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            data_collapse = np.nanmedian(data_subset, axis=0)
            err_collapse = np.sqrt(np.nanmean(err_subset**2, axis=0)) /np.sqrt(n_samples) * np.sqrt(np.pi/2) # inflate median error
    if num_frames_scaling:
        # scale up by the number of frames
        data_collapse *= tot_frames
        err_collapse *= tot_frames
    
    # dq collpase: keep all flags on
    dq_collapse = np.bitwise_or.reduce(dq_subset, axis=0)
    # except for those pixels that have been replaced with good values
    dq_collapse[np.where((dq_collapse > 0) & (~np.isnan(data_collapse)))] = 0

    # other hdus
    if other_hdus is not None:
        combined_hdus = [[] for _ in range(len(other_hdus[0]))]
        # iterate over each hdulist and append the data
        for hdul in other_hdus:
            for i, hdu in enumerate(hdul):
                combined_hdus[i].append(np.copy(hdu.data))
        # now combine each hdu data
        for i in range(len(combined_hdus)):
            # TODO: not implemented how to take means of anything beyond np.arrays (e.g., np.recarray)
            try:
                if collapse.lower() == "mean":
                    combined_hdus[i] = np.nanmean(np.array(combined_hdus[i]), axis=0)
                elif collapse.lower() == "median":
                    combined_hdus[i] = np.nanmedian(np.array(combined_hdus[i]), axis=0)
                # nothing here makes sense to scale by number of frames
                # if num_frames_scaling:
                #     combined_hdus[i] *= tot_frames
            except:
                combined_hdus[i] = combined_hdus[i][0]  # just take the first one if cannot combine
    else:
        combined_hdus = None

    return data_collapse, err_collapse, dq_collapse, combined_hdus


def combine_subexposures(input_dataset, num_frames_per_group=None, collapse="mean", num_frames_scaling=True,
                         combine_other_hdus=False):
    """
    Combines a sequence of exposures assuming a constant nubmer of frames per group. 
    The length of the dataset must be divisible by the number of frames per group.
    
    The combination is done with either the mean or median, but the collapsed image can be scaled 
    in order to ~conserve the total number of photons in the input dataset (this essentially turns a
    median into a sum)
    
    Args:
        input_dataset (corgidrp.data.Dataset): input data. 
        num_frames_per_group (int): number of subexposures per group. If None, combines all images together
        collapse (str): "mean" or "median". (default: mean) 
        num_frames_scaling (bool): Multiply by number of frames in sequence in order to ~conserve photons (default: True)
        combine_other_hdus (bool): Whether to combine other HDUs in the same way as the main data, err, DQ.
                                   Otherwise, uses the HDUs from the first frame in a subset (default: False)

    Returns:
        corgidrp.data.Dataset: dataset after combination of every "num_frames_per_group" frames together
    """
    if num_frames_per_group is None:
        num_frames_per_group = len(input_dataset)

    if len(input_dataset) % num_frames_per_group != 0:
        raise ValueError("Input dataset of length {0} cannot be grouped in sets of {1}".format(len(input_dataset), num_frames_per_group))
    
    if collapse.lower() not in ["mean", "median"]:
        raise ValueError("combine_subexposures can only collapse with mean or median")

    num_groups = len(input_dataset) // num_frames_per_group
    new_dataset = []
    for i in range(num_groups):
        data_subset = np.copy(input_dataset.all_data[num_frames_per_group*i:num_frames_per_group*(i+1)])
        err_subset = np.copy(input_dataset.all_err[num_frames_per_group*i:num_frames_per_group*(i+1)])
        dq_subset = np.copy(input_dataset.all_dq[num_frames_per_group*i:num_frames_per_group*(i+1)])

        if combine_other_hdus:
            other_hdus = [input_dataset[j].hdu_list for j in range(num_frames_per_group*i, num_frames_per_group*(i+1))]
        else:
            other_hdus = None

        data_collapse, err_collapse, dq_collapse, combined_hdus = combine_images(data_subset, err_subset, dq_subset, collapse=collapse, 
                                                                  num_frames_scaling=num_frames_scaling, other_hdus=other_hdus)


        # grab the headers from the first frame in this sub sequence
        pri_hdr = input_dataset[num_frames_per_group*i].pri_hdr.copy()
        ext_hdr = input_dataset[num_frames_per_group*i].ext_hdr.copy()
        ext_hdr["NUM_FR"] = num_frames_per_group
        err_hdr = input_dataset[num_frames_per_group*i].err_hdr.copy()
        dq_hdr = input_dataset[num_frames_per_group*i].dq_hdr.copy()
        hdulist = input_dataset[num_frames_per_group*i].hdu_list.copy()
        # update other hdus if needed
        if combine_other_hdus:
            for j, hdu in enumerate(hdulist):
                hdu.data = combined_hdus[j]
                hdu.header["NUM_FR"] = num_frames_per_group
                hdu.header['HISTORY'] = "Combined {0} frames by {1}".format(num_frames_per_group, collapse)
        new_image = data.Image(data_collapse, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err=err_collapse, dq=dq_collapse, err_hdr=err_hdr, 
                                dq_hdr=dq_hdr, input_hdulist=hdulist)
                                
        # always take the last filename in the group for the combined frame
        last_idx_in_group = num_frames_per_group*(i+1) - 1
        new_image.filename = input_dataset[last_idx_in_group].filename   

        new_image._record_parent_filenames(input_dataset[num_frames_per_group*i:num_frames_per_group*(i+1)])   
        new_dataset.append(new_image)

    new_dataset = data.Dataset(new_dataset)
    drpnfile = new_dataset[0].ext_hdr['DRPNFILE']
    # Here we change header keywords only for the combined non-coronagraphic imaging datasets
    if input_dataset[0].ext_hdr['DPAMNAME'] == 'IMAGING' and input_dataset[0].ext_hdr['LSAMNAME'] == 'OPEN':
        # average/delete header keywords as L4 involves combination of multiple frames
        pri_hdr_comb, ext_hdr_comb, _, _ = corgidrp.check.merge_headers(input_dataset, 
        last_frame_keywords=['VISITID', 'MJDEND'],
        first_frame_keywords=['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CRPIX1', 'CRPIX2'],
        deleted_keywords=['CDELT1','CDELT2','FILE0','SCTSRT','SCTEND','LOCAMT','CYCLES','LASTEXP'], #we re-add FILE0 below
        invalid_keywords=[
                        #Primary header keywords
                        'FILETIME', 'PA_V3', 'PA_APER','SVB_1', 'SVB_2', 'SVB_3', 
                        'ROLL', 'PITCH', 'YAW', 'WBJ_1', 'WBJ_2', 'WBJ_3',
                        #Extension header keywords
                        'DATETIME', 'FTIMEUTC','DATATYPE'],
        averaged_keywords=['PA_APER','EXCAMT','NOVEREXP','PROXET',
                        'FCMPOS','FSMSG1', 'FSMSG2', 'FSMSG3', 'FSMX', 'FSMY',
                        'SB_FP_DX', 'SB_FP_DY', 'SB_FS_DX', 'SB_FS_DY',
                        'Z2AVG', 'Z3AVG', 'Z4AVG', 'Z5AVG', 'Z6AVG', 'Z7AVG', 'Z8AVG', 'Z9AVG',
                        'Z10AVG', 'Z11AVG', 'Z12AVG', 'Z13AVG', 'Z14AVG',
                        'Z2RES', 'Z3RES', 'Z4RES', 'Z5RES', 'Z6RES', 'Z7RES', 'Z8RES', 'Z9RES',
                        'Z10RES', 'Z11RES',
                        'Z2VAR', 'Z3VAR'])
        # incorporate modified headers in L4 dataset
        for img in new_dataset:
            img.pri_hdr = pri_hdr_comb
            img.ext_hdr = ext_hdr_comb
            img.ext_hdr['NUM_FR'] = num_frames_per_group
            img.ext_hdr['DRPNFILE'] = drpnfile 
            img._record_parent_filenames(input_dataset)

    new_dataset.update_after_processing_step("Combine_subexposures: combined every {0} frames by {1}".format(num_frames_per_group, collapse))

    return new_dataset


def derotate_arr(data_arr,pa_aper_deg, xcen,ycen,new_center=None,astr_hdr=None,
                 is_dq=False,dq_round_threshold=0.05):
    """Derotates an array based on the provided PA_APER angle, about the provided
    center. Treats DQ arrays specially, converting to float to do the rotation, 
    and converting back to np.int64 afterwards. DQ output becomes only zeros and
    ones, so detailed DQ flag information is not preserved.

    Args:
        data_arr (np.array): an array with 2-4 dimensions
        pa_aper_deg (float): position angle (measured counter-clockwise) of the detector y axis from 
            celestial north (degrees)
        xcen (float): x-coordinate of center about which to rotate
        ycen (float): y-coordinate of center about which to rotate
        new_center (tuple, optional): tuple of x- and y- coordinate of the new center to shift to.
        astr_hdr (astropy.fits.Header, optional): WCS header which will be updated. Defaults to None.
        is_dq (bool, optional): Flag to determine if this is a DQ array. Defaults to False.
        dq_round_threshold (float, optional): value between 0-1 which determines the 
            threshold for spreading dq values to neighboring pixels after derotation.

    Returns:
        np.array: The derotated array.
    """
    # Temporarily convert dq to floats
    if is_dq:
        data_arr = data_arr.astype(np.float32)

    if data_arr.ndim == 2:
        derotated_arr = rotate(data_arr,pa_aper_deg,(xcen,ycen),
                               new_center=new_center,
                               astr_hdr=astr_hdr) # astr_hdr is corrected at above lines
    
    elif data_arr.ndim == 3:
        derotated_arr = []
        for i,im in enumerate(data_arr):
            derotated_im = rotate(im,pa_aper_deg,(xcen,ycen),
                               new_center=new_center,
                               astr_hdr=astr_hdr if (i==0) else None) # astr_hdr is corrected only once
        
            derotated_arr.append(derotated_im)

        derotated_arr = np.array(derotated_arr)
    
    elif data_arr.ndim == 4:
        derotated_arr = []
        for s,set in enumerate(data_arr):
            derotated_set = []
            for i,im in enumerate(set):
                derotated_im = rotate(im,pa_aper_deg,(xcen,ycen),
                               new_center=new_center,
                               astr_hdr=astr_hdr if (i==0 and s==0) else None) # astr_hdr is corrected only once
        
                derotated_set.append(derotated_im)
            derotated_arr.append(derotated_set)

        derotated_arr = np.array(derotated_arr)
    
    else:
        raise ValueError('derotate_arr() not configured for data with >4 dimensions')

    # convert dq_array back to ints
    if is_dq:
        derotated_arr[np.isnan(derotated_arr)] = 1 # assign nans to 1
        derotated_arr_int = (derotated_arr>dq_round_threshold).astype(np.int64)
        # import matplotlib.pyplot as plt
        # plt.imshow(derotated_arr_int,origin='lower')
        # plt.colorbar()
        # plt.title(f'round_threshold: {round_threshold}')
        # plt.show()
        return derotated_arr_int
    
    return derotated_arr


def prop_err_dq(sci_dataset,ref_dataset,mode,dq_thresh=1,new_center=None):
    """Applies logic to propagate the dq arrays and error arrays 
    in a dataset through PSF subtraction.

    Args:
        sci_dataset (corgidrp.data.Dataset): The input science dataset.
        ref_dataset (corgidrp.data.Dataset): The input reference dataset (or None if ADI only).
        mode (str): The PSF subtraction mode, e.g. "ADI", "RDI", "ADI+RDI".
        dq_thresh (int): Minimum dq flag value to be considered a bad pixel. Defaults to 1.
        new_center (tuple): New center (xy) to align all frames. Defaults to pixel closest to array center.

    Returns:
        tuple of np.array: the dq array and err array which should apply to the PSF subtraction output dataset.
    """

    # Assign master output dq & error (before derotation)
    # dq shape = (n_rolls, n_wls(optional), y, x)
    sci_input_dqs = sci_dataset.all_dq >= dq_thresh
    sci_input_errs = np.full_like(sci_dataset.all_err,np.nan) # Set errors to np.nan for now

    if new_center is None:
        new_center = [int(sci_dataset.all_data.shape[-1]//2), int(sci_dataset.all_data.shape[-2]//2)]

    # Align frames
    aligned_sci_dq_arr = []
    aligned_sci_err_arr = []
    for i,frame in enumerate(sci_dataset):
        xcen, ycen = frame.ext_hdr['STARLOCX'], frame.ext_hdr['STARLOCY']
        frame.ext_hdr['STARLOCX'], frame.ext_hdr['STARLOCY'] = new_center
        
        aligned_sci_dq = derotate_arr(sci_input_dqs[i],0, xcen,ycen,
                                      new_center=new_center,is_dq=True)
        aligned_sci_err = derotate_arr(sci_input_errs[i],0, xcen,ycen,
                                      new_center=new_center)
        
        aligned_sci_dq_arr.append(aligned_sci_dq)
        aligned_sci_err_arr.append(aligned_sci_err)
    aligned_sci_dq_arr = np.array(aligned_sci_dq_arr)
    aligned_sci_err_arr = np.array(aligned_sci_err_arr)

    if "RDI" in mode:
        ref_input_dqs = ref_dataset.all_dq >= dq_thresh
        ref_input_errs = np.full_like(ref_dataset.all_err,np.nan) # Set errors to np.nan for now


        aligned_ref_dq_arr = []
        aligned_ref_err_arr = []
        for i,frame in enumerate(ref_dataset):
            xcen, ycen = frame.ext_hdr['STARLOCX'], frame.ext_hdr['STARLOCY']

            frame.ext_hdr['STARLOCX'], frame.ext_hdr['STARLOCY'] = new_center
            
            aligned_ref_dq = derotate_arr(ref_input_dqs[i],0, xcen,ycen,
                                        new_center=new_center,is_dq=True)
            aligned_ref_err = derotate_arr(ref_input_errs[i],0, xcen,ycen,
                                        new_center=new_center)
            
            aligned_ref_dq_arr.append(aligned_ref_dq)
            aligned_ref_err_arr.append(aligned_ref_err)

        aligned_ref_dq_arr = np.array(aligned_ref_dq_arr)
        aligned_ref_err_arr = np.array(aligned_ref_err_arr)

    # If doing ADI, flag pixels that are bad in all science frames
    if 'ADI' in mode:
        aligned_sci_dq_arr[:] = np.all(aligned_sci_dq_arr,axis=0)

    # If using references, flag pixels that are bad in all the ref frames
    if 'RDI' in mode:
        ref_output_dqs_flat = np.all(aligned_ref_dq_arr,axis=0,keepdims=True)
        aligned_sci_dq_arr = np.logical_or(aligned_sci_dq_arr,ref_output_dqs_flat) 

   # Derotate dq & error
    derotated_dq_arr = []
    derotated_err_arr = []
    for i,frame in enumerate(sci_dataset):
        pa_aper_deg = frame.pri_hdr['PA_APER']
        xcen, ycen = frame.ext_hdr['STARLOCX'], frame.ext_hdr['STARLOCY']
        
        derotated_dq = derotate_arr(aligned_sci_dq_arr[i],pa_aper_deg, xcen,ycen,is_dq=True)
        derotated_err = derotate_arr(aligned_sci_err_arr[i],pa_aper_deg, xcen,ycen)
        
        derotated_dq_arr.append(derotated_dq)
        derotated_err_arr.append(derotated_err)

    # Collapse dq & error
    dq_out_collapsed = np.where(np.all(derotated_dq_arr,axis=0),1,0)
    err_out_collapsed = np.sqrt(np.sum(np.array(derotated_err_arr)**2,axis=0))

    return dq_out_collapsed, err_out_collapsed
