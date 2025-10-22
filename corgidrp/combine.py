"""
Module to support frame combination
"""
import warnings
import numpy as np
import corgidrp.data as data


def combine_images(data_subset, err_subset, dq_subset, collapse, num_frames_scaling):
    """
    Combines several images together

    Args:
        data_subset (np.array): 3-D array of N 2-D images
        err_subset (np.array): 4-D array of N 2-D error maps
        dq_subset (np.array): 3-D array of N 2-D DQ maps
        collapse (str): "mean" or "median". 
        num_frames_scaling (bool): Multiply by number of frames in sequence in order to ~conserve photons

    Returns:
        np.array: 2-D array of combined images
        np.array: 3-D array of combined error map
        np.array: 2-D array of combined DQ maps
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

    return data_collapse, err_collapse, dq_collapse



def combine_subexposures(input_dataset, num_frames_per_group=None, collapse="mean", num_frames_scaling=True):
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

        data_collapse, err_collapse, dq_collapse = combine_images(data_subset, err_subset, dq_subset, collapse=collapse, 
                                                                  num_frames_scaling=num_frames_scaling)

        # grab the headers from the first frame in this sub sequence
        pri_hdr = input_dataset[num_frames_per_group*i].pri_hdr.copy()
        ext_hdr = input_dataset[num_frames_per_group*i].ext_hdr.copy()
        ext_hdr["NUM_FR"] = num_frames_per_group
        err_hdr = input_dataset[num_frames_per_group*i].err_hdr.copy()
        dq_hdr = input_dataset[num_frames_per_group*i].dq_hdr.copy()
        hdulist = input_dataset[num_frames_per_group*i].hdu_list.copy()
        new_image = data.Image(data_collapse, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err=err_collapse, dq=dq_collapse, err_hdr=err_hdr, 
                                dq_hdr=dq_hdr, input_hdulist=hdulist)
                                
        # always take the last filename in the group for the combined frame
        last_idx_in_group = num_frames_per_group*(i+1) - 1
        new_image.filename = input_dataset[last_idx_in_group].filename   

        new_image._record_parent_filenames(input_dataset[num_frames_per_group*i:num_frames_per_group*(i+1)])   
        new_dataset.append(new_image)
    new_dataset = data.Dataset(new_dataset)
    new_dataset.update_after_processing_step("Combine_subexposures: combined every {0} frames by {1}".format(num_frames_per_group, collapse))

    return new_dataset