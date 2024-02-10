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

def get_relgains(frame, em_gain, nonlin_path):
    """For a given bias subtracted frame of dn counts, return a same sized
    array of relative gain values.

    The required format for the file specified at nonlin_path is as follows:
     - CSV
     - Minimum 2x2
     - First value (top left) must be assigned to nan
     - Row headers (dn counts) must be monotonically increasing
     - Column headers (EM gains) must be monotonically increasing
     - Data columns (relative gain curves) must straddle 1

    For example:

    [
        [nan,  1,     10,    100,   1000 ],
        [1,    0.900, 0.950, 0.989, 1.000],
        [1000, 0.910, 0.960, 0.990, 1.010],
        [2000, 0.950, 1.000, 1.010, 1.050],
        [3000, 1.000, 1.001, 1.011, 1.060],
    ],

    where the row headers [1, 1000, 2000, 3000] are dn counts, the column
    headers [1, 10, 100, 1000] are EM gains, and the first data column
    [0.900, 0.910, 0.950, 1.000] is the first of the four relative gain curves.

    Parameters
    ----------
    frame : array_like
        Array of dn count values.
    em_gain : float
        Detector EM gain.
    nonlin_path : str
        Full path of nonlinearity calibration csv.

    Returns
    -------
    array_like
        Array of relative gain values.

    Notes
    -----
    This algorithm contains two interpolations:

     - A 2d interpolation to find the relative gain curve for a given EM gain
     - A 1d interpolation to find a relative gain value for each given dn
     count value.

    Both of these interpolations are linear, and both use their edge values as
    constant extrapolations for out of bounds values.

    """

    # Get file data
    gain_ax, count_ax, relgains = _parse_file(nonlin_path)

    # Create interpolation for em gain (x), counts (y), and relative gain (z).
    # Note that this defaults to using the edge values as fill_value for
    # out of bounds values (same as specified below in interp1d)
    f = interpolate.RectBivariateSpline(gain_ax,
                                    count_ax,
                                    relgains.T,
                                    kx=1,
                                    ky=1,
    )
    # Get the relative gain curve for the given gain value
    relgain_curve = f(em_gain, count_ax)[0]

    # Create interpolation for dn counts (x) and relative gains (y). For
    # out of bounds values use edge values
    ff = interpolate.interp1d(count_ax, relgain_curve, kind='linear',
                              bounds_error=False,
                              fill_value=(relgain_curve[0], relgain_curve[-1]))
    # For each dn count, find the relative gain
    counts_flat = ff(frame.ravel())

    return counts_flat.reshape(frame.shape)