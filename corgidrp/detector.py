# Place to put detector-related utility functions

import numpy as np
import corgidrp.data as data
from scipy import interpolate
from scipy.ndimage import median_filter

from astropy.time import Time
import os
from pathlib import Path
import yaml

class ReadMetadataException(Exception):
    """Exception class for read_metadata module."""

# Set up to allow the metadata.yaml in the repo be the default
here = Path(os.path.dirname(os.path.abspath(__file__)))
meta_path = Path(here, 'util', 'metadata.yaml')

class Metadata(object):
    """Create masks for different regions of the Roman CGI EXCAM detector.

    Parameters
    ----------
    meta_path : str
        Full path of metadta yaml.

    Attributes
    ----------
    data : dict
        Data from metadata file.
    geom : SimpleNamespace
        Geometry specific data.

    B Nemati and S Miller - UAH - 03-Aug-2018

    """

    def __init__(self, meta_path=meta_path, obstype='SCI'):
        self.meta_path = meta_path

        self.data = self.get_data()
        self.obstype = obstype
        self.frame_rows = self.data[obstype]['frame_rows']
        self.frame_cols = self.data[obstype]['frame_cols']
        self.geom = self.data[obstype]['geom']

        # Get imaging area geometry
        self.rows_im, self.cols_im, self.r0c0_im = self._imaging_area_geom()
        # Make some zeros frames for initial creation of arrays
        self.imaging_area_zeros = np.zeros((self.rows_im, self.cols_im))
        self.full_frame_zeros = np.zeros((self.frame_rows, self.frame_cols))

    def get_data(self):
        """Read yaml data into dictionary.

        Returns:
            data: data from the .yaml file
        """
        with open(self.meta_path, 'r') as stream:
            data = yaml.safe_load(stream)
        return data

    def mask(self, key):
        full_frame_m = self.full_frame_zeros.copy()

        rows, cols, r0c0 = self._unpack_geom(key)
        full_frame_m[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = 1
        return full_frame_m.astype(bool)

    def slice_section(self, frame, key):
        """Slice 2d section out of frame.

        Args;

        frame : array_like
            Full frame consistent with size given in frame_rows, frame_cols.
        key : str
            Keyword referencing section to be sliced; must exist in geom.

        Returns:

        section: array_like
            Desired slice of the frame

        """
        rows, cols, r0c0 = self._unpack_geom(key)

        section = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
        if section.size == 0:
            raise ReadMetadataException('Corners invalid')
        return section

    def _unpack_geom(self, key):
        """Safely check format of geom sub-dictionary and return values.

        Args:
        key: str
            Desired section

        Returns:
        rows: int
            Number of rows of frame
        cols : int
            Number of columns of frame
        r0c0: tuple
            Tuple of (row position, column position) of corner closest to (0,0)
        """
        coords = self.geom[key]
        rows = coords['rows']
        cols = coords['cols']
        r0c0 = coords['r0c0']

        return rows, cols, r0c0

    def _imaging_area_geom(self):
        """Return geometry of imaging area (including shielded pixels)
        in reference to full frame.  Different from normal image area.

        Returns:
        rows: int
            Number of rows of imaging area
        cols : int
            Number of columns of imaging area
        r0c0: tuple
            Tuple of (row position, column position) of corner closest to (0,0)
        """
        _, cols_pre, _ = self._unpack_geom('prescan')
        _, cols_serial_ovr, _ = self._unpack_geom('serial_overscan')
        rows_parallel_ovr, _, _ = self._unpack_geom('parallel_overscan')
        #_, _, r0c0_image = self._unpack_geom('image')
        fluxmap_rows, _, r0c0_image = self._unpack_geom('image')

        rows_im = self.frame_rows - rows_parallel_ovr
        cols_im = self.frame_cols - cols_pre - cols_serial_ovr
        r0c0_im = r0c0_image.copy()
        r0c0_im[0] = r0c0_im[0] - (rows_im - fluxmap_rows)

        return rows_im, cols_im, r0c0_im

    def imaging_slice(self, frame):
        """Select only the real counts from full frame and exclude virtual.
        Includes shielded pixels.

        Use this to transform mask and embed from acting on the full frame to
        acting on only the image frame.

        Args:
        frame: array_like
            Input frame

        Returns:
        sl: array_like
            Imaging slice

        """
        rows, cols, r0c0 = self._imaging_area_geom()
        sl = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
        return sl


def create_dark_calib(dark_dataset):
    """
    Turn this dataset of image frames that were taken to measure
    the dark current into a dark calibration frame and determines the corresponding error

    Args:
        dark_dataset (corgidrp.data.Dataset): a dataset of Image frames (L2a-level)

    Returns:
        data.Dark: a dark calibration frame
    """
    combined_frame = np.nanmean(dark_dataset.all_data, axis=0)

    new_dark = data.Dark(combined_frame, pri_hdr=dark_dataset[0].pri_hdr.copy(),
                         ext_hdr=dark_dataset[0].ext_hdr.copy(), input_dataset=dark_dataset)

    # determine the standard error of the mean: stddev/sqrt(n_frames)
    new_dark.err = np.nanstd(dark_dataset.all_data, axis=0)/np.sqrt(len(dark_dataset))
    new_dark.err = new_dark.err.reshape((1,)+new_dark.err.shape) #Get it into the right dimensions

    return new_dark

def get_relgains(frame, em_gain, non_lin_correction):
    """
    For a given bias subtracted frame of dn counts, return a same sized
    array of relative gain values.

    This algorithm contains two interpolations:

    - A 2d interpolation to find the relative gain curve for a given EM gain
    - A 1d interpolation to find a relative gain value for each given dn
      count value.

    Both of these interpolations are linear, and both use their edge values as
    constant extrapolations for out of bounds values.

    Parameters:
        frame (array_like): Array of dn count values.
        em_gain (float): Detector EM gain.
        non_lin_correction (corgi.drp.NonLinearityCorrection): A NonLinearityCorrection calibration file.

    Returns:
        array_like: Array of relative gain values.
    """

    # Column headers are gains, row headers are dn counts
    gain_ax = non_lin_correction.data[0, 1:]
    count_ax = non_lin_correction.data[1:, 0]
    # Array is relative gain values at a given dn count and gain
    relgains = non_lin_correction.data[1:, 1:]

    #MMB Note: This check is maybe better placed in the code that is saving the non-linearity correction file?
    # Check for increasing axes
    if np.any(np.diff(gain_ax) <= 0):
        raise ValueError('Gain axis (column headers) must be increasing')
    if np.any(np.diff(count_ax) <= 0):
        raise ValueError('Counts axis (row headers) must be increasing')
    # Check that curves (data in columns) contain or straddle 1.0
    if (np.min(relgains, axis=0) > 1).any() or \
       (np.max(relgains, axis=0) < 1).any():
        raise ValueError('Gain curves (array columns) must contain or '
                              'straddle a relative gain of 1.0')

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

# NOTE:  Change the retrieval of rowreadtime_sec to come from a config file
def get_rowreadtime_sec(datetime=None):
    """
    Get the value of readrowtime. The EMCCD is considered sensitive to the
    effects of radiation damage and, if this becomes a problem, one of the
    mitigation techniques would be to change the row read time to reduce the
    impact of charge traps.

    There's no formal plan/timeline for this adjustment, though it is possible
    to change in the future should it need to.

    Its default value is 223.5e-6 sec.

    Args:
        datetime (astropy Time object): Observation's starting date. Its default
            value is sometime between the first collection of ground data (Full
            Functional Tests) and the duration of the Roman Coronagraph mission.

    Returns:
        float: Current value of rowreadtime in sec.

    """
    # Some datetime between the first collection of ground data (Full
    # Functional Tests) and the duration of the Roman Coronagraph mission.
    if datetime is None:
        datetime = Time('2024-03-01 00:00:00', scale='utc')

    # IIT datetime
    datetime_iit = Time('2023-11-01 00:00:00', scale='utc')
    # Date well in the future to always fall in this case, unless rowreadtime
    # gets updated. One may add more datetime_# values to keep track of changes.
    datetime_1 = Time('2040-01-01 00:00:00', scale='utc')

    if datetime < datetime_iit:
        raise ValueError('The observation datetime cannot be earlier than first collected data on ground.')
    elif datetime < datetime_1:
        rowreadtime_sec =223.5e-6
    else:
        raise ValueError('The observation datetime cannot be later than the' + \
            ' end of the mission')

    return rowreadtime_sec

def get_fwc_em_e(datetime=None):
    """
    Get the value of FWC_EM, the full-well capacity of the pixels in the EM
    gain register in units of electrions. This value will change over the
    course of the mission.

    Its default value is 100000 e-.

    Args:
        datetime (astropy Time object): Observation's starting date. Its default
        value is sometime between the first collection of ground data (Full
        Functional Tests) and the duration of the Roman Coronagraph mission.

    Returns:
        float: Value of FWC_EM in units of electrons at time of observation.

    """

    # IIT datetime
    datetime_iit = Time('2023-11-01 00:00:00', scale='utc')
    # Date well in the future to always fall in this case, unless rowreadtime
    # gets updated. One may add more datetime_# values to keep track of changes.
    datetime_end = Time('2040-01-01 00:00:00', scale='utc')

    # Default to datetime_iit.
    if datetime is None:
        datetime = Time('2023-11-01 00:00:00', scale='utc')

    if datetime < datetime_iit:
        raise ValueError('The observation datetime cannot be earlier than first collected data on ground.')
    elif datetime < datetime_end:
        fwc_em = 100000.
    else:
        raise ValueError('The observation datetime cannot be later than the' + \
            ' end of the mission')

    return fwc_em

def get_fwc_pp_e(datetime=None):
    """
    Get the value of FWC_PP, the full-well capacity of the pixels in the image
    area, before EM gain is applied in readout in units of electrions. This
    value will change over the course of the mission.

    Its default value is 90000 e-.

    Args:
        datetime (astropy Time object): Observation's starting date. Its default
        value is sometime between the first collection of ground data (Full
        Functional Tests) and the duration of the Roman Coronagraph mission.

    Returns:
        float: Value of FWC_PP in electrons at time of observation.

    """
    # Some datetime between the first collection of ground data (Full
    # Functional Tests) and the duration of the Roman Coronagraph mission.
    if datetime is None:
        datetime = Time('2024-03-01 00:00:00', scale='utc')

    # IIT datetime
    datetime_iit = Time('2023-11-01 00:00:00', scale='utc')
    # Date well in the future to always fall in this case, unless rowreadtime
    # gets updated. One may add more datetime_# values to keep track of changes.
    datetime_1 = Time('2040-01-01 00:00:00', scale='utc')

    if datetime < datetime_iit:
        raise ValueError('The observation datetime cannot be earlier than first collected data on ground.')
    elif datetime < datetime_1:
        fwc_pp = 90000.
    else:
        raise ValueError('The observation datetime cannot be later than the' + \
            ' end of the mission')

    return fwc_pp

def get_kgain(datetime=None):
    """
    Get the K gain, the conversion factor between raw counts from the detector
    and electrons coming out of the gain register, in units of e-/dN. This
    value may change over the course of the mission.

    Its default value is 8.7 e-/dN.

    Args:
        datetime (astropy Time object): Observation's starting date. Its default
        value is sometime between the first collection of ground data (Full
        Functional Tests) and the duration of the Roman Coronagraph mission.

    Returns:
        float: The K gain in electrons/dN at time of observation.

    """
    # Some datetime between the first collection of ground data (Full
    # Functional Tests) and the duration of the Roman Coronagraph mission.
    if datetime is None:
        datetime = Time('2024-03-01 00:00:00', scale='utc')

    # IIT datetime
    datetime_iit = Time('2023-11-01 00:00:00', scale='utc')
    # Date well in the future to always fall in this case, unless rowreadtime
    # gets updated. One may add more datetime_# values to keep track of changes.
    datetime_1 = Time('2040-01-01 00:00:00', scale='utc')

    if datetime < datetime_iit:
        raise ValueError('The observation datetime cannot be earlier than first collected data on ground.')
    elif datetime < datetime_1:
        kgain = 8.7
    else:
        raise ValueError('The observation datetime cannot be later than the' + \
            ' end of the mission')

    return kgain

def flag_cosmics(cube, fwc, sat_thresh, plat_thresh, cosm_filter):
    """Identify and remove saturated cosmic ray hits and tails.

    Use sat_thresh (interval 0 to 1) to set the threshold above which cosmics
    will be detected. For example, sat_thresh=0.99 will detect cosmics above
    0.99*fwc.

    Use plat_thresh (interval 0 to 1) to set the threshold under which cosmic
    plateaus will end. For example, if plat_thresh=0.85, once a cosmic is
    detected the beginning and end of its plateau will be determined where the
    pixel values drop below 0.85*fwc.

    Use cosm_filter to determine the smallest plateaus (in pixels) that will
    be identified. A reasonable value is 2.

    Args:
        cube (array_like, float):
            3D cube of image data (bias of zero).
        fwc (float):
            Full well capacity of detector *in DNs*.  Note that this may require a
            conversion as FWCs are usually specified in electrons, but the image
            is in DNs at this point.
        sat_thresh (float):
            Multiplication factor for fwc that determines saturated cosmic pixels.
        plat_thresh (float):
            Multiplication factor for fwc that determines edges of cosmic plateu.
        cosm_filter (int):
            Minimum length in pixels of cosmic plateus to be identified.

    Returns:
        array_like, int:
            Mask for pixels that have been set to zero.

    Notes
    -----
    This algorithm uses a row by row method for cosmic removal. It first finds
    streak rows, which are rows that potentially contain cosmics. It then
    filters each of these rows in order to differentiate cosmic hits (plateaus)
    from any outlier saturated pixels. For each cosmic hit it finds the leading
    ledge of the plateau and kills the plateau and the rest of the row to take
    out the tail.

    |<-------- streak row is the whole row ----------------------->|
     ......|<-plateau->|<------------------tail------------------->|

    B Nemati and S Miller - UAH - 02-Oct-2018

    """
    mask = np.zeros(cube.shape, dtype=int)

    # Do a cheap prefilter for rows that don't have anything bright
    max_rows = np.max(cube, axis=-1,keepdims=True)
    ji_streak_rows = np.transpose(np.array((max_rows >= sat_thresh*fwc).nonzero()[:-1]))

    for j,i in ji_streak_rows:
        row = cube[j,i]

        # Find if and where saturated plateaus start in streak row
        i_beg = find_plateaus(row, fwc, sat_thresh, plat_thresh, cosm_filter)

        # If plateaus exist, kill the hit and the rest of the row
        if i_beg is not None:
            mask[j,i, i_beg:] = 1
            pass

    return mask

def find_plateaus(streak_row, fwc, sat_thresh, plat_thresh, cosm_filter):
    """Find the beginning index of each cosmic plateau in a row.

    Note that i_beg is set at one pixel before first plateau pixel, as these
    pixels immediately neighboring the cosmic plateau are very often affected
    by the cosmic hit as well.

    Args:
        streak_row (array_like, float):
            Row with possible cosmics.
        fwc (float):
            Full well capacity of detector *in DNs*.  Note that this may require a
            conversion as FWCs are usually specified in electrons, but the image
            is in DNs at this point.
        sat_thresh (float):
            Multiplication factor for fwc that determines saturated cosmic pixels.
        plat_thresh (float):
            Multiplication factor for fwc that determines edges of cosmic plateu.
        cosm_filter (float):
            Minimum length in pixels of cosmic plateus to be identified.

    Returns:
        array_like, int:
            Index of plateau beginnings, or None if there is no plateau.
    """
    # Lowpass filter row to differentiate plateaus from standalone pixels
    # The way median_filter works, it will find cosmics that are cosm_filter-1
    # wide. Add 1 to cosm_filter to correct for this
    filtered = median_filter(streak_row, cosm_filter+1, mode='nearest')
    saturated = (filtered >= sat_thresh*fwc).nonzero()[0]

    if len(saturated) > 0:
        i_beg = saturated[0]
        while i_beg > 0 and streak_row[i_beg] >= plat_thresh*fwc:
            i_beg -= 1

        return i_beg
    else:
        return None

def calc_sat_fwc(emgain_arr,fwcpp_arr,fwcem_arr,sat_thresh):
    """Calculates the full well capacity saturation threshold for each frame.

    Args:
        emgain_arr (np.array): 1D array of the EM gain value for each frame.
        fwcpp_arr (np.array): 1D array of the full-well capacity in the image
            frame (before em gain readout) value for each frame.
        fwcem_arr (np.array): 1D array of the full-well capacity in the EM gain
            register for each frame.
        sat_thresh (float): Multiplier for the full-well capacity to determine
            what qualifies as saturation. A reasonable value is 0.99

    Returns:
        np.array: _description_
    """
    possible_sat_fwcs_arr = np.append((emgain_arr * fwcpp_arr)[:,np.newaxis], fwcem_arr[:,np.newaxis],axis=1)
    sat_fwcs = sat_thresh * np.min(possible_sat_fwcs_arr,axis=1)

    return sat_fwcs