"""Analysis of trap-pumped images for calibration.  Assumes trap-pumped
full frames as input.  Some of the trap-finding and parameter-fitting code
adapted from code from Nathan Bush, and his code was used for his paper, the
basis for this code:
Nathan Bush, David Hall, and Andrew Holland,
J. of Astronomical Telescopes, Instruments, and Systems, 7(1), 016003 (2021).
https://doi.org/10.1117/1.JATIS.7.1.016003
- Kevin Ludwick, UAH, 6/2022

- MMB Changes for corgi-drp: 
    - Removed everything meta and replaced with functions from detector.py
    - Replaced II&T get_relgains function with corgidrp function -> Now 
    requires a NonLinearityCorrection object to work. Defaults to None
"""
import os
from pathlib import Path
import warnings
import numpy as np
import glob

from astropy.io import fits
from corgidrp.detector import *
# from pymatreader import read_mat

import corgidrp.check as check
from corgidrp.data import TrapCalibration, Dataset

from trap_id import illumination_correction, trap_id
from trap_fitting import trap_fit, trap_fit_const, fit_cs

class TPumpAnException(Exception):
    """Exception class for tpumpanalysis."""

# does it make sense to have noncontinuous or less-than-2-temps traps,
# even for this non-EDU camera? Yes, if enough noise is fitted; also, for
#different temps, the sub-el location may or may not be the same, thus leading
# to inconsistent traps across temperatures

# TODO, v2.0:  Could make generate_test_data.py more accurate by, instead of
#choosing a range of phase time frames for length_lim, adding in add_*_dipole()
# selection of phase times in which amplitude peaks for that particular trap
# for each temperature.
# This is really only useful for both-type traps, like (77,90).  Also, I span
# a wide range of temperatures, and some traps are only really detectable in
# real life over a small range of temperatures.
# My unit tests would probably detect even more
# temperatures in a consistent way if I covered a smaller range of temperature.
def tpump_analysis(base_dir, time_head, emgain_head,
    num_pumps, non_lin_correction, mean_field = None, length_lim = 6,
    thresh_factor = 3, k_prob = 1, ill_corr = True, tfit_const = True,
    tau_fit_thresh = 0.8, tau_min = 0.7e-6, tau_max = 1.3e-2, tauc_min = 0,
    tauc_max = 1e-5, pc_min = 0, pc_max = 2, offset_min = 10,
    offset_max = 10,
    save_temps = None, load_temps = None,
    cs_fit_thresh = 0.8, E_min = 0, E_max = 1, cs_min = 0,
    cs_max = 50, bins_E = 100, bins_cs = 10, input_T = 180,
    sample_data = False):
    """This function analyzes trap-pumped frames and outputs the location of
    each radiation trap (pixel and sub-electrode location within the pixel),
    everything needed to determine the release time constant at any temperature
    (the capture cross section for holes and the energy level), trap densities
    (i.e., how many traps per pixel for each kind of trap found), and
    information about the capture time constant (for potential future
    analysis).  This function only works as intended for the EMCCD with its
    electrodes' paticular electric potential shape that will
    be used for trap pumping on the Roman telescope.  It can find up to two
    traps per sub-electrode location per pixel.  The function has an option to
    save a preliminary output that takes a long time to create (the dictionary
    called 'temps') and an option to load that in and start the function at
    that point.
    Parameters
    ----------
    base_dir : str
        Full path of base directory containing all trap-pumped FITS files. The
        metadata assumed to be in PrimaryHDU, and the first extension is
        assumed to be the ImageHDU, where the array data is located.
        - This folder should contain sub-folders for each temperature entitled
        'tempK', where 'temp' is the float or int value of the detctor
        temperature in K (e.g., 160K).  The function reads everything up to the
        last character of the filename.  Nothing else should be in this folder.
        - Each temperature sub-folder should contain sub-folders for each
        scheme entitled 'Scheme_num', where 'num' is either 1, 2, 3, or 4
        (e.g., 'Scheme_1').  The function reads the last character from the
        filename as the scheme.  Nothing else should be in this folder.
        - Each scheme sub-folder should contain a trap-pumped frame
        (FITS format) for each phase time.  There may be more than one FITS
        file for a given phase time.  All FITS files in these folders
        will be used by the function.  Each file should have headers specifying
        the phase time in microseconds and the EM gain for that file.  The data
        is assumed to be a 2-D array of counts with axis 0 going along the rows
        and axis 1 going along the columns, and the readout direction is
        assumed to be along axis 0 toward decreasing row number.

        If you are running sample data from Alfresco, you would need to
        download that data and specify its directory on your computer, and
        the sample_data parameter needs to be set to True.  (See below for
        information on sample_data.)

        If load_temps is not None, base_dir is not used at all. (See below for
        information on load_temps.)
    time_head : str
        Keyword corresponding to phase time for each FITS file.  Keyword value
        assumed to be float (units of microseconds).
    emgain_head : str
        Keyword corresponding to EM gain for each FITS file.  Keyword value
        assumed to be float (unitless).
    num_pumps : int, > 0
        Number of cycles of pumping performed for each trap-pumped frame.
    non_lin_correction: corgi.drp.NonLinearityCorrection
        A NonLinearityCorrection calibration file (if needed). Default = None
    mean_field : > 0 or None
        The mean electron level that was present in each pixel before trap
        pumping was performed (so doesn't include EM gain). The max charge that
        can be captured by a trap is limited by the mean number of electrons
        present in non-trap pixels.
        Only useful if this mean level is less than
        2500 e-, the max amplitude a trap adhering to probability function 1
        can have (used for determining k gain).  If the mean non-trap electron
        level is relatively low, the capture time constant (tauc) will be
        relatively big (still less than 1 s), and trap_fit() may provide a
        more accurate fit for the release time constant (tau)
        over trap_fit_const(), especially for pixels that contain only one
        trap (instead of two).  In general, trap_fit_const() gives better
        fits overall for most cases, and
        trap-pumped frames are typically made with high charge packets (which
        is when tauc is a constant to a good approximation; see
        tfit_const below).  If the level is 2500e- or
        higher, you can put in the number or None.
    length_lim : int, > 0, optional
        Minimum number of frames for which a dipole needs to meet threshold so
        that it goes forward for consideration as a true trap.  If it is bigger
        than the number of distinct phase times for frames in a scheme folder,
        length_lim is set to the latter by the function.  Defaults to 6,
        which agrees with Nathan Bush et al., 2021.
    thresh_factor : float, > 0, optional
        Number of standard deviations from the mean a dipole should stand out
        in order to be considered for a trap. If this is too high, you get
        dipoles with amplitude that continually increases with phase time (not
        characteristic of an actual trap).  If thresh_factor is too low,
        you get a bad fit because the resulting dipoles have amplitudes that
        are too noisy and low.  Defaults to 3, which agrees
        with Nathan Bush et al., 2021.
    k_prob : 1 or 2, optional
        The probability function used for finding the e-/DN factor.
        Probability function 1 (P1) should be tried first, and if the code
        fails with a TPumpAnException, re-run the code with 2.  Defaults to 1.
    ill_corr : bool, optional
        True:  run local illumination correction (via the function
        illumination_correction()) on each trap-pumped frame to remove
        any defects (irregularities in the supposed flat field, not considering
        the traps) by subtracting the local median from every image pixel.
        The local median for a pixel is the median of a square (containing that
        pixel) of side length equal to
        binsize pixels, where binsize is determined in the code as
        roughly the square root of the smaller dimension of the image area.  If
        the trap density is too high, the code moves forward as if ill_corr
        were False.  False:  illumination_correction() subtracts from each
        pixel the median of the entire image region instead of the local
        median.  Without
        illumination correction, the defaults for offset_min and offset_max
        (further below) may not be accurate and would probably need to be
        increased.  (See their doc strings for more details.)
        ill_corr defaults to True, and this is strongly recommended.
    tfit_const : bool, optional
        True: run the function trap_fit_const() for curve fitting, which treats
        the probability function for capture (Pc) as a constant (which is
        approximately true for the Roman trap-pumped frames, and if k gain is
        determined with low accuracy, Pc absorbs the factor of k gain and thus
        acts as a nuisance parameter in that case.)  During testing, it was
        found tfit_const=True outperformed the case of tfit_const=False.
        False:  run the function trap_fit() for curve fitting, which treats Pc
        accurately as the time-dependent function it really is.  (If k gain is
        determined with low accuracy, there is no fit parameter that can absorb
         k gain fully.)  Defaults to True.
    tau_fit_thresh : (0, 1), optional
        The minimum value required for adjusted coefficient of determination
        (adjusted R^2) for curve fitting for the release time constant
        (tau) using data for dipole amplitude vs phase time.  The closer to 1,
        the better the fit. Must be between 0 and 1.  Defaults to 0.8.
    tau_min : float, >= 0, optional
        Lower bound value for tau (release time constant) for curve fitting,
        in seconds.  Defaults to 0.7e-6, slightly below 1e-6, which is the
        smallest phase time that would be used for trap pumping. It is slightly
        below to allow for effective bounds for curve fitting.
        A trap is only likely to be active when the
        phase time is close to the release time constant, and release time
        constants in the literature are not lower that this default value.
        In theory, the phase time at which a trap is successfully fitted could
        be far from the trap's actual release time constant,
        but the chances for fitting a phenomenon unrelated to traps may also
        be higher if the lower bound is lower than the minimum phase time
        probed.
    tau_max : float, > tau_min, optional
        Upper bound value for tau (release time constant) for curve fitting,
        in seconds.  Defaults to 1.3e-2, slightly above 1e-2, the largest phase
        time that would be used for trap pumping. It is slightly
        above to allow for effective bounds for curve fitting.
        A trap is only likely to be active when the
        phase time is close to the release time constant, and release time
        constants in the literature are not higher that this default value.
        In theory, the phase time at which a trap is successfully fitted could
        be far from the trap's actual release time constant,
        but the chances for fitting a phenomenon unrelated to traps may also
        be higher if the upper bound is higher than the maximum phase time
        probed.
    tauc_min : float, >= 0, optional
        Lower bound value for tauc (capture time constant) for curve fitting,
        in seconds.  Only used if tfit_const = False.  Defaults to 0.
    tauc_max : float, > tauc_min, optional
        Upper bound value for tauc (capture time constant) for curve fitting,
        in seconds.  Only used if tfit_const = False.  Defaults to 1e-5,
        a sufficiently generous maximum value for tauc given that large charge
        packets used in trap pumping imply small tauc values.  If mean_field
        is not None, this may indicate that the average charge packet per
        non-trap pixel is low, which would indicate that tauc could be bigger.
        In that rare case, tauc is recommended to be set to 1e-2.
    pc_min : float, >= 0, optional
        Lower bound value for pc (capture probability) for curve fitting,
        in e-.  Only used if tfit_const = True.  Defaults to 0.
    pc_max : float, > pc_min, optional
        Upper bound value for tauc (capture time constant) for curve fitting,
        in e-.  Only used if tfit_const = True.  Defaults to 2,
        which is the maximum probability (1) times a factor of 0 to
        accommodate a potentially erroneous application of the e-/DN factor.
    offset_min : float, optional
        Offset lower bound value for a dipole's trap fit relative to 0
        for the fitting of data for amplitude vs phase time.
        Defaults to 10, which
        means the lower bound for the offset in the curve fit is 10 unless the
        maximum median subtracted which was
        determined by illumination_correction() is lower.  The
        lower of that and offset_min is what is chosen as the lower bound for
        offset in curve fitting amplitude vs phase time.
        The offset accommodates any unsubtracted voltage
        bias, the subtraction that occured in illumination_correction(),
        and any erroneous estimation of k gain. Ideally, the minimum
        value in e- is 0, but a non-zero buffer below may help the curve
        fitting.  It acts as a nuisance parameter in the fit.  Units of e-.
    offset_max : float, > offset_min, optional
        Offset upper bound value for a dipole's trap fit relative to 0.
        Used for the offset in the fitting of data for amplitude vs phase time.
        Defaults to 10, which
        means the upper bound for the offset in the curve fit is 10 unless the
        minimum median subtracted which was
        determined by illumination_correction() is bigger.  The
        bigger of that and offset_max is what is chosen as the upper bound for
        offset in curve fitting amplitude vs phase time.
        The offset accommodates any unsubtracted voltage
        bias, the subtraction that occured in illumination_correction(),
        and any erroneous estimation of k gain. It acts as a nuisance
        parameter in the fit.  Units of e-.
    save_temps : str or None, optional
        If input is a string, it should be the absolute path, including the
        desired filename, for where to save the temps dictionary.  The file
        will be saved as a .npy file, so your filename should end with .npy.
        If None, the temps dictionary is not saved.  Defaults to None.
    load_temps : str or None, optional
        If input is a string, it should be the absolute path of the .npy file
        containing the 'temps' dictionary.  If input is None, no file is
        loaded.  Defaults to None.
    cs_fit_thresh : (0, 1), optional
        The minimum value required for adjusted coefficient of determination
        (adjusted R^2) for curve fitting for the capture cross section
        for holes (cs) using data for tau vs temperature.  The closer to 1,
        the better the fit. Must be between 0 and 1.  Defaults to 0.8.
    E_min : float, >= 0, optional
        Lower bound for E (energy level in release time constant) for curve
        fitting, in eV.  Defaults to 0.
    E_max : float, > E_min, optional
        Upper bound for E (energy level in release time constant) for curve
        fitting, in eV.  Defaults to 1.
    cs_min : float, >= 0, optional
        Lower bound for cs (capture cross section for holes in release time
        constant) for curve fitting, in 1e-19 m^2.  Defaults to 0.
    cs_max : float, > cs_min, optional
        Upper bound for cs (capture cross section for holes in release time
        constant) for curve fitting, in 1e-19 m^2.  Defaults to 50.
    bins_E : int, > 0, optional
        Number of bins used for energy level in categorizing traps into 2-D
        bins of energy level and cross section.  Defaults to 100 (level of
        reasonable precision of difference in eV (i.e., each bin increments by
        1/100 eV = 0.01 eV) between traps from the
        literature that would likely be detectable).
    bins_cs : int, > 0, optional
        Number of bins used for cross section in categorizing traps into 2-D
        bins of energy level and cross section.  Defaults to 10 (level of
        reasonable precision of difference in 1e-14 cm^2 (i.e., each bin
        increments by 1/10 e-14 = 0.1e-14 for each bin) between traps from the
        literature that would likely be detectable).
    input_T : float, > 0, optional
        Temperature of Roman EMCCD at which to calculate the
        release time constant (in units of Kelvin).  Defaults to 180.
    sample_data : bool, optional
        True if you want to run the sample data on Alfresco, located at
        https://alfresco.jpl.nasa.gov/share/page/site/cgi/documentlibrary#
        filter=path%7C%2FRoman%2520CGI%2520Collaboration%2520Area%2F05%2520-
        %2520Project%2520Science%2FSample_trap_pump_data%7C&page=1.
        That data is not from the EDU camera and doesn't have
        the correct electric potential shape in the electrodes
        of the pixels, so the sub-electrode locating
        that this function performs will not be very meaningful.  However, some
        traps will still meet the sub-electrode location criteria and provide a
        reasonable return for the function.  When sample_data is True, the
        function accounts for the differences in reading this data (since it
        is MAT and not FITS) and extracting the bias from the non-EDU-sized
        frames.  It also reads off phase time from the names of each file.
        If you are running EDU camera data, this should be False.
        Defaults to False.
    Returns
    -------
    trap_dict : dict
        Dictionary with a key for each trap for which acceptable fits for
        the release time constant (tau) for all schemes could be made.  It has
        the following format, and an example entry for a
        pixel at location (row, col) on the right-hand side (RHS) of
        electrode 2 containg 1 of possibly two traps is shown.  The 0 in the
        key denotes that this is the first trap found at this pixel and
        sub-electrode location.  If a 2nd trap was found at this same location,
        another trap with a 1 in the key would be present in trap_dict.
        trap_dict = {
            ((row, col), 'RHSel2', 0): {'T': [160, 162, 164, 166, 168],
            'tau': [1.51e-6, 1.49e-6, 1.53e-6, 1.50e-6, 1.52e-6],
            'sigma_tau': [2e-7, 1e-7, 1e-7, 2e-7, 1e-7],
            'cap': [[cap1, cap1_err, max_amp1, cap2, cap2_err, max_amp2], ...],
            'E': 0.23,
            'sig_E': 0.02,
            'cs': 2.6e-15,
            'sig_cs': 0.3e-15,
            'Rsq': 0.96,
            'tau at input T': 1.61e-6,
            'sig_tau at input T': 2.02e-6},
            ...}
        'T': temperatures for which values of tau were successfully fit. In K.
        'tau': the tau values corresponding to these temperatures in the same
        order.  In seconds.
        'sigma_tau': overall uncetainty in tau taken from the errors in the
        fits from all the schemes that were used to specify trap location
        (corresponding to the same order as 'T' and 'tau'). In seconds.
        'cap': cap1 is either the probability of capture
        (if trap_fit_const() used) or tauc (capture time constant) if
        trap_fit() used. cap1_err is the error from fitting.  max_amp1 is the
        maximum amplitude of the dipole from the curve fit for that pixel.
        Similarly for cap2, cap2_err, and max_amp2, and there can be a 3rd set
        of parameters if a trap sub-electrode location was determined using
        3 schemes.  This data may be useful for future analysis.
        'E': energy level.  In eV.
        'sig_E': standard deviation error of energy level.  In eV.
        'cs': cross section for holes.  In cm^2.
        'sig_cs': standard deviation error of cross section for holes.
        In cm^2.
        'Rsq': adjusted R^2 for the tau vs temperature fit that was done to
        obtain cs.
        'tau at input T': tau (in seconds) evaluated at desired temperature of
        Roman EMCCD, input_T.
        'sig_tau at input T': standard deviation error of tau at desired
        temperature of Roman EMCCD, input_T.  Found by propagating error by
        utilizing 'sig_cs' and 'sig_E'.
    trap_densities : list
        A list of lists, where a list is provided for each type of trap.
        The trap density for a trap type is the # of traps in a given 2-D bin
        of E and tau divided by the total number of pixels in the image area.
        The binning by default is fine enough to distinguish all trap types
        found in the literarture. Only the bins that contain non-zero entries
        are returned here.  Each trap-type list is of the following format:
        [trap density, E, cs].
        E is the central value of the E bin, and cs is the central value of the
        cs bin.
    bad_fit_counter : int
        Number of times trap_fit() provided fits of amplitude vs phase time
        that were below fit_thresh over all schemes and temperature (which
        would include any preliminary trap identifications that were rejected
        because they weren't consistent across all schemes for sub-electrode
        locations).
    pre_sub_el_count : int
        Number of times a trap was identified before filtering them through
        sub-electrode location.  The counter is over all schemes and
        temperatures.
    unused_fit_data : int or None
        Number of times traps were identified that were not matched up for
        sub-electrode location determination.  If the input load_temps is not
        None, this will return as None since the code was called to start at
        the point at which the temps dictionary has already been generated, and
        that is after all fitting that determines unused_fit_data has been
        done.
    unused_temp_fit_data : int
        Number of times traps were identified that did not get used in
        identifying release time constant values across all temperatures
    two_or_less_count : int
        Number of traps that only appeared at 2 or fewer temperatures.
    noncontinuous_count : int
        Number of traps that appeared at a noncontinuous series of
        temperatures.
    """
    if type(sample_data) != bool:
        raise TypeError('sample_data should be True or False')
    if not sample_data: # don't need these for the Alfresco sample data
        if type(emgain_head) != str:
            raise TypeError('emgain_head must be a string')
        if type(time_head) != str:
            raise TypeError('time_head must be a string')
    check.positive_scalar_integer(num_pumps, 'num_pumps', TypeError)
    check.real_positive_scalar(thresh_factor, 'thresh_factor', TypeError)
    if mean_field is not None:
        check.real_positive_scalar(mean_field, 'mean_field', TypeError)
    check.positive_scalar_integer(length_lim, 'length_lim', TypeError)
    if k_prob != 1 and k_prob != 2:
        raise TypeError('k_prob must be either 1 or 2')
    if type(ill_corr) != bool:
        raise TypeError('ill_corr must be True or False')
    if type(tfit_const) != bool:
        raise TypeError('ill_corr must be True or False')
    check.real_nonnegative_scalar(tau_fit_thresh, 'tau_fit_thresh', TypeError)
    if tau_fit_thresh > 1:
        raise ValueError('tau_fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(tau_min, 'tau_min', TypeError)
    check.real_nonnegative_scalar(tau_max, 'tau_max', TypeError)
    if tau_max <= tau_min:
        raise ValueError('tau_max must be > tau_min')
    check.real_nonnegative_scalar(tauc_min, 'tauc_min', TypeError)
    check.real_nonnegative_scalar(tauc_max, 'tauc_max', TypeError)
    if tauc_max <= tauc_min:
        raise ValueError('tauc_max must be > tauc_min')
    check.real_nonnegative_scalar(pc_min, 'pc_min', TypeError)
    check.real_nonnegative_scalar(pc_max, 'pc_max', TypeError)
    if pc_max <= pc_min:
        raise ValueError('pc_max must be > pc_min')
    check.real_nonnegative_scalar(offset_min, 'offset_min', TypeError)
    check.real_nonnegative_scalar(offset_max, 'offset_max', TypeError)
    check.real_nonnegative_scalar(cs_fit_thresh, 'cs_fit_thresh', TypeError)
    if cs_fit_thresh > 1:
        raise ValueError('cs_fit_thresh should fall in (0,1).')
    check.real_nonnegative_scalar(E_min, 'E_min', TypeError)
    check.real_nonnegative_scalar(E_max, 'E_max', TypeError)
    if E_max <= E_min:
        raise ValueError('E_max must be > E_min')
    check.real_nonnegative_scalar(cs_min, 'cs_min', TypeError)
    check.real_nonnegative_scalar(cs_max, 'cs_max', TypeError)
    if cs_max <= cs_min:
        raise ValueError('cs_max must be > cs_min')
    check.positive_scalar_integer(bins_E, 'bins_E', TypeError)
    check.positive_scalar_integer(bins_cs, 'bins_cs', TypeError)
    check.real_positive_scalar(input_T, 'input_T', TypeError)

    # to count how many apparent dipoles that
    #didn't meet the fit threshold
    bad_fit_counter = 0
    # to count how many fit attempts occured
    num_attempted_fits = 0
    # counter for number of traps before filtering them for sub-el location
    # (over all schemes and temperatures)
    pre_sub_el_count = 0
    unused_fit_data = None
    # count # times Pc is bigger than one (relevant when trap_fit_const() used)
    pc_bigger_1 = 0
    pc_biggest = 0
    # count # times emit time constant > 1e-2 or < 1e-6. Relevant only if
    # tau_min and tau_max are outside of 1e-6 to 1e-2.
    tau_outside = 0
    if load_temps is None:
        temps = {}
        for dir in os.listdir(base_dir):
            sch_dir_path = os.path.abspath(Path(base_dir, dir))
            if os.path.isfile(sch_dir_path): # just want directories
                continue
            try:
                curr_temp = float(dir[0:-1])
            except:
                raise TPumpAnException('Temperature folder label must be '
                'a number up to the last character.')
            schemes = {}
            # initializing eperdn here; used if scheme is 1
            eperdn = 1
            # check to make sure no two scheme folders the same (1 per scheme)
            sch_list = []
            for scheme in os.listdir(sch_dir_path):
                scheme_path = os.path.abspath(Path(sch_dir_path, scheme))
                if os.path.isfile(scheme_path): # just want directories
                    continue
                sch_list.append(scheme[-1])
                try:
                    int(scheme[-1])
                except:
                    raise TPumpAnException('Last character of the scheme '
                    'folder label must be a number')
            for num in sch_list:
                if sch_list.count(num) > 1:
                    raise TPumpAnException('More than one folder for a single '
                    'scheme found.  Should only be one folder per scheme.')
            for sch_dir in sorted(os.listdir(sch_dir_path)):
                scheme_path = os.path.abspath(Path(sch_dir_path, sch_dir))
                if os.path.isfile(scheme_path): # just want directories
                    continue
                curr_sch = int(sch_dir[-1])
                # if Scheme_1 present, the following shouldn't happen
                if schemes == {} and curr_sch != 1:
                    raise TPumpAnException('Scheme 1 files must run first for'
                        ' an accurate eperdn estimation')
                frames = []
                cor_frames = []
                timings = []
                curr_sch_path = os.path.abspath(Path(sch_dir_path, sch_dir))
                for file in os.listdir(curr_sch_path):
                    f = os.path.join(curr_sch_path, file)
                    if os.path.isfile(f) and f.endswith('.fits') and \
                        not sample_data:
                        try:
                            data = fits.getdata(f, 1) #1st extension for image
                            data = data.astype(float)
                        except:
                            raise TPumpAnException('Must be .fits files with '
                            'image data in 1st extension')
                        with fits.open(f) as hdul:
                            try:
                                hdr = hdul[0].header
                                # time from FITS header in us, so convert to s
                                t = float(hdr[time_head])/10**6
                                em_gain = float(hdr[emgain_head])
                            except:
                                raise TPumpAnException('Primary header must '
                                'contain correct keys for phase time and EM '
                                'gain')
                        timings.append(t)
                        # getting image area (all physical CCD pixels)
                        d = imaging_slice(data)
                        # need to subtract bias if we are to extract eperdn
                        # getting all physical CCD pixels
                        prows, pcols, r0c0 = imaging_area_geometry()
                        if prows > np.shape(data)[0] or \
                            pcols > np.shape(data)[1] or \
                            r0c0[0] > np.shape(data)[0] or \
                            r0c0[1] > np.shape(data)[1]:
                            raise TPumpAnException('Assumed geometry from detector.py inconsistent'
                            ' with frame')
                        # Get prescan region
                        prescan = slice_section(data,'SCI','prescan')
                        # select the good cols for getting row-by-row bias
                        # we don't use Process class here to avoid having to
                        # input fwc params, etc
                        st = detector_areas['SCI']['col_start']
                        end = detector_areas['SCI']['col_end']
                        p_r0 = detector_areas['SCI']['r0c0'][0]
                        i_r0 = r0c0[0]
                        # prescan relative to image rows
                        al_prescan = prescan[(i_r0-p_r0):(i_r0-p_r0+prows), :]
                        bias_dn = np.median(al_prescan[:,st:end],
                                axis=1)[:, np.newaxis]
                        d -= bias_dn
                        #CIC also present in prescan, and signal is mainly
                        #gained CIC, so in image area, expect to have
                        # roughly 0 median

                        # nonlinearity correction done assuming
                        # row-by-row bias subtraction, too, I believe.
                        # could have non-linearity (flat field, but dipoles),
                        # incurred at ADU, after frame is read out;
                        # correct for it if needed (with residual nonlinearity)
                        if non_lin_correction is not None:
                            d *= get_relgains(d, em_gain, non_lin_correction)
                        
                        d = d/em_gain
                        frames.append(d)
                    if os.path.isfile(f) and f.endswith('.mat') and \
                         sample_data:
                        try:
                            data = read_mat(f)
                            data = (data['frame_data']).astype(float)
                        except:
                            raise TPumpAnException('Sample data from Alfresco '
                            'must be .mat type.')
                        if np.shape(data) != (2200, 1080):
                            raise TPumpAnException('Matlab data from Alfresco '
                            'sample data must be of shape '
                            '(rows, cols) = (2200, 1080).')
                        # extracting time from Nathan's file name
                        # converting from us to s, too
                        if 'us_' not in file:
                            raise TPumpAnException('Filenames for sample data '
                            'from Alfresco must contain \'us_\' in order for '
                            'phase time to be read from them.')
                        try:
                            t = float(file[31:file.find("us_")])/10**6
                        except:
                            raise TPumpAnException('Filenames for sample data '
                            'from Alfresco msut contain numerals for the phase'
                            ' times in the right position.')
                        timings.append(t)
                        # For Nathan's particular data, size it to be
                        # comparable to EDU
                        d = data[15:15+1024,28:28+1024].copy()
                        em_gain = 1

                        #n1 = d[0:15, 0:]
                        #n2 = d[15:, 0:28]
                        #bias_dn = (np.median(n1)*np.size(n1) +
                        #  np.median(n2)*np.size(n2))/(np.size(n1)+np.size(n2))
                        noise = data[2100:2150, 100:1000].copy()
                        bias_dn = np.median(noise)
                        # don't worry about row-by-row for this sample data
                        d = d - bias_dn
                        #bias_dns.append(bias_dn)

                        # now do nonlinearity correction (before averaging
                        # frames with same temp and phase time)

                        # change to actual path if nonlinearity correction
                        # desired (if EM gain known)
                        # actually, don't need to average frames with same temp
                        #  and phase time; curve_fit doesn't care
                        #if you have mutliple values for the same data point!

                        # Nathan says could be useful but doesn't deserve too
                        # much attention unless linearity is absolutely
                        # horrible; enough traps causes good self-calibrating
                        # of image

                        if non_lin_correction is not None:
                            d *= get_relgains(d, em_gain, non_lin_correction)
                        d = d/em_gain
                        #d *= 2500/d.max()
                        frames.append(d)
                # no need for cosmic ray removal since we do ill. correction
                # plus cosmics wouldn't look same as it would on regular frame,
                # and they affect the detection of dipoles (or very low chance)
                # This can also be mitigated by taking multiple frames per
                # phase time.
                # no need for flat-fielding since trap pumping will be done
                # while dark (i.e., a flat field of dark)
                if frames == []:
                    raise TPumpAnException('Scheme folder had no data '
                    'files in it.')
                # if curr_sch is 1, then this simply multiplies by 1
                frames = np.stack(frames)*eperdn
                timings = np.array(timings)
                # min number of frames for a dipole should meet threshold so
                # that it is
                # considered a potential true trap, unless the number of
                # available frames is smaller
                length_limit = min(length_lim, len(np.unique(timings)))
                #To accurately determine eperdn, the median level needs to be
                #subtracted off if one is to compare, for P1,
                # 2500e- sitting on top of nothing; if it were sitting on
                # top of the mean field (i.e.,
                #whatever injected charge is there after bias subtraction), it
                #would be more than 2500e-.
                # illumination correction only subtracts and doesn't divide,
                # so e- units unaffected; the offset introduced taken care of
                # by nuisance parameter 'offset' in curve fitting.

                #And every time trap_id() called when local_ill_max
                # and local_ill_min is not
                # None, illumination_correction() has already been performed,
                # and eperdn has already been applied, so the units are e-.
                local_ill_frs = []
                # reasonable binsize
                nrows = frames[0].shape[0]
                ncols = frames[0].shape[1]
                small = min(nrows, ncols)
                binsize = int(np.sqrt(small))
                for frame in frames:
                    img, local_ill = illumination_correction(frame,
                        binsize = binsize, ill_corr=ill_corr)
                    cor_frames.append(img)
                    local_ill_frs.append(local_ill)
                local_ill_max = np.max(np.stack(local_ill_frs), axis = 0)
                local_ill_min = np.min(np.stack(local_ill_frs), axis = 0)
                cor_frames = np.stack(cor_frames)

                rc_above, rc_below, rc_both = trap_id(cor_frames,
                    local_ill_min, local_ill_max, timings,
                    thresh_factor = thresh_factor, length_limit=length_limit)

                # no minimum binsize b/c with each trap, there's a deficit and
                # a surplus of equal amounts about the flat field, which
                # doesn't change the median

                # could fit eperdn as a nuisance param k multiplying,
                # like k*Num_pumps*Pc*Pe, but some of k could bleed into
                # Pc (and possibly Pe, but Pe can only range from 0 to 1/4 or
                # 4/27 for scheme 1 or 2; so Pc should absorb it).
                # So for accurate fitting, need to find k first.
                # TODO v2: Make list of traps sorted by amp, and try
                # trap_fit_const() on trap with highest amp.  If a 1-trap is
                # good fit, assign k to be the factor by
                # which Pc is bigger than 1.  If instead a 2-trap fits, then
                # go to the next brightests trap, and repeat until you get a
                # 1-trap fit.  Then I need to change doc string to indicate
                # pc_max and pc_min still relevant even if tfit_const = False.
                if curr_sch == 1:
                    max_a = [] # initialize
                    for i in rc_above:
                        max_a.append(np.max(rc_above[i]['amps_above']))
                    max_b = []
                    for i in rc_below:
                        max_b.append(np.max(rc_below[i]['amps_below']))
                    max_bo = []
                    for i in rc_both:
                        max_bo.append(np.max(rc_both[i]['amps_both']))
                    #peak_trap = max(max(max_a), max(max_b), max(max_bo))
                    # to account for 2-traps, which are fewer and may not be
                    # simply double the max trap amplitude, take median
                    max_all = max_a + max_b + max_bo
                    max_all = np.array(max_all)
                    max_max_all = 0
                    if len(max_all) != 0:
                        max_max_all = max(max_all)
                    if len(max_all) == 0 or max_max_all <= 0:
                        raise TPumpAnException('No traps were found in '
                        'scheme {} at temperature {} K.'.format(curr_sch,
                        curr_temp))
                    peak_trap = np.median(max_all)
                    # eperdn applicable to all frames, a value for each temp
                    # (which is expected to be the same for each temp)
                    if k_prob == 1:
                        prob_factor_eperdn = 1/4
                    if k_prob == 2:
                        prob_factor_eperdn = 4/27
                    # if mean e- per pixel is lower than 2500e-, than max amp
                    # is that mean e- per pixel amount
                    if mean_field is not None:
                        max_e = min(num_pumps*1*prob_factor_eperdn, mean_field)
                    else:
                        max_e = num_pumps*1*prob_factor_eperdn
                    eperdn = max_e/peak_trap
                    #convert to e-
                    cor_frames *= eperdn

                    for i in rc_above:
                        rc_above[i]['amps_above'] *= eperdn
                        rc_above[i]['loc_med_min'] *= eperdn
                        rc_above[i]['loc_med_max'] *= eperdn
                    for i in rc_below:
                        rc_below[i]['amps_below'] *= eperdn
                        rc_below[i]['loc_med_min'] *= eperdn
                        rc_below[i]['loc_med_max'] *= eperdn
                    for i in rc_both:
                        rc_both[i]['amps_both'] *= eperdn
                        rc_both[i]['loc_med_min'] *= eperdn
                        rc_both[i]['loc_med_max'] *= eperdn
                # below: short for no prob fit for scheme 1 above, below, both
                no_prob1a = 0
                no_prob1b = 0
                no_prob1bo = 0
                # delete coords that have None for trap_fit return
                del_a_list = []
                del_b_list = []
                del_bo_list = []
                # two- and one-trap counter over 'above', 'below', and 'both'
                #(for internal testing)
                two_trap_count = 0
                one_trap_count = 0
                # curve-fit 'above' traps
                fit_a_count = 0
                # of attempted fits that will be done
                num_attempted_fits += len(rc_above)+len(rc_below)+len(rc_both)
                for i in rc_above:
                    loc_med_min = rc_above[i]['loc_med_min']
                    loc_med_max = rc_above[i]['loc_med_max']
                    # set average offset to expected level after
                    #illumination_correction(), which is 0 (b/c no negative
                    # counts), by subtracting
                    loc_avg = (loc_med_min+loc_med_max)/2
                    off_min = 0 - max(offset_min,np.abs(loc_avg - loc_med_max))
                    off_max = 0 + max(offset_max,np.abs(loc_med_min - loc_avg))
                    if not tfit_const:
                        fd = trap_fit(curr_sch, rc_above[i]['amps_above'],
                            timings, num_pumps, tau_fit_thresh, tau_min,
                            tau_max, tauc_min, tauc_max, off_min,
                            off_max) #, k_min, k_max)
                    if tfit_const:
                        fd = trap_fit_const(curr_sch,
                            rc_above[i]['amps_above'], timings, num_pumps,
                            tau_fit_thresh, tau_min, tau_max, pc_min, pc_max,
                            off_min, off_max)
                    if fd is None:
                        bad_fit_counter += 1
                        print('bad fit above: ', i)
                        del_a_list.append(i)
                    if fd is not None:
                        fit_a_count += 1
                        # find out how many pixels were fitted for 2 traps
                        #(for internal testing)
                        temp_2_count = 0
                        for key in fd.keys():
                            # only meaningful if trap_fit_const() used
                            #(for internal testing)
                            # Pc - Pc_err > 1:
                            if fd[key][0][0]-fd[key][0][1] > 1:
                                pc_bigger_1 += 1
                                if fd[key][0][0]+fd[key][0][1] > pc_biggest:
                                    pc_biggest = fd[key][0][0]+fd[key][0][1]
                            # only meaningful if tau bounds outside 1e-6, 1e-2
                            #(for internal testing)
                            #tau + tau_err < 1e-6, tau - tau_err < 1e-2:
                            if fd[key][0][2]+fd[key][0][3] < 1e-6 or \
                                fd[key][0][2]-fd[key][0][3] > 1e-2:
                                tau_outside += 1
                            if len(fd[key]) == 2:
                                if fd[key][1][0]-fd[key][1][1] > 1:
                                    pc_bigger_1 += 1
                                    if fd[key][1][0]+fd[key][1][1] >pc_biggest:
                                        pc_biggest =fd[key][1][0]+fd[key][1][1]
                                if fd[key][1][2]+fd[key][1][3] < 1e-6 or \
                                    fd[key][1][2]-fd[key][1][3] > 1e-2:
                                    tau_outside += 1
                            temp_2_count += len(fd[key])
                        if temp_2_count == 2:
                            two_trap_count += 1
                            # overall trap count, before sub-el location
                            pre_sub_el_count += 2
                        if temp_2_count == 1:
                            one_trap_count += 1
                            # overall trap count, before sub-el location
                            pre_sub_el_count += 1
                        # check if no prob func 1 fits found
                        if curr_sch == 1 and k_prob not in fd:
                            no_prob1a += 1
                        rc_above[i]['fit_data_above'] = fd
                for no_fit_coord in del_a_list:
                    rc_above.__delitem__(no_fit_coord)
                # curve-fit 'below' traps
                fit_b_count = 0
                for i in rc_below:
                    loc_med_min = rc_below[i]['loc_med_min']
                    loc_med_max = rc_below[i]['loc_med_max']
                    # set average offset to expected level after
                    #illumination_correction(), which is 0 (b/c no negative
                    # counts), by subtracting
                    loc_avg = (loc_med_min+loc_med_max)/2
                    off_min = 0 - max(offset_min,np.abs(loc_avg - loc_med_max))
                    off_max = 0 + max(offset_max,np.abs(loc_med_min - loc_avg))
                    if not tfit_const:
                        fd = trap_fit(curr_sch, rc_below[i]['amps_below'],
                            timings, num_pumps, tau_fit_thresh, tau_min,
                            tau_max, tauc_min, tauc_max, off_min,
                            off_max) #, k_min, k_max)
                    if tfit_const:
                        fd = trap_fit_const(curr_sch,
                            rc_below[i]['amps_below'], timings, num_pumps,
                            tau_fit_thresh, tau_min, tau_max, pc_min, pc_max,
                            off_min, off_max)
                    if fd is None:
                        bad_fit_counter += 1
                        print('bad fit below: ', i)
                        del_b_list.append(i)
                    if fd is not None:
                        fit_b_count += 1
                        # find out how many pixels were fitted for 2 traps
                        temp_2_count = 0
                        for key in fd.keys():
                            # only meaningful if trap_fit_const() used
                            #(for internal testing)
                            #Pc - Pc_err >1:
                            if fd[key][0][0]-fd[key][0][1] > 1:
                                pc_bigger_1 += 1
                                if fd[key][0][0]+fd[key][0][1] > pc_biggest:
                                    pc_biggest = fd[key][0][0]+fd[key][0][1]
                            # only meaningful if tau bounds outside 1e-6, 1e-2
                            #(for internal testing)
                            #tau + tau_err < 1e-6, tau - tau_err < 1e-2
                            if fd[key][0][2]+fd[key][0][3] < 1e-6 or \
                                fd[key][0][2]-fd[key][0][3] > 1e-2:
                                tau_outside += 1
                            if len(fd[key]) == 2:
                                if fd[key][1][0]-fd[key][1][1] > 1:
                                    pc_bigger_1 += 1
                                    if fd[key][1][0]+fd[key][1][1] >pc_biggest:
                                        pc_biggest =fd[key][1][0]+fd[key][1][1]
                                if fd[key][1][2]+fd[key][1][3] < 1e-6 or \
                                    fd[key][1][2]-fd[key][1][3] > 1e-2:
                                    tau_outside += 1
                            temp_2_count += len(fd[key])
                        if temp_2_count == 2:
                            two_trap_count += 1
                            # overall trap count, before sub-el location
                            pre_sub_el_count += 2
                        if temp_2_count == 1:
                            one_trap_count += 1
                            # overall trap count, before sub-el location
                            pre_sub_el_count += 1
                        # check if no prob func 1 fits found
                        if curr_sch == 1 and k_prob not in fd:
                            no_prob1b += 1
                        rc_below[i]['fit_data_below'] = fd
                for no_fit_coord in del_b_list:
                    rc_below.__delitem__(no_fit_coord)
                # curve-fit 'both' traps
                fit_bo_count = 0
                for i in rc_both:
                    loc_med_min = rc_both[i]['loc_med_min']
                    loc_med_max = rc_both[i]['loc_med_max']
                    # set average offset to expected level after
                    #illumination_correction(), which is 0 (b/c no negative
                    # counts), by subtracting
                    loc_avg = (loc_med_min+loc_med_max)/2
                    off_min = 0 - max(offset_min,np.abs(loc_avg - loc_med_max))
                    off_max = 0 + max(offset_max,np.abs(loc_med_min - loc_avg))
                    if not tfit_const:
                        fd = trap_fit(curr_sch, rc_both[i]['amps_both'],
                            timings, num_pumps, tau_fit_thresh, tau_min,
                            tau_max, tauc_min, tauc_max, off_min,
                            off_max, #k_min, k_max,
                            both_a = rc_both[i]['above'])
                    if tfit_const:
                        fd = trap_fit_const(curr_sch, rc_both[i]['amps_both'],
                            timings, num_pumps, tau_fit_thresh, tau_min,
                            tau_max, pc_min, pc_max, off_min,
                            off_max, both_a = rc_both[i]['above'])
                    if fd is None:
                        bad_fit_counter += 1
                        print('bad fit both: ', i)
                        del_bo_list.append(i)
                    if fd is not None:
                        fit_bo_count += 1
                        # all pixels in 'both' are 2-traps
                        two_trap_count += 1
                        # overall trap count, before sub-el location
                        pre_sub_el_count += 2
                        temp_2_count = 0
                        for val in fd.values():
                            for pval in val.values():
                            # only meaningful if trap_fit_const() used
                            #(for internal testing)
                            #Pc - Pc_err > 1:
                                if pval[0][0]-pval[0][1] > 1:
                                    pc_bigger_1 += 1
                                    if pval[0][0]+pval[0][1] > pc_biggest:
                                        pc_biggest = pval[0][0]+pval[0][1]
                                # meaningful if tau bounds outside 1e-6, 1e-2
                                #(for internal testing)
                                #tau + tau_err < 1e-6, tau - tau_err < 1e-2:
                                if pval[0][2]+pval[0][3] < 1e-6 or \
                                    pval[0][2]-pval[0][3] > 1e-2:
                                    tau_outside += 1
                        # check if no prob func 1 fits found
                        if curr_sch == 1 and (k_prob not in fd['a']) and \
                            (k_prob not in fd['b']):
                            no_prob1bo += 1
                        rc_both[i]['fit_data_both'] = fd
                        # make new dictionary for this coord in 'above' since
                        # by construction it doesn't exist as an 'above'
                        # entry yet
                        rc_above[i] = {'fit_data_above': fd['a'],
                            'amps_above': rc_both[i]['above']['amp']}
                        # similarly for 'below'
                        rc_below[i] = {'fit_data_below': fd['b'],
                            'amps_below': rc_both[i]['below']['amp']}
                for no_fit_coord in del_bo_list:
                    rc_both.__delitem__(no_fit_coord)

                print('temperature: ', curr_temp, ', scheme: ', curr_sch,
                    ', number of two-trap pixels (before sub-electrode '
                    'location): ', two_trap_count, ', number of one-trap '
                    'pixels (before sub-electrode location): ',
                    one_trap_count, ', eperdn: ', eperdn)
                print('above traps found: ', rc_above.keys())
                print('below traps found: ', rc_below.keys())
                print('both traps found: ', rc_both.keys())
                print('bad fit counter: ', bad_fit_counter, '_____________')
                if curr_sch == 1 and \
                    (fit_a_count + fit_b_count + fit_bo_count -
                    (no_prob1a + no_prob1b + no_prob1bo)) == 0:
                    #so no traps for prob 1 in scheme 1, so eperdn should be
                    # set according to prob 2 in scheme 1
                    tot = fit_a_count + fit_b_count + fit_bo_count
                    # could still be traps in barrier electrodes 1 and 3, but
                    # realistically none if this fails with both choices of
                    # k_prob
                    raise TPumpAnException('No traps for probability function '
                    '1 found under scheme 1. Re-run with prob_factor_eperdn '
                    '= 4/27 and k_prob = 2. If that failed, no traps '
                    'present in frame. Total # of trap pixels found for '
                    'scheme 1 at current temperature: ', tot)
                #TODO v2.0: COULD happen that you happen traps in schemes 2, 3,
                #  and 4 but none in sch 1.  Could account for this, but
                # wouldn't have very good statistics in schemes 3 and 4 for
                # traps that approach Pc ~ 1.  In any case, no traps in sch 1
                # would be very rare.
                # collecting in convenient dictionary for further manipulation
                schemes[curr_sch] = {'rc_above': rc_above,
                    'rc_below': rc_below, 'rc_both': rc_both,
                    'timings': timings}
            # for a given scheme, the bright pixel coords for "above" CAN
            # be the same as those for "below" (for different ranges of phase
            # time or overlapping phase times)
            traps = {}
            # rc_both has been split up into rc_above and rc_below
            # For a given scheme, rc_below, rc_above, rc_both: each has
            # unique coord keys, but across dictionaries, there is overlap now
            # for the 'both' coords

            # getting coords that are shared across schemes
            def _el_loc_coords2(sch1, or1, sch2, or2):
                """Gets coordinates of dipoles shared across 2 schemes (sch1
                and sch2) with orientations or1 and or2 at a
                given temperature.
                For the purpose of sub-electrode location."""
                coords = list(set(schemes[sch1]['rc_'+or1]) -
                    (set(schemes[sch1]['rc_'+or1]) -
                    set(schemes[sch2]['rc_'+or2])))
                return coords

            def _el_loc_coords3(sch1, or1, sch2, or2, sch3, or3):
                """Gets coordinates of dipoles shared across 3 schemes (sch1,
                sch2, and sch3) with orientations or1 and or2 and or3 at a
                given temperature.
                For the purpose of sub-electrode location."""
                coords12 = _el_loc_coords2(sch1, or1, sch2, or2)
                coords23 = _el_loc_coords2(sch2, or2, sch3, or3)
                coords123 = list(set(coords12) - (set(coords12) -
                    set(coords23)))
                return coords123

            def _el_loc_2(sch1, or1, prob1, sch2, or2, prob2, i, subel_loc):
                """At a given temperature, for coordinate i shared across 2
                schemes (sch1 and sch2), match up dipoles that meet the
                orientation (or1 and or2) and probability function
                specifications (prob1 and prob2) for a sub-electrode location
                (subel_loc) with release time constant (tau) values and
                uncertainties such that uncertainty window across both schemes
                is minimized (thus identifying these data with the same
                physical trap).   It also outputs capture probability info
                which may be useful for future analysis.  It appends all this
                to the traps dictionary and deletes these dipole entries from
                the schemes dictionary so that they aren't used for matching up
                again.
                """
                max_amp1 = max(schemes[sch1]['rc_'+or1][i]['amps_'+or1])
                max_amp2 = max(schemes[sch2]['rc_'+or2][i]['amps_'+or2])
                fd1 = schemes[sch1]['rc_'+or1][i]['fit_data_'+or1]
                fd2 = schemes[sch2]['rc_'+or2][i]['fit_data_'+or2]
                if prob1 in fd1 and prob2 in fd2:
                    # to account for the case of 2 same sub-el traps for a
                    # given pixel:
                    iterations = min(len(fd1[prob1]), len(fd2[prob2]))
                    for iteration in range(iterations):
                        # initialize minimum difference b/w lower and upper
                        # bounds on tau
                        min_range = np.inf
                        j0 = None
                        k0 = None
                        tau_avg = None
                        tau_up = None
                        for j in fd1[prob1]:
                            for k in fd2[prob2]:
                                # find pair that minimizes overall uncertainty
                                # window; it's okay if individual tau windows
                                # don't overlap since systematic errors may
                                # have affecte them
                                #index 2: corresponds to tau
                                #index 3: corresponds to tau_err
                                up = max(j[2] + j[3], k[2] + k[3])
                                low = min(j[2] - j[3], k[2] - k[3])
                                if (up-low) < min_range:
                                    min_range = up - low
                                    j0 = j
                                    k0 = k
                                    tau_avg = (up + low)/2
                                    tau_up = up
                        # TODO v2.0:  seek absolute best pairings by
                        # across all traps in a given pixel by
                        # earmarking the fit data that have been
                        # selected above, and after full matching-
                        # up process, compare the earmarked ones
                        # to determine optimal sets so that the reported
                        # uncertainties for each trap's tau is overall
                        # minimized.
                        # Could append to
                        # each selected fit data list the j and k values of the
                        # other fit data set(s) it was matched with.
                        if (j0 is not None) and (k0 is not None) and (tau_avg
                            is not None) and (tau_up is not None):
                            #  Need amps in traps output? Sure, max amp value,
                            # so that the starting charge packet and the max
                            # amp after all the pumps gives a range of charge
                            # packet value corresponding to tau values
                            # (even though the fitting itself assumed constant
                            # charge packet value).
                            cap1 = j0[0]
                            cap1_err = j0[1]
                            cap2 = k0[0]
                            cap2_err = k0[1]
                            # pixel could have more than one trap,
                            #so trap[i] could already exist at given iteration
                            if i not in traps:
                                traps[i] = []
                            traps[i].append([subel_loc, tau_avg,
                                            tau_up - tau_avg,
                                            [cap1, cap1_err, max_amp1, cap2,
                                            cap2_err, max_amp2], iteration])
                            # now remove these trap data b/c they've been
                            #  matched
                            fd1[prob1].remove(j0)
                            fd2[prob2].remove(k0)

            def _el_loc_3(sch1, or1, prob1, sch2, or2, prob2, sch3, or3, prob3,
                i, subel_loc):
                """At a given temperature, for coordinate i shared across 3
                schemes (sch1, sch2, and sch3), match up dipoles that meet the
                orientation (or1,  or2, and or3) and probability function
                specifications (prob1, prob2, and prob3) for a sub-electrode
                location (subel_loc) with release time constant (tau) values
                and uncertainties such that uncertainty window across the
                schemes is minimized (thus identifying these data with the same
                physical trap).   It also outputs capture probability info
                which may be useful for future analysis.  It appends all this
                to the traps dictionary and deletes these dipole entries from
                the schemes dictionary so that they aren't used for matching up
                again.
                """
                max_amp1 = max(schemes[sch1]['rc_'+or1][i]['amps_'+or1])
                max_amp2 = max(schemes[sch2]['rc_'+or2][i]['amps_'+or2])
                max_amp3 = max(schemes[sch3]['rc_'+or3][i]['amps_'+or3])
                fd1 = schemes[sch1]['rc_'+or1][i]['fit_data_'+or1]
                fd2 = schemes[sch2]['rc_'+or2][i]['fit_data_'+or2]
                fd3 = schemes[sch3]['rc_'+or3][i]['fit_data_'+or3]
                if prob1 in fd1 and prob2 in fd2 and prob3 in fd3:
                    # to account for the case of 2 same sub-el traps for a
                    # given pixel:
                    iterations = min(len(fd1[prob1]), len(fd2[prob2]))
                    for iteration in range(iterations):
                        # initialize minimum difference b/w central values
                        min_range = np.inf
                        j0 = None
                        k0 = None
                        l0 = None
                        tau_avg = None
                        tau_up = None
                        for j in fd1[prob1]:
                            for k in fd2[prob2]:
                                for l in fd3[prob3]:
                                    # find pair that minimizes overall
                                    # uncertainty window; it's okay if
                                    # individual tau windows
                                    #  don't overlap since systematic errors
                                    # may have affected them
                                    #index 2: corresponds to tau
                                    #index 3: corresponds to tau_err
                                    up = max(j[2] + j[3], k[2] + k[3],
                                        l[2] + l[3])
                                    low= min(j[2] - j[3], k[2] - k[3],
                                        l[2] - l[3])
                                    if(up - low) < min_range:
                                        min_range = up - low
                                        j0 = j
                                        k0 = k
                                        l0 = l
                                        tau_avg = (up + low)/2
                                        tau_up = up
                        # TODO v2.0:  seek absolute best pairings by
                        # across all traps in a given pixel by
                        # earmarking the fit data that have been
                        # selected above, and after full matching
                        # up process, compare the earmarked ones
                        # (along with any fit data that didn't get
                        # chosen for traps) to determine optimal
                        # pairings so that the least amount of fit
                        # data goes unused for trap locations.  Could append to
                        # each selected fit data list the j and k values of the
                        # other fit data set(s) it was matched with.
                        if (j0 is not None) and (k0 is not None) and (l0
                            is not None) and (tau_avg is not None) and (tau_up
                            is not None):

                            cap1 = j0[0]
                            cap1_err = j0[1]
                            cap2 = k0[0]
                            cap2_err = k0[1]
                            cap3 = l0[0]
                            cap3_err = l0[1]
                            # pixel could have more than one trap,
                            #so trap[i] could already exist at given iteration
                            if i not in traps:
                                traps[i] = []
                            traps[i].append([subel_loc, tau_avg,
                                            tau_up - tau_avg,
                                            [cap1, cap1_err, max_amp1, cap2,
                                            cap2_err, max_amp2, cap3, cap3_err,
                                            max_amp3], iteration])
                            # now remove these trap data b/c they've been
                            # matched
                            fd1[prob1].remove(j0)
                            fd2[prob2].remove(k0)
                            fd3[prob3].remove(l0)

            # go through all 3-scheme sub-electrode locations first, since
            # there could be scenarios where a pixel has 2-traps in different
            # schemes that can match a 2-scheme ID and rob from a true 3-scheme
            #only do these operations if all 4 schemes present
            if set([1,2,3,4]) == set(schemes.keys()):
                # LHS of electrode 2: sch 1 'above', sch 2 'below', sch 4 'above'
                LHSel2_coords = _el_loc_coords3(1, 'above', 2, 'below',
                    4, 'above')
                # RHS of electrode 2
                RHSel2_coords = _el_loc_coords3(1, 'above', 2, 'below',
                    4, 'below')
                # LHS of electrode 4
                LHSel4_coords = _el_loc_coords3(1, 'below', 2, 'above',
                    3, 'above')
                # RHS of electrode 4
                RHSel4_coords = _el_loc_coords3(1, 'below', 2, 'above',
                    3, 'below')
                for i in LHSel2_coords:
                    # sch1 'above' prob2, sch2 'below' prob1, sch4 'above'prob3
                    _el_loc_3(1, 'above', 2, 2, 'below', 1, 4, 'above', 3,
                        i, 'LHSel2')
                for i in RHSel2_coords:
                    _el_loc_3(1, 'above', 1, 2, 'below', 2, 4, 'below', 3,
                        i, 'RHSel2')
                for i in LHSel4_coords:
                    _el_loc_3(1, 'below', 1, 2, 'above', 2, 3, 'above', 3,
                        i, 'LHSel4')
                for i in RHSel4_coords:
                    _el_loc_3(1, 'below', 2, 2, 'above', 1, 3, 'below', 3,
                        i, 'RHSel4')
                # now go through all 2-scheme sub-electrode locations
                # LHS of electrode 1: scheme 1 'below' and scheme 3 'below'
                LHSel1_coords = _el_loc_coords2(1, 'below', 3, 'below')
                # center of electrode 1
                CENel1_coords = _el_loc_coords2(3, 'below', 4, 'above')
                # RHS of electrode 1
                RHSel1_coords = _el_loc_coords2(1, 'above', 4, 'above')
                # center of electrode 2
                CENel2_coords = _el_loc_coords2(1, 'above', 2, 'below')
                # LHS of electrode 3
                LHSel3_coords = _el_loc_coords2(2, 'below', 4, 'below')
                # center of electrode 3
                CENel3_coords = _el_loc_coords2(3, 'above', 4, 'below')
                # RHS of electrode 3
                RHSel3_coords = _el_loc_coords2(2, 'above', 3, 'above')
                # center of electrode 4
                CENel4_coords = _el_loc_coords2(1, 'below', 2, 'above')
                for i in LHSel1_coords:
                    # scheme 1 'below' prob 1, scheme 3 'below' prob 3
                    _el_loc_2(1, 'below', 1, 3, 'below', 3, i, 'LHSel1')
                for i in CENel1_coords:
                    _el_loc_2(3, 'below', 2, 4, 'above', 2, i, 'CENel1')
                for i in RHSel1_coords:
                    _el_loc_2(1, 'above', 1, 4, 'above', 3, i, 'RHSel1')
                for i in CENel2_coords:
                    _el_loc_2(1, 'above', 1, 2, 'below', 1, i, 'CENel2')
                for i in LHSel3_coords:
                    _el_loc_2(2, 'below', 1, 4, 'below', 3, i, 'LHSel3')
                for i in CENel3_coords:
                    _el_loc_2(3, 'above', 2, 4, 'below', 2, i, 'CENel3')
                for i in RHSel3_coords:
                    _el_loc_2(2, 'above', 1, 3, 'above', 3, i, 'RHSel3')
                for i in CENel4_coords:
                    _el_loc_2(1, 'below', 1, 2, 'above', 1, i, 'CENel4')


            # see how many dipoles were not matched up in sub-electrode
            # location
            unused_fit_data = 0
            for sch in schemes.values():
                for coord in sch['rc_above'].values():
                    unused_fit_data += len(coord['fit_data_above'])
                for coord in sch['rc_below'].values():
                    unused_fit_data += len(coord['fit_data_below'])

            temps[curr_temp] = traps
        if save_temps is not None:
            try:
                np.save(save_temps, temps)
            except:
                warnings.warn('File path for saving temps dictionary '
                'was not recognized.')

    if load_temps is not None:
        try:
            temps = np.load(load_temps, allow_pickle=True)
            temps = temps.tolist()
        except:
            raise FileNotFoundError('load_temps file not found')



    # initializing
    trap_list = []
    for temp in temps.values():
        for pix, trs in temp.items():
            for tr in trs:
                # including tr[-1], which is 'iteration' since there could be
                #more than 1 trap in same sub-el location of a given pixel
                if (pix, tr[0], tr[-1]) not in trap_list:
                    trap_list.append((pix, tr[0], tr[-1]))
    # structural formats below:
    # trap_list = [((row,col), 'sub_el', iteration), ...]
    #temps = {T1: {(row,col): [['sub_el', tau, sigma_tau,
    # [cap info], iteration],..],..},...}
    # gives an list of temperatures in ascending order so that the selection of
    # tau values to match up in a given pixel and sub-el location possible
    sorted_temps = sorted(temps)
    #trap_dict = {((row,col),'sub_el',iteration): {'T': [temps],
    # 'tau': [corresponding taus], 'sigma_tau': [simga_taus],
    # 'cap':[[cap info at T1], [cap info at T2],..]},..}
    # initializing
    trap_dict = {}
    # pick out traps in same pixel and sub-el location, and for each next temp
    # step, the tau value that's closest to previous (i.e., that falls along
    # same tau vs temp curve) since there could be multiple traps in the same
    # sub-el location with different tau values
    for pix_el in trap_list:
        for temp in sorted_temps:
            if pix_el[0] in temps[temp]: #if coord is present in this temp
                dist_tau = np.inf # initialize
                tr0 = 0 # initialize
                tr1 = 0 # initialize
                for tr in temps[temp][pix_el[0]]:
                    if tr[0] == pix_el[1]: # if same sub-el location
                        if pix_el not in trap_dict:
                            trap_dict[pix_el] = {'T': [temp], 'tau': [tr[1]],
                            'sigma_tau': [tr[2]], 'cap': [tr[3]]}
                            tr0 = tr
                            break
                            #Euclidean distance to next tau in tau-temp space
                        elif ((trap_dict[pix_el]['T'][-1] - temp)**2 +
                            (trap_dict[pix_el]['tau'][-1] - tr[1])**2) < \
                                dist_tau:
                            dist_tau = ((trap_dict[pix_el]['T'][-1] - temp)**2
                            + (trap_dict[pix_el]['tau'][-1] - tr[1])**2)
                            tr1 = tr
                if tr0 != 0:
                    temps[temp][pix_el[0]].remove(tr0)
                if tr1 != 0:
                    trap_dict[pix_el]['T'].append(temp)
                    trap_dict[pix_el]['tau'].append(tr1[1])
                    trap_dict[pix_el]['sigma_tau'].append(tr1[2])
                    trap_dict[pix_el]['cap'].append(tr1[3])
                    temps[temp][pix_el[0]].remove(tr1)
    #see how many unused sets of fit data left when getting taus for
    # each temp (physically, should be 0 if all were associated with traps)
    unused_temp_fit_data = 0
    for temp in temps.values():
        for rc in temp.values():
            unused_temp_fit_data += len(rc)
    # count how many traps only have data for 2 or fewer temperatures
    two_or_less_count = 0
    # count how many traps have gaps in the temperatures (a trap should show
    # up over a continuous range of temps)
    noncontinuous_count = 0
    # lists for making histogram to categorize for outputting trap densities
    E_vals = []
    cs_vals = []
    for pix_el in trap_dict.values():
        if len(pix_el['T']) <= 2:
            two_or_less_count += 1
        T1 = pix_el['T'][0]
        Tf = pix_el['T'][-1]
        if len(pix_el['T']) - 1 != (sorted_temps.index(Tf) -
            sorted_temps.index(T1)):
            noncontinuous_count += 1

        taus = np.array(pix_el['tau'])
        tau_errs = np.array(pix_el['sigma_tau'])
        temperatures = np.array(pix_el['T'])
        (E, sig_E, cs, sig_cs, Rsq, tau_input_T,
         sig_tau_input_T) = fit_cs(taus, tau_errs,
            temperatures, cs_fit_thresh, E_min, E_max, cs_min, cs_max, input_T)
        pix_el['E'] = E # in eV
        pix_el['sig_E'] = sig_E # in eV
        pix_el['cs'] = cs # in cm^2
        pix_el['sig_cs'] = sig_cs # in cm^2
        pix_el['Rsq'] = Rsq
        pix_el['tau at input T'] = tau_input_T # in s
        pix_el['sig_tau at input T'] = sig_tau_input_T # in s
        if Rsq is not None:
            if Rsq >= cs_fit_thresh:
                E_vals.append(E)
                cs_vals.append(cs)
    E_vals = np.array(E_vals)
    cs_vals = np.array(cs_vals)
    #originally used for default: bins = [int(1e-19/1e-21) , int(1e-17/1e-20)]
    bins = [bins_E, bins_cs]

    trap_densities = []
    if len(E_vals) != 0 and len(cs_vals) != 0:
        H, E_edges, cs_edges = np.histogram2d(E_vals, cs_vals, bins = bins,
            range = [[0, max(1, E_vals.max()) + 0.5/bins[0]],
                [0, max(1e-14, cs_vals.max()) + 1e-14*0.5/bins[1]]])

        nrows, ncols, _ = imaging_area_geometry()
        for i in range(bins[0]):
            for j in range(bins[1]):
                # [percentage of traps out of total # pixels, avg E bin value,
                # avg cs bin value]
                if H[i, j] != 0:
                    trap_densities.append([H[i, j]/(nrows*ncols),
                        (E_edges[i+1] + E_edges[i])/2,
                        (cs_edges[j+1] + cs_edges[j])/2])

    return (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count)


def create_TrapCalibration_from_trap_dict(trap_dict,base_dir):
    '''
    A function that converts a trap dictionary into a corgidrp.data.TrapCalibration
    file

    We will recode the string parts of that dictionary specifying the 
    sub-electrode location for a given pixel to a number code. An example of such 
    a string is 'RHSel2', which denotes the right-hand side of electrode 2. It can 
    be 'RHS', 'LHS', or 'CEN' for electrodes 1 through 4. So that can be 
    converted to a number code (1: 'LHS', 2: 'CEN', 3: 'RHS'), so that 'RHSel2' 
    would be 32. 

    Args: 
        trap_dict (dict): A dictionary output by tpump_analysis
    Returns: 
        trap_cal (corgidrp.data.TrapCalibration): A trap calibration file. 
    '''
    
    n_traps = len(trap_dict)

    electrode_dict = {"LHS": 10, 
                      "CEN": 20,
                      "RHS": 30}

    #Create the main array of traps and their properties, and fill it in
    trap_cal_array = np.zeros([n_traps,10])
    for i,key in enumerate(trap_dict.keys()): 
        key_list = list(key)

        #The first two elements should be the row and column of the trap
        trap_cal_array[i,0] = key_list[0][0]
        trap_cal_array[i,1] = key_list[0][1]

        #Convert the electrode_string into a number (described in the docstring)
        split_electrode_str = key_list[1].split("el")
        trap_cal_array[i,2] = electrode_dict[split_electrode_str[0]] + \
                                int(split_electrode_str[1])

        #Which trap is this at this location? 
        trap_cal_array[i,3] = key_list[2]
        
        #Now extract the dictionary entry: 
        dict_entry = trap_dict[key]

        #Grab the first time constant 
        #TODO: It seems like there could be three of these. How do we choose which one? 
        trap_cal_array[i,4] = dict_entry['cap'][0][0]

        #The maximum amplitude of the dipole
        #TODO: There could also be three of these
        trap_cal_array[i,5] = dict_entry['cap'][0][2]

        #The energy level of the hole
        trap_cal_array[i,6] = dict_entry['E']

        #The cross section for holes
        trap_cal_array[i,7] = dict_entry['cs']

        #R^2 value of fit
        trap_cal_array[i,8] = dict_entry['Rsq']

        #Release time constant at desired Temperature
        trap_cal_array[i,9] = dict_entry['tau at input T']

    #We need headers and filenames. 
    all_files = sorted(glob.glob(base_dir+"**/*.fits",recursive=True))

    #Make a dataset with these files
    dataset = Dataset(all_files)

    #Great the header from the fist file
    first_file_pri_hdr = fits.open(all_files[0])[0].header
    first_file_ext_hdr = fits.open(all_files[0])[1].header

    trapcal = TrapCalibration(trap_cal_array,pri_hdr = first_file_pri_hdr, 
                    ext_hdr = first_file_ext_hdr, 
                    input_dataset=dataset)
    
    return trapcal
                    
# trap_dict = {
# ((row, col), 'RHSel2', 0): {'T': [160, 162, 164, 166, 168],
# 'tau': [1.51e-6, 1.49e-6, 1.53e-6, 1.50e-6, 1.52e-6],
# 'sigma_tau': [2e-7, 1e-7, 1e-7, 2e-7, 1e-7],
# 'cap': [[cap1, cap1_err, max_amp1, cap2, cap2_err, max_amp2], ...],
# 'E': 0.23,
# 'sig_E': 0.02,
# 'cs': 2.6e-15,
# 'sig_cs': 0.3e-15,
# 'Rsq': 0.96,
# 'tau at input T': 1.61e-6,
# 'sig_tau at input T': 2.02e-6},
# ...}


if __name__ == "__main__":
    fake_dict = {
        ((150, 25), 'RHSel2', 0): {'T': [160, 162, 164, 166, 168],
        'tau': [1.51e-6, 1.49e-6, 1.53e-6, 1.50e-6, 1.52e-6],
        'sigma_tau': [2e-7, 1e-7, 1e-7, 2e-7, 1e-7],
        # 'cap': [[cap1, cap1_err, max_amp1, cap2, cap2_err, max_amp2], ...],
        'cap': [[4, 5, 6, 4, 5, 6],],
        'E': 0.23,
        'sig_E': 0.02,
        'cs': 2.6e-15,
        'sig_cs': 0.3e-15,
        'Rsq': 0.96,
        'tau at input T': 1.61e-6,
        'sig_tau at input T': 2.02e-6},
        }
    
    base_dir = "/Users/maxwellmb/Data/GPI/Reduced/CE Ant/20180405_H_Pol/"
    
    test = create_TrapCalibration_from_trap_dict(fake_dict,base_dir)




# if __name__ == '__main__':
#     here = os.path.abspath(os.path.dirname(__file__))
#     base_dir = Path(here, 'test_data')
#     sub_no_noise_dir = Path(here, 'test_data_sub_frame_no_noise')
#     sub_noise_dir = Path(here, 'test_data_sub_frame_noise')
#     sub_noise_one_temp_dir = Path(here, 'test_data_sub_frame_noise_one_temp')
#     sub_noise_kprob2_dir = Path(here, 'test_data_sub_frame_noise_no_prob1')
#     sub_noise_mean_field = Path(here, 'test_data_sub_frame_noise_mean_field')
#     test_data_sample_data = Path(here, 'test_data_sample_data')
#     test_data_bad_sch = Path(here, 'test_data_bad_sch_label')
#     empty_base_dir = Path(here, 'test_data_empty_base_dir')
#     time_head = 'PHASE_T'
#     emgain_head = 'EM_GAIN'
#     #MMB: If we want to use this, then we need to pass in a corgi.drp.NonLinearityCorrection object
#     # Setting to None for now
#     non_lin_correction = None 
#     num_pumps = 10000
#     tau_fit_thresh = 0.8#0.9#0.9#0.8
#     cs_fit_thresh = 0.8
#     thresh_factor = 1.5#1.5 #3
#     length_lim = 5
#     ill_corr = True
#     tfit_const = True
#     offset_min = 10
#     offset_max = 10
#     pc_min = 0
#     pc_max = 2
#     mean_field = None#2090 #250 #e- #None
#     tauc_min = 0
#     tauc_max = 1e-5 #1e-2
#     k_prob = 1
#     bins_E = 70#100#80 # at 80% for noisy, with inj charge
#     bins_cs = 7#10#8 # at 80%
#     sample_data = False #True


#     (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
#     unused_fit_data, unused_temp_fit_data, two_or_less_count,
#     noncontinuous_count) = tpump_analysis(sub_noise_dir, time_head,
#     emgain_head, num_pumps, non_lin_correction = non_lin_correction,
#     length_lim = length_lim, thresh_factor = thresh_factor,
#     ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
#     tau_min = 0.7e-6, tau_max = 1.3e-2, tau_fit_thresh = tau_fit_thresh,
#     tauc_min = tauc_min, tauc_max = tauc_max, offset_min = offset_min,
#     offset_max = offset_max,s
#     pc_min=pc_min, pc_max=pc_max, k_prob = k_prob, mean_field = mean_field,
#     cs_fit_thresh = cs_fit_thresh, bins_E = bins_E, bins_cs = bins_cs,
#     sample_data = sample_data)