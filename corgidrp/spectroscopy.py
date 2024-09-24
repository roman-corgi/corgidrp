import numpy as np
import os
import corgidrp.data
from dataclasses import dataclass
from astropy.table import Table
from scipy.interpolate import interp1d
import scipy.ndimage as ndi
import scipy.optimize as optimize
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt

class DispersionModel():
    """ 
    Class for dispersion model parameter data structure

    Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR the 2D image data

    Attributes:
        data (numpy.lib.npyio.NpzFile): numpy Npz file

        clocking_angle (float): Clocking angle of the dispersion axis, theta,
        oriented in the direction of increasing wavelength, measured in degrees
        counterclockwise from the positive x-axis on the EXCAM data array
        (direction of increasing column index).

        clocking_angle_uncertainty (float): Uncertainty of the dispersion axis
        clocking angle in degrees.

        pos_vs_wavlen_polycoeff (numpy.ndarray): Polynomial fit to the
        source displacement on EXCAM along the dispersion axis as a function of
        wavelength, relative to the source position at the band reference
        wavelength (lambda_c = 730.0 nm for Band 3) in units of millimeters.

        pos_vs_wavlen_cov (numpy.ndarray): Covariance matrix of the
        polynomial coefficients

        wavlen_vs_pos_polycoeff (numpy.ndarray): Polynomial fit to the
        wavelength as a function of displacement along the dispersion axis on
        EXCAM, relative to the source position at the Band 3 reference
        wavelength (x_c at lambda_c = 730.0 nm) in units of nanometers. 

        wavlen_vs_pos_cov (numpy.ndarray): Covariance matrix of the
        polynomial coefficients
    """

    def __init__(self, data_or_filepath=None,
                 clocking_angle=None, clocking_angle_uncertainty=None,
                 wavlen_vs_pos_polycoeff=None, wavlen_vs_pos_cov=None,
                 pos_vs_wavlen_polycoeff=None, pos_vs_wavlen_cov=None):
        if isinstance(data_or_filepath, str):
            # a filepath is passed in
            dispersion_params = np.load(data_or_filepath)
            if 'clocking_angle' in dispersion_params: 
                self.clocking_angle = dispersion_params['clocking_angle']
            if 'clocking_angle_uncertainty' in dispersion_params: 
                self.clocking_angle_uncertainty = dispersion_params['clocking_angle_uncertainty']
            if 'pos_vs_wavlen_polycoeff' in dispersion_params:
                self.pos_vs_wavlen_polycoeff = dispersion_params['pos_vs_wavlen_polycoeff']
            if 'pos_vs_wavlen_cov' in dispersion_params:
                self.pos_vs_wavlen_cov = dispersion_params['pos_vs_wavlen_cov']
            if 'wavlen_vs_pos_polycoeff' in dispersion_params:
                self.wavlen_vs_pos_polycoeff = dispersion_params['wavlen_vs_pos_polycoeff']
            if 'wavlen_vs_pos_cov' in dispersion_params:
                self.wavlen_vs_pos_cov = dispersion_params['wavlen_vs_pos_cov']

            # parse the filepath to store the filedir and filename
            filepath_args = data_or_filepath.split(os.path.sep)
            if len(filepath_args) == 1:
                # no directory info in filepath, so current working directory
                self.filedir = "."
                self.filename = filepath_args[0]
            else:
                self.filename = filepath_args[-1]
                self.filedir = os.path.sep.join(filepath_args[:-1])
        else:
            # initialization data passed in directly
            self.data = data_or_filepath
            self.clocking_angle = clocking_angle
            self.clocking_angle_uncertainty = clocking_angle_uncertainty
            self.pos_vs_wavlen_polycoeff = pos_vs_wavlen_polycoeff
            self.pos_vs_wavlen_cov = pos_vs_wavlen_cov
            self.wavlen_vs_pos_polycoeff = wavlen_vs_pos_polycoeff
            self.wavlen_vs_pos_cov = wavlen_vs_pos_cov

    # create this field dynamically
    @property
    def filepath(self):
        return os.path.join(self.filedir, self.filename)

    def save(self, filedir=None, filename=None):
        """
        Save file to disk with user specified filepath

        Args:
            filedir (str): filedir to save to. Use self.filedir if not specified
            filename (str): filepath to save to. Use self.filename if not specified
        """
        if filename is not None:
            self.filename = filename
        if filedir is not None:
            self.filedir = filedir

        if len(self.filename) == 0:
            raise ValueError("Output filename is not defined. Please specify!")

        np.savez(self.filepath,
            clocking_angle = self.clocking_angle,
            clocking_angle_uncertainty = self.clocking_angle_uncertainty,
            pos_vs_wavlen_polycoeff = self.pos_vs_wavlen_polycoeff ,
            pos_vs_wavlen_cov = self.pos_vs_wavlen_cov,
            wavlen_vs_pos_polycoeff = self.wavlen_vs_pos_polycoeff,
            wavlen_vs_pos_cov = self.wavlen_vs_pos_cov)

@dataclass
class WavelengthZeropoint():
    """
    Class for a wavelength zero-point data structure.

    Attributes:

    prism (str): Label for the DPAM zero-deviation prism; must be either
    'PRISM3' or 'PRISM2'. 

    wavlen (float): Wavelength of the zero-point (nanometers)

    x (float): x-coordinate of the zero-point position (EXCAM array columns)

    x_err (float): x-coordinate uncertainty of the zero-point position
    (EXCAM array columns)

    y (float): y-coordinate of the zero-point position (EXCAM array rows)

    y_err (float): y-coordinate uncertainty of the zero-point position
    (EXCAM array columns)

    image_shape (tuple): shape tuple of the 2D image array
    """
    prism: str
    wavlen: float
    x: float
    x_err: float
    y: float
    y_err: float
    image_shape: tuple

def get_center_of_mass(frame):
    """
    Finds the center coordinates for a given frame.

    Args:
        frame (np.ndarray): 2D array to compute centering

    Returns:
        tuple:
            xcen (float): X centroid coordinate
            ycen (float): Y centroid coordinate

    """
    y, x = np.indices(frame.shape)
    
    ycen = np.sum(y * frame)/np.sum(frame)
    xcen = np.sum(x * frame)/np.sum(frame)
    
    return xcen, ycen

def gauss2d(x0, y0, sigma_x, sigma_y, peak):
    """
    2d guassian function for guassfit2d

    Args:
        x0,y0: center of gaussian
        peak: peak amplitude of guassian
        sigma_x,sigma_y: stddev in x and y directions
    """
    return lambda y,x: peak * np.exp(-( ((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) **2 ) / 2)

def gauss1d(x0, sigma, peak):
    """
    1d guassian function for guassfit1d

    Args:
        x0: center of gaussian
        peak: peak amplitude of guassian
        sigma: stddev
    """
    return lambda x: peak * np.exp(-( ((x - x0) / sigma) ** 2 ) / 2)

def gaussfit2d_pix(frame, xguess, yguess, xfwhm_guess=3, yfwhm_guess=6, 
                   halfwidth=3, halfheight=5, guesspeak=1, oversample=5, refinefit=True):
    """
    Fits a 2-d Gaussian to the data at point (xguess, yguess), with pixel integration.

    Args:
        frame: the data - Array of size (y,x)
        xguess, yguess: location to fit the 2d guassian to (should be within +/-1 pixel of true peak)
        xfwhm_guess: approximate x-axis fwhm to fit to
        yfwhm_guess: approximate y-axis fwhm to fit to    
        halfwidth: 1/2 the width of the box used for the fit
        halfheight: 1/2 the height of the box used for the fit
        guesspeak: approximate flux in peak pixel
        oversample: odd integer >= 3; to represent detector pixels, overample and then bin the model by this factor 
        refinefit: whether to refine the fit of the position of the guess

    Returns:
        xfit: x position (only chagned if refinefit is True)
        yfit: y position (only chagned if refinefit is True)
        xfwhm: x-axis fwhm of the PSF in pixels
        yfwhm: y-axis fwhm of the PSF in pixels
        peakflux: the peak value of the gaussian
        fitbox: 2-d array of the fitted region from the data array  
        model: 2-d array of the best-fit model
        residual: 2-d array of residuals (data - model)

    """
    if not isinstance(halfwidth, int):
        raise ValueError("halfwidth must be an integer")
    if not isinstance(halfheight, int):
        raise ValueError("halfheight must be an integer")
    if not isinstance(oversample, int) or (oversample % 2 != 1) or (oversample < 3):
        raise ValueError("oversample must be an odd integer >= 3")

    x0 = np.rint(xguess).astype(int)
    y0 = np.rint(yguess).astype(int)
    fitbox = np.copy(frame[y0 - halfheight:y0 + halfheight + 1,
                           x0 - halfwidth:x0 + halfwidth + 1])
    nrows = fitbox.shape[0]
    ncols = fitbox.shape[1]
    fitbox[np.where(np.isnan(fitbox))] = 0

    oversampled_ycoord = np.linspace(-(oversample // 2) / oversample, 
                                     nrows - 1 + (oversample // 2) / oversample + 1./oversample,
                                     nrows * oversample)
    oversampled_xcoord = np.linspace(-(oversample // 2) / oversample, 
                                     ncols - 1 + (oversample // 2) / oversample + 1./oversample,
                                     ncols * oversample)
    oversampled_grid = np.meshgrid(oversampled_ycoord, oversampled_xcoord, indexing='ij')
    
    if refinefit:
        errorfunction = lambda p: np.ravel(
                np.reshape(gauss2d(*p)(*oversampled_grid), 
                (nrows, oversample, ncols, oversample)).mean(
                axis = 1).mean(axis = 2) - fitbox)
  
        guess = (halfwidth, halfheight, 
                 xfwhm_guess/(2 * np.sqrt(2*np.log(2))),
                 yfwhm_guess/(2 * np.sqrt(2*np.log(2))),
                 guesspeak)

        p, success = optimize.leastsq(errorfunction, guess)

        xfit = p[0] + (x0 - halfwidth)
        yfit = p[1] + (y0 - halfheight)
        xfwhm = p[2] * (2 * np.sqrt(2*np.log(2)))
        yfwhm = p[3] * (2 * np.sqrt(2*np.log(2)))
        peakflux = p[4]

        model = np.reshape(gauss2d(*p)(*oversampled_grid), 
                        (nrows, oversample, ncols, oversample)).mean(
                        axis = 1).mean(axis = 2)
        residual = fitbox - model
    else:
        model = np.reshape(gauss2d(*guess)(*oversampled_grid), 
                        (nrows, oversample, ncols, oversample)).mean(
                        axis = 1).mean(axis = 2)
        residual = fitbox - model

        xfit = xfit
        yfit = yfit
        xfwhm = xfwhm_guess
        yfwhm = yfwhm_guess
        peakflux = guesspeak

    return xfit, yfit, xfwhm, yfwhm, peakflux, fitbox, model, residual

def gaussfit2d(frame, xguess, yguess, xfwhm_guess=3, yfwhm_guess=6, 
               halfwidth=3, halfheight=5, guesspeak=1, refinefit=True):
    """
    Fits a 2-d Gaussian to the data at point (xguess, yguess)

    Args:
        frame: the data - Array of size (y,x)
        xguess, yguess: location to fit the 2d guassian to (should be within +/-1 pixel of true peak)
        xfwhm_guess: approximate x-axis fwhm to fit to
        yfwhm_guess: approximate y-axis fwhm to fit to    
        halfwidth: 1/2 the width of the box used for the fit
        halfheight: 1/2 the height of the box used for the fit
        guesspeak: approximate flux in peak pixel
        refinefit: whether to refine the fit of the position of the guess

    Returns:
        xfit: x position (only chagned if refinefit is True)
        yfit: y position (only chagned if refinefit is True)
        xfwhm: x-axis fwhm of the PSF in pixels
        yfwhm: y-axis fwhm of the PSF in pixels
        peakflux: the peak value of the gaussian
        fitbox: 2-d array of the fitted region from the data array  
        model: 2-d array of the best-fit model
        residual: 2-d array of residuals (data - model)

    """
    if not isinstance(halfwidth, int):
        raise ValueError("halfwidth must be an integer")
    if not isinstance(halfheight, int):
        raise ValueError("halfheight must be an integer")
    
    x0 = np.rint(xguess).astype(int)
    y0 = np.rint(yguess).astype(int)
    #construct our searchbox
    fitbox = np.copy(frame[y0 - halfheight:y0 + halfheight + 1,
                           x0 - halfwidth:x0 + halfwidth + 1])

    #mask bad pixels
    fitbox[np.where(np.isnan(fitbox))] = 0

    #fit a least squares gaussian to refine the fit on the source, otherwise just use the guess
    if refinefit:
        #construct the residual to the fit
        errorfunction = lambda p: np.ravel(gauss2d(*p)(*np.indices(fitbox.shape)) - fitbox)
   
        #do a least squares fit. Note that we use searchrad for x and y centers since we're narrowed it to a box of size
        #(2*halfwidth+1,2*halfheight+1)

        guess = (halfwidth, halfheight, 
                 xfwhm_guess/(2 * np.sqrt(2*np.log(2))),
                 yfwhm_guess/(2 * np.sqrt(2*np.log(2))),
                 guesspeak)

        p, success = optimize.leastsq(errorfunction, guess)

        xfit = p[0] + (x0 - halfwidth)
        yfit = p[1] + (y0 - halfheight)
        xfwhm = p[2] * (2 * np.sqrt(2*np.log(2)))
        yfwhm = p[3] * (2 * np.sqrt(2*np.log(2)))
        peakflux = p[4]

        model = gauss2d(*p)(*np.indices(fitbox.shape))
        residual = fitbox - model
    else:
        model = gauss2d(xguess - x0 + halfwidth, 
                        yguess - y0 + halfheight, 
                        xfwhm_guess/(2 * np.sqrt(2*np.log(2))),
                        yfwhm_guess/(2 * np.sqrt(2*np.log(2))),
                        guesspeak)
        residual = fitbox - model

        xfit = xfit
        yfit = yfit
        xfwhm = xfwhm_guess
        yfwhm = yfwhm_guess
        peakflux = guesspeak

    return xfit, yfit, xfwhm, yfwhm, peakflux, fitbox, model, residual

def gaussfit1d(frame, xguess, fwhm_guess=6, halfwidth=5, guesspeak=1, oversample=5, refinefit=True):
    """
    Fits a Gaussian profile to a 1-d data array

    Args:
        frame: 1-d data array
        xguess: location to fit the guassian to (should be within +/-1 pixel of true peak)
        fwhm_guess: approximate x-axis fwhm to fit to
        halfwidth: 1/2 the width of the box used for the fit
        guesspeak: approximate flux in peak pixel
        oversample: odd integer >= 3; to represent detector pixels, overample and then bin the model by this factor 
        refinefit: whether to refine the fit of the position of the guess

    Returns:
        xfit: position (only chagned if refinefit is True)
        fwhm: fwhm of the PSF in pixels
        peakflux: the peak value of the gaussian
        fitwin: 1-d array of the fitted region from the data array  
        model: 1-d array of the best-fit model
        residual: 1-d array of residuals (data - model)

    """
    if not isinstance(halfwidth, int):
        raise ValueError("halfwidth must be an integer")
    if not isinstance(oversample, int) or (oversample % 2 != 1) or (oversample < 3):
        raise ValueError("oversample must be an odd integer >= 3")

    x0 = np.rint(xguess).astype(int)
    fitwin = np.copy(frame[x0 - halfwidth:x0 + halfwidth + 1])
    npix = fitwin.shape[0]
    fitwin[np.where(np.isnan(fitwin))] = 0

    overampled_coord = np.linspace(-(oversample // 2) / oversample, 
                                   npix - 1 + (oversample // 2) / oversample,
                                   npix * oversample)
    
    if refinefit:
        errorfunction = lambda p: np.reshape(gauss1d(*p)(overampled_coord), (npix, oversample)).mean(axis=1) - fitwin

        guess = (halfwidth, fwhm_guess/(2 * np.sqrt(2*np.log(2))), guesspeak)
        p, success = optimize.leastsq(errorfunction, guess)

        xfit = p[0] + (x0 - halfwidth)
        fwhm = p[1] * (2 * np.sqrt(2*np.log(2)))
        peakflux = p[2]

        model = np.reshape(gauss1d(*p)(overampled_coord), (npix, oversample)).mean(axis=1)
        residual = fitwin - model
    else:
        model = np.reshape(gauss1d(*guess)(oversampled_coord), (npix, oversample)).mean(axis=1)
        residual = fitwin - model

        xfit = xfit
        fwhm = fwhm_guess
        peakflux = guesspeak

    return xfit, fwhm, peakflux, fitwin, model, residual

def fit_line_spread_function(image, wave_cal_map, zeropt, halfwidth = 1, halfheight = 9):
    """
    Fit the line spread function 

    Args:

        image (numpy.ndarray): 2-D image array containg a narrowband filter + prism PSF

        wavlen_map (numpy.ndarray): 2-D wavelength calibration map. Each image
        pixel value is a wavelength in units of nanometers, computed for the
        dispersion profile, zero-point position, coordinates, and image shape
        specified in the input wavelength zero-point object.

        zeropt (spectroscopy.WavelengthZeropoint): Wavelength zero-point data
        object containing the image array coordinates and center wavelength of
        the narrowband signal.

        halfwidth (int): The width of the fitting region is 2 * halfwidth + 1 pixels.
        
        halfheight (int): The height of the fitting region is 2 * halfwidth + 1 pixels.

    Returns:
        
        wavlens (numpy.ndarray)
        
        flux_profile (numpy.ndarray) 

        fwhm_fit (float)
        
        mean_fit (float)

        peak_fit (float)

    """
    xcent_round, ycent_round = (int(np.rint(zeropt.x)), int(np.rint(zeropt.y)))
    image_cutout = image[ycent_round - halfheight:ycent_round + halfheight + 1,
                         xcent_round - halfwidth:xcent_round + halfwidth + 1]

    wave_cal_map_cutout = wave_cal_map[ycent_round - halfheight:ycent_round + halfheight + 1,
                                       xcent_round - halfwidth:xcent_round + halfwidth + 1]

    flux_profile = np.sum(image_cutout, axis=1) / np.sum(image_cutout)
    wavlens = np.mean(wave_cal_map_cutout, axis=1)

    g_init = models.Gaussian1D(amplitude = np.max(flux_profile),
                               mean = wavlens[halfheight], 
                               stddev = 10./(2 * np.sqrt(2*np.log(2))))
    fit_g = fitting.LevMarLSQFitter()
    g_func = fit_g(g_init, x = wavlens, y = flux_profile)
    fwhm_fit_nm = 2 * np.sqrt(2*np.log(2)) * g_func.stddev.value
    mean_wavlen_fit_nm = g_func.mean.value
    peak_fit = g_func.amplitude

    return wavlens, flux_profile, fwhm_fit_nm, mean_wavlen_fit_nm, peak_fit

def rotate_points(points, angle_rad, pivot_point):
    """ 
    Rotate an array of (x,y) coordinates by an angle about a pivot point.

    Args:
        points (tuple): Two-element tuple of (x,y) coordinates. 
                The first element is an array of x values; 
                the second element is an array of y values. 
        angle_rad (float): Rotation angle in radians
        pivot_point (tuple): Tuple of (x,y) coordinates of the pivot point.

    Returns:
        Two-element tuple of rotated (x,y) coordinates.
    """
    rotated_points = (points[0] - pivot_point[0], points[1] - pivot_point[1]) 
    rotated_points = (rotated_points[0] * np.cos(angle_rad) - rotated_points[1] * np.sin(angle_rad),
                      rotated_points[0] * np.sin(angle_rad) + rotated_points[1] * np.cos(angle_rad))
    rotated_points = (rotated_points[0] + pivot_point[0], rotated_points[1] + pivot_point[1])
    return rotated_points

def shift_and_scale_2darray(array, xshift, yshift, amp):
    """
    Evaluate x,y shift and amplitude scale parameters 
    for a least-squares PSF fit.

    Args:
        array (numpy.ndarray): input data array, 2d
        xshift (float): x-axis shift in pixels
        yshift (float): y-axis shift in pixels
        amp (float): amplitude scale factor

    Returns:
        Flattened array of values after applying the shift and scale parameters
    """
    return np.ravel(amp * ndi.shift(array, (yshift, xshift), order=1, prefilter=False))
    
def psf_registration_costfunc(p, template, data):
    """
    Cost function for a least-squares fit to register a PSF with a fitting template.

    Args:
        p (tuple): shift and scale parameters: 
                    (x-axis shift in pixels, y-axis shift in pixels, 
                     amplitude scale factor)
        template (numpy.ndarray): PSF tempate array, 2d
        data (numpy.ndarray): PSF data array, 2d

    Returns:
        The sum of squares of differences between the data array and the shifted
        and scaled template.
    """
    xshift = p[0]
    yshift = p[1]
    amp = p[2]
    shifted_template = amp * ndi.shift(template, (yshift, xshift), order=1, prefilter=False)
    return np.sum((data - shifted_template)**2)

def fit_psf_centroid(psf_data, psf_template,
                     xcent_template = None, ycent_template = None,
                     xcent_guess = None, ycent_guess = None,
                     halfwidth = 10, halfheight = 10, 
                     fwhm_major_guess = 3, fwhm_minor_guess = 6,
                     gauss2d_oversample = 9):
    """
    Fit the centroid of a PSF image with a template.
    
    Args:
        psf_data (np.ndarray): PSF data, 2D array
        psf_template (np.ndarray): PSF template, 2D array
        xcent_template (float): true x centroid of the template PSF; for accurate 
                results this must be determined in advance.
        ycent_template (float): true y centroid of the template PSF; for accurate
                results this must be determined in advance.
        xcent_guess (int): Estimate of the x centroid of the data array, pixels
        ycent_guess (int): Estimate of the y centroid of the data array, pixels
        halfwidth (int): Half-width of the fitting region, pixels
        halfheight (int): Half-height of the fitting region, pixels
        fwhm_major_guess (float): guess for FWHM value along major axis of PSF, pixels
        fwhm_minor_guess (float): guess for FWHM value along minor axis of PSF, pixels
        gauss2d_oversample (int): upsample factor for 2-D Gaussian PSF fit;
                this must be an odd number.
    Returns:
        xfit (float): Data PSF x centroid obtained from the template fit, 
                array pixels
        yfit (float): Data PSF y centroid obtained from the template fit, 
                array pixels
        gauss2d_xfit (float): Data PSF x centroid estimated by a 2-D Gaussian fit to 
                the main lobe of the PSF
        gauss2d_yfit (float): Data PSF y centroid estimated by a 2-D Gaussian fit to 
                the main lobe of the PSF
        peakpix_snr (float): Peak-pixel signal-to-noise ratio
        x_precis (float): Statistical precision of the x centroid fit, estimated from
                peak-pixel S/N ratio
        y_precis (float): Statistical precision of the y centroid fit, estimated from
                peak-pixel S/N ratio
    """

    # Use the center of mass as a starting point if positions were not provided.
    if xcent_template == None or ycent_template == None: 
        xcom_template, ycom_template = np.rint(get_center_of_mass(psf_template))
    else:
        xcom_template, ycom_template = (np.rint(xcent_template), np.rint(ycent_template))

    if xcent_guess == None or ycent_guess == None:
        median_filt_psf = ndi.median_filter(psf_data, size=2)
        xcom_data, ycom_data = np.rint(get_center_of_mass(median_filt_psf))
    else:
        xcom_data, ycom_data = (np.rint(xcent_guess), np.rint(ycent_guess))
    
    xmin_template_cut, xmax_template_cut = (int(xcom_template) - halfwidth, int(xcom_template) + halfwidth)
    ymin_template_cut, ymax_template_cut = (int(ycom_template) - halfheight, int(ycom_template) + halfheight) 
    
    xmin_data_cut, xmax_data_cut = (int(xcom_data) - halfwidth, int(xcom_data) + halfwidth)
    ymin_data_cut, ymax_data_cut = (int(ycom_data) - halfheight, int(ycom_data) + halfheight) 
    
    template_stamp = psf_template[ymin_template_cut:ymax_template_cut+1, xmin_template_cut:xmax_template_cut+1]
    data_stamp = psf_data[ymin_data_cut:ymax_data_cut+1, xmin_data_cut:xmax_data_cut+1]
    
    xoffset_guess, yoffset_guess = (0.0, 0.0)
    amp_guess = np.sum(psf_data) / np.sum(psf_template)
    guess_params = (xoffset_guess, yoffset_guess, amp_guess)

    registration_result = optimize.minimize(psf_registration_costfunc, guess_params, 
                                            args = (template_stamp, data_stamp), method='Powell')
    xfit = xcent_template + (xcom_data - xcom_template) + registration_result.x[0]
    yfit = ycent_template + (ycom_data - ycom_template) + registration_result.x[1]

    #(fit_popt, pcov, 
    # infodict, mesg, ier) = optimize.curve_fit(shift_and_scale_2darray, template_stamp, 
    #                                           np.ravel(data_stamp), p0=guess_params, full_output=True)
    
    psf_data_bkg = psf_data.copy()
    psf_data_bkg[ymin_data_cut:ymax_data_cut+1, xmin_data_cut:xmax_data_cut+1] = np.nan
    psf_peakpix_snr = np.max(psf_data) / np.nanstd(psf_data_bkg)
    
    (gauss2d_xfit, gauss2d_yfit, xfwhm, yfwhm, gauss2d_peakfit,
     fitted_data_stamp, model, residual) = gaussfit2d_pix(psf_data,
                                                xguess = xfit,
                                                yguess = yfit,
                                                xfwhm_guess = fwhm_minor_guess, 
                                                yfwhm_guess = fwhm_major_guess,
                                                halfwidth = 1, halfheight = halfheight, 
                                                guesspeak = np.max(psf_data), oversample = gauss2d_oversample, 
                                                refinefit = True)
    
    (x_precis, y_precis) = (np.abs(xfwhm) / (2 * np.sqrt(2 * np.log(2))) / psf_peakpix_snr,
                            np.abs(yfwhm) / (2 * np.sqrt(2 * np.log(2))) / psf_peakpix_snr)

    return xfit, yfit, gauss2d_xfit, gauss2d_yfit, psf_peakpix_snr, x_precis, y_precis

def estimate_dispersion_clocking_angle(xpts, ypts, weights):
    """ 
    Estimate the clocking angle of the dispersion axis based on the centroids of
    the sub-band filter PSFs.

    Args:
        xpts (numpy.ndarray): Array of x coordinates in EXCAM pixels
        ypts (numpy.ndarray): Array of y coordinates in EXCAM pixels
        weights (numpy.ndarray): Array of weights for line fit

    Returns:
        clocking_angle, clocking_angle_uncertainty
    """
    linear_fit, V = np.polyfit(ypts, xpts, deg=1, w=weights, cov=True)
    
    theta = np.arctan(1/linear_fit[0])
    if theta > 0:
        clocking_angle = np.rad2deg(theta - np.pi)
    else:
        clocking_angle = np.rad2deg(theta)
    clocking_angle_uncertainty = np.abs(np.rad2deg(np.arctan(linear_fit[0] + np.sqrt(V[0,0]))) - 
                                        np.rad2deg(np.arctan(linear_fit[0] - np.sqrt(V[0,0])))) / 2

    return clocking_angle, clocking_angle_uncertainty 

def fit_dispersion_polynomials(wavlens, xpts, ypts, cent_errs, clock_ang, ref_wavlen, pixel_pitch_um=13.0):
    """ 
    Given arrays of wavlengths and positions, fit two polynomials:  
    1. Displacement from a reference wavelength along the dispersion axis, 
       in millimeters as a function of wavelength  
    2. Wavelength as a function of displacement along the dispersion axis

    Args:
        wavlens (numpy.ndarray): Array of wavelengths corresponding to the
        centroid data points

        xpts (numpy.ndarray): Array of x coordinates in EXCAM pixels
        
        ypts (numpy.ndarray): Array of y coordinates in EXCAM pixels

        cent_errs (numpy.ndarray): Array of centroid uncertainties in EXCAM pixels

        clock_ang (float): Clocking angle of the dispersion axis in degrees

        ref_wavlen (float): Reference wavelength of the bandpass, in nanometers

        pixel_pitch_um (float): EXCAM pixel pitch in microns

    Returns:
        pfit_pos_vs_wavlen (numpy.ndarray): polynomial coefficients for the
        position vs wavelength fit

        cov_pos_vs_wavlen (numpy.ndarray): covariance matrix of the polynomial
        coefficients for the position vs wavelength fit

        pfit_wavlen_vs_pos (numpy.ndarray): polynomial coefficients for the
        wavelength vs position fit

        cov_wavlen_vs_pos (numpy.ndarray): covariance matrix of the polynomial
        coefficients for the wavelength vs position fit
    """
    pixel_pitch_mm = pixel_pitch_um * 1E-3

    # Rotate the centroid coordinates so the dispersion axis is horizontal
    # to define a rotation pivot point, select the filter closest to the nominal 
    # zero deviation wavelength
    refidx = np.argmin(np.abs(wavlens - ref_wavlen))
    (x_rot, y_rot) = rotate_points((xpts, ypts), -np.deg2rad(clock_ang), 
                                    pivot_point = (xpts[refidx], ypts[refidx]))
    
    # Fit an intermediate polynomial to wavelength versus position
    delta_x = (x_rot - x_rot[refidx]) * pixel_pitch_mm
    pos_err = cent_errs * pixel_pitch_mm
    weights = 1 / pos_err
    lambda_func_x = np.poly1d(np.polyfit(x = delta_x, y = wavlens, deg = 2, w = weights))
    # Determine the position at the reference wavelength
    poly_roots = (np.poly1d(lambda_func_x) - ref_wavlen).roots
    real_roots = poly_roots[np.isreal(poly_roots)]
    root_select_ind = np.argmin(np.abs(poly_roots[np.isreal(poly_roots)]))
    pos_ref = np.real(real_roots[root_select_ind])
    np.testing.assert_almost_equal(lambda_func_x(pos_ref), ref_wavlen)
    displacements_mm = delta_x - pos_ref

    # Fit two polynomials:  
    # 1. Displacement from the band center along the dispersion axis as a 
    #    function of wavelength  
    # 2. Wavelength as a function of displacement along the dispersion axis
    (pfit_pos_vs_wavlen,
     cov_pos_vs_wavlen) = np.polyfit(x = (wavlens - ref_wavlen) / ref_wavlen,
                                      y = displacements_mm, deg = 3, w = weights, cov=True)
    
    (pfit_wavlen_vs_pos,
     cov_wavlen_vs_pos) = np.polyfit(x = displacements_mm, y = wavlens, deg = 3, 
                                      w = weights, cov=True)

    return pfit_pos_vs_wavlen, cov_pos_vs_wavlen, pfit_wavlen_vs_pos, cov_wavlen_vs_pos

def create_wave_cal_map(disp_params, zeropt, ref_wavlen, pixel_pitch_um=13.0):
    """
    Create a wavelength calibration map and a wavelength-position lookup table,
    given a dispersion model and a wavelength zero-point.

    Args:
        disp_params (spectroscopy.DispersionModel): Dispersion model object

        zeropt (spectroscopy.WavelengthZeropoint): Wavelength zero-point data object
        
        ref_wavlen (float): Reference wavelength of the bandpass, in nanometers

        pixel_pitch_um (float): EXCAM pixel pitch in microns
    
    Returns:

        wavlen_map (numpy.ndarray): 2-D wavelength calibration map. Each image
        pixel value is a wavelength in units of nanometers, computed for the
        dispersion profile, zero-point position, coordinates, and image shape
        specified in the input wavelength zero-point object.
        
        wavlen_uncertainty (numpy.ndarray): 2-D array of wavelength calibration map
        uncertainty values in units of nanometers.

        pos_lookup_table (astropy.table.table.Table): Wavelength-to-position
        lookup table, computed for the dispersion profile, zero-point position,
        coordinates, and image shape specified in the input wavelength
        zero-point object. The table contains 5 columns: wavelength, x, x
        uncertainty, y, y uncertainty.

    """

    pos_vs_wavlen_poly = np.poly1d(disp_params.pos_vs_wavlen_polycoeff)
    wavlen_vs_pos_poly = np.poly1d(disp_params.wavlen_vs_pos_polycoeff)
    wavlen_c = ref_wavlen
    d_zp_mm = pos_vs_wavlen_poly((zeropt.wavlen - wavlen_c) / wavlen_c)

    pixel_pitch_mm = pixel_pitch_um * 1E-3
    theta = np.deg2rad(disp_params.clocking_angle)
    x_c, y_c = (zeropt.x - d_zp_mm * np.cos(theta) / pixel_pitch_mm,
                zeropt.y - d_zp_mm * np.sin(theta) / pixel_pitch_mm)

    yy, xx = np.indices(zeropt.image_shape)
    dd_mm = (xx - x_c) * np.cos(theta) + (yy - y_c) * np.sin(theta) * pixel_pitch_mm
    wavlen_map = wavlen_vs_pos_poly(dd_mm)

    bandpass_frac = 0.17
    delta_wav = 0.5
    n_wav = int(ref_wavlen * bandpass_frac / delta_wav)
    n_wav_odd = n_wav + (n_wav % 2 == 0) # force odd array length
    wavlen_beg = ref_wavlen - n_wav_odd // 2 * delta_wav
    wavlen_end = ref_wavlen + n_wav_odd // 2 * delta_wav
    
    wavlens = np.linspace(wavlen_beg, wavlen_end, n_wav_odd)
    np.testing.assert_almost_equal(wavlens[n_wav_odd // 2], ref_wavlen)
        
    # Use a Monte Carlo error propagation to estimate the uncertainties of the
    # values in the wavelength calibration map and the position lookup table. 
    ntrials = 1000
    polyfit_order = len(disp_params.pos_vs_wavlen_polycoeff) - 1
    prand_wavlen_pos = np.zeros((ntrials, polyfit_order + 1))
    prand_pos_wavlen = np.zeros((ntrials, polyfit_order + 1))
    
    # Add the wavelength zero-point position error to the dispersion profile uncertainty 
    d_zp_err_mm = np.sqrt((zeropt.x_err * np.cos(theta))**2 + (zeropt.y_err * np.sin(theta))**2) * pixel_pitch_mm
    # To translate the position uncertainty to wavelength uncertainty, use the second coefficient of the
    # the wavelength(x) polynomial, which is the linear dispersion coefficient (units nm/mm).
    d_zp_err_nm = disp_params.wavlen_vs_pos_polycoeff[2] * d_zp_err_mm 
    disp_params.pos_vs_wavlen_cov[polyfit_order, polyfit_order] += d_zp_err_mm**2
    disp_params.wavlen_vs_pos_cov[polyfit_order, polyfit_order] += d_zp_err_nm**2 

    # Generate random polynomial coefficients consistent with the covariance
    # matrix in the dispersion profile.
    for ii in range(ntrials):
        prand_wavlen_pos[ii] = np.random.multivariate_normal(disp_params.wavlen_vs_pos_polycoeff, cov=disp_params.wavlen_vs_pos_cov)
        prand_pos_wavlen[ii] = np.random.multivariate_normal(disp_params.pos_vs_wavlen_polycoeff, cov=disp_params.pos_vs_wavlen_cov)

    ws = (wavlens - wavlen_c) / wavlen_c
    ds_mm = pos_vs_wavlen_poly(ws)
    
    ds_vander = np.vander(ds_mm, N=polyfit_order+1, increasing=False)
    ws_vander = np.vander(ws, N=polyfit_order+1, increasing=False)
    
    wavlen_rand_eval = prand_wavlen_pos.dot(ds_vander.T)
    pos_rand_eval = prand_pos_wavlen.dot(ws_vander.T)
    pos_eval_std = np.std(pos_rand_eval, axis=0)
    wavlen_eval_std = np.std(wavlen_rand_eval, axis=0)
    
    pos_vs_wavlen_err_func = interp1d(wavlens, pos_eval_std, fill_value="extrapolate")
    wavlen_vs_pos_err_func = interp1d(ds_mm, wavlen_eval_std, fill_value="extrapolate")

    # Wavelength uncertainty map
    wavlen_uncertainty_map = wavlen_vs_pos_err_func(dd_mm)

    # Build the position lookup table 
    ds_eval = pos_vs_wavlen_poly((wavlens - wavlen_c) / wavlen_c) / pixel_pitch_mm
    xs_eval, ys_eval = (x_c + ds_eval * np.cos(theta),
                        y_c + ds_eval * np.sin(theta))
    pos_lookup_1d = (wavlens, xs_eval, ys_eval)
    
    xs_uncertainty, ys_uncertainty = (np.abs(pos_vs_wavlen_err_func(wavlens) / pixel_pitch_mm * np.cos(theta)),
                                      np.abs(pos_vs_wavlen_err_func(wavlens) / pixel_pitch_mm * np.sin(theta)))

    pos_lookup_table = Table((wavlens, xs_eval, xs_uncertainty, ys_eval, ys_uncertainty),
                             names=('Wavelength (nm)', 'x (column)', 'x uncertainty', 'y (row)', 'y uncertainty'))

    return wavlen_map, wavlen_uncertainty_map, pos_lookup_table
