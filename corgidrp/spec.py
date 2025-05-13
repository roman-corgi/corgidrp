import numpy as np
import scipy.ndimage as ndi
import scipy.optimize as optimize
from corgidrp.data import Dataset, SpectroscopyCentroidPSF, Image
from corgidrp import mocks
import os
from astropy.io import fits
from corgidrp.data import Dataset



def gauss2d(x0, y0, sigma_x, sigma_y, peak):
    """
    2d gaussian function for gaussfit2d

    Args:
        x0: center of gaussian
        y0: center of gaussian
        peak: peak amplitude of gaussian
        sigma_x: stddev in x direction
        sigma_y: stddev in y direction
    
    Returns:
        function evaluated at coordinate tuple y,x
    """
    return lambda y,x: peak * np.exp(-( ((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) **2 ) / 2)

def gaussfit2d_pix(frame, xguess, yguess, xfwhm_guess=3, yfwhm_guess=6, 
                   halfwidth=3, halfheight=5, guesspeak=1, oversample=5, refinefit=True):
    """
    Fits a 2-d Gaussian to the data at point (xguess, yguess), with pixel integration.

    Args:
        frame: the data - Array of size (y,x)
        xguess: location to fit the 2d guassian to (should be within +/-1 pixel of true peak)
        yguess: location to fit the 2d guassian to (should be within +/-1 pixel of true peak)
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
        xguess: location to fit the 2d guassian to (should be within +/-1 pixel of true peak)
        yguess: location to fit the 2d guassian to (should be within +/-1 pixel of true peak)
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
                                         args=(template_stamp, data_stamp), 
                                         method='Nelder-Mead',  # Change to Nelder-Mead for better convergence
                                         options={'maxiter': 5000})
    
    if not registration_result.success:
        print(f"Warning: Registration optimization did not converge: {registration_result.message}")
    
    # Don't round the template positions
    xcom_template, ycom_template = xcent_template, ycent_template
    xcom_data, ycom_data = xcent_guess, ycent_guess
    
    xfit = xcom_template + registration_result.x[0]
    yfit = ycom_template + registration_result.x[1]

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

def compute_psf_centroid(dataset, initial_cent, verbose=False, halfwidth=10, halfheight=10):
    """
    Compute PSF centroids for a grid of PSFs and return them as a calibration object.

    Args:
        dataset (Dataset): Dataset containing 2D PSF images. Each image must include pri_hdr and ext_hdr.
        initial_cent (dict): Dictionary with initial guesses for PSF centroids.
                             Must include keys 'xcent' and 'ycent', each mapping to an array of shape (N,).
        verbose (bool): If True, prints fitted centroid values for each frame.
        halfwidth (int): Half-width of the PSF fitting box.
        halfheight (int): Half-height of the PSF fitting box.

    Returns:
        SpectroscopyCentroidPSF: Calibration object with fitted (x, y) centroids.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError("Input must be a corgidrp.data.Dataset object.")

    xcent = np.asarray(initial_cent.get("xcent"))
    ycent = np.asarray(initial_cent.get("ycent"))

    if xcent is None or ycent is None:
        raise ValueError("initial_cent dictionary must contain 'xcent' and 'ycent' arrays.")
    if len(dataset) != len(xcent) or len(dataset) != len(ycent):
        raise ValueError("Mismatch between dataset length and centroid guess arrays.")

    centroids = np.zeros((len(dataset), 2))

    pri_hdr = dataset[0].pri_hdr.copy()
    ext_hdr = dataset[0].ext_hdr.copy()

    for idx, frame in enumerate(dataset):
        psf_data = frame.data
        xguess = xcent[idx]
        yguess = ycent[idx]

        xfit, yfit, *_ = fit_psf_centroid(
            psf_data, psf_data,
            xcent_template=xguess,
            ycent_template=yguess,
            xcent_guess=xguess,
            ycent_guess=yguess,
            halfwidth=halfwidth,
            halfheight=halfheight
        )

        centroids[idx] = [xfit, yfit]

        if verbose:
            print(f"Slice {idx}: x = {xfit:.3f}, y = {yfit:.3f}")

    calibration = SpectroscopyCentroidPSF(
        centroids,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        input_dataset=dataset
    )
    
    calibration.ext_hdr["EXTNAME"] = "CENTROIDS"
    
    return calibration
