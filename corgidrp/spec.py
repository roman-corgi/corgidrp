import warnings
import glob
import numpy as np
import scipy.ndimage as ndi
import scipy.optimize as optimize
from scipy.interpolate import interp1d, LinearNDInterpolator
from corgidrp.data import Dataset, SpectroscopyCentroidPSF, DispersionModel, LineSpread, SpecFluxCal, SpecFilterOffset, SlitTransmission
import os
from astropy.io import ascii, fits
from astropy.table import Table
import astropy.modeling.models as models
import astropy.modeling.fitting as fitting
import corgidrp
from corgidrp.fluxcal import get_filter_name, read_cal_spec, read_filter_curve, get_calspec_file


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
        xguess: location to fit the 2d gaussian to (should be within +/-1 pixel of true peak)
        yguess: location to fit the 2d gaussian to (should be within +/-1 pixel of true peak)
        xfwhm_guess: approximate x-axis fwhm to fit to
        yfwhm_guess: approximate y-axis fwhm to fit to    
        halfwidth: 1/2 the width of the box used for the fit
        halfheight: 1/2 the height of the box used for the fit
        guesspeak: approximate flux in peak pixel
        oversample: odd integer >= 3; to represent detector pixels, oversample and then bin the model by this factor 
        refinefit: whether to refine the fit of the position of the guess

    Returns:
        xfit: x position (only changed if refinefit is True)
        yfit: y position (only changed if refinefit is True)
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
    if not isinstance(halfheight, int):
        raise ValueError("halfheight must be an integer")
    # Use the center of mass as a starting point if positions were not provided.
    if xcent_template is None or ycent_template is None:
        xcom_template, ycom_template = np.rint(get_center_of_mass(psf_template))
        xcent_template, ycent_template = xcom_template, ycom_template
    else:
        xcom_template, ycom_template = (np.rint(xcent_template), np.rint(ycent_template))

    if xcent_guess is None or ycent_guess is None:
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
                                         method='Powell')

    if not registration_result.success:
        print(f"Warning: Registration optimization did not converge: {registration_result.message}")

    xfit = xcent_template + (xcom_data - xcom_template) + registration_result.x[0]
    yfit = ycent_template + (ycom_data - ycom_template) + registration_result.x[1]

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

def get_template_dataset(dataset):
    """
    return the default template dataset from the data/spectroscopy/templates files

    Args:
        dataset (Dataset): Dataset containing 2D PSF images. Each image must include pri_hdr and ext_hdr.

    Returns:
        Dataset: template dataset
        boolean: filtersweep true or false
    """
    template_dir = os.path.join(os.path.dirname(__file__), "data", "spectroscopy", "templates")
    filtersweep = False
    cfamname = []
    slits = []
    for frames in dataset.frames:
        dpamname = frames.ext_hdr['DPAMNAME']
        fsamname = frames.ext_hdr['FSAMNAME']
        if dpamname != "PRISM3":
            raise AttributeError("currently we only have template files for PRISM3, not for "+ dpamname)

        cfamname.append (frames.ext_hdr['CFAMNAME'])
        slits.append (fsamname)
    if len(np.unique(slits)) != 1:
        raise AttributeError("currently we only have template files for no slit or R1C2, not for "+ slits)
    if len(np.unique(cfamname)) == 1:
        band = cfamname[0]
        if not band.startswith ("3"):
            raise AttributeError("currently we only have template files for the filter band 3, not for "+ band)
        slit = slits[0]
        if slit == "R1C2":
            filenames = sorted(glob.glob(os.path.join(template_dir,"spec_unocc_r1c2slit_offset_prism3_3d_*.fits")))
        elif slit == "OPEN":
            filenames = sorted(glob.glob(os.path.join(template_dir,"spec_unocc_noslit_offset_prism3_3d_*.fits")))
        else:
            raise AttributeError("we do not (yet) have template files for slit " + slit)
    else:
        #filtersweep
        filenames = sorted(glob.glob(os.path.join(template_dir, "spec_unocc_noslit_prism3_filtersweep_*.fits")))
        filtersweep = True
    return Dataset(filenames), filtersweep

def compute_psf_centroid(dataset, template_dataset = None, initial_cent = None, filtersweep = False, halfwidth=10, halfheight=10, verbose = False):
    """
    Compute PSF centroids for a grid of PSFs and return them as a calibration object.

    Args:
        dataset (Dataset): Dataset containing 2D PSF images. Each image must include pri_hdr and ext_hdr.
        template_dataset (Dataset): dataset of the template PSF, if None, a simulated PSF from the data/spectroscopy/template path is taken
        initial_cent (dict): Dictionary with initial guesses for PSF centroids.
                             Must include keys 'xcent' and 'ycent', each mapping to an array of shape (N,).
        filtersweep (bool): If True, it uses a filter sweep/scan dataset, this parameter is only relevant if template_dataset is not None.
        halfwidth (int): Half-width of the PSF fitting box.
        halfheight (int): Half-height of the PSF fitting box.
        verbose (bool): If True, prints fitted centroid values for each frame.
    
    Returns:
        SpectroscopyCentroidPSF: Calibration object with fitted (x, y) centroids.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError("Input must be a corgidrp.data.Dataset object.")

    if initial_cent is None:
        xcent, ycent = None, None
    else:
        xcent = np.asarray(initial_cent.get("xcent"))
        ycent = np.asarray(initial_cent.get("ycent"))
        if xcent is None or ycent is None:
            raise ValueError("initial_cent dictionary must contain 'xcent' and 'ycent' arrays.")
        if len(dataset) != len(xcent) or len(dataset) != len(ycent):
            raise ValueError("Mismatch between dataset length and centroid guess arrays.")

    if template_dataset is None:
        template_dataset, filtersweep = get_template_dataset(dataset)

    xcent_temp = []
    ycent_temp = []
    for frame in template_dataset:
        if "XCENT" in frame.ext_hdr:
            xcent_temp.append(frame.ext_hdr["XCENT"])
        else:
            xcent_temp.append(None)
        if "YCENT" in frame.ext_hdr:
            ycent_temp.append(frame.ext_hdr["YCENT"])
        else:
            ycent_temp.append(None)
    xcent_temp = np.asarray(xcent_temp)
    ycent_temp = np.asarray(ycent_temp)
    centroids = np.zeros((len(dataset), 2))
    centroids_err = np.zeros((len(dataset), 2))

    pri_hdr_centroid = dataset[0].pri_hdr.copy()
    ext_hdr_centroid = dataset[0].ext_hdr.copy()

    filters = []
    for idx, frame in enumerate(dataset):
        cfam = frame.ext_hdr['CFAMNAME']
        filters.append(cfam)
        psf_data = frame.data
        if xcent is None:
            xguess, yguess = None, None
        else:
            xguess = xcent[idx]
            yguess = ycent[idx]

        if filtersweep:
            found_cfam = False
            for k, temp_frame in enumerate(template_dataset):
                cfam_temp = temp_frame.ext_hdr['CFAMNAME']
                if cfam == cfam_temp:
                    temp_psf_data = temp_frame.data
                    temp_x = xcent_temp[k]
                    temp_y = ycent_temp[k]
                    found_cfam = True
                    break
            if found_cfam == False:
                raise AttributeError("no template image found for filter: "+cfam)
        else:
            if idx < len(template_dataset):
                temp_psf_data = template_dataset[idx].data
                temp_x = xcent_temp[idx]
                temp_y = ycent_temp[idx]
            else:
                temp_psf_data = template_dataset[-1].data
                temp_x = xcent_temp[-1]
                temp_y = ycent_temp[-1]
        # larger fitting stamp needed for broadband filter
        if cfam == '2' or cfam == '3':
            halfheight = 30

        xfit, yfit, gauss2d_xfit, gauss2d_yfit, psf_peakpix_snr, x_precis, y_precis = fit_psf_centroid(
            psf_data, temp_psf_data,
            xcent_template=temp_x,
            ycent_template=temp_y,
            xcent_guess=xguess,
            ycent_guess=yguess,
            halfwidth=halfwidth,
            halfheight=halfheight
        )

        centroids[idx] = [xfit, yfit]
        centroids_err[idx] = [x_precis, y_precis]

        if verbose:
            print(f"Slice {idx}: x = {xfit:.3f}, y = {yfit:.3f}")

    if filtersweep:
        ext_hdr_centroid['FILTERS'] = ",".join(filters)
    else:
        ext_hdr_centroid['FILTERS'] = filters[0]
    calibration = SpectroscopyCentroidPSF(
        centroids,
        pri_hdr=pri_hdr_centroid,
        ext_hdr=ext_hdr_centroid,
        err_hdr = fits.Header(),
        err = centroids_err,
        input_dataset=dataset
    )

    return calibration

def read_cent_wave(band, filter_file = None):
    """
    read the csv filter file containing the band names and the central wavelength in nm.
    There are 6 columns: the CFAM filter name, the (Phase C) center wavelength, the TVAC TV-40b measured center wavelength 
    and the FWHM for the 4 broad bands, and xoffset, yoffset between the bands
    The TVAC wavelengths are not measured for all filters, but are the preferred value if available.
    
    Args:
       filter_file (str): file name of the filter file
       band (str): name of the filter band
       
    Returns:
       list: [central wavelength of the filter band, fwhm, xoffset, yoffset]
    """
    if filter_file is None:
        filter_file = os.path.join(os.path.dirname(__file__), "data", "spectroscopy", "CGI_bandpass_centers.csv")
    data = ascii.read(filter_file, format = 'csv', data_start = 1)
    filter_names = data.columns[0]
    band = band.upper()
    if band not in filter_names:
        raise ValueError("{0} is not in table band names {1}".format(band, filter_names))
    ret_list = []
    if data.columns[2][filter_names == band]:
        cen_wave = data.columns[2][filter_names == band][0]
    else:
        cen_wave = data.columns[1][filter_names == band][0]
    ret_list.append(cen_wave)
    for i in range(3, 6):
        ret_list.append(data.columns[i][filter_names == band][0])
    return ret_list

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
    Given arrays of wavelengths and positions, fit two polynomials:  
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


def calibrate_dispersion_model(centroid_psf, spec_filter_offset, band_center_file = None, pixel_pitch_um = 13.0):
    """ 
    Generate a DispersionModel of the spectral dispersion profile of the CGI ZOD prism.

    Args:
       centroid_psf (data.SpectroscopyCentroidPsf): instance of SpectroscopyCentroidPsf calibration class
       spec_filter_offset (data.SpecFilterOffset): instance of SpecFilterOffset calibration class
       band_center_file (str): file name of the band centers, optional, default is in data/spectroscopy
       pixel_pitch_um (float): EXCAM pixel pitch in micron, default 13 micron
    
    Returns:
       data.DispersionModel: DispersionModel calfile object with the fit results including errors of the spectral trace and the dispersion
    """
    prism = centroid_psf.ext_hdr['DPAMNAME']
    if prism not in ['PRISM2', 'PRISM3']:
        raise ValueError("prism must be PRISM2 or PRISM3")

    #PRISM2 not yet available
    if prism == 'PRISM2':
        subband_list = ['2A', '2B', '2C', '2F']
        ref_cfam = '2'
        ref_wavlen = 660.
    else:
        subband_list = ['3A', '3B', '3C', '3D', '3E', '3G']
        ref_cfam = '3'
        ref_wavlen = 730.

    ##bandpass_frac = fwhm/cen_wave, needed for the wavelength calibration
    band_center, fwhm, _, _ = read_cent_wave(ref_cfam, filter_file = band_center_file)
    #needed to consider the position offsets between different filters
    offset_band = spec_filter_offset.get_offsets(ref_cfam)
    xoff_band = offset_band[0]
    yoff_band = offset_band[1]
    bandpass_frac = fwhm/band_center
    if 'FILTERS' not in centroid_psf.ext_hdr:
        raise AttributeError("there should be a FILTERS header keyword in the filtersweep SpectroscopyCentroidPsf")
    filters = centroid_psf.ext_hdr['FILTERS'].upper().split(",")
    center_wavel = []
    xoff = []
    yoff = []
    for band in filters:
        band_str = band.strip()
        if band_str == ref_cfam:
            pass
        elif band_str not in subband_list:
            warnings.warn("measured band {0} is not in the sub band list {1} of the used prism".format(band_str, subband_list))
        else:
            cen_wave = read_cent_wave(band_str, filter_file = band_center_file)
            center_wavel.append(cen_wave[0])
            offset_sub = spec_filter_offset.get_offsets(band_str)
            xoff.append(offset_sub[0] - xoff_band)
            yoff.append(offset_sub[1] - yoff_band)
    if len(center_wavel) < 4:
        raise ValueError ("number of measured sub-bands {0} is too small to model the dispersion".format(len(center_wavel)))
    if len(center_wavel) != len(centroid_psf.xfit) -1:
        raise ValueError ("number of measured sub-bands {0} does not fit to the measured number of centroids {1}".format(len(center_wavel), len(centroid_psf.xfit)))
    center_wavel = np.array(center_wavel)
    xfit = centroid_psf.xfit[:-1] - np.array(xoff)
    yfit = centroid_psf.yfit[:-1] - np.array(yoff)
    xfit_err = centroid_psf.xfit_err[:-1]
    yfit_err = centroid_psf.yfit_err[:-1]
    (clocking_angle,
     clocking_angle_uncertainty) = estimate_dispersion_clocking_angle(xfit, yfit, weights = 1. / yfit_err)

    (pfit_pos_vs_wavlen, cov_pos_vs_wavlen,
     pfit_wavlen_vs_pos, cov_wavlen_vs_pos) = fit_dispersion_polynomials(
            center_wavel, xfit, yfit,
            yfit_err, clocking_angle, ref_wavlen, pixel_pitch_um = pixel_pitch_um
     )

    disp_dict = {"clocking_angle": clocking_angle,
        "clocking_angle_uncertainty": clocking_angle_uncertainty,
        "pos_vs_wavlen_polycoeff": pfit_pos_vs_wavlen,
        "pos_vs_wavlen_cov": cov_pos_vs_wavlen,
        "wavlen_vs_pos_polycoeff": pfit_wavlen_vs_pos,
        "wavlen_vs_pos_cov": cov_wavlen_vs_pos}
    pri_hdr = centroid_psf.pri_hdr.copy()
    ext_hdr = centroid_psf.ext_hdr.copy()
    ext_hdr["REFWAVE"] = ref_wavlen
    ext_hdr["BAND"] = ref_cfam
    ext_hdr["BANDFRAC"] = bandpass_frac
    corgi_dispersion_profile = DispersionModel(
        disp_dict, pri_hdr = pri_hdr, ext_hdr = ext_hdr
    )

    return corgi_dispersion_profile


def create_wave_cal(disp_model, wave_zeropoint, pixel_pitch_um=13.0, ntrials = 1000):
    """
    Create a wavelength calibration map and a wavelength-position lookup table,
    given a dispersion model and a wavelength zero-point

    Args:
        disp_model (data.DispersionModel): Dispersion model calibration object
        wave_zeropoint (dict): Wavelength zero-point dictionary
        pixel_pitch_um (float): EXCAM pixel pitch in microns
        ntrials (int): number of trials when applying a Monte Carlo error propagation to estimate the uncertainties of the
                       values in the wavelength calibration map
    
    Returns:
        wavlen_map (numpy.ndarray): 2-D wavelength calibration map. Each image
        pixel value is a wavelength in units of nanometers, computed for the
        dispersion profile, zero-point position, coordinates, and image shape
        specified in the input wavelength zero-point object.
        wavlen_uncertainty (numpy.ndarray): 2-D array of wavelength calibration map
        uncertainty values in units of nanometers.
        pos_lookup_table (astropy.table.Table): Wavelength-to-position
        lookup table, computed for the dispersion profile, zero-point position,
        coordinates, and image shape specified in the input wavelength
        zero-point object. The table contains 5 columns: wavelength, x, x
        uncertainty, y, y uncertainty.
        x_refwav, y_refwave (float): coordinates of the source at the reference wavelength

    """
    ref_wavlen = disp_model.ext_hdr["REFWAVE"]
    #bandpass_frac = fwhm/cen_wave
    bandpass_frac = disp_model.ext_hdr["BANDFRAC"]
    pos_vs_wavlen_poly = np.poly1d(disp_model.pos_vs_wavlen_polycoeff)
    wavlen_vs_pos_poly = np.poly1d(disp_model.wavlen_vs_pos_polycoeff)
    wavlen_c = ref_wavlen
    d_zp_mm = pos_vs_wavlen_poly((wave_zeropoint.get('wavlen') - wavlen_c) / wavlen_c)

    pixel_pitch_mm = pixel_pitch_um * 1E-3
    theta = np.deg2rad(disp_model.clocking_angle)
    x_refwav, y_refwav = (wave_zeropoint.get('x') - d_zp_mm * np.cos(theta) / pixel_pitch_mm,
                wave_zeropoint.get('y') - d_zp_mm * np.sin(theta) / pixel_pitch_mm)

    yy, xx = np.indices((wave_zeropoint.get('shapex'), wave_zeropoint.get('shapey')))
    dd_mm = (xx - x_refwav) * np.cos(theta) + (yy - y_refwav) * np.sin(theta) * pixel_pitch_mm
    wavlen_map = wavlen_vs_pos_poly(dd_mm)

    delta_wav = 0.5
    n_wav = int(ref_wavlen * bandpass_frac / delta_wav)
    n_wav_odd = n_wav + (n_wav % 2 == 0) # force odd array length
    wavlen_beg = ref_wavlen - n_wav_odd // 2 * delta_wav
    wavlen_end = ref_wavlen + n_wav_odd // 2 * delta_wav

    wavlens = np.linspace(wavlen_beg, wavlen_end, n_wav_odd)
    #np.testing.assert_almost_equal(wavlens[n_wav_odd // 2], ref_wavlen)

    # Use a Monte Carlo error propagation to estimate the uncertainties of the
    # values in the wavelength calibration map and the position lookup table.
    polyfit_order = len(disp_model.pos_vs_wavlen_polycoeff) - 1
    prand_wavlen_pos = np.zeros((ntrials, polyfit_order + 1))
    prand_pos_wavlen = np.zeros((ntrials, polyfit_order + 1))

    # Add the wavelength zero-point position error to the dispersion profile uncertainty
    d_zp_err_mm = np.sqrt((wave_zeropoint.get('xerr') * np.cos(theta))**2 + (wave_zeropoint.get('yerr') * np.sin(theta))**2) * pixel_pitch_mm
    # To translate the position uncertainty to wavelength uncertainty, use the second coefficient of the
    # the wavelength(x) polynomial, which is the linear dispersion coefficient (units nm/mm).
    d_zp_err_nm = disp_model.wavlen_vs_pos_polycoeff[2] * d_zp_err_mm
    disp_model.pos_vs_wavlen_cov[polyfit_order, polyfit_order] += d_zp_err_mm**2
    disp_model.wavlen_vs_pos_cov[polyfit_order, polyfit_order] += d_zp_err_nm**2

    # Generate random polynomial coefficients consistent with the covariance
    # matrix in the dispersion profile.
    for ii in range(ntrials):
        prand_wavlen_pos[ii] = np.random.multivariate_normal(disp_model.wavlen_vs_pos_polycoeff, cov=disp_model.wavlen_vs_pos_cov)
        prand_pos_wavlen[ii] = np.random.multivariate_normal(disp_model.pos_vs_wavlen_polycoeff, cov=disp_model.pos_vs_wavlen_cov)

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
    xs_eval, ys_eval = (x_refwav + ds_eval * np.cos(theta),
                        y_refwav + ds_eval * np.sin(theta))

    xs_uncertainty, ys_uncertainty = (np.abs(pos_vs_wavlen_err_func(wavlens) / pixel_pitch_mm * np.cos(theta)),
                                      np.abs(pos_vs_wavlen_err_func(wavlens) / pixel_pitch_mm * np.sin(theta)))

    pos_lookup_table = Table((wavlens, xs_eval, xs_uncertainty, ys_eval, ys_uncertainty),
                             names=('Wavelength (nm)', 'x (column)', 'x uncertainty', 'y (row)', 'y uncertainty'))

    return wavlen_map, wavlen_uncertainty_map, pos_lookup_table, x_refwav, y_refwav


def get_shift_correlation(
    img_data,
    img_template,
    ):
    """ Find the array shift that maximizes the phase correlation between two
      images.

    Args:
      img_data (array): first two dimensional array.
      img_template (array): second two dimensional array. Its size must be the same or
      less than img1, because img2 is the noiseless template used to find the
      spectrum on the L2b data and it is a cropped frame.

    Returns:
      Image shift in image pixels that maximizes the phase correlation of the
      first image with the second one.
    """
    if np.any(img_data.shape < img_template.shape):
        raise Exception('The template image cannot have a larger size then the data one')  

    # Pad img_template to be the same size as img_data
    img2 = np.zeros_like(img_data)
    img2[img_data.shape[0]//2-img_template.shape[0]//2:img_data.shape[0]//2+img_template.shape[0]//2 + 1,
        img_data.shape[1]//2-img_template.shape[1]//2:img_data.shape[1]//2+img_template.shape[1]//2 + 1] = img_template

    dft1 = np.fft.fftshift(np.fft.fft2(img_data))
    dft2 = np.fft.fftshift(np.fft.fft2(img2))

    # Cross-power spectrum (Add epsilon to avoid division by zero, from Google Labs)
    R = (dft1 * np.conj(dft2)) / (np.abs(dft1 * np.conj(dft2)) + 1e-10)

    # Inverse FFT: Imaginary part are numerical residuals. The original data are real.
    poc_real = np.real(np.fft.ifft2(R))

    # Find the peak location (shift)
    shift = np.unravel_index(np.argmax(np.abs(poc_real)), poc_real.shape)

    return shift

def star_spec_registration(
    dataset_fsm,
    pathfiles_template,
    slit_align_err=0,
    halfheight=40):
    """ This function addresses:

      CGI-REQT-5465 – Given (1) a series of cleaned images of a prism-dispersed
      unocculted star observed through the FSAM slit mask, observed with the
      same CFAM filter, and acquired over a grid of FSM offsets and (2) an
      estimate of the spectroscopic target source position on EXCAM and its
      alignment error from the FSAM slit, the CTC GSW should identify the
      dispersed star image whose PSF-to-FSAM slit alignment most closely matches
      that of the target source.

      NOTE: This calibration is repeated for each roll angle in the observation
      campaign
  
    Args:
      dataset_fsm (Dataset): Dataset containing a series of L2b cleaned images of a
        prism-dispersed unocculted star observed through the FSAM slit mask,
        observed with the same CFAM filter, and acquired over a grid of FSM
        offsets. By default, the grid of FSM offsets spans a 3×3 FSM offset grid. 
        Each of the L2b images must have the following header keywords:
          – FSMX, FSMY (float64)
          – CFAMNAME (same for all images)
          – FSAMNAME = OPEN, R1C2, R6C5, R3C1
      pathfiles_template (array): array of path and filenames containing the 
        simulated star spectrum that are used as a template to find the image
        in dataset_fsm that best matches it.
      slit_align_err (float64): Distance between the source and the center of
        the slit aperture, measured along the narrow axis of the slit aperture,
        in units of mas. It is determined after each observation by
        looking at the data.
      halfheight: 1/2 the height of the box used for the fit.

    Returns:
      Filenames with the star image whose PSF-to-FSAM slit alignment most
      closely matches that of the target source.
      
    """
    # Confirm spectroscopy configuration for different PAMs
    # CFAM
    cfam_name = dataset_fsm[0].ext_hdr['CFAMNAME'].upper()
    if cfam_name.find('3') != -1:
        dpam_name = 'PRISM3'
        # fsam_name = []
    elif cfam_name.find('2') != -1:
        dpam_name = 'PRISM2'
        # fsam_name = []
    else:
        raise ValueError(f'{cfam_name} is not a spectroscopy filter')
    # DPAM
    if dataset_fsm[0].ext_hdr['DPAMNAME'] != dpam_name:
        raise ValueError(f'DPAMNAME should be {dpam_name}')
    # FPAM
    fpam_name = dataset_fsm[0].ext_hdr['FPAMNAME'].upper()
    if (fpam_name != 'OPEN' and fpam_name != 'ND225' and fpam_name != 'ND475'):
        raise ValueError('FPAMNAME should be either OPEN, ND225 or ND475')
    # SPAM
    spam_name = dataset_fsm[0].ext_hdr['SPAMNAME'].upper()
    if spam_name[0:4] != 'SPEC':
        raise ValueError('SPAMNAME should be SPEC')
    # LSAM
    lsam_name = dataset_fsm[0].ext_hdr['LSAMNAME'].upper()
    if lsam_name[0:4] != 'SPEC':
        raise ValueError('LSAMNAME should be SPEC')
    # FSAM
    fsam_name = dataset_fsm[0].ext_hdr['FSAMNAME'].upper()
    if (fsam_name != 'OPEN' and fsam_name != 'R1C2' and fsam_name != 'R6C5' and
        fsam_name != 'R3C1'):
        raise ValueError('FSAMNAME should be either OPEN, R1C2, R6C5 or R3C1')

    # All images must have the same setup
    for img in dataset_fsm:
        exthdr = img.ext_hdr
        assert exthdr['CFAMNAME'].upper() == cfam_name, f"CFAMNAME={exthdr['CFAMNAME']} differs from expected value: {cfam_name}"
        assert exthdr['DPAMNAME'].upper() == dpam_name, f"DPAMNAME={exthdr['DPAMNAME']} differs from expected value: {dpam_name}"
        assert exthdr['FPAMNAME'].upper() == fpam_name, f"FPAMNAME={exthdr['FPAMNAME']} differs from expected value: {fpam_name}"
        assert exthdr['SPAMNAME'].upper() == spam_name, f"SPAMNAME={exthdr['SPAMNAME']} differs from expected value: {spam_name}"
        assert exthdr['LSAMNAME'].upper() == lsam_name, f"LSAMNAME={exthdr['LSAMNAME']} differs from expected value: {lsam_name}"
        assert exthdr['FSAMNAME'].upper() == fsam_name, f"FSAMNAME={exthdr['FSAMNAME']} differs from expected values: {fsam_name}"
        # Confirm presence of FSMX, FSMY
        assert 'FSMX' in exthdr.keys() and 'FSMY' in exthdr.keys(), 'Missing FSMX/Y'

    # Templates
    yoffset_arr = []
    for file in pathfiles_template:
        # Make sure that all template files exist
        if os.path.exists(file) == False:
            raise Exception(f'Template file {file} not found.')
        # Collect FSAM offsets
        yoffset_arr += [fits.open(file)[0].header['FSM_OFF']]

    # Find closest template offset to the one measured in the data
    slit_idx = int(np.abs(slit_align_err - yoffset_arr).argmin())
    
    # Template data
    temp = fits.open(pathfiles_template[slit_idx])[0]
    temp_data = temp.data
    # Associated zeropoint
    try:
        wv0_x = temp.header['WV0_X']
    except:
        raise ValueError(f'WV0_X missing from {pathfiles_template[slit_idx]:s}')
    try:
        wv0_y = temp.header['WV0_Y']
    except:
        raise ValueError(f'WV0_Y missing from {pathfiles_template[slit_idx]:s}')
    
    # Split FSM dataset according to their FSM values
    dataset_list, fsm_values = dataset_fsm.split_dataset(exthdr_keywords=['FSMY'])
    
    # Combine frames with the same FSMY to increase SNR before the analysis
    fsm_combined = []
    for dataset in dataset_list:
        fsm_tmp = []
        for img in dataset:
            fsm_tmp += [img.data]
        fsm_combined += [np.median(np.array(fsm_tmp), axis=0)]

    # Find best PSF centroid fit for each image compared to the template
    # Cost function: Start with any large value that cannot happen. P.S. Units
    # are EXCAM pixels
    zeropt_dist = 1e8
    idx_best = None
    # cross-correlate data with expected slit error with template
    shift = get_shift_correlation(fsm_combined[slit_idx], temp_data)
    for idx_img, img in enumerate(fsm_combined):
        # Bring img_data on top of img_template
        img = np.roll(img, (-shift[0], -shift[1]), axis=(0,1))
        # Crop it to match img2 size
        img_cropped = img[img.shape[0]//2-temp_data.shape[0]//2:img.shape[0]//2+temp_data.shape[0]//2+1,
        img.shape[1]//2-temp_data.shape[1]//2:img.shape[1]//2+temp_data.shape[1]//2+1]
        # Find best centroid
        x_fit, y_fit = fit_psf_centroid(img_cropped, temp_data,
            xcent_template = wv0_x,
            ycent_template = wv0_y,
            halfheight = halfheight)[0:2]
        # best-matching image is wrt zero-point
        zeropt_dist_img = np.sqrt((x_fit - wv0_x)**2 + (y_fit - wv0_y)**2)

        # Keep track of absolute minimum
        if zeropt_dist_img < zeropt_dist:
            zeropt_dist = zeropt_dist_img
            idx_best = idx_img

    # Check that there's at least one solution
    assert idx_best != None, 'No suitable best image found.'        
    
    # List of filenames of frames with the same FSM value that best matches the
    # stellar template
    list_of_best_fsm = []
    for img in dataset_list[idx_best]:
        list_of_best_fsm += [img.filename]

    return list_of_best_fsm

def fit_line_spread_function(dataset, halfwidth = 2, halfheight = 9, guess_fwhm = 15.):
    """
    Fit the line spread function to a wavelength calibrated (averaged) dataset, by reading 
    the wavelength map extension and wavelength zeropoint header

    Args:
        dataset (corgidrp.data.Dataset): dataset containg a narrowband filter + prism PSF
        halfwidth (int): The width of the fitting region is 2 * halfwidth + 1 pixels.
        halfheight (int): The height of the fitting region is 2 * halfheight + 1 pixels.
        guess_fwhm (float): guess value of the fwhm of the line

    Returns:
        corgidrp.data.LineSpread: LineSpread object containing
        wavlens (numpy.ndarray)
        flux_profile (numpy.ndarray) 
        fwhm_fit (float)
        mean_fit (float)
        peak_fit (float)

    """
    # Assumed that only narrowband filter (includes sat spots) frames are taken to fit the line spread function LSF
    narrow_dataset, band = dataset.split_dataset(exthdr_keywords=["CFAMNAME"])
    band = np.array([s.upper() for s in band])

    if "3D" in band:
        nar_dataset = narrow_dataset[int(np.nonzero(band == "3D")[0].item())]
    elif "2C" in band:
        nar_dataset = narrow_dataset[int(np.nonzero(band == "2C")[0].item())]
    else:
        raise AttributeError("No narrowband frames found in input dataset")

    wave = []
    wave_err = []
    fwhm = []
    fwhm_err = []
    peak = []
    peak_err = []
    wavlens = []
    flux_profile = []
    for image in nar_dataset:
        xcent_round, ycent_round = (int(np.rint(image.ext_hdr["WV0_X"])), int(np.rint(image.ext_hdr["WV0_Y"])))
        image_cutout = image.data[ycent_round - halfheight:ycent_round + halfheight + 1,
                                  xcent_round - halfwidth:xcent_round + halfwidth + 1]
        dq_cutout = image.dq[ycent_round - halfheight:ycent_round + halfheight + 1,
                                  xcent_round - halfwidth:xcent_round + halfwidth + 1]
        wave_cal_map_cutout = image.hdu_list["WAVE"].data[ycent_round - halfheight:ycent_round + halfheight + 1,
                                                          xcent_round - halfwidth:xcent_round + halfwidth + 1]
        bad_ind = np.where(dq_cutout > 0)
        image_cutout[bad_ind] = np.nan
        flux_p = np.nansum(image_cutout, axis=1) / np.nansum(image_cutout)
        wav = np.mean(wave_cal_map_cutout, axis=1)
        flux_profile.append(flux_p)
        wavlens.append(wav)
        g_init = models.Gaussian1D(amplitude = np.max(flux_p),
                                   mean = wav[halfheight],
                                   stddev = guess_fwhm/(2 * np.sqrt(2*np.log(2))))
        fit_g = fitting.LevMarLSQFitter(calc_uncertainties=True)
        g_func = fit_g(g_init, x = wav, y = flux_p)
        fwhm.append(2 * np.sqrt(2*np.log(2)) * g_func.stddev.value)
        wave.append(g_func.mean.value)
        peak.append(g_func.amplitude.value)
        errors = np.diagonal(fit_g.fit_info.get("param_cov"))
        peak_err.append(errors[0])
        wave_err.append(errors[1])
        fwhm_err.append(errors[2] * 8 * np.log(2))

    mean_peak = np.mean(np.array(peak))
    mean_fwhm = np.mean(np.array(fwhm))
    mean_wave = np.mean(np.array(wave))
    mean_peak_err = np.sqrt(np.sum(np.array(peak_err)))
    mean_wave_err = np.sqrt(np.sum(np.array(wave_err)))
    mean_fwhm_err = np.sqrt(np.sum(np.array(fwhm_err)))
    mean_flux_profile = np.mean(np.array(flux_profile), axis = 0)
    mean_wavlens = np.mean(np.array(wavlens), axis = 0)
    prihdr = nar_dataset[0].pri_hdr.copy()
    exthdr = nar_dataset[0].ext_hdr.copy()

    ls_data = np.array([mean_wavlens, mean_flux_profile])
    gauss_profile = np.array([mean_peak, mean_wave, mean_fwhm, mean_peak_err, mean_wave_err, mean_fwhm_err])

    line_spread = LineSpread(ls_data, pri_hdr = prihdr, ext_hdr = exthdr, gauss_par = gauss_profile, input_dataset = nar_dataset)
    return line_spread

def slit_transmission(
    dataset_slit,
    dataset_open,
    target_pix=None,
    x_range=[40.,42],
    y_range=[32.,34],
    n_gridx=10,
    n_gridy=10,
    kind='linear',
    average='mean',
    ):
    """ This step function addresses:
  
      CGI-REQT-5475 – Given (1) a series of cleaned images of a prism-dispersed
      unocculted star observed through the FSAM slit mask, observed with a CFAM
      filter, and acquired over one or more FSM offsets, (2) a series of cleaned
      images of the same prism-dispersed unocculted star observed with the FSAM
      slit mask removed (FSAM in OPEN position), the same CFAM filter, and
      acquired over one or more FSM offsets, the CTC GSW should compute the slit
      transmission map.

    Args:
      dataset_slit (Dataset): Dataset containing a set of extracted spectra for
        some set of FSM positions with the FSAM slit in its position. There can
        be a different number of frames for each FSM position.
      dataset_open (Dataset): Dataset containing a set of extracted spectra for
        some set of FSM positions with the FSAM slit in OPEN position. There can
        be a different number of frames for each FSM position.
      target_pix (array) (optional): a user-defined Mx2 array containing the
        pixel positions for M target pixels where the slit transmission will be
        derived by interpolation. The target pixels are measured with respect
        the zero-point in (fractional) EXCAM pixels. Default is None. In this
        case, a rectangular grid of pixel positions is used. 
      x_range (array): Two values [xmin, xmax] specifying the range of pixels to
        be considered. Units are EXCAM pixels measured with respect the zero-point
        solution along EXCAM +X direction.
      y_range (array): Two values [ymin, ymax] specifying the range of pixels to
        be considered. Units are EXCAM pixels measured with respect the zero-point
        solution along EXCAM +Y direction.
      n_gridx (int) (optional): Number of positions when pos_range is set.
      n_gridy (int) (optional): Number of positions when pos_range is set.
      kind (string): Specifies the kind of interpolation. See scipy documentation.
        Default is piecewise linear.
      average (str): The type of average (first momentum) applied to each subset
        of spectra. The slitless spectra are all averaged at once regardless of
        their FSMX, FSMY values. The spectra with the slit in are averaged over
        subsets with the same FSMX, FSMY values. Options are 'mean' and 'median'.

    Returns:
      SlitTransmission calibration product containing:
        1/ Slit transmission map derived at different locations by interpolation.
        2/ Corresponding locations along EXCAM +X direction with respect to the
          zero-point in (fractional) EXCAM pixels where the slit transmission has
          been derived.
        3/ Corresponding locations along EXCAM +Y direction with respect to the
          zero-point in (fractional) EXCAM pixels where the slit transmission has
          been derived.
    """
    # Confirm spectroscopy configuration for different PAMs
    # CFAM
    cfam_name = dataset_slit[0].ext_hdr['CFAMNAME'].upper()
    if cfam_name.find('3') != -1:
        dpam_name = 'PRISM3'
        # fsam_name = []
    elif cfam_name.find('2') != -1:
        dpam_name = 'PRISM2'
        # fsam_name = []
    else:
        raise ValueError(f'{cfam_name} is not a spectroscopy filter')
    # DPAM
    if dataset_slit[0].ext_hdr['DPAMNAME'] != dpam_name:
        raise ValueError(f'DPAMNAME should be {dpam_name}')
    # FPAM
    fpam_name = dataset_slit[0].ext_hdr['FPAMNAME'].upper()
    if (fpam_name != 'OPEN' and fpam_name != 'ND225' and fpam_name != 'ND475'):
        raise ValueError('FPAMNAME should be either OPEN, ND225 or ND475')
    # SPAM
    spam_name = dataset_slit[0].ext_hdr['SPAMNAME'].upper()
    if spam_name[0:4] != 'SPEC':
        raise ValueError('SPAMNAME should be SPEC')
    # LSAM
    lsam_name = dataset_slit[0].ext_hdr['LSAMNAME'].upper()
    if lsam_name[0:4] != 'SPEC':
        raise ValueError('LSAMNAME should be SPEC')
    # FSAM: slit in
    fsam_name = dataset_slit[0].ext_hdr['FSAMNAME'].upper()
    if (fsam_name != 'R1C2' and fsam_name != 'R6C5' and fsam_name != 'R3C1'):
        raise ValueError('FSAMNAME with the slit in must be either R1C2, R6C5 or R3C1')
    # FSAM: slitless
    if dataset_open[0].ext_hdr['FSAMNAME'] != 'OPEN':
        raise ValueError('FSAMNAME must be OPEN for slitless observations.')

    # All images with the slit in must have the same setup
    for image in dataset_slit:
        exthdr = image.ext_hdr
        assert exthdr['CFAMNAME'].upper() == cfam_name, f"CFAMNAME={exthdr['CFAMNAME']} differs from expected value: {cfam_name}"
        assert exthdr['DPAMNAME'].upper() == dpam_name, f"DPAMNAME={exthdr['DPAMNAME']} differs from expected value: {dpam_name}"
        assert exthdr['FPAMNAME'].upper() == fpam_name, f"FPAMNAME={exthdr['FPAMNAME']} differs from expected value: {fpam_name}"
        assert exthdr['SPAMNAME'].upper() == spam_name, f"SPAMNAME={exthdr['SPAMNAME']} differs from expected value: {spam_name}"
        assert exthdr['LSAMNAME'].upper() == lsam_name, f"LSAMNAME={exthdr['LSAMNAME']} differs from expected value: {lsam_name}"
        assert exthdr['FSAMNAME'].upper() == fsam_name, f"FSAMNAME={exthdr['FSAMNAME']} differs from expected value: {fsam_name}"

    # All images without the slit must have the same setup as with the slit, but
    # for FSAMNAME=OPEN
    for image in dataset_open:
        exthdr = image.ext_hdr
        assert exthdr['CFAMNAME'].upper() == cfam_name, f"CFAMNAME={exthdr['CFAMNAME']} differs from expected value: {cfam_name}"
        assert exthdr['DPAMNAME'].upper() == dpam_name, f"DPAMNAME={exthdr['DPAMNAME']} differs from expected value: {dpam_name}"
        assert exthdr['FPAMNAME'].upper() == fpam_name, f"FPAMNAME={exthdr['FPAMNAME']} differs from expected value: {fpam_name}"
        assert exthdr['SPAMNAME'].upper() == spam_name, f"SPAMNAME={exthdr['SPAMNAME']} differs from expected value: {spam_name}"
        assert exthdr['LSAMNAME'].upper() == lsam_name, f"LSAMNAME={exthdr['LSAMNAME']} differs from expected value: {lsam_name}"
        # It can only be OPEN
        assert exthdr['FSAMNAME'].upper() == 'OPEN', f"FSAMNAME={exthdr['FSAMNAME']} differs from expected value: OPEN"

    # Split each subset with the slit in by FSMX/Y values (FSM values are not used)
    dataset_slit_subsets = []
    dataset_slit_y = dataset_slit.split_dataset(exthdr_keywords=['FSMY'])[0]
    for ds_1 in dataset_slit_y:
        dataset_slit_subsets += ds_1.split_dataset(exthdr_keywords=['FSMX'])[0]
        
    # Average all spectra of the images with FSAM=OPEN
    if average.lower() == 'mean': 
        spec_open = np.mean([ds.hdu_list["SPEC"].data for ds in dataset_open], axis=0)
    elif average.lower() == 'median':
        spec_open = np.median([ds.hdu_list["SPEC"].data for ds in dataset_open], axis=0)
    else:
        raise ValueError(f'Averaging method {average} not recognized.')

    # Average all spectra of the images with the slit in by FSM position and get
    # the wavelength zero-point solution for each one
    slit_trans_fsm = []
    slit_pos_x = []
    slit_pos_y = []
    for subset in dataset_slit_subsets:
        slit_pos_x += [subset[0].ext_hdr['WV0_X']]
        slit_pos_y += [subset[0].ext_hdr['WV0_Y']]
        if average.lower() == 'mean':
            slit_trans_fsm += [np.mean([ds.hdu_list["SPEC"].data/spec_open for ds in subset], axis=0)]
        # At this point average can only take 'mean' and 'median' values
        else:
            slit_trans_fsm += [np.median([ds.hdu_list["SPEC"].data/spec_open for ds in subset], axis=0)]
    # Double check they all have the same length
    slit_pos_x = np.array(slit_pos_x)
    slit_pos_y = np.array(slit_pos_y)
    slit_trans_fsm = np.array(slit_trans_fsm) 
    if not (len(slit_pos_x) == len(slit_pos_y) == len(slit_trans_fsm)):
        raise ValueError('The lengths of distinct FSM positions and averaged spectra is different.')

    # If there's only one position, there's no interpolation
    if len(np.unique(slit_pos_y)) == len(np.unique(slit_pos_x)) == 1:
        print('Only one unique position in the data. Returning slit transmission at that position.')
        return (slit_trans_fsm,
            slit_pos_x,
            slit_pos_y)

    # If no target pixels are provided, create a series
    if target_pix == None:
        # If the FSM images are along one direction:
        if len(np.unique(slit_pos_x)) == 1:
            x_tmp = np.ones(n_gridy) * np.unique(slit_pos_x)
            y_tmp = np.linspace(y_range[0], y_range[1], n_gridy)
        elif len(np.unique(slit_pos_y)) == 1:
            x_tmp = np.linspace(x_range[0], x_range[1], n_gridx)
            y_tmp = np.ones(n_gridx) * np.unique(slit_pos_y)
        # If the FSM images span a 2-d grid:
        else:
            x_tmp = np.linspace(x_range[0], x_range[1], n_gridx)
            y_tmp = np.linspace(y_range[0], y_range[1], n_gridy)
        target_pix = np.array(np.meshgrid(x_tmp, y_tmp)).reshape(2, n_gridx*n_gridy)

    # Derive slit transmission at desired locations 
    # 1-d cases: The positions along one of the slit dimensions is constant.
    # P.S. scipy takes care of raising exceptions if there's any extrapolation
    if len(np.unique(slit_pos_y)) == 1:
        interpolant = interp1d(slit_pos_x, slit_trans_fsm, axis=0, kind=kind)
        slit_trans_interp = interpolant(target_pix[0])
    elif len(np.unique(slit_pos_x)) == 1:
        interpolant = interp1d(slit_pos_y, slit_trans_fsm, axis=0, kind=kind)
        slit_trans_interp = interpolant(target_pix[1])
    else:
    # 2-d grid:
        try:
            if kind.lower() != 'linear':
                raise ValueError('Only linear interpolation is available for',
                    'two dimensional scattered data.')
            else:
                interpolator = LinearNDInterpolator(np.c_[slit_pos_x, slit_pos_y],
                    slit_trans_fsm)
                slit_trans_interp = interpolator(target_pix[0], target_pix[1])
            # If there's some extrapolation, redefine target points to be within limits
            if np.sum(np.isnan(slit_trans_interp) == True):
                raise ValueError('Some target points require extrapolation.'
                    'Make sure all target points are within the interpolator support.')   
        except:
            raise ValueError('Not enough independent values to derive a slit transmission map')

    # Raise ValueError if all values are NaN
    if np.all(np.isnan(slit_trans_interp)):
        raise ValueError('There are no valid target positions within the ' +
            'range of input PSF locations')
    
    pri_hdr, ext_hdr, _, _ = corgidrp.check.merge_headers(
        dataset_slit,
        any_true_keywords=['DESMEAR', 'CTI_CORR'],
        invalid_keywords=[
                    'FRMTYPE',
                    'EACQ_ROW', 'EACQ_COL', 'SB_FP_DX', 'SB_FP_DY', 'SB_FS_DX', 'SB_FS_DY',
                    'Z2AVG', 'Z3AVG', 'Z4AVG', 'Z5AVG', 'Z6AVG', 'Z7AVG', 'Z8AVG', 'Z9AVG',
                    'Z10AVG', 'Z11AVG', 'Z12AVG', 'Z13AVG', 'Z14AVG',
                    'Z2RES', 'Z3RES', 'Z4RES', 'Z5RES', 'Z6RES', 'Z7RES', 'Z8RES', 'Z9RES',
                    'Z10RES', 'Z11RES',
                    'Z2VAR', 'Z3VAR',
                    'FWC_PP_E', 'FWC_EM_E', 'WV0_X', 'WV0_Y'
                ]
        )
    input_dataset = Dataset([frame for frame in dataset_slit] + [frame for frame in dataset_open])
    slit_trans =  SlitTransmission(slit_trans_interp, pri_hdr = pri_hdr, ext_hdr = ext_hdr, x_offset = target_pix[0], y_offset = target_pix[1], input_dataset = input_dataset) 
    return slit_trans


def star_pos_spec(
    dataset,
    r_lamD=3,
    phi_deg=0,
    ):
    """ Find the position of the star using the information from the satellite
      spot. The position of the satellite spot on EXCAM is given by the
      zero-point solution. Using the information of the commanded position of
      the satellite spot with respect the occulted star, one can infer the
      location of the occulted star.
      The relative of the satellite spot with respect the occulted star is given
      in polar coordinates. The radial distance of the satellite spot is measured
      in units lambda/D, with lambda the band reference wavelength, either 730 nm
      (band 3) or 660 nm (band 2), and D=2.4 m. The polar angle is measured in
      degrees, with 0 degrees meaning +X and 90 degrees meaning +Y. The polar
      coordinates of the satellite spot are translated into (X,Y) EXCAM pixel
      coordinates, which can then be subtracted from the zero-point solution to
      infer the location of the occulted star.

      Args:
        dataset (Dataset): A Dataset with L3 spectroscopy frames.
        r_lamD (float): Radial distance of the satellite spot on EXCAM with respect
        the occulted star in units of lambda/D.
        phi_deg (float): Polar angle of the satellite spot on EXCAM with respect
        the occulted star in degrees, with 0 degrees meaning +X and 90 degrees
        meaning +Y.

      Returns:
          Input Dataset with updated keywords recording the satellite position
          in EXCAM pixels.
    """ 
    # Primary diameter of Roman Space Telescope in meters
    D_m=2.4
    # Basic checks
    if r_lamD < 0:
        raise ValueError('r_lamD cannot be negative. Usual range is 3-20.')

    dataset_cp = dataset.copy()
    for img in dataset_cp:
        # Check it is L3
        if img.ext_hdr['DATALVL'] != 'L3':
            raise ValueError(f"The data level must be L3 and it is {img.ext_hdr['DATALVL']}")
        # Extract satellite spot wavelength from L3 extended header (it must be present)
        try:
            lam_sat_nm = float(img.ext_hdr['WAVLEN0'])
        except:
            raise ValueError(f'WAVLEN0 keyword missing in L3 frame.')
        
        fsmlos = img.ext_hdr['FSMLOS']
        # shift of star location only for coronagraphic observations
        if fsmlos == 1:
            # Conversion from EXCAM pixels to milliarsec
            plate_scale_mas = img.ext_hdr['PLTSCALE']
            # Conversion from radians to milliarsec (mas/rad)
            rad2mas = 180/np.pi*3600*1e3
            # lam/D in radians
            lamDrad = 1e-9*lam_sat_nm/D_m
            # lam/D to EXCAM pixels
            r_pix = r_lamD*lamDrad*rad2mas/plate_scale_mas
            # EXCAM (X,Y) coordinates
            X_pix = r_pix * np.cos(phi_deg*np.pi/180)
            Y_pix = r_pix * np.sin(phi_deg*np.pi/180)
            # Update estimated location of the occulted star
            img.ext_hdr['STARLOCX'] = img.ext_hdr['WV0_X'] - X_pix
            img.ext_hdr['STARLOCY'] = img.ext_hdr['WV0_Y'] - Y_pix
        else:
            img.ext_hdr['STARLOCX'] = img.ext_hdr['WV0_X']
            img.ext_hdr['STARLOCY'] = img.ext_hdr['WV0_Y']

    return dataset_cp


def spec_fluxcal(dataset_or_image, calspec_file = None):
    """
    generates the SpecFluxCal calibration product for one band,
    calculates the spectral flux calibration or spectro-photometric calibration, that describes the
    sensitivity of the spectrometer, i.e. how an input power is
    converted into how many photoelectrons per wavelength, with the final unit erg/(s * cm^2 * AA)/(photoelectron/s/bin).
    The input is expected to be the dataset of an extracted spectrum of a 
    CALSPEC photometric standard star with units photoelectron/s/bin.
    
    The band flux values of the input calspec data files are divided by 
    the spectral extracted photoelectrons/s/bin interpolated on the available wavelengths.
    Propagates also errors to the spectral flux calibration file.
    
    Parameters:
        dataset_or_image (corgidrp.data.Dataset or corgidrp.data.Image): Image(s) to compute 
            the calibration factor. Should already be normalized for exposure time. Output of extract_spec.
        calspec_file (str, optional): file path to the calspec fits file of the observed star. 
                                      If None, it is downloaded from the calspec database

    Returns:
        SpecFluxCal (corgidrp.data.SpecFluxCal): A calibration object containing the computed 
            flux calibration factor in units erg/(s * cm^2 * AA)/(photoelectron/s/bin)
    """
    d_or_i = dataset_or_image.copy()
    if isinstance(d_or_i, Dataset):
        image = d_or_i[0]
        dataset = d_or_i
    else:
        image = d_or_i
        dataset = Dataset([image])
    if image.ext_hdr['BUNIT'] != "photoelectron/s":
        raise ValueError("input dataset must have unit photoelectron/s for the calibration, not {0}".format(image.ext_hdr['BUNIT']))
    
    if "SPEC" not in image.hdu_names:
        raise AttributeError("input dataset has no spectral extracted data and has not run through extract_spec")
    
    filter_name = image.ext_hdr["CFAMNAME"]
    filter_file = get_filter_name(image)
    
    # Read filter and CALSPEC data.
    wave, filter_trans = read_filter_curve(filter_file)
    
    if calspec_file is not None:
        calspec_filepath = calspec_file
        calspec_filename = os.path.basename(calspec_file)
    else:
        star_name = image.pri_hdr["TARGET"]
        calspec_filepath, calspec_filename = get_calspec_file(star_name)
    
    flux_ref = read_cal_spec(calspec_filepath, wave)
    #is this correct, do we need to consider the filter transmission at all?
    flux = flux_ref #* filter_trans
    if len(dataset) == 1:
        spec = image.hdu_list["SPEC"].data
        spec_dq = image.hdu_list["SPEC_DQ"].data
        spec_err = image.hdu_list["SPEC_ERR"].data
        spec_wave = image.hdu_list["SPEC_WAVE"].data
        spec_wave_err = image.hdu_list["SPEC_WAVE_ERR"].data
    else:
        spec = []
        spec_dq = []
        spec_err = []
        spec_wave = []
        spec_wave_err = []
        for frame in dataset:
            spec.append(frame.hdu_list["SPEC"].data)
            spec_dq.append(frame.hdu_list["SPEC_DQ"].data)
            spec_err.append(frame.hdu_list["SPEC_ERR"].data)
            spec_wave.append(frame.hdu_list["SPEC_WAVE"].data)
            spec_wave_err.append(frame.hdu_list["SPEC_WAVE_ERR"].data)
        
        spec = np.mean(np.array(spec),0)
        spec_err = np.mean(np.array(spec_err),0)
        spec_wave = np.mean(np.array(spec_wave), 0)
        spec_wave_err = np.mean(np.array(spec_wave_err), 0)
        spec_dq = np.bitwise_or.reduce(np.array(spec_dq),axis = 0)
    
    #interpolate on the extracted wavelength in nm (AA/10)
    wave_nm = wave/10.
    inter = interp1d(wave_nm, flux, fill_value="extrapolate")
    spec_flux = inter(spec_wave)/spec

    spec_flux_err = spec_flux / spec**2 * spec_err[0]

    data = np.array([spec_wave, spec_flux])
    error = np.array([spec_wave_err, spec_flux_err])
    spec_fluxcal_obj = SpecFluxCal(
        data,
        err=error,
        dq = np.tile(spec_dq, (2,1)),
        pri_hdr=image.pri_hdr,
        ext_hdr=image.ext_hdr,
        input_dataset=dataset
    )

    # Append to or create a HISTORY entry in the header.
    history_entry = "spectral flux calibration was determined by spectral extraction using SED file {0}".format(calspec_filename)
    spec_fluxcal_obj.ext_hdr.add_history(history_entry)

    return spec_fluxcal_obj


def generate_filter_offset(offset_file = None):
    """
    read the csv filter file containing at least the band names and the pixel x/y offsets 
    between the filters and generate a new SpecFilterOffset product.
    
    Args:
       offset_file (str): file name of the filter file, if none it takes data/spectroscopy/CGI_bandpass_centers.csv
       
    Returns:
       corgidrp.data.SpecFilterOffset: SpecFilterOffset product
    """
    if offset_file is None:
        offset_file = os.path.join(os.path.dirname(__file__), "data", "spectroscopy", "CGI_bandpass_centers.csv")
    table = ascii.read(offset_file, format = 'csv')
    
    offset_dict = {}
    for i, col in enumerate(table.colnames):
        if "filter" in col:
            filter_name = table.columns[i].value
        if "xoffset" in col:
            xoffset = table.columns[i].value
        if "yoffset" in col:
            yoffset = table.columns[i].value
    for i, filter in enumerate(filter_name):
        offset_dict[str(filter)] = [float(xoffset[i]), float(yoffset[i])]
    return SpecFilterOffset(offset_dict)
