# This module is written to do an absolute flux calibration observing a standard star having CALSPEC data.
import glob
import os
import numpy as np
from astropy.io import fits, ascii
from astropy import wcs
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
import corgidrp
from photutils.aperture import CircularAperture
from photutils.background import LocalBackground
from photutils.psf import fit_2dgaussian
from scipy import integrate
from corgidrp.astrom import centroid
import urllib

# Dictionary of anticipated bright and dim CASLPEC standard star names and corresponding fits names
calspec_names= {
# bright standards
'109 Vir': '109vir_stis_005.fits',
'Vega': 'alpha_lyr_stis_011.fits',
'Eta Uma': 'etauma_stis_008.fits',
'Lam Lep': 'lamlep_stis_008.fits',
'KSI2 CETI': 'ksi2ceti_stis_008.fits',
# dim standards
'TYC 4433-1800-1': '1808347_stiswfc_006.fits',
'TYC 4205-1677-1': '1812095_stisnic_008.fits',
'TYC 4212-455-1': '1757132_stiswfc_006.fits',
'TYC 4209-1396-1': '1805292_stisnic_008.fits',
'TYC 4413-304-1': 'p041c_stisnic_010.fits',
'UCAC3 313-62260': 'kf08t3_stisnic_005.fits',
'BPS BS 17447-0067': '1802271_stiswfcnic_006.fits',
'TYC 4424-1286-1': '1732526_stisnic_009.fits',
'GSC 02581-02323': 'p330e_stiswfcnic_007.fits',
'TYC 4207-219-1': '1740346_stisnic_005.fits'
}

calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/'

def get_calspec_file(star_name):
    """
    download the corresponding CALSPEC fits file and return the file path
    
    Args:
        star_name (str): 
    
    Returns:
        str: file path
    """
    if star_name not in calspec_names:
        raise ValueError('{0} is not in list of anticipated standard stars {1}, please check naming'.format(star_name, calspec_names.keys()) )
    fits_name = calspec_names.get(star_name)
    # TODO: be flexible with the version of the calspec fits file, so essentially, the number in the name should not matter
    fits_url = calspec_url + fits_name
    try:
        calspec_dir = os.path.join(os.path.dirname(corgidrp.config_filepath), "calspec_data")
        if not os.path.exists(calspec_dir):
            os.mkdir(calspec_dir)
        file_name, headers = urllib.request.urlretrieve(fits_url, filename =  os.path.join(calspec_dir, fits_name))
    except:
        raise Exception("cannot access CALSPEC archive web page and/or download {0}".format(fits_name))
    return file_name

def get_filter_name(image):
    """
    return the name of the transmission curve csv file of used color filter
    
    Args:
        image (corgidrp.image): image of the observed calstar
    
    Returns:
        str: filepath of the selected filter curve
    """
    datadir = os.path.join(os.path.dirname(__file__), "data", "filter_curves")
    filters = os.path.join(datadir, "*.csv")
    filter = image.ext_hdr['CFAMNAME']
    filter_names = os.listdir(datadir)

    filter_name = [name for name in filter_names if filter in name]
    if filter_name == []:
        raise ValueError("there is no filter available with name {0}".format(filter))
    else:
        return os.path.join(datadir,filter_name[0])

def read_filter_curve(filter_filename):
    """
    read the transmission curve csv file of the color filters
    
    Args:
        filter_filename (str): file name of the transmission curve data
    
    Returns:
        lambda_nm (np.array): wavelength in unit Angstroem
        transmission (np.array): transmission of the filter < 1
    """
    tab = ascii.read(filter_filename, format='csv', header_start = 3, data_start = 4)
    lambda_nm = tab['lambda_nm'].data #unit nm
    transmission = tab['%T'].data
    return lambda_nm * 10 , transmission/100.

def read_cal_spec(calspec_filename, filter_wavelength):
    """
    read the calspec flux density data interpolated on the wavelength grid of the transmission curve
    
    Args:
        calspec_filename (str): file name of the CALSPEC fits file
        filter_wavelength (np.array): wavelength grid of the transmission curve in unit Angstroem
    
    Returns:
        np.array: flux density in Jy interpolated on the wavelength grid of the transmission curve 
        in CALSPEC units erg/(s * cm^2 * AA)
    """
    hdulist = fits.open(calspec_filename)
    data = hdulist[1].data
    hdulist.close()
    w = data['WAVELENGTH'] #wavelength in Angstroem
    flux = data['FLUX']
    flux = flux[(w<=filter_wavelength[-1]) & (w>=filter_wavelength[0])] #erg/(s*cm^2*AA)
    w = w[(w<=filter_wavelength[-1]) & (w>=filter_wavelength[0])]

    #interpolate on transmission curve wavelengths
    flux_inter = np.interp(filter_wavelength, w, flux)
    
    return flux_inter

def calculate_band_flux(filter_curve, calspec_flux, filter_wavelength):
    """
    calculate the average band flux of a calspec source in the filter band, see convention A in Gordon et al. (2022)
    TBC if needed at all
    
    Args:
        filter_curve (np.array): filter transmission curve over the filter_wavelength
        calspec_flux (np.array): converted flux in units of erg/(s*cm^2*AA) of the calpec source in the filter band
        filter_wavelength (np.array): wavelengths in units Angstroem in the filter band 
    
    Returns:
        float: average band flux of the calspec star in unit erg/(s*cm^2*AA)
    """
    multi_flux = calspec_flux * filter_curve * filter_wavelength
    multi_band = filter_curve * filter_wavelength
    aver_flux = integrate.simpson(multi_flux, x=filter_wavelength)/integrate.simpson(multi_band, x=filter_wavelength)
    
    return aver_flux

def calculate_effective_lambda(filter_curve, calspec_flux, filter_wavelength):
    """
    calculate the effective wavelength of a calspec source in the filter band, see convention A in Gordon et al. (2022)
    TBC if needed at all
    
    Args:
        filter_curve (np.array): filter transmission curve over the filter_wavelength
        calspec_flux (np.array): converted flux in units of the calpec source in the filter band
        filter_wavelength (np.array): wavelengths in units nm in the filter band 
    
    Returns:
        float: effective wavelength in unit Angstroem
    """
    multi_flux = calspec_flux * filter_curve * np.square(filter_wavelength)
    multi_band = calspec_flux * filter_curve * filter_wavelength
    eff_lambda = integrate.simpson(multi_flux, x=filter_wavelength)/integrate.simpson(multi_band, x=filter_wavelength)
    
    return eff_lambda

def calculate_pivot_lambda(filter_curve, filter_wavelength):
    """
    calculate the reference pivot wavelength of the filter band, see convention B in Gordon et al. (2022)
    
    Args:
        filter_curve (np.array): filter transmission curve over the filter_wavelength
        filter_wavelength (np.array): wavelengths in unit Angstroem in the filter band 
    
     Returns:
        float: pivot wavelength in unit Angstroem
    """
    multi_flux = filter_curve * filter_wavelength
    multi_band = filter_curve / filter_wavelength
    piv_lambda = np.sqrt(integrate.simpson(multi_flux, x=filter_wavelength)/integrate.simpson(multi_band, x=filter_wavelength))
    
    return piv_lambda

def calculate_flux_ref(filter_wavelength, calspec_flux, wave_ref):
    """
    calculate the flux at the reference wavelength of the filter band
    
    Args:
        filter_wavelength (np.array): wavelengths in unit Angstroem in the filter band 
        calspec_flux (np.array): converted flux in units of the calpec source in the filter band
        wave_ref (float): reference wavelength in unit Angstroem
    
    Returns:
        float: flux at reference wavelength in unit erg/(s*cm^2*AA)
    """
    
    flux_ref = np.interp(wave_ref, filter_wavelength, calspec_flux)
    return flux_ref

def compute_color_cor(filter_curve, filter_wavelength , flux_ref, wave_ref, flux_source):
    """
    Compute the color correction factor K given the filter bandpass, reference spectrum (CALSPEC),
    and source spectrum model.  To use this color correction, divide the flux density
    for a band by K.  Such color corrections are needed to compute the correct
    flux density at the reference wavelength for a source with the flux_source
    spectral shape in the photometric convention that provides the flux density
    at a reference wavelength (convention B, see Gordon et al. 2022, The Astronomical Journal 163:267, for details).
    Thus the flux density value found by applying the calibration factor on the found detected electrons 
    of an arbitrary source should be divided by K (for the appropriate filter and spectral shape) 
    to produce the flux density at the reference wavelength of the filter. 
    The color correction adjusts the calibration factor to align the reference spectral shape 
    with the current source, which results in the correct flux density at the reference wavelength.

    Args:
    filter_curve (np.array): transmission of the filter bandpass
    filter_wavelength (np.array): the wavelengths of the filter bandpass, flux_ref, and flux_source in unit Angstroem
    flux_ref (np.array): reference flux density F(lambda) as a function of wavelength
    wave_ref (float): reference wavelength in unit Angstroem
    flux_source (np.array): source flux density F(lambda) as a function of wavelength in CALSPEC unit erg/(s * cm^2 * AA)
    
    Returns:
        float: color correction factor K
    """
    # get the flux densities at the reference wavelength
    flux_source_lambda_ref = calculate_flux_ref(filter_wavelength, flux_source, wave_ref)
    flux_ref_lambda_ref = calculate_flux_ref(filter_wavelength, flux_ref, wave_ref)

    # compute the top and bottom integrals
    int_source = integrate.simpson(filter_wavelength * filter_curve * flux_source / flux_source_lambda_ref, x=filter_wavelength)
    int_ref = integrate.simpson(filter_wavelength * filter_curve * flux_ref / flux_ref_lambda_ref, x=filter_wavelength)

    return int_source / int_ref

def calculate_band_irradiance(filter_curve, calspec_flux, filter_wavelength):
    """
    calculate the integrated band flux, irradiance of a calspec source in the filter band
    to determine the apparent magnitude
    
    Args:
        filter_curve (np.array): filter transmission curve over the filter_wavelength
        calspec_flux (np.array): converted flux in units of erg/(s*cm^2*AA) of the calpec source in the filter band
        filter_wavelength (np.array): wavelengths in units Angstroem in the filter band 
    
    Returns:
        float: band irradiance of the calspec star in unit erg/(s*cm^2)
    """
    multi_flux = calspec_flux * filter_curve
    irrad = integrate.simpson(multi_flux, x=filter_wavelength)
    
    return irrad


def aper_phot(image, encircled_radius, frac_enc_energy=1., method='subpixel', subpixels=5,
              background_sub=False, r_in=5, r_out=10, centering_method='xy'):
    """
    Returns the flux in photo-electrons of a point source using aperture photometry,
    with an option for background subtraction via an annulus.
    
    Parameters:
        image: corgidrp.data.Image – the source image.
        encircled_radius (float): Radius for the aperture.
        frac_enc_energy (float): Fraction of energy in the aperture.
        method (str): Overlap method, e.g. 'subpixel'.
        subpixels (int): Subpixel resolution.
        background_sub (bool): If True, subtract background.
        r_in (float): Inner annulus radius.
        r_out (float): Outer annulus radius.
        centering_method (str): 'xy' for centroiding or 'wcs' for WCS-based centering.
    
    Returns:
        tuple: (flux, flux_err) or (flux, flux_err, back) if background_sub is True.
    """
    if frac_enc_energy <= 0 or frac_enc_energy > 1:
        raise ValueError("frac_enc_energy {0} should be within 0 < fee <= 1".format(str(frac_enc_energy)))
    
    # Work on a copy so that background subtraction does not alter original data.
    dat = image.data.copy()
    
    # Determine the center position using either WCS or centroid method.
    if centering_method == 'wcs':
        ra = image.pri_hdr['RA']
        dec = image.pri_hdr['DEC']
        target_skycoord = SkyCoord(ra=ra, dec=dec, unit='deg')
        w = wcs.WCS(image.ext_hdr)
        pos = wcs.utils.skycoord_to_pixel(target_skycoord, w, origin=1)
    elif centering_method == 'xy':
        x_center, y_center = centroid(image.data)
        pos = (x_center, y_center)
    else:
        raise ValueError("Invalid centering_method. Choose 'xy' or 'wcs'.")
    
    # Optionally subtract the background.
    if background_sub:
        bkg = LocalBackground(r_in, r_out)
        back = bkg(dat, pos[0], pos[1], mask=image.dq.astype(bool))
        dat -= back

    # Create the circular aperture and compute photometry.
    aper = CircularAperture(pos, encircled_radius)
    aperture_sums, aperture_sums_errs = aper.do_photometry(
        dat,
        error=image.err[0],
        mask=image.dq.astype(bool),
        method=method,
        subpixels=subpixels
    )
    
    flux = aperture_sums[0] / frac_enc_energy
    flux_err = aperture_sums_errs[0] / frac_enc_energy
    
    if background_sub:
        return flux, flux_err, back
    else:
        return flux, flux_err


def phot_by_gauss2d_fit(image, fwhm, fit_shape=None, background_sub=False, r_in=5,
                        r_out=10, centering_method='xy'):
    """
    Returns the flux in photo-electrons using a 2D Gaussian fit.
    Allows optional background subtraction and selection of centering method.
    
    Parameters:
        image: corgidrp.data.Image – the source image.
        fwhm (float): Estimated full-width at half maximum.
        fit_shape (int or tuple, optional): Fitting region shape.
        background_sub (bool): If True, subtract background.
        r_in (float): Inner annulus radius.
        r_out (float): Outer annulus radius.
        centering_method (str): 'xy' or 'wcs' centering.
    
    Returns:
        tuple: (flux, flux_err)
    """
    # Determine the star center via WCS or centroid method.
    if centering_method == 'wcs':
        ra = image.pri_hdr['RA']
        dec = image.pri_hdr['DEC']
        target_skycoord = SkyCoord(ra=ra, dec=dec, unit='deg')
        w = wcs.WCS(image.ext_hdr)
        pos = wcs.utils.skycoord_to_pixel(target_skycoord, w, origin=1)
    elif centering_method == 'xy':
        x_center, y_center = centroid(image.data)
        pos = (x_center, y_center)
    else:
        raise ValueError("Invalid centering_method. Choose 'xy' or 'wcs'.")

    data_for_fit = image.data.copy()
    if background_sub:
        bkg = LocalBackground(r_in, r_out)
        back = bkg(data_for_fit, pos[0], pos[1], mask=image.dq.astype(bool))
        data_for_fit = data_for_fit - back

    err = image.err[0].copy()
    err[err == 0] = np.finfo(np.float32).eps

    if fit_shape is None:
        fit_shape = image.data.shape[0] - 1

    psf_phot = fit_2dgaussian(data_for_fit, xypos=pos, fwhm=fwhm, fit_shape=fit_shape,
                              mask=image.dq.astype(bool), error=err)
    flux = psf_phot.results['flux_fit'][0]
    flux_err = psf_phot.results['flux_err'][0]
    return flux, flux_err


def calibrate_fluxcal_aper(dataset_or_image, flux_or_irr = 'flux', phot_kwargs=None):
    """
    Computes a flux calibration factor using aperture photometry.
    
    The photometry parameters are controlled via the `phot_kwargs` dictionary.
    Defaults are provided below if these parameters are not defined. 
    Accepted keywords:
        'encircled_radius' (float): The radius of the circular aperture used for photometry.
        'frac_enc_energy' (float): The fraction of the total flux expected to be enclosed 
            within the aperture. Must be in the range (0, 1].
        'method' (str): The photometry method to use. For example, 'subpixel' indicates subpixel 
            sampling for the aperture.
        'subpixels' (int): The number of subpixels per pixel to use in the photometry calculation 
            or improved resolution.
        'background_sub' (bool): Flag indicating whether to subtract background using an annulus.
        'r_in' (float): The inner radius of the annulus used for background estimation.
        'r_out' (float): The outer radius of the annulus used for background estimation.
        'centering_method' (str): The method for determining the star's center. Options include 
            'xy' for centroiding or 'wcs' for WCS-based centering.
    
    Parameters:
        dataset_or_image:
            A corgidrp.data.Dataset or a single corgidrp.data.Image from which to compute 
            the calibration factor.
        flux_or_irr (str, optional): Whether flux ('flux') or in-band irradiance ('irr) should 
            be used.
        phot_kwargs (dict, optional):
            A dictionary of keyword arguments controlling the aperture photometry.

    Returns:
        FluxcalFactor:
            A calibration object containing the computed flux calibration factor and its 
            associated error.
    """
    if isinstance(dataset_or_image, corgidrp.data.Dataset):
        image = dataset_or_image[0]
        dataset = dataset_or_image
    else:
        image = dataset_or_image
        dataset = corgidrp.data.Dataset([image])
    
    if phot_kwargs is None:
        phot_kwargs = {
            'encircled_radius': 5,
            'frac_enc_energy': 1.0,
            'method': 'subpixel',
            'subpixels': 5,
            'background_sub': False,
            'r_in': 5,
            'r_out': 10,
            'centering_method': 'xy'
        }
    
    star_name = image.ext_hdr["TARGET"]
    exptime = image.ext_hdr["EXPTIME"]
    filter_name = image.ext_hdr["CFAMNAME"]
    filter_file = get_filter_name(image)
    
    # Read filter and CALSPEC data.
    wave, filter_trans = read_filter_curve(filter_file)
    calspec_filepath = get_calspec_file(star_name) 
    flux_ref = read_cal_spec(calspec_filepath, wave)
    
    if flux_or_irr == 'flux':
        flux = calculate_band_flux(filter_trans, flux_ref, wave)
    elif flux_or_irr == 'irr':
        flux = calculate_band_irradiance(filter_trans, flux_ref, wave)
    else:
        raise ValueError("Invalid flux method. Choose 'flux' or 'irr'.")
    
    result = aper_phot(image, **phot_kwargs)
    if phot_kwargs.get('background_sub', False):
        ap_sum, ap_sum_err, back = result
    else:
        ap_sum, ap_sum_err = result

    fluxcal_fac = (flux * exptime) / ap_sum
    fluxcal_fac_err = (flux * exptime) / ap_sum**2 * ap_sum_err

    fluxcal_obj = corgidrp.data.FluxcalFactor(
        np.array([[fluxcal_fac]]),
        err=np.array([[[fluxcal_fac_err]]]),
        pri_hdr=image.pri_hdr,
        ext_hdr=image.ext_hdr,
        input_dataset=dataset
    )
    fluxcal_obj.ext_hdr["TARGET"] = star_name
    fluxcal_obj.ext_hdr["CFAMNAME"] = filter_name
    
    return fluxcal_obj



def calibrate_fluxcal_gauss2d(dataset_or_image, flux_or_irr = 'flux', phot_kwargs=None):
    """
    Computes a flux calibration factor using a 2D Gaussian fit.
    All photometry settings are provided via the phot_kwargs dictionary.

    Defaults are provided below if these parameters are not defined. 
    Accepted keywords:
        'fwhm' (float): The expected full width at half maximum.
        'fit_shape' (int or tuple): Fitting region shape.
        'background_sub' (bool): Flag indicating whether to subtract background using an annulus.
        'r_in' (float): The inner radius of the annulus used for background estimation.
        'r_out' (float): The outer radius of the annulus used for background estimation.
        'centering_method' (str): The method for determining the star's center. Options include 
            'xy' for centroiding or 'wcs' for WCS-based centering.

    Parameters:
    dataset_or_image:
        A corgidrp.data.Dataset or a single corgidrp.data.Image from which to compute 
        the calibration factor.
    flux_or_irr (str, optional): Whether flux ('flux') or in-band irradiance ('irr) should 
        be used.
    phot_kwargs (dict, optional):
        A dictionary of keyword arguments controlling the aperture photometry.
            
    Returns:
        FluxcalFactor calibration object.
    """
    if isinstance(dataset_or_image, corgidrp.data.Dataset):
        image = dataset_or_image[0]
        dataset = dataset_or_image
    else:
        image = dataset_or_image
        dataset = corgidrp.data.Dataset([image])
    
    if phot_kwargs is None:
        phot_kwargs = {
        'fwhm': 3,
        'fit_shape': None,
        'background_sub': False,
        'r_in': 5,
        'r_out': 10,
        'centering_method': 'xy'
    }

    star_name = image.ext_hdr["TARGET"]
    exptime = image.ext_hdr["EXPTIME"]
    filter_file = get_filter_name(image)
    
    wave, filter_trans = read_filter_curve(filter_file)
    calspec_filepath = get_calspec_file(star_name)
    flux_ref = read_cal_spec(calspec_filepath, wave)
    
    if flux_or_irr == 'flux':
        flux = calculate_band_flux(filter_trans, flux_ref, wave)
    elif flux_or_irr == 'irr':
        flux = calculate_band_irradiance(filter_trans, flux_ref, wave)
    else:
        raise ValueError("Invalid flux method. Choose 'flux' or 'irr'.")
    
    flux_sum, flux_sum_err = phot_by_gauss2d_fit(image, **phot_kwargs)
    
    # Factor in exposure time.
    fluxcal_fac = (flux * exptime) / flux_sum
    fluxcal_fac_err = (flux * exptime) / flux_sum**2 * flux_sum_err

    fluxcal_obj = corgidrp.data.FluxcalFactor(
        np.array([[fluxcal_fac]]),
        err=np.array([[[fluxcal_fac_err]]]),
        pri_hdr=image.pri_hdr,
        ext_hdr=image.ext_hdr,
        input_dataset=dataset
    )
    
    return fluxcal_obj
