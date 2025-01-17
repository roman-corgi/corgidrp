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
from photutils.psf import fit_2dgaussian
from scipy import integrate
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

def aper_phot(image, encircled_radius, frac_enc_energy, method = 'exact', subpixels = 5):
    """
    returns the flux in photo-electrons of a point source at the target Ra/Dec position
    and using a circular aperture by applying aperture_photometry of photutils.
    We assume that background subtraction is already done.
    
    Args:
        image (corgidrp.data.Image): combined source exposure image
        encircled_radius (float): pixel radius of the circular aperture to sum the flux
        frac_enc_energy (float): fraction of encircled energy inside the encircled_radius of the PSF, inverse aperture correction, 0...1
        method (str): {‘exact’, ‘center’, ‘subpixel’}, The method used to determine the overlap of the aperture on the pixel grid, 
        default is 'exact'. For detailed description see https://photutils.readthedocs.io/en/stable/api/photutils.aperture.CircularAnnulus.html
        subpixels (int): For the 'subpixel' method, resample pixels by this factor in each dimension. That is, each pixel is divided 
                         into subpixels**2 subpixels. This keyword is ignored unless method='subpixel', default is 5
    
    Returns:
        float: integrated flux of the point source in unit photo-electrons and corresponding error
    """
    #calculate the x and y pixel positions using the RA/Dec target position and applying WCS conversion
    ra = image.pri_hdr['RA']
    dec = image.pri_hdr['DEC']
    
    target_skycoord = SkyCoord(ra = ra, dec = dec, unit='deg')
    w = wcs.WCS(image.ext_hdr)
    pix = wcs.utils.skycoord_to_pixel(target_skycoord, w, origin = 1)
    aper = CircularAperture(pix, encircled_radius)
    aperture_sums, aperture_sums_errs = \
        aper.do_photometry(image.data, error = image.err[0], mask = image.dq.astype(bool), method = method, subpixels = subpixels)
    return aperture_sums[0]/frac_enc_energy, aperture_sums_errs[0]/frac_enc_energy

def phot_by_gauss2d_fit(image, fwhm, fit_shape = None):
    """
    returns the flux in photo-electrons of a point source at the target Ra/Dec position
    and using a circular aperture by applying aperture_photometry of photutils
    We assume that background subtraction is already done.
    
    Args:
        image (corgidrp.data.Image): combined source exposure image
        fwhm (float): estimated fwhm of the point source
        fit_shape (int or tuple of two ints): optional
            The shape of the fitting region. If a scalar, then it is assumed
            to be a square. If `None`, then the shape of the input ``data``.
            It must be an odd value and should be much bigger than fwhm.
    
    Returns:
        float: integrated flux of the Gaussian2d fit of the point source in unit photo-electrons and corresponding error
    """
    ra = image.pri_hdr['RA']
    dec = image.pri_hdr['DEC']
    
    target_skycoord = SkyCoord(ra = ra, dec = dec, unit='deg')
    w = wcs.WCS(image.ext_hdr)
    pix = wcs.utils.skycoord_to_pixel(target_skycoord, w, origin = 1)
    # fit_2dgaussian: error weighting raises exception if error is zero
    err = image.err[0]
    err[err == 0] = np.finfo(np.float32).eps
    
    if fit_shape == None:
        fit_shape = np.shape(image.data)[0] -1
    
    psf_phot = fit_2dgaussian(image.data, xypos = pix, fwhm = fwhm, fit_shape = fit_shape, mask = image.dq.astype(bool), error = err)
    flux = psf_phot.results['flux_fit'][0]
    flux_err = psf_phot.results['flux_err'][0]
    return flux, flux_err

def calibrate_fluxcal_aper(image, encircled_radius, frac_enc_energy, method = 'exact', subpixels = 5):
    """
    fills the FluxcalFactors calibration product values for one filter band,
    calculates the flux calibration factors by aperture photometry.
    The band flux values are divided by the found photoelectrons.
    Propagates also errors to flux calibration factor calfile.
    
    Args:
        image (corgidrp.data.Image): combined source exposure image
        encircled_radius (float): pixel radius of the circular aperture to sum the flux
        frac_enc_energy (float): fraction of encircled energy inside the encircled_radius of the PSF, inverse aperture correction, 0...1
        method (str): {‘exact’, ‘center’, ‘subpixel’}, The method used to determine the overlap of the aperture on the pixel grid, 
        default is 'exact'. For detailed description see https://photutils.readthedocs.io/en/stable/api/photutils.aperture.CircularAnnulus.html
        subpixels (int): For the 'subpixel' method, resample pixels by this factor in each dimension. That is, each pixel is divided 
                         into subpixels**2 subpixels. This keyword is ignored unless method='subpixel', default is 5
    
    Returns:
        corgidrp.data.FluxcalFactor: FluxcalFactor calibration object with the value and error of the corresponding filter
    """
    star_name = image.ext_hdr["TARGET"]
    
    filter_name = image.ext_hdr["CFAMNAME"]
    filter_file = get_filter_name(image)
    # read the transmission curve from the color filter file
    wave, filter_trans = read_filter_curve(filter_file)
    
    calspec_filepath = get_calspec_file(star_name)
    
    # calculate the flux from the user given CALSPEC file binned on the wavelength grid of the filter
    flux_ref = read_cal_spec(calspec_filepath, wave)
    flux = calculate_band_flux(filter_trans, flux_ref, wave)
    
    ap_sum, ap_sum_err = aper_phot(image, encircled_radius, frac_enc_energy, method = method, subpixels = subpixels)
    
    fluxcal_fac = flux/ap_sum
    fluxcal_fac_err = flux/ap_sum**2 * ap_sum_err
    
    dataset = corgidrp.data.Dataset([image])
    fluxcal = corgidrp.data.FluxcalFactor(np.array([[fluxcal_fac]]), err = np.array([[[fluxcal_fac_err]]]), pri_hdr = image.pri_hdr, ext_hdr = image.ext_hdr, input_dataset = dataset)
    
    return fluxcal
    
def calibrate_fluxcal_gauss2d(image, fwhm, fit_shape = None):
    """
    fills the FluxcalFactors calibration product values for one filter band,
    calculates the flux calibration factors by fitting a 2D Gaussian.
    The band flux values are divided by the found photoelectrons.
    Propagates also errors to flux calibration factor calfile.
    
    Args:
        image (corgidrp.data.Image): combined source exposure image
        fwhm (float): estimated fwhm of the point source
        fit_shape (int or tuple of two ints): optional
            The shape of the fitting region. If a scalar, then it is assumed
            to be a square. If `None`, then the shape of the input ``data``.
            It must be an odd value and should be much bigger than fwhm.
    
    Returns:
        corgidrp.data.FluxcalFactor: FluxcalFactor calibration object with the value and error of the corresponding filter
    """
    star_name = image.ext_hdr["TARGET"]
    filter_name = image.ext_hdr["CFAMNAME"]
    filter_file = get_filter_name(image)
    # read the transmission curve from the color filter file
    wave, filter_trans = read_filter_curve(filter_file)
    
    calspec_filepath = get_calspec_file(star_name)
    
    # calculate the flux from the user given CALSPEC file binned on the wavelength grid of the filter
    flux_ref = read_cal_spec(calspec_filepath, wave)
    flux = calculate_band_flux(filter_trans, flux_ref, wave)
    
    flux_sum, flux_sum_err = phot_by_gauss2d_fit(image, fwhm, fit_shape = fit_shape)
    
    fluxcal_fac = flux/flux_sum
    fluxcal_fac_err = flux/flux_sum**2 * flux_sum_err
    
    dataset = corgidrp.data.Dataset([image])
    fluxcal = corgidrp.data.FluxcalFactor(np.array([[fluxcal_fac]]), err = np.array([[[fluxcal_fac_err]]]), pri_hdr = image.pri_hdr, ext_hdr = image.ext_hdr, input_dataset = dataset)
    
    return fluxcal
