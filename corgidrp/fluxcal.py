# This module is written to do an absolute flux calibration observing a standard star having CALSPEC data.
import glob
import os
import json
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
from corgidrp.astrom import centroid_with_roi
from urllib.request import Request, urlopen, urlretrieve

# Dictionary of anticipated bright and dim CASLPEC standard star names and corresponding fits names
calspec_names= {
# bright standards
'109 vir': '109vir_stis_005.fits',
'vega': 'alpha_lyr_stis_011.fits',
'eta uma': 'etauma_stis_008.fits',
'lam lep': 'lamlep_stis_008.fits',
'ksi2 ceti': 'ksi2ceti_stis_008.fits',
# dim standards
'tyc 4433-1800-1': '1808347_stiswfc_006.fits',
'tyc 4205-1677-1': '1812095_stisnic_008.fits',
'tyc 4212-455-1': '1757132_stiswfc_006.fits',
'tyc 4209-1396-1': '1805292_stisnic_008.fits',
'tyc 4413-304-1': 'p041c_stisnic_010.fits',
'ucac3 313-62260': 'kf08t3_stisnic_005.fits',
'bps bs 17447-0067': '1802271_stiswfcnic_006.fits',
'tyc 4424-1286-1': '1732526_stisnic_009.fits',
'gsc 02581-02323': 'p330e_stiswfcnic_007.fits',
'tyc 4207-219-1': '1740346_stisnic_005.fits'
}

calspec_url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/'

def get_calspec_file(star_name):
    """
    look up the calspec fits file name in the names dict and look for 
    the corresponding file in the .corgidrp/  calspec_data. If not available 
    download the corresponding CALSPEC fits file and return the file path
    
    Args:
        star_name (str): 
    
    Returns:
        str: file path
        str: fits file name
    """
    
    calspec_dir = os.path.join(os.path.dirname(corgidrp.config_filepath), "calspec_data")
    names_file = os.path.join(calspec_dir, "calspec_names.json")
    calspec_names_ff = calspec_names
    if not os.path.exists(calspec_dir):
        os.mkdir(calspec_dir)
        with open(names_file, 'w') as f:
            json.dump(calspec_names_ff, f)
    else:
        if not os.path.exists(names_file):
            with open(names_file, 'w') as f:
                json.dump(calspec_names_ff, f)
        else:
            with open(names_file, 'r') as f:
                calspec_names_ff = json.load(f)
    if star_name.lower() not in calspec_names_ff.keys():
        raise ValueError('{0} is not in list of anticipated standard stars \n {1},\n please check naming'.format(star_name, [*calspec_names_ff])) 
    fits_name = calspec_names_ff.get(star_name.lower())
    if fits_name in os.listdir(calspec_dir):
        file_name = os.path.join(calspec_dir, fits_name)
        name =  fits_name
    else:
        basic_name = fits_name.split('stis')[0]+'stis'
        #to be flexible with the version of the calspec fits file, so essentially, the number in the name should not matter
        req = Request(calspec_url)
        list = urlopen(req).readlines()
        for line in list:
            str_line = str(line)
            if basic_name in str_line: 
                name = str_line.split(".fits")[1].split(">")[-1] + ".fits"
                break

        fits_url = calspec_url + name
        try:
            file_name, headers = urlretrieve(fits_url, filename =  os.path.join(calspec_dir, name))
        except:
            raise Exception("cannot access CALSPEC archive web page and/or download {0}".format(name))
    return file_name, name

def get_filter_name(image):
    """
    return the name of the transmission curve csv file of used color filter
    
    Args:
        image (corgidrp.image): image of the observed calibration star
    
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

def read_cal_spec(calspec_file, filter_wavelength):
    """
    read the calspec flux density data interpolated on the wavelength grid of the transmission curve
    
    Args:
        calspec_file (str): file path to the CALSPEC fits file
        filter_wavelength (np.array): wavelength grid of the transmission curve in unit Angstroem
    
    Returns:
        np.array: flux density in Jy interpolated on the wavelength grid of the transmission curve 
        in CALSPEC units erg/(s * cm^2 * AA)
    """
    hdulist = fits.open(calspec_file)
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
        calspec_flux (np.array): converted flux in units of erg/(s*cm^2*AA) of the calspec source in the filter band
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
        calspec_flux (np.array): converted flux in units of the calspec source in the filter band
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
        calspec_flux (np.array): converted flux in units of the calspec source in the filter band
        wave_ref (float): reference wavelength in unit Angstroem
    
    Returns:
        float: flux at reference wavelength in unit erg/(s*cm^2*AA)
    """
    
    flux_ref = np.interp(wave_ref, filter_wavelength, calspec_flux)
    return flux_ref

def calculate_vega_mag(source_flux, filter_file):
    """
    determine the apparent Vega magnitude of the source with known flux in CALSPEC units erg/(s * cm^2 *AA)
    in the used filter band.
    
    Args:
        source_flux (float): source flux usually determined by applying the FluxcalFactor
        filter_file (str): name of the file with the transmission data of the corresponding color filter
    
    Returns:
        float: the apparent VEGA magnitude
    """
    
    wave, filter_trans = read_filter_curve(filter_file)
    # calculate the flux of VEGA and the source star from the user given CALSPEC file binned on the wavelength grid of the filter
    vega_filepath = get_calspec_file('Vega')[0]
    vega_sed = read_cal_spec(vega_filepath, wave)

    vega_flux = calculate_band_flux(filter_trans, vega_sed, wave)
    #calculate apparent vega magnitude
    vega_mag = -2.5 * np.log10(source_flux/vega_flux)
    
    return vega_mag


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
        calspec_flux (np.array): converted flux in units of erg/(s*cm^2*AA) of the calspec source in the filter band
        filter_wavelength (np.array): wavelengths in units Angstroem in the filter band 
    
    Returns:
        float: band irradiance of the calspec star in unit erg/(s*cm^2)
    """
    multi_flux = calspec_flux * filter_curve
    irrad = integrate.simpson(multi_flux, x=filter_wavelength)
    
    return irrad


def aper_phot(image, encircled_radius, frac_enc_energy=1., method='subpixel', subpixels=5,
              background_sub=False, r_in=5, r_out=10, centering_method='xy', centroid_roi_radius=5,
              centering_initial_guess=None):
    """
    Returns the flux in photo-electrons of a point source, either by placing an aperture using a 
        centroiding method, or by using WCS information.
    Background subtraction can be done optionally using a user defined circular annulus.
    
    Parameters:
        image (corgidrp.data.Image): combined source exposure image
        encircled_radius (float): pixel radius of the circular aperture to sum the flux
        frac_enc_energy (float): fraction of encircled energy inside the encircled_radius of the PSF, inverse aperture correction, 0...1
        method (str): {‘exact’, ‘center’, ‘subpixel’}, The method used to determine the overlap of the aperture on the pixel grid, 
        default is 'exact'. For detailed description see https://photutils.readthedocs.io/en/stable/api/photutils.aperture.CircularAnnulus.html
        subpixels (int): For the 'subpixel' method, resample pixels by this factor in each dimension. That is, each pixel is divided 
                         into subpixels**2 subpixels. This keyword is ignored unless method='subpixel', default is 5
        background_sub (boolean): background can be determine from a circular annulus (default: False).
        r_in (float): inner radius of circular annulus in pixels, (default: 5)
        r_out (float): outer radius of circular annulus in pixels, (default: 10)
        centering_method (str): 'xy' for centroiding or 'wcs' for WCS-based centering.
        centroid_roi_radius (int or float): Half-size of the box around the peak,
                                   in pixels. Adjust based on desired λ/D.
        centering_initial_guess (tuple): (Optional) (x,y) initial guess to perform centroiding.  
    
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
        x_center, y_center = centroid_with_roi(image.data, centroid_roi_radius, centering_initial_guess)
        pos = (x_center, y_center)
    else:
        raise ValueError("Invalid centering_method. Choose 'xy' or 'wcs'.")
    
    # Optionally subtract the background.
    if background_sub:
        #This is essentially the median in a circular annulus 
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
                        r_out=10, centering_method='xy', centroid_roi_radius=5,
                        centering_initial_guess=None):
    """
    Returns the flux in photo-electrons using a 2D Gaussian fit. Finds the star
        center either by placing an aperture using a centroiding method, or by using 
        WCS information.
    Allows optional background subtraction and selection of centering method.
    
    Parameters:
        image (corgidrp.data.Image): the source image.
        fwhm (float): Estimated full-width at half maximum.
        fit_shape (int or tuple, optional): Fitting region shape. If a scalar, then 
            it is assumed to be a square. If `None`, then the shape of the input 'data'.
            It must be an odd value and should be much bigger than fwhm.
        background_sub (bool): If True, subtract background.
        r_in (float): Inner annulus radius.
        r_out (float): Outer annulus radius.
        centering_method (str): 'xy' or 'wcs' centering.
        centroid_roi_radius (int or float): Half-size of the box around the peak,
                                   in pixels. Adjust based on desired λ/D.
        centering_initial_guess (tuple): (Optional) (x,y) initial guess to perform centroiding.
    
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
        x_center, y_center = centroid_with_roi(image.data, centroid_roi_radius, centering_initial_guess)
        pos = (x_center, y_center)
    else:
        raise ValueError("Invalid centering_method. Choose 'xy' or 'wcs'.")

    # Work on a copy so that background subtraction does not alter original data.
    dat = image.data.copy()

    if background_sub:
        bkg = LocalBackground(r_in, r_out)
        back = bkg(dat, pos[0], pos[1], mask=image.dq.astype(bool))
        dat -= back

    # fit_2dgaussian: error weighting raises exception if error is zero
    err = image.err[0].copy()
    err[err == 0] = np.finfo(np.float32).eps

    if fit_shape is None:
        fit_shape = image.data.shape[0] - 1

    psf_phot = fit_2dgaussian(dat, xypos=pos, fwhm=fwhm, fit_shape=fit_shape,
                              mask=image.dq.astype(bool), error=err)
    flux = psf_phot.results['flux_fit'][0]
    flux_err = psf_phot.results['flux_err'][0]

    if background_sub:
        return [flux, flux_err, back]
    else:
        return [flux, flux_err]


def calibrate_fluxcal_aper(dataset_or_image, calspec_file = None, flux_or_irr = 'flux', phot_kwargs=None):
    """
    fills the FluxcalFactors calibration product values for one filter band,
    calculates the flux calibration factors by aperture photometry.
    The band flux values are divided by the found photoelectrons/s.
    Propagates also errors to flux calibration factor calfile.
    Background subtraction can be done optionally using a user defined circular annulus.
    
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
        'centroid_roi_radius' (int or float): Half-size of the box around the peak,
                                   in pixels. Adjust based on desired λ/D.
        'centering_initial_guess' (tuple): (Optional) (x,y) initial guess to perform centroiding.
    
    Parameters:
        dataset_or_image (corgidrp.data.Dataset or corgidrp.data.Image): Image(s) to compute 
            the calibration factor. Should already be normalized for exposure time.
        calspec_file (str, optional): file path to the calspec fits file of the observed star
        flux_or_irr (str, optional): Whether flux ('flux') or in-band irradiance ('irr) should 
            be used.
        phot_kwargs (dict, optional): A dictionary of keyword arguments controlling the aperture 
            photometry function.

    Returns:
        FluxcalFactor (corgidrp.data.FluxcalFactor): A calibration object containing the computed 
            flux calibration factor in (TO DO: what units should this be in)
    """
    d_or_i = dataset_or_image.copy()
    if isinstance(d_or_i, corgidrp.data.Dataset):
        image = d_or_i[0]
        dataset = d_or_i
    else:
        image = d_or_i
        dataset = corgidrp.data.Dataset([image])
    if image.ext_hdr['BUNIT'] != "photoelectron/s":
        raise ValueError("input dataset must have unit photoelectron/s for the calibration, not {0}".format(image.ext_hdr['BUNIT']))
    if phot_kwargs is None:
        phot_kwargs = {
            'encircled_radius': 5,
            'frac_enc_energy': 1.0,
            'method': 'subpixel',
            'subpixels': 5,
            'background_sub': False,
            'r_in': 5,
            'r_out': 10,
            'centering_method': 'xy',
            'centroid_roi_radius': 5
        }
    
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
    
    if flux_or_irr == 'flux':
        flux = calculate_band_flux(filter_trans, flux_ref, wave)
        image.ext_hdr['BUNIT'] = 'erg/(s * cm^2 * AA)/(photoelectron/s)'
        image.err_hdr['BUNIT'] = 'erg/(s * cm^2 * AA)/(photoelectron/s)'
    elif flux_or_irr == 'irr':
        flux = calculate_band_irradiance(filter_trans, flux_ref, wave)
        image.ext_hdr['BUNIT'] = 'erg/(s * cm^2)/(photoelectron/s)'
        image.err_hdr['BUNIT'] = 'erg/(s * cm^2)/(photoelectron/s)'
    else:
        raise ValueError("Invalid flux method. Choose 'flux' or 'irr'.")
    
    result = aper_phot(image, **phot_kwargs)
    if phot_kwargs.get('background_sub', False):
        ap_sum, ap_sum_err, back = result
    else:
        ap_sum, ap_sum_err = result

    fluxcal_fac = flux / ap_sum
    fluxcal_fac_err = flux / ap_sum**2 * ap_sum_err

    fluxcal_obj = corgidrp.data.FluxcalFactor(
        fluxcal_fac,
        err=fluxcal_fac_err,
        pri_hdr=image.pri_hdr,
        ext_hdr=image.ext_hdr,
        input_dataset=dataset
    )

    # If background subtraction was performed, set the LOCBACK keyword.
    if phot_kwargs.get('background_sub', False):
        # Here, "back" is the third value returned from phot_by_gauss2d_fit.
        fluxcal_obj.ext_hdr['LOCBACK'] = back

    # Append to or create a HISTORY entry in the header.
    history_entry = "Flux calibration factor was determined by aperture photometry using SED file {0}".format(calspec_filename)
    fluxcal_obj.ext_hdr.add_history(history_entry)

    return fluxcal_obj

def calibrate_pol_fluxcal_aper(dataset_or_image,
                               image_center_x=512,
                               image_center_y=512,
                               separation_diameter_arcsec=7.5, 
                               alignment_angle=None,
                               calspec_file = None,
                               flux_or_irr = 'flux',
                               phot_kwargs=None):
    """
    Same overall process as calibrate_fluxcal_aper, adapted for polarimetric images 
    from WP1 or WP2 with two apertures instead of one.

    fills the FluxcalFactors calibration product values for one filter band,
    calculates the flux calibration factors by aperture photometry.
    The band flux values are divided by the found photoelectrons/s.
    Propagates also errors to flux calibration factor calfile.
    Background subtraction can be done optionally using a user defined circular annulus.
    
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
        'centroid_roi_radius' (int or float): Half-size of the box around the peak,
                                   in pixels. Adjust based on desired λ/D.
    
    Parameters:
        dataset_or_image (corgidrp.data.Dataset or corgidrp.data.Image): Image(s) to compute 
            the calibration factor. Should already be normalized for exposure time. Images must
            be from polarimetric observations taken with WP1 or WP2 in the DPAM.
        image_center_x (int): X pixel coordinate of where the two wollaston spots are 
            centered around
        image_center_y (int): Y pixel coordinate of where the two wollaston spots are 
            centered around
        separation_diameter_arcsec (optional, float): Distance between the centers of the two polarized images on the detector in arcsec, 
            default for Roman CGI is 7.5"
        alignment_angle (optional, float): the angle in degrees of how the two polarized images are aligned with respect to the horizontal,
            defaults to 0 for WP1 and 45 for WP2
        calspec_file (str, optional): file path to the calspec fits file of the observed star
        flux_or_irr (str, optional): Whether flux ('flux') or in-band irradiance ('irr) should 
            be used.
        phot_kwargs (dict, optional): A dictionary of keyword arguments controlling the aperture 
            photometry function.

    Returns:
        FluxcalFactor (corgidrp.data.FluxcalFactor): A calibration object containing the computed 
            flux calibration factor in (TO DO: what units should this be in)
    """
    d_or_i = dataset_or_image.copy()
    if isinstance(d_or_i, corgidrp.data.Dataset):
        image = d_or_i[0]
        dataset = d_or_i
    else:
        image = d_or_i
        dataset = corgidrp.data.Dataset([image])
    if image.ext_hdr['BUNIT'] != "photoelectron/s":
        raise ValueError("input dataset must have unit photoelectron/s for the calibration, not {0}".format(image.ext_hdr['BUNIT']))
    #estimate the centers of the wollaston spots based on relative position from image center
    #polarized images separated 7.5" or 344 pix on the detector by default (1"=0.0218 pix)
    #WP1 output is aligned horizontally across the image center by default
    #WP2 output is algined diagonally across the image center by default
    image_center = (image_center_x, image_center_y)
    dpamname = image.ext_hdr['DPAMNAME']
    if dpamname not in ['POL0', 'POL45']:
        raise ValueError('input dataset must be a polarimetric observation')
    if alignment_angle is None:
        if dpamname == 'POL0':
            alignment_angle = 0
        else:
            alignment_angle = 45
    angle_rad = alignment_angle * (np.pi / 180)
    displacement_x = int(round((separation_diameter_arcsec * np.cos(angle_rad)) / (2 * 0.0218)))
    displacement_y = int(round((separation_diameter_arcsec * np.sin(angle_rad)) / (2 * 0.0218)))
    #estimate where the centers are based on alignment angle and separation
    centering_initial_guess_beam_1 = (image_center[0] - displacement_x, image_center[1] + displacement_y)
    centering_initial_guess_beam_2 = (image_center[0] + displacement_x, image_center[1] - displacement_y)

    #ensure xy centering method is used with estimated centers for aperture photometry
    if phot_kwargs is None:
        phot_kwargs = {
            'encircled_radius': 5,
            'frac_enc_energy': 1.0,
            'method': 'subpixel',
            'subpixels': 5,
            'background_sub': False,
            'r_in': 5,
            'r_out': 10,
            'centroid_roi_radius': 5,
        }
    #update parameters to ensure centering is performed correctly
    phot_kwargs_beam_1 = phot_kwargs.copy()
    phot_kwargs_beam_2 = phot_kwargs.copy()
    phot_kwargs_beam_1.update({
        'centering_method': 'xy',
        'centering_initial_guess': centering_initial_guess_beam_1
    })
    phot_kwargs_beam_2.update({
        'centering_method': 'xy',
        'centering_initial_guess': centering_initial_guess_beam_2
    })
    
    result_beam_1, result_beam_2 = measure_aper_flux_pol(
        image,
        image_center_x=image_center_x,
        image_center_y=image_center_y,
        separation_diameter_arcsec=separation_diameter_arcsec,
        alignment_angle=alignment_angle,
        phot_kwargs=phot_kwargs
    )
    
    #Optionally subtract a local background 
    if phot_kwargs.get('background_sub', False):
        ap_sum_beam_1, ap_sum_err_beam_1, back_beam_1 = result_beam_1
        ap_sum_beam_2, ap_sum_err_beam_2, back_beam_2 = result_beam_2
    else:
        ap_sum_beam_1, ap_sum_err_beam_1 = result_beam_1
        ap_sum_beam_2, ap_sum_err_beam_2 = result_beam_2

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

    if flux_or_irr == 'flux':
        flux = calculate_band_flux(filter_trans, flux_ref, wave)
        image.ext_hdr['BUNIT'] = 'erg/(s * cm^2 * AA)/(photoelectron/s)'
        image.err_hdr['BUNIT'] = 'erg/(s * cm^2 * AA)/(photoelectron/s)'
    elif flux_or_irr == 'irr':
        flux = calculate_band_irradiance(filter_trans, flux_ref, wave)
        image.ext_hdr['BUNIT'] = 'erg/(s * cm^2)/(photoelectron/s)'
        image.err_hdr['BUNIT'] = 'erg/(s * cm^2)/(photoelectron/s)'
    else:
        raise ValueError("Invalid flux method. Choose 'flux' or 'irr'.")

    fluxcal_fac = flux / (ap_sum_beam_1 + ap_sum_beam_2)
    fluxcal_fac_err = flux / (ap_sum_beam_1 + ap_sum_beam_2)**2 * np.sqrt((ap_sum_err_beam_1**2) + (ap_sum_err_beam_2**2))

    fluxcal_obj = corgidrp.data.FluxcalFactor(
        fluxcal_fac,
        err=fluxcal_fac_err,
        pri_hdr=image.pri_hdr,
        ext_hdr=image.ext_hdr,
        input_dataset=dataset
    )

    # If local background subtraction was performed, set the LOCBACK keyword.
    if phot_kwargs.get('background_sub', False):
        #average medium background photoelectrons from both beams
        back = 0.5 * (back_beam_1 + back_beam_2)
        # Here, "back" is the third value returned from phot_by_gauss2d_fit.
        fluxcal_obj.ext_hdr['LOCBACK'] = back

    # Append to or create a HISTORY entry in the header.
    history_entry = "Flux calibration factor was determined by aperture photometry using SED file {0}".format(calspec_filename)
    fluxcal_obj.ext_hdr.add_history(history_entry)

    return fluxcal_obj

def measure_aper_flux_pol(image,
                          image_center_x=512,
                          image_center_y=512,
                          separation_diameter_arcsec=7.5, 
                          alignment_angle=None,
                          phot_kwargs=None,
                          pixel_scale = 0.0218):
    """
    Measure the individual flux from both polarized images in a polarimetric observation
    using aperture photometry.

    Parameters:
        image (corgidrp.data.Image): the source image.
        image_center_x (int): X pixel coordinate of where the two wollaston spots are centered around
        image_center_y (int): Y pixel coordinate of where the two wollaston spots are centered around
        separation_diameter_arcsec (optional, float): Distance between the centers of the two polarized images on the detector in arcsec,
            default for Roman CGI is 7.5"
        alignment_angle (optional, float): the angle in degrees of how the two polarized images are aligned with respect to the horizontal,
            defaults to 0 for WP1 and 45 for WP2
        phot_kwargs (dict, optional): A dictionary of keyword arguments controlling the aperture
            photometry function. See calibrate_pol_fluxcal_aper for accepted keywords.
        pixel_scale (float): pixel scale in arcsec/pixel, default for Roman CGI is 0.0218"/pix

    Returns:
        tuple: (result_beam_1, result_beam_2) where each result is either (flux, flux_err) or (flux, flux_err, back)
    """

    if image.ext_hdr['BUNIT'] != "photoelectron/s":
        raise ValueError("input dataset must have unit photoelectron/s, not {0}".format(image.ext_hdr['BUNIT']))
    #estimate the centers of the wollaston spots based on relative position from image center
    #polarized images separated 7.5" or 344 pix on the detector by default (1"=0.0218 pix)
    #WP1 output is aligned horizontally across the image center by default
    #WP2 output is algined diagonally across the image center by default
    image_center = (image_center_x, image_center_y)
    dpamname = image.ext_hdr['DPAMNAME']
    if dpamname not in ['POL0', 'POL45']:
        raise ValueError('input dataset must be a polarimetric observation')
    if alignment_angle is None:
        if dpamname == 'POL0':
            alignment_angle = 0
        else:
            alignment_angle = 45
    angle_rad = alignment_angle * (np.pi / 180)

    displacement_x = int(round((separation_diameter_arcsec * np.cos(angle_rad)) / (2 * pixel_scale)))
    displacement_y = int(round((separation_diameter_arcsec * np.sin(angle_rad)) / (2 * pixel_scale)))
    #estimate where the centers are based on alignment angle and separation
    centering_initial_guess_beam_1 = (image_center[0] - displacement_x, image_center[1] + displacement_y)
    centering_initial_guess_beam_2 = (image_center[0] + displacement_x, image_center[1] - displacement_y)

    #ensure xy centering method is used with estimated centers for aperture photometry
    if phot_kwargs is None:
        phot_kwargs = {
            'encircled_radius': 5,
            'frac_enc_energy': 1.0,
            'method': 'subpixel',
            'subpixels': 5,
            'background_sub': False,
            'r_in': 5,
            'r_out': 10,
            'centroid_roi_radius': 5,
        }
    #update parameters to ensure centering is performed correctly
    phot_kwargs_beam_1 = phot_kwargs.copy()
    phot_kwargs_beam_2 = phot_kwargs.copy()
    phot_kwargs_beam_1.update({
        'centering_method': 'xy',
        'centering_initial_guess': centering_initial_guess_beam_1
    })
    phot_kwargs_beam_2.update({
        'centering_method': 'xy',
        'centering_initial_guess': centering_initial_guess_beam_2
    })
    
    #calculate flux from both apertures
    result_beam_1 = aper_phot(image, **phot_kwargs_beam_1)
    result_beam_2 = aper_phot(image, **phot_kwargs_beam_2)
    
    return result_beam_1, result_beam_2



def calibrate_fluxcal_gauss2d(dataset_or_image, calspec_file = None, flux_or_irr = 'flux', phot_kwargs=None):
    """
    fills the FluxcalFactors calibration product values for one filter band,
    calculates the flux calibration factors by fitting a 2D Gaussian.
    The band flux values are divided by the found photoelectrons/s.
    Propagates also errors to flux calibration factor calfile.
    Background subtraction can be done optionally using a user defined circular annulus.
    
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
        'centroid_roi_radius' (int or float): Half-size of the box around the peak,
            in pixels. Adjust based on desired λ/D.
        'centering_initial_guess' (tuple): (Optional) (x,y) initial guess to perform centroiding.

    Parameters:
        dataset_or_image (corgidrp.data.Dataset or corgidrp.data.Image): Image(s) to compute 
            the calibration factor. Should already be normalized for exposure time.
        calspec_file (str, optional): file path to the calspec fits file of the observed star
        flux_or_irr (str, optional): Whether flux ('flux') or in-band irradiance ('irr) should 
            be used.
        phot_kwargs (dict, optional): A dictionary of keyword arguments controlling the Gaussian 
            photometry function.
                
    Returns:
        FluxcalFactor (corgidrp.data.FluxcalFactor): A calibration object containing the computed 
            flux calibration factor in (TO DO: what units should this be in)
    """
    d_or_i = dataset_or_image.copy()
    if isinstance(d_or_i, corgidrp.data.Dataset):
        image = d_or_i[0]
        dataset = d_or_i
    else:
        image = d_or_i
        dataset = corgidrp.data.Dataset([image])
    if image.ext_hdr['BUNIT'] != "photoelectron/s":
        raise ValueError("input dataset must have unit photoelectron/s for the calibration, not {0}".format(image.ext_hdr['BUNIT']))
    if phot_kwargs is None:
        phot_kwargs = {
        'fwhm': 3,
        'fit_shape': None,
        'background_sub': False,
        'r_in': 5,
        'r_out': 10,
        'centering_method': 'xy',
        'centroid_roi_radius': 5
    }

    filter_file = get_filter_name(image)
    wave, filter_trans = read_filter_curve(filter_file)
    
    if calspec_file is not None:
        calspec_filepath = calspec_file
        calspec_filename = os.path.basename(calspec_file)
    else:
        star_name = image.pri_hdr["TARGET"]
        calspec_filepath, calspec_filename = get_calspec_file(star_name)
    flux_ref = read_cal_spec(calspec_filepath, wave)
    
    if flux_or_irr == 'flux':
        flux = calculate_band_flux(filter_trans, flux_ref, wave)
        image.ext_hdr['BUNIT'] = 'erg/(s * cm^2 * AA)/(photoelectron/s)'
        image.err_hdr['BUNIT'] = 'erg/(s * cm^2 * AA)/(photoelectron/s)'
    elif flux_or_irr == 'irr':
        flux = calculate_band_irradiance(filter_trans, flux_ref, wave)
        image.ext_hdr['BUNIT'] = 'erg/(s * cm^2)/(photoelectron/s)'
        image.err_hdr['BUNIT'] = 'erg/(s * cm^2)/(photoelectron/s)'
    else:
        raise ValueError("Invalid flux method. Choose 'flux' or 'irr'.")
    
    if phot_kwargs.get('background_sub', False):
        gauss_sum, gauss_sum_err, back = phot_by_gauss2d_fit(image, **phot_kwargs)
    else:
        gauss_sum, gauss_sum_err = phot_by_gauss2d_fit(image, **phot_kwargs)
    
    fluxcal_fac = flux / gauss_sum
    fluxcal_fac_err = flux / gauss_sum**2 * gauss_sum_err

    fluxcal_obj = corgidrp.data.FluxcalFactor(
        fluxcal_fac,
        err=fluxcal_fac_err,
        pri_hdr=image.pri_hdr,
        ext_hdr=image.ext_hdr,
        input_dataset=dataset
    )
    
    # If background subtraction was performed, set the LOCBACK keyword.
    if phot_kwargs.get('background_sub', False):
        # Here, "back" is the third value returned from phot_by_gauss2d_fit.
        fluxcal_obj.ext_hdr['LOCBACK'] = back

    # Append to or create a HISTORY entry in the header.
    history_entry = "Flux calibration factor was determined by a Gaussian 2D fit photometry using SED file {0}".format(calspec_filename)
    fluxcal_obj.ext_hdr.add_history(history_entry)
    
    return fluxcal_obj
