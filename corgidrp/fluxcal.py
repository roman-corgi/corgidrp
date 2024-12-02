# This module is written to do an absolute flux calibration observing a standard star having CALSPEC data.
import glob
import os
import numpy as np
from astropy.io import fits, ascii
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
        calspec_dir = os.path.join(os.path.dirname(__file__), "data", "calspec_data")
        if not os.path.exists(calspec_dir):
            os.mkdir(calspec_dir)
        file_name, headers = urllib.request.urlretrieve(fits_url, filename =  os.path.join(calspec_dir, fits_name))
    except:
        raise Exception("cannot access CALSPEC archive web page and/or download {0}".fits_name)
    return file_name

def get_filter_name(dataset):
    """
    return the name of the transmission curve csv file of used color filter
    
    Args:
        dataset (corgidrp.Dataset): dataset of the observed calstar
    
    Returns:
        str: filepath of the selected filter curve
    """
    datadir = os.path.join(os.path.dirname(__file__), "data", "filter_curves")
    filters = os.path.join(datadir, "*.csv")
    filter = dataset[0].ext_hdr['CFAMNAME']
    filter_names = os.listdir(datadir)
    f_avail = False
    for name in filter_names:
        if filter in name:
            f_avail = True
            filter_name= os.path.join(datadir, name)
            break
        else:
            pass
    if f_avail:
        return filter_name
    else:
        raise Exception("there is no filter available with name {0}".format(filter))

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
