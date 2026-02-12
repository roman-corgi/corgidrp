"""
Module to hold input-checking functions to minimize repetition

Copied over from the II&T pipeline
"""
import datetime
import numbers
import numpy as np
import logging
import os
import glob
import warnings
import astropy.io.fits as fits
import pandas as pd
from corgidrp import mocks

# Set up module-level logger
logger = logging.getLogger(__name__)

class CheckException(Exception):
    pass

# String check support
string_types = (str, bytes)

# Int check support
int_types = (int, np.integer)

def _checkname(vname):
    """
    Internal check that we can use vname as a string for printing

    Args:
    vname: str
        variable name
    """
    if not isinstance(vname, string_types):
        raise CheckException('vname must be a string when fed to check ' + \
                             'functions')
    pass


def _checkexc(vexc):
    """
    Internal check that we can raise from the vexc object

    Args:
    vexc: str
        exception name
    """
    if not isinstance(vexc, type): # pre-check it is class-like
        raise CheckException('vexc must be a Exception, or an object ' + \
                             'descended from one when fed to check functions')
    if not issubclass(vexc, Exception):
        raise CheckException('vexc must be a Exception, or an object ' + \
                             'descended from one when fed to check functions')
    pass


def real_positive_scalar(var, vname, vexc):
    """
    Checks whether an object is a real positive scalar.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be real')
    if var <= 0:
        raise vexc(vname + ' must be positive')
    return var


def real_array(var, vname, vexc):
    """
    Checks whether an object is a real numpy array, or castable to one.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.asarray(var) # cast to array
    if len(var.shape) == 0:
        raise vexc(vname + ' must have length > 0')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be a real array')
    # skip 'c' as we don't want complex; rest are non-numeric
    if not var.dtype.kind in ['b', 'i', 'u', 'f']:
        raise vexc(vname + ' must be a real numeric type to be real')
    return var


def oneD_array(var, vname, vexc):
    """
    Checks whether an object is a 1D numpy array, or castable to one.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.asarray(var) # cast to array
    if len(var.shape) != 1:
        raise vexc(vname + ' must be a 1D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 1D array')
    return var


def twoD_array(var, vname, vexc):
    """
    Checks whether an object is a 2D numpy array, or castable to one.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.asarray(var) # cast to array
    if len(var.shape) != 2:
        raise vexc(vname + ' must be a 2D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 2D array')
    return var


def twoD_square_array(var, vname, vexc):
    """
    Checks whether an object is a 2D square array_like.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.asarray(var) # cast to array
    if len(var.shape) != 2:
        raise vexc(vname + ' must be a 2D array')
    else: # is 2-D
        if not var.shape[0] == var.shape[1]:
            raise vexc(vname + ' must be a square 2D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex square 2D array')
    return var


def threeD_array(var, vname, vexc):
    """
    Checks whether an object is a 3D numpy array, or castable to one.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.asarray(var) # cast to array
    if len(var.shape) != 3:
        raise vexc(vname + ' must be a 3D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 3D array')
    return var



def real_scalar(var, vname, vexc):
    """
    Checks whether an object is a real scalar.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be real')
    return var


def real_nonnegative_scalar(var, vname, vexc):
    """
    Checks whether an object is a real nonnegative scalar.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be real')
    if var < 0:
        raise vexc(vname + ' must be nonnegative')
    return var


def positive_scalar_integer(var, vname, vexc):
    """
    Checks whether an object is a positive scalar integer.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not isinstance(var, int_types):
        raise vexc(vname + ' must be integer')
    if var <= 0:
        raise vexc(vname + ' must be positive')
    return var


def nonnegative_scalar_integer(var, vname, vexc):
    """
    Checks whether an object is a nonnegative scalar integer.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not isinstance(var, int_types):
        raise vexc(vname + ' must be integer')
    if var < 0:
        raise vexc(vname + ' must be nonnegative')
    return var


def scalar_integer(var, vname, vexc):
    """
    Checks whether an object is a scalar integer (no sign dependence).

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not isinstance(var, int_types):
        raise vexc(vname + ' must be integer')
    return var


def string(var, vname, vexc):
    """
    Checks whether an object is a string.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, string_types):
        raise vexc(vname + ' must be a string')
    return var


def boolean(var, vname, vexc):
    """
    Checks whether an object is a bool.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, bool):
        raise vexc(vname + ' must be a bool')
    return var


def dictionary(var, vname, vexc):
    """
    Checks whether an object is a dictionary.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, dict):
        raise vexc(vname + ' must be a dict')
    return var

# ================================================================================
# Validation functions for E2E tests
# ================================================================================

def check_filename_convention(filename, expected_pattern, frame_info="",
     logger=None, data_level="l2b"):
    """Check if filename follows the expected naming convention.

    Args:
        filename (str): Filename to check
        expected_pattern (str): Expected pattern (e.g., 'cgi_*_l2b.fits')
        frame_info (str): Additional info for logging (e.g., "Frame 0")
        logger: Logger instance to use. If None, uses module logger.
        data_level (str): data level to be considered. 

    Returns:
        bool: True if filename matches convention
    """
    if logger is None:
        logger = globals()['logger']
    
    if not filename:
        logger.info(f"{frame_info}: No filename. Naming convention FAIL.")
        return False
    
    # Basic pattern check
    if expected_pattern == f'cgi_*_{data_level}.fits':
        parts = filename.split('_')
        valid = (len(parts) >= 4 and 
                parts[0] == 'cgi' and 
                len(parts[2]) == 16 and parts[2][8] == 't' and 
                parts[2][:8].isdigit() and parts[2][9:].isdigit() and
                filename.endswith(f'_{data_level}.fits'))
    elif expected_pattern == 'cgi_*_dpm_cal.fits':
        valid = filename.startswith('cgi_') and '_dpm_cal.fits' in filename
    elif expected_pattern == 'cgi_*_lsf_cal.fits':
        valid = filename.startswith('cgi_') and '_lsf_cal.fits' in filename
    elif expected_pattern == 'cgi_*_mmx_cal.fits':
        valid = filename.startswith('cgi_') and '_mmx_cal.fits' in filename
    elif expected_pattern == 'cgi_*_ndm_cal.fits':
        valid = filename.startswith('cgi_') and '_ndm_cal.fits' in filename
    else:
        valid = expected_pattern in filename
    
    status = "PASS" if valid else "FAIL"
    logger.info(f"{frame_info}: Filename: {filename}. Naming convention {status}.")
    return valid

def check_dimensions(data, expected_shape, frame_info="", logger=None):
    """Check if data has expected dimensions.

    Args:
        data (numpy.ndarray): Data array to check
        expected_shape (tuple): Expected shape tuple
        frame_info (str): Additional info for logging (e.g., "Frame 0")
        logger: Logger instance to use. If None, uses module logger.

    Returns:
        bool: True if dimensions match
    """
    if logger is None:
        logger = globals()['logger']
    
    if data.shape == expected_shape:
        logger.info(f"{frame_info}: Shape={data.shape}. Expected: {expected_shape}. PASS.")
        return True
    else:
        logger.info(f"{frame_info}: Shape={data.shape}. Expected: {expected_shape}. FAIL.")
        return False

def verify_hdu_count(hdul, expected_count, frame_info="", logger=None):
    """Verify that the number of HDUs in the FITS file is as expected.

    Args:
        hdul (astropy.io.fits.HDUList): FITS HDUList object
        expected_count (int): Expected number of HDUs
        frame_info (str): Additional info for logging (e.g., "Frame 0")
        logger: Logger instance to use. If None, uses module logger.

    Returns:
        bool: True if HDU count matches expected count
    """
    if logger is None:
        logger = globals()['logger']
    
    actual_count = len(hdul)
    if actual_count == expected_count:
        logger.info(f"{frame_info}: HDU count={actual_count}. Expected: {expected_count}. PASS.")
        return True
    else:
        logger.info(f"{frame_info}: HDU count={actual_count}. Expected: {expected_count}. FAIL.")
        return False

def verify_header_keywords(header, required_keywords, frame_info="", logger=None):
    """Verify that required header keywords are present and have expected values.

    Args:
        header (astropy.io.fits.Header): FITS header object
        required_keywords (dict or list): Dictionary of {keyword: expected_value} or list of keywords
        frame_info (str): Additional info for logging (e.g., "Frame 0")
        logger: Logger instance to use. If None, uses module logger.

    Returns:
        bool: True if all keywords are valid
    """
    if logger is None:
        logger = globals()['logger']
    
    all_valid = True
    
    if isinstance(required_keywords, dict):
        # Check keyword-value pairs
        for keyword, expected_value in required_keywords.items():
            if keyword not in header:
                logger.error(f"{frame_info}: Missing required keyword {keyword}!")
                all_valid = False
            else:
                actual_value = header[keyword]
                if actual_value == expected_value:
                    logger.info(f"{frame_info}: {keyword}={actual_value}. Expected {keyword}: {expected_value}. PASS.")
                else:
                    logger.info(f"{frame_info}: {keyword}={actual_value}. Expected {keyword}: {expected_value}. FAIL.")
                    all_valid = False
    else:
        # Just check if keywords exist
        for keyword in required_keywords:
            if keyword not in header:
                logger.error(f"{frame_info}: Missing required keyword {keyword}!")
                all_valid = False
            else:
                logger.info(f"{frame_info}: {keyword}={header[keyword]},")
    
    return all_valid

def validate_binary_table_fields(hdu1, required_fields, logger=None):
    """Validate binary table fields with consistent error reporting.

    Args:
        hdu1 (astropy.io.fits.BinTableHDU): FITS binary table HDU
        required_fields (list): List of required field names
        logger: Logger instance to use. If None, uses module logger.

    Returns:
        bool: True if all fields are valid
    """
    if logger is None:
        logger = globals()['logger']
    
    if isinstance(hdu1, fits.BinTableHDU):
        logger.info("HDU1: Binary table format. Expected: BinTableHDU. PASS.")
        
        for field in required_fields:
            if field in hdu1.data.names:
                data = hdu1.data[field]
                # Check if dtype is 64-bit float (ignoring endianness)
                is_float64 = (data.dtype.kind == 'f' and data.dtype.itemsize == 8)
                status = "PASS" if is_float64 else "FAIL"
                logger.info(f"HDU1: Table field '{field}' present. Data type {data.dtype}. Expected: 64-bit float. {status}.")
                
                # Additional shape validation for polynomial coefficients
                if field == 'pos_vs_wavlen_polycoeff':
                    expected_shape = (1, 4)
                    shape_status = "PASS" if data.shape == expected_shape else "FAIL"
                    logger.info(f"HDU1: {field} shape {data.shape}. Expected: {expected_shape}. {shape_status}.")
            else:
                logger.info(f"HDU1: Field '{field}' missing. Expected: field present. FAIL.")
        return True
    else:
        logger.info(f"HDU1: Format {type(hdu1)}. Expected: BinTableHDU. FAIL.")
        # Report all field failures when format is wrong
        for field in required_fields:
            logger.info(f"HDU1: Field '{field}' missing. Expected: field present. FAIL.")
            if field == 'pos_vs_wavlen_polycoeff':
                logger.info(f"HDU1: Field '{field}' missing. Expected: 64-bit float. FAIL.")
                logger.info(f"HDU1: Field '{field}' missing. Expected: 1×4 array. FAIL.")
        return False

def get_latest_cal_file(e2eoutput_path, pattern, logger=None):
    """Get the most recent calibration file matching the pattern.

    Args:
        e2eoutput_path (str): Directory to search for calibration files
        pattern (str): Pattern to match (e.g., '*_dpm_cal.fits')
        logger: Logger instance to use. If None, uses module logger.

    Returns:
        str: Path to the most recent calibration file
    """
    if logger is None:
        logger = globals()['logger']
    
    cal_files = sorted(glob.glob(os.path.join(e2eoutput_path, pattern)), key=os.path.getmtime, reverse=True)
    assert len(cal_files) > 0, f'No {pattern} files found in {e2eoutput_path}!'
    return cal_files[0]

def generate_fits_excel_documentation(fits_filepath, output_excel_path):
    """
    Generate an Excel file documenting the structure and headers of a FITS file.
    
    Args:
        fits_filepath (str): Path to the FITS file to document
        output_excel_path (str): Path where the Excel file should be saved
        
    Returns:
        str: Path to the generated Excel file
        
    Raises:
        ImportError: If pandas is not available
        FileNotFoundError: If the FITS file doesn't exist
    """
    
    if not os.path.exists(fits_filepath):
        raise FileNotFoundError(f"FITS file not found: {fits_filepath}")
    
    # Load keyword descriptions from RST documentation files if available
    keyword_descriptions = {}
    try:
        import re
        current_dir = os.path.dirname(os.path.abspath(__file__))
        docs_dir = os.path.join(current_dir, '..', 'docs', 'source', 'data_formats')
        
        if os.path.exists(docs_dir):
            # Read all RST files and extract keyword descriptions
            for rst_file in os.listdir(docs_dir):
                if rst_file.endswith('.rst') and rst_file != 'index.rst':
                    rst_path = os.path.join(docs_dir, rst_file)
                    with open(rst_path, 'r') as f:
                        for line in f:
                            # Match table rows: | KEYWORD | datatype | value | description |
                            match = re.match(r'^\|\s+([A-Z0-9_]+)\s+\|\s+\S+\s+\|\s+.+?\s+\|\s+(.+?)\s+\|$', line)
                            if match:
                                keyword = match.group(1)
                                description = match.group(2).strip()
                                # Only store if we have a real description (not empty, not just structural info)
                                if description and description not in ['0', '1', '2', '3', '4']:
                                    # Prefer longer descriptions (keep the most complete one)
                                    if keyword not in keyword_descriptions or len(description) > len(keyword_descriptions[keyword]):
                                        keyword_descriptions[keyword] = description
    except Exception as e:
        # If we can't load RST files, just continue without reference descriptions
        pass
    
    # Open the FITS file
    with fits.open(fits_filepath) as hdulist:
        
        # Create Excel writer
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            
            # Sheet 1: Extensions Overview
            extensions_data = []
            for i, hdu in enumerate(hdulist):
                # Determine extension name
                if i == 0:
                    ext_name = "Primary"
                    description = "Primary header (no data)"
                elif hasattr(hdu, 'name') and hdu.name:
                    ext_name = hdu.name
                else:
                    ext_name = f"Extension {i}"
                
                # Determine description based on extension type
                if i == 0:
                    description = "Primary header (no data)"
                elif hasattr(hdu, 'data') and hdu.data is not None:
                    if len(hdu.data.shape) == 0:
                        description = "Scalar data"
                    elif len(hdu.data.shape) == 1:
                        description = "1D array data"
                    elif len(hdu.data.shape) == 2:
                        description = "2D image data"
                    elif len(hdu.data.shape) == 3:
                        description = "3D data cube"
                    else:
                        description = f"{len(hdu.data.shape)}D data array"
                else:
                    description = "Header only"
                
                # Get data type
                if hasattr(hdu, 'data') and hdu.data is not None:
                    datatype = str(hdu.data.dtype)
                else:
                    datatype = "None"
                
                # Get array size
                if hasattr(hdu, 'data') and hdu.data is not None:
                    array_size = str(hdu.data.shape)
                else:
                    array_size = "0"
                
                extensions_data.append({
                    'Index': i,
                    'Extension Name': ext_name,
                    'Description': description,
                    'Data Type': datatype,
                    'Array Size': array_size
                })
            
            # Create DataFrame and save to first sheet
            extensions_df = pd.DataFrame(extensions_data)
            extensions_df.to_excel(writer, sheet_name='Extensions_Overview', index=False)
            
            # Sheets 2+: Header keywords for each extension
            for i, hdu in enumerate(hdulist):
                # Determine sheet name
                if i == 0:
                    sheet_name = "Primary_Header"
                elif hasattr(hdu, 'name') and hdu.name:
                    sheet_name = f"{hdu.name}_Header"
                else:
                    sheet_name = f"Extension_{i}_Header"
                
                # Ensure sheet name is valid (Excel limits)
                sheet_name = sheet_name[:31]  # Excel sheet name limit
                
                # Extract header information
                header_data = []
                for keyword in hdu.header:
                    value = hdu.header[keyword]
                    fits_comment = hdu.header.comments[keyword]
                    
                    # Use RST description if available, otherwise use FITS comment
                    if keyword in keyword_descriptions:
                        description = keyword_descriptions[keyword]
                    elif fits_comment and fits_comment.strip():
                        description = fits_comment
                    else:
                        description = ''
                    
                    # Determine data type
                    if isinstance(value, bool):
                        dtype = "boolean"
                    elif isinstance(value, int):
                        dtype = "integer"
                    elif isinstance(value, float):
                        dtype = "float"
                    elif isinstance(value, str):
                        dtype = "string"
                    else:
                        dtype = "other"
                    
                    # Determine if auto-populated by FITS
                    fits_auto_keywords = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 
                                        'EXTEND', 'PCOUNT', 'GCOUNT', 'TFIELDS', 'TTYPE1', 'TFORM1', 
                                        'TTYPE2', 'TFORM2', 'TTYPE3', 'TFORM3', 'TTYPE4', 'TFORM4',
                                        'TTYPE5', 'TFORM5', 'TTYPE6', 'TFORM6', 'TTYPE7', 'TFORM7',
                                        'TTYPE8', 'TFORM8', 'TTYPE9', 'TFORM9', 'TTYPE10', 'TFORM10']
                    is_fits_auto = keyword in fits_auto_keywords
                    
                    # Determine if optional based on data level and keyword
                    # Get data level from Image HDU if available
                    datalvl = None
                    if len(hdulist) > 1 and 'DATALVL' in hdulist[1].header:
                        datalvl = hdulist[1].header['DATALVL']
                    
                    # Trap pump keywords are optional for all levels
                    trap_pump_keywords = ['TPINJCYC', 'TPOSCCYC', 'TPTAU', 'TPSCHEM1', 'TPSCHEM2', 'TPSCHEM3', 'TPSCHEM4']
                    
                    # L2b-specific optional keywords
                    l2b_optional_keywords = ['PCTHRESH', 'NUM_FR']
                    
                    # Check if this keyword should be marked as optional
                    is_optional = False
                    if keyword in trap_pump_keywords:
                        is_optional = True
                    elif datalvl == 'L2b' and keyword in l2b_optional_keywords:
                        is_optional = True
                    elif keyword == 'COMMENT' or (keyword.startswith('FILE') and keyword[4:].isdigit()):
                        is_optional = True
                    
                    header_data.append({
                        'Keyword': keyword,
                        'Value': str(value),
                        'Data Type': dtype,
                        'FITS Auto-populated': is_fits_auto,
                        'Optional': is_optional,
                        'Description': description
                    })
                
                # Create DataFrame and save to sheet
                header_df = pd.DataFrame(header_data)
                header_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return output_excel_path


# Default keyword sets for merge_headers
first_frame_keywords_default = ['MJDSRT']
last_frame_keywords_default = ['VISITID', 'MJDEND',  'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'NAXIS4']
averaged_keywords_default = (
    ['RA'] + ['DEC'] + ['RAPM'] + ['DECPM'] + ['PA_V3'] + ['PA_APER'] + ['SVB_1'] + ['SVB_2'] + ['SVB_3'] +
    ['ROLL'] + ['PITCH'] + ['YAW'] +
    ['FSMSG1'] + ['FSMSG2'] + ['FSMSG3'] + ['FSMX'] + ['FSMY'] +
    ['EXCAMT'] + ['NOVEREXP'] + ['PROXET'] +
    [f'Z{i}AVG' for i in range(2, 15)] +
    [f'Z{i}VAR' for i in range(2, 15)] +
    [f'Z{i}RES' for i in range(2, 15)]
)
deleted_keywords_default = ['SCTSRT', 'SCTEND', 'LOCAMT', 'CYCLES', 'LASTEXP']
invalid_keywords_default = ['FTIMEUTC', 'PROXET', 'DATETIME']
# FILETIME is the only one calculated in merge_heades, the others are calculated elsewhere in the
# pipeline but are included here so that they are exempt from the identical check
calculated_value_keywords_default = (
    ['FILETIME', 'NUM_FR', 'DRPCTIME', 'DRPNFILE', 'COMMENT', 'HISTORY', 'FILENAME', 'RECIPE']
    + [f'FILE{i}' for i in range(100)]
)
any_true_keywords_default = ['OVEREXP']

def merge_headers(
    input_dataset,
    first_frame_keywords=None,
    last_frame_keywords=None,
    averaged_keywords=None,
    deleted_keywords=None,
    invalid_keywords=None,
    calculated_value_keywords=None,
    any_true_keywords=None,
):
    """
    Merge headers from multiple input frames into a single header set.

    Used when building combined frames or calibration products from a dataset of frames.

    Frames are sorted by MJDSRT (ascending); the chronologically last frame
    provides the base headers and values for last_frame_keywords.

    Header keywords are handled according to which set they belong to:

    1. first_frame_keywords: use value from the chronologically first frame
    2. last_frame_keywords: use value from the chronologically last frame
    3. averaged_keywords: average across all frames (Z{i}VAR uses pooled-variance)
    4. deleted_keywords: remove keywords from output headers entirely
    5. invalid_keywords: assign -999 with type matching original (int/float/str)
    6. calculated_value_keywords: compute value for output (eg FILETIME = current time)
    7. any_true_keywords: if any frame has a true value for the keyword, use True. else False
    8. all other keywords: must be identical across frames; raise error if not

    Note/ TODO - Only the primary, image extension, error, and DQ headers are merged here.
    Any additional HDUs listed in Image.hdu_list are not handled.

    Args:
        input_dataset (corgidrp.data.Dataset): Dataset of input frames to merge.
        first_frame_keywords (list or set, optional): Keywords to take from the first
            frame.
        last_frame_keywords (list or set, optional): Keywords to take from the last
            frame.
        averaged_keywords (list or set, optional): Keywords to average across frames.
        deleted_keywords (list or set, optional): Keywords to remove from output headers.
        invalid_keywords (list or set, optional): Keywords to set to -999 with type
            matching original (int/float/str).
        calculated_value_keywords (list or set, optional): Keywords whose value is computed
            for the merged output (e.g. FILETIME = current UTC time).
        any_true_keywords (list or set, optional): Keywords where output is True if any
            frame has a truthy value, else False (e.g. DESMEAR, CTI_CORR).

    Returns:
        tuple: (merged_pri_hdr, merged_ext_hdr, merged_err_hdr, merged_dq_hdr)
   
    """
    if first_frame_keywords is None:
        first_frame_keywords = first_frame_keywords_default
    if last_frame_keywords is None:
        last_frame_keywords = last_frame_keywords_default
    if averaged_keywords is None:
        averaged_keywords = averaged_keywords_default
    if deleted_keywords is None:
        deleted_keywords = deleted_keywords_default
    if invalid_keywords is None:
        invalid_keywords = invalid_keywords_default
    if calculated_value_keywords is None:
        calculated_value_keywords = calculated_value_keywords_default

    first_frame_keywords = set(first_frame_keywords)
    last_frame_keywords = set(last_frame_keywords)
    averaged_keywords = set(averaged_keywords)
    deleted_keywords = set(deleted_keywords)
    invalid_keywords = set(invalid_keywords)
    calculated_value_keywords = set(calculated_value_keywords)
    any_true_keywords = set(any_true_keywords) if any_true_keywords else set()

    # Dataset may not be time-ordered, so sort by MJDSRT to define the last frame
    # and define the header starting point
    mjd_vals = [float(f.ext_hdr['MJDSRT']) for f in input_dataset]
    sort_idx = np.argsort(mjd_vals)
    time_ordered = input_dataset[sort_idx]
    first = time_ordered[0]
    last = time_ordered[-1]
    pri_hdr = last.pri_hdr.copy()
    ext_hdr = last.ext_hdr.copy()
    err_hdr = (
        last.err_hdr.copy()
        if hasattr(last, 'err_hdr') and last.err_hdr is not None
        else fits.Header()
    )
    dq_hdr = (
        last.dq_hdr.copy()
        if hasattr(last, 'dq_hdr') and last.dq_hdr is not None
        else fits.Header()
    )

    headers = (pri_hdr, ext_hdr, err_hdr, dq_hdr)
    last_hdrs = (last.pri_hdr, last.ext_hdr, getattr(last, 'err_hdr', None) or fits.Header(), getattr(last, 'dq_hdr', None) or fits.Header())
    first_hdrs = (first.pri_hdr, first.ext_hdr, getattr(first, 'err_hdr', None) or fits.Header(), getattr(first, 'dq_hdr', None) or fits.Header())

    # Keyword set 1 & 2: use values from last/first frame
    for out_hdr, last_src, first_src in zip(headers, last_hdrs, first_hdrs):
        for key in last_frame_keywords:
            if key in last_src:
                out_hdr[key] = last_src[key]
        for key in first_frame_keywords:
            if key in first_src:
                out_hdr[key] = first_src[key]

    # Keyword set 3: average averaged_keywords (using mean or pooled-variance for Z{i}VAR)
    for key in averaged_keywords:
        is_zvar = (len(key) > 4 and key.startswith('Z') and key.endswith('VAR') and
                   key[1:-3].isdigit())
        if is_zvar:
            i = int(key[1:-3])
            avg_key, var_key = f'Z{i}AVG', key
            avg_vals = [float(f.ext_hdr[avg_key]) for f in input_dataset if avg_key in f.ext_hdr]
            var_vals = [float(f.ext_hdr[var_key]) for f in input_dataset if var_key in f.ext_hdr]
            if not var_vals:
                continue
            if avg_vals:
                mu = float(np.mean(avg_vals))
                existing_comment = ext_hdr.comments[var_key] if var_key in ext_hdr else None
                if existing_comment and "from previous" in existing_comment:
                    existing_comment = existing_comment.split("from previous")[0] + "across input frames"
                ext_hdr.set(
                    var_key,
                    float(np.mean(var_vals) + np.mean((np.array(avg_vals) - mu) ** 2)),
                    comment=existing_comment,
                )
            else:
                existing_comment = ext_hdr.comments[var_key] if var_key in ext_hdr else None
                if existing_comment and "from previous" in existing_comment:
                    existing_comment = existing_comment.split("from previous")[0] + "pooled variance across input frames"
                ext_hdr.set(
                    var_key,
                    float(np.mean(var_vals)),
                    comment=existing_comment,
                )
        else:
            values = [float(frame.ext_hdr[key]) for frame in input_dataset if key in frame.ext_hdr]
            if values:
                existing_comment = ext_hdr.comments[key] if key in ext_hdr else None
                if existing_comment and "from previous" in existing_comment:
                    existing_comment = existing_comment.split("from previous")[0] + "averaged across input frames"
                ext_hdr.set(
                    key,
                    float(np.mean(values)),
                    comment=existing_comment,
                )


    # Keyword set 4: remove deleted_keywords from headers
    for hdr in headers:
        for key in list(hdr.keys()):
            if key in deleted_keywords:
                del hdr[key]


    # Keyword set 5: assign -999 to invalid_keywords
    for key in invalid_keywords:
        sample = None
        # figure out the datatype
        for f in input_dataset:
            for attr in ('ext_hdr', 'pri_hdr', 'err_hdr', 'dq_hdr'):
                h = getattr(f, attr, None)
                if h is not None and key in h:
                    sample = h[key]
                    break
            if sample is not None:
                break
        if sample is None:
            inv_val = "-999"
        elif isinstance(sample, (int, np.integer)) or isinstance(sample, bool):
            # TODO: what to do about bool?
            inv_val = -999
        elif isinstance(sample, (float, np.floating)):
            inv_val = -999.0
        else:
            inv_val = "-999"
        for hdr in headers:
            if key in hdr:
                hdr[key] = inv_val

    # Keyword set 6: calculated_value_keywords - compute value
    for key in calculated_value_keywords:
        if key == 'FILETIME':
            val = datetime.datetime.now(datetime.timezone.utc).isoformat()
            pri_hdr[key] = val
            # add others as necessary

    # Keyword set 7: any_true_keywords - true if any frame has true value, else false
    header_attr_pairs = [
        ('pri_hdr', pri_hdr), ('ext_hdr', ext_hdr),
        ('err_hdr', err_hdr), ('dq_hdr', dq_hdr)
    ]
    for key in any_true_keywords:
        for attr, out_hdr in header_attr_pairs:
            values = []
            for f in input_dataset:
                h = getattr(f, attr, None)
                if h is not None and key in h:
                    values.append(h[key])
            if values:
                any_true = any(bool(v) for v in values)
                out_hdr[key] = any_true

    # All other keywords: must be identical across frames, error if not
    exempt = (first_frame_keywords | last_frame_keywords | averaged_keywords |
              deleted_keywords | invalid_keywords | calculated_value_keywords |
              any_true_keywords)

    header_attrs = ('pri_hdr', 'ext_hdr', 'err_hdr', 'dq_hdr')
    for header_attr, out_header in zip(header_attrs, headers):
        for key in list(out_header.keys()):
            if key in exempt:
                continue
            values = []
            for f in input_dataset:
                h = getattr(f, header_attr, None)
                if h is not None and key in h:
                    values.append(h[key])
            if len(values) > 1 and len(set(values)) > 1:
                warnings.warn(
                    f"Keyword {key} not identical across frames. Found: {set(values)}",
                    RuntimeWarning,
                )

    return pri_hdr, ext_hdr, err_hdr, dq_hdr


def fix_hdrs_for_tvac(list_of_fits, output_dir, header_template=None):
    """
    Overwrite FITS headers with mock defaults while preserving certain values from originals.

    Used for TVAC (and similar) data. Writes updated files to output_dir; does not modify originals.

    Args:
        list_of_fits (list): FITS file paths to update
        output_dir (str): Directory to write updated FITS files to
        header_template (callable, optional): Function returning headers.
            Defaults to mocks.create_default_L1_headers.

    Returns:
        list: Updated FITS file paths written to output_dir
    """
    preserve_pri_keys = [
        'VISITID', 'CDMSVERS', 'FSWDVERS', 'ORIGIN', 'FILETIME',
        'VISTYPE', 'DATAVERS', 'PROGNUM', 'EXECNUM', 'CAMPAIGN',
        'SEGMENT', 'OBSNUM', 'VISNUM', 'TARGET', 'FILENAME',
        'PSFREF',
    ]
    preserve_img_keys = [
        'XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
        'PCOUNT', 'GCOUNT', 'BSCALE', 'BZERO', 'BUNIT',
        'ARRTYPE', 'SCTSRT', 'SCTEND', 'STATUS', 'HVCBIAS',
        'OPMODE', 'EXPTIME', 'EMGAIN_C', 'UNITYG', 'EMGAINA1',
        'EMGAINA2', 'EMGAINA3', 'EMGAINA4', 'EMGAINA5', 'GAINTCAL',
        'EXCAMT', 'LOCAMT', 'EMGAIN_A', 'KGAINPAR', 'CYCLES',
        'LASTEXP', 'BLNKTIME', 'BLNKCYC', 'EXPCYC', 'OVEREXP',
        'NOVEREXP', 'ISPC', 'PROXET', 'FCMLOOP', 'FCMPOS',
        'FSMINNER', 'FSMLOS', 'FSMPRFL', 'FSMRSTR', 'FSMSG1',
        'FSMSG2', 'FSMSG3', 'FSMX', 'FSMY', 'EACQ_ROW',
        'EACQ_COL', 'SB_FP_DX', 'SB_FP_DY', 'SB_FS_DX', 'SB_FS_DY',
        'DMZLOOP', '1SVALID', 'Z2AVG', 'Z2RES', 'Z2VAR',
        'Z3AVG', 'Z3RES', 'Z3VAR', '10SVALID', 'Z4AVG',
        'Z4RES', 'Z5AVG', 'Z5RES', 'Z6AVG', 'Z6RES',
        'Z7AVG', 'Z7RES', 'Z8AVG', 'Z8RES', 'Z9AVG',
        'Z9RES', 'Z10AVG', 'Z10RES', 'Z11AVG', 'Z11RES',
        'Z12AVG', 'Z13AVG', 'Z14AVG', 'SPAM_H', 'SPAM_V',
        'SPAMNAME', 'SPAMSP_H', 'SPAMSP_V', 'FPAM_H', 'FPAM_V',
        'FPAMNAME', 'FPAMSP_H', 'FPAMSP_V', 'LSAM_H', 'LSAM_V',
        'LSAMNAME', 'LSAMSP_H', 'LSAMSP_V', 'FSAM_H', 'FSAM_V',
        'FSAMNAME', 'FSAMSP_H', 'FSAMSP_V', 'CFAM_H', 'CFAM_V',
        'CFAMNAME', 'CFAMSP_H', 'CFAMSP_V', 'DPAM_H', 'DPAM_V',
        'DPAMNAME', 'DPAMSP_H', 'DPAMSP_V', 'DATETIME', 'FTIMEUTC',
        'DATALVL', 'MISSING', 'DATATYPE'
    ]

    if header_template is None:
        header_template = mocks.create_default_L1_headers

    updated_files = []
    for file in list_of_fits:
        with fits.open(file) as hdul:
            orig_pri_hdr = hdul[0].header.copy()
            orig_img_hdr = hdul[1].header.copy()

            header_result = header_template()
            mock_pri_hdr, mock_img_hdr = header_result[0], header_result[1]

            for key in preserve_pri_keys:
                if key in orig_pri_hdr:
                    mock_pri_hdr[key] = orig_pri_hdr[key]
            for key in preserve_img_keys:
                if key in orig_img_hdr:
                    mock_img_hdr[key] = orig_img_hdr[key]

            if 'EMGAIN_A' in mock_img_hdr and 'HVCBIAS' in mock_img_hdr:
                if float(mock_img_hdr['EMGAIN_A']) == 1 and mock_img_hdr['HVCBIAS'] <= 0:
                    # SSC TVAC files default EMGAIN_A=1 regardless of commanded gain
                    mock_img_hdr['EMGAIN_A'] = -1
            if 'EMGAIN_C' in mock_img_hdr and isinstance(mock_img_hdr['EMGAIN_C'], str):
                mock_img_hdr['EMGAIN_C'] = float(mock_img_hdr['EMGAIN_C'])

            hdul_copy = fits.HDUList([hdu.copy() for hdu in hdul])
            hdul_copy[0].header = mock_pri_hdr
            hdul_copy[1].header = mock_img_hdr

            output_path = os.path.join(output_dir, os.path.basename(file))
            hdul_copy.writeto(output_path, overwrite=True)
            updated_files.append(output_path)

    return updated_files
