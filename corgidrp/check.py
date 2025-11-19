"""
Module to hold input-checking functions to minimize repetition

Copied over from the II&T pipeline
"""
import numbers
import numpy as np
import logging
import os
import glob
import astropy.io.fits as fits
import pandas as pd

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
