import os
import csv
from pathlib import Path
import numpy as np
import warnings
import math
import re
import datetime
import scipy.ndimage
import pandas as pd
import astropy.io.fits as fits
from astropy.io.fits import Header
from astropy.time import Time
from astropy.io.fits import Header
import astropy.io.ascii as ascii
from astropy.coordinates import SkyCoord
import astropy.wcs as wcs
from astropy.table import Table
from astropy.convolution import convolve_fft
from astropy.modeling import models
import corgidrp
import astropy.units as u
from astropy.modeling.models import Gaussian2D
import photutils.centroids as centr
import corgidrp.data as data
from corgidrp.data import Image, Dataset, DetectorParams, FpamFsamCal, FluxcalFactor
import corgidrp.detector as detector
import corgidrp.flat as flat
from corgidrp.detector import imaging_area_geom, unpack_geom
from corgidrp.pump_trap_calibration import (P1, P1_P1, P1_P2, P2, P2_P2, P3, P2_P3, P3_P3, tau_temp)
from pyklip.instruments.utils.wcsgen import generate_wcs
from corgidrp import measure_companions, corethroughput
from corgidrp.astrom import get_polar_dist, seppa2dxdy, seppa2xy
import datetime
import glob
import shutil
from corgidrp import pol

from emccd_detect.emccd_detect import EMCCDDetect
from emccd_detect.util.read_metadata_wrapper import MetadataWrapper

detector_areas_test= {
'SCI' : { #used for unit tests; enables smaller memory usage with frames of scaled-down comparable geometry
        'frame_rows': 120, 
        'frame_cols': 220,
        'image': {
            'rows': 104,
            'cols': 105,
            'r0c0': [2, 108]
            },
        'prescan_reliable': {
            'rows': 120,
            'cols': 108,
            'r0c0': [0, 0]
        },        

        'prescan': {
            'rows': 120,
            'cols': 108,
            'r0c0': [0, 0],
            'col_start': 0, #10
            'col_end': 108, #100
        }, 

        'serial_overscan' : {
            'rows': 120,
            'cols': 5,
            'r0c0': [0, 215]
        },
        'parallel_overscan': {
            'rows': 14,
            'cols': 107,
            'r0c0': [106, 108]
        }
        },
'ENG' : { #used for unit tests; enables smaller memory usage with frames of scaled-down comparable geometry
        'frame_rows' : 220,
        'frame_cols' : 220,
        'image' : {
            'rows': 102,
            'cols': 102,
            'r0c0': [13, 108]
            },
        'prescan' : {
            'rows': 220,
            'cols': 108,
            'r0c0': [0, 0],
            'col_start': 0, #10
            'col_end': 108, #100
            },
        'prescan_reliable' : {
            'rows': 220,
            'cols': 20,
            'r0c0': [0, 80]
            },
        'parallel_overscan' : {
            'rows': 116,
            'cols': 105,
            'r0c0': [104, 108]
            },
        'serial_overscan' : {
            'rows': 220,
            'cols': 5,
            'r0c0': [0, 215]
            },
        }
}

def parse_csv_table(csv_file_path, section_name, key_col="Keyword",
                    value_col="Example Value", datatype_col="Datatype"):
    """
    Parse a combined CSV (with a Section column) and extract keywords and values
    from a specified section.

    Args:
        csv_file_path (str): Path to the combined CSV file.
        section_name (str): Name of the section to filter on
                            (eg, "Primary Header (HDU 0)" or "Image Header (HDU 1)").
        key_col (str): Column name holding the keyword (default: "Keyword").
        value_col (str): Column name holding the example/value (default: "Example Value").
        datatype_col (str): Column name holding the datatype (default: "Datatype").

    Returns:
        dict: values are coerced using the datatype column. If the section is not a header 
                table or required columns are missing, returns an empty dict.
    """
    def coerce(val_str, dtype_str):
        if val_str is None:
            return None
        s = str(val_str).strip()
        # strip surrounding quotes if present
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()

        dtype = (dtype_str or "").strip().lower()
        if dtype == "int":
            try:
                return int(float(s))  # tolerate "0.0"
            except ValueError:
                return 0
        if dtype == "float":
            try:
                return float(s)
            except ValueError:
                return 0.0
        if dtype == "bool":
            return s.lower() in ("true", "1", "yes", "y", "t")
        # default: string
        return s

    out = {}

    if not os.path.exists(csv_file_path):
        print(f"Warning: CSV file not found at {csv_file_path}")
        return out

    with open(csv_file_path, newline="") as f:
        reader = csv.DictReader(f)
        # Ensure required columns exist
        required_cols = {"Section", key_col, value_col, datatype_col}
        if not required_cols.issubset(reader.fieldnames or []):
            # Not a header-style section or wrong CSV
            return out

        for row in reader:
            if row.get("Section") != section_name:
                continue
            key = (row.get(key_col) or "").strip()
            if not key or key.lower() in {"keyword", "datatype", "example value", "description"}:
                continue
            val = coerce(row.get(value_col), row.get(datatype_col))
            out[key] = val

    return out


def make_mock_fluxcal_factor(value, err=0.0, cfam_name='3D',
                             dpam_name='PRISM3', fsam_name='R1C2'):
    """Create a lightweight FluxcalFactor for unit testing.

    Args:
        value (float): Absolute flux calibration factor to store.
        err (float, optional): Uncertainty on the calibration factor.
        cfam_name (str, optional): CFAM filter name recorded in the header.
        dpam_name (str, optional): DPAM name recorded in the header.
        fsam_name (str, optional): FSAM name recorded in the header.

    Returns:
        FluxcalFactor: Calibration object referencing a dummy dataset so tests
            can exercise downstream logic without building full calibration files.
    """
    pri_hdr, ext_hdr, err_hdr, dq_hdr = create_default_L3_headers()
    ext_hdr['CFAMNAME'] = cfam_name
    ext_hdr['DPAMNAME'] = dpam_name
    ext_hdr['FSAMNAME'] = fsam_name
    dummy_data = np.zeros((2, 2))
    dummy_err = np.zeros((1, 2, 2))
    dummy_dq = np.zeros((2, 2), dtype=int)
    dummy_img = Image(dummy_data, pri_hdr=pri_hdr.copy(), ext_hdr=ext_hdr.copy(),
                      err=dummy_err, dq=dummy_dq, err_hdr=err_hdr.copy(),
                      dq_hdr=dq_hdr.copy())
    return FluxcalFactor(value, err=err, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                         input_dataset=Dataset([dummy_img]))


def create_default_L1_headers(arrtype="SCI", vistype="CGIVST_TDD_OBS"):
    """
    Creates default L1 headers by reading values from the l1.csv documentation file.
    
    Args:
        arrtype (str): Array type ("SCI" or "ENG"). Defaults to "SCI".
        vistype (str): Visit type. Defaults to "CGIVST_TDD_OBS".
    
    Returns:
        tuple: 
            prihdr (fits.Header): Primary FITS header with L1 keywords
            exthdr (fits.Header): Extension FITS header with L1 keywords
    
    """
    # Create empty headers
    prihdr = fits.Header()
    exthdr = fits.Header()
    
    # Set up dynamic values
    dt = datetime.datetime.now()
    dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
    ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
    # Override NAXIS values to match test expectations (1024x1024)
    # L1.rst documents actual detector dimensions (2200x1200 for SCI, 2200x2200 for ENG)
    NAXIS1 = 1024
    NAXIS2 = 1024
    

    # Get the path to the RST file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, 'data', 'header_formats', 'l1.csv')
    
    # Parse primary header values
    primary_values = parse_csv_table(csv_file_path, "Primary Header (HDU 0)")
    
    # Fill in primary header with values from RST
    for keyword, value in primary_values.items():
        prihdr[keyword] = value
    
    # Override some values that should be dynamic
    prihdr['FILETIME'] = dt_str
    prihdr['VISTYPE'] = vistype
    
    # Parse image header values  
    image_values = parse_csv_table(csv_file_path, "Image Header (HDU 1)")
    
    # Fill in extension header with values from RST
    for keyword, value in image_values.items():
        exthdr[keyword] = value
    
    # Override some values that should be dynamic
    exthdr['NAXIS1'] = NAXIS1
    exthdr['NAXIS2'] = NAXIS2
    exthdr['ARRTYPE'] = arrtype
    exthdr['DATETIME'] = dt_str
    exthdr['FTIMEUTC'] = dt_str
    prihdr['FILENAME'] = f"cgi_{prihdr['VISITID']}_{ftime}_l1_.fits"

    return prihdr, exthdr


def create_default_L1_TrapPump_headers(arrtype="SCI"):
    """
    Creates default L1 trap pump headers by reading values from the l1.csv documentation file.
    
    Args:
        arrtype (str): Array type ("SCI" or "ENG"). Defaults to "SCI".
    
    Returns:
        tuple: 
            prihdr (fits.Header): Primary FITS header with L1 trap pump keywords
            exthdr (fits.Header): Extension FITS header with L1 trap pump keywords
    
    """
    # Create empty headers
    prihdr = fits.Header()
    exthdr = fits.Header()
    
    # Set up dynamic values
    dt = datetime.datetime.now()
    dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
    ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
    # Override NAXIS values to match test expectations (1024x1024)
    # L1.rst documents actual detector dimensions (2200x1200 for SCI, 2200x2200 for ENG)
    NAXIS1 = 1024
    NAXIS2 = 1024
    


    # Get the path to the RST file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, 'data', 'header_formats', 'l1.csv')
    
    # Parse primary header values
    primary_values = parse_csv_table(csv_file_path, "Primary Header (HDU 0)")
    
    # Fill in primary header with values from RST
    for keyword, value in primary_values.items():
        prihdr[keyword] = value
    
    # Override some values that should be dynamic
    prihdr['FILETIME'] = dt_str
    prihdr['VISTYPE'] = 'TPUMP'  # Trap pump specific
    
    # Parse image header values  
    image_values = parse_csv_table(csv_file_path, "Image Header (HDU 1)")
    
    # Fill in extension header with values from RST
    for keyword, value in image_values.items():
        exthdr[keyword] = value
    
    # Override some values that should be dynamic
    exthdr['NAXIS1'] = NAXIS1
    exthdr['NAXIS2'] = NAXIS2
    exthdr['ARRTYPE'] = arrtype
    exthdr['DATETIME'] = dt_str
    exthdr['FTIMEUTC'] = dt_str

    prihdr['FILENAME'] = f"cgi_{prihdr['VISITID']}_{ftime}_l1_.fits"
    
    # Override BUNIT for trap pump data (different from regular L1)
    exthdr['BUNIT'] = 'detected EM electron'
    
    # Add trap pumping specific keywords that aren't in the RST
    exthdr['TPINJCYC'] = 0               # Number of cycles for TPUMP injection
    exthdr['TPOSCCYC'] = 0               # Number of cycles for charge oscillation (TPUMP)
    exthdr['TPTAU'] = 0                  # Length of one step in a trap pumping scheme (microseconds)
    exthdr['TPSCHEM1'] = 0               # Number of cycles for TPUMP pumping SCHEME_1
    exthdr['TPSCHEM2'] = 0               # Number of cycles for TPUMP pumping SCHEME_2
    exthdr['TPSCHEM3'] = 0               # Number of cycles for TPUMP pumping SCHEME_3
    exthdr['TPSCHEM4'] = 0               # Number of cycles for TPUMP pumping SCHEME_4

    return prihdr, exthdr


def create_default_L2a_headers(arrtype="SCI"):
    """
    Creates an empty primary header and an Image extension header with currently
        defined keywords.

    Args:
        arrtype (str): Array type (SCI or ENG). Defaults to "SCI". 

    Returns:
        tuple:
            prihdr (fits.Header): Primary FITS Header
            exthdr (fits.Header): Extension FITS Header
            errhdr (fits.Header): Error FITS Header
            dqhdr (fits.Header): Data quality FITS Header
            biashdr (fits.Header): Bias FITS Header


    """
    dt = datetime.datetime.now()
    dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
    ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]

    prihdr, exthdr = create_default_L1_headers(arrtype)
    
    errhdr = fits.Header()
    dqhdr = fits.Header()
    biashdr = fits.Header()

    prihdr['ORIGIN']        = 'DRP'         # Who is responsible for the data
    prihdr['FILENAME']      = f"cgi_{prihdr['VISITID']}_{ftime}_l2a.fits"

    del(exthdr['BSCALE'])
    del(exthdr['BZERO'])
    exthdr['NAXIS1']        = 1024          # Axis 1 size
    exthdr['NAXIS2']        = 1024          # Axis 2 size
    exthdr['DATALVL']       = 'L2a'         # Data level (e.g., 'L1', 'L2a', 'L2b')

    exthdr['DESMEAR']       = False         # Whether desmearing is used
    exthdr['CTI_CORR']      = False         # Whether CTI correction was applied to the frame
    exthdr['IS_BAD']        = False         # Whether frame is bad
    exthdr['FWC_PP_E']      = 0.0           # Full well capacity of detector EM gain register
    exthdr['FWC_EM_E']      = 0             # Full well capacity of detector image area pixel
    exthdr['SAT_DN']        = 0.0           # DN saturation
    exthdr['RECIPE']        = ''            # DRP recipe and steps used to generate this data product
    exthdr['DRPVERSN']      = '2.2'         # Version of DRP software
    exthdr['DRPCTIME']      = dt_str        # DRP clock time
    exthdr['HISTORY']       = ''            # History comments
    exthdr['FTIMEUTC']      = dt_str

    errhdr['XTENSION']    = 'IMAGE'         # Image Extension (FITS format keyword)
    errhdr['BITPIX']      = 16              # Array data type – instrument data is unsigned 16-bit
    errhdr['NAXIS']       = 3               # Number of array dimensions
    errhdr['NAXIS1']      = 1024            # Axis 1 size
    errhdr['NAXIS2']      = 1024            # Axis 2 size
    errhdr['NAXIS3']      = 1               # Axis 3 size
    errhdr['PCOUNT']      = 0               # Number of parameters (FITS keyword)
    errhdr['GCOUNT']      = 1               # Number of groups (FITS keyword)
    errhdr['EXTNAME']     = 'ERR'           # Extension name
    errhdr['TRK_ERRS']    = False           # Whether or not errors are tracked
    errhdr['LAYER_1']     = 'combined_error' # The type of error reported in this slice
    errhdr['HISTORY']       = ''            # History comments

    dqhdr['XTENSION']    = 'IMAGE'         # Image Extension (FITS format keyword)
    dqhdr['BITPIX']      = 16              # Array data type – instrument data is unsigned 16-bit
    dqhdr['NAXIS']       = 2               # Number of array dimensions
    dqhdr['NAXIS1']      = 1024            # Axis 1 size
    dqhdr['NAXIS2']      = 1024            # Axis 2 size
    dqhdr['PCOUNT']      = 0               # Number of parameters (FITS keyword)
    dqhdr['GCOUNT']      = 1               # Number of groups (FITS keyword)
    dqhdr['EXTNAME']     = 'DQ'           # Extension name

    biashdr['XTENSION']    = 'IMAGE'         # Image Extension (FITS format keyword)
    biashdr['BITPIX']      = 16              # Array data type – instrument data is unsigned 16-bit
    biashdr['NAXIS']       = 1               # Number of array dimensions
    biashdr['NAXIS1']      = 1024            # Axis 1 size
    biashdr['PCOUNT']      = 0               # Number of parameters (FITS keyword)
    biashdr['GCOUNT']      = 1               # Number of groups (FITS keyword)
    biashdr['EXTNAME']     = 'BIAS'           # Extension name

    return prihdr, exthdr, errhdr, dqhdr, biashdr

def create_default_L2a_TrapPump_headers(arrtype="SCI"):
    """
    Creates an empty primary header and an Image extension header with currently
        defined keywords.

    Args:
        arrtype (str): Array type (SCI or ENG). Defaults to "SCI". 

    Returns:
        tuple:
            prihdr (fits.Header): Primary FITS Header
            exthdr (fits.Header): Extension FITS Header
            errhdr (fits.Header): Error FITS Header
            dqhdr (fits.Header): Data quality FITS Header
            biashdr (fits.Header): Bias FITS Header

    """
    # TO DO: Update this once L2a headers have been finalized
    dt = datetime.datetime.now()
    dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
    ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]

    prihdr, exthdr = create_default_L1_TrapPump_headers(arrtype)
    
    errhdr = fits.Header()
    dqhdr = fits.Header()
    biashdr = fits.Header()

    prihdr['ORIGIN']        = 'DRP'         # Who is responsible for the data
    prihdr['FILENAME']      = f"cgi_{prihdr['VISITID']}_{ftime}_l2a.fits"

    del(exthdr['BSCALE'])
    del(exthdr['BZERO'])
    exthdr['NAXIS1']        = 1024          # Axis 1 size
    exthdr['NAXIS2']        = 1024          # Axis 2 size
    exthdr['DATALVL']       = 'L2a'         # Data level (e.g., 'L1', 'L2a', 'L2b')

    exthdr['DESMEAR']       = False         # Whether desmearing is used
    exthdr['CTI_CORR']      = False         # Whether CTI correction was applied to the frame
    exthdr['IS_BAD']        = False         # Whether frame is bad
    exthdr['FWC_PP_E']      = 0.0           # Full well capacity of detector EM gain register
    exthdr['FWC_EM_E']      = 0             # Full well capacity of detector image area pixel
    exthdr['SAT_DN']        = 0.0           # DN saturation
    exthdr['RECIPE']        = ''            # DRP recipe and steps used to generate this data product
    exthdr['DRPVERSN']      = '2.2'         # Version of DRP software
    exthdr['DRPCTIME']      = dt_str        # DRP clock time
    exthdr['HISTORY']       = ''            # History comments

    errhdr['XTENSION']    = 'IMAGE'         # Image Extension (FITS format keyword)
    errhdr['BITPIX']      = 16              # Array data type – instrument data is unsigned 16-bit
    errhdr['NAXIS']       = 3               # Number of array dimensions
    errhdr['NAXIS1']      = 1024            # Axis 1 size
    errhdr['NAXIS2']      = 1024            # Axis 2 size
    errhdr['NAXIS3']      = 1               # Axis 3 size
    errhdr['PCOUNT']      = 0               # Number of parameters (FITS keyword)
    errhdr['GCOUNT']      = 1               # Number of groups (FITS keyword)
    errhdr['EXTNAME']     = 'ERR'           # Extension name
    errhdr['TRK_ERRS']    = False           # Whether or not errors are tracked
    errhdr['LAYER_1']     = 'combined_error' # The type of error reported in this slice
    errhdr['HISTORY']       = ''            # History comments

    dqhdr['XTENSION']    = 'IMAGE'         # Image Extension (FITS format keyword)
    dqhdr['BITPIX']      = 16              # Array data type – instrument data is unsigned 16-bit
    dqhdr['NAXIS']       = 2               # Number of array dimensions
    dqhdr['NAXIS1']      = 1024            # Axis 1 size
    dqhdr['NAXIS2']      = 1024            # Axis 2 size
    dqhdr['PCOUNT']      = 0               # Number of parameters (FITS keyword)
    dqhdr['GCOUNT']      = 1               # Number of groups (FITS keyword)
    dqhdr['EXTNAME']     = 'DQ'           # Extension name

    biashdr['XTENSION']    = 'IMAGE'         # Image Extension (FITS format keyword)
    biashdr['BITPIX']      = 16              # Array data type – instrument data is unsigned 16-bit
    biashdr['NAXIS']       = 1               # Number of array dimensions
    biashdr['NAXIS1']      = 1024            # Axis 1 size
    biashdr['PCOUNT']      = 0               # Number of parameters (FITS keyword)
    biashdr['GCOUNT']      = 1               # Number of groups (FITS keyword)
    biashdr['EXTNAME']     = 'BIAS'           # Extension name

    return prihdr, exthdr, errhdr, dqhdr, biashdr


def create_default_L2b_headers(arrtype="SCI"):
    """
    Creates an empty primary header and an Image extension header with currently
        defined keywords.

    Args:
        arrtype (str): Array type (SCI or ENG). Defaults to "SCI". 

    Returns:
        tuple:
            prihdr (fits.Header): Primary FITS Header
            exthdr (fits.Header): Extension FITS Header
            errhdr (fits.Header): Error FITS Header
            dqhdr (fits.Header): Data quality FITS Header
            biashdr (fits.Header): Bias FITS Header

    """

    prihdr, exthdr, errhdr, dqhdr, biashdr = create_default_L2a_headers(arrtype)

    dt = datetime.datetime.now()
    dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
    ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
    exthdr['DRPCTIME']      = dt_str        # DRP clock time
    prihdr['FILENAME']      = f"cgi_{prihdr['VISITID']}_{ftime}_l2b.fits"

    exthdr['BUNIT'] = 'photoelectron'   # Physical unit of the array (brightness unit)
    exthdr['DATALVL']       = 'L2b'         # Data level (e.g., 'L1', 'L2a', 'L2b')

    exthdr['KGAIN_ER']      = 0.0           # Kgain error
    exthdr['RN']            = ''            # Read noise
    exthdr['RN_ERR']        = ''            # Read noise error
    exthdr['FRMSEL01'] = (1, "Bad Pixel Fraction < This Value. Doesn't include DQflags summed to 0") # record selection criteria
    exthdr['FRMSEL02'] = (False, "Are we selecting on the OVEREXP flag?") # record selection criteria
    exthdr['FRMSEL03'] = (None, "tip rms (Z2VAR) threshold") # record selection criteria
    exthdr['FRMSEL04'] = (None, "tilt rms (Z3VAR) threshold") # record selection criteria
    exthdr['FRMSEL05'] = (None, "tip bias (Z2RES) threshold") # record selection criteria
    exthdr['FRMSEL06'] = (None, "tilt bias (Z3RES) threshold") # record selection criteria
    exthdr.add_history("Marked 0 frames as bad: ") # history message tracking bad frames

    errhdr['BUNIT']         = 'photoelectron'   # Unit of error map
    errhdr['KGAINPAR']      = exthdr['KGAINPAR'] # Calculated kgain parameter (copied from exthdr)
    errhdr['KGAIN_ER']      = exthdr['KGAIN_ER'] # Kgain error (copied from exthdr)
    errhdr['RN']            = exthdr['RN']       # Kgain error (copied from exthdr)
    errhdr['DESMEAR']       = exthdr['DESMEAR']  # Whether desmearing was used (copied from exthdr)
    errhdr['LAYER_1']       = 'combined_error' # The type of error reported in this slice

    return prihdr, exthdr, errhdr, dqhdr, biashdr


def create_default_L2b_TrapPump_headers(arrtype="SCI"):
    """
    Creates an empty primary header and an Image extension header with currently
        defined keywords.

    Args:
        arrtype (str): Array type (SCI or ENG). Defaults to "SCI". 

    Returns:
        tuple:
            prihdr (fits.Header): Primary FITS Header
            exthdr (fits.Header): Extension FITS Header
            errhdr (fits.Header): Error FITS Header
            dqhdr (fits.Header): Data quality FITS Header
            biashdr (fits.Header): Bias FITS Header

    """

    prihdr, exthdr, errhdr, dqhdr, biashdr = create_default_L2a_TrapPump_headers(arrtype)

    dt = datetime.datetime.now()
    dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
    ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
    exthdr['DRPCTIME']      = dt_str        # DRP clock time
    prihdr['FILENAME']      = f"cgi_{prihdr['VISITID']}_{ftime}_l2b.fits"

    exthdr['BUNIT'] = 'photoelectron'   # Physical unit of the array (brightness unit)
    exthdr['DATALVL']       = 'L2b'         # Data level (e.g., 'L1', 'L2a', 'L2b')

    exthdr['KGAIN_ER']      = 0.0           # Kgain error
    exthdr['RN']            = ''            # Read noise
    exthdr['RN_ERR']        = ''            # Read noise error
    exthdr['FRMSEL01'] = (1, "Bad Pixel Fraction < This Value. Doesn't include DQflags summed to 0") # record selection criteria
    exthdr['FRMSEL02'] = (False, "Are we selecting on the OVEREXP flag?") # record selection criteria
    exthdr['FRMSEL03'] = (None, "tip rms (Z2VAR) threshold") # record selection criteria
    exthdr['FRMSEL04'] = (None, "tilt rms (Z3VAR) threshold") # record selection criteria
    exthdr['FRMSEL05'] = (None, "tip bias (Z2RES) threshold") # record selection criteria
    exthdr['FRMSEL06'] = (None, "tilt bias (Z3RES) threshold") # record selection criteria
    exthdr.add_history("Marked 0 frames as bad: ") # history message tracking bad frames
    exthdr['PCTHRESH']      = 0.0           # Photon counting threshold applied
    exthdr['NUM_FR']        = 0             # Number of frames which were PC processed

    errhdr['BUNIT']         = exthdr['BUNIT']   # Unit of error map
    errhdr['KGAINPAR']      = exthdr['KGAINPAR'] # Calculated kgain parameter (copied from exthdr)
    errhdr['KGAIN_ER']      = exthdr['KGAIN_ER'] # Kgain error (copied from exthdr)
    errhdr['RN']            = exthdr['RN']       # Kgain error (copied from exthdr)
    errhdr['DESMEAR']       = exthdr['DESMEAR']  # Whether desmearing was used (copied from exthdr)

    return prihdr, exthdr, errhdr, dqhdr, biashdr


def create_default_L3_headers(arrtype="SCI"):
    """
    Creates an empty primary header and an Image extension header with currently
        defined keywords.

    Args:
        arrtype (str): Array type (SCI or ENG). Defaults to "SCI". 

    Returns:
        tuple:
            prihdr (fits.Header): Primary FITS Header
            exthdr (fits.Header): Extension FITS Header
            errhdr (fits.Header): Error FITS Header
            dqhdr (fits.Header): Data quality FITS Header

    """
    # TO DO: Update this once L3 headers have been finalized
    prihdr, exthdr, errhdr, dqhdr, biashdr = create_default_L2b_headers(arrtype)

    dt = datetime.datetime.now()
    dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
    ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
    exthdr['DRPCTIME']      = dt_str        # DRP clock time
    exthdr['HISTORY']       = ''            # History comments
    prihdr['FILENAME']      = f"cgi_{prihdr['VISITID']}_{ftime}_l3_.fits"
    
    exthdr['BUNIT'] = 'photoelectron/s'   # Physical unit of the array (brightness unit)
    exthdr['CD1_1'] = 0
    exthdr['CD1_2'] = 0
    exthdr['CD2_1'] = 0
    exthdr['CD2_2'] = 0
    exthdr['CRPIX1'] = 0
    exthdr['CRPIX2'] = 0
    exthdr['CTYPE1'] = 'RA---TAN'
    exthdr['CTYPE2'] = 'DEC--TAN'
    exthdr['CDELT1'] = 0
    exthdr['CDELT2'] = 0
    exthdr['CRVAL1'] = 0
    exthdr['CRVAL2'] = 0
    exthdr['PLTSCALE'] = 21.8             # mas/ pixel
    exthdr['DATALVL']    = 'L3'           # Data level (e.g., 'L1', 'L2a', 'L2b')

    errhdr['LAYER_1']       = 'combined_error' # The type of error reported in this slice

    return prihdr, exthdr, errhdr, dqhdr


def create_default_L4_headers(arrtype="SCI"):
    """
    Creates an empty primary header and an Image extension header with currently
        defined keywords.

    Args:
        arrtype (str): Array type (SCI or ENG). Defaults to "SCI". 

    Returns:
        tuple:
            prihdr (fits.Header): Primary FITS Header
            exthdr (fits.Header): Extension FITS Header
            errhdr (fits.Header): Error FITS Header
            dqhdr (fits.Header): Data quality FITS Header

    """
    # TO DO: Update this once L4 headers have been finalized
    prihdr, exthdr, errhdr, dqhdr = create_default_L3_headers(arrtype)

    dt = datetime.datetime.now()
    dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
    ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
    exthdr['DRPCTIME']      = dt_str        # DRP clock time
    exthdr['HISTORY']       = ''            # History comments
    prihdr['FILENAME']      = f"cgi_{prihdr['VISITID']}_{ftime}_l4_.fits"
    
    exthdr['NUM_FR']        = 2             # Number of frames that were used in the combine_subexposures step
    exthdr['DRPNFILE']      = 2             # Num raw files used in final image combination
    exthdr['FILE0']         = 'mockfile0.fits' 	#File name for the Nth science file used in PSF subtraction
    exthdr['FILE1']         = 'mockfile1.fits' 	#File name for the Nth science file used in PSF subtraction
    exthdr['PSFSUB']        = ''            # PSF subtraction algorithm used (coronagraphic only)
    exthdr['PYKLIPV']       = ''            # pyKLIP version used (coronagraphic only)
    exthdr['KLMODE0']       = ''            # Number of KL modes used in the Nth slice (coronagraphic only)
    exthdr['STARLOCX']      = 512           # X location of the of the target star (coronagraphic only)
    exthdr['STARLOCY']      = 512           # Y location of the of the target star (coronagraphic only)
    exthdr['DETPIX0X']      = ''            #  Position of the 0th column of the data array on the 1024x1024 EXCAM detector
    exthdr['DETPIX0Y']      = ''            # Position of the 0th row of the data array on the 1024x1024 EXCAM detector 
    exthdr['CTCALFN']       = ''            # Core throughput linked file for calibration
    exthdr['FLXCALFN']      = ''            # Abs flux file linked for calibration
    exthdr['DATALVL']       = 'L4'          # Data level (e.g., 'L1', 'L2a', 'L2b')

    errhdr['LAYER_1']       = 'combined_error' # The type of error reported in this slice

    return prihdr, exthdr, errhdr, dqhdr


def create_default_calibration_product_headers():
    '''
    This function creates the basic primary and extension headers that
        would be used in a calibration product. Each individual calibration
        product should add additional headers as required.

    Returns:
        tuple:
            prihdr (fits.Header): Primary FITS Header
            exthdr (fits.Header): Extension FITS Header
            errhdr (fits.Header): Error FITS Header
            dqhdr (fits.Header): Data quality FITS Header
    '''
    # TO DO: update when this has been more defined
    prihdr, exthdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    exthdr['DATALVL']    = 'CAL'
    exthdr['DATATYPE']    = 'Image'              # What type of calibration product, just do image for now, mock codes will update

    return prihdr, exthdr, errhdr, dqhdr


def create_noise_maps(FPN_map, FPN_map_err, FPN_map_dq, CIC_map, CIC_map_err, CIC_map_dq, DC_map, DC_map_err, DC_map_dq):
    '''
    Create simulated noise maps for test_masterdark_from_noisemaps.py.

    Arguments:
        FPN_map: 2D np.array for fixed-pattern noise (FPN) data array
        FPN_map_err: 2D np.array for FPN err array
        FPN_map_dq: 2D np.array for FPN DQ array
        CIC_map: 2D np.array for clock-induced charge (CIC) data array
        CIC_map_err: 2D np.array for CIC err array
        CIC_map_dq: 2D np.array for CIC DQ array
        DC_map: 2D np.array for dark current data array
        DC_map_err: 2D np.array for dark current err array
        DC_map_dq: 2D np.array for dark current DQ array

    Returns:
        corgidrp.data.DetectorNoiseMaps instance
    '''

    prihdr, exthdr, errhdr, dqhdr = create_default_calibration_product_headers()
    # taken from end of calibrate_darks_lsq()

    exthdr['EMGAIN_A']    = 0.0             # "Actual" gain computed from coefficients and calibration temperature
    exthdr['EMGAIN_C']    = 1.0             # Commanded gain computed from coefficients and calibration temperature
    exthdr['DATALVL']      = 'CalibrationProduct'
    exthdr['DATATYPE']      = 'DetectorNoiseMaps'
    exthdr['DRPNFILE']      = 2         # Number of files used to create this calibration product 
    exthdr['FILE0']         = "Mock0.fits"
    exthdr['FILE1']         = "Mock1.fits"
    exthdr['B_O'] = 0.01
    exthdr['B_O_UNIT'] = 'DN'
    exthdr['B_O_ERR'] = 0.001

    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    exthdr['DATATYPE'] = 'DetectorNoiseMaps'
    input_data = np.stack([FPN_map, CIC_map, DC_map])
    err = np.stack([[FPN_map_err, CIC_map_err, DC_map_err]])
    dq = np.stack([FPN_map_dq, CIC_map_dq, DC_map_dq])
    noise_maps = data.DetectorNoiseMaps(input_data, pri_hdr=prihdr, ext_hdr=exthdr, err=err,
                              dq=dq, err_hdr=err_hdr)
    return noise_maps


def create_synthesized_master_dark_calib(detector_areas):
    '''
    Create simulated data specifically for test_calibrate_darks_lsq.py.

    Args:
        detector_areas: dict
        a dictionary of detector geometry properties.  Keys should be as found
        in detector_areas in detector.py.


    Returns:
        dataset: corgidrp.data.Dataset instances
    The simulated dataset
    '''

    dark_current = 8.33e-4 #e-/pix/s
    cic=0.02  # e-/pix/frame
    read_noise=100 # e-/pix/frame
    bias=2000 # e-
    eperdn = 7 # e-/DN conversion; used in this example for all stacks
    EMgain_picks = (np.linspace(2, 5000, 7))
    exptime_picks = (np.linspace(2, 100, 7))
    grid = np.meshgrid(EMgain_picks, exptime_picks)
    EMgain_arr = grid[0].ravel()
    exptime_arr = grid[1].ravel()
    #added in after emccd_detect makes the frames (see below)
    # The mean FPN that will be found is eperdn*(FPN//eperdn)
    # due to how I simulate it and then convert the frame to uint16
    FPN = 21 # e
    # the bigger N is, the better the adjusted R^2 per pixel becomes
    N = 30 #Use N=600 for results with better fits (higher values for adjusted
    # R^2 per pixel)
    # image area, including "shielded" rows and cols:
    imrows, imcols, imr0c0 = imaging_area_geom('SCI', detector_areas)
    prerows, precols, prer0c0 = unpack_geom('SCI', 'prescan', detector_areas)
    
    frame_list = []
    for i in range(len(EMgain_arr)):
        for l in range(N): #number of frames to produce
            # Simulate full dark frame (image area + the rest)
            frame_rows = detector_areas['SCI']['frame_rows']
            frame_cols = detector_areas['SCI']['frame_cols']
            frame_dn_dark = np.zeros((frame_rows, frame_cols))
            im = np.random.poisson(cic*EMgain_arr[i]+
                                exptime_arr[i]*EMgain_arr[i]*dark_current,
                                size=(frame_rows, frame_cols))
            frame_dn_dark = im
            # prescan has no dark current
            pre = np.random.poisson(cic*EMgain_arr[i],
                                    size=(prerows, precols))
            frame_dn_dark[prer0c0[0]:prer0c0[0]+prerows,
                            prer0c0[1]:prer0c0[1]+precols] = pre
            rn = np.random.normal(0, read_noise,
                                    size=(frame_rows, frame_cols))
            with_rn = frame_dn_dark + rn + bias

            frame_dn_dark = with_rn/eperdn
            # simulate a constant FPN in image area (not in prescan
            # so that it isn't removed when bias is removed)
            frame_dn_dark[imr0c0[0]:imr0c0[0]+imrows,imr0c0[1]:
            imr0c0[1]+imcols] += FPN/eperdn # in DN
            # simulate telemetry rows, with the last 5 column entries with high counts
            frame_dn_dark[-1,-5:] = 100000 #DN
            # take raw frames and process them to what is needed for input
            # No simulated pre-processing bad pixels or cosmic rays, so just subtract bias
            # and multiply by k gain
            frame_dn_dark -= bias/eperdn
            frame_dn_dark *= eperdn

            # Now make this into a bunch of corgidrp.Dataset stacks
            prihdr, exthdr, errhdr, dqhdr = create_default_calibration_product_headers()
            frame = data.Image(frame_dn_dark, pri_hdr=prihdr,
                            ext_hdr=exthdr)
            frame.ext_hdr['EMGAIN_C'] = EMgain_arr[i]
            frame.ext_hdr['EXPTIME'] = exptime_arr[i]
            frame.ext_hdr['KGAINPAR'] = eperdn
            frame_list.append(frame)
    dataset = data.Dataset(frame_list)

    return dataset


def create_dark_calib_files(filedir=None, numfiles=10):
    """
    Create simulated data to create a master dark.
    Assume these have already undergone L1 processing and are L2a level products

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 10.

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset
    """
    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_L1_headers(arrtype="SCI")
        prihdr["OBSNUM"] = 000
        exthdr['KGAINPAR'] = 7
        exthdr['BUNIT'] = "detected electron"
        #np.random.seed(456+i); 
        sim_data = np.random.poisson(lam=150., size=(1200, 2200)).astype(np.float64)
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            # Generate unique, properly formatted filename
            visitid = prihdr["VISITID"]
            base_time = datetime.datetime.now()
            time_offset = datetime.timedelta(seconds=i)
            unique_time = base_time + time_offset
            time_str = data.format_ftimeutc(unique_time.isoformat())
            filename = f"cgi_{visitid}_{time_str}_l1_.fits"
            frame.save(filedir=filedir, filename=filename)
        frames.append(frame)
    dataset = data.Dataset(frames)
    return dataset


def create_simflat_dataset(filedir=None, numfiles=10):
    """
    Create simulated data to check the flat division

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 10.

    Returns:
        corgidrp.data.Dataset:
        The simulated dataset
    """
    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_L1_headers()
        # generate images in normal distribution with mean 1 and std 0.01
        #np.random.seed(456+i); 
        sim_data = np.random.poisson(lam=150., size=(1024, 1024)).astype(np.float64)
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            # Generate unique, properly formatted filename
            visitid = prihdr["VISITID"]
            base_time = datetime.datetime.now()
            time_offset = datetime.timedelta(seconds=i)
            unique_time = base_time + time_offset
            time_str = data.format_ftimeutc(unique_time.isoformat())
            filename = f"cgi_{visitid}_{time_str}_l1_.fits"
            frame.save(filedir=filedir, filename=filename)
        frames.append(frame)
    dataset = data.Dataset(frames)
    return dataset


def create_raster(mask,data,dither_sizex=None,dither_sizey=None,row_cent = None,col_cent = None,n_dith=None,mask_size=420,snr=250,planet=None, band=None, radius=None, snr_constant=None):
    """Performs raster scan of Neptune or Uranus images
    
    Args:
        mask (int): (Required)  Mask used for the image. (Size of the HST images, 420 X 420 pixels with random values mean=1, std=0.03)
        data (float):(Required) Data in array npixels*npixels format to be raster scanned
        dither_sizex (int):(Required) Size of the dither in X axis in pixels (number of pixels across the planet (neptune=50 and uranus=65))
        dither_sizey (int):(Required) Size of the dither in X axis in pixels (number of pixels across the planet (neptune=50 and uranus=65))
        row_cent (int): (Required)  X coordinate of the centroid
        col_cent (int): (Required)  Y coordinate of the centroid
        n_dith (int): number of dithers required
        mask_size (int): Size of the mask in pixels  (Size of the HST images, 420 X 420 pixels with random values mean=1, std=0.03)
        snr (int): Required SNR in the planet images (=250 in the HST images)
        planet (str): neptune or uranus
        band (str): 1 or 4
        radius (int): radius of the planet in pixels (radius=54 for neptune, radius=90)
        snr_constant (int): constant for snr reference  (4.95 for band1 and 9.66 for band4)
        
	Returns:
    	dither_stack_norm (np.array): stacked dithers of the planet images
    	cent (np.array): centroid of images 
    	
        
    """  
 
    cents = []
    
    data_display = data.copy()
    col_max = int(col_cent) + int(mask_size/2)
    col_min = int(col_cent) - int(mask_size/2)
    row_max = int(row_cent) + int(mask_size/2)
    row_min = int(row_cent) - int(mask_size/2)
    dithers = []
    
    if dither_sizey == None:
        dither_sizey = dither_sizex

    
    for i in np.arange(-n_dith,n_dith):
        for j in np.arange(-n_dith,n_dith):
            mask_data = data.copy()
            new_image_row_coords = np.arange(row_min + (dither_sizey * j), row_max + (dither_sizey * j))
            new_image_col_coords = np.arange(col_min + (dither_sizex * i), col_max + (dither_sizex * i))
            new_image_col_coords, new_image_row_coords = np.meshgrid(new_image_col_coords, new_image_row_coords)
            image_data = scipy.ndimage.map_coordinates(mask_data, [new_image_row_coords, new_image_col_coords], mode="constant", cval=0)
            # image_data = mask_data[row_min + (dither_sizey * j):row_max + (dither_sizey * j), col_min + (dither_sizex * i):col_max + (dither_sizex * i)]
            cents.append(((mask_size/2) + (row_cent - int(row_cent)) - (dither_sizey//2) - (dither_sizey * j), (mask_size/2) + (col_cent - int(col_cent)) - (dither_sizex//2) - (dither_sizex * i)))
            # try:
            new_image_data = image_data * mask
            
            snr_ref = snr/np.sqrt(snr_constant)

            u_centroid = centr.centroid_1dg(new_image_data)
            uxc = int(u_centroid[0])
            uyc = int(u_centroid[1])

            modified_data = new_image_data

            nx = np.arange(0,modified_data.shape[1])
            ny = np.arange(0,modified_data.shape[0])
            nxx,nyy = np.meshgrid(nx,ny)
            nrr = np.sqrt((nxx-uxc)**2 + (nyy-uyc)**2)

            planmed = np.median(modified_data[nrr<radius])
            modified_data[nrr<=radius] = np.random.normal(modified_data[nrr<=radius], (planmed/snr_ref) * np.abs(modified_data[nrr<=radius]/planmed))
            
            new_image_data_snr = modified_data
            # except ValueError:
            #     print(image_data.shape)
            #     print(mask.shape)
            dithers.append(new_image_data_snr)

    dither_stack_norm = []
    for dither in dithers:
        dither_stack_norm.append(dither) 
    dither_stack = None 
    
    median_dithers = None 
    final = None 
    full_mask = mask 
    
    return dither_stack_norm,cents
    

def create_onsky_rasterscans(dataset,filedir=None,planet=None,band=None, im_size=420, d=None, n_dith=3, radius=None, snr=250, snr_constant=None, flat_map=None, raster_radius=40, raster_subexps=1):
    """
    Create simulated data to check the flat division
    
    Args:
       dataset (corgidrp.data.Dataset): dataset of HST images of neptune and uranus
       filedir (str): Full path to directory to save the raster scanned images.
       planet (str): neptune or uranus
       band (str): 1 or 4
       im_size (int): x-dimension of the planet image (in pixels= 420 for the HST images)
       d (int): number of pixels across the planet (neptune=50 and uranus=65)
       n_dith (int): Number of dithers required (Default is 3)
       radius (int): radius of the planet in pixels (radius=54 for neptune, radius=90 in HST images)
       snr (int): SNR required for the planet image (default is 250 for the HST images)
       snr_constant (int): constant for snr reference  (4.95 for band1 and 9.66 for band4)
       flat_map (np.array): a user specified flat map. Must have shape (im_size, im_size). Default: None; assumes each pixel drawn from a normal distribution with 3% rms scatter
       raster_radius (float): radius of circular raster done to smear out image during observation, in pixels
       raster_subexps (int): number of subexposures that consist of a singular raster. Currently just duplicates images and does not simulate partial rasters
        
    Returns: 
    	corgidrp.data.Dataset:
        The simulated dataset of raster scanned images of planets uranus or neptune
    """
    n = im_size

    if flat_map is None:
        qe_prnu_fsm_raster = np.random.normal(1,.03,(n,n))
    else:
        qe_prnu_fsm_raster = flat_map

    pred_cents=[]
    planet_rot_images=[]
    
    for i in range(len(dataset)):
        target=dataset[i].pri_hdr['TARGET']
        filter=dataset[i].pri_hdr['FILTER']
        if planet==target and band==filter: 
            planet_image=dataset[i].data
            centroid=centr.centroid_com(planet_image)
            xc=centroid[0]
            yc=centroid[1]
            planet_image = convolve_fft(planet_image, flat.raster_kernel(raster_radius, planet_image))
            if planet == 'neptune':
                planetrad=radius; snrcon=snr_constant
                planet_repoint_current = create_raster(qe_prnu_fsm_raster,planet_image,row_cent=yc+(d//2),col_cent=xc+(d//2), dither_sizex=d, dither_sizey=d,n_dith=n_dith,mask_size=n,snr=snr,planet=target,band=filter,radius=planetrad, snr_constant=snrcon)
            elif planet == 'uranus':
                planetrad=radius; snrcon=snr_constant     
                planet_repoint_current = create_raster(qe_prnu_fsm_raster,planet_image,row_cent=yc,col_cent=xc, dither_sizex=d, dither_sizey=d,n_dith=n_dith,mask_size=n,snr=snr,planet=target,band=filter,radius=planetrad, snr_constant=snrcon)
    
    numfiles = len(planet_repoint_current[0])
    for j in np.arange(numfiles):
        for k in range(raster_subexps):
            # don't know how to simualate partial rasters, so we just append the same image multiple times
            # it's ok to append the same noise as well because we simulated the full raster to reach the SNR after combining subexps
            planet_rot_images.append(planet_repoint_current[0][j])
            pred_cents.append(planet_repoint_current[1][j])

    frames=[]
    # Generate base timestamp once for consistency
    base_dt = datetime.datetime.now()
    base_visitid = None
    
    for i in range(numfiles*raster_subexps):
        prihdr, exthdr = create_default_L1_headers()
        
        # Get VISITID (should be consistent across frames)
        if base_visitid is None:
            base_visitid = prihdr.get('VISITID', '0000000000000000000')
        
        # Generate unique timestamp for each frame (add microseconds for uniqueness)
        dt = base_dt + datetime.timedelta(microseconds=i*100)
        ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
        
        sim_data=planet_rot_images[i]
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        pl=planet
        band=band
        frame.pri_hdr.set('TARGET', pl)
        frame.ext_hdr.append(('CFAMNAME', "{0}F".format(band)), end=True)
        
        # Set proper L2a filename
        filename = f"cgi_{base_visitid}_{ftime}_l2a.fits"
        
        if filedir is not None:
            frame.save(filedir=filedir, filename=filename)
        else:
            frame.filename = filename
        frames.append(frame)
    raster_dataset = data.Dataset(frames)
    return raster_dataset


def create_flatfield_dummy(filedir=None, numfiles=2):

    """
    Turn this flat field dataset of image frames that were taken for performing the flat calibration and
    to make one master flat image

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 1 to create the dummy flat can be changed to any number

    Returns:
        corgidrp.data.Dataset:
        a set of flat field images
    """
    ## Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    frames=[]
    for i in range(numfiles):
        prihdr, exthdr = create_default_L1_headers()
        #np.random.seed(456+i); 
        sim_data = np.random.normal(loc=1.0, scale=0.01, size=(1024, 1024))
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            # Generate CGI filename with incrementing datetime
            visitid = prihdr["VISITID"]
            base_time = datetime.datetime.now()
            time_offset = datetime.timedelta(seconds=i)
            unique_time = base_time + time_offset
            time_str = data.format_ftimeutc(unique_time.isoformat())
            filename = f"cgi_{visitid}_{time_str}_l1_.fits"
            frame.save(filedir=filedir, filename=filename)
        frames.append(frame)
    flatfield = data.Dataset(frames)
    return flatfield

def create_nonlinear_dataset(nonlin_filepath, filedir=None, numfiles=2,em_gain=2000):
    """
    Create simulated data to non-linear data to test non-linearity correction.

    Args:
        nonlin_filepath (str): path to FITS file containing nonlinear calibration data (e.g., tests/test_data/nonlin_sample.fits)
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 2 (not creating the cal here, just testing the function)
        em_gain (int): The EM gain to use for the simulated data.  Defaults to 2000.

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset
    """

    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_L1_headers()
        #Add the commanded gain to the headers
        exthdr['EMGAIN_C'] = em_gain
        exthdr['OBSNAME'] = 'NONLIN'
        # Create a default
        size = 1024
        sim_data = np.zeros([size,size])
        data_range = np.linspace(800,65536,size)
        # Generate data for each row, where the mean increase from 10 to 65536
        for x in range(size):
            #np.random.seed(120+x); 
            sim_data[:, x] = np.random.poisson(data_range[x], size).astype(np.float64)

        non_linearity_correction = data.NonLinearityCalibration(nonlin_filepath)

        #Apply the non-linearity to the data. When we correct we multiple, here when we simulate we divide
        #This is a bit tricky because when we correct the get_relgains function takes the current state of 
        # the data as input, which when actually used will be the non-linear data. Here we try to get close 
        # to that by calculating the relative gains after applying the relative gains one time. This won't be 
        # perfect, but it'll be closer than just dividing by the straight simulated data. 

        sim_data_tmp = sim_data/detector.get_relgains(sim_data,em_gain,non_linearity_correction)

        sim_data /= detector.get_relgains(sim_data_tmp,em_gain,non_linearity_correction)

        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)
        if filedir is not None:
            # Generate unique, properly formatted filename
            visitid = prihdr["VISITID"]
            base_time = datetime.datetime.now()
            time_offset = datetime.timedelta(seconds=i)
            unique_time = base_time + time_offset
            time_str = data.format_ftimeutc(unique_time.isoformat())
            filename = f"cgi_{visitid}_{time_str}_l1_.fits"
            frame.save(filedir=filedir, filename=filename)
        frames.append(frame)
    dataset = data.Dataset(frames)
    return dataset


def create_cr_dataset(nonlin_filepath, filedir=None, obs_datetime=None, numfiles=2, em_gain=500, numCRs=5, plateau_length=10):
    """
    Create simulated non-linear data with cosmic rays to test CR detection.

    Args:
        nonlin_filepath (str): path to FITS file containing nonlinear calibration data (e.g., tests/test_data/nonlin_sample.fits)
        filedir (str): (Optional) Full path to directory to save to.
        obs_datetime (astropy.time.Time): (Optional) Date and time of the observations to simulate.
        numfiles (int): Number of files in dataset.  Defaults to 2 (not creating the cal here, just testing the function)
        em_gain (int): The EM gain to use for the simulated data.  Defaults to 2000.
        numCRs (int): The number of CR hits to inject. Defaults to 5.
        plateau_length (int): The minimum length of a CR plateau that will be flagged by the filter.

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset.
    """

    if obs_datetime is None:
        obs_datetime = Time('2024-01-01T11:00:00.000Z')

    detector_params = data.DetectorParams({}, date_valid=Time("2023-11-01 00:00:00"))

    kgain = detector_params.params['KGAINPAR']
    fwc_em_dn = detector_params.params['FWC_EM_E'] / kgain
    fwc_pp_dn = detector_params.params['FWC_PP_E'] / kgain
    fwc = np.min([fwc_em_dn,em_gain*fwc_pp_dn])
    dataset = create_nonlinear_dataset(nonlin_filepath, filedir=None, numfiles=numfiles,em_gain=em_gain)

    im_width = dataset.all_data.shape[-1]

    # Overwrite dataset with a poisson distribution
    #np.random.seed(123)
    dataset.all_data[:,:,:] = np.random.poisson(lam=150,size=dataset.all_data.shape).astype(np.float64)

    # Loop over images in dataset
    for i in range(len(dataset.all_data)):

        # Save the date
        dataset[i].ext_hdr['DATETIME'] = str(obs_datetime)

        # Pick random locations to add a cosmic ray
        for x in range(numCRs):
            #np.random.seed(123+x)
            loc = np.round(np.random.uniform(0,im_width-1, size=2)).astype(int)

            # Add the CR plateau
            tail_start = np.min([loc[1]+plateau_length,im_width])
            dataset.all_data[i,loc[0],loc[1]:tail_start] += fwc

            if tail_start < im_width-1:
                tail_len = im_width-tail_start
                cr_tail = [fwc/(j+1) for j in range(tail_len)]
                dataset.all_data[i,loc[0],tail_start:] += cr_tail

        # generate unique, properly formatted filename
        visitid = dataset[i].pri_hdr["VISITID"]
        base_time = datetime.datetime.now()
        time_offset = datetime.timedelta(seconds=i)
        unique_time = base_time + time_offset
        time_str = data.format_ftimeutc(unique_time.isoformat())
        dataset[i].filename = f"cgi_{visitid}_{time_str}_l1_.fits"

    # Save frame if desired
    if filedir is not None:
        dataset.save(filedir=filedir)

    return dataset


def create_prescan_files(filedir=None, numfiles=2, arrtype="SCI"):
    """
    Create simulated raw data.

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset.  Defaults to 2.
        arrtype (str): Observation type. Defaults to "SCI".

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset
    """
    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    if arrtype == "SCI":
        size = (1200, 2200)
    elif arrtype == "ENG":
        size = (2200, 2200)
    elif arrtype == "CAL":
        size = (2200,2200)
    else:
        raise ValueError(f'Arrtype {arrtype} not in ["SCI","ENG","CAL"]')

    frames = []
    for i in range(numfiles):
        prihdr, exthdr = create_default_L1_headers(arrtype=arrtype)
        sim_data = np.random.poisson(lam=150., size=size).astype(np.float64)
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)

        if filedir is not None:
            # Generate unique, properly formatted filename
            visitid = prihdr["VISITID"]
            base_time = datetime.datetime.now()
            time_offset = datetime.timedelta(seconds=i)
            unique_time = base_time + time_offset
            time_str = data.format_ftimeutc(unique_time.isoformat())
            filename = f"cgi_{visitid}_{time_str}_l1_.fits"
            frame.save(filedir=filedir, filename=filename)

        frames.append(frame)

    dataset = data.Dataset(frames)

    return dataset


def create_badpixelmap_files(filedir=None, col_bp=None, row_bp=None):
    """
    Create simulated bad pixel map data. Code value is 4.

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        col_bp (array): (Optional) Array of column indices where bad detector
            pixels are found.
        row_bp (array): (Optional) Array of row indices where bad detector
            pixels are found.

    Returns:
        corgidrp.data.BadPixelMap:
            The simulated dataset
    """
    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)

    prihdr, exthdr, errhdr, dqhdr = create_default_calibration_product_headers()
    exthdr['DATATYPE']      = 'BadPixelMap'

    sim_data = np.zeros([1024,1024], dtype = np.uint16)
    if col_bp is not None and row_bp is not None:
        for i_col in col_bp:
            for i_row in row_bp:
                sim_data[i_col, i_row] += 4
    frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr)

    if filedir is not None:
        # Generate unique, properly formatted filename
        visitid = prihdr["VISITID"]
        base_time = datetime.datetime.now()
        time_str = data.format_ftimeutc(base_time.isoformat())
        filename = f"cgi_{visitid}_{time_str}_bpm_cal.fits"
        frame.save(filedir=filedir, filename=filename)

    badpixelmap = data.Dataset([frame])

    return badpixelmap


def nonlin_coefs(filename,EMgain,order):
    """
    Reads TVAC nonlinearity table from location specified by ‘filename’.
    The column in the table closest to the ‘EMgain’ value is selected and fits
    a polynomial of order ‘order’. The coefficients of the fit are adjusted so
    that the polynomial function equals unity at 3000 DN. Outputs array polynomial
    coefficients, array of DN values from the TVAC table, and an array of the
    polynomial function values for all the DN values.

    Args:
      filename (string): file name
      EMgain (int): em gain value
      order (int): polynomial order

    Returns:
      np.array: fit coefficients
      np.array: DN values
      np.array: fit values
    """
    # filename is the name of the csv text file containing the TVAC nonlin table
    # EM gain selects the closest column in the table
    # Load the specified file
    bigArray = pd.read_csv(filename, header=None).values
    EMgains = bigArray[0, 1:]
    DNs = bigArray[1:, 0]

    # Find the closest EM gain available to what was requested
    iG = (np.abs(EMgains - EMgain)).argmin()

    # Fit the nonlinearity numbers to a polynomial
    vals = bigArray[1:, iG + 1]
    coeffs = np.polyfit(DNs, vals, order)

    # shift so that function passes through unity at 3000 DN for these tests
    fitVals0 = np.polyval(coeffs, DNs)
    ind = np.where(DNs == 3000)
    unity_val = fitVals0[ind][0]
    coeffs[3] = coeffs[3] - (unity_val-1.0)
    fitVals = np.polyval(coeffs,DNs)

    return coeffs, DNs, fitVals


def nonlin_factor(coeffs,DN):
    """ 
    Takes array of nonlinearity coefficients (from nonlin_coefs function)
    and an array of DN values and returns the nonlinearity values array. If the
    DN value is less 800 DN, then the nonlinearity value at 800 DN is returned.
    If the DN value is greater than 10000 DN, then the nonlinearity value at
    10000 DN is returned.
    
    Args:
       coeffs (np.array): nonlinearity coefficients
       DN (int): DN value
       
    Returns:
       float: nonlinearity value
    """
    # input coeffs from nonlin_ceofs and a DN value and return the
    # nonlinearity factor
    min_value = 800.0
    max_value = 10000.0
    f_nonlin = np.polyval(coeffs, DN)
    # Control values outside the min/max range
    f_nonlin = np.where(DN < min_value, np.polyval(coeffs, min_value), f_nonlin)
    f_nonlin = np.where(DN > max_value, np.polyval(coeffs, max_value), f_nonlin)

    return f_nonlin


def make_fluxmap_image(f_map, bias, kgain, rn, emgain, time, coeffs, nonlin_flag=False,
        divide_em=False):
    """ 
    This function makes a SCI-sized frame with simulated noise and a fluxmap. It
    also performs bias-subtraction and division by EM gain if required. It is used
    in the unit tests test_nonlin.py and test_kgain_cal.py

    Args:
        f_map (np.array): fluxmap in e/s/px. Its size is 1024x1024 pixels.
        bias (float): bias value in electrons.
        kgain (float): value of K-Gain in electrons per DN.
        rn (float): read noise in electrons.
        emgain (float): calue of EM gain. 
        time (float):  exposure time in sec.
        coeffs (np.array): array of cubic polynomial coefficients from nonlin_coefs.
        nonlin_flag (bool): (Optional) if nonlin_flag is True, then nonlinearity is applied.
        divide_em (bool): if divide_em is True, then the emgain is divided
        
    Returns:
        corgidrp.data.Image
    """
    # Generate random values of rn in electrons from a Gaussian distribution
    random_array = np.random.normal(0, rn, (1200, 2200)) # e-
    # Generate random values from fluxmap from a Poisson distribution
    Poiss_noise_arr = emgain*np.random.poisson(time*f_map) # e-
    signal_arr = np.zeros((1200,2200))
    start_row = 10
    start_col = 1100
    signal_arr[start_row:start_row + Poiss_noise_arr.shape[0],
                start_col:start_col + Poiss_noise_arr.shape[1]] = Poiss_noise_arr
    temp = random_array + signal_arr # e-
    if nonlin_flag:
        temp2 = nonlin_factor(coeffs, signal_arr/kgain)
        frame = np.round((bias + random_array + signal_arr/temp2)/kgain) # DN
    else:
        frame = np.round((bias+temp)/kgain) # DN

    # Subtract bias and divide by EM gain if required. TODO: substitute by
    # prescan_biassub step function in l1_to_l2a.py and the em_gain_division
    # step function in l2a_to_l2b.py    
    offset_colroi1 = 799
    offset_colroi2 = 1000
    offset_colroi = slice(offset_colroi1,offset_colroi2)
    row_meds = np.median(frame[:,offset_colroi], axis=1)
    row_meds = row_meds[:, np.newaxis]
    frame -= row_meds
    if divide_em:
        frame = frame/emgain

    # TO DO: Determine what level this image should be
    prhd, exthd, errhdr, dqhdr, biashdr= create_default_L2b_headers()
    # Record actual commanded EM
    exthd['EMGAIN_C'] = emgain
    # Record actual exposure time
    exthd['EXPTIME'] = time
    # Mock error maps
    err = np.ones([1200,2200]) * 0.5
    dq = np.zeros([1200,2200], dtype = np.uint16)
    image = Image(frame, pri_hdr = prhd, ext_hdr = exthd, err = err,
        dq = dq)
    # Use a an expected filename
    visitid = prhd["VISITID"]
    base_time = datetime.datetime.now()
    time_str = data.format_ftimeutc(base_time.isoformat())
    image.filename = f"cgi_{visitid}_{time_str}_l2b.fits"
    return image

def create_astrom_data(field_path, filedir=None, image_shape=(1024, 1024), target=(80.553428801, -69.514096821), offset=(0,0), subfield_radius=0.03, platescale=21.8, rotation=45, add_gauss_noise=True, 
                       distortion_coeffs_path=None, dither_pointings=0, bpix_map=None, sim_err_map=False):
    """
    Create simulated data for astrometric calibration.

    Args:
        field_path (str): Full path to directory with test field data (ra, dec, vmag, etc.)
        filedir (str): (Optional) Full path to directory to save to. (default: None)
        image_shape (tuple of ints): The desired shape of the image (num y pixels, num x pixels), (default: (1024, 1024))
        target (tuple): The original pointing target in RA/DEC [deg] (default: (80.553428801, -69.514096821))
        offset (tuple): The RA/DEC [deg] injected offset from the target pointing (default: (0,0))
        subfield_radius (float): The radius [deg] around the target coordinate for creating a subfield to produce the image from (default: 0.03 [deg])
        platescale (float): The plate scale of the created image data (default: 21.8 [mas/pixel])
        rotation (float): The north angle of the created image data (default: 45 [deg])
        add_gauss_noise (boolean): Argument to determine if gaussian noise should be added to the data (default: True)
        distortion_coeffs_path (str): Full path to csv with the distortion coefficients and the order of polynomial used to describe distortion (default: None))
        dither_pointings (int): Number of dithers to include with the dataset. Dither offset is assumed to be half the FoV. (default: 0)
        bpix_map (np.array): 2D bad pixel map to apply to simulated data (default: None)
        sim_err_map (boolean): If True, simulates an error map (default: False) 

    Returns:
        corgidrp.data.Dataset:
            The simulated dataset

    """
    if type(field_path) != str:
        raise TypeError('field_path must be a str')

    # Make filedir if it does not exist
    if (filedir is not None) and (not os.path.exists(filedir)):
        os.mkdir(filedir)
    
    # hard coded image properties
    sim_data = np.zeros(image_shape)
    ny, nx = image_shape
    center = [nx //2, ny //2]
    fwhm = 3
    
    # load in the field data and restrict to 0.02 [deg] radius around target
    cal_field = ascii.read(field_path)
    subfield = cal_field[((cal_field['RA'] >= target[0] - subfield_radius) & (cal_field['RA'] <= target[0] + subfield_radius) & (cal_field['DEC'] >= target[1] - subfield_radius) & (cal_field['DEC'] <= target[1] + subfield_radius))]
    cal_SkyCoords = SkyCoord(ra= subfield['RA'], dec= subfield['DEC'], unit='deg', frame='icrs')  # save these subfield skycoords somewhere

    # create the simulated image header
    vert_ang = np.radians(rotation)
    pc = np.array([[-np.cos(vert_ang), np.sin(vert_ang)], [np.sin(vert_ang), np.cos(vert_ang)]])
    cdmatrix = pc * (platescale * 0.001) / 3600.

    new_hdr = {}
    new_hdr['CD1_1'] = cdmatrix[0,0]
    new_hdr['CD1_2'] = cdmatrix[0,1]
    new_hdr['CD2_1'] = cdmatrix[1,0]
    new_hdr['CD2_2'] = cdmatrix[1,1]

    new_hdr['CRPIX1'] = center[0]
    new_hdr['CRPIX2'] = center[1]

    new_hdr['CTYPE1'] = 'RA---TAN'
    new_hdr['CTYPE2'] = 'DEC--TAN'

    new_hdr['CDELT1'] = (platescale * 0.001) / 3600
    new_hdr['CDELT2'] = (platescale * 0.001) / 3600

    new_hdr['CRVAL1'] = target[0] + offset[0]
    new_hdr['CRVAL2'] = target[1] + offset[1]

    w = wcs.WCS(new_hdr)

    # create the image data
    xpix_full, ypix_full = wcs.utils.skycoord_to_pixel(cal_SkyCoords, wcs=w)

    frame_xpixels = []  # place to hold the source pixel locations for different frames
    frame_ypixels = []
    frame_ras = []      # place to hold matching ra/decs for guesses.csv file
    frame_decs = []
    frame_amps = []
    frame_mags = []
    frame_targs = []

    # compute pixel positions and sky locations for the undithered image
    pix_inds = np.where((xpix_full >= 0) & (xpix_full <= nx) & (ypix_full >= 0) & (ypix_full <= ny))[0]
    xpix = xpix_full[pix_inds]
    ypix = ypix_full[pix_inds]
    ras = cal_SkyCoords[pix_inds]
    decs = cal_SkyCoords[pix_inds]
    mags = subfield['VMAG'][pix_inds]
    amplitudes = np.power(10, ((mags - 22.5) / (-2.5))) * 10 

    frame_xpixels.append(np.array(xpix))    # add pixel locations to all frame list
    frame_ypixels.append(np.array(ypix))
    frame_ras.append(ras)
    frame_decs.append(decs)
    frame_amps.append(np.array(amplitudes))
    frame_mags.append(np.array(mags))
    frame_targs.append(np.array(target))

    # find the dither RA/DEC pointings (assume we know this)
    # one FoV roughly translates to 
    ra_fov = 0.01741774460001011  #[deg]
    dec_fov = 0.00617760699999792  #[deg]
    ## assume the target coord has moved by half ra/dec fov based on direction
    dither_target_ras = [target[0], target[0], target[0]+(ra_fov/2), target[0]-(ra_fov/2)]
    dither_target_decs = [target[1]+(dec_fov/2), target[1]-(dec_fov/2), target[1], target[1]]


    # create dithered images if dither_pointings > 0
    for i in range(dither_pointings):

        # simulate header with same image properties but around the dither target coord
        new_hdr = {}
        new_hdr['CD1_1'] = cdmatrix[0,0]
        new_hdr['CD1_2'] = cdmatrix[0,1]
        new_hdr['CD2_1'] = cdmatrix[1,0]
        new_hdr['CD2_2'] = cdmatrix[1,1]
        
        new_hdr['CRPIX1'] = center[0]
        new_hdr['CRPIX2'] = center[1]
        
        new_hdr['CTYPE1'] = 'RA---TAN'
        new_hdr['CTYPE2'] = 'DEC--TAN'
        
        new_hdr['CDELT1'] = (platescale * 0.001) / 3600
        new_hdr['CDELT2'] = (platescale * 0.001) / 3600
        
        new_hdr['CRVAL1'] = dither_target_ras[i] + offset[0]
        new_hdr['CRVAL2'] = dither_target_decs[i] + offset[1]
        
        w = wcs.WCS(new_hdr)
        
        # create the image data
        xpix_full, ypix_full = wcs.utils.skycoord_to_pixel(cal_SkyCoords, wcs=w)

        dither_inds = np.where((xpix_full >= 0) & (xpix_full <= 1024) & (ypix_full >= 0) & (ypix_full <= 1024))[0]

        dxpix = xpix_full[dither_inds]
        dypix = ypix_full[dither_inds]
        dras = cal_SkyCoords[dither_inds]
        ddecs = cal_SkyCoords[dither_inds]
        dmags = subfield['VMAG'][dither_inds]
        damplitudes = np.power(10, ((dmags - 22.5) / (-2.5))) * 10
   
        frame_xpixels.append(np.array(dxpix))
        frame_ypixels.append(np.array(dypix))
        frame_ras.append(dras)
        frame_decs.append(ddecs)
        frame_amps.append(np.array(damplitudes))
        frame_mags.append(np.array(dmags))
        frame_targs.append(np.array([dither_target_ras[i], dither_target_decs[i]]))


    # create a place to save the image frames
    image_frames = []

    for i, (xp, yp, amps) in enumerate(zip(frame_xpixels, frame_ypixels, frame_amps)):
        sim_data = np.zeros(image_shape)

        # inject gaussian psf stars
        for xpos, ypos, amplitude in zip(xp, yp, amps):  
            stampsize = int(np.ceil(3 * fwhm))
            sigma = fwhm/ (2.*np.sqrt(2*np.log(2)))
            
            # coordinate system
            y, x = np.indices([stampsize, stampsize])
            y -= stampsize // 2
            x -= stampsize // 2
            
            # find nearest pixel
            x_int = int(xpos)
            y_int = int(ypos)
            x += x_int
            y += y_int
            
            xmin = x[0][0]
            xmax = x[-1][-1]
            ymin = y[0][0]
            ymax = y[-1][-1]
            
            psf = amplitude * np.exp(-((x - xpos)**2. + (y - ypos)**2.) / (2. * sigma**2))

            # crop the edge of the injection at the edge of the image
            if xmin <= 0:
                psf = psf[:, -xmin:]
                xmin = 0
            if ymin <= 0:
                psf = psf[-ymin:, :]
                ymin = 0
            if xmax >= nx:
                psf = psf[:, :-(xmax-nx + 1)]
                xmax = nx - 1
            if ymax >= ny:
                psf = psf[:-(ymax-ny + 1), :]
                ymax = ny - 1

            # inject the stars into the image
            sim_data[ymin:ymax + 1, xmin:xmax + 1] += psf

        if add_gauss_noise:
            # add Gaussian random noise
            noise_rng = np.random.default_rng(10)
            gain = 1
            ref_flux = 10
            noise = noise_rng.normal(scale= ref_flux/gain * 0.1, size= image_shape)
            sim_data = sim_data + noise
            
        if sim_err_map:
            
            # Create an error map estimating the measurement noise to be about 5% of the flux. Rather arbitrary values, feel free to change.
            err_rng = np.random.default_rng(10)
            err_map = err_rng.normal(loc=sim_data*0.05, scale=1, size=sim_data.shape)
      
        # add distortion (optional)
        if distortion_coeffs_path is not None:
            # load in distortion coeffs and fitorder
            coeff_data = np.genfromtxt(distortion_coeffs_path, delimiter=',', dtype=None)
            fitorder = int(coeff_data[-1])

            # convert legendre polynomials into distortin maps in x and y 
            yorig, xorig = np.indices(image_shape)
            y0, x0 = image_shape[0]//2, image_shape[1]//2
            yorig -= y0
            xorig -= x0

            # get the number of fitting params from the order
            fitparams = (fitorder + 1)**2
            
            # reshape the coeff arrays
            best_params_x = coeff_data[:fitparams]
            best_params_x = best_params_x.reshape(fitorder+1, fitorder+1)
            
            total_orders = np.arange(fitorder+1)[:,None] + np.arange(fitorder+1)[None, :]
            
            best_params_x = best_params_x / 500**(total_orders)

            # evaluate the polynomial at all pixel positions
            x_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), best_params_x)
            x_corr = x_corr.reshape(xorig.shape)
            distmapx = x_corr - xorig
            
            # reshape and evaluate the same for y
            best_params_y = coeff_data[fitparams:-1]
            best_params_y = best_params_y.reshape(fitorder+1, fitorder+1)
        
            best_params_y = best_params_y / 500**(total_orders)

            # evaluate the polynomial at all pixel positions
            y_corr = np.polynomial.legendre.legval2d(xorig.ravel(), yorig.ravel(), best_params_y)
            y_corr = y_corr.reshape(yorig.shape)
            distmapy = y_corr - yorig

            ## distort image based on coeffs
            if (nx >= ny): imgsize = nx
            else: imgsize = ny

            gridx, gridy = np.meshgrid(np.arange(imgsize), np.arange(imgsize))
            gridx = gridx + distmapx
            gridy = gridy + distmapy

            sim_data = scipy.ndimage.map_coordinates(sim_data, [gridy, gridx])
            
            if sim_err_map:
                # transform the error map
                err_map = scipy.ndimage.map_coordinates(err_map, [gridy, gridx])
            
            # translated_pix = scipy.ndimage.map_coordinates()
            # transform the source coordinates
            dist_xpix, dist_ypix = [], []
            for (x,y) in zip(xp, yp):
                x_new = x - distmapx[int(y)][int(x)]
                y_new = y - distmapy[int(y)][int(x)]

                dist_xpix.append(x_new)
                dist_ypix.append(y_new)

            frame_xpixels[i] = np.array(dist_xpix)
            frame_ypixels[i] = np.array(dist_ypix)
            
        # apply bad pixel map if provided (optional)
        if bpix_map is not None:
            if bpix_map.shape[0] == 3:
                frame_bpix = bpix_map[i]
                sim_data[frame_bpix.astype(bool)] = np.nan 
                if sim_err_map:
                    err_map[frame_bpix.astype(bool)] = np.nan
                dq_map = frame_bpix
            else:
                sim_data[bpix_map.astype(bool)] = np.nan
                if sim_err_map:
                    err_map[bpix_map.astype(bool)] = np.nan
                dq_map = bpix_map
            
        # image_frames.append(sim_data)

        # TO DO: Determine what level this image should be
        prihdr, exthdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
        prihdr['VISTYPE'] = 'CGIVST_CAL_BORESIGHT'
        prihdr['RA'] = np.array(frame_targs).T[0][i]  # assume we will know something about the dither RA/DEC pointing
        prihdr['DEC'] = np.array(frame_targs).T[1][i]
        prihdr['ROLL'] = 0   ## assume a telescope roll = 0 for now

        ## save as an Image object
        err_map = None if not sim_err_map else err_map
        dq_map = None if bpix_map is None else dq_map
        frame = data.Image(sim_data, pri_hdr= prihdr, ext_hdr= exthdr, err=err_map, dq=dq_map)
        # Generate unique, properly formatted filename
        visitid = prihdr["VISITID"]
        base_time = datetime.datetime.now()
        time_str = data.format_ftimeutc(base_time.isoformat())
        filename = f"cgi_{visitid}_{time_str}_l2b.fits"
        frame.filename = filename
        
        if filedir is not None:
            # save source SkyCoord locations and pixel location estimates
            guess = Table()
            guess['x'] = frame_xpixels[i]
            guess['y'] = frame_ypixels[i]
            guess['RA'] = frame_ras[i].ra
            guess['DEC'] = frame_decs[i].dec
            guess['VMAG'] = frame_mags[i]
            # guessname = "guesses.csv"
            ascii.write(guess, filedir+'/guesses'+str(i)+'.csv', overwrite=True)

            frame.save(filedir=filedir, filename=filename)

        image_frames.append(frame)

    # frames.append(frame)
    dataset = data.Dataset(image_frames)

    return dataset


def create_not_normalized_dataset(filedir=None, numfiles=10):
    """
    Create simulated data not normalized for the exposure time.

    Args:
        filedir (str): (Optional) Full path to directory to save to.
        numfiles (int): Number of files in dataset. Default is 10.

    Returns:
        corgidrp.data.Dataset:
            the simulated dataset
    """
    frames = []
    for i in range(numfiles):
        # TO DO: Determine what level this image should be
        prihdr, exthdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()

        sim_data = np.asarray(np.random.poisson(lam=150.0, size=(1024,1024)), dtype=float)
        sim_err = np.asarray(np.random.poisson(lam=1.0, size=(1024,1024)), dtype=float)
        sim_dq = np.asarray(np.zeros((1024,1024)), dtype=int)
        frame = data.Image(sim_data, pri_hdr=prihdr, ext_hdr=exthdr, err=sim_err, dq=sim_dq, err_hdr = errhdr, dq_hdr = dqhdr)
        # frame = data.Image(sim_data, pri_hdr = prihdr, ext_hdr = exthdr, err = sim_err, dq = sim_dq)
        if filedir is not None:
            # Generate CGI filename with incrementing datetime
            visitid = prihdr["VISITID"]
            base_time = datetime.datetime.now()
            time_offset = datetime.timedelta(seconds=i)
            unique_time = base_time + time_offset
            time_str = data.format_ftimeutc(unique_time.isoformat())
            filename = f"cgi_{visitid}_{time_str}_l2b.fits"
            frame.save(filedir=filedir, filename=filename)
        frames.append(frame)
    dataset = data.Dataset(frames)

    return dataset


def generate_mock_pump_trap_data(output_dir,meta_path, EMgain=10, 
                                 read_noise = 100, eperdn = 6, e2emode=False, 
                                 nonlin_path=None, arrtype='SCI'):
    """
    Generate mock pump trap data, save it to the output_directory
    
    Args:
        output_dir (str): output directory
        meta_path (str): metadata path
        EMgain (float): desired EM gain for frames
        read_noise (float): desired read noise for frames
        eperdn (float):  desired k gain (e-/DN conversion factor)
        e2emode (bool):  If True, e2e simulated data made instead of data for the unit test.  
            Difference b/w the two: 
            This e2emode data differs from the data generated when e2emode is False in the following ways:
            -The bright pixel of each trap is simulated in a more realistic way (i.e., at every phase time frame).
            -Simulated readout is more realistic (read noise, EM gain, k gain, nonlinearity, bias invoked after traps simulated).  
            In the other dataset (when e2emode is False), readout was simulated before traps were added, and no nonlinearity was applied.  
            Also, the number of electrons in the dark pixels of the dipoles can no longer be negative, and this condition is enforced.
            -The number of pumps and injected charge are much higher in these frames so that traps stand out above the read noise.  
            This was not an issue in the other dataset since read noise was added to frames that were EM-gained before charge was injected, which suppressed the effective read noise.
            -The EM gain used is 1.5.  For a large injected charge amount, the EM gain cannot be very high because of the risk of saturation.  
            -The number of phase times is 10 per scheme, to reduce the dataset size (compared to 100 when e2emode is False).
            -The frame format is ENG, as real trap-pump data is.
        nonlin_path (str): Path of nonlinearity correction file to use.  
            The inverse is applied, implementing rather than correcting nonlinearity.  
            If None, no nonlinearity is applied.  Defaults to None.
        arrtype (str): array type (for this function, choice of 'SCI' or 'ENG')
    """

    #If output_dir doesn't exist then make it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # here = os.path.abspath(os.path.dirname(__file__))
    # meta_path = Path(here, '..', 'util', 'metadata_test.yaml')
    #meta_path = Path(here, '..', 'util', 'metadata.yaml')
    meta = MetadataWrapper(meta_path)
    num_pumps = 10000
    multiple = 1
    #nrows, ncols, _ = meta._imaging_area_geom()
    # the way emccd_detect works is that it takes an input for the selected
    # image area within the viable CCD pixels, so my input here must be that
    # smaller size (below) as opposed to the full useable CCD pixel size
    # (commented out above)
    nrows, ncols, _ = meta._unpack_geom('image')
    #EM gain
    g = EMgain
    cic = 200  
    rn = read_noise 
    dc = {180: 0.163, 190: 0.243, 200: 0.323, 210: 0.403,
          220: 0.483}
    # dc = {180: 0, 190: 0, 195: 0, 200: 0, 210: 0, 220: 0}
    bias = 1000 
    inj_charge = 500 # 0
    full_well_image=50000.  # e-
    full_well_serial=50000.
    # trap-pumping done when CGI is secondary instrument (i.e., dark):
    fluxmap = np.zeros((nrows, ncols))
    # frametime for pumped frames: 1000ms, or 1 s
    frametime = 1
    # set these to have no effect, then use these with their input values at the end
    later_eperdn = eperdn
    if e2emode: 
        eperdn = 1
        cic = 0.02
        num_pumps = 50000 #120000#90000#15000#5000
        inj_charge = 27000 #31000#70000#45000#8000 #num_pumps/2 # more than num_pumps/4, so no mean_field input needed
        multiple = 1
        g = 1
        rn = 0
        bias = 0
        full_well_image=105000.  # e-
        full_well_serial=105000.
        phase_times = 10
    bias_dn = bias/eperdn
    nbits = 14 #1
    
    def _ENF(g, Nem):
        """
        Returns the ENF.

        Args:
            g (float): gain
            Nem (int): Nem

        Returns:
            float: ENF

        """
        return np.sqrt(2*(g-1)*g**(-(Nem+1)/Nem) + 1/g)
    # std dev in e-, before gain division
    std_dev = np.sqrt(100**2 + _ENF(g,604)**2*g**2*(cic+ 1*dc[220]))
    fit_thresh = 3 #standard deviations above mean for trap detection
    #Offset ensures detection.  Physically, shouldn't have to add offset to
    #frame to meet threshold for detection, but in a small-sized frame, the
    # addition of traps increases the std dev a lot
    # (gain divided, w/o offset: from 22 e- before traps to 73 e- after adding)
    # If I run code with lower threshold, though, I can do an offset of 0.
    # For regular full-sized, definitely shouldn't have to add in offset.
    # Also, a trap can't capture more than mean per pixel e-, which is 200e-
    # in this case.  So max amp P1 trap will not be 2500e- but rather the
    #mean e- per pixel!  But this discrepancy doesn't affect validity of tests.

    offset_u = 0
    # offset_u = (bias_dn + ((cic+1*dc[220])*g + fit_thresh*std_dev/g)/eperdn+\
    #    inj_charge/eperdn)
    # #offset_l = bias_dn + ((cic+1*dc[220])*g - fit_thresh*std_dev/g)/eperdn
    # gives these 0 offset in the function (which gives e-), then add it in
    # by hand and convert to DN
    # and I increase dark current with temp linearly (even though it should
    # be exponential, but dc really doesn't affect anything here)
    emccd = {}
    # leaving out 170K
    #170K: gain of 10-20; gives g*CIC ~ 2000 e-
    # emccd[170] = EMCCDDetect(
    #         em_gain=1,#10,
    #         full_well_image=50000.,  # e-
    #         full_well_serial=50000.,  # e-
    #         dark_current=0.083,  # e-/pix/s
    #         cic=200, # e-/pix/frame; lots of CIC from all the prep clocking
    #         read_noise=100.,  # e-/pix/frame
    #         bias=bias,  # e-
    #         qe=0.9,
    #         cr_rate=0.,  # hits/cm^2/s
    #         pixel_pitch=13e-6,  # m
    #         eperdn=7.,
    #         nbits=14,
    #         numel_gain_register=604,
    #         meta_path=meta_path
    #    )
    #180K: gain of 10-20
    emccd[180] = EMCCDDetect(
            em_gain=g,#10,
            full_well_image=full_well_image,  # e-
            full_well_serial=full_well_serial,  # e-
            dark_current=dc[180], #0.163,  # e-/pix/s
            cic=cic, # e-/pix/frame; lots of CIC from all the prep clocking
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #190K: gain of 10-20
    emccd[190] = EMCCDDetect(
            em_gain=g,#10,
            full_well_image=full_well_image,  # e-
            full_well_serial=full_well_serial,  # e-
            dark_current= dc[190],#0.243,  # e-/pix/s
            cic=cic, # e-/pix/frame
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #195K: gain of 10-20
    # emccd[195] = EMCCDDetect(
    #         em_gain=g,#10,
    #         full_well_image=50000.,  # e-
    #         full_well_serial=50000.,  # e-
    #         dark_current= dc[195],#0.263,  # e-/pix/s
    #         cic=cic, # e-/pix/frame
    #         read_noise=rn,  # e-/pix/frame
    #         bias=bias,  # e-
    #         qe=0.9,
    #         cr_rate=0.,  # hits/cm^2/s
    #         pixel_pitch=13e-6,  # m
    #         eperdn=eperdn,
    #         nbits=nbits,
    #         numel_gain_register=604,
    #         meta_path=meta_path
    #     )
    #200K: gain of 10-20
    emccd[200] = EMCCDDetect(
            em_gain=g,#10,
            full_well_image=full_well_image,  # e-
            full_well_serial=full_well_serial,  # e-
            dark_current=dc[200], #0.323,  # e-/pix/s
            cic=cic, # e-/pix/frame
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #210K: gain of 10-20
    emccd[210] = EMCCDDetect(
            em_gain=g, #10,
            full_well_image=full_well_image,  # e-
            full_well_serial=full_well_serial,  # e-
            dark_current=dc[210], #0.403,  # e-/pix/s
            cic=cic, # e-/pix/frame
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #220K: gain of 10-20
    emccd[220] = EMCCDDetect(
            em_gain=g, #10,
            full_well_image=full_well_image,  # e-
            full_well_serial=full_well_serial,  # e-
            dark_current=dc[220], #0.483,  # e-/pix/s
            cic=cic, # e-/pix/frame; divide by 15 to get the same 1000
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )

    #when tauc is 3e-3, that gives a mean e- field of 2090 e-
    tauc = 1e-8 #3e-3
    tauc2 = 1.2e-8 # 3e-3
    tauc3 = 1e-8 # 3e-3
    # tried for mean field test, but gave low amps that got lost in noise
    tauc4 = 1e-3 #constant Pc over time not a great approximation in theory
    #In order of amplitudes overall (given comparable tau and tau2):
    # P1 biggest, then P3, then P2
    # E,E3 and cs,cs3 params below chosen to ensure a P1 trap found at its
    # peak amp for good eperdn determination
    # E3,cs3: will give tau outside of 1e-6,1e-2
    # for all temps except 220K; we'll just make sure it's present in all
    # scheme 1 stacks for all temps to ensure good eperdn for all temps;
    # E, cs: will give tau outside of 1e-6, 1e-2
    # for just 170K, which I took out of temp_data
    # E2, cs2: fine for all temps
    E = 0.32 #eV
    E2 = 0.28 #0.24 # eV
    E3 = 0.4 #eV
    # tried mean field test (gets tau = 1e-4 for 180K)
    E4 = 0.266 #eV
    cs = 2 #in 1e-19 m^2
    cs2 = 12 #3 #8 # in 1e-19 m^2
    cs3 = 2 # in 1e-19 m^2
    # for mean field test
    cs4 = 4 # in 1e-19 m^2
    #temp_data = np.array([170, 180, 190, 200, 210, 220])
    temp_data = np.array([180, 190, 195, 200, 210, 220])
    #temp_data = np.array([180])
    taus = {}
    taus2 = {}
    taus3 = {}
    taus4 = {}
    for i in temp_data:
        taus[i] = tau_temp(i, E, cs)
        taus2[i] = tau_temp(i, E2, cs2)
        taus3[i] = tau_temp(i, E3, cs3)
        taus4[i] = tau_temp(i, E4, cs4)
    #tau = 7.5e-3
    #tau2 = 8.8e-3
    if e2emode:
        time_data = (np.logspace(-6, -2, phase_times))*10**6 # in us 
    else:
        time_data = (np.logspace(-6, -2, 100))*10**6 # in us 
    #time_data = (np.linspace(1e-6, 1e-2, 50))*10**6 # in us
    time_data = time_data.astype(float)
    # make one phase time a repitition
    time_data[-1] = time_data[-2]
    time_data = np.array(time_data.tolist()*multiple)
    time_data_s = time_data/10**6 # in s
    # half the # of frames for length limit
    length_limit = 5 #int(np.ceil((len(time_data)/2)))
    # mean of these frames will be a bit more than 2000e-, which is gain*CIC
    # std dev: sqrt(rn^2 + ENF^2 * g^2(e- signal))

    # with offset_u non-zero in below, I expect to get eperdn 4.7 w/ the code
    amps1 = {}; amps2 = {}; amps3 = {}
    amps1_k = {}; amps1_tau2 = {}; amps3_tau2 = {}; amps1_mean_field = {}
    amps2_mean_field = {}
    amps11 = {}; amps12 = {}; amps22 = {}; amps23 = {}; amps33 = {}; amps21 ={}
    for i in temp_data:
        amps1[i] = offset_u + g*P1(time_data_s, 0, tauc, taus[i], num_pumps)/eperdn
        amps11[i] = offset_u + g*P1_P1(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_pumps)/eperdn
        amps2[i] = offset_u + g*P2(time_data_s, 0, tauc, taus[i], num_pumps)/eperdn
        amps12[i] = offset_u + g*P1_P2(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_pumps)/eperdn
        amps22[i] = offset_u + g*P2_P2(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_pumps)/eperdn
        amps3[i] = offset_u + g*P3(time_data_s, 0, tauc, taus[i], num_pumps)/eperdn
        amps33[i] = offset_u + g*P3_P3(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_pumps)/eperdn
        amps23[i] = offset_u + g*P2_P3(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i], num_pumps)/eperdn
        # just for (98,33)
        amps21[i] =  offset_u + g*P1_P2(time_data_s, 0, tauc2, taus2[i],
            tauc, taus[i], num_pumps)/eperdn
        # now a special amps just for ensuring good eperdn determination
        # actually, doesn't usually meet trap_id thresh, but no harm
        # including it
        amps1_k[i] = offset_u + g*P1(time_data_s, 0, tauc3, taus3[i], num_pumps)/eperdn
        # for the case of (89,2) with a single trap with tau2
        amps1_tau2[i] = offset_u + g*P1(time_data_s, 0, tauc2, taus2[i], num_pumps)/eperdn
        # for the case of (77,90) with a single trap with tau2
        amps3_tau2[i] = offset_u + g*P3(time_data_s, 0, tauc2, taus2[i], num_pumps)/eperdn
        #amps1_k[i] = g*2500/eperdn
        # make a trap for the mean_field test (when mean field=400e- < 2500e-)
        #this trap peaks at 250 e-
        amps1_mean_field[i] = offset_u + \
            g*P1(time_data_s,0,tauc4,taus4[i], num_pumps)/eperdn
        amps2_mean_field[i] = offset_u + \
            g*P2(time_data_s,0,tauc4,taus4[i], num_pumps)/eperdn
    amps_1_trap = {1: amps1, 2: amps2, 3: amps3, 'sp': amps1_k,
            '1b': amps1_tau2, '3b': amps3_tau2, 'mf1': amps1_mean_field,
            'mf2': amps2_mean_field}
    amps_2_trap = {11: amps11, 12: amps12, 21: amps21, 22: amps22, 23: amps23,
        33: amps33}

    #r0c0[0]: starting row for imaging area (physical CCD pixels)
    #r0c0[1]: starting col for imaging area (physical CCD pixels)
    _, _, r0c0 = meta._imaging_area_geom()

    def add_1_dipole(img_stack, row, col, ori, prob, start, end, temp):
        """Adds a dipole to an image stack img_stack at the location of the
        bright pixel given by row and col (relative to image area coordinates)
        that is of orientation 'above' or
        'below' (specified by ori) for a number of unique phase times
        going from start to end (inclusive; don't use -1 for end; 0 for start
        means first frame, length of time array means last frame), and the
        dipole is of the probability function prob (which can be 1, 2, 3,
        'sp', '1b', '3b', 'mf1', or 'mf2').
        The temperature is specified by temp (in K).
        
        When e2emode is True, the amount subtracted from the dark pixel and added to the bright 
        pixel of a given dipole is constrained so that a pixel is not left with a negative number of electrons. 
        See doc string of generate_mock_pump_trap_data for full e2emode details.

        Args: 
            img_stack (np.array): image stack
            row (int): row
            col (int): col
            ori (str): orientation
            prob (int): probability
            start (int): start
            end (int): end
            temp (int): temperature

        Returns:
            np.array: image stack
        """
        # length limit controlled by how 'long' deficit pixel is since
        #threshold should be met for all frames for bright pixel
        if ori == 'above':
            #img_stack[start:end,r0c0[0]+row+1,r0c0[1]+col] = offset_l
            region = img_stack[start:end,r0c0[0]+row+1,r0c0[1]+col]
            region_c = img_stack[start:end,r0c0[0]+row+1,r0c0[1]+col].copy()
        if ori == 'below':
            #img_stack[start:end,r0c0[0]+row-1,r0c0[1]+col] = offset_l
            region = img_stack[start:end,r0c0[0]+row-1,r0c0[1]+col]
            region_c = img_stack[start:end,r0c0[0]+row-1,r0c0[1]+col].copy()
        region -= amps_1_trap[prob][temp][start:end]
        if e2emode:
            # can't draw more e- than what's there
            neg_inds = np.where(region < 0)
            good_inds = np.where(region >= 0)
            if neg_inds[0].size > 0:
                print(neg_inds[0].size)
                pass
            region[neg_inds[0]] = 0
            img_stack[start:end,r0c0[0]+row,r0c0[1]+col][good_inds[0]] += amps_1_trap[prob][temp][start:end][good_inds[0]]
            img_stack[start:end,r0c0[0]+row,r0c0[1]+col][neg_inds[0]] += region_c[neg_inds[0]]
        else:
            img_stack[: ,r0c0[0]+row,r0c0[1]+col] += amps_1_trap[prob][temp][:]

        return img_stack

    def add_2_dipole(img_stack, row, col, ori1, ori2, prob, start1, end1,
        start2, end2, temp):
        """Adds a 2-dipole to an image stack img_stack at the location of the
        bright pixel given by row and col (relative to image area coordinates)
        that is of orientation 'above' or
        'below' (specified by ori1 and ori2).  The 1st dipole is for a number
        of unique phase times going from start1 to end1, and
        the 2nd dipole starts from start2 and ends at end2 (inclusive; don't
        use -1 for end; 0 for start means first frame, length of time array
        means last frame). The 2-dipole is of probability function
        prob.  Valid values for prob are 11, 12, 22, 23, and 33.
        The temperature is specified by temp (in K).

        When e2emode is True, the amount subtracted from the dark pixel and added to the bright 
        pixel of a given dipole is constrained so that a pixel is not left with a negative number of electrons. 
        Also, start2:end2 should not overlap with start1:end1, and the ranges should 
        cover the whole 0:10 frames.  This condition allows for the simulation of the probability 
        distribution across all phase times.
        See doc string of generate_mock_pump_trap_data for full e2emode details.
        
        Args:
            img_stack (np.array): image stack
            row (int): row
            col (int): col
            ori1 (str): orientation 1
            ori2 (str): orientation 2
            prob (int): probability
            start1 (int): start 1
            end1 (int): end 1
            start2 (int): start 2
            end2 (int): end 2  
            temp (int): temperature

        Returns:
            np.array: image stack    
        """
        # length limit controlled by how 'long' deficit pixel is since
        #threshold should be met for all frames for bright pixel
        if ori1 == 'above':
            region1 = img_stack[start1:end1,r0c0[0]+row+1,r0c0[1]+col]
            region1_c = img_stack[start1:end1,r0c0[0]+row+1,r0c0[1]+col].copy()
            #img_stack[start1:end1,r0c0[0]+row+1,r0c0[1]+col] = offset_l
        if ori1 == 'below':
            #img_stack[start1:end1,r0c0[0]+row-1,r0c0[1]+col] = offset_l
            region1 = img_stack[start1:end1,r0c0[0]+row-1,r0c0[1]+col] 
            region1_c = img_stack[start1:end1,r0c0[0]+row-1,r0c0[1]+col].copy()
        if ori2 == 'above':
            #img_stack[start2:end2,r0c0[0]+row+1,r0c0[1]+col] = offset_l
            region2 = img_stack[start2:end2,r0c0[0]+row+1,r0c0[1]+col]
            region2_c = img_stack[start2:end2,r0c0[0]+row+1,r0c0[1]+col].copy()
        if ori2 == 'below':
            region2 = img_stack[start2:end2,r0c0[0]+row-1,r0c0[1]+col]
            region2_c = img_stack[start2:end2,r0c0[0]+row-1,r0c0[1]+col].copy()
        # technically, should subtract 1 prob distribution at at time (amps_1_trap), but I'm just subtracting 
        # a bit more than I'm supposed to, and doesn't matter too much since these 
        # are the deficit pixels (or pixel) next to the bright pixel, which is what counts for doing fits
        region1 -= amps_2_trap[prob][temp][start1:end1]
        region2 -= amps_2_trap[prob][temp][start2:end2]
        if e2emode:
            # can't draw more e- than what's there
            neg_inds1 = np.where(region1 < 0)
            if neg_inds1[0].size > 0:
                print(neg_inds1[0].size)
                pass
            good_inds1 = np.where(region1 >= 0)
            region1[neg_inds1] = 0
            img_stack[start1:end1,r0c0[0]+row,r0c0[1]+col][good_inds1[0]] += amps_2_trap[prob][temp][start1:end1][good_inds1[0]]
            img_stack[start1:end1,r0c0[0]+row,r0c0[1]+col][neg_inds1[0]] += region1_c[neg_inds1[0]]
        
            # can't draw more e- than what's there
            neg_inds2 = np.where(region2 < 0)
            if neg_inds2[0].size > 0:
                print(neg_inds2[0].size)
                pass
            good_inds2 = np.where(region2 >= 0)
            region2[neg_inds2] = 0
            img_stack[start2:end2,r0c0[0]+row,r0c0[1]+col][good_inds2[0]] += amps_2_trap[prob][temp][start2:end2][good_inds2[0]]
            img_stack[start2:end2,r0c0[0]+row,r0c0[1]+col][neg_inds2[0]] += region2_c[neg_inds2[0]]
        
        else:
            img_stack[:,r0c0[0]+row,r0c0[1]+col] += amps_2_trap[prob][temp][:]
        # technically, if there is overlap b/w start1:end1 and start2:end2,
        # then you are physically causing too big of a deficit since you're
        # saying more emitted than the amount captured in bright pixel, so
        # avoid this
        return img_stack

    def make_scheme_frames(emccd_inst, phase_times = time_data,
        inj_charge = inj_charge ):
        """Makes a series of frames according to the emccd_detect instance
        emccd_inst, one for each element in the array phase_times (assumed to
        be in s).

        Args:
            emccd_inst (EMCCDDetect): emccd instance
            phase_times (np.array): phase times
            inj_charge (int): injection charge

        Returns:
            np.array: full frames
        """
        full_frames = []
        for i in range(len(phase_times)):
            full = (emccd_inst.sim_full_frame(fluxmap,frametime)).astype(float)
            full_frames.append(full)
        # inj charge is before gain, but since it has no variance,
        # g*0 = no noise from this
        full_frames = np.stack(full_frames)
        # lazy and not putting in the last image row and col, but doesn't
        #matter since I only use prescan and image areas
        # add to just image area so that it isn't wiped with bias subtraction
        full_frames[:,r0c0[0]:,r0c0[1]:] += inj_charge
        return full_frames

    def add_defect(sch_imgs, prob, ori, temp):
        """Adds to all frames of an image stack sch_imgs a defect area with
        local mean above image-area mean such that a
        dipole in that area that isn't detectable unless ill_corr is True.
        The dipole is a single trap with orientation
        ori ('above' or 'below') and is of probability function prob
        (can be 1, 2, or 3).  The temperature is specified by temp (in K).

        Note: If a defect region is arbitrarily small (e.g., a 2x2 region of
        very bright pixels hiding a trap dipole), that trap simply will not
        be found since the illumination correction bin size is not allowed to
        be less than 5.  In v2.0, a moving median subtraction can be
        implemented that would be more likely to catch cases similar to that.
        However, physically, a defect region of such a small number of rows is
        improbable; even a cosmic ray hit, which could have this signature for
        perhaps 1 phase time, is very unlikely to hit the same region while
        data for each phase time is being taken.
        
        When e2emode is True, the amount subtracted from the dark pixel and added to the bright 
        pixel of a given dipole is constrained so that a pixel is not left with a negative number of electrons. 
        This condition allows for the simulation of the probability 
        distribution across all phase times.
        See doc string of generate_mock_pump_trap_data for full e2emode details.

        Args: 
            sch_imgs (np.array): scheme images
            prob (int): probability
            ori (str): orientation
            temp (int): temperature

            
        Returns:
            np.array: scheme images
            
        """
        # area with defect (high above mean),
        # but no dipole that stands out enough without ill_corr = True
        amount = 9000
        if e2emode:
            amount = inj_charge*2
        sch_imgs[:,r0c0[0]+12:r0c0[0]+22,r0c0[1]+17:r0c0[1]+27]=g*amount/eperdn
        # now a dipole that meets threshold around local mean doesn't meet
        # threshold around frame mean; would be detected only after
        # illumination correction
        if ori == 'above':
            region = sch_imgs[:,r0c0[0]+13+1, r0c0[1]+21] 
            region_c = region.copy()
        if ori == 'below':
            region = sch_imgs[:,r0c0[0]+13-1, r0c0[1]+21] 
            region_c = region.copy()
                # 2*offset_u - fit_thresh*std_dev/eperdn
        region -= amps_1_trap[prob][temp][:]
        if e2emode: # realistic handling:  can't trap more charge than what's there in a pixel
            neg_inds = np.where(region < 0)
            if neg_inds[0].size > 0:
                print(neg_inds[0].size)
            good_inds = np.where(region >= 0)
            region[neg_inds[0]] = 0
            sch_imgs[good_inds[0],r0c0[0]+13, r0c0[1]+21] += amps_1_trap[prob][temp][good_inds[0]]
            sch_imgs[neg_inds[0],r0c0[0]+13,r0c0[1]+21] += region_c[neg_inds[0]]
        else:
            sch_imgs[:,r0c0[0]+13, r0c0[1]+21] += amps_1_trap[prob][temp][:]

        return sch_imgs
    
    #initializing
    sch = {1: None, 2: None, 3: None, 4: None}
    #temps = {170: sch, 180: sch, 190: sch, 200: sch, 210: sch, 220: sch}
    # change from last iteration: make copies of sch below b/c make_scheme_frames() below was changing sch present in 
    # EVERY temp for every iteration in the temps for loop; however, no actual change in the output since 
    # the output .fits files were saved before the next iteration's make_scheme_frames() is called. So, Max's
    # unit test is unchanged. 
    temps = {180: sch, 190: sch.copy(), 200: sch.copy(), 210: sch.copy(), 220: sch.copy()}
    #temps = {180: sch}

    # first, get rid of files already existing in the folders where I'll put
    # the simulated data
    # for temp in temps.keys():
    #     for sch in [1,2,3,4]:
    #         curr_sch_dir = Path(here, 'test_data_sub_frame_noise', str(temp)+'K',
    #             'Scheme_'+str(sch))
    #         for file in os.listdir(curr_sch_dir):
    #             os.remove(Path(curr_sch_dir, file))

    for temp in temps.keys():
        for sc in [1,2,3,4]:
            temps[temp][sc] = make_scheme_frames(emccd[temp])
        # 14 total traps (15 with the (13,19) defect trap); at least 1 in every
        # possible sub-electrode location
        # careful not to add traps in defect region; do that with add_defect()
        # careful not to add, e.g., bright pixel of one trap in the deficit
        # pixel of another trap since that would negate the original trap

        # add in 'LHSel1' trap in midst of defect for all phase times
        # (only detectable with ill_corr)
        add_defect(temps[temp][1], 1, 'below', temp)
        add_defect(temps[temp][3], 3, 'below', temp)
        #this defect was used for k_prob=2 case instead of the 2 lines above
        # 'LHSel2':
    #    add_defect(temps[temp][1], 2, 'above', temp)
    #    add_defect(temps[temp][2], 1, 'below', temp)
    #    add_defect(temps[temp][4], 3, 'above', temp)
        # add in 'special' max amp trap for good eperdn determination
        # has tau value outside of 1e-6 to 1e-2, but provides a peak trap
        # actually, doesn't meet threshold usually to count as trap, but
        #no harm leaving it in
        if not e2emode:
            add_1_dipole(temps[temp][1], 33, 77, 'below', 'sp', 0, 100, temp)
            # add in 'CENel1' trap for all phase times
        #    add_1_dipole(temps[temp][3], 26, 28, 'below', 'mf2', 0, 100, temp)
        #    add_1_dipole(temps[temp][4], 26, 28, 'above', 'mf2', 0, 100, temp)
            add_1_dipole(temps[temp][3], 26, 28, 'below', 2, 0, 100, temp)
            add_1_dipole(temps[temp][4], 26, 28, 'above', 2, 0, 100, temp)
            # add in 'RHSel1' trap for more than length limit (but diff lengths)
            #unused sch2 in this same pixel that is compatible with another trap
            add_1_dipole(temps[temp][1], 50, 50, 'above', 1, 0, 100, temp)
            add_1_dipole(temps[temp][4], 50, 50, 'above', 3, 3, 98, temp)
            add_1_dipole(temps[temp][2], 50, 50, 'below', 1, 2, 99, temp)
            # FALSE TRAPS: 'LHSel2' trap that doesn't meet length limit of unique
            # phase times even though the actual length is met for first 2
            # (and/or doesn't pass trap_id(), but I've already tested this case in
            # its unit test file)
            # (3rd will be 'unused')
            add_1_dipole(temps[temp][1], 71, 84, 'above', 2, 95, 100, temp)
            add_1_dipole(temps[temp][2], 71, 84, 'below', 1, 95, 100, temp)
            add_1_dipole(temps[temp][4], 71, 84, 'above', 3, 9, 20, temp)
            # 'LHSel2' trap
            add_1_dipole(temps[temp][1], 60, 80, 'above', 2, 1, 100, temp)
            add_1_dipole(temps[temp][2], 60, 80, 'below', 1, 1, 100, temp)
            add_1_dipole(temps[temp][4], 60, 80, 'above', 3, 1, 100, temp)
            # 'CENel2' trap
            add_1_dipole(temps[temp][1], 68, 67, 'above', 1, 0, 100, temp)
            add_1_dipole(temps[temp][2], 68, 67, 'below', 1, 0, 100, temp)
        #    add_1_dipole(temps[temp][1], 68, 67, 'above', 'mf1', 0, 100, temp)
        #    add_1_dipole(temps[temp][2], 68, 67, 'below', 'mf1', 0, 100, temp)
            # 'RHSel2' and 'LHSel3' traps in same pixel (could overlap phase time),
            # but good detectability means separation of peaks
            add_1_dipole(temps[temp][1], 98, 33, 'above', 1, 0, 100, temp)
            add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
                60, 100, 0, 40, temp) #80, 100, 0, 20, temp)
            add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
                60, 100, 0, 40, temp)
            # old:
            # add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
            #     50, 100, 0, 50, temp) #80, 100, 0, 20, temp)
            # add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
            #     50, 100, 0, 50, temp)
            # 'CENel3' trap (where sch3 has a 2-trap where one goes unused)
            add_2_dipole(temps[temp][3], 41, 15, 'above', 'above', 23,
            30, 100, 0, 30, temp)
            add_1_dipole(temps[temp][4], 41, 15, 'below', 2, 30, 100, temp)
            # 'RHSel3' and 'LHSel4'
            add_1_dipole(temps[temp][1], 89, 2, 'below', '1b', 0, 100, temp)
            add_2_dipole(temps[temp][2], 89, 2, 'above', 'above', 12,
                60, 100, 0, 30, temp) #30 was 40 in the past
            add_2_dipole(temps[temp][3], 89, 2, 'above', 'above', 33,
                60, 100, 0, 40, temp)
            # 2 'LHSel4' traps; whether the '0' or '1' trap gets assigned tau2 is
            # somewhat random; if one has an earlier starting temp than the other,
            # it would get assigned tau
            add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
                0, 40, 63, 100, temp)
            add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
                0, 40, 63, 100, temp)
            add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
                0, 40, 63, 100, temp) #30, 60, 100
            # old:
            # add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
            #     0, 40, 50, 100, temp)
            # 'CENel4' trap
            add_1_dipole(temps[temp][1], 56, 56, 'below', 1, 1, 100, temp)
            add_1_dipole(temps[temp][2], 56, 56, 'above', 1, 3, 99, temp)
            #'RHSel4' and 'CENel2' trap (tests 'a' and 'b' splitting in trap_fit_*)
            add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
                60, 100, 0, 40, temp)
            add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
                60, 100, 0, 40, temp)
            add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, 40, temp)
            # old:
            # add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
            #     30, 100, 0, 30, temp)
            # add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
            #     53, 100, 0, 53, temp)
            # add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, 30, temp)

        if e2emode: # full range should be covered if trap present
            add_1_dipole(temps[temp][1], 33, 77, 'below', 'sp', 0, phase_times, temp)
            # add in 'CENel1' trap for all phase times
        #    add_1_dipole(temps[temp][3], 26, 28, 'below', 'mf2', 0, 100, temp)
        #    add_1_dipole(temps[temp][4], 26, 28, 'above', 'mf2', 0, 100, temp)
            add_1_dipole(temps[temp][3], 26, 28, 'below', 2, 0, phase_times, temp)
            add_1_dipole(temps[temp][4], 26, 28, 'above', 2, 0, phase_times, temp)
            # add in 'RHSel1' trap for more than length limit (but diff lengths)
            #unused sch2 in this same pixel that is compatible with another trap
            add_1_dipole(temps[temp][1], 50, 50, 'above', 1, 0, phase_times, temp)
            add_1_dipole(temps[temp][4], 50, 50, 'above', 3, 3, phase_times, temp)
            add_1_dipole(temps[temp][2], 50, 50, 'below', 1, 2, phase_times, temp)
            # FALSE TRAPS: 'LHSel2' trap that doesn't meet length limit of unique
            # phase times even though the actual length is met for first 2
            # (and/or doesn't pass trap_id(), but I've already tested this case in
            # its unit test file)
            # (3rd will be 'unused')
            add_1_dipole(temps[temp][1], 71, 84, 'above', 2, 95, phase_times, temp)
            add_1_dipole(temps[temp][2], 71, 84, 'below', 1, 95, phase_times, temp)
            add_1_dipole(temps[temp][4], 71, 84, 'above', 3, 9, phase_times, temp)
            # 'LHSel2' trap
            add_1_dipole(temps[temp][1], 60, 80, 'above', 2, 1, phase_times, temp)
            add_1_dipole(temps[temp][2], 60, 80, 'below', 1, 1, phase_times, temp)
            add_1_dipole(temps[temp][4], 60, 80, 'above', 3, 1, phase_times, temp)
            # 'CENel2' trap
            add_1_dipole(temps[temp][1], 68, 67, 'above', 1, 0, phase_times, temp)
            add_1_dipole(temps[temp][2], 68, 67, 'below', 1, 0, phase_times, temp)
        #    add_1_dipole(temps[temp][1], 68, 67, 'above', 'mf1', 0, 100, temp)
        #    add_1_dipole(temps[temp][2], 68, 67, 'below', 'mf1', 0, 100, temp)
            # 'RHSel2' and 'LHSel3' traps in same pixel (could overlap phase time),
            # but good detectability means separation of peaks
            add_1_dipole(temps[temp][1], 98, 33, 'above', 1, 0, phase_times, temp)
            add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp) #80, 100, 0, 20, temp)
            add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            # old:
            # add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
            #     50, 100, 0, 50, temp) #80, 100, 0, 20, temp)
            # add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
            #     50, 100, 0, 50, temp)
            # 'CENel3' trap (where sch3 has a 2-trap where one goes unused)
            add_2_dipole(temps[temp][3], 41, 15, 'above', 'above', 23,
            int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            add_1_dipole(temps[temp][4], 41, 15, 'below', 2, 0, phase_times, temp)
            # 'RHSel3' and 'LHSel4'
            add_1_dipole(temps[temp][1], 89, 2, 'below', '1b', 0, phase_times, temp)
            add_2_dipole(temps[temp][2], 89, 2, 'above', 'above', 12,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp) #30 was 40 in the past
            add_2_dipole(temps[temp][3], 89, 2, 'above', 'above', 33,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            # 2 'LHSel4' traps; whether the '0' or '1' trap gets assigned tau2 is
            # somewhat random; if one has an earlier starting temp than the other,
            # it would get assigned tau
            add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
                0, int(phase_times/2), int(phase_times/2), phase_times, temp)
            add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
                0, int(phase_times/2), int(phase_times/2), phase_times, temp)
            add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
                0, int(phase_times/2), int(phase_times/2), phase_times, temp) #30, 60, 100
            # old:
            # add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
            #     0, 40, 50, 100, temp)
            # add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
            #     0, 40, 50, 100, temp)
            # 'CENel4' trap
            add_1_dipole(temps[temp][1], 56, 56, 'below', 1, 1, phase_times, temp)
            add_1_dipole(temps[temp][2], 56, 56, 'above', 1, 3, phase_times, temp)
            #'RHSel4' and 'CENel2' trap (tests 'a' and 'b' splitting in trap_fit_*)
            add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
                int(phase_times/2), phase_times, 0, int(phase_times/2), temp)
            add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, phase_times, temp)
            # old:
            # add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
            #     30, 100, 0, 30, temp)
            # add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
            #     53, 100, 0, 53, temp)
            # add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, 30, temp)
        pass
        if e2emode:
            readout_emccd = EMCCDDetect(
                em_gain=EMgain, #10,
                full_well_image=full_well_image,  # e-
                full_well_serial=full_well_serial,  # e-
                dark_current=0,  # e-/pix/s
                cic=0, # e-/pix/frame
                read_noise=read_noise,  # e-/pix/frame
                bias=1000,  # e-
                qe=1, # no QE hit here; just simulating readout
                cr_rate=0.,  # hits/cm^2/s
                pixel_pitch=13e-6,  # m
                eperdn=later_eperdn,
                nbits=nbits,
                numel_gain_register=604,
                meta_path=meta_path,
                nonlin_path=nonlin_path
                )
        # save to FITS files
        for sc in [1,2,3,4]:
            for i in range(len(temps[temp][sc])):
                if e2emode:
                    if temps[temp][sc][i].any() >= full_well_image:
                        raise Exception('Saturated before EM gain applied.')
                    # Now apply readout things for e2e mode 
                    gain_counts = np.reshape(readout_emccd._gain_register_elements(temps[temp][sc][i].ravel()),temps[temp][sc][i].shape)
                    if gain_counts.any() >= full_well_serial:
                        raise Exception('Saturated after EM gain applied.')
                    output_dn = readout_emccd.readout(gain_counts)
                else:
                    output_dn = temps[temp][sc][i]
                prihdr, exthdr = create_default_L1_TrapPump_headers(arrtype)
                prim = fits.PrimaryHDU(header = prihdr)
                hdr_img = fits.ImageHDU(output_dn, header=exthdr)
                hdul = fits.HDUList([prim, hdr_img])
                ## Fill in the headers that matter to corgidrp
                hdul[1].header['EXCAMT']  = temp
                hdul[1].header['EMGAIN_C'] = EMgain
                hdul[1].header['ARRTYPE'] = arrtype
                for j in range(1, 5):
                    if sc == j:
                        hdul[1].header['TPSCHEM' + str(j)] = num_pumps
                    else:
                        hdul[1].header['TPSCHEM' + str(j)] = 0
                hdul[1].header['TPTAU'] = time_data[i]
                
                t = time_data[i]
                # curr_sch_dir = Path(here, 'test_data_sub_frame_noise', str(temp)+'K',
                # 'Scheme_'+str(sch))

                # if os.path.isfile(Path(output_dir,
                # str(temp)+'K'+'Scheme_'+str(sch)+'TPUMP_Npumps_10000_gain'+str(g)+'_phasetime'+str(t)+'.fits')):
                #     hdul.writeto(Path(output_dir,
                #     str(temp)+'K'+'Scheme_'+str(sch)+'TPUMP_Npumps_10000_gain'+str(g)+'_phasetime'+
                #     str(t)+'_2.fits'), overwrite = True)
                # else: 
                # Note: have to use old filename format for now and overwrite later because setting
                # the filename affects data generation
                mult_counter = 0
                filename = Path(output_dir,
                    str(temp)+'K'+'Scheme_'+str(sc)+'TPUMP_Npumps_'+str(int(num_pumps))+'_gain'+str(EMgain)+'_phasetime'+str(t)+'.fits')
                if os.path.exists(filename):
                    mult_counter += 1
                    hdul.writeto(str(filename)[:-4]+'_'+str(mult_counter)+'.fits', overwrite = True)
                else:
                    hdul.writeto(filename, overwrite = True)
    
    # After all data generation is complete, rename files to CGI format, because changing the filename
    # in the function above somehow affects the content of the file
    rename_files_to_cgi_format(pattern=os.path.join(output_dir, "*K*Scheme_*TPUMP*.fits"), level_suffix="l1")

def create_photon_countable_frames(Nbrights=30, Ndarks=40, EMgain=5000, kgain=7, exptime=0.05, cosmic_rate=0, full_frame=True, smear=True, flux=1, bad_frames=0, cic=0.01, dark_current=8.33e-4, read_noise=100., bias=20000, qe=0.9):
    '''This creates mock L1 Dataset containing frames with large gain and short exposure time, illuminated and dark frames.
    Used for unit tests for photon counting.  
    
    Args:
        Nbrights (int):  number of illuminated frames to simulate
        Ndarks (int):  number of dark frames to simulate
        EMgain (float): EM gain
        kgain (float): k gain (e-/DN)
        exptime (float): exposure time (in s)
        cosmic_rate: (float) simulated cosmic rays incidence, hits/cm^2/s
        full_frame: (bool) If True, simulated frames are SCI full frames.  If False, 50x50 images are simulated.  Defaults to True.
        smear: (bool) If True, smear is simulated.  Defaults to True.
        flux: (float) Number of photons/s per pixel desired.  Defaults to 1.
        bad_frames (int): Number of simulated bad frames (with primary header keyword 'OVEREXP' set to True) to include in output datasets that frame_select would catch by default. Defaults to 0.
        cic (float): simulated clock-induced charge (CIC) in e-/pix/frame.  Defaults to 0.01.
        dark_current (float): simulated dark current in e-/pix/s.  Defaults to 8.33e-4.
        read_noise (float): simulated read noise in e-/pix/frame.  Defaults to 100.
        bias (float): simulated bias in e-.  Defaults to 20000.
        qe (float): quantum efficiency, e-/photon.  Defaults to 0.9.
    
    Returns:
        ill_dataset (corgidrp.data.Dataset): Dataset containing the illuminated frames
        dark_dataset (corgidrp.data.Dataset): Dataset containing the dark frames
        ill_mean (float): mean electron count value simulated in the illuminated frames
        dark_mean (float): mean electron count value simulated in the dark frames
    '''
    pix_row = 1024 #number of rows and number of columns
    fluxmap = flux*np.ones((pix_row,pix_row)) #photon flux map, photons/s

    emccd = EMCCDDetect(
        em_gain=EMgain,
        full_well_image=60000.,  # e-
        full_well_serial=100000.,  # e-
        dark_current=dark_current,  # e-/pix/s
        cic=cic,  # e-/pix/frame
        read_noise=read_noise,  # e-/pix/frame
        bias=bias,  # e-
        qe=qe,  # quantum efficiency, e-/photon
        cr_rate=cosmic_rate,  # cosmic rays incidence, hits/cm^2/s
        pixel_pitch=13e-6,  # m
        eperdn=kgain,  
        nbits=64, # number of ADU bits
        numel_gain_register=604 #number of gain register elements
        )

    thresh = emccd.em_gain/10 # threshold

    if np.average(exptime*fluxmap) > 0.1:
        warnings.warn('average # of photons/pixel is > 0.1.  Decrease frame '
        'time to get lower average # of photons/pixel.')

    if emccd.read_noise <=0:
        warnings.warn('read noise should be greater than 0 for effective '
        'photon counting')
    if thresh < 4*emccd.read_noise:
        warnings.warn('thresh should be at least 4 or 5 times read_noise for '
        'accurate photon counting')

    avg_ph_flux = np.mean(fluxmap)
    # theoretical electron flux for brights
    ill_mean = avg_ph_flux*emccd.qe*exptime + emccd.dark_current*exptime + emccd.cic
    # theoretical electron flux for darks
    dark_mean = emccd.dark_current*exptime + emccd.cic

    if smear:
        #simulate smear to fluxmap
        detector_params = DetectorParams({})
        rowreadtime = detector_params.params['ROWREADT']
        smear = np.zeros_like(fluxmap)
        m = len(smear)
        for r in range(m):
            columnsum = 0
            for i in range(r+1):
                columnsum = columnsum + rowreadtime*fluxmap[i,:]
            smear[r,:] = columnsum
        
        fluxmap = fluxmap + smear/exptime
    
    frame_e_list = []
    frame_e_dark_list = []
    prihdr, exthdr = create_default_L1_headers()
    for i in range(Nbrights):
        # Simulate bright
        if full_frame:
            frame_dn = emccd.sim_full_frame(fluxmap, exptime)
        else:
            frame_dn = emccd.sim_sub_frame(fluxmap[:50,:50], exptime)
        frame = data.Image(frame_dn, pri_hdr=prihdr, ext_hdr=exthdr)
        frame.ext_hdr['EMGAIN_C'] = EMgain
        frame.ext_hdr['EXPTIME'] = exptime
        frame.ext_hdr['RN'] = 100
        frame.ext_hdr['KGAINPAR'] = kgain
        frame.pri_hdr['PHTCNT'] = 1
        frame.ext_hdr['ISPC'] = 1
        frame.pri_hdr["VISTYPE"] = "CGIVST_TDD_OBS"
        # Generate CGI filename with incrementing datetime
        visitid = frame.pri_hdr["VISITID"]
        base_time = datetime.datetime.now()
        time_offset = datetime.timedelta(seconds=i)
        unique_time = base_time + time_offset
        time_str = data.format_ftimeutc(unique_time.isoformat())
        frame.filename = f'cgi_{visitid}_{time_str}_l1_.fits'
        frame_e_list.append(frame)

    for i in range(Ndarks):
        # Simulate dark
        if full_frame:
            frame_dn_dark = emccd.sim_full_frame(np.zeros_like(fluxmap), exptime)
        else:
            frame_dn_dark = emccd.sim_sub_frame(np.zeros_like(fluxmap[:50,:50]), exptime)
        frame_dark = data.Image(frame_dn_dark, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
        frame_dark.ext_hdr['EMGAIN_C'] = EMgain
        frame_dark.ext_hdr['EXPTIME'] = exptime
        frame_dark.ext_hdr['RN'] = 100
        frame_dark.ext_hdr['KGAINPAR'] = kgain
        frame_dark.pri_hdr['PHTCNT'] = 1
        frame_dark.ext_hdr['ISPC'] = 1
        frame_dark.pri_hdr["VISTYPE"] = "CGIVST_CAL_DRK"
        # Generate CGI filename with incrementing datetime for dark frames
        visitid = frame_dark.pri_hdr["VISITID"]
        base_time = datetime.datetime.now()
        time_offset = datetime.timedelta(seconds=i + 1000)  # Offset to avoid conflicts with bright frames
        unique_time = base_time + time_offset
        time_str = data.format_ftimeutc(unique_time.isoformat())
        frame_dark.filename = f'cgi_{visitid}_{time_str}_l1_.fits'
        frame_e_dark_list.append(frame_dark)

    for i in range(bad_frames):
        bad_frame = frame.copy()
        bad_frame.ext_hdr['OVEREXP'] = True
        # Generate CGI filename for bad bright frames
        visitid = bad_frame.pri_hdr["VISITID"]
        base_time = datetime.datetime.now()
        time_offset = datetime.timedelta(seconds=Nbrights + i)
        unique_time = base_time + time_offset
        time_str = data.format_ftimeutc(unique_time.isoformat())
        bad_frame.filename = f'cgi_{visitid}_{time_str}_l1_.fits'
        frame_e_list.append(bad_frame)
        bad_dark_frame = frame_dark.copy()
        bad_dark_frame.ext_hdr['OVEREXP'] = True
        # Generate CGI filename for bad dark frames
        time_offset = datetime.timedelta(seconds=Ndarks + i + 1000)  # Offset to avoid conflicts
        unique_time = base_time + time_offset
        time_str = data.format_ftimeutc(unique_time.isoformat())
        bad_dark_frame.filename = f'cgi_{visitid}_{time_str}_l1_.fits'
        frame_e_dark_list.append(bad_dark_frame)

    ill_dataset = data.Dataset(frame_e_list)
    dark_dataset = data.Dataset(frame_e_dark_list)

    return ill_dataset, dark_dataset, ill_mean, dark_mean

def gaussian_array(array_shape=[50,50],sigma=2.5,amp=100.,xoffset=0.,yoffset=0.):
    """Generate a 2D square array with a centered gaussian surface (for mock PSF data).

    Args:
        array_shape (int, optional): Shape of desired array in pixels. Defaults to [50,50].
        sigma (float, optional): Standard deviation of the gaussian curve, in pixels. Defaults to 5.
        amp (float,optional): Amplitude (peak) of gaussian curve. Defaults to 1.
        xoffset (float,optional): x offset of gaussian from array center. Defaults to 0.
        yoffset (float,optional): y offset of gaussian from array center. Defaults to 0.
        
    Returns:
        np.array: 2D array of a gaussian surface.
    """
    x, y = np.meshgrid(np.linspace(-array_shape[0]/2+0.5, array_shape[0]/2-0.5, array_shape[0]),
                        np.linspace(-array_shape[1]/2+0.5, array_shape[1]/2-0.5, array_shape[1]))
    dst = np.sqrt((x-xoffset)**2+(y-yoffset)**2)

    # Calculate Gaussian 
    gauss = np.exp(-((dst)**2 / (2.0 * sigma**2))) * amp
    
    return gauss

def create_flux_image(star_flux, fwhm, cal_factor, filter='3C', fpamname = 'HOLE', target_name='Vega', fsm_x=0.0, 
                      fsm_y=0.0, exptime=1.0, filedir=None, platescale=21.8, 
                      background=0, add_gauss_noise=True, noise_scale=1., file_save=False):
    """
    Create simulated data for absolute flux calibration. This is a point source with a 2D-Gaussian PSF
    and Gaussian noise.

    Args:
        star_flux (float): Flux of the point source in erg/(s*cm^2*AA)
        fwhm (float): Full width at half max (FWHM) of the centroid
        cal_factor (float): Calibration factor erg/(s*cm^2*AA)/electron/s
        filter (str): (Optional) The CFAM filter used.
        fpamname (str): (Optional) Position of the FPAM
        target_name (str): (Optional) Name of the calspec star
        fsm_x (float): (Optional) X position shift in milliarcseconds (mas)
        fsm_y (float): (Optional) Y position shift in milliarcseconds (mas)
        exptime (float): (Optional) Exposure time (s)
        filedir (str): (Optional) Directory path to save the output file
        platescale (float): Plate scale in mas/pixel (default: 21.8 mas/pixel)
        background (float): optional additive background value
        add_gauss_noise (bool): Whether to add Gaussian noise to the data (default: True)
        noise_scale (float): Spread of the Gaussian noise
        file_save (bool): Whether to save the image (default: False)

    Returns:
        corgidrp.data.Image: The simulated image
    """

    # Create directory if needed
    if filedir is not None and not os.path.exists(filedir):
        os.mkdir(filedir)

    # Image properties
    size = (1024, 1024)
    sim_data = np.zeros(size)
    ny, nx = size
    center = [nx // 2, ny // 2]  # Default image center
    target_location = (80.553428801, -69.514096821)

    # Convert FSM shifts from mas to pixels
    fsm_x_shift = fsm_x * 0.001 / (platescale * 0.001)  # Convert mas to degrees, then to pixels
    fsm_y_shift = fsm_y * 0.001 / (platescale * 0.001)

    # New star position
    xpos = center[0] + fsm_x_shift
    ypos = center[1] + fsm_y_shift

    # Convert flux from calspec units to photo-electrons/s
    flux = star_flux / cal_factor

    # Inject Gaussian PSF star
    stampsize = int(np.ceil(3 * fwhm))
    sigma = fwhm/ (2.*np.sqrt(2*np.log(2)))

    # coordinate system
    y, x = np.indices([stampsize, stampsize])
    y -= stampsize // 2
    x -= stampsize // 2

    # Find nearest pixel
    x_int = int(round(xpos))
    y_int = int(round(ypos))
    x += x_int
    y += y_int
    
    xmin = x[0][0]
    xmax = x[-1][-1]
    ymin = y[0][0]
    ymax = y[-1][-1]
        
    psf = gaussian_array((stampsize,stampsize),sigma,flux) / (2.0 * np.pi * sigma**2)

    # Inject the star into the image
    sim_data[ymin:ymax + 1, xmin:xmax + 1] += psf

    # Add background
    sim_data += background

    # Add Gaussian noise
    if add_gauss_noise:
        # add Gaussian random noise
        noise_rng = np.random.default_rng(10)
        noise = noise_rng.normal(scale=noise_scale, size=size)
        sim_data += noise

    # Error map
    err = np.full(size, noise_scale)

    # Get FPAM positions, not strictly necessary but
    if fpamname == 'HOLE':
        fpam_h = 40504.4
        fpam_v = 9616.8
    elif fpamname == 'ND225':
        fpam_h = 61507.8
        fpam_v = 25612.4
    elif fpamname == 'ND475':
        fpam_h = 2503.7
        fpam_v = 6124.9

    # Create image object
    prihdr, exthdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    prihdr['VISTYPE'] = 'CGIVST_CAL_ABSFLUX_BRIGHT'
    prihdr['RA'] = target_location[0]
    prihdr['DEC'] = target_location[1]
    prihdr['TARGET'] = target_name

    exthdr['CFAMNAME'] = filter             # Using the variable 'filter' (ensure it's defined)
    exthdr['FPAMNAME'] = fpamname
    exthdr['FPAM_H']   = 2503.7
    exthdr['FPAM_V']   = 6124.9
    exthdr['FSMX']    = fsm_x              # Ensure fsm_x is defined
    exthdr['FSMY']    = fsm_y              # Ensure fsm_y is defined
    exthdr['EXPTIME']  = exptime            # Ensure exptime is defined       # Ensure color_cor is defined
    exthdr['CRPIX1']   = xpos               # Ensure xpos is defined
    exthdr['CRPIX2']   = ypos               # Ensure ypos is defined
    exthdr['CTYPE1']   = 'RA---TAN'
    exthdr['CTYPE2']   = 'DEC--TAN'
    exthdr['CDELT1']   = (platescale * 0.001) / 3600  # Ensure platescale is defined
    exthdr['CDELT2']   = (platescale * 0.001) / 3600
    exthdr['CRVAL1']   = target_location[0]  # Ensure target_location is a defined list/tuple
    exthdr['CRVAL2']   = target_location[1]
    exthdr['BUNIT'] = 'photoelectron'
    frame = data.Image(sim_data, err=err, pri_hdr=prihdr, ext_hdr=exthdr)
   
    # Set filename
    ftimeutc = data.format_ftimeutc(exthdr['FTIMEUTC'])
    filename = f'cgi_{prihdr["VISITID"]}_{ftimeutc}_l2b.fits'
    frame.filename = filename
    
    # Save file if requested
    if filedir is not None and file_save:
        frame.save(filedir=filedir, filename=filename)

    return frame

def create_pol_flux_image(star_flux_left, star_flux_right, fwhm, cal_factor, filter='3C', dpamname = 'POL0', fpamname = 'HOLE', 
                      target_name='Vega', fsm_x=0.0, fsm_y=0.0, exptime=1.0, filedir=None, platescale=21.8, 
                      background=0, add_gauss_noise=True, noise_scale=1., file_save=False):
    """
    Create simulated data for polarimetric absolute flux calibration. Two point sources to
    simulate images split by polarization, with a 2D-Gaussian PSF and Gaussian noise.

    Args:
        star_flux_left (float): Flux of the point source on the left size of the image in erg/(s*cm^2*AA)
        star_flux_right (float): Flux of the point source on the right size of the image in erg/(s*cm^2*AA)
        fwhm (float): Full width at half max (FWHM) of the centroid
        cal_factor (float): Calibration factor erg/(s*cm^2*AA)/electron/s
        filter (str): (Optional) The CFAM filter used.
        dpamname (str): (Optional) The wollaston prism being used
        fpamname (str): (Optional) Position of the FPAM
        target_name (str): (Optional) Name of the calspec star
        fsm_x (float): (Optional) X position shift in milliarcseconds (mas)
        fsm_y (float): (Optional) Y position shift in milliarcseconds (mas)
        exptime (float): (Optional) Exposure time (s)
        filedir (str): (Optional) Directory path to save the output file
        platescale (float): Plate scale in mas/pixel (default: 21.8 mas/pixel)
        background (float): optional additive background value
        add_gauss_noise (bool): Whether to add Gaussian noise to the data (default: True)
        noise_scale (float): Spread of the Gaussian noise
        file_save (bool): Whether to save the image (default: False)

    Returns:
        corgidrp.data.Image: The simulated image
    """

    # Create directory if needed
    if filedir is not None and not os.path.exists(filedir):
        os.mkdir(filedir)

    # Image properties
    size = (1024, 1024)
    sim_data = np.zeros(size)
    ny, nx = size
    center = [nx // 2, ny // 2]  # Default image center
    target_location = (80.553428801, -69.514096821)

    # Convert FSM shifts from mas to pixels
    fsm_x_shift = fsm_x * 0.001 / (platescale * 0.001)  # Convert mas to degrees, then to pixels
    fsm_y_shift = fsm_y * 0.001 / (platescale * 0.001)

    # New star position
    xpos = center[0] + fsm_x_shift
    ypos = center[1] + fsm_y_shift

    # Find nearest pixel
    x_int = int(round(xpos))
    y_int = int(round(ypos))


    # Convert flux from calspec units to photo-electrons/s
    flux_left = star_flux_left / cal_factor
    flux_right = star_flux_right / cal_factor

    if dpamname == 'POL0':
        x_int_left = x_int - 172
        x_int_right = x_int + 172
        y_int_left = y_int
        y_int_right = y_int
        dpam_h = 8991.3
        dpam_v = 1261.3
    elif dpamname == 'POL45':
        x_int_left = x_int - 122
        x_int_right = x_int + 122
        y_int_left = y_int + 122
        y_int_right = y_int - 122
        dpam_h = 44660.1
        dpam_v = 1261.3
    else:
        raise ValueError('dpamname have to be "POL0" or "POL45"')


    # Inject Gaussian PSF star
    stampsize = int(np.ceil(3 * fwhm))
    sigma = fwhm/ (2.*np.sqrt(2*np.log(2)))

    # coordinate system
    y, x = np.indices([stampsize, stampsize])
    y -= stampsize // 2
    x -= stampsize // 2

    x_left = x + x_int_left
    x_right = x + x_int_right
    y_left = y + y_int_left
    y_right = y + y_int_right
    
    xmin_left = x_left[0][0]
    xmax_left = x_left[-1][-1]
    xmin_right = x_right[0][0]
    xmax_right = x_right[-1][-1]
    ymin_left = y_left[0][0]
    ymax_left = y_left[-1][-1]
    ymin_right = y_right[0][0]
    ymax_right = y_right[-1][-1]

    psf_left = gaussian_array((stampsize,stampsize),sigma,flux_left) / (2.0 * np.pi * sigma**2)
    psf_right = gaussian_array((stampsize,stampsize),sigma,flux_right) / (2.0 * np.pi * sigma**2)

    # Inject the star into the image
    sim_data[ymin_left:ymax_left + 1, xmin_left:xmax_left + 1] += psf_left
    sim_data[ymin_right:ymax_right + 1, xmin_right:xmax_right + 1] += psf_right

    # Add background
    sim_data += background

    # Add Gaussian noise
    if add_gauss_noise:
        # add Gaussian random noise
        noise_rng = np.random.default_rng(10)
        noise = noise_rng.normal(scale=noise_scale, size=size)
        sim_data += noise

    # Error map
    err = np.full(size, noise_scale)

    # Get FPAM positions, not strictly necessary but
    if fpamname == 'HOLE':
        fpam_h = 40504.4
        fpam_v = 9616.8
    elif fpamname == 'ND225':
        fpam_h = 61507.8
        fpam_v = 25612.4
    elif fpamname == 'ND475':
        fpam_h = 2503.7
        fpam_v = 6124.9

    # Create image object
    prihdr, exthdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    prihdr['VISTYPE'] = 'CGIVST_CAL_ABSFLUX_BRIGHT'
    prihdr['TARGET'] = target_name

    exthdr['CFAMNAME'] = filter             # Using the variable 'filter' (ensure it's defined)
    exthdr['FPAMNAME'] = fpamname
    exthdr['DPAMNAME'] = dpamname
    exthdr['DPAM_H'] = dpam_h
    exthdr['DPAM_V'] = dpam_v
    exthdr['FPAM_H']   = 2503.7
    exthdr['FPAM_V']   = 6124.9
    exthdr['FSMX']    = fsm_x              # Ensure fsm_x is defined
    exthdr['FSMY']    = fsm_y              # Ensure fsm_y is defined
    exthdr['EXPTIME']  = exptime            # Ensure exptime is defined       # Ensure color_cor is defined
    exthdr['CRPIX1']   = xpos               # Ensure xpos is defined
    exthdr['CRPIX2']   = ypos               # Ensure ypos is defined
    exthdr['CTYPE1']   = 'RA---TAN'
    exthdr['CTYPE2']   = 'DEC--TAN'
    exthdr['CDELT1']   = (platescale * 0.001) / 3600  # Ensure platescale is defined
    exthdr['CDELT2']   = (platescale * 0.001) / 3600
    exthdr['CRVAL1']   = target_location[0]  # Ensure target_location is a defined list/tuple
    exthdr['CRVAL2']   = target_location[1]
    frame = data.Image(sim_data, err=err, pri_hdr=prihdr, ext_hdr=exthdr)
   
    # Set filename
    ftimeutc = data.format_ftimeutc(exthdr['FTIMEUTC'])
    filename = f'cgi_{prihdr["VISITID"]}_{ftimeutc}_l2b.fits'
    frame.filename = filename
    
    # Save file if requested
    if filedir is not None and file_save:
        frame.save(filedir=filedir, filename=filename)

    return frame

def generate_reference_star_dataset_with_flux(
    n_frames=3,
    roll_angles=None,
    # Following arguments match create_flux_image
    flux_erg_s_cm2=1e-13,
    fwhm=3.0,
    cal_factor=1e10,            # [ e- / erg ]
    optical_throughput=1.0,
    color_cor=1.0,
    filter='3C',
    fpamname='HOLE',
    target_name='Vega',
    fsm_x=0.0,
    fsm_y=0.0,
    exptime=1.0,
    pltscale_mas=21.8,     
    background=0,
    add_gauss_noise=True,
    noise_scale=1.,
    filedir=None,
    file_save=False,
    shape=(1024, 1024)  # <-- new shape argument
):
    """
    Generate simulated reference star dataset with flux calibration.
    This function creates multiple frames of a reference star (with no planet)
    using create_flux_image(), and assigns unique roll angles to each frame's header.
    The generated frames can then be used for RDI or ADI+RDI processing in pyKLIP.
    
    Args:
        n_frames (int): Number of frames to generate.
        roll_angles (list or None): Roll angles (in degrees) for each frame. If None, all frames use 0.0.
        flux_erg_s_cm2 (float): Stellar flux in erg s⁻¹ cm⁻².
        fwhm (float): Full Width at Half Maximum of the star's PSF.
        cal_factor (float): Calibration factor [e⁻/erg] to convert flux to electron counts.
        optical_throughput (float): Overall optical throughput factor.
        color_cor (float): Color correction factor.
        filter (str): Filter identifier.
        fpamname (str): FPAM name indicating the pupil mask configuration.
        target_name (str): Name of the target star.
        fsm_x (float): Field Stabilization Mirror (FSM) x-offset.
        fsm_y (float): Field Stabilization Mirror (FSM) y-offset.
        exptime (float): Exposure time in seconds.
        pltscale_mas (float): Plate scale (e.g., mas/pixel or arcsec/pixel) of the image.
        background (int): Background level counts.
        add_gauss_noise (bool): If True, add Gaussian noise to the image.
        noise_scale (float): Scaling factor for the Gaussian noise.
        filedir (str or None): Directory to save the generated files if file_save is True.
        file_save (bool): Flag to save each generated frame to disk.
        shape (tuple): Shape (ny, nx) of the generated images.
    
    Returns:
        Dataset (corgidrp.Data.Dataset): A Dataset object containing the generated reference star frames.
    """

    if roll_angles is None:
        roll_angles = [0.0]*n_frames
    elif len(roll_angles) != n_frames:
        raise ValueError("roll_angles must match n_frames or be None.")

    frames = []
    for i in range(n_frames):
        # 1) Create a single flux image with the star alone
        frame = create_flux_image(
            flux_erg_s_cm2=flux_erg_s_cm2,
            fwhm=fwhm,
            cal_factor=cal_factor,
            optical_throughput=optical_throughput,
            color_cor=color_cor,
            filter=filter,
            fpamname=fpamname,
            target_name=target_name,
            fsm_x=fsm_x,
            fsm_y=fsm_y,
            exptime=exptime,
            filedir=filedir,
            platescale=pltscale_mas,
            background=background,
            add_gauss_noise=add_gauss_noise,
            noise_scale=noise_scale,
            file_save=False,
            shape=shape            # <--- pass the shape argument here
        )   

        # 2) Mark primary header as "PSFREF=1" so do_psf_subtraction sees it as reference
        frame.pri_hdr["PSFREF"] = 1

        # 3) Set this frame's roll angle in pri_hdr
        frame.pri_hdr["ROLL"] = roll_angles[i]

        # 4) Set star center for reference
        #    create_flux_image puts the star around (shape[1]//2, shape[0]//2).
        #    If fsm_x=0, fsm_y=0. 
        nx = shape[1]
        ny = shape[0]
        x_center = nx // 2 + (fsm_x * 0.001 / (pltscale_mas * 0.001))
        y_center = ny // 2 + (fsm_y * 0.001 / (pltscale_mas * 0.001))

        frame.ext_hdr['PLTSCALE'] = pltscale_mas
        frame.ext_hdr["STARLOCX"] = x_center  
        frame.ext_hdr["STARLOCY"] = y_center  
        
        # Generate CGI filename with incrementing datetime
        visitid = frame.pri_hdr["VISITID"]
        base_time = datetime.datetime.now()
        time_offset = datetime.timedelta(seconds=i)
        unique_time = base_time + time_offset
        time_str = data.format_ftimeutc(unique_time.isoformat())
        filename = f"cgi_{visitid}_{time_str}_l2b.fits"
        frame.pri_hdr["FILENAME"] = filename

        # 5) Optionally save each file
        if filedir is not None and file_save:
            frame.save(filedir=filedir, filename=filename)

        frames.append(frame)

    return Dataset(frames)

def create_ct_psfs(fwhm_mas, cfam_name='1F', n_psfs=10, e2e=False):
    """
    Create simulated data for core throughput calibration. This is a set of
    individual, noiseless 2D Gaussians, one per image.  

    Args:
        fwhm_mas (float): PSF's FWHM in mas
        cfam_name (str) (optional): CFAM filter name.
        n_psfs (int) (optional): Number of simulated PSFs.
        e2e (bool) (optional): Whether these simulated data are for the CT e2e
          test or not. If they are, the files are L2b. Otherwise, they are L3. 

    Returns:
        corgidrp.data.Image: The simulated PSF Images
        np.array: PSF locations
        np.array: PSF CT values
    """
    # Default headers
    if e2e:
        prhd, exthd, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    else:
        prhd, exthd, errhdr, dqhdr = create_default_L3_headers()
    # These data are for CT calibration
    prhd['VISTYPE'] = 'CGIVST_CAL_CORETHRPT'
    # cfam filter
    exthd['CFAMNAME'] = cfam_name
    # Mock ERR
    err = np.ones([1024,1024])
    # Mock DQ
    dq = np.zeros([1024,1024], dtype = np.uint16)

    fwhm_pix = int(np.ceil(fwhm_mas/21.8))
    # PSF/PSF_peak > 1e-10 for +/- 3FWHM around the PSFs center
    imshape = (6*fwhm_pix+1, 6*fwhm_pix+1)
    y, x = np.indices(imshape)

    # Following astropy documentation:
    # Generate random source model list. Random amplitudes and centers within a pixel
    # PSF's final location on SCI frame is moved by more than one pixel below. This
    # is the fractional part that only needs a smaller array of non-zero values
    # Set seed for reproducibility of mock data
    rng = np.random.default_rng(0)
    model_params = [
        dict(amplitude=rng.uniform(1,10),
        x_mean=rng.uniform(imshape[0]//2,imshape[0]//2+1),
        y_mean=rng.uniform(imshape[0]//2,imshape[0]//2+1),
        x_stddev=fwhm_mas/21.8/2.335,
        y_stddev=fwhm_mas/21.8/2.335)
        for _ in range(n_psfs)]

    model_list = [models.Gaussian2D(**kwargs) for kwargs in model_params]
    # Render models to image using full evaluation
    psf_loc = []
    half_psf = []
    data_psf = []
    for model in model_list:
        # Skip any PSFs with 0 amplitude (if any)
        if model.amplitude == 0:
            continue
        psf = np.zeros(imshape)
        model.bounding_box = None
        model.render(psf)
        image = np.zeros([1024, 1024])
        # Insert PSF at random location within the SCI frame
        y_image, x_image = rng.integers(100), rng.integers(100)
        image[512+y_image-imshape[0]//2:512+y_image+imshape[0]//2+1,
            512+x_image-imshape[1]//2:512+x_image+imshape[1]//2+1] = psf
        # List of known positions and list of known PSF volume
        psf_loc += [[512+x_image+model.x_mean.value-imshape[0]//2,
            512+y_image+model.y_mean.value-imshape[0]//2]]
        # Add half PSF volume for 2D Gaussian (numerator of core throughput)
        half_psf += [np.pi*model.amplitude.value*model.x_stddev.value*model.y_stddev.value]
        # Build up the Dataset
        data_psf += [Image(image,pri_hdr=prhd, ext_hdr=exthd, err=err, dq=dq)]
        # Add some filename following the file convention:
        # cgi_<VisitID: PPPPPCCAAASSSOOOVVV>_<TimeUTC>_l2b.fits
        data_psf[-1].filename = 'cgi_0200001001001001001_20250415t0305102_l2b.fits'
        
    return data_psf, np.array(psf_loc), np.array(half_psf)

def create_ct_psfs_with_mask(fwhm_mas, cfam_name='1F', n_psfs=10, image_shape=(1024,1024),
                   apply_mask=True, total_counts=1e4):
    """
    Create simulated data for core throughput calibration. This is a set of
    individual, noiseless 2D Gaussians with a spatially varying throughput
    that mimics a central occulting mask when apply_mask=True.
    
    Args:
        fwhm_mas (float): PSF FWHM in mas.
        cfam_name (str): CFAM filter name.
        n_psfs (int): Number of PSFs to generate.
        image_shape (tuple): Full image shape.
        apply_mask (bool): If True, apply the mask transmission function. If False,
            the transmission is set to 1 everywhere.
        total_counts (float): The desired total integrated counts in the unmasked PSF.
    
    Returns:
        data_psf (list): List of Image objects with the PSF stamp inserted.
        psf_loc (np.array): Array of PSF locations.
        half_psf (np.array): Array of “half” throughput values (roughly total_counts/2 after mask).
    """
    # Set up headers, error, and dq arrays.
    prhd, exthd, errhdr, dqhdr = create_default_L3_headers()
    exthd['CFAMNAME'] = cfam_name
    err = np.ones(image_shape)
    dq = np.zeros(image_shape, dtype=np.uint16)
    
    # Calculate the image center.
    center_x = image_shape[1] // 2
    center_y = image_shape[0] // 2
    image_center = (center_x, center_y)
    exthd['STARLOCX'] = center_x
    exthd['STARLOCY'] = center_y
    
    # Determine the stamp size for the PSF: +/- 3 FWHM in pixels.
    fwhm_pix = int(np.ceil(fwhm_mas / 21.8))
    stamp_shape = (6 * fwhm_pix + 1, 6 * fwhm_pix + 1)
    
    # PSF parameters.
    # Compute the standard deviations (assuming FWHM = 2.355 sigma).
    x_stddev = fwhm_mas / 21.8 / 2.355
    y_stddev = fwhm_mas / 21.8 / 2.355
    x_mean = stamp_shape[1] // 2
    y_mean = stamp_shape[0] // 2
    
    # Calculate the amplitude such that the integrated flux equals total_counts.
    # Integrated flux of a 2D Gaussian: 2 * pi * amplitude * sigma_x * sigma_y
    amplitude = total_counts / (2 * np.pi * x_stddev * y_stddev)
    
    constant_model = models.Gaussian2D(amplitude=amplitude,
                                       x_mean=x_mean,
                                       y_mean=y_mean,
                                       x_stddev=x_stddev,
                                       y_stddev=y_stddev)
    
    # Use a random generator for PSF placement.
    rng = np.random.default_rng(0)
    psf_loc = []
    half_psf = []
    data_psf = []
    
    # Define mask transmission function; if apply_mask is False, return 1.
    def mask_transmission(x_val, y_val, center=image_center, r0=30, sigma=10):
        if not apply_mask:
            return 1.0
        r = ((x_val - center[0])**2 + (y_val - center[1])**2)**0.5
        return 1 / (1 + np.exp(-(r - r0) / sigma))
    
    # Compute allowed offsets so that the stamp fits within the image.
    x_offset_min = stamp_shape[1] // 2 - center_x
    x_offset_max = image_shape[1] - stamp_shape[1] - center_x + stamp_shape[1] // 2
    y_offset_min = stamp_shape[0] // 2 - center_y
    y_offset_max = image_shape[0] - stamp_shape[0] - center_y + stamp_shape[0] // 2

    # Restrict offsets to ±100 pixels.
    desired_range = 100
    x_offset_min = max(x_offset_min, -desired_range)
    x_offset_max = min(x_offset_max, desired_range)
    y_offset_min = max(y_offset_min, -desired_range)
    y_offset_max = min(y_offset_max, desired_range)
    
    for i in range(n_psfs):
        # Render the PSF stamp.
        psf_stamp = np.zeros(stamp_shape)
        constant_model.bounding_box = None
        constant_model.render(psf_stamp)
        
        # Create a full image and insert the stamp.
        image = np.zeros(image_shape)
        y_offset = rng.integers(y_offset_min, y_offset_max + 1)
        x_offset = rng.integers(x_offset_min, x_offset_max + 1)
        x_start = center_x + x_offset - stamp_shape[1] // 2
        y_start = center_y + y_offset - stamp_shape[0] // 2
        
        final_x = x_start + x_mean
        final_y = y_start + y_mean
        psf_loc.append([final_x, final_y])
        
        # Compute the base throughput (unmasked core flux) for reference.
        # For a perfect Gaussian, roughly 50% of the flux is above half maximum.
        base_throughput = np.pi * amplitude * x_stddev * y_stddev  # equals total_counts/2
        transmission = mask_transmission(final_x, final_y)
        half_psf.append(base_throughput * transmission)
        
        # Apply transmission if requested.
        psf_stamp = psf_stamp * transmission
        
        image[y_start:y_start+stamp_shape[0], x_start:x_start+stamp_shape[1]] = psf_stamp
        
        data_psf.append(Image(image, pri_hdr=prhd, ext_hdr=exthd, err=err, dq=dq))
    
    return data_psf, np.array(psf_loc), np.array(half_psf)



def create_ct_cal(fwhm_mas, cfam_name='1F',
                  cenx = 50.5,ceny=50.5,
                  nx=21,ny=21,
                  psfsize=None):
    """
    Creates a mock CoreThroughputCalibration object with gaussian PSFs.

    Args:
        fwhm_mas (float): FWHM in milliarcseconds
        cfam_name (str, optional): CFAM name, defaults to '1F'.
        cenx (float, optional): EXCAM mask center X location (measured from bottom left corner of bottom left pixel)
        ceny (float, optional): EXCAM mask center Y location (measured from bottom left corner of bottom left pixel)
        nx (int, optional): Number of x positions at which to simulate mock PSFs. Must be an odd number. 
            PSFs will be generated in the center of each pixel within nx/2 pixels of the mask center. Defaults to 21.
        ny (int, optional): Number of y positions at which to simulate mock PSFs. Must be an odd number. 
            PSFs will be generated in the center of each pixel within nx/2 pixels of the mask center. Defaults to 21.
        psfsize (int,optional): Size of psf model array in pixels. Must be an odd number. Defaults to 6 * the FWHM.
    
    Returns:
        ct_cal (corgidrp.data.CoreThroughputCalibration): mock CoreThroughputCalibration object 
    """
    # Default headers
    prhd, exthd, errhdr, dqhdr = create_default_L3_headers()
    # cfam filter
    exthd['CFAMNAME'] = cfam_name
    exthd.set('EXTNAME','PSFCUBE')

    # Need nx, ny to be odd
    assert nx%2 == 1, 'nx must be an odd integer'
    assert ny%2 == 1, 'ny must be an odd integer'

    x_arr = []
    y_arr = []
    r_arr = []

    for x in np.linspace(cenx-(nx-1)/2,cenx+(nx-1)/2,nx):
        for y in np.linspace(ceny-(ny-1)/2,ceny+(ny-1)/2,ny):
            x_arr.append(x)
            y_arr.append(y)
            r_arr.append(np.sqrt((x - cenx)**2 + (y - ceny)**2))
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)

    n_psfs = len(x_arr)

    fwhm_pix = int(np.ceil(fwhm_mas/21.8))
    sig_pix = fwhm_pix / (2 * np.sqrt(2. * np.log(2.)))

    # PSF/PSF_peak > 1e-10 for +/- 3FWHM around the PSFs center
    if psfsize is None:
        imshape = (6*fwhm_pix+1, 6*fwhm_pix+1)
    else:
        assert psfsize%2 == 1, 'psfsize must be an odd integer'
        imshape = (psfsize,psfsize)

    psf = gaussian_array(array_shape=imshape,sigma=sig_pix,amp=1.,xoffset=0.,yoffset=0.)
    scale_factors = np.interp(r_arr, [0, np.max(r_arr)], [1, 0.01]) # throughput falls off linearly radially

    psf_cube = np.ones((n_psfs,*imshape))
    psf_cube *= psf
    amps = scale_factors
    psf_cube = np.array([psf_cube[i] * amps[i] for i in range(len(psf_cube))])

    err_cube = np.zeros_like(psf_cube)
    err_hdr = fits.Header()
    dq_cube = np.zeros_like(psf_cube)
    dq_hdr = fits.Header()

    cts = scale_factors
    ct_excam = np.array([x_arr,y_arr,cts])
    ct_hdr = fits.Header()
    ct_hdu_list = [fits.ImageHDU(data=ct_excam, header=ct_hdr, name='CTEXCAM')]

    fpam_hv = [0., 0.]
    fpam_hdr = fits.Header()
    fpam_hdr['COMMENT'] = 'FPAM H and V values during the core throughput observations'
    fpam_hdr['UNITS'] = 'micrometer'
    ct_hdu_list += [fits.ImageHDU(data=fpam_hv, header=fpam_hdr, name='CTFPAM')]

    fsam_hv = [0., 0.]
    fsam_hdr = fits.Header()
    fsam_hdr['COMMENT'] = 'FSAM H and V values during the core throughput observations'
    fsam_hdr['UNITS'] = 'micrometer'
    ct_hdu_list += [fits.ImageHDU(data=fsam_hv, header=fsam_hdr, name='CTFSAM')]

    ct_cal = data.CoreThroughputCalibration(psf_cube,
        pri_hdr=prhd,
        ext_hdr=exthd,
        input_hdulist=ct_hdu_list,
        dq=dq_cube,
        dq_hdr=dq_hdr,
        input_dataset=data.Dataset([data.Image(np.array([0.]),
                                 pri_hdr=fits.Header(),
                                 ext_hdr=fits.Header())]))

    return ct_cal



def create_ct_interp(
    n_radii=9,
    n_azimuths=5,
    min_angle=0,
    max_angle=6.2831853072,
    norm=1,
    fpm_x=0,
    fpm_y=0,
    pop_index=None,
    ):
    """
    Create simulated data to test the class function that does core throughput
    interpolation. We want to set the CT to a known function. We accomplish it
    by using a synthetic PSF shape with a well defined CT profile. The PSF is
    needed because the interpolation function takes a CT cal file that is
    generated from a Dataset of PSF images.

    Args:
        n_radii (int): Number of divisions along a radial direction.
        n_azimuths (int): Number of divisions along the azimuth, from
          zero to max_angle.
        min_angle (float): Minimum angle in radians to be considered.
        max_angle (float): Maximum angle in radians to be considered.
        norm (float): Factor to multiply the CT profile. Useful if one
          wants the CT to be between 0 and 1 after the division by the total counts
          that happens when estimating the CT of the Dataset in corethroughput.py.
        fpm_x (float): FPM location in EXCAM (fractional) pixels. First dimension.
        fpm_y (float): FPM location in EXCAM (fractional) pixels. Second dimension.
        pop_index (int) (optional): the Dataset skips the PSF with this index.
          Useful when testing interpolation by popping some PSFs and comparing
          the interpolated values with the original ones at the same location.

    Returns:
        corgidrp.data.Image: The simulated PSF Images
        np.array: PSF locations
        np.array: PSF CT values
    """
    if max_angle > 2*np.pi:
        print('You may have set a maximum angle in degrees instead of radians. '
            'Please check the value of max_angle is the one intended.')

    # The shape of all PSFs is the same, and irrelevant
    # Their amplitude will be adjusted to a specific CT profiledepending on
    # their location
    imshape=(7,7)
    psf_model = np.zeros(imshape)
    # Set of known values at selected locations
    psf_model[imshape[1]//2 - 3:imshape[1]//2 + 4,
        imshape[0]//2 - 3:imshape[0]//2 + 4] = 1
    psf_model[imshape[1]//2 - 2:imshape[1]//2 + 3,
        imshape[0]//2 - 2:imshape[0]//2 + 3] = 2
    psf_model[imshape[1]//2 - 1:imshape[1]//2 + 2,
        imshape[0]//2 - 1:imshape[0]//2 + 2] = 3
    psf_model[imshape[1]//2, imshape[0]//2] = 4

    # Default headers
    prhd, exthd, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    exthd['CFAMNAME'] = '1F'
    # Mock error
    err = np.ones([1024,1024])

    # Simulate PSFs within two radii
    psf_loc = []
    half_psf = []
    data_psf = []

    #Create a dataset
    # From 2 to 9 lambda/D
    radii = np.logspace(np.log10(2), np.log10(9),n_radii)
    # lambda/D ~ 2.3 EXCAM pixels for Band 1 and HLC
    radii *= 2.3
    # Threefold symmetry
    azimuths = np.linspace(min_angle, max_angle, n_azimuths)
    
    # Create 2D grids for the radii and azimuths
    r_grid, theta_grid = np.meshgrid(radii, azimuths)
    
    # Convert polar coordinates to Cartesian coordinates
    x_grid = np.round(fpm_x + r_grid * np.cos(theta_grid)).flatten()
    y_grid = np.round(fpm_y + r_grid * np.sin(theta_grid)).flatten()
    # Derive the final radial distance from the FPM's center
    r_grid_from_fpm = np.sqrt((x_grid-fpm_x)**2 + (y_grid-fpm_y)**2)
    # Make up a core throughput dataset
    core_throughput = r_grid_from_fpm.flatten()/r_grid_from_fpm.max()
    # Normalize to 1 by accounting for the contribution of the PSF to the CT
    core_throughput /= psf_model[psf_model>=psf_model.max()/2].sum()
    # Optionally, take into account an additional factor
    core_throughput *= norm
    for i_psf in range(r_grid.size):
        if pop_index is not None:
            if i_psf == pop_index:
                print('Skipping 1 PSF (interpolation test)')
                continue
        image = np.zeros([1024, 1024])
        # Insert PSF at random location within the SCI frame
        x_image = int(x_grid[i_psf])
        y_image = int(y_grid[i_psf])
        image[y_image-imshape[0]//2:y_image+imshape[0]//2+1,
            x_image-imshape[1]//2:x_image+imshape[1]//2+1] = psf_model
        # Adjust intensity following some radial profile
        image *= core_throughput[i_psf]
        # List of known positions
        psf_loc += [[x_image-imshape[0]//2, y_image-imshape[0]//2]]
        # Add numerator of core throughput
        half_psf += [image[image>=image.max()/2].sum()]
        # Build up the Dataset
        data_psf += [Image(image,pri_hdr=prhd, ext_hdr=exthd, err=err)]

    return data_psf, np.array(psf_loc), np.array(half_psf)

default_wcs_string = """WCSAXES =                    2 / Number of coordinate axes                      
CRPIX1  =                  0.0 / Pixel coordinate of reference point            
CRPIX2  =                  0.0 / Pixel coordinate of reference point            
CDELT1  =                  1.0 / Coordinate increment at reference point        
CDELT2  =                  1.0 / Coordinate increment at reference point        
CRVAL1  =                  0.0 / Coordinate value at reference point            
CRVAL2  =                  0.0 / Coordinate value at reference point            
LATPOLE =                 90.0 / [deg] Native latitude of celestial pole        
MJDREF  =                  0.0 / [d] MJD of fiducial time
"""

def create_psfsub_dataset(n_sci,n_ref,roll_angles,darkhole_scifiles=None,darkhole_reffiles=None,
                          wcs_header = None,
                          data_shape = [100,100],
                          centerxy = None,
                          outdir = None,
                          st_amp = 100.,
                          noise_amp = 1.,
                          fwhm_pix = 2.5,
                          ref_psf_spread=1. ,
                          pl_contrast=1e-3,
                          pl_sep = 10.
                          ):
    """Generate a mock science and reference dataset ready for the PSF subtraction step.
    TODO: reference a central pixscale number, rather than hard code.

    Args:
        n_sci (int): number of science frames, must be >= 1.
        n_ref (int): nummber of reference frames, must be >= 0.
        roll_angles (list-like): list of the roll angles of each science and reference 
            frame, with the science frames listed first. 
        darkhole_scifiles (list of str, optional): Filepaths to the darkhole science frames. 
            If not provided, a noisy 2D gaussian will be used instead. Defaults to None.
        darkhole_reffiles (list of str, optional): Filepaths to the darkhole reference frames. 
            If not provided, a noisy 2D gaussian will be used instead. Defaults to None.
        wcs_header (astropy.fits.Header, optional): Fits header object containing WCS 
            information. If not provided, a mock header will be created. Defaults to None.
        data_shape (list of int): desired shape of data array, with the last two axes in xy order. 
            Must have length 2 or 3. Defaults to [100,100].
        centerxy (list of float): Desired PSF center in xy order. Must have length 2. Defaults 
            to image center.
        outdir (str, optional): Desired output directory. If not provided, data will not be 
            saved. Defaults to None.
        st_amp (float): Amplitude of stellar psf added to fake data. Defaults to 100.
        fwhm_pix (float): FWHM of the stellar (and optional planet) PSF. Defaults to 2.5.
        noise_amp (float): Amplitude of gaussian noise added to fake data. Defaults to 1.
        ref_psf_spread (float): Fractional increase in gaussian PSF width between science and 
            reference PSFs. Defaults to 1.
        pl_contrast (float): Flux ratio between planet and starlight incident on the detector. 
            Defaults to 1e-3.
        pl_sep (float): Planet-star separation in pixels. Defaults to 10.

        
    Returns:
        tuple: corgiDRP science Dataset object and reference Dataset object.
    """

    assert len(data_shape) == 2 or len(data_shape) == 3
    
    if roll_angles is None:
        roll_angles = [0.] * (n_sci+n_ref)

    # mask_center = np.array(data_shape)/2
    # star_pos = mask_center
    pixscale = 21.8 # milli-arcsec

    # Build each science/reference frame
    sci_frames = []
    ref_frames = []
    for i in range(n_sci+n_ref):

        # Create default headers
        prihdr, exthdr, errhdr, dqhdr = create_default_L3_headers()
        
        # Read in darkhole data, if provided
        if i<n_sci and not darkhole_scifiles is None:
            fpath = darkhole_scifiles[i]
            _,fname = os.path.split(fpath)
            darkhole = fits.getdata(fpath)
            
            fill_value = np.nanmin(darkhole)
            img_data = np.full(data_shape[-2:],fill_value)

            # Overwrite center of array with the darkhole data
            cr_psf_pix = np.array(darkhole.shape) / 2 - 0.5
            if centerxy is None:
                full_arr_center = np.array(img_data.shape) // 2 
            else:
                full_arr_center = (centerxy[1],centerxy[0])
            start_psf_ind = full_arr_center - np.array(darkhole.shape) // 2
            img_data[start_psf_ind[0]:start_psf_ind[0]+darkhole.shape[0],start_psf_ind[1]:start_psf_ind[1]+darkhole.shape[1]] = darkhole
            psfcenty, psfcentx = cr_psf_pix + start_psf_ind
        
        elif i>=n_sci and not darkhole_reffiles is None:
            fpath = darkhole_reffiles[i-n_sci]
            _,fname = os.path.split(fpath)
            darkhole = fits.getdata(fpath)
            fill_value = np.nanmin(darkhole)
            img_data = np.full(data_shape[-2:],fill_value)

            # Overwrite center of array with the darkhole data
            cr_psf_pix = np.array(darkhole.shape) / 2 - 0.5
            if centerxy is None:
                full_arr_center = np.array(img_data.shape) // 2 
            else:
                full_arr_center = (centerxy[1],centerxy[0])
            start_psf_ind = full_arr_center - np.array(darkhole.shape) // 2
            img_data[start_psf_ind[0]:start_psf_ind[0]+darkhole.shape[0],start_psf_ind[1]:start_psf_ind[1]+darkhole.shape[1]] = darkhole
            psfcenty, psfcentx = cr_psf_pix + start_psf_ind

        # Otherwise generate a 2D gaussian for a fake PSF
        else:
            sci_fwhm = fwhm_pix
            ref_fwhm = sci_fwhm * ref_psf_spread
            pl_amp = st_amp * pl_contrast

            label = 'ref' if i>= n_sci else 'sci'
            fwhm = ref_fwhm if i>= n_sci else sci_fwhm
            sigma = fwhm / (2 * np.sqrt(2. * np.log(2.)))
            
            # Generate CGI filename with incrementing datetime
            visitid = prihdr["VISITID"]
            base_time = datetime.datetime.now()
            time_offset = datetime.timedelta(seconds=i)
            unique_time = base_time + time_offset
            time_str = data.format_ftimeutc(unique_time.isoformat())
            fname = f"cgi_{visitid}_{time_str}_l3_.fits"
            arr_center = np.array(data_shape[-2:]) / 2 - 0.5
            if centerxy is None:
                psfcenty,psfcentx = arr_center
            else:
                psfcentx,psfcenty = centerxy
            
            psf_off_xy = (psfcentx-arr_center[1],psfcenty-arr_center[0])
            img_data = gaussian_array(array_shape=data_shape[-2:],
                                      xoffset=psf_off_xy[0],
                                      yoffset=psf_off_xy[1],
                                      sigma=sigma,
                                      amp=st_amp)
            
            # Add some noise
            rng = np.random.default_rng(seed=123+2*i)
            noise = rng.normal(0,noise_amp,img_data.shape)
            img_data += noise

            # Add fake planet to sci files
            if i<n_sci:
                pa_deg = -roll_angles[i]
                xoff,yoff = pl_sep * np.array([-np.sin(np.radians(pa_deg)),np.cos(np.radians(pa_deg))])
                planet_psf = gaussian_array(array_shape=data_shape[-2:],
                                            amp=pl_amp,
                                            sigma=sigma,
                                            xoffset=xoff+psf_off_xy[0],
                                            yoffset=yoff+psf_off_xy[1])
                img_data += planet_psf
        
                # Assign PSFREF flag
                prihdr['PSFREF'] = 0
            else:
                prihdr['PSFREF'] = 1

        # Add necessary header keys
        prihdr['TELESCOP'] = 'ROMAN'
        prihdr['INSTRUME'] = 'CGI'
        prihdr['XOFFSET'] = 0.0
        prihdr['YOFFSET'] = 0.0
        prihdr["ROLL"] = roll_angles[i]
        
        exthdr['BUNIT'] = 'photoelectron/s'
        exthdr['STARLOCX'] = psfcentx
        exthdr['STARLOCY'] = psfcenty
        exthdr['EACQ_COL'] = psfcentx
        exthdr['EACQ_ROW'] = psfcentx
        exthdr['PLTSCALE'] = pixscale # This is in milliarcseconds!
        exthdr["HIERARCH DATA_LEVEL"] = 'L3'
        
        # Add WCS header info, if provided
        if wcs_header is None:
            wcs_header = generate_wcs(roll_angles[i], 
                                      [psfcentx,psfcenty],
                                      platescale=0.0218).to_header()
            
            # wcs_header._cards = wcs_header._cards[-1]
        exthdr.extend(wcs_header)

        # Make a corgiDRP Image frame
        if len(data_shape)==3:
            frame_data = np.zeros([data_shape[0],data_shape[2],data_shape[1]])
            frame_data[:] = img_data
        else:
            frame_data = img_data

        frame = data.Image(frame_data, pri_hdr=prihdr, ext_hdr=exthdr)
        frame.filename = fname

        # Add it to the correct dataset
        if i < n_sci:
            sci_frames.append(frame)
        else:
            ref_frames.append(frame)

    sci_dataset = data.Dataset(sci_frames)

    if len(ref_frames) > 0:
        ref_dataset = data.Dataset(ref_frames)
    else:
        ref_dataset = None

        # Save datasets if outdir was provided
        if not outdir is None:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
                
            # Generate CGI filenames for dataset saves
            visitid = prihdr["VISITID"]
            base_time = datetime.datetime.now()
            time_str = data.format_ftimeutc(base_time.isoformat())
            sci_filename = f"cgi_{visitid}_{time_str}_l2b.fits"
            ref_filename = f"cgi_{visitid}_{time_str}_l2b.fits"
            
            sci_dataset.save(filedir=outdir, filenames=[sci_filename])
            if len(ref_frames) > 0:
                # Offset time for reference dataset to avoid filename conflict
                ref_time = base_time + datetime.timedelta(seconds=1)
                ref_time_str = data.format_ftimeutc(ref_time.isoformat())
                ref_filename = f"cgi_{visitid}_{ref_time_str}_l2b.fits"
                ref_dataset.save(filedir=outdir, filenames=[ref_filename])

    return sci_dataset,ref_dataset


def generate_coron_dataset_with_companions(
    n_frames=1,
    shape=(1024, 1024),
    host_star_center=None,
    host_star_counts=1e5,
    roll_angles=None,
    companion_sep_pix=None,   # Companion separation in pixels (explicit) or list of separations
    companion_pa_deg=None,    # Companion position angle in degrees (counterclockwise from north) or list
    companion_counts=100.0,   # Total flux (counts) for the companion or list
    filter='1F',
    pltscale_as=21.8,
    add_noise=False,
    noise_std=1.0e-2,
    outdir=None,
    darkhole_file=None,   # darkhole_file is ignored in this version
    apply_coron_mask=True,
    coron_mask_radius=5,
    throughput_factors=1,
):
    """
    Create a mock coronagraphic dataset with a star and (optionally) one or more companions.
    
    In this version, a Gaussian star is always injected at host_star_center.
    When the coronagraph mask is applied, the flux within the inner mask region
    is scaled down by a factor of 1e-3.
    
    Args:
      n_frames (int): Number of frames.
      shape (tuple): (ny, nx) image shape.
      host_star_center (tuple): (x, y) pixel coordinates of the host star. If None, uses image center.
      host_star_counts (float): Total counts for the star.
      roll_angles (list of float): One roll angle per frame (in degrees).
      companion_sep_pix (float or list): On-sky separation(s) of the companion(s) in pixels.
      companion_pa_deg (float or list): On-sky position angle(s) of the companion(s) (counterclockwise from north).
      companion_counts (float or list): Total flux (counts) of the companion(s).
      filter (str): Filter name.
      pltscale_as (float): Plate scale in arcsec per pixel.
      add_noise (bool): Whether to add random noise.
      noise_std (float): Stddev of the noise.
      outdir (str or None): If given, saves the frames to disk.
      darkhole_file (Image): Ignored in this version.
      apply_coron_mask (bool): Whether to apply the simulated coronagraph mask.
      coron_mask_radius (int): Coronagraph mask radius in pixels.
      throughput_factors (float): Optical throughput of companion due to the presence of a coronagraph mask.
    
    Returns:
      Dataset: A dataset of frames (each an Image) with the star and companion(s) injected.
    """
    ny, nx = shape
    if host_star_center is None:
        host_star_center = (nx / 2, ny / 2)  # (x, y)

    if roll_angles is None:
        roll_angles = [0.0] * n_frames
    elif len(roll_angles) != n_frames:
        raise ValueError("roll_angles must have length n_frames or be None.")

    # If only one companion is provided, wrap parameters into lists.
    if companion_sep_pix is not None and companion_pa_deg is not None:
        if not isinstance(companion_sep_pix, list):
            companion_sep_pix = [companion_sep_pix]
        if not isinstance(companion_pa_deg, list):
            companion_pa_deg = [companion_pa_deg]
        if not isinstance(companion_counts, list):
            companion_counts = [companion_counts] * len(companion_sep_pix)

    frames = []

    for i in range(n_frames):
        angle_i = roll_angles[i]

        # Build an empty image frame.
        data_arr = np.zeros((ny, nx), dtype=np.float32)

        # (B) Insert the star as a 2D Gaussian centered at host_star_center.
        xgrid, ygrid = np.meshgrid(np.arange(nx), np.arange(ny))
        sigma_star = 1.2
        r2 = (xgrid - host_star_center[0])**2 + (ygrid - host_star_center[1])**2
        star_gaus = np.exp(-0.5 * r2 / sigma_star**2)
        star_gaus /= np.sum(star_gaus)         # Normalize the Gaussian.
        star_gaus *= host_star_counts          # Scale to the total star counts.
        data_arr += star_gaus

        # (Optional) Apply the coronagraph mask by scaling down counts in the inner region.
        if apply_coron_mask:
            data_arr[r2 < coron_mask_radius**2] *= 1e-3

        # (C) Insert the companion(s) if specified.
        companion_keywords = {}
        if companion_sep_pix is not None and companion_pa_deg is not None:
            for idx, (sep, pa, counts, throughput_factor) in enumerate(zip(companion_sep_pix, companion_pa_deg, 
                                                                           companion_counts, throughput_factors)):
                # Adjust the companion position based on the roll angle.
                rel_pa = pa - angle_i      
                # Use the helper function to convert separation and position angle to dx, dy.
                dx, dy = seppa2dxdy(sep, rel_pa)
                xcomp = host_star_center[0] + dx
                ycomp = host_star_center[1] + dy

                # Inject the companion as a small Gaussian PSF.
                sigma_c = 1.0
                rc2 = (xgrid - xcomp)**2 + (ygrid - ycomp)**2
                companion_gaus = np.exp(-0.5 * rc2 / sigma_c**2)
                companion_gaus /= np.sum(companion_gaus)
                companion_flux = counts * throughput_factor
                companion_gaus *= companion_flux
                data_arr += companion_gaus

                # Record the companion location in the header.
                # Create keys like SNYX001, SNYX002, etc.
                key = f"SNYX{idx+1:03d}"
                companion_keywords[key] = f"5.0,{ycomp:.2f},{xcomp:.2f}"

        # (D) Add noise if requested.
        if add_noise:
            noise = np.random.normal(0., noise_std, data_arr.shape)
            data_arr += noise.astype(np.float32)

        # (E) Build headers and create the Image.
        # Assume create_default_L3_headers() and generate_wcs() are defined elsewhere.
        prihdr, exthdr, errhdr, dqhdr = create_default_L3_headers()
        
        # Generate CGI filename with incrementing datetime
        visitid = prihdr["VISITID"]
        base_time = datetime.datetime.now()
        time_offset = datetime.timedelta(seconds=i)
        unique_time = base_time + time_offset
        time_str = data.format_ftimeutc(unique_time.isoformat())
        filename = f"cgi_{visitid}_{time_str}_l3_.fits"
        prihdr["FILENAME"] = filename
        prihdr["ROLL"] = angle_i
        prihdr['TELESCOP'] = 'ROMAN'
        exthdr["CFAMNAME"] = filter
        exthdr["PLTSCALE"] = pltscale_as*1000 #in milliarcsec
        exthdr["STARLOCX"] = host_star_center[0]
        exthdr["STARLOCY"] = host_star_center[1]
        exthdr["DATALVL"]  = "L3"
        exthdr['LSAMNAME'] = 'NFOV'
        exthdr['FPAMNAME'] = 'HLC12_C2R1'
        # Optional WCS generation.
        wcs_obj = generate_wcs(angle_i, [host_star_center[0], host_star_center[1]], platescale=pltscale_as)
        wcs_header = wcs_obj.to_header()
        exthdr.update(wcs_header)

        # Add companion header entries if any.
        if companion_keywords:
            for key, value in companion_keywords.items():
                exthdr[key] = value

        frame = Image(data_arr, pri_hdr=prihdr, ext_hdr=exthdr)
        frames.append(frame)

    dataset = Dataset(frames)

    # (F) Optionally save the dataset.
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        
        # Generate CGI filenames for all frames
        visitid = prihdr["VISITID"]
        base_time = datetime.datetime.now()
        file_list = []
        for i in range(n_frames):
            time_offset = datetime.timedelta(seconds=i)
            unique_time = base_time + time_offset
            time_str = data.format_ftimeutc(unique_time.isoformat())
            filename = f"cgi_{visitid}_{time_str}_l3_.fits"
            file_list.append(filename)
        
        dataset.save(filedir=outdir, filenames=file_list)
        print(f"Saved {n_frames} frames to {outdir}")

    return dataset


def create_mock_fpamfsam_cal(
    fpam_matrix=None,
    fsam_matrix=None,
    date_valid=None,
    save_file=False,
    output_dir=None,
    filename=None
):
    """
    Create and optionally save a mock FpamFsamCal object.

    Args:
        fpam_matrix (np.ndarray of shape (2,2) or None): The custom transformation matrix 
            from FPAM to EXCAM. If None, defaults to FpamFsamCal.fpam_to_excam_modelbased.
        fsam_matrix (np.ndarray of shape (2,2) or None): The custom transformation matrix 
            from FSAM to EXCAM. If None, defaults to FpamFsamCal.fsam_to_excam_modelbased.
        date_valid (astropy.time.Time or None): Date/time from which this calibration is 
            valid. If None, defaults to the current time.
        save_file (bool, optional): If True, save the generated calibration file to disk.
        output_dir (str, optional): Directory in which to save the file if save_file=True. 
            Defaults to current dir.
        filename (str, optional): Filename to use if saving to disk. If None, a default 
            name is generated.

    Returns:
        FpamFsamCal (corgidrp.data.FpamFsamCal object): The newly-created FpamFsamCal 
            object (in memory).
    """
    if fpam_matrix is None:
        fpam_matrix = FpamFsamCal.fpam_to_excam_modelbased
    if fsam_matrix is None:
        fsam_matrix = FpamFsamCal.fsam_to_excam_modelbased

    # Ensure the final shape is (2, 2, 2):
    # [ [fpam_matrix], [fsam_matrix] ]
    combined_array = np.array([fpam_matrix, fsam_matrix])  # shape (2,2,2)

    # Create the calibration object in-memory
    fpamfsam_cal = FpamFsamCal(data_or_filepath=combined_array, date_valid=date_valid)

    if save_file:
        # By default, use the filename from the object's .filename unless overridden
        if not filename:
            filename = fpamfsam_cal.filename  # e.g. "FpamFsamCal_<ISOTIME>.fits"

        if not output_dir:
            output_dir = '.'

        # Save the calibration file
        filepath = os.path.join(output_dir, filename)
        fpamfsam_cal.save(filedir=output_dir, filename=filename)
        print(f"Saved FpamFsamCal to {filepath}")

    return fpamfsam_cal

def create_mock_ct_dataset_and_cal_file(
    fwhm=50,
    n_psfs=100,
    cfam_name='1F',
    pupil_value_1=1,
    pupil_value_2=3,
    seed=None,
    save_cal_file=False,
    cal_filename=None,
    image_shape=(1024, 1024),
    total_counts = 1e4
):
    """
    Create simulated data for core throughput calibration.
    This function generates a mock dataset consisting of off-axis PSF images and pupil images,
    which are used to compute a core throughput calibration file. Two sets of PSF images are created:
    one with a simulated coronagraph mask (throughput reduced) and one without (unmasked). A calibration
    object is then generated from the masked dataset. Optionally, the calibration file can be saved to disk.
    
    Args:
        fwhm (float): Full Width at Half Maximum of the off-axis PSFs.
        n_psfs (int): Number of PSF images to generate.
        cfam_name (str): CFAM filter name used in the image headers.
        pupil_value_1 (int): Pixel value to assign to the first pupil image patch.
        pupil_value_2 (int): Pixel value to assign to the second pupil image patch.
        seed (int or None): Random seed for reproducibility. If provided, sets the NumPy random seed.
        save_cal_file (bool): If True, save the generated core throughput calibration file to disk.
        cal_filename (str or None): Filename for the calibration file. If None, a filename is generated based on the current time.
        image_shape (tuple): Shape (ny, nx) of the generated images.
        total_counts (float): Total counts assigned to the PSF images.
    
    Returns:
        dataset_ct_masked (Dataset): Dataset of masked (throughput reduced) PSF images.
        ct_cal (CoreThroughputCalibration): Generated core throughput calibration object.
        dataset_ct_nomask (Dataset): Dataset of unmasked PSF images.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # ----------------------------
    # A) Create the base headers
    # ----------------------------
    prihdr, exthd, errhdr, dqhdr = create_default_L3_headers()
    
    # Generate CGI filename
    visitid = prihdr["VISITID"]
    base_time = datetime.datetime.now()
    time_str = data.format_ftimeutc(base_time.isoformat())
    filename = f"cgi_{visitid}_{time_str}_ctp_cal.fits"
    prihdr["FILENAME"] = filename
    exthd['DRPCTIME'] = Time.now().isot
    exthd['DRPVERSN'] = corgidrp.__version__
    exthd['CFAMNAME'] = cfam_name

    exthd['FPAM_H'] = 1
    exthd['FPAM_V'] = 1
    exthd['FSAM_H'] = 1
    exthd['FSAM_V'] = 1

    exthd_pupil = exthd.copy()
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'

    # ----------------------------
    # B) Create the unocculted/pupil frames
    # ----------------------------
    pupil_image_1 = np.zeros(image_shape)
    pupil_image_2 = np.zeros(image_shape)
    ny, nx = image_shape
    y_center = ny // 2
    x_center = nx // 2
    patch_half_size = 10
    pupil_image_1[y_center - patch_half_size:y_center + patch_half_size,
                  x_center - patch_half_size:x_center + patch_half_size] = pupil_value_1
    pupil_image_2[y_center - patch_half_size:y_center + patch_half_size,
                  x_center - patch_half_size:x_center + patch_half_size] = pupil_value_2

    err = np.ones(image_shape)
    im_pupil1 = Image(pupil_image_1, pri_hdr=prihdr, ext_hdr=exthd_pupil, err=err)
    im_pupil2 = Image(pupil_image_2, pri_hdr=prihdr, ext_hdr=exthd_pupil, err=err)

    # ----------------------------
    # C) Create a set of off-axis PSFs (masked and unmasked)
    # ----------------------------
    data_psf_masked, psf_locs, half_psf = create_ct_psfs_with_mask(
        fwhm, cfam_name=cfam_name, n_psfs=n_psfs, image_shape=image_shape, apply_mask=True, 
        total_counts=total_counts
    )
    data_psf_nomask, _, _ = create_ct_psfs_with_mask(
        fwhm, cfam_name=cfam_name, n_psfs=n_psfs, image_shape=image_shape, apply_mask=False,
        total_counts=total_counts
    )

    # Combine frames for CT dataset (pupil frames + masked PSFs)
    data_ct_masked_temp = [im_pupil1, im_pupil2] + data_psf_masked
    dataset_ct_masked_temp = Dataset(data_ct_masked_temp)

    # ----------------------------
    # D) Generate the CT cal file
    # ----------------------------
    ct_cal_tmp = corethroughput.generate_ct_cal(dataset_ct_masked_temp)
    ct_cal_tmp.ext_hdr['STARLOCX'] = x_center
    ct_cal_tmp.ext_hdr['STARLOCY'] = y_center

    if save_cal_file:
        if not cal_filename:
            # Generate CGI filename for calibration file
            visitid = prihdr["VISITID"]
            base_time = datetime.datetime.now()
            time_str = data.format_ftimeutc(base_time.isoformat())
            cal_filename = f"cgi_{visitid}_{time_str}_ctp_cal.fits"
        cal_filepath = os.path.join(corgidrp.default_cal_dir, cal_filename)
        ct_cal_tmp.save(filedir=corgidrp.default_cal_dir, filename=cal_filename)
        print(f"Saved CT cal file to: {cal_filepath}")

    # Return datasets without the pupil images
    dataset_ct_nomask = Dataset(data_psf_nomask)
    dataset_ct_masked = Dataset(data_psf_masked)

    # Return both datasets and the calibration object.
    return dataset_ct_masked, ct_cal_tmp, dataset_ct_nomask



def generate_reference_star_dataset(
    n_frames=3,
    shape=(200, 200),
    star_center=(100,100),
    host_star_counts=1e5,
    roll_angles=None,
    add_noise=False,
    noise_std=1.0e-2,
    outdir=None
):
    """
    Generate a simulated reference star dataset for RDI or ADI+RDI processing.
    This function creates a set of mock frames of a reference star behind a coronagraph
    (with no planet), represented as a 2D Gaussian with a central masked (throughput-reduced)
    region. The resulting frames are assembled into a Dataset object and can optionally be saved to disk.
    
    Args:
        n_frames (int): Number of frames to generate.
        shape (tuple): Image shape (ny, nx) for each generated frame.
        star_center (tuple): Pixel coordinates (x, y) of the star center in the image.
        host_star_counts (float): Total integrated counts for the host star.
        roll_angles (list or None): Roll angles (in degrees) for each frame. If None, all frames are assigned 0.0.
        add_noise (bool): If True, add Gaussian noise to the images.
        noise_std (float): Standard deviation of the Gaussian noise.
        outdir (str or None): Directory to which the frames are saved if provided.
    
    Returns:
        dataset (corgidrp.Data.Dataset): A Dataset object containing the generated reference star frames.
    """

    # We'll adapt the same logic but no companion injection
    ny, nx = shape
    if roll_angles is None:
        roll_angles = [0.0]*n_frames

    frames = []
    for i in range(n_frames):
        data_arr = np.zeros((ny, nx), dtype=np.float32)

        # Insert a star behind coronagraph, e.g. as a 2D Gaussian with some hole
        xgrid, ygrid = np.meshgrid(np.arange(nx), np.arange(ny))
        sigma = 1.2
        r2 = (xgrid - star_center[0])**2 + (ygrid - star_center[1])**2
        star_gaus = (host_star_counts / (2*np.pi*sigma**2)) * np.exp(-0.5*r2/sigma**2)

        # Fake 5-pixel mask radius at center
        hole_mask = r2 < 5**2
        star_gaus[hole_mask] *= 1e-5

        data_arr += star_gaus.astype(np.float32)

        # Optional noise
        if add_noise:
            noise = np.random.normal(0., noise_std, data_arr.shape)
            data_arr += noise.astype(np.float32)

        # Build minimal headers
        prihdr, exthdr, errhdr, dqhdr = create_default_L3_headers()
        
        # Generate CGI filename with incrementing datetime
        visitid = prihdr["VISITID"]
        base_time = datetime.datetime.now()
        time_offset = datetime.timedelta(seconds=i)
        unique_time = base_time + time_offset
        time_str = data.format_ftimeutc(unique_time.isoformat())
        filename = f"cgi_{visitid}_{time_str}_l3_.fits"
        prihdr["FILENAME"] = filename
        # Mark these frames as reference
        prihdr["PSFREF"] = 1 
        prihdr["ROLL"] = roll_angles[i]
        exthdr["STARLOCX"] = star_center[0]
        exthdr["STARLOCY"] = star_center[1]

        # Make an Image
        frame = Image(data_arr, pri_hdr=prihdr, ext_hdr=exthdr)
        frames.append(frame)

    dataset = Dataset(frames)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        
        # Generate CGI filenames for all frames
        visitid = prihdr["VISITID"]
        base_time = datetime.datetime.now()
        file_list = []
        for i in range(n_frames):
            time_offset = datetime.timedelta(seconds=i)
            unique_time = base_time + time_offset
            time_str = data.format_ftimeutc(unique_time.isoformat())
            filename = f"cgi_{visitid}_{time_str}_l3_.fits"
            file_list.append(filename)
        
        dataset.save(filedir=outdir, filenames=file_list)
    return dataset


def create_synthetic_satellite_spot_image(
    image_shape,
    bg_sigma,
    bg_offset,
    gaussian_fwhm,
    separation,
    star_center=None,
    angle_offset=0,
    amplitude_multiplier=10,
):
    """
    Creates a synthetic 2D image with Gaussian background noise, a constant background,
    and four symmetric Gaussians.

    Args:
        image_shape (tuple of int):  
            The (ny, nx) shape of the image.
        bg_sigma (float):  
            Standard deviation of the background Gaussian noise.
        bg_offset (float):  
            Constant background level added to the entire image.
        gaussian_fwhm (float):  
            Full width at half maximum (FWHM) for the 2D Gaussian sources (in pixels).
        separation (float):  
            Radial separation (in pixels) of each Gaussian from the specified center.
        star_center (tuple of float, optional):  
            Absolute (x, y) position in the image at which the four Gaussians will be centered.
            If None, defaults to the image center (nx//2, ny//2).
        angle_offset (float, optional):  
            An additional angle (in degrees) to add to each default position angle.  
            The default Gaussians are at [0, 90, 180, 270] degrees; the final angles will be these 
            plus the `angle_offset`. Positive offsets rotate the Gaussians counterclockwise.
        amplitude_multiplier (float, optional):  
            Multiplier for the amplitude of the Gaussians relative to `bg_sigma`. By default, each 
            Gaussian’s amplitude is 10 * `bg_sigma`.

    Returns:
        numpy.ndarray:  
            The synthetic image (2D NumPy array) with background noise, a constant background, 
            and four added Gaussians.
    """
    # Create the background noise image with an added constant offset.
    image = np.random.normal(loc=0, scale=bg_sigma, size=image_shape) + bg_offset

    ny, nx = image_shape

    # Determine the center for the satellite spots. Default to image center if not specified.
    if star_center is None:
        center_x = nx // 2
        center_y = ny // 2
    else:
        center_x, center_y = star_center

    # Define the default position angles (in degrees) and add any additional angle offset.
    default_angles_deg = np.array([0, 90, 180, 270])
    angles_rad = np.deg2rad(default_angles_deg + angle_offset)

    # Compute the amplitude and convert FWHM to standard deviation.
    amplitude = amplitude_multiplier * bg_sigma
    # FWHM = 2 * sqrt(2 * ln(2)) * stddev  --> stddev = FWHM / (2*sqrt(2*ln(2)))
    stddev = gaussian_fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Create a grid of (x, y) pixel coordinates.
    y_indices, x_indices = np.indices(image_shape)

    # Add four Gaussians at the computed positions.
    for angle in angles_rad:
        dx = separation * np.cos(angle)
        dy = separation * np.sin(angle)
        gauss_center_x = center_x + dx
        gauss_center_y = center_y + dy

        gauss = Gaussian2D(
            amplitude=amplitude,
            x_mean=gauss_center_x,
            y_mean=gauss_center_y,
            x_stddev=stddev,
            y_stddev=stddev,
            theta=0,
        )
        image += gauss(x_indices, y_indices)

    return image


def rename_files_to_cgi_format(list_of_fits=None, output_dir=None, level_suffix="l1", pattern=None):
    """
    Renames FITS files to match CGI filename convention. Extracts visit ID and filetime 
    from headers and creates proper CGI format filenames.
    
    Args:
        list_of_fits (list, optional): List of FITS file paths or Image objects to rename.
                                      If None, will search for files using pattern.
        output_dir (str, optional): Directory to write renamed files to. 
                                  If None, files are renamed in-place.
        level_suffix (str, optional): Level suffix for filenames (e.g., "l1", "l2a", etc.)
        pattern (str, optional): Glob pattern to find files if list_of_fits is None.
                               Used for pump trap data renaming.
    
    Returns:
        list: Updated list of file paths with new CGI filenames
    """
    
    if list_of_fits is not None:
        files_to_process = list_of_fits
    elif pattern is not None:
        files_to_process = glob.glob(pattern)
        files_to_process.sort()  # Ensure consistent ordering
    else:
        raise ValueError("Either list_of_fits or pattern must be provided")
    
    renamed_files = []
    
    for i, file in enumerate(files_to_process):
        # Handle both file paths and Image objects
        if hasattr(file, 'pri_hdr') and hasattr(file, 'ext_hdr'):
            # Image object
            prihdr = file.pri_hdr
            exthdr = file.ext_hdr
            is_image_object = True
        else:
            # File path
            fits_file = fits.open(file)
            prihdr = fits_file[0].header
            exthdr = fits_file[1].header
            is_image_object = False
        
        # Visit ID from primary header VISITID keyword
        visitid = prihdr.get('VISITID', None)
        if visitid is not None:
            # Convert to string and pad to 19 digits if necessary
            visitid = str(visitid).zfill(19)
        else:
            # Fallback: try to extract from filename or use file index
            if hasattr(file, 'filename'):
                # Image object with filename attribute
                current_filename = file.filename
            else:
                # File path string
                current_filename = os.path.basename(file)
            if f'_{level_suffix}_' in current_filename:
                # Extract the frame number after the level suffix
                frame_number = current_filename.split(f'_{level_suffix}_')[-1].replace('.fits', '')
                visitid = frame_number.zfill(19)  
            elif current_filename.replace('.fits', '').isdigit():
                # Handle numbered files like 90500.fits
                frame_number = current_filename.replace('.fits', '')
                visitid = frame_number.zfill(19)  
            else:
                visitid = f"{i:019d}"  # Fallback: use file index padded to 19 digits
        
        # For pump trap data, create deterministic timestamps based on file metadata
        # Use EXCAMT (temperature), TPSCHEM (scheme), and TPTAU (phase time) to create unique timestamps
        excamt = exthdr.get('EXCAMT', None)
        tptau = exthdr.get('TPTAU', None)
        # Find which scheme this is (TPSCHEM1-4)
        scheme = None
        for j in range(1, 5):
            if exthdr.get(f'TPSCHEM{j}', 0) > 0:
                scheme = j
                break
        
        # Use file index as primary increment, but if we have pump trap metadata, use that for better uniqueness
        if excamt is not None and scheme is not None and tptau is not None:
            # Create unique timestamp based on temperature, scheme, and phase time
            # Start from a fixed base time
            base_dt = datetime.datetime(2025, 1, 1, 0, 0, 0)
            # Add seconds based on: temperature*10000 + scheme*1000 + file_index
            # This spreads files across a wide timestamp range
            temp_offset = int(float(excamt)) * 10  # e.g., 180K → 1800 seconds
            scheme_offset = scheme * 2000  # Each scheme gets 2000 seconds
            unique_dt = base_dt + datetime.timedelta(seconds=temp_offset + scheme_offset + i)
        else:
            # Fallback for non-pump-trap data
            filetime_hdr = exthdr.get('FILETIME', prihdr.get('FILETIME', None))
            if filetime_hdr and 'T' in filetime_hdr:
                try:
                    dt = datetime.datetime.fromisoformat(filetime_hdr.replace('Z', '+00:00'))
                    base_dt = dt
                except:
                    base_dt = datetime.datetime(2025, 1, 1, 0, 0, 0)
            else:
                base_dt = datetime.datetime(2025, 1, 1, 0, 0, 0)
            unique_dt = base_dt + datetime.timedelta(seconds=i)
        
        # Format as YYYYMMDDtHHMMSSd (deciseconds = 1 digit)
        filetime = unique_dt.strftime('%Y%m%dt%H%M%S%f')[:-5]
        
        # Create new filename with correct convention
        if level_suffix in ['l2a', 'l2b']:
            filename_template = f'cgi_{visitid}_{filetime}_{level_suffix}.fits'
        elif level_suffix.endswith('_cal'):
            # Calibration files should not have trailing underscore
            filename_template = f'cgi_{visitid}_{filetime}_{level_suffix}.fits'
        else:
            filename_template = f'cgi_{visitid}_{filetime}_{level_suffix}_.fits'
        
        if output_dir:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            new_filename = os.path.join(output_dir, filename_template)
        else:
            # Rename in same directory
            file_dir = os.path.dirname(file)
            new_filename = os.path.join(file_dir, filename_template)
        
        # Check if file already exists (collision detection)
        if os.path.exists(new_filename):
            warnings.warn(f"File collision detected: {os.path.basename(new_filename)} already exists! This file will be overwritten.")
        
        if is_image_object:
            # Update headers in the Image object
            file.pri_hdr['FILENAME'] = os.path.basename(new_filename)
            file.pri_hdr['VISITID'] = visitid
            
            # Save the Image
            file.save(filedir=output_dir or os.path.dirname(new_filename), 
                     filename=os.path.basename(new_filename))
        else:
            # Update headers in the HDUList object
            fits_file[0].header['FILENAME'] = os.path.basename(new_filename)
            fits_file[0].header['VISITID'] = visitid
            
            # Update FITS file with new filename
            fits_file.writeto(new_filename, overwrite=True)
            fits_file.close()
            
            # Remove old file only if renaming in-place
            # Don't remove original files when copying to a different directory
            if file != new_filename and os.path.exists(file) and not output_dir:
                os.remove(file)
        
        # Update the file in the list
        renamed_files.append(new_filename)
    
    return renamed_files


def create_satellite_spot_observing_sequence(
        n_sci_frames, n_satspot_frames, 
        image_shape=(201, 201), bg_sigma=1.0, bg_offset=10.0,
        gaussian_fwhm=5.0, separation=14.79, star_center=None, angle_offset=0,
        amplitude_multiplier=100, observing_mode='NFOV'):
    """
    Creates a single dataset of synthetic observing frames. The dataset contains:

        • Science frames (with amplitude_multiplier=0), simulating no satellite spots.
        • Satellite spot frames (with amplitude_multiplier > 0), simulating the presence of spots.

    Synthetic frames are generated using the create_synthetic_satellite_spot_image function, 
    with added Gaussian noise and adjustable parameters for background level, spot separation, 
    and overall amplitude scaling.

    Args:
        n_sci_frames (int): 
            Number of science frames without satellite spots.
        n_satspot_frames (int): 
            Number of frames with satellite spots.
        image_shape (tuple, optional): 
            Shape of the synthetic image (height, width). Defaults to (201, 201).
        bg_sigma (float, optional): 
            Standard deviation of the background noise. Defaults to 1.0.
        bg_offset (float, optional): 
            Offset of the background noise. Defaults to 10.0.
        gaussian_fwhm (float, optional): 
            Full width at half maximum of the Gaussian spot. Defaults to 5.0.
        separation (float, optional): 
            Separation between the satellite spots. Defaults to 14.79.
        star_center (tuple of float, optional):  
            Absolute (x, y) position in the image at which the four Gaussians will be centered.
            If None, defaults to the image center (nx//2, ny//2).
        angle_offset (float, optional): 
            Offset of the spot angles. Defaults to 0.
        amplitude_multiplier (int, optional): 
            Amplitude multiplier for the satellite spots. Defaults to 100.
        observing_mode (str, optional): 
            Observing mode. Must be one of ['NFOV', 'WFOV', 'SPEC660', 'SPEC730']. 
            Defaults to 'NFOV'.

    Returns:
        data.Dataset: 
            A single dataset object containing both science frames (no satellite spots) 
            and satellite spot frames. The science frames have header value "SATSPOTS" set to 0, 
            while the satellite spot frames have "SATSPOTS" set to 1.
    """

    assert len(image_shape) == 2, "Data shape needs to have two values"
    assert observing_mode in ['NFOV', 'WFOV', 'SPEC660', 'SPEC730'], \
        "Invalid mode. Mode has to be one of 'NFOV', 'WFOV', 'SPEC660', 'SPEC730'"

    sci_frames = []
    satspot_frames = []
    
    # Example of setting up headers
    prihdr, exthdr, errhdr, dqhdr = create_default_L3_headers(arrtype="SCI")
    prihdr['NAXIS1'] = image_shape[1]
    prihdr['NAXIS2'] = image_shape[0]
    prihdr["SATSPOTS"] = 0  # 0 if no satellite spots, 1 if satellite spots
    exthdr['FSMPRFL'] = f'{observing_mode}'  # Needed for initial guess of satellite spot parameters

    # Make science images (no satellite spots)
    for i in range(n_sci_frames):
        sci_image = create_synthetic_satellite_spot_image(
            image_shape, bg_sigma, bg_offset, gaussian_fwhm,
            separation, star_center, angle_offset,
            amplitude_multiplier=0
        )
        sci_frame = data.Image(sci_image, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
        sci_frame.pri_hdr["SATSPOTS"] = 0
        
        # Generate CGI filename with incrementing datetime for science frames
        visitid = sci_frame.pri_hdr["VISITID"]
        base_time = datetime.datetime.now()
        time_offset = datetime.timedelta(seconds=i)
        unique_time = base_time + time_offset
        time_str = data.format_ftimeutc(unique_time.isoformat())
        sci_frame.filename = f"cgi_{visitid}_{time_str}_l3_.fits"
        sci_frames.append(sci_frame)

    # Make satellite spot images
    for i in range(n_satspot_frames):
        satspot_image = create_synthetic_satellite_spot_image(
            image_shape, bg_sigma, bg_offset, gaussian_fwhm,
            separation, star_center, angle_offset, amplitude_multiplier
        )
        satspot_frame = data.Image(satspot_image, pri_hdr=prihdr.copy(), ext_hdr=exthdr.copy())
        satspot_frame.pri_hdr["SATSPOTS"] = 1
        
        # Generate CGI filename with incrementing datetime for satellite spot frames
        visitid = satspot_frame.pri_hdr["VISITID"]
        base_time = datetime.datetime.now()
        time_offset = datetime.timedelta(seconds=i + 1000)  # Offset to avoid conflicts with science frames
        unique_time = base_time + time_offset
        time_str = data.format_ftimeutc(unique_time.isoformat())
        satspot_frame.filename = f"cgi_{visitid}_{time_str}_l3_.fits"
        satspot_frames.append(satspot_frame)
    
    all_frames = sci_frames + satspot_frames
    dataset = data.Dataset(all_frames)

    return dataset

def get_formatted_filename(dt, visitid):
    """
    Generate filename with proper format: cgi_VISITID_YYYYMMDDtHHMMSSS_l2b.fits
    
    Args:
        dt (datetime): Datetime object
        visitid (str): Visit ID

    Returns:
        str: Formatted filename
    """
    timestamp = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]  # Remove microseconds, keep milliseconds
    return f"cgi_{visitid}_{timestamp}_l2b.fits"

def create_spatial_pol(dataset,filedir=None,nr=None,pfov_size=174,image_center_x=512,image_center_y=512,separation_diameter_arcsec=7.5,alignment_angle_WP1=0,alignment_angle_WP2=45,planet=None,band=None,dpamname='POL0'):
    """Turns a dataset of neptune or uranus images with single planet images into the images observed through the wollaston prisms also incorporates the spatial variation of polarization on the 
        surface of the planet

    
        Args:
            dataset (corgidrp.data.Dataset): a dataset of image frames that are raster scanned (L2a-level)
            filedir (str): Full path to directory to save the raster scanned images.
            nr (int): planet radius
            pfov_size (int): size of the image created for the polarization variation, typically ~ 2 x nr
            image_center_x (int): x coordinate of the center pixel of the final image with two orthogonal pol components
            image_center_y (int): y coordinate of the center pixel of the final image with two orthogonal pol components
            separation_diameter_arcsec (float): separation between the two orthogonal pol components in arcsecs
            alignment_angle_WP1 (int): wollaston prism angle for Pol0 - 0
            alignment_angle_WP2 (int): wollaston prism angle for Pol45- 45
            planet (str): neptune or uranus
            band (str): 1F or 4F
            dpamname (str): wollaston prism pol0 or pol45
            
    	Returns:
    		data.Dataset: dataset of uranus or neptune with spatial variation of polarization corresponding to specific wollaston prism
    		
	"""
    
    assert dpamname in ['POL0', 'POL45'], \
        "Invalid prism selected, must be 'POL0' or 'POL45'"
    
    
    # Size of the square array used for introducing spatial variation of polarization - equal to/greater than the twice of planet radius (to make sure that planet pixels are not cropped)
    pfov = pfov_size
    polar_fov = np.ones((pfov,pfov))
    x = np.arange(0,pfov)
    y = np.arange(0,pfov)
    xx, yy = np.meshgrid(x,y)
    nrr = np.sqrt((xx-(pfov//2))**2 + (yy-(pfov//2))**2)
    #Divide the pfov_size image into 4 quadrants with ones (true) and zeros (false) - true and flase quadrants are assigned specific pol values for uranus and neptune in band 1 and 4 based on 
    # the previous ground based observations
    polar_filter = np.logical_and(yy<.99*xx,yy<.99*(pfov-xx)) + np.logical_and(yy>.99*xx,yy>.99*(pfov-xx))


    
    # read in the medium combined HST images of neptune ad uranus
    for i in range(len(dataset)):
        target=dataset[i].pri_hdr['TARGET']
        filter=dataset[i].pri_hdr['FILTER']
        if planet==target and band==filter: 
            #image data corresponding to the planet and band required
            planet_image=dataset[i].data
    
    # add the spatial variation of polarization 
    if planet == 'uranus' and band=="1":
        polar_fov[polar_filter==True] = .0115
        polar_fov[polar_filter==False] = -0.0115
    elif planet == 'uranus' and band=="4" or planet == 'neptune' and band=="4":
        polar_fov[polar_filter==True] = .005
        polar_fov[polar_filter==False] = -0.005
    elif planet == 'neptune' and band=="1":
        polar_fov[polar_filter==True] = .006
        polar_fov[polar_filter==False] = -0.006
    r_xy = polar_fov
    
    n_rad=nr
    if n_rad is None:
        if planet.lower() =='neptune':
             n_rad = 60
        elif planet.lower() == 'uranus':
             n_rad = 95
    
    #make all the pixels greater than the planet radius as zero
    r_xy[nrr>=n_rad] = 0
    
    # the HST images contain only one image of neptune or uranus wheras the pol images through POL0 and POL45 have two images. Make two copies of the planet images
    u_data=planet_image
    I_1 = u_data.copy()
    I_2 = u_data.copy()
    centroid_init = centr.centroid_1dg(u_data)
    xc_init=int(centroid_init[0])
    yc_init=int(centroid_init[1])

    
    # estimate the angle and displacement according to the dpam position
    if dpamname == 'POL0':
    #place image according to specified angle
        angle_rad = (alignment_angle_WP1 * np.pi) / 180
    else:
        angle_rad = (alignment_angle_WP2 * np.pi) / 180

    
     
    # fixed plate scale is used here since there are the mock HST images before raster scanned.
    displacement_x = int(round((separation_diameter_arcsec * np.cos(angle_rad)) / (2 * 0.0218)))
    displacement_y = int(round((separation_diameter_arcsec * np.sin(angle_rad)) / (2 * 0.0218)))
    center_left = (image_center_x - displacement_x, image_center_y + displacement_y)
    center_right = (image_center_x + displacement_x, image_center_y - displacement_y)

    # create the pol image with zeros
    WP_image=np.ones(shape=(1024, 1024))
    

    image_radius = pfov_size // 2
    start_left = (center_left[0] - image_radius, center_left[1] - image_radius)
    start_right = (center_right[0] - image_radius, center_right[1] - image_radius)

    y, x = np.indices([np.shape(WP_image)[0], np.shape(WP_image)[1]])
    
    # insert the two pol images with spatial variation at the specified location. wp_pol is the POL0 or POL45 image of neptune/uranus that has to be raster scanned.
    WP_pol=WP_image.copy() 
    WP_pol[start_left[1]:start_left[1]+pfov, start_left[0]:start_left[0]+pfov]=I_1[yc_init - (pfov//2):yc_init + (pfov//2),xc_init - (pfov//2):xc_init + (pfov//2)]* 0.5 * (1+(2*r_xy)) 
    WP_pol[start_right[1]:start_right[1]+pfov, start_right[0]:start_right[0]+pfov]=I_2[yc_init - (pfov//2):yc_init + (pfov//2),xc_init - (pfov//2):xc_init + (pfov//2)]* 0.5 * (1-(2*r_xy))

    # create the default headers and modify the header keywords
    prihdr, exthdr = create_default_L1_headers()
    prihdr['TARGET']=planet
    exthdr['DPAMNAME'] = dpamname
    
    # Generate proper filename with current timestamp
    dt = datetime.datetime.now()
    ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
    visitid = prihdr.get('VISITID', '0000000000000000000')
    
    image = data.Image(WP_pol, pri_hdr=prihdr, ext_hdr=exthdr)
    image.pri_hdr.append(('FILTER',band), end=True)
    
    # Set proper L2a filename
    filename = f"cgi_{visitid}_{ftime}_l2a.fits"
    
    if filedir is not None:
        image.save(filedir=filedir, filename=filename)
    else:
        image.filename = filename
    pol_image=data.Dataset([image])
    return (pol_image) 

def create_mock_l2b_polarimetric_image(image_center=(512, 512), dpamname='POL0', observing_mode='NFOV',
                                       left_image_value=1, right_image_value=1, image_separation_arcsec=7.5, alignment_angle=None):
    """
    Creates mock L2b polarimetric data with two polarized images placed on the larger
    detector frame. Image size and placement depends on the wollaston used and the observing mode.

    Args:
        image_center (optional, tuple(int, int)): pixel location of where the two images are centered on the detector
        dpamname (optional, string): name of the wollaston prism used, accepted values are 'POL0' and 'POL45'
        observing_mode (optional, string): observing mode of the coronagraph
        left_image_value (optional, int): value to fill inside the radius of the left image, corresponding to 0 or 45 degree polarization
        right_image_value (optional, int): value to fill inside the radius of the right image, corresponding to 90 or 135 degree polarization
        image_separation_arcsec (optional, float): Separation between the two polarized images in arcseconds.        
        alignment_angle (optional, float): the angle in degrees of how the two polarized images are aligned with respect to the horizontal,
            defaults to 0 for WP1 and 45 for WP2
    
    Returns:
        corgidrp.data.Image: The simulated L2b polarimetric image
    """
    assert dpamname in ['POL0', 'POL45'], \
        "Invalid prism selected, must be 'POL0' or 'POL45'"
    
    # create initial blank frame
    image_data = np.zeros(shape=(1024, 1024))

    pixel_scale = 0.0218 #arcsec/pixel
    primary_d = 2.363114 #meters

    image_separation_arcsec = 7.5

    arcseconds_per_radian = 180 * 3600 / np.pi

    #determine radius of the images
    if observing_mode == 'NFOV':
        cfamname = '1F'
        outer_radius_lambda_over_d = 9.7
        central_wavelength = 0.5738e-6 #meters
        radius = int(round((outer_radius_lambda_over_d * ((central_wavelength) / primary_d) * arcseconds_per_radian) / pixel_scale))
    elif observing_mode == 'WFOV':
        cfamname = '4F'
        outer_radius_lambda_over_d = 20.1
        central_wavelength = 0.8255e-6 #meters
        radius = int(round((outer_radius_lambda_over_d * ((central_wavelength) / primary_d) * arcseconds_per_radian) / pixel_scale))
    else:
        cfamname = '1F'
        radius = int(round(1.9 / pixel_scale))
    
    #determine the center of the two images
    if alignment_angle is None:
        if dpamname == 'POL0':
            alignment_angle = 0
        else:
            alignment_angle = 45
   
    center_left, center_right = get_pol_image_centers(image_separation_arcsec, alignment_angle, pixel_scale, image_center)

    #fill the location where the images are with 1s
    y, x = np.indices([1024, 1024])
    image_data[((x - center_left[0])**2) + ((y - center_left[1])**2) <= radius**2] = left_image_value
    image_data[((x - center_right[0])**2) + ((y - center_right[1])**2) <= radius**2] = right_image_value
    
    #create L2b headers
    prihdr, exthdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    #define necessary header keywords
    exthdr['CFAMNAME'] = cfamname
    exthdr['DPAMNAME'] = dpamname
    exthdr['LSAMNAME'] = observing_mode
    exthdr['FSMPRFL'] = observing_mode

    image = data.Image(image_data, pri_hdr=prihdr, ext_hdr=exthdr)

    return image

def create_mock_l2b_polarimetric_image_with_satellite_spots(
    image_center=(512, 512), 
    dpamname='POL0', 
    observing_mode='NFOV',
    left_image_value=1, 
    right_image_value=1,
    image_separation_arcsec = 7.5,
    alignment_angle=None,
    image_shape =(1024,1024),
    star_center = None, 
    bg_sigma = 1e-4,  #Default values from test_l2b_to_l3
    bg_offset = 0,  #Default values from test_l2b_to_l3
    gaussian_fwhm = 2,  #Default values from test_l2b_to_l3
    separation = 14.79,  #Default values from test_l2b_to_l3
    angle_offset=0, #Default values from test_l2b_to_l3
    amplitude_multiplier=10):
    """
    Creates a mock L2b polarimetric image with two separated polarized channels (left and right), 
    where each channel contains four synthetic Gaussian satellite spots.

    The function first establishes the geometry and background of the dual-channel image
    and then overlays the satellite spot pattern, centered on the middle of each channel.

    Args:
        image_center (optional, tuple(int, int)): Pixel location (x, y) where the two images 
            are centered on the larger detector frame.
        dpamname (optional, string): Name of the Wollaston prism used, accepted values are 
            'POL0' and 'POL45'.
        observing_mode (optional, string): Observing mode of the coronagraph.
        left_image_value (optional, int): Constant value to fill inside the radius of the left 
            polarized image (0 or 45 degree polarization), before adding spots.
        right_image_value (optional, int): Constant value to fill inside the radius of the right 
            polarized image (90 or 135 degree polarization), before adding spots.
        image_separation_arcsec (optional, float): Separation between the two polarized images in arcseconds.        
        alignment_angle (optional, float): The angle in degrees of how the two polarized images 
            are aligned with respect to the horizontal. Defaults to 0 for POL0 and 45 for POL45.
        image_shape (tuple of int, optional): The (ny, nx) shape of the detector array.
        star_center (list of tuple of float, optional):  
            displacement (dx, dy) from the center of each channel at which the four Gaussians will be centered for each slice.
            If None, defaults to the center of each channel.
        bg_sigma (float, optional): Standard deviation of the background Gaussian noise applied 
            to the entire image.
        bg_offset (float, optional): Constant background level added to the entire image.
        gaussian_fwhm (float, optional): Full width at half maximum (FWHM, in pixels) for the 
            2D Gaussian satellite spots.
        separation (float, optional): Radial separation (in pixels) of each satellite spot 
            from the center of its respective polarized image.
        angle_offset (float, optional): An additional angle (in degrees) to rotate the four 
            satellite spots in each channel (counterclockwise).
        amplitude_multiplier (float, optional): Multiplier for the amplitude of the Gaussians 
            relative to `bg_sigma`. By default, amplitude is 10 * `bg_sigma`.
    
    Returns:
        corgidrp.data.Image: The simulated L2b polarimetric image containing satellite spots.
    """

    # Create polarimetric image
    # Adapted from create_mock_l2b_polarimetric_image
    assert dpamname in ['POL0', 'POL45'], \
    "Invalid prism selected, must be 'POL0' or 'POL45'"
    
    # create initial frame
    image_data = np.random.normal(loc=0, scale=bg_sigma, size=image_shape) + bg_offset

    pixel_scale = 0.0218 #arcsec/pixel
    primary_d = 2.363114 #meters
    arcseconds_per_radian = 180 * 3600 / np.pi
    #determine radius of the images
    if observing_mode == 'NFOV':
        cfamname = '1F'
        outer_radius_lambda_over_d = 9.7
        central_wavelength =0.5738 * 1e-6
        radius = int(round((outer_radius_lambda_over_d * (central_wavelength / primary_d) * arcseconds_per_radian) / pixel_scale))
    elif observing_mode == 'WFOV':
        cfamname = '4F'
        outer_radius_lambda_over_d = 20.1
        central_wavelength = 0.8255e-6 #meters
        radius = int(round((outer_radius_lambda_over_d * ((central_wavelength) / primary_d) * arcseconds_per_radian) / pixel_scale))
    else:
        cfamname = '1F'
        radius = int(round(1.9 / pixel_scale))
    
    #determine the center of the two images
    if alignment_angle is None:
        if dpamname == 'POL0':
            alignment_angle = 0
        else:
            alignment_angle = 45
    angle_rad = alignment_angle * (np.pi / 180)
    displacement_x = int(round((image_separation_arcsec * np.cos(angle_rad)) / (2 * pixel_scale)))
    displacement_y = int(round((image_separation_arcsec * np.sin(angle_rad)) / (2 * pixel_scale)))
    center_left = (image_center[0] - displacement_x, image_center[1] + displacement_y)
    center_right = (image_center[0] + displacement_x, image_center[1] - displacement_y)

    #fill the location where the images are with 1s
    y, x = np.indices(image_shape)
    image_data[((x - center_left[0])**2) + ((y - center_left[1])**2) <= radius**2] = left_image_value
    image_data[((x - center_right[0])**2) + ((y - center_right[1])**2) <= radius**2] = right_image_value
    
    # Add satellite spots in each image
    # Adapted from create_synthetic_satellite_spot_image

    # Define the default position angles (in degrees) and add any additional angle offset.
    default_angles_deg = np.array([0, 90, 180, 270])
    angles_rad = np.deg2rad(default_angles_deg + angle_offset)

    # Compute the amplitude and convert FWHM to standard deviation.
    amplitude = amplitude_multiplier * bg_sigma
    # FWHM = 2 * sqrt(2 * ln(2)) * stddev  --> stddev = FWHM / (2*sqrt(2*ln(2)))
    stddev = gaussian_fwhm / (2 * np.sqrt(2 * np.log(2)))
    y_indices, x_indices = np.indices(image_shape)

    for idx, center in enumerate([center_left, center_right]):
        center_x, center_y = center
        if star_center is not None:
            center_x  = center_x + star_center [idx][0]
            center_y = center_y + star_center[idx][1]

        for angle in angles_rad:
            dx = separation * np.cos(angle)
            dy = separation * np.sin(angle)
            gauss_center_x = center_x + dx
            gauss_center_y = center_y + dy

            gauss = Gaussian2D(
                amplitude=amplitude,
                x_mean=gauss_center_x,
                y_mean=gauss_center_y,
                x_stddev=stddev,
                y_stddev=stddev,
                theta=0,
            )
            image_data += gauss(x_indices, y_indices)

    #create L2b headers
    prihdr, exthdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    #define necessary header keywords
    exthdr['CFAMNAME'] = cfamname
    exthdr['DPAMNAME'] = dpamname
    exthdr['LSAMNAME'] = observing_mode
    exthdr['FSMPRFL'] = observing_mode
    prihdr["SATSPOTS"] = 1
    image = data.Image(image_data, pri_hdr=prihdr, ext_hdr=exthdr)

    return image
    
def create_mock_stokes_image_l4(
        image_size=256,
        fwhm=3,
        I0=1e4,
        badpixel_fraction=1e-3,
        p=0.1,
        theta_deg=20.0,
        seed=None
):
    """
    Generate mock L4 Stokes cube with Gaussian source and controlled polarization.

    Args:
        image_size (int): H x W size
        fwhm (float): Gaussian FWHM in pixels
        I0 (float): Peak intensity
        badpixel_fraction (float): Fraction of bad pixels
        p (float): Fractional polarization
        theta_deg (float): Polarization angle in degrees
        seed (int, optional): Random seed

    Returns:
        Image: Stokes cube Image object with data, err, dq, and headers
    """
    rng = np.random.default_rng(seed)

    # Gaussian source
    y, x = np.mgrid[0:image_size, 0:image_size]
    x0 = y0 = image_size / 2.0
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    I_map = I0 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))
    I_map_err = np.sqrt(I_map)  # simple photon noise

    # bad pixels
    n_pixels = I_map.size
    n_bad = int(n_pixels * badpixel_fraction)
    dq = np.zeros_like(I_map, dtype=int)
    if n_bad > 0:
        idx_bad = rng.choice(n_pixels, size=n_bad, replace=False)
        dq.flat[idx_bad] = 1
        I_map.flat[idx_bad] *= -1

    theta_obs = np.radians(theta_deg)
    Q_map = I_map * p * np.cos(2 * theta_obs)
    U_map = I_map * p * np.sin(2 * theta_obs)
    stokes_cube = np.stack([I_map, Q_map, U_map])

    stokes_err = np.stack([
        I_map_err,
        I_map_err,
        I_map_err
    ])
    stokes_cube += rng.normal(0.0, stokes_err)

    # headers
    prihdr, exthdr, errhdr, dqhdr = create_default_L4_headers()

    dq_out = np.broadcast_to(dq, stokes_cube.shape).copy()

    stokes_image = Image(
        stokes_cube,
        pri_hdr=prihdr,
        ext_hdr=exthdr,
        err=stokes_err,
        dq=dq_out,
        err_hdr=errhdr,
        dq_hdr=dqhdr
    )

    # add throughput extensions
    kl_thru = np.ones((image_size, image_size), dtype=float)
    ct_thru = np.ones((image_size, image_size), dtype=float)
    # not adding any particular extra keywords to the headers for now
    stokes_image.add_extension_hdu('KL_THRU', data=kl_thru, header=fits.Header())
    stokes_image.add_extension_hdu('CT_THRU', data=ct_thru, header=fits.Header())

    return stokes_image

def create_mock_stokes_i_image(total_counts, target_name, col_cor=None, seed=0, wv0_x=0.0, wv0_y=0.0, is_coronagraphic=False):
    """Create a mock L4 Stokes I image from a mock L4 Stokes cube.
    
    Args:
        total_counts (float): Total counts in the image
        target_name (str): Name of the target
        col_cor (float, optional): Color correction factor
        seed (int, optional): Random seed
        wv0_x (float, optional): Wavelength of the x-axis
        wv0_y (float, optional): Wavelength of the y-axis
        is_coronagraphic (bool, optional): Whether the image is coronagraphic

    Returns:
        Image: Mock Image object with data of shape [4, n, m], err and dq arrays included.
    """
    base_img = create_mock_stokes_image_l4(
        image_size=64,
        fwhm=3,
        I0=1e4,
        badpixel_fraction=0.0,
        p=0.0,
        theta_deg=0.0,
        seed=seed,
    )
    profile = gaussian_array(
        array_shape=(base_img.data.shape[1], base_img.data.shape[2]),
        sigma=3.0,
        amp=total_counts / (2.0 * np.pi * 3.0**2),
        xoffset=0.0,
        yoffset=0.0,
    )
    base_img.data[0] = profile
    base_img.data[1:] = 0.0
    base_img.err[0] = np.maximum(np.sqrt(np.abs(base_img.data[0])), 1.0)
    base_img.err[1:] = base_img.err[0]
    base_img.dq[:] = 0
    base_img.pri_hdr['TARGET'] = target_name
    base_img.ext_hdr['BUNIT'] = 'photoelectron/s'
    base_img.ext_hdr['DATALVL'] = 'L4'
    base_img.ext_hdr.setdefault('CFAMNAME', '3C')
    base_img.ext_hdr.setdefault('DPAMNAME', 'POL0')
    base_img.ext_hdr.setdefault('LSAMNAME', 'NFOV')
    base_img.ext_hdr['WV0_X'] = wv0_x
    base_img.ext_hdr['WV0_Y'] = wv0_y
    base_img.ext_hdr.setdefault('STARLOCX', 0.0)
    base_img.ext_hdr.setdefault('STARLOCY', 0.0)
    base_img.ext_hdr.setdefault('FPAM_H', 0.0)
    base_img.ext_hdr.setdefault('FPAM_V', 0.0)
    base_img.ext_hdr.setdefault('FSAM_H', 0.0)
    base_img.ext_hdr.setdefault('FSAM_V', 0.0)
    base_img.ext_hdr['FSMLOS'] = 1 if is_coronagraphic else 0
    if col_cor is not None:
        base_img.ext_hdr['COL_COR'] = col_cor
    return base_img

def create_mock_IQUV_image(n=64, m=64, fwhm=20, amp=1.0, pfrac=0.1, bg=0.0):
    """
    Create a mock Image with [I, Q, U, V] planes for testing.

    Args:
        n (int): Image height (pixels).
        m (int): Image width (pixels).
        fwhm (float): FWHM of the Gaussian PSF used for I.
        amp (float): Peak amplitude of the Gaussian PSF.
        pfrac (float): Polarization fraction. Q, U are scaled by this fraction.
        bg (float): Background level added to the image.

    Returns:
        Image: Mock Image object with data of shape [4, n, m], err and dq arrays included.
    """


    y, x = np.mgrid[0:n, 0:m]
    x0, y0 = 0.5*(m-1), 0.5*(n-1)

    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    r2 = (x-x0)**2 + (y-y0)**2
    I = bg + amp * np.exp(-0.5*r2/sigma**2)

    phi = np.arctan2(y-y0, x-x0)
    Q = -pfrac * I * np.cos(2*phi)
    U = -pfrac * I * np.sin(2*phi)
    V = np.zeros_like(I)

    cube = np.stack([I, Q, U, V], axis=0)

    pri_hdr = Header()
    ext_hdr = Header()
    ext_hdr["STARLOCX"] = float(x0)
    ext_hdr["STARLOCY"] = float(y0)

    return Image(
        cube,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        err=np.zeros_like(cube),
        dq=np.zeros(cube.shape, dtype=np.uint16),
        err_hdr=Header(),
        dq_hdr=Header(),
    )

def create_mock_polarization_l3_dataset(
        image_size=1024,
        fwhm=100.0,
        I0=1e4,
        badpixel_fraction=1e-3,
        fractional_error=None,
        p=0.1,
        theta_deg=20.0,
        roll_angles=None,
        prisms=None,
        seed=None,
        return_image_list=False
):
    """
    Generate mock L3 polarimetric datasets with controlled fractional polarization
    and polarization angles, including optional bad pixels and configurable intensity.

    Each dataset can contain multiple images corresponding to different Wollaston
    prisms and roll angles. For each image, a dual-beam simulation is performed
    to produce the two analyzer channels (e.g., 0/90 deg for POL0, 45/135 deg for POL45),
    and observational noise is applied according to the specified fractional error
    or photon noise.

    Args:
        image_size (int): Size of the square image (H x W).
        fwhm (float): Full width at half maximum of the Gaussian source in pixels.
        I0 (float): Peak intensity of the Gaussian source.
        badpixel_fraction (float): Fraction of randomly placed bad pixels (0-1).
        fractional_error (float or None): Fractional Gaussian noise; if None, photon noise is used.
        p (float): Fractional polarization (0-1).
        theta_deg (float): Polarization angle in degrees.
        roll_angles (list of float, optional): Roll angles per prism. Defaults to [-15, 15, -15, 15].
        prisms (list of str, optional): Prism orientations ('POL0', 'POL45'). Defaults to ['POL0','POL0','POL45','POL45'].
        seed (int, optional): Random seed.
        return_image_list (bool): If True, return list of Image objects instead of Dataset.

    Returns:
        Dataset: Synthetic Dataset object containing Image objects with data, error maps,
                 and data quality arrays.

    Raises:
        ValueError: If roll_angles and prisms lengths mismatch or prism name is invalid.
    """

    # --- defaults ---
    if roll_angles is None:
        roll_angles = [-15, 15, -15, 15]
    if prisms is None:
        prisms = ['POL0', 'POL0', 'POL45', 'POL45']

    if len(roll_angles) != len(prisms):
        raise ValueError("roll_angles and prisms must have the same length")

    rng = np.random.default_rng(seed)

    # --- Gaussian source ---
    y, x = np.mgrid[0:image_size, 0:image_size]
    x0 = y0 = image_size / 2.0
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    I_map = I0 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))

    # --- bad pixels ---
    n_pixels = I_map.size
    n_bad = int(n_pixels * badpixel_fraction)
    dq = np.zeros_like(I_map, dtype=int)
    if n_bad > 0:
        idx_bad = rng.choice(n_pixels, size=n_bad, replace=False)
        dq.flat[idx_bad] = 1
        I_map.flat[idx_bad] *= -1

    cubes_out = []
    cubes_out_err = []

    for roll, prism in zip(roll_angles, prisms):
        theta_obs = np.radians(theta_deg + roll)
        Q_map = I_map * p * np.cos(2 * theta_obs)
        U_map = I_map * p * np.sin(2 * theta_obs)

        # dual-beam prism simulation
        if prism == 'POL0':
            pair_cube = np.stack([0.5 * (I_map + Q_map),
                                  0.5 * (I_map - Q_map)])
        elif prism == 'POL45':
            pair_cube = np.stack([0.5 * (I_map + U_map),
                                  0.5 * (I_map - U_map)])
        else:
            raise ValueError(f"Invalid prism name: {prism}")

        # error map
        if fractional_error is not None:
            pair_err = abs(pair_cube) * fractional_error
        else:
            pair_err = np.sqrt(abs(pair_cube))

        pair_cube += rng.normal(loc=0.0, scale=pair_err)

        cubes_out.append(pair_cube)
        cubes_out_err.append(pair_err)

    # convert to arrays
    cubes_out = np.array(cubes_out)
    cubes_out_err = np.array(cubes_out_err)

    # --- headers ---
    try:
        prihdr, exthdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    except:
        prihdr = exthdr = errhdr = dqhdr = biashdr = Header()

    # --- broadcast dq ---
    dq_out = np.broadcast_to(dq, cubes_out.shape).copy()

    Image_out = []
    for i, (roll, prism) in enumerate(zip(roll_angles, prisms)):
        prihdr_i = prihdr.copy()
        exthdr_i = exthdr.copy()
        prihdr_i['ROLL'] = roll
        exthdr_i['DPAMNAME'] = prism
        Image_out.append(
            Image(
                cubes_out[i],
                pri_hdr=prihdr_i,
                ext_hdr=exthdr_i,
                err=cubes_out_err[i],
                dq=dq_out[i],
                err_hdr=errhdr,
                dq_hdr=dqhdr
            )
        )
    
    if return_image_list:
        return Image_out
    else: 
        Dataset_out = Dataset(Image_out)
        return Dataset_out

def get_pol_image_centers(image_separation_arcsec, alignment_angle, pixel_scale = 0.0218, image_center=(512, 512)):
    """
    Calculate the centers of the two polarized images based on the separation and alignment angle.

    Args:
        image_separation_arcsec (float): Separation between the two polarized images in arcseconds.
        alignment_angle (float): Angle in degrees of how the two polarized images are aligned with respect to the horizontal.
        pixel_scale (float): Plate scale in arcseconds per pixel.
        image_center (tuple(int, int), optional): Pixel location of where the two images are centered on the detector.

    Returns:
        tuple: Pixel locations of the centers of the two polarized images.
    """
    angle_rad = alignment_angle * (np.pi / 180)
    displacement_x = int(round((image_separation_arcsec * np.cos(angle_rad)) / (2 * pixel_scale)))
    displacement_y = int(round((image_separation_arcsec * np.sin(angle_rad)) / (2 * pixel_scale)))
    center_left = (image_center[0] - displacement_x, image_center[1] + displacement_y)
    center_right = (image_center[0] + displacement_x, image_center[1] - displacement_y)

    return center_left, center_right

def generate_mock_polcal_dataset(path_to_pol_ref_file, read_noise=200,
                            image_separation_arcsec=7.5, q_inst=0.5,u_inst=-0.1,
                            q_eff=0.8,uq_ct=0.05,u_eff=0.7,qu_ct=0.03):
    '''
    Generate a mock L2b polarimetric dataset for polcal testing

    Args:
        path_to_pol_ref_file (str): Path to the CSV file containing the reference polarization values
        read_noise (float): Read noise to be added to the images
        image_separation_arcsec (float): Separation between the two polarized images in arcseconds
        q_inst (float): Instrumental Q polarization in percentage
        u_inst (float): Instrumental U polarization in percentage
        q_eff (float): Q efficiency
        uq_ct (float): U to Q crosstalk
        u_eff (float): U efficiency
        qu_ct (float): Q to U crosstalk 

    Returns:
        corgidrp.data.Dataset: The simulated L2b polarimetric dataset for polcal testing
    '''
    
    #Read in the test polarization stellar database from test_data/
    pol_ref = pd.read_csv(path_to_pol_ref_file, skipinitialspace=True)
    pol_ref_targets = pol_ref["TARGET"].tolist()
    #Create mock data for three targets in the database - for each target inject known polarization
    image_list = []
    for i, target in enumerate(pol_ref_targets):
        #create two mock L2b polarimetric images for each target, one for each Wollaston prism angle
        #set left and right image values to zero so that only injected polarization is measured
        pol0 = create_mock_l2b_polarimetric_image(dpamname='POL0', 
                                                        observing_mode='NFOV', left_image_value=0, right_image_value=0)
        pol0.pri_hdr['TARGET'] = target
        pol45 = create_mock_l2b_polarimetric_image(dpamname='POL45', 
observing_mode='NFOV', left_image_value=0, right_image_value=0)
        pol45.pri_hdr['TARGET'] = target

        pol0.err = (np.ones_like(pol0.data) * 1)[None,:]
        pol45.err = (np.ones_like(pol45.data) * 1)[None,:]

        #Add Random Roll - This should still work everywhere. 
        random_roll = np.random.randint(0,360)
        pol0.pri_hdr['ROLL'] = random_roll
        pol45.pri_hdr['ROLL'] = random_roll

        #get the q and u values from the reference polarization degree and angle
        q, u = pol.get_qu_from_p_theta(pol_ref["P"].values[i]/100.0, pol_ref["PA"].values[i]+random_roll)
        q_meas = q * q_eff + u * uq_ct + q_inst/100.0
        u_meas = u * u_eff + q * qu_ct + u_inst/100.0
        # generate four gaussians scaled appropriately for the target's polarization
        gauss_array_shape = [26,26]
        gauss1 = gaussian_array(array_shape=gauss_array_shape,amp=1000000) * (1 + q_meas)/2 #left image, POL0
        gauss2 = gaussian_array(array_shape=gauss_array_shape,amp=1000000) * (1 - q_meas)/2 #right image, POL0
        gauss3 = gaussian_array(array_shape=gauss_array_shape,amp=1000000) * (1 + u_meas)/2 #left image, POL45
        gauss4 = gaussian_array(array_shape=gauss_array_shape,amp=1000000) * (1 - u_meas)/2 #right image, POL45
        #add the gaussians to the mock images
        center_left0, center_right0 = get_pol_image_centers(image_separation_arcsec, 0)
        center_left45, center_right45 = get_pol_image_centers(image_separation_arcsec, 45)
        pol0.data[center_left0[1]-gauss_array_shape[1]//2:center_left0[1]+gauss_array_shape[1]//2,
                  center_left0[0]-gauss_array_shape[0]//2:center_left0[0]+gauss_array_shape[0]//2] += gauss1
        pol0.data[center_right0[1]-gauss_array_shape[1]//2:center_right0[1]+gauss_array_shape[1]//2,
                  center_right0[0]-gauss_array_shape[0]//2:center_right0[0]+gauss_array_shape[0]//2] += gauss2
        pol45.data[center_left45[1]-gauss_array_shape[1]//2:center_left45[1]+gauss_array_shape[1]//2,
                   center_left45[0]-gauss_array_shape[0]//2:center_left45[0]+gauss_array_shape[0]//2] += gauss3
        pol45.data[center_right45[1]-gauss_array_shape[1]//2:center_right45[1]+gauss_array_shape[1]//2,
                   center_right45[0]-gauss_array_shape[0]//2:center_right45[0]+gauss_array_shape[0]//2] += gauss4
        
        pol0.err = (np.sqrt(pol0.data+read_noise**2))[None,:]
        pol45.err = (np.sqrt(pol45.data+read_noise**2))[None,:]

        image_list.append(pol0)
        image_list.append(pol45)

    mock_dataset = data.Dataset(image_list)
    for frame in mock_dataset.frames: 
        frame.pri_hdr['VISTYPE'] = "CGIVST_CAL_POL_SETUP"

    return mock_dataset