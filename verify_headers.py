"""
General header verification script for all CORGIDRP pipeline stages.

Verifies headers against documentation at:
- L1: https://corgidrp.readthedocs.io/en/main/data_formats/l1.html
- L2a: https://corgidrp.readthedocs.io/en/main/data_formats/l2a.html
- L2b: https://corgidrp.readthedocs.io/en/main/data_formats/l2b.html
- L3: https://corgidrp.readthedocs.io/en/main/data_formats/l3.html
- L4: https://corgidrp.readthedocs.io/en/main/data_formats/l4.html

Usage:
    python verify_headers.py --level L2a
    python verify_headers.py --level L1 --file path/to/file.fits
"""

import os
import argparse
import astropy.io.fits as fits

# =============================================================================
# L2a EXPECTED HEADERS WITH DTYPES
# =============================================================================

# Format: (keyword, expected_dtype) or just keyword for backward compatibility
# dtype can be: bool, int, float, str, or None for optional/variable types

L2A_PRIMARY_HEADERS = [
    # Identification
    ('SIMPLE', bool), ('BITPIX', int), ('NAXIS', int), ('EXTEND', bool),
    # Observation metadata
    ('VISITID', str), ('TELESCOP', str), ('INSTRUME', str), ('DETECTOR', str), ('TARGET', str),
    # Program information
    ('PROGNUM', str), ('EXECNUM', str), ('CAMPAIGN', str), ('SEGMENT', str), ('OBSNUM', str), ('VISNUM', str),
    # Pointing coordinates
    ('RA', str), ('DEC', str), ('EQUINOX', float), ('RAPM', str), ('DECPM', str),
    ('ROLL', str), ('PITCH', str), ('YAW', str),
    # Configuration parameters
    ('OPGAIN', str), ('PHTCNT', str), ('FRAMET', str), ('SATSPOTS', str), ('ISHOWFSC', str), ('HOWFSLNK', str),
    # Processing metadata
    ('CDMSVERS', int), ('FSWDVERS', str), ('ORIGIN', str), ('FILETIME', str),
    ('DATAVERS', int), ('VISTYPE', str), ('FILENAME', str),
    # Associated files
    ('CPGSFILE', type(None)), ('AUXFILE', type(None)),
    # Additional
    ('OBSNAME', type(None)), ('PSFREF', str)
]

L2A_IMAGE_HEADERS = [
    # Data structure
    ('XTENSION', str), ('BITPIX', int), ('NAXIS', int), ('NAXIS1', int), ('NAXIS2', int),
    ('PCOUNT', int), ('GCOUNT', int), ('BUNIT', str), ('ARRTYPE', str),
    # Timing
    ('SCTSRT', str), ('SCTEND', str), ('DATETIME', str), ('FTIMEUTC', str),
    # Exposure parameters
    ('EXPTIME', float), ('EXPCYC', int), ('BLNKTIME', float), ('BLNKCYC', int), ('LASTEXP', int),
    # Gain/calibration
    ('EMGAIN_C', float), ('EMGAIN_A', int), ('KGAINPAR', float),
    ('EMGAINA1', float), ('EMGAINA2', float), ('EMGAINA3', float),
    ('EMGAINA4', float), ('EMGAINA5', float), ('GAINTCAL', float), ('UNITYG', int),
    # Temperature data
    ('EXCAMT', str), ('LOCAMT', str),
    # Operational modes
    ('OPMODE', str), ('ISPC', int), ('STATUS', int), ('HVCBIAS', int),
    # Hardware status
    ('CYCLES', int), ('OVEREXP', int), ('NOVEREXP', int), ('PROXET', str),
    # Control systems
    ('FCMLOOP', str), ('FCMPOS', int), ('FSMINNER', str), ('FSMLOS', str),
    ('FSMPRFL', str), ('FSMRSTR', int), ('DMZLOOP', int),
    # Alignment offsets
    ('FSMX', float), ('FSMY', float), ('FSMSG1', float), ('FSMSG2', float), ('FSMSG3', float),
    ('SB_FP_DX', float), ('SB_FP_DY', float), ('SB_FS_DX', float), ('SB_FS_DY', float),
    ('EACQ_ROW', float), ('EACQ_COL', float),
    # Wavefront sensing - Zernike coefficients
    ('Z2AVG', float), ('Z2RES', float), ('Z2VAR', float),
    ('Z3AVG', float), ('Z3RES', float), ('Z3VAR', float),
    ('Z4AVG', float), ('Z4RES', float),
    ('Z5AVG', float), ('Z5RES', float),
    ('Z6AVG', float), ('Z6RES', float),
    ('Z7AVG', float), ('Z7RES', float),
    ('Z8AVG', float), ('Z8RES', float),
    ('Z9AVG', float), ('Z9RES', float),
    ('Z10AVG', float), ('Z10RES', float),
    ('Z11AVG', float), ('Z11RES', float),
    ('Z12AVG', float), ('Z13AVG', float), ('Z14AVG', float),
    ('1SVALID', int), ('10SVALID', int),
    # Actuator positions
    ('SPAM_H', float), ('SPAM_V', float), ('SPAMNAME', str), ('SPAMSP_H', float), ('SPAMSP_V', float),
    ('FPAM_H', float), ('FPAM_V', float), ('FPAMNAME', str), ('FPAMSP_H', float), ('FPAMSP_V', float),
    ('LSAM_H', float), ('LSAM_V', float), ('LSAMNAME', str), ('LSAMSP_H', float), ('LSAMSP_V', float),
    ('FSAM_H', float), ('FSAM_V', float), ('FSAMNAME', str), ('FSAMSP_H', float), ('FSAMSP_V', float),
    ('CFAM_H', float), ('CFAM_V', float), ('CFAMNAME', str), ('CFAMSP_H', float), ('CFAMSP_V', float),
    ('DPAM_H', float), ('DPAM_V', float), ('DPAMNAME', str), ('DPAMSP_H', float), ('DPAMSP_V', float),
    # Processing flags - L2a specific
    ('DATALVL', str), ('MISSING', bool), ('DESMEAR', bool), ('CTI_CORR', bool), ('IS_BAD', bool),
    # Saturation info - L2a specific
    ('FWC_PP_E', float), ('FWC_EM_E', float), ('SAT_DN', float),
    # Processing history - L2a specific
    ('RECIPE', str), ('DRPVERSN', str), ('DRPCTIME', str), ('HISTORY', str)
]

L2A_ERR_HEADERS = [
    ('XTENSION', str), ('BITPIX', int), ('NAXIS', int), ('NAXIS1', int), ('NAXIS2', int), ('NAXIS3', int),
    ('PCOUNT', int), ('GCOUNT', int), ('EXTNAME', str), ('TRK_ERRS', bool), ('LAYER_1', str), ('HISTORY', str)
]

L2A_DQ_HEADERS = [
    ('XTENSION', str), ('BITPIX', int), ('NAXIS', int), ('NAXIS1', int), ('NAXIS2', int),
    ('PCOUNT', int), ('GCOUNT', int), ('BSCALE', int), ('BZERO', int), ('EXTNAME', str)
]

L2A_BIAS_HEADERS = [
    ('XTENSION', str), ('BITPIX', int), ('NAXIS', int), ('NAXIS1', int),
    ('PCOUNT', int), ('GCOUNT', int), ('EXTNAME', str)
]

# =============================================================================
# L1 EXPECTED HEADERS (to be filled in when L1 verification is needed)
# =============================================================================

L1_PRIMARY_HEADERS = []  # TODO: Fill in from L1 documentation
L1_IMAGE_HEADERS = []    # TODO: Fill in from L1 documentation

# =============================================================================
# L2b EXPECTED HEADERS (to be filled in when L2b verification is needed)
# =============================================================================

L2B_PRIMARY_HEADERS = []  # TODO: Fill in from L2b documentation
L2B_IMAGE_HEADERS = []    # TODO: Fill in from L2b documentation

# =============================================================================
# Header verification configurations by data level
# =============================================================================

VERIFICATION_CONFIG = {
    'L2a': {
        'hdus': {
            0: {'name': 'Primary', 'headers': L2A_PRIMARY_HEADERS},
            1: {'name': 'Image', 'headers': L2A_IMAGE_HEADERS},
            2: {'name': 'ERR', 'headers': L2A_ERR_HEADERS},
            3: {'name': 'DQ', 'headers': L2A_DQ_HEADERS},
            4: {'name': 'BIAS', 'headers': L2A_BIAS_HEADERS}
        },
        'check_history': True
    },
    'L1': {
        'hdus': {
            0: {'name': 'Primary', 'headers': L1_PRIMARY_HEADERS},
            1: {'name': 'Image', 'headers': L1_IMAGE_HEADERS}
        },
        'check_history': False
    },
    'L2b': {
        'hdus': {
            0: {'name': 'Primary', 'headers': L2B_PRIMARY_HEADERS},
            1: {'name': 'Image', 'headers': L2B_IMAGE_HEADERS}
        },
        'check_history': True
    }
    # Add L3, L4, etc. as needed
}


def normalize_dtype(value_type):
    """
    Normalize Python types to match FITS header value types.

    Args:
        value_type: Python type of the header value

    Returns:
        Normalized type (bool, int, float, str, or NoneType)
    """
    if value_type == type(None) or value_type is None or (hasattr(value_type, '__name__') and value_type.__name__ == 'NoneType'):
        return type(None)
    elif value_type == bool or (hasattr(value_type, '__name__') and value_type.__name__ == 'bool_'):
        return bool
    elif value_type == int or (hasattr(value_type, '__name__') and value_type.__name__ in ['int', 'int32', 'int64']):
        return int
    elif value_type == float or (hasattr(value_type, '__name__') and value_type.__name__ in ['float', 'float32', 'float64']):
        return float
    elif value_type == str or (hasattr(value_type, '__name__') and value_type.__name__ in ['str', 'str_']):
        return str
    else:
        return value_type


def verify_hdu_headers(hdu, expected_headers, hdu_name):
    """
    Verify headers in a single HDU, including dtype checking.

    Args:
        hdu: FITS HDU object
        expected_headers (list): List of expected headers (keyword or (keyword, dtype))
        hdu_name (str): Name of the HDU for reporting

    Returns:
        dict: {'present': list, 'missing': list, 'dtype_mismatch': list}
    """
    results = {'present': [], 'missing': [], 'dtype_mismatch': []}

    for header_spec in expected_headers:
        # Handle both tuple format (keyword, dtype) and plain keyword
        if isinstance(header_spec, tuple):
            keyword, expected_dtype = header_spec
        else:
            keyword = header_spec
            expected_dtype = None

        if keyword in hdu.header:
            results['present'].append(keyword)

            # Check dtype if specified
            if expected_dtype is not None:
                actual_value = hdu.header[keyword]
                actual_dtype = normalize_dtype(type(actual_value))
                expected_dtype_norm = normalize_dtype(expected_dtype)

                if actual_dtype != expected_dtype_norm:
                    # Format type names for display
                    expected_name = 'NoneType' if expected_dtype_norm == type(None) else expected_dtype_norm.__name__
                    actual_name = 'NoneType' if actual_dtype == type(None) else actual_dtype.__name__

                    results['dtype_mismatch'].append({
                        'keyword': keyword,
                        'expected': expected_name,
                        'actual': actual_name,
                        'value': actual_value
                    })
        else:
            results['missing'].append(keyword)

    return results


def verify_headers(fits_file, data_level):
    """
    Verify that all expected headers exist in the FITS file for the given data level.

    Args:
        fits_file (str): Path to FITS file
        data_level (str): Data level (e.g., 'L1', 'L2a', 'L2b', 'L3', 'L4')

    Returns:
        dict: Verification results
    """
    if data_level not in VERIFICATION_CONFIG:
        raise ValueError(f"Data level '{data_level}' not configured. Available: {list(VERIFICATION_CONFIG.keys())}")

    config = VERIFICATION_CONFIG[data_level]

    print(f"\n{'='*80}")
    print(f"Verifying {data_level} Headers: {os.path.basename(fits_file)}")
    print(f"{'='*80}\n")

    results = {}
    summary = {}
    total_dtype_mismatches = 0

    with fits.open(fits_file) as hdul:
        print(f"Total HDUs in file: {len(hdul)}\n")

        # Verify each configured HDU
        for hdu_idx, hdu_config in config['hdus'].items():
            hdu_name = hdu_config['name']
            expected_headers = hdu_config['headers']

            print(f"Checking {hdu_name} HDU (HDU {hdu_idx})...")

            if len(hdul) <= hdu_idx:
                print(f"  ✗ HDU {hdu_idx} not found in file!")
                results[hdu_name] = {'present': [], 'missing': expected_headers, 'dtype_mismatch': []}
                summary[f'{hdu_name.lower()}_total'] = len(expected_headers)
                summary[f'{hdu_name.lower()}_present'] = 0
                summary[f'{hdu_name.lower()}_missing'] = len(expected_headers)
                summary[f'{hdu_name.lower()}_dtype_mismatch'] = 0
                continue

            if not expected_headers:
                print(f"  ⚠ No expected headers configured for {hdu_name} HDU")
                results[hdu_name] = {'present': [], 'missing': [], 'dtype_mismatch': []}
                continue

            hdu_results = verify_hdu_headers(hdul[hdu_idx], expected_headers, hdu_name)
            results[hdu_name] = hdu_results

            # Report results
            total = len(expected_headers)
            present = len(hdu_results['present'])
            missing = len(hdu_results['missing'])
            dtype_mismatch = len(hdu_results['dtype_mismatch'])

            print(f"  Present: {present}/{total}")
            if hdu_results['missing']:
                print(f"  Missing: {missing} headers")
                for m in hdu_results['missing']:
                    print(f"    - {m}")
            else:
                print(f"  ✓ All expected {hdu_name} headers present!")

            if hdu_results['dtype_mismatch']:
                print(f"  ⚠ Dtype mismatches: {dtype_mismatch}")
                for dm in hdu_results['dtype_mismatch']:
                    print(f"    - {dm['keyword']}: expected {dm['expected']}, got {dm['actual']} (value={dm['value']})")
                total_dtype_mismatches += dtype_mismatch

            summary[f'{hdu_name.lower()}_total'] = total
            summary[f'{hdu_name.lower()}_present'] = present
            summary[f'{hdu_name.lower()}_missing'] = missing
            summary[f'{hdu_name.lower()}_dtype_mismatch'] = dtype_mismatch

        # Check HISTORY entries if configured
        if config.get('check_history', False):
            print("\nChecking HISTORY entries...")
            img_hdr = hdul[1].header
            history_entries = [h for h in img_hdr if h == 'HISTORY']
            if history_entries:
                print(f"  ✓ Found {len(history_entries)} HISTORY entries")
                for i, entry in enumerate(img_hdr['HISTORY'], 1):
                    print(f"    {i}. {entry}")
                summary['history_count'] = len(history_entries)
            else:
                print(f"  ✗ No HISTORY entries found (expected for {data_level})")
                summary['history_count'] = 0
        else:
            summary['history_count'] = 0

    summary['total_dtype_mismatches'] = total_dtype_mismatches
    results['summary'] = summary
    return results


def print_summary(results, data_level):
    """Print overall verification summary."""
    print(f"\n{'='*80}")
    print(f"{data_level} VERIFICATION SUMMARY")
    print(f"{'='*80}")

    summary = results['summary']
    total_missing = 0
    total_dtype_mismatches = summary.get('total_dtype_mismatches', 0)

    # Print each HDU's results
    for key in summary:
        if key.endswith('_total'):
            hdu_name = key.replace('_total', '').upper()
            total = summary[f'{hdu_name.lower()}_total']
            present = summary[f'{hdu_name.lower()}_present']
            missing = summary[f'{hdu_name.lower()}_missing']
            dtype_mismatch = summary.get(f'{hdu_name.lower()}_dtype_mismatch', 0)

            if total > 0:
                print(f"\n{hdu_name} HDU:")
                print(f"  Expected:        {total}")
                print(f"  Present:         {present} ({present/total*100:.1f}%)")
                print(f"  Missing:         {missing}")
                print(f"  Dtype mismatch:  {dtype_mismatch}")
                total_missing += missing

    if summary.get('history_count', 0) > 0:
        print(f"\nProcessing History:")
        print(f"  HISTORY entries: {summary['history_count']}")

    # Overall verdict
    print(f"\n{'='*80}")
    if total_missing == 0 and total_dtype_mismatches == 0:
        print(f"✓ ALL EXPECTED HEADERS PRESENT WITH CORRECT DTYPES FOR {data_level}")
    else:
        issues = []
        if total_missing > 0:
            issues.append(f"{total_missing} headers missing")
        if total_dtype_mismatches > 0:
            issues.append(f"{total_dtype_mismatches} dtype mismatches")
        print(f"⚠ {', '.join(issues)}")
    print(f"{'='*80}\n")


def save_results(results, data_level, fits_file, output_file):
    """Save verification results to file."""
    with open(output_file, 'w') as f:
        f.write(f"{data_level} Header Verification Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"File: {os.path.basename(fits_file)}\n\n")

        # Write missing headers for each HDU
        for hdu_name, hdu_results in results.items():
            if hdu_name == 'summary':
                continue

            f.write(f"{hdu_name} HDU:\n")
            f.write("-"*40 + "\n")

            if hdu_results['missing']:
                f.write("Missing Headers:\n")
                for h in hdu_results['missing']:
                    f.write(f"  - {h}\n")
            else:
                f.write("Missing Headers: None\n")

            if hdu_results.get('dtype_mismatch'):
                f.write("\nDtype Mismatches:\n")
                for dm in hdu_results['dtype_mismatch']:
                    f.write(f"  - {dm['keyword']}: expected {dm['expected']}, got {dm['actual']} (value={dm['value']})\n")
            else:
                f.write("Dtype Mismatches: None\n")
            f.write("\n")

        # Write summary
        summary = results['summary']
        f.write("="*80 + "\n")
        f.write("Summary:\n")
        for key in summary:
            if key.endswith('_total'):
                hdu_name = key.replace('_total', '').capitalize()
                total = summary[key]
                present = summary[f'{key.replace("_total", "_present")}']
                missing = summary[f'{key.replace("_total", "_missing")}']
                dtype_mismatch = summary.get(f'{key.replace("_total", "_dtype_mismatch")}', 0)
                if total > 0:
                    f.write(f"  {hdu_name:8s}: {present}/{total} present, {missing} missing, {dtype_mismatch} dtype mismatches\n")
        if summary.get('history_count', 0) > 0:
            f.write(f"  History : {summary['history_count']} entries\n")

        f.write(f"\nTotal dtype mismatches across all HDUs: {summary.get('total_dtype_mismatches', 0)}\n")

    print(f"Detailed results saved to: {output_file}")


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description='Verify CORGIDRP pipeline data headers')
    parser.add_argument('--level', type=str, default='L2a',
                        help='Data level to verify (L1, L2a, L2b, L3, L4)')
    parser.add_argument('--file', type=str, default=None,
                        help='Specific FITS file to verify (otherwise uses first file from pipeline_output)')
    args = parser.parse_args()

    data_level = args.level

    # Find file to verify
    if args.file:
        fits_file = args.file
    else:
        # Auto-detect from pipeline_output
        level_dir_map = {
            'L1': 'input_l1',
            'L2a': 'l1_to_l2a',
            'L2b': 'l2a_to_l2b',
            'L3': 'l2b_to_l3',
            'L4': 'l3_to_l4'
        }

        if data_level not in level_dir_map:
            print(f"Error: Data level '{data_level}' not recognized")
            return

        data_dir = os.path.join(os.path.dirname(__file__), 'pipeline_output', level_dir_map[data_level])

        if not os.path.exists(data_dir):
            print(f"Error: Directory not found: {data_dir}")
            return

        # Find FITS files
        fits_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.fits') and '_cal' not in f and 'recipe' not in f
        ]

        if not fits_files:
            print(f"No FITS files found in {data_dir}")
            return

        fits_file = fits_files[0]

    # Verify headers
    try:
        results = verify_headers(fits_file, data_level)
        print_summary(results, data_level)

        # Save results
        output_file = os.path.join(
            os.path.dirname(__file__),
            'pipeline_output',
            f'{data_level}_HEADER_VERIFICATION.txt'
        )
        save_results(results, data_level, fits_file, output_file)

    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nConfigured data levels: {list(VERIFICATION_CONFIG.keys())}")


if __name__ == "__main__":
    main()
