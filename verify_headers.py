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

# Format: (keyword, expected_dtype)
# dtype can be: bool, int, float, str, or None for optional/variable types

L2A_PRIMARY_HEADERS = [
    # Identification
    ('SIMPLE', bool), ('BITPIX', int), ('NAXIS', int), ('EXTEND', bool),
    # Observation metadata
    'VISITID', 'TELESCOP', 'INSTRUME', 'DETECTOR', 'TARGET',
    # Program information
    'PROGNUM', 'EXECNUM', 'CAMPAIGN', 'SEGMENT', 'OBSNUM', 'VISNUM',
    # Pointing coordinates
    'RA', 'DEC', 'EQUINOX', 'RAPM', 'DECPM', 'ROLL', 'PITCH', 'YAW',
    # Configuration parameters
    'OPGAIN', 'PHTCNT', 'FRAMET', 'SATSPOTS', 'ISHOWFSC', 'HOWFSLNK',
    # Processing metadata
    'CDMSVERS', 'FSWDVERS', 'ORIGIN', 'FILETIME', 'DATAVERS', 'VISTYPE', 'FILENAME',
    # Associated files
    'CPGSFILE', 'AUXFILE',
    # Additional
    'MOCK', 'OBSNAME', 'PSFREF'
]

L2A_IMAGE_HEADERS = [
    # Data structure
    'XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'PCOUNT', 'GCOUNT', 'BUNIT', 'ARRTYPE',
    # Timing
    'SCTSRT', 'SCTEND', 'DATETIME', 'FTIMEUTC',
    # Exposure parameters
    'EXPTIME', 'EXPCYC', 'BLNKTIME', 'BLNKCYC', 'LASTEXP',
    # Gain/calibration
    'EMGAIN_C', 'EMGAIN_A', 'KGAINPAR', 'EMGAINA1', 'EMGAINA2', 'EMGAINA3', 'EMGAINA4', 'EMGAINA5', 'GAINTCAL',
    # Temperature data
    'EXCAMT',
    # Operational modes
    'OPMODE', 'ISPC', 'STATUS', 'HVCBIAS',
    # Hardware status
    'CYCLES', 'OVEREXP', 'NOVEREXP', 'PROXET',
    # Control systems
    'FCMLOOP', 'FCMPOS', 'FSMINNER', 'FSMLOS', 'FSMPRFL', 'FSMRSTR', 'DMZLOOP',
    # Alignment offsets
    'FSMX', 'FSMY', 'FSMSG1', 'FSMSG2', 'FSMSG3',
    'SB_FP_DX', 'SB_FP_DY', 'SB_FS_DX', 'SB_FS_DY', 'EACQ_ROW', 'EACQ_COL',
    # Wavefront sensing - Zernike coefficients
    'Z2AVG', 'Z2RES', 'Z2VAR',
    'Z3AVG', 'Z3RES', 'Z3VAR',
    'Z4AVG', 'Z4RES',
    'Z5AVG', 'Z5RES',
    'Z6AVG', 'Z6RES',
    'Z7AVG', 'Z7RES',
    'Z8AVG', 'Z8RES',
    'Z9AVG', 'Z9RES',
    'Z10AVG', 'Z10RES',
    'Z11AVG', 'Z11RES',
    'Z12AVG', 'Z13AVG', 'Z14AVG',
    '1SVALID', '10SVALID',
    # Actuator positions
    'SPAM_H', 'SPAM_V', 'SPAMNAME', 'SPAMSP_H', 'SPAMSP_V',
    'FPAM_H', 'FPAM_V', 'FPAMNAME', 'FPAMSP_H', 'FPAMSP_V',
    'LSAM_H', 'LSAM_V', 'LSAMNAME', 'LSAMSP_H', 'LSAMSP_V',
    'FSAM_H', 'FSAM_V', 'FSAMNAME', 'FSAMSP_H', 'FSAMSP_V',
    'CFAM_H', 'CFAM_V', 'CFAMNAME', 'CFAMSP_H', 'CFAMSP_V',
    'DPAM_H', 'DPAM_V', 'DPAMNAME', 'DPAMSP_H', 'DPAMSP_V',
    # Processing flags - L2a specific
    'DATALVL', 'MISSING', 'DESMEAR', 'CTI_CORR', 'IS_BAD',
    # Saturation info - L2a specific
    'FWC_PP_E', 'FWC_EM_E', 'SAT_DN',
    # Processing history - L2a specific
    'RECIPE', 'DRPVERSN', 'DRPCTIME'
]

L2A_ERR_HEADERS = [
    'XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
    'PCOUNT', 'GCOUNT', 'EXTNAME', 'TRK_ERRS', 'LAYER_1'
]

L2A_DQ_HEADERS = [
    'XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
    'PCOUNT', 'GCOUNT', 'BSCALE', 'BZERO', 'EXTNAME'
]

L2A_BIAS_HEADERS = [
    'XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1',
    'PCOUNT', 'GCOUNT', 'EXTNAME'
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


def verify_hdu_headers(hdu, expected_headers, hdu_name):
    """
    Verify headers in a single HDU.

    Args:
        hdu: FITS HDU object
        expected_headers (list): List of expected header keywords
        hdu_name (str): Name of the HDU for reporting

    Returns:
        dict: {'present': list, 'missing': list}
    """
    results = {'present': [], 'missing': []}

    for keyword in expected_headers:
        if keyword in hdu.header:
            results['present'].append(keyword)
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

    with fits.open(fits_file) as hdul:
        print(f"Total HDUs in file: {len(hdul)}\n")

        # Verify each configured HDU
        for hdu_idx, hdu_config in config['hdus'].items():
            hdu_name = hdu_config['name']
            expected_headers = hdu_config['headers']

            print(f"Checking {hdu_name} HDU (HDU {hdu_idx})...")

            if len(hdul) <= hdu_idx:
                print(f"  ✗ HDU {hdu_idx} not found in file!")
                results[hdu_name] = {'present': [], 'missing': expected_headers}
                summary[f'{hdu_name.lower()}_total'] = len(expected_headers)
                summary[f'{hdu_name.lower()}_present'] = 0
                summary[f'{hdu_name.lower()}_missing'] = len(expected_headers)
                continue

            if not expected_headers:
                print(f"  ⚠ No expected headers configured for {hdu_name} HDU")
                results[hdu_name] = {'present': [], 'missing': []}
                continue

            hdu_results = verify_hdu_headers(hdul[hdu_idx], expected_headers, hdu_name)
            results[hdu_name] = hdu_results

            # Report results
            total = len(expected_headers)
            present = len(hdu_results['present'])
            missing = len(hdu_results['missing'])

            print(f"  Present: {present}/{total}")
            if hdu_results['missing']:
                print(f"  Missing: {missing} headers")
                for m in hdu_results['missing']:
                    print(f"    - {m}")
            else:
                print(f"  ✓ All expected {hdu_name} headers present!")

            summary[f'{hdu_name.lower()}_total'] = total
            summary[f'{hdu_name.lower()}_present'] = present
            summary[f'{hdu_name.lower()}_missing'] = missing

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

    results['summary'] = summary
    return results


def print_summary(results, data_level):
    """Print overall verification summary."""
    print(f"\n{'='*80}")
    print(f"{data_level} VERIFICATION SUMMARY")
    print(f"{'='*80}")

    summary = results['summary']
    total_missing = 0

    # Print each HDU's results
    for key in summary:
        if key.endswith('_total'):
            hdu_name = key.replace('_total', '').upper()
            total = summary[f'{hdu_name.lower()}_total']
            present = summary[f'{hdu_name.lower()}_present']
            missing = summary[f'{hdu_name.lower()}_missing']

            if total > 0:
                print(f"\n{hdu_name} HDU:")
                print(f"  Expected: {total}")
                print(f"  Present:  {present} ({present/total*100:.1f}%)")
                print(f"  Missing:  {missing}")
                total_missing += missing

    if summary.get('history_count', 0) > 0:
        print(f"\nProcessing History:")
        print(f"  HISTORY entries: {summary['history_count']}")

    # Overall verdict
    print(f"\n{'='*80}")
    if total_missing == 0:
        print(f"✓ ALL EXPECTED HEADERS PRESENT IN ALL HDUs FOR {data_level}")
    else:
        print(f"⚠ {total_missing} headers missing across all HDUs")
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

            f.write(f"Missing {hdu_name} Headers:\n")
            if hdu_results['missing']:
                for h in hdu_results['missing']:
                    f.write(f"  - {h}\n")
            else:
                f.write("  None\n")
            f.write("\n")

        # Write summary
        summary = results['summary']
        f.write("Summary:\n")
        for key in summary:
            if key.endswith('_total'):
                hdu_name = key.replace('_total', '').capitalize()
                total = summary[key]
                present = summary[f'{key.replace("_total", "_present")}']
                if total > 0:
                    f.write(f"  {hdu_name:8s}: {present}/{total} present\n")
        if summary.get('history_count', 0) > 0:
            f.write(f"  History : {summary['history_count']} entries\n")

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
