import os
import argparse
import difflib
import glob
import pytest
import numpy as np
import astropy.io.fits as fits
from datetime import datetime, timedelta
from corgidrp.check import generate_fits_excel_documentation
import pandas as pd

thisfile_dir = os.path.dirname(__file__) # this file's folder

def generate_template(hdulist, dtype_name=None):
    """
    Generates an rst documentation page of the data entries

    Args:
        hdulist (astropy.io.fits.HDUList): hdulist from fits file to be documented
        dtype_name (str): if not None, custom name to use for page title and label

    Returns:
        str: the rst page contents
    """

    datalvl = hdulist[1].header['DATALVL']

    if dtype_name is not None:
        datatype = dtype_name
    elif datalvl.lower() == "cal":
        datatype = hdulist[1].header['DATATYPE']
    else:
        datatype = datalvl

        

    template_filepath = os.path.join(thisfile_dir, "data_format_template.rst")
    with open(template_filepath, "r") as f:
        template = f.read()

    # make the data format table
    hdu_table = generate_hdustructure(hdulist)

    # make the header tables
    hdr_tables = ""

    for i, hdu in enumerate(hdulist):
        # name
        if i == 0:
            hdu_name = "Primary"
        elif i == 1:
            hdu_name = "Image"
        else:
            hdu_name = hdu.header['EXTNAME']
        title = "{0} Header (HDU {1})".format(hdu_name, i)
        title_delim = "".join(["^" for _ in range(len(title))])
        this_hdr_table = generate_header_table(hdu)

        hdr_tables += title
        hdr_tables += "\n"
        hdr_tables += title_delim
        hdr_tables += "\n\n"
        hdr_tables += this_hdr_table
        hdr_tables += "\n\n"


    doc = template.format(datatype.lower(), datatype, hdu_table, hdr_tables)

    return doc

    

def generate_hdustructure(hdulist):
    """
    Generates the hdulist structure rst table

    Args:
        hdulist (astropy.io.fits.HDUList): hdulist from fits file to be documented

    Returns:
        str: rst table with hdulist structure
    """

    hdu_table = '''
+-------+------------------+----------+----------------------+
| Index | Name             | Datatype | Array Size           |
+=======+==================+==========+======================+
'''

    row_template = "| {0:<5} | {1:<16} | {2:<8} | {3:<20} |"
    row_delimiter = "+-------+------------------+----------+----------------------+"

    for i, hdu in enumerate(hdulist):
        # name
        if i == 0:
            hdu_name = "Primary"
        elif i == 1:
            hdu_name = "Image"
        else:
            hdu_name = hdu.header['EXTNAME']

        # datatype
        if hdu.data is None:
            datatype = "None"
            arr_size = 0
        elif isinstance(hdu.data, np.ndarray):
            datatype = hdu.data.dtype.name
            arr_size = str(hdu.data.shape)
        else:
            datatype = type(hdu.data).split("'")[1]
            arr_size = '1'

        hdu_table += row_template.format(i, hdu_name, datatype, arr_size)
        hdu_table += "\n"
        hdu_table += row_delimiter
        hdu_table += "\n"

    return hdu_table
        
def generate_header_table(hdu):
    """
    Generates the hdulist structure rst table

    Args:
        hdu (astropy.io.fits.hdu): Single hdu from fits file to be documented

    Returns:
        str: rst table with hdulist structure
    """

    header_table = '''
+------------+------------+--------------------------------+----------------------------------------------------+
| Keyword    | Datatype   | Example Value                  | Description                                        |
+============+============+================================+====================================================+
'''
    row_template = "| {0:<10} | {1:<10} | {2:<30} | {3:<50} |"
    row_delimiter = "+------------+------------+--------------------------------+----------------------------------------------------+"

    history_recorded = False
    comment_recorded = False
    filen_recorded = False

    hdr = hdu.header
    for key in hdr:
        if key == "HISTORY":
            if history_recorded:
                # only need to record one history entry
                continue
            else:
                history_recorded = True
                datatype = "str"
        elif key == "COMMENT":
            if comment_recorded:
                # only need to record one history entry
                continue
            else:
                comment_recorded = True
                datatype = "str"
        else:
            datatype = str(type(hdr[key])).split("'")[1]

        example_value = str(hdr[key]).replace("\n", " ")
        if len(example_value) > 30:
            # truncate string
            example_value = example_value[:27] + "..."

        description = hdr.comments[key]
        if len(description) > 50:
            # truncate string
            description = description[:47] + "..."

        if key[:4] == "FILE" and key[4:].isdigit():
            if filen_recorded:
                continue
            else:
                description = "File name for the n-th science file used"
                filen_recorded = True


        header_table += row_template.format(key, datatype, example_value, description)
        header_table += "\n"
        header_table += row_delimiter
        header_table += "\n"   

    return header_table

def validate_cgi_filename(filepath, expected_suffix):
    """
    Validate that a FITS file follows the CGI filename convention:
    cgi_VISITID_YYYYMMDDtHHMMSSS_suffix.fits
    
    Args:
        filepath (str): Path to the FITS file
        expected_suffix (str): Expected suffix (e.g., 'l2a', 'l2b', 'ast_cal', 'bpm_cal', etc.)
    
    Returns:
        bool: True if filename is valid
    
    Raises:
        AssertionError: If filename doesn't match expected format
    """
    import re
    from astropy.io import fits
    
    filename = os.path.basename(filepath)
    
    # Check that all alphabetical characters are lowercase
    assert filename == filename.lower(), \
        f"Filename '{filename}' contains uppercase characters. All letters must be lowercase."
    
    # Pattern: cgi_VISITID(19 digits)_YYYYMMDDtHHMMSSS_SUFFIX.fits
    # VISITID should be 19 digits
    # Timestamp should be YYYYMMDDtHHMMSSS
    pattern = r'^cgi_(\d{19})_(\d{8}t\d{7})_(.+)\.fits$'
    
    match = re.match(pattern, filename)
    assert match, f"Filename '{filename}' doesn't match CGI convention 'cgi_VISITID_YYYYMMDDtHHMMSSS_SUFFIX.fits' (8 digits + 't' + 7 digits)"
    
    visitid_in_filename, timestamp, suffix = match.groups()
    
    # Check suffix matches expected
    assert suffix.lower() == expected_suffix.lower(), \
        f"Filename suffix '{suffix}' doesn't match expected '{expected_suffix}'"
    
    # Validate timestamp format, should be YYYYMMDDtHHMMSSs
    ts_pattern = r'^(\d{4})(\d{2})(\d{2})t(\d{2})(\d{2})(\d{2})(\d{1})$'
    ts_match = re.match(ts_pattern, timestamp)
    assert ts_match, f"Timestamp '{timestamp}' doesn't match YYYYMMDDtHHMMSSS format (8 digits + 't' + 7 digits)"
    
    year, month, day, hour, minute, second, decisec = ts_match.groups()
    
    assert 1 <= int(month) <= 12, f"Invalid month '{month}' in timestamp"
    assert 1 <= int(day) <= 31, f"Invalid day '{day}' in timestamp"
    assert 0 <= int(hour) <= 23, f"Invalid hour '{hour}' in timestamp"
    assert 0 <= int(minute) <= 59, f"Invalid minute '{minute}' in timestamp"
    
    # Open FITS and check VISITID matches (do not do this for now because of mocks)
    '''
    with fits.open(filepath) as hdul:
        visitid_in_header = str(hdul[0].header.get('VISITID', '')).strip()
        # VISITID in header should match filename (allow for leading zeros or string conversion)
        if visitid_in_header:
            assert visitid_in_header.zfill(19) == visitid_in_filename, \
                f"VISITID in header '{visitid_in_header}' doesn't match filename VISITID '{visitid_in_filename}'"
    '''

    print(f"Filename validation passed: {filename}")
    return True

custom_header_keys = ['DRPCTIME', 'DRPVERSN', 'RECIPE', 'FILE0', 'DATETIME', 'FTIMEUTC', 'DETPIX0X', 'DETPIX0Y', 'PYKLIPV']
def compare_docs(ref_doc, new_doc):
    """
    Compare reference doc to new doc. Checks that all headers are present regardless of order

    Args:
        ref_doc (str): full content of reference doc
        new_doc (str): full content of new document
    """
    # ignore beginning and ending whitespace
    # split lines
    ref_lines = ref_doc.strip().splitlines()
    new_lines = new_doc.strip().splitlines()

    # Get header entries (keyword, datatype, hdu) from both documents
    # Ignore order, values, and comments. Only check that keywords and datatypes match
    def extract_headers(lines):
        headers = set()
        current_hdu = None
        in_hdu_structure_table = False
        for line in lines:
            # Check if we're in the HDU structure table (first table in doc)
            if 'Index' in line and 'Name' in line and 'Datatype' in line:
                in_hdu_structure_table = True
            elif in_hdu_structure_table and (line.strip().endswith('^^^') or 'Header (HDU' in line):
                in_hdu_structure_table = False
            
            # Check which HDU section we're in 
            if 'Header (HDU' in line and not line.strip().startswith('|'):
                # Get HDU name
                try:
                    hdu_name = line.split('Header')[0].strip()
                    hdu_num = line.split('HDU ')[1].split(')')[0]
                    current_hdu = f"{hdu_name} (HDU {hdu_num})"
                except:
                    pass 
            elif '|' in line and line.count('|') >= 4 and not in_hdu_structure_table:
                line_args = line.split("|")
                if len(line_args) >= 5:
                    name = line_args[1].strip()
                    dtype = line_args[2].strip()
                    # Skip table header/delimiter rows
                    if name and dtype and name != 'Keyword' and name != '=' * len(name) and name != '-' * len(name) and not name.isdigit():
                        # Store keyword name, datatype, and HDU
                        headers.add((name, dtype, current_hdu or 'Unknown'))
        return headers
    
    ref_headers = extract_headers(ref_lines)
    new_headers = extract_headers(new_lines)
    
    # Find headers that are in reference but not in fits file
    missing_headers = ref_headers - new_headers
    # Find headers that are in fits file but not in reference
    extra_headers = new_headers - ref_headers
    
    if missing_headers or extra_headers:
        print("\n=== Header comparison failed ===")
        if missing_headers:
            print("\nHeaders in reference documentation but missing from FITS output:")
            for header in sorted(missing_headers):
                print(f"  - {header[0]} ({header[1]}) in {header[2]}")
        if extra_headers:
            print("\nHeaders in FITS output but missing from reference documentation:")
            for header in sorted(extra_headers):
                print(f"  + {header[0]} ({header[1]}) in {header[2]}")
        assert False, "Header mismatch"
    else:
        print(f"Header comparison passed: All {len(ref_headers)} headers match between reference and actual output")
    

 
###########################
### Begin Tests ###########
###########################


@pytest.mark.e2e
def test_l2a_dataformat_e2e(e2edata_path, e2eoutput_path):

    l2a_data_dir = os.path.join(thisfile_dir, "l1_to_l2a_e2e")
    #l2a_data_file = os.path.join(l2a_data_dir, "90499.fits")
    fits_files = glob.glob(os.path.join(l2a_data_dir, "*.fits"))
    l2a_data_file = max(fits_files, key=os.path.getmtime)
    
    validate_cgi_filename(l2a_data_file, 'l2a')
    
    generate_fits_excel_documentation(l2a_data_file, os.path.join(l2a_data_dir, "l2a_documentation.xlsx"))

    doc_output_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_output_dir):
        os.mkdir(doc_output_dir)


    with fits.open(l2a_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_output_dir, "l2a.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "l2a.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_l2b_analog_dataformat_e2e(e2edata_path, e2eoutput_path):

    l2b_data_dir = os.path.join(thisfile_dir, "l1_to_l2b_e2e")
    fits_files = glob.glob(os.path.join(l2b_data_dir, "*.fits"))
    l2b_data_file = max(fits_files, key=os.path.getmtime)
    
    validate_cgi_filename(l2b_data_file, 'l2b')
    
    generate_fits_excel_documentation(l2b_data_file, os.path.join(l2b_data_dir, "l2b_analog_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(l2b_data_file) as hdulist:
        doc_contents = generate_template(hdulist, dtype_name="L2b-Analog")

    doc_filepath = os.path.join(doc_dir, "l2b_analog.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "l2b_analog.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_l2b_pc_dataformat_e2e(e2edata_path, e2eoutput_path):

    l2b_data_dir = os.path.join(thisfile_dir, "photon_count_e2e", "l2a_to_l2b")
    l2b_data_file = glob.glob(os.path.join(l2b_data_dir, "*_l2b.fits"))[0]
    
    validate_cgi_filename(l2b_data_file, 'l2b')
    
    generate_fits_excel_documentation(l2b_data_file, os.path.join(l2b_data_dir, "l2b_pc_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(l2b_data_file) as hdulist:
        doc_contents = generate_template(hdulist, dtype_name="L2b-PhotonCounting")

    doc_filepath = os.path.join(doc_dir, "l2b_pc.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "l2b_pc.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)


@pytest.mark.e2e
def test_l3_dataformat_e2e(e2edata_path, e2eoutput_path):

    l3_data_dir = os.path.join(thisfile_dir, "l2b_to_l4_e2e", "l2b_to_l3")
    l3_data_file = glob.glob(os.path.join(l3_data_dir, "*_l3_.fits"))[0]
    
    validate_cgi_filename(l3_data_file, 'l3_')
    
    generate_fits_excel_documentation(l3_data_file, os.path.join(l3_data_dir, "l3_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(l3_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "l3.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "l3.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)


@pytest.mark.e2e
def test_l4_coron_dataformat_e2e(e2edata_path, e2eoutput_path):

    l4_data_dir = os.path.join(thisfile_dir, "l2b_to_l4_e2e")
    l4_data_file = glob.glob(os.path.join(l4_data_dir, "*_l4_.fits"))[0]
    
    validate_cgi_filename(l4_data_file, 'l4_')
    
    generate_fits_excel_documentation(l4_data_file, os.path.join(l4_data_dir, "l4_coron_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(l4_data_file) as hdulist:
        doc_contents = generate_template(hdulist, dtype_name="L4-Coronagraphic")

    doc_filepath = os.path.join(doc_dir, "l4coron.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "l4coron.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_l4_noncoron_dataformat_e2e(e2edata_path, e2eoutput_path):

    kgain_data_file = glob.glob(os.path.join(thisfile_dir, "l2b_to_l4_noncoron_e2e", "*_l4_.fits"))[0]
    
    validate_cgi_filename(kgain_data_file, 'l4_')
    
    generate_fits_excel_documentation(kgain_data_file, os.path.join(thisfile_dir, "l2b_to_l4_noncoron_e2e", "l4_noncoron_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(kgain_data_file) as hdulist:
        doc_contents = generate_template(hdulist, dtype_name="L4-Noncoron")

    doc_filepath = os.path.join(doc_dir, "l4noncoron.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "l4noncoron.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_astrom_dataformat_e2e(e2edata_path, e2eoutput_path):

    astrom_data_file = glob.glob(os.path.join(thisfile_dir, "astrom_cal_e2e", "*_ast_cal.fits"))[0]
    
    validate_cgi_filename(astrom_data_file, 'ast_cal')
    
    generate_fits_excel_documentation(astrom_data_file, os.path.join(thisfile_dir, "astrom_cal_e2e", "ast_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(astrom_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "astrom.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "astrom.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_bpmap_dataformat_e2e(e2edata_path, e2eoutput_path):

    bpmap_data_file = glob.glob(os.path.join(thisfile_dir, "bp_map_cal_e2e", "bp_map_master_dark", "*_bpm_cal.fits"))[0]
    
    validate_cgi_filename(bpmap_data_file, 'bpm_cal')
    
    generate_fits_excel_documentation(bpmap_data_file, os.path.join(thisfile_dir, "bp_map_cal_e2e", "bpm_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(bpmap_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "bpmap.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "bpmap.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)


@pytest.mark.e2e
def test_flat_dataformat_e2e(e2edata_path, e2eoutput_path):

    flat_data_file = glob.glob(os.path.join(thisfile_dir, "flatfield_cal_e2e", "flat_neptune_output", "*_flt_cal.fits"))[0]
    
    validate_cgi_filename(flat_data_file, 'flt_cal')
    
    generate_fits_excel_documentation(flat_data_file, os.path.join(thisfile_dir, "flatfield_cal_e2e", "flat_neptune_output", "flt_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(flat_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "flat.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "flat.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)


@pytest.mark.e2e
def test_ct_dataformat_e2e(e2edata_path, e2eoutput_path):

    ct_data_file = glob.glob(os.path.join(thisfile_dir, "corethroughput_cal_e2e", "band3_spc_data", "*_ctp_cal.fits"))[0]
    
    validate_cgi_filename(ct_data_file, 'ctp_cal')
    
    generate_fits_excel_documentation(ct_data_file, os.path.join(thisfile_dir, "corethroughput_cal_e2e", "band3_spc_data", "ctp_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(ct_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "corethroughput.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "corethroughput.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_ctmap_dataformat_e2e(e2edata_path, e2eoutput_path):

    ctmap_data_file = glob.glob(os.path.join(thisfile_dir, "ctmap_cal_e2e", "*_ctm_cal.fits"))[0]
    
    validate_cgi_filename(ctmap_data_file, 'ctm_cal')
    
    generate_fits_excel_documentation(ctmap_data_file, os.path.join(thisfile_dir, "ctmap_cal_e2e", "ctm_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(ctmap_data_file) as hdulist:
        doc_contents = generate_template(hdulist, dtype_name="CoreThroughputMap")

    doc_filepath = os.path.join(doc_dir, "corethroughput_map.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "corethroughput_map.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_fluxcal_dataformat_e2e(e2edata_path, e2eoutput_path):

    fluxcal_data_file = glob.glob(os.path.join(thisfile_dir, "flux_cal_e2e", "*_abf_cal.fits"))[0]
    
    validate_cgi_filename(fluxcal_data_file, 'abf_cal')
    
    generate_fits_excel_documentation(fluxcal_data_file, os.path.join(thisfile_dir, "flux_cal_e2e", "abf_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(fluxcal_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "fluxcal.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "fluxcal.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_kgain_dataformat_e2e(e2edata_path, e2eoutput_path):

    kgain_data_file = glob.glob(os.path.join(thisfile_dir, "kgain_cal_e2e", "*_krn_cal.fits"))[0]
    
    validate_cgi_filename(kgain_data_file, 'krn_cal')
    
    generate_fits_excel_documentation(kgain_data_file, os.path.join(thisfile_dir, "kgain_cal_e2e", "krn_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(kgain_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "kgain.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "kgain.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)


@pytest.mark.e2e
def test_nonlin_dataformat_e2e(e2edata_path, e2eoutput_path):

    nonlin_data_file = glob.glob(os.path.join(thisfile_dir, "nonlin_cal_e2e", "*_nln_cal.fits"))[0]
    
    validate_cgi_filename(nonlin_data_file, 'nln_cal')
    
    generate_fits_excel_documentation(nonlin_data_file, os.path.join(thisfile_dir, "nonlin_cal_e2e", "nln_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(nonlin_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "nonlin.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "nonlin.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_ndfilter_dataformat_e2e(e2edata_path, e2eoutput_path):

    nonlin_data_file = glob.glob(os.path.join(thisfile_dir, "nd_filter_cal_e2e", "*_ndf_cal.fits"))[0]
    
    validate_cgi_filename(nonlin_data_file, 'ndf_cal')
    
    generate_fits_excel_documentation(nonlin_data_file, os.path.join(thisfile_dir, "nd_filter_cal_e2e", "ndf_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(nonlin_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "ndfilter.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "ndfilter.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_noisemaps_dataformat_e2e(e2edata_path, e2eoutput_path):

    noisemaps_data_file = glob.glob(os.path.join(thisfile_dir, "noisemap_cal_e2e", "l1_to_dnm", "*_dnm_cal.fits"))[0]
    
    validate_cgi_filename(noisemaps_data_file, 'dnm_cal')
    
    generate_fits_excel_documentation(noisemaps_data_file, os.path.join(thisfile_dir, "noisemap_cal_e2e", "l1_to_dnm", "dnm_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(noisemaps_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "noisemaps.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "noisemaps.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)


@pytest.mark.e2e
def test_dark_dataformat_e2e(e2edata_path, e2eoutput_path):

    dark_data_file = glob.glob(os.path.join(thisfile_dir, "trad_dark_e2e", "trad_dark_full_frame", "*_drk_cal.fits"))[0]
    
    validate_cgi_filename(dark_data_file, 'drk_cal')
    
    generate_fits_excel_documentation(dark_data_file, os.path.join(thisfile_dir, "trad_dark_e2e", "trad_dark_full_frame", "drk_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(dark_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "dark.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "dark.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)


@pytest.mark.e2e
def test_tpump_dataformat_e2e(e2edata_path, e2eoutput_path):

    tpump_data_file = glob.glob(os.path.join(thisfile_dir, "trap_pump_cal_e2e", "*_tpu_cal.fits"))[0]
    
    validate_cgi_filename(tpump_data_file, 'tpu_cal')
    
    generate_fits_excel_documentation(tpump_data_file, os.path.join(thisfile_dir, "trap_pump_cal_e2e", "tpu_cal_documentation.xlsx"))

    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)


    with fits.open(tpump_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "tpump.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "tpump.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)


@pytest.mark.e2e
def test_mueller_matrix_dataformat_e2e(e2edata_path, e2eoutput_path):
    polcal_data_file = glob.glob(os.path.join(thisfile_dir, "polcal_e2e", "*_mmx_cal.fits"))[0]
    
    validate_cgi_filename(polcal_data_file, 'mmx_cal')
    
    generate_fits_excel_documentation(polcal_data_file, os.path.join(thisfile_dir, "polcal_e2e", "mmx_cal_documentation.xlsx"))
    
    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)

    with fits.open(polcal_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "mmx_cal.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "mmx_cal.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_nd_mueller_dataformat_e2e(e2edata_path, e2eoutput_path):
    polcal_data_file = glob.glob(os.path.join(thisfile_dir, "polcal_e2e", "*_ndm_cal.fits"))[0]
    
    validate_cgi_filename(polcal_data_file, 'ndm_cal')
    
    generate_fits_excel_documentation(polcal_data_file, os.path.join(thisfile_dir, "polcal_e2e", "ndm_cal_documentation.xlsx"))
    
    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)

    with fits.open(polcal_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "ndm_cal.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "ndm_cal.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)


@pytest.mark.e2e
def test_spec_linespread_dataformat_e2e(e2edata_path, e2eoutput_path):
    spec_linespread_data_file = glob.glob(os.path.join(thisfile_dir, "spec_linespread_cal_e2e", "*_lsf_cal.fits"))[0]
    
    validate_cgi_filename(spec_linespread_data_file, 'lsf_cal')
    
    generate_fits_excel_documentation(spec_linespread_data_file, os.path.join(thisfile_dir, "spec_linespread_cal_e2e", "lsf_cal_documentation.xlsx"))
    
    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)

    with fits.open(spec_linespread_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "lsf_cal.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "lsf_cal.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_spec_prism_disp_dataformat_e2e(e2edata_path, e2eoutput_path):
    spec_prism_disp_data_file = glob.glob(os.path.join(thisfile_dir, "spec_prism_disp_cal_e2e", "*_dpm_cal.fits"))[0]
    
    validate_cgi_filename(spec_prism_disp_data_file, 'dpm_cal')
    
    generate_fits_excel_documentation(spec_prism_disp_data_file, os.path.join(thisfile_dir, "spec_prism_disp_cal_e2e", "dpm_cal_documentation.xlsx"))
    
    doc_dir = os.path.join(thisfile_dir, "data_format_docs")
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)

    with fits.open(spec_prism_disp_data_file) as hdulist:
        doc_contents = generate_template(hdulist)

    doc_filepath = os.path.join(doc_dir, "dpm_cal.rst")
    with open(doc_filepath, "w") as f:
        f.write(doc_contents)

    ref_doc_dir = os.path.join(thisfile_dir, "..", "..", "docs", "source", "data_formats")
    ref_doc = os.path.join(ref_doc_dir, "dpm_cal.rst")
    if os.path.exists(ref_doc):
        with open(ref_doc, "r") as f2:
            ref_doc_contents = f2.read()
        # diff the two outputs
        compare_docs(ref_doc_contents, doc_contents)

@pytest.mark.e2e
def test_header_crossreference_e2e(e2edata_path, e2eoutput_path):
    """
    Create a cross-reference Excel file showing which headers appear in which data products.
    Each sheet represents an HDU extension, with rows for each unique header keyword and columns 
    for each data product, marked with 'X' where the header is present.

    Args:
        e2edata_path (str): Path to the test data
        e2eoutput_path (str): Path to the output directory
    """
    
    # Load L1 keyword definitions from CSV
    l1_keywords_by_hdu = {'Primary': set(), 'Image': set()}
    try:
        import csv
        csv_path = os.path.join(thisfile_dir, '..', '..', 'corgidrp', 'data', 'header_formats', 'l1.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    keyword = row.get('Keyword', '').strip()
                    section = row.get('Section', '').strip()
                    
                    if keyword:
                        if 'Primary Header' in section:
                            l1_keywords_by_hdu['Primary'].add(keyword)
                        elif 'Image Header' in section:
                            l1_keywords_by_hdu['Image'].add(keyword)
    except Exception as e:
        print(f"Warning: Could not load L1 keywords from CSV: {e}")
    
    # Define all data products and their file locations
    data_products = {
        'L2a': glob.glob(os.path.join(thisfile_dir, "l1_to_l2a_e2e", "*.fits")),
        'L2b_Analog': glob.glob(os.path.join(thisfile_dir, "l1_to_l2b_e2e", "*.fits")),
        'L2b_PC': glob.glob(os.path.join(thisfile_dir, "photon_count_e2e", "l2a_to_l2b", "*_l2b.fits")),
        'L3': glob.glob(os.path.join(thisfile_dir, "l2b_to_l4_e2e", "l2b_to_l3", "*_l3_.fits")),
        'L4_Coron': glob.glob(os.path.join(thisfile_dir, "l2b_to_l4_e2e", "*_l4_.fits")),
        'L4_Noncoron': glob.glob(os.path.join(thisfile_dir, "l2b_to_l4_noncoron_e2e", "*_l4_.fits")),
        'Astrom': glob.glob(os.path.join(thisfile_dir, "astrom_cal_e2e", "*_ast_cal.fits")),
        'BPMap': glob.glob(os.path.join(thisfile_dir, "bp_map_cal_e2e", "bp_map_master_dark", "*_bpm_cal.fits")),
        'Flat': glob.glob(os.path.join(thisfile_dir, "flatfield_cal_e2e", "flat_neptune_output", "*_flt_cal.fits")),
        'CoreThroughput': glob.glob(os.path.join(thisfile_dir, "corethroughput_cal_e2e", "band3_spc_data", "*_ctp_cal.fits")),
        'CoreThroughputMap': glob.glob(os.path.join(thisfile_dir, "ctmap_cal_e2e", "*_ctm_cal.fits")),
        'FluxCal': glob.glob(os.path.join(thisfile_dir, "flux_cal_e2e", "*_abf_cal.fits")),
        'FluxCalPol': glob.glob(os.path.join(thisfile_dir, "fluxcal_pol_e2e", "WP1","*_abf_cal.fits")),
        'KGain': glob.glob(os.path.join(thisfile_dir, "kgain_cal_e2e", "*_krn_cal.fits")),
        'MuellerMatrix': glob.glob(os.path.join(thisfile_dir, "polcal_e2e", "*_mmx_cal.fits")),
        'NonLin': glob.glob(os.path.join(thisfile_dir, "nonlin_cal_e2e", "*_nln_cal.fits")),
        'NDFilter': glob.glob(os.path.join(thisfile_dir, "nd_filter_cal_e2e", "*_ndf_cal.fits")),
        'NDMueller': glob.glob(os.path.join(thisfile_dir, "polcal_e2e", "*_ndm_cal.fits")),
        'NoiseMaps': glob.glob(os.path.join(thisfile_dir, "noisemap_cal_e2e", "l1_to_dnm", "*_dnm_cal.fits")),
        'Dark': glob.glob(os.path.join(thisfile_dir, "trad_dark_e2e", "trad_dark_full_frame", "*_drk_cal.fits")),
        'TrapPump': glob.glob(os.path.join(thisfile_dir, "trap_pump_cal_e2e", "*_tpu_cal.fits")),
    }
    
    # Get the most recent file for each data product
    data_files = {}
    for product_name, file_list in data_products.items():
        if file_list:
            data_files[product_name] = max(file_list, key=os.path.getmtime)
        else:
            print(f"Warning: No files found for {product_name}")
    
    # Collect all headers from all data products, organized by HDU
    # Structure: {hdu_name: {keyword: {product_name: True/False}}}
    all_headers = {}
    hdu_names_by_product = {}  # Track HDU names for each product
    
    for product_name, filepath in data_files.items():
        with fits.open(filepath) as hdulist:
            hdu_names_by_product[product_name] = []
            
            for i, hdu in enumerate(hdulist):
                # Get HDU name
                if i == 0:
                    hdu_name = "Primary"
                elif i == 1:
                    hdu_name = "Image"
                else:
                    hdu_name = hdu.header.get('EXTNAME', f'HDU{i}')
                
                hdu_names_by_product[product_name].append(hdu_name)
                
                # Initialize this HDU if not seen before
                if hdu_name not in all_headers:
                    all_headers[hdu_name] = {}
                
                # Collect all keywords from this HDU
                for keyword in hdu.header.keys():
                    # Skip FILE* keywords (FILE0, FILE1, etc.)
                    if keyword.startswith('FILE') and len(keyword) > 4 and keyword[4:].isdigit():
                        continue
                    
                    if keyword not in all_headers[hdu_name]:
                        all_headers[hdu_name][keyword] = {}
                    all_headers[hdu_name][keyword][product_name] = True
    
    # Add L1 keywords to the all_headers structure
    for hdu_name, keywords in l1_keywords_by_hdu.items():
        if hdu_name not in all_headers:
            all_headers[hdu_name] = {}
        for keyword in keywords:
            if keyword not in all_headers[hdu_name]:
                all_headers[hdu_name][keyword] = {}
    
    # Create Excel file with one sheet per HDU
    output_file = os.path.join(thisfile_dir, "header_crossreference.xlsx")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Process each HDU
        for hdu_name in sorted(all_headers.keys()):
            # Get all keywords for this HDU
            keywords = sorted(all_headers[hdu_name].keys())
            
            ordered_products = ['L1']  # Start with L1
            
            # Add L2a
            if 'L2a' in data_files:
                ordered_products.append('L2a')
            
            # Add L2b variants (Analog before PC)
            if 'L2b_Analog' in data_files:
                ordered_products.append('L2b_Analog')
            if 'L2b_PC' in data_files:
                ordered_products.append('L2b_PC')
            
            # Add L3
            if 'L3' in data_files:
                ordered_products.append('L3')
            
            # Add L4
            if 'L4_Coron' in data_files:
                ordered_products.append('L4_Coron')
            if 'L4_Noncoron' in data_files:
                ordered_products.append('L4_Noncoron')
            
            # Add remaining products alphabetically
            remaining_products = sorted([p for p in data_files.keys() if p not in ordered_products])
            ordered_products.extend(remaining_products)
            
            # Create a DataFrame with keywords as rows and products as columns
            data = []
            for keyword in keywords:
                row = {'Keyword': keyword}
                # Add columns in the specified order
                for product in ordered_products:
                    if product == 'L1':
                        # L1 column from CSV
                        row['L1'] = 'X' if keyword in l1_keywords_by_hdu.get(hdu_name, set()) else ''
                    else:
                        # Data product columns
                        row[product] = 'X' if all_headers[hdu_name][keyword].get(product, False) else ''
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Write to Excel sheet (sheet names can't have special chars)
            sheet_name = hdu_name.replace('/', '_').replace('\\', '_')[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                ) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)
    
    print(f"Header cross-reference created: {output_file}")

    # Verify file was created
    assert os.path.exists(output_file), "Cross-reference Excel file was not created"


if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    #e2edata_dir =  '/home/jwang/Desktop/CGI_TVAC_Data/'
    e2edata_dir = '/Users/kevinludwick/Documents/ssc_tvac_test/E2E_Test_Data2' #'/Users/kevinludwick/Documents/ssc_tvac_test/'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_astrom_dataformat_e2e(e2edata_dir, outputdir)
    test_bpmap_dataformat_e2e(e2edata_dir, outputdir)
    test_ct_dataformat_e2e(e2edata_dir, outputdir)
    test_ctmap_dataformat_e2e(e2edata_dir, outputdir)
    test_flat_dataformat_e2e(e2edata_dir, outputdir)
    test_fluxcal_dataformat_e2e(e2edata_dir, outputdir)
    test_kgain_dataformat_e2e(e2edata_dir, outputdir)
    test_l2a_dataformat_e2e(e2edata_dir, outputdir)
    test_l2b_analog_dataformat_e2e(e2edata_dir, outputdir)
    test_l2b_pc_dataformat_e2e(e2edata_dir, outputdir)
    test_l3_dataformat_e2e(e2edata_dir, outputdir)
    test_l4_coron_dataformat_e2e(e2edata_dir, outputdir)
    test_l4_noncoron_dataformat_e2e(e2edata_dir, outputdir)
    test_mueller_matrix_dataformat_e2e(e2edata_dir, outputdir)
    test_ndfilter_dataformat_e2e(e2edata_dir, outputdir)
    test_noisemaps_dataformat_e2e(e2edata_dir, outputdir)
    test_nonlin_dataformat_e2e(e2edata_dir, outputdir)
    test_nd_mueller_dataformat_e2e(e2edata_dir, outputdir)
    test_spec_linespread_dataformat_e2e(e2edata_dir, outputdir)
    test_spec_prism_disp_dataformat_e2e(e2edata_dir, outputdir)
    test_dark_dataformat_e2e(e2edata_dir, outputdir)
    test_tpump_dataformat_e2e(e2edata_dir, outputdir)
    test_header_crossreference_e2e(e2edata_dir, outputdir)