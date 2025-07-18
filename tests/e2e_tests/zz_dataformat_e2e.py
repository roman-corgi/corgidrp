import os
import argparse
import difflib
import glob
import pytest
import numpy as np
import astropy.io.fits as fits

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

custom_header_keys = ['DRPCTIME', 'DRPVERSN', 'RECIPE', 'FILE0', 'DATETIME', 'FTIMEUTC', 'DETPIX0X', 'DETPIX0Y', 'PYKLIPV']
def compare_docs(ref_doc, new_doc):
    """
    Compare reference doc to new doc

    Args:
        ref_doc (str): full content of reference doc
        new_doc (str): full content of new document
    """
    # ignore beginning and ending whitespace
    # split lines
    ref_lines = ref_doc.strip().splitlines()
    new_lines = new_doc.strip().splitlines()

    # certain rows need custom checking
    mod_ref_lines = []
    mod_new_lines = []

    # grab all the rows that need special checking out of the way
    for i, line in enumerate(ref_lines):
        if '|' in line: # check if sphinx table
            line_args = line.split("|")
            name = line_args[1].strip()
            dtype = line_args[2].strip()
            mask_value = (name.upper() in custom_header_keys) or (dtype.lower() == "float")
            if mask_value:
                # remove the value of the entry from the line
                line_args[3] = '[value ignored]'
                # remove the comment value from the line as well
                line_args[4] = '[comment ignored]'
                mod_line = "|".join(line_args)
                # move into the custom checking section
                mod_ref_lines.append(mod_line)

    for i, line in enumerate(new_lines):
        if '|' in line: # check if sphinx table
            line_args = line.split("|")
            name = line_args[1].strip()
            dtype = line_args[2].strip()
            mask_value = (name.upper() in custom_header_keys) or (dtype.lower() == "float")
            if mask_value:
                # remove the value of the entry from the line
                line_args[3] = '[value ignored]'
                # remove the comment value from the line as well
                line_args[4] = '[comment ignored]'
                mod_line = "|".join(line_args)
                # move into the custom checking section
                mod_new_lines.append(mod_line)

    # diff the regular part of the doc
    diff = difflib.unified_diff(mod_ref_lines, mod_new_lines)
    diff = list(diff)
    diff_output = "\n".join(diff)
    print(diff_output)
    assert len(diff) == 0

 
###########################
### Begin Tests ###########
###########################


@pytest.mark.e2e
def test_l2a_dataformat_e2e(e2edata_path, e2eoutput_path):

    l2a_data_dir = os.path.join(thisfile_dir, "l1_to_l2b_output", "l2a")
    l2a_data_file = os.path.join(l2a_data_dir, "90499.fits")

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
def test_l2banalog_dataformat_e2e(e2edata_path, e2eoutput_path):

    l2b_data_dir = os.path.join(thisfile_dir, "l1_to_l2b_output", "l2b")
    l2b_data_file = os.path.join(l2b_data_dir, "90499.fits")

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
def test_l2bpc_dataformat_e2e(e2edata_path, e2eoutput_path):

    l2b_data_dir = os.path.join(thisfile_dir, "pc_sim_test_data")
    l2b_data_file = glob.glob(os.path.join(l2b_data_dir, "pc_frame_ill_*.fits"))[0]

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

    l3_data_dir = os.path.join(thisfile_dir, "l2b_to_l3_output")
    l3_data_file = glob.glob(os.path.join(l3_data_dir, "CGI_*_L3_.fits"))[0]

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
def test_l4coron_dataformat_e2e(e2edata_path, e2eoutput_path):

    l4_data_dir = os.path.join(thisfile_dir, "l3_to_l4_output")
    l4_data_file = glob.glob(os.path.join(l4_data_dir, "CGI_*_L4_.fits"))[0]

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
def test_l4noncoron_dataformat_e2e(e2edata_path, e2eoutput_path):

    kgain_data_file = glob.glob(os.path.join(thisfile_dir, "l3_to_l4_noncoron_output", "CGI_*_L4_.fits"))[0]

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

    astrom_data_file = glob.glob(os.path.join(thisfile_dir, "astrom_cal_output", "CGI_*_AST_CAL.fits"))[0]

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

    bpmap_data_file = glob.glob(os.path.join(thisfile_dir, "flat_neptune_output", "CGI_*BPM_CAL*.fits"))[0]

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

    flat_data_file = glob.glob(os.path.join(thisfile_dir, "flat_neptune_output", "CGI_*FLT_CAL*.fits"))[0]

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

    ct_data_file = glob.glob(os.path.join(thisfile_dir, "l2b_to_corethroughput_output", "*CTP_CAL.fits"))[0]

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

    ctmap_data_file = os.path.join(thisfile_dir, "l2a_to_ct_map", "corethroughput_map.fits")

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

    fluxcal_data_file = glob.glob(os.path.join(thisfile_dir, "l2b_to_fluxcal_factor_output", "*ABF_CAL.fits"))[0]

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

    kgain_data_file = glob.glob(os.path.join(thisfile_dir, "nonlin_and_kgain_output", "CGI_*KRN_CAL*.fits"))[0]

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

    nonlin_data_file = glob.glob(os.path.join(thisfile_dir, "nonlin_and_kgain_output", "CGI_*NLN_CAL*.fits"))[0]

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

    nonlin_data_file = glob.glob(os.path.join(thisfile_dir, "nd_filter_e2e_output", "CGI_*NDF_CAL.fits"))[0]

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

    noisemaps_data_file = glob.glob(os.path.join(thisfile_dir, "noisemap_output", "CGI_*DNM_CAL.fits"))[0]

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

    dark_data_file = glob.glob(os.path.join(thisfile_dir, "build_trad_dark_output", "CGI_*_DRK_CAL.fits"))[0]

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

    tpump_data_file = glob.glob(os.path.join(thisfile_dir, "trap_pump_cal_output", "*_TPU_CAL.fits"))[0]

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


if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    e2edata_dir =  '/home/jwang/Desktop/CGI_TVAC_Data/'
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_l2bpc_dataformat_e2e(e2edata_dir, outputdir)
    test_dark_dataformat_e2e(e2edata_dir, outputdir)
    test_l2a_dataformat_e2e(e2edata_dir, outputdir)
    test_l2banalog_dataformat_e2e(e2edata_dir, outputdir)
    test_l3_dataformat_e2e(e2edata_dir, outputdir)
    test_l4coron_dataformat_e2e(e2edata_dir, outputdir)
    test_l4noncoron_dataformat_e2e(e2edata_dir, outputdir)
    test_astrom_dataformat_e2e(e2edata_dir, outputdir)
    test_bpmap_dataformat_e2e(e2edata_dir, outputdir)
    test_flat_dataformat_e2e(e2edata_dir, outputdir)
    test_ct_dataformat_e2e(e2edata_dir, outputdir)
    test_ctmap_dataformat_e2e(e2edata_dir, outputdir)
    test_fluxcal_dataformat_e2e(e2edata_dir, outputdir)
    test_kgain_dataformat_e2e(e2edata_dir, outputdir)
    test_nonlin_dataformat_e2e(e2edata_dir, outputdir)
    test_ndfilter_dataformat_e2e(e2edata_dir, outputdir)
    test_noisemaps_dataformat_e2e(e2edata_dir, outputdir)
    test_tpump_dataformat_e2e(e2edata_dir, outputdir)