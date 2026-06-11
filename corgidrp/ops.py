import corgidrp
import corgidrp.caldb as caldb
import corgidrp.walker as walker


def step_1_initialize(user_templates_dir=None):
    """
    Initialize corgidrp and it's caldb again

    Args:
        user_templates_dir (str): the path to the user recipe template directory (optional)

    Returns: 
        caldb.CalDB: an instance of an initialized caldb object
    """
    corgidrp.create_config_dir()
    corgidrp.update_pipeline_settings()
    if user_templates_dir is not None:
        corgidrp.user_templates_dir = user_templates_dir
    corgidrp.enforce_template_structure = True
    caldb.initialize()
    this_caldb = caldb.CalDB()
    return this_caldb

def step_2_load_cal(this_caldb, main_cal_dir):
    """

    Takes the initialized caldb and loads the calibration files from the main_cal_dir

    Args:
        this_caldb (caldb.CalDB): an instance of an initialized caldb object
        main_cal_dir (str): the path to the main calibration directory

    Returns:
        caldb.CalDB: an instance of a caldb object with the calibration files loaded
    """
    this_caldb.scan_dir_for_new_entries(main_cal_dir)
    return this_caldb

def step_3_process_data(input_filelist, cpgs_xml_filepath, outputdir, template=None):
     """
     
        Process the input file list by autodetecting a template, or accepting an optional template argument. 
        Data will be written out to the output directory.

        Args:
            input_filelist (list): a list of filepaths to the input files
            cpgs_xml_filepath (str): the path to the cpgs xml file
            outputdir (str): the path to the output directory
            template (str): the path to the template file (optional)

     
     """
     walker.walk_corgidrp(input_filelist, cpgs_xml_filepath, outputdir, template=template)