
import corgidrp.caldb as caldb
import corgidrp.walker as walker
import argparse


def step_1_initialize():
    this_caldb = caldb.CalDB()
    return this_caldb

def step_2_load_cal(this_caldb, main_cal_dir):
    this_caldb.scan_dir_for_new_entries(main_cal_dir)
    return this_caldb

def step_3_process_data(input_filelist, cpgs_xml_filepath, outputdir, template =None):
     walker.walk_corgidrp(input_filelist, cpgs_xml_filepath, outputdir, template=template)