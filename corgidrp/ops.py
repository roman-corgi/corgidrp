
import corgidrp.caldb as caldb
import corgidrp.walker as walker
import argparse


def step_1_for_all_data_initialize():
    this_caldb = caldb.CalDB()
    return this_caldb

def step_2_for_all_data_load_cal(this_caldb, main_cal_dir):
    this_caldb.scan_dir_for_new_entries(main_cal_dir)
    return this_caldb

def step_3_for_all_data_process(input_filelist, cpgs_xml_filepath, outputdir, template =None):
     walker.walk_corgidrp(input_filelist, cpgs_xml_filepath, outputdir, template=template)