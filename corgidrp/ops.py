
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data using corgidrp.")
    parser.add_argument("main_cal_dir", type=str, help="Directory containing calibration data.")
    parser.add_argument("input_filelist", type=str, help="List of input files to process.")
    parser.add_argument("cpgs_xml_filepath", type=str, help="Path to the CPGS XML file.")
    parser.add_argument("outputdir", type=str, help="Directory to save the output.")
    parser.add_argument("--template", type=str, default=None, help="Optional template file.")

    args = parser.parse_args()

    # Step 1: Initialize
    this_caldb = step_1_for_all_data_initialize()

    # Step 2: Load calibration data
    this_caldb = step_2_for_all_data_load_cal(this_caldb, args.main_cal_dir)

    # Step 3: Process data
    step_3_for_all_data_process(args.input_filelist, args.cpgs_xml_filepath, args.outputdir, template=args.template)