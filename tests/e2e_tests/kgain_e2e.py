import argparse
import os, shutil
import glob
import pytest
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.walker as walker
import corgidrp.caldb as caldb
from corgidrp.sorting import sort_pupilimg_frames
from corgidrp.calibrate_nonlin import nonlin_kgain_dataset_2_stack

import warnings


try:
    from cal.kgain.calibrate_kgain import calibrate_kgain
    import cal
except:
    # For tests to pass. Is it not necessary? See 'default_config_file' below
    print('Install e2e dependencies with pip install -r requirements_e2etests.txt')
    pass

thisfile_dir = os.path.dirname(__file__) # this file's folder


def set_vistype_for_tvac(
    list_of_fits,
    ):
    """ Adds proper values to VISTYPE for non-linearity calibration.

    This function is unnecessary with future data because data will have
    the proper values in VISTYPE. Hence, the "tvac" string in its name.

    Args:
    list_of_fits (list): list of FITS files that need to be updated.
    """
    print("Adding VISTYPE='CGIVST_CAL_PUPIL_IMAGING' to TVAC data")
    for file in list_of_fits:
        fits_file = fits.open(file)
        prihdr = fits_file[0].header
        # Adjust VISTYPE
        if prihdr['VISTYPE'] == 'N/A' or prihdr['VISTYPE'] == "PUPILIMG":
            prihdr['VISTYPE'] = 'CGIVST_CAL_PUPIL_IMAGING'
        exthdr = fits_file[1].header
        if exthdr['EMGAIN_A'] == 1:
            exthdr['EMGAIN_A'] = -1 #for new SSC-updated TVAC files which have EMGAIN_A by default as 1 regardless of the commanded EM gain
        # Update FITS file
        fits_file.writeto(file, overwrite=True)

# tvac_kgain: 8.49404981510777 e-/DN, result from new iit code with specified file input order; used to be #8.8145 #e/DN,
# tvac_readnoise: 121.76070832489948 e-, result from new iit code with specified file input order; used to be 130.12 e-

@pytest.mark.e2e
def test_l1_to_kgain(e2edata_path, e2eoutput_path):

    # sort and prepare raw files to run through both II&T and DRP
    default_config_file = os.path.join(cal.lib_dir, 'kgain', 'config_files', 'kgain_parms.yaml')
    stack_arr2_f = []
    stack_arr_f = []
    box_data = os.path.join(e2edata_path, 'TV-20_EXCAM_noise_characterization', 'nonlin', 'kgain') 
    # for f in os.listdir(box_data):
    #     file = os.path.join(box_data, f)
    #     if not file.endswith('.fits'):
    #         continue
    #     for i in range(51841, 51871):
    #         if str(i) in file:
    #             stack_arr2_f.append(file)
    #             with fits.open(file, mode='update') as hdus:
    #                 try:
    #                     hdus[0].header['VISTYPE'] = 'CGIVST_CAL_PUPIL_IMAGING'
    #                     hdus[0].header['OBSTYPE'] = 'MNFRAME'
    #                 except:
    #                     pass
    #                 try:
    #                     hdus[1].header['OBSTYPE'] = 'MNFRAME'
    #                 except:
    #                     pass
    #             exit
    #     for i in range(51731, 51841):
    #         if str(i) in file:
    #             stack_arr_f.append(file)
    #             with fits.open(file, mode='update') as hdus:
    #                 try:
    #                     hdus[0].header['VISTYPE'] = 'CGIVST_CAL_PUPIL_IMAGING'
    #                     hdus[0].header['OBSTYPE'] = 'KGAIN'
    #                 except:
    #                     pass
    #                 try:
    #                     hdus[1].header['OBSTYPE'] = 'KGAIN'
    #                 except:
    #                     pass
    #             exit
    file_list = []
    for f in os.listdir(box_data):
        file = os.path.join(box_data, f)
        if not file.lower().endswith('.fits'):
            continue
        file_list.append(file)
    file_dataset = data.Dataset(file_list)
    out_dataset = sort_pupilimg_frames(file_dataset, cal_type='k-gain')
    cal_list, mean_frame_list, exp_arr, _, _, _, datetimes_sort_inds, truncated_set_len = nonlin_kgain_dataset_2_stack(out_dataset, apply_dq = False, cal_type='kgain')
    cal_arr = cal_list[0]
    split_arr = np.arange(0,len(cal_arr), truncated_set_len)[1:]
    cal_ed_list = np.split(cal_arr, split_arr)
    stack_arr = np.stack(cal_ed_list)
    stack_arr2 = np.stack(mean_frame_list)

    #stack_arr2 = np.stack(stack_arr2)
    # fileorder_filepath = os.path.join(os.path.split(box_data)[0], 'results', 'TVAC_kgain_file_order.npy')
    #np.save(fileorder_filepath, stack_arr_f+stack_arr2_f)
    # stack_arr_f = sorted(stack_arr_f)
    # stack_dat = data.Dataset(stack_arr_f)
    # stack2_dat = data.Dataset(stack_arr2_f)

    ####### ordered_filelist is simply the combination of the the two ordered stacks that are II&T inputs is the input needed for the DRP calibration
    #ordered_filelist = stack_arr_f+stack_arr2_f
    ordered_filelist = []
    for f in os.listdir(box_data):
        if not f.lower().endswith('.fits'):
            continue
        ordered_filelist.append(os.path.join(box_data, f))

    ##### Fix TVAC headers
    #fix_headers_for_tvac(ordered_filelist)

    ########## Calling II&T code
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        (tvac_kgain, tvac_readnoise, mean_rn_std_e, ptc) = calibrate_kgain(stack_arr, stack_arr2, emgain=1, min_val=800, max_val=3000, 
                        binwidth=68, config_file=default_config_file, 
                        mkplot=None, verbose=None)
    
    ########### Now run the DRP

    # make DRP output directory if needed
    kgain_outputdir = os.path.join(e2eoutput_path, "kgain_cal_e2e")
    if os.path.exists(kgain_outputdir):
        shutil.rmtree(kgain_outputdir)
    os.makedirs(kgain_outputdir)

    # Create input_data subfolder
    input_data_dir = os.path.join(kgain_outputdir, 'input_l1')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    # Copy files to input_data directory with proper naming
    for i, file_path in enumerate(ordered_filelist):
        shutil.copy2(file_path, input_data_dir)
    
    # Update ordered_filelist to point to new files
    ordered_filelist = []
    for f in os.listdir(input_data_dir):
        if f.endswith('.fits'):
            ordered_filelist.append(os.path.join(input_data_dir, f))
    
    set_vistype_for_tvac(ordered_filelist)

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    
    ####### Run the DRP walker
    print('Running walker')
    #walker.walk_corgidrp(ordered_filelist, "", kgain_outputdir, template="l1_to_kgain.json")
    recipe = walker.autogen_recipe(ordered_filelist, kgain_outputdir)
    ### Modify they keywords of some of the steps
    for step in recipe[1]['steps']:
        if step['name'] == "calibrate_kgain":
            step['keywords']['apply_dq'] = False #do not apply the cosmics in e2etests
    walker.run_recipe(recipe[1], save_recipe_file=True)

    ####### Load in the output data. It should be the latest kgain file produced.
    possible_kgain_files = glob.glob(os.path.join(kgain_outputdir, '*_krn_cal*.fits'))
    kgain_file = max(possible_kgain_files, key=os.path.getmtime) # get the one most recently modified

    kgain = data.KGain(kgain_file)
    
    ##### compare II&T ("TVAC") results with DRP results
    new_kgain = kgain.value
    new_readnoise = kgain.ext_hdr["RN"]
    print("determined kgain:", new_kgain)
    print("determined read noise", new_readnoise)    
    
    diff_kgain = new_kgain - tvac_kgain
    diff_readnoise = new_readnoise - tvac_readnoise
    print ("difference to TVAC kgain:", diff_kgain)
    print ("difference to TVAC read noise:", diff_readnoise)
    print ("error of kgain:", kgain.error)
    print ("error of readnoise:", kgain.ext_hdr["RN_ERR"])

    # Update tolerance to account for float32 precision differences
    assert np.abs(diff_kgain) < 1e-6
    assert np.abs(diff_readnoise) < 1e-6 

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)


    
if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    e2edata_dir = '/Users/jmilton/Documents/CGI/E2E_Test_Data2'#"/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/Working_Folder/"#'/home/jwang/Desktop/CGI_TVAC_Data/''/home/jwang/Desktop/CGI_TVAC_Data/'  
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->kgain end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_l1_to_kgain(e2edata_dir, outputdir)
