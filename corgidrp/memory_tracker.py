"""
Recipes used

- l1_to_l2b_pc
- l2b_to_l3
- l3_to_l4
"""
import corgidrp
import corgidrp.data as data
from corgidrp.data import Image
import corgidrp.mocks as mocks
from corgidrp.mocks import (create_default_L2b_headers, create_synthetic_satellite_spot_image)
import corgidrp.caldb as caldb
import corgidrp.detector as detector
import corgidrp.astrom as astrom
from corgidrp import corethroughput
import corgidrp.ops as ops
import os
import astropy.time as time
from astropy.io import fits
import numpy as np
from astropy.time import Time
import datetime
from itertools import islice
import psutil
import time
import threading
import argparse
import sys
import warnings 
import shutil
from tqdm import tqdm

##Keeps track of memory usage while running corgidrp from L1 to L4

#records maximum memory usage of the program when initialized
class MemoryTracker:
    __slots__ = ('process', 'max_rss', 'interval', 'running', 'thread')

    def __init__(self, interval=0.1):
        self.process = psutil.Process(os.getpid())
        self.max_rss = 0
        self.interval = interval
        self.running = False
        self.thread = None

    def _track(self):
        get_rss = self.process.memory_info
        max_rss = self.max_rss
        sleep = time.sleep
        while self.running:
            rss = get_rss().rss
            if rss > max_rss:
                max_rss = rss
            sleep(self.interval)
        self.max_rss = max_rss

    def start(self):
        self.max_rss = 0
        self.running = True
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_max_usage_mb(self):
        return self.max_rss / (1024 ** 2)

function_memory_usage = {}
current_tracker = {}
TRACK_MODE = "steps"  # "total" or "steps"
global_tracker = None
STEPS = []

#calls MemoryTracker for each function in STEPS if tracking in steps mode
def profiler(frame, event, arg):
    if TRACK_MODE != "steps":
        return
    if event == 'call':
        func_name = frame.f_code.co_name
        if func_name in STEPS:
            tracker = MemoryTracker()
            current_tracker[frame] = tracker
            tracker.start()

    elif event == 'return':
        func_name = frame.f_code.co_name
        if func_name in STEPS and frame in current_tracker:
            tracker = current_tracker.pop(frame)
            tracker.stop()
            prev_max = function_memory_usage.get(func_name, 0)
            function_memory_usage[func_name] = max(prev_max, tracker.get_max_usage_mb())

#begins ram tracking
def start_tracking(mode="total", steps_list=[]):
    global TRACK_MODE, global_tracker, STEPS
    TRACK_MODE = mode
    STEPS = steps_list
    if mode == "total":
        global_tracker = MemoryTracker()
        global_tracker.start()
    elif mode == "steps":
        sys.setprofile(profiler)
    else:
        raise ValueError("Invalid mode. Use 'total' or 'steps'.")

#stops ram tracking, prints max usage
def stop_tracking():
    if TRACK_MODE == "total" and global_tracker:
        global_tracker.stop()
        print(f"\n[RESULT] Total peak memory usage: {global_tracker.get_max_usage_mb():.2f} MB")
    elif TRACK_MODE == "steps":
        sys.setprofile(None)
        print("\n[RESULT] Peak memory usage per step (MB):")
        for step in STEPS:
            if step in function_memory_usage:
                print(f"{step}: {function_memory_usage[step]:.2f} MB")
    else:
        warnings.warn("No tracking results.")

#delete files of a certain type in a specified directory
def delete_files_in_directory(dir_path, extension):
    for filename in os.listdir(dir_path):
        if filename.endswith(extension):
            path = os.path.join(dir_path, filename)
            os.remove(path)

#duplicate files found in a directory a certain number of times
def duplicate_files(directory, times):
    if not os.path.isdir(directory):
        raise ValueError(f"The path {directory} is not a valid directory.")
    if times < 1:
        raise ValueError("Number of times must be at least 1.")

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(filename)
            
            for i in range(1, times + 1):
                new_filename = f"{name}_copy{i}{ext}"
                new_file_path = os.path.join(directory, new_filename)
                shutil.copy2(file_path, new_file_path)
                print(f"Created: {new_file_path}")

#processes l1_to_l2b_pc for a batch of input data
def l1_to_l2b(data_dir_in, data_dir_out, n_files):
    #only grab up to n_files number of files in input directory even if there's more files there
    l1_data_filelist = [os.path.join(data_dir_in, f) for f in islice(os.listdir(data_dir_in), n_files)]
    print("l1_to_l2b processing num files:", len(l1_data_filelist))

    # TODO: What is the purpose of this my_caldb and the this_caldb? They look like the same thing
    # Cal DB is like some metadata csv about the calibration process
    # Why is this just not a variable, why a csv?
    # connection to cal DB
    my_caldb = caldb.CalDB() 
    # remove other KGain calibrations that may exist in case they don't have the added header keywords
    for i in range(len(my_caldb._db['Type'])):
        if my_caldb._db['Type'][i] == 'KGain':
            my_caldb._db = my_caldb._db.drop(i)
        elif my_caldb._db['Type'][i] == 'Dark':
            my_caldb._db = my_caldb._db.drop(i)
    my_caldb.save()

    # TODO: Understand what each of these systematic errors are.
    # KGain
    kgain = data.KGain('./calibration_data/mock_kgain.fits')
    my_caldb.create_entry(kgain)

    # NoiseMap
    noise_map = data.DetectorNoiseMaps('./calibration_data/mock_detnoisemaps.fits')
    my_caldb.create_entry(noise_map)

    #nonlinearity
    new_nonlinearity = data.NonLinearityCalibration('./calibration_data/nonlin_table_TVAC.fits')
    my_caldb.create_entry(new_nonlinearity)

    #flat
    flat = data.FlatField('./calibration_data/flat_mock.fits')
    my_caldb.create_entry(flat)

    #bp
    bp_map = data.BadPixelMap('./calibration_data/bad_pix_mock.fits')
    my_caldb.create_entry(bp_map)

    #dark
    dark_map = data.Dark('./calibration_data/pc_frame_dark_16_DRK_CAL.fits')
    my_caldb.create_entry(dark_map)

    #run pipeline
    # We basically reload the calibdation_data folder into this_caldb
    this_caldb = ops.step_1_initialize()
    ops.step_2_load_cal(this_caldb, './calibration_data')
    ops.step_3_process_data(l1_data_filelist, '', data_dir_out, template="l1_to_l2b_pc.json")
    
    #remove calibration entries
    my_caldb.remove_entry(kgain)
    my_caldb.remove_entry(noise_map)
    my_caldb.remove_entry(new_nonlinearity)
    my_caldb.remove_entry(flat)
    my_caldb.remove_entry(bp_map)
    my_caldb.remove_entry(dark_map)
    delete_files_in_directory(data_dir_out, 'json')

#processes L1 data for all data batches
def run_l1_to_l2b_iterative(data_dir, n_files):
    #number of total batches
    #only one batch is necessary if testing ram
    ref_batches = 100
    target_batches = 100
    target_rolls = [-13, 13]
    
    #process L1->L2a pc for ref star
    for i in tqdm(range(ref_batches)):
        data_dir_ref = f'{data_dir}/ref_star/batch_{i}'
        l1_to_l2b(data_dir_ref, './L2b_photon_counted_data/ref_star', n_files)

    #process L1->L2a pc for target star
    for roll in tqdm(target_rolls):
        for i in tqdm(range(target_batches)):
            data_dir_target = f'{data_dir}/target_star/roll_{roll}/batch_{i}'
            l1_to_l2b(data_dir_target, f'./L2b_photon_counted_data/target_star_roll_{roll}', n_files)

#generate satallite spot data for L2b->L4
#Code taken from l2b_to_l4_e2e.py from corgidrp
def gen_sat_spot(n_files):
    satellite_spot_image = create_synthetic_satellite_spot_image([55,55],1e-4,0,2,14.79,amplitude_multiplier=1000)
    big_array_size = [1024,1024]
    big_array = np.zeros(big_array_size)
    big_rows, big_cols = big_array_size
    small_rows, small_cols = satellite_spot_image.shape
    # Find the middle indices for the big array
    row_start = (big_rows - small_rows) // 2
    col_start = (big_cols - small_cols) // 2
    # Insert the small array into the middle of the big array
    big_array[row_start:row_start + small_rows, col_start:col_start + small_cols] = satellite_spot_image
    mock_satspot_pri_header, mock_satspot_ext_header, _, _, _ = create_default_L2b_headers()
    mock_satspot_pri_header['SATSPOTS'] = 1
    mock_satspot_ext_header['FSMPRFL']='NFOV'
    sat_spot_image = Image(big_array, mock_satspot_pri_header, mock_satspot_ext_header)
    sat_spot_image.filename ="CGI_0200001999001000{:03d}_20250415T0305102_L2b.fits".format(n_files + 1)
    # generate sat_spot folder
    sat_spot_dir = './L2b_photon_counted_data/sat_spot'
    if not os.path.exists(sat_spot_dir):
        os.makedirs(sat_spot_dir, exist_ok=True)
    sat_spot_image.save('./L2b_photon_counted_data/sat_spot')

#processes l2b_to_l3 and l3_to_l4
def l2b_to_l4(data_dir, n_files):
    #create satallite spot
    gen_sat_spot(n_files)

    #ensures reference star, and target star at both roll angles are present in input data
    n_ref = (n_files // 25) * 7
    n_target = (n_files // 25) * 9
    data_dir_ref = f'{data_dir}/ref_star'
    data_dir_target_roll_m13 = f'{data_dir}/target_star_roll_-13'
    data_dir_target_roll_p13 = f'{data_dir}/target_star_roll_13'
    data_dir_sat = f'{data_dir}/sat_spot'
    l2b_ref = [os.path.join(data_dir_ref, f) for f in islice(os.listdir(data_dir_ref), n_ref)]
    l2b_target_m13 = [os.path.join(data_dir_target_roll_m13, f) for f in islice(os.listdir(data_dir_target_roll_m13), n_target)]
    l2b_target_p13 = [os.path.join(data_dir_target_roll_p13, f) for f in islice(os.listdir(data_dir_target_roll_p13), n_target)]
    l2b_sat = [os.path.join(data_dir_sat, f) for f in os.listdir(data_dir_sat)]
    l2b_data_filelist = l2b_ref + l2b_target_m13 + l2b_target_p13 + l2b_sat
    print(len(l2b_data_filelist))

    #calibration database
    my_caldb = caldb.CalDB()

    #astrometric calibration
    astrom_cal = data.AstrometricCalibration('./calibration_data/mock_astro.fits')
    my_caldb.create_entry(astrom_cal)

    #ct calibration
    ct_cal = data.CoreThroughputCalibration('./calibration_data/mock_ct.fits')
    my_caldb.create_entry(ct_cal)

    #flux calibration
    fluxcal_fac = data.FluxcalFactor('./calibration_data/mock_fluxcal.fits')
    my_caldb.create_entry(fluxcal_fac)

    #process L2b->L3
    this_caldb = ops.step_1_initialize()
    ops.step_2_load_cal(this_caldb, './calibration_data')
    ops.step_3_process_data(l2b_data_filelist, '', './L3_data', template="l2b_to_l3.json")
    delete_files_in_directory('./L3_data', 'json')

    #process L3->L4
    l3_data_filelist = [os.path.join('./L3_data', f) for f in os.listdir('./L3_data')]
    ops.step_3_process_data(l3_data_filelist, '', './L4_output_data', template="l3_to_l4.json")

    #clean up entries
    my_caldb.remove_entry(astrom_cal)
    my_caldb.remove_entry(ct_cal)
    my_caldb.remove_entry(fluxcal_fac)
    
if __name__ == '__main__':
    #L1->L2b step functions
    steps_l1_to_l2b = [
        "prescan_biassub",
        "detect_cosmic_rays",
        "correct_nonlinearity",
        "update_to_l2a",
        "frame_select",
        "convert_to_electrons",
        "get_pc_mean",
        "desmear",
        "cti_correction",
        "flat_division",
        "correct_bad_pixels",
        "update_to_l2b",
    ]

    #L2b->L4 step functions
    steps_l2b_to_l4 = [
        "create_wcs",
        "divide_by_exptime",
        "update_to_l3",
        "distortion_correction",
        "find_star",
        "do_psf_subtraction",
        "update_to_l4"
    ]

    ##change mode to steps to get individual step function data
    ##if using mode='steps' then run L1->L2b and L2b->L4 individually instead of together
    #run and track L1->L2b
    start_tracking(mode=TRACK_MODE, steps_list=steps_l1_to_l2b)
    run_l1_to_l2b_iterative('./L1_input_data', 100)
    stop_tracking()

    # # #create enough copies for testing psf subtraction
    # duplicate_files('./L2b_photon_counted_data/ref_star', 3) # 18
    # duplicate_files('./L2b_photon_counted_data/target_star_roll_-13', 2) # 16
    # duplicate_files('./L2b_photon_counted_data/target_star_roll_13', 2) # 16
    # exit()
    #run and track L2b->L4
    start_tracking(mode=TRACK_MODE, steps_list=steps_l2b_to_l4)
    l2b_to_l4('./L2b_photon_counted_data', 100)
    stop_tracking()
