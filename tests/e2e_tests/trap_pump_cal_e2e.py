import argparse
import os
import pytest
import json
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.detector as detector
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.check as check
import shutil
try:
    from corgidrp.pump_trap_calibration import rebuild_dict
    from cal.tpumpanalysis.tpump_final import tpump_analysis
except:
    pass
import warnings
# Adjust the system's limit of open files. We need to load 200 files at once.
# some systems don't like that.
try: # this will fail on Windows
    import resource
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (6001, hard_limit))
except:
    pass

thisfile_dir = os.path.dirname(__file__) # this file's folder
metadata_path = os.path.join(thisfile_dir, '..', 'test_data', "metadata_eng.yaml")
#metadata_path = os.path.join(os.path.abspath(os.path.dirname(__name__)), 'tests', 'test_data', "metadata_test.yaml")

def fix_headers_for_tvac(
    list_of_fits,
    ):
    """
    Fixes TVAC headers to be consistent with flight headers.
    Writes headers back to disk

    Args:
        list_of_fits (list): list of FITS files that need to be updated.
    """
    print("Fixing TVAC headers")
    for file in list_of_fits:
        fits_file = fits.open(file)
        prihdr = fits_file[0].header
        exthdr = fits_file[1].header
        # Adjust VISTYPE
        if 'OBSNUM' not in prihdr:
            prihdr['OBSNUM'] = prihdr['OBSID']
        if 'EMGAIN_C' not in exthdr:
            exthdr['EMGAIN_C'] = exthdr['CMDGAIN']
        exthdr['EMGAIN_A'] = -1
        if 'DATALVL' not in exthdr:
            exthdr['DATALVL'] = exthdr['DATA_LEVEL']
        # exthdr['KGAINPAR'] = exthdr['KGAIN']
        if 'OBSNAME' not in prihdr:
            prihdr["OBSNAME"] = prihdr['OBSTYPE']
        prihdr['PHTCNT'] = "False"
        exthdr['ISPC'] = 0
        # Update FITS file
        fits_file.writeto(file, overwrite=True)

@pytest.mark.e2e
def test_trap_pump_cal(e2edata_path, e2eoutput_path):
    '''Data is simulated using mocks.generate_mock_pump_trap_data() function, run through the DRP's trap pump calibration, and
    compared to the results from the II&T trap pump analysis code.

    Args:
        e2edata_path (str): path to TVAC data root directory.
        e2eoutput_path (str): path to output files made by this test.
    '''
    # make output directory if needed
    trap_pump_outputdir = os.path.join(e2eoutput_path, "trap_pump_cal_e2e")
    if not os.path.exists(trap_pump_outputdir):
        os.makedirs(trap_pump_outputdir)

    # Create separate directory for trap pump mock data (for tpump_analysis)
    trap_pump_datadir = os.path.join(trap_pump_outputdir, "input_l1")
    if not os.path.exists(trap_pump_datadir):
        os.mkdir(trap_pump_datadir)
        
    # Create calibrations subfolder for mock calibration products
    calibrations_dir = os.path.join(trap_pump_outputdir, "calibrations")
    if not os.path.exists(calibrations_dir):
        os.mkdir(calibrations_dir)
    
    # Remove all files ending with .json and .fits in the mock data directory
    for root, _, files in os.walk(trap_pump_datadir):
        for fname in files:
            if fname.endswith('.json') or fname.endswith('.fits'):
                os.remove(os.path.join(root, fname))
    np.random.seed(39)
    e2e = True
    # Generate mock trap pump data with standard filenames
    mocks.generate_mock_pump_trap_data(trap_pump_datadir, metadata_path, EMgain=1.5, e2emode=e2e, arrtype='ENG')
    
    # Organize the generated files into temperature/scheme directories for tpump_analysis
    all_files = [f for f in os.listdir(trap_pump_datadir) if f.endswith('.fits') and f.startswith('cgi_')]
    
    # Create a mapping of files to their temperature/scheme info by reading headers
    file_info = {}
    
    for filename in all_files:
        filepath = os.path.join(trap_pump_datadir, filename)
        with fits.open(filepath) as hdul:
            # Read temperature and scheme from extension header
            if len(hdul) > 1 and 'EXCAMT' in hdul[1].header:
                temp = str(hdul[1].header['EXCAMT']) 
                if temp not in file_info:
                    file_info[temp] = []
                file_info[temp].append(filename)
    
    # Organize files into temperature directories and create scheme subdirectories
    for temp, temp_files in file_info.items():
        temp_dir = os.path.join(trap_pump_datadir, temp)
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        
        # Sort files within each temperature to maintain consistent order
        temp_files.sort()
        
        # Group files into schemes (assuming 4 schemes with equal number of files each)
        files_per_scheme = len(temp_files) // 4
        schemes = [1, 2, 3, 4]
        
        for i, sc in enumerate(schemes):
            sch_dir = os.path.join(temp_dir, f'Scheme_{sc}')
            if not os.path.exists(sch_dir):
                os.mkdir(sch_dir)
            
            # Move files for this scheme
            start_idx = i * files_per_scheme
            end_idx = start_idx + files_per_scheme if i < 3 else len(temp_files)  # Last scheme gets remaining files
            
            for j in range(start_idx, end_idx):
                if j < len(temp_files):
                    filename = temp_files[j]
                    old_filepath = os.path.join(trap_pump_datadir, filename)
                    organized_filepath = os.path.join(sch_dir, filename)
                    
                    # Move file to organized directory structure
                    shutil.move(old_filepath, organized_filepath)

    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_current_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")

    ####### run II&T code
    time_head = 'TPTAU'#PHASE_T'
    emgain_head = 'EMGAIN_C'#'EM_GAIN'
    meta_path_eng = os.path.join(os.path.split(thisfile_dir)[0], 'test_data', 'metadata_eng.yaml')
    tau_fit_thresh = 0.8#0.9#0.9#0.8
    cs_fit_thresh = 0.8
    thresh_factor = 1.5#1.5 #3
    length_lim = 5
    ill_corr = True
    tfit_const = True
    offset_min = 10
    offset_max = 10
    pc_min = 0
    pc_max = 2
    mean_field = None#2090 #250 #e- #None
    tauc_min = 0
    tauc_max = 1e-5 #1e-2
    k_prob = 1
    bins_E = 50#70#100#80 # at 80% for noisy, with inj charge
    bins_cs = 5#7#10#8 # at 80%
    sample_data = False #True
    #num_pumps = {1:10000,2:10000,3:10000,4:10000}
    num_pumps = {1:50000,2:50000,3:50000,4:50000}

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        (TVAC_trap_dict, TVAC_trap_densities, TVAC_bad_fit_counter, TVAC_pre_sub_el_count,
        TVAC_unused_fit_data, TVAC_unused_temp_fit_data, TVAC_two_or_less_count,
        TVAC_noncontinuous_count) = tpump_analysis(trap_pump_datadir, time_head,
        emgain_head, num_pumps, meta_path_eng, nonlin_path = nonlin_path,
        length_lim = length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2, tau_fit_thresh = tau_fit_thresh,
        tauc_min = tauc_min, tauc_max = tauc_max, offset_min = offset_min,
        offset_max = offset_max,
        pc_min=pc_min, pc_max=pc_max, k_prob = k_prob, mean_field = mean_field,
        cs_fit_thresh = cs_fit_thresh, bins_E = bins_E, bins_cs = bins_cs,
        sample_data = sample_data)
    ######################

    # define the raw science data to process
    trap_pump_data_filelist = []
    trap_cal_filename = None
    
    # Collect all files from the organized directory structure
    for root, dirs, files in os.walk(trap_pump_datadir):
        for name in files:
            if not name.endswith('.fits') or not name.startswith('cgi_'):
                continue
            if trap_cal_filename is None:
                trap_cal_filename = name # get first filename fed to walk_corgidrp for finding cal file later
            f = os.path.join(root, name)
            trap_pump_data_filelist.append(f)

    # update headers from TVAC data
    #fix_headers_for_tvac(trap_pump_data_filelist)

    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    # dummy data; basically just need the header info to combine with II&T nonlin calibration
    l1_datadir = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "L1")
    mock_cal_filelist = [os.path.join(l1_datadir, os.listdir(l1_datadir)[i]) for i in [-2,-1]]
    #mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]]
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")


    # Load and combine noise maps from various calibration files into a single array
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_current_path) as hdulist:
        dark_current_dat = hdulist[0].data

    # Combine all noise data into one 3D array
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_current_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'],
                              detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img

    # Initialize additional noise map parameters
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    # from CGI_TVAC_Data/TV-20_EXCAM_noise_characterization/tvac_noisemap_original_data/results/bias_offset.txt
    ext_hdr['B_O'] = 0. # bias offset not simulated in the data, so set to 0;  -0.0394 DN from tvac_noisemap_original_data/results
    ext_hdr['B_O_ERR'] = 0. # was not estimated with the II&T code

    # Create a DetectorNoiseMaps object and save it
    noise_maps = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                        input_dataset=mock_input_dataset, err=noise_map_noise,
                                        dq=noise_map_dq, err_hdr=err_hdr)
    mocks.rename_files_to_cgi_format(list_of_fits=[noise_maps], output_dir=calibrations_dir, level_suffix="dnm_cal")


    # add calibration files to caldb
    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()
    if e2e:
        this_caldb.create_entry(nonlinear_cal)
        this_caldb.create_entry(noise_maps)
    # load in files in order they were run on II&T code for exactly the same results
    # trap_pump_data_filelist = np.load(os.path.join(e2edata_path, 'TV-20_EXCAM_noise_characterization', "results", 'tpump_e2e_filelist_order.npy'), allow_pickle=True)
    # trap_pump_data_filelist = trap_pump_data_filelist.tolist()
    # tempp = trap_pump_data_filelist[4]
    # trap_pump_data_filelist[4] = trap_pump_data_filelist[3]
    # trap_pump_data_filelist[3] = tempp

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    
    ####### Run the walker on some test_data
    if not e2e: # if you want to test older simulated data
        template = json.load(open(os.path.join(thisfile_dir, "trap_pump_cal_small_size_e2e.json"), 'r'))
        recipe = walker.autogen_recipe(trap_pump_data_filelist, trap_pump_outputdir, template=template)
        walker.run_recipe(recipe)
    if e2e:
        #template = json.load(open(os.path.join(thisfile_dir,"trap_pump_cal_e2e.json"), 'r'))
        template_path = template = os.path.join(thisfile_dir, '..', '..', 'corgidrp', 'recipe_templates', "trap_pump_cal.json")
        template = json.load(open(template_path, 'r'))
        recipe = walker.autogen_recipe(trap_pump_data_filelist, trap_pump_outputdir, template=template)
        ### Modify they keywords of some of the steps
        for step in recipe['steps']:
            if step['name'] == "calibrate_trap_pump":
                step['keywords'] = {}
                step['keywords']['bin_size'] = None
        walker.run_recipe(recipe)

    # find cal file (naming convention for data.TrapCalibration class)
    for f in os.listdir(trap_pump_outputdir):
        if f.endswith('_tpu_cal.fits'):
            generated_trapcal_file = f
            break
    generated_trapcal_file = os.path.join(trap_pump_outputdir, generated_trapcal_file)
    # Load
    tpump_calibration = data.Image(generated_trapcal_file)
    ##### Check against TVAC trap pump dictionary

    #Convert the output back to a dictionary for more testing.
    e2e_trap_dict = rebuild_dict(tpump_calibration.data)
    e2e_trap_dict_keys = list(e2e_trap_dict.keys())

    #Extract the extra info.
    unused_fit_data = tpump_calibration.ext_hdr['unfitdat']
    unused_temp_fit_data = tpump_calibration.ext_hdr['untempfd']
    two_or_less_count = tpump_calibration.ext_hdr['twoorles']
    noncontinuous_count = tpump_calibration.ext_hdr['noncontc']
    pre_sub_el_count = tpump_calibration.ext_hdr['prsbelct']

    #####
    # Run many of the tests from test_tfit_const_True_sub_noise_ill in ut_tpump_final.py
    # these things are true of the TVAC result, so make sure they are also true of this output
    # assert(unused_fit_data > 0)
    # assert(unused_temp_fit_data == 0)
    # assert(two_or_less_count > 0)
    # assert(noncontinuous_count >= 0)
    # assert(pre_sub_el_count > 0)

    #assert(unused_fit_data == TVAC_unused_fit_data)
    assert(unused_temp_fit_data == TVAC_unused_temp_fit_data)
    assert(two_or_less_count == TVAC_two_or_less_count)
    assert(noncontinuous_count == TVAC_noncontinuous_count)
    assert(pre_sub_el_count == TVAC_pre_sub_el_count)

    for t in e2e_trap_dict_keys:
        assert(t in TVAC_trap_dict)


    #Note: removed several tests about sig_E and sig_cs, since we're not saving sig_E and sig_cs currently
    for t in list(TVAC_trap_dict.keys()):
        assert(t in e2e_trap_dict_keys)
        #assert(((TVAC_trap_dict[t]['E'] is None and np.isnan(e2e_trap_dict[t]['E']))) or
        #       np.abs((TVAC_trap_dict[t]['E'] - e2e_trap_dict[t]['E'])/TVAC_trap_dict[t]['E']) == 0)#< 1e-4)
        #assert(((TVAC_trap_dict[t]['cs'] is None and np.isnan(e2e_trap_dict[t]['cs']))) or
        #       np.abs((TVAC_trap_dict[t]['cs']-e2e_trap_dict[t]['cs'])/TVAC_trap_dict[t]['cs']) == 0)#< 1e-4)
        #assert(((TVAC_trap_dict[t]['tau at input T'] is None and np.isnan(e2e_trap_dict[t]['tau at input T']))) or
        #       np.abs((TVAC_trap_dict[t]['tau at input T']-e2e_trap_dict[t]['tau at input T'])/TVAC_trap_dict[t]['tau at input T']) == 0)#< 1e-4)
    pass
    # trap densities should all match if the above passes; that was tested in II&T tests mainly
    # b/c all the outputs of the trap-pump function were tested

    check.compare_to_mocks_hdrs(generated_trapcal_file)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)


if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.

    e2edata_dir = '/Users/kevinludwick/Documents/DRP_E2E_Test_Files_v2/E2E_Test_Data'#'/Users/jmilton/Documents/CGI/E2E_Test_Data2' #'/home/jwang/Desktop/CGI_TVAC_Data/'

    if False: # making e2e simulated data, which is ENG and includes nonlinearity
        nonlin_path = os.path.join(e2edata_dir, "TV-36_Coronagraphic_Data", "Cals", "nonlin_table_240322.txt")
        #nonlin_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'test_data', "nonlin_table_TVAC.txt")
        metadata_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','test_data', "metadata_eng.yaml")
        output_dir = r"/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/TV-20_EXCAM_noise_characterization/simulated_e2e_trap_pumped_frames/"
        np.random.seed(39)
        mocks.generate_mock_pump_trap_data(output_dir, metadata_file, e2emode=True, nonlin_path=nonlin_path, EMgain=1.5, read_noise=100, arrtype='ENG')

    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the trap pump cal end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args_here = ['--e2edata_dir', e2edata_dir, '--outputdir', outputdir]
    args = ap.parse_args()
    #args = ap.parse_args(args_here)
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_trap_pump_cal(e2edata_dir, outputdir)
    print('e2e test for trap_pump_cal calibration passed')

