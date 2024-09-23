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
from corgidrp.pump_trap_calibration import rebuild_dict

# Adjust the system's limit of open files. We need to load 2000 files at once. 
# some systems don't like that. 
import resource
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6001, hard_limit))

thisfile_dir = os.path.dirname(__file__) # this file's folder

@pytest.mark.e2e
def test_trap_pump_cal(tvacdata_path, e2eoutput_path, e2e=True):
    '''When e2e=True, the end-to-end test is run, which uses more realistic simulated data compared 
    to when e2e=False, in which case the less-realistic simulated data for the unit test in test_trap_pump_calibration is used.
    The recipe for that scaled-down data for the unit test is stored in the e2e_tests folder.

    Args:
        tvacdata_path (str): path to TVAC data root directory
        e2eoutput_path (str): path to output files made by this test
        e2e (bool): If True, run this for the official end-to-end test.  If False, run the scaled-down simulated data used for the unit test for pump_trap_calibration.
    '''
    # figure out paths, assuming everything is located in the same relative location
    if not e2e: # if you want to test older simulated data
        trap_pump_datadir = os.path.join(tvacdata_path, 'TV-20_EXCAM_noise_characterization', 'simulated_trap_pumped_frames')
        sim_traps = os.path.join(tvacdata_path, 'TV-20_EXCAM_noise_characterization', "results", "tpump_results.npy")
    if e2e:
        trap_pump_datadir = os.path.join(tvacdata_path, 'TV-20_EXCAM_noise_characterization', 'simulated_e2e_trap_pumped_frames')
        sim_traps = os.path.join(tvacdata_path, 'TV-20_EXCAM_noise_characterization', "results", "tpump_e2e_results.npy")
    # this is a .npy file; read it in as a dictionary
    td = np.load(sim_traps, allow_pickle=True)
    TVAC_trap_dict = dict(td[()])
    processed_cal_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals")
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_current_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")

    # make output directory if needed
    trap_pump_outputdir = os.path.join(e2eoutput_path, "trap_pump_cal_output")
    if not os.path.exists(trap_pump_outputdir):
        os.mkdir(trap_pump_outputdir)

    # define the raw science data to process
    trap_pump_data_filelist = []
    trap_cal_filename = None
    for root, _, files in os.walk(trap_pump_datadir):
        for name in files:
            if not name.endswith('.fits'):
                continue
            if trap_cal_filename is None:
                trap_cal_filename = name # get first filename fed to walk_corgidrp for finding cal file later
            f = os.path.join(root, name)
            trap_pump_data_filelist.append(f)

    ###### Setup necessary calibration files
    # Create necessary calibration files
    # we are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    # dummy data; basically just need the header info to combine with II&T nonlin calibration
    l1_datadir = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "L1")
    mock_cal_filelist = [os.path.join(l1_datadir, "{0}.fits".format(i)) for i in [90526, 90527]]
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=trap_pump_outputdir, filename="mock_nonlinearcal.fits" )


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
    err_hdr['BUNIT'] = 'detected EM electrons'
    # from CGI_TVAC_Data/TV-20_EXCAM_noise_characterization/tvac_noisemap_original_data/results/bias_offset.txt
    ext_hdr['B_O'] = 0 # bias offset not simulated in the data, so set to 0;  -0.0394 DN from tvac_noisemap_original_data/results
    ext_hdr['B_O_ERR'] = 0 # was not estimated with the II&T code

    # Create a DetectorNoiseMaps object and save it
    noise_maps = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                        input_dataset=mock_input_dataset, err=noise_map_noise,
                                        dq=noise_map_dq, err_hdr=err_hdr)
    noise_maps.save(filedir=trap_pump_outputdir, filename="mock_detnoisemaps.fits")
    

    # add calibration files to caldb
    this_caldb = caldb.CalDB()
    if e2e:
        this_caldb.create_entry(nonlinear_cal)
        this_caldb.create_entry(noise_maps)

    ####### Run the walker on some test_data
    if not e2e: # if you want to test older simulated data
        template = json.load(open(os.path.join(thisfile_dir, "trap_pump_cal_small_size_e2e.json"), 'r'))
        recipe = walker.autogen_recipe(trap_pump_data_filelist, trap_pump_outputdir, template=template)
        walker.run_recipe(recipe)
    if e2e:
        template = json.load(open(os.path.join(thisfile_dir,"trap_pump_cal_e2e.json"), 'r'))
        recipe = walker.autogen_recipe(trap_pump_data_filelist, trap_pump_outputdir, template=template)
        walker.run_recipe(recipe)


    # clean up by removing entry
    this_caldb.remove_entry(nonlinear_cal)
    this_caldb.remove_entry(noise_maps)
    # find cal file (naming convention for data.TrapCalibration class)
    generated_trapcal_file = trap_cal_filename[:-5]+'_trapcal.fits'
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
    assert(unused_fit_data > 0)
    assert(unused_temp_fit_data == 0)
    assert(two_or_less_count > 0)
    assert(noncontinuous_count >= 0)
    assert(pre_sub_el_count > 0)

    for t in e2e_trap_dict_keys:
        assert(t in TVAC_trap_dict)
        
    
    #Note: removed several tests about sig_E and sig_cs, since we're not saving sig_E and sig_cs currently
    for t in list(TVAC_trap_dict.keys()):
        assert(t in e2e_trap_dict_keys)
        assert(((TVAC_trap_dict[t]['E'] is None and np.isnan(e2e_trap_dict[t]['E']))) or 
               np.abs((TVAC_trap_dict[t]['E'] - e2e_trap_dict[t]['E'])/TVAC_trap_dict[t]['E']) < 1e-4)
        assert(((TVAC_trap_dict[t]['cs'] is None and np.isnan(e2e_trap_dict[t]['cs']))) or 
               np.abs((TVAC_trap_dict[t]['cs']-e2e_trap_dict[t]['cs'])/TVAC_trap_dict[t]['cs']) < 1e-4)
        assert(((TVAC_trap_dict[t]['tau at input T'] is None and np.isnan(e2e_trap_dict[t]['tau at input T']))) or 
               np.abs((TVAC_trap_dict[t]['tau at input T']-e2e_trap_dict[t]['tau at input T'])/TVAC_trap_dict[t]['tau at input T']) < 1e-4)
    pass
    # trap densities should all match if the above passes; that was tested in II&T tests mainly 
    # b/c all the outputs of the trap-pump function were tested

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.

    tvacdata_dir = "/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/"

    if False: # making e2e simulated data, which is ENG and includes nonlinearity
        nonlin_path = os.path.join(tvacdata_dir, "TV-36_Coronagraphic_Data", "Cals", "nonlin_table_240322.txt")
        #nonlin_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'test_data', "nonlin_table_TVAC.txt")
        metadata_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','test_data', "metadata_eng.yaml")
        output_dir = r"/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/TV-20_EXCAM_noise_characterization/simulated_e2e_trap_pumped_frames/"
        np.random.seed(39)
        mocks.generate_mock_pump_trap_data(output_dir, metadata_file, e2emode=True, nonlin_path=nonlin_path, EMgain=1.5, read_noise=100, arrtype='ENG')

    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the trap pump cal end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    ap.add_argument('-e2e', '--e2e_flag', default=True, help="True if testing newer simulated data, false if testing older scaled-down data")
    args_here = ['--tvacdata_dir', tvacdata_dir, '--outputdir', outputdir]#, '--e2e_flag',False]
    #args = ap.parse_args()
    args = ap.parse_args(args_here)
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    e2e = args.e2e_flag
    test_trap_pump_cal(tvacdata_dir, outputdir, e2e)
