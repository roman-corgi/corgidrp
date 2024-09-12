import argparse
import os
import pytest
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
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
def test_trap_pump_cal(tvacdata_path, e2eoutput_path):
    # figure out paths, assuming everything is located in the same relative location
    trap_pump_datadir = os.path.join(tvacdata_path, 'TV-20_EXCAM_noise_characterization', 'simulated_trap_pumped_frames')
    sim_traps = os.path.join(tvacdata_path, 'TV-20_EXCAM_noise_characterization', "results", "tpump_results.npy")
    # this is a .npy file; read it in as a dictionary
    td = np.load(sim_traps, allow_pickle=True)
    TVAC_trap_dict = dict(td[()])
    nonlin_path = os.path.join(tvacdata_path, "TV-36_Coronagraphic_Data", "Cals", "nonlin_table_240322.txt")

    # make output directory if needed
    trap_pump_outputdir = os.path.join(e2eoutput_path, "trap_pump_cal_output")
    if not os.path.exists(trap_pump_outputdir):
        os.mkdir(trap_pump_outputdir)

    # define the raw science data to process
    trap_pump_data_filelist = []
    for root, _, files in os.walk(trap_pump_datadir):
        for name in files:
            if not name.endswith('.fits'):
                continue
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

    # add calibration file to caldb
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(nonlinear_cal)

    ####### Run the walker on some test_data

    walker.walk_corgidrp(trap_pump_data_filelist[0:2], "", trap_pump_outputdir, template="trap_pump_cal.json")

    # clean up by removing entry
    this_caldb.remove_entry(nonlinear_cal)

    
    ##### Check against TVAC trap pump dictionary
    with fits.open(trap_pump_outputdir) as hdulist:
        tpump_calibration = hdulist[1]
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
    assert(noncontinuous_count > 0)
    assert(pre_sub_el_count > 0)

    for t in e2e_trap_dict_keys:
        assert(t in TVAC_trap_dict)
        
    
    #Note: removed several tests about sig_E and sig_cs, since we're not saving them.
    for i in range(len(TVAC_trap_dict)):
        t = TVAC_trap_dict[i]
        assert(t in e2e_trap_dict_keys)
        assert(TVAC_trap_dict[t]['E'] == e2e_trap_dict[t]['E'])
        assert(TVAC_trap_dict[t]['cs'] == e2e_trap_dict[t]['cs'])
        assert(TVAC_trap_dict[t]['tau at input T'] == e2e_trap_dict[t]['tau at input T'])
            
    # trap densities should all match if the above passes; that was tested in II&T tests mainly 
    # b/c all the outputs of the trap-pump function were tested

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    tvacdata_dir = "/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the trap pump cal end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    test_trap_pump_cal(tvacdata_dir, outputdir)
