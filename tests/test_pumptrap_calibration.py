import re
import os
import glob
import pickle
import numpy as np
# from corgidrp.mocks import generate_mock_pump_trap_data
import corgidrp.mocks as mocks
from corgidrp.detector import imaging_area_geom
from corgidrp.data import Dataset, TrapCalibration
from corgidrp.l1_to_l2a import prescan_biassub
from corgidrp.l2a_to_l2b import em_gain_division
from corgidrp.pump_trap_calibration import tpump_analysis, tau_temp, rebuild_dict



# Adjust the system's limit of open files. We need to load 2000 files at once. 
# some systems don't like that. 
import resource
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6001, hard_limit))


def test_tpump_analysis():
    '''
    After reading in the data and some setup, then run the subset of appropriate tests from
    test_tfit_const_True_sub_noise_ill in ut_tpump_final.py
    '''

    # Set the seed - II&T ut tests don't work everytime, so let's fix it. 
    np.random.seed(39)
    #Generate the mock data:
    test_data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_data', "pump_trap_data_test")
    metadata_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_data', "metadata_test.yaml")
    output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_data')
    print("Generating mock data")
    mocks.generate_mock_pump_trap_data(test_data_dir, metadata_file)
    print("Done generating mock data")

    #Read in all the data. 
    # test_data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_data', "pump_trap_data")
    data_filenames = sorted(glob.glob(os.path.join(test_data_dir, "*.fits")))
    pump_trap_dataset = Dataset(data_filenames)

    #Parse the first three characters of each filename into a temperature
    # temps = [int(os.path.basename(f)[:3]) for f in data_filenames]
    # npumps = [int(os.path.basename(f).split("_")[5]) for f in data_filenames]
    # scheme = [int(os.path.basename(f).split("_")[2]) for f in data_filenames]
    #Hack in some missing header parameters. 
    arrtype = 'SCI'
    # em_gain = 10 #The default in generate_test_data.py

    for j,frame in enumerate(pump_trap_dataset):
        frame.ext_hdr['ARRTYPE'] = arrtype
        # frame.ext_hdr['CMDGAIN'] = em_gain
        # frame.ext_hdr['EXCAMT'] = temps.pop(0)

        #Get the scheme the the "scheme" list. For the current scheme set the header keyword TPSCHEM* 
        # equal to the npumps for this filename, where * is equal to the scheme. For the other schemes (up to * =4) set TPSCHEM* =0.
        # for i in range(1, 5):
        #     if scheme[j] == i:
        #         frame.ext_hdr['TPSCHEM' + str(i)] = npumps.pop(0)
        #     else:
        #         frame.ext_hdr['TPSCHEM' + str(i)] = 0

        #Get the phase time from the filename: its between the string "phasetime" and the ".fits" extension at the end
        # phase_time = os.path.basename(data_filenames[j]).split("phasetime")[1].split(".fits")[0]
        # frame.ext_hdr['TPTAU'] = float(phase_time)

    #Run the bias subtraction
    # #TODO Figure out which detector regions to pass in here. 
    # arrtype = pump_trap_dataset[0].ext_hdr['ARRTYPE']

    #Detector regions for the smaller pump_trap_data - taken from metadata_test.yaml

    #Subtract the prescane Bias
    bias_subbed_dataset = prescan_biassub(pump_trap_dataset, detector_regions=mocks.detector_areas_test,use_imaging_area=True)

    ## Note the data were not generated with non-linearity
    #Correct for non-linearity - use the fits file derived from nonlin_sample.csv
    # nonlin_fits_filepath = os.path.join(os.path.dirname(__file__), "test_data", "nonlin_sample.fits")
    # non_linearity_correction = NonLinearityCalibration(nonlin_fits_filepath)
    # linear_dataset = correct_nonlinearity(bias_subbed_dataset, non_linearity_correction)

    #Divide by EM gain
    emgain_divided_dataset = em_gain_division(bias_subbed_dataset)

    #Done preliminary data processing. Now running the tpump_analysis

    length_lim = 5
    tau_fit_thresh = .8#.5#0.8#0.65 #0.8
    cs_fit_thresh = .8#.5#0.8# 0.2 #0.65 #0.8#0.65 #0.8
    thresh_factor = 1.5 #.5
    ill_corr = True
    tfit_const = True
    input_T = 185
    bins_E = 50
    bins_cs = 5
    mean_field = None #500

    tpump_calibration = tpump_analysis(emgain_divided_dataset,
                        mean_field=mean_field,
                        length_lim = length_lim, thresh_factor = thresh_factor,
                        ill_corr = ill_corr, tfit_const = tfit_const,
                        tau_min = 0.7e-6, tau_max = 1.3e-2,
                        tau_fit_thresh = tau_fit_thresh,
                        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
                        pc_min=0, pc_max=2,
                        cs_fit_thresh = cs_fit_thresh, 
                        input_T=input_T,
                        bins_E=bins_E, bins_cs=bins_cs)
    # filename check
    test_filename = emgain_divided_dataset.frames[-1].filename.split('.fits')[0] + '_tpu_cal.fits'
    test_filename = re.sub('_L[0-9].', '', test_filename)
    assert tpump_calibration.filename == test_filename

    #Extract the extra info. 
    unused_fit_data = tpump_calibration.ext_hdr['unfitdat']
    unused_temp_fit_data = tpump_calibration.ext_hdr['untempfd']
    two_or_less_count = tpump_calibration.ext_hdr['twoorles']
    noncontinuous_count = tpump_calibration.ext_hdr['noncontc']
    pre_sub_el_count = tpump_calibration.ext_hdr['prsbelct']
    trap_densities = tpump_calibration.hdu_list[tpump_calibration.hdu_names.index('trap_densities')-2].data

    #####
    # Run many of the tests from test_tfit_const_True_sub_noise_ill in ut_tpump_final.py
    
    assert(unused_fit_data > 0)
    assert(unused_temp_fit_data == 0)
    assert(two_or_less_count > 0)
    assert(noncontinuous_count >= 0)
    assert(pre_sub_el_count > 0)

    #Convert the output back to a dictionary for more testing. 
    trap_dict = rebuild_dict(tpump_calibration.data)
    trap_dict_keys = list(trap_dict.keys())

    #Truth values for the sim dataset from ut_tpump_final.py
    test_trap_dict_keys = [((26, 28), 'CENel1', 0),
            ((50, 50), 'RHSel1', 0), ((60, 80), 'LHSel2', 0),
            ((68, 67), 'CENel2', 0), ((98, 33), 'LHSel3', 0),
            ((98, 33), 'RHSel2', 0), ((41, 15), 'CENel3', 0),
            ((89, 2), 'RHSel3', 0), ((89, 2), 'LHSel4', 0),
            [((10, 10), 'LHSel4', 0), ((10, 10), 'LHSel4', 1)],
            ((56, 56), 'CENel4', 0), ((77, 90), 'RHSel4', 0),
            ((77, 90), 'CENel2', 0), ((13, 21), 'LHSel1', 0)]
    trap_dict_E = [0.32, 0.32, 0.32, 0.32, 0.28, 0.32, 0.32, 0.32,
            0.28, [0.32, 0.28], 0.32, 0.28, 0.32, 0.32]
    trap_dict_cs = [2e-15, 2e-15, 2e-15, 2e-15, 12e-15, 2e-15, 2e-15,
            2e-15, 12e-15, [2e-15, 12e-15], 2e-15, 12e-15, 2e-15, 2e-15]

    #Note: removed several tests about sig_E and sig_cs, since we're not saving them.
    for i in range(len(test_trap_dict_keys)):
        if i!= 9:
            t = test_trap_dict_keys[i]
            assert(t in trap_dict_keys)
            # A good uncertainty for a single-measured value (e.g., 1 set
            # of trap-pumped frames from which we extract 1 tau per trap)
            # is the standard deviation (std dev) from the fit for tau,
            # assuming random sources of noise.  However, since we have
            # non-normal noise from the detector, some of the tests below
            # occasionally fail if we only consider 1 std dev.  In light
            # of that, we use 2 standard deviations instead.
            assert(np.isclose(trap_dict[t]['E'], trap_dict_E[i], atol = 0.05))
            assert(np.isclose(trap_dict[t]['cs'], trap_dict_cs[i], rtol = 0.1))
            # must multiply cs (in cm^2) by 1e15 to get the cs input to
            # tau_temp() to be as expected, which is 1e-19 m^2
            assert(np.isclose(trap_dict[t]['tau at input T'],  tau_temp(input_T,trap_dict_E[i],trap_dict_cs[i]*1e15), rtol = 0.1))
            
        if i==9: #special case of (10,10)
            t1, t2 = test_trap_dict_keys[i]
            assert(t1 in trap_dict)
            assert(t2 in trap_dict)
            # check closeness and within error for t1
            assert(np.isclose(trap_dict[t1]['E'], trap_dict_E[i][0], atol = 0.05) or np.isclose(trap_dict[t1]['E'], trap_dict_E[i][1], atol = 0.05))
            assert(np.isclose(trap_dict[t1]['cs'], trap_dict_cs[i][0], rtol = 0.1) or np.isclose(trap_dict[t1]['cs'], trap_dict_cs[i][1], rtol = 0.1))
            assert(np.isclose(trap_dict[t1]['tau at input T'], tau_temp(input_T, trap_dict_E[i][0], trap_dict_cs[i][0]*1e15), rtol = 0.1) or np.isclose(trap_dict[t1]['tau at input T'], tau_temp(input_T, trap_dict_E[i][1], trap_dict_cs[i][1]*1e15), rtol = 0.1))
            # check closeness and within error for t2
            assert(np.isclose(trap_dict[t2]['E'], trap_dict_E[i][0], atol = 0.05) or np.isclose(trap_dict[t2]['E'], trap_dict_E[i][1], atol = 0.05))
            assert(np.isclose(trap_dict[t2]['cs'], trap_dict_cs[i][0], rtol = 0.1) or np.isclose(trap_dict[t2]['cs'], trap_dict_cs[i][1], rtol = 0.1))
            assert(np.isclose(trap_dict[t2]['tau at input T'], tau_temp(input_T, trap_dict_E[i][0], trap_dict_cs[i][0]*1e15), rtol = 0.1) or np.isclose(trap_dict[t2]['tau at input T'], tau_temp(input_T, trap_dict_E[i][1], trap_dict_cs[i][1]*1e15), rtol = 0.1))
                

    nrows, ncols, _ = imaging_area_geom('SCI')
    assert(len(trap_densities) == 2)

    for tr in trap_densities:
        # self.assertTrue(np.isclose(tr[0], 11/(nrows*ncols), atol=1e-4) \
        #    tra or np.isclose(tr[0], 4/(nrows*ncols), atol=1e-4))
        assert(tr[0] == 11/(nrows*ncols) or tr[0] == 4/(nrows*ncols))
        # 11 traps that have 0.32eV, 2e-15 cm^2
        #if np.isclose(tr[0], 11/(nrows*ncols), atol=1e-4):
        if tr[0] == 11/(nrows*ncols):

            # with atol aligning with bins trainput to tpump_final()
            assert(np.isclose(tr[1], 0.32, atol=0.05))
            assert(np.isclose(tr[2], 2e-15, rtol=0.1))
        # 4 traps that have 0.28eV, 12e-15 cm^2
        #if np.isclose(tr[0], 4/(nrows*ncols), atol=1e-4):
        if tr[0] == 4/(nrows*ncols):
            assert(np.isclose(tr[1], 0.28, atol=0.05))
            assert(np.isclose(tr[2], 12e-15, rtol=0.1))
        

    # check they can be pickled (for CTC operations)
    pickled = pickle.dumps(tpump_calibration)
    pickled_trap = pickle.loads(pickled)
    assert np.all((tpump_calibration.data == pickled_trap.data) | np.isnan(tpump_calibration.data)) 

    # save to disk and reload and try to pickle again
    tpump_calibration.save(filedir=output_dir, filename="trap_cal.fits")
    tpump_calibration_2 = TrapCalibration(os.path.join(output_dir, "trap_cal.fits"))
    pickled = pickle.dumps(tpump_calibration_2)
    pickled_trap = pickle.loads(pickled)
    assert np.all((tpump_calibration.data == pickled_trap.data) | np.isnan(tpump_calibration.data)) # check against the original

    
if __name__ == "__main__":
    test_tpump_analysis()
    
