# Import modules required for data processing and testing
import numpy as np
import shutil
import corgidrp.check as check
import corgidrp.mocks as mocks
import astropy.time as time
import corgidrp
import corgidrp.data as data
import corgidrp.caldb as caldb # This part can cause trouble if imported earlier.
import corgidrp.detector as detector
import astropy.io.fits as fits
import corgidrp.walker as walker
import corgidrp.astrom as astrom
from corgidrp import corethroughput
from astropy.io import fits
import os
import pytest
import argparse

try:
    from proc_cgi_frame.gsw_process import Process
except:
    pass

this_file_dir = os.path.dirname(__file__) # this file's folder

def create_and_clean_folder(folder_name):
    # Create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # Clean up the folder by removing any old files
    for file in os.listdir(folder_name):
        os.remove(os.path.join(folder_name,file))
        
def separate_sci_and_satspots(filelist):
    '''
    This scripts separates the files with satspots from the files without.
    
    Args:
        filelist: A list of files, in no particular order, that may or may not
                  have satspots
                  
    Returns:
        sci_images: A list of images without satspots
        satspots_images: A list of images with satspots
    '''
    sci_images=[]
    satspots_images=[]
    for file in filelist:
        hdu_temp = fits.open(file)
        if hdu_temp[1].header['SATSPOTS']==1:
            satspots_images.append(file)
        elif hdu_temp[1].header['SATSPOTS']==0:
            sci_images.append(file)
    return sci_images, satspots_images  
        
def process_l1_to_l2a(filelist,l2a_outputdir):
    '''
    This script separates the files in the provided list into those with satspots
    and those without. It then processess the two subsets of files from L1 to L2a.
    
    Args:
        filelist: a list of the data files to process
        l2a_outputdir: the directory to store the output l2a files
    '''
    images_sci, images_spots = separate_sci_and_satspots(filelist)
    walker.walk_corgidrp(images_spots, '', l2a_outputdir)
    walker.walk_corgidrp(images_sci, '', l2a_outputdir)
    
def process_l2a_to_l2b(filelist,l2b_outputdir):
    '''
    This script separates the files in the provided list into those with satspots
    and those without. It then processess the two subsets of files from L2a to L2b.
    
    Args:
        filelist: a list of data files to process
        l2b_outputdir: the directory to store the output l2b files
    '''
    images_sci, images_spots = separate_sci_and_satspots(filelist)
    walker.walk_corgidrp(images_spots, '', l2b_outputdir)
    walker.walk_corgidrp(images_sci, '', l2b_outputdir)

@pytest.mark.e2e
def test_l1_to_l4_analog_HLC_band1_e2e(l1_datadir,test_data_dir,e2edata_path,e2eoutput_path):
    '''
    This test function loads simulated HLC band 1 analog data and processes it
    from L1 to L4.
    
    Args:
        l1_datadir: location of the folder containing the L1 data for the test
        test_data: location of the test_data folder that contains JWST_CALFIELD202.csv
        e2edata_path: location of the TVAC test data folder containing the data used for mock calibrations
        e2eoutput_path: directory of e2e output data
    '''
    # Define the file paths for the TVAC test data (used for example calibrations)
    processed_cal_path = os.path.join(e2edata_path, "TV-36_Coronagraphic_Data", "Cals")
    TVAC_datadir = os.path.join(e2edata_path,"TV-36_Coronagraphic_Data","L1")
    
    # Create output directories for the test

    # Top-level output folder
    test_outputdir = os.path.join(e2eoutput_path,"l1_to_l4_e2e")
    for d in [test_outputdir]:
        os.makedirs(d, exist_ok=True)

    # Create input_data subfolder
    input_data_dir = os.path.join(test_outputdir,'input_l1')
    create_and_clean_folder(input_data_dir)
     
    # Create calibrations subfolder
    calibrations_dir = os.path.join(test_outputdir,'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)
        
    # Create separate L2a and L2b outputdirs
    l2a_outputdir = os.path.join(test_outputdir,'l1_to_l2a')
    create_and_clean_folder(l2a_outputdir)
    
    l2b_outputdir = os.path.join(test_outputdir,'l1_to_l2b')
    create_and_clean_folder(l2b_outputdir)
   
    # Create L3 outputdir
    l3_outputdir = os.path.join(test_outputdir,'l1_to_l3')
    create_and_clean_folder(l3_outputdir)
    
    # Create L4 outputdir
    l4_outputdir = os.path.join(test_outputdir,'l1_to_l4')
    create_and_clean_folder(l4_outputdir)
    
    #--------------------------------------------------------------------------
    # Adapt TVAC test data for use with the corgidrp as sample calibration data

    # Calibration files
    nonlin_path = os.path.join(processed_cal_path, "nonlin_table_240322.txt")
    dark_path = os.path.join(processed_cal_path, "dark_current_20240322.fits")
    flat_path = os.path.join(processed_cal_path, "flat.fits")
    fpn_path = os.path.join(processed_cal_path, "fpn_20240322.fits")
    cic_path = os.path.join(processed_cal_path, "cic_20240322.fits")
    bp_path = os.path.join(processed_cal_path, "bad_pix.fits")
        
    mock_cal_filelist = [os.path.join(TVAC_datadir, os.listdir(TVAC_datadir)[i]) for i in [-2,-1]]
        
    # Copy and fix mock cal headers
    mock_cal_dir = os.path.join(test_outputdir,'mock_cal_input')
    os.makedirs(mock_cal_dir,exist_ok=True)
    mock_cal_filelist = [
        shutil.copy2(f,os.path.join(mock_cal_dir,os.path.basename(f)))
        for f in mock_cal_filelist]
    mock_cal_filelist = check.fix_hdrs_for_tvac(mock_cal_filelist,mock_cal_dir)
    #-------------------------------------------------------------------------------
    # Set up necessary calibration files
    pri_hdr, ext_hdr, errhdr, dqhdr = mocks.create_default_calibration_product_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(mock_cal_filelist)
    
    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_l1tol4_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB() # connection to cal DB
        
    # Nonlinearity calibration
    nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[nonlinear_cal], output_dir=calibrations_dir, level_suffix="nln_cal")
    this_caldb.create_entry(nonlinear_cal)
        
    # KGain
    kgain_val = 8.7 # 8.7 is what is in the TVAC headers
    signal_array = np.linspace(0, 50)
    noise_array = np.sqrt(signal_array)
    ptc = np.column_stack([signal_array, noise_array])
    kgain = data.KGain(kgain_val, ptc=ptc, pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                       input_dataset=mock_input_dataset)
    mocks.rename_files_to_cgi_format(list_of_fits=[kgain], output_dir=calibrations_dir, level_suffix="krn_cal")
    this_caldb.create_entry(kgain)
    
    # NoiseMap
    with fits.open(fpn_path) as hdulist:
        fpn_dat = hdulist[0].data
    with fits.open(cic_path) as hdulist:
        cic_dat = hdulist[0].data
    with fits.open(dark_path) as hdulist:
        dark_dat = hdulist[0].data
    noise_map_dat_img = np.array([fpn_dat, cic_dat, dark_dat])
    noise_map_dat = np.zeros((3, detector.detector_areas['SCI']['frame_rows'], detector.detector_areas['SCI']['frame_cols']))
    rows, cols, r0c0 = detector.unpack_geom('SCI', 'image')
    noise_map_dat[:, r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] = noise_map_dat_img
    noise_map_noise = np.zeros([1,] + list(noise_map_dat.shape))
    noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electron'
    ext_hdr['B_O'] = 0
    ext_hdr['B_O_ERR'] = 0
    noise_map = data.DetectorNoiseMaps(noise_map_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
                                       input_dataset=mock_input_dataset, err=noise_map_noise,
                                       dq = noise_map_dq, err_hdr=err_hdr)
    mocks.rename_files_to_cgi_format(list_of_fits=[noise_map], output_dir=calibrations_dir, level_suffix="dnm_cal")
    this_caldb.create_entry(noise_map)
    
    # Flat field
    with fits.open(flat_path) as hdulist:
        flat_dat = hdulist[0].data
    # The predefined ext_hdr assigns the FPAMNAME to be 'HLC12_R2C5'. It should
    # be 'OPEN_12' for the flat. Manually correct this. Also need to switch the
    # DPAMNAME to 'IMAGING' instead of 'IMAGING,IMAGING_FFT'
    flat = data.FlatField(flat_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=mock_input_dataset)
    flat.ext_hdr['FPAMNAME'] = 'OPEN_12'
    flat.ext_hdr['DPAMNAME'] = 'IMAGING'
    mocks.rename_files_to_cgi_format(list_of_fits=[flat], output_dir=calibrations_dir, level_suffix="flt_cal")
    this_caldb.create_entry(flat)
    
    # Bad pixel map
    with fits.open(bp_path) as hdulist:
        bp_dat = hdulist[0].data
    # Make sure BPM includes a dark(-like) frame
    bp_dark_pri, bp_dark_ext, _, _ = mocks.create_default_calibration_product_headers()
    bp_dark_ext['EXPTIME'] = float(mock_input_dataset.frames[0].ext_hdr.get('EXPTIME', 0.0))
    bp_dark_ext['EMGAIN_C'] = float(mock_input_dataset.frames[0].ext_hdr.get('EMGAIN_C', 1.0))
    bp_dark_ext['DRPNFILE'] = 1
    bp_dark = data.Dark(np.zeros_like(bp_dat, dtype=float), pri_hdr=bp_dark_pri, ext_hdr=bp_dark_ext,
                        input_dataset=mock_input_dataset,
                        err=np.zeros((1,) + bp_dat.shape, dtype=float),
                        dq=np.zeros(bp_dat.shape, dtype='uint16'),
                        err_hdr=fits.Header())
    bp_map_inputs = data.Dataset([bp_dark, flat])
    bp_map = data.BadPixelMap(bp_dat, pri_hdr=pri_hdr, ext_hdr=ext_hdr, input_dataset=bp_map_inputs)
    mocks.rename_files_to_cgi_format(list_of_fits=[bp_map], output_dir=calibrations_dir, level_suffix="bpm_cal")
    this_caldb.create_entry(bp_map)
    #--------------------------------------------------------------------------
    # Define the raw data to process
    l1_data_filelist = [os.path.join(l1_datadir,f) for f in os.listdir(l1_datadir) if f.endswith('.fits')]
    # Copy the raw data to the L1 folder
    for file in l1_data_filelist:
        shutil.copy2(file,os.path.join(input_data_dir,os.path.basename(file)))
        
    bad_pix = np.zeros((1024,1024)) # what is used in DRP
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)
    det_params = this_caldb.get_calib(None, data.DetectorParams)
    fwc_pp_e = int(det_params.params['FWC_PP_E']) # same as what is in DRP's DetectorParams
    fwc_em_e = int(det_params.params['FWC_EM_E']) # same as what is in DRP's DetectorParams
    telem_rows_start = det_params.params['TELRSTRT']
    telem_rows_end = det_params.params['TELREND']
    telem_rows = slice(telem_rows_start, telem_rows_end)
    for j, file in enumerate(l1_data_filelist):
        frame_data = fits.getdata(file)
        ext_hdr = fits.getheader(file, ext=1)
        exptime = ext_hdr['EXPTIME']
        em_gain = float(ext_hdr['EMGAIN_C'])
        eperdn = float(ext_hdr['KGAINPAR'])
        b_offset = 0 # what is used in DRP by default
        md_data = fpn_dat/em_gain + exptime*dark_dat + cic_dat
        proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                       b_offset, em_gain, exptime,
                       nonlin_path, dark=md_data, flat=flat.data, desmear_flag=True)
        l2a_im, bp_im, _, _, l2a_fr, bp_fr, _ = proc.L1_to_L2a(frame_data)
    #--------------------------------------------------------------------------
    # Run the walker to process from L1 to L2a
    # Note that the reference star images must be processed separately from the
    # target star images. We will also separate out the images with satspots.
    
    car114_images = [f for f in l1_data_filelist if f[-27]=='1']
    process_l1_to_l2a(car114_images, l2a_outputdir)
    
    car115_images = [f for f in l1_data_filelist if f[-27]=='2']
    process_l1_to_l2a(car115_images, l2a_outputdir)    
    
    car116_images = [f for f in l1_data_filelist if f[-27]=='3']
    process_l1_to_l2a(car116_images, l2a_outputdir)
    
    car117_images = [f for f in l1_data_filelist if f[-27]=='4']
    process_l1_to_l2a(car117_images, l2a_outputdir)
    
    new_l2a_filenames = [os.path.join(l2a_outputdir, f) for f in os.listdir(l2a_outputdir) if f.endswith('l2a.fits')]
    print('Completed processing L1 to L2a')
    
    # Run the walker to process from L2a to L2b
    # Again, the reference and target star images must be processed separately
    # We will also separately process images with and without spots
    car114_l2a = [f for f in new_l2a_filenames if f[-27]=='1']
    process_l2a_to_l2b(car114_l2a, l2b_outputdir)
    
    car115_l2a = [f for f in new_l2a_filenames if f[-27]=='2']
    process_l2a_to_l2b(car115_l2a, l2b_outputdir)
    
    car116_l2a = [f for f in new_l2a_filenames if f[-27]=='3']
    process_l2a_to_l2b(car116_l2a, l2b_outputdir)
    
    car117_l2a = [f for f in new_l2a_filenames if f[-27]=='4']
    process_l2a_to_l2b(car117_l2a, l2b_outputdir)
    
    new_l2b_filenames = [os.path.join(l2b_outputdir, f) for f in os.listdir(l2b_outputdir) if f.endswith('l2b.fits')]
    print('Completed processing L2a to L2b')
    #--------------------------------------------------------------------------
    # Processing from L2b to L3 requires three calibrations:
    #   astrometric calibration
    #   core throughput calibration
    #   flux calibration
    #
    # We will generate mock calibrations.
    
    # Simulated image
    field_path = os.path.join(test_data_dir,'JWST_CALFIELD2020.csv')
    
    # Astrometric calibration input
    astrom_input_dir = os.path.join(calibrations_dir,'astrom_cal_input')
    create_and_clean_folder(astrom_input_dir)
    
    mock_dataset = mocks.create_astrom_data(field_path=field_path,filedir=None,rotation=0)
    mock_dataset.save(filedir=astrom_input_dir)

    # Add calibration file to caldb
    astrom_cal = astrom.boresight_calibration(input_dataset=mock_dataset, field_path=field_path, find_threshold=5)
    astrom_cal.save(filedir=calibrations_dir)

    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(astrom_cal)

    # Core Throughput Calibration
    # Dataset with some CT profile defined in create_ct_interp
    # The DRP will return an error if it does not find at least one pupil image
    # Pupil image
    pupil_image = np.zeros([1024, 1024])
    # Set it to some known value for a selected range of pixels
    pupil_image[510:530, 510:530]=1
    prhd, exthd_pupil, errhdr, dqhdr = mocks.create_default_L3_headers()
    # DRP
    # cfam filter
    exthd_pupil['CFAMNAME'] = '1F'
    # Add specific values for pupil images:
        # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'
        
    data_psf, psf_loc_in, half_psf = mocks.create_ct_psfs(50, cfam_name='1F',
                                                          n_psfs=100)
    
    err = np.ones([1024,1024]) 
    data_ct_interp = [data.Image(pupil_image,pri_hdr = prhd,
                                 ext_hdr = exthd_pupil, err = err)]
    # Set of off-axis PSFs with a CT profile defined in create_ct_interp
    # First, we need the CT FPM center to create the CT radial profile
    # We can use a miminal dataset to get to know it
    data_ct_interp += [data_psf[0]]
    ct_cal_tmp = corethroughput.generate_ct_cal(corgidrp.data.Dataset(data_ct_interp))
    # Change the FPAMNAME to match the sim data
    ct_cal_tmp.ext_hdr['FPAMNAME']='HLC12_C2R5'
    mocks.rename_files_to_cgi_format(list_of_fits=[ct_cal_tmp], output_dir=calibrations_dir, level_suffix="ctm_cal")
    this_caldb.create_entry(ct_cal_tmp)

    # Flux Calibration
    #Create a mock flux calibration file
    fluxcal_factor = 2e-12
    fluxcal_factor_error = 1e-14
    prhd, exthd, errhd, dqhd = mocks.create_default_L3_headers()
    # Set consistent header values for flux calibration factor
    exthd['CFAMNAME'] = '1F'
    exthd['DPAMNAME'] = 'PUPIL'
    exthd['LSAMNAME'] = 'OPEN'
    exthd['FSAMNAME'] = 'OPEN'
    exthd['FPAMNAME'] = 'OPEN_12'
    fluxcal_fac = corgidrp.data.FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, err_hdr = errhd, input_dataset = mock_dataset)

    mocks.rename_files_to_cgi_format(list_of_fits=[fluxcal_fac], output_dir=calibrations_dir, level_suffix="abf_cal")
    this_caldb.create_entry(fluxcal_fac)

    # Run the walker to process from L2b to L3
    # Now all of the images could be processed together, except that for a 
    # full dataset (7,222 files), that takes so much memory (235+ GB) and time that 
    # it takes hours to even get to the printing of the first walker step.
    # Divide up the files by car
    car114_l2b = [f for f in new_l2b_filenames if f[-27]=='1']
    car115_l2b = [f for f in new_l2b_filenames if f[-27]=='2']
    car116_l2b = [f for f in new_l2b_filenames if f[-27]=='3']
    car117_l2b = [f for f in new_l2b_filenames if f[-27]=='4']
    print('Starting walker for CAR114 L2b to L3')
    walker.walk_corgidrp(car114_l2b,'',l3_outputdir)
    print('Starting walker for CAR115 L2b to L3')
    walker.walk_corgidrp(car115_l2b,'',l3_outputdir)
    print('Starting walker for CAR116 L2b to L3')
    walker.walk_corgidrp(car116_l2b,'',l3_outputdir)
    print('Starting walker for CAR117 L2b to L3')
    walker.walk_corgidrp(car117_l2b,'',l3_outputdir)
    #walker.walk_corgidrp(new_l2b_filenames,'',l3_outputdir)
    new_l3_filenames = [os.path.join(l3_outputdir, f) for f in os.listdir(l3_outputdir) if f.endswith('l3_.fits') ]
    print('Completed processing L2b to L3')

    # Run the walker to process from L3 to L4
    walker.walk_corgidrp(new_l3_filenames,'',l4_outputdir)
    new_l4_filenames = [os.path.join(l4_outputdir, f) for f in os.listdir(l4_outputdir) if f.endswith('l4.fits') ]
    print('Completed processing L3 to L4')
    

if __name__=='__main__':
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    
    # Location of the simulated L1 data
    l1_datadir = os.path.join(this_file_dir,'../test_data/input_l1_HLC_Band1_e2e/')
    # Location of the test_data folder that contains JWST_CALFIELD202.csv
    test_data_dir = os.path.join(this_file_dir,'../test_data/')
    e2edata_dir = os.path.join(this_file_dir,'../../../TVAC_Test_Data/E2E_Test_Data/')
    outputdir = this_file_dir
    
    ap = argparse.ArgumentParser(description='run the l1->l4 end-to-end test')
    ap.add_argument("-l1", "--l1_datadir",default=l1_datadir,
                    help="Path to the simulated L1 data [%(default)s]")
    ap.add_argument("-test_data","--test_data_dir",default=test_data_dir,
                    help="Path to the test data folder containing JWST_CALFIELD202.csv [%(default)s]")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    l1_datadir = args.l1_datadir
    test_data_dir = args.test_data_dir
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir
    test_l1_to_l4_analog_HLC_band1_e2e(l1_datadir,test_data_dir,e2edata_dir,outputdir)