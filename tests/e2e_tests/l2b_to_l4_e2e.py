import argparse
import copy
import os
import shutil
import numpy as np
from astropy.io import fits
import corgidrp
from corgidrp.data import Image
from corgidrp.mocks import (create_default_L2b_headers, create_default_L3_headers, 
                            create_synthetic_satellite_spot_image, create_ct_psfs)
from corgidrp.l4_to_tda import find_source
import corgidrp.mocks as mocks
import corgidrp.caldb as caldb
import corgidrp.astrom as astrom
import corgidrp.data as corgidata
import corgidrp.walker as walker
import corgidrp.check as check
from corgidrp import corethroughput
import pytest
import glob
from datetime import datetime, timedelta

thisfile_dir = os.path.dirname(__file__) # this file's folder


def visitid_by_pa_aper(pa_aper_values, visitids):
    """Assign visit IDs to PA_APER groups in sorted PA_APER order.

    Args:
        pa_aper_values (array-like): PA_APER values, one per frame.
        visitids (tuple or list): Visit IDs to assign to the sorted unique
            PA_APER values.

    Returns:
        list: Visit ID for each PA_APER value in the input order.
    """
    pa_aper_values = [round(float(pa_aper), 10) for pa_aper in pa_aper_values]
    unique_pa_apers = sorted(set(pa_aper_values))
    if len(unique_pa_apers) != len(visitids):
        raise ValueError(
            f"Expected {len(visitids)} PA_APER values, but found "
            f"{len(unique_pa_apers)}: {unique_pa_apers}"
        )
    return [visitids[unique_pa_apers.index(pa_aper)] for pa_aper in pa_aper_values]


def make_satspot_frame_for_visit(visitid, pa_aper, psfref, sctsrt,
                                 source_image=None, satspot_data=None,
                                 satspot_pri_header=None,
                                 satspot_ext_header=None, err_hdr=None,
                                 dq_hdr=None):
    """Create one L2b satellite-spot frame assigned to a visit split.

    Args:
        visitid (str): VISITID to write into the primary header.
        pa_aper (float): PA_APER value for this visit.
        psfref (int): Primary-header PSFREF value, 0 for science and 1 for
            reference-star frames.
        sctsrt (datetime.datetime): Timestamp used for ``SCTSRT`` and filename
            ordering across no-offset/+offset/-offset groups.
        source_image (corgidrp.data.Image, optional): Existing L2b frame to copy
            for the no-offset satellite-spot group.
        satspot_data (numpy.ndarray, optional): Synthetic image data for the
            +offset/-offset satellite-spot groups.
        satspot_pri_header (astropy.io.fits.Header, optional): Base primary
            header for synthetic satellite-spot frames.
        satspot_ext_header (astropy.io.fits.Header, optional): Base image
            extension header for synthetic satellite-spot frames.
        err_hdr (astropy.io.fits.Header, optional): Base ERR header for
            synthetic satellite-spot frames.
        dq_hdr (astropy.io.fits.Header, optional): Base DQ header for synthetic
            satellite-spot frames.

    Returns:
        corgidrp.data.Image: L2b satellite-spot frame with visit-specific
        headers and filename metadata.
    """
    if source_image is not None:
        satspot_frame = copy.deepcopy(source_image)
        satspot_frame.ext_hdr['SATSPOTS'] = True
        satspot_frame.ext_hdr['FSMPRFL'] = 'NFOV'
    else:
        required_inputs = {
            'satspot_data': satspot_data,
            'satspot_pri_header': satspot_pri_header,
            'satspot_ext_header': satspot_ext_header,
            'err_hdr': err_hdr,
            'dq_hdr': dq_hdr,
        }
        missing_inputs = [
            name for name, value in required_inputs.items() if value is None
        ]
        if missing_inputs:
            raise ValueError(
                "Missing required inputs for synthetic satellite-spot frame: "
                + ", ".join(missing_inputs)
            )

        satspot_frame = Image(
            satspot_data.copy(),
            satspot_pri_header.copy(),
            satspot_ext_header.copy(),
            err_hdr=err_hdr.copy(),
            dq_hdr=dq_hdr.copy(),
        )

    satspot_frame.pri_hdr['VISITID'] = visitid
    satspot_frame.pri_hdr['PA_APER'] = pa_aper
    satspot_frame.pri_hdr['PSFREF'] = psfref
    satspot_frame.ext_hdr['SATSPOTS'] = True
    satspot_frame.ext_hdr['FSMPRFL'] = 'NFOV'
    satspot_frame.ext_hdr['SCTSRT'] = sctsrt.isoformat()
    time_str = sctsrt.strftime('%Y%m%dt%H%M%S%f')[:-5]
    satspot_frame.filename = f"cgi_{visitid}_{time_str}_l2b.fits"

    return satspot_frame


@pytest.mark.e2e
def test_l2b_to_l3(e2edata_path, e2eoutput_path):
    '''

    An end-to-end test that takes the OS11 data and runs it through the L2B to L4 pipeline.

        It checks that: 
            - The two OS11 planets are detected within 1 pixel of their expected separations
            - The calibration files are correctly associated with the output file

        Data needed: 
            - Coronagraphic dataset - taken from OS11 data
            - Reference star dataset - taken from OS11 data
            - Satellite spot dataset - created in the test
        
        Calibrations needed: 
            - AstrometricCalibration
            - CoreThroughputCalibration
            - FluxCalibration
    
    Args:
        e2edata_path (str): Path to the test data
        e2eoutput_path (str): Path to the output directory


    '''

    main_output_dir = os.path.join(e2eoutput_path, "l2b_to_l4_e2e")
    if os.path.exists(main_output_dir):

        shutil.rmtree(main_output_dir)
    os.makedirs(main_output_dir)
    calibrations_dir = os.path.join(main_output_dir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)
    
    # Create input_data subfolder
    input_data_dir = os.path.join(main_output_dir, 'input_l2b')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    ##################################################
    #### Generate an astrometric calibration file ####
    ##################################################

    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not

    field_path = os.path.join(os.path.dirname(__file__),"..","test_data", "JWST_CALFIELD2020.csv")

    # Create separate directory for astrometric calibration input (L1 data)
    astrom_input_dir = os.path.join(main_output_dir, 'astrom_cal_input')
    if not os.path.exists(astrom_input_dir):
        os.makedirs(astrom_input_dir)
    
    #Set rotation to 0 for this test. 
    mock_dataset = mocks.create_astrom_data(field_path=field_path, filedir=None,rotation=0)
    mock_dataset.save(filedir=astrom_input_dir)

    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=mock_dataset, field_path=field_path, find_threshold=5)
    astrom_cal.save(filedir=calibrations_dir)

    # add calibration file to caldb
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(astrom_cal)

    ############################################
    #### Put the OS 11 data into L2B format ####
    ############################################

    #Read in the PSFs
    input_file = 'hlc_os11_frames_with_planets.fits'
    input_hdul = fits.open(os.path.join(e2edata_path,"hlc_os11_v3",input_file))
    input_images = input_hdul[0].data
    header = input_hdul[0].header
    psf_center_x = header['XCENTER']
    psf_center_y = header['YCENTER']
    
    #Get the auxilliary data
    data = np.loadtxt(os.path.join(e2edata_path,"hlc_os11_v3",'hlc_os11_batch_info.txt'), skiprows=2)
    batch = data[:,0].astype(int)
    star = data[:,2].astype(int)
    rotation = data[:,3]
    frame_exptime_sec = data[:,4]
    nframes = data[:,6].astype(int)

    nbatch = 22
    istart = 0

    big_array_size = [1024,1024]
    science_visitids = ("1111111111111111101", "1111111111111111201")
    reference_visitid = "2222222222222222222"

    science_visitids_by_frame = visitid_by_pa_aper(rotation[star != 2], science_visitids)

    image_list = []
    last_sci_image_by_pa = {}
    last_ref_image = None
    science_frame_index = 0
    for ibatch in range(nbatch): 
        # iend = istart + nframes_per_batch
        iend = istart + nframes[ibatch] 

        batch_images = input_images[istart:iend,:,:]

        #collapse these images and stick them into a big array.
        psf_image = np.nanmedian(batch_images, axis=0)

        big_array = np.zeros(big_array_size)
        big_rows, big_cols = big_array_size
        small_rows, small_cols = psf_image.shape

        # Find the middle indices for the big array
        row_start = (big_rows - small_rows) // 2
        col_start = (big_cols - small_cols) // 2

        # Insert the small array into the middle of the big array
        big_array[row_start:row_start + small_rows, col_start:col_start + small_cols] = psf_image

        #Update the PSF Center. Might have columns and row mixed up; doesn't matter for now since psf_center_x and psf_center_y are the same. 
        new_psf_center_x = psf_center_x + col_start
        new_psf_center_y = psf_center_y + row_start

        #Make a BIAS HDU
        bias_hdu = fits.ImageHDU(data=np.zeros_like(big_array[0]))
        bias_hdu.nane = 'BIAS'
        bias_hdu.header['PCOUNT'] = 0
        bias_hdu.header['GCOUNT'] = 1
        bias_hdu.header['EXTNAME'] = 'BIAS'

        #Create the new Image object
        mock_pri_header, mock_ext_header, errhdr, dqhdr, biashdr = create_default_L2b_headers()
        new_image = Image(big_array, mock_pri_header, mock_ext_header, err_hdr=errhdr, dq_hdr=dqhdr,  input_hdulist=[bias_hdu])
        # new_image.ext_hdr.set('PSF_CEN_X', new_psf_center_x)
        # new_image.ext_hdr.set('PSF_CEN_Y', new_psf_center_y)
        new_image.pri_hdr.set('FRAMET', str(frame_exptime_sec[ibatch]))
        new_image.ext_hdr.set('EXPTIME', frame_exptime_sec[ibatch])
        new_image.pri_hdr.set('PA_APER',rotation[ibatch])
        new_image.ext_hdr.set('FSMPRFL','NFOV')
        new_image.ext_hdr.set('LSAMNAME','NFOV')
        new_image.ext_hdr.set('CFAMNAME','1F')
        new_image.ext_hdr.set('FSMLOS', "1") # tip/tilt enabled only in coronagraphic images
        new_image.ext_hdr.set('FPAMNAME', 'HLC12_C2R1')
        new_image.ext_hdr.set('EACQ_ROW', big_cols/2.0)
        new_image.ext_hdr.set('EACQ_COL', big_cols/2.0)
        new_image.err_hdr.set('LAYER_1','combined error') 

        #If Reference star then flag it. 
        if star[ibatch] == 2:
            new_image.pri_hdr.set('VISITID', reference_visitid)
            new_image.pri_hdr.set('PSFREF', 1)
        else:
            visitid = science_visitids_by_frame[science_frame_index]
            science_frame_index += 1
            new_image.pri_hdr.set('VISITID', visitid)
            new_image.pri_hdr.set('PSFREF', 0)

        # Generate proper filename with visitid and current time
        unique_time = (datetime.now() + timedelta(milliseconds=ibatch*100)).strftime('%Y%m%dt%H%M%S%f')[:-5]
        new_image.filename = f"cgi_{new_image.pri_hdr['VISITID']}_{unique_time}_l2b.fits"
        #Save the last science filename for later. 
        if star[ibatch] != 2:
            last_sci_image_by_pa[round(float(new_image.pri_hdr['PA_APER']), 10)] = copy.deepcopy(new_image)
        else:
            last_ref_image = copy.deepcopy(new_image)


        image_list.append(new_image)

        istart = iend

    ########################################################
    #### Add three sat spot images to the dataset       ####
    #### (no-offset, +offset, -offset acquisition groups) ####
    ########################################################

    # Build the +offset and -offset satspot image (identical for mock purposes).
    satellite_spot_image = create_synthetic_satellite_spot_image([55,55],1e-4,0,2,14.79,amplitude_multiplier=1000)
    big_array = np.zeros(big_array_size)
    big_rows, big_cols = big_array_size
    small_rows, small_cols = satellite_spot_image.shape
    row_start = (big_rows - small_rows) // 2
    col_start = (big_cols - small_cols) // 2
    big_array[row_start:row_start + small_rows, col_start:col_start + small_cols] = satellite_spot_image

    mock_satspot_pri_header, mock_satspot_ext_header, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    mock_satspot_ext_header['SATSPOTS'] = True
    mock_satspot_ext_header['FSMPRFL'] = 'NFOV'

    # Base time for ascending SCTSRT across the three satspot groups.
    satspot_base_time = datetime(2024, 1, 2, 0, 0, 0)
    science_pa_apers = sorted({round(float(pa_aper), 10) for pa_aper in rotation[star != 2]})
    science_visitids_for_pa = visitid_by_pa_aper(science_pa_apers, science_visitids)
    if set(last_sci_image_by_pa) != set(science_pa_apers):
        raise ValueError("No science frames were found for one or more PA_APER satellite-spot groups.")
    if last_ref_image is None:
        raise ValueError("No reference frames were found for the reference satellite-spot group.")

    satspot_frame_specs = []
    for pa_aper, visitid in zip(science_pa_apers, science_visitids_for_pa):
        satspot_frame_specs.append((pa_aper, visitid, 0, last_sci_image_by_pa[pa_aper]))
    ref_pa_aper = round(float(last_ref_image.pri_hdr['PA_APER']), 10)
    satspot_frame_specs.append((ref_pa_aper, reference_visitid, 1, last_ref_image))

    ### This block creates the mock satellite-spot frames needed for each science PA_APER / science VISITID
    ### and for the reference VISITID. In short: for each visit, it creates 3 extra frames:
    ###   - a no-offset frame
    ###   - a positive-offset satellite-spot frame
    ###   - a negative-offset satellite-spot frame
    ### So if there are 2 science PA_APER values plus 1 reference VISITID, this creates 9
    ### satellite-spot frames total.
    satellite_spot_frames = []
    for group_index in range(3):
        for i_frame, frame_spec in enumerate(satspot_frame_specs):
            pa_aper, visitid, psfref, nooffset_source_image = frame_spec
            frame_time = satspot_base_time + timedelta(
                seconds=group_index * len(satspot_frame_specs) + i_frame
            )
            if group_index == 0:
                # No-offset frame: copy the last matching frame for this visit.
                satspot_frame = make_satspot_frame_for_visit(
                    visitid,
                    pa_aper,
                    psfref,
                    frame_time,
                    source_image=nooffset_source_image,
                )
            else:
                satspot_frame = make_satspot_frame_for_visit(
                    visitid,
                    pa_aper,
                    psfref,
                    frame_time,
                    satspot_data=big_array,
                    satspot_pri_header=mock_satspot_pri_header,
                    satspot_ext_header=mock_satspot_ext_header,
                    err_hdr=errhdr,
                    dq_hdr=dqhdr,
                )
            satellite_spot_frames.append(satspot_frame)

    image_list.extend(satellite_spot_frames)
    science_visitids_found = {image.pri_hdr['VISITID'] for image in image_list if image.pri_hdr['PSFREF'] == 0}
    reference_visitids_found = {image.pri_hdr['VISITID'] for image in image_list if image.pri_hdr['PSFREF'] == 1}
    assert science_visitids_found == set(science_visitids)
    assert reference_visitids_found == {reference_visitid}
    assert science_visitids_found.isdisjoint(reference_visitids_found)
    science_images = [image for image in image_list if image.pri_hdr['PSFREF'] == 0]
    expected_science_visitids = visitid_by_pa_aper(
        [image.pri_hdr['PA_APER'] for image in science_images],
        science_visitids,
    )
    for image, expected_visitid in zip(science_images, expected_science_visitids):
        assert image.pri_hdr['VISITID'] == expected_visitid
    for image in image_list:
        if image.pri_hdr['PSFREF'] == 1:
            assert image.pri_hdr['VISITID'] == reference_visitid

    #########################################
    #### Save the dataset to a directory ####
    #########################################

    mock_dataset = corgidata.Dataset(image_list)
    mock_dataset.save(filedir=input_data_dir)

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    ## Next step run things through the walker. 
    
    #####################################
    #### Pass the data to the walker ####
    #####################################

    l2b_data_filelist = sorted(glob.glob(os.path.join(input_data_dir, "*.fits")))

    # Organize L3 files into l2b_to_l3 subfolder
    l2b_to_l3_dir = os.path.join(main_output_dir, "l2b_to_l3")
    if not os.path.exists(l2b_to_l3_dir):
        os.mkdir(l2b_to_l3_dir)
    
    walker.walk_corgidrp(l2b_data_filelist, "",l2b_to_l3_dir)


    #Read in an L3 file (now from l2b_to_l3 subfolder)
    l3_filename = glob.glob(os.path.join(l2b_to_l3_dir, "*l3_.fits"))[0]
    l3_image = Image(l3_filename)

    #Check if there's a WCS header
    assert l3_image.ext_hdr['CTYPE1'] == 'RA---TAN'
    assert l3_image.ext_hdr['CTYPE2'] == 'DEC--TAN'

    #Check if the Bunit is correct
    assert l3_image.ext_hdr['BUNIT'] == 'photoelectron/s'
    
    check.compare_to_mocks_hdrs(l3_filename)

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)
    # shutil.rmtree(e2e_data_path)
    # shutil.rmtree(e2eoutput_path)
    

@pytest.mark.e2e
def test_l3_to_l4(e2eoutput_path):
    '''
    An end-to-end test that takes the L3 data and runs it through the L3 to L4 pipeline.

        It checks that: 
            - The two OS11 planets are detected within 1 pixel of their expected separations
            - The calibration files are correctly associated with the output file

        Data needed: 
            - L3 data - taken from the L3 data
            - Satellite spot dataset - created in the test
        
        Calibrations needed: 
            - AstrometricCalibration
            - CoreThroughputCalibration
            - FluxCalibration
    
    Args:
        e2eoutput_path (str): Path to the output directory
    '''

    main_output_dir = os.path.join(e2eoutput_path, "l2b_to_l4_e2e")
    e2eintput_path = os.path.join(main_output_dir, "l2b_to_l3")
    calibrations_dir = os.path.join(main_output_dir, 'calibrations')
    if not os.path.exists(calibrations_dir):
        os.makedirs(calibrations_dir)


    ##################################################
    #### Generate an astrometric calibration file ####
    ##################################################

    # create a simulated image with source guesses and true positions
    # check that the simulated image folder exists and create if not

    field_path = os.path.join(os.path.dirname(__file__),"..","test_data", "JWST_CALFIELD2020.csv")
    
    # Create separate directory for astrometric calibration input (L1 data)
    astrom_input_dir = os.path.join(main_output_dir, 'astrom_cal_input')
    if not os.path.exists(astrom_input_dir):
        os.makedirs(astrom_input_dir)
    
    mock_dataset = mocks.create_astrom_data(field_path=field_path, filedir=None)
    mock_dataset.save(filedir=astrom_input_dir)
    

    # perform the astrometric calibration
    astrom_cal = astrom.boresight_calibration(input_dataset=mock_dataset, field_path=field_path, find_threshold=5)
    astrom_cal.save(filedir=calibrations_dir)

    # add calibration file to caldb
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)
    this_caldb = caldb.CalDB()
    this_caldb.create_entry(astrom_cal)

    ###########################
    #### Make dummy CT cal ####
    ###########################

        # Dataset with some CT profile defined in create_ct_interp
    # Pupil image
    pupil_image = np.zeros([1024, 1024])
    # Set it to some known value for a selected range of pixels
    pupil_image[510:530, 510:530]=1
    prhd, exthd_pupil, errhdr, dqhdr = create_default_L3_headers()
    # DRP
    # cfam filter
    exthd_pupil['CFAMNAME'] = '1F'
    # Add specific values for pupil images:
    # DPAM=PUPIL, LSAM=OPEN, FSAM=OPEN and FPAM=OPEN_12
    exthd_pupil['DPAMNAME'] = 'PUPIL'
    exthd_pupil['LSAMNAME'] = 'OPEN'
    exthd_pupil['FSAMNAME'] = 'OPEN'
    exthd_pupil['FPAMNAME'] = 'OPEN_12'

    data_psf, psf_loc_in, half_psf = create_ct_psfs(50, cfam_name='1F',
    n_psfs=100)
    
    err = np.ones([1024,1024]) 
    data_ct_interp = [Image(pupil_image,pri_hdr = prhd,
        ext_hdr = exthd_pupil, err = err)]
    # Set of off-axis PSFs with a CT profile defined in create_ct_interp
    # First, we need the CT FPM center to create the CT radial profile
    # We can use a miminal dataset to get to know it
    data_ct_interp += [data_psf[0]]
    ct_cal_tmp = corethroughput.generate_ct_cal(corgidata.Dataset(data_ct_interp))
    mocks.rename_files_to_cgi_format(list_of_fits=[ct_cal_tmp], output_dir=calibrations_dir, level_suffix="ctm_cal")
    this_caldb.create_entry(ct_cal_tmp)

    ##########################################
    #### Generate a flux calibration file ####
    ##########################################

    #Create a mock flux calibration file
    fluxcal_factor = 2e-12
    fluxcal_factor_error = 1e-14
    prhd, exthd, errhd, dqhd = create_default_L3_headers()
    # Set consistent header values for flux calibration factor
    exthd['CFAMNAME'] = '1F'
    exthd['DPAMNAME'] = 'PUPIL'
    exthd['LSAMNAME'] = 'OPEN'
    exthd['FSAMNAME'] = 'OPEN'
    exthd['FPAMNAME'] = 'OPEN_12'
    fluxcal_fac = corgidata.FluxcalFactor(fluxcal_factor, err = fluxcal_factor_error, pri_hdr = prhd, ext_hdr = exthd, err_hdr = errhd, input_dataset = mock_dataset)

    mocks.rename_files_to_cgi_format(list_of_fits=[fluxcal_fac], output_dir=calibrations_dir, level_suffix="abf_cal")
    this_caldb.create_entry(fluxcal_fac)

    #####################################
    #### Read in the L3 data and run ####
    #####################################

    # now get any default cal files that might be needed; if any reside in the folder that are not 
    # created by caldb.initialize(), doing the line below AFTER having added in the ones in the previous lines
    # means the ones above will be preferentially selected
    this_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    l3_data_filelist = sorted(glob.glob(os.path.join(e2eintput_path, "*l3_.fits")))

    walker.walk_corgidrp(l3_data_filelist, "", main_output_dir)

    ########################################################################
    #### Read in the psf_subtracted images and test for source detection ###
    ########################################################################

    l4_filename = glob.glob(os.path.join(main_output_dir, "*l4_.fits"))[0]
    psf_subtracted_image = Image(l4_filename)
    psf_subtracted_image.data = psf_subtracted_image.data[-1,:,:] #Just pick one of the KL modes for now
    
    #Find the sources and get their distances from the center
    psf_subtracted_image_with_source = find_source(psf_subtracted_image)
    source_header = psf_subtracted_image_with_source.ext_hdr

    snyx = np.array([list(map(float, source_header[key].split(','))) for key in source_header if key.startswith("SNYX")])
    xcen = psf_subtracted_image_with_source.ext_hdr['STARLOCX']
    ycen = psf_subtracted_image_with_source.ext_hdr['STARLOCY']
    source_distances =np.sort(np.sqrt((snyx[:,1] - xcen)**2 + (snyx[:,2] - ycen)**2))

    ### Get the expected distances
    pixel_scale = 21 #mas/pixel
    lambda_c = 575e-9 #center wavelength of band 1 in m
    roman_D = 2.37 #roman primary diameter
    expected_separations_lambda_over_D = np.array([3.5, 4.5]) #OS11 injected planets
    expected_separations_arcsec = expected_separations_lambda_over_D *  lambda_c / roman_D * 206265
    expected_separations_pixels = expected_separations_arcsec / pixel_scale * 1000 

    import warnings
    from astropy.wcs import WCS, FITSFixedWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FITSFixedWarning)
        wcs_after = WCS(psf_subtracted_image.ext_hdr)
    cd12 = wcs_after.wcs.cd[0,1]
    cd22 = wcs_after.wcs.cd[1,1]

    angle = np.arctan2(-cd12, cd22)
    assert angle == pytest.approx(0.0, abs=1e-6), "WCS CD matrix not properly rotated to North-up East-left."

    #Check that the detected sources are within 1 pixel of the expected separations
    #Assumes that only the correct sources were detected. 
    for i,source_distance in enumerate(expected_separations_pixels):
        assert np.isclose(source_distance, source_distances[i], atol=1)
    print("Found all the sources!")

    # Filename format will be checked in data format test
    check.compare_to_mocks_hdrs(l4_filename)

    print('e2e test for l3_to_l4 calibration passed')

    # remove temporary caldb file
    os.remove(tmp_caldb_csv)
    # shutil.rmtree(e2eoutput_path_l4)
    # shutil.rmtree(e2eintput_path)
    # shutil.rmtree(os.path.join(pathlib.Path.home(), ".corgidrp",'KLIP_SUB'))



if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.


    outputdir = thisfile_dir #"/Users/jmilton/Documents/CGI/E2E_Test_Data2"
    #This folder should contain an OS11 folder: ""hlc_os11_v3" with the OS11 data in it.
    e2edata_dir = '/Users/kevinludwick/Documents/DRP_E2E_Test_Files_v2/E2E_Test_Data' #'/Users/jmilton/Documents/CGI/E2E_Test_Data2'
    #Not actually TVAC Data, but we can put it in the TVAC data folder. 
    ap = argparse.ArgumentParser(description="run the l2b->l4 end-to-end test")

    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    e2edata_dir = args.e2edata_dir
    outputdir = args.outputdir

    test_l2b_to_l3(e2edata_dir, outputdir)
    test_l3_to_l4(outputdir)
    
