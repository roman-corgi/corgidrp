import argparse
import os, shutil
import glob
import pytest
import numpy as np

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.fluxcal as fluxcal
from corgidrp import caldb
from corgidrp.check import (check_filename_convention, check_dimensions, 
                           verify_hdu_count, verify_header_keywords, 
                           validate_binary_table_fields, get_latest_cal_file)

@pytest.mark.e2e
def test_expected_results_e2e(e2edata_path, e2eoutput_path):
    # create output dir
    output_dir = os.path.join(e2eoutput_path, 'pol_flux_sim_test_data')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    #mock a point source image
    fwhm = 3
    star_flux = 1.5e-09 #erg/(s*cm^2*AA)
    cal_factor = star_flux/200
    # split the flux unevenly by polarization
    star_flux_left = 0.6 * star_flux
    star_flux_right = 0.4 * star_flux
    flux_image_WP1 = mocks.create_pol_flux_image(star_flux_left, star_flux_right, fwhm, cal_factor, dpamname='POL0', filedir=output_dir, file_save=True)
    flux_image_WP1.ext_hdr['BUNIT'] = 'photoelectron'
    flux_dataset_WP1 = data.Dataset([flux_image_WP1, flux_image_WP1])
    flux_image_WP2 = mocks.create_pol_flux_image(star_flux_left, star_flux_right, fwhm, cal_factor, dpamname='POL45', filedir=output_dir, file_save=True)
    flux_image_WP2.ext_hdr['BUNIT'] = 'photoelectron'
    flux_dataset_WP2 = data.Dataset([flux_image_WP2, flux_image_WP2])

    # check input dataset info
    for i, frame in enumerate(flux_dataset_WP1):
        frame_name = getattr(frame, 'filename', None)
        assert check_filename_convention(frame_name, 'cgi_*_l2b.fits')
        assert check_dimensions(frame.data, (1024, 1024))
        # check all images have the same CFAMNAME value
        assert verify_header_keywords(frame.ext_hdr, {'CFAMNAME': flux_dataset_WP1.frames[0].ext_hdr['CFAMNAME']})
        # check all images have POL0 as DPAMNAME value
        assert verify_header_keywords(frame.ext_hdr, {'DPAMNAME': 'POL0'})
        assert verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L2b'})
        # print CFAMNAME
        print(f'Frame {frame_name} in flux_dataset_WP1 have CFAMNAME {frame.ext_hdr['CFAMNAME']}')
        # prin FSMX and FSMY
        print(f'Frame {frame_name} in flux_dataset_WP1 have FSMX {frame.ext_hdr['FSMX']}')
        print(f'Frame {frame_name} in flux_dataset_WP1 have FSMY {frame.ext_hdr['FSMY']}')
    # same checks for other dataset
    for i, frame in enumerate(flux_dataset_WP2):
        frame_name = getattr(frame, 'filename', None)
        assert check_filename_convention(frame_name, 'cgi_*_l2b.fits')
        assert check_dimensions(frame.data, (1024, 1024))
        # check all images have the same CFAMNAME value
        assert verify_header_keywords(frame.ext_hdr, {'CFAMNAME': flux_dataset_WP2.frames[0].ext_hdr['CFAMNAME']})
        # check all images have POL45 as DPAMNAME value
        assert verify_header_keywords(frame.ext_hdr, {'DPAMNAME': 'POL45'})
        assert verify_header_keywords(frame.ext_hdr, {'DATALVL': 'L2b'})
        # print CFAMNAME
        print(f'Frame {frame_name} in flux_dataset_WP2 have CFAMNAME {frame.ext_hdr['CFAMNAME']}')
        # prin FSMX and FSMY
        print(f'Frame {frame_name} in flux_dataset_WP2 have FSMX {frame.ext_hdr['FSMX']}')
        print(f'Frame {frame_name} in flux_dataset_WP2 have FSMY {frame.ext_hdr['FSMY']}')

    output_dir_WP1 = os.path.join(output_dir, 'WP1')
    output_dir_WP2 = os.path.join(output_dir, 'WP2')
    os.mkdir(output_dir_WP1)
    os.mkdir(output_dir_WP2)
    flux_dataset_WP1.save(output_dir_WP1, ['flux_e2e_WP1_{0}.fits'.format(i) for i in range(len(flux_dataset_WP1))])
    flux_dataset_WP2.save(output_dir_WP2, ['flux_e2e_WP2_{0}.fits'.format(i) for i in range(len(flux_dataset_WP2))])

    data_filelist_WP1 = []
    data_filelist_WP2 = []

    for f in os.listdir(output_dir_WP1):
        data_filelist_WP1.append(os.path.join(output_dir_WP1, f))
    
    for f in os.listdir(output_dir_WP2):
        data_filelist_WP2.append(os.path.join(output_dir_WP2, f))
    
    # make DRP output directory if needed
    fluxcal_outputdir = os.path.join(e2eoutput_path, "l2b_to_pol_fluxcal_factor_output")
    if os.path.exists(fluxcal_outputdir):
        shutil.rmtree(fluxcal_outputdir)
    os.mkdir(fluxcal_outputdir)

    fluxcal_outputdir_WP1 = os.path.join(fluxcal_outputdir, 'WP1')
    fluxcal_outputdir_WP2 = os.path.join(fluxcal_outputdir, 'WP2')
    os.mkdir(fluxcal_outputdir_WP1)
    os.mkdir(fluxcal_outputdir_WP2)

    ####### Run the DRP walker for WP1
    print('Running walker for WP1')
    walker.walk_corgidrp(data_filelist_WP1, '', fluxcal_outputdir_WP1)
    fluxcal_file_WP1 = glob.glob(os.path.join(fluxcal_outputdir_WP1, '*abf_cal*.fits'))[0]
    fluxcal_image_WP1 = data.Image(fluxcal_file_WP1)

    ## check that the calibration file is configured correctly
    # check HDU0 have no data
    assert fluxcal_image_WP1.pri_hdr['NAXIS'] == 0
    # check HDU1 data is a single float
    assert fluxcal_image_WP1.data.dtype.type == corgidrp.image_dtype
    # check err and dq haave the right dimension
    assert fluxcal_image_WP1.err.shape == (1,)
    assert fluxcal_image_WP1.dq.shape == (1,)
    # check filename convention
    assert check_filename_convention(getattr(fluxcal_image_WP1, 'filename', None), 'abf_cal.fits')
    # check header keyword values match with what is expected
    assert verify_header_keywords(fluxcal_image_WP1.ext_hdr, {'DATALVL': 'CAL'})
    assert verify_header_keywords(fluxcal_image_WP1.ext_hdr, {'DATATYPE': 'FluxcalFactor'})
    assert verify_header_keywords(fluxcal_image_WP1.ext_hdr, {'DPAMNAME': flux_image_WP1.ext_hdr['DPAMNAME']})
    assert verify_header_keywords(fluxcal_image_WP1.ext_hdr, {'CFAMNAME': flux_image_WP1.ext_hdr['CFAMNAME']})
    

    #output values
    flux_fac_WP1 = data.FluxcalFactor(fluxcal_file_WP1)
    print("used color filter", flux_fac_WP1.filter)
    print("used ND filter", flux_fac_WP1.nd_filter)
    print("fluxcal factor", flux_fac_WP1.fluxcal_fac)
    print("fluxcal factor error", flux_fac_WP1.fluxcal_err)
    assert flux_fac_WP1.fluxcal_fac == pytest.approx(cal_factor, abs = 1.5 * flux_fac_WP1.fluxcal_err)


    ####### Run the DRP walker for WP2
    print('Running walker for WP2')
    walker.walk_corgidrp(data_filelist_WP2, '', fluxcal_outputdir_WP2)
    fluxcal_file_WP2 = glob.glob(os.path.join(fluxcal_outputdir_WP2, '*abf_cal*.fits'))[0]
    fluxcal_image_WP2 = data.Image(fluxcal_file_WP2)

    ## check that the calibration file is configured correctly
    # check HDU0 have no data
    assert fluxcal_image_WP2.pri_hdr['NAXIS'] == 0
    # check HDU1 data is a single float
    assert fluxcal_image_WP2.data.dtype.type == corgidrp.image_dtype
    # check err and dq haave the right dimension
    assert fluxcal_image_WP2.err.shape == (1,)
    assert fluxcal_image_WP2.dq.shape == (1,)
    # check filename convention
    print(getattr(fluxcal_image_WP2, 'filename', None))
    assert check_filename_convention(getattr(fluxcal_image_WP2, 'filename', None), 'abf_cal.fits')
    # check header keyword values match with what is expected
    assert verify_header_keywords(fluxcal_image_WP2.ext_hdr, {'DATALVL': 'CAL'})
    assert verify_header_keywords(fluxcal_image_WP2.ext_hdr, {'DATATYPE': 'FluxcalFactor'})
    assert verify_header_keywords(fluxcal_image_WP2.ext_hdr, {'DPAMNAME': flux_image_WP2.ext_hdr['DPAMNAME']})
    assert verify_header_keywords(fluxcal_image_WP2.ext_hdr, {'CFAMNAME': flux_image_WP2.ext_hdr['CFAMNAME']})

    #output values
    flux_fac_WP2 = data.FluxcalFactor(fluxcal_file_WP2)
    print("used color filter", flux_fac_WP2.filter)
    print("used ND filter", flux_fac_WP2.nd_filter)
    print("fluxcal factor", flux_fac_WP2.fluxcal_fac)
    print("fluxcal factor error", flux_fac_WP2.fluxcal_err)
    assert flux_fac_WP2.fluxcal_fac == pytest.approx(cal_factor, abs = 1.5 * flux_fac_WP2.fluxcal_err)

    #check the flux values are similar regardless of the wollaston used
    assert flux_fac_WP1.fluxcal_fac == pytest.approx(flux_fac_WP2.fluxcal_fac, rel=0.05)

    # clean up
    this_caldb = caldb.CalDB()
    this_caldb.remove_entry(flux_fac_WP1)
    this_caldb.remove_entry(flux_fac_WP2)



if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the user to edit the file if that is their preferred
    # workflow.
    thisfile_dir = os.path.dirname(__file__)
    outputdir = thisfile_dir
    e2edata_dir =  "/home/ericshen/corgi/E2E_Test_Data/"

    ap = argparse.ArgumentParser(description="run the l2b-> PolFluxcalFactor end-to-end test")
    ap.add_argument("-tvac", "--e2edata_dir", default=e2edata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    outputdir = args.outputdir
    test_expected_results_e2e(e2edata_dir, outputdir)