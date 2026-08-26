import os, shutil, glob, argparse
import pytest
import warnings
import numpy as np
import corgidrp
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.fluxcal as fluxcal
import corgidrp.walker as walker
from corgidrp import caldb, check
from photutils.aperture import CircularAperture, aperture_photometry

# this file's folder
thisfile_dir = os.path.dirname(__file__)

@pytest.mark.e2e
def test_l1_to_fluxcal_pol_e2e(e2edata_path, e2eoutput_path):
    # grab input L1 data, consisting of dim (no ND) and bright (ND 475) stars
    l1_input_data_dir = os.path.join(e2edata_path, "POL_fluxcal_sims", "L1")
    l1_input_data_list = glob.glob(os.path.join(l1_input_data_dir, "*_l1_*.fits"))

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)

    # grab polarimetric calibration files
    cal_dir =os.path.join(e2edata_path, "POL_fluxcal_sims", "Cals") 
    db = caldb.CalDB()
    db.scan_dir_for_new_entries(cal_dir)

    # create empty output directory
    test_outputdir = os.path.join(e2eoutput_path, "l1_to_fluxcal_pol_e2e")
    if os.path.exists(test_outputdir):
        shutil.rmtree(test_outputdir)
    os.makedirs(test_outputdir)
    l2b_outputdir = os.path.join(test_outputdir, "l2b_results")
    os.makedirs(l2b_outputdir)

    with warnings.catch_warnings():
        # suppress warning about number of detectornoisemap frames
        warnings.simplefilter("ignore", category=UserWarning)
        # suppress warnings about frames having different header values
        warnings.simplefilter("ignore", category=RuntimeWarning)
        walker.walk_corgidrp(l1_input_data_list, "", l2b_outputdir)

    # grab calibration file
    fluxcal_file = glob.glob(os.path.join(l2b_outputdir, '*abf_cal*.fits'))[0]
    flux_fac = data.FluxcalFactor(fluxcal_file)
    
    # check color filter and ND filter are assigned correctly
    assert flux_fac.filter == "1F"
    assert flux_fac.nd_filter == "ND475"
    print("fluxcal factor", flux_fac.fluxcal_fac)
    print("fluxcal factor error", flux_fac.fluxcal_err)

    #sanity check using the real flux of the target calstar.
    new_l2b_filenames = [os.path.join(l2b_outputdir, f) for f in os.listdir(l2b_outputdir) if f.endswith('l2b.fits')] 
    dataset=data.Dataset(new_l2b_filenames)
    image = dataset[0]
    image_data = np.nanmedian(dataset.all_data, 0)
    #estimate expected flux of calspec standard
    filter_file = fluxcal.get_filter_name(image)
    wave, filter_trans = fluxcal.read_filter_curve(filter_file)
    
    star_name = image.pri_hdr["TARGET"]
    calspec_filepath, calspec_filename = fluxcal.get_calspec_file(star_name)
    flux_ref = fluxcal.read_cal_spec(calspec_filepath, wave)
    flux = fluxcal.calculate_band_flux(filter_trans, flux_ref, wave)

    # crude aperture photometry on the wollaston o and e beams to estimate polarimetric photoelectron count
    pixel_scale = 0.0218
    separation_diameter_arcsec = 7.5
    center = 512

    # figure out where the wollaston beams are
    dpamname = image.ext_hdr['DPAMNAME']
    alignment_angle = 0 if dpamname == 'POL0' else (np.pi / 4)
    dx = int(round((separation_diameter_arcsec * np.cos(alignment_angle)) / (2 * pixel_scale)))
    dy = int(round((separation_diameter_arcsec * np.sin(alignment_angle)) / (2 * pixel_scale)))
    o_x, o_y = center - dx, center + dy
    e_x, e_y = center + dx, center - dy

    # obtain counts and cross check with actual flux
    aper_pos = [(o_x, o_y), (e_x, e_y)]
    apertures = CircularAperture(aper_pos, r=5)
    phot = aperture_photometry(image_data, apertures, method='center')
    # combine counts from o and e beam, divide by exposure time
    counts = (phot['aperture_sum'][0] + phot['aperture_sum'][1]) / image.ext_hdr["EXPTIME"]
    flux_count = flux_fac.fluxcal_fac * counts
    assert flux == pytest.approx(flux_count, rel = 0.05)

    # check headers
    check.compare_to_mocks_hdrs(fluxcal_file)
    assert flux_fac.ext_hdr["DATATYPE"] == "FluxcalFactor"
    assert flux_fac.ext_hdr["DATALVL"] == "CAL"

    os.remove(tmp_caldb_csv)

    print("L1 to pol fluxcal E2E test passed")

if __name__ == "__main__":
    outputdir = thisfile_dir
    e2edata_path = '/home/eshen12345/dev/E2E_Test_Data'

    ap = argparse.ArgumentParser(description='run the l1 to polarimetric absolute flux calibration end-to-end test')
    ap.add_argument('-e2e', '--e2edata_dir', default=e2edata_path,
                    help='Path to test Data Folder [%(default)s]')
    ap.add_argument('-o', '--outputdir', default=outputdir,
                    help='directory to write results to [%(default)s]')
    args = ap.parse_args()
    outputdir = args.outputdir
    test_l1_to_fluxcal_pol_e2e(args.e2edata_dir, args.outputdir)