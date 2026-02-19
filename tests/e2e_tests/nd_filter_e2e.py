import os, shutil, glob, argparse
import pytest

import corgidrp
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.walker as walker
import corgidrp.nd_filter_calibration as nd_filter_calibration
from corgidrp import caldb, check
import time

# ----------------------------------------------------------------------
@pytest.mark.e2e
def test_nd_filter_e2e(e2edata_path, e2eoutput_path):
    # 1. Synthetic “dim star” frames (no ND)
    fwhm = 3  # pix PSF width
    true_flux_dim = nd_filter_calibration.compute_expected_band_irradiance('TYC 4424-1286-1', '3C')
    cal_factor = true_flux_dim / 200  # erg / (e‑ s⁻¹)

    dim_frames = mocks.create_flux_image(
        true_flux_dim, fwhm, cal_factor, target_name='TYC 4424-1286-1'
    )
    dim_frames.ext_hdr['BUNIT'] = 'photoelectron'
    dim_frames = [dim_frames] if not isinstance(dim_frames, list) else dim_frames

    # sleep for 2 seconds so the next file has a different timestamp
    time.sleep(2)

    # 2. Synthetic “bright star” frames (with ND)
    true_flux_bright = nd_filter_calibration.compute_expected_band_irradiance('Vega', '3C')
    cal_factor = true_flux_dim / 200  # erg / (e‑ s⁻¹)
    od_truth = 2.0                   # optical density to recover
    attenuated_flux = true_flux_bright / (10 ** od_truth)

    fsm_positions = [(0.0, 0.0),                     
                     (1.0, -1.0)]                   

    bright_frames = []
    for fsm_x, fsm_y in fsm_positions:
        frame = mocks.create_flux_image(attenuated_flux, fwhm, cal_factor, fpamname='ND225',     
            target_name='Vega', fsm_x=fsm_x, fsm_y=fsm_y)
        frame.ext_hdr['BUNIT'] = 'photoelectron'
        bright_frames.append(frame)

    # 3. Save raw files for the walker
    simdata_dir = os.path.join(e2eoutput_path, "nd_filter_cal_e2e")
    shutil.rmtree(simdata_dir, ignore_errors=True)
    os.makedirs(simdata_dir)

    # Create input_data subfolder
    input_data_dir = os.path.join(simdata_dir, 'input_l3')
    if not os.path.exists(input_data_dir):
        os.makedirs(input_data_dir)

    # Save all frames and collect file paths
    all_frames = dim_frames + bright_frames
    filelist = []
    for frame in all_frames:
        frame.save(input_data_dir)
        filelist.append(frame.filepath)

    # Initialize a connection to the calibration database
    tmp_caldb_csv = os.path.join(corgidrp.config_folder, 'tmp_e2e_test_caldb.csv')
    corgidrp.caldb_filepath = tmp_caldb_csv
    # remove any existing caldb file so that CalDB() creates a new one
    if os.path.exists(corgidrp.caldb_filepath):
        os.remove(tmp_caldb_csv)

    # 4. Run the DRP walker with outputs saved in the current folder (e2eoutput_path)
    # Remove old NDF cal files first
    for old_file in glob.glob(os.path.join(simdata_dir, "*ndf_cal.fits")):
        os.remove(old_file)
    walker.walk_corgidrp(filelist, "", simdata_dir)

    # 5. Load product & assert if calculated OD matches the input
    nd_file = glob.glob(os.path.join(simdata_dir, "*_ndf_cal*.fits"))
    nd_cal  = data.NDFilterSweetSpotDataset(nd_file[0])

    recovered_od = float(nd_cal.od_values[0])  # use the first entry for the check
    print("Calculated OD:", recovered_od)
    print("Input OD:", od_truth)
    assert recovered_od == pytest.approx(od_truth, abs=1e-1)

    check.compare_to_mocks_hdrs(nd_file[0], mocks.create_default_L2b_headers)
    
    # remove temporary caldb file
    os.remove(tmp_caldb_csv)

    print("ND‑filter E2E test passed")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-tvac", "--e2edata_dir", default=here)
    parser.add_argument("-o",    "--outputdir",   default=here)
    args = parser.parse_args()
    test_nd_filter_e2e(args.e2edata_dir, args.outputdir)
