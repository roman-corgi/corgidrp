import os, shutil, glob, argparse
import pytest

import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.walker as walker
import corgidrp.nd_filter_calibration as nd_filter_calibration
from corgidrp import caldb

# ----------------------------------------------------------------------
@pytest.mark.e2e
def test_nd_filter_e2e(e2edata_path):
    # 1. Synthetic “dim star” frames (no ND)
    fwhm = 3  # pix PSF width
    true_flux_dim = nd_filter_calibration.compute_expected_band_irradiance('TYC 4424-1286-1', '3C')
    cal_factor = true_flux_dim / 200  # erg / (e‑ s⁻¹)

    dim_frames = mocks.create_flux_image(
        true_flux_dim, fwhm, cal_factor, target_name='TYC 4424-1286-1'
    )
    dim_frames = [dim_frames] if not isinstance(dim_frames, list) else dim_frames

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
        bright_frames.append(frame)

    # 3. Save raw files for the walker
    simdata_dir = os.path.join(os.path.dirname(e2edata_path), "nd_filter_e2e_output")
    shutil.rmtree(simdata_dir, ignore_errors=True)
    os.makedirs(simdata_dir)

    for i, frame in enumerate(dim_frames + bright_frames):
        input_prihdr = frame.pri_hdr
        input_exthdr = frame.ext_hdr
        frame.save(simdata_dir, f"CGI_{input_prihdr['VISITID']}_{data.format_ftimeutc(input_exthdr['FTIMEUTC'])}_L3_.fits")

    filelist = [os.path.join(simdata_dir, f) for f in os.listdir(simdata_dir)]

    # 4. Run the DRP walker with outputs saved in the current folder (e2eoutput_path)
    # Remove old NDF cal files first
    for old_file in glob.glob(os.path.join(simdata_dir, "*NDF_CAL.fits")):
        os.remove(old_file)
    walker.walk_corgidrp(filelist, "", simdata_dir)

    # 5. Load product & assert if calculated OD matches the input
    nd_file = glob.glob(os.path.join(simdata_dir, "*_NDF_CAL*.fits"))
    nd_cal  = data.NDFilterSweetSpotDataset(nd_file[0])

    recovered_od = float(nd_cal.od_values[0])  # use the first entry for the check
    print("Calculated OD:", recovered_od)
    print("Input OD:", od_truth)
    assert recovered_od == pytest.approx(od_truth, abs=1e-1)

    # Clean up CAL‑DB entry
    caldb.CalDB().remove_entry(nd_cal)

    print("ND‑filter E2E test passed")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-tvac", "--e2edata_dir", default=here)
    parser.add_argument("-o",    "--outputdir",   default=here)
    args = parser.parse_args()
    test_nd_filter_e2e(args.e2edata_dir)
