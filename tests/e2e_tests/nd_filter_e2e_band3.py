import os, shutil, glob, argparse
import pytest
from datetime import datetime, timedelta

import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.walker as walker
import corgidrp.nd_filter_calibration as nd_filter_calibration
from corgidrp import caldb
from corgidrp.check import generate_fits_excel_documentation
import time

# ----------------------------------------------------------------------
@pytest.mark.e2e
def test_nd_filter_e2e(e2edata_path, e2eoutput_path):
    # 1. Synthetic “dim star” frames (no ND)
    fwhm = 3  # pix PSF width
    # Test Band 3 imaging
    cfam_name = '3F'
    # Changing filter to 3F to reproduce imaging in band 3
    true_flux_dim = nd_filter_calibration.compute_expected_band_irradiance('TYC 4424-1286-1', cfam_name)
    cal_factor = true_flux_dim / 200  # erg / (e‑ s⁻¹)

    dim_frames = mocks.create_flux_image(
        true_flux_dim, fwhm, cal_factor, target_name='TYC 4424-1286-1'
    )
    dim_frames.ext_hdr['BUNIT'] = 'photoelectron'
    # Changes to test Band 3F
    # There's nothing specific to band 3 in the primary header
    # In the extension header, we can set
    dim_frames.ext_hdr['CFAMNAME']= cfam_name
    # These settings are related to imaging, which is what is tested now. They
    # do not need to be changed. No other settings in the extension header are
    # specific to band 3
    #FSAMNAME= 'R1C1'
    #LSAMNAME= 'NFOV'
    #FPAMNAME= 'HOLE'
    #SPAMNAME= 'OPEN
    dim_frames = [dim_frames] if not isinstance(dim_frames, list) else dim_frames

    # sleep for 2 seconds so the next file has a different timestamp
    time.sleep(2)

    # 2. Synthetic “bright star” frames (with ND)
    true_flux_bright = nd_filter_calibration.compute_expected_band_irradiance('Vega', cfam_name)
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
        # Changes to test Band 3F
        frame.ext_hdr['CFAMNAME']= cfam_name
        bright_frames.append(frame)

    # 3. Create output directory structure
    main_output_dir = os.path.join(e2eoutput_path, "nd_filter_cal_band3_e2e")
    shutil.rmtree(main_output_dir, ignore_errors=True)
    os.makedirs(main_output_dir)
    
    # Create input_data subfolder
    input_data_dir = os.path.join(main_output_dir, 'input_l3')
    os.makedirs(input_data_dir)

    # Save raw files with proper filename conventions
    base_time = datetime.now()
    visitid = "0200001999001000001"  # Use consistent visitid
    
    for i, frame in enumerate(dim_frames + bright_frames):
        # Generate unique timestamp for each frame
        unique_time = (base_time + timedelta(milliseconds=i*100)).strftime('%Y%m%dt%H%M%S%f')[:-5]
        frame.filename = f"cgi_{visitid}_{unique_time}_l3_.fits"
        frame.save(filedir=input_data_dir)

    filelist = sorted(glob.glob(os.path.join(input_data_dir, "*.fits")))

    # 4. Run the DRP walker with outputs saved in the main output directory
    # Remove old NDF cal files first
    for old_file in glob.glob(os.path.join(main_output_dir, "*ndf_cal.fits")):
        os.remove(old_file)
    walker.walk_corgidrp(filelist, "", main_output_dir)

    # 5. Load product & assert if calculated OD matches the input
    nd_file = glob.glob(os.path.join(main_output_dir, "*_ndf_cal*.fits"))
    nd_cal  = data.NDFilterSweetSpotDataset(nd_file[0])

    recovered_od = float(nd_cal.od_values[0])  # use the first entry for the check
    print("Calculated OD:", recovered_od)
    print("Input OD:", od_truth)
    assert recovered_od == pytest.approx(od_truth, abs=1e-1)

    # Clean up CAL‑DB entry
    caldb.CalDB().remove_entry(nd_cal)

    # Generate Excel documentation for the ND filter calibration product (Band 3)
    nd_filter_file = glob.glob(os.path.join(main_output_dir, "*_ndf_cal.fits"))[0]
    excel_output_path = os.path.join(main_output_dir, "ndf_cal_documentation.xlsx")
    generate_fits_excel_documentation(nd_filter_file, excel_output_path)
    print(f"Excel documentation generated: {excel_output_path}")

    print("ND‑filter E2E test passed")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-tvac", "--e2edata_dir", default=here)
    parser.add_argument("-o",    "--outputdir",   default=here)
    args = parser.parse_args()
    test_nd_filter_e2e(args.e2edata_dir, args.outputdir)
