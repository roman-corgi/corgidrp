import os
import numpy as np
from astropy.io import fits
from corgidrp.data import Dataset, SpectroscopyCentroidPSF
import corgidrp.spec as steps
from corgidrp.mocks import create_default_headers

def test_psf_centroid():
    """
    Test PSF centroid computation with mock data and validate results
    """
    # Define file path to test input
    file_path = os.path.abspath("tests/test_data/spectroscopy/g0v_vmag6_spc-spec_band3_unocc_CFAM3d_NOSLIT_PRISM3_offset_array.fits")
    print(f"Attempting to load FITS file: {file_path}")

    # Load FITS data and table (contains true values for testing)
    with fits.open(file_path) as hdul:
        psf_array = hdul[0].data
        table_data = hdul[1].data
        print(f"FITS file loaded. PSF Array shape: {psf_array.shape}")
        print(f"Metadata rows: {len(table_data)}")

    # Convert table data to dictionary for initial_cent
    initial_cent = {
        "xcent": np.array(table_data["xcent"]),
        "ycent": np.array(table_data["ycent"])
    }

    # Create Dataset object for traceability
    dataset = Dataset([file_path])

    # Ensure output directory exists
    output_dir = "calibrations"
    os.makedirs(output_dir, exist_ok=True)

    # Create mock headers for testing
    pri_hdr, ext_hdr = create_default_headers()

    # Run the step function to produce the calibration output
    print("Running the PSF Centroid Step...")
    calibration = steps.compute_psf_centroid(
        psf_array=psf_array,
        initial_cent=initial_cent,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        input_dataset=dataset,
        output_path=output_dir,  
        verbose=True
    )

    # Validation section
    print("\nValidating PSF Centroid Accuracy...")

    # Get fitted centroids from calibration
    xfit = calibration.xfit
    yfit = calibration.yfit

    # Get true centroids from test data
    xtrue = np.array(initial_cent["xcent"])
    ytrue = np.array(initial_cent["ycent"])

    x_error = xfit - xtrue
    y_error = yfit - ytrue

    print(f"Mean centroid error (x, y): {np.mean(x_error):.3e}, {np.mean(y_error):.3e} pixels")
    print(f"Std dev of centroid error (x, y): {np.std(x_error):.2E}, {np.std(y_error):.2E} pixels")

    assert np.abs(np.mean(x_error)) < 1e-3, "X centroid mean error too large"
    assert np.abs(np.mean(y_error)) < 1e-3, "Y centroid mean error too large"
    assert np.std(x_error) < 1e-2, "X centroid error spread too large"
    assert np.std(y_error) < 1e-2, "Y centroid error spread too large"

    print("PSF centroid fit accuracy test passed.")

    calibration_path = os.path.join(output_dir, calibration.filename)
    assert os.path.exists(calibration_path), "Calibration file not created"

    # Test loading the calibration file
    calibration_dataset = Dataset([calibration_path])
    print(f"\nSuccessfully loaded the calibration FITS as a Dataset.")
    print(f"Number of frames in dataset: {len(calibration_dataset)}")

    with fits.open(calibration_path) as hdul:
        hdul.info()
        print("\nPrimary Header:")
        print(repr(hdul[0].header))
        if len(hdul) > 1:
            print("\nExtension Header:")
            print(repr(hdul[1].header))
            print("\nData (first 5 rows):")
            print(hdul[1].data[:5])

if __name__ == "__main__":
    test_psf_centroid()
