import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from corgidrp.data import Dataset, SpectroscopyCentroidPSF, Image
import corgidrp.spec as steps
from corgidrp.mocks import create_default_headers

def test_psf_centroid():
    """
    Test PSF centroid computation with mock data and assert correctness of output FITS structure.
    """
    datadir = os.path.join(os.path.dirname(__file__), "test_data", "spectroscopy")
    file_path = os.path.join(datadir, "g0v_vmag6_spc-spec_band3_unocc_CFAM3d_NOSLIT_PRISM3_offset_array.fits")
    assert os.path.exists(file_path), f"Test FITS file not found: {file_path}"

    with fits.open(file_path) as hdul:
        psf_array = hdul[0].data
        psf_table = Table(hdul[1].data)
        pri_hdr = hdul[0].header
        ext_hdr = hdul[1].header

    assert psf_array.ndim == 3, "Expected 3D PSF array"
    assert "xcent" in psf_table.colnames and "ycent" in psf_table.colnames, "Missing centroid columns"

    initial_cent = {
        "xcent": np.array(psf_table["xcent"]),
        "ycent": np.array(psf_table["ycent"])
    }

    psf_images = []
    for i in range(psf_array.shape[0]):
        data_2d = np.copy(psf_array[i])
        err = np.zeros_like(data_2d)
        dq = np.zeros_like(data_2d, dtype=int)
        image = Image(
            data_or_filepath=data_2d,
            pri_hdr=pri_hdr.copy(),
            ext_hdr=ext_hdr.copy(),
            err=err,
            dq=dq
        )
        psf_images.append(image)

    dataset = Dataset(psf_images)

    output_dir = os.path.join(os.path.dirname(__file__), "calibrations")
    os.makedirs(output_dir, exist_ok=True)

    calibration = steps.compute_psf_centroid(
        dataset=dataset,
        initial_cent=initial_cent,
        verbose=False
    )

    # Manually assign filedir and filename before saving
    calibration.filename = "centroid_calibration.fits"
    calibration.filedir = output_dir
    calibration.save()

    output_file = os.path.join(output_dir, calibration.filename)
    assert os.path.exists(output_file), "Calibration file not created"

    # Validate calibration file structure and contents
    with fits.open(output_file) as hdul:
        assert len(hdul) >= 2, "Expected at least 2 HDUs (primary + extension)"
        assert isinstance(hdul[0].header, fits.Header), "Missing primary header"
        assert isinstance(hdul[1].header, fits.Header), "Missing extension header"
        assert "EXTNAME" in hdul[1].header, "Extension header missing EXTNAME"
        assert hdul[1].header["EXTNAME"] == "CENTROIDS", "EXTNAME should be 'CENTROIDS'"

        centroid_data = hdul[1].data
        assert centroid_data is not None, "Centroid data missing"
        assert centroid_data.shape[1] == 2, "Centroid data should be shape (N, 2)"

        print(f"Centroid FITS file validated: {centroid_data.shape[0]} rows")

if __name__ == "__main__":
    test_psf_centroid()
