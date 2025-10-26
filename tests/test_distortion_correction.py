import numpy as np
import pytest
import os
from corgidrp import data, mocks, astrom
from corgidrp.l3_to_l4 import distortion_correction

def test_distortion_correction():
    """
    Unit test of the distortion correction function. The test checks that a distorted mock image has been accurately corrected for by comparing it to an undistorted mock image.
    
    """

    # Create a bad pixel map for testing
    datashape = (1024, 1024)  # Example shape, adjust as needed
    
    bpixmap = np.zeros(datashape) # Assuming bad pixels have been corrected before this step.

    # create an undistorted dataset
    field_path = os.path.join(os.path.dirname(__file__), "test_data", "JWST_CALFIELD2020.csv")

    no_distortion_dataset = mocks.create_astrom_data(field_path, bpix_map=bpixmap, sim_err_map=True)
    
    # add distortion to the dataset
    distortion_coeffs_path = os.path.join(os.path.dirname(__file__), "test_data", "distortion_expected_coeffs.csv")
    distortion_coeffs = np.genfromtxt(distortion_coeffs_path)
   
    distortion_dataset = mocks.create_astrom_data(field_path, distortion_coeffs_path=distortion_coeffs_path, bpix_map=bpixmap, sim_err_map=True)
    
    # assume a ground truth AstrometricCalibration file for this dataset (zero offset from pointing, platescale=21.8, northangle=45)
    astromcal_data = np.concatenate((np.array([80.553428801, -69.514096821, 21.8, 45, 0, 0]), distortion_coeffs), axis=0)
    astrom_cal = data.AstrometricCalibration(astromcal_data, pri_hdr=distortion_dataset[0].pri_hdr, ext_hdr=distortion_dataset[0].ext_hdr, input_dataset=distortion_dataset)

    # use the distortion correction function to undistort
    undistorted_dataset = distortion_correction(distortion_dataset, astrom_cal)

    # compare the undistorted data to the original dataset with no distortion
    for frame, ref_frame in zip(undistorted_dataset, no_distortion_dataset):
        assert np.nanmean(frame.data - ref_frame.data) == pytest.approx(0, abs=0.005)
        assert np.sum(np.isnan(frame.data)) == np.sum(np.isnan(ref_frame.data)) # to ensure no new bad pixels
        assert np.all(frame.dq == ref_frame.dq) # to ensure the bad pixel map didn't change
        assert np.nanmean(frame.err - ref_frame.err) == pytest.approx(0, abs=0.00005)


def test_distortion_correction_pol():
    """
    Unit test of the distortion correction function for polarization data.
    Tests that the function can handle (2, 1024, 1024) input and return (2, 1024, 1024) output.
    """

    # Create a bad pixel map for testing
    datashape = (2, 1024, 1024)  # pol data shape
    bpixmap = np.zeros(datashape)

    ## CREATE UNDISTORTED POL DATASET
    field_path = os.path.join(os.path.dirname(__file__), "test_data", "JWST_CALFIELD2020.csv")
    no_distortion_dataset_base = mocks.create_astrom_data(field_path, bpix_map=bpixmap[0], sim_err_map=True)

    # Stack to create (2, 1024, 1024)
    no_distortion_dataset_base.frames[0].data = np.stack([no_distortion_dataset_base.frames[0].data,
                                                          no_distortion_dataset_base.frames[0].data])
    original_err = no_distortion_dataset_base.frames[0].err
    no_distortion_dataset_base.frames[0].err = np.stack([original_err, original_err], axis=1)
    no_distortion_dataset_base.frames[0].dq = np.stack([no_distortion_dataset_base.frames[0].dq,
                                                        no_distortion_dataset_base.frames[0].dq])
    no_distortion_dataset_base.all_data = np.array([frame.data for frame in no_distortion_dataset_base.frames])
    no_distortion_dataset_base.all_err = np.array([frame.err for frame in no_distortion_dataset_base.frames])
    no_distortion_dataset_base.all_dq = np.array([frame.dq for frame in no_distortion_dataset_base.frames])

    # CREATE DISTORTED POL DATASET
    distortion_coeffs_path = os.path.join(os.path.dirname(__file__), "test_data", "distortion_expected_coeffs.csv")
    distortion_coeffs = np.genfromtxt(distortion_coeffs_path)
    distortion_dataset_base = mocks.create_astrom_data(field_path, distortion_coeffs_path=distortion_coeffs_path,
                                                       bpix_map=bpixmap[0], sim_err_map=True)

    # Stack to create (2, 1024, 1024)
    distortion_dataset_base.frames[0].data = np.stack([distortion_dataset_base.frames[0].data,
                                                       distortion_dataset_base.frames[0].data])
    original_err = distortion_dataset_base.frames[0].err
    distortion_dataset_base.frames[0].err = np.stack([original_err, original_err], axis=1)
    distortion_dataset_base.frames[0].dq = np.stack([distortion_dataset_base.frames[0].dq,
                                                     distortion_dataset_base.frames[0].dq])
    distortion_dataset_base.all_data = np.array([frame.data for frame in distortion_dataset_base.frames])
    distortion_dataset_base.all_err = np.array([frame.err for frame in distortion_dataset_base.frames])
    distortion_dataset_base.all_dq = np.array([frame.dq for frame in distortion_dataset_base.frames])

    # VERIFY INPUT SHAPE
    assert distortion_dataset_base.frames[0].data.shape == (2, 1024, 1024)
    # Create astrom cal
    astromcal_data = np.concatenate((np.array([80.553428801, -69.514096821, 21.8, 45, 0, 0]), distortion_coeffs),
                                    axis=0)
    astrom_cal = data.AstrometricCalibration(astromcal_data,
                                             pri_hdr=distortion_dataset_base[0].pri_hdr,
                                             ext_hdr=distortion_dataset_base[0].ext_hdr,
                                             input_dataset=distortion_dataset_base)

    # CALL DISTORTION CORRECTION WITH (2, 1024, 1024) DATA
    undistorted_dataset = distortion_correction(distortion_dataset_base, astrom_cal)
    # VERIFY OUTPUT SHAPE
    assert undistorted_dataset.frames[0].data.shape == (2, 1024, 1024), \
        f"Expected output shape (2, 1024, 1024), got {undistorted_dataset.frames[0].data.shape}"

    # NOW VERIFY EACH POL MODE IS BEING PROCESSED CORRECTED BY FUNCTION
    for pol_idx in range(2):
        # Data comparison
        undistorted_pol = undistorted_dataset.frames[0].data[pol_idx]
        reference_pol = no_distortion_dataset_base.frames[0].data[pol_idx]
        diff_mean = np.nanmean(undistorted_pol - reference_pol)
        assert diff_mean == pytest.approx(0, abs=0.005)  # mean difference should be ~0
        assert np.sum(np.isnan(undistorted_pol)) == np.sum(np.isnan(reference_pol))  # no new bad pixels
        # DQ comparison
        undistorted_dq = undistorted_dataset.frames[0].dq[pol_idx]
        reference_dq = no_distortion_dataset_base.frames[0].dq[pol_idx]
        assert np.all(undistorted_dq == reference_dq)  # bad pixel map shouldn't change
        # Error comparison
        undistorted_err = undistorted_dataset.frames[0].err[:, pol_idx]
        reference_err = no_distortion_dataset_base.frames[0].err[:, pol_idx]
        err_diff_mean = np.nanmean(undistorted_err - reference_err)
        assert err_diff_mean == pytest.approx(0, abs=0.00005)  # error mean difference should be ~0


if __name__ == "__main__":
    test_distortion_correction()
    test_distortion_correction_pol()
