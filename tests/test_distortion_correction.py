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



if __name__ == "__main__":
    test_distortion_correction()