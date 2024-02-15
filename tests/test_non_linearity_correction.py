import os
import glob
import numpy as np
import corgidrp.mocks as mocks
import corgidrp.data as data
import corgidrp.l1_to_l2a as l1_to_l2a

def test_non_linearity_correction():
    """
    Generate a non-linearity correction calibration and test the correction 
    
    Ported from II&T Pipeline
    """
    ###### create a simulated dataset that is non-linear
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    
    mocks.create_nonlinear_dataset(filedir=datadir)

    ####### open up the files
    sim_data_filenames = glob.glob(os.path.join(datadir, "simcal_nonlin*.fits"))
    nonlinear_dataset = data.Dataset(sim_data_filenames)
    assert len(nonlinear_dataset) == 2

    ######## perform non-linearity correction
    non_linearity_correction = data.NonLinearityCalibration(os.path.join(os.path.dirname(__file__),'nonlin_sample.fits'))
    linear_dataset = l1_to_l2a.correct_nonlinearity(nonlinear_dataset, non_linearity_correction)

    #The data was generated with a ramp in the x-direction going from 10 to 65536
    expected_ramp = np.linspace(10,65536,1024)
    #Let's collapse the data and see if there's a ramp. 
    collapsed_data = np.mean(linear_dataset.all_data, axis=(0,1))

    #Relative correction
    relative_correction = (collapsed_data-expected_ramp)/collapsed_data

    #We are happy if the relative correction is less than 1% [TBC]
    assert np.all(relative_correction < 1e-2)

if __name__ == "__main__":
    test_non_linearity_correction