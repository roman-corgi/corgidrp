#%%
import os
import glob
import pytest
import numpy as np
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.detector as detector
import corgidrp.l2b_to_l3 as l2b_to_l3

#%%

def test_exptime_normalization():
    """
    Generate mock input data and pass into exposure_time_normalization function
    """
    ###### create simulated data
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    
    #create simulated data
    not_normalized_dataset = mocks.create_not_normalized_dataset(filedir=datadir)

    # extract the exposure time from each Image and check that it is >0
    exposure_times = np.zeros(len(not_normalized_dataset.frames))
    for i in range(len(not_normalized_dataset.frames)):
        assert not_normalized_dataset.frames[i].ext_hdr['EXPTIME'] > 0
        exposure_times[i] = not_normalized_dataset.frames[i].ext_hdr['EXPTIME']

    # divide the simulated data by the exposure time
    norm_dataset = l2b_to_l3.divide_by_exptime(not_normalized_dataset)

    # test if the function works
    for i in range(len(not_normalized_dataset.frames)):

        # test that the unit in the header is in electrons/s
        assert norm_dataset.frames[i].ext_hdr['BUNIT'] == "photoelectron/s"

        #check that the quality flag has exists and it's 1 everywhere
        assert np.mean(not_normalized_dataset.frames[i].dq) == pytest.approx(0, abs=1e-6)

        # check that, for each frame, if you multiply the output by the exposure time you can recover the input
        assert not_normalized_dataset.frames[i].data == pytest.approx(exposure_times[i] * norm_dataset.frames[i].data, abs=1e-5)
        assert not_normalized_dataset.frames[i].err == pytest.approx(exposure_times[i] * norm_dataset.frames[i].err, abs=1e-5)

        # check that the output in .frames.data and .frames.err correspond to .all_data and .all_err
        assert norm_dataset.frames[i].data == pytest.approx(norm_dataset.all_data[i,:,:], abs = 1e-5)
        assert norm_dataset.frames[i].err == pytest.approx(norm_dataset.all_err[i,:,:], abs = 1e-5)

#%%    

if __name__ == "__main__":
    test_exptime_normalization()
# %%
