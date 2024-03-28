import os
import glob
import pytest
import numpy as np
from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset
import corgidrp.l2a_to_l2b as l2a_to_l2b

def test_emgain_div():
    data = np.ones([1024,1024]) * 2 
    err = np.ones([1,1024,1024]) *0.5
    prhd, exthd = create_default_headers()
    exthd["CMDGAIN"] = 1000
    image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err)
    image2 = image1.copy()
    dataset=Dataset([image1, image2])
    
    ###### perform em_gain division
    gain_dataset = l2a_to_l2b.em_gain_division(dataset)
    assert("em_gain" in str(gain_dataset[0].ext_hdr["HISTORY"]))

    emgain = gain_dataset[0].ext_hdr['CMDGAIN']
    
    # check the level of the dataset
    #assert np.mean(gain_dataset.all_data) == pytest.approx(np.mean(dark_dataset.all_data)/emgain, abs=1e-3)
    #assert np.mean(gain_dataset.all_err) == pytest.approx(np.mean(dark_dataset.all_err)/emgain, abs=1e-3)
    assert gain_dataset[0].ext_hdr["BUNIT"] == "e/phot"
    assert gain_dataset[0].err_hdr["BUNIT"] == "e/phot"
  
    print(gain_dataset[0].ext_hdr)
    

if __name__ == "__main__":
    test_emgain_div()