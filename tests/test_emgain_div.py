import pytest
import numpy as np
from corgidrp.mocks import create_default_L1_headers
from corgidrp.data import Image, Dataset
import corgidrp.l2a_to_l2b as l2a_to_l2b

def test_emgain_div():
    data = np.ones([1024,1024]) * 2 
    err = np.ones([1,1024,1024]) *0.5
    prhd, exthd = create_default_L1_headers()
    exthd["EMGAIN_C"] = 1000
    #kgain conversion must have been done already
    exthd["BUNIT"] = "detected EM electron"
    image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err)
    image2 = image1.copy()
    dataset=Dataset([image1, image2])
    
    ###### perform em_gain division
    gain_dataset = l2a_to_l2b.em_gain_division(dataset)
    assert("same" in str(gain_dataset[0].ext_hdr["HISTORY"]))

    emgain = gain_dataset[0].ext_hdr['EMGAIN_C']
    
    # check the level of the dataset
    assert np.mean(gain_dataset.all_data) == pytest.approx(np.mean(dataset.all_data)/emgain, abs=1e-3)
    assert np.mean(gain_dataset.all_err) == pytest.approx(np.mean(dataset.all_err)/emgain, abs=1e-3)
    assert gain_dataset[0].ext_hdr["BUNIT"] == "detected electron"
    assert gain_dataset[0].err_hdr["BUNIT"] == "detected electron"
    
    # check non-unique emgain
    emgain1 = 100
    dataset[1].ext_hdr['EMGAIN_C'] = emgain1
    gain_dataset = l2a_to_l2b.em_gain_division(dataset)
    assert("different" in str(gain_dataset[0].ext_hdr["HISTORY"]))
    assert np.mean(gain_dataset.all_data[0]) == pytest.approx(np.mean(dataset.all_data[0])/emgain, abs=1e-3)
    assert np.mean(gain_dataset.all_data[1]) == pytest.approx(np.mean(dataset.all_data[1])/emgain1, abs=1e-3)
    assert np.mean(gain_dataset.all_err[0]) == pytest.approx(np.mean(dataset.all_err[0])/emgain, abs=1e-3)
    assert np.mean(gain_dataset.all_err[1]) == pytest.approx(np.mean(dataset.all_err[1])/emgain1, abs=1e-3)
    

if __name__ == "__main__":
    test_emgain_div()