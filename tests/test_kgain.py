#A file to test the kgain conversion
import os
import numpy as np
from corgidrp.mocks import create_default_headers
import corgidrp.data as data
import corgidrp.l2a_to_l2b as l2a_to_l2b
import pytest

def test_kgain():
    """
    test the KGain class and the calibration file and the unit conversion
    """
    #test KGain class and cal file
    prhd, exthd = create_default_headers()
    dat = np.ones([1024,1024]) * 2 
    err = np.ones([1,1024,1024]) * 0.5
    image1 = data.Image(dat,pri_hdr = prhd, ext_hdr = exthd, err = err)
    image2 = image1.copy()
    image1.filename = "test1"
    image2.filename = "test2"
    dataset= data.Dataset([image1, image2])
    
    gain_value = np.array([[9.55]])
    kgain = data.KGain(gain_value, pri_hdr = prhd, ext_hdr = exthd, input_dataset = dataset)
    assert kgain.value == gain_value[0,0]
    assert kgain.data[0,0] == gain_value[0,0]

    # save KGain
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    kgain_filename = "kgain_calib.fits"
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    kgain.save(filedir=calibdir, filename=kgain_filename)
        
    kgain_filepath = os.path.join(calibdir, kgain_filename)
    kgain_open = data.KGain(kgain_filepath)
    assert kgain_open.value == gain_value[0,0]
        
    kgain_copy = kgain_open.copy()
    assert kgain_copy.value == gain_value[0,0]
    kgain_copy = kgain_open.copy(copy_data = False)
    assert kgain_copy.value == gain_value[0,0]
    
    # test convert_to_electrons
    k_gain = kgain.value
    
    gain_dataset = l2a_to_l2b.convert_to_electrons(dataset, kgain)
    
    assert np.mean(gain_dataset[0].data) == pytest.approx(k_gain * np.mean(dataset[0].data), abs = 1e-4)
    assert np.mean(gain_dataset[0].err) == pytest.approx(k_gain * np.mean(dataset[0].err), abs = 1e-4)
    
    #test header updates
    assert gain_dataset[0].ext_hdr["BUNIT"] == "detected EM electrons"
    assert gain_dataset[0].err_hdr["BUNIT"] == "detected EM electrons"
    assert gain_dataset[0].ext_hdr["KGAIN"] == k_gain
    assert gain_dataset[0].err_hdr["KGAIN"] == k_gain
    assert("converted" in str(gain_dataset[0].ext_hdr["HISTORY"]))
    
if __name__ == "__main__":
    test_kgain()
