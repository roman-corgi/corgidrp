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
    
    gain_value = np.array([[9.55]])
    kgain = data.KGain(gain_value, pri_hdr = prhd, ext_hdr = exthd)
    kgain.save("kgain.fits")
    assert kgain.value == gain_value[0,0]
    assert kgain.data[0,0] == gain_value[0,0]
    
    kgain_open = data.KGain("kgain.fits")
    assert kgain_open.value == gain_value[0,0]
    os.remove('kgain.fits')
    
    kgain_copy = kgain_open.copy()
        
    assert kgain_copy.value == gain_value[0,0]
 
    # test convert_to_electrons
    image1 = data.Image(dat,pri_hdr = prhd, ext_hdr = exthd, err = err)
    image2 = image1.copy()
    dataset= data.Dataset([image1, image2])
    
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