#A file to test the kgain conversion
import os
import pickle
import numpy as np
from corgidrp.mocks import create_default_L1_headers
import corgidrp.data as data
import corgidrp.l2a_to_l2b as l2a_to_l2b
import pytest
import astropy.io.fits as fits

def test_kgain():
    """
    test the KGain class and the calibration file and the unit conversion
    """
    #test KGain class and cal file
    prhd, exthd = create_default_L1_headers()
    dat = np.ones([1024,1024]) * 2
    err = np.ones([1,1024,1024]) * 0.5
    ptc = np.ones([2,1024])
    exthd["RN"] = 100.0
    exthd["RN_ERR"] = 2
    ptc_hdr = fits.Header()
    image1 = data.Image(dat,pri_hdr = prhd, ext_hdr = exthd, err = err)
    image2 = image1.copy()
    image1.filename = "test1_l1_.fits"
    image2.filename = "test2_l1_.fits"
    dataset= data.Dataset([image1, image2])

    prhd_kgain = prhd.copy()
    exthd_kgain = exthd.copy()
    gain_value = 9.55
    gain_err = 1.
    #int input is not allowed
    gain_value_int = 9
    gain_err_int = 1
    with pytest.raises(ValueError):
        kgain_int = data.KGain(gain_value_int, pri_hdr = prhd_kgain, ext_hdr = exthd_kgain, input_dataset = dataset)
    with pytest.raises(ValueError):
        kgain_int = data.KGain(gain_value, err = gain_err_int, pri_hdr = prhd_kgain, ext_hdr = exthd_kgain, input_dataset = dataset)   
        
    kgain = data.KGain(gain_value, pri_hdr = prhd_kgain, ext_hdr = exthd_kgain, input_dataset = dataset)

    assert kgain.filename.split(".")[0] == "test2_krn_cal"
    assert kgain.value == gain_value
    assert kgain.data[0] == gain_value
    
    #test ptc and error extension
    kgain_ptc = data.KGain(gain_value, err = gain_err, ptc = ptc, pri_hdr = prhd_kgain, ext_hdr = exthd_kgain, ptc_hdr = ptc_hdr, input_dataset = dataset)
    assert kgain_ptc.error == gain_err
    assert kgain_ptc.ptc[0,0] == 1.
    assert kgain_ptc.ptc_hdr is not None
    
    # check the kgain can be pickled (for CTC operations)
    pickled = pickle.dumps(kgain_ptc)
    pickled_kgain = pickle.loads(pickled)
    assert np.all((kgain_ptc.data == pickled_kgain.data))

    #test copy and save
    kgain_ptc_copy = kgain_ptc.copy(copy_data = False)
    assert kgain_ptc_copy.value == gain_value
    kgain_ptc_copy = kgain_ptc.copy()
    assert kgain_ptc_copy.value == gain_value
    
    assert kgain_ptc_copy.ptc[0,0] == 1.
    assert kgain_ptc_copy.error == gain_err
    assert kgain_ptc_copy.ptc_hdr is not None
    assert kgain_ptc_copy.err_hdr is not None
    
    # save KGain
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    kgain_filename = "kgain_calib.fits"
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    kgain_ptc.save(filedir=calibdir, filename=kgain_filename)        
        
    kgain_filepath = os.path.join(calibdir, kgain_filename)
    kgain_open = data.KGain(kgain_filepath)
    # use isclose to deal with 64 bit vs 32 bit precision 
    assert np.isclose(kgain_open.value, gain_value, rtol=1e-6)
    assert np.isclose(kgain_open.error, gain_err, rtol=1e-6)
    assert np.isclose(kgain_open.ptc[0,0], 1., rtol=1e-6)
    assert kgain_open.ptc_hdr["EXTNAME"] == "PTC"
    assert kgain_open.err_hdr is not None
    
    # check the kgain can be pickled (for CTC operations)
    pickled = pickle.dumps(kgain_open)
    pickled_kgain = pickle.loads(pickled)
    assert np.all((kgain_open.data == pickled_kgain.data))

    # test convert_to_electrons
    k_gain = kgain.value

    gain_dataset = l2a_to_l2b.convert_to_electrons(dataset, kgain)

    assert np.mean(gain_dataset[0].data) == pytest.approx(k_gain * np.mean(dataset[0].data), abs = 1e-4)
    assert np.mean(gain_dataset[0].err) == pytest.approx(k_gain * np.mean(dataset[0].err), abs = 1e-4)

    # test error propagation
    assert gain_dataset[0].err[0,10,10] == np.sqrt(dataset[0].err[0,10,10]**2 * k_gain**2 + kgain.err**2 * gain_dataset[0].data[10,10]**2)
     
    #test header updates
    assert gain_dataset[0].ext_hdr["BUNIT"] == "detected EM electron"
    assert gain_dataset[0].err_hdr["BUNIT"] == "detected EM electron"
    assert gain_dataset[0].ext_hdr["KGAINPAR"] == k_gain
    assert gain_dataset[0].ext_hdr["KGAIN_ER"] == kgain.error
    assert gain_dataset[0].ext_hdr["RN"] > 0
    assert gain_dataset[0].ext_hdr["RN_ERR"] > 0
    assert("converted" in str(gain_dataset[0].ext_hdr["HISTORY"]))

if __name__ == "__main__":
    test_kgain()
