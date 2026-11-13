import os
import pickle
import numpy as np
import astropy.time as time
import corgidrp
import corgidrp.data as data


expected_rowreadtime = 0.0002235
expected_fwc_em = 100000.
expected_fwc_pp = 90000.
expected_kgain = 8.7

def test_default_params():
    """
    Tests creation of the default file
    """
    default_detparams = data.DetectorParams({}, date_valid=time.Time("2023-11-01 00:00:00", scale='utc'))

    assert default_detparams.params['ROWREADT'] == expected_rowreadtime
    assert default_detparams.params['FWC_EM_E'] == expected_fwc_em
    assert default_detparams.params['FWC_PP_E'] == expected_fwc_pp
    assert default_detparams.params['KGAINPAR'] == expected_kgain

def test_hashing():
    """
    Tests the hash function produces the expected results
    """
    default_detparams = data.DetectorParams({}, date_valid=time.Time("2023-11-01 00:00:00", scale='utc'))
    default_detparams_2 = data.DetectorParams({}, date_valid=time.Time("2023-11-01 00:00:00", scale='utc'))

    assert default_detparams.get_hash() == default_detparams_2.get_hash()

    new_detparams = data.DetectorParams({'FWC_EM_E' : 200000}, date_valid=time.Time("2023-11-01 00:00:00", scale='utc'))
    assert default_detparams.get_hash() != new_detparams.get_hash()

def test_pickling():
    """
    Test detector params can be pickled
    """
    filename = os.path.join(corgidrp.default_cal_dir, "DetectorParams_2023-11-01T00.00.00.000.fits")
    default_detparams = data.DetectorParams(filename)
        
    # check they can be pickled (for CTC operations)
    pickled = pickle.dumps(default_detparams)
    pickled_detparams = pickle.loads(pickled)
    assert np.all((default_detparams.data == pickled_detparams.data))

    if __name__ == '__main__':
        test_default_params()
        test_hashing()
        test_pickling()