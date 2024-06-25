import astropy.time as time
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

    assert default_detparams.params['rowreadtime'] == expected_rowreadtime
    assert default_detparams.params['fwc_em'] == expected_fwc_em
    assert default_detparams.params['fwc_pp'] == expected_fwc_pp
    assert default_detparams.params['kgain'] == expected_kgain

def test_hashing():
    """
    Tests the hash function produces the expected results
    """
    default_detparams = data.DetectorParams({}, date_valid=time.Time("2023-11-01 00:00:00", scale='utc'))
    default_detparams_2 = data.DetectorParams({}, date_valid=time.Time("2023-11-01 00:00:00", scale='utc'))

    assert default_detparams.get_hash() == default_detparams_2.get_hash()

    new_detparams = data.DetectorParams({'fwc_em' : 200000}, date_valid=time.Time("2023-11-01 00:00:00", scale='utc'))
    assert default_detparams.get_hash() != new_detparams.get_hash()

    