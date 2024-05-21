from astropy.time import Time
from corgidrp.detector import get_rowreadtime_sec

# Value of readrowtime
rowreadtime_sec = 0.0002235

def test_rowreadtime():
    # Default datetime
    rowreadtime_sec_test = get_rowreadtime_sec()
    
    assert rowreadtime_sec == rowreadtime_sec_test

    # Another valid datetime
    rowreadtime_sec_test = get_rowreadtime_sec(datetime=Time('2024-05-16 00:00:00',
        scale='utc'))

    assert rowreadtime_sec == rowreadtime_sec_test

if __name__ == '__main__':
    test_rowreadtime()

