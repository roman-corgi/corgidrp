import os
import pickle
import pytest
import numpy as np
import astropy.io.fits as fits
import corgidrp
from corgidrp.data import Image, Dataset
from corgidrp.mocks import create_default_L1_headers, create_dark_calib_files

np.random.seed(123)

data = np.ones([1024,1024]) * 2
err = np.zeros([1024,1024])
err1 = np.ones([1024,1024])
err2 = err1.copy()
err3 = np.ones([1,1024,1024]) * 0.5
dq = np.zeros([1024,1024], dtype = int)
dq1 = dq.copy()
dq1[0,0] = 1
prhd, exthd = create_default_L1_headers()
errhd = fits.Header()
errhd["CASE"] = "test"
dqhd = fits.Header()
dqhd["CASE"] = "test"


def test_hashing():
    """
    Test hashing works on data, err, and dq at the same time
    Two images with same data should be the same
    """
    # identical images should get the same hash
    image1 = Image(data, err = err, dq = dq, pri_hdr = prhd, ext_hdr = exthd)
    image2 = Image(np.copy(data), err = np.copy(err), dq = np.copy(dq), pri_hdr = prhd, ext_hdr = exthd)

    assert image1.get_hash() == image2.get_hash()

    # modifying the data should result in different hashes
    image2.data += 1

    assert image1.get_hash() != image2.get_hash()

    # take image 2 and modify the error. should be different hash from before
    old_hash = image2.get_hash()

    image2.err += 1
    assert old_hash != image2.get_hash()

    # take image 2 and modify the dq frame. should be different hash from before
    old_hash = image2.get_hash()

    image2.dq[0] = 1
    assert old_hash != image2.get_hash()

def test_split_dataset():
    """
    Test splitting dataset into sub datasets based on header keywords
    """
    image1 = Image(np.copy(data), err = np.copy(err), dq = np.copy(dq), pri_hdr = prhd.copy(), ext_hdr = exthd.copy())
    image2 = Image(np.copy(data), err = np.copy(err), dq = np.copy(dq), pri_hdr = prhd.copy(), ext_hdr = exthd.copy())
    image3 = Image(np.copy(data), err = np.copy(err), dq = np.copy(dq), pri_hdr = prhd.copy(), ext_hdr = exthd.copy())
    image4 = Image(np.copy(data), err = np.copy(err), dq = np.copy(dq), pri_hdr = prhd.copy(), ext_hdr = exthd.copy())

    orig_dataset = Dataset([image1, image2, image3, image4])

    # defaults
    # exthdr['EXPTIME'] = 60.0
    # prihdr['OBSID'] = 0

    ## slice it into 2
    image1.pri_hdr['OBSNUM'] = 0
    image1.ext_hdr['EXPTIME'] = 60.0

    image2.pri_hdr['OBSNUM'] = 1
    image2.ext_hdr['EXPTIME'] = 120.

    image3.pri_hdr['OBSNUM'] = 0
    image3.ext_hdr['EXPTIME'] = 60.0

    image4.pri_hdr['OBSNUM'] = 1
    image4.ext_hdr['EXPTIME'] = 120.

    sliced_datasets, unique_combos = orig_dataset.split_dataset(exthdr_keywords=['EXPTIME',], prihdr_keywords=['OBSNUM',])
    assert len(sliced_datasets) == 2

    sliced_datasets, unique_combos = orig_dataset.split_dataset(exthdr_keywords=['EXPTIME',])
    assert len(sliced_datasets) == 2

    sliced_datasets, unique_combos = orig_dataset.split_dataset(prihdr_keywords=['OBSNUM',])
    assert len(sliced_datasets) == 2

    ## slice it into 3
    image1.pri_hdr['OBSNUM'] = 0
    image1.ext_hdr['EXPTIME'] = 60.0

    image2.pri_hdr['OBSNUM'] = 1
    image2.ext_hdr['EXPTIME'] = 60.0

    image3.pri_hdr['OBSNUM'] = 0
    image3.ext_hdr['EXPTIME'] = 60.0

    image4.pri_hdr['OBSNUM'] = 1
    image4.ext_hdr['EXPTIME'] = 120.


    sliced_datasets, unique_combos = orig_dataset.split_dataset(exthdr_keywords=['EXPTIME',], prihdr_keywords=['OBSNUM',])
    assert len(sliced_datasets) == 3

    ## slice it into 4
    image1.pri_hdr['OBSNUM'] = 0
    image1.ext_hdr['EXPTIME'] = 60.0

    image2.pri_hdr['OBSNUM'] = 1
    image2.ext_hdr['EXPTIME'] = 60.0

    image3.pri_hdr['OBSNUM'] = 0
    image3.ext_hdr['EXPTIME'] = 120.

    image4.pri_hdr['OBSNUM'] = 1
    image4.ext_hdr['EXPTIME'] = 120.

    sliced_datasets, unique_combos = orig_dataset.split_dataset(exthdr_keywords=['EXPTIME',], prihdr_keywords=['OBSNUM',])
    assert len(sliced_datasets) == 4

    sliced_datasets, unique_combos = orig_dataset.split_dataset(exthdr_keywords=['EXPTIME',])
    assert len(sliced_datasets) == 2



def test_pickling():
    """
    Test that datasets and images can be pickled
    """
    ###### create simulated data
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    ####### test data architecture
    dark_dataset = create_dark_calib_files(filedir=datadir)

    pickle_filename = os.path.join(datadir, "simcal_dataset.pkl")
    pickled = pickle.dumps(dark_dataset)
    pickled_dark_dataset = pickle.loads(pickled)

    assert np.all(dark_dataset[0].data == pickled_dark_dataset[0].data)


if __name__ == "__main__":
    test_hashing()
    test_split_dataset()
    test_pickling()