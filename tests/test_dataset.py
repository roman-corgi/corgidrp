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


def test_split_dataset_tolerances():
    """
    Test splitting a dataset on a keyword that drifts, such as the FSM dither position, by
    passing a tolerance instead of requiring the values to match exactly
    """
    images = [Image(np.copy(data), err=np.copy(err), dq=np.copy(dq),
                    pri_hdr=prhd.copy(), ext_hdr=exthd.copy()) for _ in range(6)]
    orig_dataset = Dataset(images)

    # Two commanded dither positions, each reported with a little drift, and two targets
    # observed at both of them
    fsmx_vals = [-20.3, 19.8, -19.6, 20.4, -20.1, 20.1]
    fsmy_vals = [20.2, -19.7, 19.6, -20.4, 20.1, -20.2]
    targets = ['star1', 'star1', 'star2', 'star2', 'star3', 'star3']
    for image, fsmx, fsmy, target in zip(images, fsmx_vals, fsmy_vals, targets):
        image.ext_hdr['FSMX'] = fsmx
        image.ext_hdr['FSMY'] = fsmy
        image.pri_hdr['TARGET'] = target

    # Without a tolerance every frame has its own unique position, so nothing groups
    sliced_datasets, _ = orig_dataset.split_dataset(exthdr_keywords=['FSMX', 'FSMY'])
    assert len(sliced_datasets) == 6

    # With a tolerance wider than the drift but narrower than the dither throw, the frames
    # group into the two dither positions
    sliced_datasets, unique_combos = orig_dataset.split_dataset(
        exthdr_keywords=['FSMX', 'FSMY'], tolerances={'FSMX': 5., 'FSMY': 5.})
    assert len(sliced_datasets) == 2
    assert all(len(sub_dataset) == 3 for sub_dataset in sliced_datasets)

    # The unique values reported for a keyword with a tolerance are each group's mean
    assert sorted(round(float(combo[0]), 6) for combo in unique_combos) == \
        [round(np.mean([-20.3, -19.6, -20.1]), 6), round(np.mean([19.8, 20.4, 20.1]), 6)]

    # Splitting on the dither position and the target together separates every frame again,
    # and a tolerance on one keyword leaves the others matching exactly
    sliced_datasets, _ = orig_dataset.split_dataset(
        prihdr_keywords=['TARGET'], exthdr_keywords=['FSMX', 'FSMY'],
        tolerances={'FSMX': 5., 'FSMY': 5.})
    assert len(sliced_datasets) == 6

    # The frames keep their own reported positions, they are only grouped by the representative
    assert [image.ext_hdr['FSMX'] for image in orig_dataset] == fsmx_vals
    assert [image.ext_hdr['FSMY'] for image in orig_dataset] == fsmy_vals

    # A tolerance on a keyword that is not being split on is a mistake, not a no-op
    with pytest.raises(ValueError):
        orig_dataset.split_dataset(exthdr_keywords=['FSMX'], tolerances={'FSMY': 5.})

    # Single linkage: values that each fall within the tolerance of the next form one group,
    # even though the ends of the run are further apart than the tolerance. 
    # Not the ideal behavior, but OK for things that drift about a center value. 
    for image, fsmx in zip(images, [0., 4., 8., 12., 16., 20.]):
        image.ext_hdr['FSMX'] = fsmx
    sliced_datasets, _ = orig_dataset.split_dataset(exthdr_keywords=['FSMX'],
                                                    tolerances={'FSMX': 5.})
    assert len(sliced_datasets) == 1


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
    test_split_dataset_tolerances()
    test_pickling()