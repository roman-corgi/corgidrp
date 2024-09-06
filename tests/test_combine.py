import os
import pytest
import numpy as np
import astropy.io.fits as fits
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.combine as combine

img1 = np.ones([100, 100])
err1 = np.ones([100, 100])
dq = np.zeros([100, 100], dtype = int)
prhd, exthd = mocks.create_default_headers()
errhd = fits.Header()
errhd["CASE"] = "test"
dqhd = fits.Header()
dqhd["CASE"] = "test"


def test_mean_combine_subexposures():
    """
    Test mean combine of subexposures
    """

    image1 = data.Image(img1, err=err1, dq=dq, pri_hdr = prhd, ext_hdr = exthd)
    image1.filename = "1.fits"
    image2 = image1.copy()
    image2.filename = "2.fits"
    image3 = image1.copy()
    image3.filename = "3.fits"
    image4 = image1.copy()
    image4.filename = "4.fits"

    dataset = data.Dataset([image1, image2, image3, image4])

    combined_dataset = combine.combine_subexposures(dataset, 2)

    assert(len(combined_dataset) == 2)
    assert(np.all(combined_dataset[0].data == 2))
    assert(np.all(combined_dataset[0].err == pytest.approx(np.sqrt(2))))
    assert(np.all(combined_dataset[0].dq == 0))

    assert combined_dataset[0].ext_hdr['DRPNFILE'] == 2
    assert combined_dataset[0].ext_hdr['FILE0'] in ["2.fits", "1.fits"]
    assert combined_dataset[0].ext_hdr['FILE1'] in ["2.fits", "1.fits"]

    assert combined_dataset[1].ext_hdr['DRPNFILE'] == 2
    assert combined_dataset[1].ext_hdr['FILE0'] in ["3.fits", "4.fits"]
    assert combined_dataset[1].ext_hdr['FILE1'] in ["3.fits", "4.fits"]

    # combine again
    combined_dataset_2 = combine.combine_subexposures(combined_dataset, 2)
     
    assert(len(combined_dataset_2) == 1)
    assert(np.all(combined_dataset_2[0].data == 4))
    assert(np.all(combined_dataset_2[0].err == pytest.approx(2)))
    assert(np.all(combined_dataset_2[0].dq == 0))

    assert combined_dataset_2[0].ext_hdr['DRPNFILE'] == 4
    assert combined_dataset_2[0].ext_hdr['FILE0'] in ["2.fits", "1.fits", "3.fits", "4.fits"]
    assert combined_dataset_2[0].ext_hdr['FILE1'] in ["2.fits", "1.fits", "3.fits", "4.fits"]
    assert combined_dataset_2[0].ext_hdr['FILE2'] in ["2.fits", "1.fits", "3.fits", "4.fits"]
    assert combined_dataset_2[0].ext_hdr['FILE3'] in ["2.fits", "1.fits", "3.fits", "4.fits"]


def test_mean_combine_subexposures_with_bad():
    """
    Test mean combine of subexposures with bad pixels
    """
    # use copies since we are going to modify their values
    image1 = data.Image(np.copy(img1), err=np.copy(err1), dq=np.copy(dq), 
                        pri_hdr = prhd, ext_hdr = exthd)
    image1.filename = "1.fits"
    image2 = image1.copy()
    image2.filename = "2.fits"
    image3 = image1.copy()
    image3.filename = "3.fits"
    image4 = image1.copy()
    image4.filename = "4.fits"

    # (0,0) has one bad frame
    image1.dq[0][0] = 1
    # (0,1) has both pixels bad
    image1.dq[0][1] = 1
    image2.dq[0][1] = 1

    dataset = data.Dataset([image1, image2, image3, image4])

    combined_dataset = combine.combine_subexposures(dataset, 2)

    assert(len(combined_dataset) == 2)
    # the pixel with one bad pixel should have same value but higher error
    assert combined_dataset[0].data[0][0] == 2
    assert combined_dataset[0].err[0][0][0] == pytest.approx(2)
    # compare against a pixel without any bad pixels
    assert combined_dataset[1].data[0][0] == 2
    assert combined_dataset[1].err[0][0][0] == pytest.approx(np.sqrt(2))

    # the pixel with two bad pixels should be nan
    assert np.isnan(combined_dataset[0].data[0][1])
    assert np.isnan(combined_dataset[0].err[0][0][1])
    assert combined_dataset[0].dq[0][1] == 1

def test_median_combine_subexposures():
    """
    Test median combine of subexposures
    """

    image1 = data.Image(img1, err=err1, dq=dq, pri_hdr = prhd, ext_hdr = exthd)
    image1.filename = "1.fits"
    image2 = image1.copy()
    image2.filename = "2.fits"
    image3 = image1.copy()
    image3.filename = "3.fits"
    image4 = image1.copy()
    image4.filename = "4.fits"

    dataset = data.Dataset([image1, image2, image3, image4])

    combined_dataset = combine.combine_subexposures(dataset, 2, collapse="median")

    assert(len(combined_dataset) == 2)
    assert(np.all(combined_dataset[0].data == 2))
    assert(np.all(combined_dataset[0].err == pytest.approx(np.sqrt(np.pi))))
    assert(np.all(combined_dataset[0].dq == 0))


if __name__ == "__main__":
    test_mean_combine_subexposures()