"""
Test the data level specification
"""
import pytest
import numpy as np
import corgidrp.mocks as mocks
import corgidrp.l1_to_l2a as l1_to_l2a
import corgidrp.l2a_to_l2b as l2a_to_l2b
import corgidrp.l2b_to_l3 as l2b_to_l3
import corgidrp.l3_to_l4 as l3_to_l4
from corgidrp.data import Image

np.random.seed(456)

def test_l1_to_l4():
    """
    Tests a successful upgrade of L1 to L4 in bookkeeping only
    """
    l1_dataset = mocks.create_prescan_files(arrtype="SCI", numfiles=2)
    # edit filenames
    fname_template = "CGI_L1_100_0200001001001100001_20270101T120000_{0:03d}.fits"
    for i, image in enumerate(l1_dataset):
        image.filename = fname_template.format(i)

    l2a_dataset = l1_to_l2a.update_to_l2a(l1_dataset)

    for frame in l2a_dataset:
        assert frame.ext_hdr['DATALVL'] == "L2a"
        assert "L2a" in frame.filename
    

    l2b_dataset = l2a_to_l2b.update_to_l2b(l2a_dataset)

    for frame in l2b_dataset:
        assert frame.ext_hdr['DATALVL'] == "L2b"
        assert "L2b" in frame.filename

    l3_dataset = l2b_to_l3.update_to_l3(l2b_dataset)

    for frame in l3_dataset:
        assert frame.ext_hdr['DATALVL'] == "L3"
        assert "L3" in frame.filename

    #Create dummy dataset to pass in to update_to_l4 (which needs filenames)
    pri_hdr, ext_hdr = mocks.create_default_L3_headers()
    test = Image(np.array([1,1]),pri_hdr = pri_hdr, ext_hdr = ext_hdr)
    # expect an exception

    l4_dataset = l3_to_l4.update_to_l4(l3_dataset, test, test)

    for frame in l4_dataset:
        assert frame.ext_hdr['DATALVL'] == "L4"
        assert "L4" in frame.filename

def test_l1_to_l2a_bad():
    """
    Tests an unsuccessful upgrade of L1 to L2a data because the input is not L1
    """
    l2a_dataset = mocks.create_dark_calib_files(numfiles=2)
    for frame in l2a_dataset:
        frame.ext_hdr['DATALVL'] = "L2a"
    
    # expect an exception
    with pytest.raises(ValueError):
        l2a_dataset_2 = l1_to_l2a.update_to_l2a(l2a_dataset)

def test_l1a_to_l2b_bad():
    """
    Tests an unsuccessful upgrade because input is not L2a
    """
    l1_dataset = mocks.create_dark_calib_files(numfiles=2)
    
    # expect an exception
    with pytest.raises(ValueError):
        l2b_dataset = l2a_to_l2b.update_to_l2b(l1_dataset)

def test_l2b_to_l3_bad():
    """
    Tests an unsuccessful upgrade because input is not L2b
    """
    l1_dataset = mocks.create_dark_calib_files(numfiles=2)
    
    # expect an exception
    with pytest.raises(ValueError):
        _ = l2b_to_l3.update_to_l3(l1_dataset)

def test_l3_to_l4_bad():
    """
    Tests an unsuccessful upgrade because input is not L3
    """
    l1_dataset = mocks.create_dark_calib_files(numfiles=2)
    
    #Create dummy dataset to pass in to update_to_l4 (which needs filenames)
    pri_hdr, ext_hdr = mocks.create_default_L3_headers()
    test = Image(np.array([1,1]),pri_hdr = pri_hdr, ext_hdr = ext_hdr)
    # expect an exception
    with pytest.raises(ValueError):
        _ = l3_to_l4.update_to_l4(l1_dataset, test, test)


if __name__ == "__main__":
    test_l1_to_l4()
    test_l1_to_l2a_bad()
    test_l1a_to_l2b_bad()
    test_l2b_to_l3_bad()
    test_l3_to_l4_bad()