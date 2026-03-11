import pytest
import numpy as np
import corgidrp.mocks as mocks
from corgidrp.l1_to_l2a import discard_setup_frames

np.random.seed(456)

def test_no_discard():
    """No frames have setup keywords set, so all should survive."""
    dataset = mocks.create_dark_calib_files(numfiles=5)
    result = discard_setup_frames(dataset, keywords_to_check=["ISACQ"])
    assert len(result) == 5

def test_discard_acquisition():
    """ISACQ=1 on 2 of 5 frames -> 3 remain."""
    dataset = mocks.create_dark_calib_files(numfiles=5)
    dataset[0].ext_hdr["ISACQ"] = 1
    dataset[3].ext_hdr["ISACQ"] = 1

    result = discard_setup_frames(dataset, keywords_to_check=["ISACQ"])
    assert len(result) == 3
    assert dataset[0].filename in result[0].ext_hdr["HISTORY"][-1]
    assert "ISACQ" in result[0].ext_hdr["HISTORY"][-1]

def test_discard_speckle_balance():
    """SPBAL=1 on some frames."""
    dataset = mocks.create_dark_calib_files(numfiles=5)
    dataset[1].ext_hdr["SPBAL"] = 1
    dataset[4].ext_hdr["SPBAL"] = 1

    result = discard_setup_frames(dataset, keywords_to_check=["SPBAL"])
    assert len(result) == 3
    assert "SPBAL" in result[0].ext_hdr["HISTORY"][-1]

def test_discard_howfsc():
    """ISHOWFSC=1 on some frames."""
    dataset = mocks.create_dark_calib_files(numfiles=5)
    dataset[2].ext_hdr["ISHOWFSC"] = 1

    result = discard_setup_frames(dataset, keywords_to_check=["ISHOWFSC"])
    assert len(result) == 4
    assert "ISHOWFSC" in result[0].ext_hdr["HISTORY"][-1]

def test_discard_multiple_keywords():
    """Mix of SPBAL and ISHOWFSC on different frames."""
    dataset = mocks.create_dark_calib_files(numfiles=5)
    dataset[0].ext_hdr["SPBAL"] = 1
    dataset[2].ext_hdr["ISHOWFSC"] = 1
    dataset[4].ext_hdr["SPBAL"] = 1
    dataset[4].ext_hdr["ISHOWFSC"] = 1  # both set

    result = discard_setup_frames(dataset, keywords_to_check=["SPBAL", "ISHOWFSC"])
    assert len(result) == 2
    history = result[0].ext_hdr["HISTORY"][-1]
    assert "SPBAL" in history
    assert "ISHOWFSC" in history

def test_all_discarded_raises():
    """All frames have ISACQ=1 -> ValueError."""
    dataset = mocks.create_dark_calib_files(numfiles=5)
    for frame in dataset:
        frame.ext_hdr["ISACQ"] = 1

    with pytest.raises(ValueError):
        discard_setup_frames(dataset, keywords_to_check=["ISACQ"])

def test_noop_when_no_keywords():
    """keywords_to_check=None or [] -> all survive."""
    dataset = mocks.create_dark_calib_files(numfiles=5)
    # Set a keyword that would normally cause discard
    dataset[0].ext_hdr["ISACQ"] = 1

    result_none = discard_setup_frames(dataset, keywords_to_check=None)
    assert len(result_none) == 5

    result_empty = discard_setup_frames(dataset, keywords_to_check=[])
    assert len(result_empty) == 5

def test_missing_keyword_safe():
    """Keyword not in header -> treated as 0 (not discarded)."""
    dataset = mocks.create_dark_calib_files(numfiles=5)
    # Don't set ISACQ on any frame
    result = discard_setup_frames(dataset, keywords_to_check=["ISACQ"])
    assert len(result) == 5

if __name__ == "__main__":
    test_no_discard()
    test_discard_acquisition()
    test_discard_speckle_balance()
    test_discard_howfsc()
    test_discard_multiple_keywords()
    test_all_discarded_raises()
    test_noop_when_no_keywords()
    test_missing_keyword_safe()
