import pytest
import numpy as np
import corgidrp.mocks as mocks
from corgidrp.l2a_to_l2b import frame_select

np.random.seed(123)

def test_no_selection():
    """
    Tests defautl case where no frame selection is done
    """
    # use mock darks for now, doesn't really matter right now
    default_dataset = mocks.create_dark_calib_files(numfiles=5)

    pruned_dataset = frame_select(default_dataset)

    assert len(pruned_dataset) == len(default_dataset)
    assert len(pruned_dataset) == 5
    
    # assert headers are filled out
    assert pruned_dataset[0].ext_hdr['FRMSEL01'] == 1
    assert pruned_dataset[0].ext_hdr['FRMSEL02'] == False
    assert pruned_dataset[0].ext_hdr['FRMSEL03'] == None
    assert pruned_dataset[0].ext_hdr['FRMSEL04'] == None
    assert pruned_dataset[0].ext_hdr['FRMSEL05'] == None
    assert pruned_dataset[0].ext_hdr['FRMSEL06'] == None

def test_bpfrac_cutoff():
    """
    Tests the bad pixel fraction cutoff
    Cutoff is a > comparison, not >=
    """
    # use mock darks for now, doesn't really matter right now
    default_dataset = mocks.create_dark_calib_files(numfiles=5)
    default_dataset[0].dq[0:int(default_dataset.all_data.shape[-2]/2)] += 1

    # at exactly 50%, it should not drop a frame. behavior is >
    pruned_dataset = frame_select(default_dataset, bpix_frac=0.5)
    assert len(pruned_dataset) == len(default_dataset)
    assert len(pruned_dataset) == 5

    # one pixel over 50%, we should drop the first frame
    default_dataset[0].dq[-1,-1] += 1
    pruned_dataset = frame_select(default_dataset, bpix_frac=0.5)
    assert len(pruned_dataset) != len(default_dataset)
    assert len(pruned_dataset) == 4
    assert default_dataset[0].filename in pruned_dataset[0].ext_hdr['HISTORY'][-1] # test history
    assert pruned_dataset[0].ext_hdr['FRMSEL01'] == 0.5

    # allowing DQ = 1 should make no frames get dropped
    pruned_dataset = frame_select(default_dataset, bpix_frac=0.5, allowed_bpix=1)
    assert len(pruned_dataset) == len(default_dataset)
    assert len(pruned_dataset) == 5

    # allowing DQ = 1 should not affect DQ = 2
    default_dataset[0].dq[0:int(default_dataset.all_data.shape[-2]/2)] = 2
    default_dataset[0].dq[-1,-1] = 2
    pruned_dataset = frame_select(default_dataset, bpix_frac=0.5, allowed_bpix=1)
    assert len(pruned_dataset) != len(default_dataset)
    assert len(pruned_dataset) == 4
    assert default_dataset[0].filename in pruned_dataset[0].ext_hdr['HISTORY'][-1] # test history


def test_overexp():
    """
    Tests the overexp keyword
    """
    default_dataset = mocks.create_dark_calib_files(numfiles=5)
    default_dataset[0].ext_hdr['OVEREXP'] = True

    # doest nothing
    pruned_dataset = frame_select(default_dataset)
    assert len(pruned_dataset) == len(default_dataset)
    assert len(pruned_dataset) == 5

    # remove overexp
    pruned_dataset = frame_select(default_dataset, overexp=True)
    assert len(pruned_dataset) != len(default_dataset)
    assert len(pruned_dataset) == 4
    assert pruned_dataset[0].ext_hdr['FRMSEL02'] == True

def test_tt_rms():
    """
    Tests for tt rms jitter
    """
    default_dataset = mocks.create_dark_calib_files(numfiles=5)
    # add tt rms header
    tt_rms = 0
    for frame in default_dataset:
        frame.ext_hdr['Z2VAR'] = tt_rms
        frame.ext_hdr['Z3VAR'] = tt_rms
        tt_rms += 1

    # does nothing
    pruned_dataset = frame_select(default_dataset)
    assert len(pruned_dataset) == len(default_dataset)
    assert len(pruned_dataset) == 5

    # removes 2 frames
    pruned_dataset = frame_select(default_dataset, tt_rms_thres=2.5)
    assert len(pruned_dataset) != len(default_dataset)
    assert len(pruned_dataset) == 3
    assert pruned_dataset[0].ext_hdr['FRMSEL03'] == 2.5
    assert pruned_dataset[0].ext_hdr['FRMSEL04'] == 2.5

def test_tt_bias():
    """
    Tests for tt offset
    """
    default_dataset = mocks.create_dark_calib_files(numfiles=5)
    # add tt rms header
    tt_rms = 0
    for frame in default_dataset:
        frame.ext_hdr['Z2RES'] = tt_rms
        frame.ext_hdr['Z3RES'] = tt_rms
        tt_rms += 1

    # does nothing
    pruned_dataset = frame_select(default_dataset)
    assert len(pruned_dataset) == len(default_dataset)
    assert len(pruned_dataset) == 5

    # removes 2 frames
    pruned_dataset = frame_select(default_dataset, tt_bias_thres=2.5)
    assert len(pruned_dataset) != len(default_dataset)
    assert len(pruned_dataset) == 3
    assert pruned_dataset[0].ext_hdr['FRMSEL05'] == 2.5
    assert pruned_dataset[0].ext_hdr['FRMSEL06'] == 2.5

def test_remove_all():
    """
    Uses multiple methods to remove frames.
    Tests that error is raised if all frames is removed
    """
    # use mock darks for now, doesn't really matter right now
    default_dataset = mocks.create_dark_calib_files(numfiles=5)
    # make the first frame have 50% bad pixels
    default_dataset[0].dq[0:int(default_dataset.all_data.shape[-2]/2)] += 1
    # add tt rms header
    tt_rms = 0
    for frame in default_dataset:
        frame.ext_hdr['Z2VAR'] = tt_rms
        frame.ext_hdr['Z3VAR'] = tt_rms
        tt_rms += 1
    # add overexp
    default_dataset[1].ext_hdr['OVEREXP'] = True

    # keep only 1 frame
    # bpix_frac removes index 0
    # overexp removes index 1
    # tt_rms_thres removes indicies 3,4
    pruned_dataset = frame_select(default_dataset, bpix_frac=0.1, overexp=True, tt_rms_thres=2.5)
    assert len(pruned_dataset) == 1

    # removes all frames
    with pytest.raises(ValueError):
        pruned_dataset = frame_select(default_dataset, bpix_frac=0.1, overexp=True, tt_rms_thres=1.5)

def test_marking():
    """
    Tests marking frames as bad instead of completely dropping them
    """
    default_dataset = mocks.create_dark_calib_files(numfiles=5)
    # add tt rms header
    tt_rms = 0
    for frame in default_dataset:
        frame.ext_hdr['Z2RES'] = tt_rms
        frame.ext_hdr['Z3RES'] = tt_rms
        tt_rms += 1

    # does nothing
    pruned_dataset = frame_select(default_dataset, discard_bad=False)
    assert len(pruned_dataset) == len(default_dataset)
    assert len(pruned_dataset) == 5
    for frame in pruned_dataset:
        assert frame.ext_hdr['IS_BAD'] == False

    # marks 2 frames as bad
    pruned_dataset = frame_select(default_dataset, tt_bias_thres=2.5, discard_bad=False)
    assert len(pruned_dataset) == len(default_dataset)
    # but first 3 are good
    for frame in pruned_dataset[:3]:
        assert frame.ext_hdr['IS_BAD'] == False
    # and last 2 are bad
    for frame in pruned_dataset[3:]:
        assert frame.ext_hdr['IS_BAD'] == True

if __name__ == "__main__":
    test_no_selection()
    test_bpfrac_cutoff()
    test_overexp()
    test_tt_rms()
    test_remove_all()