import pytest
import corgidrp.mocks as mocks
from corgidrp.l2a_to_l2b import frame_select


def test_no_selection():
    """
    Tests defautl case where no frame selection is done
    """
    # use mock darks for now, doesn't really matter right now
    default_dataset = mocks.create_dark_calib_files(numfiles=5)

    pruned_dataset = frame_select(default_dataset)

    assert len(pruned_dataset) == len(default_dataset)
    assert len(pruned_dataset) == 5

def test_bpfrac_cutoff():
    """
    Tests the bad pixel fraction cutoff
    Cutoff is a > comparison, not >=
    """
    # use mock darks for now, doesn't really matter right now
    default_dataset = mocks.create_dark_calib_files(numfiles=5)
    default_dataset[0].dq[0:512] += 1

    # at exactly 50%, it should not drop a frame. behavior is > 
    pruned_dataset = frame_select(default_dataset, bpix_frac=0.5)
    assert len(pruned_dataset) == len(default_dataset)
    assert len(pruned_dataset) == 5

    # one pixel over 50%, we should drop the first frame
    default_dataset[0].dq[-1,-1] += 1
    pruned_dataset = frame_select(default_dataset, bpix_frac=0.5)
    assert len(pruned_dataset) != len(default_dataset)
    assert len(pruned_dataset) == 4

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

def test_tt_rms():
    """
    Tests for tt rms jitter
    """
    default_dataset = mocks.create_dark_calib_files(numfiles=5)
    # add tt rms header
    tt_rms = 0
    for frame in default_dataset:
        frame.ext_hdr['RESZ2RMS'] = tt_rms
        tt_rms += 1
          
    # doest nothing
    pruned_dataset = frame_select(default_dataset)
    assert len(pruned_dataset) == len(default_dataset)
    assert len(pruned_dataset) == 5

    # removes 2 frames
    pruned_dataset = frame_select(default_dataset, tt_thres=2.5)
    assert len(pruned_dataset) != len(default_dataset)
    assert len(pruned_dataset) == 3 

def test_remove_all():
    """
    Uses multiple methods to remove frames. 
    Tests that error is raised if all frames is removed
    """
    # use mock darks for now, doesn't really matter right now
    default_dataset = mocks.create_dark_calib_files(numfiles=5)
    # make the first frame have 50% bad pixels
    default_dataset[0].dq[0:512] += 1
    # add tt rms header
    tt_rms = 0
    for frame in default_dataset:
        frame.ext_hdr['RESZ2RMS'] = tt_rms
        tt_rms += 1
    # add overexp
    default_dataset[1].ext_hdr['OVEREXP'] = True

    # keep only 1 frame
    # bpix_frac removes index 0
    # overexp removes index 1
    # tt_thres removes indicies 3,4
    pruned_dataset = frame_select(default_dataset, bpix_frac=0.1, overexp=True, tt_thres=2.5)
    assert len(pruned_dataset) == 1

    # removes all frames
    with pytest.raises(ValueError):
        pruned_dataset = frame_select(default_dataset, bpix_frac=0.1, overexp=True, tt_thres=1.5)


if __name__ == "__main__":
    test_bpfrac_cutoff()
