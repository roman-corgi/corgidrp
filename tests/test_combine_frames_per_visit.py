"""
Test the combine_frames_per_visit function in combine
"""
import pytest
import numpy as np
import corgidrp.mocks as mocks
import corgidrp.l1_to_l2a as l1_to_l2a
import corgidrp.l2a_to_l2b as l2a_to_l2b
import corgidrp.l2b_to_l3 as l2b_to_l3
import corgidrp.combine as combine
from corgidrp.data import Dataset, Image

def test_combine_frames_per_visit_split_groups():
    """
    Tests that combine_frames_per_visit combines each VISITID/DPAMNAME subset independently.
    """
    group_specs = [
        ("1111111111111111111", "POL0", [1.0, 2.0], ["v1_pol0_a.fits", "v1_pol0_b.fits"]),
        ("1111111111111111111", "POL45", [10.0, 20.0], ["v1_pol45_a.fits", "v1_pol45_b.fits"]),
        ("2222222222222222222", "POL0", [100.0, 200.0], ["v2_pol0_a.fits", "v2_pol0_b.fits"]),
    ]

    frames = []
    expected_by_group = {}
    for visitid, dpamname, values, filenames in group_specs:
        expected_by_group[(visitid, dpamname)] = {
            "mean": np.mean(values),
            "err": 1 / np.sqrt(len(values)),
            "filenames": set(filenames),
            "snr": sum(values) / np.sqrt(len(values)),
        }
        for value, filename in zip(values, filenames):
            pri_hdr, ext_hdr, err_hdr, dq_hdr = mocks.create_default_L3_headers()
            pri_hdr["VISITID"] = visitid
            ext_hdr["DPAMNAME"] = dpamname
            image = Image(
                np.full((4, 4), value),
                err=np.ones((1, 4, 4)),
                dq=np.zeros((4, 4), dtype=np.uint16),
                pri_hdr=pri_hdr,
                ext_hdr=ext_hdr,
                err_hdr=err_hdr,
                dq_hdr=dq_hdr,
            )
            image.filename = filename
            image.pri_hdr["FILENAME"] = filename
            frames.append(image)
            
    combined_dataset = combine.combine_frames_per_visit(
        Dataset(frames),
        collapse="mean",
        num_frames_scaling=False,
    )

    assert len(combined_dataset) == len(group_specs)

    for frame in combined_dataset:
        group_key = (frame.pri_hdr["VISITID"], frame.ext_hdr["DPAMNAME"])
        expected = expected_by_group[group_key]

        assert np.all(frame.data == pytest.approx(expected["mean"]))
        assert np.all(frame.err[0] == pytest.approx(expected["err"]))
        assert np.all(frame.data / frame.err[0] == pytest.approx(expected["snr"]))
        assert frame.ext_hdr["NUM_FR"] == 2
        assert frame.ext_hdr["DRPNFILE"] == 2
        assert {frame.ext_hdr["FILE0"], frame.ext_hdr["FILE1"]} == expected["filenames"]


def test_combine_frames_per_visit_num_frames_within_split_groups():
    """
    Tests that combine_frames_per_visit applies num_frames_per_group within each
    VISITID/DPAMNAME subset.
    """
    group_specs = [
        ("1111111111111111111", "POL0", [1.0, 3.0, 5.0, 7.0],
         ["v1_pol0_a.fits", "v1_pol0_b.fits", "v1_pol0_c.fits", "v1_pol0_d.fits"]),
        ("2222222222222222222", "POL45", [10.0, 14.0], ["v2_pol45_a.fits", "v2_pol45_b.fits"]),
    ]

    frames = []
    expected_by_output = {
        "v1_pol0_b.fits": {
            "mean": 2.0,
            "err": 1 / np.sqrt(2),
            "snr": 2.0 * np.sqrt(2),
            "filenames": {"v1_pol0_a.fits", "v1_pol0_b.fits"},
        },
        "v1_pol0_d.fits": {
            "mean": 6.0,
            "err": 1 / np.sqrt(2),
            "snr": 6.0 * np.sqrt(2),
            "filenames": {"v1_pol0_c.fits", "v1_pol0_d.fits"},
        },
        "v2_pol45_b.fits": {
            "mean": 12.0,
            "err": 1 / np.sqrt(2),
            "snr": 12.0 * np.sqrt(2),
            "filenames": {"v2_pol45_a.fits", "v2_pol45_b.fits"},
        },
    }

    for visitid, dpamname, values, filenames in group_specs:
        for value, filename in zip(values, filenames):
            pri_hdr, ext_hdr, err_hdr, dq_hdr = mocks.create_default_L3_headers()
            pri_hdr["VISITID"] = visitid
            ext_hdr["DPAMNAME"] = dpamname
            image = Image(
                np.full((4, 4), value),
                err=np.ones((1, 4, 4)),
                dq=np.zeros((4, 4), dtype=np.uint16),
                pri_hdr=pri_hdr,
                ext_hdr=ext_hdr,
                err_hdr=err_hdr,
                dq_hdr=dq_hdr,
            )
            image.filename = filename
            image.pri_hdr["FILENAME"] = filename
            frames.append(image)

    combined_dataset = combine.combine_frames_per_visit(
        Dataset(frames),
        collapse="mean",
        num_frames_per_group=2,
        num_frames_scaling=False,
    )

    assert len(combined_dataset) == 3

    for frame in combined_dataset:
        expected = expected_by_output[frame.filename]

        assert np.all(frame.data == pytest.approx(expected["mean"]))
        assert np.all(frame.err[0] == pytest.approx(expected["err"]))
        assert np.all(frame.data / frame.err[0] == pytest.approx(expected["snr"]))
        assert frame.ext_hdr["NUM_FR"] == 2
        assert frame.ext_hdr["DRPNFILE"] == 2
        assert {frame.ext_hdr["FILE0"], frame.ext_hdr["FILE1"]} == expected["filenames"]


def test_combine_frames_per_visit_custom_split_keywords():
    """
    Tests that combine_frames_per_visit adds caller-provided split keywords.
    """
    group_specs = [
        ("1111111111111111111", "POL0", "target_a", "FILT1", [1.0, 3.0], ["visit1_target_a_filt1_a.fits", "visit1_target_a_filt1_b.fits"]),
        ("2222222222222222222", "POL0", "target_a", "FILT1", [10.0, 14.0], ["visit2_target_a_filt1_a.fits", "visit2_target_a_filt1_b.fits"]),
        ("1111111111111111111", "POL45", "target_a", "FILT1", [100.0, 200.0], ["visit1_pol45_target_a_filt1_a.fits", "visit1_pol45_target_a_filt1_b.fits"]),
        ("1111111111111111111", "POL0", "target_b", "FILT2", [1000.0, 2000.0], ["visit1_target_b_filt2_a.fits", "visit1_target_b_filt2_b.fits"]),
    ]

    frames = []
    expected_by_group = {}
    for visitid, dpamname, target, filter_name, values, filenames in group_specs:
        expected_by_group[(visitid, dpamname, target, filter_name)] = np.mean(values)
        for value, filename in zip(values, filenames):
            pri_hdr, ext_hdr, err_hdr, dq_hdr = mocks.create_default_L3_headers()
            pri_hdr["VISITID"] = visitid
            pri_hdr["TARGET"] = target
            ext_hdr["DPAMNAME"] = dpamname
            ext_hdr["CFAMNAME"] = filter_name
            image = Image(
                np.full((4, 4), value),
                err=np.ones((1, 4, 4)),
                dq=np.zeros((4, 4), dtype=np.uint16),
                pri_hdr=pri_hdr,
                ext_hdr=ext_hdr,
                err_hdr=err_hdr,
                dq_hdr=dq_hdr,
            )
            image.filename = filename
            image.pri_hdr["FILENAME"] = filename
            frames.append(image)

    combined_dataset = combine.combine_frames_per_visit(
        Dataset(frames),
        collapse="mean",
        num_frames_scaling=False,
        pri_split_keywords=["TARGET"],
        ext_split_keywords=["CFAMNAME"],
    )

    assert len(combined_dataset) == len(group_specs)
    for frame in combined_dataset:
        group_key = (
            frame.pri_hdr["VISITID"],
            frame.ext_hdr["DPAMNAME"],
            frame.pri_hdr["TARGET"],
            frame.ext_hdr["CFAMNAME"],
        )
        assert np.all(frame.data == pytest.approx(expected_by_group[group_key]))
        assert frame.ext_hdr["NUM_FR"] == 2
        assert frame.ext_hdr["DRPNFILE"] == 2


def test_combine_frames_per_visit_auto_override_num_frames_per_group():
    """
    Tests that combine_frames_per_visit increases num_frames_per_group when a split subset
    would otherwise produce more than max_combined output frames.
    """
    frames = []
    visitid = "3333333333333333333"
    dpamname = "POL0"
    for i in range(240):
        filename = f"big_pol0_{i:03d}.fits"
        pri_hdr, ext_hdr, err_hdr, dq_hdr = mocks.create_default_L3_headers()
        pri_hdr["VISITID"] = visitid
        ext_hdr["DPAMNAME"] = dpamname
        image = Image(
            np.full((4, 4), float(i)),
            err=np.ones((1, 4, 4)),
            dq=np.zeros((4, 4), dtype=np.uint16),
            pri_hdr=pri_hdr,
            ext_hdr=ext_hdr,
            err_hdr=err_hdr,
            dq_hdr=dq_hdr,
        )
        image.filename = filename
        image.pri_hdr["FILENAME"] = filename
        frames.append(image)

    combined_dataset = combine.combine_frames_per_visit(
        Dataset(frames),
        collapse="mean",
        num_frames_per_group=2,
        num_frames_scaling=False,
    )

    assert len(combined_dataset) == 80
    for frame in combined_dataset:
        assert frame.ext_hdr["NUM_FR"] == 3
        assert frame.ext_hdr["DRPNFILE"] == 3
        assert np.all(frame.err[0] == pytest.approx(1 / np.sqrt(3)))

    first_frame = next(frame for frame in combined_dataset if frame.filename == "big_pol0_002.fits")
    last_frame = next(frame for frame in combined_dataset if frame.filename == "big_pol0_239.fits")

    assert np.all(first_frame.data == pytest.approx(1.0))
    assert np.all(first_frame.data / first_frame.err[0] == pytest.approx(np.sqrt(3)))
    assert {first_frame.ext_hdr["FILE0"], first_frame.ext_hdr["FILE1"], first_frame.ext_hdr["FILE2"]} == {
        "big_pol0_000.fits",
        "big_pol0_001.fits",
        "big_pol0_002.fits",
    }

    assert np.all(last_frame.data == pytest.approx(238.0))
    assert {
        last_frame.ext_hdr["FILE0"],
        last_frame.ext_hdr["FILE1"],
        last_frame.ext_hdr["FILE2"],
    } == {
        "big_pol0_237.fits",
        "big_pol0_238.fits",
        "big_pol0_239.fits",
    }

    max_combined_dataset = combine.combine_frames_per_visit(
        Dataset(frames),
        collapse="mean",
        num_frames_per_group=2,
        num_frames_scaling=False,
        max_combined=60,
    )

    assert len(max_combined_dataset) == 60
    for frame in max_combined_dataset:
        assert frame.ext_hdr["NUM_FR"] == 4
        assert frame.ext_hdr["DRPNFILE"] == 4


if __name__ == "__main__":
    test_combine_frames_per_visit_split_groups()
    test_combine_frames_per_visit_num_frames_within_split_groups()
    test_combine_frames_per_visit_custom_split_keywords()
    test_combine_frames_per_visit_auto_override_num_frames_per_group()
