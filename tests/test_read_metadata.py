# -*- coding: utf-8 -*-
"""Unit tests for read_metadata."""
from __future__ import absolute_import, division, print_function

import os
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import yaml

from corgidrp.detector import Metadata, ReadMetadataException

# Metadata path
corgi_path = Path(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'corgidrp')))
meta_path_default = Path(corgi_path, 'util', 'metadata.yaml')
# in case a different file wants to be specified below, define default path
# above separately
meta_path = Path(corgi_path, 'util', 'metadata.yaml')
meta_path_test = Path(corgi_path, 'util', 'metadata_test.yaml')
# Read yaml
with open(meta_path, 'r') as stream:
    yaml_dict = yaml.safe_load(stream)

# Instantiate class for testing

meta = Metadata(meta_path)
obstype = meta.obstype


class TestMetadata(unittest.TestCase):
    """Unit tests for __init__ method."""

    def test_input(self):
        """Check that input value is assigned correctly."""
        self.assertEqual(meta.meta_path, meta_path)

    def test_data(self):
        """Check that data attribute is the same as the output of get_data."""
        self.assertEqual(meta.data, meta.get_data())

    def test_attributes(self):
        """Check that other attributes are correct."""
        self.assertEqual(meta.frame_rows, meta.data[obstype]['frame_rows'])
        self.assertEqual(meta.frame_cols, meta.data[obstype]['frame_cols'])
        self.assertEqual(meta.geom, meta.data[obstype]['geom'])
        self.assertEqual(meta.obstype, 'SCI') # default one

    def test_default(self):
        """
        Check that the default if no path is specified is the metadata.yaml
        in the repo.
        """
        m2 = Metadata()
        self.assertEqual(meta_path_default, m2.meta_path)


class TestGetData(unittest.TestCase):
    """Unit tests for get_data method."""

    def test_data_values(self):
        """Check that data is being read in properly."""
        data = meta.get_data()
        self.assertEqual(data, yaml_dict)


class TestSliceSection(unittest.TestCase):
    """Unit tests for slice_section method."""

    @patch.object(Metadata, '_unpack_geom')
    def test_slice_section(self, mock_unpack_geom):
        """Verify method returns correct slice."""
        rows = 4
        cols = 5
        ul = (1, 2)
        sub_frame = np.ones((rows, cols))
        sub_frame[0, 0] = 2
        check_frame = np.zeros((9, 10))
        check_frame[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols] = sub_frame

        mock_unpack_geom.return_value = (rows, cols, ul)
        m2 = Metadata(meta_path)
        sliced = m2.slice_section(check_frame, 'test')

        self.assertEqual(sliced.tolist(), sub_frame.tolist())

    @patch.object(Metadata, '_unpack_geom')
    def test_slice_section_exception(self, mock_unpack_geom):
        """Verify method returns exception as expected."""
        rows = 0
        cols = 5
        ul = (1, 2)
        # sub_frame = np.ones((rows, cols))
        # sub_frame[0, 0] = 2
        check_frame = np.zeros((9, 10))
        #check_frame[ul[0]:ul[0]+rows, ul[1]:ul[1]+cols] = sub_frame

        mock_unpack_geom.return_value = (rows, cols, ul)
        m2 = Metadata(meta_path)

        with self.assertRaises(ReadMetadataException):
            m2.slice_section(check_frame, 'test')

class TestImagingSlice(unittest.TestCase):
    """Unit tests for imaging_slice method."""

    def test_imaging_slice_section(self):
        """Verifying method returns correct imaging slice, which includes
        all physical CCD pixels."""
        # total for frame (numbers from metadata_test.yaml
        rows = 120
        cols = 220
        # parallel overscan: (po_rows, po_cols, (106, 108))
        po_rows = 14
        po_cols = 107
        # prescan: (ps_rows, ps_cols, (0, 0))
        ps_rows = 120
        ps_cols = 108
        # serial overscan: (so_rows, so_cols, (0, 215))
        so_rows = 120
        so_cols = 5
        # image: (im_rows, im_cols, (2, 108))
        im_rows = 104
        im_cols = 105
        #starts at col right after ps, but at row above 0
        im_ul = (2, 108)
        ccd_rows = rows - po_rows
        ccd_cols = cols - ps_cols - so_cols

        m2 = Metadata(meta_path_test)
        check_frame = np.zeros((120, 220))
        sub_frame = np.ones((ccd_rows, ccd_cols))
        ccd_ul = (im_ul[0] - (ccd_rows - im_rows), im_ul[1])
        sub_frame[0, 0] = 2
        check_frame[ccd_ul[0]:ccd_ul[0]+ccd_rows,
                    ccd_ul[1]:ccd_ul[1]+ccd_cols] = sub_frame

        sliced = m2.imaging_slice(check_frame)

        self.assertEqual(sliced.tolist(), sub_frame.tolist())


if __name__ == '__main__':
    unittest.main()
