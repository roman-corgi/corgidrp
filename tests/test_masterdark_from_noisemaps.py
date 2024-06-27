import unittest

import numpy as np
import os
from pathlib import Path

from corgidrp.masterdark_from_noisemaps import build_dark
from corgidrp.mocks import create_noise_maps
from corgidrp.detector import unpack_geom, detector_areas

im_rows, im_cols, _ = unpack_geom(detector_areas, 'SCI', 'image')
rows = detector_areas['SCI']['frame_rows']
cols = detector_areas['SCI']['frame_cols']

class TestBuildDark(unittest.TestCase):
    """
    Tests for building the dark
    """

    def setUp(self):

        self.Fd = np.ones((rows, cols))
        self.Dd = 3/3600*np.ones((rows, cols))
        self.Cd = 0.02*np.ones((rows, cols))
        self.F, self.C, self.D = create_noise_maps(self.Fd, None, None, self.Cd,
                                                   None, None, self.Dd, None, None)
        self.g = 1
        self.t = 1
        pass


    def test_success(self):
        """Good inputs complete as expected"""
        build_dark(self.F, self.D, self.C, self.g, self.t)
        build_dark(self.F, self.D, self.C, self.g, self.t, detector_areas, full_frame=True)
        pass


    def test_output_size(self):
        """output is correct size"""
        M = build_dark(self.F, self.D, self.C, self.g, self.t)
        self.assertTrue(M.data.shape == (im_rows, im_cols))
        pass


    def test_exact_case(self):
        """Exact case produces expected result.  Use full frame this time."""
        tol = 1e-13

        Fd = 5*np.ones((rows, cols))
        Ferr = 0.1*Fd
        Fdq = np.zeros((rows, cols))
        Fdq[3,3] = 1
        Dd = 1/7*np.ones((rows, cols))
        Derr = 0.1*Dd
        Ddq = np.zeros((rows, cols))
        Ddq[3,3] = 1 # Fdq has this one, too
        Ddq[2,2] = 1
        Cd = 1*np.ones((rows, cols))
        Cerr = 0.1*Cd
        Cdq = np.zeros((rows, cols))
        Cdq[1,2] = 1
        g = 5
        t = 7
        target = 3*np.ones((rows, cols))
        exp_err = np.sqrt(Derr**2*g**2*t**2 + Cerr**2*g**2 + Ferr**2)/g
        exp_dq = np.zeros((rows, cols))
        exp_dq[3,3] = 1
        exp_dq[2,2] = 1
        exp_dq[1,2] = 1
        F, C, D = create_noise_maps(Fd, Ferr, Fdq, Cd, Cerr, Cdq, Dd, Derr, Ddq)
        M = build_dark(F, D, C, g, t, full_frame=True)
        self.assertTrue(np.max(np.abs(M.data - target)) < tol)
        self.assertTrue(np.max(np.abs(M.err - exp_err)) < tol)
        self.assertTrue(np.max(np.abs(M.dq - exp_dq)) < tol)
        self.assertTrue(M.err_hdr['BUNIT'] == 'detected electrons')
        self.assertTrue(M.ext_hdr['EXPTIME'] == t)
        self.assertTrue(M.ext_hdr['CMDGAIN'] == g)
        self.assertTrue(M.ext_hdr['DATATYPE'] == 'Dark')
        self.assertTrue(M.data.shape == (rows, cols))
        self.assertTrue(M.ext_hdr['NAXIS1'] == rows)
        self.assertTrue(M.ext_hdr['NAXIS2'] == cols)
        self.assertTrue(M.ext_hdr['DRPNFILE'] == 3)
        self.assertTrue(M.filename == '0_FPN_NoiseMap_dark.fits')
        self.assertTrue('EM gain = '+str(g) in str(M.ext_hdr['HISTORY']))
        self.assertTrue('exptime = '+str(t) in str(M.ext_hdr['HISTORY']))

        M_copy = M.copy()
        self.assertTrue(np.max(np.abs(M_copy.data - target)) < tol)
        self.assertTrue(np.max(np.abs(M_copy.err - exp_err)) < tol)
        self.assertTrue(np.max(np.abs(M_copy.dq - exp_dq)) < tol)
        self.assertTrue(M_copy.err_hdr['BUNIT'] == 'detected electrons')
        self.assertTrue(M_copy.ext_hdr['EXPTIME'] == t)
        self.assertTrue(M_copy.ext_hdr['CMDGAIN'] == g)
        self.assertTrue(M_copy.ext_hdr['DATATYPE'] == 'Dark')
        self.assertTrue(M_copy.data.shape == (rows, cols))
        self.assertTrue(M_copy.ext_hdr['NAXIS1'] == rows)
        self.assertTrue(M_copy.ext_hdr['NAXIS2'] == cols)
        self.assertTrue(M_copy.ext_hdr['DRPNFILE'] == 3)
        self.assertTrue(M_copy.filename == '0_FPN_NoiseMap_dark.fits')
        self.assertTrue('EM gain = '+str(g) in str(M_copy.ext_hdr['HISTORY']))
        self.assertTrue('exptime = '+str(t) in str(M_copy.ext_hdr['HISTORY']))
        pass


    def test_gain_goes_as_1overg(self):
        """change in dark goes as 1/g"""
        tol = 1e-13
        F0, C, D = create_noise_maps(0*self.Fd, None, None, self.Cd,
                                            None, None, self.Dd, None, None)
        F0 = build_dark(F0, D, C, 1, self.t)
        M1 = build_dark(self.F, self.D, self.C, 1, self.t)
        M2 = build_dark(self.F, self.D, self.C, 2, self.t)
        M4 = build_dark(self.F, self.D, self.C, 4, self.t)
        dg1 = M1.data-F0.data
        dg2 = M2.data-F0.data
        dg4 = M4.data-F0.data

        self.assertTrue(np.max(np.abs(dg2 - dg1/2)) < tol)
        self.assertTrue(np.max(np.abs(dg4 - dg2/2)) < tol)
        pass


    def test_exptime_goes_as_t(self):
        """change in dark goes as t"""
        tol = 1e-13

        M0 = build_dark(self.F, self.D, self.C, self.g, 0)
        M2 = build_dark(self.F, self.D, self.C, self.g, 2)
        M4 = build_dark(self.F, self.D, self.C, self.g, 4)
        dg2 = M2.data-M0.data
        dg4 = M4.data-M0.data

        self.assertTrue(np.max(np.abs(dg4 - dg2*2)) < tol)
        pass


    def test_c_doesnt_change_with_g_or_t(self):
        """F = 0 and D = 0 implies C is constant"""
        tol = 1e-13
        F0, C, D0 = create_noise_maps(0*self.Fd, None, None, self.Cd,
                                            None, None, 0*self.Dd, None, None)
        M = build_dark(F0, D0, C, 1, self.t)
        G2 = build_dark(F0, D0, C, 2, self.t)
        G4 = build_dark(F0, D0, C, 4, self.t)
        T2 = build_dark(F0, D0, C, self.g, 2)
        T4 = build_dark(F0, D0, C, self.g, 4)
        dg2 = G2.data-M.data
        dg4 = G4.data-M.data
        dt2 = T2.data-M.data
        dt4 = T4.data-M.data

        self.assertTrue(np.max(np.abs(dg2)) < tol)
        self.assertTrue(np.max(np.abs(dg4)) < tol)
        self.assertTrue(np.max(np.abs(dt2)) < tol)
        self.assertTrue(np.max(np.abs(dt4)) < tol)
        pass


    def test_bias_subtracted(self):
        """check there is no bias when all three noise terms are 0"""
        F0, C0, D0 = create_noise_maps(0*self.Fd, None, None, 0*self.Cd,
                                            None, None, 0*self.Dd, None, None)
        M = build_dark(F0, D0, C0, 1, self.t)
        G2 = build_dark(F0, D0, C0, 2, self.t)
        G4 = build_dark(F0, D0, C0, 4, self.t)
        T2 = build_dark(F0, D0, C0, self.g, 2)
        T4 = build_dark(F0, D0, C0, self.g, 4)

        self.assertTrue((M.data == 0).all())
        self.assertTrue((G2.data == 0).all())
        self.assertTrue((G4.data == 0).all())
        self.assertTrue((T2.data == 0).all())
        self.assertTrue((T4.data == 0).all())

        pass

    def test_invalid_F_shape(self):
        """Invalid inputs caught as expected"""
        for perr in [np.ones((1024, 1024)), np.ones((1, 1024)), np.ones((2, 2))]:
            perrF, _, _ = create_noise_maps(perr, None, None, self.Cd,
                                            None, None, self.Dd, None, None)
            with self.assertRaises(TypeError):
                build_dark(perr, self.D, self.C, self.g, self.t)
            pass
        pass


    def test_invalid_D_shape(self):
        """Invalid inputs caught as expected"""
        for perr in [np.ones((1024, 1024)), np.ones((1, 1024)), np.ones((2, 2))]:
            _, _, perrD = create_noise_maps(self.Fd, None, None, self.Cd,
                                            None, None, perr, None, None)
            with self.assertRaises(TypeError):
                build_dark(self.F, perrD, self.C, self.g, self.t)
            pass
        pass

    def test_invalid_C_shape(self):
        """Invalid inputs caught as expected"""
        for perr in [np.ones((1024, 1024)), np.ones((1, 1024)), np.ones((2, 2))]:
            _, perrC, _ = create_noise_maps(self.Fd, None, None, perr,
                                            None, None, self.Dd, None, None)
            with self.assertRaises(TypeError):
                build_dark(self.F, self.Dd, perrC, self.g, self.t)
            pass
        pass

    def test_invalid_D_range(self):
        """Invalid inputs caught as expected"""
        for perr in [-1*np.ones_like(self.Dd)]:
            _, _, perrD = create_noise_maps(self.Fd, None, None, self.Cd,
                                            None, None, perr, None, None)
            with self.assertRaises(TypeError):
                build_dark(self.F, perrD, self.C, self.g, self.t)
            pass
        pass


    def test_invalid_C_range(self):
        """Invalid inputs caught as expected"""
        for perr in [-1*np.ones_like(self.Cd)]:
            _, perrC, _ = create_noise_maps(self.Fd, None, None, perr,
                                            None, None, self.Dd, None, None)
            with self.assertRaises(TypeError):
                build_dark(self.F, self.D, perrC, self.g, self.t)
            pass
        pass


    def test_invalid_g(self):
        """Invalid inputs caught as expected"""
        for perr in [-1.5, None, 1j, 'txt', np.ones((1024,)),
                     np.ones((1024, 1024)), np.ones((1, 1024, 1024))]:
            with self.assertRaises(TypeError):
                build_dark(self.F, self.D, self.C, perr, self.t)
            pass
        pass


    def test_invalid_t(self):
        """Invalid inputs caught as expected"""
        for perr in [-1.5, None, 1j, 'txt', np.ones((1024,)),
                     np.ones((1024, 1024)), np.ones((1, 1024, 1024))]:
            with self.assertRaises(TypeError):
                build_dark(self.F, self.D, self.C, self.g, perr)
            pass
        pass


    def test_g_range_correct(self):
        """gain is valid >= 1 only"""

        for perr in [-10, -1, 0, 0.999]:
            with self.assertRaises(TypeError):
                build_dark(self.F, self.D, self.C, perr, self.t)
            pass

        for perr in [1, 1.5, 10]:
            build_dark(self.F, self.D, self.C, perr, self.t)
            pass
        pass



if __name__ == '__main__':
    unittest.main()
