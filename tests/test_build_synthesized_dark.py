import pytest

import numpy as np

from corgidrp.darks import build_synthesized_dark
from corgidrp.mocks import create_noise_maps, create_dark_calib_files
from corgidrp.detector import unpack_geom, detector_areas

im_rows, im_cols, _ = unpack_geom('SCI', 'image', detector_areas)
rows = detector_areas['SCI']['frame_rows']
cols = detector_areas['SCI']['frame_cols']

Fd = np.ones((rows, cols))
Dd = 3/3600*np.ones((rows, cols))
Cd = 0.02*np.ones((rows, cols))

Ferr = np.zeros((rows, cols))
Derr = np.zeros((rows, cols))
Cerr = np.zeros((rows, cols))
Fdq = Ferr.copy().astype(int)
Ddq = Derr.copy().astype(int)
Cdq = Cerr.copy().astype(int)
noise_maps = create_noise_maps(Fd, Ferr, Fdq, Cd,
                                            Cerr, Cdq, Dd, Derr, Ddq)
F = noise_maps.FPN_map
C = noise_maps.CIC_map
D = noise_maps.DC_map
# just need some a dataset for input
dataset = create_dark_calib_files()
# values used in create_dark_calib_files()
g = 1
t = 60
for fr in dataset.frames:
    fr.ext_hdr['KGAIN'] = 7

# convenience function for these tests
def reset_g_t(d, g, t):
    '''
    For a dataset d, reset the EM gain to g and the exposure time to t for
    all frames in d.

    Args:
        d (corgidrp.data.Dataset):  input dataset
        g (float): desired EM gain
        t (float): desired exposure time

    Returns:
        dset (corgidrp.data.Dataset):  dataset with the EM gain and exposure time reset
    '''
    dset = d.copy()
    for fr in dset.frames:
        fr.ext_hdr['CMDGAIN'] = g
        fr.ext_hdr['EXPTIME'] = t

    return dset

def main():

    def test_success():
        """Good inputs complete as expected"""
        build_synthesized_dark(dataset, noise_maps)
        build_synthesized_dark(dataset, noise_maps, detector_areas, full_frame=True)
        pass

    test_success()

    def test_output_size():
        """output is correct size"""
        M = build_synthesized_dark(dataset, noise_maps)
        assert(M.data.shape == (im_rows, im_cols))
        pass

    test_output_size()

    def test_exact_case():
        """Exact case produces expected result.  Use full frame this time."""
        tol = 1e-13

        Fd = 5*np.ones((rows, cols))
        Ferr = 0.1*Fd
        Fdq = np.zeros((rows, cols)).astype(int)
        Fdq[3,3] = 1
        Dd = 1/7*np.ones((rows, cols))
        Derr = 0.1*Dd
        Ddq = np.zeros((rows, cols)).astype(int)
        Ddq[3,3] = 1 # Fdq has this one, too
        Ddq[2,2] = 1
        Cd = 1*np.ones((rows, cols))
        Cerr = 0.1*Cd
        Cdq = np.zeros((rows, cols)).astype(int)
        Cdq[1,2] = 1
        g = 5
        t = 7
        target = 3*np.ones((rows, cols))
        exp_err = np.sqrt(Derr**2*g**2*t**2 + Cerr**2*g**2 + Ferr**2)/g
        exp_dq = np.zeros((rows, cols))
        exp_dq[3,3] = 1
        exp_dq[2,2] = 1
        exp_dq[1,2] = 1
        n_maps = create_noise_maps(Fd, Ferr, Fdq, Cd, Cerr, Cdq, Dd, Derr, Ddq)
        dset = reset_g_t(dataset, g, t)

        M = build_synthesized_dark(dset, n_maps, full_frame=True)
        assert(np.max(np.abs(M.data - target)) < tol)
        assert(np.max(np.abs(M.err - exp_err)) < tol)
        assert(np.max(np.abs(M.dq - exp_dq)) < tol)
        assert(M.err_hdr['BUNIT'] == 'detected electrons')
        assert(M.ext_hdr['EXPTIME'] == t)
        assert(M.ext_hdr['CMDGAIN'] == g)
        assert(M.ext_hdr['DATATYPE'] == 'Dark')
        assert(M.data.shape == (rows, cols))
        assert(M.ext_hdr['NAXIS1'] == rows)
        assert(M.ext_hdr['NAXIS2'] == cols)
        assert(M.ext_hdr['DRPNFILE'] == 1) #made from 1 DetectorNoiseMaps file
        assert(M.filename == '0_DetectorNoiseMaps_dark.fits')
        assert('EM gain = '+str(g) in str(M.ext_hdr['HISTORY']))
        assert('exptime = '+str(t) in str(M.ext_hdr['HISTORY']))

        M_copy = M.copy()
        assert(np.max(np.abs(M_copy.data - target)) < tol)
        assert(np.max(np.abs(M_copy.err - exp_err)) < tol)
        assert(np.max(np.abs(M_copy.dq - exp_dq)) < tol)
        assert(M_copy.err_hdr['BUNIT'] == 'detected electrons')
        assert(M_copy.ext_hdr['EXPTIME'] == t)
        assert(M_copy.ext_hdr['CMDGAIN'] == g)
        assert(M_copy.ext_hdr['DATATYPE'] == 'Dark')
        assert(M_copy.data.shape == (rows, cols))
        assert(M_copy.ext_hdr['NAXIS1'] == rows)
        assert(M_copy.ext_hdr['NAXIS2'] == cols)
        assert(M_copy.ext_hdr['DRPNFILE'] == 1) #made from 1 DetectorNoiseMaps file
        assert(M_copy.filename == '0_DetectorNoiseMaps_dark.fits')
        assert('EM gain = '+str(g) in str(M_copy.ext_hdr['HISTORY']))
        assert('exptime = '+str(t) in str(M_copy.ext_hdr['HISTORY']))
        pass

    test_exact_case()

    def test_gain_goes_as_1overg():
        """change in dark goes as 1/g"""
        tol = 1e-13
        noise_maps0 = create_noise_maps(0*Fd, Ferr, Fdq, Cd,
                                            Cerr, Cdq, Dd, Derr, Ddq)
        dset = reset_g_t(dataset, g, t)

        F0 = build_synthesized_dark(reset_g_t(dataset, 1, t), noise_maps0)
        M1 = build_synthesized_dark(reset_g_t(dataset, 1, t), noise_maps)
        M2 = build_synthesized_dark(reset_g_t(dataset, 2, t), noise_maps)
        M4 = build_synthesized_dark(reset_g_t(dataset, 4, t), noise_maps)
        dg1 = M1.data-F0.data
        dg2 = M2.data-F0.data
        dg4 = M4.data-F0.data

        assert(np.max(np.abs(dg2 - dg1/2)) < tol)
        assert(np.max(np.abs(dg4 - dg2/2)) < tol)
        pass

    test_gain_goes_as_1overg()

    def test_exptime_goes_as_t():
        """change in dark goes as t"""
        tol = 1e-13

        M0 = build_synthesized_dark(reset_g_t(dataset, g, 0), noise_maps)
        M2 = build_synthesized_dark(reset_g_t(dataset, g, 2), noise_maps)
        M4 = build_synthesized_dark(reset_g_t(dataset, g, 4), noise_maps)
        dg2 = M2.data-M0.data
        dg4 = M4.data-M0.data

        assert(np.max(np.abs(dg4 - dg2*2)) < tol)
        pass

    test_exptime_goes_as_t()

    def test_c_doesnt_change_with_g_or_t():
        """F = 0 and D = 0 implies C is constant"""
        tol = 1e-13
        noise_maps0 = create_noise_maps(0*Fd, Ferr, Fdq, Cd,
                                            Cerr, Cdq, 0*Dd, Derr, Ddq)

        M = build_synthesized_dark(reset_g_t(dataset, 1, t), noise_maps0)
        G2 = build_synthesized_dark(reset_g_t(dataset, 2, t), noise_maps0)
        G4 = build_synthesized_dark(reset_g_t(dataset, 4, t), noise_maps0)
        T2 = build_synthesized_dark(reset_g_t(dataset, g, 2), noise_maps0)
        T4 = build_synthesized_dark(reset_g_t(dataset, g, 4), noise_maps0)
        dg2 = G2.data-M.data
        dg4 = G4.data-M.data
        dt2 = T2.data-M.data
        dt4 = T4.data-M.data

        assert(np.max(np.abs(dg2)) < tol)
        assert(np.max(np.abs(dg4)) < tol)
        assert(np.max(np.abs(dt2)) < tol)
        assert(np.max(np.abs(dt4)) < tol)
        pass

    test_c_doesnt_change_with_g_or_t()

    def test_bias_subtracted():
        """check there is no bias when all three noise terms are 0"""
        noise_maps0 = create_noise_maps(0*Fd, Ferr, Fdq, 0*Cd,
                                            Cerr, Cdq, 0*Dd, Derr, Ddq)

        M = build_synthesized_dark(reset_g_t(dataset, 1, t), noise_maps0)
        G2 = build_synthesized_dark(reset_g_t(dataset, 2, t), noise_maps0)
        G4 = build_synthesized_dark(reset_g_t(dataset, 4, t), noise_maps0)
        T2 = build_synthesized_dark(reset_g_t(dataset, g, 2), noise_maps0)
        T4 = build_synthesized_dark(reset_g_t(dataset, g, 4), noise_maps0)

        assert((M.data == 0).all())
        assert((G2.data == 0).all())
        assert((G4.data == 0).all())
        assert((T2.data == 0).all())
        assert((T4.data == 0).all())

        pass

    test_bias_subtracted()

    def test_invalid_D_range():
        """Invalid inputs caught as expected"""
        for perr in [-1*np.ones_like(Dd)]:
            nm = create_noise_maps(Fd, Ferr, Fdq, Cd,
                                            Cerr, Cdq, perr, Derr, Ddq)
            with pytest.raises(TypeError):
                build_synthesized_dark(dataset, nm)
            pass
        pass

    test_invalid_D_range()

    def test_invalid_C_range():
        """Invalid inputs caught as expected"""
        for perr in [-1*np.ones_like(Cd)]:
            nm = create_noise_maps(Fd, Ferr, Fdq, perr,
                                            Cerr, Cdq, Dd, Derr, Ddq)
            with pytest.raises(TypeError):
                build_synthesized_dark(dataset, nm)
            pass
        pass

    test_invalid_C_range()

    def test_g_range_correct():
        """gain is valid >= 1 only"""

        for perr in [-10, -1, 0, 0.999]:
            with pytest.raises(TypeError):
                build_synthesized_dark(reset_g_t(dataset, perr, t), noise_maps)
            pass

        for perr in [1, 1.5, 10]:
            build_synthesized_dark(reset_g_t(dataset, perr, t), noise_maps)
            pass
        pass

    test_g_range_correct()


if __name__ == '__main__':
    main()
