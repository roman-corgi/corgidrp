# tests/test_compute_QphiUphi.py
import numpy as np
from astropy.io.fits import Header
from corgidrp.data import Image
from corgidrp.mocks import create_mock_IQUV_image
from corgidrp.l4_to_tda import compute_QphiUphii


def test_compute_QphiUphi_center_correct():
    img = create_mock_IQUV_image()
    res = compute_QphiUphi(img)

    # check shape
    assert res.data.shape[0] == 6, "Output should have 6 planes (I,Q,U,V,Q_phi,U_phi)"

    # check U_phi ~ 0 when center is correct
    U_phi = res.data[5]
    assert np.allclose(U_phi, 0.0, atol=1e-6), "U_phi should be ~0 for correct center"


def test_compute_QphiUphi_center_wrong():
    img = create_mock_IQUV_image()
    # overwrite header center with wrong value
    img.ext_hdr["STARLOCX"] += 5.0
    img.ext_hdr["STARLOCY"] += 5.0

    res = compute_QphiUphi(img)

    U_phi = res.data[5]
    assert not np.allclose(U_phi, 0.0, atol=1e-6), "U_phi should be nonzero for wrong center"


def test_compute_QphiUphi_err_shape():
    img = create_mock_IQUV_image()
    res = compute_QphiUphi(img)

    # check that err array is consistent with data
    assert res.err.shape == res.data.shape, "err should have the same shape as data"
    assert res.dq.shape == res.data.shape, "dq should have the same shape as data"

def test_dq_propagation():
    """
    Verify that the dq of Q_phi and U_phi propagates as the bitwise-OR of
    the input dq for Q and U. We set distinct bits on all pixels of Q and U
    so the expected OR relationship holds regardless of geometry.
    """
    img = create_mock_IQUV_image()

    # Expect at least (I, Q, U, V) planes in the input dq
    assert img.dq.shape[0] >= 4, "mock image should have I,Q,U,V planes"

    # Distinct bits for Q and U (non-overlapping)
    BIT_Q = 1 << 2
    BIT_U = 1 << 5

    # Add bits to Q and U while preserving existing dq
    dq_mod = img.dq.copy()
    dq_mod[1] = dq_mod[1] | BIT_Q  # Q plane
    dq_mod[2] = dq_mod[2] | BIT_U  # U plane
    img.dq = dq_mod

    # Compute Q_phi and U_phi
    res = compute_QphiUphi(img)

    # Expect (I, Q, U, V, Q_phi, U_phi) -> 6 planes
    assert res.dq.shape[0] == 6, "Output dq should have 6 planes"

    expected_or = img.dq[1] | img.dq[2]

    # Q_phi dq should include bits from Q and U (bitwise OR)
    np.testing.assert_array_equal(
        res.dq[4] & (BIT_Q | BIT_U),
        expected_or & (BIT_Q | BIT_U),
        err_msg="Q_phi dq should include bits from Q and U (OR)."
    )

    # U_phi dq should include bits from Q and U (bitwise OR)
    np.testing.assert_array_equal(
        res.dq[5] & (BIT_Q | BIT_U),
        expected_or & (BIT_Q | BIT_U),
        err_msg="U_phi dq should include bits from Q and U (OR)."
    )

if __name__ == "__main__":
    test_compute_QphiUphi_center_correct()
    test_compute_QphiUphi_center_wrong()
    test_compute_QphiUphi_err_shape()
    test_dq_propagation()
