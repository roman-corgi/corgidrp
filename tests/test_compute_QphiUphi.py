# tests/test_compute_QphiUphi.py
import numpy as np
from astropy.io.fits import Header
from corgidrp.data import Image
from corgidrp.l4_to_tda import compute_QphiUphi

def make_mock_IQUV_image(n=64, m=64, fwhm=20, amp=1.0, pfrac=0.1, bg=0.0):
    """
    Create a mock Image with [I, Q, U, V] planes for testing.

    Args:
        n (int): Image height (pixels).
        m (int): Image width (pixels).
        fwhm (float): FWHM of the Gaussian PSF used for I.
        amp (float): Peak amplitude of the Gaussian PSF.
        pfrac (float): Polarization fraction. Q, U are scaled by this fraction.
        bg (float): Background level added to the image.

    Returns:
        Image: Mock Image object with data of shape [4, n, m], err and dq arrays included.
    """


    y, x = np.mgrid[0:n, 0:m]
    x0, y0 = 0.5*(m-1), 0.5*(n-1)

    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    r2 = (x-x0)**2 + (y-y0)**2
    I = bg + amp * np.exp(-0.5*r2/sigma**2)

    phi = np.arctan2(y-y0, x-x0)
    Q = -pfrac * I * np.cos(2*phi)
    U = -pfrac * I * np.sin(2*phi)
    V = np.zeros_like(I)

    cube = np.stack([I, Q, U, V], axis=0)

    pri_hdr = Header()
    ext_hdr = Header()
    ext_hdr["STARLOCX"] = float(x0)
    ext_hdr["STARLOCY"] = float(y0)

    return Image(
        cube,
        pri_hdr=pri_hdr,
        ext_hdr=ext_hdr,
        err=np.zeros_like(cube),
        dq=np.zeros(cube.shape, dtype=np.uint16),
        err_hdr=Header(),
        dq_hdr=Header(),
    )

def test_compute_QphiUphi_center_correct():
    img = make_mock_IQUV_image()
    res = compute_QphiUphi(img)

    # check shape
    assert res.data.shape[0] == 6, "Output should have 6 planes (I,Q,U,V,Q_phi,U_phi)"

    # check U_phi ~ 0 when center is correct
    U_phi = res.data[5]
    assert np.allclose(U_phi, 0.0, atol=1e-6), "U_phi should be ~0 for correct center"


def test_compute_QphiUphi_center_wrong():
    img = make_mock_IQUV_image()
    # overwrite header center with wrong value
    img.ext_hdr["STARLOCX"] += 5.0
    img.ext_hdr["STARLOCY"] += 5.0

    res = compute_QphiUphi(img)

    U_phi = res.data[5]
    assert not np.allclose(U_phi, 0.0, atol=1e-6), "U_phi should be nonzero for wrong center"


def test_compute_QphiUphi_err_shape():
    img = make_mock_IQUV_image()
    res = compute_QphiUphi(img)

    # check that err array is consistent with data
    assert res.err.shape == res.data.shape, "err should have the same shape as data"
    assert res.dq.shape == res.data.shape, "dq should have the same shape as data"


if __name__ == "__main__":
    test_compute_QphiUphi_center_correct()
    test_compute_QphiUphi_center_wrong()
    test_compute_QphiUphi_err_shape()
