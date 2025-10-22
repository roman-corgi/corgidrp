import pytest

from corgidrp.mocks import create_mock_polimage
from corgidrp.pol import calc_pol_p_and_pa_image


def test_calc_pol_p_and_pa_image(p_input=0.1, theta_input=10.0):
    """Test calc_pol_p_and_pa_image using a mock Stokes cube."""

    # Generate mock Stokes cube
    Image_polmock = create_mock_polimage(
        p=p_input,
        theta_deg=theta_input,
        roll_angles=[0],
        prisms=['POL0'],
        return_stokes=True
    )

    # Compute polarization products
    image_pol = calc_pol_p_and_pa_image(Image_polmock)

    p_map = image_pol.data[0]       # fractional polarization
    evpa_map = image_pol.data[3]    # EVPA

    assert p_map == pytest.approx(p_input)
    assert evpa_map == pytest.approx(theta_input)


if __name__ == "__main__":
    test_calc_pol_p_and_pa_image()
