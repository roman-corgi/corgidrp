import numpy as np
import os
from corgidrp import mocks, astrom
from corgidrp.l2b_to_l3 import create_wcs

def test_create_wcs():
    """
    Unit test of the create WCS function. The test checks that all WCS keywords (['CD{}_{}'], ['CRPIX{}'], ['CTYPE{}'], ['CDELT{}'], ['CRVAL{}']) exist in the updated ext_hdr 
    and that the values are consistent with those derived from the AstrometricCalibration file.
    """

    # create mock dataset (arbitrary northangle)
    field_path = os.path.join(os.path.dirname(__file__), "test_data", "JWST_CALFIELD2020.csv")
    mock_dataset = mocks.create_astrom_data(field_path, platescale=21.8, rotation=20, dither_pointings=2)

    # run the boresight calibration to get an AstrometricCalibration file
    astrom_cal = astrom.boresight_calibration(mock_dataset, field_path, find_threshold=200, find_distortion=True, position_error=0.5)

    # create the wcs
    updated_dataset = create_wcs(mock_dataset, astrom_cal)

    # check that all wcs keywords exist in the ext_hdrs of the mock dataset
    # and that the values are as expected from the AstrometricCalibration file
    platescale = astrom_cal.platescale
    northangle = astrom_cal.northangle
    ra_offset, dec_offset = astrom_cal.avg_offset

    for mock_frame, updated_frame in zip(mock_dataset, updated_dataset):
        roll_ang = mock_frame.pri_hdr['PA_APER']
        data = mock_frame.data
        image_shape = data.shape
        center_pixel = [(image_shape[1]-1) // 2, (image_shape[0]-1) // 2]
        target_ra, target_dec = mock_frame.pri_hdr['RA'], mock_frame.pri_hdr['DEC']

        pc = np.array([[-np.cos(np.radians(northangle + roll_ang)), np.sin(np.radians(northangle + roll_ang))], [np.sin(np.radians(northangle + roll_ang)), np.cos(np.radians(northangle + roll_ang))]])
        # assuming platescale is the same along both axes
        matrix = pc * (platescale * 0.001) / 3600.
        
        # gather expected values in a dictionary
        expected = {}
        expected['CD1_1'] = matrix[0,0]
        expected['CD1_2'] = matrix[0,1]
        expected['CD2_1'] = matrix[1,0]
        expected['CD2_2'] = matrix[1,1]

        expected['CRPIX1'] = center_pixel[0]
        expected['CRPIX2'] = center_pixel[1]

        expected['CTYPE1'] = 'RA---TAN'
        expected['CTYPE2'] = 'DEC--TAN'

        expected['CDELT1'] = (platescale * 0.001) / 3600  ## converting to degrees
        expected['CDELT2'] = (platescale * 0.001) / 3600

        expected['CRVAL1'] = target_ra - ra_offset      # the corrected target pointing based on astrom_cal
        expected['CRVAL2'] = target_dec - dec_offset

        # gather the wcs values from the updated dataset
        wcs = {}
        for key in expected.keys():
            wcs[key] = updated_frame.ext_hdr[key]

        # compare the expected dictionary to the updated dateset output
        assert wcs.items() == expected.items()
if __name__ == "__main__":
    test_create_wcs()