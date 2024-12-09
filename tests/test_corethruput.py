
# Use
# with fits.open(unocc_psf_filepath) as hdulist:
# assert A == pytest.approx(min, max)
# with pytest.raises(ValueError):

import pytest
import os
import numpy as np
from astropy.io import fits

from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset
import corgidrp.corethroughput as ct

ct_filepath = os.path.join(os.path.dirname(__file__), 'test_data')
# Mock error
err = np.ones([1,1024,1024]) * 0.5
# Default headers
prhd, exthd = create_default_headers()

def setup_module():
    """
    Create a dataset with some representative psf responses. 
    """
    # corgidrp dataset
    global dataset_psf
    # arbitrary set of PSF positions to be tested in EXCAM pixels referred to (0,0)
    global psf_position_x, psf_position_y
    psf_position_x = [512, 522, 532, 542, 552, 562, 522, 532, 542, 552, 562]
    psf_position_y = [512, 522, 532, 542, 552, 562, 502, 492, 482, 472, 462]

    data_unocc = np.zeros([1024, 1024])
    # unocculted PSF
    unocc_psf_filepath = os.path.join(ct_filepath, 'hlc_os11_no_fpm.fits')
    # os11 unocculted PSF is sampled at the same pixel pitch as EXCAM
    unocc_psf = fits.getdata(unocc_psf_filepath)
    # Insert PSF at its location
    idx_0_0 = psf_position_x[0] - unocc_psf.shape[0]
    idx_0_1 = idx_0_0 + unocc_psf.shape[0]
    idx_1_0 = psf_position_y[0] - unocc_psf.shape[1]
    idx_1_1 = idx_1_0 + unocc_psf.shape[1]
    data_unocc[idx_0_0:idx_0_1, idx_1_0:idx_1_1] = unocc_psf
    
    i_psf = 0
    data = [data_unocc]
    for x_psf in psf_position_x[1:]:
        for y_psf in psf_position_y[1:]:
            
            i_psf += 1

    breakpoint()
#image1 = Image(data,pri_hdr = prhd, ext_hdr = exthd, err = err)
#dataset=Dataset([image1])

def test_psf_pix_and_ct():
    """
    Test 1090881 - Given a core throughput dataset consisting of M clean frames
    (nominally 1024x1024) taken at different FSM positions, the CTC GSW shall
    estimate the pixel location and core throughput of each PSF.

    NOTE: the list of M clean frames may be a subset of the frames collected during
    core throughput data collection, to allow for the removal of outliers.
    """


def test_unocc():
    """ Test array position of the unocculted PSF in the data array """
    
    # pass test with the dataset defined by setup_module() 

    # do not pass if the order is different (swap first two)
    
    
if __name__ == '__main__':
    test_psf_pix_and_ct()




