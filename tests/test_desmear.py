import os
import pytest
import corgidrp
import numpy as np
import corgidrp.detector as detector
from astropy.time import Time
from corgidrp.l2a_to_l2b import desmear
from corgidrp.mocks import create_default_L1_headers
from corgidrp.data import Image, Dataset, DetectorParams

old_err_tracking = corgidrp.track_individual_errors

def test_desmear():
    # Tolerance for comparisons
    tol = 1e-12

    corgidrp.track_individual_errors = False

    print("Testing desmear step function")

    detector_params = DetectorParams({}, date_valid=Time("2023-11-01 00:00:00"))
    rowreadtime_sec = detector_params.params['ROWREADT']

    #make a flux map
    size = 1024
    background_flux = 10
    foreground_flux = 100
    xx, yy = np.mgrid[:size, :size]
    circle = ((xx - size//2)**2 + (yy - size//2)**2 ) < (size//4)**2
    flux = background_flux * np.ones([size,size]) + (foreground_flux - background_flux) * circle

    #make a truth frame
    err = np.ones([1024,1024]) *0.5
    dq = np.zeros([1024,1024], dtype = np.uint16)
    prhd, exthd = create_default_L1_headers()
    e_t=exthd['EXPTIME']
    unsmeared_frame = e_t*flux

    #simulate the smearing
    smear = np.zeros_like(flux)
    m = len(smear)
    for r in range(m):
        columnsum = 0
        for i in range(r+1):
            columnsum = columnsum + rowreadtime_sec*flux[i,:] 
        smear[r,:] = columnsum
    smeared_frame = unsmeared_frame + smear

    image1 = Image(smeared_frame, pri_hdr = prhd, ext_hdr = exthd, err = err,         dq = dq)
    dataset_smeared = Dataset([image1])
    
    assert type(dataset_smeared) == corgidrp.data.Dataset

    # check the header keyword hasn't been toggled yet
    for frame in dataset_smeared:
        assert not frame.ext_hdr['DESMEAR']

    # Apply desmear correction
    dataset_desmear = desmear(dataset_smeared, detector_params)

    assert type(dataset_desmear) == corgidrp.data.Dataset

    assert(np.max(np.abs(dataset_desmear.all_data[0] - unsmeared_frame)) < tol)

    # check the header keyword is toggled
    for frame in dataset_desmear:
        assert frame.ext_hdr['DESMEAR']

if __name__ == '__main__':
    test_desmear()
    
