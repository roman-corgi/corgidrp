import os
import pytest
import corgidrp
import numpy as np
import corgidrp.detector as detector
from astropy.time import Time
from corgidrp.l2a_to_l2b import desmear
from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset, BadPixelMap

old_err_tracking = corgidrp.track_individual_errors

def test_desmear():
    # Tolerance for comparisons
    tol = 1e-12

    corgidrp.track_individual_errors = False

    print("Testing desmear step function")

    rowreadtime_sec = detector.get_rowreadtime_sec()

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
    prhd, exthd = create_default_headers()
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

    # Apply desmear correction
    dataset_desmear = desmear(dataset_smeared)

    assert type(dataset_desmear) == corgidrp.data.Dataset

    assert(np.max(np.abs(dataset_desmear.all_data[0] - unsmeared_frame)) < tol)

if __name__ == '__main__':
    test_desmear()
    
