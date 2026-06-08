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
    err = np.ones([1200,2200]) *0.5
    dq = np.zeros([1200,2200], dtype = np.uint16)
    prhd, exthd = create_default_L1_headers()
    e_t=exthd['EXPTIME']
    unsmeared_image = e_t*flux
    unsmeared_frame = detector.embed(e_t*flux, 'SCI', 'image')

    #simulate the smearing
    smear = np.zeros_like(flux)
    m = len(smear)
    for r in range(m):
        columnsum = 0
        for i in range(r+1):
            columnsum = columnsum + rowreadtime_sec*flux[i,:] 
        smear[r,:] = columnsum
    smeared_image = unsmeared_image + smear

    smeared_frame = detector.embed(smeared_image, 'SCI', 'image')
    image1 = Image(smeared_frame, pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq)
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
    
    # now add a cosmic rays, including at corners
    dq_im = detector.slice_section(dq, 'SCI', 'image')
    smeared_image[0:2, 0:2] += 200000
    dq_im[0:2, 0:2] = 160 # cosmic ray (128) and saturated (32)
    dq_im[4,3] = 32 # merely saturated
    # cosmic rays at other corners
    smeared_image[-2:, -2:] += 200000
    dq_im[-2:, -2:] = 128 
    smeared_image[-2:, 0:2] += 200000
    dq_im[-2:, 0:2] = 160
    smeared_image[0:2, -2:] += 200000
    dq_im[0:2, -2:] = 160
    # cosmic ray in circle area but not marked in DQ
    smeared_image[500, 500] += 200000

    # truth frame in this case
    unsmeared_image[0:2, 0:2] += 200000
    unsmeared_image[-2:, -2:] += 200000
    unsmeared_image[-2:, 0:2] += 200000
    unsmeared_image[0:2, -2:] += 200000
    unsmeared_image[500, 500] += 200000
    unsmeared_frame = detector.embed(unsmeared_image, 'SCI', 'image')

    smeared_frame = detector.embed(smeared_image, 'SCI', 'image')
    image1 = Image(smeared_frame, pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq)
    dataset_smeared = Dataset([image1])

    # Apply desmear correction
    dataset_desmear = desmear(dataset_smeared, detector_params)
    # desmearing should be incorrect b/c of the cosmic ray in the circle that wasn't dq'ed
    assert(np.max(np.abs(dataset_desmear.all_data[0] - unsmeared_frame)) > tol)

    # now account for cosmic in dq so that desmearing can account for the cosmic 
    dq_im[500, 500] = 160
    image1 = Image(smeared_frame, pri_hdr = prhd, ext_hdr = exthd, err = err, dq = dq)
    dataset_smeared = Dataset([image1])
    dataset_desmear = desmear(dataset_smeared, detector_params)
    # slightly bigger discrepancy than previous tolerance b/c cosmic-flagged pixels take on the 
    # value of nearest neighbors, which may be in a different row, which would include a different 
    # amount of smear than the original row
    assert(np.max(np.abs(dataset_desmear.all_data[0] - unsmeared_frame)) < 1e-7)

    # test same thing with auto-decide on
    dataset_desmear = desmear(dataset_smeared, detector_params)
    assert(np.max(np.abs(dataset_desmear.all_data[0] - unsmeared_frame)) < 1e-7)

    # now test a frame that shouldn't be desmeared
    new_rn_thresh = e_t*foreground_flux
    # set the read noise so that foreground of image only 2 times read noise
    detector_params.params['READ_N'] = new_rn_thresh/2
    dataset_desmear = desmear(dataset_smeared, detector_params, auto_decide=True, rn_factor=5)
    # using default factor of 5 above read noise should mean no desmearing here:
    assert np.array_equal(dataset_desmear[0].data, dataset_smeared[0].data)
    assert dataset_desmear[0].ext_hdr['DESMEAR'] == False
    # now setting signal_factor to something < 2:
    dataset_desmear = desmear(dataset_smeared, detector_params, auto_decide=True, rn_factor=1.5)
    # desmearing should have happened:
    assert(np.max(np.abs(dataset_desmear.all_data[0] - unsmeared_frame)) < 1e-7)
    # nbins should be >= 3 in order to find a local minimum in the histogram
    with pytest.raises(ValueError):
        desmear(dataset_smeared, detector_params, auto_decide=True, nbins=2, rn_factor=1.5)
    print('Passed')
if __name__ == '__main__':
    test_desmear()
    
