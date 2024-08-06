import os
import glob
import pytest
import numpy as np
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.detector as detector
import corgidrp.l2a_to_l2b as l2a_to_l2b
from corgidrp.darks import build_trad_dark

old_err_tracking = corgidrp.track_individual_errors
# use default parameters
detector_params = data.DetectorParams({})

# make a mock DetectorNoiseMaps instance (to get the bias offset input)
im_rows, im_cols, _ = detector.unpack_geom('SCI', 'image')
rows = detector.detector_areas['SCI']['frame_rows']
cols = detector.detector_areas['SCI']['frame_cols']

Fd = np.ones((rows, cols))
Dd = 3/3600*np.ones((rows, cols))
Cd = 0.02*np.ones((rows, cols))

Ferr = np.zeros((rows, cols))
Derr = np.zeros((rows, cols))
Cerr = np.zeros((rows, cols))
Fdq = Ferr.copy().astype(int)
Ddq = Derr.copy().astype(int)
Cdq = Cerr.copy().astype(int)
noise_maps = mocks.create_noise_maps(Fd, Ferr, Fdq, Cd,
                                            Cerr, Cdq, Dd, Derr, Ddq)

def test_dark_sub():
    """
    Generate mock input data and pass into dark subtraction function
    """
    corgidrp.track_individual_errors = True # this test uses individual error components

    ###### create simulated data
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    mocks.create_dark_calib_files(filedir=datadir)

    ####### test data architecture
    dark_filenames = glob.glob(os.path.join(datadir, "simcal_dark*.fits"))

    dark_dataset = data.Dataset(dark_filenames)

    assert len(dark_dataset) == 10

    # check that data is consistently modified
    dark_dataset.all_data[0,0,0] = 0
    assert dark_dataset[0].data[0,0] == 0

    dark_dataset[0].data[0,0] = 1
    assert dark_dataset.all_data[0,0,0] == 1

    ###### create dark
    dark_frame = build_trad_dark(dark_dataset, detector_params, detector_regions=None, full_frame=True)

    # check the level of dark current is approximately correct; leave off last row, telemetry row
    assert np.mean(dark_frame.data[:-1]) == pytest.approx(150, abs=1e-2)

    # leave out telemetry row of full frame
    dark_dataset_notelem = dark_dataset.all_data[:, 0:-1, :]
    assert np.array_equal(np.std(dark_dataset_notelem, axis = 0)/np.sqrt(len(dark_dataset_notelem)), dark_frame.err[0][0:-1, :])

    # save dark
    calibdir = os.path.join(os.path.dirname(__file__), "testcalib")
    dark_filename = "sim_dark_calib.fits"
    if not os.path.exists(calibdir):
        os.mkdir(calibdir)
    dark_frame.save(filedir=calibdir, filename=dark_filename)

    ###### perform dark subtraction
    # load in the dark
    dark_filepath = os.path.join(calibdir, dark_filename)
    new_dark = data.Dark(dark_filepath)
    # subtract darks from itself; also saves to testcalib folder
    darkest_dataset = l2a_to_l2b.dark_subtraction(dark_dataset, new_dark, outputdir=calibdir)
    assert(dark_filename in str(darkest_dataset[0].ext_hdr["HISTORY"]))

    # check the level of the dataset is now approximately 0, leaving off telemetry row
    assert np.mean(darkest_dataset.all_data[:,:-1,:]) == pytest.approx(0, abs=1e-2)

    # check that image area option works
    dark_frame = build_trad_dark(dark_dataset, detector_params, detector_regions=None, full_frame=False)
    # compare, but first get image area from input:
    dark_dataset_im = dark_dataset.copy()
    dark_im_frames = []
    for fr in dark_dataset_im.all_data:
        fr = detector.slice_section(fr, 'SCI', 'image')
        dark_im_frames.append(fr)
    dark_im_frames = np.stack(dark_im_frames)
    # check that the error is determined correctly (no telemetry row in image area)
    assert np.array_equal(np.std(dark_im_frames, axis = 0)/np.sqrt(len(dark_im_frames)), dark_frame.err[0])

    # check the propagated errors
    assert darkest_dataset[0].err_hdr["Layer_2"] == "dark_error"
    assert(np.mean(darkest_dataset.all_err) == pytest.approx(np.mean(dark_frame.err), abs = 1e-2))
    #print("mean of all data:", np.mean(darkest_dataset.all_data))
    #print("mean of all errors:", np.mean(darkest_dataset.all_err))
    assert darkest_dataset[0].ext_hdr["BUNIT"] == "photoelectrons"
    assert darkest_dataset[0].err_hdr["BUNIT"] == "photoelectrons"
    #print(darkest_dataset[0].ext_hdr)

    # If too many masked in a stack for a given pixel, warning raised. Checks
    # that dq values are as expected, too.
    ds = dark_dataset.copy()
    # tag as bad pixel all the
    # way through for one pixel (7,8)
    # And mask (10,12) to get flag value of 256
    ds.all_dq[:,7,8] = 4
    ds.all_dq[:int(1+len(ds)/2),10,12] = 2

    with pytest.warns(UserWarning):
        master_dark = build_trad_dark(ds, detector_params, full_frame=True)
    assert master_dark.dq[7,8] == 1
    assert master_dark.dq[10,12] == 256

    # now input a DetectorNoiseMaps instance and subtract dark from itself; here outputdir will do nothing since this is not a Dark instance
    EMgain = 10
    exptime = 4
    frame = (noise_maps.FPN_map + noise_maps.CIC_map*EMgain + noise_maps.DC_map*exptime*EMgain)/EMgain
    prihdr, exthdr = mocks.create_default_headers()
    image_frame = data.Image(frame, prihdr, exthdr)
    image_frame.ext_hdr['CMDGAIN'] = EMgain
    image_frame.ext_hdr['EXPTIME'] = exptime
    image_frame.ext_hdr['KGAIN'] = 7
    dataset_from_noisemap = data.Dataset([image_frame])
    nm_dataset0 = l2a_to_l2b.dark_subtraction(dataset_from_noisemap, noise_maps, outputdir=calibdir)
    # check the level of the dataset is now approximately 0, leaving off telemetry row
    assert np.mean(nm_dataset0.all_data[:,:-1,:]) == pytest.approx(0, abs=1e-2)

    corgidrp.track_individual_errors = old_err_tracking

if __name__ == "__main__":
    test_dark_sub()
