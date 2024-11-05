import glob
import os

import corgidrp
import corgidrp.data as data
from corgidrp.l1_to_l2a import prescan_biassub
import corgidrp.mocks as mocks
from corgidrp.detector import detector_areas, unpack_geom, imaging_area_geom

import numpy as np
import yaml
from astropy.io import fits
from pathlib import Path

from pytest import approx

old_err_tracking = corgidrp.track_individual_errors

# make a mock DetectorNoiseMaps instance (to get the bias offset input)
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
noise_maps = mocks.create_noise_maps(Fd, Ferr, Fdq, Cd,
                                            Cerr, Cdq, Dd, Derr, Ddq)

# Expected output image shapes
shapes = {
    'SCI' : {
        True : (1200,2200),
        False : (1024,1024)
    },
    'ENG' : {
        True: (2200,2200),
        False : (1024,1024)
    }
}

# Copy-pasted II&T code

# Metadata code from https://github.com/roman-corgi/cgi_iit_drp/blob/main/proc_cgi_frame_NTR/proc_cgi_frame/read_metadata.py

class ReadMetadataException(Exception):
    """Exception class for read_metadata module."""

# Set up to allow the metadata.yaml in the repo be the default
here = Path(os.path.dirname(os.path.abspath(__file__)))
meta_path = Path(here,'test_data','metadata.yaml')

class Metadata(object):
    """ II&T pipeline class to store metadata.

    B Nemati and S Miller - UAH - 03-Aug-2018

    Args:
        meta_path (str): Full path of metadta yaml.

    Attributes:
        data (dict):
            Data from metadata file.
        geom (SimpleNamespace):
            Geometry specific data.
    """

    def __init__(self, meta_path=meta_path):
        self.meta_path = meta_path

        self.data = self.get_data()
        self.frame_rows = self.data['frame_rows']
        self.frame_cols = self.data['frame_cols']
        self.geom = self.data['geom']

    def get_data(self):
        """Read yaml data into dictionary.

        Returns:
            data (dict): Metadata dictionary.
        """
        with open(self.meta_path, 'r') as stream:
            data = yaml.safe_load(stream)
        return data

    def slice_section(self, frame, key):
        """Slice 2d section out of frame.

        Args:
            frame (array_like):
                Full frame consistent with size given in frame_rows, frame_cols.
            key (str):
                Keyword referencing section to be sliced; must exist in geom.

        Returns:
            section (array_like): Section of frame
        """
        rows, cols, r0c0 = self._unpack_geom(key)

        section = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
        if section.size == 0:
            raise ReadMetadataException('Corners invalid')
        return section

    def _unpack_geom(self, key):
        """Safely check format of geom sub-dictionary and return values.

        Args:
            key (str): Keyword referencing section to be sliced; must exist in geom.

        Returns:
            rows (int): Number of rows in section.
            cols (int): Number of columns in section.
            r0c0 (tuple): Initial row and column of section.
        """
        coords = self.geom[key]
        rows = coords['rows']
        cols = coords['cols']
        r0c0 = coords['r0c0']

        return rows, cols, r0c0

    #added in from MetadataWrapper
    def _imaging_area_geom(self):
        """Return geometry of imaging area in reference to full frame.

        Returns:
            rows_im (int): Number of rows corresponding to image frame.
            cols_im (int): Number of columns in section.
            r0c0_im (tuple): Initial row and column of section.
        """

        _, cols_pre, _ = self._unpack_geom('prescan')
        _, cols_serial_ovr, _ = self._unpack_geom('serial_overscan')
        rows_parallel_ovr, _, _ = self._unpack_geom('parallel_overscan')
        #_, _, r0c0_image = self._unpack_geom('image')
        fluxmap_rows, _, r0c0_image = self._unpack_geom('image')

        rows_im = self.frame_rows - rows_parallel_ovr
        cols_im = self.frame_cols - cols_pre - cols_serial_ovr
        r0c0_im = r0c0_image.copy()
        r0c0_im[0] = r0c0_im[0] - (rows_im - fluxmap_rows)

        return rows_im, cols_im, r0c0_im

    def imaging_slice(self, frame):
        """Select only the real counts from full frame and exclude virtual.

        Use this to transform mask and embed from acting on the full frame to
        acting on only the image frame.

        Args:
            frame (array_like):
                Full frame consistent with size given in frame_rows, frame_cols.

        Returns:
            slice (array_like):
                Science image area of full frame.
        """
        rows, cols, r0c0 = self._imaging_area_geom()

        slice = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]

        return slice

# EMCCDFrame code from https://github.com/roman-corgi/cgi_iit_drp/blob/main/proc_cgi_frame_NTR/proc_cgi_frame/gsw_emccd_frame.py#L9

class EMCCDFrameException(Exception):
    """Exception class for emccd_frame module."""

class EMCCDFrame:
    """Get data from EMCCD frame and subtract the bias and bias offset.

    S Miller - UAH - 16-April-2019

    Args:
        frame_dn (array_like):
            Raw EMCCD full frame (DN).
        meta (instance):
            Instance of Metadata class containing detector metadata.
        fwc_em (float):
            Detector EM gain register full well capacity (DN).
        fwc_pp (float):
            Detector image area per-pixel full well capacity (DN).
        em_gain (float):
            Gain from EM gain register, >= 1 (unitless).
        bias_offset (float):
            Median number of counts in the bias region due to fixed non-bias noise
            not in common with the image region.  Basically we compute the bias
            for the image region based on the prescan from each frame, and the
            bias_offset is how many additional counts the prescan had from extra
            noise not captured in the master dark fit.  This value is subtracted
            from each measured bias.  Units of DN.

    Attributes:
        image (array_like):
            Image section of frame (DN).
        prescan (array_like):
            Prescan section of frame (DN).
        al_prescan (array_like):
            Prescan with row numbers relative to the first image row (DN).
        frame_bias (array_like):
            Column vector with each entry the median of the prescan row minus the
            bias offset (DN).
        bias (array_like):
            Column vector with each entry the median of the prescan row relative
            to the first image row minus the bias offset (DN).
        frame_bias0 (array_like):
            Total frame minus the bias (row by row) minus the bias offset (DN).
        image_bias0 (array_like):
            Image area minus the bias (row by row) minus the bias offset (DN).
    """

    def __init__(self, frame_dn, meta, fwc_em, fwc_pp, em_gain, bias_offset):
        self.frame_dn = frame_dn
        self.meta = meta
        self.fwc_em = fwc_em
        self.fwc_pp = fwc_pp
        self.em_gain = em_gain
        self.bias_offset = bias_offset

        # Divide frame into sections
        try:
            self.image = self.meta.slice_section(self.frame_dn, 'image')
            self.prescan = self.meta.slice_section(self.frame_dn, 'prescan')
        except Exception:
            raise EMCCDFrameException('Frame size inconsistent with metadata')

        # Get the part of the prescan that lines up with the image, and do a
        # row-by-row bias subtraction on it
        i_r0 = self.meta.geom['image']['r0c0'][0]
        p_r0 = self.meta.geom['prescan']['r0c0'][0]
        i_nrow = self.meta.geom['image']['rows']
        # select the good cols for getting row-by-row bias
        st = self.meta.geom['prescan']['col_start']
        end = self.meta.geom['prescan']['col_end']
        # over all prescan rows
        medbyrow_tot = np.median(self.prescan[:,st:end], axis=1)[:, np.newaxis]
        # prescan relative to image rows
        self.al_prescan = self.prescan[(i_r0-p_r0):(i_r0-p_r0+i_nrow), :]
        medbyrow = np.median(self.al_prescan[:,st:end], axis=1)[:, np.newaxis]

        # Get data from prescan (image area)
        self.bias = medbyrow - self.bias_offset
        self.image_bias0 = self.image - self.bias

        # over total frame
        self.frame_bias = medbyrow_tot - self.bias_offset
        self.frame_bias0 = self.frame_dn[p_r0:, :] -  self.frame_bias

# Run tests

def test_prescan_sub():
    """
    Generate mock raw data ('SCI' & 'ENG') and pass into prescan processing function.
    Check output dataset shapes, maintain pointers in the Dataset and Image class,
    and check that output is consistent with results II&T code.
    """

    tol = 0.01

    ###### create simulated data
    datadir = os.path.join(os.path.dirname(__file__), "simdata")

    for arrtype in ['SCI','ENG']:
        # create simulated data
        dataset = mocks.create_prescan_files(filedir=datadir, arrtype=arrtype)

        filenames = glob.glob(os.path.join(datadir, f"sim_prescan_{arrtype}*.fits"))

        dataset = data.Dataset(filenames)
        assert len(dataset) == 2

        iit_images = []
        iit_frames = []

        # II&T version
        for fname in filenames:

            l1_data = fits.getdata(fname)

            # Read in data
            meta_path = Path(here,'test_data','metadata.yaml') if arrtype == 'SCI' else Path(here,'test_data','metadata_eng.yaml')
            meta = Metadata(meta_path = meta_path)
            frameobj = EMCCDFrame(l1_data,
                                    meta,
                                    1., # fwc_em_dn
                                    1., # fwc_pp_dn
                                    1., # em_gain
                                    0.) # bias_offset

            # Subtract bias and bias offset and get cosmic mask
            iit_images.append(frameobj.image_bias0) # Science area
            iit_frames.append(frameobj.frame_bias0) # Full frame

        if len(dataset) != 2:
            raise Exception(f"Mock dataset is an unexpected length ({len(dataset)}).")

        for return_full_frame in [True, False]:
            for return_imaging_area in [True, False]:
                if return_full_frame is True and return_imaging_area is True:
                    continue # don't test this case b/c function not can't work
                output_dataset = prescan_biassub(dataset, noise_maps, return_full_frame=return_full_frame, use_imaging_area=return_imaging_area)

                # Check that output shape is as expected
                output_shape = output_dataset[0].data.shape
                if return_imaging_area is False:
                    shape_compare = shapes[arrtype][return_full_frame]
                else:
                    r, c, _ = imaging_area_geom(arrtype) 
                    shape_compare = (r,c)
                if output_shape != shape_compare:
                    raise Exception(f"Shape of output frame for {arrtype}, return_full_frame={return_full_frame} is {output_shape}, \nwhen {shapes[arrtype][return_full_frame]} was expected.")

                # Check that bias extension has the right size, dtype
                for i, frame in enumerate(output_dataset):

                    try: 
                        frame_bias = frame.hdu_list['BIAS'].data
                    except KeyError:
                        raise Exception(f"BIAS extension not found in frame {i}.")
                    
                    if frame_bias.shape != (frame.data.shape[0],):
                        raise Exception(f"Bias of frame {i} has shape {frame.bias.shape} when we expected {(frame.data.shape[0],)}.")
                    
                    if frame_bias.dtype != np.float32:
                        raise Exception(f"Bias of frame {i} does not have datatype np.float32.")

                # Check that corgiDRP and II&T pipeline produce the same result
                corgidrp_result = output_dataset[0].data
                if return_imaging_area is True:
                    continue
                iit_result = iit_frames[0] if return_full_frame else iit_images[0]
                if np.nanmax(np.abs(corgidrp_result-iit_result)) > tol:
                    raise Exception(f"corgidrp result does not match II&T result for generated mock data, arrtype={arrtype}, return_full_frame={return_full_frame}.")

                # check that data, err, and dq arrays are consistently modified
                output_dataset.all_data[0, 0, 0] = 0.
                if output_dataset[0].data[0, 0] != 0. :
                    raise Exception("Modifying dataset.all_data did not modify individual frame data.")

                output_dataset[0].data[0,0] = 1.
                if output_dataset.all_data[0,0,0] != 1. :
                    raise Exception("Modifying individual frame data did not modify dataset.all_data.")

                output_dataset.all_err[0, 0, 0, 0] = 0.
                if output_dataset[0].err[0, 0, 0] != 0. :
                    raise Exception("Modifying dataset.all_err did not modify individual frame err.")

                output_dataset[0].err[0, 0, 0] = 1.
                if output_dataset.all_err[0, 0, 0, 0] != 1. :
                    raise Exception("Modifying individual frame err did not modify dataset.all_err.")

                output_dataset.all_dq[0, 0, 0] = 0.
                if output_dataset[0].dq[0, 0] != 0. :
                    raise Exception("Modifying dataset.all_dq did not modify individual frame dq.")

                output_dataset[0].dq[0,0] = 1.
                if output_dataset.all_dq[0,0,0] != 1. :
                    raise Exception("Modifying individual frame dq did not modify dataset.all_dq.")

def test_bias_zeros_frame():
    """Verify prescan_biassub does not break for a frame of all zeros
    (should return all zeros)."""

    tol = 1e-13

    ###### create simulated data
    datadir = os.path.join(os.path.dirname(__file__), "simdata")

    for arrtype in ['SCI', 'ENG']:
        # create simulated data
        dataset = mocks.create_prescan_files(filedir=datadir, arrtype=arrtype,numfiles=1)

        # Overwrite data with zeros
        dataset.all_data[:,:,:] = 0.

        for return_full_frame in [True, False]:

            output_dataset = prescan_biassub(dataset, noise_maps=None, return_full_frame=return_full_frame)

            if np.max(np.abs(output_dataset.all_data)) > tol:
                raise Exception(f'Operating on all zero frame did not return all zero frame.')

            if np.max(np.abs(output_dataset.all_err)) > tol:
                raise Exception(f'Operating on all zero frame did not return all zero error.')           
            
            for i,frame in enumerate(output_dataset):
                try: 
                    frame_bias = frame.hdu_list['BIAS'].data
                except KeyError:
                    raise Exception(f"BIAS extension not found in frame {i}.".format(i))
                
                if np.max(np.abs(frame_bias)) > tol:
                    raise Exception(f'Operating on all zero frame did not return all zero bias.')

def test_bias_hvoff():
    """
    Verify that the function finds bias for gaussian distribution, with no
    contribution from the effect of gain ("hv" is the voltage applied in
    the EM gain register).
    The error tolerance is set by the standard error on the median of
    the Gaussian noise, not the mean.
    """
    corgidrp.track_individual_errors = True # needs to run with error tracking on

    # Set tolerance
    tol = 1.
    err_tol = 0.02
    bval = 100.
    sig = 1.
    seed = 12346

    ###### create simulated data
    datadir = os.path.join(os.path.dirname(__file__), "simdata")

    for arrtype in ['SCI', 'ENG']:
        # create simulated data
        dataset = mocks.create_prescan_files(filedir=datadir, arrtype=arrtype,
                                             numfiles=1)

        # Overwrite data with normal distribution
        rng = np.random.default_rng(seed)
        dataset.all_data[:,:,:] = rng.normal(bval, sig,
                                             size=dataset.all_data.shape)

        for return_full_frame in [True, False]:

            output_dataset = prescan_biassub(dataset, noise_maps, return_full_frame=return_full_frame)

            # Compare bias measurement to expectation
            if np.any(np.abs(output_dataset[0].hdu_list['BIAS'].data - bval) > tol):
                raise Exception(f'Higher than expected error in bias measurement for hvoff distribution.')

            # Compare error to expected standard error of the median
            std_err = sig / np.sqrt(detector_areas[arrtype]['prescan_reliable']['cols']) * np.sqrt(np.pi / 2.)
            if np.max(np.abs(output_dataset[0].err[1]) - std_err) > err_tol:
                raise Exception(f'Higher than expected std. error in bias measurement for hvoff distribution: \n{np.max(np.abs(output_dataset[0].err[1]))} when we expect {std_err} +- {err_tol} ')

    corgidrp.track_individual_errors = old_err_tracking

def test_bias_hvon():
    """
    Verify that the function finds bias for a gaussian distribution, plus
    additional contributions from the effect of gain ("hv" is the voltage
    applied in the EM gain register), approximated as an exponential distribution
    + inflated values for the "unreliable" prescan region, minus the mean of the
    exponential distribution to keep the DC contribution 0.
    Also tests that only the good columns are used for the bias.
    """

    # Set tolerance
    tol = 6.
    bval = 100.
    expmean = 10.
    seed = 12346

    ###### create simulated data
    datadir = os.path.join(os.path.dirname(__file__), "simdata")

    for arrtype in ['SCI', 'ENG']:
        # create simulated dataset
        dataset = mocks.create_prescan_files(filedir=datadir, arrtype=arrtype,numfiles=1)

        # Generate bias with inflated values in the bad columns
        bias = np.full_like(dataset.all_data,bval)
        col_start = detector_areas[arrtype]['prescan_reliable']['r0c0'][1]
        bias[:,:,0:col_start] = bval * 5

        # Overwrite dataset with normal + exponential + bias
        rng = np.random.default_rng(seed)
        dataset.all_data[:,:,:] = (rng.normal(0, 1, size=dataset.all_data.shape)
                                   + rng.exponential(expmean, size=dataset.all_data.shape)
                                   - expmean # to keep DC contribution 0
                                   + bias)

        for return_full_frame in [True, False]:
            output_dataset = prescan_biassub(dataset, noise_maps, return_full_frame=return_full_frame)
            if np.any(np.abs(output_dataset[0].hdu_list['BIAS'].data - bval) > tol):
                raise Exception(f'Higher than expected error in bias measurement for hvon distribution.')

def test_bias_uniform_value():
    """Verify that function finds bias for uniform value."""

    tol = 1e-13
    bval = 1.

    ###### create simulated dataset
    datadir = os.path.join(os.path.dirname(__file__), "simdata")

    for arrtype in ['SCI', 'ENG']:
        # create simulated dataset
        dataset = mocks.create_prescan_files(filedir=datadir, arrtype=arrtype,numfiles=1)

        # Overwrite dataset with normal + exponential + bias
        dataset.all_data[:,:,:] = bval

        for return_full_frame in [True, False]:
            output_dataset = prescan_biassub(dataset, noise_maps=None, return_full_frame=return_full_frame)
            if np.max(np.abs(output_dataset.all_data)) > tol:
                raise Exception(f'Higher than expected error in bias measurement for uniform value.')

            if np.max(np.abs(output_dataset.all_err)) > tol:
                raise Exception(f'Higher than expected std. error in bias measurement for uniform value.')

def test_bias_offset():
    """Verify bias offset incorporated as expected"""

    # 10 counts higher than the bias in the image region.
    bias_offset = 10
    tol = 1e-13

    ###### create simulated dataset
    datadir = os.path.join(os.path.dirname(__file__), "simdata")

    for arrtype in ['SCI', 'ENG']:
        # create simulated dataset with 0 bias offset
        dataset_0 = mocks.create_prescan_files(filedir=datadir, arrtype=arrtype,numfiles=1)
        dataset_0.all_data[:,:,:] = 0.

        # create simulated dataset with 10 bias offset
        # bias_offset = 10 means the bias, as measured in the prescan, is
        # 10 counts higher than the bias in the image region.
        dataset_10 = dataset_0.copy()
        r0c0 = detector_areas[arrtype]['prescan']['r0c0']
        rows = detector_areas[arrtype]['prescan']['rows']
        cols = detector_areas[arrtype]['prescan']['cols']
        dataset_10.all_data[:,r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] += bias_offset

        noise_maps0 = noise_maps.copy()
        noise_maps0.bias_offset = 0
        noise_maps10 = noise_maps.copy()
        noise_maps10.bias_offset = 10
        for return_full_frame in [True,False]:
            output_dataset_0 = prescan_biassub(dataset_0, noise_maps0, return_full_frame=return_full_frame)
            output_dataset_10 = prescan_biassub(dataset_10, noise_maps10, return_full_frame=return_full_frame)

            # Compare science image region only
            if return_full_frame:
                r0c0 = detector_areas[arrtype]['image']['r0c0']
                rows = detector_areas[arrtype]['image']['rows']
                cols = detector_areas[arrtype]['image']['cols']
                image_slice_0 = output_dataset_0.all_data[0,r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
                image_slice_10 = output_dataset_10.all_data[0,r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
            else:
                image_slice_0 = output_dataset_0.all_data[0]
                image_slice_10 = output_dataset_10.all_data[0]

            if not np.nanmax(np.abs(image_slice_0 - image_slice_10)) < tol:
                raise Exception(f"Bias offset subtraction did not produce the correct result. absmax value : {np.nanmax(np.abs(image_slice_0 - image_slice_10))}")


if __name__ == "__main__":
    test_prescan_sub()
    test_bias_zeros_frame()
    test_bias_hvoff()
    test_bias_hvon()
    test_bias_uniform_value()
    test_bias_offset()