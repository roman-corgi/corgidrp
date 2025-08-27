import os
import re
import numpy as np
import numpy.ma as ma
import astropy.io.fits as fits
import warnings
from astropy.io.fits.card import VerifyWarning
import astropy.time as time
import pandas as pd
from astropy.table import Table
import pyklip
from pyklip.instruments.Instrument import Data as pyKLIP_Data
from pyklip.instruments.utils.wcsgen import generate_wcs
from scipy.interpolate import LinearNDInterpolator
from astropy import wcs
import copy
import corgidrp
from datetime import datetime, timedelta, timezone

class Dataset():
    """
    A sequence of data of the same kind. Can be indexed and looped over

    Args:
        frames_or_filepaths (list): list of either filepaths or data objects (e.g., Image class)

    Attributes:
        all_data (np.array): an array with all the data combined together. First dimension is always number of images
        frames (np.array): list of data objects (probably corgidrp.data.Image)
    """
    def __init__(self, frames_or_filepaths):
        """
        Args:
            frames_or_filepaths (list): list of either filepaths or data objects (e.g., Image class)
        """
        if len(frames_or_filepaths) == 0:
            raise ValueError("Empty list passed in")

        if isinstance(frames_or_filepaths[0], str):
            # list of filepaths
            # TODO: do some auto detection of the filetype, but for now assume it is an image file
            self.frames = []
            for filepath in frames_or_filepaths:
                self.frames.append(Image(filepath))
        else:
            # list of frames
            self.frames = frames_or_filepaths

        # turn lists into np.array for indexing behavior
        if isinstance(self.frames, list):
            self.frames = np.array(self.frames) # list of objects

        # create 3-D cube of all the data
        self.all_data = np.array([frame.data for frame in self.frames])
        self.all_err = np.array([frame.err for frame in self.frames])
        self.all_dq = np.array([frame.dq for frame in self.frames])
        # do a clever thing to point all the individual frames to the data in this cube
        # this way editing a single frame will also edit the entire datacube
        for i, frame in enumerate(self.frames):
            frame.data = self.all_data[i]
            frame.err = self.all_err[i]
            frame.dq = self.all_dq[i]

    def __iter__(self):
        return self.frames.__iter__()

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # return a single element of the data
            return self.frames[indices]
        else:
            # return a subset of the dataset
            return Dataset(self.frames[indices])

    def __len__(self):
        return len(self.frames)

    def save(self, filedir=None, filenames=None):
        """
        Save each file of data in this dataset into directory

        Args:
            filedir (str): directory to save the files. Default: the existing filedir for each file
            filenames (list): a list of output filenames for each file. Default: unchanged filenames

        """
        # if filenames are not passed, use the default ones
        if filenames is None:
            filenames = []
            for frame in self.frames:
                filename = frame.filename
                filenames.append(frame.filename)

        for filename, frame in zip(filenames, self.frames):
            frame.save(filename=filename, filedir=filedir)

    def update_after_processing_step(self, history_entry, new_all_data=None, new_all_err = None, new_all_dq = None, header_entries = None):
        """
        Updates the dataset after going through a processing step

        Args:
            history_entry (str): a description of what processing was done. Mention reference files used.
            new_all_data (np.array): (optional) Array of new data. Needs to be the same shape as `all_data`
            new_all_err (np.array): (optional) Array of new err. Needs to be the same shape as `all_err` except of second dimension
            new_all_dq (np.array): (optional) Array of new dq. Needs to be the same shape as `all_dq`
            header_entries (dict): (optional) a dictionary {} of ext_hdr and err_hdr entries to add or update
        """
        # update data if necessary
        if new_all_data is not None:
            if new_all_data.shape != self.all_data.shape:
                raise ValueError("The shape of new_all_data is {0}, whereas we are expecting {1}".format(new_all_data.shape, self.all_data.shape))
            self.all_data[:] = new_all_data # specific operation overwrites the existing data rather than changing pointers
        if new_all_err is not None:
            if new_all_err.shape[-2:] != self.all_err.shape[-2:] or new_all_err.shape[0] != self.all_err.shape[0]:
                raise ValueError("The shape of new_all_err is {0}, whereas we are expecting {1}".format(new_all_err.shape, self.all_err.shape))
            self.all_err = new_all_err
            for i in range(len(self.frames)):
                self.frames[i].err = self.all_err[i]
        if new_all_dq is not None:
            if new_all_dq.shape != self.all_dq.shape:
                raise ValueError("The shape of new_all_dq is {0}, whereas we are expecting {1}".format(new_all_dq.shape, self.all_dq.shape))
            self.all_dq[:] = new_all_dq # specific operation overwrites the existing data rather than changing pointers

        # update history and header entries
        for img in self.frames:
            img.ext_hdr['HISTORY'] = history_entry
            if header_entries:
                for key, value in header_entries.items():
                    img.ext_hdr[key] = value
                    img.err_hdr[key] = value


    def copy(self, copy_data=True):
        """
        Make a copy of this dataset, including all data and headers.
        Data copying can be turned off if you only want to modify the headers
        Headers should always be copied as we should modify them any time we make new edits to the data

        Args:
            copy_data (bool): (optional) whether the data should be copied. Default is True

        Returns:
            corgidrp.data.Dataset: a copy of this dataset
        """
        # there's a smarter way to manage memory, but to keep the API simple, we will avoid it for now
        new_frames = [frame.copy(copy_data=copy_data) for frame in self.frames]
        new_dataset = Dataset(new_frames)

        return new_dataset

    def add_error_term(self, input_error, err_name):
        """
        Calls Image.add_error_term() for each frame.
        Updates Dataset.all_err.

        Args:
          input_error (np.array): per-frame or per-dataset error layer
          err_name (str): name of the uncertainty layer
        """
        if input_error.ndim == self.all_data.ndim:
            for i,frame in enumerate(self.frames):
                frame.add_error_term(input_error[i], err_name)

        elif input_error.ndim == self.all_data.ndim - 1:
            for frame in self.frames:
                frame.add_error_term(input_error, err_name)

        else:
            raise ValueError("input_error is not either a 2D or 3D array for 2D data, or a 3D or 4D array for 3D data.")

        # Preserve pointer links between Dataset.all_err and Image.err
        self.all_err = np.array([frame.err for frame in self.frames])
        for i, frame in enumerate(self.frames):
            frame.err = self.all_err[i]

    def rescale_error(self, input_error, err_name):
        """
        Calls Image.rescale_errors() for each frame.
        Updates Dataset.all_err

        Args:
          input_error (np.array): 2-d error layer or 3-d layer
          err_name (str): name of the uncertainty layer
        """
        if input_error.ndim == 3:
            for i,frame in enumerate(self.frames):
                frame.rescale_error(input_error[i], err_name)

        elif input_error.ndim ==2:
            for frame in self.frames:
                frame.rescale_error(input_error, err_name)

        else:
            raise ValueError("input_error is not either a 2D or 3D array.")

        # Preserve pointer links between Dataset.all_err and Image.err
        self.all_err = np.array([frame.err for frame in self.frames])
        for i, frame in enumerate(self.frames):
            frame.err = self.all_err[i]

    def split_dataset(self, prihdr_keywords=None, exthdr_keywords=None):
        """
        Splits up this dataset into multiple smaller datasets that have the same set of header keywords
        The code uses all keywords together to determine an unique group

        Args:
            prihdr_keywords (list of str): list of primary header keywords to split
            exthdr_keywords (list of str): list of 1st extension header keywords to split on

        Returns:
            list of datasets: list of sub datasets
            list of tuples: list of each set of unique header keywords. pri_hdr keywords occur before ext_hdr keywords
        """
        if prihdr_keywords is None and exthdr_keywords is None:
            raise ValueError("No prihdr or exthdr keywords passed in to split dataset")

        col_names = []
        col_vals = []
        if prihdr_keywords is not None:
            for key in prihdr_keywords:
                dataset_vals = [frame.pri_hdr[key] for frame in self.frames]

                col_names.append(key)
                col_vals.append(dataset_vals)

        if exthdr_keywords is not None:
            for key in exthdr_keywords:
                dataset_vals = [frame.ext_hdr[key] for frame in self.frames]

                col_names.append(key)
                col_vals.append(dataset_vals)

        all_data = np.array(col_vals).T

        # track all combinations
        df = pd.DataFrame(data=all_data, columns=col_names)

        grouped = df.groupby(col_names)

        unique_vals = list(grouped.indices.keys()) # each unique set of values
        split_datasets = []
        for combo in grouped.indices:
            dataset_indices = grouped.indices[combo]
            sub_dataset = self[dataset_indices]
            split_datasets.append(sub_dataset)

        return split_datasets, unique_vals

class Image():
    """
    Base class for 2-D image data. Data can be created by passing in the data/header explicitly, or
    by passing in a filepath to load a FITS file from disk

    Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR the 2D image data
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw 2D data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw 2D data is passed in)
        err (np.array): 2-D/3-D uncertainty data
        dq (np.array): 2-D data quality, 0: good. Other values track different causes for bad pixels and other pixel-level effects in accordance with the DRP implementation document.x
        err_hdr (astropy.io.fits.Header): the error extension header
        dq_hdr (astropy.io.fits.Header): the data quality extension header
        hdu_list (astropy.io.fits.HDUList): an astropy HDUList object that contains any other extension types. 

    Attributes:
        data (np.array): 2-D data for this Image
        err (np.array): 2-D uncertainty
        dq (np.array): 2-D data quality
        pri_hdr (astropy.io.fits.Header): primary header
        ext_hdr (astropy.io.fits.Header): image extension header. Generally this header will be edited/added to
        err_hdr (astropy.io.fits.Header): the error extension header
        dq_hdr (astropy.io.fits.Header): the data quality extension header
        hdu_list (astropy.io.fits.HDUList): an astropy HDUList object that contains any other extension types.
        filename (str): the filename corresponding to this Image
        filedir (str): the file directory on disk where this image is to be/already saved.
        filepath (str): full path to the file on disk (if it exists)
    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, err = None, dq = None, err_hdr = None, dq_hdr = None, input_hdulist = None):
        if isinstance(data_or_filepath, str):
            # a filepath is passed in
            with fits.open(data_or_filepath, ignore_missing_simple=True) as hdulist:
                
                #Pop out the primary header
                self.pri_hdr = hdulist.pop(0).header
                #Pop out the image extension
                first_hdu = hdulist.pop(0)
                self.ext_hdr = first_hdu.header
                self.data = first_hdu.data

                #A list of extensions
                self.hdu_names = [hdu.name for hdu in hdulist]

                # we assume that if the err and dq array is given as parameter they supersede eventual err and dq extensions
                if err is not None:
                    if isinstance(err, float):
                        if np.size(self.data) != 1:
                            raise ValueError("err can only be a float if data is a float value")
                        self.err = np.array([err])
                    elif np.shape(self.data) != np.shape(err)[-self.data.ndim:]:
                        raise ValueError("The shape of err is {0} while we are expecting shape {1}".format(err.shape[-self.data.ndim:], self.data.shape))
                    #we want to have an extra dimension in the error array
                    elif err.ndim == self.data.ndim+1:
                        self.err = err
                    else:
                        self.err = err.reshape((1,)+err.shape)
                elif "ERR" in self.hdu_names:
                    err_hdu = hdulist.pop("ERR")
                    self.err = err_hdu.data
                    self.err_hdr = err_hdu.header
                    if self.err.ndim != 1 and self.err.ndim == self.data.ndim:
                        self.err = self.err.reshape((1,)+self.err.shape)
                else:
                    self.err = np.zeros((1,)+self.data.shape)

                if dq is not None:
                    if np.shape(self.data) != np.shape(dq):
                        raise ValueError("The shape of dq is {0} while we are expecting shape {1}".format(dq.shape, self.data.shape))
                    self.dq = dq
                
                elif "DQ" in self.hdu_names:
                    dq_hdu = hdulist.pop("DQ")
                    self.dq = dq_hdu.data
                    self.dq_hdr = dq_hdu.header
                else:
                    self.dq = np.zeros(self.data.shape, dtype = int)


                if input_hdulist is not None:
                    this_hdu_list = [hdu.copy() for hdu in input_hdulist]
                else: 
                    #After the data, err and dqs are popped out, the rest of the hdulist is stored in hdu_list
                    this_hdu_list = [hdu.copy() for hdu in hdulist]
                self.hdu_list = fits.HDUList(this_hdu_list)
                

            # parse the filepath to store the filedir and filename
            filepath_args = data_or_filepath.split(os.path.sep)
            if len(filepath_args) == 1:
                # no directory info in filepath, so current working directory
                self.filedir = "."
                self.filename = filepath_args[0]
            else:
                self.filename = filepath_args[-1]
                self.filedir = os.path.sep.join(filepath_args[:-1])

        else:
            # data has been passed in directly
            # creation of a new file in DRP eyes
            if isinstance(data_or_filepath, float):
                self.data = np.array([data_or_filepath])
                if err is not None:
                    if isinstance(err, float):
                        self.err = np.array([err])
                    else:
                        raise ValueError("err value must be float")
                else:
                    self.err = np.array([0.])
            elif hasattr(data_or_filepath, "__len__"):
                self.data = data_or_filepath
                if err is not None:
                    if np.shape(self.data) != np.shape(err)[-self.data.ndim:]:
                        raise ValueError("The shape of err is {0} while we are expecting shape {1}".format(err.shape[-self.data.ndim:], self.data.shape))
                    #we want to have a 3 dim error array
                    if err.ndim == self.data.ndim + 1:
                        self.err = err
                    else:
                        self.err = err.reshape((1,)+err.shape)
                else:
                    self.err = np.zeros((1,)+self.data.shape)
            else:
                raise ValueError("input must be an array or float")
            
            if pri_hdr is None or ext_hdr is None:
                raise ValueError("Missing primary and/or extension headers, because you passed in raw data")
            self.pri_hdr = pri_hdr
            self.ext_hdr = ext_hdr
            self.filedir = "."
            self.filename = ""

            if dq is not None:
                if np.shape(self.data) != np.shape(dq):
                    raise ValueError("The shape of dq is {0} while we are expecting shape {1}".format(dq.shape, self.data.shape))
                self.dq = dq
            else:
                self.dq = np.zeros(self.data.shape, dtype = int)

            #The default hdu extensions
            self.hdu_names = ["ERR", "DQ"]

            #Take the input hdulist or make a blank one. 
            if input_hdulist is not None:
                this_hdu_list = [hdu.copy() for hdu in input_hdulist]
                self.hdu_list = fits.HDUList(this_hdu_list)
                #Keep track of the names 
                for hdu in input_hdulist:
                    self.hdu_names.append(hdu.name)
            else: 
                self.hdu_list = fits.HDUList()

            # record when this file was created and with which version of the pipeline
            self.ext_hdr.set('DRPVERSN', corgidrp.__version__, "corgidrp version that produced this file")
            self.ext_hdr.set('DRPCTIME', time.Time.now().isot, "When this file was saved")

        
        # we assume that if the err_hdr and dq_hdr is given as parameter they supersede eventual existing err_hdr and dq_hdr
        if err_hdr is not None:
            self.err_hdr = err_hdr
        if dq_hdr is not None:
            self.dq_hdr = dq_hdr
        if not hasattr(self, 'err_hdr'):
            self.err_hdr = fits.Header()
        self.err_hdr["EXTNAME"] = "ERR"
        if not hasattr(self, 'dq_hdr'):
            self.dq_hdr = fits.Header()
        self.dq_hdr["EXTNAME"] = "DQ"

        # discard individual errors if we aren't tracking them but multiple error terms are passed in
        if not corgidrp.track_individual_errors and self.err.shape[0] > 1:
            num_errs = self.err.shape[0] - 1
            # delete keywords specifying the error of each individual slice
            for i in range(num_errs):
                del self.err_hdr['Layer_{0}'.format(i + 2)]
            self.err = self.err[:1] # only save the total err, preserve 3-D shape
        self.err_hdr['TRK_ERRS'] = corgidrp.track_individual_errors # specify whether we are tracing errors

        # the DRP needs to make sure certain keywords are set in its reduced products
        # check those here, and if not, set them. 
        # by default, assume desmear and CTI correction are not applied by default
        # and they can be toggled to true after their step functions are run
        if not 'DESMEAR' in self.ext_hdr:
            self.ext_hdr.set('DESMEAR', False, "Was desmear applied to this frame?")
        if not 'CTI_CORR' in self.ext_hdr:
            self.ext_hdr.set('CTI_CORR', False, "Was CTI correction applied to this frame?")
        if not 'IS_BAD' in self.ext_hdr:
            self.ext_hdr.set('IS_BAD', False, "Was this frame deemed bad?")

        # the DRP has touched this file so it's origin is now this DRP
        self.pri_hdr['ORIGIN'] = 'DRP'


    # create this field dynamically
    @property
    def filepath(self):
        return os.path.join(self.filedir, self.filename)


    def save(self, filedir=None, filename=None):
        """
        Save file to disk with user specified filepath

        Args:
            filedir (str): filedir to save to. Use self.filedir if not specified
            filename (str): filepath to save to. Use self.filename if not specified
        """
        if filename is not None:
            self.filename = filename
        if filedir is not None:
            self.filedir = filedir

        if len(self.filename) == 0:
            raise ValueError("Output filename is not defined. Please specify!")

        prihdu = fits.PrimaryHDU(header=self.pri_hdr)
        exthdu = fits.ImageHDU(data=self.data, header=self.ext_hdr)
        hdulist = fits.HDUList([prihdu, exthdu])

        errhdu = fits.ImageHDU(data=self.err, header = self.err_hdr)
        hdulist.append(errhdu)

        dqhdu = fits.ImageHDU(data=self.dq, header = self.dq_hdr)
        hdulist.append(dqhdu)

        for hdu in self.hdu_list:
            hdulist.append(hdu)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=VerifyWarning) # fits save card length truncated warning
            hdulist.writeto(self.filepath, overwrite=True)
        hdulist.close()

    def _record_parent_filenames(self, input_dataset):
        """
        Record what input dataset was used to create this Image.
        This assumes many Images were used to make this single Image.
        Also tracks images that were used to make the images that make this image.
        Record is stored in the ext header.

        Args:
            input_dataset (corgidrp.data.Dataset): the input dataset that were combined together to make this image
        """

        parent_filenames = set()
        # go through filenames and also check each frames parents
        for img in input_dataset:
            parent_filenames.add(img.filename)
            # also check if this frame has parent frames. keep trakc of them too
            if 'DRPNFILE' in img.ext_hdr:
                for j in range(img.ext_hdr['DRPNFILE']):
                    parent_filenames.add(img.ext_hdr['FILE{0}'.format(j)])
        
        for i, filename in enumerate(parent_filenames):
            self.ext_hdr.set('FILE{0}'.format(i), filename, "File #{0} filename used to create this frame".format(i))
        self.ext_hdr.set('DRPNFILE', len(parent_filenames), "# of files used to create this processed frame")

    def copy(self, copy_data=True):
        """
        Make a copy of this image file. including data and headers.
        Data copying can be turned off if you only want to modify the headers
        Headers should always be copied as we should modify them any time we make new edits to the data

        Args:
            copy_data (bool): (optional) whether the data should be copied. Default is True

        Returns:
            corgidrp.data.Image: a copy of this Image
        """
        if copy_data:
            new_img = copy.deepcopy(self)
        else:
            new_img = copy.copy(self)
            # copy the hdu_list and hdu_names list, but not their pointers
            new_img.hdu_list = self.hdu_list.copy()
            new_img.hdu_names = copy.copy(self.hdu_names)

        # update DRP version tracking
        new_img.ext_hdr['DRPVERSN'] =  corgidrp.__version__
        new_img.ext_hdr['DRPCTIME'] =  time.Time.now().isot

        return new_img

    def get_masked_data(self):
        """
        Uses the dq array to generate a numpy masked array of the data

        Returns:
            numpy.ma.MaskedArray: the data masked
        """
        mask = self.dq>0
        return ma.masked_array(self.data, mask=mask)

    def add_error_term(self, input_error, err_name):
        """
        Add a layer of a specific additive uncertainty on the 3- or 4-dim error array extension
        and update the combined uncertainty in the first layer.
        Update the error header and assign the error name.

        Only tracks individual errors if the "track_individual_errors" setting is set to True
        in the configuration file

        Args:
          input_error (np.array): error layer with same shape as data
          err_name (str): name of the uncertainty layer
        """
        ndim = self.data.ndim
        if not (input_error.ndim==2 or input_error.ndim==3) or input_error.shape != self.data.shape:
            raise ValueError("we expect a 2-dimensional or 3-dimensional error layer with dimensions {0}".format(self.data.shape))

        #first layer is always the updated combined error
        if ndim == 2:
            self.err[0,:,:] = np.sqrt(self.err[0,:,:]**2 + input_error**2)
        elif ndim == 3:
            self.err[0,:,:,:] = np.sqrt(self.err[0,:,:,:]**2 + input_error**2)
        self.err_hdr["Layer_1"] = "combined_error"

        if corgidrp.track_individual_errors:
            #append new error as layer on 3D or 4D cube
            self.err=np.append(self.err, [input_error], axis=0)

            layer = str(self.err.shape[0])
            self.err_hdr["Layer_" + layer] = err_name

        # record history since 2-D error map doesn't track individual terms
        self.err_hdr['HISTORY'] = "Added error term: {0}".format(err_name)

    def rescale_error(self, input_error, err_name):
        """
        Add a layer of a specific additive uncertainty on the 3-dim error array extension
        and update the combined uncertainty in the first layer.
        Update the error header and assign the error name.

        Only tracks individual errors if the "track_individual_errors" setting is set to True
        in the configuration file

        Args:
          input_error (np.array): 2-d error layer
          err_name (str): name of the uncertainty layer
        """
        if input_error.ndim != 2 or input_error.shape != self.data.shape:
            raise ValueError("we expect a 2-dimensional error layer with dimensions {0}".format(self.data.shape))

        #first layer is always the updated combined error
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning) # catch any invalid value encountered in multiply
            self.err = self.err*input_error
        self.err_hdr["Layer_1"] = "combined_error"

        # record history since 2-D error map doesn't track individual terms
        self.err_hdr['HISTORY'] = "Errors rescaled by: {0}".format(err_name)


    def get_hash(self):
        """
        Computes the hash of the data, err, and dq. Does not use the header information.

        Returns:
            str: the hash of the data, err, and dq
        """
        data_bytes = self.data.data.tobytes()
        err_bytes = self.err.data.tobytes()
        dq_bytes = self.dq.data.tobytes()

        total_bytes = data_bytes + err_bytes + dq_bytes

        return str(hash(total_bytes))

    def add_extension_hdu(self, name, data = None, header=None):
        """

        Create a new hdu extension and append it to the hdu_list

        Args:
            name (str): The name of the new extension
            data (array, optional): Some kind of data. Defaults to None.
            header (astropy.io.fits.Header, optional): _description_. Defaults to None.
        """
        new_hdu = fits.ImageHDU(data=data, header=header, name=name)

        if name in self.hdu_names:
            raise ValueError("Extension name already exists in HDU list")
        else: 
            self.hdu_names.append(name)
            self.hdu_list.append(new_hdu)

class Dark(Image):
    """
    Dark calibration frame for a given exposure time and EM gain.

     Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR the 2D image data
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw 2D data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw 2D data is passed in)
        input_dataset (corgidrp.data.Dataset): the Image files combined together to make this dark (required only if raw 2D data is passed in and if raw data filenames not already archived in ext_hdr)
        err (np.array): the error array (required only if raw data is passed in)
        err_hdr (astropy.io.fits.Header): the error header (required only if raw data is passed in)
        dq (np.array): the DQ array (required only if raw data is passed in)
    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, input_dataset=None, err = None, dq = None, err_hdr = None):
       # run the image class contructor
        super().__init__(data_or_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err=err, dq=dq, err_hdr=err_hdr)

        # if this is a new dark, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if it is a new dark
        if ext_hdr is not None:
            if input_dataset is None and 'DRPNFILE' not in ext_hdr.keys():
                # error check. this is required in this case
                raise ValueError("This appears to be a new dark. The dataset of input files needs to be passed in to the input_dataset keyword to record history of this dark.")
            self.ext_hdr['DATATYPE'] = 'Dark' # corgidrp specific keyword for saving to disk
            self.ext_hdr['BUNIT'] = 'detected electron'
            # TO-DO: check PC_STAT and whether this will be in L2s
            if 'PC_STAT' not in ext_hdr:
                self.ext_hdr['PC_STAT'] = 'analog master dark'
            # log all the data that went into making this calibration file
            if 'DRPNFILE' not in ext_hdr.keys() and input_dataset is not None:
                self._record_parent_filenames(input_dataset)

            # add to history
            self.ext_hdr['HISTORY'] = "Dark with exptime = {0} s and commanded EM gain = {1} created from {2} frames".format(self.ext_hdr['EXPTIME'], self.ext_hdr['EMGAIN_C'], self.ext_hdr['DRPNFILE'])

            # give it a default filename using the last input file as the base
            # strip off everything starting at .fits
            if input_dataset is not None:
                orig_input_filename = input_dataset[-1].filename.split(".fits")[0]
                self.filename = "{0}_drk_cal.fits".format(orig_input_filename)
                self.filename = re.sub('_l[0-9].', '', self.filename)
                # dnm_cal fed directly into drk_cal when doing build_synthesized_dark, so this will delete that string if it's there:
                self.filename = self.filename.replace("_dnm_cal", "")
            else:
                if self.filename == '':
                    self.filename = "drk_cal.fits" # we shouldn't normally be here, but we default to something just in case. 
                else:
                    self.filename = self.filename.replace("_dnm_cal", "_drk_cal")
            self.pri_hdr['FILENAME'] = self.filename
            # Enforce data level = CAL
            self.ext_hdr['DATALVL']    = 'CAL'
        
        if 'PC_STAT' not in self.ext_hdr:
            self.ext_hdr['PC_STAT'] = 'analog master dark'

        if err_hdr is not None:
            self.err_hdr['BUNIT'] = 'detected electron'

        # double check that this is actually a dark file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a Dark file.")
        if self.ext_hdr['DATATYPE'] != 'Dark':
            raise ValueError("File that was loaded was not a Dark file.")

class FlatField(Image):
    """
    Master flat generated from raster scan of uranus or Neptune.

     Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR the 2D image data
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw 2D data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw 2D data is passed in)
        input_dataset (corgidrp.data.Dataset): the Image files combined together to make this flat file (required only if raw 2D data is passed in)
    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, input_dataset=None):
        # run the image class contructor
        super().__init__(data_or_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr)

        # if this is a new master flat, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if it is a new masterflat
        if ext_hdr is not None:
            if input_dataset is None:
                # error check. this is required in this case
                raise ValueError("This appears to be a master flat. The dataset of input files needs to be passed in to the input_dataset keyword to record history of this flat")
            self.ext_hdr['DATATYPE'] = 'FlatField' # corgidrp specific keyword for saving to disk
            self.ext_hdr['BUNIT'] = '' # flat field is dimensionless

            # log all the data that went into making this flat
            self._record_parent_filenames(input_dataset)

            # add to history
            self.ext_hdr['HISTORY'] = "Flat with exptime = {0} s created from {1} frames".format(self.ext_hdr['EXPTIME'], self.ext_hdr['DRPNFILE'])

            # give it a default filename using the last input file as the base
            self.filename = re.sub('_l[0-9].', '_flt_cal', input_dataset[-1].filename)
            self.pri_hdr['FILENAME'] = self.filename

            # Enforce data level = CAL
            self.ext_hdr['DATALVL']    = 'CAL'

        # double check that this is actually a masterflat file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a FlatField file.")
        if self.ext_hdr['DATATYPE'] != 'FlatField':
            raise ValueError("File that was loaded was not a FlatField file.")

class SpectroscopyCentroidPSF(Image):
    """
    Calibration product that stores fitted PSF centroid (x, y) positions
    for a grid of simulated PSFs.

    Args:
        data_or_filepath (str or np.ndarray): 2D array of (x, y) centroid positions 
                                              with shape (N, 2), where N is the number of PSFs.
        err (np.ndarray): 2D array of (x,y) errors of centroid positions with shape (N,2)
        pri_hdr (fits.Header): Primary header.
        ext_hdr (fits.Header): Extension header.
        err_hdr (fits.Header): error extension header
        input_dataset (Dataset): Dataset of raw PSF images used to generate this calibration.
    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, err_hdr = None, err = None, input_dataset=None):
        super().__init__(data_or_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err_hdr = err_hdr, err = err)


        # if this is a new SpectroscopyCentroidPSF, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if it is a new SpectroscopyCentroidPSF 
        if ext_hdr is not None:
            if input_dataset is None:
                raise ValueError("Must pass `input_dataset` to create new PSFCentroidCalibration.")
            
            self.ext_hdr["EXTNAME"] = "CENTROIDS"

            self.ext_hdr['DATATYPE'] = 'PSFCentroidCalibration'
            self.ext_hdr['DATALVL'] = 'CAL'
            self._record_parent_filenames(input_dataset)
            self.ext_hdr['HISTORY'] = "Stored PSF centroid calibration results."

            # Generate default output filename
            base = input_dataset[0].filename.split(".fits")[0]
            self.filename = re.sub('_l[0-9].', '_scp_cal', input_dataset[-1].filename)
            self.pri_hdr['FILENAME'] = self.filename
            if err is None:
                self.err = np.zeros(self.data.shape)
                self.err_hdr = fits.Header

        if 'DATATYPE' not in self.ext_hdr or self.ext_hdr['DATATYPE'] != 'PSFCentroidCalibration':
            raise ValueError("This file is not a valid PSFCentroidCalibration.")

        self.xfit = self.data[:, 0]
        self.yfit = self.data[:, 1]
        self.xfit_err = self.err[0][:, 0]
        self.yfit_err = self.err[0][:, 1]


class DispersionModel(Image):
    """ 
    Class for dispersion model parameter data structure

    Args:
        data_or_filepath (str or dict): either the filepath to the FITS file to read in OR the dictionary containing the dispersion data
        pri_hdr (fits.Header): Primary header.
        ext_hdr (fits.Header): Extension header.
        
    Attributes:
        data (dict): table containing the dispersion data
        clocking_angle (float): Clocking angle of the dispersion axis, theta,
        oriented in the direction of increasing wavelength, measured in degrees
        counterclockwise from the positive x-axis on the EXCAM data array
        (direction of increasing column index).
        clocking_angle_uncertainty (float): Uncertainty of the dispersion axis
        clocking angle in degrees.
        pos_vs_wavlen_polycoeff (numpy.ndarray): Polynomial fit to the
        source displacement on EXCAM along the dispersion axis as a function of
        wavelength, relative to the source position at the band reference
        wavelength (lambda_c = 730.0 nm for Band 3) in units of millimeters.
        pos_vs_wavlen_cov (numpy.ndarray): Covariance matrix of the
        polynomial coefficients
        wavlen_vs_pos_polycoeff (numpy.ndarray): Polynomial fit to the
        wavelength as a function of displacement along the dispersion axis on
        EXCAM, relative to the source position at the Band 3 reference
        wavelength (x_c at lambda_c = 730.0 nm) in units of nanometers. 
        wavlen_vs_pos_cov (numpy.ndarray): Covariance matrix of the
        polynomial coefficients
        params_key (list): key names of the parameters
    """
    
    params_key = ['clocking_angle', 'clocking_angle_uncertainty', 'pos_vs_wavlen_polycoeff', 'pos_vs_wavlen_cov', 'wavlen_vs_pos_polycoeff', 'wavlen_vs_pos_cov']
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None):
        if isinstance(data_or_filepath, str):
            # run the image class contructor
            super().__init__(data_or_filepath)
            # double check that this is actually a DispersionModel file that got read in
            # since if only a filepath was passed in, any file could have been read in
            if 'DATATYPE' not in self.ext_hdr:
                raise ValueError("File that was loaded was not a DispersionModel file.")
            if self.ext_hdr['DATATYPE'] != 'DispersionModel':
                raise ValueError("File that was loaded was not a DispersionModel file.")
        else:
            if not isinstance(data_or_filepath, dict):
                raise ValueError("Input should either be a dictionary or a filepath string")
            if pri_hdr == None:
                pri_hdr = fits.Header()
            if ext_hdr == None:
                ext_hdr = fits.Header()
            ext_hdr['DRPCTIME'] =  time.Time.now().isot
            ext_hdr['DRPVERSN'] =  corgidrp.__version__
            self.pri_hdr = pri_hdr
            self.ext_hdr = ext_hdr
            self.ext_hdr['DATATYPE'] = 'DispersionModel' # corgidrp specific keyword for saving to disk
            # add to history
            self.ext_hdr['HISTORY'] = "DispersionModel file created"
            #check that all parameters are available in the input dict
            for key in self.params_key:
                if key not in data_or_filepath:
                    raise ValueError("parameter {0} is missing in the data".format(key))
            data_list = Table(rows = [data_or_filepath])
            self.data = data_list
            self.filedir = "."
            # Use the last input file's name if available, else timestamp
            filetime = format_ftimeutc(pri_hdr['FILETIME'])
            self.filename = f"cgi_{pri_hdr['VISITID']}_{filetime}_dpm_cal.fits"
            self.pri_hdr['FILENAME'] = self.filename

        # initialization data passed in
        self.clocking_angle = self.data["clocking_angle"][0]
        self.clocking_angle_uncertainty = self.data["clocking_angle_uncertainty"][0]
        self.pos_vs_wavlen_polycoeff = np.array(self.data["pos_vs_wavlen_polycoeff"][0])
        self.pos_vs_wavlen_cov = np.array(self.data["pos_vs_wavlen_cov"][0])
        self.wavlen_vs_pos_polycoeff = np.array(self.data["wavlen_vs_pos_polycoeff"][0])
        self.wavlen_vs_pos_cov = np.array(self.data["wavlen_vs_pos_cov"][0])
        
        # Add err and dq attributes for walker compatibility (set to None since DispersionModel doesn't have these)
        self.err = None
        self.dq = None


    def save(self, filedir=None, filename=None):
        """
        Save file to disk with user specified filepath

        Args:
            filedir (str): filedir to save to. Use self.filedir if not specified
            filename (str): filepath to save to. Use self.filename if not specified
        """
        if filename is not None:
            self.filename = filename
        if filedir is not None:
            self.filedir = filedir

        if len(self.filename) == 0:
            raise ValueError("Output filename is not defined. Please specify!")

        prihdu = fits.PrimaryHDU(header=self.pri_hdr)
        exthdu = fits.BinTableHDU(data=self.data, header=self.ext_hdr)
        hdulist = fits.HDUList([prihdu, exthdu])

        hdulist.writeto(self.filepath, overwrite=True)
        hdulist.close()

class NonLinearityCalibration(Image):
    """
    Class for non-linearity calibration files. Although it's not strictly an image that you might look at, it is a 2D array of data

    The required format for calibration data is as follows:
     - Minimum 2x2
     - First value (top left) must be assigned to nan
     - Row headers (dn counts) must be monotonically increasing
     - Column headers (EM gains) must be monotonically increasing
     - Data columns (relative gain curves) must straddle 1
     - The first row will provide the the Gain axis values (accesssed via 
        gain_ax = non_lin_correction.data[0, 1:])
     - The first column will provide the "count" axis value (accessed via 
        count_ax = non_lin_correction.data[1:, 0])
     - The rest of the array will be the calibration data (accessed via 
     relgains = non_lin_correction.data[1:, 1:])

    For example:
    [
        [nan,  1,     10,    100,   1000 ], <- gain axis
        [1,    0.900, 0.950, 0.989, 1.000],
        [1000, 0.910, 0.960, 0.990, 1.010],
        [2000, 0.950, 1.000, 1.010, 1.050],
        [3000, 1.000, 1.001, 1.011, 1.060],
         ^
         count axis
    ],

    where the row headers [1, 1000, 2000, 3000] are dn counts, the column
    headers [1, 10, 100, 1000] are EM gains, and the first data column
    [0.900, 0.910, 0.950, 1.000] is the first of the four relative gain curves.

     Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file 
        to read in OR the 2D calibration data. See above for the required format.
        pri_hdr (astropy.io.fits.Header): the primary header (required only if 
        raw 2D data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required 
        only if raw 2D data is passed in)
        input_dataset (corgidrp.data.Dataset): the Image files combined 
        together to make this NonLinearityCalibration file (required only if 
        raw 2D data is passed in)
    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, 
                 input_dataset=None):

        # run the image class contructor
        super().__init__(data_or_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr)

        # File format checks - Ported from II&T
        nonlin_raw = self.data
        if nonlin_raw.ndim < 2 or nonlin_raw.shape[0] < 2 or \
        nonlin_raw.shape[1] < 2:
            raise ValueError('The non-linearity calibration array must be at' 
                             'least 2x2 (room for x and y axes and one data' 
                             'point)')
        if not np.isnan(nonlin_raw[0, 0]):
            raise ValueError('The first value of the non-linearity calibration '
                             'array  (upper left) must be set to "nan"')


        # additional bookkeeping for a calibration file
        # if this is a new calibration file, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if 
        # it is a new NonLinearityCalibration file
        if ext_hdr is not None:
            if input_dataset is None:
                # error check. this is required in this case
                raise ValueError("This appears to be a new Non Linearity "
                                 "Correction. The dataset of input files needs" 
                                 "to be passed in to the input_dataset keyword" 
                                 "to record history of this calibration file.")
            # corgidrp specific keyword for saving to disk
            self.ext_hdr['DATATYPE'] = 'NonLinearityCalibration' 

            # log all the data that went into making this calibration file
            self._record_parent_filenames(input_dataset)

            # add to history
            self.ext_hdr['HISTORY'] = "Non Linearity Calibration file created"

            # Enforce data level = CAL
            self.ext_hdr['DATALVL'] = 'CAL'

            # Follow filename convention as of R3.0.2
            self.filedir = '.'
            self.filename = re.sub('_l[0-9].', '_nln_cal', input_dataset[-1].filename)
            self.pri_hdr['FILENAME'] = self.filename

        # double check that this is actually a NonLinearityCalibration file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a NonLinearityCalibration file.")
        if self.ext_hdr['DATATYPE'] != 'NonLinearityCalibration':
            raise ValueError("File that was loaded was not a NonLinearityCalibration file.")
        self.dq_hdr['COMMENT'] = 'DQ not meaningful for this calibration; just present for class consistency' 
        
class KGain(Image):
    """
    Class for KGain calibration file. Until further insights it is just one float value.

    Args:
        data_or_filepath (str or float): either the filepath to the FITS file to read in OR the calibration data. See above for the required format.
        err (float): uncertainty value of kgain factor
        ptc (np.array): 2 column array with the photon transfer curve
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw data is passed in)
        err_hdr (astropy.io.fits.Header): the err extension header (required only if raw data is passed in)
        ptc_hdr (astropy.io.fits.Header): the ptc extension header (required only if raw data is passed in)
        input_dataset (corgidrp.data.Dataset): the Image files combined together to make this KGain file (required only if raw 2D data is passed in)

    Attrs:
        value: the getter of the kgain value
        _kgain (float): the value of kgain
        error: the getter of the kgain error value
        _kgain_error (float): the value of kgain error
    """
    def __init__(self, data_or_filepath, err = None, ptc = None, pri_hdr=None, ext_hdr=None, err_hdr = None, ptc_hdr = None, input_dataset = None):
       # run the image class contructor
        super().__init__(data_or_filepath, err=err, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err_hdr=err_hdr)

        # initialize these headers that have been recently added so that older calib files still contain this keyword when initialized and allow for tests that don't require 
        # these values to run smoothly; if these values are actually required for 
        # a particular process, the user would be alerted since these values below would result in an error as they aren't numerical
        if 'RN' not in self.ext_hdr:
            self.ext_hdr['RN'] = ''
        if 'RN_ERR' not in self.ext_hdr:
            self.ext_hdr['RN_ERR'] = ''
        # File format checks
        if self.data.shape != (1,):
            raise ValueError('The KGain calibration data should be just one float value')
        
        self._kgain = self.data[0] 
        self._kgain_error = self.err[0]
        
        if isinstance(data_or_filepath, str):
            # a filepath is passed in
            with fits.open(data_or_filepath) as hdulist:
                self.ptc_hdr = hdulist[3].header
                # ptc data is in FITS extension
                self.ptc = hdulist[3].data
        
        else:
            if ptc is not None:
                self.ptc = ptc
            else:
               self.ptc = np.zeros([2,0])
            if ptc_hdr is not None:
                self.ptc_hdr = ptc_hdr
            else:
                self.ptc_hdr = fits.Header()
        
        self.ptc_hdr["EXTNAME"] = "PTC"
        # additional bookkeeping for a calibration file
        # if this is a new calibration file, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if it is a new KGain file
        if ext_hdr is not None:
            if input_dataset is None:
                if 'DRPNFILE' not in ext_hdr:
                    # error check. this is required in this case
                    raise ValueError("This appears to be a new kgain. The dataset of input files needs to be passed in to the input_dataset keyword to record history of this kgain.")
                else:
                    pass
            else:
                # log all the data that went into making this calibration file
                self._record_parent_filenames(input_dataset)
                # give it a default filename using the last input file as the base
                self.filename = re.sub('_l[0-9].', '_krn_cal', input_dataset[-1].filename)
                self.pri_hdr['FILENAME'] = self.filename

            self.ext_hdr['DATATYPE'] = 'KGain' # corgidrp specific keyword for saving to disk
            self.ext_hdr['BUNIT'] = 'detected EM electron/DN'
            # add to history
            self.ext_hdr['HISTORY'] = "KGain Calibration file created"

            # Enforce data level = CAL
            self.ext_hdr['DATALVL']    = 'CAL'
        
        if err_hdr is not None:
            self.err_hdr['BUNIT'] = 'detected EM electron/DN'
        # double check that this is actually a KGain file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a KGain Calibration file.")
        if self.ext_hdr['DATATYPE'] != 'KGain':
            raise ValueError("File that was loaded was not a KGain Calibration file.")

    @property
    def value(self):
        return self._kgain
    
    @property
    def error(self):
        return self._kgain_error

    def save(self, filedir=None, filename=None):
        """
        Save file to disk with user specified filepath

        Args:
            filedir (str): filedir to save to. Use self.filedir if not specified
            filename (str): filepath to save to. Use self.filename if not specified
        """
        if filename is not None:
            self.filename = filename
        if filedir is not None:
            self.filedir = filedir

        if len(self.filename) == 0:
            raise ValueError("Output filename is not defined. Please specify!")

        prihdu = fits.PrimaryHDU(header=self.pri_hdr)
        exthdu = fits.ImageHDU(data=self.data, header=self.ext_hdr)
        hdulist = fits.HDUList([prihdu, exthdu])

        errhdu = fits.ImageHDU(data=self.err, header = self.err_hdr)
        hdulist.append(errhdu)

        ptchdu = fits.ImageHDU(data=self.ptc, header = self.ptc_hdr)
        hdulist.append(ptchdu)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=VerifyWarning) # fits save card length truncated warning
            hdulist.writeto(self.filepath, overwrite=True)
        hdulist.close()

class BadPixelMap(Image):
    """
    Class for bad pixel map. The bad pixel map indicates which pixels are hot
    pixels and thus unreliable. Note: These bad pixels are bad due to inherent
    nonidealities in the detector (applicable to any frame taken) and are
    separate from pixels marked per frame as contaminated by cosmic rays.

     Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR the 2D image data
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw 2D data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw 2D data is passed in)
        input_dataset (corgidrp.data.Dataset): the Image files combined together to make this bad pixel map (required only if raw 2D data is passed in)
    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, input_dataset=None):
        # run the image class contructor
        super().__init__(data_or_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr)

        # if this is a new bad pixel map, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if it is a new bad pixel map
        if ext_hdr is not None:
            if input_dataset is None and 'DRPNFILE' not in ext_hdr.keys():
                # error check. this is required in this case
                raise ValueError("This appears to be a new bad pixel map. The dataset of input files needs to be passed in to the input_dataset keyword to record history of this bad pixel map.")
            self.ext_hdr['DATATYPE'] = 'BadPixelMap'

            # log all the data that went into making this bad pixel map
            self._record_parent_filenames(input_dataset)

            # add to history
            self.ext_hdr['HISTORY'] = "Bad Pixel map created"

            # check whether we're making the bpmap from a flat only, or from L1/2 files. 
            if "_flt_cal" in input_dataset[-1].filename:
                self.filename = input_dataset[-1].filename.replace("_flt_cal", "_bpm_cal")
            else:
                self.filename = re.sub('_l[0-9].', '_bpm_cal', input_dataset[-1].filename)
            self.pri_hdr['FILENAME'] = self.filename          
            
            # Enforce data level = CAL
            self.ext_hdr['DATALVL']    = 'CAL'


        # double check that this is actually a bad pixel map that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a BadPixelMap file.")
        if self.ext_hdr['DATATYPE'] != 'BadPixelMap':
            raise ValueError("File that was loaded was not a BadPixelMap file.")
        self.dq_hdr['COMMENT'] = 'DQ not meaningful for this calibration; just present for class consistency' 
        self.err_hdr['COMMENT'] = 'err not meaningful for this calibration; just present for class consistency' 

class DetectorNoiseMaps(Image):
    """
    Class for DetectorNoiseMaps calibration file. The data is a 3-D stack of 3 frames, each of which is a full SCI frame of fitted
    values for a given noise type at a given temperature.  The 4th calibration product is bias offset, which is stored in the header.
    The 3 frames in the stack are in this order:
    index 0 for the fixed-pattern noise (FPN) map,
    index 1 for the clock-induced charge (CIC) map,
    index 2 for the dark current (DC) map.
    The input err should be a 4-D stack with first dimension of 1 and the next 3 corresponding to a 3-D stack with this order above.
    The input dq should be a 3-D stack corresponding to the order above.
    Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR the 3-D calibration data. See above for the required format.
        pri_hdr (astropy.io.fits.Header): the primary header (required only if data is passed in for data_or_filepath)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if data is passed in for data_or_filepath)
        input_dataset (corgidrp.data.Dataset): the input data combined together to make the noise maps (required only if data is passed in for data_or_filepath and if the filenames for the raw data used to make the calibration data are not already archived in ext_hdr)
        err (np.array): the error 3-D array (required only if data is passed in for data_or_filepath)
        dq (np.array): the 3-D DQ array (required only if data is passed in for data_or_filepath)
        err_hdr (astropy.io.fits.Header): the error header (required only if data is passed in for data_or_filepath)

    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, input_dataset=None, err = None, dq = None, err_hdr = None):
       # run the image class contructor
        super().__init__(data_or_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err=err, dq=dq, err_hdr=err_hdr)

        # File format checks
        if self.data.ndim != 3 or self.data.shape[0] != 3:
            raise ValueError('The DetectorNoiseMaps calibration data should be a 3-D array with the first dimension size equal to 3.')
        if self.err.ndim != 4 or self.err.shape[0] != 1: # conforms to usual format the Image class expects, with 1 as the first element of the shape for err
            raise ValueError('The DetectorNoiseMaps err data should be a 4-D array with the first dimension size equal to 4.')
        if self.dq.ndim != 3 or self.dq.shape[0] != 3:
            raise ValueError('The DetectorNoiseMaps dq data should be a 3-D array with the first dimension size equal to 3.')

        # required inputs, whether or not ext_hdr is None
        if "B_O" not in self.ext_hdr.keys() or "B_O_ERR" not in self.ext_hdr.keys():
                raise ValueError('Calibrated bias offset and its error should be present in header.')

        # additional bookkeeping for a calibration file
        # if this is a new calibration file, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if it is a new calibration file
        if ext_hdr is not None:
            if input_dataset is None and 'DRPNFILE' not in ext_hdr.keys():
                # error check. this is required in this case
                raise ValueError("This appears to be a new DetectorNoiseMaps instance. The dataset of input files needs to be passed in to the input_dataset keyword to record the history of the files that made the calibration products.")

            self.ext_hdr['DATATYPE'] = 'DetectorNoiseMaps' # corgidrp specific keyword for saving to disk
            self.ext_hdr['BUNIT'] = 'detected electron'
            # bias offset
            self.ext_hdr['B_O_UNIT'] = 'DN' # err unit is also in DN

            # log all the data that went into making this calibration file
            if 'DRPNFILE' not in ext_hdr.keys():
                self._record_parent_filenames(input_dataset)
            # add to history
            self.ext_hdr['HISTORY'] = "DetectorNoiseMaps calibration file created"

            # give it a default filename
            if input_dataset is not None:
                orig_input_filename = input_dataset.frames[-1].filename.split(".fits")[0]
            else:
                #running the calibration code gets the name right (based on last filename in input dataset); this is a standby
                orig_input_filename = self.ext_hdr['FILE0'].split(".fits")[0] 
            
            self.filename = "{0}_dnm_cal.fits".format(orig_input_filename)
            self.filename = re.sub('_l[0-9].', '', self.filename)
            self.pri_hdr['FILENAME'] = self.filename
            # Enforce data level = CAL
            self.ext_hdr['DATALVL']    = 'CAL'

        if err_hdr is not None:
            self.err_hdr['BUNIT'] = 'detected electron'

        # double check that this is actually a DetectorNoiseMaps file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a DetectorNoiseMaps Calibration file.")
        if self.ext_hdr['DATATYPE'] != 'DetectorNoiseMaps':
            raise ValueError("File that was loaded was not a DetectorNoiseMaps Calibration file.")

        #convenient attributes
        self.bias_offset = self.ext_hdr["B_O"]
        self.bias_offset_err = self.ext_hdr["B_O_ERR"]
        self.FPN_map = self.data[0]
        self.CIC_map = self.data[1]
        self.DC_map = self.data[2]
        self.FPN_err = self.err[0][0]
        self.CIC_err = self.err[0][1]
        self.DC_err = self.err[0][2]

class DetectorParams(Image):
    """
    Class containing detector parameters that may change over time. 

    To create a new instance of DetectorParams, you only need to pass in the values you would like to change from default values:
        new_valid_date = astropy.time.Time("2027-01-01")
        new_det_params = DetectorParams({'gmax' : 7500.0 }, date_valid=new_valid_date). 

    Args:
        data_or_filepath (dict or str): either a filepath string corresponding to an 
                                        existing DetectorParams file saved to disk or a
                                        dictionary of parameters to modify from default values
        date_valid (astropy.time.Time): date after which these parameters are valid

    Attributes:
        params (dict): the values for various detector parameters specified here
        default_values (dict): default values for detector parameters (fallback values)
        back_compat_mapping (dict): values to make test FITS files comply with new header standard
    """
    # default detector params
    default_values = {
        'KGAINPAR' : 8.7,
        'FWC_PP_E' : 90000.,
        'FWC_EM_E' : 100000.,
        'ROWREADT' : 223.5e-6,  # seconds
        'NEMGAIN': 604,         # number of EM gain register stages
        'TELRSTRT': -1,         # slice of rows that are used for telemetry
        'TELREND': None,        #goes to the end, in other words
        'CRHITRT': 5.0e+04,     # cosmic ray hit rate (hits/m**2/sec)
        'PIXAREA': 1.69e-10,    # pixel area (m**2/pixel)
        'GAINMAX': 8000.0,      # Maximum allowable EM gain
        'DELCNST': 1.0e-4,      # tolerance in exposure time calculator
        'OVERHEAD': 3,          # Overhead time, in seconds, for each collected frame.  Used to compute total wall-clock time for data collection
        'PCECNTMX': 0.1,        # Maximum allowed electrons/pixel/frame for photon counting
        'TFACTOR': 5,            # number of read noise standard deviations at which to set the photon-counting threshold
    }

    back_compat_mapping = {
        "KGAINPAR" : "KGAIN",
        "FWC_PP_E" : "FWC_PP",
        'FWC_EM_E' : 'FWC_EM',
        'ROWREADT' : "rowreadtime",
        "NEMGAIN" : "NEM",
        "TELRSTRT" : "telem_rows_start",
        "TELREND" : "telem_rows_end",
        "CRHITRT" : "X",
        "PIXAREA" : "A",
        "GAINMAX" : "GMAX",
        "DELCNST" : "delta_constr",
        "PCECNTMX" : "pc_ecount_max",
        "TFACTOR" : "T_FACTOR"
    }

    def __init__(self, data_or_filepath, date_valid=None):
        if date_valid is None:
            date_valid = time.Time.now()
        # if filepath passed in, just load in from disk as usual
        if isinstance(data_or_filepath, str):
            # run the image class contructor
            super().__init__(data_or_filepath)

            # double check that this is actually a DetectorParams file that got read in
            # since if only a filepath was passed in, any file could have been read in
            if 'DATATYPE' not in self.ext_hdr:
                raise ValueError("File that was loaded was not a DetectorParams file.")
            if self.ext_hdr['DATATYPE'] != 'DetectorParams':
                raise ValueError("File that was loaded was not a DetectorParams file.")
        else:
            if not isinstance(data_or_filepath, dict):
                raise ValueError("Input should either be a dictionary or a filepath string")
            pri_hdr = fits.Header()
            ext_hdr = fits.Header()
            ext_hdr['SCTSRT'] = date_valid.isot # use this for validity date
            ext_hdr['DRPVERSN'] =  corgidrp.__version__
            ext_hdr['DRPCTIME'] =  time.Time.now().isot

            # fill caldb required keywords with dummy data
            pri_hdr["OBSNUM"] = 000     
            ext_hdr["EXPTIME"] = 1
            ext_hdr['OPMODE'] = ""
            ext_hdr['EMGAIN_C'] = 1.0
            ext_hdr['EXCAMT'] = 40.0

            # Enforce data level = CAL?
            ext_hdr['DATALVL']    = 'CAL'

            # write default values to headers
            for key, value in self.default_values.items():
                ext_hdr[key] = value
            # overwrite default values
            for key, value in data_or_filepath.items():
                # to avoid VerifyWarning from fits
                ext_hdr[key] = value

            self.pri_hdr = pri_hdr
            self.ext_hdr = ext_hdr
            self.data = np.zeros([1,1])
            self.dq = np.zeros([1,1])
            self.err = np.zeros([1,1])

            self.err_hdr = fits.Header()
            self.dq_hdr = fits.Header()

            self.hdu_list = fits.HDUList()

        # make a dictionary that's easy to use
        self.params = {}
        # load back in all the values from the header
        for key in self.default_values:
            # if this key is not in the header, try the backwards compatability mapping
            new_key = key
            if key not in self.ext_hdr:
                key = self.back_compat_mapping[key]

            if len(key) > 8:
                # to avoid VerifyWarning from fits
                self.params[new_key] = self.ext_hdr['HIERARCH ' + key]
            else:
                self.params[new_key] = self.ext_hdr[key]


        # for backwards compatability:
        if "OBSID" in self.pri_hdr:
            self.pri_hdr['OBSNUM'] = self.pri_hdr['OBSID']
        if "CMDGAIN" in self.ext_hdr:
            self.ext_hdr["EMGAIN_C"] = self.ext_hdr['CMDGAIN']

        # if this is a new DetectorParams file, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if it is a new DetectorParams file
        if isinstance(data_or_filepath, dict):
            self.ext_hdr['DATATYPE'] = 'DetectorParams' # corgidrp specific keyword for saving to disk

            # add to history
            self.ext_hdr['HISTORY'] = "Detector Params file created"

            # use the start date for the filename by default
            self.filedir = "."
            self.filename = "DetectorParams_{0}.fits".format(self.ext_hdr['SCTSRT'])
            self.pri_hdr['FILENAME'] = self.filename

    def get_hash(self):
        """
        Computes the hash of the detector param values

        Returns:
            str: the hash of the detector parameters
        """
        hashing_str = "" # make a string that we can actually hash
        for key in self.params:
            hashing_str += str(self.params[key])

        return str(hash(hashing_str))
        
            
class AstrometricCalibration(Image):
    """
    Class for astrometric calibration file. 
    
    Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR a single array of calibration measurements of the following lengths (boresight: length 2 (RA, DEC), 
        plate scale: length 1 (float), north angle: length 1 (float), average offset: length 2 (floats) of average boresight offset in RA/DEC [deg],
        distortion coeffs: length dependent on order of polynomial fit but the last value should be an int describing the polynomial order). For a 
        3rd order distortion fit the input array should be length 37.
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw 2D data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw 2D data is passed in)
        
    Attrs:
        boresight (np.array): the corrected RA/DEC [deg] position of the detector center
        platescale (float): the platescale value in [mas/pixel]
        northangle (float): the north angle value in [deg]
        avg_offset (np.array): the average offset [deg] from the detector center
        distortion_coeffs (np.array): the array of legendre polynomial coefficients that describe the distortion map, where the last value of the array is the order of polynomial used

    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, err=None, input_dataset=None):
        # run the image class constructor
        super().__init__(data_or_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err=err)

        # File format checks
        if type(self.data) != np.ndarray:
            raise ValueError("The AstrometricCalibration data should be an array of calibration measurements")
        else:
            self.boresight = self.data[:2]
            self.platescale = self.data[2]
            self.northangle = self.data[3]
            self.avg_offset = self.data[4:6]
            self.distortion_coeffs = self.data[6:]
            
        # if this is a new astrometric calibration file, bookkeep it in the header
        # we need to check if it is new
        if ext_hdr is not None:
            if input_dataset is None:
                raise ValueError("This appears to be a new astrometric calibration file. The dataset of input files needs to be passed in to the input_dataset keyword to record its history.")
            self.ext_hdr['DATATYPE'] = 'AstrometricCalibration'

            # record all the data that went into making this calibration file
            self._record_parent_filenames(input_dataset)

            # add to history
            self.ext_hdr['HISTORY'] = "Astrometric Calibration file created"
            
            # give it a default filename using the first input file as the base
            # strip off everything starting at .fits
            orig_input_filename = input_dataset[-1].filename.split(".fits")[0]
            self.filename = "{0}_ast_cal.fits".format(orig_input_filename)
            self.filename = re.sub('_l[0-9].', '', self.filename)
            self.pri_hdr['FILENAME'] = self.filename
            
            # Enforce data level = CAL
            self.ext_hdr['DATALVL']    = 'CAL'

        # check that this is actually an AstrometricCalibration file that was read in
        if 'DATATYPE' not in self.ext_hdr or self.ext_hdr['DATATYPE'] != 'AstrometricCalibration':
            raise ValueError("File that was loaded was not an AstrometricCalibration file.")    
        self.dq_hdr['COMMENT'] = 'DQ not meaningful for this calibration; just present for class consistency'     
    
class TrapCalibration(Image):
    """

    Class for data related to charge traps that cause charge transfer inefficiency. 
    The calibration is generated by trap-pumped data. 

    The format will be [n,10], where each entry will have: 
    [row, column, sub-electrode location, index numnber of trap at this pixel/electrode, 
    capture time constant, maximum amplitude of the dipole, energy level of hole, 
    cross section for holes, R^2 value of fit, release time constant]

     Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR the 2D image data
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw 2D data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw 2D data is passed in)
        input_dataset (corgidrp.data.Dataset): the Image files combined together to make the trap calibration
    """
    def __init__(self,data_or_filepath, pri_hdr=None,ext_hdr=None, input_dataset=None):
        # run the image class constructor
        super().__init__(data_or_filepath,pri_hdr=pri_hdr, ext_hdr=ext_hdr)
        
        # if this is a new calibration, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if it is a new cal
        if ext_hdr is not None:
            if input_dataset is None:
                # error check. this is required in this case
                raise ValueError("This appears to be a new TrapCalibration. The dataset of input files needs to be "
                                 "passed in to the input_dataset keyword to record history of this TrapCalibration.")
            self.ext_hdr['DATATYPE'] = 'TrapCalibration' # corgidrp specific keyword for saving to disk

            # log all the data that went into making this dark
            self._record_parent_filenames(input_dataset)

            # add to history
            self.ext_hdr['HISTORY'] = "TrapCalibration created from {0} frames".format(self.ext_hdr['DRPNFILE'])

            # give it a default filename using the first input file as the base
            # strip off everything starting at .fits
            orig_input_filename = input_dataset[-1].filename.split(".fits")[0]
            self.filename = "{0}_tpu_cal.fits".format(orig_input_filename)
            self.filename = re.sub('_l[0-9].', '', self.filename)
            self.pri_hdr['FILENAME'] = self.filename

            # Enforce data level = CAL
            self.ext_hdr['DATALVL']    = 'CAL'

        # double check that this is actually a dark file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr or self.ext_hdr['DATATYPE'] != 'TrapCalibration':
            raise ValueError("File that was loaded was not a TrapCalibration file.")
        self.dq_hdr['COMMENT'] = 'DQ not meaningful for this calibration; just present for class consistency' 

class FluxcalFactor(Image):
    """
    Class containing the flux calibration factor (and corresponding error) for each band in unit erg/(s * cm^2 * AA)/photo-electrons/s. 

    To create a new instance of FluxcalFactor, you need to pass the value and error and the filter name in the ext_hdr:

    Args:
        data_or_filepath (str or float): either a filepath string corresponding to an 
                                        existing FluxcalFactor file saved to disk or the data and error float values of the
                                        flux cal factor of a certain filter defined in the header
        err (float): uncertainty value of fluxcal factor
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw data is passed in)
        err_hdr (astropy.io.fits.Header): the err extension header (required only if raw data is passed in)
        input_dataset (corgidrp.data.Dataset): the Image files combined together to make this FluxcalFactor file (required only if raw 2D data is passed in)
    
    Attributes:
        filter (str): used filter name
        nd_filter (str): used neutral density filter or "No"
        fluxcal_fac (float): the value of the flux cal factor for the corresponding filter
        fluxcal_err (float): the error of the flux cal factor for the corresponding filter
    """
    def __init__(self, data_or_filepath, err = None, pri_hdr=None, ext_hdr=None, err_hdr = None, input_dataset = None):
       # run the image class contructor
        super().__init__(data_or_filepath, err=err, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err_hdr=err_hdr)
        # if filepath passed in, just load in from disk as usual
        # File format checks
        if self.data.shape != (1,):
            raise ValueError('The FluxcalFactor calibration data should be just one float value')
        
        #TBC
        self.nd_filter = "ND0" #no neutral density filter in beam
        if 'FPAMNAME' in self.ext_hdr:
            name = self.ext_hdr['FPAMNAME']
            if name.startswith("ND"):
                self.nd_filter = name
        elif 'FSAMNAME' in self.ext_hdr:
            name = self.ext_hdr['FSAMNAME']
            if name.startswith("ND"):
                self.nd_filter = name
        else:
            raise ValueError('The FluxcalFactor calibration has no keyword FPAMNAME or FSAMNAME in the header')
        
        if 'CFAMNAME' in self.ext_hdr:
            self.filter = self.ext_hdr['CFAMNAME']
        else:
            raise ValueError('The FluxcalFactor calibration has no filter keyword CFAMNAME in the header')


        if isinstance(data_or_filepath, str):
            # double check that this is actually a FluxcalFactor file that got read in
            # since if only a filepath was passed in, any file could have been read in
            if 'DATATYPE' not in self.ext_hdr:
                raise ValueError("File that was loaded was not a FluxcalFactor file.")
            if self.ext_hdr['DATATYPE'] != 'FluxcalFactor':
                raise ValueError("File that was loaded was not a FluxcalFactor file.")
        else:
            self.ext_hdr['DRPVERSN'] =  corgidrp.__version__
            self.ext_hdr['DRPCTIME'] =  time.Time.now().isot
            
        # make some attributes to be easier to use
        self.fluxcal_fac = self.data[0]
        self.fluxcal_err =  self.err[0]

        # if this is a new FluxcalFactors file, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if it is a new FluxcalFactors file
        if ext_hdr is not None:
            if input_dataset is None:
                if 'DRPNFILE' not in ext_hdr:
                    # error check. this is required in this case
                    raise ValueError("This appears to be a new FluxcalFactor. The dataset of input files needs to be passed \
                                     in to the input_dataset keyword to record history of this FluxcalFactor file.")
                else:
                    pass
            else:
                # log all the data that went into making this calibration file
                self._record_parent_filenames(input_dataset)
                # give it a default filename using the first input file as the base
                # strip off everything starting at .fits
                orig_input_filename = input_dataset[-1].filename.split(".fits")[0]
  
            self.ext_hdr['DATATYPE'] = 'FluxcalFactor' # corgidrp specific keyword for saving to disk
            # JM: moved the below to fluxcal.py since it varies depending on the method
            #self.ext_hdr['BUNIT'] = 'erg/(s * cm^2 * AA)/(photoelectron/s)'
            #self.err_hdr['BUNIT'] = 'erg/(s * cm^2 * AA)/(photoelectron/s)'
            # add to history
            self.ext_hdr['HISTORY'] = "Flux calibration file created"

            # Enforce data level = CAL
            self.ext_hdr['DATALVL']    = 'CAL'

            # use the start date for the filename by default
            self.filedir = "."
            # slight hack for old mocks not in the stardard filename format
            self.filename = "{0}_abf_cal.fits".format(orig_input_filename)
            self.filename = re.sub('_L[0-9].', '', self.filename)
            self.pri_hdr['FILENAME'] = self.filename

class FpamFsamCal(Image):
    """
    Class containing the FPAM to EXCAM and FSAM to EXCAM transformation matrices.
    CGI model was consistent with FFT/TVAC tests. Transformation matrices are
    a 2x2 array with real values. Model cases are fpam_to_excam_modelbased and
    fsam_to_excam_modelbased, see below.

    Args:
        data_or_filepath (dict or str): either a filepath string corresponding to an
                                        existing FpamFsamCal file saved to disk or an
                                        array with the FPAM and FSAM rotation matrices
        date_valid (astropy.time.Time): date after which these parameters are valid

    Attributes:
         fpam_to_excam_modelbased (array): default values for FPAM rotation matrices.
         fsam_to_excam_modelbased (array): default values for FSAM rotation matrices.
         default_trans (array): array collecting fpam_to_excam_modelbased and
           fsam_to_excam_modelbased.
    """
    # default transformation matrices (model is consistent with FFT/TVAC tests)
    # Signs +/- have been double checked against FFT/TVAC data
    fpam_to_excam_modelbased = np.array([[ 0.        ,  0.12285012],
       [-0.12285012, 0.        ]], dtype=float)
    # Signs -/- have been double checked against FFT/TVAC data
    fsam_to_excam_modelbased = np.array([[-0.        , -0.09509319],
       [-0.09509319, 0.        ]], dtype=float)
    default_trans = np.array([fpam_to_excam_modelbased, fsam_to_excam_modelbased])

    ###################
    ### Constructor ###
    ###################

    def __init__(self, data_or_filepath, date_valid=None):
        if date_valid is None:
            date_valid = time.Time.now()
        # if filepath passed in, just load in from disk as usual
        if isinstance(data_or_filepath, str):
            # run the image class contructor
            super().__init__(data_or_filepath)

            # double check that this is actually a FpamFsamCal file that got read in
            # since if only a filepath was passed in, any file could have been read in
            if 'DATATYPE' not in self.ext_hdr:
                raise ValueError('File that was loaded was not a FpamFsamCal file.')
            if self.ext_hdr['DATATYPE'] != 'FpamFsamCal':
                raise ValueError('File that was loaded was not a FpamFsamCal file.')
        else:
            if len(data_or_filepath) == 0:
                data_or_filepath = self.default_trans
            elif not isinstance(data_or_filepath, np.ndarray):
                raise ValueError('Input should either be an array or a filepath string.')
            if data_or_filepath.shape != (2,2,2):
                raise ValueError('FpamFsamCal must be a 2x2x2 array')
            prihdr = fits.Header()
            exthdr = fits.Header()
            exthdr['SCTSRT'] = date_valid.isot # use this for validity date
            exthdr['DRPVERSN'] =  corgidrp.__version__
            exthdr['DRPCTIME'] =  time.Time.now().isot

            # fill caldb required keywords with dummy data
            prihdr['OBSNUM'] = 0
            exthdr["EXPTIME"] = 0
            exthdr['OPMODE'] = ""
            exthdr['EMGAIN_C'] = 1.0
            exthdr['EXCAMT'] = 40.0

            self.pri_hdr = prihdr
            self.ext_hdr = exthdr
            self.data = data_or_filepath
            self.dq = self.data * 0
            self.err = self.data * 0

            self.err_hdr = fits.Header()
            self.dq_hdr = fits.Header()

            self.hdu_list = fits.HDUList()

        # if this is a new FpamFsamCal file, we need to bookkeep it in the
        # header b/c of logic in the super.__init__, we just need to check this
        # to see if it is a new FpamFsamCal file
        if isinstance(data_or_filepath, np.ndarray):
            # corgidrp specific keyword for saving to disk
            self.ext_hdr['DATATYPE'] = 'FpamFsamCal' 

            # add to history
            self.ext_hdr['HISTORY'] = 'FPAM/FSAM rotation matrices file created'

            # use the start date for the filename by default
            self.filedir = '.'
            self.filename = "FpamFsamCal_{0}.fits".format(self.ext_hdr['SCTSRT'])
            self.pri_hdr['FILENAME'] = self.filename

            # Enforce data level = CAL
            self.ext_hdr['DATALVL']    = 'CAL'

class CoreThroughputCalibration(Image):
    """
    Class containing a core throughput calibration file

    A CoreThroughput calibration file has two main data arrays:

    3-d cube of PSF images, e.g, a N1xN1xN array where N1= +/- 3l/D about
      PSF's centroid in EXCAM pixels. The N PSF images are the ones in the CT
      dataset.

    N sets of (x, y, CT measurements). The (x, y) are pixel coordinates of the
      PSF images in the CT dataset wrt EXCAM (0,0) pixel during core throughput
      observation.

    Args:
      data_or_filepath (array or str): either a filepath string corresponding
        to an existing CoreThroughputCalibration file saved to disk or an array
        with the elements of the core throughput calibration file.
      hdu_list (astropy.io.fits.HDUList): an astropy HDUList object that
        contains the elements of the core throughput calibration file.
    """

    ###################
    ### Constructor ###
    ###################

    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, err=None,
        dq=None, err_hdr=None, dq_hdr=None, input_hdulist=None,
        input_dataset=None):
        # run the image class contructor
        super().__init__(data_or_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr,
            err=err, dq=dq, err_hdr=err_hdr, dq_hdr=dq_hdr,
            input_hdulist=input_hdulist)

        # Verify the extension header corresponds to the PSF cube
        if self.ext_hdr['EXTNAME'] != 'PSFCUBE':
            raise ValueError('The input data does not seem to contain the PSF '
                'cube of measurements')

        # CT array measurements on EXCAM
        idx_hdu = 0
        if self.hdu_list[idx_hdu].name == 'CTEXCAM':
            self.ct_excam = self.hdu_list[idx_hdu].data
            self.ct_excam_hdr = self.hdu_list[idx_hdu].header
        else:
            raise ValueError('The HDU list does not seem to contain the CT '
                'array of measurements')
        # FPAM positions during CT observations
        idx_hdu = 1
        if self.hdu_list[idx_hdu].name == 'CTFPAM':
            self.ct_fpam = self.hdu_list[idx_hdu].data
            self.ct_fpam_hdr = self.hdu_list[idx_hdu].header
        else:
            raise ValueError('The HDU list does not seem to contain the FPAM '
                'value during core throughput observations')
        # FSAM positions during CT observations
        idx_hdu = 2
        if self.hdu_list[idx_hdu].name == 'CTFSAM':
            self.ct_fsam = self.hdu_list[idx_hdu].data
            self.ct_fsam_hdr = self.hdu_list[idx_hdu].header
        else:
            raise ValueError('The HDU list does not seem to contain the FPAM '
                'value during core throughput observations')

        # File format checks 
        # Check PSF basis is a 3D set
        if self.data.ndim != 3:
            raise ValueError('The PSF basis is an (N,N1,N1) array with N PSFs, '
                'each with N1 pixels x N1 pixels.')
        # Check CT array is a 3xN set
        if len(self.ct_excam) != 3:
            raise ValueError('The core throughput array of measurements is a '
                'Nx3 array.')
        # Check the CT map has the same number of elements as the PSF cube
        if self.ct_excam.shape[1] != self.data.shape[0]:
            raise ValueError('The core throughput map must have one PSF location '
                'and CT value for each PSF.')

        # Additional bookkeeping for a calibration file:
        # If this is a new calibration file, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if
        # it is a new CoreThroughputCalibration file
        if ext_hdr is not None:
            if input_dataset is None:
                # error check. this is required in this case
                raise ValueError('This appears to be a new Core Throughput calibration'
                                 'File. The dataset of input files needs'
                                 'to be passed in to the input_dataset keyword'
                                 'to record history of this calibration file.')
            # corgidrp specific keyword for saving to disk
            self.ext_hdr['DATATYPE'] = 'CoreThroughputCalibration'

            # log all the data that went into making this calibration file
            self._record_parent_filenames(input_dataset)

            # add to history if not present
            if not 'HISTORY' in self.ext_hdr:
                self.ext_hdr['HISTORY'] = ('Core Throughput calibration derived '
                    f'from a set of frames on {self.ext_hdr["DATETIME"]}')

            # Default convention: replace _l3_.fits from the filename of the
            # input dataset by _ctp_cal.fits
            self.filedir = '.'
            self.filename = re.sub('_l[0-9].', '_ctp_cal', input_dataset[-1].filename)
            self.pri_hdr['FILENAME'] = self.filename

            # Enforce data level = CAL
            self.ext_hdr['DATALVL']    = 'CAL'

        # double check that this is actually a NonLinearityCalibration file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a CoreThroughputCalibration file.")
        if self.ext_hdr['DATATYPE'] != 'CoreThroughputCalibration':
            raise ValueError("File that was loaded was not a CoreThroughputCalibration file.")

    ###############
    ### Methods ###
    ###############

    def GetCTFPMPosition(self,
        corDataset,
        fpamfsamcal):
        """ Gets the FPM's center during a Core throughput observing sequence.

        The use of the delta FPAM/FSAM positions and the rotation matrices is
        based on the prescription provided on 1/14/25: "H/V values to EXCAM
        row/column pixels"

          delta_pam = np.array([[dh], [dv]]) # fill these in
          M = np.array([[ M00, M01], [M10, M11]], dtype=float32)
          delta_pix = M @ delta_pam
        
        Args:
          corDataset (corgidrp.data.Dataset): a dataset containing some
              coronagraphic observations.
          fpamfsamcal (corgidrp.data.FpamFsamCal): an instance of the
              FpamFsamCal class.

        Returns:
            Returns the FPM's center during a Core throughput observing sequence.
        """
        # Read FPM location during the coronagraphic observations
        cor_fpm_center = np.array([corDataset[0].ext_hdr['STARLOCX'],
            corDataset[0].ext_hdr['STARLOCY']])
        # Read FPAM and FSAM values during the coronagraphic observations
        cor_fpam = np.array([corDataset[0].ext_hdr['FPAM_H'],
            corDataset[0].ext_hdr['FPAM_V']])
        cor_fsam = np.array([corDataset[0].ext_hdr['FSAM_H'],
            corDataset[0].ext_hdr['FSAM_V']])
        # Compute delta FPAM and delta FSAM
        delta_fpam_um = self.ct_fpam - cor_fpam
        delta_fsam_um = self.ct_fsam - cor_fsam
        # Follow the array prescription from the doc string
        delta_fpam_um = np.array([delta_fpam_um]).transpose()
        delta_fsam_um = np.array([delta_fsam_um]).transpose()
        # Get the shift in EXCAM pixels for FPAM and FSAM
        delta_fpam_excam = (fpamfsamcal.data[0] @ delta_fpam_um).transpose()[0]
        delta_fsam_excam = (fpamfsamcal.data[1] @ delta_fsam_um).transpose()[0]
        # Return the FPAM and FSAM centers during the core throughput observations
        return cor_fpm_center + delta_fpam_excam, cor_fpm_center + delta_fsam_excam

    def InterpolateCT(self,
        x_cor,
        y_cor,
        corDataset,
        fpamfsamcal,
        logr=False):
        """ Interpolate CT value at a desired position of a coronagraphic
            observation.

        First implementation based on Max Millar-Blanchaer's suggestions
        https://collaboration.ipac.caltech.edu/display/romancoronagraph/Max%27s+Interpolation+idea

        Here we assume that the core throughput measurements with the star
        located along a series of radial spikes at various azimuths. 

        It throws an error if the radius of the new points is outside the range
        of the input radii. If the azimuth is greater than the maximum azimuth
        of the core throughput dataset, it will mod the azimuth to be within
        the range of the input azimuths.

        Assumes that the input core_thoughput is between 0 and 1.

        # TODO: review accuracy of the method with simulated data that are more
        # representative of future mission data, including error budget and 
        # expected azimuthal dependence on the CT.

        Args:
          x_cor (numpy.ndarray): Values of the first dimension of the
              target locations where the CT will be interpolated. Locations are
              EXCAM pixels measured with respect to the FPM's center.
          y_cor (numpy.ndarray): Values of the second dimension of the
              target locations where the CT will be interpolated. Locations are
              EXCAM pixels measured with respect to the FPM's center.
          corDataset (corgidrp.data.Dataset): a dataset containing some
              coronagraphic observations.
          fpamfsamcal (corgidrp.data.FpamFsamCal): an instance of the
              FpamFsamCal calibration class.
          logr (bool) (optional): If True, radii are mapped into their logarithmic
              values before constructing the interpolant.

        Returns:
          Returns interpolated value of the CT, first, and positions for valid
            locations as a numpy ndarray.
        """
        if isinstance(x_cor, np.ndarray) is False:
            if isinstance(x_cor, int) or isinstance(x_cor, float):
                x_cor = np.array([x_cor])
            elif isinstance(x_cor, list):
                x_cor = np.array(x_cor)
            else:
                raise ValueError('Target locations must be a scalar '
                    '(int or float), list of int or float values, or '
                    ' a numpy.ndarray')
        if isinstance(y_cor, np.ndarray) is False:
            if isinstance(y_cor, int) or isinstance(y_cor, float):
                y_cor = np.array([y_cor])
            elif isinstance(y_cor, list):
                y_cor = np.array(y_cor)
            else:
                raise ValueError('Target locations must be a scalar '
                    '(int or float), list of int or float values, or '
                    ' a numpy.ndarray')

        if len(x_cor) != len(y_cor):
            raise ValueError('Target locations must have the same number of '
                'elements')
  
        # Get FPM's center during CT observations
        fpam_ct_pix_out = self.GetCTFPMPosition(
                corDataset,
                fpamfsamcal)[0]
        # Get CT measurements relative to CT FPM's center
        x_grid = self.ct_excam[0,:] - fpam_ct_pix_out[0]
        y_grid = self.ct_excam[1,:] - fpam_ct_pix_out[1]
        core_throughput = self.ct_excam[2,:]
        # Algorithm
        radii = np.sqrt(x_grid**2 + y_grid**2)

        # We'll need to mod the input azimuth, so let's subtract the
        # minimum azimuth so we are relative to zero
        azimuths = np.arctan2(y_grid, x_grid)
        azimuth0 = azimuths.min()
        azimuths = azimuths - azimuth0
        # Now we can create a 2D array of the radii and azimuths
        
        # Calculate the new datapoint in the right radius and azimuth coordinates
        radius_cor = np.sqrt(x_cor**2 + y_cor**2)
       
        # Remove interpolation locations that are outside the radius range
        r_good = radius_cor >= radii.min()
        
        if len(x_cor[r_good]) == 0:
            raise ValueError('All target radius are less than the minimum '
                'radius in the core throughout data: {:.2f} EXCAM pixels'.format(radii.min()))
        radius_cor = radius_cor[r_good]
        # Update x_cor and y_cor
        x_cor = x_cor[r_good]
        y_cor = y_cor[r_good]
        r_good = radius_cor <= radii.max()
        if len(x_cor[r_good]) == 0:
            raise ValueError('All target radius are greater than the maximum '
                'radius in the core throughout data: {:.2f} EXCAM pixels'.format(radii.max()))
        radius_cor = radius_cor[r_good]
        # Update x_cor and y_cor
        x_cor = x_cor[r_good]
        y_cor = y_cor[r_good]

        # We'll need to mod the input azimuth, so let's subtract the minimum azimuth so we are relative to zero.
        azimuth_cor = np.arctan2(y_cor, x_cor) - azimuth0
       
        # MOD this azimuth so that we're in the right range: all angles will be
        # within [0, azimuths.max()), including negative values
        azimuth_cor = azimuth_cor % azimuths.max()
       
        if logr: 
            radii = np.log10(radii)
            radius_cor = np.log10(radius_cor)
       
        rad_az = np.c_[radii, azimuths]
        # Make the interpolator
        interpolator = LinearNDInterpolator(rad_az, core_throughput)
        # Now interpolate: 
        interpolated_values = interpolator(radius_cor, azimuth_cor)

        # Raise ValueError if CT < 0, CT> 1
        if np.any(interpolated_values < 0) or np.any(interpolated_values > 1): 
            raise ValueError('Some interpolated core throughput values are '
                f'out of bounds (0,1): ({interpolated_values.min():.2f}, '
                f'{interpolated_values.max():.2f})')

        # Edge case:
        # If a target location happens to be part of the CT dataset (i.e., the
        # interpolator) and its azimuth is equal to the maximum azimuth in the
        # CT dataset, the interpolated CT may sometimes be assigned to NaN, while
        # it should simply be the same inout CT value at the same location
        idx_az_max = np.argwhere(np.isnan(interpolated_values))
        for idx in idx_az_max:
            idx_x_arr = np.argwhere(x_cor[idx] == x_grid)
            idx_y_arr = np.argwhere(y_cor[idx] == y_grid)
            for idx_x in idx_x_arr:
                # If and only if the same index is in both, it's the same location
                if idx_x in idx_y_arr:
                    interpolated_values[idx_x] = core_throughput[idx_x]
            
        # Raise ValueError if all CT are NaN
        if np.all(np.isnan(interpolated_values)):
            raise ValueError('There are no valid target positions within the ' +
                'range of input PSF locations')

        # Extrapolation: Remove NaN values
        is_valid = np.where(np.isnan(interpolated_values) == False)[0]
        return np.array([interpolated_values[is_valid],
            x_cor[is_valid],
            y_cor[is_valid]])

    def GetPSF(self,
        x_cor,
        y_cor,
        corDataset,
        fpamfsamcal,
        method='nearest-polar'):
        """ Get a PSF at a given (x,y) location on HLC in a coronagraphic
        observation given a CT calibration file and the PAM transformation from
        encoder values to EXCAM pixels.

        First implementation: nearest PSF in a polar sense. See below.

        # TODO: Implement an interpolation method that takes into account other
        # PSF than the nearest one. Comply with any required precision from the
        # functions that will use the interpolated PSF. 

        Args:
          x_cor (numpy.ndarray): Values of the first dimension of the
              target locations where the CT will be interpolated. Locations are
              EXCAM pixels measured with respect to the FPM's center.
          y_cor (numpy.ndarray): Values of the second dimension of the
              target locations where the CT will be interpolated. Locations are
              EXCAM pixels measured with respect to the FPM's center.
          corDataset (corgidrp.data.Dataset): a dataset containing some
              coronagraphic observations.
          fpamfsamcal (corgidrp.data.FpamFsamCal): an instance of the
              FpamFsamCal class. That is, a FpamFsamCal calibration file.
          method (str): Interpolation method that will be used:
              'polar-nearest': Given an (x,y) position wrt FPM's center, the
               associated PSF is the one in the CT calibration dataset whose
               radial distance to the FPM's center is the closest to
               sqrt(x**2+y**2). If there is more than one CT PSF at the same
               radial distance, choose the one whose angular distance to the
               (x,y) location is the smallest.
              
        Returns:
          psf_interp_list (array): Array of interpolated PSFs for the valid
              target locations.
          x_interp_list (array): First dimension of the list of valid target positions. 
          y_interp_list (array): Second dimension of the list of valid target positions.
        """
        if isinstance(x_cor, np.ndarray) is False:
            if isinstance(x_cor, int) or isinstance(x_cor, float):
                x_cor = np.array([x_cor])
            elif isinstance(x_cor, list):
                x_cor = np.array(x_cor)
            else:
                raise ValueError('Target locations must be a scalar '
                    '(int or float), list of int or float values, or '
                    ' a numpy.ndarray')
        if isinstance(y_cor, np.ndarray) is False:
            if isinstance(y_cor, int) or isinstance(y_cor, float):
                y_cor = np.array([y_cor])
            elif isinstance(y_cor, list):
                y_cor = np.array(y_cor)
            else:
                raise ValueError('Target locations must be a scalar '
                    '(int or float), list of int or float values, or '
                    ' a numpy.ndarray')

        if len(x_cor) != len(y_cor):
            raise ValueError('Target locations must have the same number of '
                'elements')

        # We need to translate the PSF locations in the CT cal file to be with
        # respect to the FPM's center during CT observations:
        # Get FPM's center during CT observations
        fpam_ct_pix_out = self.GetCTFPMPosition(
                corDataset,
                fpamfsamcal)[0]
        # Get CT measurements relative to CT FPM's center
        x_grid = self.ct_excam[0,:] - fpam_ct_pix_out[0]
        y_grid = self.ct_excam[1,:] - fpam_ct_pix_out[1]
        # Algorithm:
        # Radial distances wrt FPM's center
        radii = np.sqrt(x_grid**2 + y_grid**2)
        # Azimuths
        azimuths = np.arctan2(y_grid, x_grid)

        # Radial distances of the target locations
        radius_cor = np.sqrt(x_cor**2 + y_cor**2)

        # Remove interpolation locations that are outside the radius range
        r_good = radius_cor >= radii.min()
        if len(x_cor[r_good]) == 0:
            raise ValueError('All target radius are less than the minimum '
                'radius in the core throughout data: {:.2f} EXCAM pixels'.format(radii.min()))
        radius_cor = radius_cor[r_good]
        # Update x_cor and y_cor
        x_cor = x_cor[r_good]
        y_cor = y_cor[r_good]
        r_good = radius_cor <= radii.max()
        if len(x_cor[r_good]) == 0:
            raise ValueError('All target radius are either less than the minimum'
                ' radius or greater than the maximum radius in the core throughout'
                ' data: {:.2f} EXCAM pixels'.format(radii.max()))
        radius_cor = radius_cor[r_good]
        # Update x_cor and y_cor
        x_cor = x_cor[r_good]
        y_cor = y_cor[r_good]
        r_cor = np.sqrt(x_cor**2 + y_cor**2)

        psf_interp_list = []
        x_interp_list = []
        y_interp_list = []
        if method.lower() == 'nearest-polar':
            for i_psf in range(len(x_cor)):
                # Agreeement for this nearest method is that radial distances are
                # binned at 1/10th of a pixel. This will be unnecessary as soon as
                # there's any other interpolation method than the 'nearest' one.
                # Find the nearest radial position in the CT file (argmin()
                # returns the first occurence only)
                diff_r_abs = np.round(10*np.abs(r_cor[i_psf] - radii)/10)
                idx_near = np.argwhere(diff_r_abs == diff_r_abs.min())
                # If there's more than one case, select that one with the
                # smallest angular distance
                if len(idx_near) > 1:
                    print("More than one PSF found with the same radial distance from the FPM's center")
                    # Difference in angle b/w target and grid
                    # We want to distinguish PSFs at different quadrants
                    az_grid = np.arctan2(y_grid[idx_near], x_grid[idx_near])
                    az_cor = np.arctan2(y_cor[i_psf], x_cor[i_psf])
                    # Flatten into a 1-D array
                    diff_az_abs = np.abs(az_cor - az_grid).transpose()[0]
                    # Azimuth binning consistent with the binning of the radial distance
                    bin_az_fac = 1/10/r_cor[i_psf]
                    diff_az_abs = bin_az_fac * np.round(diff_az_abs/bin_az_fac)
                    # Closest angular location to the target location within equal radius
                    idx_near_az = np.argwhere(diff_az_abs == diff_az_abs.min())
                    # If there are two locations (half angle), choose the average (agreement)
                    if len(idx_near_az) == 2: 
                        psf_interp = np.squeeze(self.data[idx_near[idx_near_az]]).mean(axis=0)
                    # Otherwise, this is the PSF
                    elif len(idx_near_az) == 1:
                        psf_interp = np.squeeze(self.data[idx_near[idx_near_az[0]]])
                    else:
                        raise ValueError(f'There are {len(idx_near_az):d} PSFs ',
                            'equally near the target PSF. This should not happen.')
                # Otherwise this is the interpolated PSF (nearest)
                elif len(idx_near) == 1:
                    psf_interp = np.squeeze(self.data[idx_near[0]])
                # This should not happen b/c there should always be a closest radius
                else:
                    raise Exception('No closest radial distance found. This should not happen.')

                # Add valid case
                psf_interp_list += [psf_interp]
                x_interp_list += [x_cor[i_psf]]
                y_interp_list += [y_cor[i_psf]]
        else:
            raise ValueError(f'Unidentified method for the interpolation: {method}')

        return np.array(psf_interp_list), np.array(x_interp_list), np.array(y_interp_list)

class CoreThroughputMap(Image):
    """ Class containing a corethroughput map.

    The corethroughput map consists of M sets of (x, y, CT estimated). The
      (x, y) are pixel coordinates wrt the FPM's center. More details about the
      corethroughput map array can be found in the class method create_ct_map().

    Args:
      data_or_filepath (array or str): either the filepath to the FITS file to
      read in OR the 2D image data. The FITS file or data must be from a
      coronagraphic observation because the FPM's center is needed during the
      creation of the corethroughput map.
    """

    ###################
    ### Constructor ###
    ###################

    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, err=None, input_dataset=None):
        # run the image class constructor
        super().__init__(data_or_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr, err=err)

        # Check it has the FPM's center information 
        if ('STARLOCX' not in self.ext_hdr) or ('STARLOCY' not in self.ext_hdr):
            raise ValueError('The input dataset does not contain the information'
                'about the FPM center')
        # Check data have the expected format (x,y,ct)
        if isinstance(data_or_filepath, str) is False:
            data_or_filepath.shape[0] == 3
            if data_or_filepath[2,:].max() > 1 or data_or_filepath[2,:].min() < 0:
                raise ValueError('Corethroughput map values should be within 0 and 1')

        # Additional bookkeeping for a calibration file:
        # If this is a new calibration file, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if
        # it is a new CoreThroughputMap file
        if ext_hdr is not None:
            if input_dataset is None:
                # error check. this is required in this case
                raise ValueError('This appears to be a new CoreThroughputMap '
                                 'file. The dataset of input files needs '
                                 'to be passed in to the input_dataset keyword '
                                 'to record history of this calibration file.')
            # corgidrp specific keyword for saving to disk
            self.ext_hdr['DATATYPE'] = 'CoreThroughputMap'

            # log all the data that went into making this calibration file
            self._record_parent_filenames(input_dataset)

            # add to history if not present
            if not 'HISTORY' in self.ext_hdr:
                self.ext_hdr['HISTORY'] = ('CoreThroughputMap derived '
                    f'from a set of frames on {self.ext_hdr["DATETIME"]}')

            # The corethroughput map is not a calibration product as of writing
            # this class. The filename does not follow the convention for
            # calibration files
            self.filedir = '.'
            self.filename = 'corethroughput_map.fits'
            self.pri_hdr['FILENAME'] = self.filename

            # Enforce data level = L3
            self.ext_hdr['DATALVL']    = 'L3'

        # Keep track of the coronagraphic files used to create the CT map
        if input_dataset is not None:
            self._record_parent_filenames(input_dataset)
        # double check that this is actually a NonLinearityCalibration file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a CoreThroughputMap file.")
        if self.ext_hdr['DATATYPE'] != 'CoreThroughputMap':
            raise ValueError("File that was loaded was not a CoreThroughputMap file.")

class PyKLIPDataset(pyKLIP_Data):
    """
    A pyKLIP instrument class for Roman Coronagraph Instrument data.

    # TODO: Add more bandpasses, modes to self.wave_hlc
    #       Add wcs header info!

    Attrs:
        input: Input corgiDRP dataset.
        centers: Star center locations.
        filenums: file numbers.
        filenames: file names.
        PAs: position angles.
        wvs: wavelengths.
        wcs: WCS header information. Currently None.
        IWA: inner working angle.
        OWA: outer working angle.
        psflib: corgiDRP dataset containing reference PSF observations.
        output: PSF subtracted pyKLIP dataset

    """
    
    ####################
    ### Constructors ###
    ####################
    
    def __init__(self,
                 dataset,
                 psflib_dataset=None,
                 highpass=False):
        """
        Initialize the pyKLIP instrument class for space telescope data.
        # TODO: Determine inner working angle based on PAM positions
                    - Inner working angle based on Focal plane mask (starts with HLC) + color filter ('1F') for primary mode
                    - Outer working angle based on field stop? (should be R1C1 or R1C3 for primary mode)
        
        Args:
            dataset (corgidrp.data.Dataset):
                Dataset containing input science observations.
            psflib_dataset (corgidrp.data.Dataset, optional):
                Dataset containing input reference observations. The default is None.
            highpass (bool, optional):
                Toggle to do highpass filtering. Defaults fo False.
        """
        
        # Initialize pyKLIP Data class.
        super(PyKLIPDataset, self).__init__()

        # Set filter wavelengths
        self.wave_hlc = {'1F': 575e-9} # meters
            
        # Read science and reference files.
        self.readdata(dataset, psflib_dataset, highpass)
        
        pass
    
    ################################
    ### Instance Required Fields ###
    ################################
    
    @property
    def input(self):
        return self._input
    @input.setter
    def input(self, newval):
        self._input = newval
    
    @property
    def centers(self):
        return self._centers
    @centers.setter
    def centers(self, newval):
        self._centers = newval
    
    @property
    def filenums(self):
        return self._filenums
    @filenums.setter
    def filenums(self, newval):
        self._filenums = newval
    
    @property
    def filenames(self):
        return self._filenames
    @filenames.setter
    def filenames(self, newval):
        self._filenames = newval
    
    @property
    def PAs(self):
        return self._PAs
    @PAs.setter
    def PAs(self, newval):
        self._PAs = newval
    
    @property
    def wvs(self):
        return self._wvs
    @wvs.setter
    def wvs(self, newval):
        self._wvs = newval
    
    @property
    def wcs(self):
        return self._wcs
    @wcs.setter
    def wcs(self, newval):
        self._wcs = newval
    
    @property
    def IWA(self):
        return self._IWA
    @IWA.setter
    def IWA(self, newval):
        self._IWA = newval
    
    @property
    def OWA(self):
        return self._OWA
    @OWA.setter
    def OWA(self, newval):
        self._OWA = newval
    
    @property
    def psflib(self):
        return self._psflib
    @psflib.setter
    def psflib(self, newval):
        self._psflib = newval
    
    @property
    def output(self):
        return self._output
    @output.setter
    def output(self, newval):
        self._output = newval
    
    ###############
    ### Methods ###
    ###############
    
    def readdata(self,
                 dataset,
                 psflib_dataset,
                 highpass=False):
        """
        Read the input science observations.
        
        Args:
            dataset (corgidrp.data.Dataset):
                Dataset containing input science observations.
            psflib_dataset (corgidrp.data.Dataset, optional):
                Dataset containing input reference observations. The default is None.
            highpass (bool, optional):
                Toggle to do highpass filtering. Defaults fo False.
        """
        
        # Check input.
        if not isinstance(dataset, corgidrp.data.Dataset):
            raise UserWarning('Input dataset is not a corgidrp Dataset object.')
        if len(dataset) == 0:
            raise UserWarning('No science frames in the input dataset.')
        
        if not psflib_dataset is None:
            if not isinstance(psflib_dataset, corgidrp.data.Dataset):
                raise UserWarning('Input psflib_dataset is not a corgidrp Dataset object.')
        
        # Loop through frames.
        input_all = []
        centers_all = []  # pix
        filenames_all = []
        PAs_all = []  # deg
        wvs_all = []  # m
        wcs_all = []
        PIXSCALE = []  # arcsec

        psflib_data_all = []
        psflib_centers_all = []  # pix
        psflib_filenames_all = []

        # Iterate over frames in dataset

        for i, frame in enumerate(dataset):
            
            phead = frame.pri_hdr
            shead = frame.ext_hdr
                
            if 'TELESCOP' in phead:
                TELESCOP = phead['TELESCOP']
                if TELESCOP != "ROMAN":
                    raise UserWarning('Data is not from Roman Space Telescope Coronagraph Instrument. TELESCOP = {0}'.format(TELESCOP))
            INSTRUME = phead['INSTRUME']
            if INSTRUME != "CGI":
                raise UserWarning('Data is not from Roman Space Telescope Coronagraph Instrument. INSTRUME = {0}'.format(INSTRUME))
            
            CFAMNAME = shead['CFAMNAME']
            data = frame.data
            if data.ndim == 2:
                data = data[np.newaxis, :]
            if data.ndim != 3:
                raise UserWarning('Requires 2D/3D data cube')
            NINTS = data.shape[0]
            pix_scale = shead['PLTSCALE'] * 1000. # arcsec
            PIXSCALE += [pix_scale] 

            # Get centers.
            centers = np.array([shead['STARLOCX'], shead['STARLOCY']] * NINTS)

            # Get metadata.
            input_all += [data]
            centers_all += [centers]
            filenames_all += [os.path.split(frame.filename)[1] + '_INT%.0f' % (j + 1) for j in range(NINTS)]
            PAs_all += [phead['ROLL']] * NINTS

            # Get center wavelengths
            try:
                CWAVEL = self.wave_hlc[CFAMNAME]
            except:
                raise UserWarning(f'CFAM position {CFAMNAME} is not configured in corgidrp.data.PyKLIPDataset .')
            
            # Rounding error introduced here?
            wvs_all += [CWAVEL] * NINTS

            # pyklip will look for wcs.cd, so make sure that attribute exists
            wcs_obj = wcs.WCS(header=shead)

            if not hasattr(wcs_obj.wcs,'cd'):
                wcs_obj.wcs.cd = wcs_obj.wcs.pc * wcs_obj.wcs.cdelt
            
            for j in range(NINTS):
                wcs_all += [wcs_obj.deepcopy()]
                
        try:
            input_all = np.concatenate(input_all)
        except ValueError:
            raise UserWarning('Unable to concatenate images. Some science files do not have matching image shapes')
        centers_all = np.concatenate(centers_all).reshape(-1, 2)
        filenames_all = np.array(filenames_all)
        filenums_all = np.array(range(len(filenames_all)))
        PAs_all = np.array(PAs_all)
        wvs_all = np.array(wvs_all).astype(np.float32)
        wcs_all = np.array(wcs_all)
        PIXSCALE = np.unique(np.array(PIXSCALE))
        if len(PIXSCALE) != 1:
            raise UserWarning('Some science files do not have matching pixel scales')
        iwa_all = np.min(wvs_all) / 6.5 * 180. / np.pi * 3600. / PIXSCALE[0]  # pix
        owa_all = np.sum(np.array(input_all.shape[1:]) / 2.)  # pix

        # Recenter science images so that the star is at the center of the array.
        new_center = (np.array(data.shape[1:])-1)/ 2.
        new_center = new_center[::-1]
        for i, image in enumerate(input_all):
            recentered_image = pyklip.klip.align_and_scale(image, new_center=new_center, old_center=centers_all[i])
            input_all[i] = recentered_image
            centers_all[i] = new_center
        
        # Assign pyKLIP variables.
        self._input = input_all
        self._centers = centers_all
        self._filenames = filenames_all
        self._filenums = filenums_all
        self._PAs = PAs_all
        self._wvs = wvs_all
        self._wcs = wcs_all
        self._IWA = iwa_all
        self._OWA = owa_all

        # Prepare reference library
        if not psflib_dataset is None:
            psflib_data_all = []
            psflib_centers_all = []  # pix
            psflib_filenames_all = []

            for i, frame in enumerate(psflib_dataset):
                
                phead = frame.pri_hdr
                shead = frame.ext_hdr
                    
                data = frame.data
                if data.ndim == 2:
                    data = data[np.newaxis, :]
                if data.ndim != 3:
                    raise UserWarning('Requires 2D/3D data cube')
                NINTS = data.shape[0]
                pix_scale = shead['PLTSCALE'] * 1000. # arcsec
                PIXSCALE += [pix_scale] 

                # Get centers.
                centers = np.array([shead['STARLOCX'], shead['STARLOCY']] * NINTS)

                psflib_data_all += [data]
                psflib_centers_all += [centers]
                psflib_filenames_all += [os.path.split(frame.filename)[1] + '_INT%.0f' % (j + 1) for j in range(NINTS)]
            
            psflib_data_all = np.concatenate(psflib_data_all)
            if psflib_data_all.ndim != 3:
                raise UserWarning('Some reference files do not have matching image shapes')
            psflib_centers_all = np.concatenate(psflib_centers_all).reshape(-1, 2)
            psflib_filenames_all = np.array(psflib_filenames_all)
            
            # Recenter reference images.
            new_center = (np.array(data.shape[1:])-1)/ 2.
            new_center = new_center[::-1]
            for i, image in enumerate(psflib_data_all):
                recentered_image = pyklip.klip.align_and_scale(image, new_center=new_center, old_center=psflib_centers_all[i])
                psflib_data_all[i] = recentered_image
                psflib_centers_all[i] = new_center
            
            # Append science data.
            psflib_data_all = np.append(psflib_data_all, self._input, axis=0)
            psflib_centers_all = np.append(psflib_centers_all, self._centers, axis=0)
            psflib_filenames_all = np.append(psflib_filenames_all, self._filenames, axis=0)
            
            # Initialize PSF library.
            psflib = pyklip.rdi.PSFLibrary(psflib_data_all, new_center, psflib_filenames_all, compute_correlation=True, highpass=highpass)
            
            # Prepare PSF library.
            psflib.prepare_library(self)
            
            # Assign pyKLIP variables.
            self._psflib = psflib
        
        else:
            self._psflib = None
        
        pass
    
    def savedata(self,
                 filepath,
                 data,
                 klipparams=None,
                 filetype='',
                 zaxis=None,
                 more_keywords=None):
        """
        Function to save the data products that will be called internally by
        pyKLIP.
        
        Args:
            filepath (path): 
                Path of the output FITS file.
            data (3D-array): 
                KLIP-subtracted data of shape (nkl, ny, nx).
            klipparams (str, optional): 
                PyKLIP keyword arguments used for the KLIP subtraction. The default
                is None.
            filetype (str, optional): 
                Data type of the pyKLIP product. The default is ''.
            zaxis (list, optional): 
                List of KL modes used for the KLIP subtraction. The default is
                None.
            more_keywords (dict, optional): 
                Dictionary of additional header keywords to be written to the
                output FITS file. The default is None.
        """
        
        # Make FITS file.
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(data))
        
        # Write all used files to header. Ignore duplicates.
        filenames = np.unique(self.filenames)
        Nfiles = np.size(filenames)
        hdul[0].header['DRPNFILE'] = (Nfiles, 'Num raw files used in pyKLIP')
        for i, filename in enumerate(filenames):
            if i < 1000:
                hdul[0].header['FILE_{0}'.format(i)] = filename + '.fits'
            else:
                print('WARNING: Too many files to be written to header, skipping')
                break
        
        # Write PSF subtraction parameters and pyKLIP version to header.
        try:
            pyklipver = pyklip.__version__
        except:
            pyklipver = 'unknown'
        hdul[0].header['PSFSUB'] = ('pyKLIP', 'PSF Subtraction Algo')
        hdul[0].header.add_history('Reduced with pyKLIP using commit {0}'.format(pyklipver))
        hdul[0].header['CREATOR'] = 'pyKLIP-{0}'.format(pyklipver)
        hdul[0].header['pyklipv'] = (pyklipver, 'pyKLIP version that was used')
        if klipparams is not None:
            hdul[0].header['PSFPARAM'] = (klipparams, 'KLIP parameters')
            hdul[0].header.add_history('pyKLIP reduction with parameters {0}'.format(klipparams))
        
        # Write z-axis units to header if necessary.
        if zaxis is not None:
            if 'KL Mode' in filetype:
                hdul[0].header['CTYPE3'] = 'KLMODES'
                for i, klmode in enumerate(zaxis):
                    hdul[0].header['KLMODE{0}'.format(i)] = (klmode, 'KL Mode of slice {0}'.format(i))

        # Write extra keywords to header if necessary.
        if more_keywords is not None:
            for hdr_key in more_keywords:
                hdul[0].header[hdr_key] = more_keywords[hdr_key]
        
        # Update image center.
        center = self.output_centers[0]
        hdul[0].header.update({'PSFCENTX': center[0], 'PSFCENTY': center[1]})
        hdul[0].header.update({'CRPIX1': center[0] + 1, 'CRPIX2': center[1] + 1})
        hdul[0].header.add_history('Image recentered to {0}'.format(str(center)))
        
        # Write FITS file.
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=VerifyWarning) # fits save card length truncated warning
                hdul.writeto(filepath, overwrite=True)
        except TypeError:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=VerifyWarning) # fits save card length truncated warning
                hdul.writeto(filepath, clobber=True)
        hdul.close()
        
        pass
    
class NDFilterSweetSpotDataset(Image):
    """
    Class for an ND filter sweet spot dataset product.
    Typically stores an N×3 array of data:
      [OD, x_center, y_center] for each measurement.
    Args:
        data_or_filepath (str or np.array): Either the filepath to the FITS file 
            to read in OR the 2D array of ND filter sweet-spot data (N×3).
        pri_hdr (astropy.io.fits.Header): The primary header (required only if 
            raw 2D data is passed in).
        ext_hdr (astropy.io.fits.Header): The image extension header (required 
            only if raw 2D data is passed in).
        input_dataset (corgidrp.data.Dataset): The input dataset used to produce 
            this calibration file (optional). If this is a new product, you should 
            pass in the dataset so that the parent filenames can be recorded.
        err (np.array): Optional 3D error array for the data.
        dq (np.array): Optional 2D data-quality mask for the data.
        err_hdr (astropy.io.fits.Header): Optional error extension header.
    Attributes:
        od_values (np.array): Array of OD measurements (length N).
        x_values (np.array): Array of x-centroid positions (length N).
        y_values (np.array): Array of y-centroid positions (length N).
    """

    def __init__(
        self,
        data_or_filepath,
        pri_hdr=None,
        ext_hdr=None,
        input_dataset=None,
        err=None,
        dq=None,
        err_hdr=None
    ):
        # Run the standard Image constructor.
        super().__init__(
            data_or_filepath,
            pri_hdr=pri_hdr,
            ext_hdr=ext_hdr,
            err=err,
            dq=dq,
            err_hdr=err_hdr
        )

        # 1. Check data shape: expect N×3 array for the sweet-spot dataset.
        if self.data.ndim != 2 or self.data.shape[1] != 3:
            raise ValueError(
                "NDFilterSweetSpotDataset data must be a 2D array of shape (N, 3). "
                f"Received shape {self.data.shape}."
            )

        # 2. Parse the columns into convenient attributes.
        #    Column 0: OD, Column 1: x_center, Column 2: y_center.
        self.od_values = self.data[:, 0]
        self.x_values = self.data[:, 1]
        self.y_values = self.data[:, 2]

        # 3. If creating a new product (i.e. ext_hdr was passed in), record metadata.
        if ext_hdr is not None:
            if input_dataset is not None:
                self._record_parent_filenames(input_dataset)
                self.filename = re.sub('_l[0-9].', '_ndf_cal', input_dataset[-1].filename)
            # if no input_dataset is given, do we want to set the filename manually using 
            # header values?

            self.ext_hdr['DATATYPE'] = 'NDFilterSweetSpotDataset'
            self.ext_hdr['HISTORY'] = (
                f"NDFilterSweetSpotDataset created from {self.ext_hdr.get('DRPNFILE','?')} frames"
            )

            # Enforce data level = CAL
            self.ext_hdr['DATALVL']    = 'CAL'

        # 4. If reading from a file, verify that the header indicates the correct DATATYPE.
        if 'DATATYPE' not in self.ext_hdr or self.ext_hdr['DATATYPE'] != 'NDFilterSweetSpotDataset':
            raise ValueError("File that was loaded is not labeled as an NDFilterSweetSpotDataset file.")

    def interpolate_od(self, x, y, method="nearest"):
        """
        Interpolates the data to get the OD at the requested x/y location

        Args:
            x (float): x detector pixel location
            y (float): y detector pixel location
            method (str): only "nearest" supported currently
        
        Returns:
            float: the OD at the requested point
        """
        interpolator = LinearNDInterpolator(np.array([self.x_values, self.y_values]).T, self.od_values)

        return interpolator(x, y)

def format_ftimeutc(ftime_str):
    """
    Round the input FTIMEUTC time to the nearest 0.01 sec and reformat as:
    yyyymmddthhmmsss.

    Args:
        ftime_str (str): Time string in ISO format, e.g. "2025-04-15T03:05:10.21".

    Returns:
        formatted_time (str): Reformatted time string in yyyymmddthhmmsss format.
    """
    # Parse the input using fromisoformat, which can handle timezone offsets.
    try:
        ftime = datetime.fromisoformat(ftime_str)
    except ValueError as e:
        raise ValueError(f"Could not parse FTIMEUTC: {ftime_str}") from e

    # If the datetime is timezone aware, convert to UTC and remove tzinfo.
    if ftime.tzinfo is not None:
        ftime = ftime.astimezone(timezone.utc).replace(tzinfo=None)
    
    # Round to nearest 0.01 seconds (10,000 microseconds)
    rounding_interval = 10000
    rounded_microsec = int((ftime.microsecond + rounding_interval / 2) // rounding_interval * rounding_interval)
    
    # Handle rollover: if rounding reaches or exceeds 1,000,000 microseconds increment the second
    if rounded_microsec >= 1000000:
        ftime = ftime.replace(microsecond=0) + timedelta(seconds=1)
    else:
        ftime = ftime.replace(microsecond=rounded_microsec)
    
    # Format seconds with exactly 3 digits total
    # We want the seconds part to be exactly 3 digits
    # Format: sss where sss = seconds (00-59) + first digit of hundredths (0-9)
    # Example: 10.21 becomes 102, 5.05 becomes 505, 59.99 becomes 599
    sec_int = ftime.second
    hundredths = int(ftime.microsecond / 10000)  # (0-99)
    
    # Take only the first digit of hundredths (0-9)
    first_hundredth = hundredths // 10
    
    # Combine seconds and first hundredth: ss * 10 + h
    # This gives us a 3-digit number where the first 2 digits are seconds and last digit is tenths
    combined_seconds = sec_int * 10 + first_hundredth
    
    # Format as yyyymmddthhmmsss (17 characters total)
    # Use :03d to ensure exactly 3 digits with leading zeros if needed
    formatted_time = ftime.strftime("%Y%m%dt%H%M") + f"{combined_seconds:03d}"
    return formatted_time


datatypes = { "Image" : Image,
              "Dark" : Dark,
              "NonLinearityCalibration" : NonLinearityCalibration,
              "KGain" : KGain,
              "BadPixelMap" : BadPixelMap,
              "DetectorNoiseMaps": DetectorNoiseMaps,
              "FlatField" : FlatField,
              "DetectorParams" : DetectorParams,
              "AstrometricCalibration" : AstrometricCalibration,
              "TrapCalibration" : TrapCalibration,
              "FluxcalFactor" : FluxcalFactor,
              "FpamFsamCal" : FpamFsamCal,
              "CoreThroughputCalibration": CoreThroughputCalibration,
              "CoreThroughputMap": CoreThroughputMap,
              "PSFCentroidCalibration": SpectroscopyCentroidPSF,
              "NDFilterSweetSpotDataset": NDFilterSweetSpotDataset,
              "SpectroscopyCentroidPSF": SpectroscopyCentroidPSF,
              "DispersionModel": DispersionModel
              }

def autoload(filepath):
    """
    Loads the supplied FITS file filepath using the appropriate data class

    Should be used sparingly to avoid accidentally loading in data of the wrong type

    Args:
        filepath (str): path to FITS file

    Returns:
        corgidrp.data.* : an instance of one of the data classes specified here
    """

    with fits.open(filepath) as hdulist:
        # check the exthdr for datatype
        if 'DATATYPE' in hdulist[1].header:
            dtype = hdulist[1].header['DATATYPE']
        else:
            # datatype not specified. Check if it's 2D
            if len(hdulist[1].data.shape) == 2:
                # a standard image (possibly a science frame)
                dtype = "Image"
            else:
                errmsg = "Could not determine datatype for {0}. Data shape of {1} is not 2-D"
                raise ValueError(errmsg.format(filepath, dtype))

    # if we got here, we have a datatype
    data_class = datatypes[dtype]

    # use the class constructor to load in the data
    frame = data_class(filepath)

    return frame

def unpackbits_64uint(arr, axis):
    """
    Unpacking bits from a 64-bit unsigned integer array

    Args:
        arr (np.ndarray): the array to unpack
        axis (int): axis to unpack

    Returns:
        np.ndarray of bits
    """
    arr = arr.astype('>u8')
    n = arr.view('u1')
    return np.unpackbits(n, axis=axis, bitorder='big')

def packbits_64uint(arr, axis):
    """
    Packing bits into a 64-bit unsigned integer array

    Args:
        arr (np.ndarray): the array to pack 
        axis (int): axis to pack

    Returns:
        np.ndarray of 64-bit unsigned integers
    """
    return np.packbits(arr, axis=axis, bitorder='big').view('>u8')

def get_flag_to_bit_map():
    """
    Returns a dictionary mapping flag names to bit positions.
    
    Returns:
        dict: A dictionary with flag names as keys and bit positions (int) as values.
    """
    return {
        "bad_pixel_unspecified": 0,
        "data_replaced_by_filled_value": 1,
        "bad_pixel": 2,
        "hot_pixel": 3,
        "not_used": 4,
        "full_well_saturated_pixel": 5,
        "non_linear_pixel": 6,
        "pixel_affected_by_cosmic_ray": 7,
        "TBD": 8,
    }

def get_flag_to_value_map():
    """
    Returns a dictionary mapping flag names to their decimal flag values. Example usage is as follows:
    
    FLAG_TO_VALUE_MAP = get_flag_to_value_map()
    flag_value = FLAG_TO_VALUE_MAP["TBD"]  # Gives the decimal value corresponding to "TBD"

    Returns:
        dict: A dictionary with flag names as keys and decimal values (int) as values.
    """
    return {name: (1 << bit) for name, bit in get_flag_to_bit_map().items()}

def get_value_to_flag_map():
    """
    Returns a dictionary mapping flag decimal values to flag names. Example usage is as follows:
    
    FLAG_TO_BIT_MAP = get_flag_to_bit_map()
    bit_position = FLAG_TO_BIT_MAP["TBD"]  # Gives which index it should be in with the key. 
    
    Expected position of bit_position of the unpacked array = 63 - bit_position
    
    Returns:
        dict: A dictionary with decimal values (int) as keys and flag names as values.
    """
    return {value: name for name, value in get_flag_to_value_map().items()}

def get_bit_to_flag_map():
    """
    Returns a dictionary mapping bit positions to flag names. Example usage is as follows:

    BIT_TO_FLAG_MAP = get_bit_to_flag_map()
    flag_name_from_bit = BIT_TO_FLAG_MAP[8]  # Expected: "TBD"
    
    Returns:
        dict: A dictionary with bit positions (int) as keys and flag names as values.
    """
    return {bit: name for name, bit in get_flag_to_bit_map().items()}
