import os
import numpy as np
import numpy.ma as ma
import astropy.io.fits as fits
import astropy.time as time
import pandas as pd

import corgidrp

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
          input_error (np.array): 2-d or 3-d error layer
          err_name (str): name of the uncertainty layer
        """
        if input_error.ndim == 3:
            for i,frame in enumerate(self.frames):
                frame.add_error_term(input_error[i], err_name)

        elif input_error.ndim ==2:
            for frame in self.frames:
                frame.add_error_term(input_error, err_name)

        else:
            raise ValueError("input_error is not either a 2D or 3D array.")

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
                    if np.shape(self.data) != np.shape(err)[-self.data.ndim:]:
                        raise ValueError("The shape of err is {0} while we are expecting shape {1}".format(err.shape[-self.data.ndim:], self.data.shape))
                    #we want to have a 3 dim error array
                    if err.ndim > 2:
                        self.err = err
                    else:
                        self.err = err.reshape((1,)+err.shape)
                elif "ERR" in self.hdu_names:
                    err_hdu = hdulist.pop("ERR")
                    self.err = err_hdu.data
                    self.err_hdr = err_hdu.header
                    if self.err.ndim == 2:
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
            if pri_hdr is None or ext_hdr is None:
                raise ValueError("Missing primary and/or extension headers, because you passed in raw data")
            self.pri_hdr = pri_hdr
            self.ext_hdr = ext_hdr
            self.data = data_or_filepath
            self.filedir = "."
            self.filename = ""

            # self.hdu_names = [hdu.name for hdu in self.hdu_list]

            if err is not None:
                if np.shape(self.data) != np.shape(err)[-self.data.ndim:]:
                    raise ValueError("The shape of err is {0} while we are expecting shape {1}".format(err.shape[-self.data.ndim:], self.data.shape))
                #we want to have a 3 dim error array
                if err.ndim > 2:
                    self.err = err
                else:
                    self.err = err.reshape((1,)+err.shape)
            else:
                self.err = np.zeros((1,)+self.data.shape)

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

            
            
            #A list of extensions
            

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
            new_data = np.copy(self.data)
            new_err = np.copy(self.err)
            new_dq = np.copy(self.dq)
            new_hdulist = self.hdu_list.copy()
        else:
            new_data = self.data # this is just pointer referencing
            new_err = self.err
            new_dq = self.dq
            new_hdulist = self.hdu_list
        new_img = Image(new_data, pri_hdr=self.pri_hdr.copy(), ext_hdr=self.ext_hdr.copy(), err = new_err, dq = new_dq, 
                        input_hdulist = new_hdulist, err_hdr = self.err_hdr.copy(), dq_hdr = self.dq_hdr.copy())

        # annoying, but we got to manually update some parameters. Need to keep track of which ones to update
        new_img.filename = self.filename
        new_img.filedir = self.filedir

        # update DRP version tracking
        self.ext_hdr['DRPVERSN'] =  corgidrp.__version__
        self.ext_hdr['DRPCTIME'] =  time.Time.now().isot

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
        self.err[0,:,:] = np.sqrt(self.err[0,:,:]**2 + input_error**2)
        self.err_hdr["Layer_1"] = "combined_error"

        if corgidrp.track_individual_errors:
            #append new error as layer on 3D cube
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
        input_dataset (corgidrp.data.Dataset): the Image files combined together to make this noise map (required only if raw 2D data is passed in and if raw data filenames not already archived in ext_hdr)
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
            self.ext_hdr['BUNIT'] = 'detected electrons'

            # log all the data that went into making this calibration file
            if 'DRPNFILE' not in ext_hdr.keys() and input_dataset is not None:
                self._record_parent_filenames(input_dataset)

            # add to history
            self.ext_hdr['HISTORY'] = "Dark with exptime = {0} s and commanded EM gain = {1} created from {2} frames".format(self.ext_hdr['EXPTIME'], self.ext_hdr['CMDGAIN'], self.ext_hdr['DRPNFILE'])

            # give it a default filename using the first input file as the base
            # strip off everything starting at .fits
            if input_dataset is not None:
                orig_input_filename = input_dataset[0].filename.split(".fits")[0]
                self.filename = "{0}_dark.fits".format(orig_input_filename)

        if err_hdr is not None:
            self.err_hdr['BUNIT'] = 'detected electrons'

        # double check that this is actually a dark file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a Dark file.")
        if self.ext_hdr['DATATYPE'] != 'Dark':
            raise ValueError("File that was loaded was not a Dark file.")

    def copy(self, copy_data = True):
        """
        Make a copy of this Dark file, including data and headers.
        Data copying can be turned off if you only want to modify the headers
        Headers should always be copied as we should modify them any time we make new edits to the data

        Args:
            copy_data (bool): (optional) whether the data should be copied. Default is True

        Returns:
            new_nm (corgidrp.data.NoiseMap): a copy of this Dark
        """
        if copy_data:
            new_data = np.copy(self.data)
            new_err = np.copy(self.err)
            new_dq = np.copy(self.dq)
        else:
            new_data = self.data # this is just pointer referencing
            new_err = self.err
            new_dq = self.dq
        new_dark = Dark(new_data, pri_hdr=self.pri_hdr.copy(), ext_hdr=self.ext_hdr.copy(), err = new_err, dq = new_dq, err_hdr = self.err_hdr.copy())

        # annoying, but we got to manually update some parameters. Need to keep track of which ones to update
        new_dark.filename = self.filename
        new_dark.filedir = self.filedir

        # update DRP version tracking
        self.ext_hdr['DRPVERSN'] =  corgidrp.version
        self.ext_hdr['DRPCTIME'] =  time.Time.now().isot

        return new_dark

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

            # log all the data that went into making this flat
            self._record_parent_filenames(input_dataset)

            # add to history
            self.ext_hdr['HISTORY'] = "Flat with exptime = {0} s created from {1} frames".format(self.ext_hdr['EXPTIME'], self.ext_hdr['DRPNFILE'])

            # give it a default filename using the first input file as the base
            orig_input_filename = input_dataset[0].filename.split(".fits")[0]
            self.filename = "{0}_flatfield.fits".format(orig_input_filename)


        # double check that this is actually a masterflat file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a FlatField file.")
        if self.ext_hdr['DATATYPE'] != 'FlatField':
            raise ValueError("File that was loaded was not a FlatField file.")


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

            # give it a default filename using the first input file as the base
            # strip off everything starting at .fits
            orig_input_filename = input_dataset[0].filename.split(".fits")[0]
            self.filename = "{0}_NonLinearityCalibration.fits".format(orig_input_filename)


        # double check that this is actually a NonLinearityCalibration file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a NonLinearityCalibration file.")
        if self.ext_hdr['DATATYPE'] != 'NonLinearityCalibration':
            raise ValueError("File that was loaded was not a NonLinearityCalibration file.")

class KGain(Image):
    """
    Class for KGain calibration file. Until further insights it is just one float value.

    Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR the calibration data. See above for the required format.
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

        # File format checks
        if self.data.shape != (1,1):
            raise ValueError('The KGain calibration data should be just one float value')

        self._kgain = self.data[0,0] 
        self._kgain_error = self.err[0,0]
        
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
                # give it a default filename using the first input file as the base
                # strip off everything starting at .fits
                orig_input_filename = input_dataset[0].filename.split(".fits")[0]
                self.filename = "{0}_kgain.fits".format(orig_input_filename)

            self.ext_hdr['DATATYPE'] = 'KGain' # corgidrp specific keyword for saving to disk
            self.ext_hdr['BUNIT'] = 'detected electrons/DN'
            # add to history
            self.ext_hdr['HISTORY'] = "KGain Calibration file created"

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
    
    def copy(self, copy_data = True):
        """
        Make a copy of this KGain file. including data and headers.
        Data copying can be turned off if you only want to modify the headers
        Headers should always be copied as we should modify them any time we make new edits to the data

        Args:
            copy_data (bool): (optional) whether the data should be copied. Default is True

        Returns:
            corgidrp.data.KGain: a copy of this KGain
        """
        if copy_data:
            new_data = np.copy(self.data)
            new_ptc = np.copy(self.ptc)
            new_err = np.copy(self.err)
        else:
            new_data = self.data # this is just pointer referencing
            new_ptc = self.ptc
            new_err = np.copy(self.err)
        
        new_kg = KGain(new_data, err = new_err, ptc = new_ptc, pri_hdr=self.pri_hdr.copy(), ext_hdr=self.ext_hdr.copy(), err_hdr = self.err_hdr.copy(), ptc_hdr = self.ptc_hdr.copy())
        
        # annoying, but we got to manually update some parameters. Need to keep track of which ones to update
        new_kg.filename = self.filename
        new_kg.filedir = self.filedir

        # update DRP version tracking
        self.ext_hdr['DRPVERSN'] =  corgidrp.__version__
        self.ext_hdr['DRPCTIME'] =  time.Time.now().isot
        
        return new_kg


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
            self.ext_hdr['DATATYPE'] = 'BadPixelMap' # corgidrp specific keyword for saving to disk

            # log all the data that went into making this bad pixel map
            self._record_parent_filenames(input_dataset)

            # add to history
            self.ext_hdr['HISTORY'] = "Bad Pixel map created"

            # give it a default filename using the first input file as the base
            # strip off everything starting at .fits
            orig_input_filename = input_dataset[0].filename.split(".fits")[0]
            self.filename = "{0}_bad_pixel_map.fits".format(orig_input_filename)


        # double check that this is actually a bad pixel map that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr:
            raise ValueError("File that was loaded was not a BadPixelMap file.")
        if self.ext_hdr['DATATYPE'] != 'BadPixelMap':
            raise ValueError("File that was loaded was not a BadPixelMap file.")

    def copy(self, copy_data = True):
        """
        Make a copy of this BadPixelMap file. including data and headers.
        Data copying can be turned off if you only want to modify the headers
        Headers should always be copied as we should modify them any time we make new edits to the data

        Args:
            copy_data (bool): (optional) whether the data should be copied. Default is True

        Returns:
            corgidrp.data.BadPixelMap: a copy of this BadPixelMap
        """
        if copy_data:
            new_data = np.copy(self.data)
        else:
            new_data = self.data # this is just pointer referencing
        new_bp = BadPixelMap(new_data, pri_hdr=self.pri_hdr.copy(), ext_hdr=self.ext_hdr.copy())

        # we got to manually update some parameters. Need to keep track of which ones to update
        new_bp.filename = self.filename
        new_bp.filedir = self.filedir

        # update DRP version tracking
        self.ext_hdr['DRPVERSN'] =  corgidrp.__version__
        self.ext_hdr['DRPCTIME'] =  time.Time.now().isot

        return new_bp

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
            self.ext_hdr['BUNIT'] = 'detected electrons'
            # bias offset
            self.ext_hdr['B_0_UNIT'] = 'DN' # err unit is also in DN

            # log all the data that went into making this calibration file
            if 'DRPNFILE' not in ext_hdr.keys():
                self._record_parent_filenames(input_dataset)
            # add to history
            self.ext_hdr['HISTORY'] = "DetectorNoiseMaps calibration file created"

            # give it a default filename
            orig_input_filename = self.ext_hdr['FILE0'].split(".fits")[0]
            self.filename = "{0}_DetectorNoiseMaps.fits".format(orig_input_filename)

        if err_hdr is not None:
            self.err_hdr['BUNIT'] = 'detected electrons'

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


    def copy(self, copy_data = True):
        """
        Make a copy of this DetectorNoiseMaps file, including data and headers.
        Data copying can be turned off if you only want to modify the headers
        Headers should always be copied as we should modify them any time we make new edits to the data

        Args:
            copy_data (bool): (optional) whether the data should be copied. Default is True

        Returns:
            new_nm (corgidrp.data.DetectorNoiseMaps): a copy of this DetectorNoiseMaps
        """
        if copy_data:
            new_data = np.copy(self.data)
            new_err = np.copy(self.err)
            new_dq = np.copy(self.dq)
        else:
            new_data = self.data # this is just pointer referencing
            new_err = self.err
            new_dq = self.dq
        new_nm = DetectorNoiseMaps(new_data, pri_hdr=self.pri_hdr.copy(), ext_hdr=self.ext_hdr.copy(), err = new_err, dq = new_dq, err_hdr = self.err_hdr.copy())

        # annoying, but we got to manually update some parameters. Need to keep track of which ones to update
        new_nm.filename = self.filename
        new_nm.filedir = self.filedir

        # update DRP version tracking
        self.ext_hdr['DRPVERSN'] =  corgidrp.version
        self.ext_hdr['DRPCTIME'] =  time.Time.now().isot

        return new_nm

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
    """
     # default detector params
    default_values = {
        'kgain' : 8.7,
        'fwc_pp' : 90000.,
        'fwc_em' : 100000.,
        'rowreadtime' : 223.5e-6, # seconds
        # number of EM gain register stages
        'Nem': 604,
        # slice of rows that are used for telemetry
        'telem_rows_start': -1,
        'telem_rows_end': None, #goes to the end, in other words
        # pixel full well (e-)
        'fwc': 90000,
        # serial full well (e-) in EM gain register in EXCAM EMCCD
        'fwc_em': 100000,
        # cosmic ray hit rate (hits/m**2/sec)
        'X': 5.0e+04,
        # pixel area (m**2/pixel)
        'a': 1.69e-10,
        # Maximum allowable EM gain
        'gmax': 8000.0,
        # tolerance in exposure time calculator
        'delta_constr': 1.0e-4,
        # Overhead time, in seconds, for each collected frame.  Used to compute
        # total wall-clock time for data collection
        'overhead': 3,
        # Maximum allowed electrons/pixel/frame for photon counting
        'pc_ecount_max': 0.1,
        # number of read noise standard deviations at which to set the
        # photon-counting threshold
        'T_factor': 5,
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
            prihdr = fits.Header()
            exthdr = fits.Header()
            exthdr['SCTSRT'] = date_valid.isot # use this for validity date
            exthdr['DRPVERSN'] =  corgidrp.__version__
            exthdr['DRPCTIME'] =  time.Time.now().isot

            # fill caldb required keywords with dummy data
            prihdr['OBSNUM'] = 0
            exthdr["EXPTIME"] = 0
            exthdr['OPMODE'] = ""
            exthdr['CMDGAIN'] = 1.0
            exthdr['EXCAMT'] = 40.0

            # write default values to headers
            for key in self.default_values:
                if len(key) > 8:
                    # to avoid VerifyWarning from fits
                    exthdr['HIERARCH ' + key] = self.default_values[key]
                else:
                    exthdr[key] = self.default_values[key]
            # overwrite default values
            for key in data_or_filepath:
                # to avoid VerifyWarning from fits
                if len(key) > 8:
                    exthdr['HIERARCH ' + key] = data_or_filepath[key]
                else:
                    exthdr[key] = data_or_filepath[key]

            self.pri_hdr = prihdr
            self.ext_hdr = exthdr
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
            if len(key) > 8:
                # to avoid VerifyWarning from fits
                self.params[key] = self.ext_hdr['HIERARCH ' + key]
            else:
                self.params[key] = self.ext_hdr[key]


        # if this is a new DetectorParams file, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if it is a new DetectorParams file
        if isinstance(data_or_filepath, dict):
            self.ext_hdr['DATATYPE'] = 'DetectorParams' # corgidrp specific keyword for saving to disk

            # add to history
            self.ext_hdr['HISTORY'] = "Detector Params file created"

            # use the start date for the filename by default
            self.filedir = "."
            self.filename = "DetectorParams_{0}.fits".format(self.ext_hdr['SCTSRT'])

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
        data_or_filepath (str or np.array); either the filepath to the FITS file to read in OR the 1D array of calibration measurements
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw 2D data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw 2D data is passed in)
        
    Attrs:
        boresight (np.array): the [(RA, Dec)] of the center pixel in ([deg], [deg])
        platescale (float): the platescale value in [mas/pixel]
        northangle (float): the north angle value in [deg]

    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, input_dataset=None):
        # run the image class constructor
        super().__init__(data_or_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr)

        # File format checks
        if self.data.shape != (4,):
            raise ValueError("The AstrometricCalibration data should be a 1D array of four values")
        else:
            self.boresight = self.data[:2]
            self.platescale = self.data[2]
            self.northangle = self.data[3]
            
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
            
            # give a default filename
            self.filename = "AstrometricCalibration.fits"

        # check that this is actually an AstrometricCalibration file that was read in
        if 'DATATYPE' not in self.ext_hdr or self.ext_hdr['DATATYPE'] != 'AstrometricCalibration':
            raise ValueError("File that was loaded was not an AstrometricCalibration file.")    
        
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
            orig_input_filename = input_dataset[0].filename.split(".fits")[0]
            self.filename = "{0}_trapcal.fits".format(orig_input_filename)


        # double check that this is actually a dark file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr or self.ext_hdr['DATATYPE'] != 'TrapCalibration':
            raise ValueError("File that was loaded was not a TrapCalibration file.")
        

class NDFilterCalibration(Image):
    """
    Class for an ND filter calibration product. Typically stores N×3 data:
    [OD, x_center, y_center] for each measurement.

    Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file 
            to read in OR the 2D array of ND filter calibration data (N×3).
        pri_hdr (astropy.io.fits.Header): the primary header (required only if 
            raw 2D data is passed in).
        ext_hdr (astropy.io.fits.Header): the image extension header (required 
            only if raw 2D data is passed in).
        input_dataset (corgidrp.data.Dataset): the input dataset used to produce 
            this calibration file (optional). If this is a new ND filter calibration, 
            you should pass in the dataset so that the parent filenames can be 
            recorded in the headers.
        err (np.array): optional 3D error array for the data
        dq (np.array): optional 2D data-quality mask for the data
        err_hdr (astropy.io.fits.Header): optional error extension header

    Attributes:
        od_values (np.array): the array of OD measurements (length N).
        x_values (np.array): the array of x-centroid positions (length N).
        y_values (np.array): the array of y-centroid positions (length N).
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
        # Run the standard Image constructor
        super().__init__(
            data_or_filepath,
            pri_hdr=pri_hdr,
            ext_hdr=ext_hdr,
            err=err,
            dq=dq,
            err_hdr=err_hdr
        )

        # 1. Check data shape: expect Nx3 for ND filter calibration
        if self.data.ndim != 2 or self.data.shape[1] != 3:
            raise ValueError(
                "NDFilterCalibration data must be a 2D array of shape (N, 3). "
                f"Received shape {self.data.shape}."
            )

        # 2. Parse columns to attributes for convenience
        #    Column 0: OD, Column 1: x_center, Column 2: y_center
        self.od_values = self.data[:, 0]
        self.x_values = self.data[:, 1]
        self.y_values = self.data[:, 2]

        # 3. If this is a new NDFilterCalibration (i.e., ext_hdr was passed in),
        #    record the metadata and set DATATYPE.
        #    If reading from a file on disk, ext_hdr might already have DATATYPE set.
        if ext_hdr is not None:
            # If the user provided an input dataset, record the parent filenames
            if input_dataset is not None:
                self._record_parent_filenames(input_dataset)

            # Mark it as an NDFilterCalibration product
            self.ext_hdr['DATATYPE'] = 'NDFilterCalibration'

            # TODO: store some summary metadata in the header?

            # Add history
            self.ext_hdr['HISTORY'] = (
                f"NDFilterCalibration created from {self.ext_hdr.get('DRPNFILE','?')} frames"
            )

            # Give it a default filename if you like
            if input_dataset is not None and len(input_dataset) > 0:
                base_name = input_dataset[0].filename.split(".fits")[0]
                self.filename = f"{base_name}_ndfcal.fits"
            else:
                self.filename = "NDFilterCalibration.fits"

        # 4. If reading from a file, verify it’s actually an NDFilterCalibration
        if 'DATATYPE' not in self.ext_hdr or self.ext_hdr['DATATYPE'] != 'NDFilterCalibration':
            raise ValueError("File that was loaded is not labeled as an NDFilterCalibration file.")


datatypes = { "Image" : Image,
             "Dark" : Dark,
              "NonLinearityCalibration" : NonLinearityCalibration,
              "KGain" : KGain,
              "BadPixelMap" : BadPixelMap,
              "DetectorNoiseMaps": DetectorNoiseMaps,
              "FlatField" : FlatField,
              "DetectorParams" : DetectorParams,
              "AstrometricCalibration" : AstrometricCalibration,
              "TrapCalibration": TrapCalibration,
            "NDFilterCalibration": NDFilterCalibration}

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
