import os
import numpy as np
import astropy.io.fits as fits

class Dataset():
    """
    A sequence of data of the same kind. Can be indexed and looped over

    Args:
        frames_or_filepaths (list): list of either filepaths or data objects (e.g., Image class)
    """
    def __init__(self, frames_or_filepaths):
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
        
        # turn lists into np.array for indiexing behavior
        if isinstance(self.frames, list):
            self.frames = np.array(self.frames) # list of objects

        # create 3-D cube of all the data
        self.all_data = np.array([frame.data for frame in self.frames])

        # do a clever thing to point all the individual frames to the data in this cube
        # this way editing a single frame will also edit the entire datacube
        for i, frame in enumerate(self.frames):
            frame.data = self.all_data[i]
        
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

    def save(self, filedir, filenames=None):
        """
        Save each file of data in this dataset into directory

        Args:
            filedir (str): directory to save the files
            filenames (list): a list of output filenames for each file

        """
        # if filenames are not passed, use the default ones
        if filenames is None:
            filenames = []
            for frame in self.frames:
                filename = frame.filename
                filenames.append(frame.filename)

        for filename, frame in zip(filenames, self.frames):
            frame.save(filename=filename, filedir=filedir)

    def update_after_processing_step(self, history_entry, new_all_data=None):
        """
        Updates the dataset after going through a processing step

        Args:
            history_entry (str): a description of what processing was done. Mention reference files used.
            new_all_data (np.array): (optional) Array of new data. Needs to be the same shape as `all_data`
        Returns:
            corgidrp.data.Dataset: updated dataset. Maybe the same as self! (implementation still being finalized)
        """
        # update data if necessary
        if new_all_data is not None:
            if new_all_data.shape != self.all_data.shape:
                raise ValueError("The shape of new_all_data is {0}, whereas we are expecting {1}".format(new_all_data.shape, self.all_data.shape))
            self.all_data[:] = new_all_data # specific operation overwrites the existing data rather than changing pointers

        # update history
        for img in self.frames:
            img.ext_hdr['HISTORY'] = history_entry

        return self # not sure if we should be returning new copies of the dataset, so function signature is such


class Image():
    """
    Base class for 2-D image data. Data can be created by passing in the data/header explicitly, or
    by passing in a filepath to load a FITS file from disk

    Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR the 2D image data
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw 2D data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw 2D data is passed in)
    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None):
        if isinstance(data_or_filepath, str):
            # a filepath is passed in
            with fits.open(data_or_filepath) as hdulist:
                self.pri_hdr = hdulist[0].header
                # image data is in FITS extension
                self.ext_hdr = hdulist[1].header
                self.data = hdulist[1].data
            
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
            if pri_hdr is None or ext_hdr is None:
                raise ValueError("Missing primary and/or extension headers, because you passed in raw data")
            self.pri_hdr = pri_hdr
            self.ext_hdr = ext_hdr
            self.data = data_or_filepath
            self.filedir = "."
            self.filename = ""

        # can do fancier things here if needed or storing more meta data

    # create this field dynamically 
    @property
    def filepath(self):
        return os.path.join(self.filedir, self.filename)
    
    
    def save(self, filename=None, filedir=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
            filedir (str): filedir to save to. Use self.filedir if not specified
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
        hdulist.writeto(self.filepath, overwrite=True)
        hdulist.close()

    def _record_parent_filenames(self, input_dataset):
        """
        Record what input dataset was used to create this Image.
        This assumes many Images were used to make this single Image.
        Record is stored in the ext header. 

        Args:
            input_dataset (corgidrp.data.Dataset): the input dataset that were combined together to make this image
        """
        self.ext_hdr['DRPNFILE'] = len(input_dataset) # corgidrp specific keyword
        for i, img in enumerate(input_dataset):
            self.ext_hdr['FILE{0}'.format(i)] = img.filename


class Dark(Image):
    """
    Dark calibration frame for a given exposure time. 

     Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR the 2D image data
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw 2D data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw 2D data is passed in)
        input_dataset (corgidrp.data.Dataset): the Image files combined together to make this dark file (required only if raw 2D data is passed in)
    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, input_dataset=None):
        # run the image class contructor
        super().__init__(data_or_filepath, pri_hdr=pri_hdr, ext_hdr=ext_hdr)
        # additional bookkeeping for Dark

        # if this is a new dark, we need to bookkeep it in the header
        # b/c of logic in the super.__init__, we just need to check this to see if it is a new dark
        if ext_hdr is not None:
            if input_dataset is None:
                # error check. this is required in this case
                raise ValueError("This appears to be a new dark. The dataset of input files needs to be passed in to the input_dataset keyword to record history of this dark.")
            self.ext_hdr['DATATYPE'] = 'Dark' # corgidrp specific keyword for saving to disk

            # log all the data that went into making this dark
            self._record_parent_filenames(input_dataset)

            # add to history
            self.ext_hdr['HISTORY'] = "Dark with exptime = {0} s created from {1} frames".format(self.ext_hdr['EXPTIME'], self.ext_hdr['DRPNFILE'])

            # give it a default filename using the first input file as the base
            # strip off everything starting at .fits
            orig_input_filename = input_dataset[0].filename.split(".fits")[0]
            self.filename = "{0}_dark.fits".format(orig_input_filename)


        # double check that this is actually a dark file that got read in
        # since if only a filepath was passed in, any file could have been read in
        if 'DATATYPE' not in self.ext_hdr or self.ext_hdr['DATATYPE'] != 'Dark':
            raise ValueError("File that was loaded was not a Dark file.")