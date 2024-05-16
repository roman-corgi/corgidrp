"""
Calibration tracking system. Modified from kpicdrp caldb implmentation (Copyright (c) 2024, KPIC Team)
"""
import os
import numpy as np
import pandas as pd
import corgidrp
import corgidrp.data as data
import astropy.time as time


column_names = [
    "Filepath",
    "Type",
    "MJD",
    "EXPTIME",
    "Files Used",
    "Date Created",
    "DRPVERSN",
    "OBSID",
    "NAXIS1",
    "NAXIS2",
    "OPMODE",
    "CMDGAIN",
    "EXCAMT",
]

labels = {data.Dark: "Dark", data.NonLinearityCalibration: "NonLinearityCalibration", data.BadPixelMap: "BadPixelMap", }


class CalDB:
    """
    Database for tracking calibration files saved to disk. Modified from the kpicdrp version

    Note that database is not parallelism-safe, but should be ok in most cases.
    (Jason: look at using posix_ipc to guarantee thread safety if we really need it)

    Args:
        filepath (str): [optional] filepath to a CSV file with an existing database

    Fields:
        columns (list): column names of dataframe
        filepath(str): full filepath to data
    """

    def __init__(self, filepath=""):
        """
        Args:
            filepath (str): [optional] filepath to a CSV file with an existing database
        """

        # If filepath is not passed in, use the default (majority of case)
        if len(filepath) == 0:
            self.filepath = corgidrp.caldb_filepath
        else:
            # possibly edge case where we want to use specialized caldb
            self.filepath = filepath

        # if database does't exist, create a blank one
        if not os.path.exists(self.filepath):
            # new database
            self.columns = column_names
            self._db = pd.DataFrame(columns=self.columns)
            self.save()
        else:
            # already a database exists
            self.load()
            self.columns = list(self._db.columns.values)

    def load(self):
        """
        Load/update db from filepath
        """
        self._db = pd.read_csv(self.filepath)

    def save(self):
        """
        Save file without numbered index to disk with user specified filepath as a CSV file
        """
        self._db.to_csv(self.filepath, index=False)

    def _get_values_from_entry(self, entry, is_calib=True):
        """
        Extract the properties from this data entry to ingest them into the database

        Args:
            entry (corgidrp.data.Image subclass): calibration frame to add to the database
            is_calib (bool): is a calibration frame. if Not, it won't look up filetype.
                             Only used in get_calib() to grab metadata for science frames

        Returns:
            tuple:
                row (list):
                    List of data entry properties
                row_dict (dict):
                    Dictionary of data entry properties keyed by column names

        """
        filepath = os.path.abspath(entry.filepath)
        if is_calib:
            datatype = labels[entry.__class__]  # get the database str representation
        else:
            datatype = "Sci"
        mjd = time.Time(entry.ext_hdr["SCTSRT"]).mjd
        exptime = entry.ext_hdr["EXPTIME"]

        # check if this exists. will be a keyword written by corgidrp
        if "DRPNFILE" in entry.ext_hdr:
            files_used = entry.ext_hdr["DRPNFILE"]
        else:
            files_used = 0

        date_created = time.Time(entry.ext_hdr["DRPCTIME"]).mjd
        drp_version = entry.ext_hdr["DRPVERSN"]

        obsid = entry.pri_hdr["OBSID"]

        # this only works for 2D images. may need to adapt for non-2D calibration frames
        naxis1 = entry.data.shape[-1]
        naxis2 = entry.data.shape[-2]

        row = [
            filepath,
            datatype,
            mjd,
            exptime,
            files_used,
            date_created,
            drp_version,
            obsid,
            naxis1,
            naxis2,
        ]

        # rest are ext_hdr keys we can copy
        start_index = len(row)
        for i in range(start_index, len(self.columns)):
            row.append(entry.ext_hdr[self.columns[i]])  # add value staright from header

        row_dict = {}
        for key, val in zip(self.columns, row):
            row_dict[key] = val

        return row, row_dict

    def create_entry(self, entry, to_disk=True):
        """
        Add a new entry to or update an existing one in the database. Note that function by default will load and save db to disk

        Args:
            entry (corgidrp.data.Image subclass): calibration frame to add to the database
            to_disk (bool): True by default, will update DB from disk before adding entry and saving it back to disk
        """
        new_row, row_dict = self._get_values_from_entry(entry)

        # update database from disk in case anything changed
        if to_disk:
            self.load()

        # use filepath as key to see if it's already in database
        if row_dict["Filepath"] in self._db.values:
            row_index = self._db[
                self._db["Filepath"] == row_dict["Filepath"]
            ].index.values
            self._db.loc[row_index, self.columns] = new_row
        # otherwise create new entry
        else:
            new_entry = pd.DataFrame([new_row], columns=self.columns)
            self._db = pd.concat([self._db, new_entry], ignore_index=True)

        # save to disk to update changes
        if to_disk:
            self.save()

    def remove_entry(self, entry, to_disk=True):
        """
        Remove an entry from the database. Removes the entire row

        Args:
            entry (corgidrp.data.Image subclass): calibration frame to add to the database
            to_disk (bool): True by default, will update DB from disk before adding entry and saving it back to disk
        """
        new_row, row_dict = self._get_values_from_entry(entry)

        # update database from disk in case anything changed
        if to_disk:
            self.load()

        if row_dict["Filepath"] in self._db.values:
            entry_index = self._db[
                self._db["Filepath"] == row_dict["Filepath"]
            ].index.values
            self._db = self._db.drop(self._db.index[entry_index])
            self._db = self._db.reset_index(drop=True)
        else:
            raise ValueError("No filepath found so could not remove.")

        # save to disk to update changes
        if to_disk:
            self.save()

    def get_calib(self, frame, dtype, to_disk=True):
        """
        Outputs the best calibration file of the given type for the input sciene frame.

        Args:
            frame (corgidrp.data.Image): an image frame to request a calibratio for
            dtype (corgidrp.data Class): for example: corgidrp.data.Dark (TODO: document the entire list of options)
            to_disk (bool): True by default, will update DB from disk before matching

        Returns:
            corgidrp.data.*: an instance of the appropriate calibration type (Exact type depends on calibration type)
        """
        if dtype not in labels:
            raise ValueError(
                "Requested calibration dtype of {0} not a valid option".format(dtype)
            )
        dtype_label = labels[dtype]

        # get values for this science frame
        _, frame_dict = self._get_values_from_entry(frame, is_calib=False)

        # update database from disk in case anything changed
        if to_disk:
            self.load()

        # downselect to only calibs of this type
        calibdf = self._db[self._db["Type"] == dtype_label]

        if dtype_label in ["Dark"]:
            # general selection criteria for 2D image frames. Can use different selection criteria for different dtypes
            options = calibdf.loc[
                (
                    (calibdf["EXPTIME"] == frame_dict["EXPTIME"])
                    & (calibdf["NAXIS1"] == frame_dict["NAXIS1"])
                    & (calibdf["NAXIS2"] == frame_dict["NAXIS2"])
                )
            ]
        else:
            options = calibdf

        # select the one closest in time
        result_index = np.abs(options["MJD"] - frame_dict["MJD"]).argmin()
        calib_filepath = options.iloc[result_index, 0]

        # load the object from disk and return it
        return dtype(calib_filepath)

    def scan_dir_for_new_entries(self, filedir, look_in_subfolders=True, to_disk=True):
        """
        Scan a folder and subfolder for calibration files and add them all to the caldb

        Args:
            filedir (str): path to folder to scan (includes all subfolders by default)
            look_in_subfolders (bool): whether to look in subfolders for files. True by default
            to_disk (bool): True by default, will update DB from disk before adding entry and saving it back to disk
        """
        calib_frames = []
        # walk the directory to find all the calibration files
        for dirpath, subfolders, filenames in os.walk(filedir):
            for filename in filenames:
                # hard coded check only for files that end in .fits
                if filename[-5:] != ".fits":
                    continue

                filepath = os.path.join(dirpath, filename)
                frame = data.autoload(filepath)

                # check what class it has been loaded as. only save frames that fall into calibration classes
                if frame.__class__ in labels:
                    calib_frames.append(frame)

            # the first iteration looks in the basedir
            # if we don't wnat to look in subdirs now, we should break
            if not look_in_subfolders:
                break

        # load all these files into the caldb
        for calib_frame in calib_frames:
            self.create_entry(calib_frame, to_disk=to_disk)
