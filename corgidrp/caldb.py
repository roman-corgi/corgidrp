"""
Calibration tracking system. Modified from kpicdrp caldb implmentation (Copyright (c) 2024, KPIC Team)
"""
import os
import numpy as np
import pandas as pd
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.spec as spec
import corgidrp.flat as flat
import corgidrp.detector as detector

import astropy.time as time
import astropy.io.fits as fits
from astropy.table import Table
import datetime

column_dtypes = {
    "Filepath": str,
    "Type": str,
    "MJD": float,
    "EXPTIME": float,
    "Files Used": int,
    "Date Created": float,
    "Hash": str,
    "DRPVERSN": str,
    "OBSNUM": str,
    "NAXIS1": int,
    "NAXIS2": int,
    "OPMODE": str,
    "EMGAIN_C": float,
    "EXCAMT": float,
    "CFAMNAME": str,
    "DPAMNAME": str,
    "FPAMNAME": str,
    "FSAMNAME": str,
    "SPAMNAME": str,
    "PC_STAT": str
}

default_values = {
    str : "",
    float : np.nan,
    int : -1
}

column_names = list(column_dtypes.keys())

labels = {data.Dark: "Dark",
          data.NonLinearityCalibration: "NonLinearityCalibration",
          data.KGain : "KGain",
          data.BadPixelMap: "BadPixelMap",
          data.DetectorNoiseMaps: "DetectorNoiseMaps",
          data.FlatField : "FlatField",
          data.FlatFieldPOL0 : "FlatField",
          data.FlatFieldPOL45 : "FlatField",
          data.DetectorParams : "DetectorParams",
          data.AstrometricCalibration : "AstrometricCalibration",
          data.TrapCalibration : "TrapCalibration",
          data.FluxcalFactor : "FluxcalFactor",
          data.FluxcalFactorPOL0 : "FluxcalFactor",
          data.FluxcalFactorPOL45 : "FluxcalFactor",
          data.FpamFsamCal : "FpamFsamCal",
          data.CoreThroughputCalibration: "CoreThroughputCalibration",
          data.NDFilterSweetSpotDataset: "NDFilterSweetSpot",
          data.NDSpectroscopy: "NDSpectroscopy",
          data.SpectroscopyCentroidPSF: "SpectroscopyCentroidPSF",
          data.DispersionModel: "DispersionModel",
          data.MuellerMatrix: "MuellerMatrix",
          data.NDMuellerMatrix: "NDMuellerMatrix",
          data.SpecFilterOffset: "SpecFilterOffset",
          data.SpecFluxCal: "SpecFluxCal",
          data.SlitTransmission: "SlitTransmission",
          data.LineSpread: "LineSpread"
          }

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
        # Scan the database for any columns that might be missing, fill in missing columns with default values if necessary
        for col in column_names:
            if col not in self._db.columns:
                self._db[col] = default_values[column_dtypes[col]]
        # fill any NA values with defaults and enforce column types
        for col, dtype in column_dtypes.items():
            if col in self._db.columns:
                self._db[col] = self._db[col].fillna(default_values[dtype]).astype(dtype)

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
        # return a dummy entry if nothing is passed in
        if entry is None:
            time_now = time.Time.now()
            row_dict = {
                "Filepath" : "",
                "Type" : "Sci",
                "MJD" : time_now.mjd,
                "EXPTIME" : 0.,
                "Files Used" : 0,
                "Date Created" : time_now.mjd,
                "Hash" : hash(time_now),
                "DRPVERSN" : "0.0",
                "OBSNUM" : "000",
                "NAXIS1": 0,
                "NAXIS2" : 0,
                "OPMODE" : "",
                "EMGAIN_C" : 0.,
                "EXCAMT" : 0,
                "CFAMNAME": ""
            }
            return list(row_dict.values()), row_dict

        filepath = os.path.abspath(entry.filepath)
        if is_calib:
            datatype = labels[entry.__class__]  # get the database str representation
        else:
            datatype = "Sci"
        mjd = float(entry.ext_hdr["MJDSRT"])

        exptime = entry.ext_hdr["EXPTIME"]
        if exptime is None:
            exptime = np.nan

        # check if this exists. will be a keyword written by corgidrp
        if "DRPNFILE" in entry.ext_hdr:
            files_used = entry.ext_hdr["DRPNFILE"]
            if files_used is None:
                files_used = -1
        else:
            files_used = -1

        if "DRPCTIME" in entry.ext_hdr:
            date_created = time.Time(entry.ext_hdr["DRPCTIME"]).mjd
            if date_created is None:
                date_created = np.nan
        else:
            date_created = np.nan

        if "DRPVERSN" in entry.ext_hdr:
            drp_version = entry.ext_hdr["DRPVERSN"]
            if drp_version is None:
                drp_version = ""
        else:
            drp_version = ""
        
        if "OBSNUM" in entry.pri_hdr:
            obsid = str(entry.pri_hdr["OBSNUM"]) # force to be str
            if obsid is None:
                obsid = ""
        else:
            obsid = ""

        hash_val = entry.get_hash()

        # this only works for 2D images. may need to adapt for non-2D calibration frames
        # import IPython; IPython.embed()

        entry_shape = entry.data.shape
        if len(entry_shape) < 2:
            naxis1 = entry.data.shape[-1]
            naxis2 = 0
        else:
            naxis1 = entry.data.shape[-1]
            naxis2 = entry.data.shape[-2]

        # naxis1 = entry.data.shape[-1]
        # naxis2 = entry.data.shape[-2]

        row = [
            filepath,
            datatype,
            mjd,
            exptime,
            files_used,
            date_created,
            hash_val,
            drp_version,
            obsid,
            naxis1,
            naxis2,
        ]

        # rest are ext_hdr keys we can copy
        start_index = len(row)
        for i in range(start_index, len(self.columns)):
            if self.columns[i] not in entry.ext_hdr:
                row.append(default_values[column_dtypes[self.columns[i]]])
            else:
                val = entry.ext_hdr[self.columns[i]]
                if val is not None:
                    row.append(val)  # add value staright from header
                else:
                    # if value is not in header, use default value
                    row.append(default_values[column_dtypes[self.columns[i]]])

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
        if not os.path.exists(entry.filepath):
            raise FileNotFoundError("Calibration file {0} does not exist on disk; save it before calling create_entry.".format(entry.filepath))

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
            if len(self._db) == 0:
                self._db = new_entry
            else:
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
        Outputs the best calibration file of the given type for the input science frame.

        Args:
            frame (corgidrp.data.Image): an image frame to request a calibration for. If None is passed in, looks for the 
                                         most recently created calibration. 
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
        if len(calibdf) == 0:
            raise ValueError("No valid {0} calibration in caldb located at {1}".format(dtype_label, self.filepath))

        # different logic for different cases
        # each if/else statement sets options_sorted: candidates ordered by preference (best first)
        if frame is None:
            # no frame is passed in, get the most recently created
            options_sorted = calibdf.sort_values("Date Created", ascending=False)

        elif dtype_label in ["Dark"]:
            # match on exposure time and EM gain
            options = self.filter_calib(calibdf, "EXPTIME", frame_dict["EXPTIME"], err_if_none=True)
            options = self.filter_calib(options, "EMGAIN_C", frame_dict["EMGAIN_C"], err_if_none=True)

            # for analog frames, exclude PC master darks. for PC frames, prefer them
            is_pc = frame.ext_hdr.get('ISPC', 0)
            if is_pc:
                # prefer PC master dark if available, otherwise fall back to any matching dark
                pc_options = options[options['PC_STAT'] == 'photon-counted master dark']
                if len(pc_options) > 0:
                    options = pc_options
            else:
                # analog frame: do not select a matching PC master dark
                options = options[options['PC_STAT'] != 'photon-counted master dark']
                if len(options) == 0:
                    raise ValueError(
                        "No valid analog Dark calibration found in caldb located at {0} "
                        "(no analog master darks with EXPTIME={1} and EMGAIN_C={2}).".format(
                            self.filepath, frame_dict["EXPTIME"], frame_dict["EMGAIN_C"]
                        )
                    )

            # sort by closest in time
            options_sorted = options.iloc[np.argsort(np.abs(options["MJD"] - frame_dict["MJD"]))]
        elif dtype_label in ['FluxcalFactor']:
            # filter by color filter and DPAM
            # FluxcalFactorPOL0/FluxcalFactorPOL45 force the DPAMNAME for lookup
            if dtype == data.FluxcalFactorPOL0:
                dpamname = "POL0"
            elif dtype == data.FluxcalFactorPOL45:
                dpamname = "POL45"
            else:
                dpamname = frame_dict['DPAMNAME']
            options = self.filter_calib(calibdf, "CFAMNAME", frame_dict['CFAMNAME'], err_if_none=True)
            options = self.filter_calib(options, "DPAMNAME", dpamname, err_if_none=True)

            # sort by closest in time
            options_sorted = options.iloc[np.argsort(np.abs(options["MJD"] - frame_dict["MJD"]))]
            # FluxcalFactorPOL0/FluxcalFactorPOL45 are looked up as FluxcalFactor entries on disk
            dtype = data.FluxcalFactor

        elif dtype_label in ['SpecFluxCal']:
            # filter by color filter and DPAM
            if frame_dict['CFAMNAME'] in ['2F', '3F', '2A', '2B', '2C', '3A', '3B', '3C', '3D', '3E', '3G']:
                value = list(frame_dict['CFAMNAME'])[0] + 'F'
                options = self.filter_calib(calibdf, "CFAMNAME", value, err_if_none=True)
            else:
                options = self.filter_calib(calibdf, "CFAMNAME", frame_dict['CFAMNAME'], err_if_none=True)
            options = self.filter_calib(options, "DPAMNAME", frame_dict['DPAMNAME'], err_if_none=True)

            # sort by closest in time
            options_sorted = options.iloc[np.argsort(np.abs(options["MJD"] - frame_dict["MJD"]))]
            dtype = data.SpecFluxCal
            
        elif dtype_label in ['CoreThroughputCalibration']:
            # filter by focal plane mask
            options = self.filter_calib(calibdf, "FPAMNAME", frame_dict['FPAMNAME'], err_if_none=True)
            options = self.filter_calib(options, "CFAMNAME", frame_dict['CFAMNAME'], err_if_none=True)

            # sort by closest in time
            options_sorted = options.iloc[np.argsort(np.abs(options["MJD"] - frame_dict["MJD"]))]
        elif dtype_label in ['FlatField']:
            # DPAM: IMAGING, POL0, or POL45 only. All other DPAM settings, including PUPIL, use flat = ones
            # CFAM: spectroscopy bands (2F/3F and their sub-filters) and CLEAR use flat = ones
            #       all other non-imaging CFAM positions should use a custom recipe, not auto-selection
            # Sub-bands 1A/1B/1C and 4A/4B/4C are mapped to their corresponding broadband (1F, 4F)
            # FPAM: FPM-in data uses the open-substrate flat for the appropriate band pair (OPEN_12 or OPEN_34)
            spectroscopy_cfams = {'2F', '3F', '2A', '2B', '2C', '3A', '3B', '3C', '3D', '3E', '3G'}
            imaging_cfams = {'1F', '1A', '1B', '1C', '4F', '4A', '4B', '4C'}
            cfam_subband_map = {'1A': '1F', '1B': '1F', '1C': '1F',
                                '4A': '4F', '4B': '4F', '4C': '4F'}
            ones_flat_cfams = spectroscopy_cfams | {'CLEAR'}
            if frame_dict['DPAMNAME'] not in ['IMAGING', 'POL0', 'POL45'] or frame_dict['CFAMNAME'] in ones_flat_cfams:
                # Use an all-ones flat file identified with FPAMNAME='ONES'
                options = calibdf[calibdf["FPAMNAME"] == "ONES"]
                if len(options) == 0:
                    raise ValueError("No ones-flat FlatField calibration found in caldb at {0}".format(self.filepath))
                # Any ones-flat is equivalent, choose the most recently created
                options_sorted = options.sort_values("MJD", ascending=False)
            elif frame_dict['CFAMNAME'] not in imaging_cfams:
                raise ValueError(
                    "No flat defined for CFAMNAME={0}, use a custom recipe".format(frame_dict['CFAMNAME'])
                )
            else:
                # sub-bands map to their corresponding broadband flat
                cfam_lookup = cfam_subband_map.get(frame_dict['CFAMNAME'], frame_dict['CFAMNAME'])
                options = self.filter_calib(calibdf, "CFAMNAME", cfam_lookup, err_if_none=True)
                options = self.filter_calib(options, "DPAMNAME", frame_dict['DPAMNAME'], err_if_none=True)

                # FPAM should be either OPEN_12 for bands 1&2, or OPEN_34 for bands 3&4
                options = options[options["FPAMNAME"].str.startswith("OPEN")]
                if len(options) == 0:
                    raise ValueError("No FlatField with OPEN FPAM found in caldb at {0} for CFAMNAME={1}, DPAMNAME={2}".format(
                        self.filepath, cfam_lookup, frame_dict['DPAMNAME']))
                options_sorted = options.iloc[np.argsort(np.abs(options["MJD"] - frame_dict["MJD"]))]
            # FlatFieldPOL0/FlatFieldPOL45 are looked up as FlatField entries on disk
            dtype = data.FlatField
        elif dtype_label in ['MuellerMatrix']:
            # filter by color filter
            options = self.filter_calib(calibdf, "CFAMNAME", frame_dict['CFAMNAME'], err_if_none=True)

            # sort by closest in time
            options_sorted = options.iloc[np.argsort(np.abs(options["MJD"] - frame_dict["MJD"]))]
        elif dtype_label in ['SlitTransmission']:
            # filter by slit mask (FSAM), prism (DPAM), and color filter (CFAM)
            options = self.filter_calib(calibdf, "FSAMNAME", frame_dict['FSAMNAME'], err_if_none=True)
            options = self.filter_calib(options, "DPAMNAME", frame_dict['DPAMNAME'], err_if_none=True)
            options = self.filter_calib(options, "CFAMNAME", frame_dict['CFAMNAME'], err_if_none=True)

            # sort by closest in time
            options_sorted = options.iloc[np.argsort(np.abs(options["MJD"] - frame_dict["MJD"]))]
        elif dtype_label in ['LineSpread']:
            # filter by prism (DPAM) and color filter (CFAM)
            options = self.filter_calib(calibdf, "DPAMNAME", frame_dict['DPAMNAME'], err_if_none=True)
            options = self.filter_calib(options, "CFAMNAME", frame_dict['CFAMNAME'], err_if_none=True)

            # sort by closest in time
            options_sorted = options.iloc[np.argsort(np.abs(options["MJD"] - frame_dict["MJD"]))]
        elif dtype_label in ['SpectroscopyCentroidPSF']:
            # filter by prism (DPAM) and color filter (CFAM)
            options = self.filter_calib(calibdf, "DPAMNAME", frame_dict['DPAMNAME'], err_if_none=True)
            options = self.filter_calib(options, "CFAMNAME", frame_dict['CFAMNAME'], err_if_none=True)

            # sort by closest in time
            options_sorted = options.iloc[np.argsort(np.abs(options["MJD"] - frame_dict["MJD"]))]
        else:
            # sort by closest in time
            options_sorted = calibdf.iloc[np.argsort(np.abs(calibdf["MJD"] - frame_dict["MJD"]))]

        # load the object from disk and return it
        calib_filepath = self._pick_existing(options_sorted, dtype_label)
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
                try:
                    frame = data.autoload(filepath)
                except Exception:
                    continue

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

    def _pick_existing(self, options, dtype_label):
        """Walk candidates in preference order and return the first filepath that exists on disk.

        Args:
            options (pd.DataFrame): candidate calibration rows sorted in preference order (best first)
            dtype_label (str): calibration type label used in error messages

        Returns:
            str: filepath of the best existing calibration

        Raises:
            ValueError: if no candidate file exists on disk
        """
        for filepath in options["Filepath"]:
            if os.path.exists(filepath):
                return filepath
            print("Calibration file {0} no longer exists on disk, trying next best option.".format(filepath))
        raise ValueError(
            "No valid {0} calibration in caldb located at {1}: "
            "all matching entries reference files that no longer exist on disk".format(
                dtype_label, self.filepath
            )
        )

    def filter_calib(self, calibdf, col_name, value, err_if_none=True):
        '''
        Takes in a calibration dataframe, filters them so that
        only the files with matching header values are returned. If none is found,
        this function is omitted and the original list is returned or an error is 
        thrown depending on the err_if_none parameter.

        Args:
            calibdf (pd.DataFrame): database containing the potential calibration files 
            col_name (string): name of the column that we want to look for matches in
            value (string/float/int): value of the column entry to filter by
            err_if_none (optional, boolean): tells the function whether to throw an error
            or not if no matches are found. 

        Returns:
            filtered_calibdf (pd.DataFrame): database containing only the calibration files
            with matching values, or the original database if no matches are found and 
            err_if_none is set to false. 

        '''

        filtered_calibdf = calibdf.loc[
            (
                (calibdf[col_name] == value)
            )
        ]

        if len(filtered_calibdf) == 0:
            # throws an error if err_if_none=True
            if err_if_none:
                raise ValueError(f"No valid calibration with {col_name}={value})")
            else:
                print(f"No valid calibration with {col_name}={value}")
                return calibdf
        
        return filtered_calibdf

def initialize():
    """
    Creates default calibrations and caldb if it doesn't exist

    """
    global initialized

    ### Create set of default calibrations
    rescan_needed = False
    # Add default detector_params calibration file if it doesn't exist
    if not os.path.exists(os.path.join(corgidrp.default_cal_dir, "DetectorParams_2023-11-01T00.00.00.000.fits")):
        date_valid = time.Time("2023-11-01 00:00:00", scale='utc')
        sctsrt_str = date_valid.to_datetime().strftime("%Y-%m-%dT%H:%M:%S")
        default_detparams = data.DetectorParams({}, date_valid=date_valid)
        default_detparams.ext_hdr['SCTSRT'] = sctsrt_str
        default_detparams.ext_hdr['SCTEND'] = sctsrt_str
        default_detparams.save(filedir=corgidrp.default_cal_dir, filename="DetectorParams_2023-11-01T00.00.00.000.fits")
        rescan_needed = True
    # Add default FpamFsamCal calibration file if it doesn't exist
    if not os.path.exists(os.path.join(corgidrp.default_cal_dir, "FpamFsamCal_2024-02-10T00.00.00.000.fits")):
        date_valid = time.Time("2024-02-10 00:00:00", scale='utc')
        sctsrt_str = date_valid.to_datetime().strftime("%Y-%m-%dT%H:%M:%S")
        fpamfsam_2excam = data.FpamFsamCal([], date_valid=date_valid)
        fpamfsam_2excam.ext_hdr['MJDSRT'] = float(date_valid.mjd)
        fpamfsam_2excam.ext_hdr['SCTSRT'] = sctsrt_str
        fpamfsam_2excam.ext_hdr['SCTEND'] = sctsrt_str
        fpamfsam_2excam.save(filedir=corgidrp.default_cal_dir)
        rescan_needed = True
    # Add default SpecFilterOffset calibration file if it doesn't exist
    if not os.path.exists(os.path.join(corgidrp.default_cal_dir, "SpecFilterOffset_2025-12-10T00.00.00.000.fits")):
        date_valid = time.Time("2025-12-10 00:00:00", scale='utc')
        sctsrt_str = date_valid.to_datetime().strftime("%Y-%m-%dT%H:%M:%S")
        spec_filter = data.SpecFilterOffset({}, date_valid=date_valid)
        spec_filter.ext_hdr['MJDSRT'] = float(date_valid.mjd)
        spec_filter.ext_hdr['SCTSRT'] = sctsrt_str
        spec_filter.ext_hdr['SCTEND'] = sctsrt_str
        spec_filter.save(filedir=corgidrp.default_cal_dir)
        rescan_needed = True
    # Add default DispersionModel calibration file if it doesn't exist
    if not os.path.exists(os.path.join(corgidrp.default_cal_dir, 'cgi_0200001001001001001_20240210t0000000_dpm_cal.fits')):
        spec_datadir = os.path.join(os.path.split(corgidrp.__file__)[0], "data", "spectroscopy")
        output_dir = corgidrp.default_cal_dir
        prihdr, exthdr, errhdr, dqhdr, biashdr = mocks.create_default_L2b_headers()
        dt_time = time.Time("2024-02-10 00:00:00", scale='utc')
        dt = dt_time.to_datetime()
        dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
        ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
        disp_filename = f"cgi_{prihdr['VISITID']}_{ftime}_dpm_cal.fits"
        prihdr['FILETIME'] = dt_str
        prihdr['FILENAME'] = disp_filename
        exthdr['DATETIME'] = dt_str
        exthdr['FTIMEUTC'] = dt_str
        exthdr['MJDSRT'] = float(dt_time.mjd)
        exthdr['SCTSRT'] = dt_str
        exthdr['SCTEND'] = dt_str
        # not physically relevant since we are just constructing the calibration product for the dispersion model, not 
        # the observations that produced it, but just to avoid confusion, we set the values to something sensible
        exthdr['DPAMNAME'] = 'PRISM3' 
        exthdr['CFAMNAME'] = '3F'
        exthdr['FSAMNAME'] = 'OPEN'
        # these below, however, are needed for the DispersionModel calibration 
        exthdr["REFWAVE"] = 730.
        exthdr["BAND"] = '3'
        band_list = spec.read_cent_wave('3')
        band_center = band_list[0]
        fwhm = band_list[1]
        bandpass_frac = fwhm/band_center
        exthdr["BANDFRAC"] = bandpass_frac
        disp_file_path = os.path.join(spec_datadir, "TVAC_PRISM3_dispersion_profile.npz")
        disp_params = np.load(disp_file_path)
        disp_dict = {'clocking_angle': disp_params['clocking_angle'],
                    'clocking_angle_uncertainty': disp_params['clocking_angle_uncertainty'],
                    'pos_vs_wavlen_polycoeff': disp_params['pos_vs_wavlen_polycoeff'],
                    'pos_vs_wavlen_cov' : disp_params['pos_vs_wavlen_cov'],
                    'wavlen_vs_pos_polycoeff': disp_params['wavlen_vs_pos_polycoeff'],
                    'wavlen_vs_pos_cov': disp_params['wavlen_vs_pos_cov']}
        
        disp_model = data.DispersionModel(disp_dict, pri_hdr = prihdr, ext_hdr = exthdr)
        disp_model.save(output_dir, disp_model.filename)
        rescan_needed = True

    # Add default ones-flat FlatField calibration file
    fixed_ones_flat_filename = "cgi_0000000000000000000_20000101t0000000_flt_cal.fits"
    fixed_ones_flat_filepath = os.path.join(corgidrp.default_cal_dir, fixed_ones_flat_filename)

    if not os.path.exists(fixed_ones_flat_filepath):
        ones_flat_dataset = mocks.create_flatfield_dummy(numfiles=1)
        ones_flat_dataset[0].data[:] = 1.0
        ones_flat = flat.create_flatfield(ones_flat_dataset)
        ones_flat.ext_hdr["FPAMNAME"] = "ONES"
        ones_flat.ext_hdr["MJDSRT"] = float(time.Time("2026-01-01").mjd)
        # Write to a fixed CGI-formatted filename so there's no need to search for the latest ones-flat
        ones_flat.filename = fixed_ones_flat_filename
        ones_flat.pri_hdr["FILENAME"] = fixed_ones_flat_filename
        ones_flat.save(filedir=corgidrp.default_cal_dir, filename=fixed_ones_flat_filename)
        rescan_needed = True

    # Add TVAC default calibration files built from raw TVAC data packaged in corgidrp/data/default_calibs/.
    # Fixed filenames encode the synthetic timestamp (20260101) so the existence check is stable across runs.
    tvac_raw_dir = os.path.join(os.path.split(corgidrp.__file__)[0], "data", "default_calibs")
    tvac_nln_filename = "cgi_0000000000000000000_20260101t0000000_nln_cal.fits"
    tvac_krn_filename = "cgi_0000000000000000000_20260101t0000001_krn_cal.fits"
    tvac_dnm_filename = "cgi_0000000000000000000_20260101t0000002_dnm_cal.fits"
    tvac_flt_filename = "cgi_0000000000000000000_20260101t0000003_flt_cal.fits"
    tvac_bpm_filename = "cgi_0000000000000000000_20260101t0000004_bpm_cal.fits"
    tvac_pol0_flt_filename = "cgi_0000000000000000000_20260101t0000005_flt_cal.fits"
    tvac_pol45_flt_filename = "cgi_0000000000000000000_20260101t0000006_flt_cal.fits"
    tvac_flt4_filename = "cgi_0000000000000000000_20260101t0000007_flt_cal.fits"
    tvac_pol0_flt4_filename = "cgi_0000000000000000000_20260101t0000008_flt_cal.fits"
    tvac_pol45_flt4_filename = "cgi_0000000000000000000_20260101t0000009_flt_cal.fits"

    tvac_cal_filenames = [tvac_nln_filename, tvac_krn_filename, tvac_dnm_filename,
                          tvac_flt_filename, tvac_bpm_filename,
                          tvac_pol0_flt_filename, tvac_pol45_flt_filename,
                          tvac_flt4_filename, tvac_pol0_flt4_filename, tvac_pol45_flt4_filename]
    tvac_cals_missing = any(
        not os.path.exists(os.path.join(corgidrp.default_cal_dir, f))
        for f in tvac_cal_filenames
    )

    if tvac_cals_missing:
        pri_hdr, ext_hdr, _, _ = mocks.create_default_calibration_product_headers()
        mjd_2026 = float(time.Time("2026-01-01").mjd)
        isot_2026 = "2026-01-01T00:00:00.000"
        ext_hdr["MJDSRT"] = mjd_2026
        ext_hdr["DATETIME"] = isot_2026
        ext_hdr["FTIMEUTC"] = isot_2026
        ext_hdr["DRPCTIME"] = isot_2026
        ext_hdr["DRPVERSN"] = corgidrp.__version__
        # Minimal mock input dataset for bookkeeping requirements in calibration constructors
        mock_dataset = mocks.create_flatfield_dummy(numfiles=2)

        # NonLinearityCalibration from packaged TVAC nonlinearity table
        if not os.path.exists(os.path.join(corgidrp.default_cal_dir, tvac_nln_filename)):
            nonlin_path = os.path.join(tvac_raw_dir, "nonlin_table_240322.txt")
            nonlin_dat = np.genfromtxt(nonlin_path, delimiter=",")
            nonlinear_cal = data.NonLinearityCalibration(
                nonlin_dat, pri_hdr=pri_hdr.copy(), ext_hdr=ext_hdr.copy(),
                input_dataset=mock_dataset,
            )
            nonlinear_cal.ext_hdr["MJDSRT"] = mjd_2026
            nonlinear_cal.ext_hdr["DATETIME"] = isot_2026
            nonlinear_cal.ext_hdr["FTIMEUTC"] = isot_2026
            nonlinear_cal.save(filedir=corgidrp.default_cal_dir, filename=tvac_nln_filename)

        # KGain — 8.7 e/DN and read noise from TVAC measurements
        if not os.path.exists(os.path.join(corgidrp.default_cal_dir, tvac_krn_filename)):
            kgain_val = 8.7
            signal_array = np.linspace(0, 50)
            noise_array = np.sqrt(signal_array)
            ptc = np.column_stack([signal_array, noise_array])
            ext_hdr_krn = ext_hdr.copy()
            ext_hdr_krn["RN"] = 121.76     # read noise in electrons from TVAC measurement
            ext_hdr_krn["RN_ERR"] = 2.0    # read noise uncertainty in electrons
            kgain = data.KGain(
                kgain_val, ptc=ptc, pri_hdr=pri_hdr.copy(), ext_hdr=ext_hdr_krn,
                input_dataset=mock_dataset,
            )
            kgain.ext_hdr["MJDSRT"] = mjd_2026
            kgain.ext_hdr["DATETIME"] = isot_2026
            kgain.ext_hdr["FTIMEUTC"] = isot_2026
            kgain.save(filedir=corgidrp.default_cal_dir, filename=tvac_krn_filename)

        # DetectorNoiseMaps from packaged TVAC noise component files.
        # The 1024x1024 science-image data is embedded in the full detector frame.
        if not os.path.exists(os.path.join(corgidrp.default_cal_dir, tvac_dnm_filename)):
            with fits.open(os.path.join(tvac_raw_dir, "fpn_20240322.fits")) as hdulist:
                fpn_dat = hdulist[0].data.astype(float)
            with fits.open(os.path.join(tvac_raw_dir, "cic_20240322.fits")) as hdulist:
                cic_dat = hdulist[0].data.astype(float)
            with fits.open(os.path.join(tvac_raw_dir, "dark_current_20240322.fits")) as hdulist:
                dark_dat = hdulist[0].data.astype(float)
            frame_rows = detector.detector_areas["SCI"]["frame_rows"]
            frame_cols = detector.detector_areas["SCI"]["frame_cols"]
            img_rows, img_cols, r0c0 = detector.unpack_geom("SCI", "image")
            r0, c0 = r0c0
            noise_map_dat = np.zeros((3, frame_rows, frame_cols))
            noise_map_dat[0, r0:r0 + img_rows, c0:c0 + img_cols] = fpn_dat
            noise_map_dat[1, r0:r0 + img_rows, c0:c0 + img_cols] = cic_dat
            noise_map_dat[2, r0:r0 + img_rows, c0:c0 + img_cols] = dark_dat
            noise_map_err = np.zeros((1,) + noise_map_dat.shape)
            noise_map_dq = np.zeros(noise_map_dat.shape, dtype=int)
            noise_err_hdr = fits.Header()
            noise_err_hdr["BUNIT"] = "detected electron"
            ext_hdr_dnm = ext_hdr.copy()
            ext_hdr_dnm["B_O"] = 0.0
            ext_hdr_dnm["B_O_ERR"] = 0.0
            noise_map = data.DetectorNoiseMaps(
                noise_map_dat,
                pri_hdr=pri_hdr.copy(),
                ext_hdr=ext_hdr_dnm,
                input_dataset=mock_dataset,
                err=noise_map_err,
                dq=noise_map_dq,
                err_hdr=noise_err_hdr,
            )
            noise_map.ext_hdr["MJDSRT"] = mjd_2026
            noise_map.ext_hdr["DATETIME"] = isot_2026
            noise_map.ext_hdr["FTIMEUTC"] = isot_2026
            noise_map.ext_hdr["DRPNFILE"] = 96
            noise_map.save(filedir=corgidrp.default_cal_dir, filename=tvac_dnm_filename)

        # FlatField — all-ones array for the default imaging configuration.
        # FPAMNAME='OPEN_12' and CFAMNAME='1F' match the caldb lookup for
        # standard DPAMNAME='IMAGING' observations in bands 1 & 2.
        if not os.path.exists(os.path.join(corgidrp.default_cal_dir, tvac_flt_filename)):
            flat_dat = np.ones((1024, 1024))
            flat_mock_dataset = mocks.create_flatfield_dummy(numfiles=1)
            tvac_flat = data.FlatField(flat_dat, pri_hdr=pri_hdr.copy(), ext_hdr=ext_hdr.copy(),
                                       input_dataset=flat_mock_dataset)
            tvac_flat.ext_hdr["FPAMNAME"] = "OPEN_12"
            tvac_flat.ext_hdr["CFAMNAME"] = "1F"
            tvac_flat.ext_hdr["DPAMNAME"] = "IMAGING"
            tvac_flat.ext_hdr["MJDSRT"] = mjd_2026
            tvac_flat.ext_hdr["DATETIME"] = isot_2026
            tvac_flat.ext_hdr["FTIMEUTC"] = isot_2026
            tvac_flat.save(filedir=corgidrp.default_cal_dir, filename=tvac_flt_filename)

        # BadPixelMap from packaged TVAC bad pixel data.
        # A synthetic Dark frame is required by BadPixelMap as the base header source.
        if not os.path.exists(os.path.join(corgidrp.default_cal_dir, tvac_bpm_filename)):
            with fits.open(os.path.join(tvac_raw_dir, "bad_pix.fits")) as hdulist:
                bp_dat = hdulist[0].data.astype(float)
            bp_dark_pri, bp_dark_ext, _, _ = mocks.create_default_calibration_product_headers()
            bp_dark_ext["EXPTIME"] = 1.0
            bp_dark_ext["EMGAIN_C"] = 1.0
            bp_dark_ext["DRPNFILE"] = 1
            bp_dark_ext["MJDSRT"] = mjd_2026
            bp_dark_ext["DATETIME"] = isot_2026
            bp_dark_ext["FTIMEUTC"] = isot_2026
            bp_dark = data.Dark(
                np.zeros_like(bp_dat, dtype=float),
                pri_hdr=bp_dark_pri,
                ext_hdr=bp_dark_ext,
                input_dataset=mock_dataset,
                err=np.zeros((1,) + bp_dat.shape, dtype=float),
                dq=np.zeros(bp_dat.shape, dtype="uint16"),
                err_hdr=fits.Header(),
            )
            bp_map = data.BadPixelMap(
                bp_dat,
                pri_hdr=pri_hdr.copy(),
                ext_hdr=ext_hdr.copy(),
                input_dataset=data.Dataset([bp_dark]),
            )
            bp_map.ext_hdr["MJDSRT"] = mjd_2026
            bp_map.ext_hdr["DATETIME"] = isot_2026
            bp_map.ext_hdr["FTIMEUTC"] = isot_2026
            bp_map.save(filedir=corgidrp.default_cal_dir, filename=tvac_bpm_filename)

        # POL0 FlatField — all-ones for the default polarimetry configuration.
        # FPAMNAME='OPEN_12', CFAMNAME='1F', DPAMNAME='POL0' for caldb lookup.
        if not os.path.exists(os.path.join(corgidrp.default_cal_dir, tvac_pol0_flt_filename)):
            flat_dat = np.ones((1024, 1024))
            flat_mock_dataset = mocks.create_flatfield_dummy(numfiles=1)
            pol0_flat = data.FlatField(flat_dat, pri_hdr=pri_hdr.copy(), ext_hdr=ext_hdr.copy(),
                                       input_dataset=flat_mock_dataset)
            pol0_flat.ext_hdr["FPAMNAME"] = "OPEN_12"
            pol0_flat.ext_hdr["CFAMNAME"] = "1F"
            pol0_flat.ext_hdr["DPAMNAME"] = "POL0"
            pol0_flat.ext_hdr["MJDSRT"] = mjd_2026
            pol0_flat.ext_hdr["DATETIME"] = isot_2026
            pol0_flat.ext_hdr["FTIMEUTC"] = isot_2026
            pol0_flat.save(filedir=corgidrp.default_cal_dir, filename=tvac_pol0_flt_filename)

        # POL45 FlatField — all-ones for the default polarimetry configuration.
        # FPAMNAME='OPEN_12', CFAMNAME='1F', DPAMNAME='POL45' for caldb lookup.
        if not os.path.exists(os.path.join(corgidrp.default_cal_dir, tvac_pol45_flt_filename)):
            flat_dat = np.ones((1024, 1024))
            flat_mock_dataset = mocks.create_flatfield_dummy(numfiles=1)
            pol45_flat = data.FlatField(flat_dat, pri_hdr=pri_hdr.copy(), ext_hdr=ext_hdr.copy(),
                                        input_dataset=flat_mock_dataset)
            pol45_flat.ext_hdr["FPAMNAME"] = "OPEN_12"
            pol45_flat.ext_hdr["CFAMNAME"] = "1F"
            pol45_flat.ext_hdr["DPAMNAME"] = "POL45"
            pol45_flat.ext_hdr["MJDSRT"] = mjd_2026
            pol45_flat.ext_hdr["DATETIME"] = isot_2026
            pol45_flat.ext_hdr["FTIMEUTC"] = isot_2026
            pol45_flat.save(filedir=corgidrp.default_cal_dir, filename=tvac_pol45_flt_filename)

        # Band-4 IMAGING FlatField — all-ones, FPAMNAME='OPEN_34', CFAMNAME='4F', DPAMNAME='IMAGING'.
        if not os.path.exists(os.path.join(corgidrp.default_cal_dir, tvac_flt4_filename)):
            flat_dat = np.ones((1024, 1024))
            flat_mock_dataset = mocks.create_flatfield_dummy(numfiles=1)
            flt4 = data.FlatField(flat_dat, pri_hdr=pri_hdr.copy(), ext_hdr=ext_hdr.copy(),
                                  input_dataset=flat_mock_dataset)
            flt4.ext_hdr["FPAMNAME"] = "OPEN_34"
            flt4.ext_hdr["CFAMNAME"] = "4F"
            flt4.ext_hdr["DPAMNAME"] = "IMAGING"
            flt4.ext_hdr["MJDSRT"] = mjd_2026
            flt4.ext_hdr["DATETIME"] = isot_2026
            flt4.ext_hdr["FTIMEUTC"] = isot_2026
            flt4.save(filedir=corgidrp.default_cal_dir, filename=tvac_flt4_filename)

        # Band-4 POL0 FlatField — all-ones, FPAMNAME='OPEN_34', CFAMNAME='4F', DPAMNAME='POL0'.
        if not os.path.exists(os.path.join(corgidrp.default_cal_dir, tvac_pol0_flt4_filename)):
            flat_dat = np.ones((1024, 1024))
            flat_mock_dataset = mocks.create_flatfield_dummy(numfiles=1)
            pol0_flt4 = data.FlatField(flat_dat, pri_hdr=pri_hdr.copy(), ext_hdr=ext_hdr.copy(),
                                       input_dataset=flat_mock_dataset)
            pol0_flt4.ext_hdr["FPAMNAME"] = "OPEN_34"
            pol0_flt4.ext_hdr["CFAMNAME"] = "4F"
            pol0_flt4.ext_hdr["DPAMNAME"] = "POL0"
            pol0_flt4.ext_hdr["MJDSRT"] = mjd_2026
            pol0_flt4.ext_hdr["DATETIME"] = isot_2026
            pol0_flt4.ext_hdr["FTIMEUTC"] = isot_2026
            pol0_flt4.save(filedir=corgidrp.default_cal_dir, filename=tvac_pol0_flt4_filename)

        # Band-4 POL45 FlatField — all-ones, FPAMNAME='OPEN_34', CFAMNAME='4F', DPAMNAME='POL45'.
        if not os.path.exists(os.path.join(corgidrp.default_cal_dir, tvac_pol45_flt4_filename)):
            flat_dat = np.ones((1024, 1024))
            flat_mock_dataset = mocks.create_flatfield_dummy(numfiles=1)
            pol45_flt4 = data.FlatField(flat_dat, pri_hdr=pri_hdr.copy(), ext_hdr=ext_hdr.copy(),
                                        input_dataset=flat_mock_dataset)
            pol45_flt4.ext_hdr["FPAMNAME"] = "OPEN_34"
            pol45_flt4.ext_hdr["CFAMNAME"] = "4F"
            pol45_flt4.ext_hdr["DPAMNAME"] = "POL45"
            pol45_flt4.ext_hdr["MJDSRT"] = mjd_2026
            pol45_flt4.ext_hdr["DATETIME"] = isot_2026
            pol45_flt4.ext_hdr["FTIMEUTC"] = isot_2026
            pol45_flt4.save(filedir=corgidrp.default_cal_dir, filename=tvac_pol45_flt4_filename)

        rescan_needed = True

    if rescan_needed:
        # add default caldb entries
        default_caldb = CalDB()
        default_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    # set initialization
    initialized = True

initialized = False
if not os.environ.get('CORGIDRP_DO_NOT_AUTO_INIT_CALDB', False):
    initialize()
