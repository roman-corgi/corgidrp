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

import astropy.time as time
from astropy.io import fits
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
    "FPAMNAME": str
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
          data.DetectorParams : "DetectorParams",
          data.AstrometricCalibration : "AstrometricCalibration",
          data.TrapCalibration : "TrapCalibration",
          data.FluxcalFactor : "FluxcalFactor",
          data.FpamFsamCal : "FpamFsamCal",
          data.CoreThroughputCalibration: "CoreThroughputCalibration",
          data.NDFilterSweetSpotDataset: "NDFilterSweetSpot",
          data.SpectroscopyCentroidPSF: "SpectroscopyCentroidPSF",
          data.DispersionModel: "DispersionModel",
          data.MuellerMatrix: "MuellerMatrix",
          data.NDMuellerMatrix: "NDMuellerMatrix",
          data.SpecFilterOffset: "SpecFilterOffset",
          data.SpecFluxCal: "SpecFluxCal"
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
        self._db = pd.read_csv(self.filepath, dtype=column_dtypes)
        # Scan the database for any columns that might be missing, fill in missing columns with default values if necessary
        for col in column_names:
            if col not in self._db.columns:
                self._db[col] = default_values[column_dtypes[col]]

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
        # each if/else statement returns a single filepath to a good calibration
        if frame is None:
            # no frame is passed in, get the most recently created 
            options = calibdf

            # select the one that was most recently created
            result_index = options["Date Created"].argmax()
            calib_filepath = options.iloc[result_index, 0]

        elif dtype_label in ["Dark"]:
            # general selection criteria for 2D image frames. Can use different selection criteria for different dtypes
            options = self.filter_calib(calibdf, "EXPTIME", frame_dict["EXPTIME"], err_if_none=True)

            # select the one closest in time
            result_index = np.abs(options["MJD"] - frame_dict["MJD"]).argmin()
            calib_filepath = options.iloc[result_index, 0]
        elif dtype_label in ['NDFilterSweetSpot']:
            # filter by color filter
            # filter_calib() is configured to not throw an error if no matches are found, so that
            # no existing e2e tests breaks, if in the future we want to strictly only use the calibration
            # files with matching headers, then set err_if_none to True
            options = self.filter_calib(calibdf, "CFAMNAME", frame_dict['CFAMNAME'], err_if_none=False)

            # select the one closest in time
            result_index = np.abs(options["MJD"] - frame_dict["MJD"]).argmin()
            calib_filepath = options.iloc[result_index, 0]
        elif dtype_label in ['FluxcalFactor']:
            # filter by color filter and DPAM
            options = self.filter_calib(calibdf, "CFAMNAME", frame_dict['CFAMNAME'], err_if_none=False)
            if frame_dict['DPAMNAME'] in ['POL0', 'POL45']:
                options = self.filter_calib(options, "DPAMNAME", frame_dict['DPAMNAME'], err_if_none=False)

            # select the one closest in time
            result_index = np.abs(options["MJD"] - frame_dict["MJD"]).argmin()
            calib_filepath = options.iloc[result_index, 0]
        elif dtype_label in ['CoreThroughputCalibration']:
            # filter by focal plane mask
            options = self.filter_calib(calibdf, "FPAMNAME", frame_dict['FPAMNAME'], err_if_none=False)

            # select the one closest in time
            result_index = np.abs(options["MJD"] - frame_dict["MJD"]).argmin()
            calib_filepath = options.iloc[result_index, 0]
        elif dtype_label in ['FlatField'] and frame_dict['DPAMNAME'] in ['POL0', 'POL45']:
            # filter by DPAM
            options = self.filter_calib(calibdf, "DPAMNAME", frame_dict['DPAMNAME'], err_if_none=False)

            # select the one closest in time
            result_index = np.abs(options["MJD"] - frame_dict["MJD"]).argmin()
            calib_filepath = options.iloc[result_index, 0]
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

    def filter_calib(self, calibdf, col_name, value, err_if_none=False):
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
        default_detparams = data.DetectorParams({}, date_valid=time.Time("2023-11-01 00:00:00", scale='utc'))
        default_detparams.save(filedir=corgidrp.default_cal_dir, filename="DetectorParams_2023-11-01T00.00.00.000.fits")
        rescan_needed = True
    # Add default FpamFsamCal calibration file if it doesn't exist
    if not os.path.exists(os.path.join(corgidrp.default_cal_dir, "FpamFsamCal_2024-02-10T00.00.00.000.fits")):
        fpamfsam_2excam = data.FpamFsamCal([],
            date_valid=time.Time("2024-02-10 00:00:00", scale='utc'))
        fpamfsam_2excam.save(filedir=corgidrp.default_cal_dir)
        rescan_needed = True
    # Add default SpecFilterOffset calibration file if it doesn't exist
    if not os.path.exists(os.path.join(corgidrp.default_cal_dir, "SpecFilterOffset_2025-12-10T00.00.00.000.fits")):
        spec_filter = data.SpecFilterOffset({},
            date_valid=time.Time("2025-12-10 00:00:00", scale='utc'))
        spec_filter.save(filedir=corgidrp.default_cal_dir)
        rescan_needed = True
    # Add default DispersionModel calibration file if it doesn't exist
    if not os.path.exists(os.path.join(corgidrp.default_cal_dir, 'cgi_0200001001001001001_20240210t0000000_dpm_cal.fits')):
        spec_datadir = os.path.join(os.path.split(corgidrp.__file__)[0], "data", "spectroscopy")
        output_dir = corgidrp.default_cal_dir
        prihdr, exthdr, errhdr, dqhdr, biashdr = mocks.create_default_L2b_headers()
        dt = time.Time("2024-02-10 00:00:00", scale='utc').to_datetime()
        dt_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
        ftime = dt.strftime("%Y%m%dt%H%M%S%f")[:-5]
        disp_filename = f"cgi_{prihdr['VISITID']}_{ftime}_l2b.fits"
        prihdr['FILETIME'] = dt_str
        prihdr['FILENAME'] = disp_filename
        exthdr['DATETIME'] = dt_str
        exthdr['FTIMEUTC'] = dt_str
        exthdr['MJDSRT'] = float(dt.mjd)
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

    if rescan_needed:
        # add default caldb entries
        default_caldb = CalDB()
        default_caldb.scan_dir_for_new_entries(corgidrp.default_cal_dir)

    # set initialization
    initialized = True

initialized = False
if not os.environ.get('CORGIDRP_DO_NOT_AUTO_INIT_CALDB', False):
    initialize()
