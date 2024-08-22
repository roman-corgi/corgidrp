import argparse
import os
import pytest
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.detector as detector
import corgidrp.darks as darks
import corgidrp.bad_pixel_calibration

box_data_path = "/Users/jmilton/Documents/CGI/CGI_TVAC_Data"
tvac_data_path = "/Users/jmilton/Documents/CGI/CGI_Data"

box_data_path = os.path.join(box_data_path, "TV-36_Coronagraphic_Data", "Cals")
tvac_data_path = os.path.join(tvac_data_path, "Cals")

# box cal files
box_dark_path = os.path.join(box_data_path, "Dark_map_240322.fits")
box_fpn_path = os.path.join(box_data_path, "FPN_map_240318.fits")
box_cic_path = os.path.join(box_data_path, "CIC_map_240322.fits")

# tvac cal files
tvac_dark_path = os.path.join(tvac_data_path, "dark_current_20240322.fits")
tvac_fpn_path = os.path.join(tvac_data_path, "fpn_20240322.fits")
tvac_cic_path = os.path.join(tvac_data_path, "cic_20240322.fits")

box_filelist = [box_cic_path] #[box_dark_path, box_fpn_path, box_cic_path]
tvac_filelist = [tvac_cic_path] #[tvac_dark_path, tvac_fpn_path, tvac_cic_path]

for file in box_filelist:
    with fits.open(file, mode='update') as hdulist:
        print("Box file: ", file)
        print("header 0: ", hdulist[0].header)
        if len(hdulist) > 1:
            print("header 1: ", hdulist[1].header)
        else:
            print("no header 1")

for file in tvac_filelist:
    with fits.open(file, mode='update') as hdulist:
        print("TVAC file: ", file)
        print("header 0: ", hdulist[0].header)
        print("header 1: ", hdulist[1].header)