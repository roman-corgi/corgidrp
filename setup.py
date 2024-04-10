#!/usr/bin/env python

import configparser
import os
import pathlib

from setuptools import setup


## Create a configuration file for the corgidrp if it doesn't exist. 
def run_init():
    homedir = pathlib.Path.home()
    config_filepath = os.path.join(homedir, ".corgidrp")
    if not os.path.exists(config_filepath):
        corgidrp_basedir = os.path.dirname(__file__)
        config = configparser.ConfigParser()
        config["PATH"] = {}
        config["PATH"]["caldb"] = os.path.join(corgidrp_basedir, "corgidrp_caldb.csv") # location to store caldb
        config["PATH"]["auxdata"] = os.path.join(corgidrp_basedir, "aux") + os.path.sep # folder for auxiliary data
        with open(config_filepath, 'w') as f:
                config.write(f)

        print("corgidrp: Configuration file written to {0}. Please edit if you want things stored in different locations.".format(config_filepath))

if __name__ == "__main__":
    run_init()
    setup()