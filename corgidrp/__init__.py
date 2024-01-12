import configparser
import os
import pathlib
import configparser

version = "0.1"

# borrowed from the kpicdrp caldb
# load in default caldbs based on configuration file
config_filepath = os.path.join(pathlib.Path.home(), ".corgidrp")
config = configparser.ConfigParser()
config.read(config_filepath)
aux_dir = config.get("PATH", "auxdir", fallback=None)
caldb_filepath = config.get("PATH", "caldb", fallback=None)
