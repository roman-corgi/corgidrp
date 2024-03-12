import configparser
import os
import pathlib
import configparser

version = "0.1"

_bool_map = {"true" : True, "false" : False}

# borrowed from the kpicdrp caldb
# load in default caldbs based on configuration file
config_filepath = os.path.join(pathlib.Path.home(), ".corgidrp", "corgidrp.cfg")
config = configparser.ConfigParser()
config.read(config_filepath)

## pipeline settings
aux_dir = config.get("PATH", "auxdir", fallback=None)
caldb_filepath = config.get("PATH", "caldb", fallback=None)
track_individual_errors = _bool_map[config.get("DATA", "track_individual_errors").lower()]
