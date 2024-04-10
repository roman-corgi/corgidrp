import configparser
import os
import pathlib

from . import _version

try:
    __version__ = _version.version
except Exception:
    __version__ = "dev"

# borrowed from the kpicdrp caldb
# load in default caldbs based on configuration file
config_filepath = os.path.join(pathlib.Path.home(), ".corgidrp")
config = configparser.ConfigParser()
config.read(config_filepath)
aux_dir = config.get("PATH", "auxdir", fallback=None)
caldb_filepath = config.get("PATH", "caldb", fallback=None)
