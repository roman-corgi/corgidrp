import configparser
import os
import pathlib
import configparser

__version__ = "0.2"
version = __version__ # temporary backwards compatability 

#### Create a configuration file for the corgidrp if it doesn't exist. 
def create_config_dir():
    """
    Checks if the default .corgidrp directory exists, and if not, it sets it up
    """
    homedir = pathlib.Path.home()
    config_folder = os.path.join(homedir, ".corgidrp")
    # replace legacy file with folder if needed
    if os.path.isfile(config_folder):
        oldconfig = configparser.ConfigParser()
        oldconfig.read(config_folder)
        os.remove(config_folder)
    else:
        oldconfig = None

    # make folder if doesn't exist
    if not os.path.isdir(config_folder):
        os.mkdir(config_folder)

        # make default calibrations folder
        default_cal_dir = os.path.join(config_folder, "default_calibs")
        if not os.path.exists(default_cal_dir):
            os.mkdir(default_cal_dir)

        # write config 
        config_filepath = os.path.join(config_folder, "corgidrp.cfg")
        if not os.path.exists(config_filepath):
            config = configparser.ConfigParser()
            config["PATH"] = {}
            config["PATH"]["caldb"] = os.path.join(config_folder, "corgidrp_caldb.csv") # location to store caldb
            config["PATH"]["default_calibs"] = default_cal_dir
            config["DATA"] = {}
            config["DATA"]["track_individual_errors"] = "False"
            # overwrite with old settings if needed
            if oldconfig is not None:
                config["PATH"]["caldb"] = oldconfig["PATH"]["caldb"]

            with open(config_filepath, 'w') as f:
                config.write(f)

            print("corgidrp: Configuration file written to {0}. Please edit if you want things stored in different locations.".format(config_filepath))
create_config_dir()
    
_bool_map = {"true" : True, "false" : False}

# borrowed from the kpicdrp caldb
# load in default caldbs based on configuration file
config_filepath = os.path.join(pathlib.Path.home(), ".corgidrp", "corgidrp.cfg")
config = configparser.ConfigParser()
config.read(config_filepath)

## pipeline settings
caldb_filepath = config.get("PATH", "caldb", fallback=None)
default_cal_dir = config.get("PATH", "default_calibs", fallback=None)
track_individual_errors = _bool_map[config.get("DATA", "track_individual_errors").lower()]
