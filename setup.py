from setuptools import setup, find_packages
import os
import pathlib
import configparser

#### Create a configuration file for the corgidrp if it doesn't exist. 

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

# write config 
config_filepath = os.path.join(config_folder, "corgidrp.cfg")
if not os.path.exists(config_filepath):
    config = configparser.ConfigParser()
    config["PATH"] = {}
    config["PATH"]["caldb"] = os.path.join(config_folder, "corgidrp_caldb.csv") # location to store caldb
    config["DATA"] = {}
    config["DATA"]["track_individual_errors"] = "False"
    # overwrite with old settings if needed
    if oldconfig is not None:
        config["PATH"]["caldb"] = oldconfig["PATH"]["caldb"]

    with open(config_filepath, 'w') as f:
        config.write(f)

    print("corgidrp: Configuration file written to {0}. Please edit if you want things stored in different locations.".format(config_filepath))

def get_requires():
    reqs = []
    for line in open("requirements.txt", "r").readlines():
        reqs.append(line)
    return reqs

setup(
    name='corgidrp',
    version='0.1',
    description='(Roman Space Telescope) CORonaGraph Instrument Data Reduction Pipeline',
    #long_description="",
    #long_description_content_type="text/markdown",
    #url='',
    author='Roman Coronagraph Instrument CPP',
    #author_email='',
    #license='BSD',
    packages=find_packages(),
    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',

        'Programming Language :: Python :: 3',
        ],
    keywords='Roman Space Telescope Exoplanets Astronomy',
    install_requires=get_requires()
    )


