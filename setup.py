from setuptools import setup, find_packages
import os
import pathlib
import configparser

## Create a configuration file for the corgidrp if it doesn't exist. 
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


