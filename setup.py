from setuptools import setup, find_packages



def get_requires():
    reqs = []
    for line in open("requirements.txt", "r").readlines():
        reqs.append(line)
    return reqs

setup(
    name='corgidrp',
    version="3.0",
    description='(Roman Space Telescope) CORonaGraph Instrument Data Reduction Pipeline',
    #long_description="",
    #long_description_content_type="text/markdown",
    url='https://github.com/roman-corgi/corgidrp',
    author='Roman Coronagraph Instrument CPP',
    #author_email='',
    license='BSD',
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

