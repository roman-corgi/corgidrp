import os
import numpy as np
import matplotlib.pyplot as plt
from pyklip import fakes
from corgidrp.data import Dataset
from corgidrp.l3_to_l4 import do_psf_subtraction

def convert_to_contrast():
    return

def meas_algo_thrupt(input_dataset,ct_cal,
                     pas=None,seps=None,
                     companions=None): 
    """Measure KLIP algorithm throughput by injecting and recovering fake planets.
    Based on https://pyklip.readthedocs.io/en/latest/contrast_curves.html

    Args:
        input_dataset (corgidrp.data.Dataset): PSF-subtracted dataset
        ct_cal (corgidrp.data.CoreThroughputCalibration): Core throughput calibration object
    """

    # TODO:
    #   - How to space out injected planets? phi spiral? 
    #   - How finely to sample? Sample more near IWA?
    #   - How to mask planets, make sure we don't overlap with known companions?
    #   - How to convert data to contrast units? CT file?

    # Read in PSF-subtracted dataset, collect KLIP parameters and input files
    filelist = input_dataset.pri_hdr['FNAMES']

    # Get pre-subtracted files
    pre_klip_dataset = Dataset(filelist)
    pre_klip_refs_dataset = Dataset(ref_filelist) # This may be None if no ref files


    # Convert dataset to contrast units
    pre_klip_dataset.all_data *= dn_to_contrast
    
    # Inject PSFs to input dataset
    input_planet_fluxes = [1e-4, 1e-5, 5e-6]
    if seps == None:
        seps = [20, 40, 60]
    if pas == None:
        pas = [0, 90, 180, 270]
    fwhm = 3.5 # pixels, approximate for GPI, update to CGI

    for input_planet_flux, sep in zip(input_planet_fluxes, seps):
        # inject 4 planets at that separation to improve noise
        # fake planets are injected in data number, not contrast units, so we need to convert the flux
        # for GPI, a convenient field dn_per_contrast can be used to convert the planet flux to raw data numbers
        injected_flux = input_planet_flux * dataset.dn_per_contrast
        for pa in pas:
            fakes.inject_planet(dataset.input, dataset.centers, injected_flux, dataset.wcs, sep, pa, fwhm=fwhm)

    # Run KLIP with identical parameters as for science reduction
    fakes_subtracted_dataset = do_psf_subtraction(pre_klip_dataset)

    # Retrieve fluxes of fake planets
    dat_with_fakes = kl_hdulist[1].data[1]
    dat_with_fakes_centers = [kl_hdulist[1].header['PSFCENTX'], kl_hdulist[1].header['PSFCENTY'] ]
    retrieved_fluxes = [] # will be populated, one for each separation

    for input_planet_flux, sep in zip(input_planet_fluxes, seps):
        fake_planet_fluxes = []
        for pa in pas:
            fake_flux = fakes.retrieve_planet_flux(dat_with_fakes, dat_with_fakes_centers, dataset.output_wcs[0], sep, pa, searchrad=7)
            fake_planet_fluxes.append(fake_flux)
        retrieved_fluxes.append(np.mean(fake_planet_fluxes))

    # Calculate algo throughput
    algo_throughput = np.array(retrieved_fluxes)/np.array(input_planet_fluxes) # a number less than 1 probably


    return algo_throughput