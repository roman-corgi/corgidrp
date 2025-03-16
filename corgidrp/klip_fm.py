import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from corgidrp.data import Dataset
from corgidrp.l3_to_l4 import do_psf_subtraction
from pyklip.fmlib import fmpsf
import pyklip.fm as fm

def get_closest_psf(ct_calibration,sep,pa):
    return ct_calibration['psfs'][0]

def meas_klip_thrupt(pyklip_dataset, # pre-psf-subtracted dataset
                     ct_calibration,
                     inject_flux,
                     numbasis,
                     seps=None, # in pixels from mask center
                     pas=None):
    
    # TODO:
        # - Figure out if we can use KLIP FM class to avoid having to rerun KLIP entirely
            # - KLIP FM reruns KLIP anyway BUT might be more accurate BUT might be slower to do each one individually...
        # - How to space out injected planets? phi spiral? 
        # - How finely to sample? Sample more near IWA?
        # - How to mask planets, make sure we don't overlap with known companions?
        # - How to convert data to contrast units? CT file?

    if pas == None:
        pas = np.array([0.,60.,120.])
    if seps == None:
        seps = np.array([10.,15.,20.]) # Some linear spacing between the IWA & OWA, around 2x the resolution element
    

    # Loop over locations:
    input_fluxes = []
    output_fluxes = []
    for sep in seps:
        this_sep = []
        for pa in pas:
            # Initialize PSF FM class from pyklip with model PSF at closest location
            # Inject fakes

            psf = get_closest_psf(ct_calibration,sep,pa) # Should be PSF cube with shape (1,Y,X)

            fm_class = fmpsf.FMPlanetPSF(pyklip_dataset.input.shape, numbasis, sep, pa, inject_flux, psf,
                                np.unique(pyklip_dataset.wvs), 
                                star_spt='G2', # Will this be in a header somewhere? Does it matter?
                                #spectrallib=[guessspec] # Do we need this?
                                )
            
            outputdir = "." # where to write the output files
            prefix = "betpic-131210-j-fmpsf" # fileprefix for the output files
            annulus_bounds = [[sep-10, sep+10]] # one annulus centered on the planet
            subsections = 1 # we are not breaking up the annulus
            padding = 0 # we are not padding our zones
            movement = 4 # should use the same as in PSF subtraction step 

            # run KLIP-FM
            
            fm.klip_dataset(pyklip_dataset, fm_class, outputdir=outputdir, fileprefix=prefix, numbasis=numbasis,
                            annuli=annulus_bounds, subsections=subsections, padding=padding, movement=movement)
            
            output_prefix = os.path.join(outputdir, prefix)
            # fm_hdu = fits.open(output_prefix + "-fmpsf-KLmodes-all.fits")
            data_hdu = fits.open(output_prefix + "-klipped-KLmodes-all.fits")

            # get data_stamp frame, use KL=7
            data_frame = data_hdu[1].data[1]
            # data_centx = data_hdu[1].header["PSFCENTX"]
            # data_centy = data_hdu[1].header["PSFCENTY"]

            fig,axes = plt.subplots(1,2,sharex=True,sharey=True,layout='constrained')
            im0 = axes[0].imshow(psf[0],origin='lower')
            plt.colorbar(im0,ax=axes[0])
            im1 = axes[1].imshow(data_frame,origin='lower')
            plt.colorbar(im1,ax=axes[1])
            plt.show()
            plt.close()
            
    return output_dataset

# def meas_klip_thrupt(input_dataset,ct_cal,
#                      pas=None,seps=None,
#                      companions=None): 
#      # TODO:

#     # Read in PSF-subtracted dataset, collect KLIP parameters and input files
#     filelist = input_dataset.pri_hdr['FNAMES']

#     # Get pre-subtracted files
#     pre_klip_dataset = Dataset(filelist)
#     pre_klip_refs_dataset = Dataset(ref_filelist) # This may be None if no ref files


#     # Convert dataset to contrast units
#     pre_klip_dataset.all_data *= dn_to_contrast
    
#     # Inject PSFs to input dataset
#     input_planet_fluxes = [1e-4, 1e-5, 5e-6]
#     if seps == None:
#         seps = [20, 40, 60]
#     if pas == None:
#         pas = [0, 90, 180, 270]
#     fwhm = 3.5 # pixels, approximate for GPI, update to CGI

#     for input_planet_flux, sep in zip(input_planet_fluxes, seps):
#         # inject 4 planets at that separation to improve noise
#         # fake planets are injected in data number, not contrast units, so we need to convert the flux
#         # for GPI, a convenient field dn_per_contrast can be used to convert the planet flux to raw data numbers
#         injected_flux = input_planet_flux * dataset.dn_per_contrast
#         for pa in pas:
#             fakes.inject_planet(dataset.input, dataset.centers, injected_flux, dataset.wcs, sep, pa, fwhm=fwhm)

#     # Run KLIP with identical parameters as for science reduction
#     fakes_subtracted_dataset = do_psf_subtraction(pre_klip_dataset)

#     # Retrieve fluxes of fake planets
#     dat_with_fakes = kl_hdulist[1].data[1]
#     dat_with_fakes_centers = [kl_hdulist[1].header['PSFCENTX'], kl_hdulist[1].header['PSFCENTY'] ]
#     retrieved_fluxes = [] # will be populated, one for each separation

#     for input_planet_flux, sep in zip(input_planet_fluxes, seps):
#         fake_planet_fluxes = []
#         for pa in pas:
#             fake_flux = fakes.retrieve_planet_flux(dat_with_fakes, dat_with_fakes_centers, dataset.output_wcs[0], sep, pa, searchrad=7)
#             fake_planet_fluxes.append(fake_flux)
#         retrieved_fluxes.append(np.mean(fake_planet_fluxes))

#     # Calculate algo throughput
#     algo_throughput = np.array(retrieved_fluxes)/np.array(input_planet_fluxes) # a number less than 1 probably


#     return algo_throughput