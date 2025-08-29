# This script was used for producing the simulated data to create corethroughput calibration files.
# Do not include this in production as it is only meant for reference.

from corgisim import scene
from corgisim import instrument
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import proper
import warnings
from corgisim import outputs
import psutil
import math

#psutil.Process().nice(19) 

import roman_preflight_proper
roman_preflight_proper.copy_here()

Vmag = 10                            # V-band magnitude of the host star
sptype = 'G2V'                      # Spectral type of the host star
ref_flag = False                    # if the target is a reference star or not, default is False
host_star_properties = {'Vmag': Vmag,
                        'spectral_type': sptype,
                        'magtype': 'vegamag',
                        'ref_flag': False,
                        }


# --- Create the Astrophysical Scene ---
# This Scene object combines the host star 
base_scene = scene.Scene(host_star_properties)

# --- Access the generated stellar spectrum ---
sp_star = base_scene.stellar_spectrum

# Setting coronagraph and imaging mode
# See examples/tutorial1_corgisim.ipynb for more ways to set these parameters. Note that
# not all combinations may work with this code. 
cgi_mode = 'spec'
cor_type = 'spc-spec_band3'
bandpass = '3F'
cases = ['2e-8']      

rootname = 'spc-spec_ni_' + cases[0]
dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )

polaxis = 10
# output_dim define the size of the output image
output_dim = 90
overfac = 5

pupil = 0 

# Range of distances and PAs you want to sample
mag = np.arange(1, 10, 0.25) # lambda/D
PAs = np.arange(0, 360, 15) # Degrees


for c, pa in enumerate(PAs): 

    print(f'Processing new PA {c+1}/{len(PAs)}')
    exptime = 0.5 # Starting exposure time for the closest in observation
    prevmax = 6000 # Previous maximum pixel value (assumes 6000 for the first iteration)
    for magn in mag:

        pa *= math.pi/180
        src_x_offset = magn*np.sin(pa)
        src_y_offset = magn*np.cos(pa)
        
        proper_keywords = {'cor_type':cor_type, 'use_errors':2, 'polaxis':polaxis, 'output_dim':output_dim, 
                  'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1, "source_y_offset": src_y_offset, "source_x_offset": src_x_offset}

                        
        optics_noslit_noprism = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=proper_keywords, if_quiet=True,
                                small_spc_grid = 0, oversample = overfac, return_oversample = False)
        
        sim_scene_noslit_noprism = optics_noslit_noprism.get_host_star_psf(base_scene)
        image_star_noslit_noprism = sim_scene_noslit_noprism.host_star_image.data
        nd_filter = 0

        # Adjusts the exposure time based on the previous max pixel value 
        mult = 1.0+(6000.0-prevmax)/7000.0 # 6000 and 7000 are arbitrary, feel free to change
        mult = max(mult, 0.05)
        exptime *= min(mult, 1.75)
        
        emccd_keywords ={"cr_rate":0} # Using default parameters except for cosmic ray rate
        detector = instrument.CorgiDetector(emccd_keywords)
        
        sim_scene = detector.generate_detector_image(sim_scene_noslit_noprism,exptime)
        image_star_noslit_noprism = sim_scene.image_on_detector.data

        final = detector.place_scene_on_detector(image_star_noslit_noprism, 512, 512)

        prevmax = np.max(final) # Update previous max pixel value

        print(f'Max pixel value: {prevmax}')
        print(f'Exposure Time: {exptime} seconds')
        
        if prevmax > 12000:
            warnings.warn(f"Image at PA {pa*180/math.pi} degrees and distance {magn} lambda/D has a max pixel value of {prevmax}, which is probably saturated")

        sim_scene = detector.generate_detector_image(sim_scene,exptime,full_frame=True,loc_x=512, loc_y=512)
        
        outdir = "ctoutputf"
        sim_scene.image_on_detector[0].header['EXPTIME'] = exptime
        
        
        primary_hdu = fits.PrimaryHDU(data=final, header=sim_scene.image_on_detector[0].header)
        extension_header = sim_scene.image_on_detector[1].header
       
        extension_hdu = fits.ImageHDU(data=None, header=extension_header)

        hdul = fits.HDUList([primary_hdu, extension_hdu])
        
        # Convert PA back to degrees for filename and save output
        pa *= 180/math.pi
        pa = int(pa)
        
        roundedmagn = round(magn, 3) 
        outputs.save_hdu_to_fits(hdul, outdir=outdir, filename=f'flux_map-3-f-{roundedmagn}-{pa}.fits', overwrite=True, write_as_L1=False)
