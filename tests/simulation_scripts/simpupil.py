# This script was used for producing the simulated data to create corethroughput calibration files.
# Do not include this in production as it is only meant for reference.

from corgisim import scene
from corgisim import instrument
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import proper
from corgisim import outputs
import psutil

#psutil.Process().nice(19)

import roman_preflight_proper
roman_preflight_proper.copy_here()

Vmag = 10                            # V-band magnitude of the host star
sptype = 'G0V'                      # Spectral type of the host star
ref_flag = False                    # if the target is a reference star or not, default is False
host_star_properties = {'Vmag': Vmag,
                        'spectral_type': sptype,
                        'magtype': 'vegamag',
                        'ref_flag': False}


# --- Create the Astrophysical Scene ---
# This Scene object combines the host star and companion(s)
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
output_dim = 1024

pupil = 1

# Exposure time and oversampling factor. Adjust as necessary.
exptime = 0.75
overfac = 5

proper_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':polaxis, 'output_dim':output_dim,\
                            'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':0, 'use_lyot_stop':1,  'use_field_stop':0,
                            "use_pupil_lens": pupil, "use_pupil_defocus": pupil}

                        
optics_noslit_noprism = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=proper_keywords, if_quiet=True,
                                small_spc_grid = 0, oversample = overfac, return_oversample = False)
nd_filter = 0

sim_scene = optics_noslit_noprism.get_host_star_psf(base_scene)
image_star_corgi = sim_scene.host_star_image.data


emccd_keywords ={"cr_rate":0}
detector = instrument.CorgiDetector(emccd_keywords)


sim_scene_new = detector.generate_detector_image(sim_scene,exptime)
image_tot_corgi_sub= sim_scene_new.image_on_detector.data
print(f'Max pixel value: {np.max(image_tot_corgi_sub)}')
final = image_tot_corgi_sub

outdir = "ctoutputf"
sim_scene = detector.generate_detector_image(sim_scene,exptime,full_frame=True,loc_x=512, loc_y=512)
        
# Headers necessary to be recognized as a pupil image
sim_scene.image_on_detector[1].header['CFAMNAME'] = '3F'
sim_scene.image_on_detector[1].header['DPAMNAME'] = 'PUPIL'
sim_scene.image_on_detector[1].header['LSAMNAME'] = 'OPEN'
sim_scene.image_on_detector[1].header['FSAMNAME'] = 'OPEN'
sim_scene.image_on_detector[1].header['FPAMNAME'] = 'OPEN_12'
   

primary_hdu = fits.PrimaryHDU(data=final, header=sim_scene.image_on_detector[0].header)
extension_header = sim_scene.image_on_detector[1].header

extension_hdu = fits.ImageHDU(data=None, header=extension_header)

hdul = fits.HDUList([primary_hdu, extension_hdu])
        
outputs.save_hdu_to_fits(hdul, outdir=outdir, filename=f'pupil.fits', overwrite=True, write_as_L1=False)
