import corgisim
from corgisim import scene
from corgisim import instrument
from corgisim import outputs
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import os
import astropy.io.fits as fits
import time
from specL1sims_utils import write_png_and_header

def create_spec_satspot_sim(outdir, Vmag, sptype, slit_name, slit_pos_mas, spot_contrast, gain=200, exptime=200, nframes=3, output_dim=121, overfac=5, loc_x=512, loc_y=512):
    """
    Generate slit + prism spectroscopic simulations with DM satellite spots.
    
    Creates L1 detector images of an occulted star with deformable mirror (DM) 
    satellite spots for wavefront sensing. Uses FPM in, slit in, prism disperser, 
    and wideband filter 3D.
    
    Parameters
    ----------
    outdir : str
        Output directory path for saving simulation files
    Vmag : float
        V-band magnitude of the star
    sptype : str
        Spectral type of the star (e.g., 'G0V', 'B3V')
    slit_name : str
        Name of the FSAM slit position (e.g., 'R1C2')
    slit_pos_mas : float
        Vertical slit offset from the FPM in milliarcseconds
    spot_contrast : float
        Contrast ratio of the satellite spots (e.g., 1E-5)
    gain : float, optional
        EMCCD gain value (default: 200)
    exptime : float, optional
        Exposure time in seconds (default: 200)
    nframes : int, optional
        Number of frames to generate (default: 3)
    output_dim : int, optional
        Dimension (width/height) of the output image in pixels (default: 121)
    overfac : int, optional
        Oversampling factor for optical simulation (default: 5)
    loc_x : int, optional
        X-coordinate location on the detector array (default: 512)
    loc_y : int, optional
        Y-coordinate location on the detector array (default: 512)
    
    Returns
    -------
    None
        Saves FITS files, PNG images, and header text files to outdir
    """
    
    # --- Host Star Properties ---
    host_star_properties = {'Vmag': Vmag,
                            'spectral_type': sptype,
                            'magtype': 'vegamag',
                            'ref_flag': True}
    
    # Define their positions relative to the host star, in milliarcseconds (mas)
    # For reference: 1 λ/D at 550 nm with a 2.3 m telescope is ~49.3 mas
    mas_per_lamD = 63.72 # Band 3
    
    base_scene = scene.Scene(host_star_properties)
    
    cgi_mode = 'spec'
    cor_type = 'spc-spec_band3'
    cases = ['2e-8']      
    rootname = 'spc-spec_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
    
    # Slit+Prism images of occulted star 
    # FPM in, R1C2 slit in, filter 3D
    
    ##  Define the polaxis parameter. Use 10 for non-polaxis cases only, as other options are not yet implemented.
    polaxis = 10
    
    slit_pos_lamod = slit_pos_mas / mas_per_lamD 
    
    occulted_star_optics_keywords = {'cor_type':cor_type, 'use_errors':1, 'polaxis':polaxis, 'output_dim':output_dim, 
            'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'use_fpm':1, 'use_lyot_stop':1,
            # 'slit':'None',
            'slit':slit_name, 'slit_y_offset_mas':slit_pos_mas,
            # 'prism':'None'}
            'prism':'PRISM3', 'wav_step_um':2E-3}

    satspot_keywords = {'num_pairs':1, 'sep_lamD':slit_pos_lamod, 'angle_deg':[90], 
                        'contrast':spot_contrast, 'wavelength_m': 730E-9}
    
    optics_without_spots = instrument.CorgiOptics(cgi_mode, bandpass='3D', optics_keywords=occulted_star_optics_keywords,
                                                  if_quiet=True, oversampling_factor = overfac, return_oversample = False)
    optics_with_spots = instrument.CorgiOptics(cgi_mode, bandpass='3D', optics_keywords=occulted_star_optics_keywords,
                                               satspot_keywords=satspot_keywords, if_quiet=True, oversampling_factor = overfac, 
                                               return_oversample = False)

    sim_satspot = optics_with_spots.get_host_star_psf(base_scene)

    emccd_keywords ={'em_gain':gain, 'cr_rate':0}
    detector = instrument.CorgiDetector(emccd_keywords)

    ##### Diagnostic plots
    # sim_scene_without_spots = optics_without_spots.get_host_star_psf(base_scene)
    # image_without_spots = sim_scene_without_spots.host_star_image.data 
    # plt.figure(figsize=(12,10))
    # plt.imshow(image_without_spots)
    # plt.colorbar()
    # plt.show()
    #
    # crop = 20
    # image_with_spots = sim_satspot.host_star_image.data 
    # plt.figure(figsize=(8,6))
    # plt.imshow(image_with_spots[crop:-crop,crop:-crop], origin='lower')
    # plt.colorbar()
    # plt.show()
    # 
    # detector.generate_detector_image(sim_satspot, exptime, full_frame=False)
    # plt.figure(figsize=(8,6))
    # plt.imshow(sim_satspot.image_on_detector.data[crop:-crop,crop:-crop], origin='lower')
    # plt.colorbar()
    # plt.show()

    for ii in range(nframes):
        detector.generate_detector_image(sim_satspot, exptime, full_frame=True, loc_x=loc_x, loc_y=loc_y)
        outputs.save_hdu_to_fits(sim_satspot.image_on_detector, outdir=outdir, write_as_L1=True)
        png_fname, header_txt_fname = write_png_and_header(sim_satspot, outdir, loc_x, loc_y, output_dim)
        time.sleep(1) # Wait 1 sec to prevent collision between filenames