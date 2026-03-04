import corgisim
from corgisim import scene
from corgisim import instrument
from corgisim import outputs
import matplotlib.pyplot as plt
import numpy as np
import proper
import roman_preflight_proper
import os
import time
import astropy.io.fits as fits
from specL1sims_utils import write_png_and_header

def create_occ_target_sim(outdir, Vmag, sptype, slit_name, slit_pos_mas, comp_dra, comp_ddec, 
                          comp_fluxratio, gain=200, exptime=200, nframes=5, output_dim=161, overfac=5, loc_x=512, loc_y=512):
    """
    Generate occulted target star science observations with companion source.
    
    Creates L1 detector images of an occulted target star with a companion point source 
    (e.g., planet or disk feature) for high-contrast spectroscopy. Uses FPM in, 
    slit in, prism disperser, and narrowband filter 3F.
    
    Parameters
    ----------
    outdir : str
        Output directory path for saving simulation files
    Vmag : float
        V-band magnitude of the target star
    sptype : str
        Spectral type of the target star (e.g., 'G0V')
    slit_name : str
        Name of the FSAM slit position (e.g., 'R1C2')
    slit_pos_mas : float
        Vertical slit offset from the FPM in milliarcseconds
    comp_dra : float
        Companion right ascension offset from host star in milliarcseconds
    comp_ddec : float
        Companion declination offset from host star in milliarcseconds
    comp_fluxratio : float
        Flux ratio of companion to host star (e.g., 2E-7 for a faint companion)
    gain : float, optional
        EMCCD gain value (default: 200)
    exptime : float, optional
        Exposure time in seconds (default: 200)
    nframes : int, optional
        Number of frames to generate (default: 5)
    output_dim : int, optional
        Dimension (width/height) of the output image in pixels (default: 161)
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

    host_star_properties = {'Vmag': Vmag,
                            'spectral_type': sptype,
                            'magtype': 'vegamag',
                            'ref_flag': False}

    mag_companion = Vmag - 2.5*np.log10(comp_fluxratio)
    print('mag_companion:', mag_companion)
    point_source_info = [
        {
            'Vmag': mag_companion,
            'magtype': 'vegamag',
            'position_x': comp_dra,
            'position_y': comp_ddec
        }
    ]
    base_scene = scene.Scene(host_star_properties, point_source_info)
    
    cgi_mode = 'spec'
    cor_type = 'spc-spec_band3'
    bandpass = '3F'
    cases = ['2e-8']      
    # cases = ['1e-9']      
    rootname = 'spc-spec_ni_' + cases[0]
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
    
    ##  Define the polaxis parameter. Use 10 for non-polaxis cases only, as other options are not yet implemented.
    polaxis = 10

    occulted_star_optics_keywords = {
            'cor_type':cor_type, 'use_errors':1, 'polaxis':polaxis, 'output_dim':output_dim, 
            'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'use_fpm':1, 'use_lyot_stop':1,
            'slit':slit_name, 'slit_dec_offset_mas':slit_pos_mas,
            'prism':'PRISM3', 'wav_step_um':2E-3}

    optics_slit_prism = instrument.CorgiOptics(cgi_mode, bandpass=bandpass, optics_keywords=occulted_star_optics_keywords,
                                              if_quiet=True, oversampling_factor = overfac, return_oversample = False)
    sim_occ_target = optics_slit_prism.get_host_star_psf(base_scene)
    optics_slit_prism.inject_point_sources(base_scene, sim_occ_target)
    image_star_slit_prism = sim_occ_target.host_star_image.data
    image_comp_slit_prism = sim_occ_target.point_source_image.data
    combined_image_slit_prism = image_star_slit_prism + image_comp_slit_prism

    # Diagnostic plot of noiseless target star companion prism image
    # crop = 20
    # plt.figure(figsize=(8,6))
    # plt.imshow(combined_image_slit_prism[crop:-crop,crop:-crop], origin='lower')
    # plt.colorbar()
    # plt.show()
    
    emccd_keywords = {'cr_rate':0.0, 'em_gain':gain}
    detector = instrument.CorgiDetector(emccd_keywords)

    # Diagnostic plot of noisy target star companion prism image
    # detector.generate_detector_image(sim_occ_target, exptime, full_frame=False)
    # plt.figure(figsize=(8,6))
    # plt.imshow(sim_occ_target.image_on_detector.data[crop:-crop,crop:-crop], origin='lower')
    # plt.colorbar()
    # plt.show()

    for ii in range(nframes):
        detector.generate_detector_image(sim_occ_target, exptime, full_frame=True, loc_x=loc_x, loc_y=loc_y)
        outputs.save_hdu_to_fits(sim_occ_target.image_on_detector, outdir=outdir, write_as_L1=True)
        png_fname, header_txt_fname = write_png_and_header(sim_occ_target, outdir, loc_x, loc_y, output_dim)
        time.sleep(1) # Wait 1 sec to prevent collision between filenames

if __name__ == '__main__':
    outdir = os.path.relpath('./spec_occ_targetstar')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    create_occ_target_sim(outdir, Vmag=5.0, sptype='G0V', slit_name='R1C2', slit_pos_mas=320.0, 
                          comp_dra=0.0, comp_ddec=320.0, comp_fluxratio=5E-7, 
                          gain=500, exptime=2000, nframes=0)




