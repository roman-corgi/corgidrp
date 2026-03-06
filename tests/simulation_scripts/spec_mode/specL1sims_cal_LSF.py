import corgisim
from corgisim import scene
from corgisim import instrument
from corgisim import outputs
import matplotlib.pyplot as plt
import numpy as np
import os
import astropy.io.fits as fits
import itertools
from specL1sims_utils import write_png_from_sceneobj

def create_spec_LSF_sims(outdir, Vmag, sptype, slit_name, slit_pos_mas, fsm_offsets_mas, gain, exptime, ref_flag=False, 
                         output_dim=121, overfac=5, loc_x=512, loc_y=512):
    """
    Generate line spread function (LSF) spectroscopic simulations for standard star visits.
    
    Creates L1 detector images for LSF characterization including both slitless prism 
    images and slit + prism images with a 2-D grid of FSM dither positions. Uses 
    FPM in, narrowband (3F) and wideband (3D) filters.
    
    Parameters
    ----------
    outdir : str
        Output directory path for saving simulation files
    Vmag : float
        V-band magnitude of the spectrophotometric standard star
    sptype : str
        Spectral type of the star (e.g., 'G0V')
    slit_name : str
        Name of the FSAM slit position (e.g., 'R1C2')
    slit_pos_mas : float
        Vertical slit offset from the FPM in milliarcseconds
    fsm_offsets_mas : list of float
        List of FSM offset values in milliarcseconds. Will be applied as a 2-D grid 
        of horizontal and vertical offsets relative to the source location
    gain : float
        EMCCD gain value
    exptime : float
        Exposure time in seconds
    ref_flag : bool, optional
        If True, marks this as a reference star observation (default: False)
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
    L1_fits_files : list of str
        List of paths to all L1 FITS files generated
    """
    # Slit+Prism images of unocculated star according to Line Spread Function visit
    host_star_properties = {'Vmag': Vmag,
                            'spectral_type': sptype,
                            'magtype': 'vegamag',
                            'ref_flag': ref_flag}
    
    base_scene = scene.Scene(host_star_properties)
    
    cgi_mode = 'spec'
    cor_type = 'spc-spec_band3'

    # FPM in, R1C2 slit in, filters 3F and 3D
    
    ##  Define the polaxis parameter. Use 10 for non-polaxis cases only, as other options are not yet implemented.
    polaxis = 10

    short_exptime = 5.0
    emccd_keywords = {'cr_rate':0.0, 'em_gain':gain}
    detector = instrument.CorgiDetector(emccd_keywords, photon_counting=False)

    L1_fits_files = []

    # Slitless prism image
    fsm_source_offset_mas = slit_pos_mas
    slitless_offset_star_optics_keywords = {'cor_type':cor_type, 'use_errors':0, 'polaxis':polaxis, 'output_dim':output_dim, 
            'use_dm1':0, 'use_dm2':0, 'use_fpm':1, 'use_lyot_stop':1, 'fsm_y_offset_mas':fsm_source_offset_mas,
            'slit':'None', 'slit_y_offset_mas':slit_pos_mas, 'prism':'PRISM3', 'wav_step_um':2E-3}

    slitless_cfam3F_optics = instrument.CorgiOptics(cgi_mode, bandpass='3F', optics_keywords=slitless_offset_star_optics_keywords,
                                                    if_quiet=True, oversampling_factor = overfac, return_oversample = False)
    slitless_cfam3D_optics = instrument.CorgiOptics(cgi_mode, bandpass='3D', optics_keywords=slitless_offset_star_optics_keywords,
                                                    if_quiet=True, oversampling_factor = overfac, return_oversample = False)
    noiseless_cfam3F_slitless_fname = os.path.join(outdir, 'standstar_{:s}_PRISM3_template_slitless_cfam3F.fits'.format(sptype))
    crop_frame_cfam3F_slitless_fname = os.path.join(outdir, 'standstar_{:s}_PRISM3_noise_slitless_cfam3F.fits'.format(sptype))

    print("Evaluating slitless filter 3D image...") 
    sim_cfam3D_slitless = slitless_cfam3D_optics.get_host_star_psf(base_scene)
    sim_hdr_comment_cfam3D_slitless = sim_cfam3D_slitless.host_star_image.header['COMMENT']
    for card in sim_hdr_comment_cfam3D_slitless:
        if 'dispersed_image_centx' in card:
            centx_cropframe_3D = float(card.split(' : ')[1].strip())
        elif 'dispersed_image_centy' in card:
            centy_cropframe_3D = float(card.split(' : ')[1].strip())
    print("Evaluating slitless filter 3F image...") 
    sim_cfam3F_slitless = slitless_cfam3F_optics.get_host_star_psf(base_scene)
    sim_hdr_comment_cfam3F_slitless = sim_cfam3F_slitless.host_star_image.header['COMMENT']
    for card in sim_hdr_comment_cfam3F_slitless:
        if 'dispersed_image_centx' in card:
            centx_cropframe_3F = float(card.split(' : ')[1].strip())
        elif 'dispersed_image_centy' in card:
            centy_cropframe_3F = float(card.split(' : ')[1].strip())

    # noiseless template image
    noiseless_image_3F_slitless = sim_cfam3F_slitless.host_star_image.data
    noiseless_image_hdr_3F_slitless = sim_cfam3F_slitless.host_star_image.header
    noiseless_image_hdr_3F_slitless['slit_dy'] = (slit_pos_mas, 'FSAM slit vertical offset from FPM, in mas')
    noiseless_image_hdr_3F_slitless['centx'] = (centx_cropframe_3F, 'x centroid of dispersed source image at filter center wavelength')
    noiseless_image_hdr_3F_slitless['centy'] = (centy_cropframe_3F, 'y centroid of dispersed source image at filter center wavelength')
    noiseless_image_hdr_3F_slitless['wv0_x'] = (centx_cropframe_3D, 'x centroid of dispersed source image at zeropoint wavelength')
    noiseless_image_hdr_3F_slitless['wv0_y'] = (centy_cropframe_3D, 'y centroid of dispersed source image at zeropoint wavelength')
    fits.writeto(noiseless_cfam3F_slitless_fname, noiseless_image_3F_slitless, header=noiseless_image_hdr_3F_slitless, overwrite=True)

    ##### Diagnostic plots
    # noisy image 
    # detector.generate_detector_image(sim_cfam3F_slitless, exptime, full_frame=False)
    # sim_cfam3F_slitless.image_on_detector.writeto(crop_frame_cfam3F_slitless_fname, overwrite=True)
    # 
    # crop = 5
    # plt.figure(figsize=(8,6))
    # plt.imshow(sim_cfam3F_slitless.image_on_detector.data[crop:-crop,crop:-crop], origin='lower')
    # plt.colorbar()
    # plt.show()

    detector.generate_detector_image(sim_cfam3F_slitless, exptime, full_frame=True, loc_x=loc_x, loc_y=loc_y)
    outputs.save_hdu_to_fits(sim_cfam3F_slitless.image_on_detector, outdir=outdir, write_as_L1=True)
    L1_fitsname = os.path.join(outdir, sim_cfam3F_slitless.image_on_detector[0].header['FILENAME'])
    L1_fits_files.append(L1_fitsname)
    png_fname = write_png_from_sceneobj(sim_cfam3F_slitless, outdir, loc_x, loc_y, output_dim)

    # FSM-dithered prism images with slit
    fsm_offset_3F_optics_list = list() 
    fsm_offset_3D_optics_list = list() 
    fsm_xoff_mas_list = list()
    fsm_yoff_mas_list = list()
    
    for ii, (xoff_mas, yoff_mas) in enumerate(itertools.product(fsm_offsets_mas, fsm_offsets_mas)):
        fsm_source_y_offset_mas = slit_pos_mas + yoff_mas
        fsm_source_x_offset_mas = xoff_mas
        offset_star_optics_keywords = {'cor_type':cor_type, 'use_errors':0, 'polaxis':polaxis, 'output_dim':output_dim, 
            'use_dm1':0, 'use_dm2':0, 'use_fpm':1, 'use_lyot_stop':1, 
            'slit':slit_name, 'slit_y_offset_mas':slit_pos_mas, 'prism':'PRISM3', 'wav_step_um':2E-3,
            'fsm_x_offset_mas':fsm_source_x_offset_mas, 'fsm_y_offset_mas':fsm_source_y_offset_mas}
    
        fsm_offset_3F_optics = instrument.CorgiOptics(cgi_mode, bandpass='3F', optics_keywords=offset_star_optics_keywords,
                                                      if_quiet=True, oversampling_factor = overfac, return_oversample = False)
        fsm_offset_3D_optics = instrument.CorgiOptics(cgi_mode, bandpass='3D', optics_keywords=offset_star_optics_keywords,
                                                      if_quiet=True, oversampling_factor = overfac, return_oversample = False)
    
        fsm_offset_3F_optics_list.append(fsm_offset_3F_optics)
        fsm_offset_3D_optics_list.append(fsm_offset_3D_optics)
        fsm_xoff_mas_list.append(fsm_source_x_offset_mas)
        fsm_yoff_mas_list.append(fsm_source_y_offset_mas)
    
    for ii, (fsm_xoff_mas, fsm_yoff_mas, optics_3F, optics_3D) in enumerate(zip(fsm_xoff_mas_list,
                                                                                fsm_yoff_mas_list, 
                                                                                fsm_offset_3F_optics_list, 
                                                                                fsm_offset_3D_optics_list)):
        print("------------------------------")
        print("Source delta x, y = {:+.1f}, {:+.1f} mas".format(optics_3F.optics_keywords['fsm_x_offset_mas'], 
                                                                      optics_3F.optics_keywords['fsm_y_offset_mas']))
        noiseless_3F_fname = os.path.join(outdir, 'standstar_{:s}_PRISM3_template_fsam{:s}_cfam3F_fsm_offset_{:02}.fits'.format(sptype, slit_name, ii))
        noiseless_3D_fname = os.path.join(outdir, 'standstar_{:s}_PRISM3_template_fsam{:s}_cfam3D_fsm_offset_{:02}.fits'.format(sptype, slit_name, ii))
        # crop_frame_3F_fname = os.path.join(outdir, 'standstar_{:s}_PRISM3_noise_fsam{:s}_cfam3F_fsm_offset_{:02}.fits'.format(sptype, slit_name, ii))
        # crop_frame_3D_fname = os.path.join(outdir, 'standstar_{:s}_PRISM3_noise_fsam{:s}_cfam3D_fsm_offset_{:02}.fits'.format(sptype, slit_name, ii))
        # full_frame_3F_fname = os.path.join(outdir, 'spec_reg_fsm_offset_template_{:02}_cfam3F_full.fits'.format(offset_mas))
        # full_frame_3D_fname = os.path.join(outdir, 'spec_reg_fsm_offset_template_{:02}_cfam3D_full.fits'.format(offset_mas))

        print("Evaluating FSM dither, filter 3D image...") 
        sim_cfam3D = optics_3D.get_host_star_psf(base_scene)
        sim_hdr_comment_3D = sim_cfam3D.host_star_image.header['COMMENT']
        for card in sim_hdr_comment_3D:
            if 'dispersed_image_centx' in card:
                centx_cropframe_3D = float(card.split(' : ')[1].strip())
            elif 'dispersed_image_centy' in card:
                centy_cropframe_3D = float(card.split(' : ')[1].strip())
            elif 'dispersed_fullframe_centx' in card:
                centx_fullframe_3D = float(card.split(' : ')[1].strip())
            elif 'dispersed_fullframe_centy' in card:
                centy_fullframe_3D = float(card.split(' : ')[1].strip())

        # noiseless template image
        noiseless_image_3D = sim_cfam3D.host_star_image.data
        noiseless_image_hdr_3D = sim_cfam3D.host_star_image.header
        noiseless_image_hdr_3D['slit_dy'] = (slit_pos_mas, 'FSAM slit vertical offset from FPM, in mas')
        noiseless_image_hdr_3D['fsm_xoff'] = (fsm_xoff_mas, 'FSM source horizontal offset in mas')
        noiseless_image_hdr_3D['fsm_yoff'] = (fsm_yoff_mas, 'FSM source vertical offset in mas')
        noiseless_image_hdr_3D['centx'] = (centx_cropframe_3D, 'x centroid of dispersed source image at filter center wavelength')
        noiseless_image_hdr_3D['centy'] = (centy_cropframe_3D, 'y centroid of dispersed source image at filter center wavelength')
        noiseless_image_hdr_3D['wv0_x'] = (centx_cropframe_3D, 'x centroid of dispersed source image at zeropoint wavelength')
        noiseless_image_hdr_3D['wv0_y'] = (centy_cropframe_3D, 'y centroid of dispersed source image at zeropoint wavelength')
        fits.writeto(noiseless_3D_fname, noiseless_image_3D, header=noiseless_image_hdr_3D, overwrite=True)

        # noisy image
        # detector.generate_detector_image(sim_cfam3D, short_exptime, full_frame=False)
        # sim_cfam3D.image_on_detector.writeto(crop_frame_3D_fname, overwrite=True)

        # detector_unitygain.generate_detector_image(sim_cfam3D, short_exptime, full_frame=True, loc_x=512, loc_y=512)
        # outputs.save_hdu_to_fits(sim_cfam3D.image_on_detector, outdir=outdir, write_as_L1=True)

        print("Evaluating FSM dither, filter 3F image...") 
        sim_cfam3F = optics_3F.get_host_star_psf(base_scene)
        sim_hdr_comment_3F = sim_cfam3F.host_star_image.header['COMMENT']
        for card in sim_hdr_comment_3F:
            if 'dispersed_image_centx' in card:
                centx_cropframe_3F = float(card.split(' : ')[1].strip())
            elif 'dispersed_image_centy' in card:
                centy_cropframe_3F = float(card.split(' : ')[1].strip())
            elif 'dispersed_fullframe_centx' in card:
                centx_fullframe_3F = float(card.split(' : ')[1].strip())
            elif 'dispersed_fullframe_centy' in card:
                centy_fullframe_3F = float(card.split(' : ')[1].strip())

        # noiseless template image
        noiseless_image_3F = sim_cfam3F.host_star_image.data
        noiseless_image_hdr_3F = sim_cfam3F.host_star_image.header
        noiseless_image_hdr_3F['slit_dy'] = (slit_pos_mas, 'FSAM slit vertical offset from FPM, in mas')
        noiseless_image_hdr_3F['fsm_xoff'] = (fsm_xoff_mas, 'FSM source horizontal offset in mas')
        noiseless_image_hdr_3F['fsm_yoff'] = (fsm_yoff_mas, 'FSM source vertical offset in mas')
        noiseless_image_hdr_3F['centx'] = (centx_cropframe_3F, 'x centroid of dispersed source image at filter center wavelength')
        noiseless_image_hdr_3F['centy'] = (centy_cropframe_3F, 'y centroid of dispersed source image at filter center wavelength')
        noiseless_image_hdr_3F['wv0_x'] = (centx_cropframe_3D, 'x centroid of dispersed source image at zeropoint wavelength')
        noiseless_image_hdr_3F['wv0_y'] = (centy_cropframe_3D, 'y centroid of dispersed source image at zeropoint wavelength')
        fits.writeto(noiseless_3F_fname, noiseless_image_3F, header=noiseless_image_hdr_3F, overwrite=True)

        # noisy image 
        # detector.generate_detector_image(sim_cfam3F, short_exptime, full_frame=False)
        # sim_cfam3F.image_on_detector.writeto(crop_frame_3F_fname, overwrite=True)

        detector.generate_detector_image(sim_cfam3F, short_exptime, full_frame=True, loc_x=loc_x, loc_y=loc_y)
        outputs.save_hdu_to_fits(sim_cfam3F.image_on_detector, outdir=outdir, write_as_L1=True)
        L1_fitsname = os.path.join(outdir, sim_cfam3F.image_on_detector[0].header['FILENAME'])
        L1_fits_files.append(L1_fitsname)
        png_fname = write_png_from_sceneobj(sim_cfam3F, outdir, loc_x, loc_y, output_dim)

    return L1_fits_files