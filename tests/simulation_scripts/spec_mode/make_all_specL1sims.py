import roman_preflight_proper
import os
from specL1sims_cal_prism_images import *
from specL1sims_cal_DMsatspot import *
from specL1sims_cal_LSF import *
from specL1sims_science_ref import *
from specL1sims_science_target import *

if __name__ == '__main__':
    data_outdir = os.path.expanduser('~/RomanCGI_E2Etest_L1_data')
    if not os.path.exists(data_outdir):
        os.mkdir(data_outdir)

    Vmag_target = 5.0                     # V-band magnitude of the target star
    sptype_target = 'A0V'                 # Spectral type of the target star
    Vmag_ref = 1.86                       # V-band magnitude of the reference/bright standard star (eta UMa)
    sptype_ref = 'B3V'                    # Spectral type of the reference/bright standard star (eta UMa)
    Vmag_stdstar = 12.0                   # V-band magnitude of faint standard star
    sptype_stdstar = 'G0V'                # Spectral type of faint standard star

    source_fluxratio = 2E-7               # Target source flux ratio
    source_y_offset = 320.0               # Source vertical (ddec) offset in millarcsec 
    slit_name = 'R1C2'                    # FSAM slit named position
    slit_pos_mas = source_y_offset        # Vertical slit offset in units of millarcsec

    ### Copy the Proper prescription file 
    roman_preflight_proper.copy_here()

    # Reference star prism images
    print("********************************************************")
    print("Slitless prism images on reference star")
    exptime = 8.0
    gain = 1
    outdir_ref_slitless_prism = os.path.join(data_outdir, 'SPEC_refstar_slitless_prism')
    if not os.path.exists(outdir_ref_slitless_prism):
        os.mkdir(outdir_ref_slitless_prism)
    create_star_slitless_prism_sims(outdir_ref_slitless_prism, Vmag_ref, sptype_ref, slit_pos_mas, gain, exptime, ref_flag=True, nd=1)
    print("********************************************************")
    print("Slit + prism images on reference star")
    slit_fsm_offsets_mas = [-10, -5, 0, 5, 10] # Vertical FSM offsets relative to slit aperture
    outdir_ref_slit_prism = os.path.join(data_outdir, 'SPEC_refstar_slit_prism')
    if not os.path.exists(outdir_ref_slit_prism):
        os.mkdir(outdir_ref_slit_prism)
    create_star_slit_prism_sims(outdir_ref_slit_prism, Vmag_ref, sptype_ref, slit_name, slit_pos_mas, slit_fsm_offsets_mas, gain, exptime, ref_flag=True, nd=1)

    # Target star prism images
    print("********************************************************")
    print("Slitless prism images on target star")
    exptime = 1.0
    gain = 1
    outdir_target_slitless_prism = os.path.join(data_outdir, 'SPEC_targetstar_slitless_prism')
    if not os.path.exists(outdir_target_slitless_prism):
        os.mkdir(outdir_target_slitless_prism)
    create_star_slitless_prism_sims(outdir_target_slitless_prism, Vmag_target, sptype_target, slit_pos_mas, gain, exptime, cycle_subband_filters=False)
    print("********************************************************")
    print("Slit + prism images on target star")
    slit_fsm_offsets_mas = [-10, -5, 0, 5, 10] # Vertical FSM offsets relative to slit aperture
    outdir_target_slit_prism = os.path.join(data_outdir, 'SPEC_targetstar_slit_prism')
    if not os.path.exists(outdir_target_slit_prism):
        os.mkdir(outdir_target_slit_prism)
    create_star_slit_prism_sims(outdir_target_slit_prism, Vmag_target, sptype_target, slit_name, slit_pos_mas, slit_fsm_offsets_mas, gain, exptime, nd=0)

    # Line Spread Function visit with standard star 
    print("********************************************************")
    print("Line spread function and slit + prism images on spectrophotometric standard star")
    stdstar_fsm_offsets_mas = [-30, 0, 30]      # FSM offsets relative to the planet source location, 
                                                # in units of mas. Will be applied in a 2-D grid of 
                                                # horizontal and vertical offsets.
    exptime = 30
    gain = 30
    outdir_LSF = os.path.join(data_outdir, 'SPEC_standstar_linespreadfunc')
    if not os.path.exists(outdir_LSF):
        os.mkdir(outdir_LSF)
    create_spec_LSF_sims(outdir_LSF, Vmag_stdstar, sptype_stdstar, slit_name, slit_pos_mas, stdstar_fsm_offsets_mas, gain, exptime)

    ### ND filter / absolute flux calibration with 3 faint standard stars and 1 bright standard star (CAR-160)
    ### No dedicated simulations for now; the reference star prism images taken with ND filter
    ### can be treated equivalently to bright standard star data.

    # DM Satellite Spot image on reference star
    print("********************************************************")
    print("DM satellite spot slit + prism images on reference star")
    spot_contrast = 1E-5
    gain = 200
    exptime = 200

    outdir_ref_satspot = os.path.join(data_outdir, 'SPEC_refstar_satspot')
    if not os.path.exists(outdir_ref_satspot):
        os.mkdir(outdir_ref_satspot)
    ref_satspot_image = create_spec_satspot_sim(outdir_ref_satspot, Vmag_ref, sptype_ref, slit_name, slit_pos_mas, 
                                                spot_contrast, gain, exptime, nframes=3)
    # DM Satellite Spot image on target star
    print("********************************************************")
    print("DM satellite spot slit + prism images on target star")
    spot_contrast = 1E-5
    gain = 500
    exptime = 800
    outdir_target_satspot = os.path.join(data_outdir, 'SPEC_targetstar_satspot')
    if not os.path.exists(outdir_target_satspot):
        os.mkdir(outdir_target_satspot)
    target_satspot_image = create_spec_satspot_sim(outdir_target_satspot, Vmag_target, sptype_target, slit_name, slit_pos_mas,
                                                   spot_contrast, gain, exptime, nframes=3)

    # Reference star occulted science images
    print("********************************************************")
    print("Science slit + prism images on occulted reference star")
    gain_ref = 500
    exptime_ref = 800
    num_frames_ref = 10

    outdir_refstar_science = os.path.join(data_outdir, 'SPEC_refstar_science')
    if not os.path.exists(outdir_refstar_science):
        os.mkdir(outdir_refstar_science)
    create_occ_refstar_sim(outdir_refstar_science, Vmag_ref, sptype_ref, slit_name, slit_pos_mas, 
                           gain_ref, exptime_ref, num_frames_ref)

    # Target star occulted science images
    print("********************************************************")
    print("Science slit + prism images on occulted target star with companion source")
    gain_target = 500
    exptime_target = 2000
    num_frames_target = 10

    outdir_targetstar_science = os.path.join(data_outdir, 'SPEC_targetstar_science')
    if not os.path.exists(outdir_targetstar_science):
        os.mkdir(outdir_targetstar_science)
    create_occ_target_sim(outdir_targetstar_science, Vmag_target, sptype_target, slit_name, slit_pos_mas, 
                          comp_dra=0.0, comp_ddec=source_y_offset, comp_fluxratio=source_fluxratio,
                          gain=gain_target, exptime=exptime_target, nframes=num_frames_target)