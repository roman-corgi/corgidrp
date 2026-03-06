import roman_preflight_proper
import os
import pandas as pd
from specL1sims_cal_prism_images import *
from specL1sims_cal_DMsatspot import *
from specL1sims_cal_LSF import *
from specL1sims_science_occ import *
from specL1sims_utils import *

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
    #%%
    # Reference star prism images
    print("********************************************************")
    print("Slitless prism images on reference star")
    exptime = 8.0
    gain = 1
    outdir_ref_slitless_prism = os.path.join(data_outdir, 'SPEC-NOM_refstar_slitless_prism')
    if not os.path.exists(outdir_ref_slitless_prism):
        os.mkdir(outdir_ref_slitless_prism)
    L1_files_ref_slitless_prism = create_star_slitless_prism_sims(outdir_ref_slitless_prism, Vmag_ref, sptype_ref, slit_pos_mas, gain, exptime, ref_flag=True, nd=1)
    update_fits_headers(L1_files_ref_slitless_prism, [('VISTYPE', 'CGIVST_CAL_SPEC_TGTREF')])
    write_headers_to_text(L1_files_ref_slitless_prism)
    print("********************************************************")
    print("Slit + prism images on reference star")
    slit_fsm_offsets_mas = [-10, -5, 0, 5, 10] # Vertical FSM offsets relative to slit aperture
    outdir_ref_slit_prism = os.path.join(data_outdir, 'SPEC-NOM_refstar_slit_prism')
    if not os.path.exists(outdir_ref_slit_prism):
        os.mkdir(outdir_ref_slit_prism)
    L1_files_ref_slit_prism = create_star_slit_prism_sims(outdir_ref_slit_prism, Vmag_ref, sptype_ref, slit_name, slit_pos_mas, slit_fsm_offsets_mas, gain, exptime, ref_flag=True, nd=1)
    update_fits_headers(L1_files_ref_slit_prism, [('VISTYPE', 'CGIVST_CAL_SPEC_TGTREF')])
    write_headers_to_text(L1_files_ref_slit_prism)
    #%%
    # Target star prism images
    print("********************************************************")
    print("Slitless prism images on target star")
    exptime = 1.0
    gain = 1
    outdir_target_slitless_prism = os.path.join(data_outdir, 'SPEC-NOM_targetstar_slitless_prism')
    if not os.path.exists(outdir_target_slitless_prism):
        os.mkdir(outdir_target_slitless_prism)
    L1_files_target_slitless_prism = create_star_slitless_prism_sims(outdir_target_slitless_prism, Vmag_target, sptype_target, slit_pos_mas, gain, exptime, cycle_subband_filters=False)
    update_fits_headers(L1_files_target_slitless_prism, [('VISTYPE', 'CGIVST_CAL_SPEC_TGTREF')])
    write_headers_to_text(L1_files_target_slitless_prism)
    print("********************************************************")
    print("Slit + prism images on target star")
    slit_fsm_offsets_mas = [-10, -5, 0, 5, 10] # Vertical FSM offsets relative to slit aperture
    outdir_target_slit_prism = os.path.join(data_outdir, 'SPEC-NOM_targetstar_slit_prism')
    if not os.path.exists(outdir_target_slit_prism):
        os.mkdir(outdir_target_slit_prism)
    L1_files_target_slit_prism = create_star_slit_prism_sims(outdir_target_slit_prism, Vmag_target, sptype_target, slit_name, slit_pos_mas, slit_fsm_offsets_mas, gain, exptime, nd=0)
    update_fits_headers(L1_files_target_slit_prism, [('VISTYPE', 'CGIVST_CAL_SPEC_TGTREF')])
    write_headers_to_text(L1_files_target_slit_prism)
    #%%
    # Line Spread Function visit with standard star 
    print("********************************************************")
    print("Line spread function and slit + prism images on spectrophotometric standard star")
    stdstar_fsm_offsets_mas = [-30, 0, 30]      # FSM offsets relative to the planet source location, 
                                                # in units of mas. Will be applied in a 2-D grid of 
                                                # horizontal and vertical offsets.
    exptime = 30
    gain = 30
    outdir_LSF = os.path.join(data_outdir, 'SPEC-NOM_standstar_linespreadfunc')
    if not os.path.exists(outdir_LSF):
        os.mkdir(outdir_LSF)
    L1_files_LSF = create_spec_LSF_sims(outdir_LSF, Vmag_stdstar, sptype_stdstar, slit_name, slit_pos_mas, stdstar_fsm_offsets_mas, gain, exptime)
    update_fits_headers(L1_files_LSF, [('VISTYPE', 'CGIVST_CAL_SPEC_LINESPREAD')])
    write_headers_to_text(L1_files_LSF)

    ### ND filter / absolute flux calibration with 3 faint standard stars and 1 bright standard star (CAR-160)
    ### No dedicated simulations for now; the reference star prism images taken with ND filter
    ### can be treated equivalently to bright standard star data.

    #%%
    # DM Satellite Spot image on reference star
    print("********************************************************")
    print("DM satellite spot slit + prism images on reference star")
    spot_contrast = 1E-5
    gain = 200
    exptime = 200

    outdir_ref_satspot = os.path.join(data_outdir, 'SPEC-NOM_refstar_satspot')
    if not os.path.exists(outdir_ref_satspot):
        os.mkdir(outdir_ref_satspot)
    L1_files_ref_satspot = create_spec_satspot_sim(outdir_ref_satspot, Vmag_ref, sptype_ref, slit_name, slit_pos_mas, 
                                                spot_contrast, gain, exptime, nframes=3)
    write_headers_to_text(L1_files_ref_satspot)
    # DM Satellite Spot image on target star
    print("********************************************************")
    print("DM satellite spot slit + prism images on target star")
    spot_contrast = 1E-5
    gain = 500
    exptime = 800
    outdir_target_satspot = os.path.join(data_outdir, 'SPEC-NOM_targetstar_satspot')
    if not os.path.exists(outdir_target_satspot):
        os.mkdir(outdir_target_satspot)
    L1_files_target_satspot = create_spec_satspot_sim(outdir_target_satspot, Vmag_target, sptype_target, slit_name, slit_pos_mas,
                                                   spot_contrast, gain, exptime, nframes=3)
    write_headers_to_text(L1_files_target_satspot)

    #%%
    # Reference star occulted science images, analog processing
    print("********************************************************")
    print("Analog science images on occulted reference star")
    gain_ref = 500
    exptime_ref = 800
    num_frames_ref = 5

    outdir_refstar_science = os.path.join(data_outdir, 'SPEC-NOM_refstar_science_analog')
    if not os.path.exists(outdir_refstar_science):
        os.mkdir(outdir_refstar_science)
    L1_files_refstar_science_analog = create_occ_refstar_sim(outdir_refstar_science, Vmag_ref, sptype_ref, slit_name, slit_pos_mas, 
                           gain_ref, exptime_ref, num_frames_ref, photon_counting=False)
    write_headers_to_text(L1_files_refstar_science_analog)

    # Target star occulted science images, analog processing
    print("********************************************************")
    print("Analog science images on occulted target star with companion source")
    gain_target = 500
    exptime_target = 2000
    num_frames_target = 5

    outdir_targetstar_science = os.path.join(data_outdir, 'SPEC-NOM_targetstar_science_analog')
    if not os.path.exists(outdir_targetstar_science):
        os.mkdir(outdir_targetstar_science)
    L1_files_targetstar_science_analog = create_occ_target_sim(outdir_targetstar_science, Vmag_target, sptype_target, slit_name, slit_pos_mas, 
                          comp_dra=0.0, comp_ddec=source_y_offset, comp_fluxratio=source_fluxratio,
                          gain=gain_target, exptime=exptime_target, nframes=num_frames_target, photon_counting=False)
    write_headers_to_text(L1_files_targetstar_science_analog)

    #%%
    # Reference star occulted science images, photon counting processing
    print("********************************************************")
    print("Photon-counting science images on occulted reference star")
    gain_ref = 5.0E4
    exptime_ref = 5
    num_frames_ref = 200

    outdir_refstar_science = os.path.join(data_outdir, 'SPEC-NOM_refstar_science_pc')
    if not os.path.exists(outdir_refstar_science):
        os.mkdir(outdir_refstar_science)
    L1_files_refstar_science_pc = create_occ_refstar_sim(outdir_refstar_science, Vmag_ref, sptype_ref, slit_name, slit_pos_mas, 
                           gain_ref, exptime_ref, num_frames_ref, photon_counting=True)
    write_headers_to_text(L1_files_refstar_science_pc)

    #%%
    # Target star occulted science images, photon counting processing
    print("********************************************************")
    print("Photon-counting science images on occulted target star with companion source")
    gain_target = 2.0E5
    exptime_target = 20
    num_frames_target = 200

    outdir_targetstar_science = os.path.join(data_outdir, 'SPEC-NOM_targetstar_science_pc')
    if not os.path.exists(outdir_targetstar_science):
        os.mkdir(outdir_targetstar_science)
    L1_files_targetstar_science_pc = create_occ_target_sim(outdir_targetstar_science, Vmag_target, sptype_target, slit_name, slit_pos_mas, 
                          comp_dra=0.0, comp_ddec=source_y_offset, comp_fluxratio=source_fluxratio,
                          gain=gain_target, exptime=exptime_target, nframes=num_frames_target, photon_counting=True)
    write_headers_to_text(L1_files_targetstar_science_pc)

    #%%
    # Take averages and counting threshold sums of the photon-counting science images
    print("********************************************************")
    print("Computing stack means and counting threshold sums of the photon-counting science images")
    outdir_proc_tests = os.path.join(data_outdir, 'SPEC-NOM_proc_tests')
    L1_mean_refstar_science_pc_filename = average_L1_images(L1_files_refstar_science_pc, outdir_proc_tests, ext=1)
    write_png_from_fits(L1_mean_refstar_science_pc_filename, loc_x=512, loc_y=512, output_dim=120, ext=1)
    L1_threshsum_refstar_science_pc_filename = threshold_sum_L1_images(L1_files_refstar_science_pc, outdir_proc_tests, ext=1)
    write_png_from_fits(L1_threshsum_refstar_science_pc_filename, loc_x=512, loc_y=512, output_dim=120, ext=1)
    print(f"Stack mean of photon-counting occulted reference star science images stored to {L1_mean_refstar_science_pc_filename}")
    print(f"Stack sum thresholded photon-counting occulted reference star science images stored to {L1_threshsum_refstar_science_pc_filename}")

    L1_mean_targetstar_science_pc_filename = average_L1_images(L1_files_targetstar_science_pc, outdir_proc_tests, ext=1)
    write_png_from_fits(L1_mean_targetstar_science_pc_filename, loc_x=512, loc_y=512, output_dim=120, ext=1)
    L1_threshsum_targetstar_science_pc_filename = threshold_sum_L1_images(L1_files_targetstar_science_pc, outdir_proc_tests, ext=1)
    write_png_from_fits(L1_threshsum_targetstar_science_pc_filename, loc_x=512, loc_y=512, output_dim=120, ext=1)
    print(f"Stack mean of photon-counting occulted target science images stored to {L1_mean_targetstar_science_pc_filename}")
    print(f"Stack sum thresholded photon-counting occulted reference star science images stored to {L1_threshsum_targetstar_science_pc_filename}")

    #%%
    # Build a summary table of all the L1 data files and store it in an Excel file
    L1_files_all = []

    table_header_keys = [(0, 'VISTYPE'), (1, 'FSMY'), (1, 'SPAMNAME'), (1, 'FPAMNAME'), (1, 'LSAMNAME'), 
                         (1, 'FSAMNAME'), (1, 'CFAMNAME'), (1, 'DPAMNAME'),
                         (1, 'EXPTIME'), (1, 'EMGAIN_C'), (0, 'PHTCNT')]

    # Reference star slitless prism
    if 'L1_files_ref_slitless_prism' in locals():
        for f in L1_files_ref_slitless_prism:
            L1_files_all.append(get_L1_config_dict(
                f, file_trunc=os.path.join('SPEC-NOM_refstar_slitless_prism', os.path.basename(f)),
                descrip='Reference star slitless prism', header_keys=table_header_keys
            ))
    # Reference star slit prism
    if 'L1_files_ref_slit_prism' in locals():
        for f in L1_files_ref_slit_prism:
            L1_files_all.append(get_L1_config_dict(
                f, file_trunc=os.path.join('SPEC-NOM_refstar_slit_prism', os.path.basename(f)),
                descrip='Reference star slit + prism', header_keys=table_header_keys
            ))
    # Target star slitless prism
    if 'L1_files_target_slitless_prism' in locals():
        for f in L1_files_target_slitless_prism:
            L1_files_all.append(get_L1_config_dict(
                f, file_trunc=os.path.join('SPEC-NOM_targetstar_slitless_prism', os.path.basename(f)),
                descrip='Target star slitless prism', header_keys=table_header_keys
            ))
    # Target star slit prism
    if 'L1_files_target_slit_prism' in locals():
        for f in L1_files_target_slit_prism:
            L1_files_all.append(get_L1_config_dict(
                f, file_trunc=os.path.join('SPEC-NOM_targetstar_slit_prism', os.path.basename(f)),
                descrip='Target star slit + prism', header_keys=table_header_keys
            ))
    # Line spread function
    if 'L1_files_LSF' in locals():
        for f in L1_files_LSF:
            L1_files_all.append(get_L1_config_dict(
                f, file_trunc=os.path.join('SPEC-NOM_standstar_linespreadfunc', os.path.basename(f)),
                descrip='Standard star line spread function', header_keys=table_header_keys
            ))
    # Reference star satellite spot
    if 'L1_files_ref_satspot' in locals():
        for f in L1_files_ref_satspot:
            L1_files_all.append(get_L1_config_dict(
                f, file_trunc=os.path.join('SPEC-NOM_refstar_satspot', os.path.basename(f)),
                descrip='Reference star DM satellite spot', header_keys=table_header_keys
            ))
    # Target star satellite spot
    if 'L1_files_target_satspot' in locals():
        for f in L1_files_target_satspot:
            L1_files_all.append(get_L1_config_dict(
                f, file_trunc=os.path.join('SPEC-NOM_targetstar_satspot', os.path.basename(f)),
                descrip='Target star DM satellite spot', header_keys=table_header_keys
            ))
    # Reference star science analog
    if 'L1_files_refstar_science_analog' in locals():
        for f in L1_files_refstar_science_analog:
            L1_files_all.append(get_L1_config_dict(
                f, file_trunc=os.path.join('SPEC-NOM_refstar_science_analog', os.path.basename(f)),
                descrip='Reference star occulted science - analog', header_keys=table_header_keys
            ))
    # Target star science analog
    if 'L1_files_targetstar_science_analog' in locals():
        for f in L1_files_targetstar_science_analog:
            L1_files_all.append(get_L1_config_dict(
                f, file_trunc=os.path.join('SPEC-NOM_targetstar_science_analog', os.path.basename(f)),
                descrip='Target star occulted science with companion - analog', header_keys=table_header_keys
            ))
    # Reference star science photon counting
    if 'L1_files_refstar_science_pc' in locals():
        for f in L1_files_refstar_science_pc:
            L1_files_all.append(get_L1_config_dict(
                f, file_trunc=os.path.join('SPEC-NOM_refstar_science_pc', os.path.basename(f)),
                descrip='Reference star occulted science - photon counting', header_keys=table_header_keys
            ))
    # Target star science photon counting
    if 'L1_files_targetstar_science_pc' in locals():
        for f in L1_files_targetstar_science_pc:
            L1_files_all.append(get_L1_config_dict(
                f, file_trunc=os.path.join('SPEC-NOM_targetstar_science_pc', os.path.basename(f)),
                descrip='Target star occulted science with companion - photon counting', header_keys=table_header_keys
            ))

    # Create DataFrame and save to Excel
    df = pd.DataFrame(L1_files_all)
    excel_file = os.path.join(data_outdir, 'SPEC-NOM_L1sims.xlsx')
    df.to_excel(excel_file, index=False)
    print("********************************************************")
    print(f"Summary table saved to {excel_file}")

# %%
