from corgisim import scene
from corgisim import instrument
import proper
from corgisim import outputs
import roman_preflight_proper

##Generate a set of L1 input frames for use with corgidrp

#global CGI settings
cgi_mode = 'excam'
cor_type = 'hlc'
bandpass = '1F'
rootname = 'hlc_ni_3e-8'
dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
polaxis = 10
output_dim = 55

#generate 6 batches of reference star data cubes and exports the data to the relevant folders
def gen_ref_star():
    #OS11 ref star properties
    Vmag = 2.25                           
    sptype = 'O4I'                     
    ref_flag = True                    
    host_star_properties = {'Vmag': Vmag,
                            'spectral_type': sptype,
                            'magtype': 'vegamag',
                            'ref_flag': ref_flag}

    # This creates the star object
    base_scene = scene.Scene(host_star_properties)

    batch_n = 100

    #actual OS11 frames per batch
    #frame_n = [1147, 991, 991, 1147, 991, 991]

    #frames per batch used for testing
    frame_n = 1

    roman_preflight_proper.copy_here()
    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':polaxis, 'output_dim':output_dim,\
                'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }

    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, roll_angle=0, if_quiet=True)

    # This gets the host star PSF viewed through the optics, noise free
    sim_scene = optics.get_host_star_psf(base_scene)

    #OS11 detector settings
    gain = 5000
    cr_rate = 0
    emccd_keywords ={'em_gain':gain, 'cr_rate':cr_rate}
    detector = instrument.CorgiDetector(emccd_keywords)

    #actual OS11 exposure time for reference star
    #exptime = 2

    #exposure time used for testing
    exptime = 0.1

    #generate L1 files and save
    for i in range(batch_n):
        for j in range(frame_n):
            sim_scene = detector.generate_detector_image(sim_scene, exptime, full_frame=True, loc_x=512, loc_y=512)
            outdir = f'./L1_input_data/ref_star/batch_{i}' 
            outputs.save_hdu_to_fits(sim_scene.image_on_detector, outdir = outdir, write_as_L1=True) 

#generate 8 batches of target star data cubes at specified roll angle and exports the data to the relevant folders
def gen_target_star(roll_angle):
    #target star properties
    Vmag = 5                            
    sptype = 'G1V'                      
    ref_flag = False                    
    host_star_properties = {'Vmag': Vmag,
                            'spectral_type': sptype,
                            'magtype': 'vegamag',
                            'ref_flag': ref_flag}
    base_scene = scene.Scene(host_star_properties)

    batch_n = 100

    #actual OS11 frames per batch
    #frame_n = 194

    #frames per batch used for testing
    frame_n = 1

    roman_preflight_proper.copy_here()
    optics_keywords ={'cor_type':cor_type, 'use_errors':2, 'polaxis':polaxis, 'output_dim':output_dim,\
                'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2,'use_fpm':1, 'use_lyot_stop':1,  'use_field_stop':1 }
    optics = instrument.CorgiOptics(cgi_mode, bandpass, optics_keywords=optics_keywords, roll_angle=roll_angle, if_quiet=True)

    sim_scene = optics.get_host_star_psf(base_scene)

    #OS11 detector settings
    gain = 5000
    cr_rate = 0
    emccd_keywords ={'em_gain':gain, 'cr_rate':cr_rate}
    detector = instrument.CorgiDetector(emccd_keywords)

    #actual OS11 exposure time for target star
    #exptime = 30

    #exposure time used for testing
    exptime = 0.1

    #generate L1 files and save
    for i in range(batch_n):
        for j in range(frame_n):
            sim_scene = detector.generate_detector_image(sim_scene, exptime, full_frame=True, loc_x=512, loc_y=512)
            outdir = f'./L1_input_data/target_star/roll_{roll_angle}/batch_{i}' 
            outputs.save_hdu_to_fits(sim_scene.image_on_detector, outdir = outdir, write_as_L1=True) 

if __name__ == '__main__':
    gen_ref_star()
    gen_target_star(-13)
    gen_target_star(13)