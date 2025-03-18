import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from corgidrp import data 
from pyklip.parallelized import klip_dataset
from scipy.ndimage import shift
from corgidrp.astrom import get_polar_dist, seppa2dxdy
from corgidrp.fluxcal import phot_by_gauss2d_fit

def get_closest_psf(ct_calibration,cenx,ceny,dx,dy):
    x_arr = ct_calibration.ct_excam[0]
    y_arr = ct_calibration.ct_excam[1]

    rel_x_arr = x_arr - cenx
    rel_y_arr = y_arr - ceny

    dists = np.sqrt((rel_x_arr-dx)**2 + (rel_y_arr-dy))

    arg_closest = np.argmin(dists)

    return ct_calibration.data[arg_closest] 

def inject_psf(frame, ct_calibration, flux, sep_pix,pa_deg):

    frame_roll = frame.ext_hdr['ROLL']
    rel_pa = frame_roll - pa_deg

    dx,dy = seppa2dxdy(sep_pix,rel_pa)

    psf_model = get_closest_psf(ct_calibration,dx,dy) * flux 
    total_counts = np.sum(psf_model)
    
    psf_cenyx = np.array(psf_model.shape)/2 - 0.5 # Assume PSF is centered in the data cutout for now
    psf_center_inframe = np.array([frame.ext_hdr['STARLOCY'],frame.ext_hdr['STARLOCX']]) + np.array([dy,dx])

    # Insert into correct frame size array 
    psf_only_frame = np.zeros_like(frame.data)
    starty, endy = (int(frame.ext_hdr['STARLOCY']),int(frame.ext_hdr['STARLOCY'])+psf_model.shape[0])
    startx, endx = (int(frame.ext_hdr['STARLOCX']),int(frame.ext_hdr['STARLOCX'])+psf_model.shape[1])
    psf_only_frame[starty:endy,startx:endx] = psf_model
    injected_psf_center = psf_cenyx + np.array([starty,startx]).astype(float)
    psf_shift = injected_psf_center - psf_center_inframe

    shifted_psf_only_frame = shift(psf_only_frame,psf_shift)

    # Add to input frame
    outframe = frame.data + shifted_psf_only_frame

    return outframe, psf_model, psf_center_inframe, total_counts

def measure_noise(frame, sep_pix, fwhm):
    """Calculates the noise (standard deviation of counts) for an 
        annulus at a given separation
        TODO: Correct for small sample statistics?
    
    Args:
        frame (corgidrp.Image): Image containing data as well as "MASKLOCX/Y" in header
        sep_pix (float): Separation (in pixels from mask center) at which to calculate the noise level
        fwhm (float): halfwidth of the annulus to use for noise calculation, based on FWHM.

    Returns:
        float: noise level at the specified separation
    """

    cenx, ceny = (frame.ext_hdr['MASKLOCX'],frame.ext_hdr['MASKLOCY'])

    # Mask data outside the specified annulus
    y, x = np.indices(frame.data.shape)
    sep_map = np.sqrt((y-ceny)**2 + (x-cenx)**2)
    r_inner = sep_pix - fwhm
    r_outer = sep_pix + fwhm
    masked_data = np.where(sep_map<r_outer and sep_map>r_inner, frame.data,np.nan)
    
    # Calculate standard deviation
    std = np.nanstd(masked_data)

    return std

def meas_klip_thrupt(sci_dataset_in,ref_dataset_in, # pre-psf-subtracted dataset
                     psfsub_dataset,
                     ct_calibration,
                     klip_params,
                     inject_snr = 5,
                     seps=None, # in pixels from mask center
                     pas=None,
                     cand_locs = [] # list of (sep_pix,pa_deg) of known off axis source locations
                     ):
    
    res_elem = 5. # pix, update this with value for NFOV, Band 1 mode
    iwa = 10. # pix, update real number
    owa = 25. # pix, update with real number

    if pas == None:
        pas = np.array([0.,60.,120.])
    if seps == None:
        seps = np.arange(iwa,owa,res_elem) # Some linear spacing between the IWA & OWA, around 2x the resolution element

    sci_dataset = sci_dataset_in.copy()
    ref_dataset = ref_dataset_in.copy()

    thrupts = []
    for k,klmode in enumerate(klip_params['numbasis']):
        
        # Inject planets:
        seppas_skipped = []
        this_klmode_injectcounts = []
        for i,frame in enumerate(sci_dataset):
            for sep in seps:

                # Measure noise at this separation in psf subtracted dataset (for this kl mode)
                noise = measure_noise(psfsub_dataset[0].data[k],sep)

                inject_counts = noise * inject_snr
                for pa in pas:
                    inject_loc = (sep,pa)
                    # Check that we're not too close to a candidate
                    for cand_loc in cand_locs:
                        dist = get_polar_dist(cand_loc,inject_loc)
                        if dist < res_elem:
                            seppas_skipped.append(inject_loc)
                            continue
                    
                    frame, psf_model, psf_center_inframe, total_counts = inject_psf(frame, ct_calibration, inject_counts, inject_loc)
        
                # Save this to divide later
                this_klmode_injectcounts.append(inject_counts)
                
            sci_dataset[i].data = frame
                    
        # Init pyklip dataset
        pyklip_dataset = data.PyKLIPDataset(sci_dataset,psflib_dataset=ref_dataset)
        
        # Run pyklip
        klip_dataset(pyklip_dataset, outputdir=klip_params['outdir'],
                                annuli=klip_params['annuli'], subsections=klip_params['subsections'], 
                                movement=klip_params['movement'], 
                                numbasis=[klmode],
                                calibrate_flux=klip_params['calibrate_flux'], mode=klip_params['mode'],
                                psf_library=pyklip_dataset._psflib,
                                fileprefix=f"FAKE_{klmode}KLMODES")
        
        # Get photometry of each injected source

        this_klmode_outcounts = []
        for sep in seps:
            this_sep_outcounts = []

            for pa in pas:
                loc = (sep,pa)
                if loc in seppas_skipped:
                    continue

                this_sep_outcounts.append(phot_by_gauss2d_fit(frame,loc))

            if len(this_sep_outcounts) == 0:
                raise Warning(f'No flux measurements at separation {sep} pixels.')
            this_klmode_outcounts.append(np.mean(this_sep_outcounts))
    
        this_klmode_thrupts = np.array(this_klmode_outcounts/this_klmode_injectcounts)

        thrupts.append(this_klmode_thrupts)

    thrupt_arr = np.array([seps,*thrupts])


