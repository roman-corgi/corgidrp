import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from corgidrp.data import PyKLIPDataset, Image 
from pyklip.parallelized import klip_dataset
from scipy.ndimage import shift
from corgidrp.astrom import get_polar_dist, seppa2dxdy, seppa2xy
from corgidrp.fluxcal import phot_by_gauss2d_fit

def get_closest_psf(ct_calibration,cenx,ceny,dx,dy):
    """_summary_
    NOTE: CT excam locations have (0,0) as the bottom left corner 
      of the bottom left pixel

    TODO: Calculate subpixel shifts if star or PSF model aren't
        perfectly in the center of a pixel

    Args:
        ct_calibration (_type_): _description_
        cenx (_type_): _description_
        ceny (_type_): _description_
        dx (_type_): _description_
        dy (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Shift so (0,0) is the center of the bottom left pixel
    x_arr = ct_calibration.ct_excam[0] - 0.5
    y_arr = ct_calibration.ct_excam[1] - 0.5

    rel_x_arr = x_arr - cenx
    rel_y_arr = y_arr - ceny

    dists = np.sqrt((rel_x_arr-dx)**2 + (rel_y_arr-dy)**2)

    arg_closest = np.argmin(dists)

    return ct_calibration.data[arg_closest] 


def inject_psf(frame_in, ct_calibration, amp, 
               sep_pix,pa_deg,
               norm='sum'):

    frame = frame_in.copy()

    # Get closest psf model
    frame_roll = frame.ext_hdr['ROLL']
    rel_pa = frame_roll - pa_deg
    dx,dy = seppa2dxdy(sep_pix,rel_pa)

    psf_model = get_closest_psf(ct_calibration,
                                frame.ext_hdr['STARLOCX'],
                                frame.ext_hdr['STARLOCY'],
                                dx,dy) 
    
    # Scale counts
    if norm == 'sum':
        total_counts = np.nansum(psf_model)
        psf_model *= amp / total_counts
    elif norm == 'peak':
        peak_count = np.nanmax(psf_model)
        psf_model *= amp / peak_count
    else:
        raise UserWarning('Invalid norm provided to inject_psf().')


    # Assume PSF is centered in the data cutout for now
    shape_arr = np.array(psf_model.shape)
    psf_cenyx_ind = (np.array(shape_arr)/2 - 0.5).astype(int) 
    psf_cenyx_inframe = np.array([frame.ext_hdr['STARLOCY'],frame.ext_hdr['STARLOCX']]) + np.array([dy,dx])
    injected_psf_cenyx_ind = np.round(psf_cenyx_inframe).astype(int)
    starty, startx = injected_psf_cenyx_ind - psf_cenyx_ind

    # Insert into correct frame size array 
    psf_only_frame = np.zeros_like(frame.data)
    psf_only_frame[starty:starty+shape_arr[0],startx:startx+shape_arr[1]] = psf_model
    
    # TODO: Calculate subpixel shift:
    psf_shift = (0,0) # Hardcode 0 shift for now
    shifted_psf_only_frame = shift(psf_only_frame,psf_shift)

    # Add to input frame
    frame.data += shifted_psf_only_frame

    psf_cenxy = [psf_cenyx_inframe[1],psf_cenyx_inframe[0]]
    return frame, psf_model, psf_cenxy


def measure_noise(frame, seps_pix, fwhm, klmode_index=None):
    """Calculates the noise (standard deviation of counts) for an 
        annulus at a given separation
        TODO: Correct for small sample statistics?
    
    Args:
        frame (corgidrp.Image): Image containing data as well as "MASKLOCX/Y" in header
        seps_pix (float): Separations (in pixels from mask center) at which to calculate the noise level
        fwhm (float): halfwidth of the annulus to use for noise calculation, based on FWHM.

    Returns:
        float: noise level at the specified separation
    """

    cenx, ceny = (frame.ext_hdr['MASKLOCX'],frame.ext_hdr['MASKLOCY'])

    # Mask data outside the specified annulus
    y, x = np.indices(frame.data.shape[1:])
    sep_map = np.sqrt((y-ceny)**2 + (x-cenx)**2)
    sep_map3d = np.ones_like(frame.data) * sep_map

    stds = []
    for sep_pix in seps_pix:
        r_inner = sep_pix - fwhm
        r_outer = sep_pix + fwhm
        masked_data = np.where(np.logical_and(sep_map3d<r_outer,sep_map3d>r_inner), frame.data,np.nan)
        
        # Calculate standard deviation
        std = np.nanstd(masked_data,axis=(1,2))
        stds.append(std)

    stds_arr = np.array(stds)

    if klmode_index != None:
        return stds_arr[:,klmode_index]
    
    return stds_arr


def meas_klip_thrupt(sci_dataset_in,ref_dataset_in, # pre-psf-subtracted dataset
                     psfsub_dataset,
                     ct_calibration,
                     klip_params,
                     inject_snr = 5,
                     seps=None, # in pixels from mask center
                     pas=None,
                     cand_locs = [] # list of tuples (sep_pix,pa_deg) of known off-axis source locations
                     ):
    
    iwa = 10. # pix, update real number
    owa = 50. # pix, update with real number

    d = 2.36 #m
    lam = 573.8e-9 #m
    pixscale_arcsec = 0.0218
    fwhm_mas = 1.22 * lam / d * 206265 * 1000
    fwhm_pix = fwhm_mas * 0.001 / pixscale_arcsec
    res_elem = 5 * fwhm_pix # pix, update this with value for NFOV, Band 1 mode
    
    if pas == None:
        pas = np.array([0.,60.,120.])
    if seps == None:
        seps = np.arange(iwa,owa,res_elem) # Some linear spacing between the IWA & OWA, around 5x the fwhm

    sci_dataset = sci_dataset_in.copy()
    ref_dataset = ref_dataset_in.copy() if not ref_dataset_in is None else None

    rolls = [frame.ext_hdr['ROLL'] for frame in sci_dataset]

    thrupts = []
    for k,klmode in enumerate(klip_params['numbasis']):
        
        # Inject planets:
        seppas_skipped = []
        this_klmode_seppas = []
        this_klmode_psfmodels = []
        for i,frame in enumerate(sci_dataset):

            for sep in seps:

                # Measure noise at this separation in psf subtracted dataset (for this kl mode)
                
                noise = measure_noise(psfsub_dataset[0],[sep],fwhm_pix,k)[0]
                
                inject_peak = noise * inject_snr

                for pa in pas:
                    inject_loc = (sep,pa)
                    # Check that we're not too close to a candidate during the first frame
                    if i==0:
                        too_close = False
                        for cand_sep, cand_pa in cand_locs:
                            # Account for telescope roll angles, skip if any are too close
                            for roll in rolls:
                                cand_pa_adj = cand_pa - roll
                                dist = get_polar_dist((cand_sep,cand_pa_adj),inject_loc)
                                if dist < res_elem:
                                    too_close=True
                                    break
                            if too_close:
                                break
                        if too_close:
                            seppas_skipped.append(inject_loc)
                            continue
                    
                    frame, psf_model, _ = inject_psf(frame, ct_calibration, 
                                                    inject_peak, *inject_loc,
                                                    norm='peak')

                    # Save these for later (only for first sci frame)
                    if i==0:
                        this_klmode_psfmodels.append(psf_model)
                        this_klmode_seppas.append(inject_loc)

            sci_dataset[i].data[:] = frame.data[:]
                    
        # Init pyklip dataset
        pyklip_dataset = PyKLIPDataset(sci_dataset,psflib_dataset=ref_dataset)
        
        # Run pyklip
        klip_dataset(pyklip_dataset, outputdir=klip_params['outdir'],
                                annuli=klip_params['annuli'], subsections=klip_params['subsections'], 
                                movement=klip_params['movement'], 
                                numbasis=[klmode],
                                calibrate_flux=klip_params['calibrate_flux'], mode=klip_params['mode'],
                                psf_library=pyklip_dataset._psflib,
                                fileprefix=f"FAKE_{klmode}KLMODES")
        
        # Get photometry of each injected source

        # Load pyklip output
        pyklip_fpath = os.path.join(klip_params['outdir'],f"FAKE_{klmode}KLMODES-KLmodes-all.fits")
        pyklip_data = fits.getdata(pyklip_fpath)[0]
        pyklip_hdr = fits.getheader(pyklip_fpath)

        # import matplotlib.pyplot as plt
        # plt.imshow(pyklip_data,origin='lower')
        
        this_klmode_thrupts = []
        for ll,loc in enumerate(this_klmode_seppas):
            
            psf_model = this_klmode_psfmodels[ll]
            
            # Crop data around location to be same as psf_model cutout
            locxy = seppa2xy(*loc,pyklip_hdr['PSFCENTX'],pyklip_hdr['PSFCENTY'])

            # Crop the data
            cutout_shape = np.array(psf_model.shape)
            start_ind = (locxy - cutout_shape//2).astype(int)
            end_ind = (locxy + cutout_shape//2 + 1).astype(int)
            x1,y1 = start_ind
            x2,y2 = end_ind
            data_cutout = pyklip_data[y1:y2,x1:x2]

            # Temporarily load into Image obj
            model_img = Image(psf_model,pri_hdr=fits.Header(),ext_hdr=fits.Header())
            data_img = Image(data_cutout,pri_hdr=fits.Header(),ext_hdr=fits.Header())

            preklip_amp, preklip_err, preklip_bg = phot_by_gauss2d_fit(model_img,fwhm_pix,background_sub=True,fit_shape=psf_model.shape)
            try:
                postklip_amp, postklip_err, postklip_bg = phot_by_gauss2d_fit(data_img,fwhm_pix,background_sub=True,fit_shape=data_cutout.shape)
            except:
                postklip_amp = np.nan
            thrupt = postklip_amp/preklip_amp

            this_klmode_thrupts.append(thrupt)

        seppas_arr = np.array(this_klmode_seppas)
        seps_arr = seppas_arr[:,0]

        mean_thrupts = []
        # TODO: If no measurements available for a given sep
        # show warning and add np.nan 
        for sep in np.unique(seps_arr):
            this_sep_thrupts = np.where(seps_arr==sep,this_klmode_thrupts,np.nan)
            mean_thrupt = np.nanmean(this_sep_thrupts)
            mean_thrupts.append(mean_thrupt)

        thrupts.append(mean_thrupts)

    thrupt_arr = np.array([seps,*thrupts])

    return thrupt_arr

