import os
import warnings
from astropy.io import fits
import numpy as np
from corgidrp.data import PyKLIPDataset, Image 
from pyklip.parallelized import klip_dataset
from pyklip.fakes import gaussfit2d, inject_planet
from scipy.ndimage import shift, rotate
from corgidrp.astrom import get_polar_dist, seppa2dxdy, seppa2xy
from corgidrp.fluxcal import phot_by_gauss2d_fit

def get_closest_psf(ct_calibration,cenx,ceny,dx,dy):
    """_summary_
    NOTE: CT excam locations have (0,0) as the bottom left corner 
      of the bottom left pixel

    TODO: Calculate subpixel shifts if star or PSF model aren't
        perfectly in the center of a pixel

    Args:
        ct_calibration (corgidrp.data.CoreThroughputCalibration): CT calibration object.
        cenx (float): x location of mask center, measured from the center of the bottom 
            left pixel of the 1024x1024 pixel science array.
        ceny (float): y location of mask center, measured from the center of the bottom 
            left pixel of the 1024x1024 pixel science array.
        dx (float): x separation from the mask center in pixels
        dy (float): y separation from the mask center in pixels

    Returns:
        np.array: 2D PSF model closest to the desired location.
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
               sep_pix,pa_deg):
    """Injects a fake psf from the CT calibration object into a corgidrp Image with 
    the desired position and amplitude. 

    Args:
        frame_in (corgidrp.data.Image): 2D image to inject a fake signal into.
        ct_calibration (corgidrp.data.CoreThroughputCalibration): CT calibration object containing PSF samples.
        amp (float): peak pixel amplitude of psf to inject.
        sep_pix (float): separation from star in pixels to inject 
        pa_deg (float): position angle to inject (counterclockise from north/up)

    Returns: 
        corgidrp.data.Image: a copy of the input Image but with a fake PSF injected.
    """

    frame = frame_in.copy()

    # Get closest psf model
    pa_aper_deg = frame.pri_hdr['PA_APER']
    rel_pa = pa_deg - pa_aper_deg
    dx,dy = seppa2dxdy(sep_pix,rel_pa)

    psf_model = get_closest_psf(ct_calibration,
                                frame.ext_hdr['STARLOCX'],
                                frame.ext_hdr['STARLOCY'],
                                dx,dy).copy() 

    # Scale counts
    peak_count = np.nanmax(psf_model)
    psf_model *= amp / peak_count

    # Assume PSF is centered in the data cutout for now
    model_shape = np.array(psf_model.shape)
    psf_cenyx_ind = (np.array(model_shape)/2 - 0.5).astype(int) 
    psf_cenyx_inframe = np.array([frame.ext_hdr['STARLOCY'],frame.ext_hdr['STARLOCX']]) + np.array([dy,dx])
    injected_psf_cenyx_ind = np.round(psf_cenyx_inframe).astype(int)
    startyx = injected_psf_cenyx_ind - psf_cenyx_ind
    endyx = startyx + model_shape

    inject_planet([frame.data], [[frame.ext_hdr['STARLOCX'], frame.ext_hdr['STARLOCY']]], [psf_model], [None], sep_pix, 0, thetas=[rel_pa + 90])

    psf_cenxy = [psf_cenyx_inframe[1],psf_cenyx_inframe[0]]
    return frame, psf_model, psf_cenxy


def measure_noise(frame, seps_pix, hw, klmode_index=None, cand_locs = []):
    """Calculates the noise (standard deviation of counts) of an 
        annulus at a given separation from the mask or star center.
        TODO: Correct for small sample statistics?
        TODO: Mask known off-axis sources.
    
    Args:
        frame (corgidrp.Image): Image containing data as well as "STARLOCX/Y" in header
        seps_pix (np.array of float): Separations (in pixels from specified center) at which to calculate 
            the noise level.
        hw (float): halfwidth of the annulus to use for noise calculation.
        klmode_index (int, optional): If provided, returns only the noise values for the KL mode with 
            the given index. I.e. klmode_index=0 would return only the values for the first KL mode 
            truncation choice.  If None (by default), all indices are returned.
        cand_locs (list of tuples, optional): Locations of known off-axis sources, so we can mask them. 
            This is a list of tuples (sep_pix,pa_degrees) for each source. Defaults to [].
        
    Returns: np.array: array of shape (number of separtions, number of KL modes) containing the annular noise.  If klmode_index 
        specified, the number of KL modes in the output array is 1.
    """
    cenx, ceny = (frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'])
    
    # Mask data outside the specified annulus
    y, x = np.indices(frame.data.shape[1:])
    sep_map = np.sqrt((y-ceny)**2 + (x-cenx)**2)
    sep_map3d = np.ones_like(frame.data) * sep_map

    stds = []
    for sep_pix in seps_pix:
        r_inner = sep_pix - hw
        r_outer = sep_pix + hw
        masked_data = np.where(np.logical_and(sep_map3d<r_outer,sep_map3d>r_inner), frame.data,np.nan)
        
        # import matplotlib.pyplot as plt
        # plt.imshow(masked_data[0],origin='lower')
        # plt.colorbar()
        # plt.title('Candlocs not masked')
        # plt.show()

        if len(cand_locs) > 0:
            for cand_loc in cand_locs:
                cand_x, cand_y = seppa2xy(*cand_loc,cenx,ceny)
                dists = np.sqrt((y-cand_y)**2 + (x-cand_x)**2)
    
                masked_data = np.where(dists > 5 * hw, masked_data.copy(),np.nan)
        
        # import matplotlib.pyplot as plt
        # plt.imshow(masked_data[0],origin='lower')
        # plt.colorbar()
        # plt.title('Candlocs masked')
        # plt.show()

        # Calculate standard deviation
        with warnings.catch_warnings():
        # warnings.filterwarnings("ignore", category=UserWarning) # catch Not all requested_separations from l4_to_tda
            warnings.filterwarnings("ignore", category=RuntimeWarning) # catch Not all requested_separations from l4_to_tda
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
                     sep_spacing = 3.,
                     n_pas = 5,
                     seps = None, # in pixels from mask center
                     pas = None,
                     cand_locs = [], # list of tuples (sep_pix,pa_deg) of known off-axis source locations,
                     num_processes = None
                    ):
    """Measures the throughput of the KLIP algorithm via injection-recovery of fake off-axis sources. 

    Args:
        sci_dataset_in (corgidrp.data.Dataset): input dataset containing science observations
        ref_dataset_in (corgidrp.data.Dataset): input dataset containing reference observations 
            (set to None for ADI-only reductions)
        psfsub_dataset (corgidrp.data.Dataset): dataset containing PSF subtraction result
        ct_calibration (corgidrp.data.CoreThroughputCalibration): core throughput calibration object 
            containing off-axis PSFs.
        klip_params (dict): dictionary containing the same KLIP parameters used for PSF subtraction. Must 
            contain the keywords 'mode','annuli','subsections','movement','numbasis','outdir'.
            See corgidrp.l3_to_l4.do_psf_subtraction() for descriptions of each of these parameters.
        inject_snr (int, optional): SNR at which to inject fake PSFs. Defaults to 5.
        sep_spacing (float, optional): multiples of the FWHM at which to space separation samples. Defaults to 3. 
            Overridden by passing in explicit separations to the seps keyword.
        n_pas (int,optional): number of evenly spaced position angles at which to inject PSFs. Defaults to 5. 
            Overridden by in explicit PAs to the pas keyword.
        seps (np.array, optional): Separations (in pixels from the star center) at which to inject fake 
            PSFs. If not provided, a linear spacing of separations between the IWA & OWA will be chosen.
        pas (np.array, optional): Position angles (in degrees counterclockwise from north/up) at which to inject fake 
            PSFs at each separation. Defaults to [0.,90.,180.,270.].
        cand_locs (list of tuples, optional): Locations of known off-axis sources, so we don't inject a fake 
            PSF too close to them. This is a list of tuples (sep_pix,pa_degrees) for each source. Defaults to [].
        num_processes (int): number of processes for parallelizing the PSF subtraction
        
    Returns: 
        np.array: array of shape (N,n_seps,2), where N is 1 + the number of KL mode truncation choices and n_seps 
        is the number of separations sampled. Index 0 contains the separations sampled, and each following index
        contains the dimensionless KLIP throughput and FWHM in pixels measured at each separation for each KL mode 
        truncation choice. An example for 4 KL mode truncation choices, using r1 and r2 for separations and n_seps=2: 
            [ [[r1,r1],[r2,r2]], 
            [[KL_thpt_r1_KL1, FWHM_r1_KL1],[KL_thpt_r2_KL1, FWHM_r2_KL1]], 
            [[KL_thpt_r1_KL2, FWHM_r1_KL2],[KL_thpt_r2_KL2, FWHM_r2_KL2]], 
            [[KL_thpt_r1_KL3, FWHM_r1_KL3],[KL_thpt_r2_KL3, FWHM_r2_KL3]], 
            [[KL_thpt_r1_KL4, FWHM_r1_KL4],[KL_thpt_r2_KL4, FWHM_r2_KL4]] ]
    """
    
    if sci_dataset_in[0].ext_hdr['CFAMNAME'] == '1F':
        lam = 573.8e-9 #m
    else:
        raise NotImplementedError("Only band 1 observations using CFAMNAME 1F are currently configured.")

    d = 2.36 #m  
    pixscale_mas = sci_dataset_in[0].ext_hdr['PLTSCALE']
    fwhm_mas = 1.22 * lam / d * 206265. * 1000.
    fwhm_pix = fwhm_mas / pixscale_mas  
    res_elem = sep_spacing * fwhm_pix # pix
    
    if seps is None:
        if sci_dataset_in[0].ext_hdr['LSAMNAME'] == 'NFOV':
            owa_mas = 450. 
            owa_pix = owa_mas / pixscale_mas   
        else:
            raise NotImplementedError("Automatic separation choices only configured for NFOV observations.")
        
        if sci_dataset_in[0].ext_hdr['FPAMNAME'] == 'HLC12_C2R1':
            iwa_mas = 140. 
            iwa_pix = iwa_mas / pixscale_mas 
        else:
            raise NotImplementedError("Automatic separation choices only configured for NFOV observations.")
        
        seps = np.arange(iwa_pix,owa_pix,res_elem) # Some linear spacing between the IWA & OWA, around 5x the fwhm
    if pas is None:
        pas = np.linspace(0.,360.,n_pas+1)[:-1] # Some linear spacing between the IWA & OWA, around 5x the fwhm

    thrupts = []
    outfwhms = []
    for k,klmode in enumerate(klip_params['numbasis']):
        
        sci_dataset = sci_dataset_in.copy()

        pa_aper_degs = [frame.pri_hdr['PA_APER'] for frame in sci_dataset]
        
        # Measure noise at each separation in psf subtracted dataset (for this kl mode)
        noise_vals = measure_noise(psfsub_dataset[0],seps,fwhm_pix,k,cand_locs)
        
        # Inject planets:
        seppas_skipped = []

        this_klmode_seppas = []
        this_klmode_psfmodels = []
        this_klmode_inject_peaks = []
        this_klmode_psfcenxy = []

        for i,frame in enumerate(sci_dataset):
            this_klmode_seppas.append([])
            this_klmode_psfmodels.append([])
            this_klmode_inject_peaks.append([])
            this_klmode_psfcenxy.append([])

            # Initialize PA offset to spiral the injections
            pa_off = 0.
            pa_step = 360. / len(pas) / 3
            
            for s,sep in enumerate(seps):
                
                noise = noise_vals[s]
                inject_peak = noise * inject_snr

                for pa in pas:
                    pa = (pa + pa_off) % 360.
                    inject_loc = (sep,pa)

                    if inject_loc in seppas_skipped:
                        continue
                    
                    # Check that we're not too close to a candidate during the first frame
                    if i==0:
                        too_close = False
                        for cand_sep, cand_pa in cand_locs:
                            # Account for rotations, skip if any are too close
                            for pa_aper_deg in pa_aper_degs:
                                # NOTE (TO DO?): rotation angle is not applied here, cand_pa_adj == cand_pa.
                                # If rotation angle should be applied, cand_pa_adj = cand_pa + pa_aper_deg
                                cand_pa_adj = cand_pa
                                dist = get_polar_dist((cand_sep,cand_pa_adj),inject_loc)
                                if dist < res_elem:
                                    too_close=True
                                    break
                            if too_close:
                                break
                        if too_close:
                            seppas_skipped.append(inject_loc)
                            continue
                    
                    frame, psf_model, psf_cenxy = inject_psf(frame, ct_calibration, 
                                                    inject_peak, *inject_loc)

                    # Save these for later
                    this_klmode_psfmodels[i].append(psf_model.copy())
                    this_klmode_seppas[i].append(inject_loc)
                    this_klmode_inject_peaks[i].append(inject_peak)
                    this_klmode_psfcenxy[i].append(psf_cenxy)
                pa_off += pa_step

            sci_dataset[i].data[:] = frame.data[:]
                    
        # Debugging things
        psfmodels_arr = np.array(this_klmode_psfmodels)
        seppas_arr = np.array(this_klmode_seppas)
        inj_peaks = np.array(this_klmode_inject_peaks)
        psfcenxys = np.array(this_klmode_psfcenxy)
        psfmodel_sums = np.sum(psfmodels_arr,axis=(2,3))
        psfmodel_peaks = np.max(psfmodels_arr,axis=(2,3))

        # Init pyklip dataset
        pyklip_dataset = PyKLIPDataset(sci_dataset,psflib_dataset=ref_dataset_in)
        
        # Run pyklip
        klip_dataset(pyklip_dataset, outputdir=klip_params['outdir'],
                                annuli=klip_params['annuli'], subsections=klip_params['subsections'], 
                                movement=klip_params['movement'], 
                                numbasis=[klmode],
                                calibrate_flux=False, mode=klip_params['mode'],
                                psf_library=pyklip_dataset._psflib,
                                fileprefix=f"FAKE_{klmode}KLMODES",
                                numthreads=num_processes)
        
        # Get photometry of each injected source

        # Load pyklip output
        pyklip_fpath = os.path.join(klip_params['outdir'],f"FAKE_{klmode}KLMODES-KLmodes-all.fits")
        pyklip_data = fits.getdata(pyklip_fpath)[0]
        pyklip_hdr = fits.getheader(pyklip_fpath)

        # Measure background via sigma clip 
        n_loops = 5
        masked_data = pyklip_data.copy()
        for n in range(n_loops):
            std = np.nanstd(masked_data)
            med = np.nanmedian(masked_data)
            clip_thresh = 3 * std
            masked_data = np.where(np.abs(masked_data-med)>clip_thresh,np.nan,masked_data)
        
        # Subtract median
        bg_level = np.nanmedian(masked_data)
        medsubtracted_data = pyklip_data - bg_level
        
        # # Plot Psf subtraction with fakes
        # if klip_params['mode'] == 'RDI':
        #     analytical_result = rotate(sci_dataset[0].data - ref_dataset_in[0].data,-rolls[0],reshape=False,cval=np.nan)
        # elif klip_params['mode'] == 'ADI':
        #     analytical_result = (rotate(sci_dataset[0].data - sci_dataset[1].data,-rolls[0],reshape=False,cval=0) + rotate(sci_dataset[1].data - sci_dataset[0].data,-rolls[1],reshape=False,cval=0)) / 2
        # elif klip_params['mode'] == 'ADI+RDI':
        #     analytical_result = (rotate(sci_dataset[0].data - (sci_dataset[1].data/2+ref_dataset_in[0].data/2),-rolls[0],reshape=False,cval=0) + rotate(sci_dataset[1].data - (sci_dataset[0].data/2+ref_dataset_in[0].data/2),-rolls[1],reshape=False,cval=0)) / 2

        # import matplotlib.pyplot as plt
        # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
        # im0 = axes[0].imshow(medsubtracted_data,origin='lower')
        # plt.colorbar(im0,ax=axes[0],shrink=0.8)
        # locsxy = seppa2xy(seppas_arr[0,:,0],seppas_arr[0,:,1],pyklip_hdr['PSFCENTX'],pyklip_hdr['PSFCENTY'])
        # axes[0].scatter(locsxy[0],locsxy[1],label='Injected PSFs',s=1,c='r',marker='x')
        # axes[0].set_title(f'Output data')
        # axes[0].legend()
        # im1 = axes[1].imshow(analytical_result,origin='lower')
        # plt.colorbar(im1,ax=axes[1],shrink=0.8)
        # axes[1].set_title('Analytical result')
        # im2 = axes[2].imshow(medsubtracted_data - analytical_result,origin='lower')
        # plt.colorbar(im2,ax=axes[2],shrink=0.8)
        # axes[2].set_title('Difference')
        # plt.suptitle(f'PSF Subtraction {klip_params["mode"]} ({klmode} KL Modes)')
        # plt.show()
        
        # After psf subtraction
        this_klmode_peakin = []
        this_klmode_peakout = []
        this_klmode_sumin = []
        this_klmode_influxs = []
        this_klmode_outfluxs = []
        this_klmode_infwhms = []
        this_klmode_outfwhms = []
        this_klmode_thrupts = []
        for ll,loc in enumerate(this_klmode_seppas[0]):
            
            psf_model = this_klmode_psfmodels[0][ll]
            
            # Pad psf model with zeros so we can measure background
            model_shape = np.array(psf_model.shape)
            cutout_shape = model_shape * 2 + 1
            cutoutcenyx = cutout_shape/2. - 0.5

            psf_model_padded = np.zeros(cutout_shape)
            start_ind_model = (cutoutcenyx-model_shape//2).astype(int)
            end_ind_model = (start_ind_model + model_shape).astype(int)
            x1_model,y1_model = start_ind_model
            x2_model,y2_model = end_ind_model

            psf_model_padded[y1_model:y2_model,
                             x1_model:x2_model] = psf_model

            # Crop data around location to be same as psf_model cutout
            locxy = seppa2xy(*loc,pyklip_hdr['PSFCENTX'],pyklip_hdr['PSFCENTY'])

            # Crop the data, pad with nans if we're cropping over the edge
            cutout = np.zeros_like(psf_model_padded)
            cutout[:] = np.nan
            cutout_starty, cutout_startx = (0,0)
            cutout_endy, cutout_endx = cutout.shape

            data_shape = medsubtracted_data.shape
            data_center_indyx = np.array([locxy[1],locxy[0]]).astype(int)
            data_start_indyx = (data_center_indyx - cutout_shape//2)
            data_end_indyx = (data_start_indyx + cutout_shape)
            data_starty,data_startx = data_start_indyx
            data_endy,data_endx = data_end_indyx
            
            if data_starty < 0:
                cutout_starty = -data_starty
                data_starty = 0
            
            if data_startx < 0:
                cutout_startx = -data_startx
                data_startx = 0
            
            if data_endy >= data_shape[0]:
                y_overhang = data_endy - medsubtracted_data.shape[0]
                cutout_endy = cutout_shape[0] - y_overhang
                data_endy = data_shape[0]

            if data_endx >= data_shape[1]:
                x_overhang = data_endx - medsubtracted_data.shape[1]
                cutout_endx = cutout_shape[1] - x_overhang
                data_endx = data_shape[1]


            cutout[cutout_starty:cutout_endy,
                        cutout_startx:cutout_endx] = medsubtracted_data[data_starty:data_endy,
                                                            data_startx:data_endx]
            
            # # Plot PSF Model
            # import matplotlib.pyplot as plt
            # fig,ax = plt.subplots(1,3,
            #                       sharey=True,
            #                       layout='constrained',
            #                       figsize=(12,4))
            # vmax = np.max(psf_model_padded)
            # im0 = ax[0].imshow(psf_model_padded,origin='lower',
            #                    vmax=vmax,vmin=-vmax)
            # plt.colorbar(im0,ax=ax[0])
            # ax[0].set_title('PSF Model')
            # im1 = ax[1].imshow(cutout,origin='lower',
            #                    vmax=vmax,vmin=-vmax)
            # plt.colorbar(im1,ax=ax[1])
            # ax[1].set_title('Data Cutout')
            # im2 = ax[2].imshow(cutout-psf_model_padded,origin='lower',
            #                    vmax=vmax,vmin=-vmax)
            # plt.colorbar(im2,ax=ax[2])
            # ax[2].set_title('Residuals')
            # plt.show()

            # Using pyklip.fakes.gaussfit2d
            preklip_peak, pre_fwhm, pre_xfit, pre_yfit = gaussfit2d(
                psf_model_padded, 
                cutoutcenyx[1], 
                cutoutcenyx[0], 
                searchrad=5, 
                guessfwhm=fwhm_pix, 
                guesspeak=inject_peak, 
                refinefit=True)

            postklip_peak, post_fwhm, post_xfit, post_yfit = gaussfit2d(
                cutout, 
                cutoutcenyx[1], 
                cutoutcenyx[0], 
                searchrad=5, 
                guessfwhm=fwhm_pix, 
                guesspeak=inject_peak, 
                refinefit=True) 
            
            # # Plot Final PSF Model
            # import matplotlib.pyplot as plt
            # post_sigma = post_fwhm / (2 * np.sqrt(2. * np.log(2.)))
            # from corgidrp.mocks import gaussian_array
            # final_model = gaussian_array(array_shape=cutout_shape,
            #                      sigma=post_sigma,
            #                      amp=postklip_peak,
            #                      xoffset=post_xfit-19.,
            #                      yoffset=post_yfit-19.)
            
            # fig,ax = plt.subplots(1,3,
            #                       sharey=True,
            #                       layout='constrained',
            #                       figsize=(12,4))
            # vmax = np.max(psf_model_padded)
            # im0 = ax[0].imshow(final_model,origin='lower',
            #                    vmax=vmax,vmin=-vmax)
            # plt.colorbar(im0,ax=ax[0])
            # ax[0].set_title('Final PSF Model')
            # im1 = ax[1].imshow(cutout,origin='lower',
            #                    vmax=vmax,vmin=-vmax)
            # plt.colorbar(im1,ax=ax[1])
            # ax[1].set_title('Data Cutout')
            # im2 = ax[2].imshow(cutout-final_model,origin='lower',
            #                    vmax=vmax,vmin=-vmax)
            # plt.colorbar(im2,ax=ax[2])
            # ax[2].set_title('Residuals')
            # plt.show()
            


            # Get total counts from 2D gaussian fit
            preklip_counts = np.pi * preklip_peak * pre_fwhm**2 / 4. / np.log(2.)
            postklip_counts = np.pi * postklip_peak * post_fwhm**2 / 4. / np.log(2.)
            
            thrupt = postklip_counts/preklip_counts

            this_klmode_peakin.append(preklip_peak)
            this_klmode_peakout.append(postklip_peak)
            this_klmode_sumin.append(np.sum(psf_model))
            this_klmode_influxs.append(preklip_counts)
            this_klmode_outfluxs.append(postklip_counts)
            this_klmode_infwhms.append(pre_fwhm)
            this_klmode_outfwhms.append(post_fwhm)
            this_klmode_thrupts.append(thrupt)

        seppas_arr = np.array(this_klmode_seppas[0])
        seps_arr = seppas_arr[:,0]

        # # Plot injected and recovered counts
        # import matplotlib.pyplot as plt
        # fig,ax = plt.subplots()
        # plt.scatter(seps_arr,this_klmode_influxs,label='Injected counts')
        # plt.scatter(seps_arr,this_klmode_outfluxs,label='Recovered counts')
        # plt.xlabel('separation (pixels)')
        # plt.legend()
        # plt.show()

        # # Plot injected and recovered peaks
        # fig,ax = plt.subplots()
        # plt.scatter(seps_arr,this_klmode_peakin,label='Injected peaks')
        # plt.scatter(seps_arr,this_klmode_peakout,label='Recovered peaks')
        # plt.xlabel('separation (pixels)')
        # plt.legend()
        # plt.show()

        mean_thrupts = []
        mean_outfwhms = []
        # TODO: If no measurements available for a given sep
        # show warning and add np.nan 
        for sep in np.unique(seps_arr):
            this_sep_thrupts = np.where(seps_arr==sep,this_klmode_thrupts,np.nan)
            mean_thrupt = np.nanmedian(this_sep_thrupts)
            mean_thrupts.append(mean_thrupt)

            this_sep_outfwhms = np.where(seps_arr==sep,this_klmode_outfwhms,np.nan)
            mean_outfwhm = np.nanmedian(this_sep_outfwhms)
            mean_outfwhms.append(mean_outfwhm)

        thrupts.append(mean_thrupts)
        outfwhms.append(mean_outfwhms)

    thrupt_arr = np.array([[(sep,sep) for sep in seps],*[[(thrupts[kk][ss],outfwhms[kk][ss]) for ss in range(len(seps))] for kk in range(len(klip_params['numbasis']))]])

    return thrupt_arr
