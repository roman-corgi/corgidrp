import os
import warnings
from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np
import matplotlib.pyplot as plt
from corgidrp.data import PyKLIPDataset, Image 
from pyklip.parallelized import klip_dataset
from pyklip.fakes import gaussfit2d
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
               sep_pix,pa_deg):
    """Injects a fake psf from the CT calibration object into a corgidrp Image with 
    the desired position and amplitude. 

    Args:
        frame_in (corgidrp.data.Image): 2D image to inject a fake signal into.
        ct_calibration (corgidrp.data.CoreThroughputCalibration): CT calibration object containing PSF samples.
        amp (float): peak pixel amplitude of psf to inject.
        sep_pix (float): separation from star in pixels to inject 
        pa_deg (float): position angle to inject (counterclockise from north/up)

    Raises:
        UserWarning: _description_

    Returns:
        _type_: _description_
    """

    frame = frame_in.copy()

    # Get closest psf model
    frame_roll = frame.ext_hdr['ROLL']
    rel_pa = pa_deg - frame_roll
    dx,dy = seppa2dxdy(sep_pix,rel_pa)

    psf_model = get_closest_psf(ct_calibration,
                                frame.ext_hdr['STARLOCX'],
                                frame.ext_hdr['STARLOCY'],
                                dx,dy).copy() 

    # Scale counts
    peak_count = np.nanmax(psf_model)
    psf_model *= amp / peak_count

    # Assume PSF is centered in the data cutout for now
    shape_arr = np.array(psf_model.shape)
    psf_cenyx_ind = (np.array(shape_arr)/2 - 0.5).astype(int) 
    psf_cenyx_inframe = np.array([frame.ext_hdr['STARLOCY'],frame.ext_hdr['STARLOCX']]) + np.array([dy,dx])
    injected_psf_cenyx_ind = np.round(psf_cenyx_inframe).astype(int)
    starty, startx = injected_psf_cenyx_ind - psf_cenyx_ind

    # Insert into correct frame size array 
    psf_only_frame = np.zeros_like(frame.data)
    psf_only_frame[starty:starty+shape_arr[0],startx:startx+shape_arr[1]] = psf_model
    
    # # TODO: Calculate subpixel shift:
    # psf_shift = (0.,0.) # Hardcode 0 shift for now
    # shifted_psf_only_frame = shift(psf_only_frame,psf_shift)

    # Add to input frame
    frame.data += psf_only_frame

    psf_cenxy = [psf_cenyx_inframe[1],psf_cenyx_inframe[0]]
    return frame, psf_model, psf_cenxy


def measure_noise(frame, seps_pix, fwhm, klmode_index=None):
    """Calculates the noise (standard deviation of counts) of an 
        annulus at a given separation from the mask center.
        TODO: Correct for small sample statistics?
    
    Args:
        frame (corgidrp.Image): Image containing data as well as "MASKLOCX/Y" in header
        seps_pix (np.array of float): Separations (in pixels from mask center) at which to calculate 
            the noise level.
        fwhm (float): halfwidth of the annulus to use for noise calculation, based on FWHM.
        klmode_index (int, optional): If provided, returns only the noise values for the KL mode with 
            the given index. I.e. klmode_index=0 would return only the values for the first KL mode 
            truncation choice.

    Returns:
        np.array: noise levels at the specified separations. Array is of shape (len(seps_pix),) 
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
        seps (np.array, optional): Separations (in pixels from the star center) at which to inject fake 
            PSFs. If not provided, a linear spacing of separations between the IWA & OWA will be chosen.
        cand_locs (list of tuples, optional): Locations of known off-axis sources, so we don't inject a fake 
            PSF too close to them. This is a list of tuples (sep_pix,pa_degrees) for each source. Defaults to [].
        
    Returns:
        np.array: _description_
    """
    
    iwa = 10. # pix, update real number
    owa = 50. # pix, update with real number

    d = 2.36 #m
    lam = 573.8e-9 #m
    pixscale_arcsec = 0.0218
    fwhm_mas = 1.22 * lam / d * 206265 * 1000
    fwhm_pix = fwhm_mas * 0.001 / pixscale_arcsec
    res_elem = 5 * fwhm_pix # pix, update this with value for NFOV, Band 1 mode
    
    if pas == None:
        pas = np.array([0.,90.,180.,270.])
    if seps == None:
        seps = np.arange(iwa,owa,res_elem) # Some linear spacing between the IWA & OWA, around 5x the fwhm

    thrupts = []
    for k,klmode in enumerate(klip_params['numbasis']):
        
        sci_dataset = sci_dataset_in.copy()
        ref_dataset = ref_dataset_in.copy() if not ref_dataset_in is None else None

        rolls = [frame.ext_hdr['ROLL'] for frame in sci_dataset]
        
        # Measure noise at each separation in psf subtracted dataset (for this kl mode)
        noise_vals = measure_noise(psfsub_dataset[0],seps,fwhm_pix,k)
        
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
                    
                    frame, psf_model, psf_cenxy = inject_psf(frame, ct_calibration, 
                                                    inject_peak, *inject_loc,
                                                    norm='peak')

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
        pyklip_dataset = PyKLIPDataset(sci_dataset,psflib_dataset=ref_dataset)
        
        # Run pyklip
        klip_dataset(pyklip_dataset, outputdir=klip_params['outdir'],
                                annuli=klip_params['annuli'], subsections=klip_params['subsections'], 
                                movement=klip_params['movement'], 
                                numbasis=[klmode],
                                calibrate_flux=False, mode=klip_params['mode'],
                                psf_library=pyklip_dataset._psflib,
                                fileprefix=f"FAKE_{klmode}KLMODES")
        
        # Get photometry of each injected source

        # Load pyklip output
        pyklip_fpath = os.path.join(klip_params['outdir'],f"FAKE_{klmode}KLMODES-KLmodes-all.fits")
        pyklip_data = fits.getdata(pyklip_fpath)[0]
        pyklip_hdr = fits.getheader(pyklip_fpath)

        # Measure and subtract background
        # Sigma clip threshold 5
        std = np.nanstd(pyklip_data)
        med = np.nanmedian(pyklip_data)
        clip_thresh = 3 * std
        masked_data = np.where(np.abs(pyklip_data-med)>clip_thresh,np.nan,pyklip_data)
        bg_level = np.nanmedian(masked_data)
        # Subtract median
        medsubtracted_data = pyklip_data - bg_level
        
        # 
        #     # Plot Psf subtraction with fakes
            
        #     if klip_params['mode'] == 'RDI':
        #         analytical_result = rotate(sci_dataset[0].data - ref_dataset[0].data,-rolls[0],reshape=False,cval=np.nan)
        #     elif klip_params['mode'] == 'ADI':
        #         analytical_result = shift((rotate(sci_dataset[0].data - sci_dataset[1].data,-rolls[0],reshape=False,cval=0) + rotate(sci_dataset[1].data - sci_dataset[0].data,-rolls[1],reshape=False,cval=0)) / 2,
        #                         [0.5,0.5],
        #                         cval=np.nan)
        #     elif klip_params['mode'] == 'ADI+RDI':
        #         analytical_result = (rotate(sci_dataset[0].data - (sci_dataset[1].data/2+ref_dataset[0].data/2),-rolls[0],reshape=False,cval=0) + rotate(sci_dataset[1].data - (sci_dataset[0].data/2+ref_dataset[0].data/2),-rolls[1],reshape=False,cval=0)) / 2
    
        #     locsxy = seppa2xy(seppas_arr[0,:,0],seppas_arr[0,:,1],pyklip_hdr['PSFCENTX'],pyklip_hdr['PSFCENTY'])
            
        #     import matplotlib.pyplot as plt

        #     fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
        #     im0 = axes[0].imshow(medsubtracted_data,origin='lower')
        #     plt.colorbar(im0,ax=axes[0],shrink=0.8)
        #     axes[0].scatter(locsxy[0],locsxy[1],label='Injected PSFs',s=1,c='r',marker='x')
        #     axes[0].set_title(f'PSF Sub Result ({klmode} KL Modes)')
        #     axes[0].legend()

        #     im1 = axes[1].imshow(analytical_result,origin='lower')
        #     plt.colorbar(im1,ax=axes[1],shrink=0.8)
        #     axes[1].set_title('Analytical result')

        #     im2 = axes[2].imshow(medsubtracted_data - analytical_result,origin='lower')
        #     plt.colorbar(im2,ax=axes[2],shrink=0.8)
        #     axes[2].set_title('Difference')

        #     plt.show()
        
        # After psf subtraction
        this_klmode_peakin = []
        this_klmode_peakout = []
        this_klmode_sumin = []
        this_klmode_influxs = []
        this_klmode_outfluxs = []
        this_klmode_thrupts = []
        for ll,loc in enumerate(this_klmode_seppas[0]):
            
            psf_model = this_klmode_psfmodels[0][ll]
            
            # Pad psf model with zeros so we can measure background
            model_shape = np.array(psf_model.shape)
            cutout_shape = model_shape
            cutoutcenyx = cutout_shape/2. - 0.5

            psf_model_padded = np.zeros(cutout_shape)
            start_ind = (cutoutcenyx-model_shape//2).astype(int)
            end_ind = (start_ind + model_shape).astype(int)
            x1,y1 = start_ind
            x2,y2 = end_ind

            psf_model_padded[y1:y2,x1:x2] = psf_model

            # Crop data around location to be same as psf_model cutout
            locxy = seppa2xy(*loc,pyklip_hdr['PSFCENTX'],pyklip_hdr['PSFCENTY'])

            # Crop the data
            start_ind = (locxy - cutout_shape//2).astype(int)
            end_ind = (locxy + cutout_shape//2 + 1).astype(int)
            x1,y1 = start_ind
            x2,y2 = end_ind
            data_cutout = medsubtracted_data[y1:y2,x1:x2]

            # if debug:
            #     import matplotlib.pyplot as plt
            #     fig,ax = plt.subplots(1,2,
            #                           sharey=True,
            #                           layout='constrained',
            #                           figsize=(8,4)
            #                         )
                
            #     im0 = ax[0].imshow(psf_model_padded,origin='lower')
            #     plt.colorbar(im0,ax=ax[0])
            #     ax[0].set_title('PSF Model')
            #     im1 = ax[1].imshow(data_cutout,origin='lower')
            #     plt.colorbar(im1,ax=ax[1])
            #     ax[1].set_title('Data Cutout')
            #     plt.show()
            #     pass
                
            # if x1<0. or y1<0. or x2>=cutout_shape[1] or y2>=cutout_shape[0]:
            #     print('!!!')
            #     pass
            
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
                data_cutout, 
                cutoutcenyx[1], 
                cutoutcenyx[0], 
                searchrad=5, 
                guessfwhm=fwhm_pix, 
                guesspeak=inject_peak, 
                refinefit=True) 

            # Get total counts from 2D gaussian fit
            preklip_counts = np.pi * preklip_peak * pre_fwhm**2 / 4. / np.log(2.)
            postklip_counts = np.pi * postklip_peak * post_fwhm**2 / 4. / np.log(2.)

            # old version using pyhot_by_gauss2d_fit
            # # Temporarily load into Image obj
            # model_img = Image(psf_model,pri_hdr=fits.Header(),ext_hdr=fits.Header())
            # data_img = Image(data_cutout,pri_hdr=fits.Header(),ext_hdr=fits.Header())

            # # Try pyklip flux measurement functions: fakes.2dgaussianfit, retrieve planet flux
            # preklip_amp, preklip_err, preklip_bg = phot_by_gauss2d_fit(model_img,fwhm_pix,background_sub=True,fit_shape=psf_model.shape)
            # try:
            #     postklip_amp, postklip_err, postklip_bg = phot_by_gauss2d_fit(data_img,fwhm_pix,background_sub=True,fit_shape=data_cutout.shape)
            # except:
            #     warnings.warn('Amplitude of fake after KLIP subtraction not recovered.')
            #     postklip_amp = np.nan
            
            
            thrupt = postklip_counts/preklip_counts

            this_klmode_peakin.append(preklip_peak)
            this_klmode_peakout.append(postklip_peak)
            this_klmode_sumin.append(np.sum(psf_model))
            this_klmode_influxs.append(preklip_counts)
            this_klmode_outfluxs.append(postklip_counts)
            this_klmode_thrupts.append(thrupt)

        seppas_arr = np.array(this_klmode_seppas[0])
        seps_arr = seppas_arr[:,0]

        # if debug:
        #     # Plot injected and recovered counts
        #     fig,ax = plt.subplots()
        #     plt.scatter(seps_arr,this_klmode_influxs,label='Injected counts')
        #     plt.scatter(seps_arr,this_klmode_outfluxs,label='Recovered counts')
        #     plt.xlabel('separation (pixels)')
        #     plt.legend()
        #     plt.show()

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

