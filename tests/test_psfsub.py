from corgidrp.mocks import create_psfsub_dataset,create_default_L1_headers
from corgidrp.l3_to_l4 import do_psf_subtraction
from corgidrp.data import PyKLIPDataset, Image, Dataset
from corgidrp.detector import nan_flags, flag_nans
from corgidrp.astrom import create_circular_mask
from scipy.ndimage import shift, rotate
import pytest
import numpy as np

## Helper functions/quantities

iwa_lod = 3.
owa_lod = 9.7
d = 2.36 #m
lam = 573.8e-9 #m
pixscale_arcsec = 0.0218

iwa_pix = iwa_lod * lam / d * 206265 / pixscale_arcsec
owa_pix = owa_lod * lam / d * 206265 / pixscale_arcsec

st_amp = 100.
noise_amp=1e-11
pl_contrast=1e-4
rel_tolerance = 0.05

## pyKLIP data class tests

def test_pyklipdata_ADI():
    """Tests that pyklip dataset centers frame, assigns rolls, and initializes PSF library properly for ADI data. 
    """

    rolls = [0,90]
    # Init with center shifted by 1 pixel in x, 2 pixels in y
    mock_sci,mock_ref = create_psfsub_dataset(2,0,rolls,
                                              centerxy=(50.5,51.5))

    pyklip_dataset = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

    # Check image is centered properly
    for i,image in enumerate(pyklip_dataset._input):

        assert mock_sci.all_data[i,2:,1:] == pytest.approx(image[:-2,:-1]), f"Frame {i} centered improperly."

    # Check roll assignments match up for sci dataset
    for r,roll in enumerate(pyklip_dataset._PAs):
        assert roll == rolls[r]
        assert mock_sci[r].pri_hdr['ROLL'] == roll, f"Incorrect roll assignment for frame {r}."
    
    # Check ref library is None
    assert pyklip_dataset.psflib is None, "pyklip_dataset.psflib is not None, even though no reference dataset was provided."

def test_pyklipdata_RDI():
    """Tests that pyklip dataset centers frame, assigns rolls, and initializes PSF library properly for RDI data. 
    """
    rolls = [45,180]
    n_sci = 1
    n_ref = 1
    # Init with center shifted
    mock_sci,mock_ref = create_psfsub_dataset(n_sci,n_ref,rolls,centerxy=(50.5,51.5))

    pyklip_dataset = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)
    
    # Check image is centered properly
    for i,image in enumerate(pyklip_dataset._input):

        assert mock_sci.all_data[i,2:,1:] == pytest.approx(image[:-2,:-1]), f"Frame {i} centered improperly."

    # Check roll assignments match up for sci dataset
    for r,roll in enumerate(pyklip_dataset._PAs):
        assert roll == rolls[r]
        assert mock_sci[r].pri_hdr['ROLL'] == roll, f"Incorrect roll assignment for frame {r}."
    
    # Check ref library shape
    assert pyklip_dataset._psflib.master_library.shape[0] == n_sci+n_ref
 
def test_pyklipdata_ADIRDI():
    """Tests that pyklip dataset centers frame, assigns rolls, and initializes PSF library properly for ADI+RDI data. 
    """
    rolls = [45,-45,180]
    n_sci = 2
    n_ref = 1
    # Init with center shifted by 1 pixel
    mock_sci,mock_ref = create_psfsub_dataset(n_sci,n_ref,rolls,
                                              centerxy=(50.5,51.5))

    pyklip_dataset = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

    # Check image is recentered properly
    for i,image in enumerate(pyklip_dataset._input):

        assert mock_sci.all_data[i,2:,1:] == pytest.approx(image[:-2,:-1]), f"Frame {i} centered improperly."

    # Check roll assignments match up for sci dataset
    for r,roll in enumerate(pyklip_dataset._PAs):
        assert roll == rolls[r]
        assert mock_sci[r].pri_hdr['ROLL'] == roll, f"Incorrect roll assignment for frame {r}."
    
    # Check ref library shape
    assert pyklip_dataset._psflib.master_library.shape[0] == n_sci+n_ref

def test_pyklipdata_badtelescope():
    """Tests that pyklip data class initialization fails if data does not come from Roman.
    """
    mock_sci,mock_ref = create_psfsub_dataset(1,1,[0,0])
    mock_sci[0].pri_hdr['TELESCOP'] = "HUBBLE"

    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

def test_pyklipdata_badinstrument():
    """Tests that pyklip data class initialization fails if data does not come from Coronagraph Instrument.
    """
    mock_sci,mock_ref = create_psfsub_dataset(1,1,[0,0])
    mock_sci[0].pri_hdr['INSTRUME'] = "WFI"

    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

def test_pyklipdata_badcfamname():
    """Tests that pyklip data class raises an error if the CFAM position is not a valid position name.
    """
    mock_sci,mock_ref = create_psfsub_dataset(1,1,[0,0])
    mock_sci[0].ext_hdr['CFAMNAME'] = "BAD"

    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

def test_pyklipdata_notdataset():
    """Tests that pyklip data class raises an error if the iput is not a corgidrp dataset object.
    """
    mock_sci,mock_ref = create_psfsub_dataset(1,0,[0])
    mock_ref = []
    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

    mock_sci = []
    mock_ref = None
    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

def test_pyklipdata_badimgshapes():
    """Tests that pyklip data class enforces all data frames to have the same shape.
    """
    mock_sci,mock_ref = create_psfsub_dataset(2,0,[0,0])
    
    mock_sci[0].data = np.zeros((5,5))
    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

def test_pyklipdata_multiplepixscales():
    """Tests that pyklip data class enforces that each frame has the same pixel scale.
    """
    mock_sci,mock_ref = create_psfsub_dataset(2,0,[0,0])
    
    mock_sci[0].ext_hdr["PLTSCALE"] = 10
    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

# DQ flagging tests

def make_test_data(frame_shape,n_frames=1,):
    """Makes a test corgidrp Dataset of all zeros with the desired
    frame shape and number of frames.

    Args:
        frame_shape (listlike of int): 2D or 3D image shape desired.
        n_frames (int, optional): Number of frames. Defaults to 1.

    Returns:
        corgidrp.Dataset: mock dataset of all zeros.
    """
    
    frames = []
    for i in range(n_frames):
        prihdr, exthdr = create_default_L1_headers()
        im_data = np.zeros(frame_shape).astype(np.float64)
        frame = Image(im_data, pri_hdr=prihdr, ext_hdr=exthdr)

        frames.append(frame)

    dataset = Dataset(frames)
    return dataset

def test_nanflags_2D():
    """Test detector.nan_flags() on 2D data.
    """

    # 2D:
    mock_dataset = make_test_data([10,10],n_frames=2,)
    mock_dataset.all_dq[0,4,6] = 1
    mock_dataset.all_dq[1,5,7] = 1

    expected_data = np.zeros([2,10,10])
    expected_data[0,4,6] = np.nan
    expected_data[1,5,7] = np.nan

    expected_err = np.zeros([2,1,10,10])
    expected_err[0,0,4,6] = np.nan
    expected_err[1,0,5,7] = np.nan

    nanned_dataset = nan_flags(mock_dataset,threshold=1)

    if not np.array_equal(nanned_dataset.all_data, expected_data,equal_nan=True):
        # import matplotlib.pyplot as plt 
        # fig,axes = plt.subplots(1,2)
        # axes[0].imshow(nanned_dataset.all_data[0,:,:])
        # axes[1].imshow(expected_data[0,:,:])
        # plt.show()
        # plt.close()
        raise Exception('2D nan_flags test produced unexpected result')

    if not np.array_equal(nanned_dataset.all_err,expected_err,equal_nan=True):
        raise Exception('2D nan_flags test produced unexpected result for ERR array')

def test_nanflags_3D():
    """Test detector.nan_flags() on 3D data.
    """

    # 3D:
    mock_dataset = make_test_data([3,10,10],n_frames=2,)
    mock_dataset.all_dq[0,0,4,6] = 1
    mock_dataset.all_dq[1,:,5,7] = 1

    expected_data = np.zeros([2,3,10,10])
    expected_data[0,0,4,6] = np.nan
    expected_data[1,:,5,7] = np.nan

    expected_err = np.zeros([2,1,3,10,10])
    expected_err[0,0,0,4,6] = np.nan
    expected_err[1,0,:,5,7] =  np.nan

    nanned_dataset = nan_flags(mock_dataset,threshold=1)

    if not np.array_equal(nanned_dataset.all_data, expected_data,equal_nan=True):
        raise Exception('2D nan_flags test produced unexpected result')
    
    if not np.array_equal(nanned_dataset.all_err, expected_err,equal_nan=True):
        raise Exception('3D nan_flags test produced unexpected result for ERR array')

def test_nanflags_mixed_dqvals():
    """Test detector.nan_flags() on 3D data with some DQ values below the threshold.
    """

    # 3D:
    mock_dataset = make_test_data([3,10,10],n_frames=2,)
    mock_dataset.all_dq[0,0,4,6] = 1
    mock_dataset.all_dq[1,:,5,7] = 2

    expected_data = np.zeros([2,3,10,10])
    expected_data[1,:,5,7] = np.nan

    expected_err = np.zeros([2,1,3,10,10])
    expected_err[1,0,:,5,7] =  np.nan

    nanned_dataset = nan_flags(mock_dataset,threshold=2)

    if not np.array_equal(nanned_dataset.all_data, expected_data,equal_nan=True):
        raise Exception('nan_flags with mixed dq values produced unexpected result')
    
    if not np.array_equal(nanned_dataset.all_err, expected_err,equal_nan=True):
        raise Exception('3D nan_flags with mixed dq values produced unexpected result for ERR array')

def test_flagnans_2D():
    """Test detector.flag_nans() on 2D data.
    """

    # 2D:
    mock_dataset = make_test_data([10,10],n_frames=2,)
    mock_dataset.all_data[0,4,6] = np.nan
    mock_dataset.all_data[1,5,7] = np.nan

    expected_dq = np.zeros([2,10,10])
    expected_dq[0,4,6] = 1
    expected_dq[1,5,7] = 1

    flagged_dataset = flag_nans(mock_dataset)

    if not np.array_equal(flagged_dataset.all_dq, expected_dq,equal_nan=True):
        raise Exception('2D nan_flags test produced unexpected result for DQ array')
    
def test_flagnans_3D():
    """Test detector.flag_nans() on 3D data.
    """

    # 3D:
    mock_dataset = make_test_data([3,10,10],n_frames=2,)
    mock_dataset.all_data[0,0,4,6] = np.nan
    mock_dataset.all_data[1,:,5,7] = np.nan

    expected_dq = np.zeros([2,3,10,10])
    expected_dq[0,0,4,6] = 1
    expected_dq[1,:,5,7] = 1
    
    flagged_dataset = flag_nans(mock_dataset)

    if not np.array_equal(flagged_dataset.all_dq, expected_dq,equal_nan=True):
        raise Exception('3D flag_nans test produced unexpected result for DQ array')

def test_flagnans_flagval2():
    """Test detector.flag_nans() on 3D data with a non-default DQ value.
    """

    # 3D:
    mock_dataset = make_test_data([3,10,10],n_frames=2,)
    mock_dataset.all_data[0,0,4,6] = np.nan
    mock_dataset.all_data[1,:,5,7] = np.nan

    expected_dq = np.zeros([2,3,10,10])
    expected_dq[0,0,4,6] = 2
    expected_dq[1,:,5,7] = 2

    flagged_dataset = flag_nans(mock_dataset,flag_val=2)

    if not np.array_equal(flagged_dataset.all_dq, expected_dq,equal_nan=True):
        raise Exception('3D nan_flags test produced unexpected result for DQ array')

## PSF subtraction step tests

def test_psf_sub_split_dataset():
    """Tests that psf subtraction step can correctly split an input dataset into
    science and reference dataset, if they are not passed in separately.
    """

    # Sci & Ref
    numbasis = [1]
    subsections = 2
    annuli = 2
    movement = 2
    rolls = [270+13,270-13,0,0]
    mock_sci,mock_ref = create_psfsub_dataset(2,2,rolls,
                                              st_amp=st_amp,
                                              noise_amp=noise_amp,
                                              pl_contrast=pl_contrast)
    
    # combine mock_sci and mock_ref into 1 dataset
    frames = [*mock_sci,*mock_ref]
    mock_sci_and_ref = Dataset(frames)
    klip_kwargs={"numbasis":numbasis,
                    "subsections":subsections,
                    "annuli":annuli,
                    "movement":movement,}
    
    # Pass combined dataset to do_psf_subtraction
    result = do_psf_subtraction(mock_sci_and_ref,
                                fileprefix='test_single_dataset',
                                
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs)
    
    # Check correct KLIP parameters are used (Should choose ADI+RDI)
    for frame in result:

        # Parse PSFPARAM header string
        psfparams = frame.ext_hdr['PSFPARAM'].split(',')
        psfparams_dict = {}
        for pair in psfparams:
            param, val = pair.strip().split('=')
            psfparams_dict[param] = val

        # Check values
        if psfparams_dict['mode'] != 'ADI+RDI':
            raise Exception(f"Chose {psfparams_dict['mode']} instead of 'ADI+RDI' when provided 2 science images and 2 references.")
        if psfparams_dict['annuli'] != str(annuli):
            raise Exception(f"Unexpected number of annuli was used in KLIP parameters.")
        if psfparams_dict['subsect'] != str(subsections):
            raise Exception(f"Unexpected number of subsections was used in KLIP parameters.")
        if psfparams_dict['minmove'] != str(movement):
            raise Exception(f"Unexpected minimum movement was used in KLIP parameters.")
        if psfparams_dict['numbasis'] != f'{numbasis}/1':
            raise Exception(f"Unexpected numbasis was used in KLIP parameters.")
        
    klip_kwargs={"numbasis":numbasis}
    # Try passing only science frames
    result = do_psf_subtraction(mock_sci,
                                fileprefix='test_sci_only_dataset',
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs)
    
    # Should choose ADI
    for frame in result:
        if not frame.pri_hdr['KLIP_ALG'] == 'ADI':
            raise Exception(f"Chose {frame.pri_hdr['KLIP_ALG']} instead of 'ADI' mode when provided 2 science images and no references.")

    klip_kwargs={"numbasis":numbasis}
    # pass only reference frames (should fail)
    with pytest.raises(UserWarning):
        _ = do_psf_subtraction(mock_ref,
                                fileprefix='test_ref_only_dataset',
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs)

def test_psf_sub_ADI():
    """Tests that psf subtraction step correctly identifies an ADI dataset (multiple rolls, no references), 
    that overall counts decrease, that the KLIP result matches the analytical expectation, and that the 
    output data shape is correct.
    """

    numbasis = [1]
    rolls = [45,-45]
    mock_sci,mock_ref = create_psfsub_dataset(2,0,rolls,
                                              st_amp=st_amp,
                                              noise_amp=noise_amp,
                                              pl_contrast=pl_contrast)

    klip_kwargs={"numbasis":numbasis}
    mock_sci.all_dq[:,55,55] = 1  # This should become flagged bc all science data after derotation will be flagged
    mock_sci.all_dq[:,55,45] = 1  # This should become flagged bc all science data after derotation will be flagged
    mock_sci.all_dq[:,18,30] = 1  # This should not become flagged
    mock_sci.all_dq[0,60,60] = 1  # This should not become flagged 
    mock_sci.all_dq[1,75,75] = 1  # This should not become flagged
    
    expected_data_shape = (1,len(numbasis),*mock_sci[0].data.shape)
    expected_err_shape = (1,1,len(numbasis),*mock_sci[0].data.shape)
    expected_dqs = np.zeros(expected_data_shape)
    expected_dqs[:,:,57,49:51] = 1
    # expected_dqs[:,:,94,5] = 1
    expected_errs = np.full(expected_err_shape,np.nan)

    result = do_psf_subtraction(mock_sci,reference_star_dataset=mock_ref,
                                fileprefix='test_ADI',
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs)

    # TODO: Do derotation with pyklip to make sure that's not the reason for the difference
    analytical_result = shift((rotate(mock_sci[0].data - mock_sci[1].data,-rolls[0],reshape=False,cval=0) + rotate(mock_sci[1].data - mock_sci[0].data,-rolls[1],reshape=False,cval=0)) / 2,
                              [0.5,0.5],
                              cval=np.nan)
    
    frame = result[0]
    for i,img in enumerate(frame.data):

        # import matplotlib.pyplot as plt

        # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
        # im0 = axes[0].imshow(img,origin='lower')
        # plt.colorbar(im0,ax=axes[0],shrink=0.8)
        # axes[0].scatter(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'])
        # axes[0].set_title(f'PSF Sub Result ({numbasis[i]} KL Modes)')

        # im1 = axes[1].imshow(analytical_result,origin='lower')
        # plt.colorbar(im1,ax=axes[1],shrink=0.8)
        # axes[1].scatter(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'])
        # axes[1].set_title('Analytical result')

        # diff = img - analytical_result
        # im2 = axes[2].imshow(diff,origin='lower')
        # plt.colorbar(im2,ax=axes[2],shrink=0.8)
        # axes[2].scatter(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'])
        # axes[2].set_title('Difference')

        # fig.suptitle('ADI')

        # plt.show()
        # plt.close()
    
        # Overall counts should decrease        
        if not np.nansum(mock_sci[0].data) > np.nansum(frame.data):
            raise Exception(f"ADI subtraction resulted in increased counts for frame {i}.")
                
        # Result should match analytical result for first KL mode       
        if np.nanmax(np.abs(frame.data[0] - analytical_result)) > np.nanmax(analytical_result) * rel_tolerance:
            raise Exception(f"Relative difference between ADI result and analytical result is greater then 5%.")
        
        # Check shape of output dq & err arrays
        assert result.all_dq[0,0,57,50] == 1
        assert result.all_err == pytest.approx(expected_errs)

    # Check expected data shape
    if not result.all_data.shape == expected_data_shape:
        raise Exception(f"Result data shape was {result.all_data.shape} instead of expected {expected_data_shape} after ADI subtraction.")

    # Parse PSFPARAM header string
    psfparams = frame.ext_hdr['PSFPARAM'].split(',')
    psfparams_dict = {}
    for pair in psfparams:
        param, val = pair.strip().split('=')
        psfparams_dict[param] = val

    # Check values
    if psfparams_dict['mode'] != 'ADI':
        raise Exception(f"Chose {psfparams_dict['mode']} instead of 'ADI' when provided 2 science images and no references.")
    if psfparams_dict['annuli'] != '1':
        raise Exception(f"Unexpected number of annuli was used in KLIP parameters.")
    if psfparams_dict['subsect'] != '1':
        raise Exception(f"Unexpected number of subsections was used in KLIP parameters.")
    if psfparams_dict['minmove'] != '1':
        raise Exception(f"Unexpected minimum movement was used in KLIP parameters.")
    if psfparams_dict['numbasis'] != f'{numbasis}/1':
        raise Exception(f"Unexpected numbasis was used in KLIP parameters.")

def test_psf_sub_RDI(): 
    """Tests that psf subtraction step correctly identifies an RDI dataset (single roll, 1 or more references), 
    that overall counts decrease, that the KLIP result matches the analytical expectation, and that the 
    output data shape is correct.
    """
    numbasis = [1]
    rolls = [13,0]

    mock_sci,mock_ref = create_psfsub_dataset(1,1,rolls,ref_psf_spread=1.,
                                centerxy=(49.5,49.5),
                                pl_contrast=pl_contrast,
                                noise_amp=noise_amp,
                                st_amp=st_amp
                                )

    klip_kwargs={"numbasis":numbasis}
                                
    result = do_psf_subtraction(mock_sci,
                                reference_star_dataset=mock_ref,
                                fileprefix='test_RDI',
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs
                                )
    analytical_result = rotate(mock_sci[0].data - mock_ref[0].data,-rolls[0],reshape=False,cval=np.nan)
    
    for i,frame in enumerate(result):

        mask = create_circular_mask(frame.data.shape[-2:],r=iwa_pix,center=(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY']))
        masked_frame = np.where(mask,np.nan,frame.data[0])

        # import matplotlib.pyplot as plt

        # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
        # im0 = axes[0].imshow(mock_sci[0].data,origin='lower')
        # plt.colorbar(im0,ax=axes[0],shrink=0.8)
        # axes[0].scatter(mock_sci[0].ext_hdr['STARLOCX'],mock_sci[0].ext_hdr['STARLOCY'])
        # axes[0].set_title(f'Sci Input')

        # im1 = axes[1].imshow(mock_ref[0].data,origin='lower')
        # plt.colorbar(im1,ax=axes[1],shrink=0.8)
        # axes[1].scatter(mock_ref[0].ext_hdr['STARLOCX'],mock_ref[0].ext_hdr['STARLOCY'])
        # axes[1].set_title('Ref Input')

        # im2 = axes[2].imshow(mock_sci[0].data - mock_ref[0].data,origin='lower')
        # plt.colorbar(im2,ax=axes[2],shrink=0.8)
        # axes[2].scatter(mock_sci[0].ext_hdr['STARLOCX'],mock_sci[0].ext_hdr['STARLOCY'])
        # axes[2].set_title('Difference')

        # fig.suptitle('Inputs')
        # plt.show()
        # plt.close()

        # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
        # im0 = axes[0].imshow(frame.data[0] - np.nanmedian(frame.data[0]),origin='lower')
        # plt.colorbar(im0,ax=axes[0],shrink=0.8)
        # axes[0].scatter(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'])
        # axes[0].set_title(f'PSF Sub Result ({numbasis[i]} KL Modes, Median Subtracted)')

        # im1 = axes[1].imshow(analytical_result,origin='lower')
        # plt.colorbar(im1,ax=axes[1],shrink=0.8)
        # axes[1].scatter(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'])
        # axes[1].set_title('Analytical result')

        # #norm = LogNorm(vmin=1e-8, vmax=1, clip=False)
        # im2 = axes[2].imshow(frame.data[0] - np.nanmedian(frame.data[0]) - analytical_result,
        #                      origin='lower',norm=None)
        # plt.colorbar(im2,ax=axes[2],shrink=0.8)
        # axes[2].scatter(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'])
        # axes[2].set_title('Difference')

        # fig.suptitle('RDI Result')

        # plt.show()
        # plt.close()
        
        # Overall counts should decrease        
        if not np.nansum(mock_sci[0].data) > np.nansum(frame.data[0]):
            raise Exception(f"RDI subtraction resulted in increased counts for frame {i}.")
        
        # The step should choose mode RDI based on having 1 roll and 1 reference.
        if not frame.pri_hdr['KLIP_ALG'] == 'RDI':
            raise Exception(f"Chose {frame.pri_hdr['KLIP_ALG']} instead of 'RDI' mode when provided 1 science image and 1 reference.")
        
        # Frame should match analytical result outside of the IWA (after correcting for the median offset)
        if not np.nanmax(np.abs((masked_frame - np.nanmedian(frame.data[0])) - analytical_result)) < 1e-5:
            raise Exception("RDI subtraction did not produce expected analytical result.")
    
    # Check expected data shape
    expected_data_shape = (1,len(numbasis),*mock_sci[0].data.shape)
    if not result.all_data.shape == expected_data_shape:
        raise Exception(f"Result data shape was {result.all_data.shape} instead of expected {expected_data_shape} after RDI subtraction.")
    
def test_psf_sub_ADIRDI():
    """Tests that psf subtraction step correctly identifies an ADI+RDI dataset (multiple rolls, 1 or more references), 
    that overall counts decrease, that the KLIP result matches the analytical expectation for 1 KL mode, and that the 
    output data shape is correct.
    """

    numbasis = [1,2,3,4]
    rolls = [13,-13,+26,-26]
    mock_sci,mock_ref = create_psfsub_dataset(2,2,rolls,
                                              st_amp=st_amp,
                                              noise_amp=noise_amp,
                                              pl_contrast=pl_contrast)
    

    analytical_result1 = (rotate(mock_sci[0].data - (mock_sci[1].data/2+mock_ref[0].data/2),-rolls[0],reshape=False,cval=0) + rotate(mock_sci[1].data - (mock_sci[0].data/2+mock_ref[0].data/2),-rolls[1],reshape=False,cval=0)) / 2
    analytical_result2 = (rotate(mock_sci[0].data - mock_sci[1].data,-rolls[0],reshape=False,cval=0) + rotate(mock_sci[1].data - mock_sci[0].data,-rolls[1],reshape=False,cval=0)) / 2                         
    analytical_results = [analytical_result1,analytical_result2]
    
    klip_kwargs={"numbasis":numbasis}
                                
    result = do_psf_subtraction(mock_sci,reference_star_dataset=mock_ref,
                                fileprefix='test_ADI+RDI',
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs)
    
    for i,frame in enumerate(result):

        
        mask = create_circular_mask(frame.data.shape[-2:],r=iwa_pix,center=(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY']))
        masked_frame = np.where(mask,np.nan,frame.data)

        # import matplotlib.pyplot as plt

        # fig,axes = plt.subplots(1,3,sharey=True,layout='constrained',figsize=(12,3))
        # im0 = axes[0].imshow(frame.data - np.nanmedian(frame.data),origin='lower')
        # plt.colorbar(im0,ax=axes[0],shrink=0.8)
        # axes[0].scatter(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'])
        # axes[0].set_title(f'PSF Sub Result ({numbasis[i]} KL Modes, Median Subtracted)')

        # im1 = axes[1].imshow(analytical_results[0],origin='lower')
        # plt.colorbar(im1,ax=axes[1],shrink=0.8)
        # axes[1].scatter(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'])
        # axes[1].set_title('Analytical result')

        # im2 = axes[2].imshow(masked_frame - np.nanmedian(frame.data) - analytical_results[0],origin='lower')
        # plt.colorbar(im2,ax=axes[2],shrink=0.8)
        # axes[2].scatter(frame.ext_hdr['STARLOCX'],frame.ext_hdr['STARLOCY'])
        # axes[2].set_title('Difference')

        # fig.suptitle('ADI+RDI')

        # plt.show()
        # plt.close()

        # Overall counts should decrease        
        if not np.nansum(mock_sci[0].data) > np.nansum(frame.data):
            raise Exception(f"ADI+RDI subtraction resulted in increased counts for frame {i}.")
        
        # Corgidrp should know to choose ADI+RDI mode
        if not frame.pri_hdr['KLIP_ALG'] == 'ADI+RDI':
            raise Exception(f"Chose {frame.pri_hdr['KLIP_ALG']} instead of 'ADI+RDI' mode when provided 2 science images and 1 reference.")
        
        # Frame should match analytical result outside of the IWA (after correcting for the median offset) for KL mode 1
        # This fails with a tolerance of 1e-5, but passes with 2e-5 after some minor changes to the mock data generation
        # that cause a small numerical difference.
        if i==0:
            diff = np.nanmax(np.abs((masked_frame - np.nanmedian(frame.data)) - analytical_results[i]))
            if not diff < 2e-5:
                raise Exception(f"ADI+RDI subtraction did not produce expected analytical result. Max difference: {diff}")
                
    # Check expected data shape
    expected_data_shape = (1,len(numbasis),*mock_sci[0].data.shape)
    if not result.all_data.shape == expected_data_shape:
        raise Exception(f"Result data shape was {result.all_data.shape} instead of expected {expected_data_shape} after ADI+RDI subtraction.")

def test_psf_sub_explicit_klip_kwargs():
    """Tests that psf subtraction step correctly identifies an ADI dataset (multiple rolls, no references), 
    that overall counts decrease, that the KLIP result matches the analytical expectation, and that the 
    output data shape is correct.
    """

    numbasis = [1]
    rolls = [270+13,270-13]
    mock_sci,mock_ref = create_psfsub_dataset(2,0,rolls,
                                              st_amp=st_amp,
                                              noise_amp=noise_amp,
                                              pl_contrast=pl_contrast)

    result = do_psf_subtraction(mock_sci,reference_star_dataset=mock_ref,
                                fileprefix='test_ADI_explicit_klip_kwargs',
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                mode='ADI',
                                annuli=2,
                                subsections=2,
                                movement=2,
                                numbasis=numbasis)

    frame = result[0]
    
    # Parse PSFPARAM header string
    psfparams = frame.ext_hdr['PSFPARAM'].split(',')
    psfparams_dict = {}
    for pair in psfparams:
        param, val = pair.strip().split('=')
        psfparams_dict[param] = val

    # Check values
    if psfparams_dict['mode'] != 'ADI':
        raise Exception(f"Chose {psfparams_dict['mode']} instead of 'ADI' when provided 2 science images and no references.")
    if psfparams_dict['annuli'] != '2':
        raise Exception(f"Unexpected number of annuli was used in KLIP parameters.")
    if psfparams_dict['subsect'] != '2':
        raise Exception(f"Unexpected number of subsections was used in KLIP parameters.")
    if psfparams_dict['minmove'] != '2':
        raise Exception(f"Unexpected minimum movement was used in KLIP parameters.")
    if psfparams_dict['numbasis'] != f'{numbasis}/1':
        raise Exception(f"Unexpected numbasis was used in KLIP parameters.")

def test_psf_sub_badmode():
    """Tests that psf subtraction step fails correctly if an unconfigured mode is supplied (e.g. SDI).
    """

    numbasis = [1,2,3,4]
    rolls = [13,-13,0]
    mock_sci,mock_ref = create_psfsub_dataset(2,1,rolls,
                                              st_amp=st_amp,
                                              noise_amp=noise_amp,
                                              pl_contrast=pl_contrast)
    
    klip_kwargs={"numbasis":numbasis,
                 "mode" : 'SDI'}
    with pytest.raises(Exception):
        _ = do_psf_subtraction(mock_sci,reference_star_dataset=mock_ref,
                                fileprefix='test_SDI',
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs)
    
def test_psf_sub_nandata():
    """Tests that psf subtraction step fails correctly if nans are present in the data.
    """

    numbasis = [1,2,3,4]
    rolls = [13,-13,0]
    klip_kwargs={"numbasis":numbasis,
                 "mode" : 'ADI+RDI'}
    
    # Test nan in science data
    mock_sci,mock_ref = create_psfsub_dataset(2,1,rolls,
                                              st_amp=st_amp,
                                              noise_amp=noise_amp,
                                              pl_contrast=pl_contrast)
    mock_sci.all_data[0,1,:] = np.nan
    
    with pytest.raises(Exception):
        _ = do_psf_subtraction(mock_sci,reference_star_dataset=mock_ref,
                                fileprefix='test_nandata',
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs)

    # Test nan in ref data
    mock_sci,mock_ref = create_psfsub_dataset(2,1,rolls,
                                              st_amp=st_amp,
                                              noise_amp=noise_amp,
                                              pl_contrast=pl_contrast)
    mock_ref.all_data[0,1,:] = np.nan
    with pytest.raises(Exception):
        _ = do_psf_subtraction(mock_sci,reference_star_dataset=mock_ref,
                                fileprefix='test_nandata',
                                measure_klip_thrupt=False,
                                measure_1d_core_thrupt=False,
                                **klip_kwargs)

    
if __name__ == '__main__':  
    # test_pyklipdata_ADI()
    # test_pyklipdata_RDI()
    # test_pyklipdata_ADIRDI()
    # test_pyklipdata_badtelescope()
    # test_pyklipdata_badinstrument()
    # test_pyklipdata_badcfamname()
    # test_pyklipdata_notdataset()
    # test_pyklipdata_badimgshapes()
    # test_pyklipdata_multiplepixscales()

    # test_nanflags_2D()
    # test_nanflags_3D() 
    # test_nanflags_mixed_dqvals()
    # test_flagnans_2D()
    # test_flagnans_3D()
    # test_flagnans_flagval2()

    # test_psf_sub_split_dataset()
    # test_psf_sub_explicit_klip_kwargs()

    test_psf_sub_ADI()
    # test_psf_sub_RDI()
    # test_psf_sub_ADIRDI()
    # test_psf_sub_nandata()
    # test_psf_sub_badmode()

    
