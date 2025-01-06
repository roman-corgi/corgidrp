from corgidrp.mocks import create_psfsub_dataset
from corgidrp.l3_to_l4 import do_psf_subtraction
from corgidrp.data import PyKLIPDataset
import pytest
import numpy as np

def test_pyklipdata_ADI():

    rolls = [0,90]
    # Init with center shifted by 1 pixel
    mock_sci,mock_ref = create_psfsub_dataset(2,0,rolls,centerxy=(30.5,30.5))

    pyklip_dataset = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

    # Check image is centered properly
    for i,image in enumerate(pyklip_dataset._input):

        assert mock_sci.all_data[i,1:,1:] == pytest.approx(image[:-1,:-1]), f"Frame {i} centered improperly."

    # Check roll assignments and filenames match up for sci dataset
    for r,roll in enumerate(pyklip_dataset._PAs):
        assert pyklip_dataset._filenames[r] == f'MOCK_sci_roll{roll}.fits_INT1', f"Incorrect roll assignment for frame {r}."
    

    # Check ref library is None
    assert pyklip_dataset.psflib is None, "pyklip_dataset.psflib is not None, even though no reference dataset was provided."

def test_pyklipdata_RDI():
    # TODO: 
        # checks for reference library
        # what is pyklip_dataset.psflib.isgoodpsf for?

    rolls = [45,180]
    # Init with center shifted by 1 pixel
    mock_sci,mock_ref = create_psfsub_dataset(1,1,rolls,centerxy=(30.5,30.5))

    pyklip_dataset = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

    # Check image is centered properly
    for i,image in enumerate(pyklip_dataset._input):

        assert mock_sci.all_data[i,1:,1:] == pytest.approx(image[:-1,:-1]), f"Frame {i} centered improperly."

    # Check roll assignments and filenames match up for sci dataset
    for r,roll in enumerate(pyklip_dataset._PAs):
        assert pyklip_dataset._filenames[r] == f'MOCK_sci_roll{roll}.fits_INT1', f"Incorrect roll assignment for frame {r}."
    

    # Check ref library
    
def test_pyklipdata_ADIRDI():
    # TODO: checks for reference library

    rolls = [45,-45,180]
    # Init with center shifted by 1 pixel
    mock_sci,mock_ref = create_psfsub_dataset(2,1,rolls,centerxy=(30.5,30.5))

    pyklip_dataset = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

    # Check image is centered properly
    for i,image in enumerate(pyklip_dataset._input):

        assert mock_sci.all_data[i,1:,1:] == pytest.approx(image[:-1,:-1]), f"Frame {i} centered improperly."

    # Check roll assignments and filenames match up for sci dataset
    for r,roll in enumerate(pyklip_dataset._PAs):
        assert pyklip_dataset._filenames[r] == f'MOCK_sci_roll{roll}.fits_INT1', f"Incorrect roll assignment for frame {r}."
    

    # Check ref library   

def test_pyklipdata_badtelescope():
    mock_sci,mock_ref = create_psfsub_dataset(1,1,[0,0])
    mock_sci[0].pri_hdr['TELESCOP'] = "HUBBLE"

    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

def test_pyklipdata_badinstrument():
    mock_sci,mock_ref = create_psfsub_dataset(1,1,[0,0])
    mock_sci[0].pri_hdr['INSTRUME'] = "WFI"

    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)


def test_pyklipdata_badcgimode():
    mock_sci,mock_ref = create_psfsub_dataset(1,1,[0,0])
    mock_sci[0].pri_hdr['MODE'] = "SPC"

    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

def test_pyklipdata_notdataset():
    mock_sci,mock_ref = create_psfsub_dataset(1,0,[0])
    mock_ref = []
    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

    mock_sci = []
    mock_ref = None
    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)


def test_pyklipdata_badimgshapes():
    mock_sci,mock_ref = create_psfsub_dataset(2,0,[0,0])
    
    mock_sci[0].data = np.zeros((5,5))
    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

def test_pyklipdata_multiplepixscales():
    mock_sci,mock_ref = create_psfsub_dataset(2,0,[0,0])
    
    mock_sci[0].ext_hdr["PIXSCALE"] = 10
    with pytest.raises(UserWarning):
        _ = PyKLIPDataset(mock_sci,psflib_dataset=mock_ref)

def test_psf_sub_ADI():

    numbasis = [1,2,4]
    rolls = [270+13,270-13]
    mock_sci,mock_ref = create_psfsub_dataset(2,0,rolls)

    result = do_psf_subtraction(mock_sci,mock_ref,
                                numbasis=numbasis,
                                fileprefix='test_ADI')
    
    for i,frame in enumerate(result):

        # import matplotlib.pyplot as plt
        # plt.imshow(frame.data,origin='lower')
        # plt.colorbar()
        # plt.title(f'{frame.ext_hdr["KLIP_ALG"]}, {frame.ext_hdr["KLMODES"]} KL MODE(S)')
        # plt.scatter(frame.ext_hdr['PSFCENTX'],frame.ext_hdr['PSFCENTY'])
        # plt.show()
        # plt.close()
    
        # Overall counts should decrease        
        if not np.nansum(mock_sci[0].data) > np.nansum(frame.data):
            print(f'sum input: {np.sum(mock_sci[0].data)}')
            print(f'sum output: {np.sum(frame.data)}')
            
            raise Exception(f"ADI subtraction resulted in increased counts for frame {i}.")
                
        if not frame.ext_hdr['KLIP_ALG'] == 'ADI':
            raise Exception(f"Chose {frame.ext_hdr['KLIP_ALG']} instead of 'ADI' mode when provided 2 science images and no references.")

    # Check expected data shape
    expected_data_shape = (len(numbasis),*mock_sci[0].data.shape)
    if not result.all_data.shape == expected_data_shape:
        raise Exception(f"Result data shape was {result.all_data.shape} instead of expected {expected_data_shape} after ADI subtraction.")
         
def test_psf_sub_RDI():

    numbasis=[1,2,4]
    rolls = [13,0,90]
    mock_sci,mock_ref = create_psfsub_dataset(1,1,rolls)

    result = do_psf_subtraction(mock_sci,mock_ref,
                                numbasis=numbasis,
                                fileprefix='test_RDI')
    
    for i,frame in enumerate(result):

        # import matplotlib.pyplot as plt
        # plt.imshow(frame.data,origin='lower')
        # plt.colorbar()
        # plt.title(f'{frame.ext_hdr["KLIP_ALG"]}, {frame.ext_hdr["KLMODES"]} KL MODE(S)')
        # plt.scatter(frame.ext_hdr['PSFCENTX'],frame.ext_hdr['PSFCENTY'])
        # plt.show()
        # plt.close()
        
        # Overall counts should decrease        
        if not np.nansum(mock_sci[0].data) > np.nansum(frame.data):
            raise Exception(f"RDI subtraction resulted in increased counts for frame {i}.")
           
        if not frame.ext_hdr['KLIP_ALG'] == 'RDI':
            raise Exception(f"Chose {frame.ext_hdr['KLIP_ALG']} instead of 'RDI' mode when provided 1 science image and 1 reference.")
    
    # Check expected data shape
    expected_data_shape = (len(numbasis),*mock_sci[0].data.shape)
    if not result.all_data.shape == expected_data_shape:
        raise Exception(f"Result data shape was {result.all_data.shape} instead of expected {expected_data_shape} after RDI subtraction.")
    
def test_psf_sub_ADIRDI():

    numbasis = [1,2,4]
    rolls = [13,-13,0,0]
    mock_sci,mock_ref = create_psfsub_dataset(2,1,rolls)

    result = do_psf_subtraction(mock_sci,mock_ref,
                                numbasis=numbasis,
                                fileprefix='test_ADI+RDI')
    
    for i,frame in enumerate(result):

        # import matplotlib.pyplot as plt
        # plt.imshow(frame.data,origin='lower')
        # plt.colorbar()
        # plt.title(f'{frame.ext_hdr["KLIP_ALG"]}, {frame.ext_hdr["KLMODES"]} KL MODE(S)')
        # plt.scatter(frame.ext_hdr['PSFCENTX'],frame.ext_hdr['PSFCENTY'])
        # plt.show()
        # plt.close()

        # Overall counts should decrease        
        if not np.nansum(mock_sci[0].data) > np.nansum(frame.data):
            raise Exception(f"ADI+RDI subtraction resulted in increased counts for frame {i}.")
        
        if not frame.ext_hdr['KLIP_ALG'] == 'ADI+RDI':
            raise Exception(f"Chose {frame.ext_hdr['KLIP_ALG']} instead of 'ADI+RDI' mode when provided 2 science images and 1 reference.")
                
    # Check expected data shape
    expected_data_shape = (len(numbasis),*mock_sci[0].data.shape)
    if not result.all_data.shape == expected_data_shape:
        raise Exception(f"Result data shape was {result.all_data.shape} instead of expected {expected_data_shape} after ADI+RDI subtraction.")

if __name__ == '__main__':                                      
    test_psf_sub_ADI()
    test_psf_sub_RDI()
    test_psf_sub_ADIRDI()
    test_pyklipdata_ADI()
    test_pyklipdata_RDI()
    test_pyklipdata_ADIRDI()
    test_pyklipdata_badtelescope()
    test_pyklipdata_badinstrument()
    test_pyklipdata_badcgimode()
    test_pyklipdata_notdataset()
    test_pyklipdata_badimgshapes()
    test_pyklipdata_multiplepixscales()
