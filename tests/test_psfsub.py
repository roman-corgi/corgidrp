from corgidrp.mocks import create_psfsub_dataset
from corgidrp.l3_to_l4 import do_psf_subtraction
import numpy as np


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
