from corgidrp import data
from corgidrp.l3_to_l4 import northup
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import glob
import os

def test_northup(save_derot_dataset=False):
    """
    test northup.py

    Returns:
    Fits containing the original mock image and the derotated image, with the roll angle recorded
    """
    
    dirname = 'test_data/'
    mockname = 'mock_northup.fits'

    mockfilepath = glob.glob(dirname+mockname)
    if not mockfilepath:
        raise FileNotFoundError(f"No mock data found at {dirname+mockname}")
    
    input_dataset = data.Dataset(mockfilepath)
    derot_dataset = northup(input_dataset)
    if save_derot_dataset is True:
    	derot_dataset.save(dirname,[mockname.split('.fits')[0]+'_derotated.fits'])

    im_input = input_dataset[0].data
    roll_angle = input_dataset[0].pri_hdr['ROLL']
    im_derot = derot_dataset[0].data

    output_data = np.array([im_input,im_derot])
    output_hdu = fits.PrimaryHDU(output_data)
    output_hdu.header['ROLL'] = roll_angle
    return output_hdu

def compare_northup(output_hdu):
    im_input = output_hdu.data[0]
    im_derot = output_hdu.data[1]
    roll_angle = output_hdu.header['ROLL']
    
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(8,5))
    
    ax0.set_title('Original Mock Data')
    ax0.imshow(im_input,origin='lower')

    ax1.set_title(f'Derotated Mock Data\n by {-roll_angle}deg counterclockwise')
    ax1.imshow(im_derot,origin='lower')

    outdir = 'test_data/'
    outfilename = 'compare_northup.png'

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    plt.savefig(outdir+outfilename) 
    
    print(f"Comparison figure saved at {outdir+outfilename}")

if __name__ == "__main__":
    output_hdu = test_northup()
    compare_northup(output_hdu)
