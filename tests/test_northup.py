from corgidrp import data
from corgidrp.l3_to_l4 import northup
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import os
from glob import glob

def test_northup(save_derot_dataset=False,save_comp_figure=False):
    """
    unit test of the northup function

    Args:
        save_derot_dataset (optional): if you want to save the derotated file at the input directory, turn True
        save_comp_figure (optional): if you want to save a comparison figure of the original mock data and the derotated data

    Returns:
        Fits containing the original mock image and the derotated image, with the roll angle recorded
    """

    # read mock file
    dirname = 'test_data'
    mockname = 'mock_northup.fits'

    mockfilepath = os.path.join(os.path.dirname(__file__),dirname,mockname)
    print(mockfilepath)
    if not mockfilepath:
        raise FileNotFoundError(f"No mock data {mockname} found")

    # running northup function
    input_dataset = data.Dataset(glob(mockfilepath))
    derot_dataset = northup(input_dataset)
    # save fits file
    if save_derot_dataset is True:
        derot_dataset.save(dirname,[mockname.split('.fits')[0]+'_derotated.fits'])

    im_input = input_dataset[0].data
    roll_angle = input_dataset[0].pri_hdr['ROLL']
    im_derot = derot_dataset[0].data

    output_data = np.array([im_input,im_derot])
    output_hdu = fits.PrimaryHDU(output_data)
    output_hdu.header['ROLL'] = roll_angle

    # save comparison figure
    if save_comp_figure is True:

        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(8,5))

        ax0.set_title('Original Mock Data')
        ax0.imshow(im_input,origin='lower')

        ax1.set_title(f'Derotated Mock Data\n by {-roll_angle}deg counterclockwise')
        ax1.imshow(im_derot,origin='lower')

        outdir = os.path.join(os.path.dirname(__file__),dirname)
        os.makedirs(outdir, exist_ok=True)
        outfilename = 'compare_northup.png'
        
        plt.savefig(os.path.join(outdir,outfilename))

        print(f"Comparison figure saved at {dirname+outfilename}")

    return output_hdu

if __name__ == "__main__":
    test_northup()
