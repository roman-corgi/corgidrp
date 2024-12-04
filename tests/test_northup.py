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
    if not mockfilepath:
        raise FileNotFoundError(f"No mock data {mockname} found")

    # running northup function
    input_dataset = data.Dataset(glob(mockfilepath))
    derot_dataset = northup(input_dataset)
    # save fits file
    if save_derot_dataset is True:
        derot_dataset.save(dirname,[mockname.split('.fits')[0]+'_derotated.fits'])

    # read the original mock file and derotated file
    im_input = input_dataset[0].data
    roll_angle = input_dataset[0].pri_hdr['ROLL']
    im_derot = derot_dataset[0].data
    dq_input = input_dataset[0].dq
    dq_derot = derot_dataset[0].dq
    # the location for test, where the mock file has 1 in DQ
    x_value1 = input_dataset[0].dq_hdr['X_1VAL']
    y_value1 = input_dataset[0].dq_hdr['Y_1VAL']

    # check if rotation works properly
    assert(im_input[y_value1,x_value1] != im_derot[y_value1,x_value1])
    assert(dq_input[y_value1,x_value1] != dq_derot[y_value1,x_value1])
    
    # check if the derotated DQ frame has no integer values (except NaN)
    non_integer_mask = (~np.isnan(dq_derot)) & (dq_derot % 1 != 0)
    non_integer_indices = np.argwhere(non_integer_mask)
    assert(len(non_integer_indices) == 0)

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

if __name__ == "__main__":
    test_northup()
