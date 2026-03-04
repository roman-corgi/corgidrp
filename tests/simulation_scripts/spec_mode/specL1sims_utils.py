"""
Utility functions for generating end-to-end (e2e) test data files.

This module contains common utility functions needed by various e2e test
data generation scripts.
"""

import os
import matplotlib.pyplot as plt


def write_png_and_header(scene, outdir, loc_x, loc_y, output_dim):
    """
    Write a PNG image and header text file for a scene's detector image.

    Parameters
    ----------
    scene : corgisim.Scene
        The scene object containing the image_on_detector attribute
    outdir : str
        Output directory path where files will be saved
    loc_x : int
        X-coordinate location on the detector array
    loc_y : int
        Y-coordinate location on the detector array
    output_dim : int
        Dimension (width/height) of the output image in pixels

    Returns
    -------
    L1_png_fname : str
        Path to the saved PNG file
    header_txt_fname : str
        Path to the saved header text file
    """
    L1_fitsname = os.path.join(outdir, scene.image_on_detector[0].header['FILENAME'])
    L1_png_fname = str.replace(L1_fitsname, '.fits', '.png')
    plt.figure(figsize=(8,6))
    plt.imshow(scene.image_on_detector[1].data[loc_y + 13 - output_dim//2:loc_y + 13 + output_dim//2,
                                               loc_x + 1088 - output_dim//2:loc_x + 1088 + output_dim//2], origin='lower')
    plt.colorbar()
    plt.savefig(L1_png_fname)
    plt.close()

    header_txt_fname = str.replace(L1_fitsname, '.fits', '_header.txt')
    with open(header_txt_fname, 'w') as f:
        f.write(repr(scene.image_on_detector[0].header))

    return L1_png_fname, header_txt_fname
