import numpy as np
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.io import fits
from matplotlib.ticker import MaxNLocator

from corgidrp.detector import Metadata


def plot_corner(ax, x, y, ha, va, xytext):
    """Plot marker with coordinates label.

    Args:
    ax: ax return of plt.subplots
    x: x coord of offset
    y: y coord of offset
    ha: horizontal placement of corner label
    va: vertical placement of corner label
    xytext: xy coords of text
    """
    ax.scatter(x, y, s=2, c='r', marker='s')
    ax.annotate(f'({x}, {y})', (x, y), size=7, ha=ha, va=va, xytext=xytext,
                textcoords='offset pixels')


def plot_im_corners(ax):
    """Plot corners of image region

    Args:
    ax: ax return of plt.subplots
    """
    rows, cols, image_r0c0 = meta._unpack_geom('image')
    image_r1c1 = (image_r0c0[0]+rows-1, image_r0c0[1]+cols-1)

    plot_corner(ax, image_r0c0[1], image_r0c0[0], 'left', 'bottom', (5, 5))
    plot_corner(ax, image_r0c0[1], image_r1c1[0], 'left', 'top', (5, -5))
    plot_corner(ax, image_r1c1[1], image_r1c1[0], 'right', 'top', (-5, -5))
    plot_corner(ax, image_r1c1[1], image_r0c0[0], 'right', 'bottom', (-5, 5))


class Formatter(object):
    """Round cursor coordinates to integers."""
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        return 'x=%i, y=%i' % (np.round(x), np.round(y))


if __name__ == '__main__':
    meta = Metadata()

    # Get masks of all regions
    image_m = meta.mask('image')
    prescan_m = meta.mask('prescan')
    parallel_overscan_m = meta.mask('parallel_overscan')
    serial_overscan_m = meta.mask('serial_overscan')

    # Assign values to each region
    values = {
        'image': 1,
        'prescan': 0.75,
        'parallel_overscan': 0.5,
        'serial_overscan': 0.25,
        'shielded/unused': 0.
    }

    # Stack masks
    mask = (
        image_m*values['image']
        + prescan_m*values['prescan']
        + parallel_overscan_m*values['parallel_overscan']
        + serial_overscan_m*values['serial_overscan']
    )

    # Plot
    origin = 'lower'  # Use origin = 'lower' to put (0, 0) at the bottom left

    # Plot image area (optional)
    plot_fits = True
    if plot_fits:
        fits_im = np.ones((1024,1024))
        fig_fits, ax_fits = plt.subplots()
        ax_fits.imshow(np.log(fits_im+10), origin=origin, cmap='Greys')
        ax_fits.set_title('SCI Frame')
        # Plot corners
        plot_im_corners(ax_fits)

        # Format plot
        ax_fits.format_coord = Formatter(fits_im)
        ax_fits.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_fits.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot masks
    fig, ax = plt.subplots()
    ax.set_title('SCI Frame Geometry')
    im = ax.imshow(mask, origin=origin)
    # Plot corners
    plot_im_corners(ax)

    # Format plot
    ax.format_coord = Formatter(im)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Make legend
    colors = {region: im.cmap(im.norm(value))
              for region, value in values.items()}
    patches = {mpatches.Patch(color=color, label=f'{region}')
               for region, color in colors.items()}
    plt.legend(handles=patches, loc='lower left')

    plt.show()
