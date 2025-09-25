'''
file containing helper functions for polarimetric step functions
'''

import numpy as np

def rotation_mueller_matrix(angle):
    '''
    constructs a rotation matrix from a given angle

    Args:
        angle (float): the angle of rotation in degrees
    returns:
        The rotation matrix for the given angle, a 4x4 numpy array 
    '''

    # convert degree to rad
    theta = angle * (np.pi / 180)
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(2*theta), np.sin(2*theta), 0],
        [0,-np.sin(2*theta), np.cos(2*theta), 0],
        [0, 0, 0, 1]
    ])