import numpy as np

def rotation_mueller_matrix(angle):
    '''
    constructs a rotation matrix from a given angle

    Args:
        angle (float): the angle of rotation in degrees
        
    Returns:
        rotation_matrix (np.array) The 4x4 mueller matrix for rotation at the given angle
    '''

    # convert degree to rad
    theta = angle * (np.pi / 180)
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(2*theta), np.sin(2*theta), 0],
        [0,-np.sin(2*theta), np.cos(2*theta), 0],
        [0, 0, 0, 1]
    ])
    return rotation_matrix

def lin_polarizer_mueller_matrix(angle):
    '''
    constructs a linear polarizer matrix from a given angle

    Args:
        angle (float): the polarization angle of the polarizer
        
    Returns:
        pol_matrix (np.array) The 4x4 mueller matrix for a linear polarizer at the given angle
    '''
    # convert degree to rad
    theta = angle * (np.pi / 180)
    cos = np.cos(2 * theta)
    sin = np.sin(2 * theta)
    pol_matrix = 0.5 * np.array([
        [1, cos, sin, 0],
        [cos, cos**2, cos * sin, 0],
        [sin, cos * sin, sin**2, 0],
        [0, 0, 0, 0]
    ])
    return pol_matrix