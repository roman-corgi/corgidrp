"""
Module to hold input-checking functions to minimize repetition

Copied over from the II&T pipeline
"""
import numbers

import numpy as np

class CheckException(Exception):
    pass

# String check support
string_types = (str, bytes)

# Int check support
int_types = (int, np.integer)

def _checkname(vname):
    """
    Internal check that we can use vname as a string for printing

    Args:
    vname: str
        variable name
    """
    if not isinstance(vname, string_types):
        raise CheckException('vname must be a string when fed to check ' + \
                             'functions')
    pass


def _checkexc(vexc):
    """
    Internal check that we can raise from the vexc object

    Args:
    vexc: str
        exception name
    """
    if not isinstance(vexc, type): # pre-check it is class-like
        raise CheckException('vexc must be a Exception, or an object ' + \
                             'descended from one when fed to check functions')
    if not issubclass(vexc, Exception):
        raise CheckException('vexc must be a Exception, or an object ' + \
                             'descended from one when fed to check functions')
    pass


def real_positive_scalar(var, vname, vexc):
    """
    Checks whether an object is a real positive scalar.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be real')
    if var <= 0:
        raise vexc(vname + ' must be positive')
    return var


def real_array(var, vname, vexc):
    """
    Checks whether an object is a real numpy array, or castable to one.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.asarray(var) # cast to array
    if len(var.shape) == 0:
        raise vexc(vname + ' must have length > 0')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be a real array')
    # skip 'c' as we don't want complex; rest are non-numeric
    if not var.dtype.kind in ['b', 'i', 'u', 'f']:
        raise vexc(vname + ' must be a real numeric type to be real')
    return var


def oneD_array(var, vname, vexc):
    """
    Checks whether an object is a 1D numpy array, or castable to one.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.asarray(var) # cast to array
    if len(var.shape) != 1:
        raise vexc(vname + ' must be a 1D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 1D array')
    return var


def twoD_array(var, vname, vexc):
    """
    Checks whether an object is a 2D numpy array, or castable to one.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.asarray(var) # cast to array
    if len(var.shape) != 2:
        raise vexc(vname + ' must be a 2D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 2D array')
    return var


def twoD_square_array(var, vname, vexc):
    """
    Checks whether an object is a 2D square array_like.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.asarray(var) # cast to array
    if len(var.shape) != 2:
        raise vexc(vname + ' must be a 2D array')
    else: # is 2-D
        if not var.shape[0] == var.shape[1]:
            raise vexc(vname + ' must be a square 2D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex square 2D array')
    return var


def threeD_array(var, vname, vexc):
    """
    Checks whether an object is a 3D numpy array, or castable to one.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    var = np.asarray(var) # cast to array
    if len(var.shape) != 3:
        raise vexc(vname + ' must be a 3D array')
    if (not np.isrealobj(var)) and (not np.iscomplexobj(var)):
        raise vexc(vname + ' must be a real or complex 3D array')
    return var



def real_scalar(var, vname, vexc):
    """
    Checks whether an object is a real scalar.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be real')
    return var


def real_nonnegative_scalar(var, vname, vexc):
    """
    Checks whether an object is a real nonnegative scalar.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not np.isrealobj(var):
        raise vexc(vname + ' must be real')
    if var < 0:
        raise vexc(vname + ' must be nonnegative')
    return var


def positive_scalar_integer(var, vname, vexc):
    """
    Checks whether an object is a positive scalar integer.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not isinstance(var, int_types):
        raise vexc(vname + ' must be integer')
    if var <= 0:
        raise vexc(vname + ' must be positive')
    return var


def nonnegative_scalar_integer(var, vname, vexc):
    """
    Checks whether an object is a nonnegative scalar integer.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not isinstance(var, int_types):
        raise vexc(vname + ' must be integer')
    if var < 0:
        raise vexc(vname + ' must be nonnegative')
    return var


def scalar_integer(var, vname, vexc):
    """
    Checks whether an object is a scalar integer (no sign dependence).

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, numbers.Number):
        raise vexc(vname + ' must be scalar')
    if not isinstance(var, int_types):
        raise vexc(vname + ' must be integer')
    return var


def string(var, vname, vexc):
    """
    Checks whether an object is a string.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, string_types):
        raise vexc(vname + ' must be a string')
    return var


def boolean(var, vname, vexc):
    """
    Checks whether an object is a bool.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, bool):
        raise vexc(vname + ' must be a bool')
    return var


def dictionary(var, vname, vexc):
    """
    Checks whether an object is a dictionary.

    Arguments:
     var: variable to check
     vname: string to output in case of error for debugging
     vexc: Exception to raise in case of error for debugging

    Returns:
     returns var

    """
    _checkname(vname)
    _checkexc(vexc)

    if not isinstance(var, dict):
        raise vexc(vname + ' must be a dict')
    return var
