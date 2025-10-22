"""
Unit tests for check.py.
For each check, test with valid and invalid inputs for all three inputs.

Test valid here as well since most other functions rely on these for
error checking
"""
import pytest

import numpy as np

from corgidrp import check


# Invalid values

# string
strlist = [1j, None, (1.,), [5, 5], -1, 0, 1.0]
# real scalar
rslist = [1j, None, (1.,), [5, 5], 'txt']
# real nonnegative scalar
rnslist = [1j, None, (1.,), [5, 5], 'txt', -1]
# real positive scalar
rpslist = [1j, None, (1.,), [5, 5], 'txt', -1, 0]
# real scalar integer
rsilist = [1j, None, (1.,), [5, 5], 'txt', 1.0]
# nonnegative scalar integer
nsilist = [1j, None, (1.,), [5, 5], 'txt', -1, 1.0]
# positive scalar integer
psilist = [1j, None, (1.,), [5, 5], 'txt', -1, 0, 1.0]
# real array
rarraylist = [1j*np.ones((5, 4)), (1+1j)*np.ones((5, 5, 5)), 'foo',
              np.array([[1, 2], [3, 4], [5, 'a']])]
# 1D array
oneDlist = [np.ones((5, 4)), np.ones((5, 5, 5)), 'foo']
# 2D array
twoDlist = [np.ones((5,)), np.ones((5, 5, 5)), [], 'foo']
# 2D square array
twoDsquarelist = [np.ones((5,)), np.ones((5, 5, 5)), np.ones((5, 4)),
                  [], 'foo']
# 3D array
threeDlist = [np.ones((5,)), np.ones((5, 5)), np.ones((2, 2, 2, 2)), [], 'foo']


class TestCheckException(Exception):
    __test__ = False # prevent PytestCollectionWarning
    pass


# real_positive_scalar
def test_real_positive_scalar_good():
    """
    Verify checker works correctly for valid input.

    Type: real positive scalar
    """
    try: 
        check.real_positive_scalar(1, 'rps', TestCheckException)
    except check.CheckException:
        pytest.xfail('real_positive_scalar failed on valid input')
    pass

def test_real_positive_scalar_bad_var():
    """
    fail on invalid variable type.
    
    Type: real positive scalar
    """
    for v0 in rpslist:
        with pytest.raises(TestCheckException):
            check.real_positive_scalar(v0, 'rps', TestCheckException)
            pass
        pass
    pass

def test_real_positive_scalar_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.real_positive_scalar(1, (1,), TestCheckException)
        pass
    pass

def test_real_positive_scalar_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.real_positive_scalar(1, 'rps', 'TestCheckException')
        pass
    pass
# real_nonnegative_scalar
def test_real_nonnegative_scalar_good():
    """
    Verify checker works correctly for valid input.
    
    Type: real nonnegative scalar
    """
    try:
        check.real_nonnegative_scalar(0, 'rps', TestCheckException)
    except check.CheckException:
        pytest.xfail('real_nonnegative_scalar failed on valid input')
    pass

def test_real_nonnegative_scalar_bad_var():
    """
    fail on invalid variable type.
    
    Type: real nonnegative scalar
    """
    for v0 in rnslist:
        with pytest.raises(TestCheckException):
            check.real_nonnegative_scalar(v0, 'rps', TestCheckException)
            pass
        pass
    pass

def test_real_nonnegative_scalar_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.real_nonnegative_scalar(0, (1,), TestCheckException)
        pass
    pass

def test_real_nonnegative_scalar_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.real_nonnegative_scalar(0, 'rps', 'TestCheckException')
        pass
    pass

# real_array
def test_real_array_good():
    """
    Verify checker works correctly for valid input.
    
    Type: real array
    """
    try:
        check.real_array(np.ones((5, 5)), 'real', TestCheckException)
    except check.CheckException:
        pytest.xfail('real_array failed on valid input')
    pass

def test_real_array_bad_var():
    """
    fail on invalid variable type.
    
    Type: real array
    """
    for v0 in rarraylist:
        with pytest.raises(TestCheckException):
            check.real_array(v0, '1D', TestCheckException)
            pass
        pass
    pass

def test_real_array_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.real_array(np.ones((5, 5)), (1,), TestCheckException)
        pass
    pass

def test_real_array_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.real_array(np.ones((5, )), 'rps', 'TestCheckException')
        pass
    pass

# oneD_array
def test_oneD_array_good():
    """
    Verify checker works correctly for valid input.
    
    Type: 1D array
    """
    try:
        check.oneD_array(np.ones((5, )), '1D', TestCheckException)
    except check.CheckException:
        pytest.xfail('oneD_array failed on valid input')
    pass

def test_oneD_array_bad_var():
    """
    fail on invalid variable type.

    Type: 1D array
    """
    for v0 in oneDlist:
        with pytest.raises(TestCheckException):
            check.oneD_array(v0, '1D', TestCheckException)
            pass
        pass
    pass

def test_oneD_array_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.oneD_array(np.ones((5, )), (1,), TestCheckException)
        pass
    pass

def test_oneD_array_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.oneD_array(np.ones((5, )), 'rps', 'TestCheckException')
        pass
    pass

# twoD_array
def test_twoD_array_good():
    """
    Verify checker works correctly for valid input.
    
    Type: 2D array
    """
    try:
        check.twoD_array(np.ones((5, 5)), '2d', TestCheckException)
    except check.CheckException:
        pytest.xfail('twoD_array failed on valid input')
    pass

def test_twoD_array_bad_var():
    """
    fail on invalid variable type.
    
    Type: 2D array
    """
    for v0 in twoDlist:
        with pytest.raises(TestCheckException):
            check.twoD_array(v0, '2d', TestCheckException)
            pass
        pass
    pass

def test_twoD_array_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.twoD_array(np.ones((5, 5)), (1,), TestCheckException)
        pass
    pass

def test_twoD_array_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.twoD_array(np.ones((5, 5)), 'rps', 'TestCheckException')
        pass
    pass

# twoD_square_array
def test_twoD_square_array_good():
    """
    Verify checker works correctly for valid input.

    Type: 2D array
    """
    try:
        check.twoD_array(np.ones((5, 5)), '2d', TestCheckException)
    except check.CheckException:
        pytest.xfail('twoD_square_array failed on valid input')
    pass

def test_twoD_square_array_bad_var():
    """
    fail on invalid variable type.

    Type: 2D array
    """
    for v0 in twoDsquarelist:
        with pytest.raises(TestCheckException):
            check.twoD_square_array(v0, '2d', TestCheckException)
            pass
        pass
    pass

def test_twoD_square_array_bad_var_shape():
    """
    fail on invalid variable type.
    
    Type: 2D square array
    """
    for v0 in [np.ones((5, 4)), np.ones((4, 6))]:
        with pytest.raises(TestCheckException):
            check.twoD_square_array(v0, '2d', TestCheckException)
            pass
        pass
    pass

def test_twoD_square_array_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.twoD_square_array(np.ones((5, 5)), (1,), TestCheckException)
        pass
    pass

def test_twoD_square_array_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.twoD_square_array(np.ones((5, 5)), 'rps',
                            'TestCheckException')
        pass
    pass

# threeD_array
def test_threeD_array_good():
    """
    Verify checker works correctly for valid input.

    Type: 3D array
    """
    try:
        check.threeD_array(np.ones((5, 5, 2)), '3d', TestCheckException)
    except check.CheckException:
        pytest.xfail('threeD_array failed on valid input')
    pass

def test_threeD_array_bad_var():
    """
    fail on invalid variable type.

    Type: 3D array
    """
    for v0 in threeDlist:
        with pytest.raises(TestCheckException):
            check.threeD_array(v0, '3d', TestCheckException)
            pass
        pass
    pass

def test_threeD_array_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.threeD_array(np.ones((5, 5, 2)), (1,), TestCheckException)
        pass
    pass

def test_threeD_array_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.threeD_array(np.ones((5, 5, 2)), 'rps', 'TestCheckException')
        pass
    pass

# real_scalar
def test_real_scalar_good():
    """
    Verify checker works correctly for valid input.

    Type: real scalar
    """
    try:
        check.real_scalar(1, 'rs', TestCheckException)
    except check.CheckException:
        pytest.xfail('real_scalar failed on valid input')
    pass

def test_real_scalar_bad_var():
    """
    fail on invalid variable type.

    Type: real scalar
    """
    for v0 in rslist:
        with pytest.raises(TestCheckException):
            check.real_scalar(v0, 'rs', TestCheckException)
            pass
        pass
    pass

def test_real_scalar_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.real_scalar(1, (1,), TestCheckException)
        pass
    pass

def test_real_scalar_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.real_scalar(1, 'rs', 'TestCheckException')
        pass
    pass

# positive_scalar_integer
def test_positive_scalar_integer_good():
    """
    Verify checker works correctly for valid input.

    Type: positive scalar integer
    """
    try:
        check.positive_scalar_integer(1, 'psi', TestCheckException)
    except check.CheckException:
        pytest.xfail('positive_scalar_integer failed on valid input')
    pass

def test_positive_scalar_integer_bad_var():
    """
    fail on invalid variable type.

    Type: positive scalar integer
    """
    for v0 in psilist:
        with pytest.raises(TestCheckException):
            check.positive_scalar_integer(v0, 'psi', TestCheckException)
            pass
        pass
    pass

def test_positive_scalar_integer_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.positive_scalar_integer(1, (1,), TestCheckException)
        pass
    pass

def test_positive_scalar_integer_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.positive_scalar_integer(1, 'psi', 'TestCheckException')
        pass
    pass

# nonnegative_scalar_integer
def test_nonnegative_scalar_integer_good():
    """
    Verify checker works correctly for valid input.

    Type: nonnegative scalar integer
    """
    for j in [0, 1, 2]:
        try:
            check.nonnegative_scalar_integer(j, 'nsi', TestCheckException)
        except check.CheckException:
            pytest.xfail('nonnegative_scalar_integer failed on valid input')
        pass
    pass

def test_nonnegative_scalar_integer_bad_var():
    """
    fail on invalid variable type.
    
    Type: nonnegative scalar integer
    """
    for v0 in nsilist:
        with pytest.raises(TestCheckException):
            check.nonnegative_scalar_integer(v0, 'nsi', TestCheckException)
            pass
        pass
    pass

def test_nonnegative_scalar_integer_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.nonnegative_scalar_integer(1, (1,), TestCheckException)
        pass
    pass

def test_nonnegative_scalar_integer_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.nonnegative_scalar_integer(1, 'nsi', 'TestCheckException')
        pass
    pass

# scalar_integer
def test_scalar_integer_good():
    """
    Verify checker works correctly for valid input.

    Type: scalar integer
    """
    for j in [-2, -1, 0, 1, 2]:
        try:
            check.scalar_integer(j, 'si', TestCheckException)
        except check.CheckException:
            pytest.xfail('scalar_integer failed on valid input')
        pass
    pass

def test_scalar_integer_bad_var():
    """
    fail on invalid variable type.

    Type: scalar integer
    """
    for v0 in rsilist:
        with pytest.raises(TestCheckException):
            check.scalar_integer(v0, 'si', TestCheckException)
            pass
        pass
    pass

def test_scalar_integer_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.scalar_integer(1, (1,), TestCheckException)
        pass
    pass

def test_scalar_integer_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.scalar_integer(1, 'si', 'TestCheckException')
        pass
    pass

def test_string_good():
    """
    Verify checker works correctly for valid input.

    Type: string
    """
    for j in ['a', '1', '.']:
        try:
            check.string(j, 'string', TestCheckException)
        except check.CheckException:
            pytest.xfail('string failed on valid input')
        pass
    pass

def test_string_bad_var():
    """
    fail on invalid variable type.

    Type: string
    """
    for v0 in strlist:
        with pytest.raises(TestCheckException):
            check.string(v0, 'string', TestCheckException)
            pass
        pass
    pass

def test_string_bad_vname():
    """fail on invalid input name for user output."""
    with pytest.raises(check.CheckException):
        check.string('a', ('a',), TestCheckException)
        pass
    pass

def test_string_bad_vexc():
    """fail on input vexc not an Exception."""
    with pytest.raises(check.CheckException):
        check.scalar_integer('a', 'string', 'TestCheckException')
        pass
    pass


if __name__ == '__main__':
    test_nonnegative_scalar_integer_bad_var()
    test_nonnegative_scalar_integer_bad_vexc()
    test_nonnegative_scalar_integer_bad_vname()
    test_nonnegative_scalar_integer_good()
    test_oneD_array_bad_var()
    test_oneD_array_bad_vexc()
    test_oneD_array_bad_vname()
    test_oneD_array_good()
    test_positive_scalar_integer_bad_var()
    test_positive_scalar_integer_bad_vexc()
    test_positive_scalar_integer_bad_vname()
    test_positive_scalar_integer_good()
    test_real_array_bad_var()
    test_real_array_bad_vexc()
    test_real_array_bad_vname()
    test_real_array_good()
    test_real_nonnegative_scalar_bad_var()
    test_real_nonnegative_scalar_bad_vexc()
    test_real_nonnegative_scalar_bad_vname()
    test_real_nonnegative_scalar_good()
    test_real_positive_scalar_bad_var()
    test_real_positive_scalar_bad_vexc()
    test_real_positive_scalar_bad_vname()
    test_real_positive_scalar_good()
    test_real_scalar_bad_var()
    test_real_scalar_bad_vexc()
    test_real_scalar_bad_vname()
    test_real_scalar_good()
    test_scalar_integer_bad_var()
    test_scalar_integer_bad_vexc()
    test_scalar_integer_bad_vname()
    test_scalar_integer_good()
    test_string_bad_var()
    test_string_bad_vexc()
    test_string_bad_vname()
    test_string_good()
    test_threeD_array_bad_var()
    test_threeD_array_bad_vexc()
    test_threeD_array_bad_vname()
    test_threeD_array_good()
    test_twoD_array_bad_var()
    test_twoD_array_bad_vexc()
    test_twoD_array_bad_vname()
    test_twoD_array_good()
    test_twoD_square_array_bad_var()
    test_twoD_square_array_bad_var_shape()
    test_twoD_square_array_bad_vexc()
    test_twoD_square_array_bad_vname() 
    test_twoD_square_array_good()
