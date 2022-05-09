from __future__ import annotations

import numpy

_axes = {
    1: 'X',
    2: 'Y',
    3: 'Z',
}
_raxes = {
    'X': 1,
    'Y': 2,
    'Z': 3,
}


class Orientation:
    def __init__(self, cols='X', rows='Y', sections='Z'):
        # values are 'X', 'Y', or 'Z'
        _cols = cols.upper()
        _rows = rows.upper()
        _sections = sections.upper()
        try:
            assert _cols in _axes.values()
            assert _rows in _axes.values()
            assert _sections in _axes.values()
        except AssertionError:
            raise ValueError(f"cols, rows, sections must be one of {', '.join(_axes.values())}")
        # ensure values are unique
        _full_axes_set = {'X', 'Y', 'Z'}
        _axes_set = set([_cols, _rows, _sections])
        try:
            assert len(_axes_set) == 3
        except AssertionError:
            raise ValueError(f"repeated axis values: {_cols, _rows, _sections}")
        self.cols = _cols
        self.rows = _rows
        self.sections = _sections
        self._data = numpy.array([_raxes[_cols], _raxes[_rows], _raxes[_sections]], dtype=int).reshape(1, 3)

    def __array__(self):
        return self._data

    @classmethod
    def from_integers(cls, integers: tuple):
        """"""
        try:
            set(integers).intersection({1, 2, 3}) == {1, 2, 3} and len(integers) == 3
        except AssertionError:
            raise ValueError(f"invalid integers {integers}: only use a 3-tuple with values from {{1, 2, 3}}")
        c, r, s = integers
        return cls(cols=_axes[c], rows=_axes[r], sections=_axes[s])

    @property
    def shape(self):
        """Orientation is a row vector in 3-space"""
        return self._data.shape

    def __repr__(self):
        return f"Orientation(cols='{self.cols}', rows='{self.rows}', sections='{self.sections}')"

    def __truediv__(self, other: Orientation):
        """Computes the permutation matrix required to convert this orientation to the specified orientation"""
        return PermutationMatrix(get_permutation_matrix(self._data, other._data))


class PermutationMatrix:
    """A square matrix with exactly only one 1 in each row and zeros everywhere else"""

    def __init__(self, data):
        # sanity check
        try:
            assert numpy.sum(data) == 3
            assert numpy.count_nonzero(data) == 3
        except AssertionError:
            raise ValueError(f"non-binary values: {data}")
        self._data = data
        self.rows, self.cols = data.shape
        # dtype
        self._data.dtype = int

    def __array__(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def __matmul__(self, other):
        """LHS matrix multiplication"""
        # other must have as many rows as self has columns
        try:
            assert self.cols == other.shape[0]
        except AssertionError:
            raise ValueError(f"invalid shapes for LHS multiplication: ({self.shape} @ {other.shape})")
        return numpy.dot(numpy.asarray(self), numpy.array(other))

    def __imatmul__(self, other):
        """iterative RHS matrix multiplication"""
        try:
            assert self.cols == other.shape[0]
        except AssertionError:
            raise ValueError(f"invalid shapes for LHS multiplication: ({self.shape} @ {other.shape})")
        return numpy.dot(numpy.asarray(self), numpy.asarray(other))

    def __rmatmul__(self, other):
        """RHS matrix multiplication"""
        # other must have as many cols as self has rows
        try:
            assert self.rows == other.shape[1]
        except AssertionError:
            raise ValueError(f"invalid shapes for RHS multiplication: ({self.shape} @ {other.shape})")
        return numpy.dot(numpy.asarray(other), numpy.asarray(self))

    def __repr__(self):
        return f"PermutationMatrix({self._data})"

    def __str__(self):
        string = ""
        for row in self._data:
            string += str(row) + "\n"
        return string

    def __eq__(self, other):
        return numpy.array_equal(numpy.asarray(self), numpy.asarray(other))


def get_orientation(mrc):
    """
    Determine the orientation of an MRC file

    :param mrc: an MRC file
    :return: a tuple
    """
    nc, nr, ns = mrc.header.nx, mrc.header.ny, mrc.header.nz
    mapc, mapr, maps = int(mrc.header.mapc), int(mrc.header.mapr), int(mrc.header.maps)
    return Orientation(cols=_axes[mapc], rows=_axes[mapr], sections=_axes[maps])


def set_orientation(mrc, orientation: Orientation):
    """
    
    :param mrc:  an MRC file
    :param orientation: 
    :return: 
    """


def get_space_handedness(mrc):
    """

    :param mrc: an MRC file
    :return:
    """


def get_permutation_matrix(orientation, new_orientation):
    """Compute the permutation matrix required to convert the sequence <orientation> to <new_orientation>

    A permutation matrix is a square matrix.
    The values of its elements are in {0, 1}.
    """

    # orientation can be a numpy array but it must be convertible to a 3-tuple
    def _convert_numpy_array_to_tuple(array):
        try:
            assert array.shape[0] == 1
        except AssertionError:
            raise ValueError(f"orientation array {array} has wrong shape; must be (1, n?) for any n")
        return tuple(array.flatten().tolist())

    if isinstance(orientation, numpy.ndarray):
        _orientation = _convert_numpy_array_to_tuple(orientation)
    elif isinstance(orientation, (tuple, list, set)):
        _orientation = tuple(orientation)
    else:
        raise TypeError(f"orientation must be a sequence type (tuple, list, set, or numpy.ndarray)")

    if isinstance(new_orientation, numpy.ndarray):
        _new_orientation = _convert_numpy_array_to_tuple(new_orientation)
    elif isinstance(new_orientation, (tuple, list, set)):
        _new_orientation = tuple(new_orientation)
    else:
        raise TypeError(f"new_orientation must be a sequence type (tuple, list, set, or numpy.ndarray)")
    # assert that the values in orientation are unique
    try:
        assert len(set(_orientation)) == len(_orientation)
    except AssertionError:
        raise ValueError(f"repeated elements in {_orientation}")
    # assert that the values in new_orientation are unique
    try:
        assert len(set(_new_orientation)) == len(_new_orientation)
    except AssertionError:
        raise ValueError(f"repeated elements in {_new_orientation}")
    # assert that the values in orientation are exactly those in new_orientation
    try:
        assert len(set(_orientation)) == len(set(_new_orientation))
    except AssertionError:
        raise ValueError(f"values differ: {_orientation} vs. {_new_orientation}")
    # compute the permutation matrix
    permutation_matrix = numpy.zeros((len(_orientation), len(_orientation)), dtype=int)
    for index, value in enumerate(_orientation):
        permutation_matrix[index, _new_orientation.index(value)] = 1
    return permutation_matrix
