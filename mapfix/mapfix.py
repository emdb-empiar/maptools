from __future__ import annotations

import struct

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
        return PermutationMatrix(get_permutation_matrix(numpy.asarray(self), numpy.asarray(other)))


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


# todo: do away with `mrcfile`!!!
class MapFile:
    def __init__(self, name):
        """"""
        self.name = name
        self._data = None
        # attributes

    def __enter__(self):
        self.handle = open(self.name, 'rb')
        # source: ftp://ftp.ebi.ac.uk/pub/databases/emdb/doc/Map-format/current/EMDB_map_format.pdf
        # number of columns (fastest changing), rows, sections (slowest changing)
        self._nc, self._nr, self._ns = struct.unpack('<iii', self.handle.read(12))
        # voxel datatype
        self._mode = struct.unpack('<I', self.handle.read(4))[0]
        # position of first column, first row, and first section (voxel grid units)
        self._ncstart, self._nrstart, self._nsstart = struct.unpack('<iii', self.handle.read(12))
        # intervals per unit cell repeat along X,Y Z
        self._nx, self._ny, self._nz = struct.unpack('<iii', self.handle.read(12))
        # Unit Cell repeats along X, Y, Z In Ångstroms
        self._x_length, self._y_length, self._z_length = struct.unpack('<fff', self.handle.read(12))
        # Unit Cell angles (degrees)
        self._alpha, self._beta, self._gamma = struct.unpack('<fff', self.handle.read(12))
        # relationship of X,Y,Z axes to columns, rows, sections
        self._mapc, self._mapr, self._maps = struct.unpack('<iii', self.handle.read(12))
        # Minimum, maximum, average density
        self._amin, self._amax, self._amean = struct.unpack('<fff', self.handle.read(12))
        # space group #
        # number of bytes in symmetry table (multiple of 80)
        # flag for skew matrix
        self._ispg, self._nsymbt, self._lskflg = struct.unpack('<iii', self.handle.read(12))
        # skew matrix-S11, S12, S13, S21, S22, S23, S31, S32, S33
        self._s11, self._s12, self._s13, self._s21, self._s22, self._s23, self._s31, self._s32, self._s33 = struct.unpack(
            '<' + 'f' * (9), self.handle.read(9 * 4))
        # skew translation-T1, T2, T3
        self._t1, self._t2, self._t3 = struct.unpack('<fff', self.handle.read(12))
        # user-defined metadata
        self._extra = struct.unpack('<15i', self.handle.read(15 * 4))
        # MRC/CCP4 MAP format identifier
        self._map = struct.unpack('<4s', self.handle.read(4))[0].decode('utf-8')
        # machine stamp
        self._machst = struct.unpack('<4s', self.handle.read(4))[0].decode('utf-8')
        # Density root-mean-square deviation
        self._rms = struct.unpack('<f', self.handle.read(4))[0]
        # number of labels
        self._nlabl = struct.unpack('<i', self.handle.read(4))[0]
        # Up to 10 user-defined labels
        for i in range(int(self._nlabl)):
            setattr(
                self, f'_label_{i}',
                struct.unpack('<80s', self.handle.read(80))[0].decode('utf-8').rstrip(' ')
            )
        # jump to the beginning of data
        if self.handle.tell() < 1024:
            self.handle.seek(1024)
        else:
            raise ValueError(f"Current byte position in file ({self.handle.tell()}) is past end of header (1024)")

        if self._mode == 0:
            self._voxel_type = 'b'
            self._voxel_size = 1
        elif self._mode == 1:
            self._voxel_type = 'h'
            self._voxel_size = 2
        elif self._mode == 2:
            self._voxel_type = 'f'
            self._voxel_size = 4
        elif self._mode == 3:
            raise ValueError("No support for complex signed integer Fourier maps")
        elif self._mode == 4:
            raise ValueError("No support for complex floating point Fourier maps")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handle.close()

    def write(self, f):
        """Write data to an EMDB Map file

        :param file f: file object
        :return int status: 0 on success; fail otherwise
        """

        string = struct.pack('<iii', self._nc, self._nr, self._ns)
        string += struct.pack('<I', self._mode)
        string += struct.pack('<iii', self._ncstart, self._nrstart, self._nsstart)
        string += struct.pack('<iii', self._nx, self._ny, self._nz)
        string += struct.pack('<fff', self._x_length, self._y_length, self._z_length)
        string += struct.pack('<fff', self._alpha, self._beta, self._gamma)
        string += struct.pack('<iii', self._mapc, self._mapr, self._maps)
        string += struct.pack('<fff', self._amin, self._amax, self._amean)
        string += struct.pack('<iii', self._ispg, self._nsymbt, self._lskflg)
        string += struct.pack('<' + 'f' * (9), self._s11, self._s12, self._s13, self._s21, self._s22, self._s23,
                              self._s31, self._s32, self._s33)
        string += struct.pack('<fff', self._t1, self._t2, self._t3)
        string += struct.pack('<15i', *self._extra)
        # string += struct.pack('<4c', self._map)
        # convert to bytes
        string += self._map.encode('utf-8')
        string += self._machst.encode('utf-8')
        string += struct.pack('<f', self._rms)

        # if inverted we will add one more label
        # if self._inverted:
        #     string += struct.pack('<i', self._nlabl + 1)
        # else:
        #     string += struct.pack('<i', self._nlabl)

        for i in range(self._nlabl):
            len_label = len(getattr(self, f'_label_{i}'))
            encoding = getattr(self, f'_label_{i}').encode('utf-8')
            string += encoding
            # pack the remaining space
            string += struct.pack(f'<{80-len_label}x')

        # todo: do something similar for fixing the orientation
        # if self._inverted:
        #     from datetime import datetime
        #     d = datetime.now()
        #     string += _encode("{:<56}{:>24}".format(
        #         "mapfix: inverted intensities",
        #         d.strftime("%d-%b-%y  %H:%M:%S     ")
        #     ), 'utf-8')

        # pad up to full header of 1024 bytes
        try:
            assert 1024 - len(string) >= 0
        except AssertionError:
            raise ValueError("Header is too long")

        string += struct.pack(
            '<' + str(1024 - len(string)) + 'x')  # dodgy line because we may need to move one byte forward or back

        # numpy.ndarray.tofile(numpy.array(self._data))
        string += struct.pack('<' + self._voxel_type * self._voxel_count, *tuple(self._voxels))

        f.write(string)
        f.flush()

        return os.EX_OK

    def __array__(self):
        if self._data is None:
            dtype = numpy.float32
            if self._mode == 0:
                dtype = numpy.int8
            elif self._mode == 1:
                dtype = numpy.float16
            elif self._mode == 2:
                dtype = numpy.float32
            elif self._mode == 3:
                dtype = numpy.complex
            elif self._mode == 4:
                dtype = numpy.complex
            elif self._mode == 6:
                dtype = numpy.uint16
            # byteorder
            # dtype = dtype.newbyteorder('<')
            self._data = numpy.frombuffer(self.handle.read(), dtype=dtype).reshape(self._ns, self._nr, self._nc)
        return self._data

    def __str__(self):
        string = f"""\
            \rCols, rows, sections: 
            \r    {self._nc}, {self._nr}, {self._ns}
            \rMode: {self._mode}
            \rStart col, row, sections: 
            \r    {self._ncstart}, {self._nrstart}, {self._nsstart}
            \rX, Y, Z: 
            \r    {self._nx}, {self._ny}, {self._nz}
            \rLengths X, Y, Z (Ångstrom): 
            \r    {self._x_length}, {self._y_length}, {self._z_length}
            \r\U000003b1, \U000003b2, \U000003b3: 
            \r    {self._alpha}, {self._beta}, {self._gamma}
            \rMap cols, rows, sections: 
            \r    {self._mapc}, {self._mapr}, {self._maps}
            \rDensity min, max, mean: 
            \r    {self._amin}, {self._amax}, {self._amean}
            \rSpace group: {self._ispg}
            \rBytes in symmetry table: {self._nsymbt}
            \rSkew matrix flag: {self._lskflg}
            \rSkew matrix:
            \r    {self._s11} {self._s12} {self._s13}
            \r    {self._s21} {self._s22} {self._s23}
            \r    {self._s31} {self._s32} {self._s33}
            \rSkew translation:
            \r    {self._t1}
            \r    {self._t2}
            \r    {self._t3}
            \rExtra: {self._extra}
            \rMap: {self._map}
            \rMach-stamp: {self._machst}
            \rRMS: {self._rms}
            \rLabel count: {self._nlabl}
            \r"""
        # if int(self._nlabl) > 0:
        for i in range(self._nlabl):
            string += f"Label {i}: {getattr(self, f'_label_{i}')}"
        return string

    @property
    def orientation(self):
        """

        :return:
        """
        return Orientation(cols=_axes[self._mapc], rows=_axes[self._mapr], sections=_axes[self._maps])

    @orientation.setter
    def orientation(self, orientation: Orientation):
        """

        :param orientation:
        :return:
        """
        self._mapc = _raxes[orientation.cols]
        self._mapr = _raxes[orientation.rows]
        self._maps = _raxes[orientation.sections]
        # rotate the volume
        permutation_matrix = self.orientation / orientation
        # matrix multiply to get the new shape
        new_shape = numpy.array(mrc.data.shape) @ permutation_matrix
        # reshape the data
        self._data = numpy.asarray(self._data).reshape(new_shape)


def get_orientation(mrc):
    """
    Determine the orientation of an MRC file

    :param mrc: an MRC file
    :return: a tuple
    """
    mapc, mapr, maps = int(mrc.header.mapc), int(mrc.header.mapr), int(mrc.header.maps)
    return Orientation(cols=_axes[mapc], rows=_axes[mapr], sections=_axes[maps])


def set_orientation(mrc, orientation: Orientation):
    """

    :param mrc:  an MRC file
    :param orientation:
    :return:
    """
    # reset the mapc, mapr, maps attributes
    mrc.header.mapc = _raxes[orientation.cols]
    mrc.header.mapr = _raxes[orientation.rows]
    mrc.header.maps = _raxes[orientation.sections]
    # reset the voxel size
    # rotate the volume
    current_orientation = get_orientation(mrc)
    permutation_matrix = current_orientation / orientation
    new_shape = numpy.array(mrc.data.shape) @ permutation_matrix
    mrc.set_data(mrc.data.reshape(new_shape))
    return mrc


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
