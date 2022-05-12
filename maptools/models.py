from __future__ import annotations

import itertools
import math
import struct
import unicodedata
import warnings

import numpy
from styled import Styled

from maptools import cli

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
        _full_axes_set = {'X', 'Y', 'Z'}  # fixme: might redundant
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

    @classmethod
    def from_string(cls, orientation_string: str):
        """"""
        try:
            set(orientation_string.upper()).intersection({'X', 'Y', 'Z'}) == {'X', 'Y', 'Z'} and len(
                orientation_string) == 3
        except AssertionError:
            raise ValueError(f"invalid orientation string {orientation_string}: only use a string with XYZ only")
        c, r, s = tuple(orientation_string)
        return cls(cols=c, rows=r, sections=s)

    def to_integers(self):
        return _raxes[self.cols], _raxes[self.rows], _raxes[self.sections]

    @property
    def shape(self):
        """Orientation is a row vector in 3-space"""
        return self._data.shape

    def __repr__(self):
        return f"Orientation(cols='{self.cols}', rows='{self.rows}', sections='{self.sections}')"

    def __truediv__(self, other: Orientation):
        """Computes the permutation matrix required to convert this orientation to the specified orientation"""
        return PermutationMatrix.from_orientations(self, other)
        # return PermutationMatrix(get_permutation_matrix(numpy.asarray(self), numpy.asarray(other)))


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

    @classmethod
    def from_orientations(cls, orientation: [tuple | list | set | numpy.ndarray | Orientation],
                          new_orientation: [tuple | list | set | numpy.ndarray | Orientation]):
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
        elif isinstance(orientation, Orientation):
            _orientation = orientation.to_integers()
        else:
            raise TypeError(f"orientation must be a sequence type (tuple, list, set, or numpy.ndarray)")

        if isinstance(new_orientation, numpy.ndarray):
            _new_orientation = _convert_numpy_array_to_tuple(new_orientation)
        elif isinstance(new_orientation, (tuple, list, set)):
            _new_orientation = tuple(new_orientation)
        elif isinstance(new_orientation, Orientation):
            _new_orientation = new_orientation.to_integers()
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
        return cls(permutation_matrix)

    def __array__(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def swap_sequences(self):
        if numpy.array_equal(self._data, numpy.fromstring('1 0 0 0 1 0 0 0 1', sep=' ', dtype=int).reshape(3, 3)):
            return []
        elif numpy.array_equal(self._data, numpy.fromstring('1 0 0 0 0 1 0 1 0', sep=' ', dtype=int).reshape(3, 3)):
            return [(0, 1)]
        elif numpy.array_equal(self._data, numpy.fromstring('0 1 0 1 0 0 0 0 1', sep=' ', dtype=int).reshape(3, 3)):
            return [(1, 2)]
        elif numpy.array_equal(self._data, numpy.fromstring('0 0 1 0 1 0 1 0 0', sep=' ', dtype=int).reshape(3, 3)):
            return [(0, 2)]
        elif numpy.array_equal(self._data, numpy.fromstring('0 0 1 1 0 0 0 1 0', sep=' ', dtype=int).reshape(3, 3)):
            return [(1, 2), (0, 2)]
        elif numpy.array_equal(self._data, numpy.fromstring('0 1 0 0 0 1 1 0 0', sep=' ', dtype=int).reshape(3, 3)):
            return [(0, 2), (0, 1)]

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
        print(f"{other = }")
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


class MapFileAttribute:
    def __init__(self, name, default=None):
        self.name = name
        self.default = default

    def __get__(self, instance, owner=None):
        value = getattr(instance, self.name)
        if value is None and self.default is not None:
            return self.default
        return value

    def __set__(self, instance, value=None):
        setattr(instance, self.name, value)

    def __delete__(self, instance):
        delattr(instance, self.name)


class MapFile:
    __attrs__ = [
        '_nc', '_nr', '_ns',
        '_nx', '_ny', '_nz', '_x_length', '_y_length', '_z_length',
        '_alpha', '_beta', '_gamma', '_mapc', '_mapr', '_maps',
        '_amin', '_amax', '_amean', '_ispg', '_nsymbt', '_lskflg',
        '_s11', '_s12', '_s13', '_s21', '_s22', '_s23', '_s31', '_s32', '_s33',
        '_t1', '_t2', '_t3', '_extra', '_map', '_machst', '_rms', '_nlabl',
        # '_label_0', '_label_1', '_label_2', '_label_3', '_label_4', '_label_5', '_label_6', '_label_7', '_label_8',
        # '_label_9',
    ]
    # descriptors
    nc = MapFileAttribute('_nc')
    nr = MapFileAttribute('_nr')
    ns = MapFileAttribute('_ns')
    nx = MapFileAttribute('_nx')
    ny = MapFileAttribute('_ny')
    nz = MapFileAttribute('_nz')
    x_length = MapFileAttribute('_x_length')
    y_length = MapFileAttribute('_y_length')
    z_length = MapFileAttribute('_z_length')
    alpha = MapFileAttribute('_alpha', 90.0)
    beta = MapFileAttribute('_beta', 90.0)
    gamma = MapFileAttribute('_gamma', 90.0)
    mapc = MapFileAttribute('_mapc', 1)
    mapr = MapFileAttribute('_mapr', 2)
    maps = MapFileAttribute('_maps', 3)
    amin = MapFileAttribute('_amin')
    amax = MapFileAttribute('_amax')
    amean = MapFileAttribute('_amean')
    ispg = MapFileAttribute('_ispg', 1)
    nsymbt = MapFileAttribute('_nsymbt', 0)
    lskflg = MapFileAttribute('_lskflg', 0)
    s11 = MapFileAttribute('_s11', 0.0)
    s12 = MapFileAttribute('_s12', 0.0)
    s13 = MapFileAttribute('_s13', 0.0)
    s21 = MapFileAttribute('_s21', 0.0)
    s22 = MapFileAttribute('_s22', 0.0)
    s23 = MapFileAttribute('_s23', 0.0)
    s31 = MapFileAttribute('_s31', 0.0)
    s32 = MapFileAttribute('_s32', 0.0)
    s33 = MapFileAttribute('_s33', 0.0)
    t1 = MapFileAttribute('_t1', 0.0)
    t2 = MapFileAttribute('_t2', 0.0)
    t3 = MapFileAttribute('_t3', 0.0)
    extra = MapFileAttribute('_extra', (0,) * 15)
    map = MapFileAttribute('_map', b'MAP ')
    machst = MapFileAttribute('_machst', bytes([68, 68, 0, 0]))
    rms = MapFileAttribute('_rms')
    nlabl = MapFileAttribute('_nlabl', 0)
    # convenience properties
    cols = MapFileAttribute('_nc')
    rows = MapFileAttribute('_nr')
    sections = MapFileAttribute('_ns')

    def __init__(
            self, name, file_mode='r',
            orientation=Orientation(cols='X', rows='Y', sections='Z'),
            voxel_size=(1.0, 1.0, 1.0),
            map_mode=2,
            start=(0, 0, 0),
            colour=False,
            verbose=False
    ):
        """"""
        # todo: validate file modes in ['r', 'r+' and 'w']
        self.name = name
        self.file_mode = file_mode
        self._data = None
        self._orientation = orientation
        if file_mode == 'w':
            self._voxel_size = tuple(
                numpy.array(voxel_size) @ PermutationMatrix.from_orientations(
                    (1, 2, 3),  # always start from the default
                    orientation.to_integers()
                )
            )
            # reset the map mode
            self._mode = map_mode
            # start
            self._start = start
        self.handle = None
        # create attributes
        for attr in self.__attrs__:
            if hasattr(self, attr):
                continue
            setattr(self, attr, None)
        # colour, verbose
        self.colour = colour
        self.verbose = verbose

    def __enter__(self):
        self.handle = open(self.name, f'{self.file_mode}b')
        if self.file_mode in ['r', 'r+']:  # since we are reading we defer to what is present
            # source: ftp://ftp.ebi.ac.uk/pub/databases/emdb/doc/Map-format/current/EMDB_map_format.pdf
            # number of columns (fastest changing), rows, sections (slowest changing)
            self._nc, self._nr, self._ns = struct.unpack('<iii', self.handle.read(12))
            # voxel datatype
            self._mode = struct.unpack('<I', self.handle.read(4))[0]
            # position of first column, first row, and first section (voxel grid units)
            self._start = struct.unpack('<iii', self.handle.read(12))
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
                '<' + 'f' * 9, self.handle.read(9 * 4))
            # skew translation-T1, T2, T3
            self._t1, self._t2, self._t3 = struct.unpack('<fff', self.handle.read(12))
            # user-defined metadata
            self._extra = struct.unpack('<15i', self.handle.read(15 * 4))
            # MRC/CCP4 MAP format identifier
            self._map = struct.unpack('<4s', self.handle.read(4))[0]
            # machine stamp
            self._machst = struct.unpack('<4s', self.handle.read(4))[0]
            # Density root-mean-square deviation
            self._rms = struct.unpack('<f', self.handle.read(4))[0]
            # number of labels
            self._nlabl = struct.unpack('<i', self.handle.read(4))[0]
            # orientation
            self._orientation = Orientation.from_integers((self._mapc, self._mapr, self._maps))
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

            # read the data
            dtype = self._mode_to_dtype()
            # todo: byteorder
            self._data = numpy.frombuffer(self.handle.read(), dtype=dtype).reshape(self.ns, self.nr, self.nc)
            # voxel size
            self._voxel_size = tuple(numpy.divide(
                numpy.array([self._x_length, self._y_length, self._z_length]),
                numpy.array([self._nc, self._nr, self._ns]),
            ).tolist())

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Depending on the file mode, this method writes (or not) changes to disk"""
        if self.file_mode in ['r+', 'w']:
            try:
                self._prepare()
            except UserWarning as uw:
                self.handle.close()
                raise ValueError(str(uw))
            # rewind
            self.handle.seek(0)
            self._write_header()
            self._write_data()
        self.handle.close()

    def __array__(self):
        if self._data is None:
            dtype = self._mode_to_dtype()
            # byteorder
            self._data = numpy.frombuffer(self.handle.read(), dtype=dtype).reshape(self.ns, self.nr, self.nc)
        return self._data

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        """"""
        try:
            assert value in cli.MAP_MODES
        except AssertionError:
            raise ValueError(f"invalid mode={value}; should be one of {', '.join(cli.MAP_MODES)}")
        if self._mode in cli.INT_MAP_MODES and value in cli.FLOAT_MAP_MODES:
            warnings.warn(f"file size will increase by converting from int to float voxels", UserWarning)
        elif self._mode in cli.FLOAT_MAP_MODES and value in cli.INT_MAP_MODES:
            warnings.warn(f"truncating data by converting from float to int voxels", UserWarning)
        if self.verbose:
            print(Styled(f"[[ '[info] changing mode from {self.mode} to {value}...'|fg-orange_3 ]]"))
        self._mode = value
        self._prepare()

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        """"""
        # tuple, list, set
        try:
            assert len(value) == 3
        except AssertionError:
            raise ValueError(f"must have only 3 values: {value}")
        if isinstance(value, (tuple, list, set)):
            _value = value
        # array
        elif isinstance(value, numpy.ndarray):
            _value = value.asdtype(int)
        else:
            raise TypeError(f"value must be a tuple, list, set or numpy array")
        self._start = _value
        self._prepare()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        dtype = self._mode_to_dtype()
        self._data = data.astype(dtype)
        self._prepare()

    @property
    def voxel_size(self):
        """"""
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, vox_size):
        if isinstance(vox_size, (int, float,)):
            x_size, y_size, z_size = (vox_size,) * 3
        elif isinstance(vox_size, (tuple, list, set)):
            try:
                assert len(vox_size) == 3
            except AssertionError:
                raise TypeError(f"voxel size should be a number or 3-tuple: {vox_size}")
            x_size, y_size, z_size = vox_size
        elif isinstance(vox_size, numpy.ndarray):
            try:
                assert vox_size.shape == (3,)
            except AssertionError:
                raise TypeError(f"voxel size should be an array of shape (3, )")
            x_size, y_size, z_size = vox_size
        self._voxel_size = x_size, y_size, z_size
        self._prepare()

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation: Orientation):
        """
        Change the orientation according to the provided Orientation object specified.

        We infer the permutation matrix by 'dividing' the current orientation by the specified orientation.
        This orientation will permute (C,R,S) to the desired arrangement. However, since the array shape is in the
        order (S,R,C) we must first reverse the shape so as to permute the correct axes. After reversing,
        we permute then reverse the new shape before applying it to the data.

        :param orientation: the new orientation
        """
        # reorient the volume
        permutation_matrix = self.orientation / orientation
        swap_sequences = permutation_matrix.swap_sequences
        for swap_sequence in swap_sequences:
            self._data = numpy.swapaxes(self._data, *swap_sequence)
        # matrix multiply to get the new shape
        # we have to reverse the shape to apply the permutation
        # reversed_current_shape = numpy.asarray(self).shape[::-1]
        # get the reverse of the new shape
        # reversed_new_shape = reversed_current_shape @ permutation_matrix
        # reverse to get the actual shape to abe applied
        # new_shape = reversed_new_shape[::-1]
        # reshape the data
        # self._data = numpy.asarray(self).reshape(new_shape)
        # set the new orientation
        self._orientation = orientation
        # also permute the voxel sizes
        # fixme: ugly
        self.voxel_size = tuple(
            (numpy.array(self.voxel_size).reshape(1, 3) @ permutation_matrix).tolist()[0]
        )
        # recalculate parameters
        self._prepare()

    def _prepare(self):
        """Ensure that all required attributes are set before write"""
        # make sure there is data because everything is derived from it
        if self._data is None:
            raise UserWarning("no data to write; set MapFile.data attribute to a numpy 3D array")
        # some attributes have default values and are excluded
        self._ns, self._nr, self._nc = self._data.shape
        self._nz, self._ny, self._nx = self._data.shape
        self._z_length, self._y_length, self._x_length = numpy.multiply(self._data.shape,
                                                                        numpy.array(self._voxel_size)[::-1])
        self._mapc, self._mapr, self._maps = self.orientation.to_integers()
        # change dtype if necessary
        dtype = self._mode_to_dtype()
        self._data = self._data.astype(dtype)
        self._amin, self._amax, self._amean = self._data.min(), self._data.max(), self._data.mean()
        self._rms = math.sqrt(numpy.mean(numpy.square(self._data)))

    def _dtype_to_mode(self):
        """"""
        dtype = numpy.asarray(self).dtype
        if dtype.name == 'int8':
            return 0
        elif dtype.name == 'int16':
            return 1
        elif dtype.name == 'float32':
            return 2
        elif dtype.name == 'int32':  # i know! this should be FFT stored a complex signed integers
            return 3
        elif dtype.name == 'complex64':
            return 4
        elif dtype.name == 'uint16':
            return 6
        elif dtype.name == 'float16':
            return 12

    def _mode_to_dtype(self):
        """"""
        # dtype = numpy.float32  # default
        if self.mode == 0:
            dtype = numpy.int8
        elif self.mode == 1:
            dtype = numpy.int16
        elif self.mode == 2:
            dtype = numpy.float32
        elif self.mode == 3:
            dtype = numpy.int32
        elif self.mode == 4:
            dtype = numpy.complex64
        elif self.mode == 6:
            dtype = numpy.uint16
        elif self.mode == 12:
            dtype = numpy.float16
        # elif self._mode == 101:
        #     dtype = numpy.dtype()
        return dtype

    def _write_header(self):
        self.handle.write(struct.pack('<iii', self.nc, self.nr, self.ns))
        self.handle.write(struct.pack('<I', self.mode))
        self.handle.write(struct.pack('<iii', *self.start))
        self.handle.write(struct.pack('<iii', self.nx, self.ny, self.nz))
        self.handle.write(struct.pack('<fff', self.x_length, self.y_length, self.z_length))
        self.handle.write(struct.pack('<fff', self.alpha, self.beta, self.gamma))
        self.handle.write(struct.pack('<iii', self.mapc, self.mapr, self.maps))
        self.handle.write(struct.pack('<fff', self.amin, self.amax, self.amean))
        self.handle.write(struct.pack('<iii', self.ispg, self.nsymbt, self.lskflg))
        self.handle.write(
            struct.pack('<9f', self.s11, self.s12, self.s13, self.s21, self.s22, self.s23, self.s31, self.s32,
                        self.s33, ))
        self.handle.write(struct.pack('<fff', self.t1, self.t2, self.t3))
        self.handle.write(struct.pack('<15i', *self.extra))
        self.handle.write(struct.pack('<4s', self.map))
        self.handle.write(struct.pack('<4s', self.machst))
        self.handle.write(struct.pack('<f', self.rms))
        self.handle.write(struct.pack('<i', self.nlabl))
        # write the remaining blanks
        # fixme: allow records to be added
        self.handle.write(struct.pack(f'<800x'))

    def _write_data(self):
        self._data.tofile(self.handle)
        self.handle.truncate()

    def __str__(self):
        alpha = unicodedata.lookup('GREEK SMALL LETTER ALPHA')
        beta = unicodedata.lookup('GREEK SMALL LETTER BETA')
        gamma = unicodedata.lookup('GREEK SMALL LETTER GAMMA')
        prec_tuple = lambda t: f"({t[0]:6f}, {t[1]:6f}, {t[2]:6f})"
        if self.colour:
            bold_yellow = lambda t: Styled(f"[[ '{t:<40}'|bold:fg-white:no-end ]]")
            bold_green = lambda t: Styled(f"[[ '{t:<40}'|bold:fg-green:no-end ]]")
            string = f"""\
                \r{bold_yellow('Cols, rows, sections:')}{self.nc, self.nr, self.ns}
                \r{bold_green('Mode:')}{self.mode} ({self._mode_to_dtype()})
                \r{bold_yellow('Start col, row, sections:')}{self.start}
                \r{bold_green('X, Y, Z:')}({self.nx}, {self.ny}, {self.nz})
                \r{bold_yellow('Voxel size:')}{prec_tuple(self.voxel_size)}
                \r{bold_green('Lengths X, Y, Z (Ångstrom):')}{self.x_length:6f}, {self.y_length:6f}, {self.z_length:6f}
                \r{bold_yellow(f'{alpha}, {beta}, {gamma}:')}{self.alpha}, {self.beta}, {self.gamma}
                \r{bold_green('Map cols, rows, sections:')}{self.mapc}, {self.mapr}, {self.maps}
                \r{bold_yellow('Density min, max, mean:')}{self.amin:6f}, {self.amax:6f}, {self.amean:6f}
                \r{bold_green('Space group:')}{self.ispg}
                \r{bold_yellow('Bytes in symmetry table:')}{self.nsymbt}
                \r{bold_green('Skew matrix flag:')}{self.lskflg}
                \r{bold_yellow('Skew matrix:')}{self.s11} {self.s12} {self.s13}
                \r{bold_yellow('')}{self.s21} {self.s22} {self.s23}
                \r{bold_yellow('')}{self.s31} {self.s32} {self.s33}
                \r{bold_green('Skew translation:')}({self.t1}, {self.t2}, {self.t3})
                \r{bold_yellow('Extra:')}{self.extra}
                \r{bold_green('Map:')}{self.map}
                \r{bold_yellow('Mach-stamp:')}{self.machst}
                \r{bold_green('RMS:')}{self.rms}
                \r{bold_yellow('Number of labels:')}{self.nlabl}\n"""
            string += str(Styled("[[ ''|yes-end ]]"))
        else:
            plain = lambda t: f"{t:<40}"
            string = f"""\
                \r{plain('Cols, rows, sections:')}{self.nc, self.nr, self.ns}
                \r{plain('Mode:')}{self.mode} ({self._mode_to_dtype()})
                \r{plain('Start col, row, sections:')}{self.start}
                \r{plain('X, Y, Z:')}({self.nx}, {self.ny}, {self.nz})
                \r{plain('Voxel size:')}{prec_tuple(self.voxel_size)}
                \r{plain('Lengths X, Y, Z (Ångstrom):')}{self.x_length:6f}, {self.y_length:6f}, {self.z_length:6f}
                \r{plain(f'{alpha}, {beta}, {gamma}:')}{self.alpha}, {self.beta}, {self.gamma}
                \r{plain('Map cols, rows, sections:')}{self.mapc}, {self.mapr}, {self.maps}
                \r{plain('Density min, max, mean:')}{self.amin:6f}, {self.amax:6f}, {self.amean:6f}
                \r{plain('Space group:')}{self.ispg}
                \r{plain('Bytes in symmetry table:')}{self.nsymbt}
                \r{plain('Skew matrix flag:')}{self.lskflg}
                \r{plain('Skew matrix:')}{self.s11} {self.s12} {self.s13}
                \r{plain('')}{self.s21} {self.s22} {self.s23}
                \r{plain('')}{self.s31} {self.s32} {self.s33}
                \r{plain('Skew translation:')}({self.t1}, {self.t2}, {self.t3})
                \r{plain('Extra:')}{self.extra}
                \r{plain('Map:')}{self.map}
                \r{plain('Mach-stamp:')}{self.machst}
                \r{plain('RMS:')}{self.rms}
                \r{plain('Number of labels:')}{self.nlabl}"""
        # labels
        for i in range(self.nlabl):
            string += f"Label {i}: {getattr(self, f'_label_{i}')}"
        return string
