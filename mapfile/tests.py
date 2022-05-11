import os
import pathlib
import random
import secrets
import sys
import unittest

import numpy

import mapfile
from mapfile import models, cli, managers, utils

BASE_DIR = pathlib.Path(__file__).parent.parent
TEST_DATA_DIR = BASE_DIR / 'test_data'


class TestCLI(unittest.TestCase):
    def test_view(self):
        """"""
        args = cli.cli(f"map view file.map")
        self.assertEqual('view', args.command)
        self.assertEqual('file.map', args.file)

    def test_edit(self):
        args = cli.cli(f"map edit file.map")
        self.assertEqual('edit', args.command)
        self.assertEqual('file.map', args.file)
        self.assertEqual('XYZ', args.orientation)
        self.assertEqual([1.0, 1.0, 1.0], args.voxel_sizes)
        self.assertEqual('r+', args.file_mode)
        self.assertEqual(2, args.map_mode)

    def test_create(self):
        args = cli.cli(f"map create file.map")
        self.assertEqual('create', args.command)
        self.assertEqual('file.map', args.file)
        self.assertEqual('XYZ', args.orientation)
        self.assertEqual([1.0, 1.0, 1.0], args.voxel_sizes)
        self.assertEqual([10, 10, 10], args.size)
        self.assertEqual('w', args.file_mode)
        self.assertEqual(2, args.map_mode)
        self.assertEqual(0, args.min)
        self.assertEqual(10, args.max)
        self.assertTrue(args.zeros)



class TestManagers(unittest.TestCase):
    def test_view(self):
        """"""
        fn = TEST_DATA_DIR / f"file-{secrets.token_urlsafe(3)}.map"
        with models.MapFile(fn, 'w') as mapf:
            mapf.data = numpy.random.rand(21, 31, 14)
        args = cli.cli(f"map view {fn}")
        ex = managers.view(args)
        os.remove(fn)

    def test_edit(self):
        """"""
        fn = TEST_DATA_DIR / f"file-{secrets.token_urlsafe(3)}.map"
        with models.MapFile(fn, 'w') as mapf:
            mapf.data = numpy.random.rand(37, 16, 62)
        # orientation
        args = cli.cli(f"map edit {fn} --orientation=YZX")
        managers.edit(args)
        with models.MapFile(fn) as mapf2:
            self.assertEqual((2, 3, 1), mapf2.orientation.to_integers())
            # todo: all the other attributes that should have been affected by the orientation
        # managers.view(args)
        os.remove(fn)


class TestExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # swap cols 0 and 1
        cls.p01 = numpy.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=int)
        # swap cols 0 and 2
        cls.p02 = numpy.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ], dtype=int)
        # swap cols 1 and 2
        cls.p12 = numpy.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ], dtype=int)

        # print(numpy.dot(p01, p02))
        # print(numpy.dot(numpy.array((1, 2, 3)), p01))
        # print(numpy.dot(numpy.array((1, 2, 3)), numpy.dot(p01, p02)))
        # print(numpy.dot(p02, p01))
        # print(numpy.dot(numpy.array((1, 2, 3)), p02))
        # print(numpy.dot(numpy.array((1, 2, 3)), numpy.dot(p02, p01)))
        # (2,1,3)->(1,2,3)->(2,3,1)
        # first (1,2,3)->(2,3,1)
        # print(
        #     numpy.dot(
        #         numpy.dot(
        #             numpy.array((2, 1, 3)),
        #             p01
        #         ),
        #         numpy.dot(p02, p01)
        #     )
        # )

        # print(numpy.dot(p12, p01))
        # print(numpy.dot(p12, p02))

    def test_get_vol(self):
        cols, rows, sects = random.sample(range(10, 30), k=3)
        # cols, rows, sects = (10, ) * 3
        print(cols, rows, sects)
        vol = utils.get_vol(cols, rows, sects)
        print(vol.strides)
        print(vol.itemsize)
        self.assertIsInstance(vol, numpy.ndarray)
        self.assertEqual((cols, rows, sects), vol.shape)
        self.assertEqual(numpy.uint8, vol.dtype)

    def test_default(self):
        """Test that the arguments to swapaxes are correct"""
        vol = utils.get_vol(10, 20, 30)
        # 1, 2, 3
        orientation = 1, 2, 3
        new_vol = utils.change_orientation(orientation, vol)
        self.assertEqual((10, 20, 30), new_vol.shape)
        # 3, 2, 1
        orientation = 3, 2, 1
        new_vol = utils.change_orientation(orientation, vol)
        self.assertEqual((30, 20, 10), new_vol.shape)
        # 2, 1, 3
        orientation = 2, 1, 3
        new_vol = utils.change_orientation(orientation, vol)
        self.assertEqual((20, 10, 30), new_vol.shape)
        # 1, 3, 2
        orientation = 1, 3, 2
        new_vol = utils.change_orientation(orientation, vol)
        self.assertEqual((10, 30, 20), new_vol.shape)
        # 3, 1, 2
        orientation = 3, 1, 2
        new_vol = utils.change_orientation(orientation, vol)
        self.assertEqual((30, 10, 20), new_vol.shape)
        # 2, 3, 1
        orientation = 2, 3, 1
        new_vol = utils.change_orientation(orientation, vol)
        self.assertEqual((20, 30, 10), new_vol.shape)

    def test_get_permutation_matrix(self):
        """Test the function for inferring permutation matrices"""
        self.assertTrue(
            numpy.array_equal(
                self.p12,
                models.PermutationMatrix.from_orientations((1, 2, 3), (1, 3, 2))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.p02,
                models.PermutationMatrix.from_orientations((1, 2, 3), (3, 2, 1))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.p01,
                models.PermutationMatrix.from_orientations((1, 2, 3), (2, 1, 3))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                numpy.dot(self.p01, self.p02),
                models.PermutationMatrix.from_orientations((1, 2, 3), (3, 1, 2))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                numpy.dot(self.p02, self.p01),
                models.PermutationMatrix.from_orientations((1, 2, 3), (2, 3, 1))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                numpy.dot(self.p01, self.p01),
                models.PermutationMatrix.from_orientations((1, 2, 3), (1, 2, 3))
            )
        )
        # sanity checks
        with self.assertRaises(ValueError):
            models.PermutationMatrix.from_orientations((1, 1, 1), (1, 2, 3))
        with self.assertRaises(ValueError):
            models.PermutationMatrix.from_orientations((1, 1, 1), (1, 2, 2))
        with self.assertRaises(ValueError):
            models.PermutationMatrix.from_orientations((1, 2, 3), (2, 3, 4))

    def test_permute_shape(self):
        """Test if permuting the shape is the same as performing a swap of axes"""
        # shape=(C,R,S), orientation=(1,2,3)
        # new_orientation=(3,2,1) => new_shape=(S,R,C)
        # permutation_matrix = models.PermutationMatrix.from_orientations(orientation, new_orientation)
        # is it true that shape * permutation_matrix = new_shape?
        vol = utils.get_vol(10, 20, 30)
        orientation = 1, 2, 3
        new_orientation = 3, 2, 1
        permutation_matrix = models.PermutationMatrix.from_orientations(orientation, new_orientation)
        new_shape = numpy.array(vol.shape).dot(permutation_matrix)
        # print(new_shape)
        new_vol = vol.reshape(new_shape)
        self.assertEqual((30, 20, 10), new_vol.shape)
        # shape=(C,R,S), orientation=(2,1,3)
        # new_orientation=(2,3,1) => new_shape=(C,S,R)?
        orientation = 2, 1, 3
        new_orientation = 2, 3, 1
        permutation_matrix = models.PermutationMatrix.from_orientations(orientation, new_orientation)
        new_shape = numpy.array(vol.shape).dot(permutation_matrix)
        new_vol = vol.reshape(new_shape)
        self.assertEqual((10, 30, 20), new_vol.shape)
        # this should be the same as (2,1,3)->(1,2,3)->(2,3,1)
        # the sequence of permutations doesn't matter
        orientation = 2, 1, 3
        intermediate_orientation = 1, 2, 3
        final_orientation = 2, 3, 1
        intermediate_permutation_matrix = models.PermutationMatrix.from_orientations(orientation,
                                                                                     intermediate_orientation)
        intermediate_shape = numpy.dot(
            numpy.array(vol.shape),
            intermediate_permutation_matrix
        )
        intermediate_vol = vol.reshape(intermediate_shape)
        self.assertEqual((20, 10, 30), intermediate_vol.shape)
        final_permutation_matrix = models.PermutationMatrix.from_orientations(intermediate_orientation,
                                                                              final_orientation)
        final_shape = numpy.array(intermediate_vol.shape).dot(final_permutation_matrix)
        final_vol = intermediate_vol.reshape(final_shape)
        self.assertEqual((10, 30, 20), final_vol.shape)
        # shape=(C,R,S), orientation=(3,1,2)
        # new_orientation=(1,2,3) => new_shape=(R,S,C)
        orientation = 3, 1, 2
        new_orientation = 1, 2, 3  # double permutation
        permutation_matrix = models.PermutationMatrix.from_orientations(orientation, new_orientation)
        new_shape = numpy.dot(numpy.array(vol.shape), permutation_matrix)
        new_vol = vol.reshape(new_shape)
        self.assertEqual((20, 30, 10), new_vol.shape)


class TestOrientation(unittest.TestCase):
    """
    `mapfile` provides a simple API to
    """

    @classmethod
    def setUpClass(cls) -> None:

        # random name
        cls.random_name = TEST_DATA_DIR / f"file-{secrets.token_urlsafe(3)}.map"
        # random size
        cls.cols, cls.rows, cls.sections = random.sample(range(10, 30), k=3)
        with models.MapFile(cls.random_name, 'w') as mapf:
            mapf.data = utils.get_vol(cls.cols, cls.rows, cls.sections)
            # mapf.data = voxel_size = 1.5

    @classmethod
    def tearDownClass(cls) -> None:
        """cleanup"""
        try:
            os.remove(cls.random_name)
        except FileNotFoundError:
            print(f"file {cls.random_name} already deleted!")

    def test_orientation(self):
        """"""
        orientation = models.Orientation(cols='X', rows='Y', sections='Z')
        self.assertEqual('X', orientation.cols)
        self.assertEqual('Y', orientation.rows)
        self.assertEqual('Z', orientation.sections)
        self.assertEqual((1, 3), orientation.shape)
        self.assertEqual("Orientation(cols='X', rows='Y', sections='Z')", repr(orientation))
        self.assertEqual("Orientation(cols='X', rows='Y', sections='Z')", str(orientation))
        # initialisation errors
        # must use x, y, z
        with self.assertRaises(ValueError):
            models.Orientation(cols='W', rows='Y', sections='Z')
        # can use lowercase
        orientation = models.Orientation(cols='x', rows='y', sections='z')
        self.assertEqual('X', orientation.cols)
        self.assertEqual('Y', orientation.rows)
        self.assertEqual('Z', orientation.sections)
        # no repetition of axes
        with self.assertRaises(ValueError):
            models.Orientation(cols='X', rows='X', sections='Y')
        # order doesn't matter
        orientation = models.Orientation(rows='Y', sections='X', cols='Z')
        self.assertEqual('Z', orientation.cols)
        self.assertEqual('Y', orientation.rows)
        self.assertEqual('X', orientation.sections)
        # create from integers using classmethod
        orientation = models.Orientation.from_integers((1, 2, 3))
        self.assertEqual('X', orientation.cols)
        self.assertEqual('Y', orientation.rows)
        self.assertEqual('Z', orientation.sections)
        # create from string using classmethod
        orientation = models.Orientation.from_string('YxZ')
        self.assertEqual('Y', orientation.cols)
        self.assertEqual('X', orientation.rows)
        self.assertEqual('Z', orientation.sections)

    def test_orientation_ops(self):
        """"""
        orientation1 = models.Orientation(cols='X', rows='Y', sections='Z')
        orientation2 = models.Orientation(cols='X', rows='Y', sections='Z')
        # trivial: the identity
        permutation_matrix = orientation1 / orientation2
        print(permutation_matrix)
        self.assertEqual(models.PermutationMatrix(numpy.eye(3, dtype=int)), permutation_matrix)
        # swap X and Z
        orientation1 = models.Orientation(cols='X', rows='Y', sections='Z')
        orientation2 = models.Orientation(cols='Z', rows='Y', sections='X')
        permutation_matrix = orientation1 / orientation2
        print(permutation_matrix)
        self.assertEqual(models.PermutationMatrix(numpy.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int)),
                         permutation_matrix)
        # Z Y X -> Z X Y
        orientation1 = models.Orientation(cols='Z', rows='Y', sections='X')
        orientation2 = models.Orientation(cols='Z', rows='X', sections='Y')
        permutation_matrix = orientation1 / orientation2
        print(permutation_matrix)
        self.assertEqual(models.PermutationMatrix(numpy.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=int)),
                         permutation_matrix)


class TestPermutationMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        # random name
        cls.random_name = TEST_DATA_DIR / f"file-{secrets.token_urlsafe(3)}.map"
        # random size
        cls.cols, cls.rows, cls.sections = random.sample(range(10, 30), k=3)
        with models.MapFile(cls.random_name, 'w') as mapf:
            mapf.data = utils.get_vol(cls.cols, cls.rows, cls.sections)
            # mapf.data = voxel_size = 1.5

    @classmethod
    def tearDownClass(cls) -> None:
        """cleanup"""
        try:
            os.remove(cls.random_name)
        except FileNotFoundError:
            print(f"file {cls.random_name} already deleted!")

    def test_permutation_matrix(self):
        """"""
        permutation_matrix = models.PermutationMatrix(numpy.eye(3, dtype=float))
        self.assertIsInstance(permutation_matrix, models.PermutationMatrix)
        self.assertEqual((3, 3), permutation_matrix.shape)
        self.assertEqual(3, permutation_matrix.rows)
        self.assertEqual(3, permutation_matrix.cols)
        self.assertEqual(int, permutation_matrix.dtype)
        # invalid data
        # non-binary
        with self.assertRaises(ValueError):
            models.PermutationMatrix(
                numpy.fromstring('1 0 0 0 2 0 0 0 1', sep=' ').reshape(3, 3)
            )

    def test_permutation_matrix_ops(self):
        """"""
        permutation_matrix1 = models.PermutationMatrix(
            numpy.fromstring('0 1 0 1 0 0 0 0 1', sep=' ', dtype=int).reshape(3, 3)
        )
        # LHS multiplication
        col_vector = numpy.fromstring('1 2 3', sep=' ', dtype=int).reshape(3, 1)
        product = permutation_matrix1 @ col_vector
        self.assertTrue(numpy.array_equal(numpy.fromstring('2 1 3', sep=' ', dtype=int).reshape(3, 1), product))
        # RHS multiplication
        row_vector = numpy.fromstring('1 2 3', sep=' ', dtype=int).reshape(1, 3)
        product = row_vector @ permutation_matrix1
        self.assertTrue(numpy.array_equal(numpy.fromstring('2 1 3', sep=' ', dtype=int).reshape(1, 3), product))
        # iRHS multiplication
        permutation_matrix1 @= permutation_matrix1
        self.assertTrue(numpy.array_equal(numpy.eye(3, dtype=int), permutation_matrix1))


class TestMapFile(unittest.TestCase):
    def setUp(self) -> None:
        self.test_fn = TEST_DATA_DIR / f"test-{secrets.token_urlsafe(3)}.map"

    def tearDown(self) -> None:
        try:
            os.remove(self.test_fn)
        except FileNotFoundError:
            print(f"test.map already deleted or not used in this test...", file=sys.stderr)

    def test_create_empty(self):
        """"""
        with self.assertRaises(ValueError):
            with models.MapFile(self.test_fn, file_mode='w') as mapf:
                # everything is None or the default value
                self.assertIsNone(mapf.nc)
                self.assertIsNone(mapf.nr)
                self.assertIsNone(mapf.ns)
                self.assertEqual(2, mapf.mode)
                self.assertEqual(0, mapf.ncstart)
                self.assertEqual(0, mapf.nrstart)
                self.assertEqual(0, mapf.nsstart)
                self.assertIsNone(mapf.nx)
                self.assertIsNone(mapf.ny)
                self.assertIsNone(mapf.nz)
                self.assertIsNone(mapf.x_length)
                self.assertIsNone(mapf.y_length)
                self.assertIsNone(mapf.z_length)
                self.assertEqual(90.0, mapf.alpha)
                self.assertEqual(90.0, mapf.beta)
                self.assertEqual(90.0, mapf.gamma)
                self.assertEqual(1, mapf.mapc)
                self.assertEqual(2, mapf.mapr)
                self.assertEqual(3, mapf.maps)
                self.assertIsNone(mapf.amin)
                self.assertIsNone(mapf.amax)
                self.assertIsNone(mapf.amean)
                self.assertEqual(1, mapf.ispg)
                self.assertEqual(0, mapf.nsymbt)
                self.assertEqual(0, mapf.lskflg)
                self.assertEqual(0.0, mapf.s11)
                self.assertEqual(0.0, mapf.s12)
                self.assertEqual(0.0, mapf.s13)
                self.assertEqual(0.0, mapf.s21)
                self.assertEqual(0.0, mapf.s22)
                self.assertEqual(0.0, mapf.s23)
                self.assertEqual(0.0, mapf.s31)
                self.assertEqual(0.0, mapf.s32)
                self.assertEqual(0.0, mapf.s33)
                self.assertEqual((0,) * 15, mapf.extra)
                self.assertEqual(b'MAP ', mapf.map)
                self.assertEqual(bytes([68, 68, 0, 0]), mapf.machst)
                self.assertEqual(0, mapf.nlabl)
                self.assertIsNone(mapf.cols)
                self.assertIsNone(mapf.rows)
                self.assertIsNone(mapf.sections)

    def test_create_with_data(self):
        """"""
        # create
        with models.MapFile(self.test_fn, file_mode='w') as mapf:
            # set data
            mapf.data = numpy.random.rand(10, 20, 30)  # sections, rows, cols
            # now the following should be automatically inferred from the data
            self.assertEqual(30, mapf.nc)
            self.assertEqual(20, mapf.nr)
            self.assertEqual(10, mapf.ns)
            self.assertEqual(2, mapf.mode)
            self.assertEqual(0, mapf.ncstart)
            self.assertEqual(0, mapf.nrstart)
            self.assertEqual(0, mapf.nsstart)
            self.assertEqual(30, mapf.nx)
            self.assertEqual(20, mapf.ny)
            self.assertEqual(10, mapf.nz)
            self.assertEqual(30.0, mapf.x_length)
            self.assertEqual(20.0, mapf.y_length)
            self.assertEqual(10.0, mapf.z_length)
            self.assertEqual(90.0, mapf.alpha)
            self.assertEqual(90.0, mapf.beta)
            self.assertEqual(90.0, mapf.gamma)
            self.assertEqual(1, mapf.mapc)
            self.assertEqual(2, mapf.mapr)
            self.assertEqual(3, mapf.maps)
            self.assertIsNotNone(mapf.amin)
            self.assertIsNotNone(mapf.amax)
            self.assertIsNotNone(mapf.amean)
            self.assertEqual(1, mapf.ispg)
            self.assertEqual(0, mapf.nsymbt)
            self.assertEqual(0, mapf.lskflg)
            self.assertEqual(0.0, mapf.s11)
            self.assertEqual(0.0, mapf.s12)
            self.assertEqual(0.0, mapf.s13)
            self.assertEqual(0.0, mapf.s21)
            self.assertEqual(0.0, mapf.s22)
            self.assertEqual(0.0, mapf.s23)
            self.assertEqual(0.0, mapf.s31)
            self.assertEqual(0.0, mapf.s32)
            self.assertEqual(0.0, mapf.s33)
            self.assertEqual((0,) * 15, mapf.extra)
            self.assertEqual(b'MAP ', mapf.map)
            self.assertEqual(bytes([68, 68, 0, 0]), mapf.machst)
            self.assertEqual(0, mapf.nlabl)
            self.assertEqual(30, mapf.cols)
            self.assertEqual(20, mapf.rows)
            self.assertEqual(10, mapf.sections)
            self.assertEqual("Orientation(cols='X', rows='Y', sections='Z')", str(mapf.orientation))
            self.assertEqual((1.0, 1.0, 1.0), mapf.voxel_size)
            self.assertEqual('float32', mapf.data.dtype.name)
        # mapfile is closed
        # read
        with models.MapFile(self.test_fn) as mapf2:
            self.assertEqual(30, mapf2.nc)
            self.assertEqual(20, mapf2.nr)
            self.assertEqual(10, mapf2.ns)
            self.assertEqual(2, mapf2.mode)
            self.assertEqual(0, mapf2.ncstart)
            self.assertEqual(0, mapf2.nrstart)
            self.assertEqual(0, mapf2.nsstart)
            self.assertEqual(30, mapf2.nx)
            self.assertEqual(20, mapf2.ny)
            self.assertEqual(10, mapf2.nz)
            self.assertEqual(30.0, mapf2.x_length)
            self.assertEqual(20.0, mapf2.y_length)
            self.assertEqual(10.0, mapf2.z_length)
            self.assertEqual(90.0, mapf2.alpha)
            self.assertEqual(90.0, mapf2.beta)
            self.assertEqual(90.0, mapf2.gamma)
            self.assertEqual(1, mapf2.mapc)
            self.assertEqual(2, mapf2.mapr)
            self.assertEqual(3, mapf2.maps)
            self.assertIsNotNone(mapf2.amin)
            self.assertIsNotNone(mapf2.amax)
            self.assertIsNotNone(mapf2.amean)
            self.assertEqual(1, mapf2.ispg)
            self.assertEqual(0, mapf2.nsymbt)
            self.assertEqual(0, mapf2.lskflg)
            self.assertEqual(0.0, mapf2.s11)
            self.assertEqual(0.0, mapf2.s12)
            self.assertEqual(0.0, mapf2.s13)
            self.assertEqual(0.0, mapf2.s21)
            self.assertEqual(0.0, mapf2.s22)
            self.assertEqual(0.0, mapf2.s23)
            self.assertEqual(0.0, mapf2.s31)
            self.assertEqual(0.0, mapf2.s32)
            self.assertEqual(0.0, mapf2.s33)
            self.assertEqual((0,) * 15, mapf2.extra)
            self.assertEqual(b'MAP ', mapf2.map)
            self.assertEqual(bytes([68, 68, 0, 0]), mapf2.machst)
            self.assertEqual(0, mapf2.nlabl)
            self.assertEqual(30, mapf2.cols)
            self.assertEqual(20, mapf2.rows)
            self.assertEqual(10, mapf2.sections)
            self.assertEqual("Orientation(cols='X', rows='Y', sections='Z')", str(mapf2.orientation))
            self.assertEqual((1.0, 1.0, 1.0), mapf2.voxel_size)
            self.assertEqual('float32', mapf2.data.dtype.name)

    def test_create_standard_then_modify_to_nonstandard(self):
        """"""
        # create
        with models.MapFile(self.test_fn, file_mode='w') as mapf:
            # set data
            mapf.data = numpy.random.rand(10, 20, 30)  # sections, rows, cols
            # change orientation to nonstandar YXZ
            mapf.orientation = models.Orientation(cols='Y', rows='X', sections='Z')
            # now the following should be automatically inferred from the data
            self.assertEqual(20, mapf.nc)
            self.assertEqual(30, mapf.nr)
            self.assertEqual(10, mapf.ns)
            # change orientation to nonstandard YZX
            # S=10, R=20, C=30
            # C=30, R=20, S=10
            # X Y Z
            # Y Z X
            # C'=20, R'=10, S'=30
            mapf.orientation = models.Orientation(cols='Y', rows='Z', sections='X')
            self.assertEqual(20, mapf.nc)
            self.assertEqual(10, mapf.nr)
            self.assertEqual(30, mapf.ns)
            # change orientation to nonstandard YZX
            mapf.orientation = models.Orientation(cols='Z', rows='Y', sections='X')
            # Z Y X
            self.assertEqual(10, mapf.nc)
            self.assertEqual(20, mapf.nr)
            self.assertEqual(30, mapf.ns)
            # let's do all the other orientation permutations
            # Z X Y
            mapf.orientation = models.Orientation(cols='Z', rows='X', sections='Y')
            self.assertEqual(10, mapf.nc)
            self.assertEqual(30, mapf.nr)
            self.assertEqual(20, mapf.ns)
            # X Z Y
            mapf.orientation = models.Orientation(cols='X', rows='Z', sections='Y')
            self.assertEqual(30, mapf.nc)
            self.assertEqual(10, mapf.nr)
            self.assertEqual(20, mapf.ns)

        # mapfile is closed
        # read
        with models.MapFile(self.test_fn) as mapf2:
            self.assertEqual(30, mapf2.nc)
            self.assertEqual(10, mapf2.nr)
            self.assertEqual(20, mapf2.ns)
            self.assertEqual(2, mapf2.mode)
            self.assertEqual(0, mapf2.ncstart)
            self.assertEqual(0, mapf2.nrstart)
            self.assertEqual(0, mapf2.nsstart)
            self.assertEqual(30, mapf2.nx)
            self.assertEqual(10, mapf2.ny)
            self.assertEqual(20, mapf2.nz)
            self.assertEqual(30.0, mapf2.x_length)
            self.assertEqual(10.0, mapf2.y_length)
            self.assertEqual(20.0, mapf2.z_length)
            self.assertEqual(90.0, mapf2.alpha)
            self.assertEqual(90.0, mapf2.beta)
            self.assertEqual(90.0, mapf2.gamma)
            self.assertEqual(1, mapf2.mapc)
            self.assertEqual(3, mapf2.mapr)
            self.assertEqual(2, mapf2.maps)
            self.assertIsNotNone(mapf2.amin)
            self.assertIsNotNone(mapf2.amax)
            self.assertIsNotNone(mapf2.amean)
            self.assertEqual(1, mapf2.ispg)
            self.assertEqual(0, mapf2.nsymbt)
            self.assertEqual(0, mapf2.lskflg)
            self.assertEqual(0.0, mapf2.s11)
            self.assertEqual(0.0, mapf2.s12)
            self.assertEqual(0.0, mapf2.s13)
            self.assertEqual(0.0, mapf2.s21)
            self.assertEqual(0.0, mapf2.s22)
            self.assertEqual(0.0, mapf2.s23)
            self.assertEqual(0.0, mapf2.s31)
            self.assertEqual(0.0, mapf2.s32)
            self.assertEqual(0.0, mapf2.s33)
            self.assertEqual((0,) * 15, mapf2.extra)
            self.assertEqual(b'MAP ', mapf2.map)
            self.assertEqual(bytes([68, 68, 0, 0]), mapf2.machst)
            self.assertEqual(0, mapf2.nlabl)
            self.assertEqual(30, mapf2.cols)
            self.assertEqual(10, mapf2.rows)
            self.assertEqual(20, mapf2.sections)
            self.assertEqual("Orientation(cols='X', rows='Z', sections='Y')", str(mapf2.orientation))
            self.assertEqual((1.0, 1.0, 1.0), mapf2.voxel_size)
            self.assertEqual('float32', mapf2.data.dtype.name)

    def test_read_and_modify(self):
        """"""
        with models.MapFile(self.test_fn, file_mode='w') as mapf:
            # set data
            mapf.data = numpy.random.rand(10, 20, 30)  # sections, rows, cols
            self.assertEqual(30, mapf.nc)
            self.assertEqual(20, mapf.nr)
            self.assertEqual(10, mapf.ns)
            self.assertEqual(2, mapf.mode)
            self.assertEqual(0, mapf.ncstart)
            self.assertEqual(0, mapf.nrstart)
            self.assertEqual(0, mapf.nsstart)
            self.assertEqual(30, mapf.nx)
            self.assertEqual(20, mapf.ny)
            self.assertEqual(10, mapf.nz)
            self.assertEqual(30.0, mapf.x_length)
            self.assertEqual(20.0, mapf.y_length)
            self.assertEqual(10.0, mapf.z_length)
            self.assertEqual(90.0, mapf.alpha)
            self.assertEqual(90.0, mapf.beta)
            self.assertEqual(90.0, mapf.gamma)
            self.assertEqual(1, mapf.mapc)
            self.assertEqual(2, mapf.mapr)
            self.assertEqual(3, mapf.maps)
            self.assertIsNotNone(mapf.amin)
            self.assertIsNotNone(mapf.amax)
            self.assertIsNotNone(mapf.amean)
            self.assertEqual(1, mapf.ispg)
            self.assertEqual(0, mapf.nsymbt)
            self.assertEqual(0, mapf.lskflg)
            self.assertEqual(0.0, mapf.s11)
            self.assertEqual(0.0, mapf.s12)
            self.assertEqual(0.0, mapf.s13)
            self.assertEqual(0.0, mapf.s21)
            self.assertEqual(0.0, mapf.s22)
            self.assertEqual(0.0, mapf.s23)
            self.assertEqual(0.0, mapf.s31)
            self.assertEqual(0.0, mapf.s32)
            self.assertEqual(0.0, mapf.s33)
            self.assertEqual((0,) * 15, mapf.extra)
            self.assertEqual(b'MAP ', mapf.map)
            self.assertEqual(bytes([68, 68, 0, 0]), mapf.machst)
            self.assertEqual(0, mapf.nlabl)
            self.assertEqual(30, mapf.cols)
            self.assertEqual(20, mapf.rows)
            self.assertEqual(10, mapf.sections)
            self.assertEqual("Orientation(cols='X', rows='Y', sections='Z')", str(mapf.orientation))
            self.assertEqual((1.0, 1.0, 1.0), mapf.voxel_size)

        # read and modify
        with models.MapFile(self.test_fn, file_mode='r+') as mapf2:
            self.assertEqual(30, mapf2.nc)
            self.assertEqual(20, mapf2.nr)
            self.assertEqual(10, mapf2.ns)
            self.assertEqual(1, mapf2.mapc)
            self.assertEqual(2, mapf2.mapr)
            self.assertEqual(3, mapf2.maps)
            # change the orientation
            mapf2.orientation = models.Orientation(cols='Z', rows='Y', sections='X')
            self.assertEqual(10, mapf2.nc)
            self.assertEqual(20, mapf2.nr)
            self.assertEqual(30, mapf2.ns)
            self.assertEqual(3, mapf2.mapc)
            self.assertEqual(2, mapf2.mapr)
            self.assertEqual(1, mapf2.maps)
            print(mapf2)

    def test_voxel_size(self):
        """"""
        # we start off with a standard orientation but anisotropic
        # then we change to nonstandard
        # examine the voxel size
        with mapfile.MapFile(self.test_fn, file_mode='w') as mapf:
            mapf.data = numpy.random.rand(3, 4, 5)
            # x=1.7, y=2.4, z=9.3
            # X=8.5, Y=9.6, Z=27.9
            mapf.voxel_size = 1.7, 2.4, 9.3
            self.assertEqual(8.5, mapf.x_length)
            self.assertEqual(9.6, mapf.y_length)
            self.assertAlmostEqual(27.9, mapf.z_length)
            # what if we change the orientation to ZYX
            mapf.orientation = models.Orientation(cols='Z', rows='Y', sections='X')
            # the voxel sizes also get permuted
            self.assertEqual((9.3, 2.4, 1.7), mapf.voxel_size)
            # but the lengths should change because we now have a different number of voxels on the same length
            self.assertAlmostEqual(27.9, mapf.x_length)
            self.assertEqual(9.6, mapf.y_length)
            self.assertAlmostEqual(8.5, mapf.z_length)

    def test_create_anisotropic_voxels(self):
        # create with anisotropic voxel sizes
        with mapfile.MapFile(self.test_fn, 'w', voxel_size=(3.7, 2.6, 1.5)) as mapf:
            mapf.data = numpy.random.rand(12, 22, 17)
            self.assertEqual((3.7, 2.6, 1.5), mapf.voxel_size)

    def test_create_with_nonstardard_and_anisotropic(self):
        """"""
        with mapfile.MapFile(
                self.test_fn, 'w',
                orientation=models.Orientation(cols='Y', rows='Z', sections='X'),
                voxel_size=(3.7, 2.6, 1.5)
        ) as mapf:
            mapf.data = numpy.random.rand(12, 22, 17)
            self.assertEqual((2, 3, 1), mapf.orientation.to_integers())
            self.assertEqual((2.6, 1.5, 3.7), mapf.voxel_size)

    def test_create_with_map_mode(self):
        """"""
        with mapfile.MapFile(
                self.test_fn,
                file_mode='w',
                map_mode=0,
                voxel_size=(8.7, 9.2, 1.2),
                orientation=models.Orientation.from_integers((2, 3, 1))
        ) as mapf:
            mapf.data = numpy.random.randint(0, 1, size=(11, 9, 16))
            self.assertEqual(0, mapf.mode)

    def test_change_map_mode_int(self):
        """"""
        # int to int valid: 0, 1, 3, 6
        # float to float valid: 2, 12
        # complex to complex: 3, 4
        # int to/from float invalid
        with mapfile.MapFile(self.test_fn, 'w', map_mode=0) as mapf:
            mapf.data = numpy.random.randint(0, 1, (5, 5, 5))
            self.assertEqual(1, mapf.data.itemsize)
            mapf.mode = 1
            self.assertEqual(2, mapf.data.itemsize)
            mapf.mode = 3
            self.assertEqual(4, mapf.data.itemsize)
            # change back
            mapf.mode = 0
            self.assertEqual(1, mapf.data.itemsize)
            mapf.mode = 1
            self.assertEqual(2, mapf.data.itemsize)

        with self.assertRaises(UserWarning):
            with mapfile.MapFile(self.test_fn, 'r+') as mapf2:
                mapf.mode = 2
                self.assertEqual(4, mapf.data.itemsize)

        with self.assertRaises(UserWarning):
            with mapfile.MapFile(self.test_fn, 'r+') as mapf2:
                mapf.mode = 12
                self.assertEqual(2, mapf.data.itemsize)

    def test_change_map_mode_float(self):
        """"""
        with mapfile.MapFile(self.test_fn, 'w') as mapf:
            mapf.data = numpy.random.rand(8, 8, 8)
            self.assertEqual(4, mapf.data.itemsize)
            mapf.mode = 12
            self.assertEqual(2, mapf.data.itemsize)

        with self.assertRaises(UserWarning):
            with mapfile.MapFile(self.test_fn, 'r+') as mapf:
                mapf.mode = 0
                self.assertEqual(1, mapf.data.itemsize)


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        # random name
        cls.random_name = TEST_DATA_DIR / f"file-{secrets.token_urlsafe(3)}.map"
        # random size
        cls.cols, cls.rows, cls.sections = random.sample(range(10, 30), k=3)
        with models.MapFile(cls.random_name, 'w') as mapf:
            mapf.data = utils.get_vol(cls.cols, cls.rows, cls.sections)
            # mapf.data = voxel_size = 1.5

    @classmethod
    def tearDownClass(cls) -> None:
        """cleanup"""
        try:
            os.remove(cls.random_name)
        except FileNotFoundError:
            print(f"file {cls.random_name} already deleted!")

    def test_get_orientation(self):
        """"""
        # by default, orientation is XYZ
        with models.MapFile(self.random_name) as mapf:
            self.assertIsInstance(mapf.orientation, models.Orientation)
            self.assertEqual('X', mapf.orientation.cols)
            self.assertEqual('Y', mapf.orientation.rows)
            self.assertEqual('Z', mapf.orientation.sections)
