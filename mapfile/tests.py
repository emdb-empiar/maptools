import os
import random
import secrets
import sys
import unittest

import numpy

from mapfile import models, cli, managers


def get_vol(cols, rows, sects, dtype=numpy.uint8):
    return numpy.empty(shape=(cols, rows, sects), dtype=dtype)


def change_orientation(axes, vol):
    """Always assumes that we start with 1, 2, 3"""
    if axes == (1, 2, 3):
        return vol
    elif axes == (3, 2, 1):
        return numpy.swapaxes(vol, 0, 2)
    elif axes == (2, 1, 3):
        return numpy.swapaxes(vol, 0, 1)
    elif axes == (1, 3, 2):
        return numpy.swapaxes(vol, 1, 2)
    elif axes == (3, 1, 2):
        # double
        inter_vol = numpy.swapaxes(vol, 0, 1)
        return numpy.swapaxes(inter_vol, 0, 2)
    elif axes == (2, 3, 1):
        # double
        inter_vol = numpy.swapaxes(vol, 0, 1)
        return numpy.swapaxes(inter_vol, 1, 2)


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


class TestManagers(unittest.TestCase):
    def test_view(self):
        """"""
        fn = f"file-{secrets.token_urlsafe(3)}.map"
        with models.MapFile(fn, 'w') as mapfile:
            mapfile.data = numpy.random.rand(21, 31, 14)
        args = cli.cli(f"map view {fn}")
        ex = managers.view(args)

    def test_edit(self):
        """"""
        fn = f"file-{secrets.token_urlsafe(3)}.map"
        with models.MapFile(fn, 'w') as mapfile:
            mapfile.data = numpy.random.rand(37, 16, 62)
        # orientation
        args = cli.cli(f"map edit {fn} --orientation=YZX")
        managers.edit(args)
        with models.MapFile(fn) as mapfile2:
            self.assertEqual((2, 3, 1), mapfile2.orientation.to_integers())
            # todo: all the other attributes that should have been affected by the orientation
        # managers.view(args)
        os.remove(fn)


class Test(unittest.TestCase):
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
        vol = get_vol(cols, rows, sects)
        print(vol.strides)
        print(vol.itemsize)
        self.assertIsInstance(vol, numpy.ndarray)
        self.assertEqual((cols, rows, sects), vol.shape)
        self.assertEqual(numpy.uint8, vol.dtype)

    def test_default(self):
        """Test that the arguments to swapaxes are correct"""
        vol = get_vol(10, 20, 30)
        # 1, 2, 3
        orientation = 1, 2, 3
        new_vol = change_orientation(orientation, vol)
        self.assertEqual((10, 20, 30), new_vol.shape)
        # 3, 2, 1
        orientation = 3, 2, 1
        new_vol = change_orientation(orientation, vol)
        self.assertEqual((30, 20, 10), new_vol.shape)
        # 2, 1, 3
        orientation = 2, 1, 3
        new_vol = change_orientation(orientation, vol)
        self.assertEqual((20, 10, 30), new_vol.shape)
        # 1, 3, 2
        orientation = 1, 3, 2
        new_vol = change_orientation(orientation, vol)
        self.assertEqual((10, 30, 20), new_vol.shape)
        # 3, 1, 2
        orientation = 3, 1, 2
        new_vol = change_orientation(orientation, vol)
        self.assertEqual((30, 10, 20), new_vol.shape)
        # 2, 3, 1
        orientation = 2, 3, 1
        new_vol = change_orientation(orientation, vol)
        self.assertEqual((20, 30, 10), new_vol.shape)

    def test_get_permutation_matrix(self):
        """Test the function for inferring permutation matrices"""
        self.assertTrue(
            numpy.array_equal(
                self.p12,
                models.get_permutation_matrix((1, 2, 3), (1, 3, 2))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.p02,
                models.get_permutation_matrix((1, 2, 3), (3, 2, 1))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.p01,
                models.get_permutation_matrix((1, 2, 3), (2, 1, 3))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                numpy.dot(self.p01, self.p02),
                models.get_permutation_matrix((1, 2, 3), (3, 1, 2))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                numpy.dot(self.p02, self.p01),
                models.get_permutation_matrix((1, 2, 3), (2, 3, 1))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                numpy.dot(self.p01, self.p01),
                models.get_permutation_matrix((1, 2, 3), (1, 2, 3))
            )
        )
        # sanity checks
        with self.assertRaises(ValueError):
            models.get_permutation_matrix((1, 1, 1), (1, 2, 3))
        with self.assertRaises(ValueError):
            models.get_permutation_matrix((1, 1, 1), (1, 2, 2))
        with self.assertRaises(ValueError):
            models.get_permutation_matrix((1, 2, 3), (2, 3, 4))

    def test_permute_shape(self):
        """Test if permuting the shape is the same as performing a swap of axes"""
        # shape=(C,R,S), orientation=(1,2,3)
        # new_orientation=(3,2,1) => new_shape=(S,R,C)
        # permutation_matrix = models.get_permutation_matrix(orientation, new_orientation)
        # is it true that shape * permutation_matrix = new_shape?
        vol = get_vol(10, 20, 30)
        orientation = 1, 2, 3
        new_orientation = 3, 2, 1
        permutation_matrix = models.get_permutation_matrix(orientation, new_orientation)
        new_shape = numpy.array(vol.shape).dot(permutation_matrix)
        # print(new_shape)
        new_vol = vol.reshape(new_shape)
        self.assertEqual((30, 20, 10), new_vol.shape)
        # shape=(C,R,S), orientation=(2,1,3)
        # new_orientation=(2,3,1) => new_shape=(C,S,R)?
        orientation = 2, 1, 3
        new_orientation = 2, 3, 1
        permutation_matrix = models.get_permutation_matrix(orientation, new_orientation)
        new_shape = numpy.array(vol.shape).dot(permutation_matrix)
        new_vol = vol.reshape(new_shape)
        self.assertEqual((10, 30, 20), new_vol.shape)
        # this should be the same as (2,1,3)->(1,2,3)->(2,3,1)
        # the sequence of permutations doesn't matter
        orientation = 2, 1, 3
        intermediate_orientation = 1, 2, 3
        final_orientation = 2, 3, 1
        intermediate_permutation_matrix = models.get_permutation_matrix(orientation, intermediate_orientation)
        intermediate_shape = numpy.dot(
            numpy.array(vol.shape),
            intermediate_permutation_matrix
        )
        intermediate_vol = vol.reshape(intermediate_shape)
        self.assertEqual((20, 10, 30), intermediate_vol.shape)
        final_permutation_matrix = models.get_permutation_matrix(intermediate_orientation, final_orientation)
        final_shape = numpy.array(intermediate_vol.shape).dot(final_permutation_matrix)
        final_vol = intermediate_vol.reshape(final_shape)
        self.assertEqual((10, 30, 20), final_vol.shape)
        # shape=(C,R,S), orientation=(3,1,2)
        # new_orientation=(1,2,3) => new_shape=(R,S,C)
        orientation = 3, 1, 2
        new_orientation = 1, 2, 3  # double permutation
        permutation_matrix = models.get_permutation_matrix(orientation, new_orientation)
        new_shape = numpy.dot(numpy.array(vol.shape), permutation_matrix)
        new_vol = vol.reshape(new_shape)
        self.assertEqual((20, 30, 10), new_vol.shape)


class TestMapfile(unittest.TestCase):
    """
    `mapfile` provides a simple API to
    """

    @classmethod
    def setUpClass(cls) -> None:

        # random name
        cls.random_name = f"file-{secrets.token_urlsafe(3)}.mapfile"
        # random size
        cls.cols, cls.rows, cls.sections = random.sample(range(10, 30), k=3)
        with models.MapFile(cls.random_name, 'w') as mapfile:
            mapfile.data = get_vol(cls.cols, cls.rows, cls.sections)
            # mapfile.data = voxel_size = 1.5

    @classmethod
    def tearDownClass(cls) -> None:
        """cleanup"""
        try:
            os.remove(cls.random_name)
        except FileNotFoundError:
            print(f"file {cls.random_name} already deleted!")

    # def test_mrcfile(self):
    #     """Test behaviour with mrcfile"""
    #     with models.MapFile('file.mrc') as mapfile:
    #         mapfile.data = numpy.zeros(shape=(10, 20, 30), dtype=numpy.float32)
    #         # voxel size (isotropic)
    #         # c=30,
    #         mapfile.voxel_size = (3.0, 2.0, 1.0)
    #         print(f"orig shape = {mapfile.data.shape}")
    #         orientation = tuple(map(int, (mapfile.header.mapc, mapfile.header.mapr, mapfile.header.maps)))
    #         print(orientation)
    #         print(type(int(orientation[0])))
    #         new_orientation = 3, 2, 1
    #         permutation_matrix = models.get_permutation_matrix(orientation, new_orientation)
    #         print(permutation_matrix)
    #         new_shape = numpy.dot(mapfile.data.shape, permutation_matrix)
    #         print(f"{new_shape = }")
    #         # reset the data
    #         mapfile.set_data(mapfile.data.reshape(new_shape))
    #         print(f"new shape = {mapfile.data.shape}")
    #         # we have to also change the values of mapc, mapr, maps
    #         mapfile.header.mapc, mapfile.header.macr, mapfile.header.maps = new_orientation
    #         # and mx, my, mz
    #         mapfile.header.mx, mapfile.header.my, mapfile.header.mz = new_shape
    #         mapfile.voxel_size = 1.0
    #         # print(mapfile.print_header())
    #         mapfile.update_header_from_data()
    #         print(f"{mapfile.voxel_size = }")

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

    def test_get_orientation(self):
        """"""
        # by default, orientation is XYZ
        with models.MapFile(self.random_name) as mapfile:
            orientation = models.get_orientation(mapfile)
            self.assertIsInstance(orientation, models.Orientation)
            self.assertEqual('X', orientation.cols)
            self.assertEqual('Y', orientation.rows)
            self.assertEqual('Z', orientation.sections)


class TestMapFile(unittest.TestCase):
    def tearDown(self) -> None:
        try:
            os.remove('test.map')
        except FileNotFoundError:
            print(f"test.map already deleted or not used in this test...", file=sys.stderr)

    def test_create_empty(self):
        """"""
        with self.assertRaises(ValueError):
            with models.MapFile('test.map', file_mode='w') as mapfile:
                # everything is None
                self.assertIsNone(mapfile.nc)
                self.assertIsNone(mapfile.nr)
                self.assertIsNone(mapfile.ns)
                self.assertEqual(2, mapfile.mode)
                self.assertEqual(0, mapfile.ncstart)
                self.assertEqual(0, mapfile.nrstart)
                self.assertEqual(0, mapfile.nsstart)
                self.assertIsNone(mapfile.nx)
                self.assertIsNone(mapfile.ny)
                self.assertIsNone(mapfile.nz)
                self.assertIsNone(mapfile.x_length)
                self.assertIsNone(mapfile.y_length)
                self.assertIsNone(mapfile.z_length)
                self.assertEqual(90.0, mapfile.alpha)
                self.assertEqual(90.0, mapfile.beta)
                self.assertEqual(90.0, mapfile.gamma)
                self.assertEqual(1, mapfile.mapc)
                self.assertEqual(2, mapfile.mapr)
                self.assertEqual(3, mapfile.maps)
                self.assertIsNone(mapfile.amin)
                self.assertIsNone(mapfile.amax)
                self.assertIsNone(mapfile.amean)
                self.assertEqual(1, mapfile.ispg)
                self.assertEqual(0, mapfile.nsymbt)
                self.assertEqual(0, mapfile.lskflg)
                self.assertEqual(0.0, mapfile.s11)
                self.assertEqual(0.0, mapfile.s12)
                self.assertEqual(0.0, mapfile.s13)
                self.assertEqual(0.0, mapfile.s21)
                self.assertEqual(0.0, mapfile.s22)
                self.assertEqual(0.0, mapfile.s23)
                self.assertEqual(0.0, mapfile.s31)
                self.assertEqual(0.0, mapfile.s32)
                self.assertEqual(0.0, mapfile.s33)
                self.assertEqual((0,) * 15, mapfile.extra)
                self.assertEqual(b'MAP ', mapfile.map)
                self.assertEqual(bytes([68, 68, 0, 0]), mapfile.machst)
                self.assertEqual(0, mapfile.nlabl)
                self.assertIsNone(mapfile.cols)
                self.assertIsNone(mapfile.rows)
                self.assertIsNone(mapfile.sections)

    def test_create_with_data(self):
        """"""
        with models.MapFile('test.map', file_mode='w') as mapfile:
            # set data
            mapfile.data = numpy.random.rand(10, 20, 30)  # sections, rows, cols
            self.assertEqual(30, mapfile.nc)
            self.assertEqual(20, mapfile.nr)
            self.assertEqual(10, mapfile.ns)
            self.assertEqual(2, mapfile.mode)
            self.assertEqual(0, mapfile.ncstart)
            self.assertEqual(0, mapfile.nrstart)
            self.assertEqual(0, mapfile.nsstart)
            self.assertEqual(30, mapfile.nx)
            self.assertEqual(20, mapfile.ny)
            self.assertEqual(10, mapfile.nz)
            self.assertEqual(30.0, mapfile.x_length)
            self.assertEqual(20.0, mapfile.y_length)
            self.assertEqual(10.0, mapfile.z_length)
            self.assertEqual(90.0, mapfile.alpha)
            self.assertEqual(90.0, mapfile.beta)
            self.assertEqual(90.0, mapfile.gamma)
            self.assertEqual(1, mapfile.mapc)
            self.assertEqual(2, mapfile.mapr)
            self.assertEqual(3, mapfile.maps)
            self.assertIsNotNone(mapfile.amin)
            self.assertIsNotNone(mapfile.amax)
            self.assertIsNotNone(mapfile.amean)
            self.assertEqual(1, mapfile.ispg)
            self.assertEqual(0, mapfile.nsymbt)
            self.assertEqual(0, mapfile.lskflg)
            self.assertEqual(0.0, mapfile.s11)
            self.assertEqual(0.0, mapfile.s12)
            self.assertEqual(0.0, mapfile.s13)
            self.assertEqual(0.0, mapfile.s21)
            self.assertEqual(0.0, mapfile.s22)
            self.assertEqual(0.0, mapfile.s23)
            self.assertEqual(0.0, mapfile.s31)
            self.assertEqual(0.0, mapfile.s32)
            self.assertEqual(0.0, mapfile.s33)
            self.assertEqual((0,) * 15, mapfile.extra)
            self.assertEqual(b'MAP ', mapfile.map)
            self.assertEqual(bytes([68, 68, 0, 0]), mapfile.machst)
            self.assertEqual(0, mapfile.nlabl)
            self.assertEqual(30, mapfile.cols)
            self.assertEqual(20, mapfile.rows)
            self.assertEqual(10, mapfile.sections)
            self.assertEqual("Orientation(cols='X', rows='Y', sections='Z')", str(mapfile.orientation))
            self.assertEqual((1.0, 1.0, 1.0), mapfile.voxel_size)
            self.assertEqual('float32', mapfile.data.dtype.name)

        # read
        with models.MapFile('test.map') as mapfile2:
            self.assertEqual(30, mapfile2.nc)
            self.assertEqual(20, mapfile2.nr)
            self.assertEqual(10, mapfile2.ns)
            self.assertEqual(2, mapfile2.mode)
            self.assertEqual(0, mapfile2.ncstart)
            self.assertEqual(0, mapfile2.nrstart)
            self.assertEqual(0, mapfile2.nsstart)
            self.assertEqual(30, mapfile2.nx)
            self.assertEqual(20, mapfile2.ny)
            self.assertEqual(10, mapfile2.nz)
            self.assertEqual(30.0, mapfile2.x_length)
            self.assertEqual(20.0, mapfile2.y_length)
            self.assertEqual(10.0, mapfile2.z_length)
            self.assertEqual(90.0, mapfile2.alpha)
            self.assertEqual(90.0, mapfile2.beta)
            self.assertEqual(90.0, mapfile2.gamma)
            self.assertEqual(1, mapfile2.mapc)
            self.assertEqual(2, mapfile2.mapr)
            self.assertEqual(3, mapfile2.maps)
            self.assertIsNotNone(mapfile2.amin)
            self.assertIsNotNone(mapfile2.amax)
            self.assertIsNotNone(mapfile2.amean)
            self.assertEqual(1, mapfile2.ispg)
            self.assertEqual(0, mapfile2.nsymbt)
            self.assertEqual(0, mapfile2.lskflg)
            self.assertEqual(0.0, mapfile2.s11)
            self.assertEqual(0.0, mapfile2.s12)
            self.assertEqual(0.0, mapfile2.s13)
            self.assertEqual(0.0, mapfile2.s21)
            self.assertEqual(0.0, mapfile2.s22)
            self.assertEqual(0.0, mapfile2.s23)
            self.assertEqual(0.0, mapfile2.s31)
            self.assertEqual(0.0, mapfile2.s32)
            self.assertEqual(0.0, mapfile2.s33)
            self.assertEqual((0,) * 15, mapfile2.extra)
            self.assertEqual(b'MAP ', mapfile2.map)
            self.assertEqual(bytes([68, 68, 0, 0]), mapfile2.machst)
            self.assertEqual(0, mapfile2.nlabl)
            self.assertEqual(30, mapfile2.cols)
            self.assertEqual(20, mapfile2.rows)
            self.assertEqual(10, mapfile2.sections)
            self.assertEqual("Orientation(cols='X', rows='Y', sections='Z')", str(mapfile2.orientation))
            self.assertEqual((1.0, 1.0, 1.0), mapfile2.voxel_size)
            self.assertEqual('float32', mapfile2.data.dtype.name)

    def test_read_and_modify(self):
        """"""
        with models.MapFile('test-other.map', file_mode='w') as mapfile:
            # set data
            mapfile.data = numpy.random.rand(10, 20, 30)  # sections, rows, cols
            self.assertEqual(30, mapfile.nc)
            self.assertEqual(20, mapfile.nr)
            self.assertEqual(10, mapfile.ns)
            self.assertEqual(2, mapfile.mode)
            self.assertEqual(0, mapfile.ncstart)
            self.assertEqual(0, mapfile.nrstart)
            self.assertEqual(0, mapfile.nsstart)
            self.assertEqual(30, mapfile.nx)
            self.assertEqual(20, mapfile.ny)
            self.assertEqual(10, mapfile.nz)
            self.assertEqual(30.0, mapfile.x_length)
            self.assertEqual(20.0, mapfile.y_length)
            self.assertEqual(10.0, mapfile.z_length)
            self.assertEqual(90.0, mapfile.alpha)
            self.assertEqual(90.0, mapfile.beta)
            self.assertEqual(90.0, mapfile.gamma)
            self.assertEqual(1, mapfile.mapc)
            self.assertEqual(2, mapfile.mapr)
            self.assertEqual(3, mapfile.maps)
            self.assertIsNotNone(mapfile.amin)
            self.assertIsNotNone(mapfile.amax)
            self.assertIsNotNone(mapfile.amean)
            self.assertEqual(1, mapfile.ispg)
            self.assertEqual(0, mapfile.nsymbt)
            self.assertEqual(0, mapfile.lskflg)
            self.assertEqual(0.0, mapfile.s11)
            self.assertEqual(0.0, mapfile.s12)
            self.assertEqual(0.0, mapfile.s13)
            self.assertEqual(0.0, mapfile.s21)
            self.assertEqual(0.0, mapfile.s22)
            self.assertEqual(0.0, mapfile.s23)
            self.assertEqual(0.0, mapfile.s31)
            self.assertEqual(0.0, mapfile.s32)
            self.assertEqual(0.0, mapfile.s33)
            self.assertEqual((0,) * 15, mapfile.extra)
            self.assertEqual(b'MAP ', mapfile.map)
            self.assertEqual(bytes([68, 68, 0, 0]), mapfile.machst)
            self.assertEqual(0, mapfile.nlabl)
            self.assertEqual(30, mapfile.cols)
            self.assertEqual(20, mapfile.rows)
            self.assertEqual(10, mapfile.sections)
            self.assertEqual("Orientation(cols='X', rows='Y', sections='Z')", str(mapfile.orientation))
            self.assertEqual((1.0, 1.0, 1.0), mapfile.voxel_size)

        # read and modify
        with models.MapFile('test-other.map', file_mode='r+') as mapfile2:
            self.assertEqual(30, mapfile2.nc)
            self.assertEqual(20, mapfile2.nr)
            self.assertEqual(10, mapfile2.ns)
            self.assertEqual(1, mapfile2.mapc)
            self.assertEqual(2, mapfile2.mapr)
            self.assertEqual(3, mapfile2.maps)
            # change the orientation
            mapfile2.set_orientation(models.Orientation(cols='Z', rows='Y', sections='X'))
            self.assertEqual(10, mapfile2.nc)
            self.assertEqual(20, mapfile2.nr)
            self.assertEqual(30, mapfile2.ns)
            self.assertEqual(3, mapfile2.mapc)
            self.assertEqual(2, mapfile2.mapr)
            self.assertEqual(1, mapfile2.maps)
            print(mapfile2)
