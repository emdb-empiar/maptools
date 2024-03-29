import os
import pathlib
import random
import secrets
import struct
import sys
import unittest

import numpy

from maptools import models, cli, managers, utils, engines

BASE_DIR = pathlib.Path(__file__).parent.parent
TEST_DATA_DIR = BASE_DIR / 'test_data'


class TestCLI(unittest.TestCase):
    def test_view(self):
        """"""
        args = cli.cli(f"map view file.map")
        self.assertEqual('view', args.command)
        self.assertEqual('file.map', args.file)
        self.assertFalse(args.verbose)
        self.assertFalse(args.colour)

    def test_edit(self):
        args = cli.cli(f"map edit file.map")
        self.assertEqual('edit', args.command)
        self.assertEqual('file.map', args.file)
        self.assertIsNone(args.orientation)
        self.assertIsNone(args.voxel_sizes)
        self.assertEqual('r+', args.file_mode)
        self.assertEqual([0, 0, 0], args.start)
        self.assertIsNone(args.map_mode)
        self.assertFalse(args.verbose)
        self.assertFalse(args.colour)

    def test_create_with_no_arguments(self):
        args = cli.cli(f"map create file.map")
        # self.assertEqual('create', args.command)
        self.assertIsNone(args.command)
        self.assertEqual('file.map', args.file)
        self.assertIsNone(args.orientation)
        self.assertIsNone(args.voxel_sizes)
        self.assertEqual([10, 10, 10], args.size)
        self.assertEqual('w', args.file_mode)
        self.assertEqual([0, 0, 0], args.start)
        self.assertIsNone(args.map_mode)
        self.assertEqual(0, args.min)
        self.assertEqual(10, args.max)
        self.assertEqual('zeros', args.voxel_values)
        self.assertFalse(args.verbose)
        self.assertFalse(args.colour)

    def test_create_with_arguments(self):
        args = cli.cli(f"map create file.map -O yzx -M 0 -V 3.0 2.0 1.0")
        self.assertEqual('create', args.command)
        self.assertEqual('file.map', args.file)
        self.assertEqual('YZX', args.orientation)
        self.assertAlmostEqual(3.0, args.voxel_sizes[0])
        self.assertAlmostEqual(2.0, args.voxel_sizes[1])
        self.assertAlmostEqual(1.0, args.voxel_sizes[2])
        self.assertEqual([10, 10, 10], args.size)
        self.assertEqual('w', args.file_mode)
        self.assertEqual([0, 0, 0], args.start)
        self.assertEqual(0, args.map_mode)
        self.assertEqual(0, args.min)
        self.assertEqual(10, args.max)
        self.assertEqual('zeros', args.voxel_values)
        self.assertFalse(args.verbose)
        self.assertFalse(args.colour)

    def test_sample(self):
        """"""
        args = cli.cli(f"map sample --factor 2 file.map")
        self.assertEqual('sample', args.command)
        self.assertEqual(2, args.factor)
        self.assertEqual('file.map', args.file)


class TestManagers(unittest.TestCase):
    def setUp(self) -> None:
        self.test_fn = TEST_DATA_DIR / f"file-{secrets.token_urlsafe(3)}.map"
        self.test_fn2 = TEST_DATA_DIR / f"file-{secrets.token_urlsafe(3)}.map"
        shape = random.choices(range(12, 52, 4), k=3)
        self.shape = shape
        with models.MapFile(self.test_fn, 'w') as mapfile:
            mapfile.data = numpy.random.rand(*shape)
            mapfile.voxel_size = 3.78

    def tearDown(self) -> None:
        try:
            os.remove(self.test_fn)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.test_fn2)
        except FileNotFoundError:
            pass

    def test_view(self):
        """"""
        args = cli.cli(f"map view {self.test_fn}")
        ex = managers.view(args)
        self.assertEqual(0, ex)

    def test_edit(self):
        """"""
        args = cli.cli(f"map edit {self.test_fn} -o {self.test_fn2}")
        managers.edit(args)
        # no changes then both files should be exactly alike
        with models.MapFile(self.test_fn, file_mode='r') as map1, models.MapFile(self.test_fn2, file_mode='r') as map2:
            self.assertEqual(map1.nc, map2.nc)
            self.assertEqual(map1.nr, map2.nr)
            self.assertEqual(map1.ns, map2.ns)
            self.assertEqual(map1.nx, map2.nx)
            self.assertEqual(map1.ny, map2.ny)
            self.assertEqual(map1.nz, map2.nz)
            self.assertEqual(map1.x_length, map2.x_length)
            self.assertEqual(map1.y_length, map2.y_length)
            self.assertEqual(map1.z_length, map2.z_length)
            self.assertEqual(map1.alpha, map2.alpha)
            self.assertEqual(map1.beta, map2.beta)
            self.assertEqual(map1.gamma, map2.gamma)
            self.assertEqual(map1.mapc, map2.mapc)
            self.assertEqual(map1.mapr, map2.mapr)
            self.assertEqual(map1.maps, map2.maps)
            self.assertEqual(map1.amin, map2.amin)
            self.assertEqual(map1.amax, map2.amax)
            self.assertEqual(map1.amean, map2.amean)
            self.assertEqual(map1.ispg, map2.ispg)
            self.assertEqual(map1.nsymbt, map2.nsymbt)
            self.assertEqual(map1.lskflg, map2.lskflg)
            self.assertEqual(map1.s11, map2.s11)
            self.assertEqual(map1.s12, map2.s12)
            self.assertEqual(map1.s13, map2.s13)
            self.assertEqual(map1.s21, map2.s21)
            self.assertEqual(map1.s22, map2.s22)
            self.assertEqual(map1.s23, map2.s23)
            self.assertEqual(map1.s31, map2.s31)
            self.assertEqual(map1.s32, map2.s32)
            self.assertEqual(map1.s33, map2.s33)
            self.assertEqual(map1.t1, map2.t1)
            self.assertEqual(map1.t2, map2.t2)
            self.assertEqual(map1.t3, map2.t3)
            self.assertEqual(map1.extra, map2.extra)
            self.assertEqual(map1.map, map2.map)
            self.assertEqual(map1.machst, map2.machst)
            self.assertEqual(map1.rms, map2.rms)
            self.assertNotEqual(map1.nlabl, map2.nlabl)
            self.assertNotEqual(map1.name, map2.name)
            self.assertTrue(numpy.array_equal(map1.data, map2.data))
            self.assertEqual(map1.orientation, map2.orientation)
            self.assertEqual(map1.mode, map2.mode)
            self.assertEqual(map1.start, map2.start)
            self.assertEqual(map1.voxel_size, map2.voxel_size)
            self.assertNotEqual(map1.labels, map2.labels)

    def test_edit_change_orientation(self):
        """"""
        # orientation
        args = cli.cli(f"map edit {self.test_fn} --orientation=YZX")
        managers.edit(args)
        with models.MapFile(self.test_fn) as mapfile:
            self.assertEqual((2, 3, 1), mapfile.orientation.to_integers())

    def test_edit_change_mode(self):
        """Test that changing the mode:

        - preserves the data
        - preserves space information
        """
        # before the change
        with models.MapFile(self.test_fn) as mapfile:
            self.assertAlmostEqual(3.78, mapfile.voxel_size[0], places=6)
            self.assertAlmostEqual(3.78, mapfile.voxel_size[1], places=6)
            self.assertAlmostEqual(3.78, mapfile.voxel_size[2], places=6)
        # edit
        args = cli.cli(f"map edit {self.test_fn} --map-mode=1 -o {self.test_fn2}")
        managers.edit(args)
        # managers.view(cli.cli(f"map view --colour {self.test_fn}"))
        # args = cli.cli(f"map edit {self.test_fn} --map-mode=1 -o {self.test_fn2}")
        # managers.edit(args)
        # after the change
        with models.MapFile(self.test_fn2) as mapfile2:
            self.assertAlmostEqual(3.78, mapfile2.voxel_size[0], places=6)
            self.assertAlmostEqual(3.78, mapfile2.voxel_size[1], places=6)
            self.assertAlmostEqual(3.78, mapfile2.voxel_size[2], places=6)
        # managers.view(cli.cli(f"map view --colour {self.test_fn}"))

    def test_file_modes(self):
        """Demonstrate that modifying a file with r+b does not truncate file. Call file.truncate() to do so."""
        with open(self.test_fn, 'wb') as f:
            f.write(struct.pack('<10f', *(0.0,) * 10))
        with open(self.test_fn, 'rb') as g:
            data = struct.unpack('<10f', g.read(10 * 4))
            print(f"before: {data}")
        with open(self.test_fn, 'r+b') as h:
            print(f"{h.tell()}")
            h.write(struct.pack('<5f', *(1.0,) * 5))
        with open(self.test_fn, 'rb') as g:
            data = struct.unpack('<10f', g.read(10 * 4))
            print(f"after: {data}")

    def test_edit_with_label(self):
        """"""
        args = cli.cli(f"map edit {self.test_fn}")
        managers.edit(args)
        with models.MapFile(self.test_fn, colour=True) as mapfile:
            self.assertRegex(mapfile.get_label(0), r".*edit.*")

    def test_edit_with_outfile(self):
        """"""
        args = cli.cli(f"map edit {self.test_fn} -c -O zyx -o {self.test_fn2}")
        managers.edit(args)
        with models.MapFile(self.test_fn2, colour=True) as mapfile:
            self.assertRegex(mapfile.get_label(0), r".*copied.*")

    def test_create(self):
        """"""
        args = cli.cli(f"map create {self.test_fn} -O XYZ -V 1 1 1 -M 2")
        managers.create(args)
        with models.MapFile(self.test_fn, colour=True) as mapfile:
            print(mapfile)
            self.assertRegex(mapfile.get_label(0), r".*creat.*")

    def test_create_ad_hoc(self):
        """"""
        args = cli.cli(f"map create {self.test_fn} -O XYZ -V 1.9 9.1 7.1 -M 2 -S -5 -5 -5 -s 30 15 28")
        managers.create(args)
        with models.MapFile(self.test_fn, colour=True) as mapfile:
            print(mapfile)
            self.assertEqual((1, 2, 3), mapfile.orientation.to_integers())
            self.assertAlmostEqual(1.9, mapfile.voxel_size[0], places=6)
            self.assertAlmostEqual(9.1, mapfile.voxel_size[1], places=6)
            self.assertAlmostEqual(7.1, mapfile.voxel_size[2], places=6)
            self.assertEqual((-5, -5, -5), mapfile.start)
            self.assertEqual((28, 15, 30), mapfile.data.shape)
            print(mapfile.data)
            self.assertTrue(numpy.array_equal(numpy.zeros(shape=args.size[::-1], dtype=numpy.int8), mapfile.data))

    def test_create_random(self):
        """"""
        args = cli.cli(f"map create {self.test_fn} -O XYZ -M 12 -V 1 1 1 --voxel-values random")
        managers.create(args)
        with models.MapFile(self.test_fn) as mapfile:
            # how do we know it's random?
            # densities: min, max, mean and rms should all be different
            # the length of the set of these values should be 4
            self.assertEqual(4, len({mapfile.amin, mapfile.amax, mapfile.amean, mapfile.rms}))

    def test_resample(self):
        """"""
        args = cli.cli(f"map sample --factor=4 {self.test_fn}")
        managers.sample(args)
        c, r, s = self.shape
        with models.MapFile(self.test_fn) as mapfile:
            self.assertEqual((c // args.factor, r // args.factor, s // args.factor), mapfile.data.shape)


class TestEngines(unittest.TestCase):
    def test_grid_resample(self):
        vol = utils.get_vol(10, 10, 10)
        factor = 2
        resampled_vol = engines.grid_resample(vol, factor)
        self.assertEqual((5, 5, 5), resampled_vol.shape)


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
        vol = utils.get_vol(cols, rows, sects)
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
        with self.assertRaises(AssertionError):
            models.PermutationMatrix.from_orientations((1, 1, 1), (1, 2, 3))
        with self.assertRaises(AssertionError):
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
        new_shape = numpy.array(vol.shape) @ permutation_matrix
        # print(new_shape)
        new_vol = vol.reshape(new_shape)
        self.assertEqual((30, 20, 10), new_vol.shape)
        # shape=(C,R,S), orientation=(2,1,3)
        # new_orientation=(2,3,1) => new_shape=(C,S,R)?
        orientation = 2, 1, 3
        new_orientation = 2, 3, 1
        permutation_matrix = models.PermutationMatrix.from_orientations(orientation, new_orientation)
        new_shape = numpy.array(vol.shape) @ permutation_matrix
        new_vol = vol.reshape(new_shape)
        self.assertEqual((10, 30, 20), new_vol.shape)
        # this should be the same as (2,1,3)->(1,2,3)->(2,3,1)
        # the sequence of permutations doesn't matter
        orientation = 2, 1, 3
        intermediate_orientation = 1, 2, 3
        final_orientation = 2, 3, 1
        intermediate_permutation_matrix = models.PermutationMatrix.from_orientations(orientation,
                                                                                     intermediate_orientation)
        intermediate_shape = numpy.array(vol.shape) @ intermediate_permutation_matrix
        intermediate_vol = vol.reshape(intermediate_shape)
        self.assertEqual((20, 10, 30), intermediate_vol.shape)
        final_permutation_matrix = models.PermutationMatrix.from_orientations(intermediate_orientation,
                                                                              final_orientation)
        final_shape = numpy.array(intermediate_vol.shape) @ final_permutation_matrix
        final_vol = intermediate_vol.reshape(final_shape)
        self.assertEqual((10, 30, 20), final_vol.shape)
        # shape=(C,R,S), orientation=(3,1,2)
        # new_orientation=(1,2,3) => new_shape=(R,S,C)
        orientation = 3, 1, 2
        new_orientation = 1, 2, 3  # double permutation
        permutation_matrix = models.PermutationMatrix.from_orientations(orientation, new_orientation)
        new_shape = numpy.array(vol.shape) @ permutation_matrix
        new_vol = vol.reshape(new_shape)
        self.assertEqual((20, 30, 10), new_vol.shape)

    def test_transpose(self):
        """"""
        vol = utils.get_vol(10, 20, 30)
        orientation = models.Orientation.from_integers((3, 1, 2))
        # 1, 2, 3
        self.assertEqual((0, 1, 2), orientation.from_integers((1, 2, 3)).to_transpose_integers())
        self.assertEqual((10, 20, 30),
                         vol.transpose(orientation.from_integers((1, 2, 3)).to_transpose_integers()).shape)
        # 1, 3, 2
        self.assertEqual((1, 0, 2), orientation.from_integers((1, 3, 2)).to_transpose_integers())
        self.assertEqual((20, 10, 30),
                         vol.transpose(orientation.from_integers((1, 3, 2)).to_transpose_integers()).shape)
        # 3, 1, 2
        self.assertEqual((1, 2, 0), orientation.from_integers((3, 1, 2)).to_transpose_integers())
        self.assertEqual((20, 30, 10),
                         vol.transpose(orientation.from_integers((3, 1, 2)).to_transpose_integers()).shape)
        # 3, 2, 1
        self.assertEqual((2, 1, 0), orientation.from_integers((3, 2, 1)).to_transpose_integers())
        self.assertEqual((30, 20, 10),
                         vol.transpose(orientation.from_integers((3, 2, 1)).to_transpose_integers()).shape)
        # 2, 1, 3
        self.assertEqual((0, 2, 1), orientation.from_integers((2, 1, 3)).to_transpose_integers())
        self.assertEqual((10, 30, 20),
                         vol.transpose(orientation.from_integers((2, 1, 3)).to_transpose_integers()).shape)
        # 2, 3, 1
        self.assertEqual((2, 0, 1), orientation.from_integers((2, 3, 1)).to_transpose_integers())
        self.assertEqual((30, 10, 20),
                         vol.transpose(orientation.from_integers((2, 3, 1)).to_transpose_integers()).shape)


class TestOrientation(unittest.TestCase):
    """
    `maptools` provides a simple API to
    """

    @classmethod
    def setUpClass(cls) -> None:

        # random name
        cls.random_name = TEST_DATA_DIR / f"file-{secrets.token_urlsafe(3)}.map"
        # random size
        cls.cols, cls.rows, cls.sections = random.sample(range(10, 30), k=3)
        with models.MapFile(cls.random_name, 'w') as mapfile:
            mapfile.data = utils.get_vol(cls.cols, cls.rows, cls.sections)
            # mapfile.data = voxel_size = 1.5

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
        self.assertIsInstance(numpy.asarray(orientation), numpy.ndarray)
        self.assertEqual((1, 3), orientation.shape)
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

    def test_from_integers(self):
        """"""
        orientation = models.Orientation.from_integers((2, 1, 3))
        self.assertIsInstance(orientation, models.Orientation)
        self.assertEqual('Y', orientation.cols)
        self.assertEqual('X', orientation.rows)
        self.assertEqual('Z', orientation.sections)

        with self.assertRaises(AssertionError):
            models.Orientation.from_integers((1, 1, 3))

    def test_from_string(self):
        """"""
        orientation = models.Orientation.from_string('XZY')
        self.assertIsInstance(orientation, models.Orientation)
        self.assertEqual('X', orientation.cols)
        self.assertEqual('Z', orientation.rows)
        self.assertEqual('Y', orientation.sections)

        with self.assertRaises(AssertionError):
            models.Orientation.from_string('XXY')

    def test_to_integers(self):
        """"""
        orientation = models.Orientation.from_string('XZY')
        self.assertEqual((1, 3, 2), orientation.to_integers())


class TestPermutationMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        # random name
        cls.random_name = TEST_DATA_DIR / f"file-{secrets.token_urlsafe(3)}.map"
        # random size
        cls.cols, cls.rows, cls.sections = random.sample(range(10, 30), k=3)
        with models.MapFile(cls.random_name, 'w') as mapfile:
            mapfile.data = utils.get_vol(cls.cols, cls.rows, cls.sections)
            # mapfile.data = voxel_size = 1.5

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
        with self.assertRaises(AssertionError):
            models.PermutationMatrix(
                numpy.fromstring('0 0 0 0 0 0 0 0 1', sep=' ').reshape(3, 3)
            )
        # non-binary
        with self.assertRaises(AssertionError):
            models.PermutationMatrix(
                numpy.fromstring('1 0 0 0 2 0 0 0 1', sep=' ').reshape(3, 3)
            )

    def test_permuation_matrix_from_orientations(self):
        """"""
        permutation_matrix = models.PermutationMatrix.from_orientations((1, 2, 3), (1, 3, 2))
        self.assertIsInstance(permutation_matrix, models.PermutationMatrix)
        self.assertEqual((3, 3), permutation_matrix.shape)
        self.assertEqual(3, permutation_matrix.rows)
        self.assertEqual(3, permutation_matrix.cols)
        self.assertEqual(int, permutation_matrix.dtype)
        self.assertTrue(numpy.array_equal(
            numpy.fromstring('1 0 0 0 0 1 0 1 0', sep=' ', dtype=int).reshape(3, 3),
            numpy.asarray(permutation_matrix)
        ))

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
        self.assertTrue(numpy.array_equal(numpy.eye(3, dtype=int), numpy.asarray(permutation_matrix1)))

    def test_swap_sequences(self):
        """Since the shape is ZYX, the swap axes should be 'inverted' i.e. if we are to swap X and Y instead of
        swapping 0 1 we swap 1 2"""
        swap_sequences = models.PermutationMatrix.from_orientations((1, 2, 3), (1, 2, 3)).swap_sequences
        self.assertEqual([], swap_sequences)
        swap_sequences = models.PermutationMatrix.from_orientations((1, 2, 3), (1, 3, 2)).swap_sequences
        self.assertEqual([(0, 1)], swap_sequences)
        swap_sequences = models.PermutationMatrix.from_orientations((1, 2, 3), (2, 1, 3)).swap_sequences
        self.assertEqual([(1, 2)], swap_sequences)
        swap_sequences = models.PermutationMatrix.from_orientations((1, 2, 3), (2, 3, 1)).swap_sequences
        self.assertEqual([(1, 2), (0, 2)], swap_sequences)
        swap_sequences = models.PermutationMatrix.from_orientations((1, 2, 3), (3, 2, 1)).swap_sequences
        self.assertEqual([(0, 2)], swap_sequences)
        swap_sequences = models.PermutationMatrix.from_orientations((1, 2, 3), (3, 1, 2)).swap_sequences
        self.assertEqual([(0, 2), (0, 1)], swap_sequences)


class TestMapFile(unittest.TestCase):
    def setUp(self) -> None:
        self.test_fn = TEST_DATA_DIR / f"test-{secrets.token_urlsafe(3)}.map"
        self.test_fn2 = TEST_DATA_DIR / f"test-{secrets.token_urlsafe(3)}.map"

    def tearDown(self) -> None:
        try:
            os.remove(self.test_fn)
        except FileNotFoundError:
            print(f"test.map already deleted or not used in this test...", file=sys.stderr)
        try:
            os.remove(self.test_fn2)
        except FileNotFoundError:
            pass

    def test_create_empty(self):
        """"""
        with self.assertRaises(ValueError):
            with models.MapFile(self.test_fn, file_mode='w') as mapfile:
                # everything is None or the default value
                self.assertIsNone(mapfile.nc)
                self.assertIsNone(mapfile.nr)
                self.assertIsNone(mapfile.ns)
                self.assertEqual(2, mapfile.mode)
                self.assertEqual((0, 0, 0), mapfile.start)
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
        # create
        with models.MapFile(self.test_fn, file_mode='w') as mapfile:
            # set data
            mapfile.data = numpy.random.rand(10, 20, 30)  # sections, rows, cols
            # now the following should be automatically inferred from the data
            self.assertEqual(30, mapfile.nc)
            self.assertEqual(20, mapfile.nr)
            self.assertEqual(10, mapfile.ns)
            self.assertEqual(2, mapfile.mode)
            self.assertEqual((0, 0, 0), mapfile.start)
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
            self.assertEqual(0.0, mapfile.t1)
            self.assertEqual(0.0, mapfile.t2)
            self.assertEqual(0.0, mapfile.t3)
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
        # mapfile is closed
        # read
        with models.MapFile(self.test_fn) as mapfile2:
            self.assertEqual(30, mapfile2.nc)
            self.assertEqual(20, mapfile2.nr)
            self.assertEqual(10, mapfile2.ns)
            self.assertEqual(2, mapfile2.mode)
            self.assertEqual((0, 0, 0), mapfile2.start)
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
            self.assertEqual(0.0, mapfile2.t1)
            self.assertEqual(0.0, mapfile2.t2)
            self.assertEqual(0.0, mapfile2.t3)
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

    def test_create_standard_then_modify_to_nonstandard(self):
        """"""
        # create
        with models.MapFile(self.test_fn, start=(-2, 3, -4), file_mode='w') as mapfile:
            # set data given default orientation
            mapfile.data = numpy.random.rand(10, 20, 30)  # sections, rows, cols
            print(f"{mapfile.data.shape = }")
            print(f"{mapfile.orientation = }")
            print(f"{mapfile.cols, mapfile.rows, mapfile.sections}")
            self.assertEqual(30, mapfile.nc)
            self.assertEqual(20, mapfile.nr)
            self.assertEqual(10, mapfile.ns)
            self.assertEqual(30, mapfile.nx)
            self.assertEqual(20, mapfile.ny)
            self.assertEqual(10, mapfile.nz)
            # change orientation to nonstandard YXZ
            mapfile.orientation = models.Orientation(cols='Y', rows='X', sections='Z')
            print(f"{mapfile.data.shape}")
            print(f"{mapfile.orientation.to_integers() = }")
            print(f"{mapfile.cols, mapfile.rows, mapfile.sections}")
            # now the following should be automatically inferred from the data
            self.assertEqual(20, mapfile.nc)
            self.assertEqual(30, mapfile.nr)
            self.assertEqual(10, mapfile.ns)
            self.assertEqual(20, mapfile.nx)
            self.assertEqual(30, mapfile.ny)
            self.assertEqual(10, mapfile.nz)
            # change orientation to nonstandard YZX
            # S=10, R=20, C=30
            # C=30, R=20, S=10
            # X Y Z
            # Y Z X
            # C'=20, R'=10, S'=30
            mapfile.orientation = models.Orientation(cols='Y', rows='Z', sections='X')
            self.assertEqual(20, mapfile.nc)
            self.assertEqual(10, mapfile.nr)
            self.assertEqual(30, mapfile.ns)
            # change orientation to nonstandard YZX
            mapfile.orientation = models.Orientation(cols='Z', rows='Y', sections='X')
            # Z Y X
            self.assertEqual(10, mapfile.nc)
            self.assertEqual(20, mapfile.nr)
            self.assertEqual(30, mapfile.ns)
            # let's do all the other orientation permutations
            # Z X Y
            mapfile.orientation = models.Orientation(cols='Z', rows='X', sections='Y')
            self.assertEqual(10, mapfile.nc)
            self.assertEqual(30, mapfile.nr)
            self.assertEqual(20, mapfile.ns)
            # X Z Y
            mapfile.orientation = models.Orientation(cols='X', rows='Z', sections='Y')
            self.assertEqual(30, mapfile.nc)
            self.assertEqual(10, mapfile.nr)
            self.assertEqual(20, mapfile.ns)

        # mapfile is closed
        # read
        with models.MapFile(self.test_fn) as mapfile2:
            self.assertEqual(30, mapfile2.nc)
            self.assertEqual(10, mapfile2.nr)
            self.assertEqual(20, mapfile2.ns)
            self.assertEqual(2, mapfile2.mode)
            self.assertEqual((-2, -4, 3), mapfile2.start)
            self.assertEqual(30, mapfile2.nx)
            self.assertEqual(10, mapfile2.ny)
            self.assertEqual(20, mapfile2.nz)
            self.assertEqual(30.0, mapfile2.x_length)
            self.assertEqual(10.0, mapfile2.y_length)
            self.assertEqual(20.0, mapfile2.z_length)
            self.assertEqual(90.0, mapfile2.alpha)
            self.assertEqual(90.0, mapfile2.beta)
            self.assertEqual(90.0, mapfile2.gamma)
            self.assertEqual(1, mapfile2.mapc)
            self.assertEqual(3, mapfile2.mapr)
            self.assertEqual(2, mapfile2.maps)
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
            self.assertEqual(10, mapfile2.rows)
            self.assertEqual(20, mapfile2.sections)
            self.assertEqual("Orientation(cols='X', rows='Z', sections='Y')", str(mapfile2.orientation))
            self.assertEqual((1.0, 1.0, 1.0), mapfile2.voxel_size)
            self.assertEqual('float32', mapfile2.data.dtype.name)

    def test_read_and_modify(self):
        """"""
        with models.MapFile(self.test_fn, file_mode='w') as mapfile:
            # set data
            mapfile.data = numpy.random.rand(10, 20, 30)  # sections, rows, cols
            self.assertEqual(30, mapfile.nc)
            self.assertEqual(20, mapfile.nr)
            self.assertEqual(10, mapfile.ns)
            self.assertEqual(2, mapfile.mode)
            self.assertEqual((0, 0, 0), mapfile.start)
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
        with models.MapFile(self.test_fn, file_mode='r+') as mapfile2:
            self.assertEqual(30, mapfile2.nc)
            self.assertEqual(20, mapfile2.nr)
            self.assertEqual(10, mapfile2.ns)
            self.assertEqual(1, mapfile2.mapc)
            self.assertEqual(2, mapfile2.mapr)
            self.assertEqual(3, mapfile2.maps)
            # change the orientation
            mapfile2.orientation = models.Orientation(cols='Z', rows='Y', sections='X')
            self.assertEqual(10, mapfile2.nc)
            self.assertEqual(20, mapfile2.nr)
            self.assertEqual(30, mapfile2.ns)
            self.assertEqual(3, mapfile2.mapc)
            self.assertEqual(2, mapfile2.mapr)
            self.assertEqual(1, mapfile2.maps)
            print(mapfile2)

    def test_voxel_size(self):
        """"""
        # we start off with a standard orientation but anisotropic
        # then we change to nonstandard
        # examine the voxel size
        with models.MapFile(self.test_fn, file_mode='w') as mapfile:
            mapfile.data = numpy.random.rand(3, 4, 5)
            # x=1.7, y=2.4, z=9.3
            # X=8.5, Y=9.6, Z=27.9
            mapfile.voxel_size = 1.7, 2.4, 9.3
            self.assertEqual(8.5, mapfile.x_length)
            self.assertEqual(9.6, mapfile.y_length)
            self.assertAlmostEqual(27.9, mapfile.z_length)
            # what if we change the orientation to ZYX
            mapfile.orientation = models.Orientation(cols='Z', rows='Y', sections='X')
            # the voxel sizes also get permuted
            self.assertEqual((9.3, 2.4, 1.7), mapfile.voxel_size)
            # but the lengths should change because we now have a different number of voxels on the same length
            self.assertAlmostEqual(27.9, mapfile.x_length)
            self.assertEqual(9.6, mapfile.y_length)
            self.assertAlmostEqual(8.5, mapfile.z_length)
            voxel_size_orig = mapfile.voxel_size

        # read
        with models.MapFile(self.test_fn, 'r+') as mapfile2:
            self.assertAlmostEqual(voxel_size_orig[0], mapfile2.voxel_size[0], places=6)
            self.assertAlmostEqual(voxel_size_orig[1], mapfile2.voxel_size[1], places=6)
            self.assertAlmostEqual(voxel_size_orig[2], mapfile2.voxel_size[2], places=6)

    def test_create_anisotropic_voxels(self):
        # create with anisotropic voxel sizes
        with models.MapFile(self.test_fn, 'w', voxel_size=(3.7, 2.6, 1.5)) as mapfile:
            mapfile.data = numpy.random.rand(12, 22, 17)
            self.assertEqual((3.7, 2.6, 1.5), mapfile.voxel_size)

    def test_create_with_nonstardard_and_anisotropic(self):
        """"""
        with models.MapFile(
                self.test_fn, 'w',
                orientation=models.Orientation(cols='Y', rows='Z', sections='X'),
                voxel_size=(3.7, 2.6, 1.5)
        ) as mapfile:
            mapfile.data = numpy.random.rand(12, 22, 17)
            self.assertEqual((2, 3, 1), mapfile.orientation.to_integers())
            self.assertEqual((2.6, 1.5, 3.7), mapfile.voxel_size)

    def test_create_with_map_mode(self):
        """"""
        with models.MapFile(
                self.test_fn,
                file_mode='w',
                map_mode=0,
                voxel_size=(8.7, 9.2, 1.2),
                orientation=models.Orientation.from_integers((2, 3, 1))
        ) as mapfile:
            mapfile.data = numpy.random.randint(0, 1, size=(11, 9, 16))
            self.assertEqual(0, mapfile.mode)

    def test_change_map_mode_int(self):
        """"""
        # int to int valid: 0, 1, 3, 6
        # float to float valid: 2, 12
        # complex to complex: 3, 4
        # int to/from float invalid
        with models.MapFile(self.test_fn, 'w', map_mode=0) as mapfile:
            mapfile.data = numpy.random.randint(0, 1, (5, 5, 5))
            self.assertEqual(1, mapfile.data.itemsize)
            mapfile.mode = 1
            self.assertEqual(2, mapfile.data.itemsize)
            mapfile.mode = 3
            self.assertEqual(4, mapfile.data.itemsize)
            # change back
            mapfile.mode = 0
            self.assertEqual(1, mapfile.data.itemsize)
            mapfile.mode = 1
            self.assertEqual(2, mapfile.data.itemsize)

        with self.assertWarns(UserWarning):
            with models.MapFile(self.test_fn, 'r+') as mapfile2:
                mapfile2.mode = 2
                self.assertEqual(4, mapfile2.data.itemsize)

        with self.assertWarns(UserWarning):
            with models.MapFile(self.test_fn, 'r+') as mapf3:
                mapf3.mode = 1
                self.assertEqual(2, mapf3.data.itemsize)

    def test_change_map_mode_float(self):
        """"""
        with models.MapFile(self.test_fn, 'w') as mapfile:
            mapfile.data = numpy.random.rand(8, 8, 8)
            self.assertEqual(4, mapfile.data.itemsize)
            mapfile.mode = 12
            self.assertEqual(2, mapfile.data.itemsize)

        with self.assertWarns(UserWarning):
            with models.MapFile(self.test_fn, 'r+') as mapfile2:
                mapfile2.mode = 0
                self.assertEqual(1, mapfile2.data.itemsize)

    def test_change_map_mode_valid_data(self):
        """"""
        # start with map_mode = 2
        with models.MapFile(
                self.test_fn, 'w',
                start=(-5, -6, -7),
                voxel_size=(1.9, 2.6, 4.2),
                colour=True,
                orientation=models.Orientation.from_string('YZX')
        ) as mapfile:
            mapfile.data = numpy.random.rand(10, 10, 10)
            print(mapfile)
            # change to map_mode = 0
            mapfile.mode = 0
            print(mapfile)
        # change to map_mode = 1
        with models.MapFile(self.test_fn, 'r+', colour=True) as mapfile2:
            mapfile2.mode = 1
            print(mapfile2)
        # change to map_mode = 2 again, but obviously we've lost the data
        with models.MapFile(self.test_fn, 'r+', colour=True) as mapfile3:
            mapfile3.mode = 2
            print(mapfile3)

    def test_start(self):
        """"""
        with models.MapFile(self.test_fn, 'w', start=(3, 9, -11)) as mapfile:
            mapfile.data = numpy.random.rand(3, 5, 2)
            print(mapfile)
            self.assertEqual((3, 9, -11), mapfile.start)
            # change start
            mapfile.start = (58, 3, 4)
            print(mapfile)
            self.assertEqual((58, 3, 4), mapfile.start)
        # read
        with models.MapFile(self.test_fn) as mapfile2:
            self.assertEqual((58, 3, 4), mapfile2.start)

    def test_handle_labels(self):
        """"""
        with models.MapFile(TEST_DATA_DIR / 'emd_5625.map', colour=True) as mapfile:
            self.assertTrue(len(mapfile.labels) == 1)
            self.assertEqual("::::EMDATABANK.org::::EMD-5625::::", mapfile.get_label(0))
            # add a new label
            mapfile.add_label("there is a new dog in town")
            self.assertEqual(2, len(mapfile.labels))
            self.assertEqual(2, mapfile.nlabl)
            # insert a label
            mapfile.insert_label("are you coming for lunch?")
            self.assertEqual(3, len(mapfile.labels))
            mapfile.add_label("extra 1")
            mapfile.add_label("extra 2")
            mapfile.add_label("extra 3")
            mapfile.add_label("extra 4")
            mapfile.add_label("extra 5")
            mapfile.add_label("extra 6")
            mapfile.add_label("extra 7")
            print(mapfile)
            with self.assertWarns(UserWarning):
                mapfile.add_label("extra 8")
            self.assertEqual(10, len(mapfile.labels))
            mapfile.del_label()
            self.assertEqual(9, len(mapfile.labels))
            mapfile.insert_label("this is where it ends", 4)
            self.assertEqual(10, len(mapfile.labels))
            with self.assertWarns(UserWarning):
                mapfile.insert_label("another way to end", 7)

            # invalid values
            with self.assertWarns(UserWarning):
                mapfile.get_label(-11)
            with self.assertWarns(UserWarning):
                mapfile.get_label(10)
            with self.assertWarns(UserWarning):
                mapfile.add_label(
                    "I'm very sure that this is much longer than 80 characters because at some point I will cross "
                    "the 80 char limit line, right?")
            with self.assertWarns(UserWarning):
                mapfile.insert_label("invalid label", 20)

            # clear all labels
            mapfile.clear_labels()
            self.assertEqual(0, len(mapfile.labels))
            print(mapfile)
            self.assertEqual(0, mapfile.nlabl)

            # unicode
            with self.assertWarns(UserWarning):
                mapfile.add_label("ニシコクマルガラスは私のクォーツのスフィンクスが大好きです")

            print(mapfile)

    def test_create_with_labels(self):
        """"""
        with models.MapFile(self.test_fn, 'w', colour=True) as mapfile:
            mapfile.data = numpy.random.rand(5, 6, 7)
            mapfile.add_label("a new label")
            self.assertEqual("a new label", mapfile.get_label(0))
            mapfile.add_label("ニシコクマルガラスは私のクォーツのスフィンクスが大好きです")

        # read
        with models.MapFile(self.test_fn, colour=True) as mapfile2:
            self.assertEqual("a new label", mapfile2.get_label(0))
            self.assertEqual("ニシコクマルガラスは私のクォーツのスフィンクスが大好", mapfile2.get_label(1))
            print(mapfile2)

    def test_calculate_rms(self):
        """Test that:
        - we do not overflow
        - we leave the type as before
        """
        # int8
        fn = f"test_{secrets.token_urlsafe(4)}.map"
        with models.MapFile(fn, file_mode='w', map_mode=0, colour=True) as mapfile:
            mapfile.data = numpy.random.randint(-128, 127, size=(10, 10, 10))
            print(mapfile)
        os.remove(fn)
        # uint8
        fn = f"test_{secrets.token_urlsafe(4)}.map"
        with models.MapFile(fn, file_mode='w', map_mode=0, colour=True) as mapfile:
            mapfile.data = numpy.random.randint(128, 255, size=(10, 10, 10))
            print(mapfile)
        os.remove(fn)
        # int16
        fn = f"test_{secrets.token_urlsafe(4)}.map"
        with models.MapFile(fn, file_mode='w', map_mode=1, colour=True) as mapfile:
            mapfile.data = numpy.random.randint(-(2 ** 15), 2 ** 15 - 1, size=(10, 10, 10))
            print(mapfile)
        os.remove(fn)
        # uint16
        fn = f"test_{secrets.token_urlsafe(4)}.map"
        with models.MapFile(fn, file_mode='w', map_mode=1, colour=True) as mapfile:
            mapfile.data = numpy.random.randint(2 ** 15, 2 ** 16 - 1, size=(10, 10, 10))
            print(mapfile)
        os.remove(fn)
        # float32
        fn = f"test_{secrets.token_urlsafe(4)}.map"
        with models.MapFile(fn, file_mode='w', map_mode=2, colour=True) as mapfile:
            mapfile.data = numpy.random.rand(10, 10, 10)
            print(mapfile)
        os.remove(fn)

    def test_copy(self):
        """"""
        mapfile1 = models.MapFile(self.test_fn, 'w')
        mapfile1.data = numpy.random.rand(2, 3, 4)
        # second one
        mapfile2 = models.MapFile(self.test_fn2, 'w')
        mapfile2.copy(mapfile1)
        self.assertEqual(mapfile1, mapfile2)
        mapfile1.close()
        mapfile2.close()


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        # random name
        cls.random_name = TEST_DATA_DIR / f"file-{secrets.token_urlsafe(3)}.map"
        # random size
        cls.cols, cls.rows, cls.sections = random.sample(range(10, 30), k=3)
        with models.MapFile(cls.random_name, 'w') as mapfile:
            mapfile.data = utils.get_vol(cls.cols, cls.rows, cls.sections)
            # mapfile.data = voxel_size = 1.5

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
        with models.MapFile(self.random_name) as mapfile:
            self.assertIsInstance(mapfile.orientation, models.Orientation)
            self.assertEqual('X', mapfile.orientation.cols)
            self.assertEqual('Y', mapfile.orientation.rows)
            self.assertEqual('Z', mapfile.orientation.sections)
