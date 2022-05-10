import os
import random
import secrets
import unittest

import mrcfile
import numpy

from mapfix import models


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


class TestMapFix(unittest.TestCase):
    """
    `mapfix` provides a simple API to
    """

    @classmethod
    def setUpClass(cls) -> None:

        # random name
        cls.random_name = f"file-{secrets.token_urlsafe(3)}.mrc"
        # random size
        cls.cols, cls.rows, cls.sections = random.sample(range(10, 30), k=3)
        with mrcfile.new(cls.random_name) as mrc:
            mrc.set_data(get_vol(cls.cols, cls.rows, cls.sections))
            mrc.voxel_size = 1.5

    @classmethod
    def tearDownClass(cls) -> None:
        """cleanup"""
        try:
            os.remove(cls.random_name)
        except FileNotFoundError:
            print(f"file {cls.random_name} already deleted!")

    def test_mrcfile(self):
        """Test behaviour with mrcfile"""
        with mrcfile.new('file.mrc', overwrite=True) as mrc:
            mrc.set_data(numpy.zeros(shape=(10, 20, 30), dtype=numpy.float32))
            # voxel size (isotropic)
            # c=30,
            mrc.voxel_size = 3.0, 2.0, 1.0
            mrc.update_header_from_data()
            print(f"orig shape = {mrc.data.shape}")
            print(mrc.print_header())
            print(f"{mrc.header.cella = }")
            orientation = tuple(map(int, (mrc.header.mapc, mrc.header.mapr, mrc.header.maps)))
            print(orientation)
            print(type(int(orientation[0])))
            new_orientation = 3, 2, 1
            permutation_matrix = models.get_permutation_matrix(orientation, new_orientation)
            print(permutation_matrix)
            new_shape = numpy.dot(mrc.data.shape, permutation_matrix)
            print(f"{new_shape = }")
            # reset the data
            mrc.set_data(mrc.data.reshape(new_shape))
            print(f"new shape = {mrc.data.shape}")
            # we have to also change the values of mapc, mapr, maps
            mrc.header.mapc, mrc.header.macr, mrc.header.maps = new_orientation
            # and mx, my, mz
            mrc.header.mx, mrc.header.my, mrc.header.mz = new_shape
            mrc.voxel_size = 1.0
            # print(mrc.print_header())
            mrc.update_header_from_data()
            print(f"{mrc.voxel_size = }")

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

    def test_mapfile(self):
        """"""
        with models.MapFile(self.random_name) as mapfile:
            print(mapfile)
            print(numpy.asarray(mapfile).shape)
            print(mapfile.orientation)
            self.assertEqual('X', mapfile.orientation.cols)
            self.assertEqual('Y', mapfile.orientation.rows)
            self.assertEqual('Z', mapfile.orientation.sections)

    def test_mapfile_reorient(self):
        """"""
        with models.MapFile(self.random_name) as mapfile:
            self.assertEqual("Orientation(cols='X', rows='Y', sections='Z')", str(mapfile.orientation))

            with open('new-file.mrc', 'wb') as newmap:
                mapfile.write(newmap)

    def test_get_orientation(self):
        """"""
        # by default, orientation is XYZ
        with mrcfile.open(self.random_name) as mrc:
            orientation = models.get_orientation(mrc)
            self.assertIsInstance(orientation, models.Orientation)
            self.assertEqual('X', orientation.cols)
            self.assertEqual('Y', orientation.rows)
            self.assertEqual('Z', orientation.sections)

    def test_set_orientation(self):
        """"""
        # read an mrc with xyz orientation
        print(self.cols, self.rows, self.sections)
        with mrcfile.open(self.random_name, 'r+') as mrc:
            mrc.print_header()
            # check the orientation is xyz
            orientation = models.get_orientation(mrc)
            self.assertEqual("Orientation(cols='X', rows='Y', sections='Z')", str(orientation))
            # check voxel size
            print(f"{mrc.voxel_size = }")
            # check cella
            print(f"{mrc.header.cella = }")
            # check dimensions
            self.assertEqual((self.cols, self.rows, self.sections), mrc.data.shape)
            # set it to zyx
            models.set_orientation(mrc, models.Orientation(cols='Z', rows='Y', sections='X'))
        # write it out
        # read it afresh
        with mrcfile.open(self.random_name) as mrc2:
            mrc2.print_header()
            # confirm that it behaves as it should
            # check the orientation is xyz
            orientation = models.get_orientation(mrc2)
            self.assertEqual("Orientation(cols='Z', rows='Y', sections='X')", str(orientation))
            # check that dimensions are changed
            self.assertEqual((self.sections, self.rows, self.cols), mrc2.data.shape)
            # check that the voxel sizes are unchanged

            # check that cella is changed


class TestMapFile(unittest.TestCase):
    def test_create(self):
        """"""
        map
