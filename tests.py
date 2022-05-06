import random
import unittest

import numpy


def get_vol(cols, rows, sects, dtype=numpy.uint8):
    return numpy.empty(shape=(cols, rows, sects), dtype=dtype)


def get_permutation_matrix(orientation, new_orientation):
    """Compute the permutation matrix required to convert the sequence <orientation> to <new_orientation>

    A permutation matrix is a square matrix.
    The values of its elements are in {0, 1}.
    """
    # assert that the values in orientation are unique
    try:
        assert len(set(orientation)) == len(orientation)
    except AssertionError:
        raise ValueError(f"repeated elements in {orientation}")
    # assert that the values in new_orientation are unique
    try:
        assert len(set(new_orientation)) == len(new_orientation)
    except AssertionError:
        raise ValueError(f"repeated elements in {new_orientation}")
    # assert that the values in orientation are exactly those in new_orientation
    try:
        assert len(set(orientation)) == len(set(new_orientation))
    except AssertionError:
        raise ValueError(f"values differ: {orientation} vs. {new_orientation}")
    permutation_matrix = numpy.zeros((len(orientation), len(orientation)), dtype=int)
    for index, value in enumerate(orientation):
        permutation_matrix[index, new_orientation.index(value)] = 1
    return permutation_matrix


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
                get_permutation_matrix((1, 2, 3), (1, 3, 2))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.p02,
                get_permutation_matrix((1, 2, 3), (3, 2, 1))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                self.p01,
                get_permutation_matrix((1, 2, 3), (2, 1, 3))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                numpy.dot(self.p01, self.p02),
                get_permutation_matrix((1, 2, 3), (3, 1, 2))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                numpy.dot(self.p02, self.p01),
                get_permutation_matrix((1, 2, 3), (2, 3, 1))
            )
        )
        self.assertTrue(
            numpy.array_equal(
                numpy.dot(self.p01, self.p01),
                get_permutation_matrix((1, 2, 3), (1, 2, 3))
            )
        )
        # sanity checks
        with self.assertRaises(ValueError):
            get_permutation_matrix((1, 1, 1), (1, 2, 3))
        with self.assertRaises(ValueError):
            get_permutation_matrix((1, 1, 1), (1, 2, 2))
        with self.assertRaises(ValueError):
            get_permutation_matrix((1, 2, 3), (2, 3, 4))

    def test_permute_shape(self):
        """Test if permuting the shape is the same as performing a swap of axes"""
        # shape=(C,R,S), orientation=(1,2,3)
        # new_orientation=(3,2,1) => new_shape=(S,R,C)
        # permutation_matrix = get_permutation_matrix(orientation, new_orientation)
        # is it true that shape * permutation_matrix = new_shape?
        vol = get_vol(10, 20, 30)
        orientation = 1, 2, 3
        new_orientation = 3, 2, 1
        permutation_matrix = get_permutation_matrix(orientation, new_orientation)
        new_shape = numpy.array(vol.shape).dot(permutation_matrix)
        # print(new_shape)
        new_vol = vol.reshape(new_shape)
        self.assertEqual((30, 20, 10), new_vol.shape)
        # shape=(C,R,S), orientation=(2,1,3)
        # new_orientation=(2,3,1) => new_shape=(C,S,R)?
        orientation = 2, 1, 3
        new_orientation = 2, 3, 1
        permutation_matrix = get_permutation_matrix(orientation, new_orientation)
        new_shape = numpy.array(vol.shape).dot(permutation_matrix)
        new_vol = vol.reshape(new_shape)
        self.assertEqual((10, 30, 20), new_vol.shape)
        # this should be the same as (2,1,3)->(1,2,3)->(2,3,1)
        # the sequence of permutations doesn't matter
        orientation = 2, 1, 3
        intermediate_orientation = 1, 2, 3
        final_orientation = 2, 3, 1
        intermediate_permutation_matrix = get_permutation_matrix(orientation, intermediate_orientation)
        intermediate_shape = numpy.dot(
            numpy.array(vol.shape),
            intermediate_permutation_matrix
        )
        intermediate_vol = vol.reshape(intermediate_shape)
        self.assertEqual((20, 10, 30), intermediate_vol.shape)
        final_permutation_matrix = get_permutation_matrix(intermediate_orientation, final_orientation)
        final_shape = numpy.array(intermediate_vol.shape).dot(final_permutation_matrix)
        final_vol = intermediate_vol.reshape(final_shape)
        self.assertEqual((10, 30, 20), final_vol.shape)
        # shape=(C,R,S), orientation=(3,1,2)
        # new_orientation=(1,2,3) => new_shape=(R,S,C)
        orientation = 3, 1, 2
        new_orientation = 1, 2, 3  # double permutation
        permutation_matrix = get_permutation_matrix(orientation, new_orientation)
        new_shape = numpy.dot(numpy.array(vol.shape), permutation_matrix)
        new_vol = vol.reshape(new_shape)
        self.assertEqual((20, 30, 10), new_vol.shape)
