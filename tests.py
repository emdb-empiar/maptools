"""
mapfix

The best way to apply the ideas in this package will be into `mrcfile` Python package.

The problem is that the mapc,mapr,maps fields can take one of six possible values.
The default is 1, 2, 3 meaning cols=X, rows=Y, sections=Z.
The goal is to make it straightforward to the user how these may be modified.

For example, on reading in a map the user should be able to query the volume orientation.

map.orientation # (cols='X', rows='Y', sections='Z')

The user should then be able to re-orient the volume

map.change_orientation(cols='Z', rows='Y', sections='X')
map.orientation # (cols='Z', rows='Y', sections='X')

Also, on reading, the user can specify a new orientation

mrcfile.open(*args, **kwargs, orientation=(cols='X', rows='Y', sections='Z'))
mrcfile.new(*args, **kwargs, orientation=(cols='X', rows='Y', sections='Z'))

By default, orientation is (cols='X', rows='Y', sections='Z')

It may be necessary to come up with another name for this attribute e.g.
- axes
- alignment
- axis_alignment

It should also be possible to come up with derivative attributes based on the orientation.

map.space_handedness # 'right' | 'left'

"""
import random
import unittest

import numpy


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


def get_swap_axes(to_orientation, from_orientation=(1, 2, 3)):
    """
    This is where the complexity comes from.

    Orientation is a triple of integers: 1, 2, 3, where 1=X, 2=Y, 3=Z. Therefore, 1, 2, 3 is X, Y, Z orientation which is standard.

    Some files have non-standard orientations. Our goal is to transform the data so that it is presented in the standard orientation.
    This means that we have to decide on the transformation required to modify data with a particular orientation to the
    standard orientation. In general, we would like to be able to move from any orientation to any other.

    The transformation will be accomplished using numpy.swapaxes(arra, <index1>, <index2>)

    For 3D data the values of <index?> are exclusively one of: 0, 1, 2

    E.g. numpy.swapaxes(arr, 0, 2) means swap the first and third dimension etc.

    We can outline the set of possible orientations. These are permutations of (1, 2, 3). There are six (6) such permutations.

    For any permutation we can swap at one of two pairs of positions: (0, 1), (0, 2) and (1, 2).

    This means that any orientation can be converted to three other orientations by only permuting two positions.
    The two remaining orientations require at least two permutations.
    The identity permutation transforms an orientation into itself.

    We can graphically describe the set of permutations as a permutohedron. (see https://en.wikipedia.org/wiki/Permutohedron)

    Permutations may be expressed using a permutation matrix (see https://en.wikipedia.org/wiki/Permutation_matrix).
    """
    pass


class Test(unittest.TestCase):
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
