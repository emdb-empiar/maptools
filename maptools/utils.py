import numpy

from maptools.models import Orientation, _axes, _raxes


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


def get_orientation(mapfile):
    """
    Determine the orientation of an MRC file

    :param mapfile: an MRC file
    :return: a tuple
    """
    mapc, mapr, maps = mapfile.mapc, mapfile.mapr, mapfile.maps
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
