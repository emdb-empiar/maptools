import numpy


def grid_resample(array: numpy.ndarray, factor: int):
    """resample the grid provided by array according to the factor specified"""
    resampled_array = array[::factor, ::factor, ::factor]
    return resampled_array
