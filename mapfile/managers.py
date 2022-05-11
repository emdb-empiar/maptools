import os
import numpy
from mapfile import models


def view(args):
    """"""
    with models.MapFile(args.file, colour=True) as mapf:
        print(mapf)
    return os.EX_OK


def edit(args):
    """"""
    # todo: handle output file
    """
    if args.output is not None:
        with models.MapFile(args.output, 'w') as mapout:
            with models.MapFile(args.file) as mapin:
                mapout.data = mapin.data
    else:
        with models.MapFile(args.file, file_mode=args.file_mode) as mapf:
            if args.orientation is not None:
                mapf.orientation = models.Orientation.from_string(args.orientation)
            if args.voxel_sizes is not None:
                mapf.voxel_size = args.voxel_sizes
            if args.map_mode is not None:
                mapf.mode = args.map_mode
    """
    with models.MapFile(args.file, args.file_mode, start=args.start) as mapf:
        if args.orientation is not None:
            mapf.orientation = models.Orientation.from_string(args.orientation)
        if args.voxel_sizes is not None:
            mapf.voxel_size = args.voxel_sizes
        if args.map_mode is not None:
            mapf.mode = args.map_mode
        print(mapf)
    return os.EX_OK


def create(args):
    """"""
    with models.MapFile(
            args.file,
            args.file_mode,
            orientation=(models.Orientation.from_string(args.orientation)),
            map_mode=args.map_mode,
            voxel_size=args.voxel_sizes
    ) as mapf:
        if args.ones:
            mapf.data = numpy.ones(args.size)
        elif args.zeros:
            mapf.data = numpy.zeros(args.size)
        elif args.empty:
            mapf.data = numpy.empty(args.size)
        elif args.random_integers:
            mapf.data = numpy.random.randit(args.min, args.max, args.size)
        elif args.random_floats:
            mapf.data = numpy.random.rand(args.size)
        print(mapf)
    return os.EX_OK
