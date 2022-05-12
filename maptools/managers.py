import os

import numpy

from maptools import models


def view(args):
    """"""
    with models.MapFile(args.file, colour=args.colour) as mapfile:
        print(mapfile)
    return os.EX_OK


def edit(args):
    """"""
    # todo: handle output file
    """
    if args.output is not None:
        with models.mapfileile(args.output, 'w') as mapout:
            with models.mapfileile(args.file) as mapin:
                mapout.data = mapin.data
    else:
        with models.mapfileile(args.file, file_mode=args.file_mode) as mapfile:
            if args.orientation is not None:
                mapfile.orientation = models.Orientation.from_string(args.orientation)
            if args.voxel_sizes is not None:
                mapfile.voxel_size = args.voxel_sizes
            if args.map_mode is not None:
                mapfile.mode = args.map_mode
    """
    with models.MapFile(
            args.file,
            args.file_mode,
            start=args.start,
            colour=args.colour,
            verbose=args.verbose
    ) as mapfile:
        if args.orientation is not None:
            mapfile.orientation = models.Orientation.from_string(args.orientation)
        if args.voxel_sizes is not None:
            mapfile.voxel_size = args.voxel_sizes
        if args.map_mode is not None:
            mapfile.mode = args.map_mode
        print(mapfile)
    return os.EX_OK


def create(args):
    """"""
    with models.MapFile(
            args.file,
            args.file_mode,
            orientation=(models.Orientation.from_string(args.orientation)),
            map_mode=args.map_mode,
            voxel_size=args.voxel_sizes
    ) as mapfile:
        if args.ones:
            mapfile.data = numpy.ones(args.size)
        elif args.zeros:
            mapfile.data = numpy.zeros(args.size)
        elif args.empty:
            mapfile.data = numpy.empty(args.size)
        elif args.random_integers:
            mapfile.data = numpy.random.randit(args.min, args.max, args.size)
        elif args.random_floats:
            mapfile.data = numpy.random.rand(args.size)
        print(mapfile)
    return os.EX_OK
