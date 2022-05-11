import os
import numpy
from mapfile import models


def view(args):
    """"""
    with models.MapFile(args.file, colour=args.colour) as mapf:
        print(mapf)
    return os.EX_OK


def edit(args):
    """"""
    with models.MapFile(args.file, args.file_mode) as mapf:
        mapf.orientation = models.Orientation.from_string(args.orientation)
        mapf.voxel_size = args.voxel_sizes
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
