import os

import numpy

from maptools import models
import sys
from styled import Styled


def view(args):
    """"""
    with models.MapFile(
            args.file,
            orientation=args.orientation,
            colour=args.colour,
            verbose=args.verbose
    ) as mapfile:
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
            file_mode=args.file_mode,
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
        mapfile.add_label(args.label)
        print(mapfile)
    return os.EX_OK


def create(args):
    """"""
    with models.MapFile(
            args.file,
            file_mode='w',
            start=args.start,
            orientation=models.Orientation.from_string(args.orientation),
            map_mode=args.map_mode,
            voxel_size=args.voxel_sizes,
            colour=args.colour,
            verbose=args.verbose
    ) as mapfile:
        if args.verbose:
            print(Styled(f"[[ '[info] creating map with {args.voxel_values}'|fg-orange_3 ]]"), file=sys.stderr)
            print(Styled(f"[[ '[info] creating map mode {args.map_mode}'|fg-orange_3 ]]"), file=sys.stderr)
        if args.voxel_values == 'zeros':
            mapfile.data = numpy.zeros(args.size[::-1])
        elif args.voxel_values == 'ones':
            mapfile.data = numpy.ones(args.size[::-1])
        elif args.voxel_values == 'empty':
            mapfile.data = numpy.empty(args.size[::-1])
        elif args.voxel_values == 'randint':
            mapfile.data = numpy.random.randint(args.min, args.max, args.size[::-1])
        elif args.voxel_values == 'random':
            mapfile.data = numpy.random.rand(*args.size[::-1])
        mapfile.add_label(args.label)
        print(mapfile)
    return os.EX_OK
