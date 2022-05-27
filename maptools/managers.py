import os
import sys

import numpy
from styled import Styled

from maptools import models


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
    """Edit in place or to another file"""
    with models.MapFile(args.file, file_mode=args.file_mode, colour=args.colour, verbose=args.verbose) as mapin:
        if args.orientation is not None:
            mapin.orientation = models.Orientation.from_string(args.orientation)
        if args.voxel_sizes is not None:
            mapin.voxel_size = args.voxel_sizes
        if args.map_mode is not None:
            mapin.mode = args.map_mode
        if args.output is not None:
            with models.MapFile(args.output, 'w', colour=args.colour, verbose=args.verbose) as mapout:
                mapout.copy(mapin)
                mapout.add_label(args.label)
            if not args.quiet:
                print(mapout)
        else:
            mapin.add_label(args.label)
            if not args.quiet:
                print(mapin)
    return os.EX_OK


def create(args):
    """Create a valid EMDB MAP file from scratch"""
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
        if not args.quiet:
            print(mapfile)
    return os.EX_OK


def sample(args):
    """Sample the MAP grid by the specified factor"""
    with models.MapFile(args.file, file_mode=args.file_mode, colour=args.colour, verbose=args.verbose) as mapin:
        from maptools.engines import grid_resample
        data = grid_resample(mapin.data, args.factor)
        if args.output is not None:
            with models.MapFile(args.output, 'w', colour=args.colour, verbose=args.verbose) as mapout:
                mapout.copy(mapin)
                mapout.data = data
                mapout.add_label(args.label)
                if not args.quiet:
                    print(mapout)
        else:
            mapin.data = data
            mapin.add_label(args.label)
            if not args.quiet:
                print(mapin)
    return os.EX_OK
