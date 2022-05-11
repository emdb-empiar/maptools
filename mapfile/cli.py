from __future__ import annotations

import argparse
import itertools
import shlex
import sys

# options
file = {
    'args': ['file'],
    'kwargs': dict(
        help="a valid EMDB MAP file"
    )
}
orientation = {
    'args': ['-O', '--orientation'],
    'kwargs': dict(
        choices=list(map(lambda t: ''.join(t), itertools.permutations('XYZ', 3))),  # all 3-permutations
        help="change the space orientation [default: XYZ]"
    )
}
voxel_sizes = {
    'args': ['--voxel-sizes'],
    'kwargs': dict(
        nargs=3,
        type=float,
        help="specify the voxel sizes in the order X, Y, Z [default: 1.0 1.0 1.0]"
    )
}

size = {
    'args': ['-s', '--size'],
    'kwargs': dict(
        default=[10, 10, 10],
        nargs=3,
        type=int,
        help="specify the EMDB MAP size in the order columns, rows, sections [default: 10 10 10]"
    )
}
INT_MAP_MODES = [0, 1, 3, 6]
FLOAT_MAP_MODES = [2, 12]
COMPLEX_MAP_MODES = [4]
MAP_MODES = INT_MAP_MODES + FLOAT_MAP_MODES + COMPLEX_MAP_MODES
map_mode = {
    'args': ['--map-mode'],
    'kwargs': dict(
        type=int,
        choices=MAP_MODES,
        help="the map mode sets the number of bytes per voxel [default: 2 (= 4 bytes per voxel)"
    )
}

file_mode = {
    'args': ['--file-mode'],
    'kwargs': dict(
        default='r',
        choices=['r', 'r+', 'w'],
        help="file access mode with which to open the specified file [default: 'r']"
    )
}

start = {
    'args': ['--start'],
    'kwargs': dict(
        default=[0, 0, 0],
        nargs=3,
        type=int,
        help="position of first column, first row and first section (voxel grid units) [default: 0 0 0]"
    )
}

outfile = {
    'args': ['-o', '--outfile'],
    'kwargs': dict(
        help="output file name"
    )
}

parser = argparse.ArgumentParser(prog="map", description="Utilities to work with EMDB MAP files")


def _add_arg(parser_: argparse.ArgumentParser, option: dict, **kwargs):
    """Add options to parser"""
    for kwarg, value in kwargs.items():
        option['kwargs'][kwarg] = value
    return parser_.add_argument(*option['args'], **option['kwargs'])


parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('-v', '--verbose', default=False, action='store_true',
                           help="verbose output to terminal in addition to log files [default: False]")
parent_parser.add_argument('--colour', default=False, action='store_true',
                           help="highlight with colours [default: False]")

subparsers = parser.add_subparsers(dest='command', title='Commands available')

# view
view_parser = subparsers.add_parser(
    'view',
    description="view MAP file",
    help="view the contents of an EMDB MAP file",
    parents=[parent_parser]
)
_add_arg(view_parser, file)

# edit
edit_parser = subparsers.add_parser(
    'edit',
    description="edit MAP file",
    help="edit the attributes of an EMDB MAP file",
    parents=[parent_parser]
)
_add_arg(edit_parser, file)
_add_arg(edit_parser, orientation)
_add_arg(edit_parser, voxel_sizes)
_add_arg(edit_parser, file_mode, default='r+')
_add_arg(edit_parser, map_mode)
_add_arg(edit_parser, start)

# create
create_parser = subparsers.add_parser(
    'create',
    description="create a MAP file",
    help="create an EMDB MAP file",
    parents=[parent_parser]
)
_add_arg(create_parser, file)
_add_arg(create_parser, orientation)
_add_arg(create_parser, voxel_sizes)
_add_arg(create_parser, size)
_add_arg(create_parser, file_mode, default='w')
_add_arg(create_parser, map_mode)
data_mutex_group = create_parser.add_mutually_exclusive_group(required=False)
data_mutex_group.add_argument(
    '--zeros', action='store_true', default=True, help="EMDB MAP with zeros"
)
data_mutex_group.add_argument(
    '--ones', action='store_true', help="EMDB MAP with ones"
)
data_mutex_group.add_argument(
    '--empty', action='store_true', help="EMDB MAP which is empty"
)
data_mutex_group.add_argument(
    '--random-integers', action='store_true', help="EMDB MAP with integers"
)
data_mutex_group.add_argument(
    '--random-floats', action='store_true', help="EMDB MAP with random floats"
)
create_parser.add_argument(
    '--min', default=0, type=int, help="minimum integer [default: 0]"
)
create_parser.add_argument(
    '--max', default=10, type=int, help="minimum integer [default: 10]"
)
_add_arg(create_parser, start)


def parse_args():
    """Parse CLI args"""
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return args
    return args


def cli(cmd):
    """CLI standin"""
    sys.argv = shlex.split(cmd)
    return parse_args()
