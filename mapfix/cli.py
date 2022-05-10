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
    'args': ['--orientation'],
    'kwargs': dict(
        default="XYZ",
        choices=list(map(lambda t: ''.join(t), itertools.permutations('XYZ', 3))),  # all 3-permutations
        help="change the space orientation [default: XYZ]"
    )
}
voxel_sizes = {
    'args': ['--voxel-sizes'],
    'kwargs': dict(
        default=[1.0, 1.0, 1.0],
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

file_mode = {
    'args': ['--file-mode'],
    'kwargs': dict(
        default='r',
        choices=['r', 'r+', 'w'],
        help="file access mode with which to open the specified file [default: 'r']"
    )
}

parser = argparse.ArgumentParser(prog="map", description="Utilities to work with EMDB MAP files")


def _add_arg(parser_: argparse.ArgumentParser, option: dict):
    """Add options to parser"""
    return parser_.add_argument(*option['args'], **option['kwargs'])


parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('-v', '--verbose', default=False, action='store_true',
                           help="verbose output to terminal in addition to log files [default: False]")

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


def parse_args():
    """Parse CLI args"""
    args = parser.parse_args()
    return args


def cli(cmd):
    """CLI standin"""
    sys.argv = shlex.split(cmd)
    return parse_args()
