import os

from mapfix import models


def view(args):
    """"""
    with models.MapFile(args.file) as mapfile:
        print(mapfile)
    return os.EX_OK


def edit(args):
    """"""
    with models.MapFile(args.file, args.file_mode) as mapfile:
        print(mapfile)
    return os.EX_OK


def create(args):
    """"""
    with models.MapFile(args.file, args.file_mode) as mapfile:
        print(mapfile)
    return os.EX_OK
