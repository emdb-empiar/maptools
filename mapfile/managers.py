import os

from mapfile import models


def view(args):
    """"""
    with models.MapFile(args.file) as mapfile:
        print(mapfile)
    return os.EX_OK


def edit(args):
    """"""
    with models.MapFile(args.file, args.file_mode) as mapfile:
        orientation = models.Orientation(**dict(zip(('cols', 'rows', 'sections'), list(args.orientation))))
        mapfile.orientation = orientation
    return os.EX_OK


def create(args):
    """"""
    with models.MapFile(args.file, args.file_mode) as mapfile:
        print(mapfile)
    return os.EX_OK
