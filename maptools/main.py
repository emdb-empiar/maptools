import os
import sys

from maptools import cli, managers


def main():
    args = cli.parse_args()
    if args.command is None:
        return os.EX_USAGE
    try:
        if args.command == 'view':
            return managers.view(args)
        elif args.command == 'edit':
            return managers.edit(args)
        elif args.command == 'create':
            return managers.create(args)
    except KeyboardInterrupt:
        return os.EX_DATAERR
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main())
