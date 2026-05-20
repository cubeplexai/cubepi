"""Entry point for the ``cubepi`` console script."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from cubepi.cli.trace import commands as trace_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cubepi")
    sub = parser.add_subparsers(dest="group", required=True)
    trace_commands.register(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    from cubepi.cli.trace.render import RichMissingError

    try:
        return handler(args)
    except RichMissingError as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
