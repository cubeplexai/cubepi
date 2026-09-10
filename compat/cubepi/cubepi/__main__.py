"""`cubepi` console script — warn, then dispatch to cubeloop.cli."""

from __future__ import annotations

import sys
from collections.abc import Sequence

_WARNING = (
    "WARNING: The 'cubepi' package has been renamed to 'cubeloop'.\n"
    "Install `cubeloop` and import from `cubeloop` instead.\n"
    "See https://cubeloop.dev/docs/migration/from-cubepi\n"
)


def main(argv: Sequence[str] | None = None) -> int:
    print(_WARNING, file=sys.stderr, end="")
    from cubeloop.cli.__main__ import main as _real

    return _real(argv)


if __name__ == "__main__":
    sys.exit(main())
