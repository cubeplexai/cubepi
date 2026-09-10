"""Transitional alias for the renamed cubeloop package."""

from __future__ import annotations

import sys
import warnings

from cubepi._redirect import install as _install_redirect

_WARNING = (
    "WARNING: The 'cubepi' package has been renamed to 'cubeloop'.\n"
    "Install `cubeloop` and import from `cubeloop` instead.\n"
    "See https://cubeloop.dev/docs/migration/from-cubepi\n"
)

_install_redirect()

if not getattr(sys, "_cubepi_rename_warned", False):
    sys._cubepi_rename_warned = True  # type: ignore[attr-defined]
    warnings.warn(
        "The 'cubepi' package has been renamed to 'cubeloop'. "
        "Install `cubeloop` and import from `cubeloop` instead. "
        "See https://cubeloop.dev/docs/migration/from-cubepi",
        DeprecationWarning,
        stacklevel=2,
    )
    print(_WARNING, file=sys.stderr, end="")

from cubeloop import *  # noqa: E402,F403
from cubeloop import __all__ as __all__  # noqa: E402
from cubeloop import __version__ as __version__  # noqa: E402
