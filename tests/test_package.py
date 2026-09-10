"""Top-level package contract tests.

Things every release must satisfy: importable, has a sane
``__version__`` matching the installed distribution metadata, doesn't
silently regress the public surface.
"""

from __future__ import annotations

import re
from importlib.metadata import version as _pkg_version


def test_version_attribute_present_and_matches_metadata() -> None:
    """``cubepi.__version__`` must exist and match the version pip
    sees for the installed distribution.

    Several integrations read it (``cubepi.tracing.Tracer`` /
    ``Meter`` use it for OTel ``instrumenting_library_version`` so
    spans/metrics carry the cubepi version). A drift between
    ``cubepi.__version__`` and the installed distribution would
    make trace metadata lie about what produced the spans.
    """
    import cubeloop

    assert hasattr(cubeloop, "__version__"), "cubepi must expose __version__"
    assert cubeloop.__version__ == _pkg_version("cubeloop"), (
        f"cubeloop.__version__ ({cubeloop.__version__!r}) drifted from the "
        f"installed distribution version ({_pkg_version('cubeloop')!r}) — "
        "check pyproject.toml's [project].version field."
    )


def test_version_string_shape() -> None:
    """Version should look like a PEP 440 release (e.g. ``0.4.0``,
    ``0.4.1.dev0``). Catches accidental ``unknown`` fallbacks landing
    in a published package."""
    import cubeloop

    assert re.match(r"^\d+\.\d+", cubeloop.__version__), (
        f"cubepi.__version__ {cubeloop.__version__!r} doesn't look like a "
        f"PEP 440 version — fallback path may have been hit."
    )


def test_version_listed_in_dunder_all() -> None:
    """``__version__`` must be in ``__all__`` so ``from cubeloop import *``
    surfaces it and editors auto-complete it."""
    import cubeloop

    assert "__version__" in cubeloop.__all__
