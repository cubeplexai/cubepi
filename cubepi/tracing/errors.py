"""``error.type`` value derivation for cubepi tracing.

The OTel GenAI semantic convention does NOT prescribe a closed enum for
``error.type``. We adopt the cubepi conventions documented in
``docs/specs/2026-05-18-cubepi-tracing-design.md`` §12.3:

- Provider HTTP 4xx/5xx → ``"<provider>.<status>"``
- Provider client class raised → fully-qualified Python class name
- Network timeout → ``"timeout"``
- Connection refused → ``"connection_error"``
- Agent aborted → ``"cubepi.aborted"``
- Tool / business error → exception class name
"""

from __future__ import annotations

import asyncio


def cubepi_error_type_for(exc: BaseException) -> str:
    """Derive an ``error.type`` value for a thrown exception.

    Cheap and best-effort; observers should rely on the
    ``gen_ai.client.operation.exception`` event for full details.
    """
    if isinstance(exc, asyncio.CancelledError):
        return "cubepi.aborted"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, TimeoutError):  # builtin in 3.11+
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection_error"

    cls = type(exc)
    module = cls.__module__
    name = cls.__qualname__
    if module in {"builtins", "__main__"}:
        return name
    return f"{module}.{name}"
