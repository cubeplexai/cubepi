"""cubepi.tracing — OpenTelemetry-compatible observability for cubepi agents.

Build a :class:`Tracer`, attach it to an :class:`~cubepi.agent.Agent`, and
spans will flow to whichever :class:`~opentelemetry.sdk.trace.export.SpanExporter`
implementations you configure (e.g. :class:`JsonlSpanExporter` for local
files, or the OTLP exporter shipped with ``opentelemetry-exporter-otlp-proto-http``).

The heavyweight members (:class:`Tracer`, :class:`Meter`,
:class:`JsonlSpanExporter`, :func:`tracing_context`, ``SCHEMA_URL``) require
the optional ``opentelemetry-sdk`` dependency; install with
``pip install cubepi[tracing]``. They are imported lazily so that pure-metadata
submodules such as :mod:`cubepi.tracing.schema` remain importable without the
SDK (used by the ``cubepi trace`` CLI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "JsonlSpanExporter",
    "Meter",
    "SCHEMA_URL",
    "Tracer",
    "tracing_context",
]

if TYPE_CHECKING:  # pragma: no cover
    from cubepi.tracing.context import tracing_context
    from cubepi.tracing.exporters import JsonlSpanExporter
    from cubepi.tracing.meter import Meter
    from cubepi.tracing.schema import SCHEMA_URL
    from cubepi.tracing.tracer import Tracer

_LAZY = {
    "tracing_context": ("cubepi.tracing.context", "tracing_context"),
    "JsonlSpanExporter": ("cubepi.tracing.exporters", "JsonlSpanExporter"),
    "Meter": ("cubepi.tracing.meter", "Meter"),
    "SCHEMA_URL": ("cubepi.tracing.schema", "SCHEMA_URL"),
    "Tracer": ("cubepi.tracing.tracer", "Tracer"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    try:
        import opentelemetry.sdk.trace  # noqa: F401
    except ImportError as exc:  # pragma: no cover — only without the extra.
        raise ImportError(
            "cubepi.tracing requires the 'opentelemetry-sdk' package. "
            "Install it via: pip install cubepi[tracing]"
        ) from exc
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(__all__)
