"""cubeloop.tracing — OpenTelemetry-compatible observability for cubepi agents.

Build a :class:`Tracer`, attach it to an :class:`~cubeloop.agent.Agent`, and
spans will flow to whichever :class:`~opentelemetry.sdk.trace.export.SpanExporter`
implementations you configure (e.g. :class:`JsonlSpanExporter` for local
files, or the OTLP exporter shipped with ``opentelemetry-exporter-otlp-proto-http``).

The heavyweight members (:class:`Tracer`, :class:`Meter`,
:class:`JsonlSpanExporter`, :func:`tracing_context`, ``SCHEMA_URL``) require
the optional ``opentelemetry-sdk`` dependency; install with
``pip install cubepi[tracing]``. They are imported lazily so that pure-metadata
submodules such as :mod:`cubeloop.tracing.schema` remain importable without the
SDK (used by the ``cubepi trace`` CLI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "JsonlSpanExporter",
    "Meter",
    "SCHEMA_URL",
    "Tracer",
    "trace",
    "tracing_context",
]

if TYPE_CHECKING:  # pragma: no cover
    from cubeloop.tracing.context import tracing_context
    from cubeloop.tracing.exporters import JsonlSpanExporter
    from cubeloop.tracing.meter import Meter
    from cubeloop.tracing.schema import SCHEMA_URL
    from cubeloop.tracing.tracer import Tracer, trace

_LAZY = {
    "tracing_context": ("cubeloop.tracing.context", "tracing_context"),
    "JsonlSpanExporter": ("cubeloop.tracing.exporters", "JsonlSpanExporter"),
    "Meter": ("cubeloop.tracing.meter", "Meter"),
    "SCHEMA_URL": ("cubeloop.tracing.schema", "SCHEMA_URL"),
    "Tracer": ("cubeloop.tracing.tracer", "Tracer"),
    "trace": ("cubeloop.tracing.tracer", "trace"),
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
            "cubeloop.tracing requires the 'opentelemetry-sdk' package. "
            "Install it via: pip install cubepi[tracing]"
        ) from exc
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(__all__)
