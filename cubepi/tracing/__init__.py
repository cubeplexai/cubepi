"""cubepi.tracing — OpenTelemetry-compatible observability for cubepi agents.

Build a :class:`Tracer`, attach it to an :class:`~cubepi.agent.Agent`, and
spans will flow to whichever :class:`~opentelemetry.sdk.trace.export.SpanExporter`
implementations you configure (e.g. :class:`JsonlSpanExporter` for local
files, or the OTLP exporter shipped with ``opentelemetry-exporter-otlp-proto-http``).

This module requires the optional ``opentelemetry-sdk`` dependency; install
with ``pip install cubepi[tracing]``.
"""

try:  # noqa: SIM105 — explicit ImportError handling for the optional extra.
    import opentelemetry.sdk.trace  # noqa: F401
except ImportError as exc:  # pragma: no cover — exercised only without the extra.
    raise ImportError(
        "cubepi.tracing requires the 'opentelemetry-sdk' package. "
        "Install it via: pip install cubepi[tracing]"
    ) from exc

from cubepi.tracing.exporters import JsonlSpanExporter
from cubepi.tracing.schema import SCHEMA_URL
from cubepi.tracing.tracer import Tracer

__all__ = [
    "JsonlSpanExporter",
    "SCHEMA_URL",
    "Tracer",
]
