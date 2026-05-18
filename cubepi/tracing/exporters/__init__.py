"""Span exporter implementations for cubepi.tracing.

JsonlSpanExporter is always available (depends only on the SDK).

OTLPSpanExporter is re-exported from
``opentelemetry-exporter-otlp-proto-http`` when the ``tracing-otlp``
optional dependency is installed. Without that extra, accessing
``cubepi.tracing.exporters.OTLPSpanExporter`` raises ``ImportError``
with an actionable install hint.
"""

from __future__ import annotations

from cubepi.tracing.exporters.jsonl import JsonlSpanExporter

__all__ = ["JsonlSpanExporter", "OTLPSpanExporter"]


def __getattr__(name: str):
    if name == "OTLPSpanExporter":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter as _OTLPSpanExporter,
            )
        except ImportError as exc:  # pragma: no cover - exercised w/o the extra
            raise ImportError(
                "OTLPSpanExporter requires the 'tracing-otlp' optional dependency. "
                "Install it via: pip install cubepi[tracing-otlp]"
            ) from exc
        return _OTLPSpanExporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
