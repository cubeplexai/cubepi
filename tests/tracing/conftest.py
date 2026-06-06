from __future__ import annotations

import pytest


@pytest.fixture
def in_memory_exporter():
    """Attach a fresh InMemorySpanExporter to the global TracerProvider for
    the duration of one test.

    OTel forbids replacing the global TracerProvider once set, so we attach
    a fresh exporter to whatever provider is already installed (installing
    one if none is). Returns the exporter so tests can call
    `get_finished_spans()`.

    Skips the test if `opentelemetry` isn't installed (it's part of the
    optional `tracing` extra).
    """
    pytest.importorskip("opentelemetry")

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    current = trace.get_tracer_provider()
    if not isinstance(current, TracerProvider):
        current = TracerProvider()
        trace.set_tracer_provider(current)
    exporter = InMemorySpanExporter()
    current.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    exporter.clear()
