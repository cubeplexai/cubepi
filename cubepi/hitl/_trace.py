from __future__ import annotations

import contextlib
from typing import Any, cast


class _NullSpan:  # pragma: no cover - only exercised when opentelemetry is not installed
    def set_attribute(self, *a, **k):
        pass

    def add_event(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@contextlib.contextmanager
def hitl_span(kind: str, **attrs):
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:  # pragma: no cover - tracing extra not installed
        yield _NullSpan()
        return
    trace = cast(Any, otel_trace)
    tracer = trace.get_tracer("cubepi.hitl")
    with tracer.start_as_current_span(f"hitl.{kind}") as span:
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(f"hitl.{k}", v)
        yield span
