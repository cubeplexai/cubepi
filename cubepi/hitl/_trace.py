from __future__ import annotations

import contextlib


class _NullSpan:
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
        from opentelemetry import trace
    except ImportError:
        yield _NullSpan()
        return
    tracer = trace.get_tracer("cubepi.hitl")
    with tracer.start_as_current_span(f"hitl.{kind}") as span:
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(f"hitl.{k}", v)
        yield span
