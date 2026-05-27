"""JsonlSpanExporter shards by trace_id, not per-span run_id."""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402

from cubepi.tracing.exporters import JsonlSpanExporter  # noqa: E402
from cubepi.tracing.schema import CUBEPI_RUN_ID  # noqa: E402


def test_jsonl_shards_by_trace_id_not_run_id(tmp_path):
    # A parent span and a nested (subagent) span share one trace_id but carry
    # DIFFERENT cubepi.run_id values — the subagent run gets its own run_id.
    exporter = JsonlSpanExporter(directory=tmp_path)
    provider = TracerProvider()
    tracer = provider.get_tracer("cubepi.tracing")
    with tracer.start_as_current_span("invoke_agent") as parent:
        parent.set_attribute(CUBEPI_RUN_ID, "parent-run")
        with tracer.start_as_current_span("invoke_agent") as child:
            # Same trace, different run id (nested subagent run).
            child.set_attribute(CUBEPI_RUN_ID, "subagent-run")

    assert parent.context.trace_id == child.context.trace_id
    assert parent.attributes[CUBEPI_RUN_ID] != child.attributes[CUBEPI_RUN_ID]

    exporter.export([parent, child])

    files = list(tmp_path.glob("*/*.jsonl"))
    assert len(files) == 1, [f.name for f in files]
    assert files[0].stem == format(parent.context.trace_id, "032x")
