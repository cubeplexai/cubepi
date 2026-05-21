from __future__ import annotations

import json

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402

from cubepi.cli.trace.loader import load_run  # noqa: E402
from cubepi.cli.trace.model import build_forest  # noqa: E402
from cubepi.tracing.exporters import JsonlSpanExporter  # noqa: E402
from cubepi.tracing.schema import CUBEPI_RUN_ID  # noqa: E402


def test_real_exporter_output_parses(tmp_path):
    exporter = JsonlSpanExporter(directory=tmp_path)
    provider = TracerProvider()
    tracer = provider.get_tracer("cubepi.tracing")
    # After the `with` blocks exit the spans have ended; the span objects are
    # ReadableSpans, so we can hand them straight to the exporter.
    with tracer.start_as_current_span("invoke_agent") as root:
        root.set_attribute(CUBEPI_RUN_ID, "roundtrip")
        with tracer.start_as_current_span("chat") as chat:
            chat.set_attribute(CUBEPI_RUN_ID, "roundtrip")
    exporter.export([root, chat])

    files = sorted(tmp_path.glob("*/roundtrip.jsonl"))
    assert files, "exporter wrote no file"
    loaded, skipped = load_run(files)
    assert skipped == 0
    names = {s.name for s in loaded}
    assert {"invoke_agent", "chat"} <= names
    forest = build_forest(loaded)
    assert any(r.span.name == "invoke_agent" for r in forest)
    # JSON shape sanity: every line is valid JSON with a context.span_id.
    for f in files:
        for line in f.read_text().splitlines():
            obj = json.loads(line)
            assert "span_id" in obj["context"]
