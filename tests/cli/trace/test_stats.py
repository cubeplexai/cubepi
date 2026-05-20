from __future__ import annotations

from cubepi.cli.trace.model import Span
from cubepi.cli.trace.stats import aggregate


def _chat(model, in_tok, out_tok, dur_ms, error=False):
    end = f"2026-05-20T00:00:{dur_ms / 1000:09.6f}Z"
    return Span(
        {
            "name": "chat",
            "context": {"trace_id": "0xt", "span_id": "0x1"},
            "parent_id": "0x0",
            "start_time": "2026-05-20T00:00:00.000000Z",
            "end_time": end,
            "status": {"status_code": "ERROR" if error else "UNSET"},
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": model,
                "gen_ai.usage.input_tokens": in_tok,
                "gen_ai.usage.output_tokens": out_tok,
            },
        }
    )


def _tool(name, dur_ms, aborted=False):
    end = f"2026-05-20T00:00:{dur_ms / 1000:09.6f}Z"
    attrs = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name}
    if aborted:
        attrs["cubepi.aborted"] = True
        attrs["error.type"] = "cubepi.aborted"
    return Span(
        {
            "name": "execute_tool",
            "context": {"trace_id": "0xt", "span_id": "0x2"},
            "parent_id": "0x0",
            "start_time": "2026-05-20T00:00:00.000000Z",
            "end_time": end,
            "status": {"status_code": "UNSET"},
            "attributes": attrs,
        }
    )


def test_aggregate_by_model_tokens():
    spans = [_chat("gpt-x", 10, 5, 100), _chat("gpt-x", 20, 7, 300)]
    rows = aggregate(spans, by="model")
    assert len(rows) == 1
    row = rows[0]
    assert row.key == "gpt-x"
    assert row.count == 2
    assert row.input_tokens == 30
    assert row.output_tokens == 12
    assert row.error_rate == 0.0


def test_aggregate_by_tool_no_tokens_counts_aborts():
    spans = [_tool("read", 50), _tool("read", 150, aborted=True)]
    rows = aggregate(spans, by="tool")
    assert len(rows) == 1
    row = rows[0]
    assert row.key == "read"
    assert row.count == 2
    assert row.input_tokens is None  # tools have no token columns
    assert row.aborted == 1
