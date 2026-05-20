from __future__ import annotations

import json

from cubepi.cli.trace.follow import format_event, is_run_complete, iter_new_spans
from cubepi.cli.trace.model import Span


def _line(span_id, parent_id, name):
    return json.dumps(
        {
            "name": name,
            "context": {"trace_id": "0xt", "span_id": span_id},
            "parent_id": parent_id,
            "start_time": "2026-05-20T00:00:00Z",
            "end_time": "2026-05-20T00:00:00.1Z",
            "status": {"status_code": "UNSET"},
            "attributes": {"gen_ai.operation.name": name},
        }
    )


def test_iter_new_spans_incremental(tmp_path):
    f = tmp_path / "r1.jsonl"
    f.write_text(_line("0x2", "0x1", "chat") + "\n")
    state = {"offset": 0, "buffer": ""}
    spans = list(iter_new_spans(f, state))
    assert [s.name for s in spans] == ["chat"]
    # Append more; only the new span is yielded.
    with f.open("a") as fh:
        fh.write(_line("0x3", "0x1", "execute_tool") + "\n")
    spans = list(iter_new_spans(f, state))
    assert [s.name for s in spans] == ["execute_tool"]


def test_partial_line_held_until_newline(tmp_path):
    f = tmp_path / "r1.jsonl"
    payload = _line("0x2", "0x1", "chat")
    f.write_text(payload)  # no trailing newline yet
    state = {"offset": 0, "buffer": ""}
    assert list(iter_new_spans(f, state)) == []  # held in buffer
    with f.open("a") as fh:
        fh.write("\n")
    spans = list(iter_new_spans(f, state))
    assert [s.name for s in spans] == ["chat"]


def test_is_run_complete_on_root():
    root = Span(json.loads(_line("0x1", None, "invoke_agent")))
    child = Span(json.loads(_line("0x2", "0x1", "chat")))
    assert is_run_complete(child) is False
    assert is_run_complete(root) is True


def test_format_event_contains_name():
    sp = Span(json.loads(_line("0x2", "0x1", "chat")))
    assert "chat" in format_event(sp)
