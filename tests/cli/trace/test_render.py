from __future__ import annotations

from cubepi.cli.trace.model import Span, build_forest
from cubepi.cli.trace.render import render_tree_to_text


def _raw(span_id, parent_id, name, attrs=None, status="UNSET"):
    return {
        "name": name,
        "context": {"trace_id": "0xt", "span_id": span_id},
        "parent_id": parent_id,
        "start_time": "2026-05-20T00:00:00.000000Z",
        "end_time": "2026-05-20T00:00:00.100000Z",
        "status": {"status_code": status},
        "attributes": attrs or {},
    }


def test_render_tree_includes_names_and_markers():
    spans = [
        Span(_raw("0x1", None, "invoke_agent",
                  {"cubepi.run_id": "r1", "gen_ai.operation.name": "invoke_agent"})),
        Span(_raw("0x2", "0x1", "execute_tool read",
                  {"gen_ai.operation.name": "execute_tool",
                   "gen_ai.tool.name": "read"}, status="ERROR")),
        Span(_raw("0x9", "0xMISSING", "chat gpt-x",
                  {"gen_ai.operation.name": "chat"})),
    ]
    forest = build_forest(spans)
    text = render_tree_to_text(forest)
    assert "invoke_agent" in text
    assert "execute_tool" in text
    assert "read" in text
    assert "orphan" in text.lower()
