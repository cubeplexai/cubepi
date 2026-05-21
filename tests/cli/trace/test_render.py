from __future__ import annotations

from pathlib import Path

from cubepi.cli.trace.loader import RunSummary
from cubepi.cli.trace.model import Span, build_forest
from cubepi.cli.trace.render import render_runs, render_tree_to_text


def _raw(
    span_id, parent_id, name, attrs=None, status="UNSET", description=None, events=None
):
    status_obj = {"status_code": status}
    if description is not None:
        status_obj["description"] = description
    return {
        "name": name,
        "context": {"trace_id": "0xt", "span_id": span_id},
        "parent_id": parent_id,
        "start_time": "2026-05-20T00:00:00.000000Z",
        "end_time": "2026-05-20T00:00:00.100000Z",
        "status": status_obj,
        "attributes": attrs or {},
        "events": events or [],
    }


def test_render_tree_includes_names_and_markers():
    spans = [
        Span(
            _raw(
                "0x1",
                None,
                "invoke_agent",
                {"cubepi.run_id": "r1", "gen_ai.operation.name": "invoke_agent"},
            )
        ),
        Span(
            _raw(
                "0x2",
                "0x1",
                "execute_tool read",
                {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "read"},
                status="ERROR",
            )
        ),
        Span(_raw("0x9", "0xMISSING", "chat gpt-x", {"gen_ai.operation.name": "chat"})),
    ]
    forest = build_forest(spans)
    text = render_tree_to_text(forest)
    assert "invoke_agent" in text
    assert "execute_tool" in text
    assert "read" in text
    assert "orphan" in text.lower()


def test_render_tree_surfaces_exception_message():
    # The full error lives in the exception event; show it without -v.
    spans = [
        Span(
            _raw(
                "0x1",
                None,
                "chat gpt-x",
                {"gen_ai.operation.name": "chat"},
                status="ERROR",
                description="truncated...",
                events=[
                    {
                        "name": "gen_ai.client.operation.exception",
                        "attributes": {
                            "exception.type": "BadRequestError",
                            "exception.message": "tool_use without tool_result blocks",
                        },
                    }
                ],
            )
        ),
    ]
    text = render_tree_to_text(build_forest(spans))
    assert "tool_use without tool_result blocks" in text


def test_render_tree_falls_back_to_status_description():
    # No exception event -> use the OTel status description.
    spans = [
        Span(
            _raw(
                "0x1",
                None,
                "execute_tool read",
                {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "read"},
                status="ERROR",
                description="boom from tool",
            )
        ),
    ]
    text = render_tree_to_text(build_forest(spans))
    assert "boom from tool" in text


def test_render_runs_includes_input_column(capsys):
    runs = [
        RunSummary(
            run_id="r1",
            files=[Path("x.jsonl")],
            start=None,
            span_count=3,
            has_error=False,
            duration_ms=12.0,
            prompt="北京明天天气如何",
        )
    ]
    render_runs(runs)
    out = capsys.readouterr().out
    assert "input" in out
    assert "北京" in out
